import asyncio
import base64
import io
import json
import logging
from typing import List, Any, Dict, Optional
from pdf2image import convert_from_bytes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from src.config import config, logger
from src.schema import ResumeData
from src.storage import MinioHandler
from utils.prompt import OCR_PROMPT, STRUCTURE_PROMPT_TEMPLATE

# -- GRAPH STATE --
class AgentState(Dict):
    file_keys: List[str]
    ocr_results: Dict[str, str] 
    final_structs: List[Dict[str, Any]]
    errors: List[str]

# -- LLM FACTORY --
def get_llm(structured: bool = False):
    llm = ChatGoogleGenerativeAI(
        model=config.model_name,
        google_api_key=config.google_api_key.get_secret_value(),
        temperature=0,
        max_output_tokens=config.max_tokens,
        top_p=config.top_p,
        thinking_budget=5000
    )
    if structured:
        return llm.with_structured_output(ResumeData)
    return llm

# -- WORKER 1: OCR --
async def ocr_worker(minio: MinioHandler, file_key: str, semaphore: asyncio.Semaphore) -> Dict:
    async with semaphore:
        try:
            logger.info(f"🔹 [OCR] Processing: {file_key}")
            pdf_bytes = await minio.download_file_bytes(file_key)
            
            # 300 DPI as requested
            images = await asyncio.to_thread(convert_from_bytes, pdf_bytes, fmt='png', dpi=300)
            
            if not images:
                logger.warning(f"⚠️ [OCR] Empty PDF: {file_key}")
                return {"file": file_key, "error": "Empty PDF"}

            content_parts = [{"type": "text", "text": OCR_PROMPT}]
            
            for img in images:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })

            llm = get_llm(structured=False)
            msg = HumanMessage(content=content_parts)
            response = await llm.ainvoke([msg])
            
            return {"file": file_key, "raw_text": response.content}
            
        except Exception as e:
            logger.error(f"❌ [OCR] Failed {file_key}: {e}")
            return {"file": file_key, "error": f"OCR Failed: {str(e)}"}

# -- WORKER 2: STRUCTURE (With Retry) --
async def structure_worker(file_key: str, raw_text: str, semaphore: asyncio.Semaphore) -> Dict:
    async with semaphore:
        prompt = STRUCTURE_PROMPT_TEMPLATE.format(raw_text=raw_text)
        llm = get_llm(structured=True)
        
        # Retry Logic: 1 initial attempt + 2 retries = 3 total attempts
        max_attempts = 3
        
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    logger.info(f"🔄 [STRUCT] Retry {attempt-1}/{max_attempts-1} for {file_key}")
                else:
                    logger.info(f"🔸 [STRUCT] Extracting: {file_key}")
                
                # We send just text here
                response: ResumeData = await llm.ainvoke([HumanMessage(content=prompt)])
                
                # Success
                return {"file": file_key, "data": response.model_dump(mode='json')}
            
            except Exception as e:
                logger.warning(f"⚠️ [STRUCT] Attempt {attempt} failed for {file_key}: {e}")
                if attempt == max_attempts:
                    # Final failure
                    error_msg = f"Structure Failed after {max_attempts} attempts: {str(e)}"
                    logger.error(f"❌ {error_msg}")
                    return {
                        "file": file_key, 
                        "error": error_msg, 
                        "fallback_raw_text": raw_text 
                    }
                await asyncio.sleep(1 * attempt)

# -- GRAPH NODES --

async def load_resumes_node(state: AgentState):
    minio = MinioHandler()
    files = await minio.list_files()
    logger.info(f"📂 Found {len(files)} resumes.")
    return {"file_keys": files, "ocr_results": {}, "final_structs": [], "errors": []}

async def ocr_node(state: AgentState):
    minio = MinioHandler()
    files = state["file_keys"]
    sem = asyncio.Semaphore(config.ocr_workers)
    
    tasks = [ocr_worker(minio, f, sem) for f in files]
    results = await asyncio.gather(*tasks)
    
    ocr_map = {}
    errors = state.get("errors", [])
    
    for r in results:
        if "error" in r:
            errors.append(f"OCR Error in {r['file']}: {r['error']}")
        else:
            ocr_map[r['file']] = r['raw_text']
            
    return {"ocr_results": ocr_map, "errors": errors}

async def structure_node(state: AgentState):
    ocr_data = state["ocr_results"]
    sem = asyncio.Semaphore(config.structure_workers)
    
    tasks = []
    for filename, text in ocr_data.items():
        tasks.append(structure_worker(filename, text, sem))
        
    results = await asyncio.gather(*tasks)
    
    final_structs = []
    errors = state.get("errors", [])
    
    for r in results:
        if "error" in r:
            fallback_name = f"fallback_{r['file']}.txt"
            try:
                with open(fallback_name, "w", encoding="utf-8") as f:
                    f.write(r.get("fallback_raw_text", "No text"))
            except Exception as save_err:
                logger.error(f"Failed to save fallback: {save_err}")
            
            errors.append(f"Structure Error in {r['file']}: {r['error']} (Saved raw to {fallback_name})")
        else:
            data = r['data']
            data['_source_file'] = r['file']
            final_structs.append(data)
            
    return {"final_structs": final_structs, "errors": errors}

# -- BUILD GRAPH --
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("load", load_resumes_node)
    workflow.add_node("ocr", ocr_node)
    workflow.add_node("structure", structure_node)
    
    workflow.set_entry_point("load")
    workflow.add_edge("load", "ocr")
    workflow.add_edge("ocr", "structure")
    workflow.add_edge("structure", END)
    
    return workflow.compile()