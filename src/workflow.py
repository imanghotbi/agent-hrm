import asyncio
import base64
import io
import math
from typing import List, Any, Dict, Annotated ,TypedDict
import operator
from pdf2image import convert_from_bytes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from src.config import config, logger
from src.schemas.resume import ResumeData
from src.schemas.hiring import HiringRequirements
from src.storage import MinioHandler
from src.schemas.evaluation import ResumeEvaluation, ScoredResume
from utils.prompt import OCR_PROMPT, STRUCTURE_PROMPT_TEMPLATE ,SCORING_PROMPT

# -- GLOBAL SEMAPHORES --
ocr_semaphore = asyncio.Semaphore(config.ocr_workers)
structure_semaphore = asyncio.Semaphore(config.structure_workers)
eval_semaphore = asyncio.Semaphore(config.eval_workers)

def _keep_first(a, b):
    # When multiple values come in, keep the first one.
    return a

# -- STATES --
# 1. State for a BATCH of resumes (One of the 10 branches)
class BatchState(TypedDict):
    batch_id: int
    files_in_batch: List[str]
    ocr_results: Dict[str, str]
    structured_results: List[Dict[str, Any]]
    hiring_reqs: HiringRequirements
    # We must explicitly define this key so it gets returned to the parent
    evaluated_results: List[Dict[str, Any]]

# 2. State for the OVERALL application
class OverallState(TypedDict):
    all_files: List[str]
    hiring_reqs: Annotated[HiringRequirements , _keep_first]
    # Reducer: Combines lists of results from the 10 branches
    evaluated_results: Annotated[List[Dict[str, Any]], operator.add]
    ocr_results: Annotated[List[Dict[str, str]], operator.add]

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

# -- CORE LOGIC (HELPER FUNCTIONS) --
# These are called by the nodes. We keep them separate to allow internal parallelism.

async def process_single_ocr(minio, file_key):
    """Downloads and extracts text from one file."""
    async with ocr_semaphore: # Wait for slot
        try:
            logger.info(f"🔹 [OCR START] {file_key}")
            pdf_bytes = await minio.download_file_bytes(file_key)
            
            # 300 DPI
            images = await asyncio.to_thread(convert_from_bytes, pdf_bytes, fmt='png', dpi=300)
            
            if not images:
                return file_key, None

            
            text_result = ""
            for img in images:
                content_parts = [{"type": "text", "text": OCR_PROMPT}]
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
                text_result += response.content + "\n"

            return file_key, text_result
        except Exception as e:
            logger.error(f"❌ [OCR ERROR] {file_key}: {e}")
            return file_key, None

async def process_single_structure(file_key, text):
    """Structures text for one file."""
    if not text:
        return None

    async with structure_semaphore: # Wait for slot
        prompt = STRUCTURE_PROMPT_TEMPLATE.format(raw_text=text)
        llm = get_llm(structured=True)
        
        for attempt in range(1, 4): # 3 retries ##TODO add this to config file
            try:
                response: ResumeData = await llm.ainvoke([HumanMessage(content=prompt)])
                data = response.model_dump(mode='json')
                data["_source_file"] = file_key
                logger.info(f"✅ [STRUCT DONE] {file_key}")
                return data
            except Exception as e:
                if attempt == 3:
                    logger.error(f"❌ [STRUCT FAILED] {file_key}: {e}")
                else:
                    await asyncio.sleep(1 * attempt)
        return None

async def process_single_evaluation(resume_dict: Dict, reqs: HiringRequirements):
    """
    1. Calls LLM to get component scores.
    2. Calculates Weighted Average.
    """
    async with eval_semaphore:
        try:
            # Reconstruct Pydantic object for validation
            resume_obj = ResumeData(**resume_dict)
            file_key = resume_dict.get("_source_file", "unknown")
            
            req_json = reqs.model_dump_json()
            res_json = resume_obj.model_dump_json()
            
            prompt = SCORING_PROMPT.format(requirements_json=req_json, resume_json=res_json)
            
            llm = ChatGoogleGenerativeAI(
                model=config.model_name,
                google_api_key=config.google_api_key,
                temperature=0
            ).with_structured_output(ResumeEvaluation)

            # 1. Get raw scores
            eval_result: ResumeEvaluation = await llm.ainvoke([HumanMessage(content=prompt)])
            
            # 2. Calculate Weighted Score
            weights = reqs.weights
            
            # Simple dot product
            weighted_sum = (
                (eval_result.hard_skills_score.score * weights.hard_skills_weight) +
                (eval_result.experience_score.score * weights.experience_weight) +
                (eval_result.education_score.score * weights.education_weight) +
                (eval_result.soft_skills_score.score * weights.soft_skills_weight) +
                (eval_result.military_status_score.score * weights.military_status_weight)
            )
            
            total_weight = (
                weights.hard_skills_weight + weights.experience_weight + 
                weights.education_weight + weights.soft_skills_weight + 
                weights.military_status_weight
            )
            
            final_score = round(weighted_sum / total_weight, 2)
            eval_result.final_weighted_score = final_score
            
            logger.info(f"⚖️ [EVAL DONE] {file_key} -> {final_score}")
            
            # Return combined structure for Mongo
            return {
                "resume": resume_dict,
                "evaluation": eval_result.model_dump(),
                "final_score": final_score
            }

        except Exception as e:
            logger.error(f"❌ [EVAL ERROR] {resume_dict.get('_source_file')}: {e}")
            return None
# -- GRAPH NODES (BATCH LEVEL) --

async def batch_ocr_node(state: BatchState):
    """
    Receives a list of files (e.g., 10 files).
    Processes them concurrently using the global semaphore.
    """
    batch_id = state["batch_id"]
    files = state["files_in_batch"]
    logger.info(f"⚙️ [Batch {batch_id}] Starting OCR for {len(files)} files...")
    
    minio = MinioHandler()
    
    # Run all OCR tasks in this batch concurrently
    # The global `ocr_semaphore` limits total system load, not this batch alone.
    tasks = [process_single_ocr(minio, f) for f in files]
    results = await asyncio.gather(*tasks)
    
    # Convert list of tuples to dict, filtering failures
    ocr_map = {k: v for k, v in results if v is not None}
    
    return {"ocr_results": ocr_map}

async def batch_structure_node(state: BatchState):
    """
    Receives OCR text for the batch.
    Processes structure concurrently.
    """
    ocr_map = state["ocr_results"]
    logger.info(f"⚙️ [Batch {state['batch_id']}] Structuring {len(ocr_map)} items...")
    
    tasks = [process_single_structure(k, v) for k, v in ocr_map.items()]
    results = await asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r is not None]
    
    # We return 'final_results' which matches the Reducer key in OverallState
    return {"structured_results": valid_results , "ocr_results": [ocr_map]}

async def batch_evaluate_node(state: BatchState):
    structured_list = state["structured_results"]
    reqs = state["hiring_reqs"]
    batch_id = state["batch_id"]
    
    if not structured_list:
        return {"evaluated_results": []}

    logger.info(f"🧠 [Batch {batch_id}] Evaluating {len(structured_list)} resumes...")
    
    tasks = [process_single_evaluation(r, reqs) for r in structured_list]
    results = await asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r is not None]
    
    return {"evaluated_results": valid_results}

# -- GRAPH NODES (MAIN LEVEL) --

async def load_and_shard(state: OverallState):
    minio = MinioHandler()
    files = await minio.list_files()
    logger.info(f"📂 Found {len(files)} total resumes.")
    return {"all_files": files}

def map_to_batches(state: OverallState):
    """
    The Sharding Logic:
    Splits the list of files into MAX 10 chunks.
    """
    files = state["all_files"]
    reqs = state["hiring_reqs"]
    total_files = len(files)
    
    if total_files == 0:
        return []
        
    # Determine number of chunks (Max 10)
    num_chunks = min(10, total_files)
    if num_chunks == 0: return []
    
    # Calculate chunk size (ceil to ensure all are covered)
    chunk_size = math.ceil(total_files / num_chunks)
    
    batch_requests = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk_files = files[start_idx:end_idx]
        
        if not chunk_files:
            continue
            
        # Create a Send object for this chunk
        batch_requests.append(
            Send("process_batch_subgraph", {
                "batch_id": i + 1,
                "files_in_batch": chunk_files,
                "hiring_reqs": reqs,
                "ocr_results": {},
                "structured_results": [],
                "evaluated_results": []
            })
        )
    
    logger.info(f"🔀 Sharded {total_files} files into {len(batch_requests)} batches (Target Max: 10).")
    return batch_requests

# -- BUILD GRAPH --

def build_graph():
    # 1. Subgraph
    workflow_batch = StateGraph(BatchState)
    workflow_batch.add_node("batch_ocr", batch_ocr_node)
    workflow_batch.add_node("batch_structure", batch_structure_node)
    workflow_batch.add_node("batch_evaluate", batch_evaluate_node) # NEW
    
    workflow_batch.add_edge(START, "batch_ocr")
    workflow_batch.add_edge("batch_ocr", "batch_structure")
    workflow_batch.add_edge("batch_structure", "batch_evaluate")
    workflow_batch.add_edge("batch_evaluate", END)

    # 2. Main Graph
    workflow_main = StateGraph(OverallState)
    workflow_main.add_node("load_and_shard", load_and_shard)
    workflow_main.add_node("process_batch_subgraph", workflow_batch.compile())
    
    workflow_main.add_edge(START, "load_and_shard")
    
    # Conditional Map
    workflow_main.add_conditional_edges(
        "load_and_shard",
        map_to_batches,
        ["process_batch_subgraph"]
    )
    
    workflow_main.add_edge("process_batch_subgraph", END)
    
    return workflow_main.compile()