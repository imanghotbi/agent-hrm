import asyncio
from langchain_core.messages import HumanMessage
from langgraph.types import Command, interrupt
from langgraph.graph import END
from langchain_core.output_parsers import StrOutputParser

from app.config.logger import logger
from app.config.config import config
from app.services.minio_service import MinioHandler
from app.services.ocr import OCRService
from app.services.llm_factory import LLMFactory
from app.workflow.state import OverallState

from utils.prompt import COMPARISON_PROMPT, COMPARE_QA_PROMPT
from utils.helper import save_token_cost

# Reuse the OCR service (or create new if you want separate limits)
parser = StrOutputParser()

def compare_input_node(state: OverallState):
    """
    Interrupts to ask user for the resumes (files) to compare.
    """
    msg = "📂 Please provide the resumes you want to compare (PDFs). You can upload up to 3 files."
    user_input = interrupt(value={"type": "compare_upload", "msg": msg, "bucket_name": config.minio_compare_bucket})
    
    if (isinstance(user_input, str) and str(user_input).lower() in ["exit", "quit"]):
        return Command(goto=END)
    
    return {"compare_files": user_input}

async def compare_process_node(state: OverallState):
    """
    1. OCR the selected files.
    2. Generate Comparison Report.
    """
    session_id = state["session_id"]
    files = state["compare_files"]
    minio = MinioHandler()
    if len(files) == 0:
        files = await minio.list_files(config.minio_compare_bucket)
        logger.info(f"📂 Found {len(files)} total resumes.")

    logger.info(f"⚖️ Comparing {len(files)} resumes...")
    
    ocr_service = OCRService(node_name='compare_process_node_ocr', session_id=session_id)

    tasks = [ocr_service.process_file(minio, config.minio_compare_bucket, f) for f in files]
    results = await asyncio.gather(*tasks)
    
    combined_text = ""
    for i, (key, text) in enumerate(results):
        if text:
            combined_text += f"\n\n--- Candidate {i+1} ({key}) ---\n{text}"
    
    if not combined_text:
        return {"comparison_context": "No text extracted from files."}

    prompt = COMPARISON_PROMPT.format(count=len(files), resumes_text=combined_text)
    
    llm = LLMFactory.get_model(temperature=0.2)
    
    report = await llm.ainvoke([HumanMessage(content=prompt)])
    asyncio.create_task(save_token_cost("compare_process_node", session_id, report))
    report_content = parser.invoke(report)
    
    print("\n" + "="*40)
    print("      📊 COMPARISON REPORT")
    print("="*40 + "\n")
    print(report_content)
    print("\n" + "="*40 + "\n")
    
    return {"comparison_context": report_content}

def compare_qa_input_node(state: OverallState):
    """Interrupt for Q&A on the comparison."""
    user_q = interrupt(value="compare_qa_input")
    if not user_q or str(user_q).lower() in ["exit", "quit"]:
        return Command(goto=END)
    return {"current_question": user_q}

async def compare_qa_process_node(state: OverallState):
    """Answers questions based on the comparison context."""
    context = state["comparison_context"]
    question = state["current_question"]
    session_id = state["session_id"]
    
    prompt = COMPARE_QA_PROMPT.format(context=context, question=question)
    llm = LLMFactory.get_model()
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    asyncio.create_task(save_token_cost("compare_qa_process_node", session_id, response))
    response_content = parser.invoke(response)
    print(f"\n🤖 Comparison Assistant: {response_content}\n")
    return {"compare_qa_answer": response_content}