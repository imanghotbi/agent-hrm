import asyncio

from app.config.logger import logger
from app.services.minio_service import MinioHandler
from app.services.mongo_service import MongoHandler
from app.services.ocr import OCRService
from app.services.analyzer import ResumeAnalyzerService
from app.workflow.state import BatchState, OverallState

# Instantiate services once to reuse semaphores across batch calls
ocr_service = OCRService()
analyzer_service = ResumeAnalyzerService()

async def batch_ocr_node(state: BatchState):
    """
    Subgraph Node: Receives a list of files and runs OCR.
    """
    batch_id = state["batch_id"]
    files = state["files_in_batch"]
    logger.info(f"⚙️ [Batch {batch_id}] Starting OCR for {len(files)} files...")
    
    minio = MinioHandler()
    
    # Process concurrently using the service
    tasks = [ocr_service.process_file(minio, f) for f in files]
    results = await asyncio.gather(*tasks)
    
    # Filter out failures
    ocr_map = {k: v for k, v in results if v is not None}
    
    return {"ocr_results": ocr_map}

async def batch_structure_node(state: BatchState):
    """
    Subgraph Node: Structures OCR text into JSON.
    """
    ocr_map = state["ocr_results"]
    logger.info(f"⚙️ [Batch {state['batch_id']}] Structuring {len(ocr_map)} items...")
    
    tasks = [analyzer_service.structure_text(k, v) for k, v in ocr_map.items()]
    results = await asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r is not None]
    
    # Pass 'ocr_results' as a list to match the Reducer type in OverallState
    return {"structured_results": valid_results, "ocr_results": [ocr_map]}

async def batch_evaluate_node(state: BatchState):
    """
    Subgraph Node: Scores structured resumes against reqs.
    """
    structured_list = state["structured_results"]
    reqs = state["hiring_reqs"]
    batch_id = state["batch_id"]
    
    if not structured_list:
        return {"evaluated_results": []}

    logger.info(f"🧠 [Batch {batch_id}] Evaluating {len(structured_list)} resumes...")
    
    tasks = [analyzer_service.evaluate_resume(r, reqs) for r in structured_list]
    results = await asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r is not None]
    
    return {"evaluated_results": valid_results}

async def load_and_shard(state: OverallState):
    """
    Loads all available file keys from MinIO.
    """
    minio = MinioHandler()
    files = await minio.list_files()
    logger.info(f"📂 Found {len(files)} total resumes.")
    return {"all_files": files}

async def save_results_node(state: OverallState):
    """
    Saves all evaluated resumes to MongoDB.
    """
    results = state["evaluated_results"]
    if not results:
        logger.warning("No results to save.")
        return
    
    logger.info(f"💾 Saving {len(results)} candidates to MongoDB...")
    mongo = MongoHandler()
    for res in results:
        await mongo.save_candidate(res)
    return