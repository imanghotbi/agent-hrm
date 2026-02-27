import asyncio
from datetime import datetime, timezone
from pathlib import Path

from app.config.logger import logger
from app.config.config import config
from app.services.mongo_service import MongoHandler
from app.services.ocr import OCRService
from app.services.analyzer import ResumeAnalyzerService
from app.schemas.hiring import HiringRequirements
from app.workflow.state import BatchState, OverallState

# Instantiate services once to reuse semaphores across batch calls
analyzer_service = ResumeAnalyzerService()

async def batch_ocr_node(state: BatchState):
    """
    Subgraph Node: Receives a list of files and runs OCR.
    """
    batch_id = state["batch_id"]
    files = state["files_in_batch"]
    session_id = state["session_id"]
    logger.info(f"⚙️ [Batch {batch_id}] Starting OCR for {len(files)} files...")
    
    ocr_service = OCRService(node_name="batch_ocr_node", session_id=session_id)
    
    # Process concurrently using the service
    tasks = [ocr_service.process_local_file(f) for f in files]
    results = await asyncio.gather(*tasks)
    
    # Filter out failures
    ocr_map = {k: v for k, v in results if v is not None}
    
    return {"ocr_results": ocr_map}

async def batch_structure_node(state: BatchState):
    """
    Subgraph Node: Structures OCR text into JSON.
    """
    ocr_map = state["ocr_results"]
    session_id = state['session_id']
    logger.info(f"⚙️ [Batch {state['batch_id']}] Structuring {len(ocr_map)} items...")
    
    tasks = [analyzer_service.structure_text(k, v, session_id) for k, v in ocr_map.items()]
    results = await asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r is not None]
    
    # Pass 'ocr_results' as a list to match the Reducer type in OverallState
    return {"structured_results": valid_results, "ocr_results": [ocr_map]}

async def batch_evaluate_node(state: BatchState):
    """
    Subgraph Node: Scores structured resumes against reqs.
    """
    structured_list = state["structured_results"]
    reqs_payload = state["hiring_reqs"]
    reqs = HiringRequirements(**reqs_payload) if isinstance(reqs_payload, dict) else reqs_payload
    batch_id = state["batch_id"]
    session_id = state['session_id']

    if not structured_list:
        return {"evaluated_results": []}

    logger.info(f"🧠 [Batch {batch_id}] Evaluating {len(structured_list)} resumes...")
    
    tasks = [analyzer_service.evaluate_resume(r, reqs, session_id) for r in structured_list]
    results = await asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r is not None]
    
    return {"evaluated_results": valid_results}

async def load_and_shard(state: OverallState):
    """
    Loads all available PDF files from a local folder.
    """
    resume_dir = state.get("resume_dir") or config.resume_source_dir
    base_dir = Path(resume_dir).expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"Resume directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Resume directory is not a folder: {base_dir}")

    files = sorted(
        str(path)
        for path in base_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    logger.info(f"📂 Found {len(files)} PDF resumes in {base_dir}.")
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
    session_id = state['session_id']
    for res in results:
        res['session_id'] = session_id
        await mongo.save_candidate(res)
    return

async def finalize_review_node(state: OverallState):
    """
    Final node: computes timing metadata and a compact summary.
    """
    completed_at = datetime.now(timezone.utc).isoformat()
    started_at = state.get("review_started_at")
    duration_seconds = None
    if started_at:
        try:
            normalized = started_at.replace("Z", "+00:00")
            duration_seconds = (datetime.fromisoformat(completed_at) - datetime.fromisoformat(normalized)).total_seconds()
        except Exception:
            logger.warning(f"Could not parse review_started_at timestamp: {started_at}")

    results = state.get("evaluated_results", [])
    ranked = sorted(results, key=lambda item: item.get("final_score", 0), reverse=True)
    top_candidates = []
    for item in ranked[:3]:
        resume = item.get("resume", {})
        personal = resume.get("personal_info", {})
        top_candidates.append({
            "name": personal.get("full_name"),
            "email": personal.get("email"),
            "source_file": resume.get("_source_file"),
            "final_score": item.get("final_score"),
        })

    summary = {
        "session_id": state.get("session_id"),
        "resume_dir": state.get("resume_dir"),
        "files_count": len(state.get("all_files", [])),
        "evaluated_count": len(results),
        "top_candidates": top_candidates,
    }
    logger.info(f"✅ Review finished. evaluated={len(results)} duration_seconds={duration_seconds}")

    return {
        "review_completed_at": completed_at,
        "review_duration_seconds": duration_seconds,
        "review_summary": summary,
    }
