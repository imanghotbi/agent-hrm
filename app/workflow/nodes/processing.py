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
    batch_id = state.get("batch_id", "unknown")
    files = state.get("files_in_batch") or []
    session_id = state.get("session_id", "unknown")
    logger.info(f"⚙️ [Batch {batch_id}] Starting OCR for {len(files)} files...")
    
    ocr_service = OCRService(node_name="batch_ocr_node", session_id=session_id)
    
    # Process concurrently using the service
    tasks = [ocr_service.process_local_file(f) for f in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out failures
    ocr_map = {}
    for item in results:
        if isinstance(item, Exception):
            logger.warning(f"⚠️ [Batch {batch_id}] OCR task failed: {item}")
            continue
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and item[1] is not None
        ):
            ocr_map[item[0]] = item[1]
    
    return {"ocr_results": ocr_map}

async def batch_structure_node(state: BatchState):
    """
    Subgraph Node: Structures OCR text into JSON.
    """
    ocr_map = state.get("ocr_results") or {}
    session_id = state.get("session_id", "unknown")
    batch_id = state.get("batch_id", "unknown")
    logger.info(f"⚙️ [Batch {batch_id}] Structuring {len(ocr_map)} items...")

    if not isinstance(ocr_map, dict) or not ocr_map:
        return {"structured_results": [], "ocr_results": [{}]}
    
    tasks = [analyzer_service.structure_text(k, v, session_id) for k, v in ocr_map.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_results = []
    for item in results:
        if isinstance(item, Exception):
            logger.warning(f"⚠️ [Batch {batch_id}] Structuring task failed: {item}")
            continue
        if item is not None:
            valid_results.append(item)
    
    # Pass 'ocr_results' as a list to match the Reducer type in OverallState
    return {"structured_results": valid_results, "ocr_results": [ocr_map]}

async def batch_evaluate_node(state: BatchState):
    """
    Subgraph Node: Scores structured resumes against reqs.
    """
    structured_list = state.get("structured_results") or []
    reqs_payload = state.get("hiring_reqs")
    batch_id = state.get("batch_id", "unknown")
    session_id = state.get("session_id", "unknown")
    try:
        reqs = HiringRequirements(**reqs_payload) if isinstance(reqs_payload, dict) else reqs_payload
    except Exception as exc:
        logger.warning(f"⚠️ [Batch {batch_id}] Invalid hiring requirements; skipping evaluation: {exc}")
        return {"evaluated_results": []}

    if not structured_list:
        return {"evaluated_results": []}

    logger.info(f"🧠 [Batch {batch_id}] Evaluating {len(structured_list)} resumes...")
    
    tasks = [analyzer_service.evaluate_resume(r, reqs, session_id) for r in structured_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_results = []
    for item in results:
        if isinstance(item, Exception):
            logger.warning(f"⚠️ [Batch {batch_id}] Evaluation task failed: {item}")
            continue
        if item is not None:
            valid_results.append(item)
    
    return {"evaluated_results": valid_results}

async def load_and_shard(state: OverallState):
    """
    Loads all available PDF files from a local folder.
    """
    resume_dir = state.get("resume_dir") or config.resume_source_dir
    try:
        base_dir = Path(resume_dir).expanduser().resolve()
    except Exception as exc:
        logger.warning(f"Could not resolve resume directory '{resume_dir}': {exc}")
        return {"all_files": []}

    if not base_dir.exists():
        logger.warning(f"Resume directory does not exist: {base_dir}. Skipping file processing.")
        return {"all_files": []}
    if not base_dir.is_dir():
        logger.warning(f"Resume directory is not a folder: {base_dir}. Skipping file processing.")
        return {"all_files": []}

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
    results = state.get("evaluated_results") or []
    if not results:
        logger.warning("No results to save.")
        return
    
    logger.info(f"💾 Saving {len(results)} candidates to MongoDB...")
    mongo = MongoHandler()
    session_id = state.get("session_id", "unknown")
    inserted_count = 0
    updated_count = 0
    unchanged_count = 0
    skipped_count = 0
    for res in results:
        try:
            if not isinstance(res, dict):
                skipped_count += 1
                logger.warning("Skipping result save: result item is not a dict.")
                continue
            res["session_id"] = session_id
            operation = await mongo.save_candidate(res)
            if operation == "inserted":
                inserted_count += 1
            elif operation == "updated":
                updated_count += 1
            elif operation == "unchanged":
                unchanged_count += 1
            else:
                skipped_count += 1
        except Exception as exc:
            skipped_count += 1
            logger.exception(f"Skipping candidate after save error: {exc}")

    total_saved_ops = inserted_count + updated_count + unchanged_count
    logger.info(
        "✅ Save complete. total_ops=%s inserted=%s updated=%s unchanged=%s skipped=%s",
        total_saved_ops,
        inserted_count,
        updated_count,
        unchanged_count,
        skipped_count,
    )
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
    valid_results = [item for item in results if isinstance(item, dict)]
    ranked = sorted(valid_results, key=lambda item: item.get("final_score", 0), reverse=True)
    top_candidates = []
    for item in ranked[:3]:
        resume = item.get("resume")
        if not isinstance(resume, dict):
            resume = {}
        personal = resume.get("personal_info")
        if not isinstance(personal, dict):
            personal = {}
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
        "evaluated_count": len(valid_results),
        "top_candidates": top_candidates,
    }
    logger.info(f"✅ Review finished. evaluated={len(valid_results)} duration_seconds={duration_seconds}")

    return {
        "review_completed_at": completed_at,
        "review_duration_seconds": duration_seconds,
        "review_summary": summary,
    }
