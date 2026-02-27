import operator
from typing import List, Dict, Any, Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from app.schemas.hiring import HiringRequirements

def update_latest(old, new):
    return new

class BatchState(TypedDict):
    session_id:str
    batch_id: int
    files_in_batch: List[str]
    ocr_results: Dict[str, str]
    structured_results: List[Dict[str, Any]]
    hiring_reqs: HiringRequirements
    evaluated_results: List[Dict[str, Any]]

class OverallState(TypedDict):
    # Run/session identifiers
    session_id : Annotated[str, update_latest]
    resume_dir: Annotated[str, update_latest]

    # Hiring requirements collection
    hiring_messages: Annotated[List[BaseMessage], add_messages]
    hiring_reqs: Annotated[HiringRequirements, update_latest]

    # Processing (Map-Reduce) accumulation
    evaluated_results: Annotated[List[Dict[str, Any]], operator.add]
    ocr_results: Annotated[List[Dict[str, str]], operator.add]
    all_files: List[str]

    # Review runtime metadata
    review_started_at: Annotated[str, update_latest]
    review_completed_at: Annotated[str, update_latest]
    review_duration_seconds: Annotated[float, update_latest]
    review_summary: Annotated[Dict[str, Any], update_latest]
