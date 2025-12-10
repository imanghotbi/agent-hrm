import operator
from typing import List, Dict, Any, Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from app.schemas.hiring import HiringRequirements
from app.schemas.job_description import JobDescriptionRequest

def update_latest(old, new):
    return new

class BatchState(TypedDict):
    batch_id: int
    files_in_batch: List[str]
    ocr_results: Dict[str, str]
    structured_results: List[Dict[str, Any]]
    hiring_reqs: HiringRequirements
    evaluated_results: List[Dict[str, Any]]

class OverallState(TypedDict):
    #run_id
    session_id : Annotated[str, update_latest]
    
    # Router
    intent: Annotated[Literal["REVIEW", "WRITE", "COMPARE"], update_latest]
    start_message: Annotated[List[BaseMessage], add_messages]
    
    # JD
    jd_messages: Annotated[List[BaseMessage], add_messages]
    jd_reqs: Annotated[JobDescriptionRequest, update_latest]
    final_jd: str
    
    # Hiring
    hiring_messages: Annotated[List[BaseMessage], add_messages]
    hiring_reqs: Annotated[HiringRequirements, update_latest]

    # Processing (Map-Reduce)
    evaluated_results: Annotated[List[Dict[str, Any]], operator.add]
    ocr_results: Annotated[List[Dict[str, str]], operator.add]
    all_files: List[str]

    # mongo Q&A
    db_structure: Annotated[Dict, update_latest]
    current_question: Annotated[str, update_latest]
    qa_answer: str

    # Compare Phase State
    compare_files: List[str]
    comparison_context: str
    compare_qa_answer: str