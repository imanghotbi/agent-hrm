import math
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from app.workflow.state import BatchState, OverallState
from app.workflow.nodes import router, hiring, jd, processing, comparison, qa
from app.config.logger import logger

def map_to_batches(state: OverallState):
    """Sharding Logic."""
    files = state["all_files"]
    reqs = state["hiring_reqs"]
    total_files = len(files)
    
    if total_files == 0:
        return []
        
    num_chunks = min(10, total_files)
    chunk_size = math.ceil(total_files / num_chunks)
    batch_requests = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = files[start_idx:end_idx]
        if chunk:
            batch_requests.append(Send("process_batch_subgraph", {
                "session_id": state["session_id"],
                "batch_id": i + 1,
                "files_in_batch": chunk,
                "hiring_reqs": reqs,
                "ocr_results": {},
                "structured_results": [],
                "evaluated_results": []
            }))

    logger.info(f"🔀 Sharded {total_files} files into {len(batch_requests)} batches (Target Max: 10).")
    return batch_requests

def define_path(state: OverallState):
    """Decides where to go after introductions."""
    intent = state.get("intent")
    if intent == "REVIEW": return "hiring_process"
    if intent == "WRITE": return "jd_process"
    if intent == "COMPARE": return "compare_input"
    return "router_input"

def should_continue_hiring(state: OverallState):
    """Decides where to go after hiring_process."""
    return "upload_resume" if state.get("hiring_reqs") else "hiring_input"

def should_continue_jd(state: OverallState):
    return "jd_writer" if state.get("jd_reqs") else "jd_input"

def build_graph():
    # 1. Subgraph
    workflow_batch = StateGraph(BatchState)
    workflow_batch.add_node("batch_ocr", processing.batch_ocr_node)
    workflow_batch.add_node("batch_structure", processing.batch_structure_node)
    workflow_batch.add_node("batch_evaluate", processing.batch_evaluate_node)
    workflow_batch.add_edge(START, "batch_ocr")
    workflow_batch.add_edge("batch_ocr", "batch_structure")
    workflow_batch.add_edge("batch_structure", "batch_evaluate")
    workflow_batch.add_edge("batch_evaluate", END)

    # 2. Main Graph
    workflow = StateGraph(OverallState)
    
    # Register Nodes
    workflow.add_node("router_process", router.router_process_node)
    workflow.add_node("router_input", router.router_input_node)
    
    workflow.add_node("hiring_process", hiring.hiring_process_node)
    workflow.add_node("hiring_input", hiring.hiring_input_node)
    workflow.add_node("upload_resume", hiring.upload_resume_node)
    
    workflow.add_node("jd_process", jd.jd_process_node)
    workflow.add_node("jd_input", jd.jd_input_node)
    workflow.add_node("jd_writer", jd.jd_writer_node)
    
    workflow.add_node("compare_input", comparison.compare_input_node)
    workflow.add_node("compare_process", comparison.compare_process_node)
    workflow.add_node("compare_qa_input", comparison.compare_qa_input_node)
    workflow.add_node("compare_qa_process", comparison.compare_qa_process_node)
    
    workflow.add_node("load_and_shard", processing.load_and_shard)
    workflow.add_node("process_batch_subgraph", workflow_batch.compile())
    workflow.add_node("save_results", processing.save_results_node)
    
    workflow.add_node("prepare_qa", qa.prepare_qa_node)
    workflow.add_node("qa_input", qa.qa_input_node)
    workflow.add_node("qa_process", qa.qa_process_node)
    workflow.add_node("top_candidates", qa.top_candidates_node)

    # Edges
    workflow.add_edge(START, "router_process")
    workflow.add_conditional_edges(
        "router_process", 
        define_path, 
        ["hiring_process", "router_input", "jd_process", "compare_input"]
    )
    workflow.add_edge("router_input", "router_process")

    workflow.add_conditional_edges(
        "hiring_process", 
        should_continue_hiring, 
        ["upload_resume", "hiring_input"]
    )
    workflow.add_edge("hiring_input", "hiring_process")

    workflow.add_conditional_edges(
        "jd_process", 
        should_continue_jd, 
        ["jd_writer", "jd_input"]
    )
    workflow.add_edge("jd_input", "jd_process")
    workflow.add_edge("jd_writer", END)

    workflow.add_edge("upload_resume","load_and_shard")
    workflow.add_conditional_edges(
        "load_and_shard", 
        map_to_batches, 
        ["process_batch_subgraph"]
    )
    workflow.add_edge("process_batch_subgraph", "save_results")
    workflow.add_edge("save_results", "top_candidates")
    workflow.add_edge("top_candidates", "prepare_qa")
    workflow.add_edge("prepare_qa", "qa_input")
    
    workflow.add_edge("qa_input", "qa_process")
    workflow.add_edge("qa_process", "qa_input")

    workflow.add_edge("compare_input", "compare_process")
    workflow.add_edge("compare_process", "compare_qa_input")
    workflow.add_edge("compare_qa_input", "compare_qa_process")
    workflow.add_edge("compare_qa_process", "compare_qa_input")

    return workflow