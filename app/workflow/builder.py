import math
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from app.workflow.state import BatchState, OverallState
from app.workflow.nodes import hiring, processing
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

def should_continue_hiring(state: OverallState):
    """Decides where to go after hiring_process."""
    return "load_and_shard" if state.get("hiring_reqs") else "hiring_input"

def should_process_files(state: OverallState):
    """Decides whether there are files to process."""
    return "dispatch_batches" if state.get("all_files") else "finalize_review"

def dispatch_batches_node(state: OverallState):
    """No-op node to branch into dynamic map stage."""
    return {}

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

    # Register nodes for hiring-only flow
    workflow.add_node("hiring_process", hiring.hiring_process_node)
    workflow.add_node("hiring_input", hiring.hiring_input_node)

    workflow.add_node("load_and_shard", processing.load_and_shard)
    workflow.add_node("dispatch_batches", dispatch_batches_node)
    workflow.add_node("process_batch_subgraph", workflow_batch.compile())
    workflow.add_node("save_results", processing.save_results_node)
    workflow.add_node("finalize_review", processing.finalize_review_node)

    # Edges
    workflow.add_edge(START, "hiring_process")

    workflow.add_conditional_edges(
        "hiring_process", 
        should_continue_hiring, 
        ["load_and_shard", "hiring_input"]
    )
    workflow.add_edge("hiring_input", "hiring_process")

    workflow.add_conditional_edges(
        "load_and_shard",
        should_process_files,
        ["dispatch_batches", "finalize_review"]
    )
    workflow.add_conditional_edges(
        "dispatch_batches",
        map_to_batches,
        ["process_batch_subgraph"]
    )
    workflow.add_edge("process_batch_subgraph", "save_results")
    workflow.add_edge("save_results", "finalize_review")
    workflow.add_edge("finalize_review", END)

    return workflow
