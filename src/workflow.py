import asyncio
import base64
import io
import math
from typing import List, Any, Dict, Annotated ,TypedDict , Optional, Literal
from enum import Enum
import operator
from pdf2image import convert_from_bytes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage , SystemMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send  , interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool


from src.database import MongoHandler
from src.config import config, logger
from src.schemas.resume import ResumeData
from src.schemas.hiring import HiringRequirements
from src.schemas.job_description import JobDescriptionRequest,WorkMode,EmploymentType,MilitaryServiceRequirement,Language,SalaryRange,EducationLevel
from src.storage import MinioHandler
from src.schemas.evaluation import ResumeEvaluation, ScoredResume
from src.schemas.hiring import HiringRequirements , SeniorityLevel , PriorityWeights
from src.matcher import ResumeQAAgent
from utils.prompt import OCR_PROMPT, STRUCTURE_PROMPT_TEMPLATE ,SCORING_PROMPT, HIRING_AGENT_PROMPT , ROUTER_PROMPT,JD_REQUIREMENTS_GATHER,JD_WRITER_PROMPT
from utils.extract_structure import ExtractSchema

# -- GLOBAL SEMAPHORES --
ocr_semaphore = asyncio.Semaphore(config.ocr_workers)
structure_semaphore = asyncio.Semaphore(config.structure_workers)
eval_semaphore = asyncio.Semaphore(config.eval_workers)

parser = StrOutputParser()

def overwrite(a, b):
    """Always keep the newest value (allows updating state)."""
    return b

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
    # -- Router Phase --
    intent: Annotated[Literal["REVIEW", "WRITE"], overwrite]
    start_message: Annotated[List[BaseMessage], add_messages]
    
    # -- Job description phase --
    jd_messages: Annotated[List[BaseMessage], add_messages]
    jd_reqs: Annotated[JobDescriptionRequest , overwrite]
    final_jd: str
    
    # -- Hiring Phase State --
    hiring_messages: Annotated[List[BaseMessage], add_messages]
    hiring_reqs: Annotated[HiringRequirements , overwrite]

    # Reducer: Combines lists of results from the 10 branches
    # -- Processing Phase State --
    evaluated_results: Annotated[List[Dict[str, Any]], operator.add]
    ocr_results: Annotated[List[Dict[str, str]], operator.add]
    all_files: List[str]

    # -- Q&A Phase State --
    db_structure: Annotated[Dict , overwrite]
    current_question: Annotated[str, overwrite]
    qa_answer: str

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

# -- Tool --
@tool
def submit_hiring_requirements(role_title: str, 
                                seniority: SeniorityLevel, 
                                essential_hard_skills: List[str], 
                                military_service_required: bool, 
                                min_experience_years: int, 
                                education_level:str,
                                weights: PriorityWeights,
                                university_tier:int,
                                nice_to_have_skills:Optional[List[str]],
                                language_proficiency:Optional[str],
                                **kwargs):
    """
    Call this tool ONLY when you have gathered ALL necessary requirements from the user.
    """
    # This function acts as a dummy to validate inputs, the real data capture happens in the run loop
    return "Requirements captured."

@tool
def submit_jd_requirements(job_title: str, 
                            seniority_level: SeniorityLevel, 
                            location: str, 
                            education_level: EducationLevel,
                            study_fields: List[str],
                            work_mode: WorkMode,
                            employment_type: EmploymentType,
                            military_service: MilitaryServiceRequirement, 
                            min_experience_years: int, 
                            days_and_hours:str,
                            hard_skills: List[str],
                            soft_skills: List[str],
                            responsibilities: List[str],
                            advantage_skills:Optional[List[str]],
                            target_language: Language,
                            benefits: List[str],
                            salary: Optional[SalaryRange],
                            **kwargs):
    """
    Call this tool ONLY when you have gathered ALL necessary requirements from the user.
    """
    # This function acts as a dummy to validate inputs, the real data capture happens in the run loop
    return "Job description Requirements captured."

class Path(str, Enum):
    REVIEW = "REVIEW"
    WRITE = "WRITE"

@tool
def router_tool(path:Path):
    """
    Call this tool ONLY Once you understand what the application is intended for and which of the paths it needs.
    """
    return "The route has been determined."


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
                (eval_result.military_status_score.score * weights.military_status_weight) + 
                (eval_result.university_tier_score.score * weights.university_tier_weight)
            )
            
            total_weight = (
                weights.hard_skills_weight + weights.experience_weight + 
                weights.education_weight + weights.soft_skills_weight + 
                weights.military_status_weight + weights.university_tier_weight
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
async def router_process_node(state: OverallState):
    """
    Acts as the Receptionist. Explains features and asks for request.
    """
    messages = state["start_message"]
    
    # Ensure system prompt is present
    if not messages or not isinstance(messages[0], SystemMessage):
        # We prepend system prompt if not there (conceptually)
        # For 'add_messages', we just add it if history is empty
        sys_msg = SystemMessage(content=ROUTER_PROMPT)
        messages = [sys_msg] + messages

    llm = ChatGoogleGenerativeAI(
        model=config.model_name,
        google_api_key=config.google_api_key.get_secret_value(),
        temperature=0.0
    ).bind_tools([router_tool])
    
    response = await llm.ainvoke(messages)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "router_tool":
            logger.info("🎯 Routing Path Defined")
            try:
                # Extract args and build the object
                args = tool_call["args"]
                
                # Return command to jump to next phase (load files)
                # We also save the reqs to state
                return {
                    "intent": args['path']
                }
            except Exception as e:
                # Validation error - send back to agent to fix
                logger.error(f"Validation Error: {e}")
                err_msg = ToolMessage(tool_call_id=tool_call['id'], content=f"Error: {str(e)}")
                return {"intent": [response, err_msg]}
    # If just text, return the response so we can show it to user and wait for input
    print(f"\n🤖 Agent Answer: {parser.invoke(response)}\n")
    return {"start_message": [response]}

def router_input_node(state:OverallState):
    """
    Stops the graph and waits for user input.
    """
    logger.info("⏳ Waiting for user question (Interrupt)...")
    user_input = interrupt(value="router_node")
    
    # Return Command to route based on input
    if not user_input or user_input.lower() in ["exit", "quit"]:
        return Command(goto=END)
    
    return {"start_message": [HumanMessage(user_input)]}

async def jd_process_node(state: OverallState):
    """
    Acts as the Receptionist. Explains features and asks for request.
    """
    if not state["jd_messages"]:
        messages = [state['start_message'][-1]]
        state["jd_messages"] = messages
    else:
        messages = state['jd_messages']
    # Ensure system prompt is present
    if not isinstance(messages[0], SystemMessage):
        # We prepend system prompt if not there (conceptually)
        # For 'add_messages', we just add it if history is empty
        sys_msg = SystemMessage(content=JD_REQUIREMENTS_GATHER)
        messages = [sys_msg] + messages

    llm = ChatGoogleGenerativeAI(
        model=config.model_name,
        google_api_key=config.google_api_key.get_secret_value(),
        temperature=0.0
    ).bind_tools([submit_jd_requirements])
    
    response = await llm.ainvoke(messages)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "submit_jd_requirements":
            logger.info("🎯 Job description requirement Defined")
            try:
                # Extract args and build the object
                args = tool_call["args"]
                reqs = JobDescriptionRequest(**args)
                
                # Return command to jump to next phase (load files)
                # We also save the reqs to state
                return {
                    "jd_messages": [response], 
                    "jd_reqs": reqs
                }
            except Exception as e:
                # Validation error - send back to agent to fix
                logger.error(f"Validation Error: {e}")
                err_msg = ToolMessage(tool_call_id=tool_call['id'], content=f"Error: {str(e)}")
                return {"intent": [response, err_msg]}
    # If just text, return the response so we can show it to user and wait for input
    print(f"\n🤖 Agent Answer: {parser.invoke(response)}\n")
    return {"jd_messages": [response]}

def jd_input_node(state:OverallState):
    """
    Stops the graph and waits for user input.
    """
    logger.info("⏳ Waiting for user question (Interrupt)...")
    user_input = interrupt(value="jd_node")
    
    # Return Command to route based on input
    if not user_input or user_input.lower() in ["exit", "quit"]:
        return Command(goto=END)
    
    return {"jd_messages": [HumanMessage(user_input)]}

async def jd_writer_node(state: OverallState):
    """
    Generates the Job Description text.
    """
    reqs = state["jd_reqs"]
    logger.info("✍️ Generating Job Description...")
    
    prompt = JD_WRITER_PROMPT.format(
        reqs_json=reqs.model_dump_json(),
    )
    
    llm = ChatGoogleGenerativeAI(
        model=config.model_name,
        google_api_key=config.google_api_key.get_secret_value(),
        temperature=0.7 # Higher temp for creativity
    )
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    text = parser.invoke(response)
    print("\n" + "="*40)
    print("      📄 GENERATED JOB DESCRIPTION")
    print("="*40 + "\n")
    print(text)
    print("\n" + "="*40 + "\n")
    
    return {"final_jd": text}

async def save_results_node(state: OverallState):
    """Saves all evaluated resumes to MongoDB."""
    results = state["evaluated_results"]
    if not results:
        logger.warning("No results to save.")
        return
    
    logger.info(f"💾 Saving {len(results)} candidates to MongoDB...")
    mongo = MongoHandler()
    for res in results:
        await mongo.save_candidate(res)
    return

async def prepare_qa_node(state: OverallState):
    """Extracts DB schema once for the session."""
    logger.info("🔍 Analyzing Database Schema for Q&A...")
    try:
        extractor = ExtractSchema(config.mongo_uri, config.mongo_db_name, config.mongo_collection)
        sample = extractor.get_random_doc()
        structure = extractor.generate_schema(sample) if sample else {}
        return {"db_structure": structure}
    except Exception as e:
        logger.error(f"Schema extraction failed: {e}")
        return {"db_structure": {}}

async def hiring_process_node(state: OverallState):
    """
    Runs the Hiring Agent logic.
    Decides whether to respond with text (ask more questions) or call the tool (finish).
    """
    if not state["hiring_messages"]:
        messages = [state['start_message'][-1]]
        state["hiring_messages"] = messages
    else:
        messages = state['hiring_messages']
    # Ensure system prompt is present
    if not isinstance(messages[0], SystemMessage):
        # We prepend system prompt if not there (conceptually)
        # For 'add_messages', we just add it if history is empty
        sys_msg = SystemMessage(content=HIRING_AGENT_PROMPT)
        messages = [sys_msg] + messages

    # Call LLM
    llm = ChatGoogleGenerativeAI(
        model=config.model_name,
        google_api_key=config.google_api_key.get_secret_value(),
        temperature=0.0
    ).bind_tools([submit_hiring_requirements])
    
    response = await llm.ainvoke(messages)
    
    # Logic: Did the agent call the tool?
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "submit_hiring_requirements":
            logger.info("🎯 Hiring Requirements Collected.")
            try:
                # Extract args and build the object
                args = tool_call["args"]
                reqs = HiringRequirements(**args)
                
                # Return command to jump to next phase (load files)
                # We also save the reqs to state
                return {
                    "hiring_messages": [response], 
                    "hiring_reqs": reqs
                }
            except Exception as e:
                # Validation error - send back to agent to fix
                logger.error(f"Validation Error: {e}")
                err_msg = ToolMessage(tool_call_id=tool_call['id'], content=f"Error: {str(e)}")
                return {"hiring_messages": [response, err_msg]}
    
    # If just text, return the response so we can show it to user and wait for input
    return {"hiring_messages": [response]}

def hiring_input_node(state: OverallState):
    """
    Interrupts execution to get the User's answer to the Recruiter's question.
    """
    # Get the last AI message to display? (Main.py handles display, but we can verify)
    
    # Interrupt!
    user_response = interrupt(value="hiring_input")
    
    # If user wants to quit
    if not user_response or user_response.lower() in ["exit", "quit"]:
        return Command(goto=END)
        
    return {"hiring_messages": [HumanMessage(content=user_response)]}

def qa_input_node(state: OverallState):
    """
    Stops the graph and waits for user input.
    """
    logger.info("⏳ Waiting for user question (Interrupt)...")
    user_input = interrupt(value="qa_input")
    
    # Return Command to route based on input
    if not user_input or user_input.lower() in ["exit", "quit"]:
        return Command(goto=END)
    
    return {"current_question": user_input}

async def qa_process_node(state: OverallState):
    """Generates answer using the ReAct Agent."""
    question = state["current_question"]
    structure = state["db_structure"]
    
    agent = ResumeQAAgent(structure)
    answer = await agent.run(question)
    
    print(f"\n🤖 Agent Answer: {answer}\n")
    return {"qa_answer": answer}

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

# -- CONDITIONAL CHECK FUNCTION --
def should_continue_hiring(state: OverallState):
    """Decides where to go after hiring_process."""
    if state.get("hiring_reqs"):
        return "load_and_shard"
    return "hiring_input"

def should_continue_jd_requirement(state: OverallState):
    """Decides where to go after hiring_process."""
    if state.get("jd_reqs"):
        return "jd_writer"
    return "jd_input"

def define_path(state: OverallState):
    """Decides where to go after hiring_process."""
    if state.get("intent") == "REVIEW":
        return "hiring_process"
    elif state.get("intent") == "WRITE":
        return "jd_process"
    return "router_input"

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

    # Nodes
    workflow_main.add_node("router_process",router_process_node)
    workflow_main.add_node("router_input",router_input_node)
    workflow_main.add_node("hiring_process", hiring_process_node)
    workflow_main.add_node("hiring_input", hiring_input_node)
    workflow_main.add_node("jd_process", jd_process_node)
    workflow_main.add_node("jd_input", jd_input_node)
    workflow_main.add_node("jd_writer", jd_writer_node)
    workflow_main.add_node("load_and_shard", load_and_shard)
    workflow_main.add_node("process_batch_subgraph", workflow_batch.compile())
    workflow_main.add_node("save_results", save_results_node)
    workflow_main.add_node("prepare_qa", prepare_qa_node)
    workflow_main.add_node("qa_input", qa_input_node)
    workflow_main.add_node("qa_process", qa_process_node)
    
    # Edges
    workflow_main.add_edge(START, "router_process")
    workflow_main.add_conditional_edges(
        "router_process",
        define_path,
        ["hiring_process","router_input","jd_process"]
    )
    workflow_main.add_edge("router_input", "router_process")

    workflow_main.add_conditional_edges(
        "hiring_process",
        should_continue_hiring,
        ["load_and_shard", "hiring_input"]
    )
    workflow_main.add_edge("hiring_input", "hiring_process")
    
    workflow_main.add_conditional_edges(
        "jd_process",
        should_continue_jd_requirement,
        ["jd_writer", "jd_input"]
    )
    workflow_main.add_edge("jd_input", "jd_process")
    workflow_main.add_edge("jd_writer", END)

    # Conditional Map
    workflow_main.add_conditional_edges(
        "load_and_shard",
        map_to_batches,
        ["process_batch_subgraph"]
    )
    
    workflow_main.add_edge("process_batch_subgraph", "save_results")
    workflow_main.add_edge("save_results", "prepare_qa")
    workflow_main.add_edge("prepare_qa", "qa_input") # Enters Q&A Loop

    # Q&A Loop
    workflow_main.add_edge("qa_input", "qa_process")
    workflow_main.add_edge("qa_process", "qa_input")

   
    checkpointer = MemorySaver()
    return workflow_main.compile(checkpointer=checkpointer)