from langgraph.types import Command, interrupt
from langgraph.graph import END

from app.config.config import config
from app.config.logger import logger
from app.services.mongo_qa import ResumeQAAgent
from app.workflow.state import OverallState
from utils.extract_structure import ExtractSchema

async def prepare_qa_node(state: OverallState):
    """
    Extracts DB schema once for the session.
    """
    logger.info("🔍 Analyzing Database Schema for Q&A...")
    try:
        extractor = ExtractSchema(config.mongo_uri, config.mongo_db_name, config.mongo_collection)
        sample = extractor.get_random_doc()
        structure = extractor.generate_schema(sample) if sample else {}
        return {"db_structure": structure}
    except Exception as e:
        logger.error(f"Schema extraction failed: {e}")
        return {"db_structure": {}}

def qa_input_node(state: OverallState):
    """
    Stops the graph and waits for user input.
    """
    user_input = interrupt(value="qa_input")
    
    if not user_input or str(user_input).lower() in ["exit", "quit"]:
        return Command(goto=END)
    
    return {"current_question": user_input}

async def qa_process_node(state: OverallState):
    """
    Generates answer using the ResumeQAAgent (ReAct).
    """
    question = state["current_question"]
    structure = state["db_structure"]
    
    # We instantiate the agent here. 
    # ResumeQAAgent from src/matcher.py is robust enough to act as a service.
    agent = ResumeQAAgent(structure)
    answer = await agent.run(question)
    
    print(f"\n🤖 Agent Answer: {answer}\n")
    return {"qa_answer": answer}