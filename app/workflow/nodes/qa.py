import asyncio
from langgraph.graph import END
from langgraph.types import Command, interrupt
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.config.config import config
from app.config.logger import logger
from app.services.mongo_qa import ResumeQAAgent
from app.services.mongo_service import MongoHandler
from app.services.llm_factory import LLMFactory
from app.workflow.state import OverallState
from utils.extract_structure import ExtractSchema
from utils.helper import save_token_cost , candidate_summary
from utils.prompt import TOP_CANDIDATE

mongo_handler = MongoHandler()
parser = StrOutputParser()

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
    session_id = state['session_id']
    agent = ResumeQAAgent(structure , session_id)
    answer = await agent.run(question)
    
    print(f"\n🤖 Agent Answer: {answer}\n")
    return {"qa_answer": answer}

async def top_candidates_node(state: OverallState):
    session_id = state['session_id']
    top_candidates = await mongo_handler.get_top_candidates(3)
    top_candidates_summary = candidate_summary(top_candidates)

    prompt = TOP_CANDIDATE.format(top_candidate_summary=top_candidates_summary)
    llm = LLMFactory.get_model()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    asyncio.create_task(save_token_cost('top_candidates_node', session_id , response))
    answer = parser.invoke(response)

    print(f"\n🤖 Agent Answer: {answer}\n")
    return {"top_candidate": answer}