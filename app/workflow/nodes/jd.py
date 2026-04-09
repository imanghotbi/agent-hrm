import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.types import Command, interrupt
from langgraph.graph import  END
from langchain_core.output_parsers import StrOutputParser

from app.config.logger import logger
from app.services.llm_factory import LLMFactory
from app.workflow.llm_tools import AgentTools
from app.schemas.job_description import JobDescriptionRequest
from app.workflow.state import OverallState
from utils.prompt import JD_REQUIREMENTS_GATHER, JD_WRITER_PROMPT
from utils.helper import save_token_cost
parser = StrOutputParser()

async def jd_process_node(state: OverallState):
    """
    Interviews the user to gather JD requirements.
    """
    try:
        messages = state.get("jd_messages")
        session_id = state.get("session_id", "unknown")
        if not messages:
            start_messages = state.get("start_message") or []
            if start_messages:
                messages = [start_messages[-1]]
            else:
                messages = [HumanMessage(content="لطفا نیازمندی‌های نوشتن آگهی شغلی را بفرمایید.")]
        
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=JD_REQUIREMENTS_GATHER)] + messages

        response = await LLMFactory.ainvoke(
            messages,
            tools=[AgentTools.submit_jd_requirements],
        )
        asyncio.create_task(save_token_cost("jd_process_node", session_id, response))

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            if tool_call.get("name") == "submit_jd_requirements":
                logger.info("🎯 Job description requirement Defined")
                try:
                    args = tool_call.get("args", {})
                    reqs = JobDescriptionRequest(**args)
                    return {
                        "jd_messages": [response], 
                        "jd_reqs": reqs
                    }
                except Exception as e:
                    logger.error(f"Validation Error: {e}")
                    err_msg = ToolMessage(tool_call_id=tool_call.get('id', 'unknown_tool_call'), content=f"Error: {str(e)}")
                    return {"jd_messages": [response, err_msg]}
    
        text = parser.invoke(response)        
        print(f"\n🤖 Agent Answer: {text}\n")            
        return {"jd_messages": [response]}
    except Exception as exc:
        logger.exception(f"jd_process_node failed; continuing to input: {exc}")
        fallback = HumanMessage(content="I hit a temporary error while preparing JD requirements. Please try again.")
        return {"jd_messages": [fallback]}

def jd_input_node(state: OverallState):
    """
    Stops the graph and waits for user input.
    """
    user_input = interrupt(value="jd_node")
    
    if not user_input or str(user_input).lower() in ["exit", "quit"]:
        return Command(goto=END)
    
    return {"jd_messages": [HumanMessage(content=user_input)]}

async def jd_writer_node(state: OverallState):
    """
    Generates the Job Description text.
    """
    try:
        reqs = state.get("jd_reqs")
        session_id = state.get("session_id", "unknown")
        logger.info("✍️ Generating Job Description...")
        if reqs is None:
            logger.warning("jd_writer_node skipped: missing jd_reqs.")
            return {"final_jd": ""}

        prompt = JD_WRITER_PROMPT.format(reqs_json=reqs.model_dump_json())
        
        # Higher temperature for creativity
        response = await LLMFactory.ainvoke(
            [HumanMessage(content=prompt)],
            temperature=0.7,
        )
        asyncio.create_task(save_token_cost("jd_writer_node", session_id, response))
        text = parser.invoke(response)
        
        print("\n" + "="*40)
        print("      📄 GENERATED JOB DESCRIPTION")
        print("="*40 + "\n")
        print(text)
        print("\n" + "="*40 + "\n")
        
        return {"final_jd": text}
    except Exception as exc:
        logger.exception(f"jd_writer_node failed: {exc}")
        return {"final_jd": ""}
