import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.types import Command, interrupt
from langgraph.graph import  END
from langchain_core.output_parsers import StrOutputParser

from app.config.logger import logger
from app.services.llm_factory import LLMFactory
from app.workflow.llm_tools import AgentTools
from app.schemas.hiring import HiringRequirements
from app.workflow.state import OverallState
from utils.prompt import HIRING_AGENT_PROMPT
from utils.extract_structure import save_token_cost

parser = StrOutputParser()

async def hiring_process_node(state: OverallState):
    """
    Runs the Hiring Agent logic.
    Decides whether to respond with text (ask more questions) or call the tool (finish).
    """
    # Get history or initialize
    messages = state.get("hiring_messages")
    session_id = state["session_id"]
    if not messages:
        # Carry over the last message from the router phase as context
        messages = [state['start_message'][-1]]
        state["hiring_messages"] = messages
    
    # Ensure system prompt is present
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=HIRING_AGENT_PROMPT)] + messages

    # Call LLM
    llm = LLMFactory.get_model(tools=[AgentTools.submit_hiring_requirements])
    response = await llm.ainvoke(messages)
    asyncio.create_task(save_token_cost("hiring_process_node", session_id, response))
    
    # Check if tool called
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "submit_hiring_requirements":
            logger.info("🎯 Hiring Requirements Collected.")
            try:
                args = tool_call["args"]
                reqs = HiringRequirements(**args)
                return {
                    "hiring_messages": [response], 
                    "hiring_reqs": reqs
                }
            except Exception as e:
                logger.error(f"Validation Error: {e}")
                err_msg = ToolMessage(tool_call_id=tool_call['id'], content=f"Error: {str(e)}")
                return {"hiring_messages": [response, err_msg]}
            
    text = parser.invoke(response)        
    print(f"\n🤖 Agent Answer: {text}\n")     
    return {"hiring_messages": [response]}

def hiring_input_node(state: OverallState):
    """
    Interrupts execution to get the User's answer.
    """
    user_response = interrupt(value="hiring_input")
    
    if not user_response or str(user_response).lower() in ["exit", "quit"]:
        return Command(goto=END)
        
    return {"hiring_messages": [HumanMessage(content=user_response)]}