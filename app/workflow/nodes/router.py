from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.types import Command, interrupt
from langgraph.graph import START , END
from langchain_core.output_parsers import StrOutputParser

from app.config.logger import logger
from app.services.llm_factory import LLMFactory
from app.workflow.llm_tools import AgentTools
from app.workflow.state import OverallState
from utils.prompt import ROUTER_PROMPT

parser = StrOutputParser()

async def router_process_node(state: OverallState):
    """
    Acts as the Receptionist. Explains features and asks for request.
    """
    messages = state["start_message"]
    # Ensure system prompt is present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=ROUTER_PROMPT)] + messages

    llm = LLMFactory.get_model(tools=[AgentTools.router_tool])
    
    response = await llm.ainvoke(messages)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "router_tool":
            logger.info("🎯 Routing Path Defined")
            try:
                args = tool_call["args"]
                # Return intent to route to next phase
                return {"intent": args['path']}
            except Exception as e:
                logger.error(f"Validation Error: {e}")
                err_msg = ToolMessage(tool_call_id=tool_call['id'], content=f"Error: {str(e)}")
                return {"start_message": [response, err_msg]}

    text = parser.invoke(response)        
    print(f"\n🤖 Agent Answer: {text}\n")     
    return {"start_message": [response]}

def router_input_node(state: OverallState):
    """
    Stops the graph and waits for user input.
    """
    user_input = interrupt(value="router_node")
    
    if not user_input or str(user_input).lower() in ["exit", "quit"]:
        return Command(goto=END)
    
    return {"start_message": [HumanMessage(content=user_input)]}