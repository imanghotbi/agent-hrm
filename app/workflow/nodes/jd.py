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

parser = StrOutputParser()

async def jd_process_node(state: OverallState):
    """
    Interviews the user to gather JD requirements.
    """
    messages = state.get("jd_messages")
    if not messages:
        messages = [state['start_message'][-1]]
        state["jd_messages"] = messages
        
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=JD_REQUIREMENTS_GATHER)] + messages

    llm = LLMFactory.get_model(tools=[AgentTools.submit_jd_requirements])
    response = await llm.ainvoke(messages)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "submit_jd_requirements":
            logger.info("🎯 Job description requirement Defined")
            try:
                args = tool_call["args"]
                reqs = JobDescriptionRequest(**args)
                return {
                    "jd_messages": [response], 
                    "jd_reqs": reqs
                }
            except Exception as e:
                logger.error(f"Validation Error: {e}")
                err_msg = ToolMessage(tool_call_id=tool_call['id'], content=f"Error: {str(e)}")
                return {"jd_messages": [response, err_msg]}
    
    text = parser.invoke(response)        
    print(f"\n🤖 Agent Answer: {text}\n")            
    return {"jd_messages": [response]}

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
    reqs = state["jd_reqs"]
    logger.info("✍️ Generating Job Description...")
    
    prompt = JD_WRITER_PROMPT.format(reqs_json=reqs.model_dump_json())
    
    # Higher temperature for creativity
    llm = LLMFactory.get_model(temperature=0.7)
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    text = parser.invoke(response)
    
    print("\n" + "="*40)
    print("      📄 GENERATED JOB DESCRIPTION")
    print("="*40 + "\n")
    print(text)
    print("\n" + "="*40 + "\n")
    
    return {"final_jd": text}