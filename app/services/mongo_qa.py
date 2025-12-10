import json
from typing import Annotated, TypedDict, List
import asyncio

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from app.services.llm_factory import LLMFactory
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser

from app.config.logger import logger
from app.services.mongo_service import MongoHandler
from utils.prompt import QA_AGENT_SYSTEM_PROMPT
from utils.helper import save_token_cost

# -- AGENT STATE --
class QAAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class ResumeQAAgent:
    def __init__(self, db_structure: dict, session_id:str):
        self.db_structure = db_structure
        self.mongo = MongoHandler()
        self.parser = StrOutputParser()
        self.session_id = session_id
        # -- DEFINE TOOLS --
        @tool
        async def search_database(query: str, projection: str = None):
            """
            Executes a MongoDB find query. 
            Input MUST be a valid JSON string.
            Example: '{"final_score": {"$gt": 80}}' 
            and other example on system on system prompt
            """
            try:
                # 1. Clean and Parse JSON
                # Sometimes LLM wraps json in ```json ... ```
                query_clean = query.strip().replace("```json", "").replace("```", "")
                proj_clean = projection.strip().replace("```json", "").replace("```", "") if projection else None
                
                query_dict = json.loads(query_clean)
                proj_dict = json.loads(proj_clean) if proj_clean else None
                
                logger.info(f"🔍 Agent Executing Query: {query_dict}")
                
                # 2. Execute
                results = await self.mongo.execute_raw_query(query_dict, proj_dict)
                
                if not results:
                    return "Database returned: No documents found matching this query."
                
                return f"Database Results: {str(results)}"
                
            except json.JSONDecodeError:
                return "Error: Invalid JSON format. Please correct the query syntax."
            except Exception as e:
                return f"Database Error: {str(e)}"

        self.tools = [search_database]
        
        # -- INITIALIZE LLM --
        self.llm = LLMFactory().get_model(tools=self.tools)


        # -- BUILD INTERNAL GRAPH (Thought -> Action Loop) --
        workflow = StateGraph(QAAgentState)
        
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")
        
        self.graph = workflow.compile()

    async def call_model(self, state: QAAgentState):
        messages = state["messages"]
        # Inject system prompt at the start
        sys_msg = SystemMessage(content=QA_AGENT_SYSTEM_PROMPT.format(structure=self.db_structure))
        # We prepend it to the context sent to API
        response = await self.llm.ainvoke([sys_msg] + messages)
        asyncio.create_task(save_token_cost("qa_process_node", self.session_id, response))
        return {"messages": [response]}

    async def run(self, user_question: str) -> str:
        """Entry point for the agent."""
        inputs = {"messages": [HumanMessage(content=user_question)]}
        result = await self.graph.ainvoke(inputs)
        return self.parser.invoke(result["messages"][-1])