import json
from typing import Annotated, TypedDict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from src.config import config, logger
from src.database import MongoHandler
from utils.prompt import QA_AGENT_SYSTEM_PROMPT


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class ResumeQAAgent:
    def __init__(self, db_structure: dict):
        self.db_structure = db_structure
        self.mongo = MongoHandler()
        
        # -- DEFINE TOOLS --
        # We define the tool inside to bind it to the specific mongo instance
        @tool
        async def search_database(query: str, projection: str = None):
            """
            Executes a MongoDB find query.
            Args:
                query: JSON string of the query filter (e.g. '{"final_score": {"$gt": 80}}').
                projection: JSON string of fields to return (optional).
            """
            try:
                # 1. Parse JSON inputs from the LLM
                query_dict = json.loads(query)
                proj_dict = json.loads(projection) if projection else None
                
                logger.info(f"🔍 Agent Executing Query: {query_dict}")
                
                # 2. Execute via MongoHandler
                results = await self.mongo.execute_raw_query(query_dict, proj_dict)
                
                if not results:
                    return "Database returned: No documents found."
                
                return f"Database Results: {str(results)}"
                
            except json.JSONDecodeError:
                return "Error: Invalid JSON format in query. Please fix quotes and brackets."
            except Exception as e:
                return f"Database Error: {str(e)}"

        self.tools = [search_database]
        
        # -- INITIALIZE LLM --
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key.get_secret_value(),
            temperature=0
        ).bind_tools(self.tools)

        # -- BUILD GRAPH --
        workflow = StateGraph(AgentState)
        
        # Nodes
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Edges
        workflow.add_edge(START, "agent")
        # 'tools_condition' checks if the LLM wants to call a tool or finish
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent") # Loop back to agent after tool usage
        
        self.graph = workflow.compile()

    async def call_model(self, state: AgentState):
        messages = state["messages"]
        
        # Inject System Prompt if it's the start of conversation
        # (Or ensure it's always the first message context)
        sys_msg = QA_AGENT_SYSTEM_PROMPT.format(structure=self.db_structure)
        
        # We prepend the system message to the current history for the API call
        # (Note: We don't necessarily add it to 'state' history to keep history clean, 
        # or we can check if it exists. For simplicity, we prepend here.)
        api_messages = [SystemMessage(content=sys_msg)] + messages
        
        response = await self.llm.ainvoke(api_messages)
        return {"messages": [response]}

    async def chat(self, user_question: str) -> str:
        """
        Entry point for the Q&A loop.
        """
        try:
            inputs = {"messages": [HumanMessage(content=user_question)]}
            
            # The graph will run: Agent -> Tool? -> Agent -> Final Answer
            result = await self.graph.ainvoke(inputs)
            
            # Extract the final response text
            final_msg = result["messages"][-1]
            str_parser = StrOutputParser()
            result = str_parser.invoke(final_msg)
            return result
            
        except Exception as e:
            logger.error(f"Agent Loop Failed: {e}")
            return "متاسفانه مشکلی در پردازش درخواست شما پیش آمد."