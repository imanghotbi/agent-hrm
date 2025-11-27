from typing import List,Union,Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from src.config import config, logger
from src.schema import HiringRequirements , SeniorityLevel
from utils.prompt import HIRING_AGENT_PROMPT


class HiringAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0.0
        ).bind_tools([self.submit_hiring_requirements])
        
        self.messages = [SystemMessage(content=HIRING_AGENT_PROMPT)]
        self.final_requirements: Union[HiringRequirements, None] = None
        self.is_complete = False

    @tool
    def submit_hiring_requirements(role_title: str, 
                                   seniority: SeniorityLevel, 
                                   essential_hard_skills: List[str], 
                                   military_service_required: bool, 
                                   min_experience_years: int, 
                                   education_level:str,
                                   nice_to_have_skills:Optional[List[str]],
                                   language_proficiency:Optional[str],
                                   **kwargs):
        """
        Call this tool ONLY when you have gathered ALL necessary requirements from the user.
        """
        # This function acts as a dummy to validate inputs, the real data capture happens in the run loop
        return "Requirements captured."

    async def run_turn(self, user_input: str) -> str:
        """
        Processes one turn of conversation.
        Returns the agent's response text.
        """
        self.messages.append(HumanMessage(content=user_input))
        
        # Invoke Model
        response = await self.llm.ainvoke(self.messages)
        
        # Check if the model decided to call the tool (meaning it's done)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "submit_hiring_requirements":
                    logger.info("🎯 Agent decided requirements are complete.")
                    
                    # Parse the arguments provided by the LLM into our Pydantic model
                    try:
                        # We merge args with any extra kwargs the model might have sent
                        args = tool_call["args"]
                        self.final_requirements = HiringRequirements(**args)
                        self.is_complete = True
                        return "Thank you. I have recorded the requirements. Starting the resume processing workflow now..."
                    except Exception as e:
                        logger.error(f"Schema Validation Error: {e}")
                        # If validation fails, we feed the error back to the model to correct itself
                        error_msg = f"Error in parameters: {str(e)}. Please ask the user to clarify or fix the parameters."
                        self.messages.append(ToolMessage(tool_call_id=tool_call['id'], content=error_msg))
                        # Recursively try again with the error context
                        retry_response = await self.llm.ainvoke(self.messages)
                        self.messages.append(retry_response)
                        return retry_response.content

        # Normal conversation response
        self.messages.append(response)
        return response.content