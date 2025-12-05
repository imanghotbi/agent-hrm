from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.config import config
from typing import Optional

class LLMFactory:
    @staticmethod
    def get_model(temperature: float = 0.0, thinking_budget:Optional[int] = None,
                top_p:Optional[float] = None, max_output_tokens:Optional[int] = None,
                structured_output=None,tools:list=[]):
        llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key.get_secret_value(),
            temperature=temperature,
            max_output_tokens=max_output_tokens or config.max_tokens,
            top_p= top_p or config.top_p,
            thinking_budget= thinking_budget or config.thinking_budget
        )
        if structured_output:
            return llm.with_structured_output(structured_output)
        if tools:
            return llm.bind_tools(tools)
        return llm