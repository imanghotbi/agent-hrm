from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from app.config.config import config
from typing import Optional, Sequence, Union


class LLMFactory:
    @staticmethod
    def get_model(
        temperature: float = 0.0,
        thinking_budget: Optional[int] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        structured_output=None,
        include_raw: bool = False,
        tools: Optional[list] = None,
        model_name: Optional[str] = None,
    ):
        resolved_model = model_name or (
            config.structured_model_name if structured_output else config.model_name
        )
        llm = ChatOpenAI(
            model=resolved_model,
            base_url=config.base_url,
            api_key=config.api_key.get_secret_value(),
            temperature=temperature,
            top_p= top_p or config.top_p,
            max_tokens= max_output_tokens or config.max_tokens,
            reasoning_effort="minimal"
        )
        if structured_output:
            return llm.with_structured_output(structured_output, include_raw=include_raw)
        if tools:
            return llm.bind_tools(tools)
        return llm

    @staticmethod
    async def ainvoke(
        input_data: Union[str, Sequence[BaseMessage]],
        temperature: float = 0.0,
        thinking_budget: Optional[int] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        structured_output=None,
        include_raw: bool = False,
        tools: Optional[list] = None,
        model_name: Optional[str] = None,
    ):
        llm = LLMFactory.get_model(
            temperature=temperature,
            thinking_budget=thinking_budget,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            structured_output=structured_output,
            include_raw=include_raw,
            tools=tools,
            model_name=model_name,
        )

        if isinstance(input_data, str):
            return await llm.ainvoke([HumanMessage(content=input_data)])
        return await llm.ainvoke(list(input_data))