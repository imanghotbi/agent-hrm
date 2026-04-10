from typing import Optional, Sequence, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from app.config.logger import logger
from app.services.llm_factory import LLMFactory

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOutputHandler:
    """
    Structured-output invoker with staged recovery:
    1) Standard `with_structured_output` call.
    2) Fix prompt that asks for raw JSON.
    3) Strict fallback prompt that asks for JSON only.
    """

    def __init__(
        self,
        schema: type[SchemaT],
        max_retries: int = 3,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        self.schema = schema
        self.max_retries = max(1, max_retries)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens

    async def ainvoke(
        self,
        input_data: str | Sequence[BaseMessage],
        fallback_prompt: Optional[str] = None,
    ) -> tuple[SchemaT, Optional[BaseMessage]]:
        messages = self._normalize_messages(input_data)
        llm = LLMFactory.get_model(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            model_name=self.model_name,
        )

        try:
            # 1) Primary attempt: tool/function calling structured output.
            logger.info("Attempting structured output for %s...", self.schema.__name__)
            chain = llm.with_structured_output(self.schema)
            out = await chain.ainvoke(messages)

            if out is None:
                raise ValueError("LLM returned None for structured output")
            return out, None

        except Exception as primary_error:
            logger.warning(
                "Structured output failed (%s: %s). Attempting recovery...",
                type(primary_error).__name__,
                primary_error,
            )

            prompt_text = self._prompt_to_text(messages)
            schema_json = self._schema_json()

            # 2) Recovery attempt: ask for valid raw JSON.
            fix_prompt = f"""
You failed to provide the correct structured output.

TASK: Return ONLY valid JSON matching this schema:
{schema_json}

RULES:
- Do not output markdown code blocks (```json ... ```).
- Just the raw JSON string.
- Use null when fields are unknown.

CONTEXT:
{prompt_text}
"""

            try:
                raw_msg = await llm.ainvoke([HumanMessage(content=fix_prompt)])
                raw = self._extract_raw_content(raw_msg)
                raw_cleaned = self._strip_markdown_json(raw)
                out = self._validate_json(raw_cleaned)
                logger.info("Structured output recovered using Fix Prompt.")
                return out, raw_msg if isinstance(raw_msg, BaseMessage) else AIMessage(content=raw)

            except (ValidationError, Exception) as recovery_error:
                logger.warning(
                    "Recovery attempt 1 failed (%s). Attempting fallback...",
                    recovery_error,
                )

                # 3) Fallback attempt: stricter JSON-only prompt.
                if fallback_prompt is None:
                    fallback_prompt = f"""
CRITICAL FAILURE RECOVERY.
Return ONLY valid JSON matching this schema:
{schema_json}
"""

                final_prompt = f"{fallback_prompt}\n\nCONTEXT:\n{prompt_text}"
                raw2_msg = await llm.ainvoke([HumanMessage(content=final_prompt)])
                raw2 = self._extract_raw_content(raw2_msg)
                raw2_cleaned = self._strip_markdown_json(raw2)
                out = self._validate_json(raw2_cleaned)
                logger.info("Structured output recovered using Fallback Prompt.")
                return out, raw2_msg if isinstance(raw2_msg, BaseMessage) else AIMessage(content=raw2)

    @staticmethod
    def _normalize_messages(input_data: str | Sequence[BaseMessage]) -> list[BaseMessage]:
        if isinstance(input_data, str):
            return [HumanMessage(content=input_data)]
        return list(input_data)

    @staticmethod
    def _extract_raw_content(raw_message: Optional[BaseMessage]) -> str:
        if raw_message is None:
            return ""
        content = getattr(raw_message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _strip_markdown_json(raw_text: str) -> str:
        return raw_text.replace("```json", "").replace("```", "").strip()

    def _validate_json(self, raw_text: str) -> SchemaT:
        try:
            if hasattr(self.schema, "model_validate_json"):
                return self.schema.model_validate_json(raw_text)
            return self.schema.parse_raw(raw_text)
        except Exception as e:
            raise ValueError(f"Failed to validate JSON response: {e}") from e

    @staticmethod
    def _prompt_to_text(prompt_value: str | Sequence[BaseMessage]) -> str:
        if isinstance(prompt_value, str):
            return prompt_value

        parts: list[str] = []
        for msg in prompt_value:
            role = msg.__class__.__name__
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                list_parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        list_parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            list_parts.append(text)
                if list_parts:
                    parts.append(f"{role}: {' '.join(list_parts)}")
        return "\n".join(parts).strip()

    def _schema_json(self) -> dict:
        if hasattr(self.schema, "model_json_schema"):
            return self.schema.model_json_schema()
        return self.schema.schema()
