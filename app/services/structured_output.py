import json
import re
from json import JSONDecodeError
from typing import Optional, Sequence, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from app.config.logger import logger
from app.services.llm_factory import LLMFactory

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOutputHandler:
    """
    Resilient structured-output invoker:
    - Uses include_raw=True so raw model output is never lost.
    - Retries with validation-error feedback.
    - Falls back to manual JSON extraction + schema validation.
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
    ) -> tuple[SchemaT, Optional[BaseMessage]]:
        messages = self._normalize_messages(input_data)
        structured_llm = LLMFactory.get_model(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            structured_output=self.schema,
            include_raw=True,
            model_name=self.model_name,
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            result = await structured_llm.ainvoke(messages)
            parsed = result.get("parsed")
            raw_message = result.get("raw")
            parsing_error = result.get("parsing_error")

            if parsed is not None:
                return parsed, raw_message

            raw_content = self._extract_raw_content(raw_message)
            manual = self._parse_raw_with_fallback(raw_content)
            if manual is not None:
                logger.warning(
                    "Structured output fallback parser recovered response on attempt %s.",
                    attempt,
                )
                return manual, raw_message

            if isinstance(parsing_error, Exception):
                last_error = parsing_error
            else:
                last_error = ValueError("Structured output parse failed without error details.")

            if attempt < self.max_retries:
                error_msg = str(last_error)
                logger.warning(
                    "Structured output parsing failed (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    error_msg,
                )
                messages.append(AIMessage(content=raw_content))
                messages.append(
                    HumanMessage(
                        content=(
                            "Your previous response failed schema validation.\n"
                            f"Validation error: {error_msg}\n"
                            "Return only valid JSON that strictly matches the required schema."
                        )
                    )
                )

        raise ValueError(
            f"Failed to parse structured output after {self.max_retries} attempts: {last_error}"
        )

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

    def _parse_raw_with_fallback(self, raw_text: str) -> Optional[SchemaT]:
        if not raw_text:
            return None
        candidate = self._extract_json_from_text(raw_text)
        if candidate is None:
            return None
        try:
            if hasattr(self.schema, "model_validate"):
                return self.schema.model_validate(candidate)
            return self.schema.parse_obj(candidate)
        except Exception:
            return None

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[dict]:
        cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
        if not cleaned:
            return None

        direct = StructuredOutputHandler._try_load_json(cleaned)
        if direct is not None:
            return direct

        decoder = json.JSONDecoder()
        for idx, ch in enumerate(cleaned):
            if ch not in "{[":
                continue
            try:
                payload, _ = decoder.raw_decode(cleaned[idx:])
                if isinstance(payload, dict):
                    return payload
            except JSONDecodeError:
                continue
        return None

    @staticmethod
    def _try_load_json(text: str) -> Optional[dict]:
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except JSONDecodeError:
            return None
        return None
