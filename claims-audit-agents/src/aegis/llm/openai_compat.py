"""OpenAI-compatible client for NVIDIA NIM (hosted or self-hosted) with JSON-schema outputs and vision."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from aegis.llm.base import Message, T

_THINK = re.compile(r"<think>.*?</think>", re.S)
_JSON = re.compile(r"\{.*\}", re.S)
_transient = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.5, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)),
)


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None,
        guided_json: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.name = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=180)
        self._guided = guided_json
        self._temperature = temperature
        self._max_tokens = max_tokens

    @_transient
    async def _create(self, messages: list[Message], extra_body: dict[str, Any] | None) -> str:
        response = await self._client.chat.completions.create(
            model=self.name,
            messages=list(messages),  # type: ignore[arg-type]
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            extra_body=extra_body,
        )
        return _THINK.sub("", response.choices[0].message.content or "").strip()

    async def complete_json(self, messages: list[Message], schema: type[T]) -> T:
        """NIM guided decoding (nvext.guided_json) when available, then prompt-and-validate with one repair."""
        json_schema = schema.model_json_schema()
        instruction: Message = {
            "role": "user",
            "content": "Respond with only a JSON object matching this schema:\n" + json.dumps(json_schema),
        }
        attempt = [*messages, instruction]
        extra = {"nvext": {"guided_json": json_schema}} if self._guided else None
        last_error = ""
        for _ in range(2):
            raw = await self._create(attempt, extra)
            match = _JSON.search(raw)
            try:
                return schema.model_validate_json(match.group(0) if match else raw)
            except ValidationError as exc:
                last_error = str(exc)[:1500]
                attempt = [
                    *attempt,
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That JSON was invalid: {last_error}. Return corrected JSON."},
                ]
        raise ValueError(f"{self.name} did not return valid {schema.__name__}: {last_error}")
