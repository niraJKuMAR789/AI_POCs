"""OpenAI-compatible chat client: NVIDIA API Catalog, self-hosted NIM, or vLLM (fine-tuned model)."""

from __future__ import annotations

import re

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from quill.llm.base import Message

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class OpenAICompatibleLLM:
    def __init__(
        self, model: str, *, base_url: str, api_key: str | None, temperature: float = 0.0, max_tokens: int = 1024
    ) -> None:
        self.name = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=120)
        self._temperature = temperature
        self._max_tokens = max_tokens

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)),
    )
    async def complete(self, messages: list[Message]) -> str:
        response = await self._client.chat.completions.create(
            model=self.name,
            messages=list(messages),  # type: ignore[arg-type]
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return _THINK_RE.sub("", response.choices[0].message.content or "").strip()
