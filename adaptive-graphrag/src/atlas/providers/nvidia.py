"""NVIDIA API Catalog / NIM providers (build.nvidia.com or self-hosted NIM).

All three model families (chat, embeddings, reranking) speak NVIDIA's
OpenAI-compatible API, so pointing `ATLAS_NVIDIA_BASE_URL` at a self-hosted NIM
moves the whole stack on-prem without a code change.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from pydantic import ValidationError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from atlas.logging import get_logger
from atlas.providers.base import Message, T

log = get_logger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.S)
_JSON_RE = re.compile(r"\{.*\}", flags=re.S)

_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),
    retry=retry_if_not_exception_type((ValidationError, ValueError)),
)


def _to_lc(messages: list[Message]) -> list[BaseMessage]:
    kinds = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [kinds[m["role"]](content=m["content"]) for m in messages]


def _strip_reasoning(text: str) -> str:
    """Reasoning models may emit <think> blocks inline; callers only want the answer."""
    return _THINK_RE.sub("", text).strip()


def _connection_kwargs(api_key: str | None, base_url: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


class NvidiaLLM:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self.name = model
        self._chat = ChatNVIDIA(
            model=model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **_connection_kwargs(api_key, base_url),
        )

    @_retry
    async def generate(self, messages: list[Message]) -> str:
        result = await self._chat.ainvoke(_to_lc(messages))
        return _strip_reasoning(str(result.content))

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        in_think = False
        async for chunk in self._chat.astream(_to_lc(messages)):
            piece = str(chunk.content)
            # Suppress streamed reasoning traces from reasoning-capable models.
            if "<think>" in piece:
                in_think = True
            if in_think:
                if "</think>" in piece:
                    in_think = False
                    piece = piece.split("</think>", 1)[1]
                else:
                    continue
            if piece:
                yield piece

    async def structured(self, messages: list[Message], schema: type[T]) -> T:
        """Constrained decoding first; fall back to schema-in-prompt JSON parsing.

        NIM enforces the JSON schema server-side (guided decoding) for models that
        support it. Some catalog models don't, so we degrade gracefully instead of
        failing the whole agent run.
        """
        try:
            result = await self._structured_native(messages, schema)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
        except Exception as exc:
            log.warning("structured_output.fallback", model=self.name, schema=schema.__name__, error=str(exc))
        return await self._structured_prompted(messages, schema)

    @_retry
    async def _structured_native(self, messages: list[Message], schema: type[T]) -> Any:
        return await self._chat.with_structured_output(schema).ainvoke(_to_lc(messages))

    @_retry
    async def _structured_prompted(self, messages: list[Message], schema: type[T]) -> T:
        instruction: Message = {
            "role": "user",
            "content": (
                "Respond with ONLY a JSON object that validates against this JSON schema. "
                "No prose, no code fences.\n" + json.dumps(schema.model_json_schema())
            ),
        }
        raw = await self.generate([*messages, instruction])
        match = _JSON_RE.search(raw)
        if not match:
            raise ValueError(f"No JSON object in model output for {schema.__name__}")
        return schema.model_validate_json(match.group(0))


class NvidiaEmbedder:
    def __init__(self, model: str, *, api_key: str | None, base_url: str | None = None) -> None:
        self.name = model
        self._embeddings = NVIDIAEmbeddings(model=model, truncate="END", **_connection_kwargs(api_key, base_url))

    @_retry
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # langchain-nvidia sets input_type="passage" here and "query" below; the
        # asymmetric encoding matters for retrieval quality on NeMo Retriever models.
        return await self._embeddings.aembed_documents(texts)

    @_retry
    async def embed_query(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text)


class NvidiaReranker:
    def __init__(self, model: str, *, api_key: str | None, base_url: str | None = None) -> None:
        self.name = model
        # top_n is set high and results sliced per call: mutating it per request would race.
        self._reranker = NVIDIARerank(model=model, truncate="END", top_n=512, **_connection_kwargs(api_key, base_url))

    @_retry
    async def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        if not texts:
            return []
        docs = [Document(page_content=t, metadata={"idx": i}) for i, t in enumerate(texts)]
        ranked = await self._reranker.acompress_documents(docs, query)
        return [(int(d.metadata["idx"]), float(d.metadata.get("relevance_score", 0.0))) for d in ranked][:top_n]
