"""Provider protocols. The agent depends only on these, never on a vendor SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel
from typing_extensions import TypedDict

T = TypeVar("T", bound=BaseModel)


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@runtime_checkable
class LLM(Protocol):
    name: str

    async def generate(self, messages: list[Message]) -> str: ...

    def stream(self, messages: list[Message]) -> AsyncIterator[str]: ...

    async def structured(self, messages: list[Message], schema: type[T]) -> T: ...


@runtime_checkable
class Embedder(Protocol):
    name: str

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        """Return (index into `texts`, relevance score) pairs, best first."""
        ...
