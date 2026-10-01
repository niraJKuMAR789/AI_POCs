from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from typing_extensions import TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@runtime_checkable
class LLM(Protocol):
    name: str

    async def complete(self, messages: list[Message]) -> str: ...
