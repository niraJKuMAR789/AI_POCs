from __future__ import annotations

from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel
from typing_extensions import TypedDict

T = TypeVar("T", bound=BaseModel)


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[dict[str, Any]]


@runtime_checkable
class LLM(Protocol):
    name: str

    async def complete_json(self, messages: list[Message], schema: type[T]) -> T: ...
