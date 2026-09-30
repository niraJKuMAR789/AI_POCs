from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def trace_reducer(existing: list[dict[str, Any]] | None, new: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Append trace steps within a turn; passing None at turn start resets the trace."""
    if new is None:
        return []
    return [*(existing or []), *new]


class AgentState(TypedDict, total=False):
    # Conversation memory (persisted per thread by the checkpointer).
    messages: Annotated[list[AnyMessage], add_messages]
    # Per-turn working state. Contexts are stored as plain dicts so checkpoints stay JSON-serializable.
    question: str
    standalone: str
    route: str
    sub_questions: list[str]
    queries: list[str]
    candidates: list[dict[str, Any]]
    contexts: list[dict[str, Any]]
    rewrites: int
    generation: str
    generation_attempts: int
    feedback: str | None
    grounded: bool
    answers_question: bool
    trace: Annotated[list[dict[str, Any]], trace_reducer]
