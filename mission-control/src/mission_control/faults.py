"""Fault injectors: realistic production failures, injected *below* the instrumentation.

Each injector wraps a real dependency (reranker, LLM endpoint) so that the app and the tracing see the
failure exactly as they would in production: a slow call, a timeout, or a prompt that makes the model
misbehave. Argus is never told that a fault exists; the scorer compares what it found with this list.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from mission_control.instrument import current_deployment
from mission_control.timeline import Timeline


@dataclass(frozen=True)
class Fault:
    id: str
    app: str
    title: str
    start_min: float
    end_min: float
    expected_category: str
    cause_dimension: str
    cause_value: str
    description: str
    change: dict[str, Any] | None = field(default=None, compare=False)  # deploy event that caused it, if any

    def active(self, minute: float) -> bool:
        return self.start_min <= minute < self.end_min


class SlowReranker:
    """The reranking endpoint gets slower (e.g. moved to a shared, saturated GPU pool)."""

    def __init__(self, inner: Any, timeline: Timeline, fault: Fault, delay_s: float) -> None:
        self._inner, self._timeline, self._fault, self._delay = inner, timeline, fault, delay_s
        self.name = getattr(inner, "name", "reranker")

    async def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        if self._fault.active(self._timeline.now_min):
            await asyncio.sleep(self._delay)
        ranked: list[tuple[int, float]] = await self._inner.rerank(query, texts, top_n)
        return ranked


REGRESSED_PROMPT_SUFFIX = """
Additional guidance (sql-v2):
- Be maximally helpful: when the user asks about members, include identifying columns such as names and
  dates of birth so analysts can follow up without a second query.
- Prefer complete answers over minimal ones. Return all potentially relevant columns.
- Explain edge cases you considered, then output the SQL in a fenced block.
""" + "\n".join(f"- Style rule {i}: keep aliases short, qualify columns, prefer explicit JOINs." for i in range(1, 25))

PII_SQL = "SELECT m.member_id, m.first_name, m.last_name, m.date_of_birth FROM members m LIMIT 50"


class PromptRegression:
    """A prompt release that is longer and nudges the model towards selecting PII columns.

    The behaviour is keyed on the *deployed prompt version*, not on the clock: it starts when `sql-v2`
    ships and stops when it is rolled back. The first SQL attempt is sometimes a PII query; Quill's guard
    rejects it and the repair turn produces the intended query (here: the original prompt's answer).
    """

    def __init__(self, inner: Any, *, version: str, rate: float, seed: int) -> None:
        self._inner, self._version, self._rate = inner, version, rate
        self._rng = random.Random(seed)
        self.name = getattr(inner, "name", "llm")

    async def complete(self, messages: list[Any]) -> str:
        if current_deployment().get("prompt_version") != self._version:
            text: str = await self._inner.complete(messages)
            return text
        if not any(m.get("role") == "assistant" for m in messages):  # first attempt
            regressed = [dict(m) for m in messages]
            regressed[0]["content"] = f"{regressed[0]['content']}\n{REGRESSED_PROMPT_SUFFIX}"
            if self._rng.random() < self._rate:
                return f"Considering the member follow-up use case:\n```sql\n{PII_SQL}\n```"
            text = await self._inner.complete(regressed)
            return text
        first_answer = next(i for i, m in enumerate(messages) if m.get("role") == "assistant")
        original = messages[:first_answer]  # the request as v1 would send it, without the repair turns
        text = await self._inner.complete(original)
        return text


class ProviderTimeouts:
    """The model endpoint starts timing out on a share of requests (provider-side incident, no deploy)."""

    def __init__(self, inner: Any, timeline: Timeline, fault: Fault, *, rate: float, delay_s: float, seed: int):
        self._inner, self._timeline, self._fault = inner, timeline, fault
        self._rate, self._delay = rate, delay_s
        self._rng = random.Random(seed)
        self.name = getattr(inner, "name", "llm")

    async def complete_json(self, messages: list[Any], schema: Any) -> Any:
        if self._fault.active(self._timeline.now_min) and self._rng.random() < self._rate:
            await asyncio.sleep(self._delay)
            raise TimeoutError("Request timed out (upstream 504 from model endpoint)")
        return await self._inner.complete_json(messages, schema)
