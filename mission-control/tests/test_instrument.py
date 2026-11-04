"""Unit tests: replay clock, instrumentation proxies and fault injectors (no app boot)."""

from __future__ import annotations

import time

import pytest
from argus import semconv
from argus.sdk.exporter import InMemoryExporter
from argus.sdk.tracer import Tracer, use_tracer
from pydantic import BaseModel

from mission_control.drill import _deployment_at, default_scenario
from mission_control.faults import PII_SQL, Fault, PromptRegression, ProviderTimeouts
from mission_control.instrument import TracedLLM, agent_span, deployment, estimate_tokens
from mission_control.timeline import MINUTE_NS, ShiftingExporter, Timeline


class Verdict(BaseModel):
    ok: bool


class FakeLLM:
    name = "fake"

    def __init__(self) -> None:
        self.seen: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.seen.append(messages)
        return "```sql\nSELECT 1\n```"

    async def complete_json(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> BaseModel:
        return schema(ok=True)


def _tracer() -> tuple[Tracer, InMemoryExporter]:
    sink = InMemoryExporter()
    return Tracer("test", ShiftingExporter(sink)), sink


async def test_spans_are_moved_onto_the_virtual_timeline_without_changing_durations() -> None:
    tracer, sink = _tracer()
    timeline = Timeline(start_ns=1_000 * MINUTE_NS)
    with timeline.request_at(42.0), use_tracer(tracer), agent_span("app", "q"):
        time.sleep(0.01)
    ((record, _),) = sink.spans
    assert abs(record.start_ns - timeline.at(42.0)) < 50 * 10**6  # starts at virtual minute 42
    assert 9 <= record.duration_ms < 500  # the real duration is preserved


async def test_traced_llm_emits_genai_chat_span_with_estimated_tokens_and_deployment() -> None:
    tracer, sink = _tracer()
    llm = TracedLLM(FakeLLM(), model="nvidia/nemotron-3-super-120b-a12b", system="nvidia_nim")
    token = deployment.set({"prompt_version": "sql-v1", "agent_version": "quill-2.1.0"})
    try:
        with use_tracer(tracer), agent_span("quill", "question"):
            await llm.complete([{"role": "user", "content": "x" * 400}])
            await llm.complete_json([{"role": "user", "content": "y"}], Verdict)
    finally:
        deployment.reset(token)
    spans = {r.attributes[semconv.SPAN_KIND]: r for r, _ in sink.spans}
    chat = next(r for r, _ in sink.spans if r.attributes.get("argus.llm.purpose") == "complete")
    assert chat.name == "chat nvidia/nemotron-3-super-120b-a12b"
    assert chat.attributes[semconv.INPUT_TOKENS] == estimate_tokens("user: " + "x" * 400)
    assert chat.attributes[semconv.PROMPT_VERSION] == "sql-v1"
    assert chat.parent_span_id == spans["agent"].span_id
    assert any(r.attributes.get("argus.llm.purpose") == "Verdict" for r, _ in sink.spans)


async def test_prompt_regression_only_affects_the_deployed_version_and_repairs_cleanly() -> None:
    inner = FakeLLM()
    llm = PromptRegression(inner, version="sql-v2", rate=1.0, seed=1)
    request = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    token = deployment.set({"prompt_version": "sql-v1"})
    assert "SELECT 1" in await llm.complete(request)
    deployment.reset(token)
    token = deployment.set({"prompt_version": "sql-v2"})
    try:
        assert PII_SQL in await llm.complete(request)  # first attempt selects PII
        repair = [*request, {"role": "assistant", "content": "bad"}, {"role": "user", "content": "rejected"}]
        assert "SELECT 1" in await llm.complete(repair)
        assert inner.seen[-1] == request  # the repair answers the original request
    finally:
        deployment.reset(token)


async def test_provider_timeouts_raise_inside_the_window_only() -> None:
    timeline = Timeline(0)
    fault = Fault("F", "aegis", "t", 10, 20, "provider_degradation", "model", "m", "")
    llm = ProviderTimeouts(FakeLLM(), timeline, fault, rate=1.0, delay_s=0, seed=1)
    timeline.now_min = 5
    assert (await llm.complete_json([], Verdict)).ok  # type: ignore[attr-defined]
    timeline.now_min = 15
    with pytest.raises(TimeoutError):
        await llm.complete_json([], Verdict)


def test_deployment_state_follows_releases_and_rollbacks() -> None:
    sc = default_scenario(360)
    f2 = next(f for f in sc.faults if f.id == "F2")
    assert _deployment_at(sc, "quill", f2.start_min - 1)["prompt_version"] == "sql-v1"
    assert _deployment_at(sc, "quill", f2.start_min + 1)["prompt_version"] == "sql-v2"
    assert _deployment_at(sc, "quill", f2.end_min + 1)["prompt_version"] == "sql-v1"
