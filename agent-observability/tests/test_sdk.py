import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx

from argus import semconv
from argus.ingest.otlp import decode
from argus.sdk import InMemoryExporter, init, span, trace_agent, trace_tool
from argus.sdk.exporter import HTTPExporter
from argus.sdk.instrument.openai import instrument_openai


def test_spans_nest_and_record_errors() -> None:
    exporter = InMemoryExporter()
    init(service="t", exporter=exporter)
    try:
        with span("root", kind="agent"), span("child", kind="tool"):
            raise ValueError("boom")
    except ValueError:
        pass
    child, root = (r for r, _ in exporter.spans)
    assert child.parent_span_id == root.span_id and child.trace_id == root.trace_id
    assert child.status == "error" and child.attributes[semconv.ERROR_TYPE] == "ValueError"
    assert root.status == "error"  # exception propagated through the parent too


async def test_context_propagates_across_asyncio_tasks() -> None:
    exporter = InMemoryExporter()
    init(service="t", exporter=exporter)

    @trace_tool("lookup")
    async def lookup(q: str) -> str:
        await asyncio.sleep(0)
        return q.upper()

    @trace_agent("bot", version="v9")
    async def agent(q: str) -> list[str]:
        return list(await asyncio.gather(lookup(q + "1"), lookup(q + "2")))

    assert await agent("x") == ["X1", "X2"]
    records = {r.name: r for r, _ in exporter.spans}
    tools = [r for r, _ in exporter.spans if r.name == "lookup"]
    assert len(tools) == 2 and all(t.parent_span_id == records["bot"].span_id for t in tools)
    assert records["bot"].attributes[semconv.AGENT_VERSION] == "v9"
    assert json.loads(tools[0].attributes[semconv.INPUT])["q"] in ("x1", "x2")
    assert tools[0].attributes[semconv.TOOL_ARGS_HASH] != tools[1].attributes[semconv.TOOL_ARGS_HASH]


def test_capture_content_off_strips_text() -> None:
    exporter = InMemoryExporter()
    init(service="t", exporter=exporter, capture_content=False)

    @trace_tool("secret_tool")
    def tool(x: str) -> str:
        return x

    tool("hello")
    attrs = exporter.spans[0][0].attributes
    assert semconv.INPUT not in attrs and semconv.OUTPUT not in attrs


def test_otlp_roundtrip() -> None:
    exporter = InMemoryExporter()
    init(service="svc", exporter=exporter)
    with span("a", kind="llm", attributes={semconv.INPUT_TOKENS: 12, "x.float": 1.5, "x.bool": True}):
        pass
    decoded = decode(exporter.payload())[0]
    assert decoded["attributes"][semconv.INPUT_TOKENS] == 12 and decoded["attributes"]["x.float"] == 1.5
    assert decoded["attributes"]["x.bool"] is True and decoded["resource"][semconv.SERVICE_NAME] == "svc"


def test_http_exporter_batches_and_counts_drops() -> None:
    received: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return httpx.Response(200 if len(received) == 1 else 400, json={})

    exporter = HTTPExporter(
        "http://collector",
        api_key="k",
        max_batch=3,
        interval_s=3600,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    init(service="t", exporter=exporter)
    for i in range(5):
        with span(f"s{i}"):
            pass
    exporter.flush()
    assert len(received) == 2  # 3 + 2
    assert exporter.sent == 3 and exporter.dropped == 2  # 4xx batches are dropped, not retried
    exporter.shutdown()


def _completion(text: str) -> Any:
    return SimpleNamespace(
        model="nvidia/nemotron-3-nano-30b-a3b",
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))],
    )


async def test_openai_instrumentation_sync_async_and_stream() -> None:
    exporter = InMemoryExporter()
    init(service="t", exporter=exporter)

    class SyncCompletions:
        def create(self, **kwargs: Any) -> Any:
            return _completion("hi")

    class AsyncCompletions:
        async def create(self, **kwargs: Any) -> Any:
            if kwargs.get("stream"):

                async def gen() -> Any:
                    for part in ("he", "llo"):
                        yield SimpleNamespace(
                            model="m", usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=part))]
                        )
                    yield SimpleNamespace(
                        model="m", usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2), choices=[]
                    )

                return gen()
            return _completion("async")

    sync_client = instrument_openai(
        SimpleNamespace(
            base_url="https://integrate.api.nvidia.com/v1", chat=SimpleNamespace(completions=SyncCompletions())
        )
    )
    async_client = instrument_openai(
        SimpleNamespace(base_url="http://localhost:8000/v1", chat=SimpleNamespace(completions=AsyncCompletions()))
    )
    sync_client.chat.completions.create(model="m1", messages=[{"role": "user", "content": "q"}])
    await async_client.chat.completions.create(model="m2", messages=[])
    stream = await async_client.chat.completions.create(model="m3", messages=[], stream=True)
    assert "".join([c.choices[0].delta.content async for c in stream if c.choices]) == "hello"
    s1, s2, s3 = (r for r, _ in exporter.spans)
    assert s1.attributes[semconv.SYSTEM] == "nvidia_nim" and s1.attributes[semconv.INPUT_TOKENS] == 11
    assert s2.attributes[semconv.SYSTEM] == "self_hosted" and s2.attributes[semconv.OUTPUT] == "async"
    assert s3.attributes[semconv.OUTPUT] == "hello" and s3.attributes[semconv.OUTPUT_TOKENS] == 2


def test_langgraph_callback_builds_span_tree() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    from argus.sdk.instrument.langchain import ArgusCallbackHandler

    exporter = InMemoryExporter()
    init(service="graph", exporter=exporter)

    @tool
    def search(query: str) -> str:
        """Search the knowledge base."""
        return f"result for {query}"

    model = FakeListChatModel(responses=["final answer"])

    class State(TypedDict):
        q: str
        answer: str

    def retrieve(state: State, config) -> dict:  # type: ignore[no-untyped-def]
        return {"q": search.invoke(state["q"], config=config)}

    def respond(state: State, config) -> dict:  # type: ignore[no-untyped-def]
        return {"answer": model.invoke(state["q"], config=config).content}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("respond", respond)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "respond")
    g.add_edge("respond", END)
    out = g.compile().invoke({"q": "refund policy", "answer": ""}, config={"callbacks": [ArgusCallbackHandler()]})
    assert out["answer"] == "final answer"
    spans = {r.name: r for r, _ in exporter.spans}
    root = next(r for r, _ in exporter.spans if r.parent_span_id is None)
    assert {"retrieve", "respond", "search"} <= set(spans)
    assert spans["search"].kind == "tool" and spans["search"].parent_span_id == spans["retrieve"].span_id
    llm = next(r for r, _ in exporter.spans if r.kind == "llm")
    assert llm.parent_span_id == spans["respond"].span_id and llm.trace_id == root.trace_id
