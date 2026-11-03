"""Trace a LangGraph app with the Argus callback handler (runs offline with a fake chat model).

ARGUS_API_KEY=... python examples/langgraph_agent.py
"""

from __future__ import annotations

import os

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from argus.sdk import get_tracer, init
from argus.sdk.instrument.langchain import ArgusCallbackHandler

init(
    endpoint=os.getenv("ARGUS_ENDPOINT", "http://localhost:8003"),
    api_key=os.environ["ARGUS_API_KEY"],
    service="langgraph-demo",
)


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's status."""
    return f"order {order_id}: shipped"


model = FakeListChatModel(responses=["Your order has shipped and should arrive tomorrow."])


class State(TypedDict):
    order_id: str
    status: str
    reply: str


def fetch(state: State, config) -> dict:  # type: ignore[no-untyped-def]
    return {"status": lookup_order.invoke(state["order_id"], config=config)}


def respond(state: State, config) -> dict:  # type: ignore[no-untyped-def]
    return {"reply": model.invoke(f"Status: {state['status']}", config=config).content}


graph = StateGraph(State)
graph.add_node("fetch", fetch)
graph.add_node("respond", respond)
graph.add_edge(START, "fetch")
graph.add_edge("fetch", "respond")
graph.add_edge("respond", END)

if __name__ == "__main__":
    result = graph.compile().invoke(
        {"order_id": "A-1001", "status": "", "reply": ""}, config={"callbacks": [ArgusCallbackHandler()]}
    )
    print(result["reply"])
    get_tracer().exporter.flush()  # type: ignore[union-attr]
