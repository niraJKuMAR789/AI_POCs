"""Atlas demo UI (Streamlit). Talks to the FastAPI service over HTTP + SSE.

streamlit run ui/app.py        # expects the API at ATLAS_API_URL (default http://localhost:8000)
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("ATLAS_API_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": os.environ["ATLAS_API_KEY"]} if os.getenv("ATLAS_API_KEY") else {}
NODE_ICONS = {
    "analyze": "🧭",
    "retrieve": "🔎",
    "grade": "⚖️",
    "rewrite": "✏️",
    "web_search": "🌐",
    "generate": "✍️",
    "reflect": "🪞",
    "finalize": "✅",
}
EXAMPLES = [
    "Who is the primary on-call lead for the service whose schema change caused INC-2041?",
    "Why did VisionCore 3.2 look better offline but fail in production?",
    "What are the main reliability themes across Helix Dynamics incidents and roadmap?",
]

st.set_page_config(page_title="Atlas: Adaptive GraphRAG", page_icon="🗺️", layout="wide")


def stream_answer(question: str, thread_id: str | None) -> Iterator[dict[str, Any]]:
    with httpx.stream(
        "POST",
        f"{API_URL}/v1/ask/stream",
        json={"question": question, "thread_id": thread_id},
        headers=HEADERS,
        timeout=180,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data:"):
                yield json.loads(line[5:].strip())


def api_get(path: str) -> Any:
    response = httpx.get(f"{API_URL}{path}", headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def render_trace(trace: list[dict[str, Any]], slot: Any) -> None:
    with slot.container():
        for step in trace:
            icon = NODE_ICONS.get(step["node"], "•")
            st.markdown(f"{icon} **{step['node']}**: {step['summary']}  \n`{step['latency_ms']} ms`")


def render_graph(limit: int) -> None:
    data = api_get(f"/v1/graph?limit={limit}")
    if not data["nodes"]:
        st.info("The knowledge graph is empty. Ingest some documents first.")
        return
    lines = [
        "graph G {",
        "layout=neato; overlap=false; splines=true;",
        'node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontsize=10];',
    ]
    for node in data["nodes"]:
        label = node["label"].replace('"', "'")
        lines.append(f'"{node["id"]}" [label="{label}"];')
    for edge in data["edges"]:
        lines.append(f'"{edge["source"]}" -- "{edge["target"]}" [penwidth={min(1 + edge["weight"] / 2, 4):.1f}];')
    lines.append("}")
    st.graphviz_chart("\n".join(lines), use_container_width=True)


# --- sidebar -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🗺️ Atlas")
    st.caption("Adaptive GraphRAG · LangGraph · NVIDIA NIM · Qdrant · MCP")
    try:
        health = api_get("/health")
        st.success(f"API online · provider: **{health['provider']}**")
        cols = st.columns(3)
        for col, key in zip(cols, ("documents", "entities", "communities"), strict=True):
            col.metric(key, health["stats"][key])
    except httpx.HTTPError as exc:
        st.error(f"API unreachable at {API_URL}: {exc}")
        st.stop()

    uploads = st.file_uploader("Add documents", type=["md", "txt", "pdf"], accept_multiple_files=True)
    if uploads and st.button("Ingest", use_container_width=True):
        with st.spinner("Chunking, embedding, extracting graph…"):
            files = [("files", (f.name, f.getvalue(), f.type or "text/plain")) for f in uploads]
            report = httpx.post(f"{API_URL}/v1/ingest/files", files=files, headers=HEADERS, timeout=600).json()
        st.json(report)

    if st.button("New conversation", use_container_width=True):
        st.session_state.clear()
        st.rerun()

chat_tab, graph_tab, communities_tab = st.tabs(["💬 Research chat", "🕸️ Knowledge graph", "🧩 Communities"])

# --- chat ----------------------------------------------------------------------------------
with chat_tab:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", None)
    chat_col, trace_col = st.columns([3, 1.3])

    with trace_col:
        st.subheader("Agent trace")
        trace_box = st.empty()
        last = next((m for m in reversed(st.session_state.messages) if m.get("trace")), None)
        if last:
            render_trace(last["trace"], trace_box)

    with chat_col:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("citations"):
                    with st.expander(f"{len(message['citations'])} sources"):
                        for c in message["citations"]:
                            st.markdown(f"**[{c['label']}] {c['title']}** · _{c['kind']}_ · `{c['source']}`")
                            st.caption(c["snippet"])

        if not st.session_state.messages:
            st.markdown("#### Try a multi-hop question")
            for example in EXAMPLES:
                if st.button(example, use_container_width=True):
                    st.session_state.pending = example
                    st.rerun()

        question = st.chat_input("Ask about the knowledge base…") or st.session_state.pop("pending", None)
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                answer, trace, final = "", [], None
                for event in stream_answer(question, st.session_state.thread_id):
                    if event["type"] == "token":
                        answer += event["text"]
                        placeholder.markdown(answer + "▌")
                    elif event["type"] == "reset":
                        answer = ""
                        placeholder.markdown("_Draft failed the grounding check; regenerating…_")
                    elif event["type"] == "step":
                        trace.append(event)
                        render_trace(trace, trace_box)
                    elif event["type"] == "final":
                        final = event["result"]
                if final:
                    placeholder.markdown(final["answer"])
                    st.session_state.thread_id = final["thread_id"]
                    badge = "🟢 grounded" if final["grounded"] else "🟠 partially verified"
                    st.caption(f"route: **{final['route']}** · {badge}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": final["answer"],
                            "citations": final["citations"],
                            "trace": trace,
                        }
                    )
                    st.rerun()

with graph_tab:
    limit = st.slider("Entities to show (by weighted degree)", 10, 150, 50, step=10)
    render_graph(limit)

with communities_tab:
    for community in api_get("/v1/communities"):
        with st.expander(f"{community['title']} · {len(community['entities'])} entities"):
            st.write(community["summary"])
