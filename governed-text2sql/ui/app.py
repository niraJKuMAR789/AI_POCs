"""Quill demo UI (Streamlit) over the HTTP API.

streamlit run ui/app.py      # API at QUILL_API_URL (default http://localhost:8001)
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("QUILL_API_URL", "http://localhost:8001")
EXAMPLES = {
    "analyst": [
        "What is the denial rate for in-network versus out-of-network providers?",
        "Which specialties have a denial rate above 12%?",
        "List the first and last names of members in Texas.",
    ],
    "claims_auditor": [
        "Show the 10 largest denied claims in 2025 with member names.",
        "Flag claim 1234 for review because of possible duplicate billing.",
    ],
    "provider_portal": [
        "How many of my claims were denied, by denial reason?",
        "Show total paid for provider 8's claims.",
    ],
}
OUTCOME_ICON = {
    "ok": "✅",
    "guard_violation": "🛡️",
    "db_error": "💥",
    "timeout": "⏱️",
    "unanswerable": "🤷",
    "write_pending": "✋",
}

st.set_page_config(page_title="Quill: Governed Text-to-SQL", page_icon="🪶", layout="wide")


def headers() -> dict[str, str]:
    h = {"X-Quill-Role": st.session_state.role}
    if st.session_state.role == "provider_portal":
        h["X-Quill-Context"] = json.dumps({"provider_id": st.session_state.provider_id})
    return h


def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(f"{API}{path}", json=body, headers=headers(), timeout=180)
    if response.status_code >= 400:
        st.error(f"{response.status_code}: {response.text}")
        st.stop()
    return response.json()


def show_result(result: dict[str, Any]) -> None:
    badge = {"ok": "🟢", "blocked": "🔴", "needs_confirmation": "🟠", "unanswerable": "⚪", "error": "🔴"}
    st.markdown(f"**{badge.get(result['status'], '•')} {result['status']}**: {result.get('answer') or ''}")
    if result.get("executed_sql"):
        st.code(result["executed_sql"], language="sql", wrap_lines=True)
    for rewrite in result.get("guard_rewrites", []):
        st.caption(f"🛡️ guard rewrite: {rewrite}")
    for warning in result.get("plan_warnings", []):
        st.caption(f"⚠️ {warning}")
    if len(result.get("attempts", [])) > 1 or result["status"] != "ok":
        with st.expander(f"{len(result['attempts'])} attempt(s)"):
            for i, a in enumerate(result["attempts"], 1):
                st.markdown(f"{OUTCOME_ICON.get(a['outcome'], '•')} **{i}. {a['outcome']}** {a['feedback']}")
                st.code(a["sql"], language="sql", wrap_lines=True)
    if result.get("columns"):
        frame = pd.DataFrame(result["rows"], columns=result["columns"])
        st.dataframe(frame, use_container_width=True, hide_index=True)
        numeric = frame.select_dtypes("number").columns
        if len(frame) > 1 and len(frame.columns) >= 2 and len(numeric) >= 1 and frame.columns[0] not in numeric:
            st.bar_chart(frame.set_index(frame.columns[0])[numeric[0]])
    if result["status"] == "needs_confirmation" and st.button("Approve and execute write", type="primary"):
        st.success(post("/v1/writes/confirm", {"sql": result["executed_sql"]}))


with st.sidebar:
    st.title("🪶 Quill")
    st.caption("Governed text-to-SQL · Agno · MCP · NVIDIA NIM")
    try:
        health = httpx.get(f"{API}/health", timeout=10).json()
        st.success(f"API online · {health['sql_model']}")
    except httpx.HTTPError as exc:
        st.error(f"API unreachable at {API}: {exc}")
        st.stop()
    st.session_state.role = st.selectbox("Role", health["roles"], index=health["roles"].index("analyst"))
    st.session_state.provider_id = st.number_input(
        "provider_id (tenant)", min_value=1, value=7, disabled=st.session_state.role != "provider_portal"
    )
    with st.expander("What this role can see"):
        st.code(httpx.get(f"{API}/v1/schema", headers=headers(), timeout=10).json()["ddl"], language="sql")

ask_tab, sql_tab = st.tabs(["💬 Ask", "🧪 SQL console (guarded)"])
with ask_tab:
    cols = st.columns(len(EXAMPLES[st.session_state.role]))
    for col, example in zip(cols, EXAMPLES[st.session_state.role], strict=True):
        if col.button(example, use_container_width=True):
            st.session_state.question = example
    question = st.text_input("Question", key="question")
    if question:
        with st.spinner("Generating, validating and executing…"):
            st.session_state.last = post("/v1/ask", {"question": question})
    if "last" in st.session_state:
        show_result(st.session_state.last)
with sql_tab:
    sql = st.text_area("SQL", "SELECT * FROM members LIMIT 5", height=120)
    if st.button("Run under role policy"):
        show_result(post("/v1/sql", {"sql": sql}))
