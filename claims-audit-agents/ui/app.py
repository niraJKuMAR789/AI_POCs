"""Aegis reviewer console (Streamlit) over the HTTP API.

streamlit run ui/app.py     # API at AEGIS_API_URL (default http://localhost:8002)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("AEGIS_API_URL", "http://localhost:8002")
LABELED = Path(os.getenv("AEGIS_LABELED", "data/claims_labeled.jsonl"))
ICON = {"pay": "🟢", "adjust": "🟡", "deny": "🔴", "review": "🟠"}

st.set_page_config(page_title="Aegis: Claims Audit", page_icon="🛡️", layout="wide")


def api(method: str, path: str, **kwargs: Any) -> Any:
    response = httpx.request(method, f"{API}{path}", timeout=300, **kwargs)
    if response.status_code >= 400:
        st.error(f"{response.status_code}: {response.text}")
        st.stop()
    return response.json()


def show_decision(decision: dict[str, Any]) -> None:
    st.subheader(f"{decision['claim_id']}: {decision['outcome'].replace('_', ' ')}")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "line": ln["line_no"],
                    "cpt": ln["cpt"],
                    "outcome": f"{ICON[ln['outcome']]} {ln['outcome']}",
                    "units": ln["allowed_units"],
                    "decided by": ln["decided_by"],
                    "reasons": " | ".join(ln["reasons"]),
                }
                for ln in decision["lines"]
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    show_findings(decision["findings"], decision["reviews"])


def show_findings(findings: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> None:
    for f in findings:
        badge = "⛔ hard" if f["severity"] == "hard" else "🔍 soft"
        st.markdown(
            f"**{f['rule_id']} {f['title']}** · {badge} · line {f['line_no'] or 'claim'} · "
            f"`{f['policy_section']}`  \n{f['message']}"
        )
    for r in reviews:
        with st.expander(
            f"🤖 {r['reviewer']} reviewer → {r['resolution']} ({r['confidence']:.2f}) · "
            f"{r['rule_id']} line {r['line_no']}"
        ):
            st.write(r["rationale"])
            for quote in r["evidence_quotes"]:
                st.markdown(f"> {quote}")
            if r["policy_citations"]:
                st.caption("cites " + ", ".join(r["policy_citations"]))
            for err in r["validation_errors"]:
                st.warning(f"validation: {err}")


with st.sidebar:
    st.title("🛡️ Aegis")
    st.caption("Rules engine + NVIDIA NIM reviewer agents + human in the loop")
    health = api("GET", "/health")
    st.success(f"API online · reviewers: {', '.join(set(health['reviewers'].values()))}")
    st.metric("Audit ledger", "verified ✅" if health["ledger_verified"] else "BROKEN ❌")

submit_tab, queue_tab, trail_tab = st.tabs(["📥 Submit claim", "🧑‍⚕️ Review queue", "🔗 Audit trail"])

with submit_tab:
    samples = [json.loads(x) for x in LABELED.read_text().splitlines()] if LABELED.exists() else []
    by_scenario = {s["scenario"]: s for s in samples}
    scenario = st.selectbox("Sample scenario (synthetic, labeled)", sorted(by_scenario)) if by_scenario else None
    default = json.dumps(by_scenario[scenario]["claim"], indent=2) if scenario else "{}"
    raw = st.text_area("Claim JSON", default, height=280, key=f"claim-{scenario}")
    if scenario:
        st.caption(f"Expected: {by_scenario[scenario]['expected_lines']}")
    if st.button("Audit claim", type="primary"):
        result = api("POST", "/v1/claims", json=json.loads(raw))
        if result["decision"]:
            show_decision(result["decision"])
        else:
            st.warning("Pended for human review: see the Review queue tab.")

with queue_tab:
    queue = api("GET", "/v1/reviews")
    if not queue:
        st.info("No claims waiting for review.")
    for item in queue:
        with st.container(border=True):
            st.subheader(f"Claim {item['claim_id']}")
            show_findings(item["findings"], item["reviews"])
            decisions = {}
            for line in item["lines"]:
                decisions[line["line_no"]] = st.radio(
                    f"Line {line['line_no']} ({line['cpt']}): " + "; ".join(line["reasons"])[:300],
                    ["pay", "deny"],
                    horizontal=True,
                    key=f"{item['claim_id']}-{line['line_no']}",
                )
            reviewer = st.text_input("Reviewer", "rn.reviewer", key=f"who-{item['claim_id']}")
            note = st.text_input("Note", key=f"note-{item['claim_id']}")
            if st.button("Submit decision", key=f"go-{item['claim_id']}"):
                done = api(
                    "POST",
                    f"/v1/reviews/{item['claim_id']}",
                    json={"reviewer": reviewer, "decisions": decisions, "note": note},
                )
                show_decision(done["decision"])

with trail_tab:
    claim_id = st.text_input("Claim id")
    if claim_id:
        trail = api("GET", f"/v1/audit/{claim_id}")
        st.markdown(f"Chain verified: **{trail['chain_verified']}**")
        for event in trail["events"]:
            with st.expander(f"#{event['seq']} {event['event']} · {event['actor']} · {event['ts'][:19]}"):
                st.code(f"prev {event['prev_hash'][:16]}…  hash {event['hash'][:16]}…")
                st.json(event["payload"])
