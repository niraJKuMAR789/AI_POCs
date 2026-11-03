"""Argus dashboard (Streamlit) over the HTTP API.

    streamlit run ui/app.py     # API at ARGUS_API_URL (default http://localhost:8003); paste a project API key

Colors follow a validated categorical palette (span kinds, fixed order) with separate light and dark
steps; status colors are reserved for errors and severities and always ship with an icon + label.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API = os.getenv("ARGUS_API_URL", "http://localhost:8003")
st.set_page_config(page_title="Argus: agent observability", page_icon="👁️", layout="wide")

LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "kinds": {"agent": "#2a78d6", "llm": "#eb6834", "tool": "#1baf7a", "retrieval": "#eda100", "chain": "#e87ba4"},
    "band": "rgba(11,11,11,0.06)",
    "line": "#2a78d6",
}
DARK = {
    "surface": "#1a1a19",
    "ink": "#f0efec",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "kinds": {"agent": "#3987e5", "llm": "#d95926", "tool": "#199e70", "retrieval": "#c98500", "chain": "#d55181"},
    "band": "rgba(255,255,255,0.08)",
    "line": "#3987e5",
}
STATUS_ICON = {"critical": "⛔", "high": "🔶", "medium": "⚠️", "good": "✅"}  # severity is never color-alone
KIND_ORDER = ["agent", "llm", "tool", "retrieval", "chain"]


def theme() -> dict[str, Any]:
    kind = getattr(getattr(st.context, "theme", None), "type", None)
    return DARK if kind == "dark" else LIGHT


def api(path: str, method: str = "GET", **kw: Any) -> Any:
    key = st.session_state.get("api_key", "")
    r = httpx.request(method, f"{API}{path}", headers={"Authorization": f"Bearer {key}"}, timeout=120, **kw)
    if r.status_code >= 400:
        st.error(f"{r.status_code}: {r.text}")
        st.stop()
    return r.json()


def ts(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, UTC)


def style(fig: go.Figure, t: dict[str, Any], title: str, height: int = 240) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": t["ink"]}, "x": 0},
        height=height,
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": t["ink2"], "size": 12},
        hoverlabel={"bgcolor": t["surface"], "font_color": t["ink"], "bordercolor": t["axis"]},
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=t["axis"], tickfont={"color": t["muted"]})
    fig.update_yaxes(gridcolor=t["grid"], zeroline=False, linecolor=t["axis"], tickfont={"color": t["muted"]})
    return fig


def series_chart(
    df: pd.DataFrame, column: str, title: str, fmt: str, incidents: list[dict[str, Any]], tickformat: str = "~s"
) -> None:
    t = theme()
    fig = go.Figure(
        go.Scatter(
            x=df["time"],
            y=df[column],
            mode="lines",
            line={"width": 2, "color": t["line"]},
            hovertemplate=f"%{{y:{fmt}}}<extra>{title}</extra>",
        )
    )
    for inc in incidents:  # incident windows: neutral band + labeled id (identity is never color alone)
        fig.add_vrect(
            x0=ts(inc["start_ns"]),
            x1=ts(inc["end_ns"]),
            fillcolor=t["band"],
            line_width=0,
            annotation_text=f"#{inc['id']}",
            annotation_position="top left",
            annotation_font={"size": 10, "color": t["ink2"]},
        )
    style(fig, t, title)
    fig.update_yaxes(rangemode="tozero", tickformat=tickformat)  # magnitudes start at zero; no negative rates
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def waterfall(spans: list[dict[str, Any]]) -> None:
    t = theme()
    spans = sorted(spans, key=lambda s: s["start_ns"])
    t0 = spans[0]["start_ns"]
    depth: dict[str, int] = {}
    by_id = {s["span_id"]: s for s in spans}
    for s in spans:
        d, p = 0, s.get("parent_span_id")
        while p and p in by_id and d < 20:
            d, p = d + 1, by_id[p].get("parent_span_id")
        depth[s["span_id"]] = d
    # One row per span (repeated tool calls stay visible); "·" marks nesting depth.
    labels = [f"{'· ' * depth[s['span_id']]}{s['name']}  #{i + 1}" for i, s in enumerate(spans)]
    end_ms = max((s["start_ns"] - t0) / 1e6 + s["duration_ms"] for s in spans)
    fig = go.Figure()
    for kind in KIND_ORDER:
        rows = [(i, s) for i, s in enumerate(spans) if s["kind"] == kind]
        if not rows:
            continue
        fig.add_trace(
            go.Bar(
                y=[labels[i] for i, _ in rows],
                x=[s["duration_ms"] for _, s in rows],
                base=[(s["start_ns"] - t0) / 1e6 for _, s in rows],
                orientation="h",
                name=kind,
                marker={"color": t["kinds"][kind], "line": {"color": t["surface"], "width": 2}},
                text=[f"{'✕ ' if s['status'] == 'error' else ''}{s['duration_ms']:,.0f} ms" for _, s in rows],
                textposition="outside",
                textfont={"color": t["ink2"], "size": 11},
                cliponaxis=False,
                customdata=[
                    [
                        s["kind"],
                        s.get("model") or s.get("tool_name") or "",
                        s["status"],
                        s.get("input_tokens", 0),
                        s.get("error_type") or "",
                    ]
                    for _, s in rows
                ],
                hovertemplate="<b>%{y}</b><br>%{customdata[0]} · %{customdata[1]}<br>%{x:,.1f} ms · "
                "status %{customdata[2]} %{customdata[4]}<br>input tokens %{customdata[3]}<extra></extra>",
            )
        )
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1], showgrid=False)
    fig.update_xaxes(
        title={"text": "ms since trace start", "font": {"color": t["muted"]}}, range=[0, end_ms * 1.15]
    )  # room for the direct duration labels
    style(fig, t, "Trace waterfall", height=max(220, 26 * len(spans) + 90))
    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        bargap=0.35,
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"color": t["ink2"]}},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def severity_label(sev: str) -> str:
    return f"{STATUS_ICON.get(sev, '•')} {sev}"


with st.sidebar:
    st.title("👁️ Argus")
    st.caption("Observability · governance · anomaly investigation for agentic apps")
    st.session_state.api_key = st.text_input(
        "Project API key", type="password", value=st.session_state.get("api_key", os.getenv("ARGUS_API_KEY", ""))
    )
    try:
        st.success(f"collector {httpx.get(f'{API}/health', timeout=5).json()['version']} online")
    except httpx.HTTPError:
        st.error(f"API unreachable at {API}")
        st.stop()
    if not st.session_state.api_key:
        st.info("Run `argus demo` and paste one of the printed keys.")
        st.stop()

overview, traces_tab, incidents_tab, governance_tab = st.tabs(["Overview", "Traces", "Incidents", "Governance"])
incidents = api("/v1/incidents")

with overview:
    summary = api("/v1/metrics/summary")
    a, b, c, d, e = st.columns(5)
    a.metric("Traces", f"{summary['traces']:,}")
    b.metric("Spend", f"${summary['cost_usd']:,.2f}")
    c.metric("Error rate", f"{summary['error_rate']:.1%}")
    d.metric("p95 latency", f"{summary['p95_ms'] / 1000:,.1f} s")
    e.metric("Open incidents", sum(i["status"] == "open" for i in incidents))
    df = pd.DataFrame(summary["series"])
    if not df.empty:
        df["time"] = [ts(n) for n in df["bucket_ns"]]
        left, right = st.columns(2)
        with left:
            series_chart(df, "traces", "Traces per 10 min", ",.0f", incidents, ",.0f")
            series_chart(df, "error_rate", "Error rate", ".1%", incidents, ".0%")
        with right:
            series_chart(df, "p95_ms", "p95 trace latency (ms)", ",.0f", incidents, ",.0f")
            series_chart(df, "cost_usd", "Spend per 10 min (USD)", "$,.4f", incidents, "$,.2f")
        st.caption("Shaded bands are detected incidents (see the Incidents tab).")

with traces_tab:
    traces = api("/v1/traces?limit=300")
    tdf = pd.DataFrame(traces)
    if tdf.empty:
        st.info("No traces yet.")
    else:
        tdf["start"] = [ts(n) for n in tdf["start_ns"]]
        only_errors = st.toggle("Only traces with errors")
        view = tdf[tdf["error_count"] > 0] if only_errors else tdf
        st.dataframe(
            view[
                [
                    "start",
                    "root_name",
                    "trace_id",
                    "duration_ms",
                    "span_count",
                    "error_count",
                    "cost_usd",
                    "max_tool_repeat",
                    "pii_spans",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            height=260,
        )
        chosen = st.selectbox("Trace", view["trace_id"].tolist())
        if chosen:
            detail = api(f"/v1/traces/{chosen}?purpose=dashboard")
            waterfall(detail["spans"])
            with st.expander(
                "Span table" + (" (content visible: this read is logged)" if detail["content_included"] else "")
            ):
                st.dataframe(
                    pd.DataFrame(detail["spans"]).drop(columns=["attributes"], errors="ignore"),
                    hide_index=True,
                    use_container_width=True,
                )

with incidents_tab:
    if not incidents:
        st.info("No incidents detected.")
    for inc in sorted(incidents, key=lambda i: i["start_ns"], reverse=True):
        minutes = (inc["end_ns"] - inc["start_ns"]) // (60 * 10**9)
        with st.expander(
            f"#{inc['id']} · {severity_label(inc['severity'])} · {inc['title']} · "
            f"{ts(inc['start_ns']):%H:%M} UTC, {minutes} min · {inc['status']}"
        ):
            detail = api(f"/v1/incidents/{inc['id']}")
            st.dataframe(
                pd.DataFrame(detail["anomalies"])[
                    ["detector", "group_key", "metric", "observed", "baseline", "score", "severity"]
                ],
                hide_index=True,
                use_container_width=True,
            )
            if st.button("Investigate", key=f"inv-{inc['id']}"):
                with st.spinner("Investigating…"):
                    api(f"/v1/incidents/{inc['id']}/investigate", method="POST")
                detail = api(f"/v1/incidents/{inc['id']}")
            for inv in detail["investigations"][-1:]:
                report = inv["report"]
                st.markdown(
                    f"**Root cause:** `{report['category']}` · `{report['cause_dimension']}="
                    f"{report['cause_value']}` · confidence {report['confidence']:.0%} · by `{inv['model']}`"
                )
                st.write(report["summary"])
                st.markdown(
                    "**Evidence**\n"
                    + "\n".join(f"- *{e['kind']}* `{e['ref']}` {e['detail']}" for e in report["evidence"])
                )
                st.markdown("**Recommended actions**\n" + "\n".join(f"- {a}" for a in report["recommended_actions"]))
                for err in inv["validation_errors"]:
                    st.warning(f"evidence check: {err}")

with governance_tab:
    violations = pd.DataFrame(api("/v1/violations?limit=2000"))
    policy = api("/v1/policy")
    left, right = st.columns([2, 1])
    with left:
        if violations.empty:
            st.success("✅ No policy violations.")
        else:
            counts = violations.groupby(["policy", "severity"]).size().reset_index(name="violations")
            counts["severity"] = counts["severity"].map(severity_label)
            st.dataframe(counts.sort_values("violations", ascending=False), hide_index=True, use_container_width=True)
            violations["time"] = [ts(n) for n in violations["ts_ns"]]
            st.dataframe(
                violations[["time", "policy", "severity", "message", "trace_id"]],
                hide_index=True,
                use_container_width=True,
                height=320,
            )
    with right:
        st.markdown("**Effective policy**")
        st.json(policy)
        st.markdown("**Content access log**")
        st.dataframe(pd.DataFrame(api("/v1/content-access-log")), hide_index=True, use_container_width=True)
