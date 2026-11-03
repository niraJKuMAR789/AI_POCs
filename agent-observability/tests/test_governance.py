import pytest

from argus import semconv
from argus.governance.pii import redact
from argus.models import TraceSummary
from argus.sdk import InMemoryExporter, init, span
from argus.service import Argus


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("mail bob@corp.io now", "EMAIL"),
        ("ssn 219-09-9999", "SSN"),
        ("cc 4242 4242 4242 4242", "CREDIT_CARD"),
        ("call (212) 555-0147", "PHONE"),
        ("key sk-proj-abcdefghijklmnopqrstuvwx", "API_KEY"),
        ("nvapi-0123456789abcdefghijklmn", "API_KEY"),
        ("AKIAIOSFODNN7EXAMPLE", "API_KEY"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "BEARER_TOKEN"),
        ("host 192.168.1.20", "IP_ADDRESS"),
    ],
)
def test_detects_sensitive_data(text: str, kind: str) -> None:
    redacted, counts = redact(text)
    assert counts[kind] == 1 and f"[REDACTED:{kind}]" in redacted


@pytest.mark.parametrize("text", ["order 1234567890123", "ssn 000-12-3456", "version 1.2.3.4.5", "999-99-9999"])
def test_ignores_lookalikes(text: str) -> None:
    assert redact(text)[1].total() == 0


def _payload(
    project_tokens: int = 100,
    model: str = "nvidia/nemotron-3-super-120b-a12b",
    tool: str | None = None,
    text: str = "hello",
) -> dict:
    exporter = InMemoryExporter()
    init(service="svc", exporter=exporter)
    with span("agent", kind="agent"):
        with span(
            "chat",
            kind="llm",
            attributes={
                semconv.REQUEST_MODEL: model,
                semconv.INPUT_TOKENS: project_tokens,
                semconv.OUTPUT_TOKENS: 10,
                semconv.INPUT: text,
            },
        ):
            pass
        if tool:
            with span(tool, kind="tool", attributes={semconv.TOOL_NAME: tool}):
                pass
    return exporter.payload()


def test_redaction_happens_before_storage_and_policy_flags(argus: Argus) -> None:
    pid = argus.store.create_project("support-bot")
    result = argus.ingestor.ingest(
        pid,
        _payload(
            model="some/unapproved-model",
            tool="delete_account",
            text="user jane@x.com key nvapi-0123456789abcdefghijklmn",
        ),
    )
    llm = next(s for s in argus.store.query("SELECT * FROM spans WHERE kind = 'llm'"))
    assert "jane@x.com" not in llm["input_text"] and "[REDACTED:EMAIL]" in llm["input_text"]
    assert "nvapi-" not in llm["input_text"]
    policies = {v.policy: v.severity for v in argus.store.violations(pid)}
    assert policies["unapproved_model"] == "high" and policies["forbidden_tool"] == "critical"
    assert policies["sensitive_data"] == "critical"  # secrets escalate severity
    assert result.accepted == 3


def test_block_policy_drops_content(argus: Argus) -> None:
    pid = argus.store.create_project("claims-agent")  # pii: block
    argus.ingestor.ingest(pid, _payload(text="SSN 219-09-9999"))
    row = argus.store.query("SELECT input_text, pii_found FROM spans WHERE kind = 'llm'")[0]
    assert row["input_text"] is None and "SSN" in row["pii_found"]


def test_reingest_is_idempotent_and_trace_limits_apply(argus: Argus) -> None:
    pid = argus.store.create_project("support-bot")  # max $0.05 per trace
    payload = _payload(project_tokens=200_000)
    first, second = argus.ingestor.ingest(pid, payload), argus.ingestor.ingest(pid, payload)
    assert first.accepted == 2 and second.accepted == 0 and second.duplicates == 2
    policies = [v.policy for v in argus.store.violations(pid)]
    assert policies.count("trace_cost_limit") == 1 and policies.count("token_limit") == 1


def test_daily_budget(argus: Argus) -> None:
    pid = argus.store.create_project("tiny", org="x")
    argus.ingestor.policies.overrides["tiny"] = {"daily_budget_usd": 0.01, "max_cost_per_trace_usd": 100}
    for _ in range(3):
        argus.ingestor.ingest(pid, _payload(project_tokens=10_000))
    assert [v.policy for v in argus.store.violations(pid)].count("daily_budget") == 1


def test_retention_scrubs_content_before_metrics(argus: Argus) -> None:
    pid = argus.store.create_project("support-bot")
    argus.ingestor.ingest(pid, _payload())
    start = argus.store.query("SELECT MIN(start_ns) FROM spans")[0][0]
    day = 86_400 * 10**9
    assert argus.store.apply_retention(pid, start + 8 * day, 30, 7)["content_scrubbed"] >= 1
    assert argus.store.query("SELECT COUNT(*) FROM spans WHERE input_text IS NOT NULL")[0][0] == 0
    assert argus.store.query("SELECT COUNT(*) FROM spans")[0][0] == 2  # metrics kept
    assert argus.store.apply_retention(pid, start + 31 * day, 30, 7)["spans_deleted"] == 2


def test_summary_uses_true_root_even_if_children_arrive_first(argus: Argus) -> None:
    from argus.ingest.pipeline import summarize

    pid = argus.store.create_project("support-bot")
    exporter = InMemoryExporter()
    init(service="s", exporter=exporter)
    with span("root", kind="agent"), span("child", kind="tool", attributes={semconv.TOOL_NAME: "t"}):
        pass
    child_only = {"resourceSpans": exporter.payload()["resourceSpans"]}
    child_only["resourceSpans"][0]["scopeSpans"][0]["spans"] = child_only["resourceSpans"][0]["scopeSpans"][0]["spans"][
        :1
    ]
    argus.ingestor.ingest(pid, child_only)
    assert argus.store.traces(pid) == []  # no root yet -> not counted as a completed trace
    argus.ingestor.ingest(pid, exporter.payload())
    t: TraceSummary = argus.store.traces(pid)[0]
    assert t.root_name == "root" and t.span_count == 2 and t.max_depth == 2
    assert summarize(pid, t.trace_id, argus.store.spans_for_trace(pid, t.trace_id)).has_root
