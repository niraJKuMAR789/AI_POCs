import json
from types import SimpleNamespace
from typing import Any

from argus.investigate.agent import LLMInvestigator
from argus.investigate.deterministic import investigate
from argus.investigate.report import Evidence, RCAReport, validate
from argus.investigate.tools import InvestigationTools
from argus.service import INTERNAL_PROJECT

EXPECTED = {
    ("support-bot", "llm_latency"): ("provider_degradation", "nvidia/nemotron-3-super-120b-a12b"),
    ("support-bot", "tool_error_rate"): ("tool_failure", "search_kb"),
    ("support-bot", "throughput"): ("traffic_surge", "surge"),
    ("research-agent", "tool_loop"): ("agent_regression", "v1.4"),
    ("claims-agent", "sensitive_data"): ("data_leak", "get_claim"),
}


def test_deterministic_rca_on_every_incident(simulated) -> None:  # type: ignore[no-untyped-def]
    argus, _, projects = simulated
    checked = 0
    for (name, detector), (category, cause) in EXPECTED.items():
        tools = InvestigationTools(argus.store, projects[name])
        inc = next(i for i in argus.store.incidents(projects[name]) if detector in i.detectors)
        report = investigate(tools, inc.id)
        assert (report.category, report.cause_value) == (category, cause), (name, report.summary)
        assert validate(report, tools, inc.id) == []
        checked += 1
    assert checked == len(EXPECTED)


def test_provider_slowdown_evidence_cites_other_projects(simulated) -> None:  # type: ignore[no-untyped-def]
    argus, _, projects = simulated
    tools = InvestigationTools(argus.store, projects["support-bot"])
    inc = next(i for i in argus.store.incidents(projects["support-bot"]) if "llm_latency" in i.detectors)
    projects_cited = {e.ref for e in investigate(tools, inc.id).evidence if e.kind == "project"}
    assert projects_cited == {"research-agent", "claims-agent"}


def test_agent_regression_links_the_release(simulated) -> None:  # type: ignore[no-untyped-def]
    argus, _, projects = simulated
    tools = InvestigationTools(argus.store, projects["research-agent"])
    inc = next(i for i in argus.store.incidents(projects["research-agent"]) if "tool_loop" in i.detectors)
    changes = [e for e in investigate(tools, inc.id).evidence if e.kind == "change"]
    assert changes and "v1.4" in changes[0].detail


def test_validation_rejects_fabricated_evidence(simulated) -> None:  # type: ignore[no-untyped-def]
    argus, _, projects = simulated
    tools = InvestigationTools(argus.store, projects["support-bot"])
    inc = argus.store.incidents(projects["support-bot"])[0]
    fake = RCAReport(
        summary="s",
        category="tool_failure",
        cause_dimension="tool_name",
        cause_value="payments_api",
        evidence=[
            Evidence(kind="trace", ref="f" * 32),
            Evidence(kind="change", ref="99999"),
            Evidence(kind="attribute", ref="model=gpt-imaginary"),
        ],
        confidence=0.9,
    )
    errors = validate(fake, tools, inc.id)
    assert len(errors) == 4  # trace, change, attribute and the cause itself


def test_tenant_isolation_in_tools(simulated) -> None:  # type: ignore[no-untyped-def]
    argus, _, projects = simulated
    other_incident = argus.store.incidents(projects["claims-agent"])[0]
    tools = InvestigationTools(argus.store, projects["support-bot"])
    try:
        tools.incident_overview(other_incident.id)
    except KeyError:
        pass
    else:
        raise AssertionError("read another project's incident")


class ScriptedClient:
    """OpenAI-compatible client double that plays a scripted sequence of tool calls."""

    def __init__(self, script: list[tuple[str, dict[str, Any]]]) -> None:
        self.script, self.requests = list(script), []
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        name, args = self.script.pop(0)
        call = SimpleNamespace(
            id=f"call{len(self.requests)}", function=SimpleNamespace(name=name, arguments=json.dumps(args))
        )
        message = SimpleNamespace(
            content=None,
            tool_calls=[call],
            model_dump=lambda **_: {
                "role": "assistant",
                "tool_calls": [
                    {"id": call.id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                ],
            },
        )
        return SimpleNamespace(
            model="scripted",
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20),
            choices=[SimpleNamespace(message=message)],
        )


async def test_llm_investigator_loop_and_dogfooding(simulated) -> None:  # type: ignore[no-untyped-def]
    argus, _, projects = simulated
    pid = projects["support-bot"]
    tools = InvestigationTools(argus.store, pid)
    inc = next(i for i in argus.store.incidents(pid) if "tool_error_rate" in i.detectors)
    trace_id = tools.sample_traces(inc.id, k=1)[0]["trace_id"]
    report = RCAReport(
        summary="search_kb outage",
        category="tool_failure",
        cause_dimension="tool_name",
        cause_value="search_kb",
        confidence=0.85,
        evidence=[Evidence(kind="trace", ref=trace_id), Evidence(kind="attribute", ref="tool_name=search_kb")],
    )
    client = ScriptedClient(
        [
            ("incident_overview", {"incident_id": inc.id}),
            ("get_trace", {"trace_id": "does-not-exist"}),  # tool error is returned, not raised
            ("attribute_contrast", {"incident_id": inc.id}),
            ("submit_report", report.model_dump()),
        ]
    )
    result = await LLMInvestigator(client, "scripted-model", tracer=argus.self_tracer).investigate(tools, inc.id)
    assert result.report and result.report.cause_value == "search_kb" and result.validation_errors == []
    assert result.steps == 4
    tool_messages = [m for m in result.transcript if m.get("role") == "tool"]
    assert "KeyError" in tool_messages[1]["content"]
    argus.self_exporter.flush()
    internal = next(p["id"] for p in argus.store.projects() if p["name"] == INTERNAL_PROJECT)
    names = {r["name"] for r in argus.store.query("SELECT name FROM spans WHERE project_id = ?", (internal,))}
    assert {"investigate_incident", "incident_overview", "attribute_contrast", "chat scripted-model"} <= names


async def test_llm_investigator_budget(simulated) -> None:  # type: ignore[no-untyped-def]
    argus, _, projects = simulated
    tools = InvestigationTools(argus.store, projects["support-bot"])
    inc = argus.store.incidents(projects["support-bot"])[0]
    client = ScriptedClient([("compare_windows", {"incident_id": inc.id})] * 3)
    result = await LLMInvestigator(client, "m", max_steps=3).investigate(tools, inc.id)
    assert result.report is None and "budget" in result.validation_errors[0]
    assert client.requests[-1]["tool_choice"]["function"]["name"] == "submit_report"  # forced on the last step
