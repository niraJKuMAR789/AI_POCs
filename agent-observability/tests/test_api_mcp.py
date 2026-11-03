import json

import pytest
from fastapi.testclient import TestClient
from mcp import Client
from pydantic import SecretStr

from argus import semconv
from argus.api.app import create_app
from argus.mcp_server import build_server
from argus.sdk import init, span
from argus.sdk.exporter import HTTPExporter


@pytest.fixture
def client(argus):  # type: ignore[no-untyped-def]
    settings = argus.settings.model_copy(update={"admin_token": SecretStr("root")})
    with TestClient(create_app(settings, argus)) as c:
        yield c


def _key(client: TestClient, project: str, scopes: list[str]) -> tuple[str, str]:
    pid = client.post("/admin/projects", json={"name": project}, headers={"X-Admin-Token": "root"}).json()["project_id"]
    key = client.post(
        f"/admin/projects/{pid}/keys", json={"name": "k", "scopes": scopes}, headers={"X-Admin-Token": "root"}
    ).json()["api_key"]
    return pid, key


def test_sdk_to_collector_end_to_end(client: TestClient) -> None:
    _, key = _key(client, "support-bot", ["ingest", "read"])
    exporter = HTTPExporter("http://testserver", api_key=key, interval_s=3600, client=client)
    init(service="support-bot", exporter=exporter)
    with (
        span("support-bot", kind="agent"),
        span(
            "chat",
            kind="llm",
            attributes={
                semconv.REQUEST_MODEL: "nvidia/nemotron-3-super-120b-a12b",
                semconv.INPUT_TOKENS: 1000,
                semconv.OUTPUT_TOKENS: 200,
                semconv.INPUT: "my email is a@b.co",
            },
        ),
    ):
        pass
    exporter.flush()
    assert exporter.sent == 2 and exporter.dropped == 0
    headers = {"Authorization": f"Bearer {key}"}
    traces = client.get("/v1/traces", headers=headers).json()
    assert len(traces) == 1 and traces[0]["cost_usd"] == pytest.approx((1000 * 0.6 + 200 * 2.4) / 1e6)
    detail = client.get(f"/v1/traces/{traces[0]['trace_id']}", headers=headers).json()
    assert detail["content_included"] is False and "input_text" not in detail["spans"][0]
    summary = client.get("/v1/metrics/summary", headers=headers).json()
    assert summary["traces"] == 1 and summary["series"]
    exporter.shutdown()


def test_auth_scopes_and_tenant_isolation(client: TestClient) -> None:
    _, ingest_only = _key(client, "a", ["ingest"])
    _, reader_b = _key(client, "b", ["ingest", "read", "read:content"])
    assert client.get("/v1/traces").status_code == 401
    assert client.get("/v1/traces", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/v1/traces", headers={"Authorization": f"Bearer {ingest_only}"}).status_code == 403
    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "a" * 32,
                                "spanId": "b" * 16,
                                "name": "root",
                                "startTimeUnixNano": "1",
                                "endTimeUnixNano": "2",
                                "attributes": [{"key": "argus.input", "value": {"stringValue": "SSN 219-09-9999"}}],
                            }
                        ]
                    }
                ],
            }
        ]
    }
    assert (
        client.post("/v1/traces", json=payload, headers={"Authorization": f"Bearer {ingest_only}"}).json()["accepted"]
        == 1
    )
    # Project b cannot see project a's trace, even with the right trace id.
    assert client.get(f"/v1/traces/{'a' * 32}", headers={"Authorization": f"Bearer {reader_b}"}).status_code == 404
    assert client.post("/admin/projects", json={"name": "x"}).status_code == 403


def test_content_reads_are_logged(client: TestClient) -> None:
    _, key = _key(client, "c", ["ingest", "read", "read:content"])
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "c" * 32,
                                "spanId": "d" * 16,
                                "name": "root",
                                "startTimeUnixNano": "1",
                                "endTimeUnixNano": "2",
                                "attributes": [
                                    {"key": "argus.input", "value": {"stringValue": "card 4242 4242 4242 4242"}}
                                ],
                            }
                        ]
                    }
                ],
            }
        ]
    }
    client.post("/v1/traces", json=payload, headers=headers)
    detail = client.get(f"/v1/traces/{'c' * 32}?purpose=ticket-123", headers=headers).json()
    assert detail["content_included"] and detail["spans"][0]["input_text"] == "card [REDACTED:CREDIT_CARD]"
    log = client.get("/v1/content-access-log", headers=headers).json()
    assert log[0]["purpose"] == "ticket-123" and log[0]["trace_id"] == "c" * 32


def test_incident_api_and_investigation(simulated) -> None:  # type: ignore[no-untyped-def]
    argus, _, projects = simulated
    key = argus.store.create_key(projects["research-agent"], "oncall", ["read", "investigate"])
    with TestClient(create_app(argus.settings, argus)) as c:
        headers = {"Authorization": f"Bearer {key}"}
        incidents = c.get("/v1/incidents", headers=headers).json()
        loop = next(i for i in incidents if "tool_loop" in i["detectors"])
        result = c.post(f"/v1/incidents/{loop['id']}/investigate", headers=headers).json()
        assert result["report"]["category"] == "agent_regression" and result["validation_errors"] == []
        detail = c.get(f"/v1/incidents/{loop['id']}", headers=headers).json()
        assert detail["investigations"][-1]["model"] == "deterministic-rca" and detail["anomalies"]
        other = argus.store.incidents(projects["support-bot"])[0]
        assert c.post(f"/v1/incidents/{other.id}/investigate", headers=headers).status_code == 404


async def test_mcp_server(simulated) -> None:  # type: ignore[no-untyped-def]
    argus, _, projects = simulated
    read_key = argus.store.create_key(projects["claims-agent"], "mcp", ["read"])
    async with Client(build_server(argus, read_key)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert "investigate" not in names  # scope-gated
        result = await client.call_tool("governance_violations", {"limit": 5})
        content = result.structured_content or json.loads(result.content[0].text)
        rows = content.get("result", content) if isinstance(content, dict) else content
        assert rows and rows[0]["policy"] in {"sensitive_data", "unapproved_model"}
    full_key = argus.store.create_key(projects["claims-agent"], "mcp2", ["read", "investigate"])
    async with Client(build_server(argus, full_key)) as client:
        assert "investigate" in {t.name for t in (await client.list_tools()).tools}
