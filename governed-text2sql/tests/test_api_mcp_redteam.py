import json

from fastapi.testclient import TestClient
from mcp import Client

from quill.api.app import create_app
from quill.config import Settings
from quill.evals.runner import run_redteam
from quill.mcp_server import build_server
from quill.pipeline import Text2SQL


def test_api_roles_from_headers_in_dev(settings: Settings, pipeline: Text2SQL) -> None:
    with TestClient(create_app(settings, pipeline)) as client:
        assert client.get("/health").json()["status"] == "ok"
        ok = client.post("/v1/ask", json={"question": "How many denied claims are there?"}).json()
        assert ok["status"] == "ok" and ok["role"] == "analyst"
        blocked = client.post("/v1/sql", json={"sql": "SELECT ssn_last4 FROM members"}).json()
        assert blocked["status"] == "blocked"
        schema = client.get("/v1/schema", headers={"X-Quill-Role": "provider_portal"}).json()
        assert "TABLE members" not in schema["ddl"] and "TABLE claims" in schema["ddl"]
        denied = client.post(
            "/v1/writes/confirm",
            json={"sql": "INSERT INTO audit_flags (claim_id, reason, flagged_by) VALUES (1, 'x', 'y')"},
        )
        assert denied.status_code == 403


def test_api_principals_bind_role_to_key(settings: Settings, pipeline: Text2SQL) -> None:
    secured = settings.model_copy(
        update={"api_principals": {"portal-key": {"role": "provider_portal", "context": {"provider_id": 7}}}}
    )
    with TestClient(create_app(secured, pipeline)) as client:
        assert client.post("/v1/sql", json={"sql": "SELECT 1"}).status_code == 401
        # The role header is ignored when keys are configured: the key decides.
        body = client.post(
            "/v1/sql",
            json={"sql": "SELECT COUNT(*) FROM claims"},
            headers={"X-API-Key": "portal-key", "X-Quill-Role": "claims_auditor"},
        ).json()
        assert body["role"] == "provider_portal" and "provider_id = 7" in body["executed_sql"]


async def test_mcp_server_is_bound_to_role(settings: Settings, pipeline: Text2SQL) -> None:
    server = build_server(settings, pipeline=pipeline, role="provider_portal", context={"provider_id": 7})
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == {"ask", "describe_schema", "list_metrics", "run_sql", "whoami"}

        def payload(result):  # type: ignore[no-untyped-def]
            content = result.structured_content
            if content is not None:
                return content.get("result", content) if isinstance(content, dict) else content
            return json.loads(result.content[0].text)

        who = payload(await client.call_tool("whoami", {}))
        assert who["role"] == "provider_portal" and who["context"] == {"provider_id": 7}
        ran = payload(await client.call_tool("run_sql", {"sql": "SELECT COUNT(*) FROM claims"}))
        assert ran["status"] == "ok" and "provider_id = 7" in ran["executed_sql"]
        blocked = payload(await client.call_tool("run_sql", {"sql": "SELECT * FROM members"}))
        assert blocked["status"] == "blocked"
        schema = (await client.call_tool("describe_schema", {})).content[0].text
        assert "TABLE members" not in schema


async def test_redteam_suite_has_no_leaks(pipeline: Text2SQL) -> None:
    rows = await run_redteam(pipeline)
    assert len(rows) == 10
    assert not [r.id for r in rows if r.leaked]


async def test_leak_detector_catches_a_broken_guard(settings: Settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from tests.conftest import scripted_pipeline

    from quill.guard.validator import GuardResult

    pipe, _ = scripted_pipeline(settings, *["```sql\nSELECT first_name, last_name FROM members\n```"] * 20)
    monkeypatch.setattr(pipe.guard, "check", lambda sql, *a, **k: GuardResult(ok=True, sql=sql, statement="SELECT"))
    rows = await run_redteam(pipe)
    assert next(r for r in rows if r.id == "pii-names").leaked
