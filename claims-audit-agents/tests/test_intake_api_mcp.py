import json
from typing import Any

from fastapi.testclient import TestClient
from mcp import Client
from tests.conftest import pick

from aegis.api.app import create_app
from aegis.config import Settings
from aegis.intake.extract import ExtractedForm, OfflineVisionLLM, VisionExtractor
from aegis.intake.forms import render_claim_form
from aegis.mcp_server import build_server
from aegis.service import AuditService


async def test_vision_intake_roundtrip_and_validation(service: AuditService, labeled: list[dict[str, Any]]) -> None:
    claim = pick(labeled, "excess_units")
    extractor = VisionExtractor(OfflineVisionLLM(), service.reference)  # type: ignore[arg-type]
    good = await extractor.extract(render_claim_form(claim))
    assert good.ok and good.claim and good.claim.lines == claim.lines

    class Misread(OfflineVisionLLM):  # simulates a VLM transcription error
        async def complete_json(self, messages, schema) -> ExtractedForm:  # type: ignore[no-untyped-def]
            form = await super().complete_json(messages, schema)
            form.lines[0].charge += 100
            form.provider_npi = form.provider_npi[:-1] + str((int(form.provider_npi[-1]) + 3) % 10)
            return form

    bad = await VisionExtractor(Misread(), service.reference).extract(render_claim_form(claim))  # type: ignore[arg-type]
    assert not bad.ok
    assert any("total charge" in i for i in bad.issues) and any("check digit" in i for i in bad.issues)


def test_api_end_to_end(settings: Settings, service: AuditService, labeled: list[dict[str, Any]]) -> None:
    with TestClient(create_app(settings, service)) as client:
        assert client.get("/health").json()["ledger_verified"] is True
        claim = pick(labeled, "unbundled_ecg")
        body = client.post("/v1/claims", json=claim.model_dump(mode="json")).json()
        assert body["decision"]["outcome"] == "partially_approved"
        form = client.post(
            "/v1/claims/form", files={"image": ("c.png", render_claim_form(pick(labeled, "age_limit")), "image/png")}
        ).json()
        assert form["status"] == "audited" and form["result"]["decision"]["outcome"] == "denied"
        trail = client.get(f"/v1/audit/{claim.claim_id}").json()
        assert trail["chain_verified"] and trail["events"][0]["event"] == "claim_received"
        assert client.get("/v1/claims/NOPE").status_code == 404
        assert client.post(f"/v1/reviews/{claim.claim_id}", json={"reviewer": "x", "decisions": {}}).status_code == 409


async def test_mcp_tools(service: AuditService, labeled: list[dict[str, Any]]) -> None:
    async with Client(build_server(service)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {
            "audit_claim",
            "claim_status",
            "search_policy",
            "review_queue",
            "audit_trail",
            "submit_human_review",
        }
        assert tools["submit_human_review"].annotations.read_only_hint is False
        result = await client.call_tool("audit_claim", {"claim": pick(labeled, "duplicate").model_dump(mode="json")})
        payload = result.structured_content or json.loads(result.content[0].text)
        assert payload["decision"]["outcome"] == "denied"
        hits = await client.call_tool("search_policy", {"query": "thunderclap headache MRI"})
        found = hits.structured_content or json.loads(hits.content[0].text)
        sections = found.get("result", found) if isinstance(found, dict) else found
        assert sections[0]["id"] == "AEG-4.2"
