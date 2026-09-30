import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from atlas.api.app import create_app
from atlas.kb import KnowledgeBase


@pytest.fixture
def client(kb: KnowledgeBase):  # type: ignore[no-untyped-def]
    with TestClient(create_app(kb.settings, kb)) as c:
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["stats"]["documents"] == 9


def test_ask(client: TestClient) -> None:
    body = client.post("/v1/ask", json={"question": "Which ISO standard is Kestrel certified to?"}).json()
    assert "ISO 3691-4" in body["answer"] and body["route"] == "local" and body["thread_id"]


def test_ask_validation(client: TestClient) -> None:
    assert client.post("/v1/ask", json={"question": ""}).status_code == 422


def test_ask_stream_sse(client: TestClient) -> None:
    with client.stream("POST", "/v1/ask/stream", json={"question": "What is the Kafka retention?"}) as response:
        events = [line.removeprefix("event: ").strip() for line in response.iter_lines() if line.startswith("event:")]
        assert response.status_code == 200
    assert events[0] == "step" and events[-1] == "final" and "token" in events


def test_ingest_and_graph(client: TestClient) -> None:
    report = client.post(
        "/v1/ingest",
        json={"documents": [{"text": "# Note\n\nZephyr Labs acquired Quill AI.", "source": "note.md"}]},
    ).json()
    assert report["documents"] == 1
    files = client.post(
        "/v1/ingest/files", files=[("files", ("x.md", b"# X\n\nHello Orion Systems.", "text/markdown"))]
    )
    assert files.status_code == 200
    bad = client.post("/v1/ingest/files", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 415
    graph = client.get("/v1/graph?limit=20").json()
    assert 0 < len(graph["nodes"]) <= 20
    assert json.dumps(client.get("/v1/communities").json())


def test_api_key_enforced(kb: KnowledgeBase) -> None:
    settings = kb.settings.model_copy(update={"api_key": SecretStr("s3cret")})
    with TestClient(create_app(settings, kb)) as c:
        assert c.get("/health").status_code == 200
        assert c.post("/v1/ask", json={"question": "hi"}).status_code == 401
        ok = c.post("/v1/ask", json={"question": "hi"}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200
