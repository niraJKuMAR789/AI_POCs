import json

from mcp import Client

from atlas.kb import KnowledgeBase
from atlas.mcp_server import build_server


def _payload(result):  # type: ignore[no-untyped-def]
    if result.structured_content is not None:
        content = result.structured_content
        return content.get("result", content) if isinstance(content, dict) else content
    return json.loads(result.content[0].text)


async def test_mcp_tools_and_resources(kb: KnowledgeBase) -> None:
    server = build_server(kb.settings, kb)
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {"ask", "search", "explore_entity", "list_documents", "ingest_text"}
        assert tools["ask"].annotations.read_only_hint is True
        assert tools["ingest_text"].annotations.read_only_hint is False

        answer = _payload(await client.call_tool("ask", {"question": "Which ISO standard is Kestrel certified to?"}))
        assert "ISO 3691-4" in answer["answer"] and answer["steps"]

        hits = _payload(await client.call_tool("search", {"query": "Kafka retention", "limit": 3}))
        assert 0 < len(hits) <= 3

        entity = _payload(await client.call_tool("explore_entity", {"name": "INC-2041"}))
        assert entity["found"] and entity["relations"]

        stats = await client.read_resource("atlas://stats")
        assert json.loads(stats.contents[0].text)["documents"] == 9
