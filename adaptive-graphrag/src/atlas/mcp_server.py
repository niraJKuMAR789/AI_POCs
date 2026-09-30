"""Model Context Protocol server: exposes the Atlas knowledge base to any MCP client
(Claude Desktop/Code, Cursor, Agno/LangGraph agents, ...).

Tools are split by intent so hosts can apply per-tool permissions:
  * read-only: `ask`, `search`, `explore_entity`, `list_documents`
  * write:     `ingest_text` (idempotent; re-ingesting unchanged text is a no-op)

Run over stdio (default) or Streamable HTTP:
    atlas mcp                       # stdio
    atlas mcp --http --port 8765    # http://localhost:8765/mcp
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

from mcp.server.mcpserver import Context, MCPServer
from mcp_types import ToolAnnotations

from atlas.agent import AtlasAgent
from atlas.config import Settings, get_settings
from atlas.ingest.loaders import load_text
from atlas.kb import KnowledgeBase
from atlas.models import normalize_entity_name
from atlas.retrieval.retriever import Retriever

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


@dataclass
class AtlasContext:
    kb: KnowledgeBase
    agent: AtlasAgent
    retriever: Retriever


def _state(ctx: Context) -> AtlasContext:
    return cast(AtlasContext, ctx.request_context.lifespan_context)


def build_server(settings: Settings | None = None, kb: KnowledgeBase | None = None) -> MCPServer:
    settings = settings or get_settings()

    # Static resources cannot receive a Context in MCP v2, so they read the lifespan state from here.
    current: dict[str, AtlasContext] = {}

    @asynccontextmanager
    async def lifespan(_: MCPServer) -> AsyncIterator[AtlasContext]:
        knowledge_base = kb or await KnowledgeBase.open(settings)
        current["state"] = AtlasContext(knowledge_base, AtlasAgent(knowledge_base), Retriever(knowledge_base))
        try:
            yield current["state"]
        finally:
            current.pop("state", None)
            if kb is None:
                await knowledge_base.close()

    server = MCPServer(
        name="atlas-graphrag",
        title="Atlas GraphRAG",
        instructions=(
            "Atlas answers questions over a private knowledge base using hybrid vector + knowledge-graph "
            "retrieval. Prefer `ask` for complete, cited answers. Use `search` or `explore_entity` when you "
            "want raw evidence to reason over yourself."
        ),
        lifespan=lifespan,
    )

    @server.tool(annotations=READ_ONLY)
    async def ask(question: str, ctx: Context, thread_id: str | None = None) -> dict[str, Any]:
        """Answer a question with the full adaptive GraphRAG agent. Returns a cited answer,
        the retrieval route taken, a groundedness verdict, and the agent's step trace.
        Pass the returned thread_id back to ask follow-up questions."""
        result = await _state(ctx).agent.ask(question, thread_id)
        return result.model_dump(exclude={"trace"}) | {"steps": [f"{t.node}: {t.summary}" for t in result.trace]}

    @server.tool(annotations=READ_ONLY)
    async def search(
        query: str, ctx: Context, mode: Literal["local", "global"] = "local", limit: int = 6
    ) -> list[dict[str, Any]]:
        """Hybrid retrieval without generation. `local` = reranked chunks + knowledge-graph facts;
        `global` = community reports summarizing whole topic clusters."""
        contexts = await _state(ctx).retriever.retrieve(query, mode)
        return [c.model_dump(include={"kind", "title", "source", "text", "score"}) for c in contexts[: max(1, limit)]]

    @server.tool(annotations=READ_ONLY)
    async def explore_entity(name: str, ctx: Context, hops: int = 1) -> dict[str, Any]:
        """Look up an entity in the knowledge graph and return its description and relationships
        within `hops` (1-3) steps."""
        kb = _state(ctx).kb
        key = normalize_entity_name(name)
        entities = await asyncio.to_thread(kb.graph.entities, [key])
        if not entities:
            matches = [k for k in kb.entity_keys if key and key in k][:10]
            return {"found": False, "suggestions": matches}
        relations = await asyncio.to_thread(kb.graph.neighborhood, [key], max(1, min(hops, 3)), 40)
        return {
            "found": True,
            "entity": entities[0].model_dump(exclude={"source_chunks"}),
            "relations": [r.as_fact() for r in relations],
        }

    @server.tool(annotations=READ_ONLY)
    async def list_documents(ctx: Context) -> list[dict[str, Any]]:
        """List ingested documents with chunk counts."""
        return _state(ctx).kb.docstore.documents()

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True))
    async def ingest_text(text: str, source: str, ctx: Context, title: str | None = None) -> dict[str, Any]:
        """Add or update a document (Markdown or plain text). `source` is its stable identifier;
        re-ingesting identical text is skipped."""
        report = await _state(ctx).kb.ingest([load_text(text, source, title)])
        return report.model_dump()

    @server.resource("atlas://stats", name="stats", mime_type="application/json")
    async def stats() -> dict[str, int]:
        """Knowledge-base size: documents, chunks, entities, relations, communities."""
        return current["state"].kb.stats()

    @server.resource("atlas://communities", name="communities", mime_type="application/json")
    async def communities() -> list[dict[str, Any]]:
        """Community reports: LLM summaries of the main topic clusters in the knowledge graph."""
        return [c.model_dump(include={"title", "summary"}) for c in current["state"].kb.graph.communities()]

    @server.prompt()
    def research_brief(topic: str) -> str:
        """Produce a cited research brief on a topic using the Atlas tools."""
        return (
            f"Write a research brief on: {topic}.\n"
            "1. Call `search` in global mode to understand the relevant topic clusters.\n"
            "2. Call `ask` for each key sub-question; use `explore_entity` to follow relationships.\n"
            "3. Synthesize a brief with sections, citing sources exactly as returned by the tools."
        )

    return server


def main(http: bool = False, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = build_server()
    if http:
        server.run("streamable-http", host=host, port=port)
    else:
        server.run()
