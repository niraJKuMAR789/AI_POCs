"""FastAPI service: ingestion, question answering (JSON + SSE streaming) and graph introspection."""

# No `from __future__ import annotations` here: FastAPI must resolve the Annotated[..., Depends(...)]
# hints that reference closures inside create_app().
import json
import secrets
import tempfile
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from atlas import __version__
from atlas.agent import AtlasAgent
from atlas.config import Settings, get_settings
from atlas.ingest.loaders import SUPPORTED, load_file, load_text
from atlas.kb import IngestReport, KnowledgeBase
from atlas.logging import configure_logging
from atlas.models import AnswerResult


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=128)


class TextDocument(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(min_length=1, max_length=512)
    title: str | None = None


class IngestRequest(BaseModel):
    documents: list[TextDocument] = Field(min_length=1, max_length=500)
    force: bool = False


def create_app(settings: Settings | None = None, kb: KnowledgeBase | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        owned = kb is None
        app.state.kb = kb or await KnowledgeBase.open(settings)
        app.state.agent = AtlasAgent(app.state.kb)
        yield
        if owned:
            await app.state.kb.close()

    app = FastAPI(
        title="Atlas: Adaptive GraphRAG",
        version=__version__,
        description="Adaptive, corrective, self-reflective GraphRAG agent on NVIDIA NIM.",
        lifespan=lifespan,
    )
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_key(key: Annotated[str | None, Security(api_key_header)]) -> None:
        expected = settings.api_key.get_secret_value() if settings.api_key else None
        if expected and not (key and secrets.compare_digest(key, expected)):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def get_kb(request: Request) -> KnowledgeBase:
        return request.app.state.kb

    def get_agent(request: Request) -> AtlasAgent:
        return request.app.state.agent

    protected = [Depends(require_key)]

    @app.get("/health")
    async def health(kb: Annotated[KnowledgeBase, Depends(get_kb)]) -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "provider": settings.provider, "stats": kb.stats()}

    @app.post("/v1/ask", response_model=AnswerResult, dependencies=protected)
    async def ask(body: AskRequest, agent: Annotated[AtlasAgent, Depends(get_agent)]) -> AnswerResult:
        return await agent.ask(body.question, body.thread_id)

    @app.post("/v1/ask/stream", dependencies=protected)
    async def ask_stream(body: AskRequest, agent: Annotated[AtlasAgent, Depends(get_agent)]) -> EventSourceResponse:
        async def events() -> AsyncIterator[dict[str, str]]:
            async for event in agent.astream(body.question, body.thread_id):
                yield {"event": event["type"], "data": json.dumps(event, default=str)}

        return EventSourceResponse(events())

    @app.post("/v1/ingest", response_model=IngestReport, dependencies=protected)
    async def ingest(body: IngestRequest, kb: Annotated[KnowledgeBase, Depends(get_kb)]) -> IngestReport:
        docs = [load_text(d.text, d.source, d.title) for d in body.documents]
        return await kb.ingest(docs, force=body.force)

    @app.post("/v1/ingest/files", response_model=IngestReport, dependencies=protected)
    async def ingest_files(
        files: Annotated[list[UploadFile], File()], kb: Annotated[KnowledgeBase, Depends(get_kb)]
    ) -> IngestReport:
        docs = []
        with tempfile.TemporaryDirectory() as tmp:
            for upload in files:
                name = Path(upload.filename or "upload.txt").name
                if Path(name).suffix.lower() not in SUPPORTED:
                    raise HTTPException(415, f"Unsupported file type: {name}")
                path = Path(tmp) / name
                path.write_bytes(await upload.read())
                doc = load_file(path)
                docs.append(doc.model_copy(update={"id": load_text("", name).id, "source": name}))
        return await kb.ingest(docs)

    @app.get("/v1/documents", dependencies=protected)
    async def documents(kb: Annotated[KnowledgeBase, Depends(get_kb)]) -> list[dict[str, Any]]:
        return kb.docstore.documents()

    @app.get("/v1/communities", dependencies=protected)
    async def communities(kb: Annotated[KnowledgeBase, Depends(get_kb)]) -> list[dict[str, Any]]:
        return [c.model_dump() for c in kb.graph.communities()]

    @app.get("/v1/graph", dependencies=protected)
    async def graph(kb: Annotated[KnowledgeBase, Depends(get_kb)], limit: int = 60) -> dict[str, Any]:
        """Top-`limit` entities by weighted degree, with the edges among them (for visualization)."""
        edges = kb.graph.weighted_edges()
        degree: defaultdict[str, float] = defaultdict(float)
        for u, v, w in edges:
            degree[u] += w
            degree[v] += w
        ranked = sorted(degree, key=degree.__getitem__, reverse=True)
        top = set(ranked[: max(1, min(limit, 300))])
        entities = {e.key: e for e in kb.graph.entities(list(top))}
        return {
            "nodes": [
                {"id": k, "label": e.name, "type": e.type, "degree": degree[k], "description": e.description}
                for k, e in entities.items()
            ],
            "edges": [{"source": u, "target": v, "weight": w} for u, v, w in edges if u in top and v in top],
        }

    return app
