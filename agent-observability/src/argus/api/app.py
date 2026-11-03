"""Argus HTTP API: OTLP ingest, change events, query, incidents, investigations, governance, admin.

Authentication: `Authorization: Bearer <api key>`. Keys belong to one project and carry scopes:
  ingest · read · read:content (prompt/completion text; every read is logged) · investigate · admin
"""

# No `from __future__ import annotations`: FastAPI resolves Depends() in Annotated hints at runtime.
import asyncio
import contextlib
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel

from argus import __version__
from argus.config import Settings, get_settings
from argus.detect.stats import quantile
from argus.models import ChangeEvent
from argus.service import Argus


class Caller(BaseModel):
    project_id: str
    key_name: str
    scopes: set[str]


class ProjectBody(BaseModel):
    name: str
    org: str = "default"


class KeyBody(BaseModel):
    name: str
    scopes: list[str] = ["ingest", "read", "investigate"]


def create_app(settings: Settings | None = None, argus: Argus | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.argus = argus or Argus(settings)
        task = None
        if settings.detect_interval_s > 0:

            async def loop() -> None:
                while True:
                    await asyncio.sleep(settings.detect_interval_s)
                    await asyncio.to_thread(app.state.argus.detect_all)

            task = asyncio.create_task(loop())
        yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Argus: agent observability", version=__version__, lifespan=lifespan)

    def service(request: Request) -> Argus:
        return request.app.state.argus

    Svc = Annotated[Argus, Depends(service)]

    def caller(svc: Svc, authorization: Annotated[str | None, Header()] = None) -> Caller:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer API key")
        row = svc.store.resolve_key(authorization.split(" ", 1)[1].strip())
        if row is None:
            raise HTTPException(401, "invalid API key")
        return Caller(project_id=row["project_id"], key_name=row["name"], scopes=set(row["scopes"].split(",")))

    def need(scope: str) -> Any:
        def check(c: Annotated[Caller, Depends(caller)]) -> Caller:
            if scope not in c.scopes and "admin" not in c.scopes:
                raise HTTPException(403, f"API key lacks the '{scope}' scope")
            return c

        return check

    def admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
        expected = settings.admin_token.get_secret_value() if settings.admin_token else None
        if not expected or not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
            raise HTTPException(403, "admin token required")

    Ingest = Annotated[Caller, Depends(need("ingest"))]
    Read = Annotated[Caller, Depends(need("read"))]
    Investigate = Annotated[Caller, Depends(need("investigate"))]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    # --- ingest -------------------------------------------------------------------------------

    @app.post("/v1/traces")
    def ingest(payload: dict[str, Any], who: Ingest, svc: Svc) -> dict[str, Any]:
        result = svc.ingestor.ingest(who.project_id, payload)
        return {"partialSuccess": {"rejectedSpans": result.rejected}, **result.__dict__}

    @app.post("/v1/events")
    def add_event(event: ChangeEvent, who: Ingest, svc: Svc) -> dict[str, Any]:
        event.project_id = who.project_id
        return {"id": svc.store.add_change(event)}

    # --- query --------------------------------------------------------------------------------

    @app.get("/v1/traces")
    def list_traces(
        who: Read,
        svc: Svc,
        start_ns: int | None = None,
        end_ns: int | None = None,
        root: str | None = None,
        limit: int = Query(200, le=5000),
    ) -> list[dict[str, Any]]:
        return [t.model_dump() for t in svc.store.traces(who.project_id, start_ns, end_ns, root, limit, "DESC")]

    @app.get("/v1/traces/{trace_id}")
    def get_trace(trace_id: str, who: Read, svc: Svc, purpose: str = "debugging") -> dict[str, Any]:
        spans = svc.store.spans_for_trace(who.project_id, trace_id)
        if not spans:
            raise HTTPException(404, "trace not found")
        with_content = "read:content" in who.scopes or "admin" in who.scopes
        if with_content:
            svc.store.log_content_access(who.project_id, who.key_name, trace_id, purpose)
        exclude = None if with_content else {"input_text", "output_text"}
        return {
            "trace_id": trace_id,
            "content_included": with_content,
            "spans": [s.model_dump(exclude=exclude) | {"duration_ms": s.duration_ms} for s in spans],
        }

    @app.get("/v1/metrics/summary")
    def summary(
        who: Read, svc: Svc, bucket_min: int = 10, start_ns: int | None = None, end_ns: int | None = None
    ) -> dict[str, Any]:
        traces = svc.store.traces(who.project_id, start_ns, end_ns)
        size = bucket_min * 60 * 10**9
        buckets: dict[int, list[Any]] = {}
        for t in traces:
            buckets.setdefault(t.start_ns - t.start_ns % size, []).append(t)
        series = [
            {
                "bucket_ns": b,
                "traces": len(ts),
                "error_rate": sum(t.error_count > 0 for t in ts) / len(ts),
                "p95_ms": quantile([t.duration_ms for t in ts], 0.95),
                "cost_usd": sum(t.cost_usd for t in ts),
                "input_tokens": sum(t.input_tokens for t in ts),
            }
            for b, ts in sorted(buckets.items())
        ]
        n = max(1, len(traces))
        return {
            "traces": len(traces),
            "cost_usd": round(sum(t.cost_usd for t in traces), 4),
            "error_rate": round(sum(t.error_count > 0 for t in traces) / n, 4),
            "p95_ms": round(quantile([t.duration_ms for t in traces], 0.95), 1),
            "tokens": sum(t.input_tokens + t.output_tokens for t in traces),
            "series": series,
        }

    # --- incidents ------------------------------------------------------------------------------

    @app.post("/v1/detect")
    def detect(who: Investigate, svc: Svc) -> dict[str, Any]:
        return {"new_anomalies": len(svc.engine.run(who.project_id))}

    @app.get("/v1/incidents")
    def incidents(who: Read, svc: Svc, status: str | None = None) -> list[dict[str, Any]]:
        return [i.model_dump() for i in svc.store.incidents(who.project_id, status)]

    @app.get("/v1/incidents/{incident_id}")
    def incident(incident_id: int, who: Read, svc: Svc) -> dict[str, Any]:
        inc = svc.store.incident(who.project_id, incident_id)
        if inc is None:
            raise HTTPException(404, "incident not found")
        return {
            "incident": inc.model_dump(),
            "anomalies": [a.model_dump() for a in svc.store.anomalies(who.project_id, incident_id)],
            "investigations": svc.store.investigations(who.project_id, incident_id),
        }

    @app.post("/v1/incidents/{incident_id}/investigate")
    async def investigate(incident_id: int, who: Investigate, svc: Svc, llm: bool | None = None) -> dict[str, Any]:
        try:
            return await svc.investigate(who.project_id, incident_id, use_llm=llm)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    # --- governance --------------------------------------------------------------------------------

    @app.get("/v1/violations")
    def violations(who: Read, svc: Svc, limit: int = 500) -> list[dict[str, Any]]:
        return [v.model_dump() for v in svc.store.violations(who.project_id, limit)]

    @app.get("/v1/policy")
    def policy(who: Read, svc: Svc) -> dict[str, Any]:
        return svc.ingestor.policy_for(who.project_id).model_dump()

    @app.get("/v1/content-access-log")
    def access_log(who: Read, svc: Svc) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in svc.store.query(
                "SELECT ts_ns, key_name, trace_id, purpose FROM content_access_log WHERE project_id = ? "
                "ORDER BY ts_ns DESC LIMIT 500",
                (who.project_id,),
            )
        ]

    # --- admin -----------------------------------------------------------------------------------

    @app.post("/admin/projects", dependencies=[Depends(admin)])
    def create_project(body: ProjectBody, svc: Svc) -> dict[str, str]:
        return {"project_id": svc.store.create_project(body.name, body.org)}

    @app.post("/admin/projects/{project_id}/keys", dependencies=[Depends(admin)])
    def create_key(project_id: str, body: KeyBody, svc: Svc) -> dict[str, str]:
        if svc.store.project(project_id) is None:
            raise HTTPException(404, "project not found")
        return {
            "api_key": svc.store.create_key(project_id, body.name, body.scopes),
            "note": "store this key now; only its hash is kept",
        }

    return app
