"""HTTP API. The caller's role and tenant context come from its API key, never from the request body."""

# No `from __future__ import annotations`: FastAPI resolves Depends() in Annotated hints at runtime.
import json
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from quill import __version__
from quill.config import Settings, get_settings
from quill.llm.base import Message
from quill.pipeline import AskResult, Text2SQL


class Principal(BaseModel):
    role: str
    context: dict[str, Any] = {}


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=10)


class SQLBody(BaseModel):
    sql: str = Field(min_length=1, max_length=10_000)


def create_app(settings: Settings | None = None, pipeline: Text2SQL | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline or Text2SQL.from_settings(settings)
        yield

    app = FastAPI(title="Quill: Governed Text-to-SQL", version=__version__, lifespan=lifespan)

    def get_pipeline(request: Request) -> Text2SQL:
        return request.app.state.pipeline

    def principal(
        x_api_key: Annotated[str | None, Header()] = None,
        x_quill_role: Annotated[str | None, Header()] = None,
        x_quill_context: Annotated[str | None, Header()] = None,
    ) -> Principal:
        if settings.api_principals:
            for key, spec in settings.api_principals.items():
                if x_api_key and secrets.compare_digest(x_api_key, key):
                    return Principal.model_validate(spec)
            raise HTTPException(401, "invalid or missing API key")
        context = json.loads(x_quill_context) if x_quill_context else {}
        return Principal(role=x_quill_role or settings.default_role, context=context)

    Caller = Annotated[Principal, Depends(principal)]
    Pipe = Annotated[Text2SQL, Depends(get_pipeline)]

    def _policy_or_403(pipe: Text2SQL, who: Principal) -> None:
        if who.role not in pipe.policies:
            raise HTTPException(403, f"unknown role '{who.role}'")

    @app.get("/health")
    def health(pipe: Pipe) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "provider": settings.provider,
            "sql_model": pipe.models.sql.name,
            "roles": sorted(pipe.policies),
        }

    @app.post("/v1/ask", response_model=AskResult)
    async def ask(body: AskBody, who: Caller, pipe: Pipe) -> AskResult:
        _policy_or_403(pipe, who)
        history: list[Message] = [
            {"role": "user" if h["role"] == "user" else "assistant", "content": h["content"]}
            for h in body.history
            if h.get("role") in {"user", "assistant"}
        ]
        return await pipe.ask(body.question, role=who.role, context=who.context, history=history)

    @app.post("/v1/sql", response_model=AskResult)
    def run_sql(body: SQLBody, who: Caller, pipe: Pipe) -> AskResult:
        _policy_or_403(pipe, who)
        return pipe.run_sql(body.sql, role=who.role, context=who.context)

    @app.post("/v1/writes/confirm")
    def confirm(body: SQLBody, who: Caller, pipe: Pipe) -> dict[str, Any]:
        _policy_or_403(pipe, who)
        try:
            return {"rows_written": pipe.confirm_write(body.sql, role=who.role, context=who.context)}
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @app.get("/v1/schema")
    def schema(who: Caller, pipe: Pipe) -> dict[str, Any]:
        _policy_or_403(pipe, who)
        policy = pipe.policy(who.role)
        tables = [t for t in policy.tables if t in pipe.layer.tables]
        return {
            "role": who.role,
            "ddl": pipe.layer.ddl(tables, hidden_columns=policy.denied),
            "metrics": {k: v.description for k, v in pipe.layer.metrics.items()},
        }

    return app
