"""HTTP API: submit claims (JSON or form image), work the human-review queue, read the audit trail."""

# No `from __future__ import annotations`: FastAPI resolves Depends() in Annotated hints at runtime.
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from aegis import __version__
from aegis.config import Settings, get_settings
from aegis.intake.extract import VisionExtractor
from aegis.models import Claim
from aegis.service import AuditResult, AuditService


class ReviewBody(BaseModel):
    reviewer: str = Field(min_length=1, max_length=100)
    decisions: dict[int, str]
    note: str = Field(default="", max_length=2000)


def create_app(
    settings: Settings | None = None, service: AuditService | None = None, extractor: VisionExtractor | None = None
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or await AuditService.create(settings)
        app.state.extractor = extractor or _default_extractor(settings, app.state.service)
        yield
        if service is None:
            await app.state.service.close()

    app = FastAPI(title="Aegis: Multi-agent Claims Auditor", version=__version__, lifespan=lifespan)

    def svc(request: Request) -> AuditService:
        return request.app.state.service

    Svc = Annotated[AuditService, Depends(svc)]

    @app.get("/health")
    def health(service: Svc) -> dict[str, Any]:
        ok, _ = service.ledger.verify()
        return {
            "status": "ok",
            "version": __version__,
            "provider": settings.provider,
            "reviewers": {k: v.name for k, v in service.llms.items()},
            "ledger_verified": ok,
        }

    @app.post("/v1/claims", response_model=AuditResult)
    async def submit(claim: Claim, service: Svc) -> AuditResult:
        return await service.audit(claim)

    @app.post("/v1/claims/form")
    async def submit_form(
        request: Request,
        service: Svc,
        image: Annotated[UploadFile, File()],
        clinical_note: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        intake = await request.app.state.extractor.extract(await image.read(), clinical_note)
        if not intake.ok:
            return {
                "status": "intake_rejected",
                "issues": intake.issues,
                "extracted": intake.extracted.model_dump(mode="json") if intake.extracted else None,
            }
        result = await service.audit(intake.claim)
        return {"status": "audited", "result": result.model_dump(mode="json")}

    @app.get("/v1/claims/{claim_id}", response_model=AuditResult)
    async def status(claim_id: str, service: Svc) -> AuditResult:
        try:
            return await service.status(claim_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/v1/reviews")
    async def queue(service: Svc) -> list[dict[str, Any]]:
        items = []
        for claim_id in service.ledger.pending_claims():
            result = await service.status(claim_id)
            if result.pending:
                items.append(result.pending.model_dump(mode="json"))
        return items

    @app.post("/v1/reviews/{claim_id}", response_model=AuditResult)
    async def decide(claim_id: str, body: ReviewBody, service: Svc) -> AuditResult:
        try:
            return await service.resume(claim_id, reviewer=body.reviewer, decisions=body.decisions, note=body.note)
        except (ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/v1/audit/{claim_id}")
    def trail(claim_id: str, service: Svc) -> dict[str, Any]:
        ok, bad = service.ledger.verify()
        return {
            "claim_id": claim_id,
            "chain_verified": ok,
            "first_bad_seq": bad,
            "events": service.ledger.events(claim_id),
        }

    return app


def _default_extractor(settings: Settings, service: AuditService) -> VisionExtractor:
    if settings.offline:
        from aegis.intake.extract import OfflineVisionLLM

        return VisionExtractor(OfflineVisionLLM(), service.reference)  # type: ignore[arg-type]
    from aegis.llm.openai_compat import OpenAICompatibleLLM

    key = settings.nvidia_api_key.get_secret_value() if settings.nvidia_api_key else None
    return VisionExtractor(
        OpenAICompatibleLLM(settings.vision_model, base_url=settings.base_url, api_key=key, guided_json=False),
        service.reference,
    )
