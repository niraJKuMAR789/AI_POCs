"""Argus service: the store, ingest pipeline, detection engine and investigators wired together."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from argus.config import Settings
from argus.detect.engine import DetectionEngine
from argus.governance.policy import PolicyBook, Pricing
from argus.ingest.pipeline import Ingestor
from argus.investigate import deterministic
from argus.investigate.report import validate
from argus.investigate.tools import InvestigationTools
from argus.sdk.otlp import encode
from argus.sdk.tracer import SpanRecord, Tracer
from argus.store.sqlite import Store

INTERNAL_PROJECT = "argus-investigator"


class InProcessExporter:
    """Sends Argus's own spans (the investigator agent) through the normal ingest pipeline."""

    def __init__(self, ingestor: Ingestor, project_id: str) -> None:
        self.ingestor, self.project_id = ingestor, project_id
        self.buffer: list[tuple[SpanRecord, dict[str, Any]]] = []

    def export(self, record: SpanRecord, resource: dict[str, Any]) -> None:
        self.buffer.append((record, resource))

    def flush(self) -> None:
        if self.buffer:
            batch, self.buffer = self.buffer, []
            self.ingestor.ingest(self.project_id, encode(batch))


@dataclass
class Argus:
    settings: Settings
    store: Store = field(init=False)
    ingestor: Ingestor = field(init=False)
    engine: DetectionEngine = field(init=False)

    def __post_init__(self) -> None:
        self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.settings.db_path)
        self.ingestor = Ingestor(
            self.store, PolicyBook.load(self.settings.policies), Pricing.load(self.settings.pricing)
        )
        self.engine = DetectionEngine(self.store)
        internal = self.store.create_project(INTERNAL_PROJECT, org="argus")
        self.self_exporter = InProcessExporter(self.ingestor, internal)
        self.self_tracer = Tracer("argus-investigator", self.self_exporter)  # scoped; never the global tracer

    def detect_all(self) -> int:
        return sum(len(self.engine.run(p["id"])) for p in self.store.projects() if p["name"] != INTERNAL_PROJECT)

    async def investigate(self, project_id: str, incident_id: int, *, use_llm: bool | None = None) -> dict[str, Any]:
        tools = InvestigationTools(self.store, project_id)
        tools.incident(incident_id)  # raises KeyError for another tenant's incident
        use_llm = (not self.settings.offline) if use_llm is None else use_llm
        if use_llm:
            from openai import AsyncOpenAI

            from argus.investigate.agent import LLMInvestigator

            key = self.settings.nvidia_api_key.get_secret_value() if self.settings.nvidia_api_key else None
            investigator = LLMInvestigator(
                AsyncOpenAI(base_url=self.settings.base_url, api_key=key),
                self.settings.investigator_model,
                tracer=self.self_tracer,
            )
            result = await investigator.investigate(tools, incident_id)
            self.self_exporter.flush()
            model, report, errors = self.settings.investigator_model, result.report, result.validation_errors
        else:
            report = deterministic.investigate(tools, incident_id)
            model, errors = "deterministic-rca", validate(report, tools, incident_id)
        payload = report.model_dump() if report else {}
        investigation_id = self.store.save_investigation(project_id, incident_id, model, payload, errors)
        return {"investigation_id": investigation_id, "model": model, "report": payload, "validation_errors": errors}
