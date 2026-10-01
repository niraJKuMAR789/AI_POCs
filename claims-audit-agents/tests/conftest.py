from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from aegis.agents.reviewers import ProposedReview, ReviewBatch
from aegis.config import Settings
from aegis.data.generate import write
from aegis.llm.base import Message, T
from aegis.models import Claim, Resolution
from aegis.service import AuditService

ROOT = Path(__file__).resolve().parents[1]


class ScriptedReviewer:
    """Reviewer LLM double: answers every finding in the prompt with a scripted resolution."""

    def __init__(
        self,
        resolution: Resolution = Resolution.OVERRIDE,
        confidence: float = 0.95,
        quote: str | None = None,
        cite_finding_section: bool = True,
        extra: dict[str, Any] | None = None,
    ):
        self.name = "scripted-reviewer"
        self.resolution, self.confidence, self.quote = resolution, confidence, quote
        self.cite, self.extra = cite_finding_section, extra or {}
        self.calls: list[list[Message]] = []

    async def complete_json(self, messages: list[Message], schema: type[T]) -> T:
        import re

        self.calls.append(messages)
        text = str(messages[-1]["content"])
        note = re.search(r"<clinical_note>\n(.*?)\n</clinical_note>", text, re.S)
        reviews = []
        for rule_id, line_no, section in re.findall(
            r'<finding rule_id="(\w+)" line_no="(\w+)" section="([\w.-]+)">', text
        ):
            quote = self.quote if self.quote is not None else (note.group(1).split(".")[0] if note else "")
            reviews.append(
                ProposedReview(
                    rule_id=rule_id,
                    line_no=None if line_no == "None" else int(line_no),
                    resolution=self.resolution,
                    confidence=self.confidence,
                    rationale="scripted",
                    policy_citations=[section] if self.cite else [],
                    evidence_quotes=[quote] if quote else [],
                    **self.extra,
                )
            )
        return ReviewBatch(reviews=reviews)  # type: ignore[return-value]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("data")
    write(out, n_per_scenario=2)
    return out


@pytest.fixture
def settings(data_dir: Path, tmp_path: Path) -> Settings:
    return Settings(
        provider="offline",
        reference=ROOT / "config" / "reference.yaml",
        policy_manual=ROOT / "config" / "policy_manual.md",
        registry=data_dir / "registry.json",
        state_dir=tmp_path / "state",
        _env_file=None,
    )  # type: ignore[call-arg]


@pytest.fixture
def labeled(data_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (data_dir / "claims_labeled.jsonl").read_text().splitlines()]


def pick(labeled: list[dict[str, Any]], scenario: str) -> Claim:
    return Claim.model_validate(next(r["claim"] for r in labeled if r["scenario"] == scenario))


@pytest.fixture
async def service(settings: Settings) -> AsyncIterator[AuditService]:
    svc = await AuditService.create(settings, durable=False)
    yield svc
    await svc.close()


async def scripted_service(settings: Settings, reviewer: ScriptedReviewer, durable: bool = False) -> AuditService:
    return await AuditService.create(settings, llms={"clinical": reviewer, "coding": reviewer}, durable=durable)
