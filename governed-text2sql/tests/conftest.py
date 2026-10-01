from __future__ import annotations

from pathlib import Path

import pytest

from quill.config import Settings
from quill.data.questions import build_examples, write_splits
from quill.db.generate import generate
from quill.guard.policy import load_policies
from quill.guard.validator import SQLGuard
from quill.llm.base import Message
from quill.llm.factory import Models
from quill.llm.offline import OfflineLLM
from quill.pipeline import Text2SQL
from quill.semantic.layer import SemanticLayer

ROOT = Path(__file__).resolve().parents[1]


class ScriptedLLM:
    """Returns queued responses in order and records every prompt it received."""

    name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[list[Message]] = []

    async def complete(self, messages: list[Message]) -> str:
        self.prompts.append(list(messages))
        return self.responses.pop(0) if self.responses else "```sql\nSELECT 1\n```"


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("warehouse")
    db = generate(root / "claims.db", members=400, providers=60, claims=4000)
    write_splits(build_examples(db, per_template=10), root / "splits")
    return root


@pytest.fixture
def settings(warehouse: Path, tmp_path: Path) -> Settings:
    # Each test gets its own copy so write tests don't interfere.
    db = tmp_path / "claims.db"
    db.write_bytes((warehouse / "claims.db").read_bytes())
    return Settings(
        provider="offline",
        db_path=db,
        examples_path=warehouse / "splits" / "train.jsonl",
        semantic_layer=ROOT / "config" / "semantic_layer.yaml",
        policies=ROOT / "config" / "policies.yaml",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def pipeline(settings: Settings) -> Text2SQL:
    return Text2SQL.from_settings(settings)


def scripted_pipeline(settings: Settings, *responses: str) -> tuple[Text2SQL, ScriptedLLM]:
    llm = ScriptedLLM(list(responses))
    return Text2SQL.from_settings(settings, Models(sql=llm, answer=OfflineLLM())), llm


@pytest.fixture(scope="session")
def guard() -> SQLGuard:
    return SQLGuard(SemanticLayer.load(ROOT / "config" / "semantic_layer.yaml").sql_schema)


@pytest.fixture(scope="session")
def policies():  # type: ignore[no-untyped-def]
    return load_policies(ROOT / "config" / "policies.yaml")
