from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from atlas.config import Settings
from atlas.ingest.loaders import iter_documents
from atlas.kb import KnowledgeBase
from atlas.providers.base import Message, T
from atlas.providers.factory import Providers
from atlas.providers.offline import HashEmbedder, OfflineLLM, OverlapReranker

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


class ScriptedLLM(OfflineLLM):
    """Offline LLM whose structured outputs can be overridden per schema (callable or queue)."""

    def __init__(self) -> None:
        super().__init__("scripted")
        self.overrides: dict[type[Any], Callable[[], Any] | list[Any]] = {}
        self.calls: list[str] = []

    async def structured(self, messages: list[Message], schema: type[T]) -> T:
        self.calls.append(schema.__name__)
        override = self.overrides.get(schema)
        if isinstance(override, list) and override:
            return override.pop(0)
        if callable(override):
            return override()
        return await super().structured(messages, schema)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(provider="offline", data_dir=tmp_path / "atlas", _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
async def kb(settings: Settings, llm: ScriptedLLM) -> AsyncIterator[KnowledgeBase]:
    providers = Providers(generator=llm, fast=llm, embedder=HashEmbedder(), reranker=OverlapReranker())
    knowledge_base = await KnowledgeBase.open(settings, providers)
    await knowledge_base.ingest(iter_documents(CORPUS))
    yield knowledge_base
    await knowledge_base.close()
