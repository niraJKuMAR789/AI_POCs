"""Domain models shared across ingestion, retrieval, the agent and the API."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, Field


def stable_id(*parts: str, length: int = 16) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:length]


def normalize_entity_name(name: str) -> str:
    """Canonical key used for entity resolution ("NVIDIA Corp." == "nvidia corp")."""
    name = re.sub(r"[^\w\s-]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


class SourceDocument(BaseModel):
    id: str
    source: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    doc_id: str
    source: str
    title: str
    section: str = ""
    text: str
    position: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    name: str
    type: str = "CONCEPT"
    description: str = ""
    source_chunks: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return normalize_entity_name(self.name)


class Relation(BaseModel):
    source: str
    target: str
    type: str = "RELATED_TO"
    description: str = ""
    weight: float = 1.0
    source_chunks: list[str] = Field(default_factory=list)

    def as_fact(self) -> str:
        detail = f" ({self.description})" if self.description else ""
        return f"{self.source} -[{self.type}]-> {self.target}{detail}"


class Community(BaseModel):
    id: str
    level: int = 0
    entities: list[str]
    title: str = ""
    summary: str = ""


ContextKind = Literal["chunk", "graph", "community", "web"]


class RetrievedContext(BaseModel):
    """A unit of evidence handed to the generator, with provenance for citations."""

    id: str
    kind: ContextKind
    text: str
    source: str
    title: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render(self, label: str) -> str:
        header = f"[{label}] ({self.kind}) {self.title or self.source}"
        return f"{header}\n{self.text.strip()}"


class Citation(BaseModel):
    label: str
    source: str
    title: str
    kind: ContextKind
    snippet: str


class TraceStep(BaseModel):
    node: str
    summary: str
    latency_ms: float
    data: dict[str, Any] = Field(default_factory=dict)


class AnswerResult(BaseModel):
    question: str
    answer: str
    route: str
    citations: list[Citation]
    grounded: bool
    trace: list[TraceStep]
    thread_id: str
