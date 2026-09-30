"""Pydantic schemas for every structured LLM call (constrained decoding on NIM)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["local", "global", "web", "direct"]


class QueryAnalysis(BaseModel):
    """Router output: where to look, and how to break the question down."""

    standalone_question: str = Field(
        description="The user question rewritten to be fully self-contained given the chat history."
    )
    route: Route = Field(
        description=(
            "local: specific facts/entities answerable from the knowledge base; "
            "global: corpus-wide themes or summaries; web: needs fresh/external info; "
            "direct: chit-chat or questions needing no retrieval."
        )
    )
    sub_questions: list[str] = Field(
        default_factory=list,
        description="1-3 focused sub-questions for multi-hop questions; empty if the question is atomic.",
    )
    rationale: str = Field(default="", description="One sentence explaining the routing decision.")


class DocumentGrades(BaseModel):
    """CRAG relevance grading over a batch of retrieved documents."""

    relevant_ids: list[str] = Field(description="IDs of documents containing information useful for answering.")
    reasoning: str = Field(default="", description="Brief justification.")


class RewrittenQuery(BaseModel):
    query: str = Field(description="A retrieval-optimized rewrite of the question.")


class GroundednessCheck(BaseModel):
    """Self-RAG reflection on a drafted answer."""

    grounded: bool = Field(description="Every factual claim in the answer is supported by the documents.")
    answers_question: bool = Field(description="The answer actually resolves the question.")
    unsupported_claims: list[str] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    name: str
    type: str = Field(description="One of PERSON, ORGANIZATION, PRODUCT, TECHNOLOGY, CONCEPT, METHOD, EVENT.")
    description: str = Field(default="", description="One sentence grounded in the text.")


class ExtractedRelation(BaseModel):
    source: str
    target: str
    type: str = Field(description="UPPER_SNAKE_CASE verb phrase, e.g. DEVELOPED_BY, PART_OF, USES.")
    description: str = ""


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class CommunityReport(BaseModel):
    title: str
    summary: str = Field(description="3-5 sentence summary of the entities and how they relate.")


class JudgeScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""
