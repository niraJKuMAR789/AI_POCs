"""Evaluation metrics: retrieval metrics are exact; answer metrics use an LLM judge."""

from __future__ import annotations

import re

from atlas.models import RetrievedContext
from atlas.prompts import render_documents
from atlas.providers.base import LLM, Message
from atlas.schemas import JudgeScore

_ABSTAIN_RE = re.compile(
    r"(does not contain|doesn't contain|not (?:enough|sufficient) information|no information|"
    r"cannot (?:find|determine|answer)|not (?:mentioned|specified|provided|available))",
    re.I,
)

CORRECTNESS_SYSTEM = """You grade answers against a reference answer.
Score 1.0 if the answer contains all key facts of the reference with no contradictions,
0.5 if it is partially correct or misses important facts, 0.0 if it is wrong or contradicts the reference.
Intermediate values are allowed. Ignore style and citation markers."""

FAITHFULNESS_SYSTEM = """You check whether an answer is faithful to the provided documents.
Score = fraction of the answer's factual claims that are directly supported by the documents (0.0-1.0).
An answer stating that information is unavailable scores 1.0."""


def source_recall(retrieved_sources: list[str], expected: list[str]) -> float | None:
    if not expected:
        return None
    return len(set(expected) & set(retrieved_sources)) / len(set(expected))


def reciprocal_rank(retrieved_sources: list[str], expected: list[str]) -> float | None:
    if not expected:
        return None
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return 1.0 / rank
    return 0.0


def abstained(answer: str) -> bool:
    return bool(_ABSTAIN_RE.search(answer))


async def correctness(judge: LLM, question: str, answer: str, reference: str) -> float:
    messages: list[Message] = [
        {"role": "system", "content": CORRECTNESS_SYSTEM},
        {
            "role": "user",
            "content": f"<question>{question}</question>\n<reference>{reference}</reference>\n"
            f"<answer>{answer}</answer>",
        },
    ]
    return (await judge.structured(messages, JudgeScore)).score


async def faithfulness(judge: LLM, answer: str, contexts: list[RetrievedContext]) -> float:
    if not contexts:
        return 1.0 if abstained(answer) else 0.0
    messages: list[Message] = [
        {"role": "system", "content": FAITHFULNESS_SYSTEM},
        {"role": "user", "content": f"{render_documents(contexts)}\n<answer>{answer}</answer>"},
    ]
    return (await judge.structured(messages, JudgeScore)).score
