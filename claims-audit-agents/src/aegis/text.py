from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "not",
        "no",
        "than",
        "at",
        "least",
        "such",
        "when",
        "which",
        "who",
        "per",
    ]
)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]
