"""Small, dependency-free text utilities used by BM25, offline providers and evals."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "into",
        "over",
        "under",
        "about",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "will",
        "shall",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "only",
        "own",
        "same",
        "such",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "any",
        "all",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "again",
        "further",
        "once",
        "tell",
        "give",
        "explain",
        "describe",
        "show",
        "list",
    ]
)


def _stem(token: str) -> str:
    if token.endswith(("ss", "us", "is")):
        return token
    for suffix in ("ies", "s"):
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = [_stem(t) for t in _TOKEN_RE.findall(text.lower())]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.replace("\n", " ")) if s.strip()]


def overlap_score(query: str, text: str) -> float:
    """Fraction of distinct query terms present in `text` (0..1)."""
    q = set(tokenize(query))
    if not q:
        return 0.0
    return len(q & set(tokenize(text))) / len(q)


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.S)
    return match.group(1) if match else ""
