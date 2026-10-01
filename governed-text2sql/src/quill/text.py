from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
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
        "what",
        "which",
        "who",
        "how",
        "many",
        "much",
        "show",
        "list",
        "give",
        "me",
        "per",
        "each",
        "all",
        "their",
        "there",
        "that",
        "this",
        "these",
        "those",
        "do",
        "does",
        "did",
        "have",
        "has",
    ]
)


def _stem(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0
