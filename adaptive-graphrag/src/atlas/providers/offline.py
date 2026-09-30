"""Deterministic offline providers.

These let the full agent graph (routing, CRAG grading, rewrites, self-RAG
reflection, graph extraction) run end-to-end without network access or an API
key: in CI, in unit tests, and for keyless demos. They are heuristic stand-ins,
not models; quality numbers in the README always come from the NVIDIA stack.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import AsyncIterator, Callable
from itertools import combinations, pairwise
from typing import Any

from atlas.providers.base import Message, T
from atlas.schemas import (
    CommunityReport,
    DocumentGrades,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    GroundednessCheck,
    JudgeScore,
    QueryAnalysis,
    RewrittenQuery,
    Route,
)
from atlas.text import extract_tag, overlap_score, split_sentences, tokenize

_DOC_RE = re.compile(r'<doc id="([^"]+)"[^>]*>\s*(.*?)\s*</doc>', flags=re.S)
_DOC_KIND_RE = re.compile(r'<doc id="([^"]+)" kind="([^"]+)"')
_CITE_RE = re.compile(r"\[[^\]]{1,6}\]")
_ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:[-\s][A-Z0-9][A-Za-z0-9]*)*)\b")
_NOT_ENTITIES = frozenset(
    [
        "The",
        "This",
        "That",
        "These",
        "Those",
        "It",
        "Its",
        "In",
        "On",
        "At",
        "For",
        "With",
        "From",
        "By",
        "As",
        "An",
        "A",
        "And",
        "Or",
        "But",
        "If",
        "When",
        "Where",
        "While",
        "However",
        "Because",
        "Although",
        "Each",
        "Every",
        "Some",
        "Many",
        "Most",
        "All",
        "Both",
        "Unlike",
        "Like",
        "Today",
        "Instead",
    ]
)
_GLOBAL_HINTS = ("overall", "themes", "main topics", "summarize", "summary of", "across the", "big picture")
_WEB_HINTS = ("latest", "today", "this week", "current price", "news", "right now")
_DIRECT_HINTS = ("hello", "hi ", "hey", "thanks", "thank you", "who are you", "what can you do")
INSUFFICIENT = "The knowledge base does not contain enough information to answer this question."


def _user_text(messages: list[Message]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


def _docs(text: str) -> list[tuple[str, str]]:
    return _DOC_RE.findall(text)


class OfflineLLM:
    def __init__(self, name: str = "offline-heuristic") -> None:
        self.name = name
        self._handlers: dict[type[Any], Callable[[str], Any]] = {
            QueryAnalysis: self._analyze,
            DocumentGrades: self._grade,
            RewrittenQuery: self._rewrite,
            GroundednessCheck: self._grounding,
            ExtractionResult: self._extract,
            CommunityReport: self._community,
            JudgeScore: self._judge,
        }

    async def generate(self, messages: list[Message]) -> str:
        text = _user_text(messages)
        question = extract_tag(text, "question")
        docs = _docs(text)
        if "<documents>" in text and not docs:
            return INSUFFICIENT
        if not docs:
            return "Hi! I'm Atlas. Ask me a question about the documents in the knowledge base."
        scored: list[tuple[float, int, str, str]] = []
        for order, (doc_id, body) in enumerate(docs):
            for sentence in (s for line in body.splitlines() for s in split_sentences(line.lstrip("- "))):
                score = overlap_score(question, sentence)
                if score > 0:
                    scored.append((score, -order, doc_id, sentence))
        if not scored:
            return INSUFFICIENT
        scored.sort(reverse=True)
        seen: set[str] = set()
        parts: list[str] = []
        for _, _, doc_id, sentence in scored:
            if sentence in seen:
                continue
            seen.add(sentence)
            parts.append(f"{sentence.rstrip('.')} [{doc_id}].")
            if len(parts) == 3:
                break
        return " ".join(parts)

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        answer = await self.generate(messages)
        for token in re.findall(r"\S+\s*", answer):
            yield token

    async def structured(self, messages: list[Message], schema: type[T]) -> T:
        handler = self._handlers.get(schema)
        if handler is None:
            raise NotImplementedError(f"OfflineLLM has no heuristic for {schema.__name__}")
        result = handler(_user_text(messages))
        return schema.model_validate(result.model_dump())

    # --- heuristics -------------------------------------------------------------

    def _analyze(self, text: str) -> QueryAnalysis:
        question = extract_tag(text, "question")
        lowered = f" {question.lower()} "
        history = extract_tag(text, "history")
        standalone = question
        if history and re.search(r"\b(it|its|they|them|that|this|those|their)\b", lowered):
            last_user = [ln[6:] for ln in history.splitlines() if ln.startswith("user: ")]
            if last_user:
                standalone = f"{question} (context: {last_user[-1]})"
        route: Route
        if any(h in lowered for h in _DIRECT_HINTS) and len(tokenize(question)) <= 3:
            route = "direct"
        elif any(h in lowered for h in _WEB_HINTS):
            route = "web"
        elif any(h in lowered for h in _GLOBAL_HINTS):
            route = "global"
        else:
            route = "local"
        parts = [p.strip(" ?,.") for p in re.split(r"\?\s+|\band how\b|\band what\b|;", question) if p.strip()]
        subs = [p for p in parts if len(tokenize(p)) >= 2] if len(parts) > 1 else []
        return QueryAnalysis(
            standalone_question=standalone,
            route=route,
            sub_questions=subs[:3] if len(subs) > 1 else [],
            rationale=f"heuristic route={route}",
        )

    def _grade(self, text: str) -> DocumentGrades:
        question = extract_tag(text, "question")
        kinds = dict(_DOC_KIND_RE.findall(text))
        is_global = any(h in question.lower() for h in _GLOBAL_HINTS)
        relevant = [
            doc_id
            for doc_id, body in _docs(text)
            if overlap_score(question, body) >= 0.3 or (is_global and kinds.get(doc_id) == "community")
        ]
        return DocumentGrades(relevant_ids=relevant, reasoning="term-overlap >= 0.3; community reports for global")

    def _rewrite(self, text: str) -> RewrittenQuery:
        question = extract_tag(text, "question")
        return RewrittenQuery(query=" ".join(dict.fromkeys(tokenize(question))))

    def _grounding(self, text: str) -> GroundednessCheck:
        answer = extract_tag(text, "answer")
        evidence = set(tokenize(" ".join(body for _, body in _docs(text))))
        unsupported = []
        for sentence in split_sentences(_CITE_RE.sub("", answer)):
            tokens = tokenize(sentence)
            if tokens and sum(t in evidence for t in tokens) / len(tokens) < 0.6:
                unsupported.append(sentence)
        return GroundednessCheck(
            grounded=not unsupported,
            answers_question=bool(answer.strip()) and INSUFFICIENT not in answer,
            unsupported_claims=unsupported,
        )

    def _extract(self, text: str) -> ExtractionResult:
        body = extract_tag(text, "text")
        sentences = split_sentences(body)
        # Title-case words seen mid-sentence are likely proper nouns; ones seen only at sentence
        # start ("After", "Service") are ordinary words.
        mid_sentence_caps = {w for s in sentences for w in re.findall(r"(?<=\s)[A-Z][A-Za-z0-9-]+", s)}

        def is_plain(word: str) -> bool:
            return word.istitle() and not any(ch.isdigit() for ch in word)

        counts: Counter[str] = Counter()
        per_sentence: list[list[str]] = []
        for sentence in sentences:
            names: list[str] = []
            for match in _ENTITY_RE.findall(sentence):
                words = [w for w in match.split() if w not in _NOT_ENTITIES]
                while words and is_plain(words[0]) and words[0] not in mid_sentence_caps:
                    words = words[1:]
                if len(words) == 1 and is_plain(words[0]) and words[0] not in mid_sentence_caps:
                    continue
                name = " ".join(words)
                if len(name) >= 3 and not name.isdigit():
                    names.append(name)
                # Identifiers such as INC-2041 or SEV-1 also stand alone as entities.
                names += [w for w in words if len(words) > 1 and re.search(r"[A-Z]{2,}.*\d|\d.*[A-Z]{2,}", w)]
            names = list(dict.fromkeys(names))
            counts.update(names)
            per_sentence.append(names)
        keep = {name for name, _ in counts.most_common(15)}
        entities = [
            ExtractedEntity(name=n, type="CONCEPT", description=f"Mentioned {counts[n]}x in source.") for n in keep
        ]
        relations: list[ExtractedRelation] = []
        for names in per_sentence:
            for a, b in combinations([n for n in names if n in keep], 2):
                relations.append(ExtractedRelation(source=a, target=b, type="CO_OCCURS_WITH"))
        return ExtractionResult(entities=entities, relations=relations[:20])

    def _community(self, text: str) -> CommunityReport:
        entities = [ln[2:] for ln in extract_tag(text, "entities").splitlines() if ln.startswith("- ")]
        facts = [ln[2:] for ln in extract_tag(text, "relations").splitlines() if ln.startswith("- ")]
        title = ", ".join(entities[:3]) or "Community"
        summary = f"This cluster groups {', '.join(entities[:6])}. " + " ".join(facts[:3])
        return CommunityReport(title=title, summary=summary.strip())

    def _judge(self, text: str) -> JudgeScore:
        answer = extract_tag(text, "answer")
        reference = extract_tag(text, "reference")
        if reference:
            return JudgeScore(score=round(overlap_score(reference, answer), 3), reason="reference term recall")
        docs = " ".join(body for _, body in _docs(text))
        if docs:
            return JudgeScore(score=round(overlap_score(answer, docs), 3), reason="answer terms found in context")
        return JudgeScore(score=round(overlap_score(extract_tag(text, "question"), answer), 3), reason="overlap")


class HashEmbedder:
    """Feature-hashed bag of unigrams + bigrams, L2-normalized."""

    def __init__(self, dimension: int = 512) -> None:
        self.name = f"hash-embed-{dimension}"
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]
        vec = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            vec[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OverlapReranker:
    def __init__(self) -> None:
        self.name = "term-overlap-reranker"

    async def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        q_terms = set(tokenize(query))
        scored = []
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            coverage = overlap_score(query, text)
            density = sum(t in q_terms for t in tokens) / (len(tokens) + 10)
            scored.append((i, round(coverage + density, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
