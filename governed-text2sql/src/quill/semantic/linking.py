"""Schema linking: prune the schema to what a question needs before prompting.

Large warehouses don't fit in a prompt, and irrelevant tables are the most common cause
of wrong joins. Tables score on name, synonym, description, column and value matches;
the selection is then closed over foreign keys so any needed join path stays available.
"""

from __future__ import annotations

from dataclasses import dataclass

from quill.semantic.layer import SemanticLayer
from quill.text import tokenize


@dataclass(frozen=True)
class LinkResult:
    tables: list[str]
    metrics: list[str]
    scores: dict[str, float]


def link_schema(layer: SemanticLayer, question: str, *, max_tables: int = 5) -> LinkResult:
    q_tokens = set(tokenize(question))
    q_text = f" {question.lower()} "
    scores: dict[str, float] = {}
    for name, table in layer.tables.items():
        score = 0.0
        if set(tokenize(name)) & q_tokens:
            score += 3
        score += sum(2.5 for s in table.synonyms if f" {s.lower()}" in q_text)
        score += 0.5 * len(set(tokenize(table.description)) & q_tokens)
        for col_name, col in table.columns.items():
            if set(tokenize(col_name)) & q_tokens:
                score += 1.0
            score += sum(1.5 for v in col.values if v.lower() in q_text)
        scores[name] = score

    metrics = [
        m
        for m, metric in layer.metrics.items()
        if any(s.lower() in q_text for s in metric.synonyms) or m.replace("_", " ") in q_text
    ]
    if metrics:
        scores["claims"] = scores.get("claims", 0) + 3  # every metric is defined over claims

    selected = [t for t, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0][:max_tables]
    if not selected:
        selected = ["claims"]
    selected = _close_over_joins(layer, selected)
    return LinkResult(tables=selected, metrics=metrics, scores=scores)


def _close_over_joins(layer: SemanticLayer, tables: list[str]) -> list[str]:
    """Add bridge tables so every selected pair is connected through foreign keys."""
    edges: dict[str, set[str]] = {t: set() for t in layer.tables}
    for left, right in layer.joins():
        a, b = left.split(".")[0], right.split(".")[0]
        edges[a].add(b)
        edges[b].add(a)
    result = list(dict.fromkeys(tables))
    anchor = result[0]
    for target in result[1:]:
        path = _shortest_path(edges, anchor, target)
        for t in path:
            if t not in result:
                result.append(t)
    return result


def _shortest_path(edges: dict[str, set[str]], start: str, goal: str) -> list[str]:
    frontier, parents = [start], {start: ""}
    while frontier:
        nxt = []
        for node in frontier:
            for nbr in edges[node]:
                if nbr not in parents:
                    parents[nbr] = node
                    nxt.append(nbr)
        if goal in parents:
            break
        frontier = nxt
    if goal not in parents:
        return []
    path, node = [], goal
    while node:
        path.append(node)
        node = parents[node]
    return path[::-1]
