"""Execution accuracy (EX): does the predicted SQL return the same result as the gold SQL?

Rows are compared as multisets (or as ordered lists when the gold query has a top-level
ORDER BY). Values inside a row are compared as a sorted multiset, which tolerates
column re-ordering, and floats are rounded to absorb harmless arithmetic differences.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import sqlglot
from sqlglot import exp


def _norm(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _row_key(row: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(sorted((_norm(v) for v in row), key=lambda v: (type(v).__name__, str(v))))


def has_top_level_order(sql: str, dialect: str = "sqlite") -> bool:
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.ParseError:
        return False
    return isinstance(tree, exp.Query) and tree.args.get("order") is not None


def results_match(pred: Sequence[Sequence[Any]], gold: Sequence[Sequence[Any]], *, ordered: bool) -> bool:
    if len(pred) != len(gold):
        return False
    pred_keys, gold_keys = [_row_key(r) for r in pred], [_row_key(r) for r in gold]
    if ordered:
        return pred_keys == gold_keys
    return Counter(pred_keys) == Counter(gold_keys)
