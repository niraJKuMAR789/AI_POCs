"""Offline retrieval baseline: nearest few-shot example + value substitution.

Not a model. It adapts the most similar example in the prompt by swapping in values
from the question, which works for seen query shapes and fails on unseen ones. That
makes it a useful non-parametric baseline for the eval report, and it lets the whole
pipeline run in CI without a key.
"""

from __future__ import annotations

import re

from quill.data.questions import POOLS
from quill.llm.base import Message
from quill.text import jaccard

_SHOT_RE = re.compile(r"Q: (.*?)\n```sql\n(.*?)\n```", re.S)
_QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.S)
_RESULT_RE = re.compile(r"<result[^>]*>\n(.*?)\n</result>", re.S)


def _values_in(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    lowered = text.lower()
    for pool, values in POOLS.items():
        for value in values:
            if re.search(rf"(?<![\w-]){re.escape(str(value).lower())}(?![\w-])", lowered):
                found.setdefault(pool, str(value))
    return found


class OfflineLLM:
    name = "offline-knn-baseline"

    async def complete(self, messages: list[Message]) -> str:
        text = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        first_user = next(m["content"] for m in messages if m["role"] == "user")
        if "<result" in first_user:
            return self._summarize(first_user)
        if "was rejected" in text:
            match = re.search(r"```sql\n(.*?)\n```", text, re.S)
            return f"```sql\n{match.group(1) if match else 'SELECT 1'}\n```"
        question_match = _QUESTION_RE.search(first_user)
        question = question_match.group(1) if question_match else ""
        shots = _SHOT_RE.findall(first_user)
        if not shots:
            return "```sql\nSELECT 'UNANSWERABLE' AS reason\n```"
        example_q, sql = max(shots, key=lambda s: jaccard(question, s[0]))
        wanted, present = _values_in(question), _values_in(example_q)
        for pool, new in wanted.items():
            old = present.get(pool)
            if old is None or old == new:
                continue
            if isinstance(POOLS[pool][0], int):
                sql = re.sub(rf"\b{re.escape(old)}\b", new, sql)
            else:
                sql = sql.replace(f"'{old}'", f"'{new}'")
        return f"```sql\n{sql}\n```"

    @staticmethod
    def _summarize(prompt: str) -> str:
        result = _RESULT_RE.search(prompt)
        lines = result.group(1).splitlines() if result else []
        rows = [ln for ln in lines[2:] if ln.startswith("|")]
        if not rows:
            return "The query returned no rows."
        return f"The query returned {len(rows)} row(s); first row: {rows[0].strip('| ').replace(' | ', ', ')}."
