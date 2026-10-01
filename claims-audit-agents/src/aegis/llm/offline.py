"""Offline keyword baseline for the reviewer agents.

Not a model: it overrides a finding when a note sentence shares most content words with one of the
policy section's bullet criteria. It ignores negation by design ("no focal neurological deficit" still
matches "focal neurological deficit"), which is exactly the error a real reviewer model must avoid. It
lets the full workflow run without a key and gives the eval report a naive baseline.
"""

from __future__ import annotations

import re

from aegis.agents.reviewers import ProposedReview, ReviewBatch
from aegis.llm.base import Message, T
from aegis.models import Resolution
from aegis.text import sentences, tokens

_FINDING = re.compile(r'<finding rule_id="(\w+)" line_no="(\w+)" section="([\w.-]+)">')
_POLICY = re.compile(r'<policy id="([\w.-]+)" title="[^"]*">\n(.*?)\n</policy>', re.S)
_NOTE = re.compile(r"<clinical_note>\n(.*?)\n</clinical_note>", re.S)
_EXTRA_CRITERIA = {  # sections whose criteria are prose rather than bullets
    "AEG-3.2": ["sudden severe symptoms", "acute neurological change", "trauma"],
    "AEG-5.1": ["different anatomic site", "separate incision", "separate session", "contralateral knee"],
    "AEG-5.3": ["decision regarding hospitalization", "intensive monitoring for toxicity", "severe exacerbation"],
    "AEG-5.4": ["bilateral procedure", "unusually prolonged service", "significant complications"],
}


class OfflineLLM:
    name = "offline-keyword-baseline"

    async def complete_json(self, messages: list[Message], schema: type[T]) -> T:
        if schema is not ReviewBatch:
            raise NotImplementedError(f"offline baseline does not implement {schema.__name__}")
        text = str(messages[-1]["content"])
        policies = {pid: body for pid, body in _POLICY.findall(text)}
        note_match = _NOTE.search(text)
        note = note_match.group(1) if note_match else ""
        has_note = bool(note) and not note.startswith("(no clinical note")
        reviews = []
        for rule_id, line_no, section in _FINDING.findall(text):
            criteria = [ln.strip()[2:] for ln in policies.get(section, "").splitlines() if ln.strip().startswith("- ")]
            criteria += _EXTRA_CRITERIA.get(section, [])
            quote = self._support(note, criteria) if has_note else None
            reviews.append(
                ProposedReview(
                    rule_id=rule_id,
                    line_no=None if line_no == "None" else int(line_no),
                    resolution=Resolution.OVERRIDE if quote else Resolution.UPHOLD,
                    confidence=0.9 if quote else 0.85,
                    rationale="Keyword match to policy criterion." if quote else "No matching criterion in the note.",
                    policy_citations=[section],
                    evidence_quotes=[quote] if quote else [],
                )
            )
        return ReviewBatch(reviews=reviews).model_copy() if schema is ReviewBatch else schema()  # type: ignore[return-value]

    @staticmethod
    def _support(note: str, criteria: list[str]) -> str | None:
        for sentence in sentences(note):
            words = set(tokens(sentence))
            for criterion in criteria:
                crit = set(tokens(criterion))
                if crit and len(crit & words) / len(crit) >= 0.6:
                    return sentence
        return None
