"""Policy manual: parsed into citable sections and searchable with BM25."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from aegis.text import tokens

_SECTION_RE = re.compile(r"^###\s+(AEG-\d+\.\d+)\s+(.+)$", re.M)


@dataclass(frozen=True)
class PolicySection:
    section_id: str
    title: str
    text: str

    def render(self) -> str:
        return f'<policy id="{self.section_id}" title="{self.title}">\n{self.text}\n</policy>'

    @property
    def criteria(self) -> list[str]:
        """Bullet-list criteria within the section (used by the offline keyword baseline)."""
        return [ln.strip()[2:].strip() for ln in self.text.splitlines() if ln.strip().startswith("- ")]


class PolicyManual:
    def __init__(self, sections: list[PolicySection]) -> None:
        self.sections = {s.section_id: s for s in sections}
        ordered = list(self.sections.values())
        self._ids = [s.section_id for s in ordered]
        self._bm25 = BM25Okapi([tokens(f"{s.title} {s.text}") for s in ordered])

    @classmethod
    def load(cls, path: Path) -> PolicyManual:
        text = path.read_text()
        matches = list(_SECTION_RE.finditer(text))
        sections = []
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = re.split(r"^##\s", text[m.end() : end], flags=re.M)[0].strip()
            sections.append(PolicySection(m.group(1), m.group(2).strip(), body))
        return cls(sections)

    def get(self, section_id: str) -> PolicySection | None:
        return self.sections.get(section_id)

    def search(self, query: str, k: int = 3) -> list[PolicySection]:
        scores = self._bm25.get_scores(tokens(query))
        ranked = sorted(zip(self._ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [self.sections[sid] for sid, score in ranked[:k] if score > 0]
