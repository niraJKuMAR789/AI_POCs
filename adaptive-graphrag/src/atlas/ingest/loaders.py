"""Load Markdown, text and PDF files into `SourceDocument`s."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from atlas.models import SourceDocument, stable_id

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}


def _title_from(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, flags=re.M)
    return match.group(1).strip() if match else fallback


def load_file(path: Path) -> SourceDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        title = (reader.metadata.title if reader.metadata and reader.metadata.title else None) or path.stem
    elif suffix in SUPPORTED:
        text = path.read_text(encoding="utf-8", errors="replace")
        title = _title_from(text, path.stem.replace("_", " ").replace("-", " ").title())
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return SourceDocument(id=stable_id(str(path.resolve())), source=path.name, title=title, text=text)


def load_text(text: str, source: str, title: str | None = None) -> SourceDocument:
    return SourceDocument(id=stable_id(source), source=source, title=title or _title_from(text, source), text=text)


def iter_documents(root: Path) -> Iterator[SourceDocument]:
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    for path in paths:
        yield load_file(path)
