"""Structure-aware chunking: split on Markdown headings first, then recursively by size.

Each chunk carries its heading path ("Guide > Setup > GPUs"), which is prepended
when embedding so that short chunks keep the context their section provides.
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from atlas.models import Chunk, SourceDocument, stable_id

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", flags=re.M)


def _sections(text: str) -> list[tuple[str, str]]:
    """Return (heading_path, body) pairs."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level, heading = len(m.group(1)), m.group(2).strip()
        stack = [(lvl, h) for lvl, h in stack if lvl < level] + [(level, heading)]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            sections.append((" > ".join(h for _, h in stack), body))
    return sections


def chunk_document(doc: SourceDocument, *, chunk_size: int = 1200, chunk_overlap: int = 150) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for section, body in _sections(doc.text):
        for piece in splitter.split_text(body):
            position = len(chunks)
            chunks.append(
                Chunk(
                    id=stable_id(doc.id, str(position), piece),
                    doc_id=doc.id,
                    source=doc.source,
                    title=doc.title,
                    section=section,
                    text=piece,
                    position=position,
                    metadata=doc.metadata,
                )
            )
    return chunks


def heading_path(chunk: Chunk) -> str:
    if chunk.section.startswith(chunk.title):
        return chunk.section
    return " > ".join(p for p in (chunk.title, chunk.section) if p)


def embedding_text(chunk: Chunk) -> str:
    header = heading_path(chunk)
    return f"{header}\n\n{chunk.text}" if header else chunk.text
