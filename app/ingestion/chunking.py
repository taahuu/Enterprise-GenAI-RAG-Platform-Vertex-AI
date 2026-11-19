"""Chunking: recursive, structure-aware splitting with overlap and per-chunk provenance.

Uses LangChain's RecursiveCharacterTextSplitter when installed; otherwise falls back to an
equivalent built-in splitter so the pipeline also runs in minimal environments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .loaders import Page

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

try:  # LangChain text splitter (preferred)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - exercised only without langchain installed
    RecursiveCharacterTextSplitter = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _recursive_split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    sep = next((s for s in seps if s and s in text), None)
    if sep is None:
        return [text[i : i + size] for i in range(0, len(text), size)]
    rest = seps[seps.index(sep) + 1 :]
    out: list[str] = []
    cur = ""
    for piece in text.split(sep):
        cand = f"{cur}{sep}{piece}" if cur else piece
        if len(cand) <= size:
            cur = cand
            continue
        if cur:
            out.append(cur)
        if len(piece) > size:
            out.extend(_recursive_split(piece, size, rest))
            cur = ""
        else:
            cur = piece
    if cur:
        out.append(cur)
    return out


def _with_overlap(parts: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(parts) < 2:
        return parts
    out = [parts[0]]
    for prev, cur in zip(parts, parts[1:]):
        tail = prev[-overlap:]
        space = tail.find(" ")
        if 0 <= space < len(tail) - 1 and len(prev) > overlap:
            tail = tail[space + 1 :]  # start overlap on a word boundary
        out.append(f"{tail} {cur}".strip())
    return out


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    overlap = max(0, min(overlap, chunk_size // 2))
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap, separators=SEPARATORS
        )
        parts = splitter.split_text(text)
    else:
        base = _recursive_split(text, max(1, chunk_size - overlap - 1), SEPARATORS)
        parts = _with_overlap([p.strip() for p in base if p.strip()], overlap)
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def chunk_pages(
    pages: list[Page],
    doc_id: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 120,
    extra_metadata: Optional[dict] = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for page in pages:
        for text in split_text(page.text, chunk_size, overlap):
            meta = {
                "doc_id": doc_id,
                "source": source,
                "page": page.page,
                "chunk_index": idx,
                **(extra_metadata or {}),
            }
            chunks.append(Chunk(chunk_id=f"{doc_id}-{idx:05d}", doc_id=doc_id, text=text, metadata=meta))
            idx += 1
    return chunks
