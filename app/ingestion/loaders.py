"""Document loaders: turn raw bytes (PDF, DOCX, HTML, TXT, MD) into clean text pages."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}


class UnsupportedFileType(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


@dataclass
class Page:
    text: str
    page: Optional[int] = None


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(data: bytes) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [Page(p.extract_text() or "", i + 1) for i, p in enumerate(reader.pages)]


def _load_docx(data: bytes) -> list[Page]:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return [Page("\n".join(parts))]


def _load_html(data: bytes) -> list[Page]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return [Page(soup.get_text("\n"))]


def load_bytes(data: bytes, filename: str) -> list[Page]:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Unsupported file type '{ext or filename}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if ext == ".pdf":
        pages = _load_pdf(data)
    elif ext == ".docx":
        pages = _load_docx(data)
    elif ext in {".html", ".htm"}:
        pages = _load_html(data)
    else:
        pages = [Page(data.decode("utf-8", errors="replace"))]

    pages = [Page(clean_text(p.text), p.page) for p in pages]
    pages = [p for p in pages if p.text]
    if not pages:
        raise EmptyDocumentError(
            f"No extractable text in '{filename}' (scanned PDFs need OCR before ingestion)."
        )
    return pages


def load_path(path: str | Path) -> list[Page]:
    p = Path(path)
    return load_bytes(p.read_bytes(), p.name)
