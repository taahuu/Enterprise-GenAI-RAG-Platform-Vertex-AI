import io

import pytest

from app.ingestion.chunking import chunk_pages, split_text
from app.ingestion.loaders import EmptyDocumentError, Page, UnsupportedFileType, load_bytes


def make_pdf(text: str) -> bytes:
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
    ]
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def test_load_txt_and_md():
    assert load_bytes(b"Hello   world\n\n\n\nnext", "a.txt")[0].text == "Hello world\n\nnext"
    assert load_bytes(b"# Title\ntext", "a.md")[0].text.startswith("# Title")


def test_load_html_strips_scripts():
    html = b"<html><script>evil()</script><body><h1>Policy</h1><p>Rule one.</p></body></html>"
    text = load_bytes(html, "p.html")[0].text
    assert "evil" not in text and "Policy" in text and "Rule one." in text


def test_load_docx():
    from docx import Document

    d = Document()
    d.add_paragraph("Leave policy: 24 days.")
    buf = io.BytesIO()
    d.save(buf)
    assert "24 days" in load_bytes(buf.getvalue(), "x.docx")[0].text


def test_load_pdf_keeps_page_numbers():
    pages = load_bytes(make_pdf("Invoices are paid within 45 days"), "x.pdf")
    assert pages[0].page == 1 and "45 days" in pages[0].text


def test_unsupported_and_empty():
    with pytest.raises(UnsupportedFileType):
        load_bytes(b"x", "a.exe")
    with pytest.raises(EmptyDocumentError):
        load_bytes(b"   \n  ", "a.txt")


def test_split_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(200))
    parts = split_text(text, chunk_size=300, overlap=60)
    assert len(parts) > 5
    assert all(len(p) <= 300 + 5 for p in parts)
    # overlap: the start of a chunk repeats words from the end of the previous one
    shared = set(parts[0].split()[-6:]) & set(parts[1].split()[:12])
    assert shared


def test_split_prefers_paragraph_boundaries():
    text = ("A" * 150 + ".") + "\n\n" + ("B" * 150 + ".")
    parts = split_text(text, chunk_size=200, overlap=0)
    assert len(parts) == 2 and parts[0].startswith("A") and parts[1].startswith("B")


def test_chunk_metadata_and_ids():
    pages = [Page("First page text that is long enough to keep.", 1), Page("Second page text long enough too.", 2)]
    chunks = chunk_pages(pages, "doc1", "f.pdf", 800, 100)
    assert [c.chunk_id for c in chunks] == ["doc1-00000", "doc1-00001"]
    assert chunks[1].metadata["page"] == 2 and chunks[0].metadata["source"] == "f.pdf"
