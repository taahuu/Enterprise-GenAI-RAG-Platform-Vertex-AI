"""Bulk-ingest files or folders into the index.

    python -m scripts.ingest                      # ingests data/sample_docs
    python -m scripts.ingest path/to/docs more.pdf
"""
import sys
from pathlib import Path

from app.config import get_settings
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.pipeline import RAGPipeline


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"skip (not found): {p}")
    return files


def main(argv: list[str]) -> int:
    files = collect(argv or ["data/sample_docs"])
    if not files:
        print("No ingestible files found.")
        return 1
    pipe = RAGPipeline(get_settings())
    for f in files:
        try:
            r = pipe.ingest_path(f)
            print(f"{r.status:<10} {r.source:<36} doc_id={r.doc_id} chunks={r.chunks}")
        except ValueError as e:
            print(f"failed     {f.name}: {e}")
    print(f"\nIndex now holds {len(pipe.store)} chunks from {len(pipe.list_documents())} documents.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
