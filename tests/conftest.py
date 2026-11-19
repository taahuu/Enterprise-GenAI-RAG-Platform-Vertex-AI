"""Tests run fully offline: hashing embeddings + extractive LLM + numpy store (no model downloads)."""
import os
from pathlib import Path

os.environ.update(
    EMBEDDING_BACKEND="hash", LLM_BACKEND="extractive", VECTOR_BACKEND="numpy",
    RAG_PRELOAD="0", MIN_SCORE="0.2", API_KEY="",
)

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.pipeline import RAGPipeline  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def pipeline(settings):
    return RAGPipeline(settings)


@pytest.fixture
def loaded(pipeline):
    for f in sorted(SAMPLE_DIR.glob("*")):
        pipeline.ingest_path(f)
    return pipeline
