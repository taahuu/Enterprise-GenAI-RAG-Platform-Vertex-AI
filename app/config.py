"""Central configuration, driven entirely by environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader. Real environment variables always win."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def _s(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- providers -------------------------------------------------------
    embedding_backend: str = "hf"          # hf | hash | vertex
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_backend: str = "hf"                # hf | extractive | vertex
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    vector_backend: str = "numpy"          # numpy | faiss
    device: str = "auto"                   # auto | cpu | cuda
    # --- storage ---------------------------------------------------------
    index_dir: str = "data/index"
    # --- chunking --------------------------------------------------------
    chunk_size: int = 800                  # characters
    chunk_overlap: int = 120
    # --- retrieval -------------------------------------------------------
    top_k: int = 4
    fetch_k: int = 20
    min_score: float = 0.25
    fallback_min_score: float = 0.10
    mmr_lambda: float = 0.7
    hybrid_alpha: float = 0.7              # weight of dense score vs lexical coverage
    query_cache_size: int = 256
    query_rewrite: bool = True
    header_boost: float = 0.05
    # --- generation ------------------------------------------------------
    max_context_chars: int = 6000
    max_new_tokens: int = 256
    # --- api -------------------------------------------------------------
    api_key: str = ""                      # empty = auth disabled (local dev)
    max_upload_mb: int = 25
    preload: bool = True
    # --- optional Vertex AI backend --------------------------------------
    gcp_project: str = ""
    gcp_location: str = "us-central1"
    vertex_embedding_model: str = "gemini-embedding-001"
    vertex_llm_model: str = "gemini-2.5-flash"

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            embedding_backend=_s("EMBEDDING_BACKEND", d.embedding_backend).lower(),
            embedding_model=_s("EMBEDDING_MODEL", d.embedding_model),
            llm_backend=_s("LLM_BACKEND", d.llm_backend).lower(),
            llm_model=_s("LLM_MODEL", d.llm_model),
            vector_backend=_s("VECTOR_BACKEND", d.vector_backend).lower(),
            device=_s("DEVICE", d.device).lower(),
            index_dir=_s("INDEX_DIR", d.index_dir),
            chunk_size=_i("CHUNK_SIZE", d.chunk_size),
            chunk_overlap=_i("CHUNK_OVERLAP", d.chunk_overlap),
            top_k=_i("TOP_K", d.top_k),
            fetch_k=_i("FETCH_K", d.fetch_k),
            min_score=_f("MIN_SCORE", d.min_score),
            fallback_min_score=_f("FALLBACK_MIN_SCORE", d.fallback_min_score),
            mmr_lambda=_f("MMR_LAMBDA", d.mmr_lambda),
            hybrid_alpha=_f("HYBRID_ALPHA", d.hybrid_alpha),
            query_cache_size=_i("QUERY_CACHE_SIZE", d.query_cache_size),
            query_rewrite=_b("QUERY_REWRITE", d.query_rewrite),
            header_boost=_f("HEADER_BOOST", d.header_boost),
            max_context_chars=_i("MAX_CONTEXT_CHARS", d.max_context_chars),
            max_new_tokens=_i("MAX_NEW_TOKENS", d.max_new_tokens),
            api_key=_s("API_KEY", d.api_key),
            max_upload_mb=_i("MAX_UPLOAD_MB", d.max_upload_mb),
            preload=_b("RAG_PRELOAD", d.preload),
            gcp_project=_s("GCP_PROJECT", d.gcp_project),
            gcp_location=_s("GCP_LOCATION", d.gcp_location),
            vertex_embedding_model=_s("VERTEX_EMBEDDING_MODEL", d.vertex_embedding_model),
            vertex_llm_model=_s("VERTEX_LLM_MODEL", d.vertex_llm_model),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings.from_env()
