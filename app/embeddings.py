"""Embedding backends behind one tiny interface.

hf      Hugging Face sentence-transformers via LangChain's HuggingFaceEmbeddings (default, local, free)
hash    Deterministic hashing embedder (offline fallback for tests / air-gapped smoke runs; lexical only)
vertex  Vertex AI embeddings through the google-genai SDK (optional)
"""
from __future__ import annotations

import zlib
from collections import OrderedDict
from typing import Protocol

import numpy as np

from .config import Settings
from .text_utils import tokenize


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


class HashingEmbedder:
    """Signed feature hashing of stemmed unigrams + bigrams. Deterministic, no model download."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for f in feats:
            counts[f] = counts.get(f, 0) + 1
        for f, c in counts.items():
            h = zlib.crc32(f.encode("utf-8"))
            sign = 1.0 if (h >> 31) & 1 else -1.0
            v[h % self.dim] += sign * (1.0 + np.log(c))
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text))


class HFEmbedder:
    """Hugging Face sentence-transformers model, wrapped by LangChain's HuggingFaceEmbeddings."""

    def __init__(self, model_name: str, device: str = "auto"):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "EMBEDDING_BACKEND=hf needs: pip install langchain-huggingface sentence-transformers"
            ) from e
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.name = model_name
        self._emb = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalize(np.array(self._emb.embed_documents(texts), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(self._emb.embed_query(text), dtype=np.float32))


class VertexEmbedder:  # pragma: no cover - needs GCP credentials
    def __init__(self, project: str, location: str, model: str, dim: int = 768):
        from google import genai

        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._model, self._dim = model, dim
        self.name = f"vertex:{model}:{dim}"

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        from google.genai import types

        out = []
        for i in range(0, len(texts), 100):
            res = self._client.models.embed_content(
                model=self._model,
                contents=texts[i : i + 100],
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=self._dim),
            )
            out.extend(e.values for e in res.embeddings)
        return _normalize(np.array(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class CachedEmbedder:
    """LRU cache for query embeddings (repeat questions skip the model entirely)."""

    def __init__(self, inner: Embedder, size: int = 256):
        self._inner, self._size = inner, size
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.name = inner.name
        self.hits = 0
        self.misses = 0

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        key = text.strip()
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        vec = self._inner.embed_query(key)
        self._cache[key] = vec
        if len(self._cache) > self._size:
            self._cache.popitem(last=False)
        return vec


def get_embedder(s: Settings) -> Embedder:
    if s.embedding_backend == "hash":
        inner: Embedder = HashingEmbedder()
    elif s.embedding_backend == "hf":
        inner = HFEmbedder(s.embedding_model, s.device)
    elif s.embedding_backend == "vertex":
        inner = VertexEmbedder(s.gcp_project, s.gcp_location, s.vertex_embedding_model)
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND '{s.embedding_backend}' (hf | hash | vertex)")
    return CachedEmbedder(inner, s.query_cache_size)
