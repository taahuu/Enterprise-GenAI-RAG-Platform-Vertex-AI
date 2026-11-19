"""Persistent vector store.

numpy  exact cosine search (vectors are L2-normalised so dot product == cosine). Zero extra deps.
faiss  same persistence, search through a FAISS IndexFlatIP (pip install faiss-cpu).

Records are plain dicts: {"id": str, "text": str, "metadata": {...}}. Everything is saved under
INDEX_DIR as vectors.npy + records.json + meta.json so an index survives restarts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .config import Settings


class IndexMismatchError(RuntimeError):
    pass


class VectorStore:
    def __init__(self, index_dir: str, embedding_name: str):
        self.dir = Path(index_dir)
        self.embedding_name = embedding_name
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.records: list[dict] = []
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> None:
        meta_p = self.dir / "meta.json"
        if not meta_p.is_file():
            return
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("embedding_name") != self.embedding_name:
            raise IndexMismatchError(
                f"Index at '{self.dir}' was built with embeddings '{meta.get('embedding_name')}' but the "
                f"current config uses '{self.embedding_name}'. Re-ingest into a new INDEX_DIR or delete the old one."
            )
        self.vectors = np.load(self.dir / "vectors.npy")
        self.records = json.loads((self.dir / "records.json").read_text(encoding="utf-8"))

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_v = self.dir / "vectors.tmp.npy"
        np.save(tmp_v, self.vectors)
        os.replace(tmp_v, self.dir / "vectors.npy")
        tmp_r = self.dir / "records.tmp.json"
        tmp_r.write_text(json.dumps(self.records, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_r, self.dir / "records.json")
        meta = {"embedding_name": self.embedding_name, "dim": int(self.vectors.shape[1]) if len(self) else 0,
                "count": len(self)}
        tmp_m = self.dir / "meta.tmp.json"
        tmp_m.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp_m, self.dir / "meta.json")

    # ------------------------------------------------------------- hooks
    def _invalidate(self) -> None:
        """Called whenever vectors change (FAISS subclass rebuilds its index lazily)."""

    # -------------------------------------------------------------- crud
    def __len__(self) -> int:
        return len(self.records)

    def add(self, records: list[dict], vectors: np.ndarray) -> None:
        if len(records) != len(vectors):
            raise ValueError("records and vectors must have the same length")
        if not records:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(self) and vectors.shape[1] != self.vectors.shape[1]:
            raise ValueError(f"Vector dim {vectors.shape[1]} != index dim {self.vectors.shape[1]}")
        new_ids = {r["id"] for r in records}
        self._drop(lambda r: r["id"] in new_ids)  # idempotent re-ingest
        self.vectors = vectors if not len(self) else np.vstack([self.vectors, vectors])
        self.records.extend(records)
        self._invalidate()

    def _drop(self, predicate) -> int:
        keep = [i for i, r in enumerate(self.records) if not predicate(r)]
        removed = len(self.records) - len(keep)
        if removed:
            self.records = [self.records[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            self._invalidate()
        return removed

    def delete_doc(self, doc_id: str) -> int:
        return self._drop(lambda r: r["metadata"].get("doc_id") == doc_id)

    def doc_ids(self) -> set[str]:
        return {r["metadata"].get("doc_id") for r in self.records}

    # ------------------------------------------------------------ search
    def _mask(self, doc_ids: Optional[Iterable[str]]) -> Optional[np.ndarray]:
        if not doc_ids:
            return None
        wanted = set(doc_ids)
        return np.array([r["metadata"].get("doc_id") in wanted for r in self.records])

    def search(self, qvec: np.ndarray, k: int, doc_ids: Optional[Iterable[str]] = None) -> list[tuple[int, float]]:
        """Return [(row_index, cosine_score)] best first."""
        if not len(self) or k <= 0:
            return []
        scores = self.vectors @ np.asarray(qvec, dtype=np.float32)
        mask = self._mask(doc_ids)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


class FaissVectorStore(VectorStore):
    def __init__(self, index_dir: str, embedding_name: str):
        self._index = None
        super().__init__(index_dir, embedding_name)

    def _invalidate(self) -> None:
        self._index = None

    def search(self, qvec, k, doc_ids=None):
        if doc_ids or not len(self):  # filtered queries use the exact numpy path
            return super().search(qvec, k, doc_ids)
        import faiss  # type: ignore

        if self._index is None:
            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(np.ascontiguousarray(self.vectors))
        scores, idx = self._index.search(np.asarray(qvec, dtype=np.float32)[None, :], min(k, len(self)))
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


def get_store(s: Settings, embedding_name: str) -> VectorStore:
    if s.vector_backend == "numpy":
        return VectorStore(s.index_dir, embedding_name)
    if s.vector_backend == "faiss":
        return FaissVectorStore(s.index_dir, embedding_name)
    raise ValueError(f"Unknown VECTOR_BACKEND '{s.vector_backend}' (numpy | faiss)")
