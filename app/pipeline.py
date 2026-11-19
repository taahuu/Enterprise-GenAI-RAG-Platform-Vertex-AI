"""RAGPipeline: ingestion -> chunking -> embeddings -> vector store -> retrieval -> prompt -> LLM -> post-process."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import Settings
from .embeddings import get_embedder
from .ingestion.chunking import chunk_pages
from .ingestion.loaders import load_bytes
from .llm import get_llm
from .postprocess import finalize
from .prompt import REFUSAL, build_prompt
from .retriever import Retriever
from .vectorstore import get_store

log = logging.getLogger("rag")


@dataclass
class IngestResult:
    doc_id: str
    source: str
    chunks: int
    status: str                      # ingested | duplicate | replaced
    replaced_doc_ids: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    request_id: str
    question: str
    answer: str
    grounded: bool                   # answer cites at least one valid context block
    sources: list[dict]
    timings_ms: dict
    models: dict
    confidence: Optional[str] = None


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.s = settings
        self._lock = threading.RLock()
        self.embedder = get_embedder(settings)
        self.store = get_store(settings, self.embedder.name)
        self.retriever = Retriever(self.store, self.embedder, settings)
        self._llm = None

    # ---------------------------------------------------------------- llm
    @property
    def llm(self):
        if self._llm is None:  # lazy: ingestion-only workflows never load the generator
            self._llm = get_llm(self.s)
        return self._llm

    def warmup(self) -> None:
        self.embedder.embed_query("warmup")
        _ = self.llm

    # ------------------------------------------------------------- ingest
    def ingest_bytes(self, data: bytes, filename: str, replace_same_source: bool = True) -> IngestResult:
        source = Path(filename).name
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        with self._lock:
            if doc_id in self.store.doc_ids():
                return IngestResult(doc_id, source, 0, "duplicate")

            pages = load_bytes(data, source)
            chunks = chunk_pages(
                pages, doc_id, source, self.s.chunk_size, self.s.chunk_overlap,
                {"ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            )
            vectors = self.embedder.embed_documents([c.text for c in chunks])

            replaced: list[str] = []
            if replace_same_source:  # re-uploading an updated file must not leave stale chunks behind
                stale = {r["metadata"]["doc_id"] for r in self.store.records
                         if r["metadata"].get("source") == source}
                for old in stale:
                    self.store.delete_doc(old)
                    replaced.append(old)

            self.store.add([{"id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in chunks], vectors)
            self.store.save()
        log.info(json.dumps({"event": "ingest", "doc_id": doc_id, "source": source, "chunks": len(chunks)}))
        return IngestResult(doc_id, source, len(chunks), "replaced" if replaced else "ingested", replaced)

    def ingest_path(self, path: str | Path, **kw) -> IngestResult:
        p = Path(path)
        return self.ingest_bytes(p.read_bytes(), p.name, **kw)

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        with self._lock:
            for r in self.store.records:
                m = r["metadata"]
                d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "source": m.get("source"),
                                                  "chunks": 0, "ingested_at": m.get("ingested_at")})
                d["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["source"] or "")

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            removed = self.store.delete_doc(doc_id)
            if removed:
                self.store.save()
        return removed

    # -------------------------------------------------------------- query
    def query(self, question: str, top_k: Optional[int] = None, doc_ids: Optional[list[str]] = None) -> QueryResult:
        rid, t0 = uuid.uuid4().hex[:12], time.perf_counter()
        question = question.strip()
        models = {"embedding": self.embedder.name, "llm": self.s.llm_model if self.s.llm_backend == "hf"
                  else self.s.llm_backend}

        with self._lock:
            hits, confidence = self.retriever.retrieve_ex(question, top_k, doc_ids)
        t1 = time.perf_counter()

        if confidence == "none":
            result = QueryResult(rid, question, REFUSAL, False, [],
                                 {"retrieve": _ms(t0, t1), "generate": 0.0, "total": _ms(t0, t1)},
                                 models, confidence)
            self._audit(result, 0)
            return result

        bundle = build_prompt(question, hits, self.s.max_context_chars, loose_context=confidence == "low")
        raw = self.llm.generate(bundle)
        t2 = time.perf_counter()
        answer, cited = finalize(raw, len(bundle.hits))

        refused = answer == REFUSAL
        # Cited blocks if the model cited; otherwise fall back to everything it was shown.
        shown = [(n, h) for n, h in enumerate(bundle.hits, 1) if not cited or n in cited] if not refused else []
        sources = [{
            "n": n, "chunk_id": h.chunk_id, "doc_id": h.metadata.get("doc_id"),
            "source": h.metadata.get("source"), "page": h.metadata.get("page"),
            "score": round(h.score, 4), "snippet": h.text[:240],
        } for n, h in shown]

        grounded = bool(shown) and not refused
        result = QueryResult(rid, question, answer, grounded, sources,
                             {"retrieve": _ms(t0, t1), "generate": _ms(t1, t2), "total": _ms(t0, t2)},
                             models, confidence)
        self._audit(result, len(hits))
        return result

    def search(self, question: str, top_k: Optional[int] = None,
               doc_ids: Optional[list[str]] = None) -> dict:
        with self._lock:
            hits, confidence = self.retriever.retrieve_ex(question.strip(), top_k, doc_ids)
        return {
            "confidence": confidence,
            "min_score": self.s.min_score,
            "fallback_min_score": self.s.fallback_min_score,
            "best_scores": {
                "dense": round(self.retriever.last_best["dense"], 6),
                "lexical": round(self.retriever.last_best["lexical"], 6),
            },
            "hits": [{
                "chunk_id": h.chunk_id,
                "source": h.metadata.get("source"),
                "page": h.metadata.get("page"),
                "dense": round(h.dense, 6),
                "lexical": round(h.lexical, 6),
                "score": round(h.score, 6),
                "text": h.text,
            } for h in hits],
        }

    def _audit(self, r: QueryResult, n_hits: int) -> None:
        # Query text is hashed, not logged, because prompts can contain sensitive data.
        log.info(json.dumps({
            "event": "query", "request_id": r.request_id,
            "query_sha256": hashlib.sha256(r.question.encode()).hexdigest()[:16],
            "retrieved": n_hits, "confidence": r.confidence, "grounded": r.grounded,
            "timings_ms": r.timings_ms,
            "best_scores": {
                "dense": round(self.retriever.last_best["dense"], 6),
                "lexical": round(self.retriever.last_best["lexical"], 6),
            },
            "min_score": self.s.min_score,
            "chunk_ids": [s["chunk_id"] for s in r.sources],
        }))

    def stats(self) -> dict:
        return {"documents": len(self.list_documents()), "chunks": len(self.store),
                "embedding_model": self.embedder.name, "vector_backend": self.s.vector_backend,
                "llm_backend": self.s.llm_backend,
                "query_cache": {"hits": getattr(self.embedder, "hits", 0),
                                "misses": getattr(self.embedder, "misses", 0)}}


def _ms(a: float, b: float) -> float:
    return round((b - a) * 1000, 2)


def result_to_dict(r: QueryResult) -> dict:
    return asdict(r)
