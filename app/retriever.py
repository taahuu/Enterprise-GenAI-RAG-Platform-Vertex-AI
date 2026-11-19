"""Retrieval: dense search -> lightweight hybrid (lexical) rerank -> confidence gate -> MMR diversity."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .config import Settings
from .embeddings import Embedder
from .query import needs_rewrite, rewrite_query
from .text_utils import tokenize
from .vectorstore import VectorStore


@dataclass
class Hit:
    chunk_id: str
    text: str
    metadata: dict
    score: float            # combined score used for ranking
    dense: float            # cosine similarity from the vector index
    lexical: float          # share of question terms present in the chunk
    _row: int = field(default=-1, repr=False)


def lexical_coverage(question_tokens: set[str], text: str) -> float:
    if not question_tokens:
        return 0.0
    return len(question_tokens & set(tokenize(text))) / len(question_tokens)


def mmr_select(rel: np.ndarray, vecs: np.ndarray, k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance: trade relevance against redundancy with already-picked chunks."""
    n = len(rel)
    if n <= k:
        return list(np.argsort(-rel))
    sim = vecs @ vecs.T
    chosen = [int(np.argmax(rel))]
    remaining = set(range(n)) - set(chosen)
    while remaining and len(chosen) < k:
        best = max(remaining, key=lambda i: lam * rel[i] - (1 - lam) * max(sim[i, j] for j in chosen))
        chosen.append(best)
        remaining.remove(best)
    return chosen


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings):
        self.store, self.embedder, self.s = store, embedder, settings
        self.last_best: dict[str, float] = {"dense": 0.0, "lexical": 0.0}

    def retrieve(self, question: str, top_k: Optional[int] = None,
                 doc_ids: Optional[Iterable[str]] = None) -> list[Hit]:
        hits, _ = self.retrieve_ex(question, top_k, doc_ids)
        return hits

    def retrieve_ex(self, question: str, top_k: Optional[int] = None,
                    doc_ids: Optional[Iterable[str]] = None) -> tuple[list[Hit], str]:
        k = top_k or self.s.top_k
        rewritten, is_identity = rewrite_query(question) if self.s.query_rewrite else (question, False)
        qvec = self.embedder.embed_query(question)
        candidate_limit = max(self.s.fetch_k, k)
        candidates = self.store.search(qvec, candidate_limit, doc_ids)

        if needs_rewrite(question, rewritten):
            rewrite_vec = self.embedder.embed_query(rewritten)
        else:
            rewrite_vec = None

        if not candidates:
            self.last_best = {"dense": 0.0, "lexical": 0.0}
            return [], "none"

        qtok = set(tokenize(rewritten))
        a = self.s.hybrid_alpha
        hits: list[Hit] = []
        for row, dense_original in candidates:
            rec = self.store.records[row]
            lex = lexical_coverage(qtok, rec["text"])
            dense = float(dense_original)
            blend = a * dense + (1 - a) * lex
            boost = self.s.header_boost if is_identity and rec["metadata"].get("chunk_index") == 0 else 0.0
            hits.append(Hit(rec["id"], rec["text"], rec["metadata"], blend + boost,
                            dense, lex, row))

        best_dense = max(h.dense for h in hits)
        best_lexical = max(h.lexical for h in hits)
        self.last_best = {"dense": float(best_dense), "lexical": float(best_lexical)}

        passed = [h for h in hits if h.dense >= self.s.min_score or h.lexical >= 0.5]
        if passed:
            confidence = "high"
            pool = passed
        elif best_dense >= self.s.fallback_min_score:
            confidence = "low"
            pool = hits
        else:
            return [], "none"

        rel = np.array([h.score for h in pool], dtype=np.float32)
        vecs = self.store.vectors[[h._row for h in pool]]
        selected_k = min(k, 2) if confidence == "low" else k
        order = mmr_select(rel, vecs, selected_k, self.s.mmr_lambda)
        return [pool[i] for i in order][:selected_k], confidence
