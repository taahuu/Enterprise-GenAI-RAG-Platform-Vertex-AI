import numpy as np

from app.config import Settings
from app.query import needs_rewrite, rewrite_query
from app.retriever import Retriever


class StubEmbedder:
    name = "stub"

    def embed_query(self, text):
        return np.array([1.0], dtype=np.float32)


class StubStore:
    def __init__(self, rows, dense_scores):
        self.records = rows
        self.vectors = np.ones((len(rows), 1), dtype=np.float32)
        self.scores = dense_scores

    def search(self, qvec, k, doc_ids=None):
        return [(i, float(self.scores[i])) for i in range(min(k, len(self.records)))]


def retriever(dense_scores, texts, min_score=0.25, fallback=0.10,
             chunk_indexes=None, top_k=4):
    rows = []
    for i, text in enumerate(texts):
        metadata = {"source": f"doc{i}.txt", "chunk_index": (chunk_indexes[i] if chunk_indexes else 0 if i == 0 else i)}
        rows.append({"id": f"c{i}", "text": text, "metadata": metadata})
    settings = Settings(min_score=min_score, fallback_min_score=fallback, top_k=top_k,
                        fetch_k=20, mmr_lambda=0.7, hybrid_alpha=0.7,
                        query_rewrite=True, header_boost=0.05)
    return Retriever(StubStore(rows, dense_scores), StubEmbedder(), settings)


def test_gate_dense_or_lexical_and_fallback_levels():
    r = retriever([0.30], ["unrelated prose"])
    hits, confidence = r.retrieve_ex("ordinary question")
    assert confidence == "high" and hits

    r = retriever([0.10], ["a b c d e"])
    hits, confidence = r.retrieve_ex("a b c d e")
    assert confidence == "high" and hits

    r = retriever([0.12], ["a"])
    hits, confidence = r.retrieve_ex("a b c d e")
    assert confidence == "low" and len(hits) <= 2

    r = retriever([0.05], ["unrelated"])
    hits, confidence = r.retrieve_ex("nothing here")
    assert confidence == "none" and hits == []


def test_rewrite_filler_synonyms_and_identity():
    rewritten, identity = rewrite_query("give me name of the person which is mention")
    assert "give me" not in rewritten.lower()
    assert "mention" not in rewritten.lower()
    assert identity
    assert rewritten.lower().count("candidate") == 1
    assert rewritten.lower().count("full name") == 1

    assert rewrite_query("What is the annual leave policy?") == ("What is the annual leave policy?", False)
    assert not needs_rewrite("What is the name?", "What is the name?")
    assert not needs_rewrite("What is the name?", "What is the name?!")


def test_identity_header_chunk_ranks_first():
    # Header boost is ranking-only: the 0.10 floor deliberately keeps this synthetic weak result in low mode.
    r = retriever([0.12, 0.11], ["Header", "Other information"],
                  fallback=0.10, chunk_indexes=[0, 1], top_k=1)
    hits, confidence = r.retrieve_ex("name of the person")
    assert confidence == "low"
    assert hits and hits[0].metadata["chunk_index"] == 0
