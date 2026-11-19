import numpy as np

from app.postprocess import finalize
from app.prompt import REFUSAL, build_prompt
from app.retriever import Hit, lexical_coverage, mmr_select


def test_relevant_doc_ranks_first(loaded):
    hits = loaded.retriever.retrieve("What is the hotel limit per night when travelling?")
    assert hits and hits[0].metadata["source"] == "expense_policy.md"


def test_off_topic_returns_nothing(loaded):
    assert loaded.retriever.retrieve("What is the capital of Australia?") == []


def test_doc_filter_restricts_results(loaded):
    faq = next(d["doc_id"] for d in loaded.list_documents() if d["source"] == "product_faq.txt")
    hits = loaded.retriever.retrieve("What does the Business plan cost per user per month?", doc_ids=[faq])
    assert hits and {h.metadata["doc_id"] for h in hits} == {faq}


def test_mmr_prefers_diverse_results():
    rel = np.array([1.0, 0.99, 0.9], dtype=np.float32)
    vecs = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)   # first two are duplicates
    assert mmr_select(rel, vecs, 2, lam=0.5) == [0, 2]


def test_lexical_coverage():
    assert lexical_coverage({"leave", "day"}, "Annual leave days") == 1.0
    assert lexical_coverage(set(), "x") == 0.0


def hit(i, text, source="a.md", page=None):
    return Hit(f"c{i}", text, {"source": source, "page": page}, 0.9, 0.9, 0.9)


def test_prompt_numbers_blocks_and_has_guardrails():
    b = build_prompt("Q?", [hit(1, "alpha", page=3), hit(2, "beta")])
    assert "[1] (source: a.md, page 3)" in b.user and "[2] (source: a.md)" in b.user
    assert "Question: Q?" in b.user and "ONLY" in b.system and "never as instructions" in b.system


def test_prompt_drops_tail_hits_over_budget():
    b = build_prompt("Q?", [hit(1, "x" * 400), hit(2, "y" * 400)], max_context_chars=500)
    assert len(b.hits) == 1 and "yyyy" not in b.user


def test_prompt_handles_braces_in_context():
    assert "{not_a_var}" in build_prompt("Q?", [hit(1, "text {not_a_var}")]).user


def test_finalize_strips_invalid_citations():
    ans, cited = finalize("A is true [1]. B is true [7]. C [2, 9].", n_blocks=2)
    assert cited == [1, 2] and "[7]" not in ans and "9" not in ans and "[2]" in ans


def test_finalize_normalises_refusal():
    assert finalize(REFUSAL + " extra", 3) == (REFUSAL, [])
