import json
from pathlib import Path

from app.pipeline import RAGPipeline
from app.prompt import REFUSAL
from scripts.evaluate import evaluate

ROOT = Path(__file__).resolve().parent.parent


def test_ingest_duplicate_and_replace(pipeline):
    r1 = pipeline.ingest_bytes(b"Refunds are issued within 14 days of purchase for all plans.", "refunds.txt")
    assert r1.status == "ingested" and r1.chunks == 1
    assert pipeline.ingest_bytes(b"Refunds are issued within 14 days of purchase for all plans.",
                                 "refunds.txt").status == "duplicate"
    r3 = pipeline.ingest_bytes(b"Refunds are issued within 30 days of purchase for all plans.", "refunds.txt")
    assert r3.status == "replaced" and r3.replaced_doc_ids == [r1.doc_id]
    assert len(pipeline.list_documents()) == 1          # no stale chunks left behind
    assert "30" in pipeline.query("When are refunds issued?").answer


def test_index_persists_across_restarts(loaded, settings):
    n = len(loaded.store)
    reopened = RAGPipeline(settings)
    assert len(reopened.store) == n and reopened.query("How long is parental leave?").grounded


def test_answer_is_grounded_with_valid_sources(loaded):
    r = loaded.query("How many days of annual leave do employees get?")
    assert r.grounded and "24" in r.answer
    assert r.sources and all(s["source"] == "hr_leave_policy.md" for s in r.sources)
    assert set(r.timings_ms) == {"retrieve", "generate", "total"}


def test_unanswerable_question_is_refused_without_sources(loaded):
    r = loaded.query("Give me a recipe for chocolate brownies")
    assert r.answer == REFUSAL and not r.grounded and r.sources == []
    assert r.confidence == "none" and r.timings_ms["generate"] == 0


def test_delete_document(loaded):
    doc = next(d for d in loaded.list_documents() if d["source"] == "expense_policy.md")
    assert loaded.delete_document(doc["doc_id"]) == doc["chunks"]
    assert loaded.delete_document(doc["doc_id"]) == 0
    assert loaded.query("What is the hotel reimbursement limit per night?").sources == [] or all(
        s["source"] != "expense_policy.md" for s in loaded.query("hotel reimbursement limit per night").sources)


def test_query_embedding_cache(loaded):
    loaded.query("What is the minimum password length?")
    before = loaded.embedder.hits
    loaded.query("What is the minimum password length?")
    assert loaded.embedder.hits == before + 1


def test_evaluation_set_quality(loaded):
    """Sanity gate over eval/queries.json (offline lexical backend; real HF models should do better)."""
    res = evaluate(loaded, json.loads((ROOT / "eval" / "queries.json").read_text()))
    assert res["hit_rate"] >= 0.85, [r["question"] for r in res["rows"] if not r["hit"]]
    assert res["keyword_rate"] >= 0.75, [r["question"] for r in res["rows"] if not r["keyword"]]
    assert res["refusal_rate"] == 1.0
