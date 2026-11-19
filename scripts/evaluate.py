"""Run the RAG pipeline over a query set and report retrieval + answer quality.

    python -m scripts.evaluate                     # uses eval/queries.json against the current index
    python -m scripts.evaluate --min-hit-rate 0.8  # exit 1 if retrieval hit-rate is below 0.8

Metrics
  hit@k          expected source document appears in the cited/returned sources
  keyword        expected fact string appears in the generated answer
  refusal        unanswerable questions are refused instead of hallucinated
"""
import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.pipeline import RAGPipeline
from app.prompt import REFUSAL


def evaluate(pipe: RAGPipeline, queries: dict) -> dict:
    rows = []
    for q in queries["answerable"]:
        r = pipe.query(q["question"])
        hit = any(s["source"] == q["expected_source"] for s in r.sources)
        kw = all(k.lower() in r.answer.lower() for k in q.get("expected_keywords", []))
        rows.append({"question": q["question"], "hit": hit, "keyword": kw, "grounded": r.grounded,
                     "confidence": r.confidence, "ms": r.timings_ms["total"], "answer": r.answer})
    unanswerable_results = [pipe.query(q) for q in queries["unanswerable"]]
    refusals = [r.answer == REFUSAL for r in unanswerable_results]
    n = max(len(rows), 1)
    return {
        "rows": rows,
        "hit_rate": sum(r["hit"] for r in rows) / n,
        "keyword_rate": sum(r["keyword"] for r in rows) / n,
        "grounded_rate": sum(r["grounded"] for r in rows) / n,
        "refusal_rate": sum(refusals) / max(len(refusals), 1),
        "avg_ms": sum(r["ms"] for r in rows) / n,
        "low_confidence_rate": sum(r["confidence"] == "low" for r in rows) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/queries.json")
    ap.add_argument("--docs", default="data/sample_docs", help="ingested first if the index is empty")
    ap.add_argument("--min-hit-rate", type=float, default=0.0)
    args = ap.parse_args()

    pipe = RAGPipeline(get_settings())
    if len(pipe.store) == 0:
        for f in sorted(Path(args.docs).glob("*")):
            pipe.ingest_path(f)
    res = evaluate(pipe, json.loads(Path(args.queries).read_text(encoding="utf-8")))

    for r in res["rows"]:
        print(f"[{'OK ' if r['hit'] and r['keyword'] else 'BAD'}] {r['ms']:7.1f} ms  {r['question']}\n"
              f"        -> {r['answer'][:160]}")
    print(f"\nhit@k={res['hit_rate']:.0%}  keyword={res['keyword_rate']:.0%}  grounded={res['grounded_rate']:.0%}  "
          f"refusal={res['refusal_rate']:.0%}  avg_latency={res['avg_ms']:.1f} ms  low_confidence={res['low_confidence_rate']:.0%}")
    return 0 if res["hit_rate"] >= args.min_hit_rate else 1


if __name__ == "__main__":
    sys.exit(main())
