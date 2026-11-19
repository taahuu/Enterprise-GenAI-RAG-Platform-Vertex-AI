# Enterprise GenAI / RAG Platform – Vertex AI

Enterprise GenAI solution using Vertex AI, retrieval-augmented generation, and vector search to produce
grounded responses from enterprise knowledge. This repo runs the whole pipeline on **Hugging Face models by
default** (free, local) and can switch to **Vertex AI (Gemini)** with two environment variables.

**Technologies:** Python, Google Cloud, Vertex AI, RAG, Vector Database, LLMs, LangChain, FastAPI

## What each project bullet maps to

| Project bullet | Where it lives |
|---|---|
| End-to-end workflow: ingestion, chunking, embeddings, vector retrieval, prompt construction, response generation | `app/pipeline.py` orchestrates `ingestion/loaders.py` → `ingestion/chunking.py` → `embeddings.py` → `vectorstore.py` → `retriever.py` → `prompt.py` → `llm.py` → `postprocess.py` |
| Integrated with backend APIs and application services | `app/main.py` (FastAPI: upload, list, delete, query, stats, API-key auth, health) |
| Improved retrieval and response generation for relevance and efficiency | hybrid rerank (dense + keyword coverage), score threshold with refusal, MMR de-duplication, query-embedding LRU cache, batched embedding, content-hash dedupe, replace-on-reupload, no LLM call when nothing relevant is found |
| Tested across different queries for retrieval, relevance and response quality | `tests/` (30 tests), `eval/queries.json` + `python -m scripts.evaluate` (hit@k, keyword, grounded and refusal rates, latency) |

## Architecture

```
upload (PDF/DOCX/HTML/TXT/MD)
   │  POST /v1/documents
   ▼
loaders ──► clean text + page numbers
   ▼
chunking (LangChain RecursiveCharacterTextSplitter, 800 chars, 120 overlap, per-page provenance)
   ▼
embeddings (HF sentence-transformers via LangChain HuggingFaceEmbeddings)  ──►  vector store (cosine; numpy or FAISS)
                                                                              persisted in data/index
question ─ POST /v1/query
   ▼
embed query (LRU cache) ─► top-K dense ─► + keyword coverage (hybrid) ─► min-score gate ─► MMR
   ▼                                                                      │ nothing relevant → refusal, no LLM call
prompt builder (numbered source-labelled context + grounding rules + injection guard)
   ▼
LLM (HF transformers: Qwen2.5 / flan-t5 / ...)  ─►  citation validation  ─►  answer + sources + timings + audit log
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                   # defaults already use Hugging Face models

python -m scripts.ingest                               # indexes data/sample_docs (models download on first run)
python -m scripts.evaluate                             # quality report over eval/queries.json
uvicorn app.main:app --port 8000                       # API + Swagger UI at http://localhost:8000/docs
```

Try it:

```bash
curl -F "file=@data/sample_docs/hr_leave_policy.md" localhost:8000/v1/documents
curl -s localhost:8000/v1/query -H 'content-type: application/json' \
     -d '{"question":"How many days of annual leave do employees get?"}' | python -m json.tool
```

Response shape:

```json
{
  "request_id": "9f2c1a…",
  "answer": "Full-time employees receive 24 days of paid annual leave per year. [1]",
  "grounded": true,
  "sources": [{"n": 1, "source": "hr_leave_policy.md", "page": null, "score": 0.71, "snippet": "…"}],
  "timings_ms": {"retrieve": 12.4, "generate": 830.1, "total": 842.5},
  "models": {"embedding": "sentence-transformers/all-MiniLM-L6-v2", "llm": "Qwen/Qwen2.5-0.5B-Instruct"}
}
```

`grounded` is true only when the answer cites at least one valid context block. Unanswerable questions return the
fixed refusal text with no sources.

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness probe (never needs a key) |
| POST | `/v1/documents` | upload and index a file (415 unsupported type, 422 no text, 413 too large) |
| GET | `/v1/documents` | list indexed documents |
| DELETE | `/v1/documents/{doc_id}` | remove a document and all its chunks |
| POST | `/v1/query` | `{"question", "top_k"?, "doc_ids"?}` → grounded answer with sources |
| POST | `/v1/search` | retrieval-only diagnostics with confidence, thresholds, scores, and hits |
| GET | `/v1/stats` | counts, models, cache hits |

Set `API_KEY` in `.env` to require an `X-API-Key` header on everything except `/health`.

## Configuration

All settings are environment variables (see `.env.example`).

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_BACKEND` / `EMBEDDING_MODEL` | `hf` / `sentence-transformers/all-MiniLM-L6-v2` | any sentence-transformers model |
| `LLM_BACKEND` / `LLM_MODEL` | `hf` / `Qwen/Qwen2.5-0.5B-Instruct` | chat models and flan-t5 both work; larger models answer better |
| `VECTOR_BACKEND` | `numpy` | `faiss` needs `pip install faiss-cpu` |
| `MIN_SCORE` | `0.25` | high-confidence dense or lexical gate; raise only if off-topic matches pass |
| `FALLBACK_MIN_SCORE` | `0.10` | weak dense matches send the top 2 chunks to the LLM as low-confidence context |
| `QUERY_REWRITE` / `HEADER_BOOST` | `1` / `0.05` | identity query expansion and first-chunk ranking boost |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | characters |

## Getting right answers from your own data

After uploading, call `POST /v1/search` with the same body as `/v1/query`. Its `confidence`, `best_scores`,
threshold fields, and returned `hits` show whether a question passed as high confidence, reached the weak-match
fallback, or was refused. Dense similarity is the primary relevance signal; lexical coverage can pass the gate
independently when it reaches `0.5`.

Start with `MIN_SCORE=0.25` and `FALLBACK_MIN_SCORE=0.10`. Measure `/v1/search` on both valid and off-topic queries;
raise `MIN_SCORE` for false positives and lower `FALLBACK_MIN_SCORE` only when real weak matches are being refused.
Increase or decrease `CHUNK_SIZE` when answers are split across chunks or one chunk contains too much unrelated text;
keep `CHUNK_OVERLAP` around 15% of chunk size. Set `LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct` or a larger model in `.env`
when short extraction answers are missed or citations are inconsistent; the default 0.5B model is intentionally lightweight.

## Switching to Vertex AI

```bash
pip install google-genai
gcloud auth application-default login
# .env
EMBEDDING_BACKEND=vertex
LLM_BACKEND=vertex
GCP_PROJECT=your-project
```

Use a fresh `INDEX_DIR` because the embedding space is different. The Vertex backends are thin wrappers around
`google-genai` and were **not** executed here (no GCP credentials), so verify model names against your project.

## Testing

```bash
python -m pytest -q          # offline: hashing embeddings + extractive answerer, no downloads
```

The suite covers loaders (txt, md, html, docx, pdf), chunking, the vector store (persistence, filters, model-mismatch
guard), retrieval, MMR, prompt building, citation validation, ingestion (dedupe, replace, restart), refusal behaviour,
and an evaluation gate. API tests in `tests/test_api.py` run automatically once FastAPI and httpx are installed.

## Docker

```bash
docker build -t rag-platform .
docker run -p 8000:8000 -e API_KEY=change-me -v $PWD/data/index:/srv/data/index rag-platform
```

## Limitations

- Scanned PDFs without a text layer are rejected (add OCR before ingestion).
- The built-in store is exact search held in memory, which is ideal up to a few hundred thousand chunks. Beyond that
  use `VECTOR_BACKEND=faiss` or move to a managed vector database such as Vertex AI Vector Search.
- Single-process index: run one API worker per index directory.
- The default 0.5B generator is chosen to run on a laptop CPU. For better answers set `LLM_MODEL` to a larger model.
