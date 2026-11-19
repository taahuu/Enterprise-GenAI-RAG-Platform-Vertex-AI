"""FastAPI backend exposing the RAG pipeline.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
import hmac
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .ingestion.loaders import EmptyDocumentError, UnsupportedFileType
from .pipeline import RAGPipeline, result_to_dict
from .vectorstore import IndexMismatchError

logging.basicConfig(level=logging.INFO, format="%(message)s")


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(get_settings())


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = get_settings().api_key
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ------------------------------------------------------------------ schemas
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    doc_ids: Optional[list[str]] = Field(default=None, description="Restrict retrieval to these documents")


class Source(BaseModel):
    n: int
    chunk_id: str
    doc_id: Optional[str]
    source: Optional[str]
    page: Optional[int]
    score: float
    snippet: str


class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    grounded: bool
    sources: list[Source]
    timings_ms: dict
    models: dict
    confidence: Optional[str] = None


class SearchHit(BaseModel):
    chunk_id: str
    source: Optional[str]
    page: Optional[int]
    dense: float
    lexical: float
    score: float
    text: str


class SearchResponse(BaseModel):
    confidence: str
    min_score: float
    fallback_min_score: float
    best_scores: dict
    hits: list[SearchHit]


class IngestResponse(BaseModel):
    doc_id: str
    source: str
    chunks: int
    status: str
    replaced_doc_ids: list[str] = []


# ---------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    if get_settings().preload:
        get_pipeline().warmup()  # load models at startup, not on the first user request
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise GenAI / RAG Platform", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/v1/stats", dependencies=[Depends(require_api_key)])
    def stats(p: RAGPipeline = Depends(get_pipeline)):
        return p.stats()

    @app.post("/v1/documents", response_model=IngestResponse, status_code=201,
              dependencies=[Depends(require_api_key)])
    def upload(file: UploadFile = File(...), p: RAGPipeline = Depends(get_pipeline)):
        data = file.file.read(get_settings().max_upload_mb * 1024 * 1024 + 1)
        if len(data) > get_settings().max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File exceeds {get_settings().max_upload_mb} MB")
        try:
            res = p.ingest_bytes(data, file.filename or "upload.txt")
        except UnsupportedFileType as e:
            raise HTTPException(415, str(e))
        except EmptyDocumentError as e:
            raise HTTPException(422, str(e))
        except IndexMismatchError as e:
            raise HTTPException(409, str(e))
        return IngestResponse(**res.__dict__)

    @app.get("/v1/documents", dependencies=[Depends(require_api_key)])
    def list_documents(p: RAGPipeline = Depends(get_pipeline)):
        return p.list_documents()

    @app.delete("/v1/documents/{doc_id}", dependencies=[Depends(require_api_key)])
    def delete_document(doc_id: str, p: RAGPipeline = Depends(get_pipeline)):
        removed = p.delete_document(doc_id)
        if not removed:
            raise HTTPException(404, "Unknown doc_id")
        return {"doc_id": doc_id, "chunks_removed": removed}

    @app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
    def query(req: QueryRequest, p: RAGPipeline = Depends(get_pipeline)):
        return result_to_dict(p.query(req.question, req.top_k, req.doc_ids))

    @app.post("/v1/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
    def search(req: QueryRequest, p: RAGPipeline = Depends(get_pipeline)):
        return p.search(req.question, req.top_k, req.doc_ids)

    return app


app = create_app()
