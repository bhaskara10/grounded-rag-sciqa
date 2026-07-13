"""
Ingest service.

Endpoints
---------
POST /documents               upload PDF: parse, chunk, embed, index
GET  /documents               list indexed documents
GET  /documents/{id}/status   per-document index status

Ingestion pipeline
------------------
1. Compute SHA-256 — identical bytes return the existing doc_id (idempotent).
2. Layout-aware parse via PyMuPDF: text blocks in reading order, font-size
   heading detection, ruled tables extracted as markdown.
3. Section-aware chunking with page/section/type metadata.
4. Bi-encoder embedding (MPS/CUDA/CPU auto-select).
5. Append to the persistent local index shared with the query service.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.query.app.core.factory import encoder_from_env, index_dir_from_env
from services.query.app.core.index_store import LocalIndexStore

from .routes.documents import router as documents_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    logger.info("ingest-service starting")
    app.state.encoder = encoder_from_env()
    app.state.store = LocalIndexStore(index_dir_from_env())
    logger.info("index directory: %s", app.state.store.root)
    yield
    logger.info("ingest-service stopping")


app = FastAPI(
    title="sciqa-ingest",
    version="0.1.0",
    description="Document ingestion service for grounded-rag-sciqa",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(documents_router, prefix="/documents", tags=["documents"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "ingest"}
