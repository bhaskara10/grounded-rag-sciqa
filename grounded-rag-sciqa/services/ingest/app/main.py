"""
Ingest service.

Endpoints
---------
POST /documents                  upload PDF, trigger ingestion
POST /documents/{id}/reingest    force re-parse (e.g. after parser upgrade)
GET  /documents/{id}/status      check parse / enrichment / index status

Ingestion pipeline
------------------
1. Compute SHA-256 — if identical version exists, return existing doc_id (idempotent).
2. Store raw file in object storage.
3. Run Docling inline — normalise to internal Document schema, produce chunks.
4. Embed body/table chunks, write to OpenSearch, emit BodyChunksIndexedEvent.
5. Publish GrobidEnrichmentRequestedEvent — async enrichment worker picks it up.
6. Enrichment worker: GROBID -> grobid_* fields -> resolver -> incremental reindex.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.documents import router as documents_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    logger.info("ingest-service starting")
    # TODO: init DB pool, OpenSearch client, object-storage client, task queue
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
