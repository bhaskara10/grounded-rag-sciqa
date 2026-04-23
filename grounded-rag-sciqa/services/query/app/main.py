"""
Query service.

Endpoints
---------
POST /qa        grounded Q&A with sentence-level citations
POST /retrieve  retrieval only (for eval / inspection)
POST /explain   retrieval trace + reranker scores (no answer generated)

Per-request pipeline
--------------------
1. Query normalisation + optional classification
2. Hybrid retrieval — BM25 + dense via OpenSearch hybrid pipeline
3. Metadata filters — gated on is_enriched_searchable flag
4. Cross-encoder reranking (top 50-100 candidates)
5. Adaptive evidence selection (score threshold + token budget + diversity)
6. Structured JSON generation via vLLM
7. Sentence-level attribution
8. Abstention check
9. Return QAResponse (answer + citations) or abstention notice

Grounding contract
------------------
Every factual sentence MUST have at least one supporting_chunk_id.
If attribution cannot be completed for a sentence:
  a) rewrite as uncertainty, OR
  b) set abstained = True for the whole response.

Abstention triggers (system decision, not prompt)
  - top reranker score below threshold
  - evidence set too small
  - evidence clusters disagree
  - no sentence-level attribution completable
  - required facts not found in retrieved context
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.explain import router as explain_router
from .routes.qa import router as qa_router
from .routes.retrieve import router as retrieve_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    logger.info("query-service starting")
    # TODO: init OpenSearch client, embedding client, reranker, vLLM client
    yield
    logger.info("query-service stopping")


app = FastAPI(
    title="sciqa-query",
    version="0.1.0",
    description="Grounded Q&A service for grounded-rag-sciqa",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(qa_router, prefix="/qa", tags=["qa"])
app.include_router(retrieve_router, prefix="/retrieve", tags=["retrieve"])
app.include_router(explain_router, prefix="/explain", tags=["explain"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "query"}
