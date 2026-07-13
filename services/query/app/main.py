"""
Query service.

Endpoints
---------
POST /qa        grounded Q&A with span-level citations
POST /retrieve  retrieval only (for eval / inspection)
POST /explain   retrieval trace + reranker scores (no answer generated)

Per-request pipeline
--------------------
1. Hybrid retrieval — Okapi BM25 + bi-encoder dense search, RRF-fused
2. Cross-encoder reranking of the fused candidates
3. Adaptive evidence selection (score threshold + token budget)
4. Extractive answer proposal (LLM generator plugs in behind the same
   Answerer protocol)
5. Grounding verification with span-level attribution
6. Return QAResponse (answer + verbatim evidence spans) or abstention

Grounding contract
------------------
Every factual sentence MUST have at least one supporting chunk, and the
verifier localizes the exact evidence characters that support it. If
attribution cannot be completed for any sentence, the whole response is
abstained — a system decision, not a prompt instruction.

Abstention triggers
  - no candidates retrieved
  - no reranked chunk clears the selection threshold (weak evidence)
  - no extractable answer sentence
  - grounding verification failed (missing citation, unknown chunk,
    numeric claim absent from evidence, insufficient span coverage)
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.factory import (
    config_from_env,
    encoder_from_env,
    index_dir_from_env,
    load_indexed_pipeline,
    reranker_from_env,
)
from .core.index_store import LocalIndexStore
from .routes.explain import router as explain_router
from .routes.qa import router as qa_router
from .routes.retrieve import router as retrieve_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    logger.info("query-service starting")
    app.state.encoder = encoder_from_env()
    app.state.reranker = reranker_from_env()
    app.state.config = config_from_env()
    store = LocalIndexStore(index_dir_from_env())
    app.state.pipeline = load_indexed_pipeline(
        store, app.state.encoder, app.state.reranker, app.state.config
    )
    if app.state.pipeline is None:
        logger.info("no persistent index at %s — /qa accepts inline passages", store.root)
    else:
        logger.info("loaded index from %s", store.root)
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
