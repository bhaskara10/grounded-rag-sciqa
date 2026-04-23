"""
Eval service.

Endpoints
---------
POST /runs/retrieval      run retrieval benchmark
POST /runs/qa             run QA / grounding benchmark
POST /runs/summarization  run summarization benchmark
GET  /runs/{id}           get run status and metrics

Three-layer evaluation (tracked separately)
-------------------------------------------
Retrieval : Precision@5, Recall@5, nDCG@5, MRR,
            context precision / context recall (Ragas)
Answer    : faithfulness, unsupported-claim rate, citation coverage,
            abstention precision, answer rate (Ragas)
Ops       : p50/p95/p99 latency, parse failure rate, avg token usage,
            reranker latency, cache hit rate

Regression gate: release only when all three layers pass configured thresholds.
All runs store: dataset_version, prompt_version, model_version, threshold_config_version
so results are reproducible.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.runs import router as runs_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    logger.info("eval-service starting")
    yield
    logger.info("eval-service stopping")


app = FastAPI(
    title="sciqa-eval",
    version="0.1.0",
    description="Evaluation harness for grounded-rag-sciqa",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(runs_router, prefix="/runs", tags=["runs"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "eval"}
