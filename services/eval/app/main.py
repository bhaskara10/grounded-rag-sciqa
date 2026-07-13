"""
Eval service.

Endpoints
---------
POST /runs        execute the harness on a dataset + corpus
GET  /runs        list stored result artifacts
GET  /runs/{name} return one stored result artifact

Three-layer evaluation (tracked separately)
-------------------------------------------
Retrieval : Precision@5, Recall@5, nDCG@5, MRR — for both the RRF-fused
            ranking and the cross-encoder ranking (reranker lift is visible)
Answer    : ROUGE-1/2/L, BLEU, METEOR, token F1, groundedness,
            unsupported-claim rate, abstention precision/recall
Ops       : p50/p95/p99 latency, per-stage latency, token usage

Every artifact records the encoder, reranker, thresholds, and corpus size,
so results are reproducible; scripts/run_eval.py is the CLI entry point.
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
