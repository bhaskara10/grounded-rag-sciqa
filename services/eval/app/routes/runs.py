"""Evaluation run routes.

POST /runs        execute the harness on a dataset + corpus (synchronous)
GET  /runs        list stored result artifacts
GET  /runs/{name} return one stored result artifact
"""
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from services.query.app.core.factory import (
    config_from_env,
    encoder_from_env,
    reranker_from_env,
)
from services.query.app.core.pipeline import RagPipeline

from ..core.harness import load_corpus, load_examples, run_eval, write_report

logger = logging.getLogger(__name__)
router = APIRouter()

RESULTS_DIR = Path("results")


class RunRequest(BaseModel):
    dataset: str
    corpus: str
    output_name: str
    k: int = 5


class RunSummary(BaseModel):
    benchmark: str
    n_questions: int
    retrieval: dict[str, Any]
    answer: dict[str, Any]
    ops: dict[str, Any]
    artifact: str


@router.post("/", response_model=RunSummary)
async def execute_run(request: RunRequest) -> RunSummary:
    """Run the eval harness and persist the result artifact."""
    dataset_path = Path(request.dataset)
    corpus_path = Path(request.corpus)
    if not dataset_path.is_file() or not corpus_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="dataset or corpus not found"
        )

    encoder = encoder_from_env()
    reranker = reranker_from_env()
    config = config_from_env()
    corpus = load_corpus(corpus_path)
    examples = load_examples(dataset_path)
    pipeline = RagPipeline(corpus, encoder, reranker=reranker, config=config)
    report = run_eval(
        pipeline,
        examples,
        benchmark=dataset_path.stem,
        k=request.k,
        config={
            "encoder": encoder.name,
            "reranker": reranker.name,
            "min_rerank_score": config.min_rerank_score,
            "n_corpus_chunks": len(corpus),
        },
    )
    artifact = RESULTS_DIR / f"{request.output_name}.json"
    write_report(report, artifact)
    logger.info("eval run complete: %s", artifact)
    return RunSummary(
        benchmark=report["benchmark"],
        n_questions=report["config"]["n_questions"],
        retrieval=report["retrieval"],
        answer=report["answer"],
        ops=report["ops"],
        artifact=str(artifact),
    )


@router.get("/", response_model=list[str])
async def list_runs() -> list[str]:
    if not RESULTS_DIR.is_dir():
        return []
    return sorted(path.stem for path in RESULTS_DIR.glob("*.json"))


@router.get("/{name}")
async def get_run(name: str) -> dict[str, Any]:
    artifact = RESULTS_DIR / f"{name}.json"
    if not artifact.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no run named '{name}'"
        )
    return json.loads(artifact.read_text())
