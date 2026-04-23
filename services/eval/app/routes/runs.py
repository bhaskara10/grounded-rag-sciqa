"""Evaluation run routes."""
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class RunRequest(BaseModel):
    dataset_version: str
    prompt_version: str
    model_version: str
    threshold_config_version: str


class RetrievalMetrics(BaseModel):
    precision_at_5: float
    recall_at_5: float
    ndcg_at_5: float
    mrr: float
    context_precision: float
    context_recall: float


class QAMetrics(BaseModel):
    faithfulness: float
    unsupported_claim_rate: float
    citation_coverage: float
    abstention_precision: float
    answer_rate: float


class OpsMetrics(BaseModel):
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    parse_failure_rate: float
    avg_token_usage: float
    cache_hit_rate: float


class RunStatus(BaseModel):
    run_id: str
    run_type: Literal["retrieval", "qa", "summarization"]
    status: Literal["pending", "running", "complete", "failed"]
    retrieval_metrics: RetrievalMetrics | None = None
    qa_metrics: QAMetrics | None = None
    ops_metrics: OpsMetrics | None = None
    passed_regression_gate: bool | None = None


@router.post("/retrieval", response_model=RunStatus, status_code=status.HTTP_202_ACCEPTED)
async def run_retrieval_eval(request: RunRequest) -> RunStatus:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not yet implemented")


@router.post("/qa", response_model=RunStatus, status_code=status.HTTP_202_ACCEPTED)
async def run_qa_eval(request: RunRequest) -> RunStatus:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not yet implemented")


@router.post("/summarization", response_model=RunStatus, status_code=status.HTTP_202_ACCEPTED)
async def run_summarization_eval(request: RunRequest) -> RunStatus:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not yet implemented")


@router.get("/{run_id}", response_model=RunStatus)
async def get_run(run_id: str) -> RunStatus:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not yet implemented")
