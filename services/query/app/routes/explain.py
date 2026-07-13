"""Retrieval trace + reranker score explanation (no answer generated)."""
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class ExplainRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None


class ExplainResponse(BaseModel):
    question: str
    lexical_candidates: int
    dense_candidates: int
    after_rrf: int
    after_rerank: int
    evidence_selected: int
    top_reranker_score: float
    reranker_scores: list[float]
    timings_ms: dict[str, float]
    request_id: str


@router.post("/", response_model=ExplainResponse)
async def explain(request: Request, body: ExplainRequest) -> ExplainResponse:
    """Return the retrieval trace for a question without generating an answer."""
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no index found — ingest documents first",
        )

    trace = pipeline.trace(body.question, doc_ids=body.doc_ids)
    reranker_scores = [round(chunk.score, 4) for chunk in trace.reranked]
    return ExplainResponse(
        question=body.question,
        lexical_candidates=len(trace.lexical),
        dense_candidates=len(trace.dense),
        after_rrf=len(trace.fused),
        after_rerank=len(trace.reranked),
        evidence_selected=len(trace.selected),
        top_reranker_score=reranker_scores[0] if reranker_scores else 0.0,
        reranker_scores=reranker_scores,
        timings_ms=trace.timings_ms,
        request_id=str(uuid.uuid4()),
    )
