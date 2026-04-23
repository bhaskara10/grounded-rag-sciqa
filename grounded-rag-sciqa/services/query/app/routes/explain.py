"""Retrieval trace + reranker score explanation (no answer generated)."""
import logging

from fastapi import APIRouter, HTTPException, status
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
    request_id: str


@router.post("/", response_model=ExplainResponse)
async def explain(request: ExplainRequest) -> ExplainResponse:
    """Return the retrieval trace for a question without generating an answer."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not yet implemented")
