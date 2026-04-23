"""Grounded Q&A route."""
import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class QARequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None   # scope to specific docs if provided
    max_chunks: int = 10               # upper bound; adaptive policy may use fewer


class CitedSentence(BaseModel):
    text: str
    supporting_chunk_ids: list[str]
    confidence: float


class QAResponse(BaseModel):
    answer: str
    sentences: list[CitedSentence]
    abstained: bool
    abstain_reason: str | None = None
    best_supporting_passages: list[str] | None = None
    suggested_question: str | None = None
    # traceability fields — stored for every request
    request_id: str
    retrieved_chunk_ids: list[str]
    reranker_scores: list[float]


@router.post("/", response_model=QAResponse)
async def answer_question(request: QARequest) -> QAResponse:
    """Answer a question with sentence-level citations.

    Returns abstained=True when grounding contract cannot be met.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not yet implemented")
