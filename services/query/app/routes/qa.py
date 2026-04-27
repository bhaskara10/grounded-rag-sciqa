"""Grounded Q&A route."""
import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sciqa_schema import EvidenceChunk, VerifiedSentence

from ..core.local_rag import answer_question_local

logger = logging.getLogger(__name__)
router = APIRouter()


class QARequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None   # scope to specific docs if provided
    max_chunks: int = 10               # upper bound; adaptive policy may use fewer
    passages: list[str] | None = None  # local demo path before persistent indexing exists


class QAResponse(BaseModel):
    answer: str
    sentences: list[VerifiedSentence]
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
    if not request.passages:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="local demo path requires inline passages until indexing is implemented",
        )

    doc_id = request.doc_ids[0] if request.doc_ids else "inline"
    chunks = [
        EvidenceChunk(
            chunk_id=f"{doc_id}:passage:{index}",
            doc_id=doc_id,
            text=passage,
        )
        for index, passage in enumerate(request.passages)
    ]
    result = answer_question_local(
        request.question,
        chunks,
        top_k=request.max_chunks,
    )

    return QAResponse(
        answer=result.answer,
        sentences=result.sentences,
        abstained=result.abstained,
        abstain_reason=result.abstain_reason,
        best_supporting_passages=result.best_supporting_passages,
        request_id=str(uuid.uuid4()),
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        reranker_scores=[],
    )
