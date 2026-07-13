"""Grounded Q&A route."""
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from sciqa_schema import EvidenceChunk, VerifiedSentence

from ..core.pipeline import RagPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


class QARequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None   # scope to specific docs if provided
    passages: list[str] | None = None  # ad-hoc mode: answer over inline passages


class QAResponse(BaseModel):
    answer: str
    sentences: list[VerifiedSentence]
    abstained: bool
    abstain_reason: str | None = None
    best_supporting_passages: list[str] | None = None
    # traceability fields — returned for every request
    request_id: str
    retrieved_chunk_ids: list[str]
    selected_chunk_ids: list[str]
    reranker_scores: list[float]
    evidence_tokens: int
    answer_tokens: int
    timings_ms: dict[str, float]


@router.post("/", response_model=QAResponse)
async def answer_question(request: Request, body: QARequest) -> QAResponse:
    """Answer a question with span-level citations.

    Returns abstained=True when the grounding contract cannot be met. With
    inline ``passages`` the question is answered over just those passages;
    otherwise the persistent index built by the ingest service is used.
    """
    if body.passages:
        doc_id = body.doc_ids[0] if body.doc_ids else "inline"
        chunks = [
            EvidenceChunk(
                chunk_id=f"{doc_id}:passage:{index}",
                doc_id=doc_id,
                text=passage,
            )
            for index, passage in enumerate(body.passages)
        ]
        pipeline = RagPipeline(
            chunks,
            request.app.state.encoder,
            reranker=request.app.state.reranker,
            config=request.app.state.config,
        )
        result = pipeline.answer(body.question)
    else:
        pipeline = request.app.state.pipeline
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="no index found — ingest documents first or pass inline passages",
            )
        result = pipeline.answer(body.question, doc_ids=body.doc_ids)

    return QAResponse(
        answer=result.answer,
        sentences=result.sentences,
        abstained=result.abstained,
        abstain_reason=result.abstain_reason,
        best_supporting_passages=result.best_supporting_passages,
        request_id=str(uuid.uuid4()),
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        selected_chunk_ids=result.selected_chunk_ids,
        reranker_scores=result.reranker_scores,
        evidence_tokens=result.evidence_tokens,
        answer_tokens=result.answer_tokens,
        timings_ms=result.timings_ms,
    )
