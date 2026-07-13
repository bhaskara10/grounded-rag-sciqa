"""Retrieval-only route (for evaluation and inspection)."""
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel
from sciqa_schema import EvidenceChunk

logger = logging.getLogger(__name__)
router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    doc_ids: list[str] | None = None
    top_k: int = 10
    rerank: bool = True


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: str
    score: float
    section_path: list[str]
    page_start: int | None = None


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]
    request_id: str
    lexical_hit_count: int
    dense_hit_count: int


@router.post("/", response_model=RetrieveResponse)
async def retrieve(request: Request, body: RetrieveRequest) -> RetrieveResponse:
    """Hybrid retrieval with optional cross-encoder reranking."""
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no index found — ingest documents first",
        )

    trace = pipeline.trace(body.query, doc_ids=body.doc_ids)
    ranked = trace.reranked if body.rerank else trace.fused
    return RetrieveResponse(
        chunks=[_to_response_chunk(chunk) for chunk in ranked[: body.top_k]],
        request_id=str(uuid.uuid4()),
        lexical_hit_count=len(trace.lexical),
        dense_hit_count=len(trace.dense),
    )


def _to_response_chunk(chunk: EvidenceChunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        chunk_type=chunk.chunk_type,
        score=round(chunk.score, 4),
        section_path=chunk.section_path,
        page_start=chunk.page_start,
    )
