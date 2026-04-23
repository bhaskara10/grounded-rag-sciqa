"""Retrieval-only route (for evaluation and inspection)."""
import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    doc_ids: list[str] | None = None
    top_k: int = 50                    # pre-rerank candidate count
    rerank: bool = True
    apply_author_filter: bool = False  # only valid when is_enriched_searchable


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: str
    score: float
    reranker_score: float | None = None
    section_path: list[str]
    page_start: int | None = None


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]
    request_id: str
    lexical_hit_count: int
    dense_hit_count: int


@router.post("/", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Hybrid retrieval with optional cross-encoder reranking."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not yet implemented")
