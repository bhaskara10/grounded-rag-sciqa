"""Document ingestion routes."""
import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sciqa_schema import ParseStatus

logger = logging.getLogger(__name__)
router = APIRouter()


class IngestResponse(BaseModel):
    doc_id: str
    version_id: str
    run_id: str
    message: str


class StatusResponse(BaseModel):
    doc_id: str
    parse_status: ParseStatus
    enrichment_status: str
    is_body_searchable: bool
    is_enriched_searchable: bool
    has_authors: bool
    has_references: bool


@router.post("/", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF to ingest")]
) -> IngestResponse:
    """Upload a PDF and trigger ingestion.

    Idempotent: re-uploading the same SHA-256 returns the existing doc_id.
    """
    # TODO: sha256 check -> store -> enqueue parse job
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not yet implemented")


@router.post(
    "/{doc_id}/reingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reingest_document(doc_id: str) -> IngestResponse:
    """Force re-parse (e.g. after Docling version upgrade).

    Creates a new version_id; does not delete the existing version.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not yet implemented")


@router.get("/{doc_id}/status", response_model=StatusResponse)
async def get_status(doc_id: str) -> StatusResponse:
    """Return parse / enrichment / index projection status."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not yet implemented")
