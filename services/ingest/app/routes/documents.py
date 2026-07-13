"""Document ingestion routes."""
import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from ..core.ingest import ingest_pdf_bytes

logger = logging.getLogger(__name__)
router = APIRouter()


class IngestResponse(BaseModel):
    doc_id: str
    title: str
    page_count: int
    chunk_count: int
    deduplicated: bool


class DocumentSummary(BaseModel):
    doc_id: str
    sha256: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    chunk_count: int
    encoder_name: str | None = None


class StatusResponse(BaseModel):
    doc_id: str
    indexed: bool
    chunk_count: int


@router.post("/", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF to ingest")],
) -> IngestResponse:
    """Upload a PDF: parse, chunk, embed, index.

    Idempotent — re-uploading the same bytes returns the existing doc_id.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty upload")
    try:
        result = ingest_pdf_bytes(
            data,
            store=request.app.state.store,
            encoder=request.app.state.encoder,
            filename=file.filename or "document.pdf",
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error

    logger.info(
        "ingested doc_id=%s chunks=%d deduplicated=%s",
        result.doc_id,
        result.chunk_count,
        result.deduplicated,
    )
    return IngestResponse(
        doc_id=result.doc_id,
        title=result.title,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        deduplicated=result.deduplicated,
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(request: Request) -> DocumentListResponse:
    """List indexed documents."""
    manifest = request.app.state.store.load_manifest()
    if manifest is None:
        return DocumentListResponse(documents=[], chunk_count=0)
    return DocumentListResponse(
        documents=[
            DocumentSummary(doc_id=doc_id, sha256=sha)
            for doc_id, sha in sorted(manifest.documents.items())
        ],
        chunk_count=manifest.chunk_count,
        encoder_name=manifest.encoder_name,
    )


@router.get("/{doc_id}/status", response_model=StatusResponse)
async def get_status(request: Request, doc_id: str) -> StatusResponse:
    """Return index status for one document."""
    manifest = request.app.state.store.load_manifest()
    if manifest is None or doc_id not in manifest.documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown doc_id '{doc_id}'"
        )
    chunks, _, _ = request.app.state.store.load()
    return StatusResponse(
        doc_id=doc_id,
        indexed=True,
        chunk_count=sum(chunk.doc_id == doc_id for chunk in chunks),
    )
