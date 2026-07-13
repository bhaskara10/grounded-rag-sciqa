"""Ingestion orchestration: bytes in, indexed document out.

SHA-256 keyed and idempotent — re-uploading identical bytes returns the
existing doc_id without touching the index. The index store and encoder are
owned by the query service's core; ingest is their write path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sciqa_common.utils import sha256_bytes

from services.query.app.core.encoders import TextEncoder
from services.query.app.core.index_store import LocalIndexStore

from .chunking import chunk_document
from .pdf_parser import parse_pdf_bytes

SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    title: str
    page_count: int
    chunk_count: int
    deduplicated: bool


def ingest_pdf_bytes(
    data: bytes,
    *,
    store: LocalIndexStore,
    encoder: TextEncoder,
    filename: str = "document.pdf",
) -> IngestResult:
    """Parse, chunk, embed, and index one PDF."""
    digest = sha256_bytes(data)
    existing_doc_id = store.doc_id_for_sha(digest)
    if existing_doc_id is not None:
        chunks, _, _ = store.load()
        return IngestResult(
            doc_id=existing_doc_id,
            title=existing_doc_id,
            page_count=0,
            chunk_count=sum(chunk.doc_id == existing_doc_id for chunk in chunks),
            deduplicated=True,
        )

    stem = filename.rsplit("/", 1)[-1].removesuffix(".pdf")
    parsed = parse_pdf_bytes(data, fallback_title=stem)
    doc_id = f"{_slug(stem)}-{digest[:8]}"
    chunks = chunk_document(parsed, doc_id=doc_id)
    if not chunks:
        raise ValueError(f"no extractable text in '{filename}'")

    embeddings = encoder.encode([chunk.text for chunk in chunks])
    store.add_document(
        doc_id=doc_id,
        sha256=digest,
        chunks=chunks,
        embeddings=embeddings,
        encoder_name=encoder.name,
    )
    return IngestResult(
        doc_id=doc_id,
        title=parsed.title,
        page_count=parsed.page_count,
        chunk_count=len(chunks),
        deduplicated=False,
    )


def _slug(text: str) -> str:
    slug = SLUG_RE.sub("-", text.lower()).strip("-")
    return slug or "doc"
