"""Docling parse worker (inline with ingestion).

Responsibilities
----------------
- Run Docling on stored PDF.
- Normalise output into internal Document schema.
- Produce section-aware chunks: abstract, paragraph, table, caption.
- Write Document + Chunks to metadata store.
- Update parse_status via validate_parse_transition().
- Emit DocumentParsedEvent.

Design notes
------------
- Docling is the PRIMARY parser — runs synchronously in the ingestion path.
- GROBID enrichment is separate and async (see enrich_worker.py).
- On reingest, only chunks whose text changed are re-embedded/re-indexed.
"""
import logging
from pathlib import Path

from sciqa_schema import Document

logger = logging.getLogger(__name__)


async def run_parse(doc_id: str, source_path: Path) -> Document:
    """Parse a PDF with Docling and return a populated Document.

    Implementation steps:
    1. Load PDF with Docling DocumentConverter.
    2. Extract sections, tables, figures into internal schema.
    3. Run section-aware chunker (structural first, windowed fallback).
    4. Transition parse_status: PENDING -> RUNNING -> COMPLETE.
    5. Set is_body_searchable = False (set True after chunks are indexed).
    """
    # TODO: implement
    raise NotImplementedError
