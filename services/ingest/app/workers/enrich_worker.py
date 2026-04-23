"""GROBID enrichment worker (async/queued).

Responsibilities
----------------
- Consume GrobidEnrichmentRequestedEvents.
- Send PDF to GROBID web-service (TEI output).
- Extract title, authors, references, coordinates.
- Write to grobid_* fields — NEVER touch docling_* fields.
- Call resolver to compute/update resolved_fields.
- Transition enrichment_status via validate_enrichment_transition().
- Set has_authors, has_references flags.
- Emit GrobidEnrichmentCompletedEvent.
- Trigger incremental reindex:
    * metadata fields only (resolved_title, resolved_authors, year, venue)
    * new GROBID reference chunks
    * body chunks NOT re-embedded unless text changed
- After reindex: emit SearchProjectionActivatedEvent.
- Set is_enriched_searchable = True.

Design notes
------------
- GROBID runs ASYNC — does not block ingestion throughput.
- Use GROBID web-service mode (recommended for production).
- Store grobid_version on Document for reproducibility.
"""
import logging

from sciqa_schema import Document, EnrichmentStatus
from sciqa_schema.transitions import validate_enrichment_transition

logger = logging.getLogger(__name__)


async def run_enrichment(doc_id: str) -> Document:
    """Enrich an already-parsed document with GROBID metadata.

    Implementation steps:
    1. Retrieve Document + source PDF from storage.
    2. POST to GROBID /api/processHeaderDocument (or full-text endpoint).
    3. Parse TEI XML response.
    4. Populate grobid_title, grobid_authors, grobid_references, grobid_biblio.
    5. Call resolver: resolve_document_fields(docling_out, grobid_out, policy).
    6. Transition enrichment_status: PENDING -> RUNNING -> COMPLETE.
    7. Set has_authors = True, has_references = True.
    8. Trigger incremental reindex (metadata + reference chunks only).
    """
    # TODO: implement
    raise NotImplementedError
