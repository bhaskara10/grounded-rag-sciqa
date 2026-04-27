"""
Pipeline event types for grounded-rag-sciqa.

Events map 1:1 to the state transitions in sciqa_schema.transitions.

Happy-path sequence
-------------------
1. document_uploaded
2. document_parsed
3. body_chunks_indexed           -> document becomes body-searchable
4. grobid_enrichment_requested
5. grobid_enrichment_completed
6. document_metadata_reindexed
7. reference_chunks_indexed
8. search_projection_activated   -> document becomes enriched-searchable

Producing / consuming events (rather than calling services directly)
gives you a clean async contract and a natural audit trail.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    version_id: str
    occurred_at: datetime = Field(default_factory=utc_now)


class DocumentUploadedEvent(BaseEvent):
    event_type: Literal["document_uploaded"] = "document_uploaded"
    source_uri: str
    sha256: str


class DocumentParsedEvent(BaseEvent):
    event_type: Literal["document_parsed"] = "document_parsed"
    docling_version: str
    chunks_created: int
    parse_duration_ms: int | None = None


class BodyChunksIndexedEvent(BaseEvent):
    """Emitted when Docling body/table chunks are written to OpenSearch.

    After this event:
      - full-text and semantic retrieval are available
      - author/citation filters remain gated until search_projection_activated
    """
    event_type: Literal["body_chunks_indexed"] = "body_chunks_indexed"
    chunks_indexed: int
    embedding_model_version: str


class GrobidEnrichmentRequestedEvent(BaseEvent):
    event_type: Literal["grobid_enrichment_requested"] = "grobid_enrichment_requested"
    run_id: str


class GrobidEnrichmentCompletedEvent(BaseEvent):
    event_type: Literal["grobid_enrichment_completed"] = "grobid_enrichment_completed"
    grobid_version: str
    references_extracted: int
    authors_extracted: int
    enrich_duration_ms: int | None = None


class DocumentMetadataReindexedEvent(BaseEvent):
    """Emitted after enrichment-derived metadata fields are written to OpenSearch.

    Only metadata fields are updated; body chunks are NOT re-embedded
    unless their text materially changed.
    """
    event_type: Literal["document_metadata_reindexed"] = "document_metadata_reindexed"
    fields_updated: list[str]


class ReferenceChunksIndexedEvent(BaseEvent):
    """Emitted after GROBID-derived reference chunks are indexed.
    Enables citation/bibliography navigation queries.
    """
    event_type: Literal["reference_chunks_indexed"] = "reference_chunks_indexed"
    chunks_indexed: int


class SearchProjectionActivatedEvent(BaseEvent):
    """Emitted when the index alias switches to the enriched projection.
    After this event, enriched-searchable filters (authors, citations) are live.
    """
    event_type: Literal["search_projection_activated"] = "search_projection_activated"
    lexical_projection_version: int
    vector_projection_version: int
    enriched_projection_applied: bool


# Discriminated union for deserialisation
PipelineEvent = Annotated[
    DocumentUploadedEvent
    | DocumentParsedEvent
    | BodyChunksIndexedEvent
    | GrobidEnrichmentRequestedEvent
    | GrobidEnrichmentCompletedEvent
    | DocumentMetadataReindexedEvent
    | ReferenceChunksIndexedEvent
    | SearchProjectionActivatedEvent,
    Field(discriminator="event_type"),
]
