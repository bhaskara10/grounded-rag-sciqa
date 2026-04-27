"""
Canonical internal schema for grounded-rag-sciqa.

Design rules
------------
1. Parser output (Docling, GROBID) never leaks beyond this layer.
2. Source-specific fields are stored separately and never overwritten.
   docling_* fields are written once by the parse step.
   grobid_* fields are written once by the enrichment step.
3. Resolved/canonical fields are a deterministic merge result computed
   by the resolver using ResolutionPolicy — never written by parsers.
4. State is always explicit — never inferred from nullable fields.
   Use enrichment_status, is_body_searchable, is_enriched_searchable.
5. Every artifact version is stored (parser, chunker, embedding model)
   so evals are reproducible and incremental reindex decisions are exact.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from .enums import ChunkSource, ChunkType, EnrichmentStatus, ParseStatus, ProjectionStatus


def utc_now() -> datetime:
    return datetime.now(UTC)


# ── sub-models ────────────────────────────────────────────────────────────────

class Span(BaseModel):
    """Physical location of text in the source PDF."""
    page: int
    bbox: list[float] | None = None        # [x0, y0, x1, y1] normalised 0–1
    char_start: int | None = None
    char_end: int | None = None


class Reference(BaseModel):
    """A single bibliography entry extracted by GROBID."""
    ref_id: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    raw_text: str | None = None


class Section(BaseModel):
    """A section node in the document hierarchy from Docling."""
    section_id: str
    title: str | None = None
    level: int                             # 1 = top-level, 2 = sub, …
    path: list[str]                        # ["Introduction", "Background"]


class Table(BaseModel):
    """A table extracted from the document."""
    table_id: str
    caption: str | None = None
    page: int | None = None
    bbox: list[float] | None = None
    rows: list[list[str]] = Field(default_factory=list)
    source: ChunkSource = ChunkSource.DOCLING


class Figure(BaseModel):
    """A figure extracted from the document."""
    figure_id: str
    caption: str | None = None
    page: int | None = None
    bbox: list[float] | None = None


# ── resolution ────────────────────────────────────────────────────────────────

class ResolutionPolicy(BaseModel):
    """Versioned merge policy for resolving source-specific fields.

    Storing policy_version on Document means you can re-resolve without
    re-parsing when heuristics improve.
    """
    policy_version: str = "1.0"
    prefer_grobid_title_threshold: float = 0.8
    merge_authors_deduplicate: bool = True
    prefer_grobid_year: bool = True


class ResolvedFields(BaseModel):
    """Canonical fields derived by applying ResolutionPolicy.

    These are the fields used by serving and indexing.
    NEVER written directly by parsers.
    """
    resolved_title: str | None = None
    resolved_authors: list[str] = Field(default_factory=list)
    resolved_year: int | None = None
    resolved_venue: str | None = None
    resolution_policy_version: str = "1.0"
    resolved_at: datetime | None = None


# ── core models ───────────────────────────────────────────────────────────────

class Document(BaseModel):
    """Canonical document record.

    Source-specific fields (docling_*, grobid_*) preserve raw parser output
    and are never overwritten. Resolved fields are the canonical serving
    values computed by the resolver from both sources.

    Queryability contract
    ---------------------
    is_body_searchable     True after body_chunks_indexed event.
                           Full-text and semantic retrieval are available.
    is_enriched_searchable True after search_projection_activated event.
                           Author/citation filters become available.

    RULE: query service MUST gate filters on these flags — never on
    nullable fields like `grobid_authors != []`.
    """
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_uri: str
    sha256: str

    # status
    parse_status: ParseStatus = ParseStatus.PENDING
    enrichment_status: EnrichmentStatus = EnrichmentStatus.PENDING
    parsed_at: datetime | None = None
    enriched_at: datetime | None = None

    # parser version tracking
    docling_version: str | None = None
    grobid_version: str | None = None

    # source-specific fields — NEVER overwrite, always preserve provenance
    docling_title: str | None = None
    docling_authors: list[str] = Field(default_factory=list)
    grobid_title: str | None = None
    grobid_authors: list[str] = Field(default_factory=list)
    grobid_references: list[Reference] = Field(default_factory=list)
    grobid_biblio: dict[str, object] | None = None  # raw TEI bibliographic dict

    # resolved / canonical fields — deterministic merge result
    resolved_fields: ResolvedFields | None = None
    has_reference_graph: bool = False

    # document content (populated by Docling primary parse)
    abstract: str | None = None
    sections: list[Section] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)

    # retry metadata
    parse_retry_count: int = 0
    parse_last_error: str | None = None
    parse_last_attempted_at: datetime | None = None
    enrich_retry_count: int = 0
    enrich_last_error: str | None = None
    enrich_last_attempted_at: datetime | None = None

    # explicit queryability flags — set by pipeline, read by query service
    is_body_searchable: bool = False
    is_enriched_searchable: bool = False
    has_authors: bool = False
    has_references: bool = False

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A single indexable unit of document content.

    source tells the query service and reindex logic which parser created
    this chunk:
      DOCLING  chunks exist after primary parse
      GROBID   chunks are added after async enrichment completes
      DERIVED  chunks are post-processing artifacts

    parent_chunk_id supports hierarchical chunking: a small serving chunk
    points to its larger parent context used during generation.

    Version fields (parser_version, chunker_version, embedding_model_version)
    are required for correct incremental reindex decisions.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    version_id: str

    text: str
    chunk_type: ChunkType
    source: ChunkSource

    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    token_count: int | None = None
    provenance: list[Span] = Field(default_factory=list)

    # metadata for filtering, retrieval policy, and debugging
    section_title: str | None = None
    paper_title: str | None = None
    year: int | None = None
    venue: str | None = None
    chunk_index: int | None = None
    parent_chunk_id: str | None = None
    contains_numbers: bool = False
    contains_table: bool = False
    citation_density: float = 0.0
    language: str = "en"
    doc_hash: str | None = None

    # version provenance — required for incremental reindex decisions
    parser_version: str | None = None
    chunker_version: str | None = None
    embedding_model_version: str | None = None

    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)


class SearchProjection(BaseModel):
    """Tracks a document's projection state in the search index.

    Separation from Document is intentional:
      enrichment_status on Document  = DB / metadata-store truth
      projection_status here         = index truth
      enriched_projection_applied    = have enrichment-derived fields
                                       been written to the index?

    Without this, you get stale-index bugs:
      "document says enriched, but author filter still misses it."
    """
    doc_id: str
    version_id: str

    lexical_projection_version: int = 0
    vector_projection_version: int = 0
    last_projected_at: datetime | None = None
    projection_status: ProjectionStatus = ProjectionStatus.PENDING
    enriched_projection_applied: bool = False

    retry_count: int = 0
    last_error: str | None = None
    last_attempted_at: datetime | None = None

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class IngestionRun(BaseModel):
    """Audit record for a single ingestion pass.

    Stores all version identifiers so evals are reproducible:
    same document + same versions = same expected output.
    """
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    version_id: str

    triggered_by: Literal["api", "reingest", "retry"] = "api"

    parse_started_at: datetime | None = None
    parse_completed_at: datetime | None = None
    enrich_requested_at: datetime | None = None
    enrich_completed_at: datetime | None = None
    index_completed_at: datetime | None = None

    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_indexed: int = 0

    # version provenance
    docling_version: str | None = None
    grobid_version: str | None = None
    chunker_version: str | None = None
    embedding_model_version: str | None = None

    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)
