"""
Canonical status enums for the grounded-rag-sciqa pipeline.

All status fields on models MUST use these enums — never raw strings.
"""
from enum import Enum


class ParseStatus(str, Enum):
    """Lifecycle of the Docling inline parse step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class EnrichmentStatus(str, Enum):
    """Lifecycle of the async GROBID enrichment step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ProjectionStatus(str, Enum):
    """Lifecycle of the OpenSearch index projection step.

    Unlike parse/enrichment, projection is NOT terminal at COMPLETE —
    it can be re-run after enrichment finishes (see transitions.py).
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ChunkType(str, Enum):
    """Semantic type of a chunk.

    Used to:
    - route chunk types to different index fields
    - apply type-specific retrieval policy (e.g. prefer TABLE for numeric Qs)
    - gate reference/citation filters on enrichment-sourced types
    """
    ABSTRACT = "abstract"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CAPTION = "caption"
    REFERENCE = "reference"
    CITATION_CONTEXT = "citation_context"


class ChunkSource(str, Enum):
    """Which parser produced this chunk.

    - DOCLING  : body/table/caption chunks from primary inline parse
    - GROBID   : reference chunks added after async enrichment
    - DERIVED  : post-processing artifacts (e.g. citation context windows)
    """
    DOCLING = "docling"
    GROBID = "grobid"
    DERIVED = "derived"
