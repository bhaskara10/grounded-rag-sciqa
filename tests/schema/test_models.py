"""
Tests for sciqa_schema models.

Validates:
- models instantiate with correct defaults
- source-specific and resolved fields are independent
- queryability flags default to False (never inferred from nullable fields)
- retry metadata initialises to zero
- version fields default to None (populated during ingestion, not at creation)
- chunk source tagging is preserved
"""
from sciqa_schema import Chunk, Document, IngestionRun, SearchProjection
from sciqa_schema.enums import (
    ChunkSource,
    ChunkType,
    EnrichmentStatus,
    ParseStatus,
    ProjectionStatus,
)


class TestDocument:
    def test_status_defaults(self):
        doc = Document(source_uri="s3://bucket/paper.pdf", sha256="abc123")
        assert doc.parse_status == ParseStatus.PENDING
        assert doc.enrichment_status == EnrichmentStatus.PENDING

    def test_queryability_flags_default_false(self):
        doc = Document(source_uri="s3://bucket/paper.pdf", sha256="abc123")
        # RULE: query service gates filters on these flags — never on nullable fields
        assert doc.is_body_searchable is False
        assert doc.is_enriched_searchable is False
        assert doc.has_authors is False
        assert doc.has_references is False

    def test_source_specific_fields_are_independent(self):
        doc = Document(
            source_uri="s3://bucket/paper.pdf",
            sha256="abc123",
            docling_title="Docling Title",
            grobid_title="GROBID Title",
        )
        # neither field must overwrite the other
        assert doc.docling_title == "Docling Title"
        assert doc.grobid_title == "GROBID Title"

    def test_resolved_fields_none_until_resolution_runs(self):
        doc = Document(source_uri="s3://bucket/paper.pdf", sha256="abc123")
        assert doc.resolved_fields is None

    def test_retry_metadata_defaults_zero(self):
        doc = Document(source_uri="s3://bucket/paper.pdf", sha256="abc123")
        assert doc.parse_retry_count == 0
        assert doc.enrich_retry_count == 0
        assert doc.parse_last_error is None
        assert doc.enrich_last_error is None

    def test_parser_versions_default_none(self):
        doc = Document(source_uri="s3://bucket/paper.pdf", sha256="abc123")
        assert doc.docling_version is None
        assert doc.grobid_version is None

    def test_unique_ids_per_instance(self):
        d1 = Document(source_uri="s3://a.pdf", sha256="aaa")
        d2 = Document(source_uri="s3://b.pdf", sha256="bbb")
        assert d1.doc_id != d2.doc_id
        assert d1.version_id != d2.version_id


class TestChunk:
    def _make_chunk(self, **kwargs):
        defaults = dict(
            doc_id="doc-1",
            version_id="v-1",
            text="Test chunk text.",
            chunk_type=ChunkType.PARAGRAPH,
            source=ChunkSource.DOCLING,
        )
        return Chunk(**{**defaults, **kwargs})

    def test_defaults(self):
        chunk = self._make_chunk()
        assert chunk.contains_numbers is False
        assert chunk.contains_table is False
        assert chunk.citation_density == 0.0
        assert chunk.language == "en"
        assert chunk.embedding_model_version is None  # set during embedding

    def test_grobid_reference_chunk(self):
        chunk = self._make_chunk(
            text="Smith et al., 2022. Attention is all you need.",
            chunk_type=ChunkType.REFERENCE,
            source=ChunkSource.GROBID,
        )
        assert chunk.source == ChunkSource.GROBID
        assert chunk.chunk_type == ChunkType.REFERENCE

    def test_parent_chunk_linkage(self):
        chunk = self._make_chunk(parent_chunk_id="parent-42")
        assert chunk.parent_chunk_id == "parent-42"


class TestSearchProjection:
    def test_defaults(self):
        proj = SearchProjection(doc_id="doc-1", version_id="v-1")
        assert proj.projection_status == ProjectionStatus.PENDING
        assert proj.enriched_projection_applied is False
        assert proj.lexical_projection_version == 0
        assert proj.vector_projection_version == 0
        assert proj.retry_count == 0


class TestIngestionRun:
    def test_defaults(self):
        run = IngestionRun(doc_id="doc-1", version_id="v-1")
        assert run.triggered_by == "api"
        assert run.chunks_created == 0
        assert run.chunks_indexed == 0
        assert run.error is None
        assert run.docling_version is None
