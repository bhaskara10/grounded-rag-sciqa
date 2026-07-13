import numpy as np
import pytest
from sciqa_schema import EvidenceChunk

from services.query.app.core.index_store import LocalIndexStore


def _chunks(doc_id: str, texts: list[str]) -> list[EvidenceChunk]:
    return [
        EvidenceChunk(chunk_id=f"{doc_id}:chunk:{i}", doc_id=doc_id, text=text)
        for i, text in enumerate(texts)
    ]


def _embeddings(count: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed=count)
    return rng.normal(size=(count, dim)).astype(np.float32)


def test_add_and_load_roundtrip(tmp_path):
    store = LocalIndexStore(tmp_path / "index")
    chunks = _chunks("doc-1", ["first chunk", "second chunk"])

    store.add_document(
        doc_id="doc-1",
        sha256="abc123",
        chunks=chunks,
        embeddings=_embeddings(2),
        encoder_name="hashing-256",
    )

    loaded_chunks, embeddings, manifest = store.load()
    assert [chunk.chunk_id for chunk in loaded_chunks] == [c.chunk_id for c in chunks]
    assert embeddings.shape == (2, 8)
    assert manifest.documents == {"doc-1": "abc123"}


def test_appending_second_document_keeps_first(tmp_path):
    store = LocalIndexStore(tmp_path / "index")
    store.add_document(
        doc_id="doc-1",
        sha256="sha-1",
        chunks=_chunks("doc-1", ["one"]),
        embeddings=_embeddings(1),
        encoder_name="hashing-256",
    )
    store.add_document(
        doc_id="doc-2",
        sha256="sha-2",
        chunks=_chunks("doc-2", ["two", "three"]),
        embeddings=_embeddings(2),
        encoder_name="hashing-256",
    )

    chunks, embeddings, manifest = store.load()
    assert manifest.chunk_count == 3
    assert len(chunks) == len(embeddings) == 3


def test_sha_lookup_supports_idempotent_ingest(tmp_path):
    store = LocalIndexStore(tmp_path / "index")
    store.add_document(
        doc_id="doc-1",
        sha256="sha-1",
        chunks=_chunks("doc-1", ["one"]),
        embeddings=_embeddings(1),
        encoder_name="hashing-256",
    )

    assert store.doc_id_for_sha("sha-1") == "doc-1"
    assert store.doc_id_for_sha("unknown") is None


def test_encoder_mismatch_is_rejected(tmp_path):
    store = LocalIndexStore(tmp_path / "index")
    store.add_document(
        doc_id="doc-1",
        sha256="sha-1",
        chunks=_chunks("doc-1", ["one"]),
        embeddings=_embeddings(1),
        encoder_name="hashing-256",
    )

    with pytest.raises(ValueError, match="encoder"):
        store.add_document(
            doc_id="doc-2",
            sha256="sha-2",
            chunks=_chunks("doc-2", ["two"]),
            embeddings=_embeddings(1),
            encoder_name="minilm",
        )


def test_duplicate_doc_id_is_rejected(tmp_path):
    store = LocalIndexStore(tmp_path / "index")
    store.add_document(
        doc_id="doc-1",
        sha256="sha-1",
        chunks=_chunks("doc-1", ["one"]),
        embeddings=_embeddings(1),
        encoder_name="hashing-256",
    )

    with pytest.raises(ValueError, match="already indexed"):
        store.add_document(
            doc_id="doc-1",
            sha256="sha-other",
            chunks=_chunks("doc-1", ["again"]),
            embeddings=_embeddings(1),
            encoder_name="hashing-256",
        )


def test_misaligned_rows_are_rejected(tmp_path):
    store = LocalIndexStore(tmp_path / "index")

    with pytest.raises(ValueError, match="row-aligned"):
        store.add_document(
            doc_id="doc-1",
            sha256="sha-1",
            chunks=_chunks("doc-1", ["one", "two"]),
            embeddings=_embeddings(1),
            encoder_name="hashing-256",
        )
