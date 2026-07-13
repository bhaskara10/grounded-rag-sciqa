from fastapi.testclient import TestClient

from services.ingest.app.core.ingest import ingest_pdf_bytes
from services.ingest.app.main import app as ingest_app
from services.query.app.core.encoders import HashingEncoder
from services.query.app.core.index_store import LocalIndexStore
from services.query.app.main import app as query_app


def test_ingest_is_idempotent_by_content_hash(tmp_path, paper_pdf_bytes):
    store = LocalIndexStore(tmp_path / "index")
    encoder = HashingEncoder()

    first = ingest_pdf_bytes(
        paper_pdf_bytes, store=store, encoder=encoder, filename="paper.pdf"
    )
    second = ingest_pdf_bytes(
        paper_pdf_bytes, store=store, encoder=encoder, filename="renamed.pdf"
    )

    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.doc_id == first.doc_id
    assert store.load_manifest().chunk_count == first.chunk_count


def test_ingested_chunks_carry_layout_metadata(tmp_path, paper_pdf_bytes):
    store = LocalIndexStore(tmp_path / "index")

    result = ingest_pdf_bytes(
        paper_pdf_bytes, store=store, encoder=HashingEncoder(), filename="paper.pdf"
    )

    chunks, embeddings, manifest = store.load()
    assert result.title == "Grounded Retrieval for Science"
    assert len(chunks) == len(embeddings) == result.chunk_count
    assert any(chunk.chunk_type == "table" for chunk in chunks)
    assert any(chunk.section_path == ["Results"] for chunk in chunks)
    assert all(chunk.page_start in (1, 2) for chunk in chunks)


def test_upload_endpoint_ingests_and_reports(monkeypatch, tmp_path, paper_pdf_bytes):
    monkeypatch.setenv("SCIQA_ENCODER", "hashing")
    monkeypatch.setenv("SCIQA_INDEX_DIR", str(tmp_path / "index"))

    with TestClient(ingest_app) as client:
        response = client.post(
            "/documents/",
            files={"file": ("paper.pdf", paper_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201
        payload = response.json()
        assert payload["deduplicated"] is False
        assert payload["chunk_count"] > 0

        listing = client.get("/documents/").json()
        assert listing["documents"][0]["doc_id"] == payload["doc_id"]

        status = client.get(f"/documents/{payload['doc_id']}/status").json()
        assert status["indexed"] is True
        assert status["chunk_count"] == payload["chunk_count"]


def test_upload_rejects_empty_and_unparseable_files(monkeypatch, tmp_path):
    monkeypatch.setenv("SCIQA_ENCODER", "hashing")
    monkeypatch.setenv("SCIQA_INDEX_DIR", str(tmp_path / "index"))

    with TestClient(ingest_app) as client:
        empty = client.post("/documents/", files={"file": ("e.pdf", b"", "application/pdf")})
        assert empty.status_code == 400

        garbage = client.post(
            "/documents/", files={"file": ("bad.pdf", b"not a pdf", "application/pdf")}
        )
        assert garbage.status_code == 422


def test_pdf_to_answer_end_to_end(monkeypatch, tmp_path, paper_pdf_bytes):
    """The demo path: ingest a PDF, then ask the query service about it."""
    monkeypatch.setenv("SCIQA_ENCODER", "hashing")
    monkeypatch.setenv("SCIQA_RERANKER", "lexical")
    monkeypatch.setenv("SCIQA_MIN_RERANK_SCORE", "0.05")
    monkeypatch.setenv("SCIQA_INDEX_DIR", str(tmp_path / "index"))

    with TestClient(ingest_app) as ingest_client:
        upload = ingest_client.post(
            "/documents/",
            files={"file": ("paper.pdf", paper_pdf_bytes, "application/pdf")},
        )
        assert upload.status_code == 201

    with TestClient(query_app) as query_client:
        response = query_client.post(
            "/qa/", json={"question": "How much did the method improve F1 on SciFact?"}
        )

    payload = response.json()
    assert payload["abstained"] is False
    assert "4.2" in payload["answer"]
    assert payload["sentences"][0]["supporting_spans"]
