import numpy as np
import pytest
from fastapi.testclient import TestClient
from sciqa_schema import EvidenceChunk

from services.query.app.core.encoders import HashingEncoder
from services.query.app.core.index_store import LocalIndexStore
from services.query.app.main import app

PASSAGES = [
    "The retrieval-augmented model improved F1 by 4.2 points on SciFact.",
    "Training used eight A100 GPUs for twelve hours.",
]


@pytest.fixture()
def deterministic_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SCIQA_ENCODER", "hashing")
    monkeypatch.setenv("SCIQA_RERANKER", "lexical")
    monkeypatch.setenv("SCIQA_MIN_RERANK_SCORE", "0.05")
    monkeypatch.setenv("SCIQA_INDEX_DIR", str(tmp_path / "no-index"))


@pytest.fixture()
def indexed_env(monkeypatch, tmp_path):
    encoder = HashingEncoder()
    chunks = [
        EvidenceChunk(chunk_id=f"doc-1:chunk:{i}", doc_id="doc-1", text=text)
        for i, text in enumerate(PASSAGES)
    ]
    store = LocalIndexStore(tmp_path / "index")
    store.add_document(
        doc_id="doc-1",
        sha256="sha-1",
        chunks=chunks,
        embeddings=np.asarray(encoder.encode([chunk.text for chunk in chunks])),
        encoder_name=encoder.name,
    )
    monkeypatch.setenv("SCIQA_ENCODER", "hashing")
    monkeypatch.setenv("SCIQA_RERANKER", "lexical")
    monkeypatch.setenv("SCIQA_MIN_RERANK_SCORE", "0.05")
    monkeypatch.setenv("SCIQA_INDEX_DIR", str(tmp_path / "index"))


def test_qa_inline_passages_returns_grounded_answer(deterministic_env):
    with TestClient(app) as client:
        response = client.post(
            "/qa/",
            json={"question": "How much did the model improve F1?", "passages": PASSAGES},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["abstained"] is False
    assert "4.2" in payload["answer"]
    assert payload["sentences"][0]["supporting_spans"]
    assert payload["timings_ms"]["total_ms"] > 0


def test_qa_without_index_or_passages_is_unavailable(deterministic_env):
    with TestClient(app) as client:
        response = client.post("/qa/", json={"question": "anything"})

    assert response.status_code == 503


def test_qa_answers_from_persistent_index(indexed_env):
    with TestClient(app) as client:
        response = client.post(
            "/qa/", json={"question": "How much did the model improve F1 on SciFact?"}
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["abstained"] is False
    assert "4.2" in payload["answer"]


def test_retrieve_returns_ranked_chunks(indexed_env):
    with TestClient(app) as client:
        response = client.post(
            "/retrieve/", json={"query": "F1 improvement on SciFact", "top_k": 1}
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["chunks"]) == 1
    assert payload["chunks"][0]["chunk_id"] == "doc-1:chunk:0"
    assert payload["lexical_hit_count"] >= 1


def test_explain_reports_stage_counts(indexed_env):
    with TestClient(app) as client:
        response = client.post(
            "/explain/", json={"question": "How much did the model improve F1?"}
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["lexical_candidates"] >= 1
    assert payload["after_rerank"] >= 1
    assert payload["reranker_scores"]
    assert payload["top_reranker_score"] == payload["reranker_scores"][0]
