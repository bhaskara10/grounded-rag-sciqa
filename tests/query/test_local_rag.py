from sciqa_schema import EvidenceChunk, GroundingVerdict
from services.query.app.core.local_rag import (
    InMemoryChunkIndex,
    answer_question_local,
    chunk_text,
)


def test_chunk_text_creates_deterministic_overlapping_chunks():
    chunks = chunk_text(
        "alpha beta gamma delta epsilon zeta eta theta",
        doc_id="doc-1",
        max_tokens=4,
        overlap=1,
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "doc-1:chunk:0",
        "doc-1:chunk:1",
        "doc-1:chunk:2",
    ]
    assert chunks[1].text == "delta epsilon zeta eta"


def test_retriever_returns_best_matching_chunk_first():
    chunks = [
        EvidenceChunk(chunk_id="a", doc_id="doc-1", text="The baseline used BM25 only."),
        EvidenceChunk(
            chunk_id="b",
            doc_id="doc-1",
            text="The proposed method improved F1 by 4.2 points.",
        ),
    ]

    results = InMemoryChunkIndex(chunks).search("What improved F1?", top_k=2)

    assert [chunk.chunk_id for chunk in results] == ["b"]


def test_local_answer_returns_verified_cited_sentence():
    chunks = [
        EvidenceChunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            text="The proposed method improved F1 by 4.2 points. The baseline used BM25.",
        )
    ]

    result = answer_question_local("How much did F1 improve?", chunks)

    assert result.abstained is False
    assert result.sentences[0].verdict == GroundingVerdict.SUPPORTED
    assert result.sentences[0].supporting_chunk_ids == ["chunk-1"]
    assert result.answer == "The proposed method improved F1 by 4.2 points."


def test_local_answer_abstains_without_retrieved_evidence():
    chunks = [EvidenceChunk(chunk_id="chunk-1", doc_id="doc-1", text="Unrelated method details.")]

    result = answer_question_local("How much did F1 improve?", chunks)

    assert result.abstained is True
    assert result.abstain_reason == "no_retrieved_evidence"


def test_local_answer_respects_top_k():
    chunks = [
        EvidenceChunk(chunk_id="chunk-1", doc_id="doc-1", text="The method improved recall."),
        EvidenceChunk(
            chunk_id="chunk-2",
            doc_id="doc-1",
            text="The method improved F1 by 4.2 points.",
        ),
    ]

    result = answer_question_local("What improved F1?", chunks, top_k=1)

    assert result.retrieved_chunk_ids == ["chunk-2"]
