from sciqa_schema import EvidenceChunk, GeneratedSentence, GroundingVerdict
from services.query.app.core.grounding import GroundingVerifier


def _chunk(chunk_id: str = "chunk-1", text: str | None = None) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        text=text or "The proposed method improved F1 by 4.2 points on SciFact.",
        score=0.92,
    )


def test_supported_sentence_passes_with_retrieved_citation():
    decision = GroundingVerifier().verify(
        [
            GeneratedSentence(
                text="The proposed method improved F1 by 4.2 points on SciFact.",
                supporting_chunk_ids=["chunk-1"],
                confidence=0.91,
            )
        ],
        [_chunk()],
    )

    assert decision.abstained is False
    assert decision.sentences[0].verdict == GroundingVerdict.SUPPORTED


def test_missing_supporting_chunk_ids_forces_abstention():
    decision = GroundingVerifier().verify(
        [GeneratedSentence(text="The method improved F1 by 4.2 points.")],
        [_chunk()],
    )

    assert decision.abstained is True
    assert decision.sentences[0].reason == "missing_supporting_chunk_ids"


def test_supporting_chunk_must_be_from_retrieved_set():
    decision = GroundingVerifier().verify(
        [
            GeneratedSentence(
                text="The method improved F1 by 4.2 points.",
                supporting_chunk_ids=["chunk-missing"],
            )
        ],
        [_chunk()],
    )

    assert decision.abstained is True
    assert decision.sentences[0].reason == "supporting_chunk_not_retrieved"


def test_low_overlap_sentence_is_rejected():
    decision = GroundingVerifier().verify(
        [
            GeneratedSentence(
                text="The model was trained on a multilingual legal corpus.",
                supporting_chunk_ids=["chunk-1"],
            )
        ],
        [_chunk(text="The proposed method improved F1 by 4.2 points on SciFact.")],
    )

    assert decision.abstained is True
    assert decision.sentences[0].reason == "insufficient_evidence_overlap"


def test_numeric_claim_must_appear_in_cited_evidence():
    decision = GroundingVerifier().verify(
        [
            GeneratedSentence(
                text="The method improved F1 by 9.8 points.",
                supporting_chunk_ids=["chunk-1"],
            )
        ],
        [_chunk(text="The proposed method improved F1 by 4.2 points on SciFact.")],
    )

    assert decision.abstained is True
    assert decision.sentences[0].reason == "numeric_claim_not_in_evidence"


def test_numeric_claim_passes_when_number_is_in_evidence():
    decision = GroundingVerifier().verify(
        [
            GeneratedSentence(
                text="The method improved F1 by 4.2 points.",
                supporting_chunk_ids=["chunk-1"],
            )
        ],
        [_chunk(text="The proposed method improved F1 by 4.2 points on SciFact.")],
    )

    assert decision.abstained is False


def test_one_unsupported_sentence_abstains_whole_answer():
    decision = GroundingVerifier().verify(
        [
            GeneratedSentence(
                text="The method improved F1 by 4.2 points.",
                supporting_chunk_ids=["chunk-1"],
            ),
            GeneratedSentence(
                text="The method is state of the art.",
                supporting_chunk_ids=[],
            ),
        ],
        [_chunk(text="The proposed method improved F1 by 4.2 points on SciFact.")],
    )

    assert decision.abstained is True
    assert decision.unsupported_sentence_count == 1


def test_supporting_chunk_ids_are_deduplicated():
    decision = GroundingVerifier().verify(
        [
            GeneratedSentence(
                text="The method improved F1 by 4.2 points.",
                supporting_chunk_ids=["chunk-1", "chunk-1"],
            )
        ],
        [_chunk(text="The proposed method improved F1 by 4.2 points on SciFact.")],
    )

    assert decision.sentences[0].supporting_chunk_ids == ["chunk-1"]
