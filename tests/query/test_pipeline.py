from sciqa_schema import EvidenceChunk, GroundingVerdict

from services.query.app.core.encoders import HashingEncoder
from services.query.app.core.pipeline import ExtractiveAnswerer, PipelineConfig, RagPipeline
from services.query.app.core.rerank import LexicalReranker

CHUNKS = [
    EvidenceChunk(
        chunk_id="paper:chunk:0",
        doc_id="paper",
        text=(
            "We evaluate on the SciFact benchmark. "
            "The proposed method improved F1 by 4.2 points on SciFact."
        ),
    ),
    EvidenceChunk(
        chunk_id="paper:chunk:1",
        doc_id="paper",
        text="Training used eight A100 GPUs for twelve hours.",
    ),
    EvidenceChunk(
        chunk_id="other:chunk:0",
        doc_id="other",
        text="Cats and dogs are common household pets.",
    ),
]


def _pipeline(**config_overrides) -> RagPipeline:
    config = PipelineConfig(**{"min_rerank_score": 0.05, **config_overrides})
    return RagPipeline(
        CHUNKS,
        HashingEncoder(),
        reranker=LexicalReranker(),
        config=config,
    )


def test_answerable_question_returns_cited_answer_with_spans():
    result = _pipeline().answer("How much did the method improve F1 on SciFact?")

    assert result.abstained is False
    assert "4.2" in result.answer
    [top_sentence] = [
        sentence
        for sentence in result.sentences
        if sentence.verdict == GroundingVerdict.SUPPORTED and "4.2" in sentence.text
    ]
    assert top_sentence.supporting_spans
    assert top_sentence.supporting_spans[0].chunk_id == "paper:chunk:0"


def test_unrelated_question_abstains_on_weak_evidence():
    result = _pipeline(min_rerank_score=0.6).answer(
        "What is the boiling point of nitrogen?"
    )

    assert result.abstained is True
    assert result.abstain_reason in {"weak_evidence", "no_retrieved_evidence"}
    assert result.answer == ""


def test_doc_filter_scopes_answering():
    result = _pipeline().answer("What pets are common?", doc_ids=["paper"])

    assert all(chunk_id.startswith("paper") for chunk_id in result.retrieved_chunk_ids)


def test_result_carries_ops_telemetry():
    result = _pipeline().answer("How much did the method improve F1?")

    assert result.timings_ms["total_ms"] > 0
    assert "retrieve_ms" in result.timings_ms
    assert "rerank_ms" in result.timings_ms
    assert result.evidence_tokens > 0
    if not result.abstained:
        assert result.answer_tokens > 0


def test_trace_reports_every_stage():
    trace = _pipeline().trace("How much did the method improve F1 on SciFact?")

    assert trace.lexical and trace.dense and trace.fused and trace.reranked
    assert len(trace.reranked) <= len(trace.fused)
    assert len(trace.selected) <= len(trace.reranked)


def test_extractive_answerer_dedupes_and_caps_sentences():
    evidence = [
        EvidenceChunk(chunk_id="c1", doc_id="d", text="The method improved F1.", score=0.9),
        EvidenceChunk(chunk_id="c2", doc_id="d", text="The method improved F1.", score=0.8),
        EvidenceChunk(chunk_id="c3", doc_id="d", text="F1 improved with reranking.", score=0.7),
    ]

    proposed = ExtractiveAnswerer(max_sentences=2).propose("Did F1 improve?", evidence)

    assert len(proposed) == 2
    assert len({sentence.text for sentence in proposed}) == 2
