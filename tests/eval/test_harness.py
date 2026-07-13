import json

from sciqa_schema import EvidenceChunk

from services.eval.app.core.harness import (
    EvalExample,
    load_corpus,
    load_examples,
    run_eval,
    write_report,
)
from services.query.app.core.encoders import HashingEncoder
from services.query.app.core.pipeline import PipelineConfig, RagPipeline
from services.query.app.core.rerank import LexicalReranker

CORPUS = [
    EvidenceChunk(
        chunk_id="paper1:chunk:0",
        doc_id="paper1",
        text="The proposed method improved F1 by 4.2 points on SciFact.",
    ),
    EvidenceChunk(
        chunk_id="paper1:chunk:1",
        doc_id="paper1",
        text="Training used eight A100 GPUs for twelve hours.",
    ),
    EvidenceChunk(
        chunk_id="paper2:chunk:0",
        doc_id="paper2",
        text="Cats and dogs are common household pets.",
    ),
]

EXAMPLES = [
    EvalExample(
        question_id="q1",
        question="How much did the method improve F1 on SciFact?",
        gold_answers=["The proposed method improved F1 by 4.2 points on SciFact."],
        gold_evidence_ids=["paper1:chunk:0"],
    ),
    EvalExample(
        question_id="q2",
        question="What is the airspeed velocity of an unladen swallow?",
        unanswerable=True,
    ),
]


def _pipeline() -> RagPipeline:
    return RagPipeline(
        CORPUS,
        HashingEncoder(),
        reranker=LexicalReranker(),
        config=PipelineConfig(min_rerank_score=0.05),
    )


def test_report_covers_all_three_layers():
    report = run_eval(_pipeline(), EXAMPLES, benchmark="unit", k=5)

    assert report["retrieval"]["n_scored"] == 1
    assert report["retrieval"]["reranked"]["precision_at_5"] > 0
    assert report["retrieval"]["fused"]["mrr"] > 0
    assert report["answer"]["n_text_scored"] == 1
    assert report["answer"]["rouge1"] > 0.5
    assert report["answer"]["token_f1"] > 0.5
    assert report["ops"]["latency_ms"]["p95"] > 0
    assert report["ops"]["tokens"]["evidence_mean"] > 0


def test_unanswerable_question_scores_abstention_quality():
    report = run_eval(_pipeline(), EXAMPLES, benchmark="unit", k=5)

    assert report["answer"]["abstention_recall"] == 1.0
    assert report["answer"]["abstention_precision"] == 1.0
    assert report["answer"]["answered_rate_on_answerable"] == 1.0


def test_groundedness_is_reported_for_proposed_sentences():
    report = run_eval(_pipeline(), EXAMPLES, benchmark="unit", k=5)

    assert 0.0 <= report["answer"]["groundedness_proposed"] <= 1.0
    assert report["answer"]["unsupported_claim_rate_published"] == 0.0


def test_report_roundtrips_through_files(tmp_path):
    report = run_eval(_pipeline(), EXAMPLES, benchmark="unit", k=5)
    output = tmp_path / "results" / "unit.json"

    write_report(report, output)

    assert json.loads(output.read_text())["benchmark"] == "unit"


def test_dataset_and_corpus_loaders(tmp_path):
    dataset_path = tmp_path / "data.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    dataset_path.write_text(EXAMPLES[0].model_dump_json() + "\n\n")
    corpus_path.write_text(CORPUS[0].model_dump_json() + "\n")

    assert load_examples(dataset_path)[0].question_id == "q1"
    assert load_corpus(corpus_path)[0].chunk_id == "paper1:chunk:0"
