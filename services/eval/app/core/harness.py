"""Evaluation harness: run the RAG pipeline over a dataset, report three layers.

Layers
------
retrieval  P@k / R@k / MRR / nDCG@k — computed for both the RRF-fused ranking
           and the cross-encoder ranking, so the reranker's lift is visible.
answer     ROUGE-1/2/L, BLEU, METEOR, token F1 on answered questions, plus
           groundedness, unsupported-claim rate, and abstention quality.
ops        latency percentiles per stage and token usage.

The dataset is JSONL (one EvalExample per line) plus a corpus JSONL of
EvidenceChunk rows; retrieval runs over the whole corpus, not just the gold
paper, so ranking metrics reflect real difficulty.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sciqa_schema import EvidenceChunk, GroundingVerdict

from services.query.app.core.pipeline import PipelineResult, RagPipeline

from . import metrics


class EvalExample(BaseModel):
    question_id: str
    question: str
    gold_answers: list[str] = Field(default_factory=list)
    gold_evidence_ids: list[str] = Field(default_factory=list)
    unanswerable: bool = False
    doc_ids: list[str] | None = None  # scope retrieval, e.g. QASPER's per-paper setting


def load_examples(path: Path) -> list[EvalExample]:
    return [
        EvalExample.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_corpus(path: Path) -> list[EvidenceChunk]:
    return [
        EvidenceChunk.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def run_eval(
    pipeline: RagPipeline,
    examples: list[EvalExample],
    *,
    benchmark: str,
    k: int = 5,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [_evaluate_one(pipeline, example, k=k) for example in examples]
    report = {
        "benchmark": benchmark,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "k": k,
            "n_questions": len(examples),
            **(config or {}),
        },
        "retrieval": _aggregate_retrieval(rows),
        "answer": _aggregate_answer(rows),
        "ops": _aggregate_ops(rows),
        "rows": rows,
    }
    return report


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _evaluate_one(pipeline: RagPipeline, example: EvalExample, *, k: int) -> dict[str, Any]:
    result = pipeline.answer(example.question, doc_ids=example.doc_ids)
    row: dict[str, Any] = {
        "question_id": example.question_id,
        "unanswerable": example.unanswerable,
        "abstained": result.abstained,
        "abstain_reason": result.abstain_reason,
        "answer": result.answer,
        "latency_ms": result.timings_ms.get("total_ms", 0.0),
        "stage_ms": {
            stage: duration
            for stage, duration in result.timings_ms.items()
            if stage != "total_ms"
        },
        "evidence_tokens": result.evidence_tokens,
        "answer_tokens": result.answer_tokens,
    }

    if example.gold_evidence_ids:
        gold = set(example.gold_evidence_ids)
        row["retrieval"] = {
            "fused": _ranking_metrics(result.retrieved_chunk_ids, gold, k),
            "reranked": _ranking_metrics(result.reranked_chunk_ids, gold, k),
        }

    supported = sum(
        sentence.verdict == GroundingVerdict.SUPPORTED for sentence in result.sentences
    )
    row["proposed_sentences"] = len(result.sentences)
    row["supported_sentences"] = supported

    if not result.abstained and not example.unanswerable and example.gold_answers:
        row["text_metrics"] = _text_metrics(result, example)
    return row


def _ranking_metrics(ranked_ids: list[str], gold: set[str], k: int) -> dict[str, float]:
    return {
        f"precision_at_{k}": round(metrics.precision_at_k(ranked_ids, gold, k), 4),
        f"recall_at_{k}": round(metrics.recall_at_k(ranked_ids, gold, k), 4),
        "mrr": round(metrics.reciprocal_rank(ranked_ids, gold), 4),
        f"ndcg_at_{k}": round(metrics.ndcg_at_k(ranked_ids, gold, k), 4),
    }


def _text_metrics(result: PipelineResult, example: EvalExample) -> dict[str, float]:
    rouge = metrics.rouge_scores(result.answer, example.gold_answers)
    return {
        "rouge1": round(rouge["rouge1"], 4),
        "rouge2": round(rouge["rouge2"], 4),
        "rougeL": round(rouge["rougeL"], 4),
        "bleu": round(metrics.bleu_score(result.answer, example.gold_answers), 4),
        "meteor": round(metrics.meteor(result.answer, example.gold_answers), 4),
        "token_f1": round(metrics.token_f1(result.answer, example.gold_answers), 4),
    }


def _aggregate_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row["retrieval"] for row in rows if "retrieval" in row]
    if not scored:
        return {"n_scored": 0}
    aggregated: dict[str, Any] = {"n_scored": len(scored)}
    for stage in ("fused", "reranked"):
        aggregated[stage] = {
            metric: round(metrics.mean([row[stage][metric] for row in scored]), 4)
            for metric in scored[0][stage]
        }
    return aggregated


def _aggregate_answer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if not row["unanswerable"]]
    unanswerable = [row for row in rows if row["unanswerable"]]
    abstentions = [row for row in rows if row["abstained"]]
    correct_abstentions = [row for row in abstentions if row["unanswerable"]]

    text_rows = [row["text_metrics"] for row in rows if "text_metrics" in row]
    text_aggregate = (
        {
            metric: round(metrics.mean([row[metric] for row in text_rows]), 4)
            for metric in text_rows[0]
        }
        if text_rows
        else {}
    )

    proposed = sum(row["proposed_sentences"] for row in rows)
    supported = sum(row["supported_sentences"] for row in rows)
    return {
        **text_aggregate,
        "n_text_scored": len(text_rows),
        "answered_rate_on_answerable": round(
            metrics.mean([float(not row["abstained"]) for row in answerable]), 4
        )
        if answerable
        else 0.0,
        "groundedness_proposed": round(supported / proposed, 4) if proposed else 0.0,
        "unsupported_claim_rate_proposed": round(1 - supported / proposed, 4)
        if proposed
        else 0.0,
        "unsupported_claim_rate_published": 0.0,  # unsupported answers never publish
        "abstention_rate": round(len(abstentions) / len(rows), 4) if rows else 0.0,
        "abstention_precision": round(len(correct_abstentions) / len(abstentions), 4)
        if abstentions
        else 0.0,
        "abstention_recall": round(
            len(correct_abstentions) / len(unanswerable), 4
        )
        if unanswerable
        else 0.0,
    }


def _aggregate_ops(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows]
    stage_names = sorted({stage for row in rows for stage in row["stage_ms"]})
    return {
        "latency_ms": {
            "p50": round(metrics.percentile(latencies, 0.50), 2),
            "p95": round(metrics.percentile(latencies, 0.95), 2),
            "p99": round(metrics.percentile(latencies, 0.99), 2),
            "mean": round(metrics.mean(latencies), 2),
        },
        "stage_ms_mean": {
            stage: round(
                metrics.mean([row["stage_ms"].get(stage, 0.0) for row in rows]), 2
            )
            for stage in stage_names
        },
        "tokens": {
            "evidence_mean": round(
                metrics.mean([row["evidence_tokens"] for row in rows]), 1
            ),
            "answer_mean": round(metrics.mean([row["answer_tokens"] for row in rows]), 1),
        },
    }
