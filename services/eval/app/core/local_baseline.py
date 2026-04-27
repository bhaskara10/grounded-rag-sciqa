"""Evaluate the deterministic local QA baseline on JSONL datasets."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel
from sciqa_schema import EvidenceChunk, GroundingVerdict
from services.query.app.core.local_rag import answer_question_local


class LocalEvalExample(BaseModel):
    question_id: str
    doc_id: str
    question: str
    passages: list[str]
    expected_answer: str = ""
    requires_abstention: bool = False


def load_examples(path: Path) -> list[LocalEvalExample]:
    """Load newline-delimited local QA examples."""
    examples: list[LocalEvalExample] = []
    for line in path.read_text().splitlines():
        if line.strip():
            examples.append(LocalEvalExample.model_validate_json(line))
    return examples


def evaluate_examples(examples: list[LocalEvalExample]) -> dict[str, Any]:
    """Run LocalRAG and return portfolio-facing summary metrics."""
    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    sentence_count = 0
    supported_sentence_count = 0
    unsupported_sentence_count = 0
    answerable_count = sum(not example.requires_abstention for example in examples)
    abstention_expected_count = sum(example.requires_abstention for example in examples)
    correct_answer_count = 0
    correct_abstention_count = 0
    predicted_abstention_count = 0

    for example in examples:
        chunks = [
            EvidenceChunk(
                chunk_id=f"{example.doc_id}:passage:{index}",
                doc_id=example.doc_id,
                text=passage,
            )
            for index, passage in enumerate(example.passages)
        ]

        started_at = perf_counter()
        result = answer_question_local(example.question, chunks)
        latency_ms = (perf_counter() - started_at) * 1000
        latencies_ms.append(latency_ms)

        predicted_abstention_count += int(result.abstained)
        if example.requires_abstention and result.abstained:
            correct_abstention_count += 1
        if not example.requires_abstention and result.answer == example.expected_answer:
            correct_answer_count += 1

        sentence_count += len(result.sentences)
        supported_sentence_count += sum(
            sentence.verdict == GroundingVerdict.SUPPORTED
            for sentence in result.sentences
        )
        unsupported_sentence_count += sum(
            sentence.verdict == GroundingVerdict.UNSUPPORTED
            for sentence in result.sentences
        )

        rows.append(
            {
                "question_id": example.question_id,
                "requires_abstention": example.requires_abstention,
                "abstained": result.abstained,
                "answer": result.answer,
                "expected_answer": example.expected_answer,
                "retrieved_chunk_ids": result.retrieved_chunk_ids,
                "latency_ms": round(latency_ms, 3),
            }
        )

    return {
        "benchmark": "local_baseline_v0",
        "n_questions": len(examples),
        "answerable_questions": answerable_count,
        "abstention_expected_questions": abstention_expected_count,
        "answered_questions": len(examples) - predicted_abstention_count,
        "abstention_rate": _safe_div(predicted_abstention_count, len(examples)),
        "exact_answer_match": _safe_div(correct_answer_count, answerable_count),
        "abstention_precision": _safe_div(
            correct_abstention_count,
            predicted_abstention_count,
        ),
        "abstention_recall": _safe_div(
            correct_abstention_count,
            abstention_expected_count,
        ),
        "citation_coverage": _safe_div(supported_sentence_count, sentence_count),
        "unsupported_claim_rate": _safe_div(unsupported_sentence_count, sentence_count),
        "median_latency_ms": round(statistics.median(latencies_ms), 3)
        if latencies_ms
        else 0.0,
        "rows": rows,
    }


def write_result(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    """Evaluate a dataset and write a stable JSON result artifact."""
    result = evaluate_examples(load_examples(dataset_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
