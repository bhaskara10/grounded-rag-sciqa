#!/usr/bin/env python
"""Build the committed eval subset from the official QASPER distribution.

QASPER (Dasigi et al., NAACL 2021) is CC-BY-4.0:
    https://allenai.org/data/qasper
    curl -sLO https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz

Usage:
    python scripts/build_eval_dataset.py --qasper qasper-dev-v0.3.json \
        --out-prefix datasets/eval_qa/qasper_subset_v1

Selection: papers are sampled with a fixed seed; questions keep QASPER's
per-paper setting (doc-scoped retrieval). Yes/no questions are excluded
(an extractive answerer cannot emit "yes"), and answerable questions must
have at least one gold evidence paragraph present in the built corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sciqa_schema import EvidenceChunk  # noqa: E402

from services.eval.app.core.harness import EvalExample  # noqa: E402


def build_corpus(
    paper_id: str, paper: dict[str, Any]
) -> tuple[list[EvidenceChunk], dict[str, str]]:
    """One chunk per QASPER paragraph; returns chunks and text -> chunk_id map."""
    chunks: list[EvidenceChunk] = []
    text_to_id: dict[str, str] = {}

    def add(text: str, section: str) -> None:
        text = text.strip()
        if not text:
            return
        chunk = EvidenceChunk(
            chunk_id=f"{paper_id}:p:{len(chunks)}",
            doc_id=paper_id,
            text=text,
            section_path=[section] if section else [],
        )
        chunks.append(chunk)
        text_to_id.setdefault(text, chunk.chunk_id)

    add(paper["abstract"], "Abstract")
    for section in paper["full_text"]:
        name = (section["section_name"] or "").strip()
        for paragraph in section["paragraphs"]:
            add(paragraph, name)
    return chunks, text_to_id


def gold_answers(qa: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    for annotation in qa["answers"]:
        answer = annotation["answer"]
        if answer["unanswerable"] or answer["yes_no"] is not None:
            continue
        if answer["free_form_answer"].strip():
            answers.append(answer["free_form_answer"].strip())
        elif answer["extractive_spans"]:
            answers.append(" ".join(span.strip() for span in answer["extractive_spans"]))
    return list(dict.fromkeys(answers))


def gold_evidence_ids(qa: dict[str, Any], text_to_id: dict[str, str]) -> list[str]:
    evidence_ids: list[str] = []
    for annotation in qa["answers"]:
        for evidence in annotation["answer"]["evidence"]:
            evidence = evidence.strip()
            if not evidence or evidence.startswith("FLOAT SELECTED"):
                continue  # figure/table captions are not in the text corpus
            chunk_id = text_to_id.get(evidence)
            if chunk_id is not None:
                evidence_ids.append(chunk_id)
    return list(dict.fromkeys(evidence_ids))


def is_unanswerable(qa: dict[str, Any]) -> bool:
    return all(annotation["answer"]["unanswerable"] for annotation in qa["answers"])


def is_yes_no(qa: dict[str, Any]) -> bool:
    return any(
        annotation["answer"]["yes_no"] is not None for annotation in qa["answers"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasper", type=Path, required=True, help="qasper-dev-v0.3.json")
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--n-answerable", type=int, default=40)
    parser.add_argument("--n-unanswerable", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    data: dict[str, Any] = json.loads(args.qasper.read_text())
    rng = random.Random(args.seed)
    paper_ids = sorted(data)
    rng.shuffle(paper_ids)

    examples: list[EvalExample] = []
    corpus: list[EvidenceChunk] = []
    n_answerable = 0
    n_unanswerable = 0

    for paper_id in paper_ids:
        if n_answerable >= args.n_answerable and n_unanswerable >= args.n_unanswerable:
            break
        paper = data[paper_id]
        chunks, text_to_id = build_corpus(paper_id, paper)
        paper_examples: list[EvalExample] = []

        for qa in paper["qas"]:
            if is_yes_no(qa):
                continue
            if is_unanswerable(qa):
                pending = sum(e.unanswerable for e in paper_examples)
                if n_unanswerable + pending >= args.n_unanswerable:
                    continue
                paper_examples.append(
                    EvalExample(
                        question_id=qa["question_id"],
                        question=qa["question"],
                        unanswerable=True,
                        doc_ids=[paper_id],
                    )
                )
                continue

            answers = gold_answers(qa)
            evidence = gold_evidence_ids(qa, text_to_id)
            if not answers or not evidence:
                continue
            if n_answerable + sum(not e.unanswerable for e in paper_examples) >= args.n_answerable:
                continue
            paper_examples.append(
                EvalExample(
                    question_id=qa["question_id"],
                    question=qa["question"],
                    gold_answers=answers,
                    gold_evidence_ids=evidence,
                    doc_ids=[paper_id],
                )
            )

        if paper_examples:
            corpus.extend(chunks)
            examples.extend(paper_examples)
            n_answerable += sum(not e.unanswerable for e in paper_examples)
            n_unanswerable += sum(e.unanswerable for e in paper_examples)

    dataset_path = args.out_prefix.with_suffix(".jsonl")
    corpus_path = args.out_prefix.parent / f"{args.out_prefix.name}_corpus.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "".join(example.model_dump_json() + "\n" for example in examples)
    )
    corpus_path.write_text("".join(chunk.model_dump_json() + "\n" for chunk in corpus))

    papers_used = {example.doc_ids[0] for example in examples if example.doc_ids}
    print(
        f"wrote {len(examples)} questions ({n_answerable} answerable, "
        f"{n_unanswerable} unanswerable) over {len(papers_used)} papers, "
        f"{len(corpus)} corpus chunks"
    )
    print(f"dataset: {dataset_path}\ncorpus:  {corpus_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
