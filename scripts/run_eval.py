#!/usr/bin/env python
"""Run the evaluation harness over a dataset + corpus.

Usage:
    python scripts/run_eval.py \
        --dataset datasets/eval_qa/qasper_subset_v1.jsonl \
        --corpus datasets/eval_qa/qasper_subset_v1_corpus.jsonl \
        --output results/qasper_subset_v1.json

Components come from the environment (SCIQA_ENCODER, SCIQA_RERANKER,
SCIQA_MIN_RERANK_SCORE); defaults are the real MiniLM bi- and cross-encoders.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.eval.app.core.harness import (  # noqa: E402
    load_corpus,
    load_examples,
    run_eval,
    write_report,
)
from services.query.app.core.factory import (  # noqa: E402
    config_from_env,
    encoder_from_env,
    reranker_from_env,
)
from services.query.app.core.pipeline import RagPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    examples = load_examples(args.dataset)
    corpus = load_corpus(args.corpus)
    encoder = encoder_from_env()
    reranker = reranker_from_env()
    config = config_from_env()

    print(f"corpus: {len(corpus)} chunks | questions: {len(examples)}")
    print(f"encoder: {encoder.name} | reranker: {reranker.name}")
    print("building index (embedding corpus)...")
    pipeline = RagPipeline(corpus, encoder, reranker=reranker, config=config)

    report = run_eval(
        pipeline,
        examples,
        benchmark=args.dataset.stem,
        k=args.k,
        config={
            "encoder": encoder.name,
            "reranker": reranker.name,
            "min_rerank_score": config.min_rerank_score,
            "retrieve_top_k": config.retrieve_top_k,
            "rerank_top_k": config.rerank_top_k,
            "n_corpus_chunks": len(corpus),
            "dataset": str(args.dataset),
        },
    )
    write_report(report, args.output)

    summary = {key: report[key] for key in ("retrieval", "answer", "ops")}
    print(json.dumps(summary, indent=2))
    print(f"\nreport written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
