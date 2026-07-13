#!/usr/bin/env python
"""Ask a grounded question against the persistent index.

Usage:
    python scripts/ask.py "How much did the method improve F1?" [--index-dir index]

Prints the answer with its verbatim supporting spans, or the abstention reason.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.query.app.core.factory import (  # noqa: E402
    config_from_env,
    encoder_from_env,
    load_indexed_pipeline,
    reranker_from_env,
)
from services.query.app.core.index_store import LocalIndexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    args = parser.parse_args()

    store = LocalIndexStore(args.index_dir)
    pipeline = load_indexed_pipeline(
        store, encoder_from_env(), reranker_from_env(), config_from_env()
    )
    if pipeline is None:
        print(f"no index at {args.index_dir} — run scripts/ingest_pdf.py first")
        return 1

    result = pipeline.answer(args.question)
    if result.abstained:
        print(f"ABSTAINED ({result.abstain_reason})")
        for passage in result.best_supporting_passages[:2]:
            print(f"  closest evidence: {passage[:160]}")
        return 0

    print(f"ANSWER: {result.answer}\n")
    for sentence in result.sentences:
        for span in sentence.supporting_spans:
            print(f'  supported by {span.chunk_id} [{span.start_char}:{span.end_char}]')
            print(f'    "{span.text}"')
    print(f"\nlatency: {result.timings_ms.get('total_ms', 0):.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
