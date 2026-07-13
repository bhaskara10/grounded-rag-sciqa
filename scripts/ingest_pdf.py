#!/usr/bin/env python
"""Ingest local PDFs into the persistent index.

Usage:
    python scripts/ingest_pdf.py paper1.pdf paper2.pdf [--index-dir index]

Set SCIQA_ENCODER=hashing for a fast, model-free smoke run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.ingest.app.core.ingest import ingest_pdf_bytes  # noqa: E402
from services.query.app.core.factory import encoder_from_env  # noqa: E402
from services.query.app.core.index_store import LocalIndexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF files to ingest")
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    args = parser.parse_args()

    store = LocalIndexStore(args.index_dir)
    encoder = encoder_from_env()
    for pdf_path in args.pdfs:
        result = ingest_pdf_bytes(
            pdf_path.read_bytes(),
            store=store,
            encoder=encoder,
            filename=pdf_path.name,
        )
        state = "already indexed" if result.deduplicated else "indexed"
        print(f"{pdf_path.name}: {state} as {result.doc_id} ({result.chunk_count} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
