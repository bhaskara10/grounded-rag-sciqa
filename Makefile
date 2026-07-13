.PHONY: install-dev test lint typecheck check eval eval-smoke

install-dev:
	pip install -e libs/schema -e libs/events -e libs/common
	pip install -e services/query -e services/eval -e services/ingest
	pip install pytest pytest-asyncio httpx ruff mypy

test:
	PYTHONPATH=libs/schema:libs/events:libs/common:. pytest tests/ -v

lint:
	ruff check libs/ services/ tests/

typecheck:
	mypy libs/schema/sciqa_schema/ --ignore-missing-imports

check: lint typecheck test

# full eval with real models (downloads two ~90MB models on first run)
eval:
	PYTHONPATH=. python scripts/run_eval.py \
		--dataset datasets/eval_qa/qasper_subset_v1.jsonl \
		--corpus datasets/eval_qa/qasper_subset_v1_corpus.jsonl \
		--output results/qasper_subset_v1.json

# same harness with deterministic components — fast, no downloads
eval-smoke:
	SCIQA_ENCODER=hashing SCIQA_RERANKER=lexical SCIQA_MIN_RERANK_SCORE=0.05 \
	PYTHONPATH=. python scripts/run_eval.py \
		--dataset datasets/eval_qa/qasper_subset_v1.jsonl \
		--corpus datasets/eval_qa/qasper_subset_v1_corpus.jsonl \
		--output results/smoke_baseline.json
