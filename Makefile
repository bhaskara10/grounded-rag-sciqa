.PHONY: install-dev test lint typecheck check

install-dev:
	pip install -e libs/schema -e libs/events -e libs/common
	pip install -e services/query -e services/eval
	pip install pytest pytest-asyncio pydantic ruff mypy

test:
	PYTHONPATH=libs/schema:libs/events:libs/common:. pytest tests/ -v

lint:
	ruff check libs/ services/ tests/

typecheck:
	mypy libs/schema/sciqa_schema/ --ignore-missing-imports

check: lint typecheck test
