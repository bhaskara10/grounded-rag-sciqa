# grounded-rag-sciqa

Production-grade grounded retrieval-augmented generation for scientific paper Q&A.

## Stack

| Layer | Technology |
|-------|------------|
| Primary parse | Docling (inline) |
| Enrichment | GROBID (async/queued) |
| Search | OpenSearch — BM25 + dense + hybrid pipeline |
| Generation | vLLM structured JSON output |
| Evaluation | Ragas + custom harness |

## Services

| Service | Port | Responsibility |
|---------|------|----------------|
| ingest | 8001 | upload · parse · chunk · embed · index |
| query | 8002 | retrieve · rerank · generate · cite · abstain |
| eval | 8003 | retrieval / QA / summarization benchmarks |

## Libs

- **sciqa_schema** — canonical document model, state enums, transition validators
- **sciqa_events** — pipeline event types
- **sciqa_common** — shared utilities

## Quick start

```bash
cp .env.example .env
docker compose -f infra/compose/docker-compose.yml up -d
uvicorn services.ingest.app.main:app --port 8001 --reload
uvicorn services.query.app.main:app --port 8002 --reload
uvicorn services.eval.app.main:app --port 8003 --reload
```

## Design principles

- **Schema is the architecture.** Parser output never leaks beyond `sciqa_schema`.
- **Source-specific fields are never overwritten.** `docling_title` and `grobid_title`
  coexist; `resolved_title` is a deterministic merge via `ResolutionPolicy`.
- **Enrichment is async.** GROBID does not block ingestion throughput.
- **State is always explicit.** `enrichment_status`, `projection_status`,
  `enriched_projection_applied` — never inferred from nullable fields.
- **Abstention is a system decision**, not a prompt instruction.

## Implementation order

1. `libs/sciqa_schema` — lock models and state machine ✓
2. `libs/sciqa_events` — pipeline event types ✓
3. Ingest service — Docling parse → chunk → embed → index
4. GROBID async enrichment worker
5. Query service — hybrid retrieval → rerank → generate → cite
6. Eval harness
7. Summarization baselines
8. Observability + regression gates
