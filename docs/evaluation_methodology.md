# Evaluation Methodology

## Goal

Evaluate the system on the failure modes that matter for scientific Q&A:

- Did retrieval surface the right evidence?
- Did the answer cite every factual sentence?
- Did the system abstain when evidence was missing or weak?
- Did latency and cost remain inside an operational budget?

## Datasets

The first benchmark should be a small curated JSONL set before scaling up:

```json
{
  "question_id": "scifact-mini-001",
  "doc_ids": ["doc_123"],
  "question": "What F1 improvement did the proposed method report?",
  "gold_chunk_ids": ["chunk_12"],
  "answer_facts": ["The method improved F1 by 4.2 points."],
  "requires_abstention": false
}
```

Recommended starting size:

- 30 answerable questions with known supporting chunks.
- 10 unanswerable questions where abstention is expected.
- 10 adversarial questions with tempting but unsupported numeric claims.

## Metrics

Retrieval:

- Precision@5
- Recall@5
- nDCG@5
- MRR

Answer grounding:

- Citation coverage: share of factual sentences with at least one citation.
- Unsupported-claim rate: share of factual sentences rejected by the verifier.
- Numeric grounding failure rate: share of numeric claims absent from evidence.
- Abstention precision: share of abstentions that were actually required.
- Answer rate: share of requests answered rather than abstained.

Operations:

- p50/p95 latency
- average retrieved chunks
- average generated tokens
- parse/index failure rate

## Regression Gate

A change should pass only if it does not degrade grounding quality:

- Citation coverage stays above the configured threshold.
- Unsupported-claim rate does not increase.
- Required-abstention cases still abstain.
- Retrieval recall on gold chunks does not regress.

The first committed result is intentionally small:

- Dataset: `datasets/eval_qa/local_baseline_v0.jsonl`
- Result: `results/local_baseline_v0.json`
- Purpose: prove the local retrieve -> cite -> verify -> abstain loop is
  measurable before scaling to SciFact/QASPER.

The next published portfolio result should compare baseline RAG versus grounded
RAG on the same curated scientific QA set.
