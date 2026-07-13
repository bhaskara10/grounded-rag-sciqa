# Evaluation methodology

Three layers, tracked separately, produced by one harness
(`services/eval/app/core/harness.py`) that runs the *real* pipeline — the
same code path the query service serves — over a JSONL dataset + corpus.
Every artifact records the encoder, reranker, thresholds, and corpus size,
so any number in the README can be regenerated with one command.

## Dataset

`datasets/eval_qa/qasper_subset_v1.jsonl`: a 50-question subset of the QASPER
dev split (CC-BY-4.0, attribution in `datasets/eval_qa/README.md`) — 40
answerable and 10 unanswerable questions over 18 NLP papers (823 paragraph
chunks). QASPER's per-paper setting is preserved: retrieval is scoped to the
paper the question is about via `doc_ids`. Gold evidence paragraphs are
mapped to corpus chunk IDs at build time; yes/no questions are excluded
because the extractive answerer cannot emit "yes".

## Layer 1 — retrieval

Precision@5, Recall@5, MRR, nDCG@5 against gold evidence chunks, reported for
**two rankings from the same request**:

- `fused` — the hybrid BM25 + dense ranking after reciprocal-rank fusion;
- `reranked` — after the cross-encoder.

Reporting both makes the reranker's contribution a measured fact rather than
a claim. Only answerable questions with mapped gold evidence are scored.

## Layer 2 — answer

- **Text quality** (answered, answerable questions only): ROUGE-1/2/L, BLEU
  (smoothed), METEOR, and SQuAD-style token F1 against gold answers, best
  reference wins. The answerer is extractive, so these are lower bounds
  relative to abstractive gold answers — reported anyway, not cherry-picked.
- **Grounding**: `groundedness_proposed` (fraction of proposed sentences that
  pass the verifier) and `unsupported_claim_rate_published`, which is 0 by
  construction — the interesting number is how often the gate fires.
- **Abstention quality**: abstention rate, precision (abstentions that were
  truly unanswerable), and recall (unanswerable questions caught). This is
  the calibration story for the selection threshold: raising
  `SCIQA_MIN_RERANK_SCORE` trades answered-rate for abstention recall.

## Layer 3 — ops

p50/p95/p99 end-to-end latency, mean per-stage latency (retrieve / rerank /
verify), and token usage (evidence tokens selected, answer tokens produced).
Measured on the machine that ran the artifact; the committed numbers are from
an Apple M4 using the MPS backend.

## Running it

```bash
python scripts/run_eval.py \
    --dataset datasets/eval_qa/qasper_subset_v1.jsonl \
    --corpus  datasets/eval_qa/qasper_subset_v1_corpus.jsonl \
    --output  results/qasper_subset_v1.json
```

`SCIQA_ENCODER=hashing SCIQA_RERANKER=lexical` runs the same harness with the
deterministic components (no model downloads) — useful as a smoke test and as
a weak baseline for comparison.
