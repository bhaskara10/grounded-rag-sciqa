# grounded-rag-sciqa

[![CI](https://github.com/bhaskara10/grounded-rag-sciqa/actions/workflows/ci.yml/badge.svg)](https://github.com/bhaskara10/grounded-rag-sciqa/actions/workflows/ci.yml)

RAG over scientific papers that has to prove its answers. Every sentence the
system returns comes with an exact quote from the source (chunk id + character
offsets), and if the evidence isn't strong enough it abstains instead of
answering.

## Why this exists

Most RAG setups look great until you ask something the corpus doesn't actually
cover, and then they make something up. On scientific papers this is the worst
possible failure mode because a hallucinated "improved F1 by 4.2 points" looks
exactly like a real one. Prompting the model to "only use the context" reduces
this but doesn't eliminate it, and you can't tell from the output which answers
to trust.

So the experiment here: treat whatever generates the answer as untrusted, and
enforce grounding as a server-side check. An answer only leaves the service if
every sentence passes verification against the retrieved evidence. Otherwise
the caller gets an explicit abstention plus the closest passages, which is more
useful than a confident guess.

## How it works

```
ingestion:  PDF -> PyMuPDF layout parse -> section-aware chunks -> embeddings -> local index

query:      question -> BM25 + dense retrieval -> reciprocal-rank fusion
                     -> cross-encoder rerank
                     -> adaptive evidence selection (score threshold + token budget)
                     -> extractive answer
                     -> grounding verifier -> answer with spans, or abstention
```

A few notes on the choices:

- **Hybrid retrieval.** BM25 (Okapi, implemented in-repo) and dense retrieval
  fail differently: BM25 misses paraphrases, embeddings miss rare exact terms
  like dataset names and numbers. Their rankings are merged with RRF since the
  score scales aren't comparable anyway.
- **Reranking.** A cross-encoder reads each (question, chunk) pair jointly and
  is noticeably better at precision than either retriever. Logits go through a
  sigmoid so the selection stage can threshold on something probability-like.
- **Adaptive selection instead of fixed top-k.** Keep reranked chunks above a
  relevance threshold until a token budget runs out. If nothing clears the
  threshold, that *is* the abstention path — no separate "should I answer?"
  heuristic.
- **Verification.** For each answer sentence the verifier checks that the cited
  chunks were actually retrieved for this request, that every number in the
  sentence appears in the cited evidence (numeric hallucinations get their own
  check on purpose), and it localizes the 1–3 evidence sentences that support
  the claim. Those spans are returned verbatim with character offsets, so a UI
  can highlight exactly what supports what. One failed sentence abstains the
  whole answer.
- **PDF ingestion.** PyMuPDF, reading order preserved, headings detected from
  font sizes, ruled tables pulled out as markdown so they survive retrieval as
  structured chunks. Ingestion is SHA-256 idempotent.

Longer write-up: [docs/design_grounding_contract.md](docs/design_grounding_contract.md).

## Which models — and where's the LLM?

There is no generative LLM in the loop right now, and that's deliberate. The
answerer is extractive: it returns the evidence sentences that best match the
question, which makes every answer attributable by definition and keeps the
whole thing runnable offline on a laptop. The two models used are small:

| Role | Model | Size |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB |

Both run on MPS (Apple silicon), CUDA, or CPU — device is auto-selected. The
`Answerer` protocol in `services/query/app/core/pipeline.py` is the seam where
a generative model plugs in later; the verifier doesn't care what produced the
sentences, so the grounding contract stays the same.

## How good is it, actually

Evaluated on 50 questions from the QASPER dev set (40 answerable, 10
unanswerable, 18 papers). The artifact is committed at
[results/qasper_subset_v1.json](results/qasper_subset_v1.json) and one command
regenerates it (below). Run on an M4 MacBook Pro.

Retrieval, against gold evidence paragraphs:

| Ranking | P@5 | R@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| hybrid (BM25 + dense, RRF) | 0.165 | 0.534 | 0.533 | 0.455 |
| after cross-encoder rerank | 0.200 | 0.647 | 0.650 | 0.558 |

The harness scores both rankings from the same request specifically so the
reranker's contribution is measurable, and it does help across the board.

Answer quality (only answered questions, vs QASPER's mostly abstractive gold
answers): ROUGE-1 0.232, ROUGE-L 0.198, BLEU 0.088, METEOR 0.280, token F1
0.214. These are modest, which is what you'd expect from an extractive
answerer scored against human-written answers — a generative answerer should
lift these considerably.

Grounding and abstention: published answers have a 0% unsupported-claim rate,
but that's by construction (unsupported answers don't publish), so the honest
numbers to watch are the abstention ones. The system answered 67.5% of
answerable questions and caught 6 of the 10 unanswerable ones. Abstention
precision is 0.32 — it's too trigger-happy on hard-but-answerable questions.
That's the weakest number in the artifact and the clearest tuning target
(`SCIQA_MIN_RERANK_SCORE` trades answer rate against abstention recall).

Latency: p50 126 ms, p95 478 ms end to end; reranking dominates (~199 ms mean).

Methodology details, including how the dataset was built:
[docs/evaluation_methodology.md](docs/evaluation_methodology.md).

## Project status

Works end to end locally: ingest real PDFs, ask questions over the persistent
index via CLI or the FastAPI services, run the eval. 130 tests (hermetic — CI
uses deterministic stand-ins for both models, so no downloads), lint and type
checks in CI.

Not there yet: no generative answerer, single-machine index only (the
OpenSearch compose scaffolding in `infra/` exists but isn't wired), and the
verifier is purely lexical.

## What I'd do next

1. **Generation behind the same contract.** vLLM with structured JSON output
   implementing the `Answerer` protocol, so generated sentences carry claimed
   citations and go through the identical verifier. This is the piece the
   architecture was shaped around.
2. **NLI on top of the lexical verifier.** Token overlap misses legitimate
   paraphrase support and can be fooled by coincidental overlap; a small
   entailment model would tighten both sides.
3. **Abstention calibration.** The 0.32 precision above; probably per-question
   thresholds or a calibration set rather than one global number.
4. **OpenSearch backend** behind the existing index interface, for corpora
   that don't fit one machine.
5. **Richer parsing** (Docling/GROBID) for references and cleaner section
   structure.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
make install-dev
make test
```

Ingest a paper and ask something (first run downloads the two models):

```bash
python scripts/ingest_pdf.py your_paper.pdf
python scripts/ask.py "How much did the method improve F1?"
```

```
ANSWER: The proposed method improved F1 by 4.2 points on SciFact.

  supported by your-paper-3f2a91bc:chunk:12 [58:116]
    "The proposed method improved F1 by 4.2 points on SciFact."
```

Services, if you want the HTTP API:

```bash
uvicorn services.ingest.app.main:app --port 8001   # POST /documents (PDF upload)
uvicorn services.query.app.main:app  --port 8002   # POST /qa /retrieve /explain
uvicorn services.eval.app.main:app   --port 8003   # POST /runs
```

Reproduce the eval (or `make eval`):

```bash
python scripts/run_eval.py \
  --dataset datasets/eval_qa/qasper_subset_v1.jsonl \
  --corpus  datasets/eval_qa/qasper_subset_v1_corpus.jsonl \
  --output  results/qasper_subset_v1.json
```

There's also a walkthrough notebook at
[notebooks/demo_grounded_qa.ipynb](notebooks/demo_grounded_qa.ipynb).

## Layout

```
libs/schema/      pydantic contracts (documents, chunks, grounding, spans)
services/ingest/  PDF upload -> parse -> chunk -> embed -> index
services/query/   /qa /retrieve /explain + the retrieval/rerank/grounding core
services/eval/    metrics + harness + /runs
datasets/eval_qa/ QASPER subset (CC-BY-4.0, attribution in its README) + build script
results/          committed eval artifacts
scripts/          ingest_pdf.py, ask.py, run_eval.py, build_eval_dataset.py
tests/            schema, retrieval, grounding, ingest, eval, routes
```

Config is all environment variables: `SCIQA_INDEX_DIR`, `SCIQA_ENCODER`,
`SCIQA_RERANKER`, `SCIQA_MIN_RERANK_SCORE`. Setting
`SCIQA_ENCODER=hashing SCIQA_RERANKER=lexical` swaps in the deterministic
components, which is what the tests use and doubles as a weak baseline.

## License

Apache 2.0 ([LICENSE](LICENSE)). QASPER data is CC-BY-4.0 from AllenAI, see
[datasets/eval_qa/README.md](datasets/eval_qa/README.md).
