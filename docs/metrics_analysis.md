# Reading the numbers: why the metrics are where they are

Everything below is computed from the committed artifact
(`results/qasper_subset_v1.json`) — 50 QASPER questions, 18 papers, MiniLM
bi-encoder + cross-encoder, M4 laptop. The goal of this doc is to explain
*why* each number is what it is and what would actually move it, because most
of the interesting lessons are in the gaps.

A scoping note first: this is **pure single-pass RAG, not an agentic system**.
One retrieval, one rerank, one answer attempt, one verification — no query
rewriting, no retrieve-reflect-retry loops, no tool use. That's intentional:
you can't attribute quality to pipeline stages if an agent loop is stirring
the pot. Agentic extensions (query decomposition, retry-on-abstain) would sit
*on top of* this pipeline and should be evaluated against it as a baseline.

## 1. The retrieval numbers are better than they look

Reranked P@5 is 0.200, which reads as "1 relevant chunk in 5." But answerable
questions here average only **1.93 gold paragraphs**, so even perfect
retrieval caps P@5 at 0.375. We're at **53% of the achievable ceiling**.
Lesson: check the metric ceiling implied by your gold data before judging a
number — P@k punishes you for gold that doesn't exist.

The more informative numbers: MRR 0.650 (the first gold paragraph typically
ranks 1st–2nd out of ~45 paragraphs per paper) and R@5 0.647 (two-thirds of
all gold evidence lands in the top 5).

## 2. What the cross-encoder buys — measured, not claimed

Both rankings are scored from the same requests, so this is a controlled
comparison:

| Metric | hybrid (RRF) | + cross-encoder | relative lift |
|---|---|---|---|
| P@5 | 0.165 | 0.200 | +21% |
| R@5 | 0.534 | 0.647 | +21% |
| MRR | 0.533 | 0.650 | +22% |
| nDCG@5 | 0.455 | 0.558 | +23% |

A uniform +21–23% relative lift on every metric, from a 90 MB model with no
fine-tuning. Why it works: the bi-encoder must compress question and passage
into vectors independently and hope they land close; the cross-encoder reads
the pair jointly with full attention, so it can resolve "does *this specific
sentence* answer *this specific question*." The price is latency — reranking
is ~199 ms of the ~252 ms mean request, by far the most expensive stage. In a
latency-critical system you'd rerank fewer candidates or distill the
cross-encoder; here the quality trade is clearly worth it.

## 3. Why the answer-text metrics are low (and what "better LLM" would fix)

ROUGE-1 0.232 / token F1 0.214 look weak. The dominant cause is a **format
mismatch, not a comprehension failure**: QASPER gold answers are short
human-written phrases ("BERT-base", "linear chain CRF"), while this system
returns whole evidence sentences, because extractive quoting is what makes
answers verifiable. Token-overlap metrics charge every extra word in the
sentence against precision. For calibration, trained LED baselines in the
QASPER paper score roughly 0.17–0.34 Answer-F1 depending on setup; an
untrained extractive pipeline at 0.21 sits inside that band.

**So does a better LLM improve the metrics? Split it by metric group:**

- **Retrieval metrics: no.** An LLM sits after retrieval. If the evidence
  isn't in the top-k, no generator can fix it. This is the most common
  misdiagnosis in RAG work — most "the LLM is bad" complaints are retrieval
  failures wearing a costume.
- **Answer-text metrics: yes, substantially.** A generator that synthesizes a
  short answer from the selected evidence matches the gold-answer style, so
  ROUGE/BLEU/METEOR/token-F1 should rise a lot. That's a metric-alignment win
  as much as a quality win — worth being honest about.
- **Groundedness: it gets *riskier*, not better.** Generation reintroduces
  the hallucination channel that extraction closes by definition. That's
  exactly why the verifier exists and why the `Answerer` protocol routes any
  future LLM through the same span-level checks. The number to watch after
  adding generation is the *proposed* unsupported-claim rate (currently 0 for
  the extractive answerer, trivially).

## 4. Abstention precision 0.32 — diagnosed

The system abstained 19 times; only 6 were truly unanswerable. Two findings
from the per-row data:

1. **All 13 wrong abstentions have the same reason: `weak_evidence`** — no
   reranked chunk cleared the 0.5 selection threshold. None came from the
   grounding verifier. So this is one knob, not a systemic problem.
2. The cross-encoder was trained on MS MARCO web search; on scientific prose
   its sigmoid scores skew low (domain shift), so a 0.5 cutoff — a sensible
   default for web passages — is too strict here.

Fixes, in increasing order of effort: sweep `SCIQA_MIN_RERANK_SCORE` on a
held-out split (likely lands near 0.3–0.4); Platt-calibrate the reranker
scores on a small in-domain set; fine-tune the cross-encoder on scientific
QA pairs. The trade is explicit and the harness measures it: lowering the
threshold raises answered-rate and abstention precision, and costs abstention
recall.

## 5. Yes, the dataset is small — here's how much that matters

50 questions is enough to see structure and compare configurations, and small
enough that single-number readings are shaky. 95% Wilson intervals from the
artifact:

| Quantity | Point estimate | 95% CI |
|---|---|---|
| Abstention precision (6/19) | 0.32 | [0.15, 0.54] |
| Abstention recall (6/10) | 0.60 | [0.31, 0.83] |
| Answered rate (27/40) | 0.675 | [0.52, 0.80] |

The abstention numbers are computed on 19 and 10 events respectively — the
CIs are wide enough that "0.32" should be read as "low, probably," not as a
third significant digit. The retrieval and text metrics average over 40
questions and are steadier, and the reranker-vs-fused comparison is paired
(same requests), which makes the +21–23% lift far more trustworthy than any
absolute number here.

Scaling the eval is cheap: `scripts/build_eval_dataset.py` can emit hundreds
of questions from the full QASPER dev set (281 papers) — that tightens the
error bars, i.e. it improves the *measurement*, not the system. Keeping the
committed subset small is a deliberate trade so the whole eval runs in
minutes on a laptop; anyone reproducing this should bump `--n-answerable`
before trusting fine-grained comparisons.

## Cheat sheet: which lever moves which metric

| Intervention | Moves | Doesn't move |
|---|---|---|
| Better/domain bi-encoder (bge, SPECTER2, E5) | R@5, MRR, everything downstream | answer style mismatch |
| Rerank more candidates / fine-tune reranker | P@5, nDCG, abstention quality | latency (worse) |
| Threshold sweep / score calibration | abstention precision, answered rate | retrieval quality |
| Generative answerer (vLLM et al.) | ROUGE/BLEU/METEOR/token-F1 | retrieval metrics; groundedness at risk |
| Single-sentence answers | token F1 precision | recall of multi-fact answers |
| Bigger eval set | error bars only | the system itself |
