# eval_qa datasets

## qasper_subset_v1

A 50-question subset of the **QASPER** dev split (40 answerable, 10
unanswerable) over 18 NLP papers — 823 paragraph chunks in the companion
corpus file. Questions keep QASPER's per-paper setting: retrieval is scoped
to the paper the question was asked about (`doc_ids`), and gold evidence
paragraphs are mapped to corpus chunk IDs. Yes/no questions are excluded
because the extractive answerer cannot emit "yes"; answerable questions must
have at least one gold evidence paragraph present in the corpus.

Rebuild deterministically (seed 13):

```bash
curl -sLO https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
tar xzf qasper-train-dev-v0.3.tgz
python scripts/build_eval_dataset.py --qasper qasper-dev-v0.3.json \
    --out-prefix datasets/eval_qa/qasper_subset_v1
```

**Attribution.** QASPER: *A Dataset of Information-Seeking Questions and
Answers Anchored in Research Papers*, Dasigi, Lo, Beltagy, Cohan, Smith,
Gardner — NAACL 2021. Distributed by the Allen Institute for AI under
**CC-BY-4.0**: <https://allenai.org/data/qasper>. The files here are a
derived subset (paragraphs re-chunked with stable IDs, answers/evidence
re-keyed) of the original dataset.
