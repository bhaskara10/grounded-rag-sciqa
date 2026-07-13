# Design: the grounding contract

## Problem

Naive RAG fails in a specific, dangerous way on scientific text: the answer
sounds right, cites something, and is wrong — a hallucinated number, a claim
stitched from unrelated passages, or a citation to a chunk that never said it.
Prompt instructions ("only answer from the context") do not eliminate this;
they just make the failures quieter.

## Principle

**No factual sentence crosses the service boundary without machine-checked
evidence.** The generator is untrusted. Whatever proposes answer sentences —
the extractive answerer, an LLM behind the same `Answerer` protocol — its
output is validated server-side against the evidence that was actually
retrieved for this request.

## The contract

Each proposed sentence names its `supporting_chunk_ids`. The verifier
(`services/query/app/core/grounding.py`) accepts a sentence only if all of the
following hold:

1. **Citation present** — at least one supporting chunk ID.
2. **Citation real** — every cited ID was retrieved *for this request*.
   A generator cannot smuggle in chunks from elsewhere.
3. **Numbers grounded** — every numeric token in the sentence appears in the
   cited evidence text. Hallucinated numbers are the most damaging failure
   mode in scientific QA, so they get a dedicated check that cannot be
   traded off against lexical overlap.
4. **Span support** — for each cited chunk, the verifier localizes the best
   supporting span (a window of 1–3 evidence sentences, scored by claim-token
   coverage with token-F1 tie-breaking toward the tightest window). The union
   of located spans must cover at least `min_claim_coverage` (default 0.35)
   of the sentence's content tokens. Coverage is computed over the union so
   claims legitimately supported by two chunks pass.

If any sentence fails, the whole answer abstains (`abstained: true`) and the
response carries the best-scoring passages instead, so a caller can still
show the user *something* — evidence, not assertion.

## Span-level attribution

Accepted sentences carry `supporting_spans`: chunk ID, character offsets, the
verbatim quote at those offsets, and the localization score. Offsets satisfy
`chunk.text[start_char:end_char] == span.text`, so a UI can highlight the
exact supporting characters without re-deriving anything.

## Abstention is a system decision

Abstention triggers, in pipeline order:

| Trigger | Stage |
|---|---|
| `no_retrieved_evidence` | hybrid retrieval returned nothing |
| `weak_evidence` | no reranked chunk cleared the adaptive selection threshold |
| `no_candidate_sentence` | nothing extractable overlapped the question |
| `grounding_contract_failed` | a proposed sentence failed checks 1–4 |

None of these consult a model's self-assessment. The consequence, visible in
the eval artifact: the unsupported-claim rate of *published* answers is 0 by
construction, and quality work happens on abstention precision/recall instead.

## Verifier design constraints

- **Deterministic.** Same inputs, same verdict — testable, debuggable,
  and free of a second model's failure modes. An NLI/LLM entailment check can
  be layered on later, but it should tighten this contract, not replace it.
- **Model-agnostic.** The verifier sees only `GeneratedSentence` objects; it
  does not care what produced them.
- **One tokenizer.** Retrieval, span localization, and verification share
  `core/text.py`; support scores are comparable across stages.
