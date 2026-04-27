# Grounding Contract Design

## Problem

Scientific RAG systems fail dangerously when answer text and citations are only
loosely coupled. A model can cite a retrieved passage while adding unsupported
numbers, causal claims, or broad "state of the art" language that the passage
does not contain.

This project treats grounding as a service contract:

1. The generator emits sentence-level citations.
2. The verifier checks citations against the retrieved evidence set.
3. The service abstains when any factual sentence fails verification.

## Contract

Each answer sentence is represented as:

```json
{
  "text": "The method improved F1 by 4.2 points.",
  "supporting_chunk_ids": ["chunk_12"],
  "confidence": 0.91
}
```

The verifier returns:

```json
{
  "text": "The method improved F1 by 4.2 points.",
  "supporting_chunk_ids": ["chunk_12"],
  "verdict": "supported",
  "support_score": 0.83
}
```

If any sentence is `unsupported`, the answer response is marked
`abstained: true`.

## Implemented Checks

The current deterministic verifier enforces four rules:

1. A sentence must include at least one `supporting_chunk_id`.
2. Every supporting chunk ID must be present in the request's retrieved chunks.
3. Sentence content must overlap the cited evidence above a policy threshold.
4. Numeric claims in the sentence must appear in the cited evidence.

These checks are intentionally conservative. They are not a full entailment
model; they are a hard guardrail that catches common RAG failures before a more
expensive verifier is added.

## Why This Is Production-Shaped

The contract is independent of the generator. OpenAI, vLLM, a local model, or an
extractive baseline can all produce candidate sentences, but none of them can
publish an answer without passing the same verifier.

This separation gives the system:

- A stable API boundary for answer generation.
- Auditable sentence-level decisions.
- Deterministic regression tests.
- A clean path to add stronger entailment/NLI verification later.

## Next Steps

1. Wire the verifier into `/qa`.
2. Add local file ingestion for deterministic demos.
3. Add an LLM generator behind the same contract.
4. Track unsupported-claim rate, citation coverage, and abstention precision in
   the eval harness.
