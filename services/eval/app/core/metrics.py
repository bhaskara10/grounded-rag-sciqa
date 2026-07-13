"""Metric implementations for the three evaluation layers.

Retrieval metrics operate on a ranked list of chunk IDs against a gold
relevance set. Text metrics compare a candidate answer against one or more
references, taking the best reference (standard multi-reference practice).
Ops metrics are percentile helpers over per-request measurements.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


# --------------------------------------------------------------------------
# retrieval layer


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranked_ids[:k]
    return sum(chunk_id in relevant_ids for chunk_id in top) / k


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return 0.0
    top = ranked_ids[:k]
    return sum(chunk_id in relevant_ids for chunk_id in top) / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1 / position
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1 / math.log2(position + 1)
        for position, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / ideal_dcg


# --------------------------------------------------------------------------
# answer text layer


def token_f1(candidate: str, references: Sequence[str]) -> float:
    """SQuAD-style bag-of-tokens F1, best reference wins."""
    candidate_tokens = _tokens(candidate)
    best = 0.0
    for reference in references:
        reference_tokens = _tokens(reference)
        if not candidate_tokens or not reference_tokens:
            score = float(candidate_tokens == reference_tokens)
        else:
            overlap = _multiset_overlap(candidate_tokens, reference_tokens)
            if overlap == 0:
                score = 0.0
            else:
                precision = overlap / len(candidate_tokens)
                recall = overlap / len(reference_tokens)
                score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def rouge_scores(candidate: str, references: Sequence[str]) -> dict[str, float]:
    """Best-reference ROUGE-1/2/L F-measures."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for reference in references:
        scored = scorer.score(reference, candidate)
        for key in best:
            best[key] = max(best[key], scored[key].fmeasure)
    return best


def bleu_score(candidate: str, references: Sequence[str]) -> float:
    """Smoothed sentence BLEU against all references."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    candidate_tokens = _tokens(candidate)
    reference_tokens = [_tokens(reference) for reference in references if reference.strip()]
    if not candidate_tokens or not reference_tokens:
        return 0.0
    return float(
        sentence_bleu(
            reference_tokens,
            candidate_tokens,
            smoothing_function=SmoothingFunction().method3,
        )
    )


def meteor(candidate: str, references: Sequence[str]) -> float:
    """Best-reference METEOR; requires the NLTK wordnet corpus."""
    from nltk.translate.meteor_score import meteor_score

    ensure_nltk_data()
    candidate_tokens = _tokens(candidate)
    if not candidate_tokens:
        return 0.0
    scores = [
        float(meteor_score([_tokens(reference)], candidate_tokens))
        for reference in references
        if reference.strip()
    ]
    return max(scores, default=0.0)


def ensure_nltk_data() -> None:
    """Fetch the wordnet corpus once; METEOR needs it."""
    import nltk

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


# --------------------------------------------------------------------------
# ops layer


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile; fraction in [0, 1]."""
    if not values:
        return 0.0
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


# --------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _multiset_overlap(left: list[str], right: list[str]) -> int:
    from collections import Counter

    left_counts = Counter(left)
    right_counts = Counter(right)
    return sum((left_counts & right_counts).values())
