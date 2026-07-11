"""Span localization: find the evidence characters that support a claim.

Given a claim sentence and a chunk of evidence text, locate the contiguous
window of evidence sentences that best supports the claim, and return its
exact character offsets. This is what turns chunk-level citations into
span-level attribution: the API can quote the supporting passage verbatim
instead of pointing at a whole chunk.
"""
from __future__ import annotations

from dataclasses import dataclass

from .text import content_tokens, sentences_with_offsets


@dataclass(frozen=True)
class LocatedSpan:
    """A candidate supporting span inside one evidence chunk."""

    start_char: int
    end_char: int
    text: str
    claim_coverage: float  # fraction of claim content tokens found in the span
    score: float  # token F1 between claim and span; rewards tight spans


def locate_best_span(
    claim: str,
    evidence_text: str,
    *,
    max_window_sentences: int = 3,
) -> LocatedSpan | None:
    """Return the evidence sentence window that best supports the claim.

    Windows of 1..max_window_sentences consecutive evidence sentences are
    scored by claim-token coverage, with token F1 breaking ties so the
    tightest sufficient span wins over a sprawling one.
    """
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return None

    sentence_offsets = sentences_with_offsets(evidence_text)
    if not sentence_offsets:
        return None

    sentence_tokens = [
        content_tokens(evidence_text[start:end]) for start, end in sentence_offsets
    ]

    best: LocatedSpan | None = None
    for first in range(len(sentence_offsets)):
        window_tokens: set[str] = set()
        for last in range(first, min(first + max_window_sentences, len(sentence_offsets))):
            window_tokens = window_tokens | sentence_tokens[last]
            candidate = _score_window(
                claim_tokens,
                window_tokens,
                start_char=sentence_offsets[first][0],
                end_char=sentence_offsets[last][1],
                evidence_text=evidence_text,
            )
            if candidate.claim_coverage == 0.0:
                continue
            if best is None or (candidate.claim_coverage, candidate.score) > (
                best.claim_coverage,
                best.score,
            ):
                best = candidate
    return best


def _score_window(
    claim_tokens: set[str],
    window_tokens: set[str],
    *,
    start_char: int,
    end_char: int,
    evidence_text: str,
) -> LocatedSpan:
    overlap = len(claim_tokens & window_tokens)
    coverage = overlap / len(claim_tokens)
    precision = overlap / len(window_tokens) if window_tokens else 0.0
    f1 = (
        2 * precision * coverage / (precision + coverage)
        if precision + coverage > 0
        else 0.0
    )
    return LocatedSpan(
        start_char=start_char,
        end_char=end_char,
        text=evidence_text[start_char:end_char],
        claim_coverage=coverage,
        score=f1,
    )
