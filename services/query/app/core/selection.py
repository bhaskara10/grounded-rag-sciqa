"""Adaptive evidence selection after reranking.

Fixed top-k either starves multi-evidence questions or pads easy ones with
noise. Selection here is adaptive: keep reranked chunks that clear a relevance
threshold, in rank order, until a token budget is spent. If nothing clears the
threshold the result is empty and the pipeline abstains — that is the
low-confidence abstention path, decided by scores rather than by prompting.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sciqa_schema import EvidenceChunk

from .text import count_tokens


@dataclass(frozen=True)
class SelectionResult:
    """Selected evidence plus the bookkeeping the explain/eval paths report."""

    chunks: list[EvidenceChunk] = field(default_factory=list)
    total_tokens: int = 0
    dropped_below_threshold: int = 0
    dropped_over_budget: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.chunks


def select_evidence(
    ranked_chunks: Sequence[EvidenceChunk],
    *,
    min_score: float = 0.5,
    token_budget: int = 1024,
    max_chunks: int = 8,
) -> SelectionResult:
    """Pick evidence adaptively from reranked chunks.

    ``ranked_chunks`` must already be sorted by descending reranker score;
    selection never reorders, it only decides where to stop.
    """
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")

    selected: list[EvidenceChunk] = []
    total_tokens = 0
    dropped_below_threshold = 0
    dropped_over_budget = 0

    for chunk in ranked_chunks:
        if chunk.score < min_score:
            dropped_below_threshold += 1
            continue
        if len(selected) >= max_chunks:
            dropped_over_budget += 1
            continue
        chunk_tokens = count_tokens(chunk.text)
        if selected and total_tokens + chunk_tokens > token_budget:
            dropped_over_budget += 1
            continue
        selected.append(chunk)
        total_tokens += chunk_tokens

    return SelectionResult(
        chunks=selected,
        total_tokens=total_tokens,
        dropped_below_threshold=dropped_below_threshold,
        dropped_over_budget=dropped_over_budget,
    )
