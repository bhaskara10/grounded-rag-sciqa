"""Cross-encoder reranking of retrieval candidates.

The hybrid retriever optimizes recall over the whole corpus; the cross-encoder
reads each (question, chunk) pair jointly and produces a calibrated relevance
probability, which downstream adaptive selection can threshold. Scores from
the two stages are never mixed.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from sciqa_schema import EvidenceChunk

from .text import content_tokens

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    """Reorders candidate chunks by joint (query, chunk) relevance in [0, 1]."""

    @property
    def name(self) -> str: ...

    def rerank(
        self, query: str, chunks: Sequence[EvidenceChunk], *, top_k: int
    ) -> list[EvidenceChunk]: ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder wrapper; lazy load, batched scoring.

    Raw logits are squashed through a sigmoid so chunk.score is a probability
    that adaptive selection can threshold.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model = None

    @property
    def name(self) -> str:
        return self.model_name

    def rerank(
        self, query: str, chunks: Sequence[EvidenceChunk], *, top_k: int
    ) -> list[EvidenceChunk]:
        if not chunks or top_k <= 0:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from .encoders import pick_device

            self._model = CrossEncoder(
                self.model_name, device=self._device or pick_device()
            )
        logits = self._model.predict(
            [(query, chunk.text) for chunk in chunks],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        rescored = [
            chunk.model_copy(update={"score": _sigmoid(float(logit))})
            for chunk, logit in zip(chunks, logits, strict=True)
        ]
        return sorted(rescored, key=lambda chunk: (-chunk.score, chunk.chunk_id))[:top_k]


class LexicalReranker:
    """Deterministic token-F1 reranker for tests and offline environments."""

    @property
    def name(self) -> str:
        return "lexical-f1"

    def rerank(
        self, query: str, chunks: Sequence[EvidenceChunk], *, top_k: int
    ) -> list[EvidenceChunk]:
        if top_k <= 0:
            return []
        query_tokens = content_tokens(query)
        rescored = [
            chunk.model_copy(update={"score": _token_f1(query_tokens, content_tokens(chunk.text))})
            for chunk in chunks
        ]
        return sorted(rescored, key=lambda chunk: (-chunk.score, chunk.chunk_id))[:top_k]


def _token_f1(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    overlap = len(query_tokens & chunk_tokens)
    if overlap == 0 or not query_tokens or not chunk_tokens:
        return 0.0
    precision = overlap / len(chunk_tokens)
    recall = overlap / len(query_tokens)
    return 2 * precision * recall / (precision + recall)


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))
