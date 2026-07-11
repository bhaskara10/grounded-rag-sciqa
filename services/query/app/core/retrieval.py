"""Hybrid retrieval: Okapi BM25 + dense cosine search, fused with RRF.

Lexical and dense retrieval fail differently — BM25 misses paraphrases, dense
retrieval misses rare exact terms (model names, dataset names, numbers).
Reciprocal-rank fusion keeps candidates that either ranker believes in without
having to calibrate their score scales against each other.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sciqa_schema import EvidenceChunk

from .encoders import TextEncoder
from .text import token_sequence


class Bm25Index:
    """Okapi BM25 over evidence chunks."""

    def __init__(self, chunks: Sequence[EvidenceChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks = list(chunks)
        self._term_counts = [Counter(token_sequence(chunk.text)) for chunk in self._chunks]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._doc_freq: Counter[str] = Counter()
        for counts in self._term_counts:
            self._doc_freq.update(counts.keys())

    def search(
        self,
        query: str,
        *,
        top_k: int = 50,
        doc_ids: Sequence[str] | None = None,
    ) -> list[EvidenceChunk]:
        query_terms = token_sequence(query)
        if not query_terms or top_k <= 0:
            return []

        allowed = set(doc_ids or [])
        scored: list[EvidenceChunk] = []
        for position, chunk in enumerate(self._chunks):
            if allowed and chunk.doc_id not in allowed:
                continue
            score = self._score(query_terms, position)
            if score > 0:
                scored.append(chunk.model_copy(update={"score": score}))
        return sorted(scored, key=lambda chunk: (-chunk.score, chunk.chunk_id))[:top_k]

    def _score(self, query_terms: Sequence[str], position: int) -> float:
        counts = self._term_counts[position]
        length_norm = 1 - self.b + self.b * (
            self._lengths[position] / self._avg_length if self._avg_length else 0.0
        )
        score = 0.0
        for term in dict.fromkeys(query_terms):
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            score += self._idf(term) * (
                frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
            )
        return score

    def _idf(self, term: str) -> float:
        doc_freq = self._doc_freq.get(term, 0)
        return math.log((len(self._chunks) - doc_freq + 0.5) / (doc_freq + 0.5) + 1)


class DenseIndex:
    """Cosine search over L2-normalized chunk embeddings."""

    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        encoder: TextEncoder,
        embeddings: np.ndarray | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self._encoder = encoder
        if embeddings is not None and len(embeddings) != len(self._chunks):
            raise ValueError("embeddings row count must match chunk count")
        self._embeddings = (
            embeddings
            if embeddings is not None
            else encoder.encode([chunk.text for chunk in self._chunks])
            if self._chunks
            else np.zeros((0, 1), dtype=np.float32)
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 50,
        doc_ids: Sequence[str] | None = None,
    ) -> list[EvidenceChunk]:
        if not self._chunks or top_k <= 0:
            return []

        query_vector = self._encoder.encode([query])[0]
        similarities = self._embeddings @ query_vector

        allowed = set(doc_ids or [])
        scored: list[EvidenceChunk] = []
        for position, chunk in enumerate(self._chunks):
            if allowed and chunk.doc_id not in allowed:
                continue
            similarity = float(similarities[position])
            if similarity > 0:
                scored.append(chunk.model_copy(update={"score": similarity}))
        return sorted(scored, key=lambda chunk: (-chunk.score, chunk.chunk_id))[:top_k]


def rrf_fuse(
    rankings: Sequence[Sequence[EvidenceChunk]],
    *,
    top_k: int = 50,
    rrf_k: int = 60,
) -> list[EvidenceChunk]:
    """Reciprocal-rank fusion: score(c) = Σ 1 / (rrf_k + rank_in_list)."""
    fused_scores: dict[str, float] = {}
    by_id: dict[str, EvidenceChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1 / (
                rrf_k + rank
            )
            by_id.setdefault(chunk.chunk_id, chunk)

    ordered = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [
        by_id[chunk_id].model_copy(update={"score": score}) for chunk_id, score in ordered
    ]


class HybridRetriever:
    """BM25 + dense retrieval with reciprocal-rank fusion."""

    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        encoder: TextEncoder,
        *,
        embeddings: np.ndarray | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.bm25 = Bm25Index(chunks)
        self.dense = DenseIndex(chunks, encoder, embeddings=embeddings)
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 50,
        doc_ids: Sequence[str] | None = None,
        candidates_per_ranker: int | None = None,
    ) -> list[EvidenceChunk]:
        pool = candidates_per_ranker or max(top_k * 2, 50)
        return rrf_fuse(
            [
                self.bm25.search(query, top_k=pool, doc_ids=doc_ids),
                self.dense.search(query, top_k=pool, doc_ids=doc_ids),
            ],
            top_k=top_k,
            rrf_k=self.rrf_k,
        )
