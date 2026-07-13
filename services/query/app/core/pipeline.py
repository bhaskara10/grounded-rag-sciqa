"""The grounded RAG pipeline.

retrieve (hybrid, recall-oriented) -> rerank (cross-encoder, precision)
-> select (adaptive threshold + token budget) -> propose answer sentences
-> verify grounding (span-level attribution) -> answer or abstain.

Every stage is injected behind a small interface, so production runs use real
models while tests swap in deterministic components. The pipeline records per-
stage latencies and token counts because the eval harness reports them as
first-class metrics.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field
from sciqa_schema import EvidenceChunk, GeneratedSentence, VerifiedSentence

from .encoders import TextEncoder
from .grounding import GroundingVerifier
from .rerank import CrossEncoderReranker, Reranker
from .retrieval import HybridRetriever, rrf_fuse
from .selection import select_evidence
from .text import content_tokens, count_tokens, sentences_with_offsets


class PipelineConfig(BaseModel):
    """Stage budgets and thresholds; defaults tuned for the cross-encoder."""

    retrieve_top_k: int = 50
    rerank_top_k: int = 12
    min_rerank_score: float = 0.5
    token_budget: int = 1024
    max_evidence_chunks: int = 8
    answer_sentences: int = 2


class PipelineResult(BaseModel):
    """Full outcome of one grounded QA request, including ops telemetry."""

    question: str
    answer: str
    sentences: list[VerifiedSentence] = Field(default_factory=list)
    abstained: bool
    abstain_reason: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    selected_chunk_ids: list[str] = Field(default_factory=list)
    reranker_scores: list[float] = Field(default_factory=list)
    best_supporting_passages: list[str] = Field(default_factory=list)
    evidence_tokens: int = 0
    answer_tokens: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    """Stage-by-stage retrieval snapshot for /retrieve and /explain."""

    question: str
    lexical: list[EvidenceChunk]
    dense: list[EvidenceChunk]
    fused: list[EvidenceChunk]
    reranked: list[EvidenceChunk]
    selected: list[EvidenceChunk]
    timings_ms: dict[str, float] = Field(default_factory=dict)


class Answerer(Protocol):
    """Proposes answer sentences with claimed citations; the verifier decides."""

    def propose(
        self, question: str, evidence: Sequence[EvidenceChunk]
    ) -> list[GeneratedSentence]: ...


@dataclass(frozen=True)
class _ScoredSentence:
    text: str
    chunk_id: str
    score: float


class ExtractiveAnswerer:
    """Deterministic answerer: quote the evidence sentences that best match.

    Extractive by design — a quoted sentence can always be attributed, so the
    default pipeline works offline with zero generation dependencies. An LLM
    generator plugs in behind the same Answerer protocol.
    """

    def __init__(self, max_sentences: int = 2) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be at least 1")
        self.max_sentences = max_sentences

    def propose(
        self, question: str, evidence: Sequence[EvidenceChunk]
    ) -> list[GeneratedSentence]:
        question_tokens = content_tokens(question)
        if not question_tokens:
            return []

        candidates: list[_ScoredSentence] = []
        for chunk in evidence:
            for start, end in sentences_with_offsets(chunk.text):
                sentence = chunk.text[start:end]
                overlap = len(question_tokens & content_tokens(sentence))
                if overlap == 0:
                    continue
                candidates.append(
                    _ScoredSentence(
                        text=sentence,
                        chunk_id=chunk.chunk_id,
                        score=(overlap / len(question_tokens)) * max(chunk.score, 1e-6),
                    )
                )

        candidates.sort(key=lambda candidate: (-candidate.score, candidate.chunk_id))
        proposed: list[GeneratedSentence] = []
        seen_texts: set[str] = set()
        for candidate in candidates:
            if candidate.text in seen_texts:
                continue
            seen_texts.add(candidate.text)
            proposed.append(
                GeneratedSentence(
                    text=candidate.text,
                    supporting_chunk_ids=[candidate.chunk_id],
                    confidence=min(candidate.score, 1.0),
                )
            )
            if len(proposed) >= self.max_sentences:
                break
        return proposed


class RagPipeline:
    """Hybrid retrieve -> cross-encoder rerank -> adaptive select -> ground."""

    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        encoder: TextEncoder,
        *,
        embeddings: np.ndarray | None = None,
        reranker: Reranker | None = None,
        answerer: Answerer | None = None,
        verifier: GroundingVerifier | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.retriever = HybridRetriever(chunks, encoder, embeddings=embeddings)
        self.reranker: Reranker = reranker or CrossEncoderReranker()
        self.answerer: Answerer = answerer or ExtractiveAnswerer(
            max_sentences=self.config.answer_sentences
        )
        self.verifier = verifier or GroundingVerifier()

    def answer(self, question: str, *, doc_ids: Sequence[str] | None = None) -> PipelineResult:
        timings: dict[str, float] = {}
        total_start = perf_counter()

        retrieved = self._timed(
            timings,
            "retrieve_ms",
            lambda: self.retriever.search(
                question, top_k=self.config.retrieve_top_k, doc_ids=doc_ids
            ),
        )
        if not retrieved:
            return self._abstain(
                question, "no_retrieved_evidence", [], [], timings, total_start
            )

        reranked = self._timed(
            timings,
            "rerank_ms",
            lambda: self.reranker.rerank(question, retrieved, top_k=self.config.rerank_top_k),
        )

        selection = select_evidence(
            reranked,
            min_score=self.config.min_rerank_score,
            token_budget=self.config.token_budget,
            max_chunks=self.config.max_evidence_chunks,
        )
        if selection.is_empty:
            return self._abstain(
                question, "weak_evidence", retrieved, reranked, timings, total_start
            )

        proposed = self.answerer.propose(question, selection.chunks)
        if not proposed:
            return self._abstain(
                question, "no_candidate_sentence", retrieved, reranked, timings, total_start
            )

        decision = self._timed(
            timings,
            "verify_ms",
            lambda: self.verifier.verify(proposed, selection.chunks),
        )

        answer = (
            ""
            if decision.abstained
            else " ".join(sentence.text for sentence in decision.sentences)
        )
        timings["total_ms"] = _elapsed_ms(total_start)
        return PipelineResult(
            question=question,
            answer=answer,
            sentences=decision.sentences,
            abstained=decision.abstained,
            abstain_reason=decision.abstain_reason,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
            selected_chunk_ids=[chunk.chunk_id for chunk in selection.chunks],
            reranker_scores=[round(chunk.score, 4) for chunk in reranked],
            best_supporting_passages=[chunk.text for chunk in selection.chunks],
            evidence_tokens=selection.total_tokens,
            answer_tokens=count_tokens(answer),
            timings_ms=timings,
        )

    def trace(self, question: str, *, doc_ids: Sequence[str] | None = None) -> RetrievalTrace:
        """Stage-by-stage retrieval snapshot without generating an answer."""
        timings: dict[str, float] = {}
        pool = max(self.config.retrieve_top_k * 2, 50)

        lexical = self._timed(
            timings,
            "lexical_ms",
            lambda: self.retriever.bm25.search(question, top_k=pool, doc_ids=doc_ids),
        )
        dense = self._timed(
            timings,
            "dense_ms",
            lambda: self.retriever.dense.search(question, top_k=pool, doc_ids=doc_ids),
        )
        fused = rrf_fuse([lexical, dense], top_k=self.config.retrieve_top_k)
        reranked = self._timed(
            timings,
            "rerank_ms",
            lambda: self.reranker.rerank(question, fused, top_k=self.config.rerank_top_k),
        )
        selection = select_evidence(
            reranked,
            min_score=self.config.min_rerank_score,
            token_budget=self.config.token_budget,
            max_chunks=self.config.max_evidence_chunks,
        )
        return RetrievalTrace(
            question=question,
            lexical=lexical,
            dense=dense,
            fused=fused,
            reranked=reranked,
            selected=selection.chunks,
            timings_ms=timings,
        )

    def _abstain(
        self,
        question: str,
        reason: str,
        retrieved: Sequence[EvidenceChunk],
        reranked: Sequence[EvidenceChunk],
        timings: dict[str, float],
        total_start: float,
    ) -> PipelineResult:
        timings["total_ms"] = _elapsed_ms(total_start)
        fallback_passages = list(reranked[:3]) or list(retrieved[:3])
        return PipelineResult(
            question=question,
            answer="",
            abstained=True,
            abstain_reason=reason,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
            reranker_scores=[round(chunk.score, 4) for chunk in reranked],
            best_supporting_passages=[chunk.text for chunk in fallback_passages],
            timings_ms=timings,
        )

    @staticmethod
    def _timed(timings: dict[str, float], key: str, stage):  # type: ignore[no-untyped-def]
        started = perf_counter()
        result = stage()
        timings[key] = _elapsed_ms(started)
        return result


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)
