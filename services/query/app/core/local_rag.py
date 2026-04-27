"""Local deterministic RAG baseline.

This module gives the project a small end-to-end path before external search or
generation services are introduced: chunk text, retrieve candidate chunks,
extract a cited sentence, and enforce the grounding verifier.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel
from sciqa_schema import EvidenceChunk, GeneratedSentence, GroundingVerdict, VerifiedSentence

from .grounding import GroundingVerifier

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)?%?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}


class LocalQAResult(BaseModel):
    """Transport-neutral result for the local QA baseline."""

    answer: str
    sentences: list[VerifiedSentence]
    abstained: bool
    abstain_reason: str | None = None
    retrieved_chunk_ids: list[str]
    best_supporting_passages: list[str]


@dataclass(frozen=True)
class _CandidateSentence:
    text: str
    chunk_id: str
    score: float


def chunk_text(
    text: str,
    *,
    doc_id: str,
    max_tokens: int = 160,
    overlap: int = 32,
    section_path: Sequence[str] | None = None,
) -> list[EvidenceChunk]:
    """Split plain text into deterministic overlapping evidence chunks."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap must be zero or positive")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    words = text.split()
    if not words:
        return []

    stride = max_tokens - overlap
    chunks: list[EvidenceChunk] = []
    for chunk_index, start in enumerate(range(0, len(words), stride)):
        chunk_words = words[start : start + max_tokens]
        if not chunk_words:
            continue
        chunks.append(
            EvidenceChunk(
                chunk_id=f"{doc_id}:chunk:{chunk_index}",
                doc_id=doc_id,
                text=" ".join(chunk_words),
                section_path=list(section_path or []),
            )
        )
        if start + max_tokens >= len(words):
            break
    return chunks


class InMemoryChunkIndex:
    """Tiny TF-IDF style retriever for local demos and regression tests."""

    def __init__(self, chunks: Sequence[EvidenceChunk]) -> None:
        self._chunks = list(chunks)
        self._doc_freq = self._document_frequencies(self._chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_ids: Sequence[str] | None = None,
    ) -> list[EvidenceChunk]:
        if top_k <= 0:
            return []

        allowed_doc_ids = set(doc_ids or [])
        query_terms = _tokens(query)
        if not query_terms:
            return []

        scored: list[EvidenceChunk] = []
        for chunk in self._chunks:
            if allowed_doc_ids and chunk.doc_id not in allowed_doc_ids:
                continue
            score = self._score(query_terms, _tokens(chunk.text))
            if score > 0:
                scored.append(chunk.model_copy(update={"score": score}))

        return sorted(scored, key=lambda chunk: (-chunk.score, chunk.chunk_id))[:top_k]

    @staticmethod
    def _document_frequencies(chunks: Sequence[EvidenceChunk]) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for chunk in chunks:
            for token in _tokens(chunk.text):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _score(self, query_terms: set[str], chunk_terms: set[str]) -> float:
        numerator = sum(self._idf(term) for term in query_terms & chunk_terms)
        denominator = sum(self._idf(term) for term in query_terms)
        return numerator / denominator if denominator else 0.0

    def _idf(self, term: str) -> float:
        return math.log((len(self._chunks) + 1) / (self._doc_freq.get(term, 0) + 1)) + 1


def answer_question_local(
    question: str,
    chunks: Sequence[EvidenceChunk],
    *,
    top_k: int = 5,
    verifier: GroundingVerifier | None = None,
) -> LocalQAResult:
    """Retrieve evidence, extract a cited answer sentence, and verify it."""
    retrieved = InMemoryChunkIndex(chunks).search(question, top_k=top_k)
    if not retrieved:
        return _abstain("no_retrieved_evidence", [])

    candidate = _select_candidate_sentence(question, retrieved)
    if candidate is None:
        return _abstain("no_candidate_sentence", retrieved)

    decision = (verifier or GroundingVerifier()).verify(
        [
            GeneratedSentence(
                text=candidate.text,
                supporting_chunk_ids=[candidate.chunk_id],
                confidence=candidate.score,
            )
        ],
        retrieved,
    )
    supported_sentences = [
        sentence
        for sentence in decision.sentences
        if sentence.verdict == GroundingVerdict.SUPPORTED
    ]

    return LocalQAResult(
        answer=" ".join(sentence.text for sentence in supported_sentences)
        if not decision.abstained
        else "",
        sentences=decision.sentences,
        abstained=decision.abstained,
        abstain_reason=decision.abstain_reason,
        retrieved_chunk_ids=decision.retrieved_chunk_ids,
        best_supporting_passages=[chunk.text for chunk in retrieved],
    )


def _select_candidate_sentence(
    question: str,
    retrieved: Sequence[EvidenceChunk],
) -> _CandidateSentence | None:
    query_terms = _tokens(question)
    candidates: list[_CandidateSentence] = []
    for chunk in retrieved:
        for sentence in _sentences(chunk.text):
            score = _overlap_score(query_terms, _tokens(sentence)) * chunk.score
            if score > 0:
                candidates.append(
                    _CandidateSentence(text=sentence, chunk_id=chunk.chunk_id, score=score)
                )
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.score, candidate.chunk_id))


def _abstain(reason: str, retrieved: Sequence[EvidenceChunk]) -> LocalQAResult:
    return LocalQAResult(
        answer="",
        sentences=[],
        abstained=True,
        abstain_reason=reason,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
        best_supporting_passages=[chunk.text for chunk in retrieved],
    )


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if _is_content_token(token)
    }


def _overlap_score(query_terms: set[str], sentence_terms: set[str]) -> float:
    return len(query_terms & sentence_terms) / len(query_terms) if query_terms else 0.0


def _is_content_token(token: str) -> bool:
    normalized = token.lower()
    return normalized not in STOPWORDS and (
        len(normalized) > 2 or any(character.isdigit() for character in normalized)
    )
