"""Deterministic grounding verifier.

This is intentionally model-agnostic: generation can come from vLLM, OpenAI, or
an extractive local baseline, but the service only returns an answer after this
policy accepts every sentence-level attribution.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from sciqa_schema import (
    EvidenceChunk,
    GeneratedSentence,
    GroundingDecision,
    GroundingVerdict,
    VerifiedSentence,
)

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)?%?")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)")

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
    "with",
}


class GroundingVerifier:
    """Enforce sentence-level citations against retrieved chunks."""

    def __init__(self, min_token_overlap: float = 0.35) -> None:
        if not 0 < min_token_overlap <= 1:
            raise ValueError("min_token_overlap must be between 0 and 1")
        self.min_token_overlap = min_token_overlap

    def verify(
        self,
        sentences: Sequence[GeneratedSentence],
        retrieved_chunks: Sequence[EvidenceChunk],
    ) -> GroundingDecision:
        """Return an abstention decision for generated sentences.

        A sentence is supported only when:
        - it names at least one supporting chunk;
        - all supporting chunks were actually retrieved for this request;
        - its content words overlap enough with the cited evidence;
        - every numeric claim in the sentence appears in the cited evidence.
        """
        retrieved_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
        verified = [
            self._verify_sentence(sentence, retrieved_by_id)
            for sentence in sentences
        ]
        unsupported_count = sum(
            sentence.verdict == GroundingVerdict.UNSUPPORTED for sentence in verified
        )

        if not verified:
            return GroundingDecision(
                sentences=[],
                abstained=True,
                abstain_reason="no_answer_sentences",
                retrieved_chunk_ids=list(retrieved_by_id),
                unsupported_sentence_count=0,
            )

        return GroundingDecision(
            sentences=verified,
            abstained=unsupported_count > 0,
            abstain_reason="grounding_contract_failed" if unsupported_count else None,
            retrieved_chunk_ids=list(retrieved_by_id),
            unsupported_sentence_count=unsupported_count,
        )

    def _verify_sentence(
        self,
        sentence: GeneratedSentence,
        retrieved_by_id: dict[str, EvidenceChunk],
    ) -> VerifiedSentence:
        supporting_ids = _dedupe(sentence.supporting_chunk_ids)
        if not sentence.text.strip():
            return self._unsupported(sentence, supporting_ids, "empty_sentence")
        if not supporting_ids:
            return self._unsupported(sentence, supporting_ids, "missing_supporting_chunk_ids")

        unknown_ids = [chunk_id for chunk_id in supporting_ids if chunk_id not in retrieved_by_id]
        if unknown_ids:
            return self._unsupported(sentence, supporting_ids, "supporting_chunk_not_retrieved")

        evidence_text = " ".join(retrieved_by_id[chunk_id].text for chunk_id in supporting_ids)
        missing_numbers = _missing_numbers(sentence.text, evidence_text)
        if missing_numbers:
            return self._unsupported(
                sentence,
                supporting_ids,
                "numeric_claim_not_in_evidence",
                support_score=_token_overlap(sentence.text, evidence_text),
            )

        support_score = _token_overlap(sentence.text, evidence_text)
        if support_score < self.min_token_overlap:
            return self._unsupported(
                sentence,
                supporting_ids,
                "insufficient_evidence_overlap",
                support_score=support_score,
            )

        return VerifiedSentence(
            text=sentence.text,
            supporting_chunk_ids=supporting_ids,
            confidence=sentence.confidence,
            verdict=GroundingVerdict.SUPPORTED,
            support_score=support_score,
        )

    @staticmethod
    def _unsupported(
        sentence: GeneratedSentence,
        supporting_ids: list[str],
        reason: str,
        support_score: float = 0.0,
    ) -> VerifiedSentence:
        return VerifiedSentence(
            text=sentence.text,
            supporting_chunk_ids=supporting_ids,
            confidence=sentence.confidence,
            verdict=GroundingVerdict.UNSUPPORTED,
            reason=reason,
            support_score=support_score,
        )


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if _is_content_token(token)
    }


def _numbers(text: str) -> set[str]:
    return {number.rstrip(".").lower() for number in NUMBER_RE.findall(text)}


def _missing_numbers(sentence_text: str, evidence_text: str) -> set[str]:
    return _numbers(sentence_text) - _numbers(evidence_text)


def _token_overlap(sentence_text: str, evidence_text: str) -> float:
    sentence_tokens = _tokens(sentence_text)
    if not sentence_tokens:
        return 0.0
    evidence_tokens = _tokens(evidence_text)
    return len(sentence_tokens & evidence_tokens) / len(sentence_tokens)


def _is_content_token(token: str) -> bool:
    normalized = token.lower()
    return normalized not in STOPWORDS and (
        len(normalized) > 2 or any(character.isdigit() for character in normalized)
    )
