"""Deterministic grounding verifier with span-level attribution.

This is intentionally model-agnostic: generation can come from vLLM, OpenAI, or
an extractive local baseline, but the service only returns an answer after this
policy accepts every sentence-level attribution. For each accepted sentence the
verifier also localizes the exact evidence characters that support it, so the
API returns quotable spans, not just chunk IDs.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from sciqa_schema import (
    EvidenceChunk,
    GeneratedSentence,
    GroundingDecision,
    GroundingVerdict,
    SupportingSpan,
    VerifiedSentence,
)

from .spans import locate_best_span
from .text import content_tokens

NUMBER_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)")


class GroundingVerifier:
    """Enforce sentence-level citations against retrieved chunks."""

    def __init__(
        self,
        min_claim_coverage: float = 0.35,
        max_span_window_sentences: int = 3,
    ) -> None:
        if not 0 < min_claim_coverage <= 1:
            raise ValueError("min_claim_coverage must be between 0 and 1")
        if max_span_window_sentences < 1:
            raise ValueError("max_span_window_sentences must be at least 1")
        self.min_claim_coverage = min_claim_coverage
        self.max_span_window_sentences = max_span_window_sentences

    def verify(
        self,
        sentences: Sequence[GeneratedSentence],
        retrieved_chunks: Sequence[EvidenceChunk],
    ) -> GroundingDecision:
        """Return an abstention decision for generated sentences.

        A sentence is supported only when:
        - it names at least one supporting chunk;
        - all supporting chunks were actually retrieved for this request;
        - every numeric claim in the sentence appears in the cited evidence;
        - localized spans in the cited chunks jointly cover enough of the
          sentence's content tokens.
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

        spans, claim_coverage = self._localize_support(
            sentence.text, supporting_ids, retrieved_by_id
        )

        evidence_text = " ".join(retrieved_by_id[chunk_id].text for chunk_id in supporting_ids)
        missing_numbers = _missing_numbers(sentence.text, evidence_text)
        if missing_numbers:
            return self._unsupported(
                sentence,
                supporting_ids,
                "numeric_claim_not_in_evidence",
                support_score=claim_coverage,
            )

        if claim_coverage < self.min_claim_coverage:
            return self._unsupported(
                sentence,
                supporting_ids,
                "insufficient_evidence_overlap",
                support_score=claim_coverage,
            )

        return VerifiedSentence(
            text=sentence.text,
            supporting_chunk_ids=supporting_ids,
            confidence=sentence.confidence,
            verdict=GroundingVerdict.SUPPORTED,
            support_score=claim_coverage,
            supporting_spans=spans,
        )

    def _localize_support(
        self,
        claim: str,
        supporting_ids: Sequence[str],
        retrieved_by_id: dict[str, EvidenceChunk],
    ) -> tuple[list[SupportingSpan], float]:
        """Locate the best span per cited chunk and their joint claim coverage.

        Coverage is computed over the union of span tokens so a sentence may be
        supported by evidence split across multiple cited chunks.
        """
        claim_tokens = content_tokens(claim)
        if not claim_tokens:
            return [], 0.0

        spans: list[SupportingSpan] = []
        covered_tokens: set[str] = set()
        for chunk_id in supporting_ids:
            chunk = retrieved_by_id[chunk_id]
            located = locate_best_span(
                claim,
                chunk.text,
                max_window_sentences=self.max_span_window_sentences,
            )
            if located is None:
                continue
            spans.append(
                SupportingSpan(
                    chunk_id=chunk_id,
                    start_char=located.start_char,
                    end_char=located.end_char,
                    text=located.text,
                    score=round(located.score, 4),
                )
            )
            covered_tokens |= claim_tokens & content_tokens(located.text)

        return spans, len(covered_tokens) / len(claim_tokens)

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


def _numbers(text: str) -> set[str]:
    return {number.rstrip(".").lower() for number in NUMBER_RE.findall(text)}


def _missing_numbers(sentence_text: str, evidence_text: str) -> set[str]:
    return _numbers(sentence_text) - _numbers(evidence_text)
