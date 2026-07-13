"""Shared grounding contract models.

These models describe the boundary between generation and verification. The
query service can use any generator, but the response is not publishable until
each sentence is checked against retrieved evidence.
"""
from enum import Enum

from pydantic import BaseModel, Field


class GroundingVerdict(str, Enum):
    """Sentence-level support decision."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class EvidenceChunk(BaseModel):
    """Retrieved chunk made available to the generator and verifier."""

    chunk_id: str
    doc_id: str
    text: str
    score: float = 0.0
    chunk_type: str = "text"
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None


class GeneratedSentence(BaseModel):
    """A generated sentence plus the chunk IDs the model claims support it."""

    text: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class SupportingSpan(BaseModel):
    """Exact evidence characters that support one answer sentence.

    Offsets index into the text of the chunk named by ``chunk_id``, and
    ``text`` is the verbatim quote at those offsets — so a client can
    highlight the supporting passage without re-deriving anything.
    """

    chunk_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str
    score: float = 0.0


class VerifiedSentence(GeneratedSentence):
    """A generated sentence after deterministic grounding checks."""

    verdict: GroundingVerdict
    reason: str | None = None
    support_score: float = 0.0
    supporting_spans: list[SupportingSpan] = Field(default_factory=list)


class GroundingDecision(BaseModel):
    """Result of enforcing the grounding contract for an answer."""

    sentences: list[VerifiedSentence]
    abstained: bool
    abstain_reason: str | None = None
    retrieved_chunk_ids: list[str]
    unsupported_sentence_count: int = 0
