"""Shared lexical utilities for retrieval and grounding.

One tokenizer for the whole query path: BM25, span localization, and the
grounding verifier must agree on what a "content token" is, otherwise support
scores are not comparable across stages.
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)?%?")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "how", "in", "is", "it", "of", "on", "or", "that",
        "the", "their", "this", "to", "was", "were", "what", "which", "with",
    }
)


def count_tokens(text: str) -> int:
    """Total token count including stopwords; used for context budgeting."""
    return len(TOKEN_RE.findall(text))


def is_content_token(token: str) -> bool:
    """Keep tokens that can carry a factual claim; drop glue words."""
    normalized = token.lower()
    return normalized not in STOPWORDS and (
        len(normalized) > 2 or any(character.isdigit() for character in normalized)
    )


def content_tokens(text: str) -> set[str]:
    """Unique lower-cased content tokens; the unit of lexical support."""
    return {token.lower() for token in TOKEN_RE.findall(text) if is_content_token(token)}


def token_sequence(text: str) -> list[str]:
    """Ordered lower-cased content tokens, duplicates kept (for BM25 term counts)."""
    return [token.lower() for token in TOKEN_RE.findall(text) if is_content_token(token)]


def sentences_with_offsets(text: str) -> list[tuple[int, int]]:
    """Character offsets of each sentence, preserving exact source positions.

    Offsets always satisfy ``text[start:end].strip() == text[start:end]`` so a
    span can be quoted verbatim from the chunk it came from.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in SENTENCE_END_RE.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))

    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            trimmed.append((start, end))
    return trimmed
