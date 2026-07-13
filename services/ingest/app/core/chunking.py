"""Metadata-rich, section-aware chunking of parsed documents.

Paragraphs accumulate within their section up to a word budget; tables become
standalone chunks so their structure survives retrieval. Every chunk carries
doc_id, section path, starting page, and chunk type — the metadata the query
service needs for filtering and for citing precisely.
"""
from __future__ import annotations

from sciqa_schema import EvidenceChunk

from .pdf_parser import BlockKind, ParsedBlock, ParsedDocument

MAX_CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 40


def chunk_document(
    parsed: ParsedDocument,
    *,
    doc_id: str,
    max_words: int = MAX_CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[EvidenceChunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    builder = _ChunkBuilder(doc_id=doc_id, max_words=max_words, overlap_words=overlap_words)
    for block in parsed.blocks:
        if block.kind == BlockKind.HEADING:
            builder.start_section(block.text)
        elif block.kind == BlockKind.TABLE:
            builder.add_table(block)
        else:
            builder.add_paragraph(block)
    return builder.finish()


class _ChunkBuilder:
    def __init__(self, *, doc_id: str, max_words: int, overlap_words: int) -> None:
        self.doc_id = doc_id
        self.max_words = max_words
        self.overlap_words = overlap_words
        self.chunks: list[EvidenceChunk] = []
        self.section: list[str] = []
        self._buffer: list[str] = []
        self._buffer_words = 0
        self._buffer_page: int | None = None

    def start_section(self, heading: str) -> None:
        self.flush()
        self.section = [heading]

    def add_paragraph(self, block: ParsedBlock) -> None:
        words = len(block.text.split())
        if self._buffer and self._buffer_words + words > self.max_words:
            self.flush()
        if self._buffer_page is None:
            self._buffer_page = block.page
        self._buffer.append(block.text)
        self._buffer_words += words

    def add_table(self, block: ParsedBlock) -> None:
        self.flush()
        self._emit(block.text, page=block.page, chunk_type="table")

    def flush(self) -> None:
        if not self._buffer:
            return
        text = " ".join(self._buffer)
        page = self._buffer_page
        self._buffer = []
        self._buffer_words = 0
        self._buffer_page = None
        for window in _word_windows(text, self.max_words, self.overlap_words):
            self._emit(window, page=page, chunk_type="text")

    def finish(self) -> list[EvidenceChunk]:
        self.flush()
        return self.chunks

    def _emit(self, text: str, *, page: int | None, chunk_type: str) -> None:
        self.chunks.append(
            EvidenceChunk(
                chunk_id=f"{self.doc_id}:chunk:{len(self.chunks)}",
                doc_id=self.doc_id,
                text=text,
                chunk_type=chunk_type,
                section_path=list(self.section),
                page_start=page,
            )
        )


def _word_windows(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Split long text into overlapping word windows; short text passes through."""
    words = text.split()
    if len(words) <= max_words:
        return [text] if words else []

    stride = max_words - overlap_words
    windows: list[str] = []
    for start in range(0, len(words), stride):
        window = words[start : start + max_words]
        windows.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return windows
