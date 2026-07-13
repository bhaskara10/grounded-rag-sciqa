"""Layout-aware PDF parsing via PyMuPDF.

Extracts text blocks in reading order, detects section headings from font
sizes, and pulls ruled tables out as structured markdown — so downstream
chunks know their page, section, and whether they are prose or a table.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pymupdf

HEADING_SIZE_RATIO = 1.15
MAX_HEADING_CHARS = 120


class BlockKind(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"


@dataclass(frozen=True)
class ParsedBlock:
    """One layout unit in reading order."""

    kind: BlockKind
    text: str
    page: int  # 1-based


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    page_count: int
    blocks: list[ParsedBlock] = field(default_factory=list)


def parse_pdf(path: Path) -> ParsedDocument:
    return parse_pdf_bytes(path.read_bytes(), fallback_title=path.stem)


def parse_pdf_bytes(data: bytes, *, fallback_title: str = "untitled") -> ParsedDocument:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as error:
        raise ValueError(f"could not parse PDF: {error}") from error
    with document:
        body_size = _body_font_size(document)
        blocks: list[ParsedBlock] = []
        for page_number, page in enumerate(document, start=1):
            blocks.extend(_parse_page(page, page_number, body_size))

        title = (document.metadata or {}).get("title") or _first_heading(blocks) or fallback_title
        return ParsedDocument(
            title=title.strip(),
            page_count=document.page_count,
            blocks=blocks,
        )


def _parse_page(page: pymupdf.Page, page_number: int, body_size: float) -> list[ParsedBlock]:
    """Interleave text blocks and tables by vertical position."""
    tables = page.find_tables()
    table_rects = [pymupdf.Rect(table.bbox) for table in tables.tables]

    ordered: list[tuple[float, ParsedBlock]] = []
    for table, rect in zip(tables.tables, table_rects, strict=True):
        markdown = table.to_markdown().strip()
        if markdown:
            ordered.append((rect.y0, ParsedBlock(BlockKind.TABLE, markdown, page_number)))

    for raw_block in page.get_text("dict")["blocks"]:
        if raw_block.get("type") != 0:  # images and drawings
            continue
        block_rect = pymupdf.Rect(raw_block["bbox"])
        if any(_center_inside(block_rect, rect) for rect in table_rects):
            continue  # table cells come through the table extractor instead
        text, max_size = _block_text(raw_block)
        if not text:
            continue
        kind = (
            BlockKind.HEADING
            if _looks_like_heading(text, max_size, body_size)
            else BlockKind.PARAGRAPH
        )
        ordered.append((block_rect.y0, ParsedBlock(kind, text, page_number)))

    return [block for _, block in sorted(ordered, key=lambda item: item[0])]


def _block_text(raw_block: dict) -> tuple[str, float]:
    lines: list[str] = []
    max_size = 0.0
    for line in raw_block.get("lines", []):
        spans = [span["text"] for span in line.get("spans", [])]
        max_size = max([max_size, *(span["size"] for span in line.get("spans", []))])
        joined = "".join(spans).strip()
        if joined:
            lines.append(joined)
    return " ".join(lines).strip(), max_size


def _body_font_size(document: pymupdf.Document) -> float:
    """Character-weighted median span size ≈ the body text size."""
    weights: Counter[float] = Counter()
    for page in document:
        for raw_block in page.get_text("dict")["blocks"]:
            if raw_block.get("type") != 0:
                continue
            for line in raw_block.get("lines", []):
                for span in line.get("spans", []):
                    weights[round(span["size"], 1)] += len(span["text"])
    if not weights:
        return 0.0

    total = sum(weights.values())
    seen = 0
    for size in sorted(weights):
        seen += weights[size]
        if seen * 2 >= total:
            return size
    return max(weights)


def _looks_like_heading(text: str, max_size: float, body_size: float) -> bool:
    return (
        body_size > 0
        and max_size >= body_size * HEADING_SIZE_RATIO
        and len(text) <= MAX_HEADING_CHARS
        and not text.endswith(".")
    )


def _first_heading(blocks: list[ParsedBlock]) -> str | None:
    for block in blocks:
        if block.kind == BlockKind.HEADING:
            return block.text
    return None


def _center_inside(inner: pymupdf.Rect, outer: pymupdf.Rect) -> bool:
    center = ((inner.x0 + inner.x1) / 2, (inner.y0 + inner.y1) / 2)
    return outer.contains(pymupdf.Point(*center))
