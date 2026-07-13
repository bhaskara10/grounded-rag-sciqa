import pytest

from services.ingest.app.core.chunking import chunk_document
from services.ingest.app.core.pdf_parser import BlockKind, ParsedBlock, ParsedDocument


def _doc(blocks: list[ParsedBlock]) -> ParsedDocument:
    return ParsedDocument(title="t", page_count=2, blocks=blocks)


def test_chunks_inherit_section_page_and_type():
    parsed = _doc(
        [
            ParsedBlock(BlockKind.HEADING, "Results", 1),
            ParsedBlock(BlockKind.PARAGRAPH, "F1 improved by 4.2 points.", 1),
            ParsedBlock(BlockKind.TABLE, "|Model|F1|\n|---|---|\n|Ours|65.5|", 1),
            ParsedBlock(BlockKind.HEADING, "Conclusion", 2),
            ParsedBlock(BlockKind.PARAGRAPH, "Attribution is verifiable.", 2),
        ]
    )

    chunks = chunk_document(parsed, doc_id="paper")

    text_chunk, table_chunk, conclusion_chunk = chunks
    assert text_chunk.section_path == ["Results"]
    assert text_chunk.page_start == 1
    assert text_chunk.chunk_type == "text"
    assert table_chunk.chunk_type == "table"
    assert "65.5" in table_chunk.text
    assert conclusion_chunk.section_path == ["Conclusion"]
    assert conclusion_chunk.page_start == 2


def test_chunk_ids_are_stable_and_sequential():
    parsed = _doc([ParsedBlock(BlockKind.PARAGRAPH, f"Paragraph {i}.", 1) for i in range(3)])

    chunks = chunk_document(parsed, doc_id="paper", max_words=2, overlap_words=0)

    assert [chunk.chunk_id for chunk in chunks] == [
        f"paper:chunk:{i}" for i in range(len(chunks))
    ]


def test_small_paragraphs_merge_into_one_chunk():
    parsed = _doc(
        [
            ParsedBlock(BlockKind.PARAGRAPH, "First sentence.", 1),
            ParsedBlock(BlockKind.PARAGRAPH, "Second sentence.", 1),
        ]
    )

    chunks = chunk_document(parsed, doc_id="paper")

    assert len(chunks) == 1
    assert chunks[0].text == "First sentence. Second sentence."


def test_long_paragraph_splits_with_overlap():
    words = " ".join(f"word{i}" for i in range(100))
    parsed = _doc([ParsedBlock(BlockKind.PARAGRAPH, words, 1)])

    chunks = chunk_document(parsed, doc_id="paper", max_words=40, overlap_words=10)

    assert len(chunks) > 1
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-10:] == second_words[:10]


def test_section_change_flushes_buffer():
    parsed = _doc(
        [
            ParsedBlock(BlockKind.HEADING, "Intro", 1),
            ParsedBlock(BlockKind.PARAGRAPH, "Intro text.", 1),
            ParsedBlock(BlockKind.HEADING, "Methods", 1),
            ParsedBlock(BlockKind.PARAGRAPH, "Methods text.", 1),
        ]
    )

    chunks = chunk_document(parsed, doc_id="paper")

    assert [chunk.section_path for chunk in chunks] == [["Intro"], ["Methods"]]


def test_invalid_budgets_are_rejected():
    parsed = _doc([])
    with pytest.raises(ValueError):
        chunk_document(parsed, doc_id="paper", max_words=0)
    with pytest.raises(ValueError):
        chunk_document(parsed, doc_id="paper", max_words=10, overlap_words=10)
