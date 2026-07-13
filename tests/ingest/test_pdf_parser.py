from services.ingest.app.core.pdf_parser import BlockKind, parse_pdf_bytes


def test_headings_are_detected_by_font_size(paper_pdf_bytes):
    parsed = parse_pdf_bytes(paper_pdf_bytes)

    headings = [block.text for block in parsed.blocks if block.kind == BlockKind.HEADING]
    assert "Grounded Retrieval for Science" in headings
    assert "Results" in headings
    assert "Conclusion" in headings


def test_body_paragraphs_are_extracted_in_order(paper_pdf_bytes):
    parsed = parse_pdf_bytes(paper_pdf_bytes)

    paragraphs = [block.text for block in parsed.blocks if block.kind == BlockKind.PARAGRAPH]
    assert any("hallucinate numbers" in text for text in paragraphs)
    assert any("improved F1 by 4.2 points" in text for text in paragraphs)

    intro = next(i for i, block in enumerate(parsed.blocks) if "hallucinate" in block.text)
    results = next(i for i, block in enumerate(parsed.blocks) if "4.2 points" in block.text)
    assert intro < results


def test_ruled_table_is_extracted_as_markdown(paper_pdf_bytes):
    parsed = parse_pdf_bytes(paper_pdf_bytes)

    [table] = [block for block in parsed.blocks if block.kind == BlockKind.TABLE]
    assert "Baseline" in table.text
    assert "65.5" in table.text
    assert table.page == 1


def test_table_text_is_not_duplicated_into_paragraphs(paper_pdf_bytes):
    parsed = parse_pdf_bytes(paper_pdf_bytes)

    paragraphs = [block.text for block in parsed.blocks if block.kind != BlockKind.TABLE]
    assert not any("61.3" in text for text in paragraphs)


def test_pages_are_tracked_one_based(paper_pdf_bytes):
    parsed = parse_pdf_bytes(paper_pdf_bytes)

    assert parsed.page_count == 2
    conclusion = next(block for block in parsed.blocks if "character level" in block.text)
    assert conclusion.page == 2


def test_title_falls_back_to_first_heading(paper_pdf_bytes):
    parsed = parse_pdf_bytes(paper_pdf_bytes, fallback_title="fallback")

    assert parsed.title == "Grounded Retrieval for Science"
