import pymupdf
import pytest


def build_paper_pdf() -> bytes:
    """A tiny two-page 'paper': title, sections, body text, and a ruled table."""
    doc = pymupdf.open()

    page = doc.new_page()
    page.insert_text((72, 80), "Grounded Retrieval for Science", fontsize=18)
    page.insert_textbox(
        pymupdf.Rect(72, 100, 540, 200),
        "We study retrieval-augmented question answering over scientific papers. "
        "Naive RAG systems cite the wrong evidence and hallucinate numbers.",
        fontsize=11,
    )
    page.insert_text((72, 220), "Results", fontsize=15)
    page.insert_textbox(
        pymupdf.Rect(72, 240, 540, 320),
        "The proposed method improved F1 by 4.2 points on SciFact. "
        "Ablations show the reranker contributes most of the gain.",
        fontsize=11,
    )
    x0, y0, width, height = 72, 340, 150, 22
    rows = [["Model", "F1"], ["Baseline", "61.3"], ["Ours", "65.5"]]
    for r in range(len(rows) + 1):
        page.draw_line(
            pymupdf.Point(x0, y0 + r * height), pymupdf.Point(x0 + 2 * width, y0 + r * height)
        )
    for c in range(3):
        page.draw_line(
            pymupdf.Point(x0 + c * width, y0),
            pymupdf.Point(x0 + c * width, y0 + len(rows) * height),
        )
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((x0 + c * width + 6, y0 + r * height + 15), cell, fontsize=10)

    page2 = doc.new_page()
    page2.insert_text((72, 80), "Conclusion", fontsize=15)
    page2.insert_textbox(
        pymupdf.Rect(72, 100, 540, 200),
        "Span-level attribution makes citations verifiable at the character level.",
        fontsize=11,
    )

    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(scope="session")
def paper_pdf_bytes() -> bytes:
    return build_paper_pdf()
