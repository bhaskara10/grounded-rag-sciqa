from services.query.app.core.spans import locate_best_span
from services.query.app.core.text import sentences_with_offsets

EVIDENCE = (
    "Prior work relied on rule-based extraction. "
    "The proposed method improved F1 by 4.2 points on SciFact. "
    "Ablations show the reranker contributes most of the gain."
)


def test_best_span_is_the_supporting_sentence():
    span = locate_best_span("The method improved F1 by 4.2 points.", EVIDENCE)

    assert span is not None
    assert span.text == "The proposed method improved F1 by 4.2 points on SciFact."
    assert EVIDENCE[span.start_char : span.end_char] == span.text
    assert span.claim_coverage == 1.0


def test_span_prefers_tightest_sufficient_window():
    span = locate_best_span("Ablations show the reranker contributes most of the gain.", EVIDENCE)

    assert span is not None
    assert span.text == "Ablations show the reranker contributes most of the gain."


def test_claim_spanning_two_sentences_widens_the_window():
    claim = "The method improved F1 by 4.2 points and the reranker contributes most of the gain."
    span = locate_best_span(claim, EVIDENCE)

    assert span is not None
    assert span.text == (
        "The proposed method improved F1 by 4.2 points on SciFact. "
        "Ablations show the reranker contributes most of the gain."
    )


def test_unrelated_claim_finds_no_span():
    span = locate_best_span("The model was trained on a multilingual legal corpus.", EVIDENCE)

    assert span is None


def test_empty_inputs_find_no_span():
    assert locate_best_span("", EVIDENCE) is None
    assert locate_best_span("The method improved F1.", "") is None


def test_sentence_offsets_are_verbatim():
    text = "First sentence. Second one!  Third?"
    offsets = sentences_with_offsets(text)

    assert [text[start:end] for start, end in offsets] == [
        "First sentence.",
        "Second one!",
        "Third?",
    ]
