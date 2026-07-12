import pytest
from sciqa_schema import EvidenceChunk

from services.query.app.core.rerank import LexicalReranker
from services.query.app.core.selection import select_evidence


def _chunk(chunk_id: str, text: str, score: float = 0.0) -> EvidenceChunk:
    return EvidenceChunk(chunk_id=chunk_id, doc_id="doc-1", text=text, score=score)


class TestLexicalReranker:
    def test_most_relevant_chunk_ranks_first(self):
        chunks = [
            _chunk("noise", "Cats and dogs are common household pets."),
            _chunk("hit", "The proposed method improved F1 by 4.2 points on SciFact."),
        ]

        ranked = LexicalReranker().rerank(
            "How much did the method improve F1 on SciFact?", chunks, top_k=2
        )

        assert ranked[0].chunk_id == "hit"
        assert ranked[0].score > ranked[1].score

    def test_top_k_truncates(self):
        chunks = [_chunk(f"c{i}", "method improved F1") for i in range(5)]

        assert len(LexicalReranker().rerank("improved F1", chunks, top_k=2)) == 2

    def test_scores_are_probabilities(self):
        chunks = [_chunk("c1", "The method improved F1 on SciFact.")]

        ranked = LexicalReranker().rerank("did the method improve F1?", chunks, top_k=1)

        assert 0.0 <= ranked[0].score <= 1.0


class TestAdaptiveSelection:
    def test_low_scoring_chunks_are_dropped(self):
        result = select_evidence(
            [_chunk("strong", "evidence", 0.9), _chunk("weak", "noise", 0.2)],
            min_score=0.5,
        )

        assert [chunk.chunk_id for chunk in result.chunks] == ["strong"]
        assert result.dropped_below_threshold == 1

    def test_all_weak_evidence_selects_nothing(self):
        result = select_evidence([_chunk("weak", "noise", 0.1)], min_score=0.5)

        assert result.is_empty

    def test_token_budget_stops_selection(self):
        long_text = "token " * 100
        result = select_evidence(
            [
                _chunk("first", long_text, 0.9),
                _chunk("second", long_text, 0.8),
                _chunk("third", long_text, 0.7),
            ],
            min_score=0.5,
            token_budget=150,
        )

        assert [chunk.chunk_id for chunk in result.chunks] == ["first"]
        assert result.dropped_over_budget == 2
        assert result.total_tokens == 100

    def test_first_chunk_always_fits_the_budget(self):
        result = select_evidence(
            [_chunk("huge", "token " * 5000, 0.9)], min_score=0.5, token_budget=100
        )

        assert not result.is_empty

    def test_max_chunks_caps_selection(self):
        chunks = [_chunk(f"c{i}", "short evidence", 0.9) for i in range(10)]

        result = select_evidence(chunks, min_score=0.5, max_chunks=3)

        assert len(result.chunks) == 3

    def test_rank_order_is_preserved(self):
        result = select_evidence(
            [_chunk("a", "text", 0.9), _chunk("b", "text", 0.8), _chunk("c", "text", 0.7)],
            min_score=0.5,
        )

        assert [chunk.chunk_id for chunk in result.chunks] == ["a", "b", "c"]

    def test_invalid_parameters_are_rejected(self):
        with pytest.raises(ValueError):
            select_evidence([], min_score=1.5)
        with pytest.raises(ValueError):
            select_evidence([], token_budget=0)
        with pytest.raises(ValueError):
            select_evidence([], max_chunks=0)
