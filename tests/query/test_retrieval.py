import numpy as np
from sciqa_schema import EvidenceChunk

from services.query.app.core.encoders import HashingEncoder
from services.query.app.core.retrieval import Bm25Index, DenseIndex, HybridRetriever, rrf_fuse


def _chunk(chunk_id: str, text: str, doc_id: str = "doc-1") -> EvidenceChunk:
    return EvidenceChunk(chunk_id=chunk_id, doc_id=doc_id, text=text)


CHUNKS = [
    _chunk("c1", "The proposed method improved F1 by 4.2 points on SciFact."),
    _chunk("c2", "We pretrain a transformer encoder on scientific abstracts."),
    _chunk("c3", "Cats and dogs are common household pets.", doc_id="doc-2"),
    _chunk("c4", "The SciFact benchmark measures claim verification in science."),
]


class TestBm25Index:
    def test_exact_term_match_ranks_first(self):
        results = Bm25Index(CHUNKS).search("F1 improvement on SciFact", top_k=2)

        assert results[0].chunk_id == "c1"
        assert results[0].score > 0

    def test_rare_terms_outweigh_common_ones(self):
        chunks = [
            _chunk("common", "the method the method the method"),
            _chunk("rare", "electrolyte additive improves cathode stability"),
        ]
        results = Bm25Index(chunks).search("cathode electrolyte additive", top_k=2)

        assert results[0].chunk_id == "rare"

    def test_doc_filter_restricts_results(self):
        results = Bm25Index(CHUNKS).search("cats and dogs", top_k=5, doc_ids=["doc-1"])

        assert results == []

    def test_empty_query_returns_nothing(self):
        assert Bm25Index(CHUNKS).search("", top_k=5) == []

    def test_no_matching_terms_returns_nothing(self):
        assert Bm25Index(CHUNKS).search("zymurgy", top_k=5) == []


class TestDenseIndex:
    def test_lexically_similar_text_ranks_first(self):
        results = DenseIndex(CHUNKS, HashingEncoder()).search(
            "improved F1 points SciFact", top_k=2
        )

        assert results[0].chunk_id == "c1"

    def test_precomputed_embeddings_are_used(self):
        encoder = HashingEncoder()
        embeddings = encoder.encode([chunk.text for chunk in CHUNKS])
        index = DenseIndex(CHUNKS, encoder, embeddings=embeddings)

        assert index.search("scientific abstracts", top_k=1)[0].chunk_id == "c2"

    def test_embedding_row_mismatch_is_rejected(self):
        try:
            DenseIndex(CHUNKS, HashingEncoder(), embeddings=np.zeros((2, 8), dtype=np.float32))
        except ValueError as error:
            assert "row count" in str(error)
        else:
            raise AssertionError("expected ValueError")


class TestRrfFusion:
    def test_chunk_ranked_by_both_lists_wins(self):
        first = [CHUNKS[0], CHUNKS[1]]
        second = [CHUNKS[3], CHUNKS[0]]

        fused = rrf_fuse([first, second], top_k=3)

        assert fused[0].chunk_id == "c1"
        assert fused[0].score > fused[1].score

    def test_single_list_order_is_preserved(self):
        fused = rrf_fuse([[CHUNKS[1], CHUNKS[2]]], top_k=2)

        assert [chunk.chunk_id for chunk in fused] == ["c2", "c3"]


class TestHybridRetriever:
    def test_hybrid_finds_lexical_and_dense_candidates(self):
        retriever = HybridRetriever(CHUNKS, HashingEncoder())

        results = retriever.search("F1 score on the SciFact benchmark", top_k=3)

        ids = [chunk.chunk_id for chunk in results]
        assert "c1" in ids
        assert "c4" in ids

    def test_doc_filter_applies_to_both_rankers(self):
        retriever = HybridRetriever(CHUNKS, HashingEncoder())

        results = retriever.search("household pets", top_k=5, doc_ids=["doc-2"])

        assert {chunk.doc_id for chunk in results} == {"doc-2"}


class TestHashingEncoder:
    def test_encoding_is_deterministic_and_normalized(self):
        encoder = HashingEncoder()
        first = encoder.encode(["The SciFact benchmark measures claim verification."])
        second = encoder.encode(["The SciFact benchmark measures claim verification."])

        assert np.allclose(first, second)
        assert np.isclose(np.linalg.norm(first[0]), 1.0)

    def test_similar_texts_are_closer_than_unrelated_ones(self):
        encoder = HashingEncoder()
        anchor, similar, unrelated = encoder.encode(
            [
                "The model improved F1 on SciFact.",
                "F1 improved for the model on SciFact.",
                "Cats and dogs are common household pets.",
            ]
        )

        assert anchor @ similar > anchor @ unrelated
