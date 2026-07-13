import math

import pytest

from services.eval.app.core import metrics

RANKED = ["a", "b", "c", "d", "e"]


class TestRetrievalMetrics:
    def test_precision_at_k(self):
        assert metrics.precision_at_k(RANKED, {"a", "c"}, 5) == pytest.approx(0.4)
        assert metrics.precision_at_k(RANKED, {"a"}, 1) == 1.0
        assert metrics.precision_at_k(RANKED, {"zzz"}, 5) == 0.0

    def test_precision_counts_missing_slots_against_score(self):
        assert metrics.precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)

    def test_recall_at_k(self):
        assert metrics.recall_at_k(RANKED, {"a", "zzz"}, 5) == pytest.approx(0.5)
        assert metrics.recall_at_k(RANKED, set(), 5) == 0.0

    def test_reciprocal_rank(self):
        assert metrics.reciprocal_rank(RANKED, {"c"}) == pytest.approx(1 / 3)
        assert metrics.reciprocal_rank(RANKED, {"zzz"}) == 0.0

    def test_ndcg_perfect_ranking_is_one(self):
        assert metrics.ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_ndcg_penalizes_late_hits(self):
        early = metrics.ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        late = metrics.ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        assert early == pytest.approx(1.0)
        assert late == pytest.approx(1 / math.log2(4))

    def test_invalid_k_is_rejected(self):
        with pytest.raises(ValueError):
            metrics.precision_at_k(RANKED, {"a"}, 0)


class TestTextMetrics:
    def test_token_f1_exact_match(self):
        assert metrics.token_f1("the answer is 42", ["the answer is 42"]) == 1.0

    def test_token_f1_takes_best_reference(self):
        score = metrics.token_f1("42", ["unrelated words", "42"])
        assert score == 1.0

    def test_token_f1_no_overlap(self):
        assert metrics.token_f1("cats", ["dogs"]) == 0.0

    def test_rouge_exact_match_is_one(self):
        scores = metrics.rouge_scores("the model improved", ["the model improved"])
        assert scores["rouge1"] == pytest.approx(1.0)
        assert scores["rougeL"] == pytest.approx(1.0)

    def test_rouge_partial_overlap_is_between_zero_and_one(self):
        scores = metrics.rouge_scores(
            "the model improved F1", ["the model degraded recall"]
        )
        assert 0.0 < scores["rouge1"] < 1.0

    def test_bleu_identical_text_scores_high(self):
        same = metrics.bleu_score(
            "the model improved f1 by four points", ["the model improved f1 by four points"]
        )
        different = metrics.bleu_score(
            "the model improved f1 by four points", ["cats are pets"]
        )
        assert same > 0.9
        assert different < 0.05

    def test_meteor_identical_text_scores_high(self):
        try:
            score = metrics.meteor("the model improved", ["the model improved"])
        except LookupError:
            pytest.skip("wordnet corpus unavailable offline")
        assert score > 0.9

    def test_empty_candidate_scores_zero(self):
        assert metrics.bleu_score("", ["reference"]) == 0.0
        assert metrics.token_f1("", ["reference"]) == 0.0


class TestOpsMetrics:
    def test_percentiles_interpolate(self):
        values = [10.0, 20.0, 30.0, 40.0]
        assert metrics.percentile(values, 0.5) == pytest.approx(25.0)
        assert metrics.percentile(values, 0.0) == 10.0
        assert metrics.percentile(values, 1.0) == 40.0
        assert metrics.percentile([], 0.95) == 0.0

    def test_mean(self):
        assert metrics.mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
        assert metrics.mean([]) == 0.0
