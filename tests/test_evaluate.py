"""Unit tests for src.evaluate: metric math checked by hand."""

import pytest

from src.evaluate import average_precision, precision_at_k, recall_at_k

RANKING = ["a", "b", "c", "d", "e"]
RELEVANT = {"a", "c", "f"}  # "f" was never retrieved


def test_precision_at_k():
    assert precision_at_k(RANKING, RELEVANT, 1) == 1.0        # [a]
    assert precision_at_k(RANKING, RELEVANT, 2) == 0.5        # [a, b]
    assert precision_at_k(RANKING, RELEVANT, 5) == pytest.approx(2 / 5)


def test_recall_at_k():
    assert recall_at_k(RANKING, RELEVANT, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKING, RELEVANT, 5) == pytest.approx(2 / 3)


def test_average_precision_by_hand():
    # Relevant docs sit at ranks 1 and 3: AP = (1/1 + 2/3) / |relevant|
    # = (1 + 0.6667) / 3 = 0.5556. The never-retrieved "f" still counts
    # in the denominator -- missing a relevant document costs you.
    assert average_precision(RANKING, RELEVANT) == pytest.approx((1 + 2 / 3) / 3)


def test_perfect_ranking_has_ap_one():
    assert average_precision(["a", "c"], {"a", "c"}) == pytest.approx(1.0)


def test_no_relevant_retrieved_scores_zero():
    assert average_precision(["x", "y"], {"a"}) == 0.0
    assert precision_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert recall_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_empty_relevant_set_is_zero_not_crash():
    assert average_precision(RANKING, set()) == 0.0
    assert recall_at_k(RANKING, set(), 5) == 0.0
