import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from amazon_recsys.evaluation.metrics import (
    compute_metrics,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RANKED = [5, 3, 9, 1, 7]
RELEVANT = {3, 1}  # found at ranks 2 and 4


def test_recall():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 2) == 0.5
    assert recall_at_k(RANKED, RELEVANT, 4) == 1.0


def test_precision():
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.5
    assert precision_at_k(RANKED, RELEVANT, 4) == 0.5
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)


def test_hit_rate():
    assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0
    assert hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0


def test_ndcg_hand_computed():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)  # hits at positions 2 and 4
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(dcg / idcg)
    # K = 2: only the hit at position 2; ideal has two hits
    assert ndcg_at_k(RANKED, RELEVANT, 2) == pytest.approx((1 / math.log2(3)) / idcg)


def test_mrr():
    assert mrr(RANKED, RELEVANT) == 0.5
    assert mrr([1, 2, 3], {3}) == pytest.approx(1 / 3)
    assert mrr([1, 2], {9}) == 0.0


def test_no_relevant_items_gives_zero():
    for fn in (recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k):
        assert fn(RANKED, set(), 3) == 0.0
    assert mrr(RANKED, set()) == 0.0


def test_k_larger_than_list():
    assert recall_at_k([3], {3, 1}, 10) == 0.5
    assert ndcg_at_k([3], {3}, 10) == 1.0
    assert precision_at_k([3], {3}, 10) == pytest.approx(1 / 10)


def test_all_relevant_list_is_perfect():
    ranked = [1, 2, 3]
    rel = {1, 2, 3}
    assert recall_at_k(ranked, rel, 3) == 1.0
    assert ndcg_at_k(ranked, rel, 3) == 1.0
    assert precision_at_k(ranked, rel, 3) == 1.0


def test_perfect_vs_reversed_ranking():
    assert ndcg_at_k([3, 1, 5, 9], {3, 1}, 4) == 1.0
    assert ndcg_at_k([5, 9, 1, 3], {3, 1}, 4) < 1.0


def test_compute_metrics_keys_and_values():
    out = compute_metrics(RANKED, RELEVANT, [2, 4])
    assert set(out) == {
        "recall@2", "ndcg@2", "hit_rate@2", "precision@2",
        "recall@4", "ndcg@4", "hit_rate@4", "precision@4",
        "mrr",
    }  # fmt: skip
    assert out["recall@4"] == 1.0 and out["mrr"] == 0.5


@given(
    st.lists(st.integers(0, 30), unique=True, max_size=30),
    st.sets(st.integers(0, 30), max_size=10),
    st.integers(1, 40),
)
def test_property_all_metrics_in_unit_interval(ranked, relevant, k):
    for value in compute_metrics(ranked, relevant, [k]).values():
        assert 0.0 <= value <= 1.0


@given(st.sets(st.integers(0, 30), min_size=1, max_size=10), st.integers(1, 40))
def test_property_perfect_ranking_has_ndcg_one(relevant, k):
    ranked = sorted(relevant) + [x for x in range(31, 45)]
    assert ndcg_at_k(ranked, relevant, k) == pytest.approx(1.0)
