import pytest
import torch

from amazon_recsys.config import SplitCfg
from amazon_recsys.data.interactions import build_index
from amazon_recsys.data.splits import make_splits
from amazon_recsys.evaluation.evaluator import Evaluator
from amazon_recsys.models.base import Recommender


class FixedScores(Recommender):
    """Test double: `score_all` returns a fixed [users, items] matrix."""

    def __init__(self, matrix: torch.Tensor):
        super().__init__(matrix.shape[0], matrix.shape[1])
        self.matrix = matrix
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    @classmethod
    def from_config(cls, cfg, graph):  # pragma: no cover - not used
        raise NotImplementedError

    def forward(self, graph=None):  # pragma: no cover - not used
        raise NotImplementedError

    def score_all(self, user_idx, graph=None):
        return self.matrix[user_idx]


@pytest.fixture
def setup(hand_split):
    df, _ = hand_split
    splits, _ = make_splits(df, SplitCfg(min_user_interactions=3))
    user_index, item_index = build_index(*splits.values())
    return splits, user_index, item_index


def _matrix(user_index, item_index, rule):
    m = torch.zeros(len(user_index), len(item_index))
    for u, ui in user_index.items():
        for i, ii in item_index.items():
            m[ui, ii] = rule(u, i)
    return m


def _oracle(splits, user_index, item_index, split, sign=1.0):
    target = set(zip(splits[split].user_id, splits[split].item_id, strict=True))
    return FixedScores(
        _matrix(user_index, item_index, lambda u, i: sign * (10.0 if (u, i) in target else 0.0))
    )


def test_build_index_is_sorted_and_covers_all_splits(setup):
    splits, users, items = setup
    assert list(users) == sorted(users) and list(items) == sorted(items)
    assert set(items) == {"i1", "i2", "i3", "i4", "i5", "i6"}  # includes cold val/test items
    assert sorted(users.values()) == list(range(len(users)))


def test_oracle_gets_perfect_score_and_reversed_gets_zero(setup):
    splits, users, items = setup
    ev = Evaluator(splits, users, items, ks=[1, 3])
    good = ev.evaluate(_oracle(splits, users, items, "test"), "test")
    assert good["hit_rate@1"] == 1.0 and good["mrr"] == 1.0 and good["ndcg@3"] == 1.0
    bad = ev.evaluate(_oracle(splits, users, items, "test", sign=-1.0), "test")
    assert bad["hit_rate@1"] == 0.0
    assert good["n_users"] == bad["n_users"] == 2.0  # u3 has no test item


def test_train_positives_are_excluded(setup):
    splits, users, items = setup
    # Give train positives the highest score and the target the second highest.
    train = set(zip(splits["train"].user_id, splits["train"].item_id, strict=True))
    test = set(zip(splits["test"].user_id, splits["test"].item_id, strict=True))

    def rule(u, i):
        return 100.0 if (u, i) in train else (50.0 if (u, i) in test else 0.0)

    model = FixedScores(_matrix(users, items, rule))
    out = Evaluator(splits, users, items, ks=[1]).evaluate(model, "val")
    # On val the test item scores 50 but val target scores 0: target is NOT first,
    # while train items (100) would have been first if they were not excluded.
    assert out["hit_rate@1"] == 0.0
    out_test = Evaluator(splits, users, items, ks=[1]).evaluate(model, "test")
    assert out_test["hit_rate@1"] == 1.0  # only reachable if train items were excluded


def test_val_positives_are_excluded_when_scoring_test(setup):
    splits, users, items = setup
    val = set(zip(splits["val"].user_id, splits["val"].item_id, strict=True))
    test = set(zip(splits["test"].user_id, splits["test"].item_id, strict=True))

    def rule(u, i):
        return 100.0 if (u, i) in val else (50.0 if (u, i) in test else 0.0)

    model = FixedScores(_matrix(users, items, rule))
    out = Evaluator(splits, users, items, ks=[1]).evaluate(model, "test")
    assert out["hit_rate@1"] == 1.0


def test_excluded_items_do_not_count_as_candidates(setup):
    splits, users, items = setup
    ev = Evaluator(splits, users, items, ks=[10])
    out = ev.evaluate(_oracle(splits, users, items, "test", sign=-1.0), "test")
    # u1: 6 items, 2 train + 1 val excluded -> 3 candidates; target ranked last => 1/3.
    # u2: 6 items, 1 train + 1 val excluded -> 4 candidates; target ranked last => 1/4.
    assert out["mrr"] == pytest.approx((1 / 3 + 1 / 4) / 2)


def test_results_identical_for_same_model_and_metrics_for_every_k(setup):
    splits, users, items = setup
    torch.manual_seed(1)
    model = FixedScores(torch.rand(len(users), len(items)))
    ev = Evaluator(splits, users, items, ks=[1, 5, 10])
    a, b = ev.evaluate(model, "val"), ev.evaluate(model, "val")
    assert a == b
    for k in (1, 5, 10):
        for m in ("recall", "ndcg", "hit_rate", "precision"):
            assert f"{m}@{k}" in a
    assert "mrr" in a


def test_ties_broken_deterministically_by_item_index(setup):
    splits, users, items = setup
    model = FixedScores(torch.zeros(len(users), len(items)))  # all ties
    ev = Evaluator(splits, users, items, ks=[1])
    assert ev.evaluate(model, "val") == ev.evaluate(model, "val")


def test_invalid_split_and_training_mode_restored(setup):
    splits, users, items = setup
    ev = Evaluator(splits, users, items, ks=[1])
    model = FixedScores(torch.zeros(len(users), len(items)))
    model.train()
    ev.evaluate(model, "val")
    assert model.training
    with pytest.raises(ValueError):
        ev.evaluate(model, "train")


def test_mf_model_end_to_end(setup, tiny_cfg):
    from types import SimpleNamespace

    import amazon_recsys.models  # noqa: F401
    from amazon_recsys.models.registry import build_model

    splits, users, items = setup
    cfg = tiny_cfg.model_copy(update={"model": tiny_cfg.model.model_copy(update={"name": "mf"})})
    graph = SimpleNamespace(kind="bipartite", num_users=len(users), num_items=len(items))
    out = Evaluator(splits, users, items, ks=[5]).evaluate(build_model(cfg, graph), "val")
    assert 0.0 <= out["recall@5"] <= 1.0 and out["n_users"] == 2.0
