import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from amazon_recsys.config import SplitCfg
from amazon_recsys.data.schemas import INTERACTIONS
from amazon_recsys.data.splits import (
    LeakageError,
    assert_no_leakage,
    make_splits,
    split_domain,
)
from amazon_recsys.utils.io import read_json, read_parquet, write_parquet


def _pairs(df):
    return set(zip(df["user_id"], df["item_id"], strict=True))


def test_leave_one_out_by_time_on_hand_split(hand_split):
    df, expected = hand_split
    splits, _ = make_splits(df, SplitCfg(min_user_interactions=3))
    for name in ("train", "val", "test"):
        assert _pairs(splits[name]) == expected[name], name


def test_input_order_does_not_matter(hand_split):
    df, expected = hand_split
    splits, _ = make_splits(df.sample(frac=1.0, random_state=3), SplitCfg())
    assert _pairs(splits["test"]) == expected["test"]


def test_no_pair_in_more_than_one_split_and_val_test_users_in_train(hand_split):
    df, _ = hand_split
    splits, manifest = make_splits(df, SplitCfg())
    pairs = [_pairs(splits[n]) for n in ("train", "val", "test")]
    assert not (pairs[0] & pairs[1]) and not (pairs[0] & pairs[2]) and not (pairs[1] & pairs[2])
    train_users = set(splits["train"]["user_id"])
    assert set(splits["val"]["user_id"]) <= train_users
    assert set(splits["test"]["user_id"]) <= train_users
    assert manifest["leakage_check"] == {"passed": True, "violations": 0}


def test_same_seed_identical_split_and_ties_are_deterministic(interactions_factory):
    # u1 has a three-way timestamp tie for the last position.
    rows = [("u1", "i1", 1), ("u1", "i2", 5), ("u1", "i3", 5), ("u1", "i4", 5)]
    df = interactions_factory(rows)
    a, _ = make_splits(df, SplitCfg(seed=7, min_user_interactions=3))
    b, _ = make_splits(df.iloc[::-1], SplitCfg(seed=7, min_user_interactions=3))
    assert _pairs(a["test"]) == _pairs(b["test"])
    assert _pairs(a["val"]) == _pairs(b["val"])
    outcomes = {
        tuple(sorted(_pairs(make_splits(df, SplitCfg(seed=s, min_user_interactions=3))[0]["test"])))
        for s in range(20)
    }
    assert len(outcomes) > 1  # the seed actually changes the tie-break


def test_users_with_too_few_interactions_stay_in_train(interactions_factory):
    df = interactions_factory([("u1", "i1", 1), ("u1", "i2", 2), ("u1", "i3", 3), ("u2", "i1", 1)])
    splits, manifest = make_splits(df, SplitCfg(min_user_interactions=3))
    assert "u2" not in set(splits["test"]["user_id"]) | set(splits["val"]["user_id"])
    assert ("u2", "i1") in _pairs(splits["train"])
    assert manifest["users"]["train_only"] == 1

    strict, _ = make_splits(df, SplitCfg(min_user_interactions=4))
    assert strict["test"].empty and strict["val"].empty
    assert len(strict["train"]) == 4


def test_repeat_pairs_keep_earliest_and_never_leak(interactions_factory):
    rows = [("u1", "i1", 1), ("u1", "i2", 2), ("u1", "i3", 3), ("u1", "i1", 9)]
    splits, manifest = make_splits(interactions_factory(rows), SplitCfg(min_user_interactions=3))
    assert manifest["dropped_repeat_pairs"] == 1
    assert _pairs(splits["test"]) == {("u1", "i3")}
    assert ("u1", "i1") in _pairs(splits["train"])


def test_manifest_counts_match_files(tiny_cfg, tmp_paths, hand_split):
    df, _ = hand_split
    write_parquet(df, tmp_paths.processed("interactions"), INTERACTIONS)
    manifest = split_domain(tiny_cfg.split, tmp_paths)
    for name in ("train", "val", "test"):
        assert manifest["counts"][name] == len(read_parquet(tmp_paths.split(name), INTERACTIONS))
    assert read_json(tmp_paths.split_manifest) == manifest
    assert sum(manifest["counts"].values()) + manifest["dropped_repeat_pairs"] == len(df)
    assert manifest["seed"] == tiny_cfg.split.seed
    assert manifest["unseen_in_train"]["test_items"] == 1  # i4 only; i6 is in train via u3
    assert manifest["unseen_in_train"]["val_items"] == 2  # i3 and i5 are not in train


def test_unknown_strategy_raises(hand_split):
    with pytest.raises(ValueError):
        make_splits(hand_split[0], SplitCfg(strategy="random"))


def test_assert_no_leakage_detects_overlap(interactions_factory):
    a = interactions_factory([("u1", "i1", 1)])
    b = interactions_factory([("u1", "i1", 2)])
    with pytest.raises(LeakageError):
        assert_no_leakage({"train": a, "val": b, "test": a.iloc[0:0]})


@given(
    st.lists(
        st.tuples(st.integers(0, 5), st.integers(0, 8), st.integers(0, 6)),
        min_size=1,
        max_size=60,
    ),
    st.integers(0, 3),
)
@settings(max_examples=50, deadline=None)
def test_property_split_invariants(rows, seed):
    data = [
        {
            "domain": "d",
            "user_id": f"u{u}",
            "item_id": f"i{i}",
            "rating": 4.0,
            "timestamp": 1_000 + t,
            "review_id": f"r{n}",
            "overall_sentiment": None,
            "judge_overall": None,
            "judge_verdict": None,
        }
        for n, (u, i, t) in enumerate(rows)
    ]
    df = pd.DataFrame(data).astype({"overall_sentiment": "float64", "judge_overall": "float64"})
    splits, manifest = make_splits(df, SplitCfg(seed=seed, min_user_interactions=3))
    assert_no_leakage(splits)
    assert sum(manifest["counts"].values()) + manifest["dropped_repeat_pairs"] == len(df)
    train_users = set(splits["train"]["user_id"])
    assert set(splits["val"]["user_id"]) <= train_users
    assert set(splits["test"]["user_id"]) <= train_users
    for name in ("val", "test"):
        assert splits[name]["user_id"].is_unique
