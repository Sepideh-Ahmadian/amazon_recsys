import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from amazon_recsys.config import FilterCfg
from amazon_recsys.data.filter import filter_domain, filter_reviews, k_core, sample_users
from amazon_recsys.data.schemas import REVIEWS
from amazon_recsys.utils.io import read_json, read_parquet, write_parquet


def _frame(pairs):
    rows = [
        {
            "review_id": f"r{n}",
            "domain": "d",
            "user_id": u,
            "item_id": i,
            "rating": 4.0,
            "title": None,
            "text": "x",
            "timestamp": 1_600_000_000_000 + n,
        }
        for n, (u, i) in enumerate(pairs)
    ]
    return pd.DataFrame(
        rows, columns=list(REVIEWS.columns)
    ).astype({"rating": "float64", "timestamp": "int64"})


def _core_plus_noise(n_core_users=6, n_items=5, n_noise=50):
    """Complete user x item block (a 5-core) plus singleton users on private items."""
    pairs = [(f"u{u}", f"i{i}") for u in range(n_core_users) for i in range(n_items)]
    pairs += [(f"noise{n}", f"x{n}") for n in range(n_noise)]
    return _frame(pairs)


def _cfg(**kw):
    base = {"min_user_interactions": 5, "min_item_interactions": 5, "k_core": True}
    return FilterCfg(**{**base, **kw})


@given(
    st.lists(
        st.tuples(st.integers(0, 12), st.integers(0, 12)), min_size=0, max_size=120
    ),
    st.integers(1, 4),
    st.integers(1, 4),
)
@settings(max_examples=60, deadline=None)
def test_k_core_every_user_and_item_meets_minimum(pairs, min_u, min_i):
    df = _frame([(f"u{u}", f"i{i}") for u, i in pairs])
    out = k_core(df, min_u, min_i)
    if len(out):
        assert out.groupby("user_id")["item_id"].nunique().min() >= min_u
        assert out.groupby("item_id")["user_id"].nunique().min() >= min_i


def test_k_core_iterates_to_fixed_point():
    # u3 has 2 items: i1 (shared) and i9 (only u3). Removing i9 (1 user) drops u3 to 1 item,
    # which a single pass would keep but the fixed point removes.
    pairs = [("u1", "i1"), ("u1", "i2"), ("u2", "i1"), ("u2", "i2"), ("u3", "i1"), ("u3", "i9")]
    df = _frame(pairs)
    single = k_core(df, 2, 2, iterate=False)
    full = k_core(df, 2, 2, iterate=True)
    assert "u3" in set(single["user_id"])
    assert set(full["user_id"]) == {"u1", "u2"}
    assert set(full["item_id"]) == {"i1", "i2"}


def test_repeated_reviews_of_one_item_do_not_count_twice():
    df = _frame([("u1", "i1")] * 5)
    assert k_core(df, 2, 1).empty


def test_sampling_is_seeded_and_reproducible():
    df = _core_plus_noise(n_core_users=20, n_items=2, n_noise=0)
    a = sample_users(df, 5, seed=1)
    b = sample_users(df, 5, seed=1)
    c = sample_users(df, 5, seed=2)
    assert a["user_id"].nunique() == 5
    assert set(a["user_id"]) == set(b["user_id"])
    assert set(a["user_id"]) != set(c["user_id"])


def test_sample_noop_when_null_or_cap_exceeds_users():
    df = _core_plus_noise()
    assert len(sample_users(df, None, 0)) == len(df)
    assert len(sample_users(df, 10_000, 0)) == len(df)


def test_sampling_happens_after_first_k_core():
    df = _core_plus_noise()  # 6 core users among 56
    out, summary = filter_reviews(df, _cfg(max_users=5, sample_seed=3))
    assert set(out["user_id"]) <= {f"u{u}" for u in range(6)}
    assert out["user_id"].nunique() == 5
    assert [s["stage"] for s in summary["stages"]] == ["input", "k_core", "sample", "k_core_2"]
    assert summary["stages"][1]["users"] == 6  # noise removed before sampling


def test_second_k_core_runs_after_sampling():
    df = _core_plus_noise()
    out, summary = filter_reviews(df, _cfg(max_users=3))
    # 3 users leave each item with 3 < 5 users, so the second k-core empties the frame.
    assert out.empty
    assert summary["stages"][-1]["stage"] == "k_core_2"
    assert summary["final"] == {"rows": 0, "users": 0, "items": 0}


def test_null_cap_keeps_whole_core_and_skips_sampling_stages():
    out, summary = filter_reviews(_core_plus_noise(), _cfg(max_users=None))
    assert out["user_id"].nunique() == 6
    assert [s["stage"] for s in summary["stages"]] == ["input", "k_core"]


def test_drop_summary_counts_add_up():
    df = _core_plus_noise()
    out, summary = filter_reviews(df, _cfg(max_users=5))
    dropped = sum(s["dropped_rows"] for s in summary["stages"])
    assert summary["stages"][0]["rows"] - dropped == len(out)
    assert summary["final"]["rows"] == len(out)


def test_filter_domain_writes_outputs(tiny_cfg, tmp_paths):
    write_parquet(_core_plus_noise(), tmp_paths.reviews, REVIEWS)
    domain = tiny_cfg.domain.model_copy(
        update={"filter": _cfg(max_users=5, sample_seed=0)}
    )
    summary = filter_domain(domain, tmp_paths)
    out = read_parquet(tmp_paths.reviews_filtered, REVIEWS)
    assert len(out) == summary["final"]["rows"] == 25
    assert read_json(tmp_paths.filter_summary)["domain"] == "all_beauty"


def test_most_active_sampling_is_deterministic_and_picks_top_users():
    pairs = [("big", f"i{k}") for k in range(6)] + [("mid", f"i{k}") for k in range(4)]
    pairs += [("a_small", "i0"), ("b_small", "i0")]
    df = _frame(pairs)
    out = sample_users(df, 2, seed=0, strategy="most_active")
    assert set(out["user_id"]) == {"big", "mid"}
    assert set(sample_users(df, 2, seed=99, strategy="most_active")["user_id"]) == {"big", "mid"}


def test_most_active_ties_break_by_user_id():
    df = _frame([("u2", "i1"), ("u1", "i1"), ("u3", "i1")])
    out = sample_users(df, 2, seed=0, strategy="most_active")
    assert set(out["user_id"]) == {"u1", "u2"}


def test_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError):
        sample_users(_core_plus_noise(), 2, seed=0, strategy="nope")


def test_relaxed_item_minimum_after_sampling_keeps_users():
    df = _core_plus_noise()
    strict, _ = filter_reviews(df, _cfg(max_users=3))
    relaxed, summary = filter_reviews(
        df, _cfg(max_users=3, min_item_interactions_after_sample=3)
    )
    assert strict.empty
    assert relaxed["user_id"].nunique() == 3  # items keep 3 users >= 3
    assert summary["min_item_interactions_after_sample"] == 3
