import pandas as pd
import pytest

from amazon_recsys.data.interactions import build_interactions, interactions_domain
from amazon_recsys.data.schemas import INTERACTIONS, REVIEWS
from amazon_recsys.utils.io import read_parquet, write_parquet


def _reviews(n=4):
    return pd.DataFrame(
        {
            "review_id": [f"r{k}" for k in range(n)],
            "domain": "d",
            "user_id": [f"u{k}" for k in range(n)],
            "item_id": [f"i{k}" for k in range(n)],
            "rating": [5.0, 4.0, 3.0, 2.0][:n],
            "title": None,
            "text": "x",
            "timestamp": pd.array([1_600_000_000_000 + k for k in range(n)], dtype="int64"),
        }
    )


def _enrichment():
    return pd.DataFrame(
        {
            "review_id": ["r0", "r1", "r2"],
            "domain": "d",
            "status": ["ok", "parse_error", "ok"],
            "overall_sentiment": [0.8, None, -0.5],
            "extractor_model": "m",
            "prompt_version": "v1",
            "prompt_hash": "h",
        }
    )


def test_left_join_keeps_reviews_without_enrichment():
    out = build_interactions(_reviews(), _enrichment())
    assert len(out) == 4
    assert out.loc[out.review_id == "r0", "overall_sentiment"].iloc[0] == 0.8
    # r1 failed parsing, r3 was never enriched: both null, not dropped
    assert out.loc[out.review_id == "r1", "overall_sentiment"].isna().all()
    assert out.loc[out.review_id == "r3", "overall_sentiment"].isna().all()


def test_judge_overall_and_verdict_joined_and_null_when_missing():
    judgements = pd.DataFrame({"review_id": ["r0"], "overall": [0.9], "verdict": ["pass"]})
    out = build_interactions(_reviews(), None, judgements).set_index("review_id")
    assert out.loc["r0", "judge_overall"] == 0.9 and out.loc["r0", "judge_verdict"] == "pass"
    assert out["judge_overall"].isna().sum() == 3
    assert out["judge_verdict"].isna().sum() == 3


def test_no_enrichment_at_all_gives_all_nulls_and_valid_schema():
    out = build_interactions(_reviews())
    assert out["overall_sentiment"].isna().all() and out["judge_overall"].isna().all()
    INTERACTIONS.validate(out)


def test_row_count_equals_review_count_and_schema_validates():
    reviews = _reviews()
    out = build_interactions(reviews, _enrichment())
    assert len(out) == len(reviews)
    INTERACTIONS.validate(out)


def test_interactions_domain_roundtrip(tiny_cfg, tmp_paths):
    write_parquet(_reviews(), tmp_paths.reviews_filtered, REVIEWS)
    out = interactions_domain(tiny_cfg.domain, tmp_paths)
    assert read_parquet(tmp_paths.processed("interactions"), INTERACTIONS).equals(out)


def test_dataset_stats_hand_computed():
    from amazon_recsys.data.interactions import dataset_stats

    df = build_interactions(_reviews(), _enrichment(), pd.DataFrame(
        {"review_id": ["r0"], "overall": [0.9], "verdict": ["pass"]}
    ))
    stats = dataset_stats(df)
    assert (stats["interactions"], stats["users"], stats["items"]) == (4, 4, 4)
    assert stats["density"] == pytest.approx(4 / 16)
    assert stats["interactions_per_user"] == {"mean": 1.0, "min": 1, "median": 1.0, "max": 1}
    assert stats["rating_counts"] == {"2.0": 1, "3.0": 1, "4.0": 1, "5.0": 1}
    assert stats["with_sentiment"] == 2 and stats["with_judgement"] == 1
    assert dataset_stats(df.iloc[0:0])["interactions"] == 0  # empty frame does not crash
