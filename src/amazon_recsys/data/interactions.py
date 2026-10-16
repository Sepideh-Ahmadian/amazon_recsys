"""Join filtered reviews with enrichment and judgements into the interactions table."""

from __future__ import annotations

from typing import Any

import pandas as pd

from amazon_recsys.config import DomainCfg
from amazon_recsys.data.schemas import ENRICHMENT, INTERACTIONS, JUDGEMENTS, REVIEWS
from amazon_recsys.utils.io import read_parquet, write_parquet
from amazon_recsys.utils.paths import Paths


def build_interactions(
    reviews: pd.DataFrame,
    enrichment: pd.DataFrame | None = None,
    judgements: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left join: every review is kept; missing enrichment or judgement yields nulls.

    Sentiment is taken only from enrichment rows with status "ok"; `judge_overall` is the
    judge's calibrated overall score (null when the judge did not score the review).
    """
    out = reviews[
        ["domain", "user_id", "item_id", "rating", "timestamp", "review_id"]
    ].copy()

    sentiment = pd.Series(float("nan"), index=out.index, dtype="float64")
    if enrichment is not None:
        ok = enrichment[enrichment["status"] == "ok"]
        mapping = ok.drop_duplicates("review_id").set_index("review_id")["overall_sentiment"]
        sentiment = out["review_id"].map(mapping).astype("float64")
    out["overall_sentiment"] = sentiment

    overall = pd.Series(float("nan"), index=out.index, dtype="float64")
    if judgements is not None:
        mapping = judgements.drop_duplicates("review_id").set_index("review_id")["overall"]
        overall = out["review_id"].map(mapping).astype("float64")
    out["judge_overall"] = overall

    verdict = pd.Series(None, index=out.index, dtype="object")
    if judgements is not None:
        mapping = judgements.drop_duplicates("review_id").set_index("review_id")["verdict"]
        verdict = out["review_id"].map(mapping).astype("object")
    out["judge_verdict"] = verdict

    return out.reset_index(drop=True)


def interactions_domain(cfg: DomainCfg, paths: Paths) -> pd.DataFrame:
    """Read reviews_filtered (+ enrichment/judgements when present) and write interactions."""
    reviews = read_parquet(paths.reviews_filtered, REVIEWS)
    enrichment = judgements = None
    if paths.processed("enrichment").exists():
        enrichment = read_parquet(paths.processed("enrichment"), ENRICHMENT)
    if paths.processed("judgements").exists():
        judgements = read_parquet(paths.processed("judgements"), JUDGEMENTS)
    interactions = build_interactions(reviews, enrichment, judgements)
    write_parquet(interactions, paths.processed("interactions"), INTERACTIONS)
    return interactions


def build_index(*frames: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    """Deterministic (user_index, item_index) over every user/item in the given frames.

    Ids are sorted, so the mapping is stable across runs. Pass train, val and test together so
    cold val/test items still get an index and can be ranked.
    """
    users = sorted({u for f in frames for u in f["user_id"].unique()})
    items = sorted({i for f in frames for i in f["item_id"].unique()})
    return {u: n for n, u in enumerate(users)}, {i: n for n, i in enumerate(items)}


def dataset_stats(interactions: pd.DataFrame) -> dict[str, Any]:
    """Headline statistics of an interactions frame (for reports and sanity checks)."""
    n_users = int(interactions["user_id"].nunique())
    n_items = int(interactions["item_id"].nunique())
    per_user = interactions.groupby("user_id").size()
    per_item = interactions.groupby("item_id").size()
    return {
        "interactions": len(interactions),
        "users": n_users,
        "items": n_items,
        "density": len(interactions) / (n_users * n_items) if n_users and n_items else 0.0,
        "interactions_per_user": {
            "mean": float(per_user.mean()) if len(per_user) else 0.0,
            "min": int(per_user.min()) if len(per_user) else 0,
            "median": float(per_user.median()) if len(per_user) else 0.0,
            "max": int(per_user.max()) if len(per_user) else 0,
        },
        "interactions_per_item": {
            "mean": float(per_item.mean()) if len(per_item) else 0.0,
            "median": float(per_item.median()) if len(per_item) else 0.0,
            "max": int(per_item.max()) if len(per_item) else 0,
        },
        "rating_counts": {
            str(k): int(v) for k, v in interactions["rating"].value_counts().sort_index().items()
        },
        "with_sentiment": int(interactions["overall_sentiment"].notna().sum()),
        "with_judgement": int(interactions["judge_overall"].notna().sum()),
    }
