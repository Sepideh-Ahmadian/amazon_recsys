"""Interaction filtering: iterative k-core, seeded user sampling, then k-core again."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from amazon_recsys.config import DomainCfg, FilterCfg
from amazon_recsys.data.schemas import REVIEWS
from amazon_recsys.utils.io import read_parquet, write_json, write_parquet
from amazon_recsys.utils.paths import Paths


def _meets_minimums(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """One pass: keep rows whose user and item meet the minimum distinct-neighbour counts."""
    user_n = df.groupby("user_id")["item_id"].transform("nunique")
    item_n = df.groupby("item_id")["user_id"].transform("nunique")
    return df[(user_n >= min_user) & (item_n >= min_item)]


def k_core(df: pd.DataFrame, min_user: int, min_item: int, iterate: bool = True) -> pd.DataFrame:
    """Drop users/items below the minimums, repeating to a fixed point when `iterate`.

    Counts are distinct items per user and distinct users per item, so repeated reviews of the
    same item do not inflate them. All rows of surviving pairs are kept.
    """
    current = df
    while True:
        filtered = _meets_minimums(current, min_user, min_item)
        if not iterate or len(filtered) == len(current):
            return filtered
        current = filtered


def sample_users(
    df: pd.DataFrame, max_users: int | None, seed: int, strategy: str = "random"
) -> pd.DataFrame:
    """Keep all rows of `max_users` users; no-op if null or not smaller than the user count.

    `random`: seeded uniform choice. `most_active`: users with the most distinct items, ties
    broken by ascending user id (deterministic; the seed is unused).
    """
    users = np.sort(df["user_id"].unique())
    if max_users is None or max_users >= len(users):
        return df
    if strategy == "random":
        chosen = np.random.default_rng(seed).choice(users, size=max_users, replace=False)
    elif strategy == "most_active":
        activity = df.groupby("user_id")["item_id"].nunique().reset_index(name="n")
        ranked = activity.sort_values(["n", "user_id"], ascending=[False, True])
        chosen = ranked["user_id"].to_numpy()[:max_users]
    else:
        raise ValueError(f"Unknown sample strategy: {strategy!r}")
    return df[df["user_id"].isin(set(chosen))]


def _stage(name: str, df: pd.DataFrame, previous_rows: int | None) -> dict[str, Any]:
    return {
        "stage": name,
        "rows": len(df),
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),
        "dropped_rows": 0 if previous_rows is None else previous_rows - len(df),
    }


def filter_reviews(df: pd.DataFrame, cfg: FilterCfg) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply k-core, then the optional `max_users` sample, then k-core again."""
    lo_u, lo_i = cfg.min_user_interactions, cfg.min_item_interactions
    stages = [_stage("input", df, None)]

    out = k_core(df, lo_u, lo_i, iterate=cfg.k_core)
    stages.append(_stage("k_core", out, stages[-1]["rows"]))

    if cfg.max_users is not None:
        out = sample_users(out, cfg.max_users, cfg.sample_seed, cfg.sample_strategy)
        stages.append(_stage("sample", out, stages[-1]["rows"]))
        lo_i_after = cfg.min_item_interactions_after_sample or lo_i
        out = k_core(out, lo_u, lo_i_after, iterate=cfg.k_core)
        stages.append(_stage("k_core_2", out, stages[-1]["rows"]))

    summary = {
        "min_user_interactions": lo_u,
        "min_item_interactions": lo_i,
        "k_core": cfg.k_core,
        "max_users": cfg.max_users,
        "sample_seed": cfg.sample_seed,
        "sample_strategy": cfg.sample_strategy,
        "min_item_interactions_after_sample": cfg.min_item_interactions_after_sample,
        "stages": stages,
        "final": {k: stages[-1][k] for k in ("rows", "users", "items")},
    }
    return out.reset_index(drop=True), summary


def filter_domain(cfg: DomainCfg, paths: Paths) -> dict[str, Any]:
    """Read reviews.parquet, filter it, write reviews_filtered.parquet and a summary JSON."""
    reviews = read_parquet(paths.reviews, REVIEWS)
    filtered, summary = filter_reviews(reviews, cfg.filter)
    write_parquet(filtered, paths.reviews_filtered, REVIEWS)
    write_json({"domain": cfg.name, **summary}, paths.filter_summary)
    return summary
