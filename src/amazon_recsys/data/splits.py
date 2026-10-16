"""Leave-one-out-by-time splits with a leakage check and a manifest."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from amazon_recsys.config import SplitCfg
from amazon_recsys.data.schemas import INTERACTIONS
from amazon_recsys.utils.hashing import hash_text
from amazon_recsys.utils.io import read_parquet, write_json, write_parquet
from amazon_recsys.utils.paths import Paths

SPLIT_NAMES = ("train", "val", "test")
STRATEGY = "leave_one_out_time"


class LeakageError(AssertionError):
    """Raised when a (user, item) pair appears in more than one split."""


def _tie_break_rank(review_ids: pd.Series, seed: int) -> pd.Series:
    """Seeded rank per review, independent of row order (permutation of sorted review ids)."""
    order = np.argsort(review_ids.to_numpy(), kind="stable")
    perm = np.random.default_rng(seed).permutation(len(review_ids))
    rank = np.empty(len(review_ids), dtype=np.int64)
    rank[order] = perm
    return pd.Series(rank, index=review_ids.index)


def _sort_chronologically(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.assign(_tie=_tie_break_rank(df["review_id"], seed))
    out = out.sort_values(["user_id", "timestamp", "_tie"], kind="stable")
    return out.drop(columns="_tie").reset_index(drop=True)


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """No (user, item) pair may appear in more than one split, or twice in one."""
    seen: set[tuple[str, str]] = set()
    for name in SPLIT_NAMES:
        pairs = list(zip(splits[name]["user_id"], splits[name]["item_id"], strict=True))
        pair_set = set(pairs)
        if len(pair_set) != len(pairs):
            raise LeakageError(f"Duplicate (user, item) pairs inside split {name!r}")
        overlap = seen & pair_set
        if overlap:
            raise LeakageError(f"{len(overlap)} pairs from {name!r} already in an earlier split")
        seen |= pair_set


def make_splits(
    interactions: pd.DataFrame, cfg: SplitCfg
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Per user: last interaction -> test, second to last -> val, the rest -> train."""
    if cfg.strategy != STRATEGY:
        raise ValueError(f"Unknown split strategy {cfg.strategy!r}; supported: {STRATEGY!r}")

    ordered = _sort_chronologically(interactions, cfg.seed)
    deduped = ordered.drop_duplicates(["user_id", "item_id"], keep="first").reset_index(drop=True)
    n_repeats = len(ordered) - len(deduped)

    size = deduped.groupby("user_id")["item_id"].transform("size")
    pos_from_end = deduped.groupby("user_id").cumcount(ascending=False)
    eligible = size >= cfg.min_user_interactions

    is_test = eligible & (pos_from_end == 0)
    is_val = eligible & (pos_from_end == 1)
    splits = {
        "train": deduped[~(is_test | is_val)].reset_index(drop=True),
        "val": deduped[is_val].reset_index(drop=True),
        "test": deduped[is_test].reset_index(drop=True),
    }
    assert_no_leakage(splits)

    train_users = set(splits["train"]["user_id"])
    train_items = set(splits["train"]["item_id"])
    manifest = {
        "strategy": cfg.strategy,
        "seed": cfg.seed,
        "min_user_interactions": cfg.min_user_interactions,
        "input_rows": len(interactions),
        "input_hash": hash_text("\n".join(sorted(interactions["review_id"]))),
        "dropped_repeat_pairs": n_repeats,
        "counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "users": {
            "total": int(deduped["user_id"].nunique()),
            "in_val_test": int(splits["test"]["user_id"].nunique()),
            "train_only": int((~eligible).groupby(deduped["user_id"]).first().sum()),
        },
        "unseen_in_train": {
            "val_items": int((~splits["val"]["item_id"].isin(train_items)).sum()),
            "test_items": int((~splits["test"]["item_id"].isin(train_items)).sum()),
            "val_test_users": int(
                (~pd.concat([splits["val"], splits["test"]])["user_id"].isin(train_users)).sum()
            ),
        },
        "leakage_check": {"passed": True, "violations": 0},
    }
    return splits, manifest


def split_domain(cfg: SplitCfg, paths: Paths) -> dict[str, Any]:
    """Read interactions.parquet, write splits/{train,val,test}.parquet and manifest.json."""
    interactions = read_parquet(paths.processed("interactions"), INTERACTIONS)
    splits, manifest = make_splits(interactions, cfg)
    for name in SPLIT_NAMES:
        write_parquet(splits[name], paths.split(name), INTERACTIONS)
    write_json(manifest, paths.split_manifest)
    return manifest
