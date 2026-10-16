"""Weighted user-item graph builder (weighting: none | rating | rating_sentiment)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from amazon_recsys.data.interactions import build_index
from amazon_recsys.graphs.base import BipartiteGraph, GraphBuilder
from amazon_recsys.graphs.registry import register_builder

WEIGHTINGS = ("none", "rating", "rating_sentiment")
MAX_RATING = 5.0


def usable_sentiment(train: pd.DataFrame, graph_cfg: Any) -> np.ndarray:
    """Which edges may use the extractor's sentiment.

    Without `min_judge_score` any non-null sentiment is used. With it, the review must also have a
    judge score >= the threshold and a verdict that is not missing, nor `uncertain` unless
    `accept_uncertain`. The edge itself is always kept; only the enrichment is filtered.
    """
    has_sentiment = train["overall_sentiment"].notna().to_numpy()
    if graph_cfg.min_judge_score is None:
        return has_sentiment
    overall = train["judge_overall"]
    verdict = train["judge_verdict"]
    passes_score = (overall >= graph_cfg.min_judge_score).fillna(False).to_numpy()
    accepted = {"pass", "fail"} | ({"uncertain"} if graph_cfg.accept_uncertain else set())
    passes_verdict = verdict.isin(accepted).to_numpy()
    return has_sentiment & passes_score & passes_verdict


def edge_weights(train: pd.DataFrame, graph_cfg: Any) -> np.ndarray:
    """One weight in [0, 1] per train interaction."""
    mode = graph_cfg.weighting
    if mode not in WEIGHTINGS:
        raise ValueError(f"Unknown weighting {mode!r}; choose from {WEIGHTINGS}")
    n = len(train)
    if mode == "none":
        return np.ones(n, dtype=np.float32)
    rating_w = train["rating"].to_numpy(dtype=np.float64) / MAX_RATING
    if mode == "rating":
        return rating_w.astype(np.float32)
    alpha = graph_cfg.alpha
    sentiment_w = (train["overall_sentiment"].fillna(0.0).to_numpy(dtype=np.float64) + 1.0) / 2.0
    mixed = alpha * rating_w + (1.0 - alpha) * sentiment_w
    return np.where(usable_sentiment(train, graph_cfg), mixed, rating_w).astype(np.float32)


@register_builder("bipartite")
class BipartiteBuilder(GraphBuilder):
    kind = "bipartite"

    def build(
        self,
        train: pd.DataFrame,
        cfg: Any,
        user_index: dict[str, int] | None = None,
        item_index: dict[str, int] | None = None,
        **_: Any,  # extra inputs (e.g. aspects) used only by other builders
    ) -> BipartiteGraph:
        if user_index is None or item_index is None:
            user_index, item_index = build_index(train)
        n_users = len(user_index)

        users = train["user_id"].map(user_index).to_numpy(dtype=np.int64)
        items = train["item_id"].map(item_index).to_numpy(dtype=np.int64) + n_users
        weights = torch.from_numpy(edge_weights(train, cfg.graph))

        forward = torch.from_numpy(np.stack([users, items]))
        edge_index = torch.cat([forward, forward.flip(0)], dim=1)
        return BipartiteGraph(
            num_users=n_users,
            num_items=len(item_index),
            user_index=dict(user_index),
            item_index=dict(item_index),
            edge_index=edge_index,
            edge_weight=torch.cat([weights, weights]),
        )
