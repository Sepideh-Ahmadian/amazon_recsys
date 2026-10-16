"""Ranking metrics over a single ranked list (best first) and a set of relevant items."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def _hits(ranked: Sequence[int], relevant: Collection[int], k: int) -> int:
    rel = set(relevant)
    return sum(1 for item in ranked[:k] if item in rel)


def recall_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return _hits(ranked, rel, k) / len(rel)


def precision_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return _hits(ranked, relevant, k) / k


def hit_rate_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    return 1.0 if _hits(ranked, relevant, k) > 0 else 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    rel = set(relevant)
    if not rel or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(pos + 2) for pos, item in enumerate(ranked[:k]) if item in rel)
    ideal = sum(1.0 / math.log2(pos + 2) for pos in range(min(len(rel), k)))
    return dcg / ideal


def mrr(ranked: Sequence[int], relevant: Collection[int]) -> float:
    """Reciprocal rank of the first relevant item in the whole list (0 if none)."""
    rel = set(relevant)
    for pos, item in enumerate(ranked):
        if item in rel:
            return 1.0 / (pos + 1)
    return 0.0


def compute_metrics(
    ranked: Sequence[int], relevant: Collection[int], ks: Sequence[int]
) -> dict[str, float]:
    """All metrics for one user, keyed like `ndcg@10`; `mrr` has no K."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(ranked, relevant, k)
        out[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
    out["mrr"] = mrr(ranked, relevant)
    return out
