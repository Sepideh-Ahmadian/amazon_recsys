"""Full-ranking evaluation. Every model is evaluated through this one code path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from amazon_recsys.evaluation.metrics import compute_metrics

_SPLITS = ("val", "test")


def _group_by_user(
    df: pd.DataFrame, user_index: Mapping[str, int], item_index: Mapping[str, int]
) -> dict[int, np.ndarray]:
    users = df["user_id"].map(user_index).to_numpy()
    items = df["item_id"].map(item_index).to_numpy()
    grouped: dict[int, list[int]] = {}
    for u, i in zip(users.tolist(), items.tolist(), strict=True):
        grouped.setdefault(u, []).append(i)
    return {u: np.asarray(v, dtype=np.int64) for u, v in grouped.items()}


class Evaluator:
    """Ranks all domain items per user, excluding the user's already-seen positives.

    Seen positives are the train items (plus val items when evaluating on test).
    """

    def __init__(
        self,
        splits: Mapping[str, pd.DataFrame],
        user_index: Mapping[str, int],
        item_index: Mapping[str, int],
        ks: Sequence[int],
        batch_size: int = 1024,
    ) -> None:
        self.ks = list(ks)
        self.batch_size = batch_size
        self.num_items = len(item_index)
        self._train = _group_by_user(splits["train"], user_index, item_index)
        self._targets = {
            name: _group_by_user(splits[name], user_index, item_index) for name in _SPLITS
        }

    def _excluded(self, user: int, split: str) -> np.ndarray:
        parts = [self._train.get(user, np.empty(0, dtype=np.int64))]
        if split == "test":
            parts.append(self._targets["val"].get(user, np.empty(0, dtype=np.int64)))
        return np.unique(np.concatenate(parts))

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, split: str = "val") -> dict[str, float]:
        """Mean metrics over users that have a target item in `split`."""
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
        targets = self._targets[split]
        users = sorted(targets)
        totals: dict[str, float] = {}
        was_training = model.training
        model.eval()
        try:
            device = next(model.parameters()).device
            for start in range(0, len(users), self.batch_size):
                batch = users[start : start + self.batch_size]
                scores = model.score_all(torch.as_tensor(batch, device=device)).float().cpu()
                for row, user in zip(scores, batch, strict=True):
                    excluded = self._excluded(user, split)
                    row = row.clone()
                    row[torch.as_tensor(excluded, dtype=torch.long)] = float("-inf")
                    # Stable descending sort: ties are broken by ascending item index.
                    order = torch.sort(row, descending=True, stable=True).indices
                    n_candidates = self.num_items - len(excluded)
                    ranked = order[:n_candidates].tolist()
                    for name, value in compute_metrics(ranked, targets[user], self.ks).items():
                        totals[name] = totals.get(name, 0.0) + value
        finally:
            model.train(was_training)

        n = len(users)
        metrics = {name: total / n for name, total in totals.items()} if n else {}
        metrics["n_users"] = float(n)
        return metrics
