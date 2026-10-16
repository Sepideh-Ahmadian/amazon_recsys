"""Uniform negative sampling that excludes each user's train positives."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor

_MAX_ROUNDS = 100


class UniformNegativeSampler:
    """Draw items uniformly from all items the user has not interacted with in train."""

    def __init__(self, users: np.ndarray, items: np.ndarray, n_items: int, seed: int) -> None:
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        self.n_items = int(n_items)
        self._keys = np.unique(users * self.n_items + items)  # sorted (user, item) keys
        per_user = np.bincount(self._keys // self.n_items)
        if per_user.size and per_user.max() >= self.n_items:
            raise ValueError("A user has interacted with every item; no negatives exist")
        self._rng = np.random.default_rng(seed)

    def _is_positive(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        keys = users * self.n_items + items
        idx = np.searchsorted(self._keys, keys)
        idx = np.minimum(idx, len(self._keys) - 1)
        return self._keys[idx] == keys

    def sample(self, users: Tensor | np.ndarray, n_neg: int = 1) -> Tensor:
        """Return an int64 tensor of shape [len(users), n_neg]."""
        u = np.asarray(users, dtype=np.int64)
        flat_users = np.repeat(u, n_neg)
        neg = self._rng.integers(0, self.n_items, size=flat_users.shape[0])
        for _ in range(_MAX_ROUNDS):
            bad = self._is_positive(flat_users, neg)
            if not bad.any():
                break
            neg[bad] = self._rng.integers(0, self.n_items, size=int(bad.sum()))
        else:  # pragma: no cover - only with pathological density
            raise RuntimeError("Negative sampling did not converge")
        return torch.from_numpy(neg.reshape(len(u), n_neg))


def epoch_batches(
    users: np.ndarray,
    items: np.ndarray,
    sampler: UniformNegativeSampler,
    batch_size: int,
    n_neg: int = 1,
    seed: int | None = None,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Yield shuffled (user, pos_item, neg_items[B, n_neg]) batches; each train row once."""
    n = len(users)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    users_t = torch.as_tensor(np.asarray(users, dtype=np.int64))
    items_t = torch.as_tensor(np.asarray(items, dtype=np.int64))
    for start in range(0, n, batch_size):
        idx = torch.as_tensor(order[start : start + batch_size])
        u, p = users_t[idx], items_t[idx]
        yield u, p, sampler.sample(u, n_neg)
