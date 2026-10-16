"""Pairwise/pointwise recommendation losses. All take (pos_scores, neg_scores) and return a scalar.

`pos_scores` has shape [B]; `neg_scores` has shape [B] or [B, n_neg].
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

LossFn = Callable[[Tensor, Tensor], Tensor]


def _align(pos: Tensor, neg: Tensor) -> tuple[Tensor, Tensor]:
    if neg.dim() == 2 and pos.dim() == 1:
        pos = pos.unsqueeze(1)
    return pos, neg


def bpr_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    """Bayesian personalised ranking: -mean(log sigmoid(pos - neg))."""
    pos, neg = _align(pos_scores, neg_scores)
    return -F.logsigmoid(pos - neg).mean()


def bce_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    """Binary cross-entropy on logits: positives -> 1, negatives -> 0, classes weighted equally."""
    pos_term = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    neg_term = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    return 0.5 * (pos_term + neg_term)


LOSSES: dict[str, LossFn] = {"bpr": bpr_loss, "bce": bce_loss}


def get_loss(name: str) -> LossFn:
    try:
        return LOSSES[name]
    except KeyError:
        raise KeyError(f"Unknown loss {name!r}. Available: {', '.join(sorted(LOSSES))}") from None
