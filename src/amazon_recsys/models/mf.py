"""BPR matrix factorisation: one learned embedding per user and item, no graph."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from amazon_recsys.models.base import Recommender
from amazon_recsys.models.registry import register_model


@register_model("mf")
class MatrixFactorisation(Recommender):
    requires_graph = None

    def __init__(self, num_users: int, num_items: int, dim: int, graph: Any = None) -> None:
        super().__init__(num_users, num_items, graph)
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    @classmethod
    def from_config(cls, cfg: Any, graph: Any) -> MatrixFactorisation:
        return cls(graph.num_users, graph.num_items, cfg.model.dim, graph)

    def forward(self, graph: Any = None) -> tuple[Tensor, Tensor]:
        return self.user_emb.weight, self.item_emb.weight


__all__ = ["MatrixFactorisation"]
