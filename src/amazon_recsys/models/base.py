"""The Recommender interface every model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class Recommender(nn.Module, ABC):
    """Produces user/item embeddings; scores are dot products of those embeddings.

    `requires_graph` is None (no graph needed), "bipartite" or "knowledge". `graph` is the
    structure passed at construction; for graph-free models it only needs `num_users` and
    `num_items`.
    """

    requires_graph: str | None = None

    def __init__(self, num_users: int, num_items: int, graph: Any = None) -> None:
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.graph = graph

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Any, graph: Any) -> Recommender:
        """Build the model from a resolved Config and the graph/size provider."""

    @abstractmethod
    def forward(self, graph: Any = None) -> tuple[Tensor, Tensor]:
        """Return (user_emb [U, d], item_emb [I, d]); `graph` defaults to the stored one."""

    def score(self, user_idx: Tensor, item_idx: Tensor, graph: Any = None) -> Tensor:
        """Score aligned (user, item) pairs; returns shape [len(user_idx)]."""
        user_emb, item_emb = self(graph)
        return (user_emb[user_idx] * item_emb[item_idx]).sum(dim=-1)

    def score_all(self, user_idx: Tensor, graph: Any = None) -> Tensor:
        """Score every item for each given user; returns shape [len(user_idx), num_items]."""
        user_emb, item_emb = self(graph)
        return user_emb[user_idx] @ item_emb.T

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_users": self.num_users,
                "num_items": self.num_items,
                "class": type(self).__name__,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload["num_users"], payload["num_items"]) != (self.num_users, self.num_items):
            raise ValueError(
                f"Checkpoint is for {payload['num_users']} users / {payload['num_items']} items, "
                f"model has {self.num_users} / {self.num_items}"
            )
        self.load_state_dict(payload["state_dict"])
