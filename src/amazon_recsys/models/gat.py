"""GAT recommender on the user-item graph, plus the reusable `GATEncoder`."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATConv

from amazon_recsys.models.base import Recommender
from amazon_recsys.models.registry import register_model


class GATEncoder(nn.Module):
    """Stack of GATConv layers with residual connections.

    `edge_dim=1` makes layers consume a scalar edge weight as `edge_attr`; `None` ignores weights.
    Shared by `gat` and `kg_gat` so the architecture is defined once.
    """

    def __init__(
        self, dim: int, layers: int, heads: int, dropout: float, edge_dim: int | None
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.dropout = dropout
        self.convs = nn.ModuleList(
            GATConv(dim, dim // heads, heads=heads, concat=True, dropout=dropout, edge_dim=edge_dim)
            for _ in range(layers)
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None) -> Tensor:
        for conv in self.convs:
            update = F.elu(conv(x, edge_index, edge_attr))
            x = x + F.dropout(update, p=self.dropout, training=self.training)
        return x


@register_model("gat")
class GATRecommender(Recommender):
    requires_graph = "bipartite"

    def __init__(
        self,
        graph: Any,
        dim: int,
        layers: int,
        heads: int,
        dropout: float,
        use_edge_weight: bool,
    ) -> None:
        super().__init__(graph.num_users, graph.num_items, graph)
        self.embedding = nn.Embedding(graph.num_users + graph.num_items, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.use_edge_weight = use_edge_weight and graph.edge_weight is not None
        self.encoder = GATEncoder(dim, layers, heads, dropout, 1 if self.use_edge_weight else None)
        self.register_buffer("edge_index", graph.edge_index.clone(), persistent=False)
        weight = graph.edge_weight.clone().unsqueeze(-1) if self.use_edge_weight else None
        self.register_buffer("edge_attr", weight, persistent=False)

    @classmethod
    def from_config(cls, cfg: Any, graph: Any) -> GATRecommender:
        m = cfg.model
        return cls(graph, m.dim, m.layers, m.heads, m.dropout, m.use_edge_weight)

    def forward(self, graph: Any = None) -> tuple[Tensor, Tensor]:
        x = self.encoder(self.embedding.weight, self.edge_index, self.edge_attr)
        return x[: self.num_users], x[self.num_users :]


__all__ = ["GATEncoder", "GATRecommender"]
