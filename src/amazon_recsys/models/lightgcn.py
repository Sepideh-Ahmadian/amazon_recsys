"""LightGCN: parameter-free propagation over the (unweighted) user-item graph."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from amazon_recsys.models.base import Recommender
from amazon_recsys.models.registry import register_model


def sym_norm(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Per-edge weight 1 / sqrt(deg(src) * deg(dst))."""
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
    inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    return inv_sqrt[edge_index[0]] * inv_sqrt[edge_index[1]]


@register_model("lightgcn")
class LightGCN(Recommender):
    requires_graph = "bipartite"

    def __init__(self, graph: Any, dim: int, layers: int) -> None:
        super().__init__(graph.num_users, graph.num_items, graph)
        self.layers = layers
        self.embedding = nn.Embedding(graph.num_users + graph.num_items, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.register_buffer("edge_index", graph.edge_index.clone(), persistent=False)
        self.register_buffer(
            "norm", sym_norm(graph.edge_index, self.embedding.num_embeddings), persistent=False
        )

    @classmethod
    def from_config(cls, cfg: Any, graph: Any) -> LightGCN:
        return cls(graph, cfg.model.dim, cfg.model.layers)

    def forward(self, graph: Any = None) -> tuple[Tensor, Tensor]:
        x = self.embedding.weight
        src, dst = self.edge_index
        outputs = [x]
        for _ in range(self.layers):
            messages = x[src] * self.norm.unsqueeze(-1)
            x = torch.zeros_like(x).index_add_(0, dst, messages)
            outputs.append(x)
        final = torch.stack(outputs).mean(dim=0)
        return final[: self.num_users], final[self.num_users :]
