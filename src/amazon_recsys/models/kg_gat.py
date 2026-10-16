"""KG-GAT: the same GAT as `gat`, with user/item embeddings initialised from KG embeddings."""

from __future__ import annotations

from typing import Any

import torch

from amazon_recsys.models.gat import GATRecommender
from amazon_recsys.models.registry import register_model


@register_model("kg_gat")
class KGGATRecommender(GATRecommender):
    requires_graph = "knowledge"

    def __init__(
        self,
        kg: Any,
        dim: int,
        layers: int,
        heads: int,
        dropout: float,
        use_edge_weight: bool,
        freeze: bool,
    ) -> None:
        super().__init__(kg.interaction_graph, dim, layers, heads, dropout, use_edge_weight)
        if kg.entity_embeddings is None:
            raise ValueError("KG embeddings are missing; call graph.prepare(cfg, paths) first")
        if kg.entity_embeddings.shape[1] != dim:
            raise ValueError(f"KG embedding dim {kg.entity_embeddings.shape[1]} != model dim {dim}")
        n_nodes = kg.num_users + kg.num_items  # entities are ordered users, items, aspects
        with torch.no_grad():
            self.embedding.weight.copy_(kg.entity_embeddings[:n_nodes])
        self.embedding.weight.requires_grad_(not freeze)
        self.freeze = freeze
        self.graph = kg

    @classmethod
    def from_config(cls, cfg: Any, graph: Any) -> KGGATRecommender:
        m = cfg.model
        return cls(graph, m.dim, m.layers, m.heads, m.dropout, m.use_edge_weight, m.freeze)
