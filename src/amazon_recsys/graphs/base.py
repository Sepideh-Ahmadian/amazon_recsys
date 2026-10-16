"""The graph contract shared by every builder and model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import pandas as pd
from torch import Tensor


class GraphData(Protocol):
    kind: str  # "bipartite", "knowledge", later "hypergraph"
    num_users: int
    num_items: int
    user_index: dict[str, int]
    item_index: dict[str, int]


@dataclass
class BipartiteGraph:
    """User-item graph. Nodes: users 0..U-1 then items U..U+I-1; edges in both directions."""

    num_users: int
    num_items: int
    user_index: dict[str, int]
    item_index: dict[str, int]
    edge_index: Tensor  # [2, E]
    edge_weight: Tensor | None  # [E], aligned with edge_index
    kind: ClassVar[str] = "bipartite"

    @property
    def num_nodes(self) -> int:
        return self.num_users + self.num_items


@dataclass
class KnowledgeGraph:
    """Triples plus the train user-item graph the GNN runs on."""

    num_users: int
    num_items: int
    user_index: dict[str, int]
    item_index: dict[str, int]
    triples: pd.DataFrame  # head, relation, tail (string ids)
    entity_index: dict[str, int]
    relation_index: dict[str, int]
    interaction_graph: BipartiteGraph
    entity_embeddings: Tensor | None = None  # [num_entities, dim], set by `prepare`
    kind: ClassVar[str] = "knowledge"

    def prepare(self, cfg: Any, paths: Any) -> None:
        """Train (or load cached) KG embeddings; user then item entities come first."""
        from amazon_recsys.graphs.kg_embeddings import get_kg_embeddings

        self.entity_embeddings = get_kg_embeddings(self, cfg, paths)


class GraphBuilder(ABC):
    kind: str

    @abstractmethod
    def build(
        self,
        train: pd.DataFrame,
        cfg: Any,
        user_index: dict[str, int] | None = None,
        item_index: dict[str, int] | None = None,
        **extras: Any,
    ) -> GraphData:
        """Build from the train split only.

        `user_index` / `item_index` may span all splits so cold val/test entities still get a
        node; edges and weights must come from `train` alone. `extras` carries optional
        tables (e.g. `aspects`, `aspect_judgements`) that a builder may use or ignore.
        """
