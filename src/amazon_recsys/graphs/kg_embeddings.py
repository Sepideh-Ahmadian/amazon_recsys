"""Train PyKEEN embeddings for a knowledge graph, cached on disk by config hash."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from amazon_recsys.graphs.base import KnowledgeGraph
from amazon_recsys.utils.hashing import hash_obj, hash_text
from amazon_recsys.utils.io import load_checkpoint, save_checkpoint

TrainFn = Callable[[KnowledgeGraph, Any, int], Tensor]


def kg_config_hash(kg: KnowledgeGraph, embedding_cfg: Any, seed: int) -> str:
    """Stable hash of everything that determines the embeddings."""
    triples = "\n".join("|".join(row) for row in kg.triples[["head", "relation", "tail"]].values)
    return hash_obj(
        {
            "triples": hash_text(triples),
            "entities": hash_obj(kg.entity_index),
            "relations": hash_obj(kg.relation_index),
            "embedding": embedding_cfg.model_dump(),
            "seed": seed,
        }
    )[:24]


def train_pykeen(kg: KnowledgeGraph, embedding_cfg: Any, seed: int) -> Tensor:
    """Train a PyKEEN model; returns [num_entities, dim] in `kg.entity_index` order."""
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    if kg.triples.empty:
        raise ValueError("The knowledge graph has no triples; nothing to embed")
    labelled = kg.triples[["head", "relation", "tail"]].to_numpy(dtype=str)
    factory = TriplesFactory.from_labeled_triples(
        labelled, entity_to_id=kg.entity_index, relation_to_id=kg.relation_index
    )
    result = pipeline(
        training=factory,
        testing=factory,
        model=embedding_cfg.model,
        model_kwargs={"embedding_dim": embedding_cfg.dim},
        training_kwargs={"num_epochs": embedding_cfg.epochs, "batch_size": 256, "use_tqdm": False},
        random_seed=seed,
        device="cpu",
    )
    embeddings = result.model.entity_representations[0](indices=None).detach().cpu()
    if embeddings.is_complex():
        embeddings = torch.view_as_real(embeddings).flatten(1)
    return embeddings.float()


def get_kg_embeddings(
    kg: KnowledgeGraph, cfg: Any, paths: Any, train_fn: TrainFn | None = None
) -> Tensor:
    """Load cached embeddings for this KG + config, or train and cache them."""
    embedding_cfg = cfg.graph.embedding
    if embedding_cfg.dim != cfg.model.dim:
        raise ValueError(
            f"graph.embedding.dim ({embedding_cfg.dim}) must equal model.dim ({cfg.model.dim})"
        )
    path = paths.embedding(kg_config_hash(kg, embedding_cfg, cfg.seed))
    if path.exists():
        return load_checkpoint(path)["embeddings"]

    embeddings = (train_fn or train_pykeen)(kg, embedding_cfg, cfg.seed)
    expected = (len(kg.entity_index), embedding_cfg.dim)
    if tuple(embeddings.shape) != expected:
        raise ValueError(f"Embeddings have shape {tuple(embeddings.shape)}, expected {expected}")
    save_checkpoint({"embeddings": embeddings, "model": embedding_cfg.model}, path)
    return embeddings


__all__ = ["TrainFn", "get_kg_embeddings", "kg_config_hash", "train_pykeen"]
