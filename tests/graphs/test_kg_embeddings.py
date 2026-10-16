import pytest
import torch

from amazon_recsys.graphs.kg_embeddings import get_kg_embeddings, kg_config_hash


def _cfg(tiny_cfg, **embedding):
    emb = tiny_cfg.graph.embedding.model_copy(update={"epochs": 1, "dim": 8, **embedding})
    return tiny_cfg.model_copy(
        update={"graph": tiny_cfg.graph.model_copy(update={"embedding": emb})}
    )


def test_pykeen_run_gives_one_embedding_per_entity_with_configured_dim(
    tiny_kg, tiny_cfg, tmp_paths
):
    cfg = _cfg(tiny_cfg)
    emb = get_kg_embeddings(tiny_kg, cfg, tmp_paths)
    assert tuple(emb.shape) == (len(tiny_kg.entity_index), 8)
    assert torch.isfinite(emb).all()
    assert len(list(tmp_paths.embeddings_dir.glob("*.pt"))) == 1


def test_second_call_loads_from_cache_without_training(tiny_kg, tiny_cfg, tmp_paths):
    cfg = _cfg(tiny_cfg)
    calls = []

    def fake_train(kg, embedding_cfg, seed):
        calls.append(1)
        return torch.randn(len(kg.entity_index), embedding_cfg.dim)

    first = get_kg_embeddings(tiny_kg, cfg, tmp_paths, train_fn=fake_train)
    second = get_kg_embeddings(tiny_kg, cfg, tmp_paths, train_fn=fake_train)
    assert len(calls) == 1 and torch.equal(first, second)


def test_config_change_produces_a_new_file(tiny_kg, tiny_cfg, tmp_paths):
    def fake_train(kg, embedding_cfg, seed):
        return torch.zeros(len(kg.entity_index), embedding_cfg.dim)

    get_kg_embeddings(tiny_kg, _cfg(tiny_cfg), tmp_paths, train_fn=fake_train)
    get_kg_embeddings(tiny_kg, _cfg(tiny_cfg, epochs=2), tmp_paths, train_fn=fake_train)
    assert len(list(tmp_paths.embeddings_dir.glob("*.pt"))) == 2
    a = kg_config_hash(tiny_kg, _cfg(tiny_cfg).graph.embedding, 0)
    assert a == kg_config_hash(tiny_kg, _cfg(tiny_cfg).graph.embedding, 0)
    assert a != kg_config_hash(tiny_kg, _cfg(tiny_cfg).graph.embedding, 1)  # seed matters


def test_dim_must_match_model_dim_and_shape_is_checked(tiny_kg, tiny_cfg, tmp_paths):
    with pytest.raises(ValueError, match="must equal model.dim"):
        get_kg_embeddings(tiny_kg, _cfg(tiny_cfg, dim=16), tmp_paths)
    with pytest.raises(ValueError, match="expected"):
        get_kg_embeddings(
            tiny_kg, _cfg(tiny_cfg), tmp_paths, train_fn=lambda kg, c, s: torch.zeros(2, 8)
        )
