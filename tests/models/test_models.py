from pathlib import Path

import pytest
import torch

import amazon_recsys.models  # noqa: F401  (registers models)
from amazon_recsys.models.registry import available_models

CONFIGS_MODELS = Path(__file__).resolve().parents[2] / "configs" / "models"
# Models whose implementation lands in a later phase; remove names as they are built.
PENDING: set[str] = set()


def test_every_config_model_is_registered():
    names = {p.stem for p in CONFIGS_MODELS.glob("*.yaml")}
    assert names - PENDING <= set(available_models())
    assert PENDING <= names  # keep PENDING honest


def test_forward_shapes(tiny_models):
    model, graph, cfg = tiny_models
    user_emb, item_emb = model()
    assert user_emb.shape == (graph.num_users, cfg.model.dim)
    assert item_emb.shape == (graph.num_items, cfg.model.dim)


def test_score_and_score_all_agree(tiny_models):
    model, graph, _ = tiny_models
    users = torch.tensor([0, 2, 4, 1])
    items = torch.tensor([6, 1, 3, 0])
    pairwise = model.score(users, items)
    full = model.score_all(users)
    assert full.shape == (4, graph.num_items)
    assert torch.allclose(pairwise, full[torch.arange(4), items], atol=1e-6)


def test_gradients_flow_to_all_parameters(tiny_models):
    model, graph, _ = tiny_models
    users = torch.arange(graph.num_users)
    items = torch.arange(graph.num_users) % graph.num_items
    model.score(users, items).sum().backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    assert grads
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    assert all(g.abs().sum() > 0 for g in grads.values())


def test_save_load_reproduces_scores(tiny_models, tmp_path):
    model, graph, cfg = tiny_models
    users = torch.arange(graph.num_users)
    before = model.score_all(users).detach().clone()
    model.save(tmp_path / "m.pt")

    from amazon_recsys.models.registry import build_model

    fresh = build_model(cfg, graph).eval()
    assert not torch.allclose(fresh.score_all(users), before)  # different random init
    fresh.load(tmp_path / "m.pt")
    assert torch.allclose(fresh.score_all(users), before)


def test_load_rejects_mismatched_sizes(tiny_models, tmp_path):
    model, graph, cfg = tiny_models
    model.save(tmp_path / "m.pt")
    from amazon_recsys.graphs.base import BipartiteGraph
    from amazon_recsys.models.registry import build_model

    n_users = graph.num_users + 1
    other_graph = BipartiteGraph(
        num_users=n_users,
        num_items=3,
        user_index={},
        item_index={},
        edge_index=torch.tensor([[0, n_users], [n_users, 0]]),
        edge_weight=torch.ones(2),
    )
    if graph.kind == "knowledge":
        from amazon_recsys.graphs.base import KnowledgeGraph

        other_graph = KnowledgeGraph(
            num_users=n_users,
            num_items=3,
            user_index={},
            item_index={},
            triples=graph.triples.iloc[:0],
            entity_index={},
            relation_index={},
            interaction_graph=other_graph,
            entity_embeddings=torch.randn(n_users + 3, cfg.model.dim),
        )
    with pytest.raises(ValueError, match="Checkpoint is for"):
        build_model(cfg, other_graph).load(tmp_path / "m.pt")


def test_mf_score_is_dot_product():
    from amazon_recsys.models.mf import MatrixFactorisation

    model = MatrixFactorisation(2, 3, dim=2)
    with torch.no_grad():
        model.user_emb.weight.copy_(torch.tensor([[1.0, 2.0], [0.0, 1.0]]))
        model.item_emb.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]]))
    assert model.score(torch.tensor([0, 1]), torch.tensor([2, 1])).tolist() == [8.0, 1.0]
    assert model.score_all(torch.tensor([0])).tolist() == [[1.0, 2.0, 8.0]]


def _two_node_graph(weight=1.0):
    from amazon_recsys.graphs.base import BipartiteGraph

    return BipartiteGraph(
        num_users=1,
        num_items=1,
        user_index={"u": 0},
        item_index={"i": 0},
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_weight=torch.full((2,), weight),
    )


def test_lightgcn_output_is_mean_of_layer_embeddings_on_two_node_graph():
    from amazon_recsys.models.lightgcn import LightGCN

    model = LightGCN(_two_node_graph(), dim=3, layers=2)
    u = torch.tensor([1.0, 2.0, 3.0])
    i = torch.tensor([4.0, 5.0, 6.0])
    with torch.no_grad():
        model.embedding.weight.copy_(torch.stack([u, i]))
    user_emb, item_emb = model()
    # degree 1 on both nodes -> norm 1: layer 1 swaps the two, layer 2 swaps back
    assert torch.allclose(user_emb[0], (u + i + u) / 3)
    assert torch.allclose(item_emb[0], (i + u + i) / 3)


def test_lightgcn_ignores_edge_weights():
    from amazon_recsys.models.lightgcn import LightGCN

    a = LightGCN(_two_node_graph(1.0), dim=4, layers=2)
    b = LightGCN(_two_node_graph(0.1), dim=4, layers=2)
    b.load_state_dict(a.state_dict())
    for x, y in zip(a(), b(), strict=True):
        assert torch.equal(x, y)


def _gat(graph, use_edge_weight, seed=0):
    from amazon_recsys.models.gat import GATRecommender
    from amazon_recsys.utils.seed import set_seed

    set_seed(seed)
    return GATRecommender(
        graph, dim=8, layers=1, heads=2, dropout=0.0, use_edge_weight=use_edge_weight
    ).eval()


def test_gat_edge_weights_change_the_output_only_when_enabled(tiny_bipartite):
    from dataclasses import replace

    ones = replace(tiny_bipartite, edge_weight=torch.ones_like(tiny_bipartite.edge_weight))
    weighted_out = _gat(tiny_bipartite, True)()
    ones_out = _gat(ones, True)()
    assert not torch.allclose(weighted_out[0], ones_out[0])  # weights matter

    assert torch.allclose(_gat(tiny_bipartite, False)()[0], _gat(ones, False)()[0])  # ignored


def test_gat_is_deterministic_in_eval_and_rejects_bad_head_count(tiny_bipartite):
    from amazon_recsys.models.gat import GATEncoder

    model = _gat(tiny_bipartite, True)
    first, second = model(), model()
    assert torch.equal(first[0], second[0])
    with pytest.raises(ValueError, match="divisible"):
        GATEncoder(dim=7, layers=1, heads=2, dropout=0.0, edge_dim=None)


def test_kg_gat_starts_from_kg_embeddings_and_respects_freeze(tiny_kg, tiny_cfg):
    from amazon_recsys.models.kg_gat import KGGATRecommender

    n = tiny_kg.num_users + tiny_kg.num_items
    frozen = KGGATRecommender(tiny_kg, 8, 1, 2, 0.0, False, freeze=True)
    assert torch.equal(frozen.embedding.weight.detach(), tiny_kg.entity_embeddings[:n])
    assert not frozen.embedding.weight.requires_grad

    before = frozen.embedding.weight.detach().clone()
    optimizer = torch.optim.Adam([p for p in frozen.parameters() if p.requires_grad], lr=0.1)
    users, items = torch.arange(tiny_kg.num_users), torch.zeros(tiny_kg.num_users, dtype=torch.long)
    for _ in range(3):
        optimizer.zero_grad()
        frozen.score(users, items).sum().backward()
        optimizer.step()
    assert torch.equal(frozen.embedding.weight.detach(), before)  # stays fixed

    free = KGGATRecommender(tiny_kg, 8, 1, 2, 0.0, False, freeze=False)
    assert free.embedding.weight.requires_grad
    assert torch.equal(free.embedding.weight.detach(), tiny_kg.entity_embeddings[:n])


def test_kg_gat_uses_the_interaction_graph_and_checks_embeddings(tiny_kg):
    from dataclasses import replace

    from amazon_recsys.models.kg_gat import KGGATRecommender

    model = KGGATRecommender(tiny_kg, 8, 1, 2, 0.0, True, freeze=False)
    assert torch.equal(model.edge_index, tiny_kg.interaction_graph.edge_index)
    with pytest.raises(ValueError, match="missing"):
        KGGATRecommender(replace(tiny_kg, entity_embeddings=None), 8, 1, 2, 0.0, False, False)
    with pytest.raises(ValueError, match="dim"):
        KGGATRecommender(tiny_kg, 4, 1, 2, 0.0, False, False)
