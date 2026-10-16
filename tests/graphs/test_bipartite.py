import numpy as np
import pytest
import torch

import amazon_recsys.graphs  # noqa: F401  (registers builders)
from amazon_recsys.data.interactions import build_index
from amazon_recsys.graphs.base import BipartiteGraph
from amazon_recsys.graphs.bipartite import edge_weights
from amazon_recsys.graphs.registry import available_builders, build_graph


def _cfg(tiny_cfg, **graph):
    g = tiny_cfg.graph.model_copy(update={"name": "bipartite", **graph})
    return tiny_cfg.model_copy(update={"graph": g})


def _train(interactions_factory, rows, **columns):
    df = interactions_factory(rows)
    for name, values in columns.items():
        df[name] = values
    return df


def test_bipartite_builder_is_registered():
    assert "bipartite" in available_builders()


def test_node_count_and_edges_in_both_directions(tiny_cfg, tiny_splits):
    train = tiny_splits["train"]
    graph = build_graph("bipartite", train, _cfg(tiny_cfg))
    assert isinstance(graph, BipartiteGraph) and graph.kind == "bipartite"
    assert graph.num_users == train.user_id.nunique()
    assert graph.num_items == train.item_id.nunique()
    assert graph.edge_index.shape == (2, 2 * len(train))
    assert graph.edge_index.max() < graph.num_users + graph.num_items

    pairs = {(int(a), int(b)) for a, b in graph.edge_index.T.tolist()}
    for u, i in zip(train.user_id, train.item_id, strict=True):
        a, b = graph.user_index[u], graph.num_users + graph.item_index[i]
        assert (a, b) in pairs and (b, a) in pairs  # once in each direction
    assert len(pairs) == 2 * len(train)  # nothing duplicated


def test_no_val_or_test_edges_but_their_entities_have_nodes(tiny_cfg, tiny_splits):
    users, items = build_index(*tiny_splits.values())
    graph = build_graph(
        "bipartite", tiny_splits["train"], _cfg(tiny_cfg), user_index=users, item_index=items
    )
    assert graph.num_items == len(items) and graph.num_users == len(users)
    pairs = {(int(a), int(b)) for a, b in graph.edge_index.T.tolist()}
    held_out = [*tiny_splits["val"].itertuples(), *tiny_splits["test"].itertuples()]
    assert held_out
    for r in held_out:
        a, b = graph.user_index[r.user_id], graph.num_users + graph.item_index[r.item_id]
        assert (a, b) not in pairs and (b, a) not in pairs


def test_index_maps_are_bijective(tiny_cfg, tiny_splits):
    graph = build_graph("bipartite", tiny_splits["train"], _cfg(tiny_cfg))
    assert sorted(graph.user_index.values()) == list(range(graph.num_users))
    assert sorted(graph.item_index.values()) == list(range(graph.num_items))
    assert len(set(graph.user_index)) == graph.num_users


def test_weighting_none_rating_and_rating_sentiment_hand_computed(tiny_cfg, interactions_factory):
    train = _train(
        interactions_factory,
        [("u1", "i1", 1, 4), ("u1", "i2", 2, 2), ("u2", "i1", 3, 5)],
        overall_sentiment=[0.5, -1.0, float("nan")],
    )
    none = build_graph("bipartite", train, _cfg(tiny_cfg, weighting="none"))
    assert torch.equal(none.edge_weight, torch.ones(6))

    rating = build_graph("bipartite", train, _cfg(tiny_cfg, weighting="rating"))
    assert rating.edge_weight[:3].tolist() == pytest.approx([0.8, 0.4, 1.0])

    mixed = build_graph("bipartite", train, _cfg(tiny_cfg, weighting="rating_sentiment", alpha=0.5))
    # 0.5 * 0.8 + 0.5 * 0.75 ; 0.5 * 0.4 + 0.5 * 0.0 ; missing sentiment -> rating only
    assert mixed.edge_weight[:3].tolist() == pytest.approx([0.775, 0.2, 1.0])
    assert torch.equal(mixed.edge_weight[:3], mixed.edge_weight[3:])  # reverse edges match
    assert mixed.edge_weight.min() >= 0 and mixed.edge_weight.max() <= 1


def test_alpha_one_ignores_sentiment(tiny_cfg, interactions_factory):
    train = _train(interactions_factory, [("u1", "i1", 1, 4)], overall_sentiment=[-1.0])
    cfg = _cfg(tiny_cfg, weighting="rating_sentiment", alpha=1.0).graph
    assert edge_weights(train, cfg).tolist() == pytest.approx([0.8])


def test_judge_filter_falls_back_to_rating_only_but_keeps_the_edge(tiny_cfg, interactions_factory):
    rows = [(f"u{n}", "i1", n, 4) for n in range(6)]
    train = _train(
        interactions_factory,
        rows,
        overall_sentiment=[1.0] * 6,
        judge_overall=[0.9, 0.5, 0.9, float("nan"), 0.9, 0.7],
        judge_verdict=["pass", "fail", "uncertain", None, "fail", "pass"],
    )
    rating_only, mixed = 0.8, 0.5 * 0.8 + 0.5 * 1.0  # 0.9
    cfg = _cfg(tiny_cfg, weighting="rating_sentiment", alpha=0.5, min_judge_score=0.7)
    graph = build_graph("bipartite", train, cfg)
    assert graph.edge_index.shape[1] == 12  # every edge is still present
    # 0: pass 0.9 -> sentiment ; 1: below threshold ; 2: uncertain ; 3: missing ;
    # 4: fail verdict but score above the explicit threshold -> sentiment ; 5: exactly at 0.7
    expected = [mixed, rating_only, rating_only, rating_only, mixed, mixed]
    assert graph.edge_weight[:6].tolist() == pytest.approx(expected)


def test_accept_uncertain_admits_uncertain_verdicts(tiny_cfg, interactions_factory):
    train = _train(
        interactions_factory,
        [("u1", "i1", 1, 4)],
        overall_sentiment=[1.0],
        judge_overall=[0.9],
        judge_verdict=["uncertain"],
    )
    base = {"weighting": "rating_sentiment", "alpha": 0.5, "min_judge_score": 0.7}
    strict = edge_weights(train, _cfg(tiny_cfg, **base).graph)
    lenient = edge_weights(train, _cfg(tiny_cfg, **base, accept_uncertain=True).graph)
    assert strict.tolist() == pytest.approx([0.8]) and lenient.tolist() == pytest.approx([0.9])


def test_filter_off_uses_all_non_null_sentiment(tiny_cfg, interactions_factory):
    train = _train(
        interactions_factory,
        [("u1", "i1", 1, 4), ("u1", "i2", 2, 4)],
        overall_sentiment=[1.0, 1.0],
        judge_overall=[0.1, float("nan")],
        judge_verdict=["fail", None],
    )
    cfg = _cfg(tiny_cfg, weighting="rating_sentiment", alpha=0.5, min_judge_score=None).graph
    assert edge_weights(train, cfg).tolist() == pytest.approx([0.9, 0.9])


def test_unknown_weighting_raises(tiny_cfg, interactions_factory):
    train = _train(interactions_factory, [("u1", "i1", 1, 4)])
    cfg = tiny_cfg.graph.model_copy(update={"weighting": "magic"})
    with pytest.raises(ValueError, match="Unknown weighting"):
        edge_weights(train, cfg)


def test_all_weights_in_unit_interval_on_tiny_splits(tiny_bipartite):
    w = tiny_bipartite.edge_weight.numpy()
    assert np.all(w >= 0) and np.all(w <= 1)
