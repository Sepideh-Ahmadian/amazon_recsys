import pandas as pd
import pytest

import amazon_recsys.graphs  # noqa: F401  (registers builders)
from amazon_recsys.config import SplitCfg
from amazon_recsys.data.interactions import build_index
from amazon_recsys.data.splits import make_splits
from amazon_recsys.graphs.base import KnowledgeGraph
from amazon_recsys.graphs.registry import available_builders, build_graph


@pytest.fixture
def world(interactions_factory):
    """Hand-worked example. Review ids are r0..r8 in this order.

    u1: i1(5) i2(1) | val i3 | test i4
    u2: i2(4)       | val i5 | test i6
    u3: i1(2) i6(3)  (too short: train only)
    """
    rows = [
        ("u1", "i1", 1, 5), ("u1", "i2", 2, 1), ("u1", "i3", 3, 4), ("u1", "i4", 4, 4),
        ("u2", "i2", 1, 4), ("u2", "i5", 2, 4), ("u2", "i6", 3, 3),
        ("u3", "i1", 1, 2), ("u3", "i6", 2, 3),
    ]  # fmt: skip
    df = interactions_factory(rows)
    df["judge_verdict"] = ["pass"] * 9
    df.loc[df.review_id == "r7", "judge_verdict"] = "fail"  # u3's review is not trustworthy
    splits, _ = make_splits(df, SplitCfg(min_user_interactions=3))
    assert set(splits["train"].review_id) == {"r0", "r1", "r4", "r7", "r8"}

    def a(review, aspect, polarity):
        return {"review_id": review, "domain": "d", "aspect": aspect, "aspect_raw": aspect,
                "polarity": polarity, "intensity": 0.5, "evidence": "e"}  # fmt: skip

    def j(review, aspect, supported=0.9, pos=0.0, neg=0.0, neu=0.0, conf=0.9):
        return {"review_id": review, "domain": "d", "aspect": aspect, "p_supported": supported,
                "p_polarity_match": 0.5, "p_positive": pos, "p_negative": neg, "p_neutral": neu,
                "polarity_confidence": conf}  # fmt: skip

    aspects = pd.DataFrame(
        [
            a("r0", "scent", 1), a("r0", "texture", -1),  # u1 on i1
            a("r1", "scent", -1),                          # u1 on i2
            a("r4", "scent", -1), a("r4", "texture", 1),   # u2 on i2
            a("r7", "scent", 1),                           # u3 on i1 (verdict fail)
            a("r2", "scent", 1),                           # val review: must never appear
        ]
    )  # fmt: skip
    judged = pd.DataFrame(
        [
            j("r0", "scent", pos=0.9, neg=0.05, neu=0.05),
            j("r0", "texture", supported=0.3, neg=0.9),               # unsupported
            j("r1", "scent", pos=0.7, neg=0.2, neu=0.1, conf=0.5),     # disagrees, low conf
            j("r4", "scent", neg=0.9, pos=0.05, neu=0.05),
            j("r4", "texture", neg=0.95, pos=0.03, neu=0.02, conf=0.95),  # overrides + to -
            j("r7", "scent", pos=0.9),
            j("r2", "scent", pos=0.9),
        ]
    )  # fmt: skip
    users, items = build_index(*splits.values())
    return splits["train"], users, items, aspects, judged


def _cfg(tiny_cfg, **graph):
    g = tiny_cfg.graph.model_copy(update={"name": "knowledge", **graph})
    return tiny_cfg.model_copy(update={"graph": g})


def _build(world, tiny_cfg, **graph):
    train, users, items, aspects, judged = world
    return build_graph(
        "knowledge", train, _cfg(tiny_cfg, **graph), user_index=users, item_index=items,
        aspects=aspects, aspect_judgements=judged,
    )  # fmt: skip


def _triples(kg):
    return set(map(tuple, kg.triples.itertuples(index=False, name=None)))


def test_builder_registered_and_returns_knowledge_graph(world, tiny_cfg):
    assert "knowledge" in available_builders()
    kg = _build(world, tiny_cfg)
    assert isinstance(kg, KnowledgeGraph) and kg.kind == "knowledge"


def test_interaction_triples_match_hand_computed_and_exclude_val_test(world, tiny_cfg):
    t = _triples(_build(world, tiny_cfg, relations=["likes", "dislikes"]))
    assert t == {
        ("u:u1", "likes", "i:i1"),
        ("u:u2", "likes", "i:i2"),
        ("u:u1", "dislikes", "i:i2"),
        ("u:u3", "dislikes", "i:i1"),
    }  # u3/i6 has rating 3: neither; val/test pairs (u1,i3) (u1,i4) (u2,i5) (u2,i6) absent


def test_aspect_triples_apply_support_verdict_and_polarity_override(world, tiny_cfg):
    t = _triples(_build(world, tiny_cfg))
    aspect_t = {x for x in t if x[1] not in ("likes", "dislikes")}
    assert aspect_t == {
        ("i:i1", "has_positive_aspect", "a:scent"),  # r0 (r7 dropped: verdict fail)
        ("i:i2", "has_negative_aspect", "a:scent"),  # r1 keeps -1 (override conf too low), r4
        ("i:i2", "has_negative_aspect", "a:texture"),  # r4: extractor +1 overridden by Jev -1
        ("u:u1", "cares_about", "a:scent"),
        ("u:u2", "cares_about", "a:scent"),
        ("u:u2", "cares_about", "a:texture"),
    }
    # r0's texture (p_supported 0.3), r7 (fail verdict) and the val review r2 contribute nothing
    assert not any(x[0] == "u:u3" and x[1] == "cares_about" for x in t)


def test_min_counts_are_applied_to_distinct_reviews(world, tiny_cfg):
    t = _triples(_build(world, tiny_cfg, min_aspect_count=2, min_user_aspect_count=2))
    aspect_t = {x for x in t if x[1] not in ("likes", "dislikes")}
    assert aspect_t == {
        ("i:i2", "has_negative_aspect", "a:scent"),  # r1 and r4
        ("u:u1", "cares_about", "a:scent"),  # r0 and r1
        ("u:u2", "cares_about", "a:scent"),  # only r4 -> dropped below
    } - {("u:u2", "cares_about", "a:scent")}


def test_unjudged_ablation_keeps_every_extracted_train_aspect(world, tiny_cfg):
    t = _triples(_build(world, tiny_cfg, judged_only=False))
    assert ("i:i1", "has_positive_aspect", "a:scent") in t  # r0 and r7
    assert ("i:i1", "has_negative_aspect", "a:texture") in t  # unsupported but kept
    assert ("i:i2", "has_positive_aspect", "a:texture") in t  # no polarity override
    assert ("u:u3", "cares_about", "a:scent") in t  # the verdict gate is off
    assert all(x[2] != "a:scent" or x[0] != "i:i3" for x in t)  # val review r2 still excluded


def test_accept_uncertain_admits_uncertain_reviews(world, tiny_cfg):
    train, users, items, aspects, judged = world
    train = train.assign(
        judge_verdict=["uncertain" if r == "r7" else "pass" for r in train.review_id]
    )
    cfg = _cfg(tiny_cfg)
    strict = build_graph("knowledge", train, cfg, user_index=users, item_index=items,
                         aspects=aspects, aspect_judgements=judged)  # fmt: skip
    lenient_cfg = _cfg(tiny_cfg, accept_uncertain=True)
    lenient = build_graph("knowledge", train, lenient_cfg, user_index=users, item_index=items,
                          aspects=aspects, aspect_judgements=judged)  # fmt: skip
    assert ("u:u3", "cares_about", "a:scent") not in _triples(strict)
    assert ("u:u3", "cares_about", "a:scent") in _triples(lenient)


def test_polarity_override_needs_enough_confidence(world, tiny_cfg):
    # r4 texture: extractor +1, Jev says negative with confidence 0.95
    # r1 scent:   extractor -1, Jev says positive with confidence 0.5
    suppressed = _triples(_build(world, tiny_cfg, min_polarity_confidence=0.99))
    assert ("i:i2", "has_positive_aspect", "a:texture") in suppressed
    assert ("i:i2", "has_negative_aspect", "a:texture") not in suppressed

    default = _triples(_build(world, tiny_cfg, min_polarity_confidence=0.7))
    assert ("i:i2", "has_negative_aspect", "a:texture") in default  # 0.95 >= 0.7: overridden
    assert ("i:i2", "has_positive_aspect", "a:scent") not in default  # 0.5 < 0.7: kept as -1

    permissive = _triples(_build(world, tiny_cfg, min_polarity_confidence=0.4))
    assert ("i:i2", "has_positive_aspect", "a:scent") in permissive  # now r1 is overridden too


def test_entity_ids_are_namespaced_and_ordered_users_items_aspects(world, tiny_cfg):
    kg = _build(world, tiny_cfg)
    train, users, items, *_ = world
    names = list(kg.entity_index)
    assert all(n.startswith(("u:", "i:", "a:")) for n in names)
    assert names[: len(users)] == [f"u:{u}" for u, _ in sorted(users.items(), key=lambda kv: kv[1])]
    assert kg.entity_index[f"u:{next(iter(users))}"] == users[next(iter(users))]
    for item, idx in items.items():
        assert kg.entity_index[f"i:{item}"] == len(users) + idx
    assert sorted(kg.entity_index.values()) == list(range(len(names)))
    assert {n for n in names if n.startswith("a:")} == {"a:scent", "a:texture"}
    # every triple endpoint is a known entity, every relation a known relation
    assert set(kg.triples["head"]) | set(kg.triples["tail"]) <= set(names)
    assert set(kg.triples["relation"]) <= set(kg.relation_index)


def test_relation_set_is_restricted_by_config(world, tiny_cfg):
    kg = _build(world, tiny_cfg, relations=["has_positive_aspect"])
    assert set(kg.triples["relation"]) == {"has_positive_aspect"}
    assert list(kg.relation_index) == ["has_positive_aspect"]


def test_interaction_graph_is_the_train_graph_and_missing_tables_are_fine(world, tiny_cfg):
    train, users, items, *_ = world
    kg = build_graph("knowledge", train, _cfg(tiny_cfg), user_index=users, item_index=items)
    assert kg.interaction_graph.edge_index.shape[1] == 2 * len(train)
    assert set(kg.triples["relation"]) <= {"likes", "dislikes"}  # no aspect tables given
