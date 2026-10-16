"""Knowledge graph builder: user/item/aspect triples from the train split only."""

from __future__ import annotations

from typing import Any

import pandas as pd

from amazon_recsys.data.interactions import build_index
from amazon_recsys.graphs.base import GraphBuilder, KnowledgeGraph
from amazon_recsys.graphs.bipartite import BipartiteBuilder
from amazon_recsys.graphs.registry import register_builder

USER, ITEM, ASPECT = "u:", "i:", "a:"
TRIPLE_COLUMNS = ["head", "relation", "tail"]
_JEV_POLARITY = {"p_positive": 1, "p_negative": -1, "p_neutral": 0}
_MENTION_COLUMNS = ["user_id", "item_id", "review_id", "aspect", "polarity"]


def aspect_mentions(
    train: pd.DataFrame,
    aspects: pd.DataFrame | None,
    aspect_judgements: pd.DataFrame | None,
    graph_cfg: Any,
) -> pd.DataFrame:
    """Aspect mentions from train reviews that survive the filters, one row per review-aspect.

    `judged_only=True`: needs a judgement with `p_supported >= min_aspect_support`, an accepted
    review verdict, and applies the polarity override. `judged_only=False` keeps everything
    with the extractor's polarity.
    """
    if aspects is None or aspects.empty:
        return pd.DataFrame(columns=_MENTION_COLUMNS)

    context = train[["review_id", "user_id", "item_id", "judge_verdict"]]
    mentions = aspects[["review_id", "aspect", "polarity"]].merge(context, on="review_id")
    if not graph_cfg.judged_only:
        return mentions[_MENTION_COLUMNS].reset_index(drop=True)
    if aspect_judgements is None or aspect_judgements.empty:
        return pd.DataFrame(columns=_MENTION_COLUMNS)

    judged = aspect_judgements[
        ["review_id", "aspect", "p_supported", "p_positive", "p_negative", "p_neutral",
         "polarity_confidence"]
    ]  # fmt: skip
    mentions = mentions.merge(judged, on=["review_id", "aspect"], how="inner")
    accepted = {"pass"} | ({"uncertain"} if graph_cfg.accept_uncertain else set())
    keep = (mentions["p_supported"] >= graph_cfg.min_aspect_support) & mentions[
        "judge_verdict"
    ].isin(accepted)
    mentions = mentions[keep].copy()

    top = mentions[list(_JEV_POLARITY)].idxmax(axis=1).map(_JEV_POLARITY)
    override = (mentions["polarity_confidence"] >= graph_cfg.min_polarity_confidence) & (
        top != mentions["polarity"]
    )
    mentions["polarity"] = top.where(override, mentions["polarity"]).astype(int)
    return mentions[_MENTION_COLUMNS].reset_index(drop=True)


def interaction_triples(train: pd.DataFrame, graph_cfg: Any) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for r in train.itertuples():
        if r.rating >= graph_cfg.like_threshold:
            triples.append((f"{USER}{r.user_id}", "likes", f"{ITEM}{r.item_id}"))
        elif r.rating <= graph_cfg.dislike_threshold:
            triples.append((f"{USER}{r.user_id}", "dislikes", f"{ITEM}{r.item_id}"))
    return triples


def aspect_triples(mentions: pd.DataFrame, graph_cfg: Any) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    if mentions.empty:
        return triples

    per_item = (
        mentions[mentions["polarity"] != 0]
        .groupby(["item_id", "aspect", "polarity"])["review_id"]
        .nunique()
        .reset_index(name="n")
    )
    relation = {1: "has_positive_aspect", -1: "has_negative_aspect"}
    for r in per_item[per_item["n"] >= graph_cfg.min_aspect_count].itertuples():
        triples.append((f"{ITEM}{r.item_id}", relation[int(r.polarity)], f"{ASPECT}{r.aspect}"))

    per_user = mentions.groupby(["user_id", "aspect"])["review_id"].nunique().reset_index(name="n")
    for r in per_user[per_user["n"] >= graph_cfg.min_user_aspect_count].itertuples():
        triples.append((f"{USER}{r.user_id}", "cares_about", f"{ASPECT}{r.aspect}"))
    return triples


@register_builder("knowledge")
class KnowledgeBuilder(GraphBuilder):
    kind = "knowledge"

    def build(
        self,
        train: pd.DataFrame,
        cfg: Any,
        user_index: dict[str, int] | None = None,
        item_index: dict[str, int] | None = None,
        aspects: pd.DataFrame | None = None,
        aspect_judgements: pd.DataFrame | None = None,
        **_: Any,
    ) -> KnowledgeGraph:
        g = cfg.graph
        if user_index is None or item_index is None:
            user_index, item_index = build_index(train)

        # Train split only: aspects of val/test reviews are dropped before anything else.
        train_ids = set(train["review_id"])
        if aspects is not None:
            aspects = aspects[aspects["review_id"].isin(train_ids)]
        if aspect_judgements is not None:
            aspect_judgements = aspect_judgements[aspect_judgements["review_id"].isin(train_ids)]

        mentions = aspect_mentions(train, aspects, aspect_judgements, g)
        raw = interaction_triples(train, g) + aspect_triples(mentions, g)
        wanted = set(g.relations)
        triples = pd.DataFrame(
            [t for t in raw if t[1] in wanted], columns=TRIPLE_COLUMNS
        ).drop_duplicates()
        triples = triples.sort_values(TRIPLE_COLUMNS).reset_index(drop=True)

        users = [f"{USER}{u}" for u, _i in sorted(user_index.items(), key=lambda kv: kv[1])]
        items = [f"{ITEM}{i}" for i, _j in sorted(item_index.items(), key=lambda kv: kv[1])]
        in_triples = set(triples["head"]) | set(triples["tail"])
        aspect_ids = sorted(e for e in in_triples if e.startswith(ASPECT))
        entity_index = {e: n for n, e in enumerate([*users, *items, *aspect_ids])}
        relation_index = {r: n for n, r in enumerate(sorted(wanted))}

        return KnowledgeGraph(
            num_users=len(user_index),
            num_items=len(item_index),
            user_index=dict(user_index),
            item_index=dict(item_index),
            triples=triples,
            entity_index=entity_index,
            relation_index=relation_index,
            interaction_graph=BipartiteBuilder().build(train, cfg, user_index, item_index),
        )
