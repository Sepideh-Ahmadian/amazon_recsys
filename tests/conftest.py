"""Shared fixtures. Later phases add: tiny_reviews, fake_llm, fake_jev, ..."""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from amazon_recsys.config import Config, load_config
from amazon_recsys.utils.paths import Paths
from amazon_recsys.utils.seed import set_seed

REAL_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(autouse=True)
def seed() -> int:
    set_seed(0)
    return 0


@pytest.fixture
def tiny_cfg() -> Config:
    """Fully resolved all_beauty config (GAT on a bipartite graph) with tiny sizes."""
    return load_config(
        "gat_plain",
        REAL_CONFIGS,
        sets=[
            "model.dim=8",
            "model.layers=1",
            "model.heads=2",
            "train.epochs=2",
            "train.batch_size=16",
            "train.eval_every=1",
            "train.patience=2",
            "eval.ks=[5,10]",
            "graph.embedding.dim=8",
            "graph.embedding.epochs=1",
        ],
        seed=0,
    )


@pytest.fixture
def tmp_paths(tmp_path: Path, tiny_cfg: Config) -> Paths:
    """Paths rooted in tmp_path so no test touches the real data/ or runs/."""
    return Paths.from_config(tiny_cfg, root=tmp_path)


@dataclass
class RawFiles:
    reviews: Path
    meta: Path
    n_reviews: int  # unique valid reviews expected after ingestion
    n_items: int
    n_lines: int
    n_malformed: int
    n_duplicates: int
    n_missing: int


def _review(user, item, rating=4.0, ts=1_600_000_000_000, **extra):
    return {
        "user_id": user,
        "parent_asin": item,
        "rating": rating,
        "title": f"title {user}",
        "text": f"text {user} {item}",
        "timestamp": ts,
        **extra,
    }


def write_gz(path: Path, lines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line if isinstance(line, str) else json.dumps(line))
            fh.write("\n")


@pytest.fixture
def raw_jsonl_gz(tmp_paths: Paths, tiny_cfg: Config) -> RawFiles:
    """Small raw review + meta files with 2 malformed lines, 1 duplicate, 1 missing field."""
    reviews = [
        _review("u1", "i1", 5.0, 1_600_000_000_000),
        _review("u1", "i2", 3.0, 1_600_000_100_000),
        _review("u2", "i1", 4.0, 1_600_000_200_000),
        _review("u2", "i3", 1.0, 1_600_000_300_000),
        _review("u3", "i2", 2.0, 1_600_000_400_000),
        _review("u3", "i3", 5.0, 1_600_000_500),  # seconds, normalised to ms
        _review("u1", "i1", 5.0, 1_600_000_000_000),  # duplicate of the first row
        "{ this is not json",  # malformed
        "[1, 2, 3]",  # valid JSON but not an object -> malformed
        {"parent_asin": "i9", "rating": 4.0, "timestamp": 1_600_000_600_000},  # no user_id
    ]
    meta = [
        {
            "parent_asin": "i1",
            "title": "Shampoo",
            "main_category": "All Beauty",
            "store": "Acme",
            "description": ["Great", "shampoo"],
            "categories": [],
        },
        {
            "parent_asin": "i2",
            "title": "Soap",
            "main_category": None,
            "store": None,
            "description": [],
            "categories": ["Beauty", "Bath"],
        },
        {
            "parent_asin": "i3",
            "title": "Lotion",
            "main_category": "All Beauty",
            "store": "Beta",
            "description": "Smooth",
            "categories": [],
        },
        {"parent_asin": "i1", "title": "Shampoo again"},  # duplicate item
        "not json either",
    ]
    reviews_path = tmp_paths.resolve(tiny_cfg.domain.reviews_file)
    meta_path = tmp_paths.resolve(tiny_cfg.domain.meta_file)
    write_gz(reviews_path, reviews)
    write_gz(meta_path, meta)
    return RawFiles(
        reviews=reviews_path,
        meta=meta_path,
        n_reviews=6,
        n_items=3,
        n_lines=10,
        n_malformed=2,
        n_duplicates=1,
        n_missing=1,
    )


INTERACTION_COLUMNS = [
    "domain",
    "user_id",
    "item_id",
    "rating",
    "timestamp",
    "review_id",
    "overall_sentiment",
    "judge_overall",
    "judge_verdict",
]


@pytest.fixture
def interactions_factory():
    """Build an interactions frame from (user, item, timestamp[, rating]) tuples."""

    def make(rows) -> pd.DataFrame:
        data = [
            {
                "domain": "d",
                "user_id": r[0],
                "item_id": r[1],
                "rating": float(r[3]) if len(r) > 3 else 4.0,
                "timestamp": int(r[2]),
                "review_id": f"r{n}",
                "overall_sentiment": None,
                "judge_overall": None,
                "judge_verdict": None,
            }
            for n, r in enumerate(rows)
        ]
        df = pd.DataFrame(data, columns=INTERACTION_COLUMNS)
        return df.astype(
            {"overall_sentiment": "float64", "judge_overall": "float64", "timestamp": "int64"}
        )

    return make


@pytest.fixture
def hand_split(interactions_factory):
    """3 users, 6 items, expected split worked out by hand.

    u1: i1 i2 i3 i4  -> train {i1,i2}  val i3  test i4
    u2: i2 i5 i6     -> train {i2}     val i5  test i6
    u3: i1 i6        -> only 2 interactions (< 3): train only
    """
    rows = [
        ("u1", "i1", 1), ("u1", "i2", 2), ("u1", "i3", 3), ("u1", "i4", 4),
        ("u2", "i2", 1), ("u2", "i5", 2), ("u2", "i6", 3),
        ("u3", "i1", 1), ("u3", "i6", 2),
    ]  # fmt: skip
    expected = {
        "train": {("u1", "i1"), ("u1", "i2"), ("u2", "i2"), ("u3", "i1"), ("u3", "i6")},
        "val": {("u1", "i3"), ("u2", "i5")},
        "test": {("u1", "i4"), ("u2", "i6")},
    }
    return interactions_factory(rows), expected


def _registered_models() -> list[str]:
    import amazon_recsys.models  # noqa: F401  (registers every model)
    from amazon_recsys.models.registry import available_models

    return available_models()


@pytest.fixture(params=_registered_models())
def tiny_models(request, tiny_cfg, tiny_bipartite, tiny_kg):
    """(model, graph, cfg) for every registered model, on a tiny problem (dim 8), in eval mode."""
    from amazon_recsys.models.registry import build_model, get_model_class

    cfg = tiny_cfg.model_copy(
        update={"model": tiny_cfg.model.model_copy(update={"name": request.param})}
    )
    needs = get_model_class(request.param).requires_graph
    graph = tiny_kg if needs == "knowledge" else tiny_bipartite
    return build_model(cfg, graph).eval(), graph, cfg


@pytest.fixture
def tiny_splits(interactions_factory):
    """8 users x 10 items; user u likes items of parity u%2 (5 each). Real leave-one-out split
    with sentiment and judge columns filled so weighting / filtering can be exercised."""
    from amazon_recsys.config import SplitCfg
    from amazon_recsys.data.splits import make_splits

    rows = []
    for u in range(8):
        items = [i for i in range(10) if i % 2 == u % 2]
        rows += [(f"u{u}", f"i{i}", 100 + n, 1 + (u + n) % 5) for n, i in enumerate(items)]
    df = interactions_factory(rows)
    df["overall_sentiment"] = [((n % 5) - 2) / 2 for n in range(len(df))]
    df["judge_overall"] = [0.5 + (n % 5) / 10 for n in range(len(df))]
    df["judge_verdict"] = ["pass" if n % 5 >= 2 else "fail" for n in range(len(df))]
    return make_splits(df, SplitCfg(min_user_interactions=3))[0]


@pytest.fixture
def tiny_bipartite(tiny_cfg, tiny_splits):
    """Bipartite graph (rating_sentiment weights) built from tiny_splits.train."""
    import amazon_recsys.graphs  # noqa: F401  (registers builders)
    from amazon_recsys.data.interactions import build_index
    from amazon_recsys.graphs.registry import build_graph

    graph_cfg = tiny_cfg.graph.model_copy(
        update={"name": "bipartite", "weighting": "rating_sentiment"}
    )
    cfg = tiny_cfg.model_copy(update={"graph": graph_cfg})
    users, items = build_index(*tiny_splits.values())
    return build_graph("bipartite", tiny_splits["train"], cfg, user_index=users, item_index=items)


@pytest.fixture
def tiny_aspects(tiny_splits):
    """Aspect + aspect-judgement tables for every train review of tiny_splits."""
    rows, judged = [], []
    for n, r in enumerate(tiny_splits["train"].itertuples()):
        for name, polarity in (("scent", 1 if n % 3 else -1), ("texture", 1)):
            pos = polarity == 1
            rows.append(
                {
                    "review_id": r.review_id,
                    "domain": "d",
                    "aspect": name,
                    "aspect_raw": name.title(),
                    "polarity": polarity,
                    "intensity": 0.5,
                    "evidence": "ev",
                }
            )
            judged.append(
                {"review_id": r.review_id, "domain": "d", "aspect": name, "p_supported": 0.9,
                 "p_polarity_match": 0.9, "p_positive": 0.8 if pos else 0.1,
                 "p_negative": 0.1 if pos else 0.8, "p_neutral": 0.1, "polarity_confidence": 0.8}
            )  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(judged)


@pytest.fixture
def tiny_kg(tiny_cfg, tiny_splits, tiny_aspects):
    """Knowledge graph from tiny_splits.train, with random (not PyKEEN) entity embeddings."""
    import torch

    import amazon_recsys.graphs  # noqa: F401  (registers builders)
    from amazon_recsys.data.interactions import build_index
    from amazon_recsys.graphs.registry import build_graph

    graph_cfg = tiny_cfg.graph.model_copy(update={"name": "knowledge", "min_aspect_count": 1})
    cfg = tiny_cfg.model_copy(update={"graph": graph_cfg})
    users, items = build_index(*tiny_splits.values())
    aspects, judged = tiny_aspects
    kg = build_graph(
        "knowledge", tiny_splits["train"], cfg, user_index=users, item_index=items,
        aspects=aspects, aspect_judgements=judged,
    )  # fmt: skip
    gen = torch.Generator().manual_seed(0)
    kg.entity_embeddings = torch.randn(len(kg.entity_index), cfg.model.dim, generator=gen)
    return kg
