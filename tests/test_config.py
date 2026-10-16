from pathlib import Path

import pytest
import yaml

from amazon_recsys.config import (
    Config,
    ConfigError,
    load_config,
    parse_set,
    require_resolved,
)


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    root = tmp_path / "configs"
    _write(root / "base.yaml", {"seed": 0, "device": "cpu"})
    _write(
        root / "domains" / "all_beauty.yaml",
        {
            "name": "all_beauty",
            "reviews_file": "data/raw/r.jsonl.gz",
            "meta_file": "data/raw/m.jsonl.gz",
            "columns": {
                "user": "user_id",
                "item": "parent_asin",
                "rating": "rating",
                "text": "text",
                "title": "title",
                "time": "timestamp",
            },
            "prompt_dir": "prompts/all_beauty",
        },
    )
    _write(root / "enrichment.yaml", {})
    _write(root / "split.yaml", {})
    _write(root / "graphs" / "bipartite.yaml", {"weighting": "none"})
    _write(root / "models" / "gat.yaml", {"dim": 8})
    _write(root / "models" / "mf.yaml", {"dim": 8})
    _write(root / "train.yaml", {"epochs": 5, "lr": 0.001})
    _write(root / "eval.yaml", {"ks": [10]})
    _write(
        root / "experiments" / "gat_plain.yaml",
        {
            "domain": "all_beauty",
            "graph": "bipartite",
            "model": "gat",
            "overrides": {"graph.weighting": "rating", "train.epochs": 7},
        },
    )
    _write(root / "experiments" / "mf.yaml", {"domain": "all_beauty", "model": "mf"})
    return root


def test_loads_and_merges_components(configs_dir):
    cfg = load_config("gat_plain", configs_dir)
    assert cfg.name == "gat_plain"
    assert cfg.model.name == "gat" and cfg.model.dim == 8
    assert cfg.graph is not None and cfg.graph.name == "bipartite"
    assert cfg.graph.weighting == "rating"  # experiment override
    assert cfg.train.epochs == 7


def test_graph_is_optional(configs_dir):
    assert load_config("mf", configs_dir).graph is None


def test_unknown_key_in_component_file_raises(configs_dir):
    _write(configs_dir / "train.yaml", {"epochs": 5, "lerning_rate": 0.1})
    with pytest.raises(ConfigError):
        load_config("mf", configs_dir)


def test_wrong_type_raises(configs_dir):
    _write(configs_dir / "train.yaml", {"epochs": "many"})
    with pytest.raises(ConfigError):
        load_config("mf", configs_dir)


def test_set_overrides_are_coerced(configs_dir):
    cfg = load_config(
        "gat_plain",
        configs_dir,
        sets=["train.lr=0.0005", "train.epochs=3", "model.use_edge_weight=true"],
        seed=4,
    )
    assert cfg.train.lr == 0.0005 and isinstance(cfg.train.lr, float)
    assert cfg.train.epochs == 3
    assert cfg.model.use_edge_weight is True
    assert cfg.seed == 4


def test_set_wins_over_experiment_override(configs_dir):
    assert load_config("gat_plain", configs_dir, sets=["train.epochs=1"]).train.epochs == 1


def test_unknown_override_path_raises(configs_dir):
    with pytest.raises(ConfigError):
        load_config("mf", configs_dir, sets=["train.nope=1"])
    with pytest.raises(ConfigError):
        load_config("mf", configs_dir, sets=["graph.weighting=rating"])  # graph is None


def test_malformed_set_raises():
    with pytest.raises(ConfigError):
        parse_set(["no_equals_sign"])


def test_env_interpolation_and_lazy_failure(configs_dir, monkeypatch):
    monkeypatch.delenv("EXTRACTOR_DEPLOYMENT", raising=False)
    cfg = load_config("mf", configs_dir)  # loads fine without the variable
    with pytest.raises(ConfigError, match="EXTRACTOR_DEPLOYMENT"):
        require_resolved(cfg.enrichment.extractor)

    monkeypatch.setenv("EXTRACTOR_DEPLOYMENT", "gpt-x")
    cfg = load_config("mf", configs_dir)
    assert cfg.enrichment.extractor.deployment == "gpt-x"
    require_resolved(cfg.enrichment.extractor)


def test_yaml_round_trip(configs_dir):
    cfg = load_config("gat_plain", configs_dir)
    assert Config.from_yaml(cfg.to_yaml()) == cfg


REAL_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENTS = sorted(p.stem for p in (REAL_CONFIGS / "experiments").glob("*.yaml"))


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_every_real_experiment_loads(exp):
    cfg = load_config(exp, REAL_CONFIGS)
    assert cfg.name == exp
    assert cfg.model.name in {"mf", "lightgcn", "gat", "kg_gat"}


def test_expected_experiments_present():
    assert set(EXPERIMENTS) == {
        "mf",
        "lightgcn",
        "gat_plain",
        "gat_rating",
        "gat_enriched",
        "gat_enriched_judged",
        "kg_gat",
        "kg_gat_unjudged",
    }


def test_judge_filter_only_on_judged_experiment():
    assert load_config("gat_enriched", REAL_CONFIGS).graph.min_judge_score is None
    assert load_config("gat_enriched_judged", REAL_CONFIGS).graph.min_judge_score == 0.7
