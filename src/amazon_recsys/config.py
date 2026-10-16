"""Typed configuration: YAML component files -> one validated, resolved `Config`."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError


class ConfigError(ValueError):
    """Raised for invalid configuration, overrides or unresolved env vars."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- base -----------------------------------------------------------------------------------


class PathsCfg(_Strict):
    data_dir: str = "data"
    runs_dir: str = "runs"
    reports_dir: str = "reports"
    prompts_dir: str = "prompts"


# --- domain ---------------------------------------------------------------------------------


class ColumnsCfg(_Strict):
    user: str
    item: str
    rating: str
    text: str
    title: str
    time: str


class FilterCfg(_Strict):
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    k_core: bool = True
    max_users: int | None = None
    sample_seed: int = 42
    sample_strategy: Literal["random", "most_active"] = "random"
    # Item minimum for the k-core that follows sampling; null reuses min_item_interactions.
    # A small user sample cannot have 5 users per item, so this is usually lower.
    min_item_interactions_after_sample: int | None = None


class DomainCfg(_Strict):
    name: str
    reviews_file: str
    meta_file: str
    columns: ColumnsCfg
    prompt_dir: str
    filter: FilterCfg = FilterCfg()


# --- enrichment -----------------------------------------------------------------------------


class ExtractorCfg(_Strict):
    provider: Literal["azure", "openrouter"] = "azure"
    # Azure: the deployment name. OpenRouter: the model slug (e.g. mistralai/mistral-small-...).
    deployment: str = "${EXTRACTOR_DEPLOYMENT}"
    prompt_version: str = "v1"
    temperature: float = 0.0
    max_tokens: int = 600
    batch_size: int = 20
    concurrency: int = 1  # parallel API calls; results are identical to sequential
    price_per_million_input_tokens: float = 0.0


class JudgeCfg(_Strict):
    provider: str = "typesafe"
    model: str = "jev-1.13.0"
    questions_version: str = "v1"
    max_state_chars: int = 6000
    max_aspects_per_request: int = 15
    weights: dict[str, float] = {
        "faithfulness": 0.35,
        "polarity_accuracy": 0.25,
        "completeness": 0.15,
        "sentiment_agreement": 0.15,
        "format_compliance": 0.10,
    }
    pass_threshold: float = 0.7
    min_confidence: float = 0.5
    sample_rate: float = 1.0
    price_per_million_input_tokens: float = 0.042
    concurrency: int = 1  # parallel API calls; results are identical to sequential


class RetryCfg(_Strict):
    max_attempts: int = 5
    backoff_seconds: float = 2.0


class RateLimitCfg(_Strict):
    requests_per_minute: int = 300


class EnrichmentCfg(_Strict):
    extractor: ExtractorCfg = ExtractorCfg()
    judge: JudgeCfg = JudgeCfg()
    retry: RetryCfg = RetryCfg()
    rate_limit: RateLimitCfg = RateLimitCfg()


# --- split / graph / model / train / eval ---------------------------------------------------


class SplitCfg(_Strict):
    strategy: str = "leave_one_out_time"
    seed: int = 0
    min_user_interactions: int = 3  # users with fewer interactions stay entirely in train


class KGEmbeddingCfg(_Strict):
    model: str = "TransE"
    dim: int = 64
    epochs: int = 50


class GraphCfg(_Strict):
    name: str  # registry key: "bipartite", "knowledge", ...
    # bipartite
    weighting: str = "none"  # none | rating | rating_sentiment
    alpha: float = 0.5
    min_judge_score: float | None = None
    accept_uncertain: bool = False
    # knowledge
    relations: list[str] = [
        "likes",
        "dislikes",
        "has_positive_aspect",
        "has_negative_aspect",
        "cares_about",
    ]
    like_threshold: float = 4.0  # rating >= this -> likes
    dislike_threshold: float = 2.0  # rating <= this -> dislikes
    judged_only: bool = True  # False keeps every extracted aspect (ablation: no judge)
    min_aspect_support: float = 0.7
    min_polarity_confidence: float = 0.7
    min_aspect_count: int = 1  # reviews that must mention an aspect for an item-level edge
    min_user_aspect_count: int = 1  # reviews by a user that must mention it for cares_about
    embedding: KGEmbeddingCfg = KGEmbeddingCfg()


class ModelCfg(_Strict):
    name: str  # registry key: "mf", "lightgcn", "gat", "kg_gat"
    dim: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    use_edge_weight: bool = False
    freeze: bool = False


class TrainCfg(_Strict):
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 1024
    n_neg: int = 1
    loss: str = "bpr"
    eval_every: int = 1
    patience: int = 10
    metric: str = "ndcg@10"
    grad_clip: float | None = 5.0


class EvalCfg(_Strict):
    ks: list[int] = [5, 10, 20]


# --- resolved config ------------------------------------------------------------------------


class Config(_Strict):
    name: str
    seed: int = 0
    default_domain: str | None = None
    device: str = "cpu"
    paths: PathsCfg = PathsCfg()
    domain: DomainCfg
    enrichment: EnrichmentCfg = EnrichmentCfg()
    split: SplitCfg = SplitCfg()
    graph: GraphCfg | None = None
    model: ModelCfg
    train: TrainCfg = TrainCfg()
    eval: EvalCfg = EvalCfg()

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> Config:
        return cls.model_validate(yaml.safe_load(text))


class PipelineConfig(_Strict):
    """Configuration for data-stage commands, which need no model or experiment."""

    seed: int = 0
    default_domain: str | None = None
    device: str = "cpu"
    paths: PathsCfg = PathsCfg()
    domain: DomainCfg
    enrichment: EnrichmentCfg = EnrichmentCfg()
    split: SplitCfg = SplitCfg()


class ExperimentFile(_Strict):
    domain: str
    model: str
    graph: str | None = None
    overrides: dict[str, Any] = {}


# --- env interpolation ----------------------------------------------------------------------

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(obj: Any) -> Any:
    """Replace ${VAR} with its value when set; leave the placeholder otherwise."""
    if isinstance(obj, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def require_resolved(section: BaseModel) -> None:
    """Raise if any string in `section` still holds an unresolved ${VAR}."""
    missing = sorted({m for s in _strings(section.model_dump()) for m in _ENV.findall(s)})
    if missing:
        raise ConfigError(f"Environment variables not set: {', '.join(missing)}")


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


# --- loading --------------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must be a mapping: {path}")
    return data


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Set dotted-path overrides on a fully dumped config dict; unknown paths raise."""
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node: Any = data
        for part in parents:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"Unknown override path: {dotted}")
            node = node[part]
        if not isinstance(node, dict) or leaf not in node:
            raise ConfigError(f"Unknown override path: {dotted}")
        node[leaf] = value
    return data


def parse_set(items: Iterable[str]) -> dict[str, Any]:
    """Parse ["a.b=1", "c=true"] into {"a.b": 1, "c": True} using YAML scalar rules."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"Override must look like key=value, got: {item!r}")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def load_config(
    exp: str,
    configs_dir: str | Path = "configs",
    sets: Iterable[str] = (),
    seed: int | None = None,
) -> Config:
    """Resolve experiment `exp` into a validated Config.

    Order of precedence (low to high): component files, experiment `overrides`, `--set`, `--seed`.
    """
    root = Path(configs_dir)
    try:
        experiment = ExperimentFile.model_validate(_read_yaml(root / "experiments" / f"{exp}.yaml"))
        base = _read_yaml(root / "base.yaml")
        data: dict[str, Any] = {
            "name": exp,
            **base,
            "domain": _read_yaml(root / "domains" / f"{experiment.domain}.yaml"),
            "enrichment": _read_yaml(root / "enrichment.yaml"),
            "split": _read_yaml(root / "split.yaml"),
            "model": {
                "name": experiment.model,
                **_read_yaml(root / "models" / f"{experiment.model}.yaml"),
            },
            "train": _read_yaml(root / "train.yaml"),
            "eval": _read_yaml(root / "eval.yaml"),
        }
        if experiment.graph:
            data["graph"] = {
                "name": experiment.graph,
                **_read_yaml(root / "graphs" / f"{experiment.graph}.yaml"),
            }
        # Expand after validation so schema defaults containing ${VAR} are expanded too.
        resolved = _expand(Config.model_validate(data).model_dump())
        overrides = {**experiment.overrides, **parse_set(sets)}
        if seed is not None:
            overrides["seed"] = seed
        return Config.model_validate(apply_overrides(resolved, overrides))
    except ValidationError as err:
        raise ConfigError(str(err)) from err


def load_pipeline_config(
    domain: str, configs_dir: str | Path = "configs", sets: Iterable[str] = ()
) -> PipelineConfig:
    """Resolve the data-stage configuration for `domain` (no experiment needed)."""
    root = Path(configs_dir)
    try:
        data = {
            **_read_yaml(root / "base.yaml"),
            "domain": _read_yaml(root / "domains" / f"{domain}.yaml"),
            "enrichment": _read_yaml(root / "enrichment.yaml"),
            "split": _read_yaml(root / "split.yaml"),
        }
        resolved = _expand(PipelineConfig.model_validate(data).model_dump())
        return PipelineConfig.model_validate(apply_overrides(resolved, parse_set(sets)))
    except ValidationError as err:
        raise ConfigError(str(err)) from err


def resolve_domain(domain: str | None, configs_dir: str | Path = "configs") -> str:
    """`domain` if given, otherwise `default_domain` from base.yaml."""
    if domain:
        return domain
    default = _read_yaml(Path(configs_dir) / "base.yaml").get("default_domain")
    if not default:
        raise ConfigError("No --domain given and base.yaml has no default_domain")
    return str(default)
