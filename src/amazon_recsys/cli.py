"""Typer CLI. Each command loads config and calls one library function."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from amazon_recsys.config import load_config, load_pipeline_config, resolve_domain
from amazon_recsys.utils.paths import Paths

app = typer.Typer(help="amazon_recsys pipeline", no_args_is_help=True, add_completion=False)

Domain = Annotated[
    str | None,
    typer.Option(help="Domain name (configs/domains/<name>.yaml); default from base.yaml"),
]
ConfigsDir = Annotated[Path, typer.Option(help="Directory holding the YAML configs")]
Root = Annotated[Path, typer.Option(help="Project root that data/ and runs/ live under")]
Sets = Annotated[list[str], typer.Option("--set", help="Override, e.g. train.lr=0.0005")]


def _emit(obj: object) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


def load_env(root: Path) -> None:
    """Load `<root>/.env` into the environment; variables already set are never overridden."""
    load_dotenv(Path(root) / ".env", override=False)


def _pipeline(domain: str | None, configs_dir: Path, root: Path, sets: list[str]):
    load_env(root)
    cfg = load_pipeline_config(resolve_domain(domain, configs_dir), configs_dir, sets)
    return cfg, Paths.from_config(cfg, root=root)


@app.command()
def ingest(
    domain: Domain = None,
    configs_dir: ConfigsDir = Path("configs"),
    root: Root = Path("."),
    sets: Sets = [],  # noqa: B006
) -> None:
    """Stream raw .jsonl.gz files into reviews.parquet and items.parquet."""
    from amazon_recsys.data.ingest import ingest as run

    cfg, paths = _pipeline(domain, configs_dir, root, sets)
    _emit(run(cfg.domain, paths))


@app.command("filter")
def filter_(
    domain: Domain = None,
    configs_dir: ConfigsDir = Path("configs"),
    root: Root = Path("."),
    sets: Sets = [],  # noqa: B006
) -> None:
    """Apply the k-core / user-sample filter to the ingested reviews."""
    from amazon_recsys.data.filter import filter_domain

    cfg, paths = _pipeline(domain, configs_dir, root, sets)
    _emit(filter_domain(cfg.domain, paths))


@app.command()
def enrich(
    domain: Domain = None,
    dry_run: Annotated[
        bool, typer.Option(help="Estimate rows, tokens and cost; no API calls")
    ] = False,
    limit: Annotated[
        int | None, typer.Option(help="Call the API for at most N uncached rows")
    ] = None,
    configs_dir: ConfigsDir = Path("configs"),
    root: Root = Path("."),
    sets: Sets = [],  # noqa: B006
) -> None:
    """Run the extractor LLM over the filtered reviews (cached and resumable)."""
    from amazon_recsys.config import require_resolved
    from amazon_recsys.enrichment.extractor import enrich_domain
    from amazon_recsys.enrichment.llm_client import make_llm_client

    cfg, paths = _pipeline(domain, configs_dir, root, sets)
    client = None
    if not dry_run:
        ext, retry, rate = cfg.enrichment.extractor, cfg.enrichment.retry, cfg.enrichment.rate_limit
        require_resolved(ext)
        client = make_llm_client(
            ext.provider,
            ext.deployment,
            max_attempts=retry.max_attempts,
            backoff_seconds=retry.backoff_seconds,
            requests_per_minute=rate.requests_per_minute,
        )
    _emit(enrich_domain(cfg, paths, client, dry_run=dry_run, limit=limit))


@app.command()
def judge(
    domain: Domain = None,
    dry_run: Annotated[
        bool, typer.Option(help="Estimate requests, tokens and cost; no API calls")
    ] = False,
    limit: Annotated[int | None, typer.Option(help="Judge at most N uncached reviews")] = None,
    configs_dir: ConfigsDir = Path("configs"),
    root: Root = Path("."),
    sets: Sets = [],  # noqa: B006
) -> None:
    """Judge each extraction with Jev (cached and resumable)."""
    from amazon_recsys.enrichment.jev_client import TypeSafeJevClient
    from amazon_recsys.enrichment.judge import judge_domain

    cfg, paths = _pipeline(domain, configs_dir, root, sets)
    client = None
    if not dry_run:
        j, retry, rate = cfg.enrichment.judge, cfg.enrichment.retry, cfg.enrichment.rate_limit
        client = TypeSafeJevClient(
            j.model,
            max_attempts=retry.max_attempts,
            backoff_seconds=retry.backoff_seconds,
            requests_per_minute=rate.requests_per_minute,
        )
    _emit(judge_domain(cfg, paths, client, dry_run=dry_run, limit=limit))


@app.command()
def calibrate(
    action: Annotated[str, typer.Argument(help="export | score | thresholds")],
    csv: Annotated[Path | None, typer.Argument(help="Labelled CSV (score, thresholds)")] = None,
    domain: Domain = None,
    n: Annotated[int, typer.Option(help="Reviews to sample on export")] = 150,
    seed: Annotated[int, typer.Option(help="Sampling seed")] = 0,
    target_precision: Annotated[float, typer.Option(help="Precision a threshold must reach")] = 0.9,
    configs_dir: ConfigsDir = Path("configs"),
    root: Root = Path("."),
    sets: Sets = [],  # noqa: B006
) -> None:
    """Export a labelling sheet, score labels against Jev, or pick thresholds from data."""
    from amazon_recsys.enrichment.calibration import calibrate_domain

    cfg, paths = _pipeline(domain, configs_dir, root, sets)
    _emit(calibrate_domain(action, paths, cfg, csv, n, seed, target_precision))


@app.command()
def interactions(
    domain: Domain = None,
    configs_dir: ConfigsDir = Path("configs"),
    root: Root = Path("."),
    sets: Sets = [],  # noqa: B006
) -> None:
    """Join filtered reviews with enrichment/judgements into interactions.parquet."""
    from amazon_recsys.data.interactions import interactions_domain

    cfg, paths = _pipeline(domain, configs_dir, root, sets)
    _emit({"rows": len(interactions_domain(cfg.domain, paths))})


@app.command()
def split(
    domain: Domain = None,
    configs_dir: ConfigsDir = Path("configs"),
    root: Root = Path("."),
    sets: Sets = [],  # noqa: B006
) -> None:
    """Write leave-one-out train/val/test splits and the manifest."""
    from amazon_recsys.data.splits import split_domain

    cfg, paths = _pipeline(domain, configs_dir, root, sets)
    _emit(split_domain(cfg.split, paths))


@app.command()
def train(
    exp: Annotated[str, typer.Option(help="Experiment name (configs/experiments/<name>.yaml)")],
    seed: Annotated[int | None, typer.Option(help="Override the config seed")] = None,
    configs_dir: ConfigsDir = Path("configs"),
    root: Root = Path("."),
    sets: Sets = [],  # noqa: B006
) -> None:
    """Train and evaluate one experiment; writes runs/<run_id>/."""
    from amazon_recsys.training.trainer import run_experiment

    load_env(root)
    cfg = load_config(exp, configs_dir, sets, seed)
    result = run_experiment(cfg, Paths.from_config(cfg, root=root))
    _emit({k: result[k] for k in ("run_dir", "best_epoch", "epochs_run", "metric", "val", "test")})


@app.command("kg-embed")
def kg_embed(
    exp: Annotated[str, typer.Option(help="Experiment with a knowledge graph, e.g. kg_gat")],
    seed: Annotated[int | None, typer.Option(help="Override the config seed")] = None,
    configs_dir: ConfigsDir = Path("configs"),
    root: Root = Path("."),
    sets: Sets = [],  # noqa: B006
) -> None:
    """Build the knowledge graph and train (or load cached) KG embeddings."""
    from amazon_recsys.training.trainer import embed_experiment

    load_env(root)
    cfg = load_config(exp, configs_dir, sets, seed)
    _emit(embed_experiment(cfg, Paths.from_config(cfg, root=root)))


@app.command()
def evaluate(
    run: Annotated[Path, typer.Option(help="Run directory, e.g. runs/mf_seed0_...")],
    root: Root = Path("."),
) -> None:
    """Re-evaluate a finished run's best model on val and test."""
    from amazon_recsys.config import Config
    from amazon_recsys.training.trainer import evaluate_run

    cfg = Config.from_yaml((run / "config.yaml").read_text())
    _emit(evaluate_run(run, Paths.from_config(cfg, root=root)))


if __name__ == "__main__":  # pragma: no cover
    app()
