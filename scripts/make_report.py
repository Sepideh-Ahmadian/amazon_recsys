"""Aggregate runs/*/metrics.json into Markdown/CSV tables and a plot under reports/.

Usage: python scripts/make_report.py [--runs runs] [--out reports] [--split test]

Each experiment is summarised as mean +/- sample standard deviation (ddof=1) over seeds. If an
experiment/seed pair has several runs, the most recent one (by run directory name) is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_METRICS = ["ndcg@10", "recall@10", "hit_rate@10", "mrr", "ndcg@20", "recall@20"]
ORDER = [
    "mf", "lightgcn", "gat_plain", "gat_rating", "gat_enriched", "gat_enriched_judged",
    "kg_gat", "kg_gat_unjudged",
]  # fmt: skip


def load_runs(runs_dir: str | Path, split: str = "test") -> pd.DataFrame:
    """One row per run: experiment, seed, run_dir and the chosen split's metrics."""
    rows = []
    for path in sorted(Path(runs_dir).glob("*/metrics.json")):
        data = json.loads(path.read_text())
        if "experiment" not in data or split not in data:
            continue
        rows.append(
            {
                "experiment": data["experiment"],
                "seed": data["seed"],
                "run_dir": path.parent.name,
                **data[split],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["experiment", "seed", "run_dir"])
    df = pd.DataFrame(rows).sort_values("run_dir")
    return df.drop_duplicates(["experiment", "seed"], keep="last").reset_index(drop=True)


def summarise(runs: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """mean, std and seed count per experiment, ordered like the experiment table."""
    present = [m for m in metrics if m in runs.columns]
    grouped = runs.groupby("experiment")[present]
    out = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    out["n_seeds"] = runs.groupby("experiment")["seed"].nunique()
    known = [e for e in ORDER if e in out.index]
    rest = sorted(set(out.index) - set(known))
    return out.loc[[*known, *rest]].reset_index()


def markdown_table(summary: pd.DataFrame, metrics: list[str]) -> str:
    present = [m for m in metrics if f"{m}_mean" in summary.columns]
    lines = [
        "| experiment | seeds | " + " | ".join(present) + " |",
        "|---|---|" + "|".join("---" for _ in present) + "|",
    ]
    for row in summary.to_dict("records"):  # not itertuples: it renames "ndcg@10_mean"
        cells = []
        for m in present:
            mean, std = row[f"{m}_mean"], row[f"{m}_std"]
            cells.append(f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} ± {std:.3f}")
        lines.append(f"| {row['experiment']} | {int(row['n_seeds'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_metric(summary: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["experiment"], summary[f"{metric}_mean"],
           yerr=summary[f"{metric}_std"].fillna(0), capsize=4, color="#4c78a8")  # fmt: skip
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} (mean ± std over seeds)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_report(
    runs_dir: str | Path, out_dir: str | Path, split: str = "test", metrics: list[str] | None = None
) -> dict[str, Path]:
    metrics = metrics or DEFAULT_METRICS
    runs = load_runs(runs_dir, split)
    if runs.empty:
        raise SystemExit(f"No runs with a '{split}' section found under {runs_dir}")
    summary = summarise(runs, metrics)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"csv": out / "results.csv", "markdown": out / "results.md"}
    summary.to_csv(paths["csv"], index=False)
    header = f"# Results ({split} split)\n\nmean ± std over seeds (sample std).\n\n"
    paths["markdown"].write_text(header + markdown_table(summary, metrics) + "\n")
    if "ndcg@10_mean" in summary.columns:
        paths["plot"] = out / "ndcg_at_10.png"
        plot_metric(summary, "ndcg@10", paths["plot"])
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args(argv)
    for kind, path in make_report(args.runs, args.out, args.split).items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
