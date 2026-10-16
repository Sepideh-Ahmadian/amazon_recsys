"""Check Jev's probabilities against human labels and pick thresholds from data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABEL_COLUMNS = ["supported", "polarity", "completeness", "format_ok"]
EXPORT_COLUMNS = [
    "review_id", "rating", "title", "text", "aspect", "aspect_raw", "evidence", *LABEL_COLUMNS,
]  # fmt: skip
_YES = {"yes", "y", "1", "true", "t"}
_NO = {"no", "n", "0", "false", "f"}
_POLARITIES = ("positive", "negative", "neutral")
N_BINS = 10


# --- export ---------------------------------------------------------------------------------


def _rating_bucket(rating: float) -> str:
    return "low" if rating <= 2 else "mid" if rating < 4 else "high"


def _confidence_bucket(confidence: float) -> str:
    return "low" if confidence < 0.5 else "mid" if confidence < 0.8 else "high"


def export_sample(
    reviews: pd.DataFrame,
    judgements: pd.DataFrame,
    aspects: pd.DataFrame,
    n: int = 150,
    seed: int = 0,
) -> pd.DataFrame:
    """Stratified review sample (verdict x rating x confidence) with one row per aspect.

    Label columns are empty and Jev's outputs are deliberately omitted, so labelling is blind.
    """
    judged = judgements[judgements["judge_status"].isin(["ok", "no_aspects"])]
    judged = judged.dropna(subset=["verdict", "confidence"])
    frame = judged.merge(reviews[["review_id", "rating", "title", "text"]], on="review_id")
    frame["stratum"] = [
        (v, _rating_bucket(r), _confidence_bucket(c))
        for v, r, c in zip(frame["verdict"], frame["rating"], frame["confidence"], strict=True)
    ]

    rng = np.random.default_rng(seed)
    pools: dict[tuple, list[str]] = {}
    for stratum, group in frame.groupby("stratum"):
        ids = sorted(group["review_id"])
        rng.shuffle(ids)
        pools[stratum] = ids
    chosen: list[str] = []
    while len(chosen) < n and any(pools.values()):  # round-robin across strata
        for stratum in sorted(pools):
            if pools[stratum] and len(chosen) < n:
                chosen.append(pools[stratum].pop())

    by_review = frame.set_index("review_id")
    rows: list[dict[str, Any]] = []
    asp = aspects.sort_values(["review_id", "aspect"])
    for review_id in sorted(chosen):
        base = by_review.loc[review_id]
        mine = asp[asp["review_id"] == review_id]
        common = {
            "review_id": review_id,
            "rating": base["rating"],
            "title": base["title"],
            "text": base["text"],
        }
        if mine.empty:
            rows.append({**common, "aspect": "", "aspect_raw": "", "evidence": ""})
        for r in mine.itertuples():
            rows.append(
                {**common, "aspect": r.aspect, "aspect_raw": r.aspect_raw, "evidence": r.evidence}
            )
    out = pd.DataFrame(rows, columns=EXPORT_COLUMNS[: -len(LABEL_COLUMNS)])
    for col in LABEL_COLUMNS:
        out[col] = ""
    return out[EXPORT_COLUMNS]


# --- labels ---------------------------------------------------------------------------------


def _parse_yes_no(value: Any, column: str) -> float:
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    text = str(value).strip().lower()
    if text in _YES:
        return 1.0
    if text in _NO:
        return 0.0
    raise ValueError(f"Column {column!r}: unrecognised label {value!r} (use yes/no)")


def load_labels(path: str | Path) -> pd.DataFrame:
    """Read a labelled CSV; blank labels become NaN, bad values raise."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["supported"] = [_parse_yes_no(v, "supported") for v in df["supported"]]
    df["format_ok"] = [_parse_yes_no(v, "format_ok") for v in df["format_ok"]]
    pol = df["polarity"].str.strip().str.lower().replace("", np.nan)
    bad = set(pol.dropna()) - set(_POLARITIES)
    if bad:
        raise ValueError(f"Column 'polarity': unrecognised labels {sorted(bad)}")
    df["polarity"] = pol
    df["completeness"] = pd.to_numeric(df["completeness"].replace("", np.nan), errors="raise")
    return df


# --- metrics --------------------------------------------------------------------------------


def cohens_kappa(y_true: Any, y_pred: Any) -> float:
    a, b = np.asarray(y_true), np.asarray(y_pred)
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in np.union1d(a, b))
    return 1.0 if pe == 1.0 else (po - pe) / (1.0 - pe)


def reliability_table(confidence: Any, correct: Any, bins: int = N_BINS) -> pd.DataFrame:
    """Per equal-width confidence bin: count, mean confidence, observed accuracy."""
    conf, hit = np.asarray(confidence, float), np.asarray(correct, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        rows.append(
            {
                "bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}",
                "n": int(mask.sum()),
                "mean_confidence": float(conf[mask].mean()) if mask.any() else np.nan,
                "accuracy": float(hit[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(confidence: Any, correct: Any, bins: int = N_BINS) -> float:
    table = reliability_table(confidence, correct, bins)
    table = table[table["n"] > 0]
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["mean_confidence"] - table["accuracy"]).abs()).sum())


def binary_metrics(y_true: Any, p_yes: Any) -> dict[str, Any]:
    """Accuracy (at 0.5), kappa, Brier and ECE for a yes/no question."""
    y, p = np.asarray(y_true, float), np.asarray(p_yes, float)
    pred = (p >= 0.5).astype(float)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "accuracy": float(np.mean(pred == y)),
        "kappa": cohens_kappa(y, pred),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(p, y),  # bins on P(yes); gap to observed yes-rate
        "reliability": reliability_table(p, y),
    }


def multiclass_metrics(y_true: Any, probs: pd.DataFrame) -> dict[str, Any]:
    """Accuracy, kappa, multiclass Brier and top-label ECE for the Choice question."""
    classes = list(probs.columns)
    pred = probs.to_numpy().argmax(axis=1)
    pred_labels = np.array([classes[i] for i in pred])
    y = np.asarray(y_true)
    onehot = np.array([[1.0 if y_i == c else 0.0 for c in classes] for y_i in y])
    top = probs.to_numpy().max(axis=1)
    correct = (pred_labels == y).astype(float)
    return {
        "n": int(len(y)),
        "accuracy": float(correct.mean()),
        "kappa": cohens_kappa(y, pred_labels),
        "brier": float(np.mean(((probs.to_numpy() - onehot) ** 2).sum(axis=1))),
        "ece": expected_calibration_error(top, correct),
        "reliability": reliability_table(top, correct),
    }


# --- scoring --------------------------------------------------------------------------------


def _review_level(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per review from its first non-null review-level labels."""
    grouped = labels.groupby("review_id")[["completeness", "format_ok"]].first()
    return grouped.reset_index()


def score_labels(
    labels: pd.DataFrame,
    judgements: pd.DataFrame,
    aspect_judgements: pd.DataFrame,
    aspects: pd.DataFrame | None = None,
    n_levels: int = 4,
) -> dict[str, Any]:
    """Compare human labels with Jev's answers, per question type."""
    out: dict[str, Any] = {}
    asp = labels[labels["aspect"] != ""].merge(aspect_judgements, on=["review_id", "aspect"])

    sup = asp.dropna(subset=["supported"])
    if len(sup):
        out["supported"] = binary_metrics(sup["supported"], sup["p_supported"])

    pol = asp.dropna(subset=["polarity"])
    if len(pol):
        probs = pol[["p_positive", "p_negative", "p_neutral"]].set_axis(_POLARITIES, axis=1)
        out["polarity"] = multiclass_metrics(pol["polarity"], probs)
        if aspects is not None:
            extracted = aspects[["review_id", "aspect", "polarity"]]
            ext = pol.merge(extracted, on=["review_id", "aspect"], suffixes=("", "_extracted"))
            names = {1: "positive", -1: "negative", 0: "neutral"}
            ext_label = ext["polarity_extracted"].map(names)
            out["extractor_polarity_accuracy"] = float((ext_label == ext["polarity"]).mean())

    rev = _review_level(labels).merge(judgements, on="review_id")
    comp = rev.dropna(subset=["completeness_x", "completeness_y"])
    if len(comp):
        human = comp["completeness_x"] / (n_levels - 1)
        jev = comp["completeness_y"]
        out["completeness"] = {
            "n": int(len(comp)),
            "mae": float((human - jev).abs().mean()),
            "level_accuracy": float(
                (np.rint(jev * (n_levels - 1)) == comp["completeness_x"]).mean()
            ),
        }
    fmt = rev.dropna(subset=["format_ok", "format_compliance"])
    if len(fmt):
        out["format"] = binary_metrics(fmt["format_ok"], fmt["format_compliance"])
    return out


# --- thresholds -----------------------------------------------------------------------------


def _precision_recall(truth: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    tp = float(np.sum(truth & predicted))
    precision = tp / predicted.sum() if predicted.sum() else float("nan")
    recall = tp / truth.sum() if truth.sum() else float("nan")
    return precision, recall


def sweep_support(
    labels: pd.DataFrame, aspect_judgements: pd.DataFrame, grid: np.ndarray | None = None
) -> pd.DataFrame:
    """Precision/recall of `p_supported >= t` for human `supported == yes`."""
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2) if grid is None else grid
    df = labels[labels["aspect"] != ""].merge(aspect_judgements, on=["review_id", "aspect"])
    df = df.dropna(subset=["supported"])
    truth = df["supported"].to_numpy() == 1.0
    rows = []
    for t in grid:
        precision, recall = _precision_recall(truth, df["p_supported"].to_numpy() >= t)
        rows.append({"threshold": float(t), "precision": precision, "recall": recall,
                     "kept": int((df["p_supported"] >= t).sum())})  # fmt: skip
    return pd.DataFrame(rows)


def review_is_good(
    labels: pd.DataFrame, n_levels: int = 4, min_completeness: float = 0.66
) -> pd.Series:
    """Ground truth for a review: every labelled aspect supported and completeness high enough."""
    asp = labels[labels["aspect"] != ""].dropna(subset=["supported"])
    supported = asp.groupby("review_id")["supported"].min() == 1.0
    comp = _review_level(labels).set_index("review_id")["completeness"] / (n_levels - 1)
    good = supported & (comp.reindex(supported.index) >= min_completeness)
    return good.dropna()


def sweep_pass_threshold(
    good: pd.Series, judgements: pd.DataFrame, grid: np.ndarray | None = None
) -> pd.DataFrame:
    """Precision/recall of `overall >= t` for 'good extraction' reviews."""
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2) if grid is None else grid
    df = judgements.set_index("review_id").loc[good.index].dropna(subset=["overall"])
    truth = good.loc[df.index].to_numpy(dtype=bool)
    rows = []
    for t in grid:
        precision, recall = _precision_recall(truth, df["overall"].to_numpy() >= t)
        rows.append({"threshold": float(t), "precision": precision, "recall": recall,
                     "kept": int((df["overall"] >= t).sum())})  # fmt: skip
    return pd.DataFrame(rows)


def sweep_min_confidence(
    good: pd.Series, judgements: pd.DataFrame, pass_threshold: float, grid: np.ndarray | None = None
) -> pd.DataFrame:
    """Coverage and precision of `pass` verdicts as `min_confidence` rises."""
    grid = np.round(np.arange(0.0, 1.0, 0.1), 2) if grid is None else grid
    df = judgements.set_index("review_id").loc[good.index].dropna(subset=["overall", "confidence"])
    truth = good.loc[df.index].to_numpy(dtype=bool)
    rows = []
    for t in grid:
        passed = (df["overall"].to_numpy() >= pass_threshold) & (df["confidence"].to_numpy() >= t)
        precision, recall = _precision_recall(truth, passed)
        rows.append({"threshold": float(t), "precision": precision, "recall": recall,
                     "coverage": float(passed.mean())})  # fmt: skip
    return pd.DataFrame(rows)


def recommend(table: pd.DataFrame, target_precision: float = 0.9) -> float | None:
    """Lowest threshold (highest recall) whose precision meets the target; None if none does."""
    ok = table[table["precision"] >= target_precision]
    return None if ok.empty else float(ok["threshold"].min())


# --- report ---------------------------------------------------------------------------------


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append(
            "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
        )
    return "\n".join(lines)


def write_report(
    path: str | Path,
    scores: dict[str, Any],
    sweeps: dict[str, pd.DataFrame],
    recommendations: dict[str, float | None],
    target_precision: float,
) -> None:
    lines = ["# Judge calibration", ""]
    lines += [f"Recommendations are the lowest threshold with precision >= {target_precision}.", ""]
    for name, value in recommendations.items():
        lines.append(f"- **{name}**: {'none reaches the target' if value is None else value}")
    lines += ["", "## Agreement with human labels", ""]
    for name, result in scores.items():
        if isinstance(result, dict):
            scalars = {k: v for k, v in result.items() if k != "reliability"}
            lines.append(f"### {name}")
            lines.append("")
            lines += [
                f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}"
                for k, v in scalars.items()
            ]
            if "reliability" in result:
                lines += ["", "Reliability (confidence bin vs observed accuracy):", ""]
                lines.append(_md_table(result["reliability"]))
            lines.append("")
        else:
            lines += [f"- {name}: {result:.3f}", ""]
    lines += ["## Threshold sweeps", ""]
    for name, table in sweeps.items():
        lines += [f"### {name}", "", _md_table(table), ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# --- domain-level entry point ---------------------------------------------------------------


def calibrate_domain(
    action: str,
    paths: Any,
    cfg: Any,
    csv: str | Path | None = None,
    n: int = 150,
    seed: int = 0,
    target_precision: float = 0.9,
) -> dict[str, Any]:
    """`export` a labelling sheet, `score` labels, or sweep `thresholds` and write the report."""
    from amazon_recsys.data.schemas import (
        ASPECT_JUDGEMENTS,
        ASPECTS,
        JUDGEMENTS,
        REVIEWS,
    )
    from amazon_recsys.utils.io import read_parquet, write_json

    judgements = read_parquet(paths.processed("judgements"), JUDGEMENTS)
    aspects = read_parquet(paths.processed("aspects"), ASPECTS)

    if action == "export":
        reviews = read_parquet(paths.reviews_filtered, REVIEWS)
        sheet = export_sample(reviews, judgements, aspects, n=n, seed=seed)
        out = paths.reports / "judge_calibration_sample.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(out, index=False)
        return {"path": str(out), "reviews": int(sheet["review_id"].nunique()), "rows": len(sheet)}

    if action not in {"score", "thresholds"}:
        raise ValueError(f"Unknown calibrate action {action!r}; use export, score or thresholds")
    if csv is None:
        raise ValueError(f"`calibrate {action}` needs the labelled CSV path")

    aspect_judgements = read_parquet(paths.processed("aspect_judgements"), ASPECT_JUDGEMENTS)
    labels = load_labels(csv)
    scores = score_labels(labels, judgements, aspect_judgements, aspects)
    printable = {
        k: ({m: v for m, v in r.items() if m != "reliability"} if isinstance(r, dict) else r)
        for k, r in scores.items()
    }
    if action == "score":
        write_json(printable, paths.reports / "judge_calibration_scores.json")
        return printable

    good = review_is_good(labels)
    support = sweep_support(labels, aspect_judgements)
    pass_sweep = sweep_pass_threshold(good, judgements)
    recommendations = {
        "min_aspect_support": recommend(support, target_precision),
        "pass_threshold": recommend(pass_sweep, target_precision),
    }
    confidence = sweep_min_confidence(
        good, judgements, recommendations["pass_threshold"] or cfg.enrichment.judge.pass_threshold
    )
    report = paths.reports / "judge_calibration.md"
    write_report(
        report,
        scores,
        {"min_aspect_support": support, "pass_threshold": pass_sweep, "min_confidence": confidence},
        recommendations,
        target_precision,
    )
    return {"report": str(report), "recommendations": recommendations, "scores": printable}
