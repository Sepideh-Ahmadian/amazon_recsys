from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amazon_recsys.data.schemas import ASPECT_JUDGEMENTS, ASPECTS, JUDGEMENTS, REVIEWS
from amazon_recsys.enrichment.calibration import (
    binary_metrics,
    calibrate_domain,
    cohens_kappa,
    expected_calibration_error,
    export_sample,
    load_labels,
    recommend,
    review_is_good,
    score_labels,
    sweep_min_confidence,
    sweep_pass_threshold,
    sweep_support,
)
from amazon_recsys.utils.io import write_parquet

LABELS = Path(__file__).resolve().parents[1] / "fixtures" / "calibration_labels.csv"

# Hand-worked fixture: (aspect, p_supported, [p_pos, p_neg, p_neu] or None, extracted polarity)
ASPECT_ROWS = {
    "r1": [("scent", 0.9, (0.7, 0.2, 0.1), 1), ("texture", 0.8, (0.1, 0.8, 0.1), -1)],
    "r2": [("price", 0.7, (0.2, 0.2, 0.6), 0), ("packaging", 0.3, (0.5, 0.3, 0.2), 1)],
    "r3": [("color", 0.2, (0.3, 0.3, 0.4), 1), ("durability", 0.6, (0.3, 0.3, 0.4), 1)],
    "r4": [("absorption", 0.4, (0.3, 0.3, 0.4), 1), ("size", 0.1, (0.3, 0.3, 0.4), -1)],
}
# review: (completeness_norm, format_compliance, overall, confidence)
REVIEW_ROWS = {
    "r1": (0.9, 0.95, 0.9, 0.9),
    "r2": (0.6, 0.8, 0.75, 0.7),
    "r3": (1.0, 0.3, 0.5, 0.4),
    "r4": (2 / 3, 0.6, 0.3, 0.8),
}


@pytest.fixture
def tables():
    aj, asp, jd = [], [], []
    for rid, items in ASPECT_ROWS.items():
        for name, ps, (pp, pn, pu), pol in items:
            aj.append(dict(review_id=rid, domain="d", aspect=name, p_supported=ps,
                           p_polarity_match=0.5, p_positive=pp, p_negative=pn, p_neutral=pu,
                           polarity_confidence=0.5))  # fmt: skip
            asp.append(dict(review_id=rid, domain="d", aspect=name, aspect_raw=name.title(),
                            polarity=pol, intensity=0.5, evidence="ev"))  # fmt: skip
        comp, fmt, overall, conf = REVIEW_ROWS[rid]
        jd.append(dict(review_id=rid, domain="d", judge_status="ok", faithfulness=0.5,
                       polarity_accuracy=0.5, completeness=comp, format_compliance=fmt,
                       sentiment_agreement=0.5, overall=overall, confidence=conf,
                       verdict="pass" if overall >= 0.7 else "fail", truncated=False,
                       jev_model="jev", questions_hash="q", extraction_hash="e",
                       input_tokens=1))  # fmt: skip
    jd = pd.DataFrame(jd).astype({"input_tokens": "Int64"})
    return jd, pd.DataFrame(aj), pd.DataFrame(asp).astype({"polarity": "int64"})


# --- metrics (hand computed) ----------------------------------------------------------------


def test_score_labels_match_hand_computed_values(tables):
    judgements, aspect_judgements, aspects = tables
    out = score_labels(load_labels(LABELS), judgements, aspect_judgements, aspects, n_levels=4)

    sup = out["supported"]
    assert sup["n"] == 8
    assert sup["accuracy"] == pytest.approx(0.75)
    assert sup["kappa"] == pytest.approx(0.5)
    assert sup["brier"] == pytest.approx(0.15)
    assert sup["ece"] == pytest.approx(0.325)

    pol = out["polarity"]
    assert pol["n"] == 4
    assert pol["accuracy"] == pytest.approx(0.75)
    assert pol["kappa"] == pytest.approx(0.3125 / 0.5625)
    assert pol["brier"] == pytest.approx(0.405)
    assert out["extractor_polarity_accuracy"] == pytest.approx(0.75)

    comp = out["completeness"]
    assert comp["mae"] == pytest.approx(0.125)
    assert comp["level_accuracy"] == pytest.approx(0.75)

    fmt = out["format"]
    assert fmt["accuracy"] == 1.0
    assert fmt["brier"] == pytest.approx(0.073125)


def test_kappa_and_ece_edge_cases():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0
    # perfectly calibrated: 80% confidence and right 4 times out of 5
    assert expected_calibration_error([0.8] * 5, [1, 1, 1, 1, 0]) == pytest.approx(0.0)
    assert binary_metrics([1, 0], [1.0, 0.0])["brier"] == 0.0


def test_labels_loading_handles_blanks_and_rejects_bad_values(tmp_path):
    labels = load_labels(LABELS)
    assert labels["supported"].notna().sum() == 8 and labels["polarity"].notna().sum() == 4
    assert labels.loc[labels.review_id == "r1", "completeness"].dropna().tolist() == [3.0]
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "review_id,rating,title,text,aspect,aspect_raw,evidence,supported,polarity,"
        "completeness,format_ok\nr1,4,t,x,scent,Scent,e,maybe,,,\n"
    )
    with pytest.raises(ValueError, match="supported"):
        load_labels(bad)


# --- thresholds -----------------------------------------------------------------------------


def test_support_sweep_values_and_monotone_recall(tables):
    _, aspect_judgements, _ = tables
    sweep = sweep_support(load_labels(LABELS), aspect_judgements)
    at = sweep.set_index("threshold")
    assert at.loc[0.5, "precision"] == pytest.approx(0.75)
    assert at.loc[0.5, "recall"] == pytest.approx(0.75)
    assert at.loc[0.75, "precision"] == 1.0 and at.loc[0.75, "recall"] == pytest.approx(0.5)
    assert sweep["recall"].is_monotonic_decreasing  # recall rises as the threshold falls
    # t=0.7 still admits 'price' (p=0.7, not supported): precision 2/3; 0.75 is the first clean one
    assert at.loc[0.7, "precision"] == pytest.approx(2 / 3)
    assert recommend(sweep, 0.9) == 0.75


def test_pass_threshold_and_confidence_sweeps(tables, tmp_path):
    judgements, _, _ = tables
    good = review_is_good(load_labels(LABELS))
    assert good.to_dict() == {"r1": True, "r2": False, "r3": False, "r4": False}

    sweep = sweep_pass_threshold(good, judgements)
    at = sweep.set_index("threshold")
    assert at.loc[0.85, "precision"] == 1.0 and at.loc[0.85, "recall"] == 1.0
    assert at.loc[0.7, "precision"] == pytest.approx(0.5)
    assert sweep["recall"].is_monotonic_decreasing
    assert recommend(sweep, 0.9) == 0.8

    conf = sweep_min_confidence(good, judgements, pass_threshold=0.7)
    row = conf.set_index("threshold").loc[0.8]
    assert row["coverage"] == pytest.approx(0.25) and row["precision"] == 1.0
    assert recommend(sweep.assign(precision=0.5), 0.9) is None


def test_calibrate_domain_writes_report_and_scores(tables, tiny_cfg, tmp_paths):
    judgements, aspect_judgements, aspects = tables
    write_parquet(judgements, tmp_paths.processed("judgements"), JUDGEMENTS)
    write_parquet(aspect_judgements, tmp_paths.processed("aspect_judgements"), ASPECT_JUDGEMENTS)
    write_parquet(aspects, tmp_paths.processed("aspects"), ASPECTS)

    scored = calibrate_domain("score", tmp_paths, tiny_cfg, LABELS)
    assert scored["supported"]["accuracy"] == pytest.approx(0.75)
    assert (tmp_paths.reports / "judge_calibration_scores.json").exists()

    result = calibrate_domain("thresholds", tmp_paths, tiny_cfg, LABELS)
    report = Path(result["report"]).read_text()
    assert result["recommendations"] == {"min_aspect_support": 0.75, "pass_threshold": 0.8}
    assert "# Judge calibration" in report and "Reliability" in report

    with pytest.raises(ValueError, match="needs the labelled CSV"):
        calibrate_domain("score", tmp_paths, tiny_cfg, None)
    with pytest.raises(ValueError, match="Unknown calibrate action"):
        calibrate_domain("nope", tmp_paths, tiny_cfg, LABELS)


# --- export ---------------------------------------------------------------------------------


@pytest.fixture
def big_tables():
    rng = np.random.default_rng(0)
    n = 90
    ids = [f"r{i:03d}" for i in range(n)]
    reviews = pd.DataFrame(
        {"review_id": ids, "domain": "d", "user_id": ids, "item_id": ids,
         "rating": rng.integers(1, 6, n).astype(float), "title": "t", "text": "text",
         "timestamp": pd.array(range(n), dtype="int64")}
    )  # fmt: skip
    judgements = pd.DataFrame(
        {"review_id": ids, "domain": "d", "judge_status": "ok",
         "verdict": rng.choice(["pass", "fail", "uncertain"], n),
         "confidence": rng.random(n), "overall": rng.random(n)}
    )  # fmt: skip
    judgements.loc[:2, "judge_status"] = "api_error"  # must never be sampled
    aspects = pd.DataFrame(
        [
            {"review_id": rid, "aspect": a, "aspect_raw": a.title(), "evidence": "ev"}
            for rid in ids[3:]
            for a in ("scent", "texture")
        ]
    )
    return reviews, judgements, aspects


def test_export_is_stratified_blind_and_reproducible(big_tables):
    reviews, judgements, aspects = big_tables
    sheet = export_sample(reviews, judgements, aspects, n=30, seed=1)
    assert sheet["review_id"].nunique() == 30
    assert set(sheet["review_id"]).isdisjoint({"r000", "r001", "r002"})
    assert (sheet["supported"] == "").all() and (sheet["completeness"] == "").all()
    assert not {"p_supported", "verdict", "overall", "confidence"} & set(sheet.columns)  # blind
    assert (sheet.groupby("review_id").size() == 2).all()  # one row per aspect

    merged = sheet.drop_duplicates("review_id").merge(judgements, on="review_id")
    merged["r"] = merged["rating"].map(lambda r: "low" if r <= 2 else "mid" if r < 4 else "high")
    merged["c"] = pd.cut(merged["confidence"], [-1, 0.5, 0.8, 2], labels=["low", "mid", "high"])
    present = set(zip(merged["verdict"], merged["r"], merged["c"], strict=True))
    full = judgements[judgements.judge_status == "ok"].merge(reviews, on="review_id")
    full["r"] = full["rating"].map(lambda r: "low" if r <= 2 else "mid" if r < 4 else "high")
    full["c"] = pd.cut(full["confidence"], [-1, 0.5, 0.8, 2], labels=["low", "mid", "high"])
    all_strata = set(zip(full["verdict"], full["r"], full["c"], strict=True))
    assert present == all_strata  # every verdict / rating / confidence stratum is represented

    again = export_sample(reviews, judgements, aspects, n=30, seed=1)
    pd.testing.assert_frame_equal(sheet, again)
    other = export_sample(reviews, judgements, aspects, n=30, seed=2)
    assert set(other["review_id"]) != set(sheet["review_id"])


def test_export_via_domain_entry_point(big_tables, tiny_cfg, tmp_paths):
    reviews, judgements, aspects = big_tables
    full_j = judgements.assign(
        faithfulness=0.5, polarity_accuracy=0.5, completeness=0.5, format_compliance=0.5,
        sentiment_agreement=0.5, truncated=False, jev_model="jev", questions_hash="q",
        extraction_hash="e", input_tokens=pd.array([1] * len(judgements), dtype="Int64"),
    )  # fmt: skip
    write_parquet(reviews, tmp_paths.reviews_filtered, REVIEWS)
    write_parquet(full_j, tmp_paths.processed("judgements"), JUDGEMENTS)
    full_aspects = aspects.assign(
        domain="d", polarity=pd.array([1] * len(aspects), dtype="int64"), intensity=0.5
    )
    write_parquet(full_aspects, tmp_paths.processed("aspects"), ASPECTS)
    out = calibrate_domain("export", tmp_paths, tiny_cfg, n=10)
    assert out["reviews"] == 10 and Path(out["path"]).exists()
