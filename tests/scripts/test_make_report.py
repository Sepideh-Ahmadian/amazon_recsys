import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_report.py"
spec = importlib.util.spec_from_file_location("make_report", SCRIPT)
make_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_report)


def _run(root, name, experiment, seed, test_ndcg, test_mrr=0.5, stamp="20260101T000000"):
    d = root / f"{name}_{stamp}"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "seed": seed,
                "val": {"ndcg@10": 0.0, "mrr": 0.0},
                "test": {"ndcg@10": test_ndcg, "mrr": test_mrr, "recall@10": test_ndcg / 2},
            }
        )
    )


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    for seed, v in enumerate([0.1, 0.2, 0.3]):
        _run(root, f"gat_plain_s{seed}", "gat_plain", seed, v)
    for seed, v in enumerate([0.4, 0.5, 0.6]):
        _run(root, f"mf_s{seed}", "mf", seed, v, test_mrr=0.25)
    return root


def test_mean_and_std_over_three_seeds(runs):
    summary = make_report.summarise(make_report.load_runs(runs), ["ndcg@10", "mrr"])
    by_exp = summary.set_index("experiment")
    assert by_exp.loc["gat_plain", "ndcg@10_mean"] == pytest.approx(0.2)
    assert by_exp.loc["gat_plain", "ndcg@10_std"] == pytest.approx(0.1)  # sample std
    assert by_exp.loc["mf", "ndcg@10_mean"] == pytest.approx(0.5)
    assert by_exp.loc["mf", "mrr_mean"] == pytest.approx(0.25)
    assert by_exp["n_seeds"].tolist() == [3, 3]
    assert summary["experiment"].tolist() == ["mf", "gat_plain"]  # table order, not alphabetical


def test_report_files_and_formatted_cells(runs, tmp_path):
    paths = make_report.make_report(runs, tmp_path / "reports", metrics=["ndcg@10", "mrr"])
    text = paths["markdown"].read_text()
    assert "| gat_plain | 3 | 0.200 ± 0.100 |" in text
    assert "| mf | 3 | 0.500 ± 0.100 |" in text
    assert pd.read_csv(paths["csv"]).shape[0] == 2
    assert paths["plot"].exists() and paths["plot"].stat().st_size > 0


def test_latest_run_wins_when_an_experiment_seed_pair_repeats(runs, tmp_path):
    _run(runs, "mf_s0_again", "mf", 0, 0.9, stamp="20260202T000000")  # newer duplicate
    df = make_report.load_runs(runs)
    mf = df[df.experiment == "mf"].set_index("seed")
    assert len(mf) == 3 and mf.loc[0, "ndcg@10"] == pytest.approx(0.9)


def test_single_seed_has_no_std_and_empty_runs_fail_clearly(tmp_path):
    root = tmp_path / "runs"
    _run(root, "kg_s0", "kg_gat", 0, 0.3)
    summary = make_report.summarise(make_report.load_runs(root), ["ndcg@10"])
    assert pd.isna(summary.loc[0, "ndcg@10_std"])
    assert "| kg_gat | 1 | 0.300 |" in make_report.markdown_table(summary, ["ndcg@10"])
    with pytest.raises(SystemExit):
        make_report.make_report(tmp_path / "nothing", tmp_path / "out")
