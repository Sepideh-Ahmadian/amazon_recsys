import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from amazon_recsys.cli import app
from amazon_recsys.data.schemas import REVIEWS
from amazon_recsys.utils.io import write_parquet

CONFIGS = str(Path(__file__).resolve().parents[1] / "configs")
runner = CliRunner(mix_stderr=False)

COMMANDS = [
    "ingest",
    "filter",
    "enrich",
    "judge",
    "calibrate",
    "interactions",
    "split",
    "train",
    "kg-embed",
    "evaluate",
]


def _invoke(*args):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_has_help(command):
    assert "Usage" in _invoke(command, "--help").output


def test_top_level_help_lists_commands():
    out = _invoke("--help").output
    for command in COMMANDS:
        assert command in out


def test_ingest_command(tmp_paths, raw_jsonl_gz):
    out = _invoke("ingest", "--configs-dir", CONFIGS, "--root", tmp_paths.root)
    assert json.loads(out.stdout)["reviews"]["rows"] == raw_jsonl_gz.n_reviews
    assert tmp_paths.reviews.exists() and tmp_paths.items.exists()


def _core_reviews(n_users=12, n_items=6) -> pd.DataFrame:
    rows = [
        {
            "review_id": f"r{u}_{i}",
            "domain": "all_beauty",
            "user_id": f"u{u}",
            "item_id": f"i{i}",
            "rating": float(1 + (u + i) % 5),
            "title": None,
            "text": "x",
            "timestamp": 1_600_000_000_000 + 1000 * i + u,
        }
        for u in range(n_users)
        for i in range(n_items)
    ]
    return pd.DataFrame(rows).astype({"timestamp": "int64"})


def test_filter_interactions_split_train_evaluate_chain(tmp_paths):
    write_parquet(_core_reviews(), tmp_paths.reviews, REVIEWS)
    base = ["--configs-dir", CONFIGS, "--root", tmp_paths.root]

    assert json.loads(_invoke("filter", *base).stdout)["final"]["users"] == 12
    assert json.loads(_invoke("interactions", *base).stdout)["rows"] == 72
    manifest = json.loads(_invoke("split", *base).stdout)
    assert manifest["counts"] == {"train": 48, "val": 12, "test": 12}

    out = _invoke("train", "--exp", "mf", "--set", "train.epochs=1", "--seed", "3", *base)
    result = json.loads(out.stdout)
    run_dir = Path(result["run_dir"])
    assert run_dir.is_dir() and (run_dir / "metrics.json").exists()
    assert result["epochs_run"] == 1
    assert "seed: 3" in (run_dir / "config.yaml").read_text()

    evaluated = json.loads(_invoke("evaluate", "--run", run_dir, "--root", tmp_paths.root).stdout)
    assert evaluated["test"]["ndcg@10"] == pytest.approx(result["test"]["ndcg@10"])
    assert (run_dir / "evaluation.json").exists()


def test_unknown_override_fails_cleanly(tmp_paths):
    result = runner.invoke(
        app,
        ["train", "--exp", "mf", "--set", "train.nope=1", "--configs-dir", CONFIGS,
         "--root", str(tmp_paths.root)],
    )  # fmt: skip
    assert result.exit_code != 0


def test_enrich_dry_run_prints_estimate_and_makes_no_client_calls(tmp_paths, mocker):
    import shutil

    import pandas as pd

    from amazon_recsys.data.schemas import ITEMS
    from amazon_recsys.enrichment import llm_client

    shutil.copytree(Path(CONFIGS).parent / "prompts" / "all_beauty", tmp_paths.prompt_dir)
    reviews = _core_reviews(2, 3)
    write_parquet(reviews, tmp_paths.reviews_filtered, REVIEWS)
    items = pd.DataFrame(
        {
            "item_id": sorted(reviews.item_id.unique()),
            "domain": "all_beauty",
            "title": "t",
            "category": None,
            "store": None,
            "description": None,
        }
    )
    write_parquet(items, tmp_paths.items, ITEMS)
    no_client = mocker.patch.object(
        llm_client.AzureOpenAIClient, "from_env", side_effect=AssertionError("client created")
    )

    out = _invoke("enrich", "--dry-run", "--configs-dir", CONFIGS, "--root", tmp_paths.root)
    report = json.loads(out.stdout)
    assert report["rows_total"] == 6 and report["rows_uncached"] == 6
    assert report["estimated_input_tokens"] > 0 and "estimated_cost" in report
    no_client.assert_not_called()


def test_judge_dry_run_prints_estimate_and_makes_no_client_calls(tmp_paths, mocker):
    import shutil

    import pandas as pd

    from amazon_recsys.data.schemas import ASPECTS, ENRICHMENT
    from amazon_recsys.enrichment import jev_client

    shutil.copytree(Path(CONFIGS).parent / "prompts" / "all_beauty", tmp_paths.prompt_dir)
    reviews = _core_reviews(2, 3)
    write_parquet(reviews, tmp_paths.reviews_filtered, REVIEWS)
    enrichment = pd.DataFrame(
        {
            "review_id": reviews.review_id,
            "domain": "all_beauty",
            "status": "ok",
            "overall_sentiment": 0.2,
            "extractor_model": "m",
            "prompt_version": "v1",
            "prompt_hash": "h",
        }
    )
    write_parquet(enrichment, tmp_paths.processed("enrichment"), ENRICHMENT)
    write_parquet(
        pd.DataFrame(
            {
                "review_id": reviews.review_id.iloc[:3],
                "domain": "all_beauty",
                "aspect": "scent",
                "aspect_raw": "Scent",
                "polarity": pd.array([1, 1, -1], dtype="int64"),
                "intensity": 0.5,
                "evidence": "x",
            }
        ),
        tmp_paths.processed("aspects"),
        ASPECTS,
    )
    ctor = mocker.patch.object(jev_client.TypeSafeJevClient, "__init__", return_value=None)

    out = _invoke("judge", "--dry-run", "--configs-dir", CONFIGS, "--root", tmp_paths.root)
    report = json.loads(out.stdout)
    assert report["rows_total"] == 6 and report["rows_uncached"] == 6
    assert report["requests"] == 6 and report["estimated_input_tokens"] > 0
    ctor.assert_not_called()


def test_env_file_is_loaded_but_never_overrides_existing_variables(tmp_path, monkeypatch):
    import os

    from amazon_recsys.cli import load_env

    monkeypatch.delenv("RECSYS_TEST_KEY_A", raising=False)
    monkeypatch.setenv("RECSYS_TEST_KEY_B", "from-shell")
    (tmp_path / ".env").write_text("RECSYS_TEST_KEY_A=from-file\nRECSYS_TEST_KEY_B=from-file\n")
    load_env(tmp_path)
    assert os.environ["RECSYS_TEST_KEY_A"] == "from-file"
    assert os.environ["RECSYS_TEST_KEY_B"] == "from-shell"
    monkeypatch.delenv("RECSYS_TEST_KEY_A")  # keep the process environment clean

    load_env(tmp_path / "missing")  # no .env there: silently nothing
