import logging
import subprocess
from datetime import datetime

import pandas as pd
import pytest
import torch

from amazon_recsys.utils import io
from amazon_recsys.utils.logging import setup_logging


def test_parquet_round_trip_preserves_dtypes(tmp_path):
    df = pd.DataFrame(
        {
            "s": ["a", "b"],
            "i": pd.array([1, 2], dtype="int64"),
            "f": [0.5, 1.5],
            "b": [True, False],
            "n": pd.array([1, None], dtype="Int64"),
        }
    )
    io.write_parquet(df, tmp_path / "sub" / "t.parquet")
    out = io.read_parquet(tmp_path / "sub" / "t.parquet")
    pd.testing.assert_frame_equal(df, out)


def test_parquet_uses_schema(tmp_path):
    class Reject:
        def validate(self, df):
            raise ValueError("bad frame")

    with pytest.raises(ValueError):
        io.write_parquet(pd.DataFrame({"a": [1]}), tmp_path / "t.parquet", Reject())
    assert not (tmp_path / "t.parquet").exists()


def test_json_round_trip(tmp_path):
    obj = {"b": 1, "a": [1, 2, {"x": 0.5}]}
    io.write_json(obj, tmp_path / "d" / "o.json")
    assert io.read_json(tmp_path / "d" / "o.json") == obj


def test_checkpoint_round_trip(tmp_path):
    state = {"epoch": 3, "w": torch.arange(4.0)}
    io.save_checkpoint(state, tmp_path / "c" / "ckpt.pt")
    out = io.load_checkpoint(tmp_path / "c" / "ckpt.pt")
    assert out["epoch"] == 3 and torch.equal(out["w"], state["w"])


def test_run_dir_contains_config_and_git_hash(tmp_path, mocker):
    mocker.patch.object(
        io.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="abc123\n")
    )
    run_dir = io.make_run_dir(tmp_path / "runs", "r1", "seed: 0\n")
    assert (run_dir / "config.yaml").read_text() == "seed: 0\n"
    assert (run_dir / "git_hash.txt").read_text().strip() == "abc123"


def test_git_hash_unknown_when_git_unavailable(mocker):
    mocker.patch.object(io.subprocess, "run", side_effect=FileNotFoundError)
    assert io.git_hash() == "unknown"


def test_run_id_is_deterministic_given_clock():
    assert io.make_run_id("mf", 2, datetime(2026, 1, 2, 3, 4, 5)) == "mf_seed2_20260102T030405"


def test_logging_writes_file_and_is_idempotent(tmp_path):
    log = tmp_path / "run" / "log.txt"
    logger = setup_logging("recsys_test_logger", log)
    setup_logging("recsys_test_logger", log)
    logger.info("hello")
    for h in logger.handlers:
        h.flush()
    assert log.read_text().count("hello") == 1
    assert len(logger.handlers) == 2
    assert logger.level == logging.INFO


def test_run_dir_never_collides(tmp_path, mocker):
    mocker.patch.object(io, "git_hash", return_value="abc")
    first = io.make_run_dir(tmp_path / "runs", "mf_seed0_x", "a: 1\n")
    second = io.make_run_dir(tmp_path / "runs", "mf_seed0_x", "a: 2\n")
    third = io.make_run_dir(tmp_path / "runs", "mf_seed0_x", "a: 3\n")
    assert len({first, second, third}) == 3
    assert (first / "config.yaml").read_text() == "a: 1\n"
    assert second.name == "mf_seed0_x-2" and third.name == "mf_seed0_x-3"
