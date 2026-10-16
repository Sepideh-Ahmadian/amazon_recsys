"""Parquet / JSON / checkpoint helpers and run-directory creation."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import torch


class _Schema(Protocol):
    def validate(self, df: pd.DataFrame) -> pd.DataFrame: ...


def write_parquet(df: pd.DataFrame, path: str | Path, schema: _Schema | None = None) -> None:
    """Validate (if a schema is given) then write; parent dirs are created."""
    if schema is not None:
        df = schema.validate(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: str | Path, schema: _Schema | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return schema.validate(df) if schema is not None else df


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def git_hash(cwd: str | Path | None = None) -> str:
    """Current commit hash, or "unknown" when git or a repository is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_run_id(exp: str, seed: int, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    return f"{exp}_seed{seed}_{stamp}"


def make_run_dir(runs_root: str | Path, run_id: str, config_yaml: str) -> Path:
    """Create runs/<run_id>/ containing the resolved config and the git hash."""
    runs_root = Path(runs_root)
    run_dir = runs_root / run_id
    suffix = 1
    while run_dir.exists():  # two runs in the same second must not collide
        suffix += 1
        run_dir = runs_root / f"{run_id}-{suffix}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(config_yaml)
    (run_dir / "git_hash.txt").write_text(git_hash() + "\n")
    return run_dir
