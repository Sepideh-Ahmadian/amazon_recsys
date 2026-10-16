"""Every filesystem path in the project is resolved here, namespaced by domain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amazon_recsys.config import Config


class PathError(ValueError):
    """Raised when a path would escape the configured root."""


@dataclass(frozen=True)
class Paths:
    root: Path
    domain: str
    data_dir: str = "data"
    runs_dir: str = "runs"
    reports_dir: str = "reports"
    prompts_dir: str = "prompts"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        if not self.domain or Path(self.domain).name != self.domain or self.domain in {".", ".."}:
            raise PathError(f"Invalid domain name: {self.domain!r}")

    @classmethod
    def from_config(cls, cfg: Config, root: str | Path | None = None) -> Paths:
        p = cfg.paths
        return cls(
            root=Path(root) if root is not None else Path.cwd(),
            domain=cfg.domain.name,
            data_dir=p.data_dir,
            runs_dir=p.runs_dir,
            reports_dir=p.reports_dir,
            prompts_dir=p.prompts_dir,
        )

    def _inside(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise PathError(f"Path escapes root {self.root}: {resolved}")
        return resolved

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a project-relative path (e.g. a file named in a domain config)."""
        return self._inside(self.root / relative)

    # --- data ---------------------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self._inside(self.root / self.data_dir / "raw")

    @property
    def interim_dir(self) -> Path:
        return self._inside(self.root / self.data_dir / self.domain / "interim")

    @property
    def processed_dir(self) -> Path:
        return self._inside(self.root / self.data_dir / self.domain / "processed")

    @property
    def embeddings_dir(self) -> Path:
        return self._inside(self.root / self.data_dir / self.domain / "embeddings")

    @property
    def splits_dir(self) -> Path:
        return self.processed_dir / "splits"

    @property
    def reviews(self) -> Path:
        return self.interim_dir / "reviews.parquet"

    @property
    def items(self) -> Path:
        return self.interim_dir / "items.parquet"

    @property
    def ingest_manifest(self) -> Path:
        return self.interim_dir / "ingest_manifest.json"

    @property
    def reviews_filtered(self) -> Path:
        return self.interim_dir / "reviews_filtered.parquet"

    @property
    def filter_summary(self) -> Path:
        return self.interim_dir / "filter_summary.json"

    @property
    def llm_cache(self) -> Path:
        return self.interim_dir / "llm_cache.sqlite"

    def processed(self, table: str) -> Path:
        """Path of a processed table, e.g. `processed("judgements")`."""
        return self.processed_dir / f"{table}.parquet"

    def split(self, name: str) -> Path:
        if name not in {"train", "val", "test"}:
            raise PathError(f"Unknown split: {name!r}")
        return self.splits_dir / f"{name}.parquet"

    @property
    def split_manifest(self) -> Path:
        return self.splits_dir / "manifest.json"

    def embedding(self, config_hash: str) -> Path:
        return self.embeddings_dir / f"{config_hash}.pt"

    # --- prompts / runs / reports -------------------------------------------------------
    @property
    def prompt_dir(self) -> Path:
        return self._inside(self.root / self.prompts_dir / self.domain)

    @property
    def runs_root(self) -> Path:
        return self._inside(self.root / self.runs_dir)

    def run_dir(self, run_id: str) -> Path:
        return self._inside(self.root / self.runs_dir / run_id)

    @property
    def reports(self) -> Path:
        return self._inside(self.root / self.reports_dir)
