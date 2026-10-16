"""Stream raw Amazon `.jsonl.gz` files into `reviews.parquet` and `items.parquet`."""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from amazon_recsys.config import ColumnsCfg, DomainCfg
from amazon_recsys.data.schemas import ITEMS, REVIEWS
from amazon_recsys.utils.hashing import hash_file, hash_obj
from amazon_recsys.utils.io import write_json, write_parquet
from amazon_recsys.utils.paths import Paths

_SECONDS_CUTOFF = 1e11  # epoch values below this are seconds (1e11 s is year 5138; 1e11 ms is 1973)

REVIEW_COLUMNS = list(REVIEWS.columns)
ITEM_COLUMNS = list(ITEMS.columns)


def make_review_id(domain: str, user_id: str, item_id: str, timestamp: int) -> str:
    """Stable id for a review: hash of (domain, user_id, item_id, timestamp)."""
    return hash_obj([domain, user_id, item_id, timestamp])[:20]


def normalise_timestamp(value: Any) -> int | None:
    """Return UTC milliseconds, accepting epoch seconds or milliseconds; None if invalid."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    if number < _SECONDS_CUTOFF:
        number *= 1000
    return int(round(number))


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _iter_jsonl_gz(path: Path, stats: dict[str, int]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects; count non-empty lines and skip malformed ones."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            stats["lines"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed"] += 1
                continue
            if not isinstance(obj, dict):
                stats["malformed"] += 1
                continue
            yield obj


def parse_review(raw: dict[str, Any], cols: ColumnsCfg, domain: str) -> dict[str, Any] | str:
    """Map a raw record to a review row, or return the skip reason
    ("missing_fields" / "invalid_values")."""
    user = _clean_str(raw.get(cols.user))
    item = _clean_str(raw.get(cols.item))
    if user is None or item is None or raw.get(cols.rating) is None or raw.get(cols.time) is None:
        return "missing_fields"
    try:
        rating = float(raw[cols.rating])
    except (TypeError, ValueError):
        return "invalid_values"
    timestamp = normalise_timestamp(raw[cols.time])
    if timestamp is None or not 1.0 <= rating <= 5.0:
        return "invalid_values"
    return {
        "review_id": make_review_id(domain, user, item, timestamp),
        "domain": domain,
        "user_id": user,
        "item_id": item,
        "rating": rating,
        "title": _clean_str(raw.get(cols.title)),
        "text": _clean_str(raw.get(cols.text)),
        "timestamp": timestamp,
    }


def ingest_reviews(path: Path, cfg: DomainCfg) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"lines": 0, "malformed": 0, "missing_fields": 0, "invalid_values": 0, "duplicates": 0}
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw in _iter_jsonl_gz(path, stats):
        parsed = parse_review(raw, cfg.columns, cfg.name)
        if isinstance(parsed, str):
            stats[parsed] += 1
        elif parsed["review_id"] in seen:
            stats["duplicates"] += 1
        else:
            seen.add(parsed["review_id"])
            rows.append(parsed)
    stats["rows"] = len(rows)
    df = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    return df.astype({"rating": "float64", "timestamp": "int64"}), stats


def _join_text(value: Any) -> str | None:
    if isinstance(value, list):
        return _clean_str(" ".join(str(v) for v in value))
    return _clean_str(value)


def parse_item(raw: dict[str, Any], cols: ColumnsCfg, domain: str) -> dict[str, Any] | None:
    item = _clean_str(raw.get(cols.item))
    if item is None:
        return None
    categories = raw.get("categories")
    fallback = categories[-1] if isinstance(categories, list) and categories else None
    return {
        "item_id": item,
        "domain": domain,
        "title": _clean_str(raw.get("title")),
        "category": _clean_str(raw.get("main_category")) or _clean_str(fallback),
        "store": _clean_str(raw.get("store")),
        "description": _join_text(raw.get("description")),
    }


def ingest_items(path: Path, cfg: DomainCfg) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"lines": 0, "malformed": 0, "missing_fields": 0, "duplicates": 0}
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw in _iter_jsonl_gz(path, stats):
        parsed = parse_item(raw, cfg.columns, cfg.name)
        if parsed is None:
            stats["missing_fields"] += 1
        elif parsed["item_id"] in seen:
            stats["duplicates"] += 1
        else:
            seen.add(parsed["item_id"])
            rows.append(parsed)
    stats["rows"] = len(rows)
    return pd.DataFrame(rows, columns=ITEM_COLUMNS), stats


def ingest(cfg: DomainCfg, paths: Paths) -> dict[str, Any]:
    """Ingest reviews and items for one domain; write parquet + a sidecar manifest."""
    reviews_src = paths.resolve(cfg.reviews_file)
    meta_src = paths.resolve(cfg.meta_file)

    reviews, review_stats = ingest_reviews(reviews_src, cfg)
    items, item_stats = ingest_items(meta_src, cfg)

    write_parquet(reviews, paths.reviews, REVIEWS)
    write_parquet(items, paths.items, ITEMS)

    manifest = {
        "domain": cfg.name,
        "reviews": {"source": cfg.reviews_file, "sha256": hash_file(reviews_src), **review_stats},
        "items": {"source": cfg.meta_file, "sha256": hash_file(meta_src), **item_stats},
    }
    write_json(manifest, paths.ingest_manifest)
    return manifest
