import gzip
import json

import pytest

from amazon_recsys.data.ingest import (
    ingest,
    make_review_id,
    normalise_timestamp,
    parse_review,
)
from amazon_recsys.data.schemas import ITEMS, REVIEWS
from amazon_recsys.utils.hashing import hash_file
from amazon_recsys.utils.io import read_json, read_parquet


def test_counts_match_input_minus_skipped(tiny_cfg, tmp_paths, raw_jsonl_gz):
    manifest = ingest(tiny_cfg.domain, tmp_paths)
    rev = manifest["reviews"]
    assert rev["lines"] == raw_jsonl_gz.n_lines
    assert rev["malformed"] == raw_jsonl_gz.n_malformed
    assert rev["missing_fields"] == raw_jsonl_gz.n_missing
    assert rev["duplicates"] == raw_jsonl_gz.n_duplicates
    assert rev["rows"] == raw_jsonl_gz.n_reviews
    assert (
        rev["rows"] + rev["malformed"] + rev["missing_fields"] + rev["duplicates"] == rev["lines"]
    )

    reviews = read_parquet(tmp_paths.reviews, REVIEWS)
    assert len(reviews) == raw_jsonl_gz.n_reviews
    items = read_parquet(tmp_paths.items, ITEMS)
    assert len(items) == raw_jsonl_gz.n_items
    assert manifest["items"]["malformed"] == 1
    assert manifest["items"]["duplicates"] == 1


def test_sidecar_manifest_has_checksums(tiny_cfg, tmp_paths, raw_jsonl_gz):
    ingest(tiny_cfg.domain, tmp_paths)
    sidecar = read_json(tmp_paths.ingest_manifest)
    assert sidecar["domain"] == "all_beauty"
    assert sidecar["reviews"]["sha256"] == hash_file(raw_jsonl_gz.reviews)
    assert sidecar["items"]["sha256"] == hash_file(raw_jsonl_gz.meta)


def test_timestamps_are_utc_milliseconds(tiny_cfg, tmp_paths, raw_jsonl_gz):
    ingest(tiny_cfg.domain, tmp_paths)
    reviews = read_parquet(tmp_paths.reviews)
    row = reviews[(reviews.user_id == "u3") & (reviews.item_id == "i3")].iloc[0]
    assert row.timestamp == 1_600_000_500_000  # was given in seconds
    assert (reviews.timestamp > 1e11).all()


def test_review_id_deterministic_and_unique(tiny_cfg, tmp_paths, raw_jsonl_gz):
    ingest(tiny_cfg.domain, tmp_paths)
    first = read_parquet(tmp_paths.reviews)
    assert first.review_id.is_unique
    ingest(tiny_cfg.domain, tmp_paths)
    assert read_parquet(tmp_paths.reviews).review_id.tolist() == first.review_id.tolist()
    assert make_review_id("d", "u", "i", 1) == make_review_id("d", "u", "i", 1)
    assert make_review_id("d", "u", "i", 1) != make_review_id("d", "u", "i", 2)
    assert make_review_id("d", "u", "i", 1) != make_review_id("e", "u", "i", 1)


def test_items_category_fallback_and_description(tiny_cfg, tmp_paths, raw_jsonl_gz):
    ingest(tiny_cfg.domain, tmp_paths)
    items = read_parquet(tmp_paths.items).set_index("item_id")
    assert items.loc["i1", "category"] == "All Beauty"
    assert items.loc["i2", "category"] == "Bath"  # main_category missing -> last of categories
    assert items.loc["i1", "description"] == "Great shampoo"
    assert items.loc["i1", "title"] == "Shampoo"  # duplicate keeps the first
    assert items.loc["i2", "description"] is None  # empty list -> null


def test_column_mapping_comes_from_domain_config(tiny_cfg, tmp_paths):
    cols = tiny_cfg.domain.columns.model_copy(update={"user": "reviewer", "item": "sku"})
    domain = tiny_cfg.domain.model_copy(update={"columns": cols})
    reviews_path = tmp_paths.resolve(domain.reviews_file)
    meta_path = tmp_paths.resolve(domain.meta_file)
    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"reviewer": "uX", "sku": "iX", "rating": 3, "timestamp": 1_600_000_000_000}
    with gzip.open(reviews_path, "wt") as fh:
        fh.write(json.dumps(row) + "\n")
    with gzip.open(meta_path, "wt") as fh:
        fh.write(json.dumps({"sku": "iX", "title": "T"}) + "\n")

    ingest(domain, tmp_paths)
    reviews = read_parquet(tmp_paths.reviews)
    assert reviews.loc[0, "user_id"] == "uX" and reviews.loc[0, "item_id"] == "iX"
    assert read_parquet(tmp_paths.items).loc[0, "item_id"] == "iX"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1_600_000_000_000, 1_600_000_000_000),
        (1_600_000_000, 1_600_000_000_000),
        (1_600_000_000.5, 1_600_000_000_500),
        ("1600000000000", 1_600_000_000_000),
        (0, None),
        (-5, None),
        (None, None),
        ("abc", None),
        (float("nan"), None),
    ],
)
def test_normalise_timestamp(value, expected):
    assert normalise_timestamp(value) == expected


def test_parse_review_skip_reasons(tiny_cfg):
    cols = tiny_cfg.domain.columns
    ok = {"user_id": "u", "parent_asin": "i", "rating": 4, "timestamp": 1_600_000_000_000}
    assert isinstance(parse_review(ok, cols, "d"), dict)
    assert parse_review({**ok, "user_id": None}, cols, "d") == "missing_fields"
    assert parse_review({**ok, "rating": 6}, cols, "d") == "invalid_values"
    assert parse_review({**ok, "rating": "bad"}, cols, "d") == "invalid_values"
    assert parse_review({**ok, "timestamp": -1}, cols, "d") == "invalid_values"
