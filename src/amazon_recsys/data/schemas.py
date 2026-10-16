"""Pandera schemas for every stage table (CLAUDE.md section 5)."""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Check, Column

_PROB = Check.in_range(0.0, 1.0)


def _str(nullable: bool = False) -> Column:
    return Column(str, nullable=nullable)


def _prob(nullable: bool = False) -> Column:
    return Column(float, _PROB, nullable=nullable)


REVIEWS = pa.DataFrameSchema(
    {
        "review_id": _str(),
        "domain": _str(),
        "user_id": _str(),
        "item_id": _str(),
        "rating": Column(float, Check.in_range(1.0, 5.0)),
        "title": _str(nullable=True),
        "text": _str(nullable=True),
        "timestamp": Column("int64"),  # UTC milliseconds
    },
    strict=True,
)

ITEMS = pa.DataFrameSchema(
    {
        "item_id": _str(),
        "domain": _str(),
        "title": _str(nullable=True),
        "category": _str(nullable=True),
        "store": _str(nullable=True),
        "description": _str(nullable=True),
    },
    strict=True,
)

ENRICHMENT = pa.DataFrameSchema(
    {
        "review_id": _str(),
        "domain": _str(),
        "status": Column(str, Check.isin(["ok", "parse_error", "api_error"])),
        "overall_sentiment": Column(float, Check.in_range(-1.0, 1.0), nullable=True),
        "extractor_model": _str(),
        "prompt_version": _str(),
        "prompt_hash": _str(),
    },
    strict=True,
)

ASPECTS = pa.DataFrameSchema(
    {
        "review_id": _str(),
        "domain": _str(),
        "aspect": _str(),
        "aspect_raw": _str(),
        "polarity": Column("int64", Check.isin([-1, 0, 1])),
        "intensity": _prob(),
        "evidence": _str(nullable=True),
    },
    strict=True,
)

JUDGEMENTS = pa.DataFrameSchema(
    {
        "review_id": _str(),
        "domain": _str(),
        "judge_status": Column(str, Check.isin(["ok", "api_error", "too_large", "no_aspects"])),
        "faithfulness": _prob(nullable=True),
        "polarity_accuracy": _prob(nullable=True),
        "completeness": _prob(nullable=True),
        "format_compliance": _prob(nullable=True),
        "sentiment_agreement": _prob(nullable=True),
        "overall": _prob(nullable=True),
        "confidence": _prob(nullable=True),
        "verdict": Column(str, Check.isin(["pass", "fail", "uncertain"]), nullable=True),
        "truncated": Column(bool),
        "jev_model": _str(nullable=True),
        "questions_hash": _str(),
        "extraction_hash": _str(),
        "input_tokens": Column("Int64", nullable=True),
    },
    strict=True,
)

ASPECT_JUDGEMENTS = pa.DataFrameSchema(
    {
        "review_id": _str(),
        "domain": _str(),
        "aspect": _str(),
        "p_supported": _prob(),
        "p_polarity_match": _prob(),
        "p_positive": _prob(),
        "p_negative": _prob(),
        "p_neutral": _prob(),
        "polarity_confidence": _prob(),
    },
    strict=True,
)

INTERACTIONS = pa.DataFrameSchema(
    {
        "domain": _str(),
        "user_id": _str(),
        "item_id": _str(),
        "rating": Column(float, Check.in_range(1.0, 5.0)),
        "timestamp": Column("int64"),
        "review_id": _str(),
        "overall_sentiment": Column(float, Check.in_range(-1.0, 1.0), nullable=True),
        "judge_overall": _prob(nullable=True),
        "judge_verdict": Column(str, Check.isin(["pass", "fail", "uncertain"]), nullable=True),
    },
    strict=True,
)

SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "reviews": REVIEWS,
    "items": ITEMS,
    "enrichment": ENRICHMENT,
    "aspects": ASPECTS,
    "judgements": JUDGEMENTS,
    "aspect_judgements": ASPECT_JUDGEMENTS,
    "interactions": INTERACTIONS,
    # train/val/test splits share the interactions schema
    "train": INTERACTIONS,
    "val": INTERACTIONS,
    "test": INTERACTIONS,
}


def get_schema(table: str) -> pa.DataFrameSchema:
    try:
        return SCHEMAS[table]
    except KeyError:
        raise KeyError(f"No schema for table {table!r}. Known: {', '.join(sorted(SCHEMAS))}") from None


def validate(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate `df` against the named table schema; raises `pandera.errors.SchemaError`."""
    return get_schema(table).validate(df)
