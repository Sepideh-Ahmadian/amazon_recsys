import pandas as pd
import pandera as pa
import pytest

from amazon_recsys.data.schemas import SCHEMAS, get_schema, validate

VALID_ROWS = {
    "reviews": dict(
        review_id="r1", domain="d", user_id="u1", item_id="i1", rating=4.0,
        title="t", text="x", timestamp=1_700_000_000_000,
    ),
    "items": dict(item_id="i1", domain="d", title="t", category="c", store="s", description="x"),
    "enrichment": dict(
        review_id="r1", domain="d", status="ok", overall_sentiment=0.5,
        extractor_model="m", prompt_version="v1", prompt_hash="h",
    ),
    "aspects": dict(
        review_id="r1", domain="d", aspect="scent", aspect_raw="Scents", polarity=1,
        intensity=0.8, evidence="smells nice",
    ),
    "judgements": dict(
        review_id="r1", domain="d", judge_status="ok", faithfulness=0.9, polarity_accuracy=0.8,
        completeness=0.5, format_compliance=1.0, sentiment_agreement=1.0, overall=0.8, confidence=0.7, verdict="pass",
        truncated=False, jev_model="jev-1.13.0", questions_hash="q", extraction_hash="e",
        input_tokens=120,
    ),
    "aspect_judgements": dict(
        review_id="r1", domain="d", aspect="scent", p_supported=0.9, p_polarity_match=0.8,
        p_positive=0.7, p_negative=0.2, p_neutral=0.1, polarity_confidence=0.6,
    ),
    "interactions": dict(
        domain="d", user_id="u1", item_id="i1", rating=4.0, timestamp=1_700_000_000_000,
        review_id="r1", overall_sentiment=0.5, judge_overall=0.8, judge_verdict="pass",
    ),
}
TABLES = list(VALID_ROWS)


def frame(table: str) -> pd.DataFrame:
    df = pd.DataFrame([VALID_ROWS[table]])
    if table == "judgements":
        df["input_tokens"] = df["input_tokens"].astype("Int64")
    return df


@pytest.mark.parametrize("table", TABLES)
def test_valid_frame_accepted(table):
    validate(table, frame(table))


@pytest.mark.parametrize("table", TABLES)
def test_missing_column_rejected(table):
    for col in VALID_ROWS[table]:
        with pytest.raises(pa.errors.SchemaError):
            validate(table, frame(table).drop(columns=[col]))


@pytest.mark.parametrize("table", TABLES)
def test_extra_column_rejected(table):
    with pytest.raises(pa.errors.SchemaError):
        validate(table, frame(table).assign(surprise=1))


@pytest.mark.parametrize("table", TABLES)
def test_wrong_dtype_rejected(table):
    for col, val in VALID_ROWS[table].items():
        bad = "not-a-number" if not isinstance(val, str) else 12345
        df = frame(table).astype({col: object})
        df[col] = [bad]
        with pytest.raises(pa.errors.SchemaError):
            validate(table, df)


@pytest.mark.parametrize("table", TABLES)
def test_null_in_non_nullable_column_rejected(table):
    schema = get_schema(table)
    for name, col in schema.columns.items():
        if col.nullable:
            continue
        df = frame(table).astype({name: object})
        df[name] = [None]
        with pytest.raises(pa.errors.SchemaError):
            validate(table, df)


@pytest.mark.parametrize("table", TABLES)
def test_nullable_columns_accept_null(table):
    for name, col in get_schema(table).columns.items():
        if not col.nullable:
            continue
        df = frame(table)
        if name == "input_tokens":
            df[name] = pd.array([None], dtype="Int64")
        elif isinstance(VALID_ROWS[table][name], str):
            df[name] = pd.Series([None], dtype=object)
        else:
            df[name] = float("nan")
        validate(table, df)


@pytest.mark.parametrize(
    ("table", "column", "bad"),
    [
        ("reviews", "rating", 6.0),
        ("reviews", "rating", 0.0),
        ("enrichment", "overall_sentiment", 1.5),
        ("enrichment", "status", "weird"),
        ("aspects", "polarity", 2),
        ("aspects", "intensity", -0.1),
        ("judgements", "overall", 1.1),
        ("judgements", "verdict", "maybe"),
        ("judgements", "judge_status", "oops"),
        ("aspect_judgements", "p_supported", -0.1),
        ("interactions", "judge_overall", 1.2),
        ("interactions", "judge_verdict", "maybe"),
        ("interactions", "overall_sentiment", -1.5),
    ],
)
def test_out_of_range_rejected(table, column, bad):
    df = frame(table)
    df[column] = [bad]
    with pytest.raises(pa.errors.SchemaError):
        validate(table, df)


def test_split_tables_share_interactions_schema():
    assert SCHEMAS["train"] is SCHEMAS["val"] is SCHEMAS["test"] is SCHEMAS["interactions"]


def test_unknown_table():
    with pytest.raises(KeyError):
        get_schema("nope")
