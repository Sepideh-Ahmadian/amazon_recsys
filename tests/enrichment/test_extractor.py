import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from amazon_recsys.config import ConfigError
from amazon_recsys.data.schemas import ASPECTS, ENRICHMENT, ITEMS, REVIEWS
from amazon_recsys.enrichment.cache import ResponseCache
from amazon_recsys.enrichment.extractor import (
    Extractor,
    ParseError,
    aspect_rows,
    enrich_domain,
    estimate_tokens,
    extract_json_object,
    normalise_aspect,
    parse_response,
    sentiment_to_polarity,
)
from amazon_recsys.enrichment.llm_client import FakeLLMClient, LLMError
from amazon_recsys.enrichment.prompts import ExtractionOutput, load_extractor_template
from amazon_recsys.utils.io import read_parquet, write_parquet

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = sorted((ROOT / "tests" / "fixtures" / "extractor").glob("*.json"))
REAL_PROMPTS = ROOT / "prompts" / "all_beauty"


def _load(path):
    return json.loads(path.read_text())


def _reviews(ids, texts=None):
    texts = texts or {}
    return pd.DataFrame(
        {
            "review_id": ids,
            "domain": "all_beauty",
            "user_id": [f"u{n}" for n in range(len(ids))],
            "item_id": [f"i{n}" for n in range(len(ids))],
            "rating": 4.0,
            "title": "title",
            "text": [texts.get(r, f"text of {r}") for r in ids],
            "timestamp": pd.array(range(1, len(ids) + 1), dtype="int64"),
        }
    )


@pytest.fixture
def make_extractor(tmp_path):
    template, prompt_hash = load_extractor_template(REAL_PROMPTS, "v1")
    caches = []

    def build(client, price=0.0, synonyms=None, concurrency=1):
        cache = ResponseCache(tmp_path / "cache.sqlite")
        caches.append(cache)
        return Extractor(
            client, cache, template, prompt_hash, "dep", "v1", "all_beauty",
            synonyms=synonyms, price_per_million_input_tokens=price, concurrency=concurrency,
        )  # fmt: skip

    yield build
    for cache in caches:
        cache.close()


# --- parsing -----------------------------------------------------------------------------


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_every_fixture_parses_as_expected(path):
    case = _load(path)
    expect = case["expect"]
    if expect["status"] == "parse_error":
        with pytest.raises(ParseError):
            parse_response(case["response"])
    else:
        out = parse_response(case["response"])
        assert len(out.aspects) == expect["n_aspects"]
        assert out.overall_sentiment == pytest.approx(expect["overall_sentiment"])


def test_extract_json_handles_fences_prose_and_rejects_non_objects():
    assert extract_json_object('prefix {"a": 1} suffix') == {"a": 1}
    assert extract_json_object('```json\n{"a": 2}\n```') == {"a": 2}
    for bad in ("no json at all", "[1, 2, 3]", '{"a": '):
        with pytest.raises(ParseError):
            extract_json_object(bad)


# --- normalisation -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Pleasant Scent", "pleasant scent"),
        ("pleasant scents", "pleasant scent"),
        ("  Pleasant   Scent. ", "pleasant scent"),
        ("Dry Skin!", "dry skin"),
        ("Heavy Glass", "heavy glass"),  # 'ss' endings are kept
        ("Sticky Berries", "sticky berry"),
        ("Strong Smell", "strong scent"),  # per-word synonym
        ("smell", "scent"),  # whole-phrase synonym
    ],
)
def test_aspect_normalisation(raw, expected):
    assert normalise_aspect(raw, {"smell": "scent"}) == expected


def test_aspect_raw_preserved_and_duplicates_keep_strongest():
    out = ExtractionOutput.model_validate(
        {
            "overall_sentiment": 0.5,
            "aspects": [
                {"aspect": "Pleasant Scent", "sentiment": 0.4, "evidence": "ok smell"},
                {"aspect": "pleasant scents", "sentiment": 0.9, "evidence": "lovely"},
                {"aspect": "!!!", "sentiment": 0.1, "evidence": "x"},
            ],
        }
    )
    rows = aspect_rows("r1", "d", out, {})
    assert len(rows) == 1
    assert rows[0]["aspect"] == "pleasant scent"
    assert rows[0]["aspect_raw"] == "pleasant scents"
    assert rows[0]["intensity"] == pytest.approx(0.9) and rows[0]["evidence"] == "lovely"


def test_polarity_mapping_boundaries():
    assert [sentiment_to_polarity(s) for s in (-1, -0.11, -0.1, 0, 0.1, 0.11, 1)] == [
        -1, -1, 0, 0, 0, 1, 1,
    ]  # fmt: skip


# --- running -----------------------------------------------------------------------------


def _canned():
    return {_load(p)["name"]: _load(p)["response"] for p in FIXTURES}


def test_run_statuses_and_raw_text_kept_for_parse_errors(make_extractor):
    canned = _canned()
    canned["r_api"] = LLMError("boom")
    ex = make_extractor(FakeLLMClient(canned))
    reviews = _reviews(list(canned))
    enrichment, aspects, stats = ex.run(reviews, {})

    status = dict(zip(enrichment.review_id, enrichment.status, strict=True))
    assert status["valid"] == status["valid_with_extra_text"] == status["empty_aspects"] == "ok"
    assert status["malformed_json"] == status["sentiment_out_of_range"] == "parse_error"
    assert status["legacy_tuple_format"] == "parse_error"
    assert status["r_api"] == "api_error"
    assert (stats.ok, stats.parse_error, stats.api_error) == (3, 3, 1)
    assert len(enrichment) == len(reviews)  # nothing dropped silently
    assert set(aspects.review_id) == {"valid", "valid_with_extra_text"}
    ENRICHMENT.validate(enrichment)
    ASPECTS.validate(aspects)

    # raw text of a failed parse is kept in the cache; API errors are not cached
    cache = ex.cache
    raw = [
        cache.get("extractor", "malformed_json", ex.prompt_hash, "dep", h)
        for h in [_input_hash(ex, reviews, "malformed_json")]
    ]
    assert raw[0] == canned["malformed_json"]
    assert cache.count("extractor") == len(canned) - 1


def _input_hash(ex, reviews, review_id):
    row = next(r for r in reviews.itertuples() if r.review_id == review_id)
    return ex._input_hash(ex._variables(row, ""))


def test_second_run_makes_zero_client_calls(make_extractor):
    fake = FakeLLMClient(_canned())
    ex = make_extractor(fake)
    reviews = _reviews(list(_canned()))
    first = ex.run(reviews, {})
    calls_after_first = fake.call_count
    assert calls_after_first == len(reviews)
    second = ex.run(reviews, {})
    assert fake.call_count == calls_after_first
    assert second[2].cached == len(reviews) and second[2].called == 0
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_limit_caps_uncached_calls_but_keeps_cached_rows(make_extractor):
    canned = {f"r{n}": '{"overall_sentiment": 0.1, "aspects": []}' for n in range(5)}
    fake = FakeLLMClient(canned)
    ex = make_extractor(fake)
    reviews = _reviews(list(canned))

    enrichment, _, stats = ex.run(reviews, {}, limit=2)
    assert fake.call_count == 2 and len(enrichment) == 2 and stats.skipped_by_limit == 3

    enrichment, _, stats = ex.run(reviews, {}, limit=2)  # 2 cached + 2 new
    assert fake.call_count == 4 and len(enrichment) == 4

    enrichment, _, _ = ex.run(reviews, {})
    assert fake.call_count == 5 and len(enrichment) == 5


def test_api_errors_are_retried_on_next_run(make_extractor):
    fake = FakeLLMClient({"r1": LLMError("down")})
    ex = make_extractor(fake)
    reviews = _reviews(["r1"])
    assert ex.run(reviews, {})[0].status.tolist() == ["api_error"]
    fake.responses["r1"] = '{"overall_sentiment": 0.2, "aspects": []}'
    assert ex.run(reviews, {})[0].status.tolist() == ["ok"]


def test_dry_run_cost_is_tokens_times_price_and_makes_no_calls(make_extractor):
    ex = make_extractor(client=None, price=0.5)
    reviews = _reviews(["r1", "r2"], {"r1": "short", "r2": "a longer review text " * 5})
    out = ex.dry_run(reviews, {})
    from amazon_recsys.enrichment.prompts import render_extractor

    expected = 0
    for row in reviews.itertuples():
        messages = render_extractor(
            ex.template, product_title="", review_title=row.title, review_text=row.text
        )
        expected += sum(estimate_tokens(m["content"]) for m in messages)
    assert out["rows_total"] == 2 and out["rows_uncached"] == 2
    assert out["estimated_input_tokens"] == expected
    assert out["estimated_cost"] == pytest.approx(expected * 0.5 / 1_000_000)


def test_dry_run_excludes_cached_rows(make_extractor):
    fake = FakeLLMClient({"r1": '{"overall_sentiment": 0, "aspects": []}'})
    ex = make_extractor(fake)
    reviews = _reviews(["r1", "r2"])
    ex.run(reviews.iloc[:1], {})
    out = ex.dry_run(reviews, {})
    assert out["rows_uncached"] == 1


# --- end to end with files -----------------------------------------------------------------


@pytest.fixture
def enrich_setup(tiny_cfg, tmp_paths):
    shutil.copytree(REAL_PROMPTS, tmp_paths.resolve(tiny_cfg.domain.prompt_dir))
    reviews = _reviews(["valid", "malformed_json", "empty_aspects"])
    write_parquet(reviews, tmp_paths.reviews_filtered, REVIEWS)
    items = pd.DataFrame(
        {
            "item_id": reviews.item_id,
            "domain": "all_beauty",
            "title": ["Lotion", None, "Soap"],
            "category": None,
            "store": None,
            "description": None,
        }
    )
    write_parquet(items, tmp_paths.items, ITEMS)
    extractor_cfg = tiny_cfg.enrichment.extractor.model_copy(update={"deployment": "fake-dep"})
    cfg = tiny_cfg.model_copy(
        update={"enrichment": tiny_cfg.enrichment.model_copy(update={"extractor": extractor_cfg})}
    )
    return cfg, tmp_paths


def test_enrich_domain_writes_validated_tables(enrich_setup):
    cfg, paths = enrich_setup
    canned = _canned()
    fake = FakeLLMClient({k: canned[k] for k in ("valid", "malformed_json", "empty_aspects")})
    summary = enrich_domain(cfg, paths, fake)
    assert summary["ok"] == 2 and summary["parse_error"] == 1
    enrichment = read_parquet(paths.processed("enrichment"), ENRICHMENT)
    aspects = read_parquet(paths.processed("aspects"), ASPECTS)
    assert len(enrichment) == 3 and len(aspects) == 2
    assert set(aspects.aspect) == {"pleasant scent", "thick texture"}
    # re-running is free
    enrich_domain(cfg, paths, fake)
    assert fake.call_count == 3


def test_enrich_domain_dry_run_needs_no_client_or_env(tiny_cfg, enrich_setup):
    cfg, paths = enrich_setup
    assert enrich_domain(cfg, paths, client=None, dry_run=True)["rows_uncached"] == 3
    # an unresolved ${EXTRACTOR_DEPLOYMENT} is fine for a dry run but not for a real one
    unresolved = enrich_setup[0].model_copy(
        update={
            "enrichment": cfg.enrichment.model_copy(
                update={
                    "extractor": cfg.enrichment.extractor.model_copy(
                        update={"deployment": "${EXTRACTOR_DEPLOYMENT}"}
                    )
                }
            )
        }
    )
    assert enrich_domain(unresolved, paths, client=None, dry_run=True)["rows_total"] == 3
    with pytest.raises(ConfigError, match="EXTRACTOR_DEPLOYMENT"):
        enrich_domain(unresolved, paths, FakeLLMClient({}))


def test_media_markers_are_removed_from_prompt_input_only():
    from amazon_recsys.enrichment.extractor import clean_review_text

    raw = "[[VIDEOID:f77804a8088c40dfbef6fdcfacf8b15a]] Three  bath bombs [[ASIN:B000]] in a set."
    assert clean_review_text(raw) == "Three bath bombs in a set."
    assert clean_review_text(None) == "" and clean_review_text("plain text") == "plain text"


# --- concurrency ----------------------------------------------------------------------------


def _canned_many(n=12):
    return {
        f"r{i}": json.dumps(
            {
                "overall_sentiment": (i % 5 - 2) / 2,
                "aspects": [{"aspect": f"Aspect {i % 3}", "sentiment": 0.5, "evidence": "ev"}],
            }
        )
        for i in range(n)
    }


def test_concurrent_run_gives_identical_tables_and_counts(make_extractor, tmp_path):
    canned = _canned_many()
    reviews = _reviews(list(canned))

    seq_client = FakeLLMClient(canned)
    seq = make_extractor(seq_client, concurrency=1).run(reviews, {})

    # fresh cache so the concurrent run really calls the client
    (tmp_path / "cache.sqlite").unlink()
    for ext in ("-wal", "-shm"):
        (tmp_path / f"cache.sqlite{ext}").unlink(missing_ok=True)
    con_client = FakeLLMClient(canned)
    con = make_extractor(con_client, concurrency=4).run(reviews, {})

    pd.testing.assert_frame_equal(seq[0], con[0])
    pd.testing.assert_frame_equal(seq[1], con[1])
    assert seq_client.call_count == con_client.call_count == len(canned)
    assert (con[2].called, con[2].cached, con[2].ok) == (len(canned), 0, len(canned))


def test_concurrent_run_respects_limit_and_resumes_from_cache(make_extractor):
    canned = _canned_many(10)
    fake = FakeLLMClient(canned)
    ex = make_extractor(fake, concurrency=4)
    reviews = _reviews(list(canned))

    enrichment, _, stats = ex.run(reviews, {}, limit=3)
    assert fake.call_count == 3 and len(enrichment) == 3 and stats.skipped_by_limit == 7

    enrichment, _, stats = ex.run(reviews, {})  # the rest; the first 3 come from the cache
    assert fake.call_count == 10 and len(enrichment) == 10
    assert (stats.called, stats.cached) == (7, 3)

    ex.run(reviews, {})
    assert fake.call_count == 10  # fully cached


def test_concurrent_api_errors_are_recorded_once_and_not_cached(make_extractor):
    canned = _canned_many(6)
    canned["r2"] = LLMError("boom")
    fake = FakeLLMClient(canned)
    ex = make_extractor(fake, concurrency=3)
    enrichment, _, stats = ex.run(_reviews(list(canned)), {})
    status = dict(zip(enrichment.review_id, enrichment.status, strict=True))
    assert status["r2"] == "api_error" and list(status.values()).count("ok") == 5
    assert fake.call_count == 6  # the failed review is not retried within the same run
    assert stats.api_error == 1 and len(stats.api_errors) == 1
    assert ex.cache.count("extractor") == 5
