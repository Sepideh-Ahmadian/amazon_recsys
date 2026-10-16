import shutil
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from amazon_recsys.config import JudgeCfg
from amazon_recsys.data.schemas import (
    ASPECT_JUDGEMENTS,
    ASPECTS,
    ENRICHMENT,
    ITEMS,
    JUDGEMENTS,
    REVIEWS,
)
from amazon_recsys.enrichment.cache import ResponseCache
from amazon_recsys.enrichment.jev_client import (
    ChoiceAnswer,
    ChoiceQuestion,
    FakeJevClient,
    JevError,
    JevResponse,
    JevTooLargeError,
    JevTransientError,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from amazon_recsys.enrichment.judge import (
    AspectIn,
    Judge,
    answer_confidence,
    aspect_is_well_formed,
    build_requests,
    build_state,
    decide_verdict,
    faithfulness,
    format_compliance,
    in_sample,
    judge_domain,
    normalise_score,
    polarity_accuracy,
    score_review,
    sentiment_agreement,
    weighted_overall,
)
from amazon_recsys.enrichment.prompts import load_judge_questions
from amazon_recsys.utils.io import read_parquet, write_parquet

ROOT = Path(__file__).resolve().parents[2]
REAL_PROMPTS = ROOT / "prompts" / "all_beauty"
SPEC, QHASH = load_judge_questions(REAL_PROMPTS, "v1")


def asp(name="pleasant scent", polarity=1, raw=None):
    return AspectIn(name, raw or name.title(), polarity, 0.8, f"evidence for {name}")


def choice(label, p=0.8, confidence=0.9):
    rest = (1 - p) / 2
    probs = {k: (p if k == label else rest) for k in ("positive", "negative", "neutral")}
    return ChoiceAnswer(label, probs, confidence)


def score(value, n, confidence=0.9):
    probs = {i: 1 / n for i in range(n)}
    return ScoreAnswer(value, probs, confidence)


# --- state and questions ---------------------------------------------------------------------


def test_state_truncates_and_embeds_aspect_objects():
    aspects = [asp("pleasant scent", 1), asp("sticky feel", -1)]
    state, truncated = build_state("Title", "x" * 50, 4.0, 0.3, aspects, max_chars=20)
    assert truncated is True and len(state["review"]["text"]) == 20
    assert state["extraction"]["aspects"][1] == {
        "aspect": "Sticky Feel",
        "polarity": "negative",
        "intensity": 0.8,
        "evidence": "evidence for sticky feel",
    }
    state, truncated = build_state("T", "short", 4.0, 0.0, [], max_chars=20)
    assert truncated is False and state["extraction"]["aspects"] == []


def test_questions_one_noul_one_choice_per_aspect_plus_review_level():
    (req,) = build_requests(SPEC, [asp("a"), asp("b")], max_per_request=15)
    assert "format_followed" not in req  # the format check is deterministic code now
    assert isinstance(req["completeness"], ScoreQuestion)
    assert isinstance(req["overall_sentiment"], ScoreQuestion)
    assert len(req["overall_sentiment"].levels) == 5
    for i in (0, 1):
        assert isinstance(req[f"aspect_{i}_supported"], NoulQuestion)
        assert isinstance(req[f"aspect_{i}_polarity"], ChoiceQuestion)
    assert len(req) == 2 + 4
    instructions = req["aspect_0_supported"].instructions
    assert instructions["extracted_aspect"] == "A" and "question" in instructions


def test_requests_split_above_max_aspects_and_review_level_only_once():
    aspects = [asp(f"a{i}") for i in range(5)]
    reqs = build_requests(SPEC, aspects, max_per_request=2)
    assert len(reqs) == 3
    assert "completeness" in reqs[0] and "completeness" not in reqs[1]
    names = [set(r) for r in reqs]
    assert names[1] == {
        "aspect_2_supported",
        "aspect_2_polarity",
        "aspect_3_supported",
        "aspect_3_polarity",
    }
    assert set().union(*names) >= {f"aspect_{i}_supported" for i in range(5)}


def test_no_aspects_gets_only_review_level_questions():
    (req,) = build_requests(SPEC, [], max_per_request=15)
    assert set(req) == {"completeness", "overall_sentiment"}


# --- pure scoring (hand computed) -----------------------------------------------------------


def test_faithfulness_is_mean_of_p_supported():
    assert faithfulness([0.9, 0.5]) == pytest.approx(0.7)
    assert faithfulness([]) is None


def test_polarity_accuracy_uses_probability_of_the_extracted_polarity():
    answers = [choice("positive", 0.8), choice("negative", 0.6)]  # others share the remainder
    # extracted: positive -> 0.8 ; neutral -> (1-0.6)/2 = 0.2
    assert polarity_accuracy(answers, [1, 0]) == pytest.approx((0.8 + 0.2) / 2)


@pytest.mark.parametrize(
    ("value", "n", "expected"), [(1.0, 3, 0.5), (3.0, 4, 1.0), (2.0, 5, 0.5), (0.0, 5, 0.0)]
)
def test_completeness_normalisation_for_3_4_5_levels(value, n, expected):
    assert normalise_score(score(value, n)) == pytest.approx(expected)


def test_sentiment_agreement_extremes():
    assert sentiment_agreement(1.0, 1.0) == 1.0  # very positive vs +1
    assert sentiment_agreement(0.0, -1.0) == 1.0
    assert sentiment_agreement(0.0, 1.0) == 0.0  # opposite
    assert sentiment_agreement(0.5, 0.0) == 1.0  # neutral vs 0


def test_weights_renormalised_when_component_missing():
    weights = {"faithfulness": 0.5, "completeness": 0.25, "sentiment_agreement": 0.25}
    full = {"faithfulness": 1.0, "completeness": 0.0, "sentiment_agreement": 0.0}
    assert weighted_overall(full, weights) == pytest.approx(0.5)
    partial = {"faithfulness": None, "completeness": 1.0, "sentiment_agreement": 0.0}
    assert weighted_overall(partial, weights) == pytest.approx(0.5)  # 0.25 / (0.25 + 0.25)
    assert weighted_overall({"faithfulness": None}, weights) is None


def test_confidence_is_minimum_across_answers_with_noul_as_two_abs_p_minus_half():
    answers = {
        "n1": NoulAnswer(0.9),  # 0.8
        "n2": NoulAnswer(0.6),  # 0.2 -> the minimum
        "c": choice("positive", confidence=0.95),
        "s": score(1.0, 3, confidence=0.7),
    }
    assert answer_confidence(answers) == pytest.approx(0.2)
    assert answer_confidence({}) is None


def test_verdict_boundaries():
    assert decide_verdict(0.7, 0.9, 0.7, 0.5) == "pass"  # exactly at threshold
    assert decide_verdict(0.6999, 0.9, 0.7, 0.5) == "fail"
    assert decide_verdict(0.99, 0.49, 0.7, 0.5) == "uncertain"  # low confidence wins
    assert decide_verdict(0.1, 0.5, 0.7, 0.5) == "fail"  # confidence exactly at the minimum
    assert decide_verdict(None, 0.9, 0.7, 0.5) is None


@st.composite
def answer_sets(draw):
    n_aspects = draw(st.integers(0, 5))
    prob = st.floats(0.0, 1.0)
    answers = {}
    aspects = []
    for i in range(n_aspects):
        aspects.append(asp(f"a{i}", draw(st.sampled_from([-1, 0, 1]))))
        weights = [draw(st.floats(0.01, 1.0)) for _ in range(3)]
        total = sum(weights)
        probs = dict(
            zip(("positive", "negative", "neutral"), (w / total for w in weights), strict=True)
        )
        answers[f"aspect_{i}_supported"] = NoulAnswer(draw(prob))
        answers[f"aspect_{i}_polarity"] = ChoiceAnswer("positive", probs, draw(prob))
    for name in ("completeness", "overall_sentiment"):
        n = draw(st.integers(2, 10))
        answers[name] = score(draw(st.floats(0.0, n - 1.0)), n, draw(prob))
    return aspects, answers, draw(st.floats(-1.0, 1.0))


@given(answer_sets())
@settings(max_examples=80, deadline=None)
def test_property_components_overall_and_confidence_stay_in_unit_interval(data):
    aspects, answers, sentiment = data
    s = score_review(answers, aspects, sentiment, JudgeCfg(), "scent texture a b text")
    for value in (s.faithfulness, s.polarity_accuracy, s.completeness, s.format_compliance,
                  s.sentiment_agreement, s.overall, s.confidence):  # fmt: skip
        assert value is None or 0.0 <= value <= 1.0
    assert s.overall is not None and s.verdict in {"pass", "fail", "uncertain"}


# --- runner ---------------------------------------------------------------------------------


def responder(p_supported=0.9, label="positive", p_choice=0.8, level_frac=0.5, confidence=0.9):
    """Answers any question set: Noul -> p_supported, Choice -> label, Score -> a fixed fraction."""

    def respond(state, questions):
        answers = {}
        for name, q in questions.items():
            if isinstance(q, NoulQuestion):
                answers[name] = NoulAnswer(p_supported)
            elif isinstance(q, ChoiceQuestion):
                answers[name] = choice(label, p_choice, confidence)
            else:
                n = len(q.levels)
                answers[name] = score(level_frac * (n - 1), n, confidence)
        return JevResponse("jev-1.13.0", answers, 100)

    return respond


def reviews_frame(ids):
    return pd.DataFrame(
        {
            "review_id": ids,
            "domain": "all_beauty",
            "user_id": [f"u{n}" for n in range(len(ids))],
            "item_id": [f"i{n}" for n in range(len(ids))],
            "rating": 4.0,
            "title": "t",
            "text": [f"text of {r}" for r in ids],
            "timestamp": pd.array(range(1, len(ids) + 1), dtype="int64"),
        }
    )


@pytest.fixture
def make_judge(tmp_path):
    caches = []

    def build(client, cfg=None, seed=0, concurrency=1):
        cache = ResponseCache(tmp_path / "cache.sqlite")
        caches.append(cache)
        return Judge(
            client, cache, SPEC, QHASH, cfg or JudgeCfg(), "all_beauty", seed,
            concurrency=concurrency,
        )  # fmt: skip

    yield build
    for c in caches:
        c.close()


def test_judges_a_review_and_stores_probabilities_as_returned(make_judge):
    fake = FakeJevClient(responder(p_supported=0.123456789))
    judge = make_judge(fake)
    out, aspects_out, stats = judge.run(
        reviews_frame(["r1"]), {"r1": 0.5}, {"r1": [asp("pleasant scent", 1)]}
    )
    row = out.iloc[0]
    assert row.judge_status == "ok" and row.jev_model == "jev-1.13.0" and row.input_tokens == 100
    assert row.faithfulness == 0.123456789  # never rounded
    assert aspects_out.iloc[0].p_supported == 0.123456789
    assert aspects_out.iloc[0].p_polarity_match == pytest.approx(0.8)
    assert row.verdict in {"pass", "fail", "uncertain"} and stats.ok == 1
    JUDGEMENTS.validate(out)
    ASPECT_JUDGEMENTS.validate(aspects_out)


def test_high_quality_extraction_passes_and_low_quality_fails(make_judge):
    good = make_judge(FakeJevClient(responder(0.95, "positive", 0.95, level_frac=0.9)))
    out, *_ = good.run(reviews_frame(["r1"]), {"r1": 0.9}, {"r1": [asp()]})
    assert out.iloc[0].verdict == "pass"
    bad = make_judge(FakeJevClient(responder(0.05, "negative", 0.95, level_frac=0.0)))
    out, *_ = bad.run(reviews_frame(["r2"]), {"r2": 0.9}, {"r2": [asp()]})
    assert out.iloc[0].verdict == "fail"


def test_api_error_too_large_and_no_aspects(make_judge):
    def errors(state, questions):
        text = state["review"]["text"]
        if "r_api" in text:
            return JevTransientError("down")
        if "r_big" in text:
            return JevTooLargeError("too big")
        return responder()(state, questions)

    judge = make_judge(FakeJevClient(errors))
    reviews = reviews_frame(["r_api", "r_big", "r_none"])
    sentiments = dict.fromkeys(reviews.review_id, 0.2)
    by_review = {"r_api": [asp()], "r_big": [asp()], "r_none": []}
    out, aspects_out, stats = judge.run(reviews, sentiments, by_review)

    status = dict(zip(out.review_id, out.judge_status, strict=True))
    assert status == {"r_api": "api_error", "r_big": "too_large", "r_none": "no_aspects"}
    assert len(out) == 3  # rows are kept
    failed = out[out.judge_status.isin(["api_error", "too_large"])]
    assert failed.overall.isna().all() and failed.verdict.isna().all()
    none_row = out[out.review_id == "r_none"].iloc[0]
    assert pd.isna(none_row.faithfulness) and not pd.isna(none_row.completeness)
    assert pd.isna(none_row.format_compliance) and none_row.verdict is not None
    assert aspects_out.empty
    JUDGEMENTS.validate(out)


def test_reviews_without_ok_extraction_are_not_judged(make_judge):
    fake = FakeJevClient(responder())
    out, *_ = make_judge(fake).run(reviews_frame(["r1", "r2"]), {"r1": 0.1}, {"r1": [asp()]})
    assert out.review_id.tolist() == ["r1"]


def test_cached_reviews_make_no_calls_and_changed_extraction_invalidates(make_judge):
    fake = FakeJevClient(responder())
    judge = make_judge(fake)
    reviews = reviews_frame(["r1"])
    judge.run(reviews, {"r1": 0.5}, {"r1": [asp("pleasant scent", 1)]})
    calls = fake.call_count
    first, *_ = judge.run(reviews, {"r1": 0.5}, {"r1": [asp("pleasant scent", 1)]})
    assert fake.call_count == calls  # cache hit
    judge.run(reviews, {"r1": 0.5}, {"r1": [asp("pleasant scent", -1)]})  # extraction changed
    assert fake.call_count == calls + 1
    assert first.iloc[0].judge_status == "ok"


def test_limit_counts_uncached_reviews_only(make_judge):
    fake = FakeJevClient(responder())
    judge = make_judge(fake)
    ids = [f"r{i}" for i in range(4)]
    sentiments = dict.fromkeys(ids, 0.0)
    by_review = {i: [asp()] for i in ids}
    out, _, stats = judge.run(reviews_frame(ids), sentiments, by_review, limit=2)
    assert fake.call_count == 2 and len(out) == 2 and stats.skipped_by_limit == 2


def test_multi_request_review_issues_one_call_per_chunk(make_judge):
    fake = FakeJevClient(responder())
    cfg = JudgeCfg(max_aspects_per_request=2)
    judge = make_judge(fake, cfg)
    aspects = [asp(f"a{i}") for i in range(5)]
    out, aspects_out, _ = judge.run(reviews_frame(["r1"]), {"r1": 0.0}, {"r1": aspects})
    assert fake.call_count == 3 and len(aspects_out) == 5
    assert out.iloc[0].input_tokens == 300  # summed over requests


def test_sample_rate_is_seeded_and_reproducible():
    ids = [f"r{i}" for i in range(400)]
    a = [i for i in ids if in_sample(i, 0.25, seed=1)]
    assert a == [i for i in ids if in_sample(i, 0.25, seed=1)]
    assert a != [i for i in ids if in_sample(i, 0.25, seed=2)]
    assert 60 < len(a) < 140
    assert all(in_sample(i, 1.0, seed=0) for i in ids)


def test_sampled_out_reviews_are_skipped(make_judge):
    ids = [f"r{i}" for i in range(40)]
    fake = FakeJevClient(responder())
    judge = make_judge(fake, JudgeCfg(sample_rate=0.5), seed=3)
    out, _, stats = judge.run(
        reviews_frame(ids), dict.fromkeys(ids, 0.0), {i: [asp()] for i in ids}
    )
    assert 0 < len(out) < 40 and stats.sampled_out == 40 - len(out)


def test_dry_run_estimates_tokens_and_cost_without_calls(make_judge):
    judge = make_judge(None, JudgeCfg(price_per_million_input_tokens=2.0))
    reviews = reviews_frame(["r1", "r2"])
    out = judge.dry_run(reviews, {"r1": 0.0, "r2": 0.0}, {"r1": [asp()], "r2": []})
    assert out["rows_uncached"] == 2 and out["requests"] == 2
    assert out["estimated_input_tokens"] > 0
    assert out["estimated_cost"] == pytest.approx(out["estimated_input_tokens"] * 2.0 / 1e6)


def test_rejects_question_spec_without_the_three_polarity_options():
    bad = SPEC.model_copy(
        update={
            "aspect_polarity": SPEC.aspect_polarity.model_copy(update={"options": ["yes", "no"]})
        }
    )
    with pytest.raises(ValueError, match="positive, negative and neutral"):
        Judge(None, None, bad, "h", JudgeCfg(), "d")


# --- files ------------------------------------------------------------------------------------


@pytest.fixture
def judge_setup(tiny_cfg, tmp_paths):
    shutil.copytree(REAL_PROMPTS, tmp_paths.resolve(tiny_cfg.domain.prompt_dir))
    reviews = reviews_frame(["r1", "r2", "r3"])
    write_parquet(reviews, tmp_paths.reviews_filtered, REVIEWS)
    enrichment = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "domain": "all_beauty",
            "status": ["ok", "ok", "parse_error"],
            "overall_sentiment": [0.5, 0.1, float("nan")],
            "extractor_model": "m",
            "prompt_version": "v1",
            "prompt_hash": "h",
        }
    )
    write_parquet(enrichment, tmp_paths.processed("enrichment"), ENRICHMENT)
    aspects = pd.DataFrame(
        {
            "review_id": ["r1", "r1"],
            "domain": "all_beauty",
            "aspect": ["pleasant scent", "thick texture"],
            "aspect_raw": ["Pleasant Scent", "Thick Texture"],
            "polarity": pd.array([1, -1], dtype="int64"),
            "intensity": [0.8, 0.4],
            "evidence": ["smells nice", "so thick"],
        }
    )
    write_parquet(aspects, tmp_paths.processed("aspects"), ASPECTS)
    items = pd.DataFrame({"item_id": reviews.item_id, "domain": "all_beauty", "title": "t",
                          "category": None, "store": None, "description": None})  # fmt: skip
    write_parquet(items, tmp_paths.items, ITEMS)
    return tiny_cfg, tmp_paths


def test_judge_domain_writes_validated_tables(judge_setup):
    cfg, paths = judge_setup
    fake = FakeJevClient(responder())
    summary = judge_domain(cfg, paths, fake)
    assert summary["ok"] == 1 and summary["no_aspects"] == 1  # r3 failed extraction: skipped
    j = read_parquet(paths.processed("judgements"), JUDGEMENTS)
    a = read_parquet(paths.processed("aspect_judgements"), ASPECT_JUDGEMENTS)
    assert set(j.review_id) == {"r1", "r2"} and set(a.aspect) == {"pleasant scent", "thick texture"}
    assert (j.questions_hash == QHASH).all()
    calls = fake.call_count
    judge_domain(cfg, paths, fake)
    assert fake.call_count == calls  # fully cached


def test_judge_domain_dry_run_and_missing_enrichment(judge_setup):
    cfg, paths = judge_setup
    report = judge_domain(cfg, paths, client=None, dry_run=True)
    assert report["rows_uncached"] == 2
    paths.processed("enrichment").unlink()
    with pytest.raises(FileNotFoundError, match="recsys enrich"):
        judge_domain(cfg, paths, client=None, dry_run=True)


def test_jev_error_base_class_is_caught_as_api_error(make_judge):
    judge = make_judge(FakeJevClient(lambda s, q: JevError("auth failed")))
    out, *_ = judge.run(reviews_frame(["r1"]), {"r1": 0.0}, {"r1": [asp()]})
    assert out.iloc[0].judge_status == "api_error"


def test_state_contains_only_json_serialisable_python_scalars():
    import json

    import numpy as np

    aspect = AspectIn("a", "A", 1, np.float64(0.5), "ev")
    state, _ = build_state("t", "text", np.float64(4.0), np.float64(0.25), [aspect], 100)
    assert json.loads(json.dumps(state)) == state  # raises on numpy scalars
    assert type(state["review"]["rating"]) is float
    assert type(state["extraction"]["overall_sentiment"]) is float


# --- deterministic format check ----------------------------------------------------------------


def test_format_check_requires_verbatim_evidence_and_a_short_label():
    text = "Smells great, and it is not too sticky!"
    ok = AspectIn("pleasant scent", "Pleasant Scent", 1, 0.8, "smells great")
    assert aspect_is_well_formed(ok, text)  # case and punctuation ignored
    invented = AspectIn("pleasant scent", "Pleasant Scent", 1, 0.8, "smells wonderful")
    assert not aspect_is_well_formed(invented, text)
    no_evidence = AspectIn("pleasant scent", "Pleasant Scent", 1, 0.8, None)
    assert not aspect_is_well_formed(no_evidence, text)
    one_word = AspectIn("scent", "Scent", 1, 0.8, "smells great")
    long_label = AspectIn("a", "A Very Long Winded Label Here", 1, 0.8, "smells great")
    assert not aspect_is_well_formed(one_word, text)
    assert not aspect_is_well_formed(long_label, text)


def test_format_compliance_is_the_fraction_of_well_formed_aspects():
    text = "Smells great but it leaves my hands sticky"
    aspects = [
        AspectIn("pleasant scent", "Pleasant Scent", 1, 0.8, "Smells great"),
        AspectIn("sticky feel", "Sticky Feel", -1, 0.5, "leaves my hands sticky"),
        AspectIn("fast absorption", "Fast Absorption", 1, 0.5, "absorbs quickly"),  # not in text
    ]
    assert format_compliance(aspects, text) == pytest.approx(2 / 3)
    assert format_compliance([], text) is None


def test_judge_uses_the_full_review_not_the_truncated_state_for_format(make_judge):
    long_text = "x " * 50 + "smells great"  # evidence sits beyond max_state_chars
    reviews = reviews_frame(["r1"]).assign(text=long_text)
    judge = make_judge(FakeJevClient(responder()), JudgeCfg(max_state_chars=20))
    aspect = AspectIn("pleasant scent", "Pleasant Scent", 1, 0.8, "smells great")
    out, *_ = judge.run(reviews, {"r1": 0.5}, {"r1": [aspect]})
    assert out.iloc[0].truncated and out.iloc[0].format_compliance == 1.0


# --- concurrency ----------------------------------------------------------------------------


def test_concurrent_judging_matches_sequential_output_exactly(make_judge, tmp_path):
    ids = [f"r{i}" for i in range(10)]
    sentiments = {i: 0.1 * n for n, i in enumerate(ids)}
    by_review = {i: [asp(f"a{n % 3}", 1 if n % 2 else -1)] for n, i in enumerate(ids)}
    reviews = reviews_frame(ids)

    sequential = make_judge(FakeJevClient(responder(0.7)), concurrency=1)
    seq_j, seq_a, _ = sequential.run(reviews, sentiments, by_review)

    (tmp_path / "cache.sqlite").unlink()
    for ext in ("-wal", "-shm"):
        (tmp_path / f"cache.sqlite{ext}").unlink(missing_ok=True)
    fake = FakeJevClient(responder(0.7))
    concurrent = make_judge(fake, concurrency=4)
    con_j, con_a, stats = concurrent.run(reviews, sentiments, by_review)

    pd.testing.assert_frame_equal(seq_j, con_j)
    pd.testing.assert_frame_equal(seq_a, con_a)
    assert fake.call_count == len(ids) and stats.called == len(ids) and stats.cached == 0


def test_concurrent_judging_limit_errors_and_resume(make_judge):
    def flaky(state, questions):
        if "r3" in state["review"]["text"]:
            return JevTooLargeError("too big")
        if "r5" in state["review"]["text"]:
            return JevTransientError("down")
        return responder()(state, questions)

    ids = [f"r{i}" for i in range(8)]
    sentiments = dict.fromkeys(ids, 0.0)
    by_review = {i: [asp()] for i in ids}
    fake = FakeJevClient(flaky)
    judge = make_judge(fake, concurrency=3)

    out, _, stats = judge.run(reviews_frame(ids), sentiments, by_review, limit=4)
    assert fake.call_count == 4 and len(out) == 4 and stats.skipped_by_limit == 4

    out, _, stats = judge.run(reviews_frame(ids), sentiments, by_review)
    status = dict(zip(out.review_id, out.judge_status, strict=True))
    assert status["r3"] == "too_large" and status["r5"] == "api_error"
    assert sum(1 for s in status.values() if s == "ok") == 6
    # first run: r0-r2 judged and cached, r3 failed (never cached); second run reuses 3
    assert (stats.too_large, stats.api_error, stats.cached) == (1, 1, 3)


def test_an_unreadable_cache_entry_is_a_miss_not_a_crash(make_judge):
    fake = FakeJevClient(responder())
    judge = make_judge(fake)
    reviews = reviews_frame(["r1"])
    sentiments, by_review = {"r1": 0.0}, {"r1": [asp()]}
    judge.run(reviews, sentiments, by_review)
    assert fake.call_count == 1

    # corrupt the stored response: it must be re-judged instead of crashing the run
    _, _, _, _, input_hash = judge._prepare(next(reviews.itertuples()), 0.0, [asp()])
    broken = '{"model": "x", "answers": {"q": {"type": "choice"}}}'
    judge.cache.put("judge", "r1", QHASH, JudgeCfg().model, input_hash, broken)
    out, _, stats = judge.run(reviews, sentiments, by_review)
    assert fake.call_count == 2 and stats.cached == 0 and out.iloc[0].judge_status == "ok"
