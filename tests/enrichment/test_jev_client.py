import json
import os
import subprocess
import sys
from pathlib import Path

import httpx2
import pytest
import typesafe_sdk as ts

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
    TypeSafeJevClient,
    build_request,
    parse_response,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "jev"
STATE = {
    "review": {"title": "Nice", "text": "Smells great but sticky.", "rating": 4.0},
    "extraction": {
        "overall_sentiment": 0.3,
        "aspects": [{"aspect": "pleasant scent", "polarity": 1}],
    },
}


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


# --- request builder (snapshot against the documented payload shape) -----------------------


def test_noul_request_matches_snapshot():
    q = {
        "aspect_0_supported": NoulQuestion(
            {
                "extracted_aspect": "pleasant scent",
                "question": "Is the extracted aspect really mentioned in the review?",
            }
        )
    }
    assert build_request("jev-1.13.0", STATE, q) == _fixture("request_noul.json")


def test_choice_request_matches_snapshot():
    q = {
        "aspect_0_polarity": ChoiceQuestion(
            ("positive", "negative", "neutral"),
            "What sentiment does the reviewer express toward this specific aspect?",
        )
    }
    assert build_request("jev-1.13.0", STATE, q) == _fixture("request_choice.json")


def test_score_request_matches_snapshot():
    q = {
        "completeness": ScoreQuestion(
            ("Misses most aspects.", "Covers some aspects.", "Covers every aspect."),
            "How completely does the extraction cover the aspects?",
        )
    }
    assert build_request("jev-1.13.0", STATE, q) == _fixture("request_score.json")


def test_request_requires_questions():
    with pytest.raises(JevError):
        build_request("jev-1.13.0", STATE, {})


# --- response parsing ---------------------------------------------------------------------


def test_parse_each_answer_type_and_capture_model_version():
    noul = parse_response(_fixture("response_noul.json"))
    assert noul.model == "jev-1.13.0" and noul.input_tokens == 212
    assert noul.answers["aspect_0_supported"] == NoulAnswer(noul=0.93)

    choice = parse_response(_fixture("response_choice.json")).answers["aspect_0_polarity"]
    assert isinstance(choice, ChoiceAnswer) and choice.choice == "positive"
    assert choice.confidence == 0.88

    score = parse_response(_fixture("response_score.json")).answers["completeness"]
    assert isinstance(score, ScoreAnswer)
    assert score.score == 1.7 and score.n_levels == 3
    assert all(isinstance(k, int) for k in score.probabilities)  # keys parsed as ints


def test_probabilities_sum_to_one():
    response = parse_response(_fixture("response_mixed.json"))
    for answer in response.answers.values():
        if isinstance(answer, ChoiceAnswer | ScoreAnswer):
            assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "mystery"},
        {"type": "noul"},
        {"type": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"a": 0.2, "b": 0.2}},
        {"type": "score", "score": 1, "confidence": 0.5, "probabilities": {"x": 1.0}},
    ],
)
def test_malformed_answers_raise(answer):
    with pytest.raises(JevError):
        parse_response({"model": "jev-1", "usage": {}, "answers": {"q": answer}})
    with pytest.raises(JevError):
        parse_response({"usage": {}, "answers": {}})  # missing model


# --- the real client, with the SDK mocked --------------------------------------------------


def _sdk_response():
    return ts.SystemOneResponse(
        model="jev-1.13.0",
        usage=ts.Usage(input_tokens=321, output_tokens=0),
        answers={
            "s": ts.NoulAnswer(noul=0.8),
            "c": ts.ChoiceAnswer(
                choice="negative",
                confidence=0.6,
                probabilities={"positive": 0.1, "negative": 0.6, "neutral": 0.3},
            ),
            "x": ts.ScoreAnswer(
                score=2.2, confidence=0.5, legend={0: "a", 1: "b", 2: "c"},
                probabilities={0: 0.1, 1: 0.2, 2: 0.7},
            ),
        },
    )  # fmt: skip


@pytest.fixture
def client(mocker):
    c = TypeSafeJevClient("jev-1.13.0", api_key="k", requests_per_minute=None, sleep=lambda s: None)
    c._client = mocker.Mock()
    return c


QUESTIONS = {"s": NoulQuestion("is it?"), "c": ChoiceQuestion(("positive", "negative"))}


def test_sdk_response_converted_to_our_types(client):
    client._client.system_one.return_value = _sdk_response()
    out = client.ask(STATE, QUESTIONS)
    assert out.model == "jev-1.13.0" and out.input_tokens == 321
    assert out.answers["s"] == NoulAnswer(0.8)
    assert out.answers["c"].choice == "negative"
    assert out.answers["x"].n_levels == 3 and out.answers["x"].probabilities[2] == 0.7
    kwargs = client._client.system_one.call_args.kwargs
    assert kwargs["model"] == "jev-1.13.0"
    assert kwargs["questions"]["s"] == {"type": "noul", "instructions": "is it?"}


def test_rate_limit_is_retried_then_succeeds(client):
    limited = ts.TypeSafeRateLimitError(429, "slow down", httpx2.Headers())
    client._client.system_one.side_effect = [limited, limited, _sdk_response()]
    assert client.ask(STATE, QUESTIONS).model == "jev-1.13.0"
    assert client._client.system_one.call_count == 3


def test_gives_up_after_max_attempts(client):
    client.max_attempts = 2
    server = ts.TypeSafeInternalServerError(503, "down", httpx2.Headers())
    client._client.system_one.side_effect = server
    with pytest.raises(JevTransientError):
        client.ask(STATE, QUESTIONS)
    assert client._client.system_one.call_count == 2


def test_oversized_input_maps_to_too_large_and_is_not_retried(client):
    for error in (
        ts.TypeSafeAPIError(413, None, httpx2.Headers()),
        ts.TypeSafeBadRequestError(400, "input exceeds the maximum length", httpx2.Headers()),
    ):
        client._client.system_one.reset_mock()
        client._client.system_one.side_effect = error
        with pytest.raises(JevTooLargeError):
            client.ask(STATE, QUESTIONS)
        assert client._client.system_one.call_count == 1


def test_recorded_real_error_body_is_detected_as_too_large(client):
    recorded = _fixture("recorded_error_too_large.json")
    assert recorded["status"] == 400
    error = ts.TypeSafeBadRequestError(recorded["status"], recorded["body"], httpx2.Headers())
    client._client.system_one.side_effect = error
    with pytest.raises(JevTooLargeError):
        client.ask(STATE, QUESTIONS)
    assert client._client.system_one.call_count == 1


def test_other_api_errors_are_permanent(client):
    client._client.system_one.side_effect = ts.TypeSafeAuthenticationError(
        401, "bad key", httpx2.Headers()
    )
    with pytest.raises(JevError) as err:
        client.ask(STATE, QUESTIONS)
    assert not isinstance(err.value, JevTransientError | JevTooLargeError)
    assert client._client.system_one.call_count == 1


def test_latest_alias_rejected_and_key_required_only_on_use(monkeypatch):
    with pytest.raises(JevError, match="Pin"):
        TypeSafeJevClient("jev-latest", api_key="k")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    c = TypeSafeJevClient("jev-1.13.0")  # construction without a key is fine
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        c.ask(STATE, QUESTIONS)


def test_import_and_construction_make_no_sdk_client():
    code = (
        "from amazon_recsys.enrichment.jev_client import TypeSafeJevClient;"
        "c = TypeSafeJevClient('jev-1.13.0');"
        "assert c._client is None"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], env={}, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --- fake ---------------------------------------------------------------------------------


def test_fake_returns_recorded_json_exceptions_and_callables():
    recorded = _fixture("response_mixed.json")
    fake = FakeJevClient(lambda state, questions: recorded)
    out = fake.ask(STATE, QUESTIONS)
    assert isinstance(out, JevResponse) and "format_followed" in out.answers
    assert fake.call_count == 1 and fake.calls[0][0] == STATE

    fake = FakeJevClient(lambda s, q: JevTooLargeError("too big"))
    with pytest.raises(JevTooLargeError):
        fake.ask(STATE, QUESTIONS)

    # property-style: answers derived from the questions asked, no files needed
    def responder(state, questions):
        return JevResponse(
            "jev-test", {name: NoulAnswer(0.5) for name in questions if name.startswith("s")}
        )

    assert set(FakeJevClient(responder).ask(STATE, QUESTIONS).answers) == {"s"}


# --- integration (real API; never run in CI) ----------------------------------------------

RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION") == "1" and os.environ.get("TYPESAFE_API_KEY")


@pytest.mark.integration
@pytest.mark.skipif(not RUN_INTEGRATION, reason="needs RUN_INTEGRATION=1 and TYPESAFE_API_KEY")
def test_one_real_call_per_question_type_matches_recorded_schema():
    model = os.environ.get("JEV_TEST_MODEL", "jev-1.13.0")
    real = TypeSafeJevClient(model)
    questions = {
        "n": NoulQuestion("Is the extracted aspect mentioned in the review?"),
        "c": ChoiceQuestion(("positive", "negative", "neutral"), "Sentiment toward the aspect?"),
        "s": ScoreQuestion(("Misses aspects.", "Covers some.", "Covers all."), "Completeness?"),
    }
    out = real.ask(STATE, questions)
    assert out.model.startswith("jev-")
    assert isinstance(out.answers["n"], NoulAnswer) and 0 <= out.answers["n"].noul <= 1
    assert isinstance(out.answers["c"], ChoiceAnswer)
    assert sum(out.answers["c"].probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert isinstance(out.answers["s"], ScoreAnswer) and out.answers["s"].n_levels == 3


# --- recorded from the real API ------------------------------------------------------------


def test_recorded_full_review_parses_with_all_question_types():
    raw = _fixture("recorded_full_review.json")
    response = parse_response(raw)
    assert response.model == "jev-1.13.0"
    kinds = {type(a) for a in response.answers.values()}
    assert kinds == {NoulAnswer, ChoiceAnswer, ScoreAnswer}
    assert {"format_followed", "completeness", "overall_sentiment"} <= set(response.answers)
    for answer in response.answers.values():
        if isinstance(answer, NoulAnswer):
            assert 0.0 <= answer.noul <= 1.0
        else:
            assert 0.0 <= answer.confidence <= 1.0
            assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-2)
    assert response.input_tokens and response.input_tokens > 0


def test_recorded_review_level_only_response():
    response = parse_response(_fixture("recorded_no_aspects.json"))
    assert set(response.answers) == {"completeness", "overall_sentiment"}
    assert all(isinstance(a, ScoreAnswer) for a in response.answers.values())
    assert response.answers["completeness"].n_levels == 4
    assert response.answers["overall_sentiment"].n_levels == 5


def test_probabilities_rounded_to_two_decimals_are_accepted():
    """Real Jev output: a 4-level score whose rounded probabilities sum to 0.99."""
    answer = {
        "type": "score",
        "score": 1.4,
        "confidence": 0.55,
        "probabilities": {"0": 0.0, "1": 0.55, "2": 0.4, "3": 0.04},
    }
    response = parse_response({"model": "jev-1.13.0", "usage": {}, "answers": {"c": answer}})
    assert response.answers["c"].n_levels == 4
    off = dict(answer, probabilities={"0": 0.0, "1": 0.5, "2": 0.3, "3": 0.04})  # sums to 0.84
    with pytest.raises(JevError):
        parse_response({"model": "jev-1.13.0", "usage": {}, "answers": {"c": off}})
