"""Judge client for TypeSafe's Jev (System One): question types, parsing, real and fake clients.

Jev does not generate text: it takes a `state` and typed questions and returns probabilities.
Answers are stored exactly as returned (never rounded or thresholded here).
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from amazon_recsys.enrichment.llm_client import RateLimiter, retry_call

PROB_TOLERANCE = 1e-6
# Jev rounds each probability to two decimals, so a distribution over n options can be off by
# up to n * 0.005 (n <= 10 for Score): accept sums within 0.06 of 1.
SUM_TOLERANCE = 0.06


class JevError(RuntimeError):
    """A permanent Jev failure (bad request, auth, malformed response)."""


class JevTransientError(JevError):
    """Worth retrying: rate limit, 5xx, timeout, connection error."""


class JevTooLargeError(JevError):
    """The state exceeded the model's input limit."""


# --- questions (SDK-independent, serialise to the documented wire shape) ----------------------


@dataclass(frozen=True)
class NoulQuestion:
    instructions: Any = None

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "noul"}
        if self.instructions is not None:
            out["instructions"] = self.instructions
        return out


@dataclass(frozen=True)
class ChoiceQuestion:
    options: tuple[str, ...]
    instructions: Any = None
    descriptions: Mapping[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        descriptions = self.descriptions or {}
        out: dict[str, Any] = {"type": "choice"}
        if self.instructions is not None:
            out["instructions"] = self.instructions
        out["criteria"] = {option: descriptions.get(option) for option in self.options}
        return out


@dataclass(frozen=True)
class ScoreQuestion:
    levels: tuple[Any, ...]  # ordered, worst first
    instructions: Any = None

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "score"}
        if self.instructions is not None:
            out["instructions"] = self.instructions
        out["criteria"] = list(self.levels)
        return out


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


def build_request(
    model: str, state: dict[str, Any] | str, questions: Mapping[str, Question]
) -> dict[str, Any]:
    """The `POST /v1/systemone` body: {"model", "state", "questions"}."""
    if not questions:
        raise JevError("at least one question is required")
    return {
        "model": model,
        "state": state,
        "questions": {name: q.to_payload() for name, q in questions.items()},
    }


# --- answers ------------------------------------------------------------------------------


@dataclass(frozen=True)
class NoulAnswer:
    noul: float  # P(yes)


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float  # expected level, 0 .. n_levels-1 (may be fractional)
    probabilities: dict[int, float]
    confidence: float

    @property
    def n_levels(self) -> int:
        return len(self.probabilities)


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class JevResponse:
    model: str  # version string returned by the API, e.g. "jev-1.13.0"
    answers: dict[str, Answer]
    input_tokens: int | None = None
    output_tokens: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _check_probabilities(name: str, probs: Mapping[Any, float]) -> None:
    total = sum(probs.values())
    if abs(total - 1.0) > SUM_TOLERANCE or any(p < -PROB_TOLERANCE for p in probs.values()):
        raise JevError(f"answer {name!r}: probabilities do not form a distribution (sum={total})")


def parse_answer(name: str, raw: Mapping[str, Any]) -> Answer:
    kind = raw.get("type")
    try:
        if kind == "noul":
            return NoulAnswer(noul=float(raw["noul"]))
        if kind == "choice":
            probs = {str(k): float(v) for k, v in raw["probabilities"].items()}
            _check_probabilities(name, probs)
            return ChoiceAnswer(str(raw["choice"]), probs, float(raw["confidence"]))
        if kind == "score":
            probs_i = {int(k): float(v) for k, v in raw["probabilities"].items()}
            _check_probabilities(name, probs_i)
            return ScoreAnswer(float(raw["score"]), probs_i, float(raw["confidence"]))
    except (KeyError, TypeError, ValueError, AttributeError) as err:
        raise JevError(f"answer {name!r} ({kind}) is malformed: {err!r}") from err
    raise JevError(f"answer {name!r} has unknown type {kind!r}")


def parse_response(payload: Mapping[str, Any]) -> JevResponse:
    """Parse the raw JSON body of a System One response."""
    try:
        usage = payload.get("usage") or {}
        answers = {name: parse_answer(name, raw) for name, raw in dict(payload["answers"]).items()}
        return JevResponse(
            model=str(payload["model"]),
            answers=answers,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
    except KeyError as err:
        raise JevError(f"response is missing {err}") from err


# --- clients ------------------------------------------------------------------------------


class JudgeClient(Protocol):
    def ask(
        self, state: dict[str, Any] | str, questions: Mapping[str, Question]
    ) -> JevResponse: ...


_TOO_LARGE = re.compile(r"too (large|long|big)|max(imum)? (input|context|length)", re.I)


def _is_too_large(err: Any) -> bool:
    """Real API: HTTP 400 with body {"detail": {"error_type": "max_tokens_exceeded"}}.
    HTTP 413 and a few message patterns are accepted as well, in case that shape changes."""
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and detail.get("error_type") == "max_tokens_exceeded":
            return True
    if err.status == 413:
        return True
    return err.status in (400, 422) and bool(_TOO_LARGE.search(str(err)))


class TypeSafeJevClient:
    """Wraps `typesafe_sdk.TypeSafeClient`; created on first use, never at import."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        max_attempts: int = 5,
        backoff_seconds: float = 2.0,
        requests_per_minute: int | None = 300,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if model.endswith("latest"):
            raise JevError("Pin a concrete Jev model version; 'jev-latest' is not reproducible")
        self.model = model
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.timeout_s = timeout_s
        self.max_attempts, self.backoff_seconds = max_attempts, backoff_seconds
        self._limiter = RateLimiter(requests_per_minute) if requests_per_minute else None
        self._sleep = sleep
        self._client: Any = None
        self._lock = threading.Lock()

    def _sdk(self) -> Any:
        with self._lock:
            if self._client is None:
                if not self.api_key:
                    raise JevError("TYPESAFE_API_KEY is not set")
                from typesafe_sdk import RetryPolicy, TypeSafeClient

                self._client = TypeSafeClient(
                    api_key=self.api_key,
                    model=self.model,
                    timeout=self.timeout_s,
                    retry=RetryPolicy(max_retries=0),  # retries are handled here
                )
            return self._client

    def _call(self, state: Any, questions: Mapping[str, Question]) -> JevResponse:
        import typesafe_sdk as ts

        if self._limiter is not None:
            self._limiter.wait()
        payload = build_request(self.model, state, questions)
        try:
            result = self._sdk().system_one(
                state=payload["state"], questions=payload["questions"], model=self.model
            )
        except (ts.TypeSafeRateLimitError, ts.TypeSafeInternalServerError) as err:
            raise JevTransientError(str(err)) from err
        except ts.TypeSafeAPIConnectionError as err:  # includes timeouts
            raise JevTransientError(str(err)) from err
        except ts.TypeSafeAPIError as err:
            if _is_too_large(err):
                raise JevTooLargeError(str(err)) from err
            raise JevError(str(err)) from err
        except ts.TypeSafeError as err:
            raise JevError(str(err)) from err
        return _from_sdk_response(result)

    def ask(self, state: dict[str, Any] | str, questions: Mapping[str, Question]) -> JevResponse:
        kwargs = {"sleep": self._sleep} if self._sleep is not None else {}
        return retry_call(
            lambda: self._call(state, questions),
            max_attempts=self.max_attempts,
            backoff_seconds=self.backoff_seconds,
            retry_on=(JevTransientError,),
            **kwargs,
        )


def _from_sdk_response(result: Any) -> JevResponse:
    answers: dict[str, Answer] = {}
    for name, ans in result.answers.items():
        if hasattr(ans, "noul"):
            answers[name] = NoulAnswer(noul=float(ans.noul))
        elif hasattr(ans, "legend"):
            answers[name] = ScoreAnswer(
                float(ans.score),
                {int(k): float(v) for k, v in ans.probabilities.items()},
                float(ans.confidence),
            )
        else:
            answers[name] = ChoiceAnswer(
                str(ans.choice),
                {str(k): float(v) for k, v in ans.probabilities.items()},
                float(ans.confidence),
            )
    usage = result.usage
    return JevResponse(
        model=result.model,
        answers=answers,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


class FakeJevClient:
    """No-network judge. `responder(state, questions)` returns a JevResponse, a raw response
    dict (parsed like the wire body), or an exception instance (raised). Calls are recorded."""

    def __init__(
        self,
        responder: Callable[[Any, Mapping[str, Question]], JevResponse | dict | BaseException],
    ) -> None:
        self._responder = responder
        self.calls: list[tuple[Any, dict[str, Question]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def ask(self, state: dict[str, Any] | str, questions: Mapping[str, Question]) -> JevResponse:
        self.calls.append((state, dict(questions)))
        out = self._responder(state, questions)
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, dict):
            return parse_response(out)
        return out
