"""Extractor LLM clients: protocol, lazy Azure OpenAI client, retries, rate limit, fake."""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


def estimate_tokens(text: str) -> int:
    """Deterministic rough token count shared by every dry-run: ceil(characters / 4)."""
    return math.ceil(len(text) / 4)


class LLMError(RuntimeError):
    """A permanent LLM failure (bad request, auth, missing configuration)."""


class TransientLLMError(LLMError):
    """A failure worth retrying (rate limit, timeout, 5xx, connection error)."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LLMClient(Protocol):
    def complete(
        self, messages: list[dict[str, str]], *, review_id: str | None = None, **params: Any
    ) -> LLMResponse: ...


# --- retries and rate limiting ------------------------------------------------------------


def retry_call(
    fn: Callable[[], T],
    *,
    max_attempts: int,
    backoff_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    retry_on: tuple[type[BaseException], ...] = (TransientLLMError,),
) -> T:
    """Call `fn`, retrying `retry_on` errors with exponential backoff; re-raise the last one."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retry_on:
            if attempt == max_attempts:
                raise
            sleep(backoff_seconds * 2 ** (attempt - 1))
    raise AssertionError("unreachable")  # pragma: no cover


class RateLimiter:
    """Blocks so that no more than `requests_per_minute` calls start per minute."""

    def __init__(
        self,
        requests_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 60.0 / requests_per_minute
        self._clock, self._sleep = clock, sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:  # reserve a slot; sleeping happens while holding it, spacing callers
            now = self._clock()
            if now < self._next:
                self._sleep(self._next - now)
                now = self._next
            self._next = now + self._interval


class RetryingLLMClient:
    """Wraps any LLMClient with retries and an optional rate limiter."""

    def __init__(
        self,
        client: LLMClient,
        max_attempts: int = 5,
        backoff_seconds: float = 2.0,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client, self._limiter, self._sleep = client, limiter, sleep
        self._max_attempts, self._backoff = max_attempts, backoff_seconds

    def complete(
        self, messages: list[dict[str, str]], *, review_id: str | None = None, **params: Any
    ) -> LLMResponse:
        def call() -> LLMResponse:
            if self._limiter is not None:
                self._limiter.wait()
            return self._client.complete(messages, review_id=review_id, **params)

        return retry_call(
            call,
            max_attempts=self._max_attempts,
            backoff_seconds=self._backoff,
            sleep=self._sleep,
        )


# --- OpenAI-SDK providers (Azure OpenAI, OpenRouter) ---------------------------------------


def _chat_complete(
    sdk: Any, model: str, messages: list[dict[str, str]], params: dict[str, Any]
) -> LLMResponse:
    """One chat-completions call through an OpenAI-SDK client; maps SDK errors to ours."""
    import openai

    transient = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )
    start = time.perf_counter()
    try:
        out = sdk.chat.completions.create(model=model, messages=messages, **params)
    except transient as err:
        raise TransientLLMError(str(err)) from err
    except openai.OpenAIError as err:
        raise LLMError(str(err)) from err
    choices = getattr(out, "choices", None)
    if not choices:  # OpenRouter reports upstream failures as an error body with no choices
        err_body = getattr(out, "error", None)
        raise TransientLLMError(f"empty response from provider: {err_body or 'no choices'}")
    usage = getattr(out, "usage", None)
    return LLMResponse(
        text=choices[0].message.content or "",
        model=getattr(out, "model", model),
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        latency_s=time.perf_counter() - start,
    )


class AzureOpenAIClient:
    """Azure OpenAI chat client. The SDK client is created on first use, never at import."""

    def __init__(
        self,
        endpoint: str | None,
        api_key: str | None,
        deployment: str,
        api_version: str = "2024-10-21",
        timeout_s: float = 60.0,
    ) -> None:
        self.endpoint, self.api_key = endpoint, api_key
        self.deployment, self.api_version, self.timeout_s = deployment, api_version, timeout_s
        self._client: Any = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, deployment: str) -> AzureOpenAIClient:
        return cls(
            os.environ.get("AZURE_OPENAI_ENDPOINT"), os.environ.get("AZURE_OPENAI_KEY"), deployment
        )

    def _sdk(self) -> Any:
        with self._lock:
            if self._client is None:
                if not self.endpoint or not self.api_key:
                    raise LLMError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set")
                from openai import AzureOpenAI

                self._client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                    timeout=self.timeout_s,
                    max_retries=0,  # retries are handled by RetryingLLMClient
                )
            return self._client

    def complete(
        self, messages: list[dict[str, str]], *, review_id: str | None = None, **params: Any
    ) -> LLMResponse:
        return _chat_complete(self._sdk(), self.deployment, messages, params)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """OpenRouter chat client (OpenAI-compatible). `model` is an OpenRouter slug such as
    `mistralai/mistral-small-24b-instruct-2501`. The SDK client is created on first use."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str = OPENROUTER_BASE_URL,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key, self.model = api_key, model
        self.base_url, self.timeout_s = base_url, timeout_s
        self._client: Any = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, model: str) -> OpenRouterClient:
        return cls(os.environ.get("OPENROUTER_API_KEY"), model)

    def _sdk(self) -> Any:
        with self._lock:
            if self._client is None:
                if not self.api_key:
                    raise LLMError("OPENROUTER_API_KEY must be set (see .env.example)")
                from openai import OpenAI

                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self.timeout_s,
                    max_retries=0,
                )
            return self._client

    def complete(
        self, messages: list[dict[str, str]], *, review_id: str | None = None, **params: Any
    ) -> LLMResponse:
        if "max_completion_tokens" in params:  # OpenRouter's parameter is max_tokens
            params["max_tokens"] = params.pop("max_completion_tokens")
        return _chat_complete(self._sdk(), self.model, messages, params)


def make_llm_client(
    provider: str,
    model: str,
    max_attempts: int = 5,
    backoff_seconds: float = 2.0,
    requests_per_minute: int | None = None,
) -> RetryingLLMClient:
    """Build the configured provider client wrapped with retries and a rate limit."""
    if provider == "azure":
        base: LLMClient = AzureOpenAIClient.from_env(model)
    elif provider == "openrouter":
        base = OpenRouterClient.from_env(model)
    else:
        raise LLMError(f"Unknown extractor provider {provider!r}; use azure or openrouter")
    limiter = RateLimiter(requests_per_minute) if requests_per_minute else None
    return RetryingLLMClient(base, max_attempts, backoff_seconds, limiter)


# --- fake for tests -----------------------------------------------------------------------


class FakeLLMClient:
    """Returns canned responses keyed by review_id; a value may be text, LLMResponse or an
    exception instance (raised). Counts calls so tests can assert on cache hits."""

    def __init__(
        self, responses: Mapping[str, str | LLMResponse | BaseException], model: str = "fake-llm"
    ) -> None:
        self.responses = dict(responses)
        self.model = model
        self.calls: list[str | None] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def complete(
        self, messages: list[dict[str, str]], *, review_id: str | None = None, **params: Any
    ) -> LLMResponse:
        self.calls.append(review_id)
        if review_id not in self.responses:
            raise KeyError(f"No canned response for review_id={review_id!r}")
        value = self.responses[review_id]
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, LLMResponse):
            return value
        tokens = sum(len(m["content"]) for m in messages) // 4
        return LLMResponse(value, self.model, tokens, len(value) // 4, 0.0)
