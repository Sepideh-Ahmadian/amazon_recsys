import subprocess
import sys
from types import SimpleNamespace

import pytest

from amazon_recsys.enrichment.llm_client import (
    AzureOpenAIClient,
    FakeLLMClient,
    LLMError,
    RateLimiter,
    RetryingLLMClient,
    TransientLLMError,
    retry_call,
)


def test_import_with_no_env_vars_creates_no_client():
    code = (
        "import amazon_recsys.enrichment.llm_client as m;"
        "c = m.AzureOpenAIClient.from_env('dep');"
        "assert c._client is None"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], env={}, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_missing_credentials_raise_only_on_use(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
    client = AzureOpenAIClient.from_env("dep")  # construction is fine
    with pytest.raises(LLMError, match="AZURE_OPENAI_ENDPOINT"):
        client.complete([{"role": "user", "content": "hi"}])


def test_azure_complete_maps_response_and_creates_sdk_lazily(mocker):
    client = AzureOpenAIClient("https://x", "key", "dep")
    created = mocker.Mock()
    created.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        model="gpt-4o",
    )
    client._client = created
    out = client.complete([{"role": "user", "content": "hi"}], review_id="r1", temperature=0.0)
    assert (out.text, out.model, out.input_tokens, out.output_tokens) == ("hello", "gpt-4o", 7, 3)
    kwargs = created.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "dep" and kwargs["temperature"] == 0.0
    assert "review_id" not in kwargs


def test_azure_maps_sdk_errors(mocker):
    import openai

    client = AzureOpenAIClient("https://x", "key", "dep")
    sdk = mocker.Mock()
    client._client = sdk
    request = mocker.Mock()
    sdk.chat.completions.create.side_effect = openai.APIConnectionError(request=request)
    with pytest.raises(TransientLLMError):
        client.complete([{"role": "user", "content": "x"}])
    sdk.chat.completions.create.side_effect = openai.OpenAIError("bad request")
    with pytest.raises(LLMError) as err:
        client.complete([{"role": "user", "content": "x"}])
    assert not isinstance(err.value, TransientLLMError)


def test_retry_backs_off_exponentially_then_succeeds():
    attempts, sleeps = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TransientLLMError("429")
        return "ok"

    out = retry_call(flaky, max_attempts=5, backoff_seconds=2, sleep=sleeps.append)
    assert out == "ok" and len(attempts) == 3 and sleeps == [2, 4]


def test_retry_gives_up_after_max_attempts_and_surfaces_last_error():
    sleeps, count = [], []

    def always():
        count.append(1)
        raise TransientLLMError(f"fail {len(count)}")

    with pytest.raises(TransientLLMError, match="fail 3"):
        retry_call(always, max_attempts=3, backoff_seconds=1, sleep=sleeps.append)
    assert len(count) == 3 and sleeps == [1, 2]


def test_permanent_errors_are_not_retried():
    count = []

    def boom():
        count.append(1)
        raise LLMError("bad key")

    with pytest.raises(LLMError):
        retry_call(boom, max_attempts=5, backoff_seconds=1, sleep=lambda s: None)
    assert len(count) == 1


def test_retrying_client_wraps_fake_and_passes_review_id():
    fake = FakeLLMClient({"r1": "answer"})
    client = RetryingLLMClient(fake, max_attempts=2, sleep=lambda s: None)
    assert client.complete([{"role": "user", "content": "x"}], review_id="r1").text == "answer"
    assert fake.calls == ["r1"]


def test_retrying_client_retries_fake_transient_error():
    class Flaky(FakeLLMClient):
        def complete(self, messages, *, review_id=None, **params):
            if self.call_count < 1:
                self.calls.append(review_id)
                raise TransientLLMError("503")
            return super().complete(messages, review_id=review_id, **params)

    flaky = Flaky({"r1": "fine"})
    sleeps = []
    out = RetryingLLMClient(flaky, max_attempts=3, backoff_seconds=1, sleep=sleeps.append)
    assert out.complete([{"role": "user", "content": "x"}], review_id="r1").text == "fine"
    assert sleeps == [1]


def test_fake_returns_canned_response_per_review_id_and_raises_errors():
    fake = FakeLLMClient({"a": "A", "b": LLMError("api down")})
    msgs = [{"role": "user", "content": "x" * 40}]
    out = fake.complete(msgs, review_id="a")
    assert out.text == "A" and out.input_tokens == 10
    with pytest.raises(LLMError):
        fake.complete(msgs, review_id="b")
    with pytest.raises(KeyError):
        fake.complete(msgs, review_id="zzz")
    assert fake.call_count == 3


def test_rate_limiter_spaces_calls():
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(60, clock=lambda: now[0], sleep=sleep)  # 1 call / second
    for _ in range(3):
        limiter.wait()
    assert sleeps == [1.0, 1.0]


# --- OpenRouter ---------------------------------------------------------------------------


def _fake_sdk(mocker, content="hi", with_choices=True):
    sdk = mocker.Mock()
    if with_choices:
        sdk.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
            model="mistralai/mistral-small-24b-instruct-2501",
        )
    else:
        sdk.chat.completions.create.return_value = SimpleNamespace(choices=[], error={"code": 502})
    return sdk


def test_openrouter_maps_params_and_response(mocker):
    from amazon_recsys.enrichment.llm_client import OpenRouterClient

    client = OpenRouterClient("key", "mistralai/mistral-small-24b-instruct-2501")
    client._client = _fake_sdk(mocker, '{"overall_sentiment": 0}')
    out = client.complete(
        [{"role": "user", "content": "x"}],
        review_id="r1",
        temperature=0.0,
        max_completion_tokens=50,
    )
    assert (
        out.text == '{"overall_sentiment": 0}' and out.input_tokens == 11 and out.output_tokens == 4
    )
    kwargs = client._client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "mistralai/mistral-small-24b-instruct-2501"
    assert kwargs["max_tokens"] == 50 and "max_completion_tokens" not in kwargs
    assert "review_id" not in kwargs


def test_openrouter_empty_choices_is_a_transient_error(mocker):
    from amazon_recsys.enrichment.llm_client import OpenRouterClient

    client = OpenRouterClient("key", "m")
    client._client = _fake_sdk(mocker, with_choices=False)
    with pytest.raises(TransientLLMError, match="empty response"):
        client.complete([{"role": "user", "content": "x"}])


def test_openrouter_requires_key_only_on_use(monkeypatch):
    from amazon_recsys.enrichment.llm_client import OpenRouterClient

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = OpenRouterClient.from_env("some/model")  # construction is fine
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        client.complete([{"role": "user", "content": "x"}])


def test_make_llm_client_selects_provider_and_rejects_unknown():
    from amazon_recsys.enrichment.llm_client import (
        AzureOpenAIClient,
        OpenRouterClient,
        RetryingLLMClient,
        make_llm_client,
    )

    router = make_llm_client("openrouter", "some/model", requests_per_minute=60)
    assert isinstance(router, RetryingLLMClient) and isinstance(router._client, OpenRouterClient)
    azure = make_llm_client("azure", "dep")
    assert isinstance(azure._client, AzureOpenAIClient)
    with pytest.raises(LLMError, match="Unknown extractor provider"):
        make_llm_client("carrier-pigeon", "x")
