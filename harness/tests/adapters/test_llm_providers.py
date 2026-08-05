"""Tests for live-provider adapter contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from trialagentbench_harness.adapters.llm_providers import (
    ChatCompletionsProvider,
    ProviderRouting,
    ResponsesProvider,
    TransientProviderResponseError,
    _is_attributed_openrouter_upstream_error,
    _retry,
    get_provider,
)


class _Completions:
    def __init__(self, *, upstream_provider: str) -> None:
        self.upstream_provider = upstream_provider
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> Any:
        self.kwargs = kwargs
        message = SimpleNamespace(content=None, tool_calls=None, model_dump=lambda mode: {"role": "assistant"})
        usage = SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            model_extra={"cost": 0.001},
        )
        return SimpleNamespace(
            id="response-1",
            model="model-1",
            created=1,
            model_extra={"provider": self.upstream_provider},
            usage=usage,
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
        )


def _provider(*, upstream_provider: str) -> tuple[ChatCompletionsProvider, _Completions]:
    provider = ChatCompletionsProvider.__new__(ChatCompletionsProvider)
    provider.model = "model-1"
    provider._routing = ProviderRouting(provider="openrouter", openrouter_provider="PinnedProvider")
    provider._send_temperature = False
    provider._decoding_seed = None
    provider._bad_request_error_type = BadRequestError
    provider._request_timeout_s = 300.0
    completions = _Completions(upstream_provider=upstream_provider)
    provider._client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    return provider, completions


class _ResponseItem(SimpleNamespace):
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return dict(self.raw)


class _Responses:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> Any:
        self.kwargs = kwargs
        reasoning = _ResponseItem(
            type="reasoning",
            raw={
                "id": "reasoning-1",
                "type": "reasoning",
                "summary": [],
                "encrypted_content": "opaque-reasoning-state",
            },
        )
        function_call = _ResponseItem(
            type="function_call",
            call_id="call-1",
            name="execute_code",
            arguments='{"code":"print(1)"}',
            raw={
                "id": "function-1",
                "type": "function_call",
                "call_id": "call-1",
                "name": "execute_code",
                "arguments": '{"code":"print(1)"}',
                "status": "completed",
            },
        )
        usage = SimpleNamespace(input_tokens=3, output_tokens=5, total_tokens=8)
        return SimpleNamespace(
            id="resp-1",
            model="model-1",
            created_at=123.0,
            status="completed",
            error=None,
            incomplete_details=None,
            output=[reasoning, function_call],
            usage=usage,
        )


def _responses_provider() -> tuple[ResponsesProvider, _Responses]:
    provider = ResponsesProvider.__new__(ResponsesProvider)
    provider.model = "model-1"
    provider._routing = ProviderRouting(provider="openai_responses")
    provider.telemetry_route = "openai_responses"
    provider._send_temperature = False
    provider._request_timeout_s = 300.0
    responses = _Responses()
    provider._client = cast(Any, SimpleNamespace(responses=responses))
    return provider, responses


def test_openrouter_pin_disables_provider_fallbacks() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")

    response = provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert response.metadata.upstream_provider == "PinnedProvider"
    assert completions.kwargs is not None
    assert completions.kwargs["extra_body"] == {"provider": {"order": ["PinnedProvider"], "allow_fallbacks": False}}


def test_openrouter_reasoning_condition_preserves_the_exact_route_pin() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    provider._reasoning_effort = "high"
    provider._exclude_reasoning = True

    provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert completions.kwargs is not None
    assert completions.kwargs["extra_body"] == {
        "provider": {"order": ["PinnedProvider"], "allow_fallbacks": False},
        "reasoning": {"effort": "high", "exclude": True},
    }


def test_responses_provider_translates_tools_and_retains_replay_state() -> None:
    provider, responses = _responses_provider()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "Run Python.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]

    response = provider.generate_turn(
        [{"role": "system", "content": "Analyse."}, {"role": "user", "content": "Begin."}],
        tools=tools,
        temperature=0.0,
        max_tokens=128,
        tool_choice="required",
    )

    assert responses.kwargs is not None
    assert responses.kwargs["store"] is False
    assert responses.kwargs["include"] == ["reasoning.encrypted_content"]
    assert responses.kwargs["max_output_tokens"] == 128
    assert responses.kwargs["tool_choice"] == "required"
    assert responses.kwargs["tools"] == [
        {
            "type": "function",
            "name": "execute_code",
            "description": "Run Python.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        }
    ]
    assert response.tool_calls[0].id == "call-1"
    assert response.usage == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    assert response.metadata.response_id == "resp-1"
    assert response.metadata.created_unix == 123
    assert isinstance(response.raw, dict)
    assert response.raw["provider_state"][0]["encrypted_content"] == "opaque-reasoning-state"


def test_responses_provider_replays_output_items_before_tool_results() -> None:
    provider, responses = _responses_provider()
    first = provider.generate_turn(
        [{"role": "user", "content": "Begin."}],
        temperature=0.0,
        max_tokens=128,
    )
    assert isinstance(first.raw, dict)

    provider.generate_turn(
        [
            {"role": "user", "content": "Begin."},
            first.raw,
            {"role": "tool", "tool_call_id": "call-1", "content": "1"},
        ],
        temperature=0.0,
        max_tokens=128,
    )

    assert responses.kwargs is not None
    input_items = cast(list[dict[str, object]], responses.kwargs["input"])
    assert input_items[1]["type"] == "reasoning"
    assert input_items[2]["type"] == "function_call"
    assert input_items[3] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "1",
    }


def test_provider_sends_declared_decoding_seed() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    provider._decoding_seed = 17

    provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert completions.kwargs is not None
    assert completions.kwargs["seed"] == 17


def test_provider_omits_undeclared_decoding_seed() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")

    provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert completions.kwargs is not None
    assert "seed" not in completions.kwargs


def test_provider_requires_one_of_the_offered_tools_when_requested() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    tools = [{"type": "function", "function": {"name": "submit_response", "parameters": {"type": "object"}}}]

    provider.generate_turn(
        [{"role": "user", "content": "submit"}],
        tools=tools,
        temperature=0.0,
        max_tokens=8,
        tool_choice="required",
    )

    assert completions.kwargs is not None
    assert completions.kwargs["tools"] == tools
    assert completions.kwargs["tool_choice"] == "required"


def test_provider_rejects_required_tool_choice_without_tools() -> None:
    provider, _ = _provider(upstream_provider="PinnedProvider")

    with pytest.raises(ValueError, match="requires at least one provider tool"):
        provider.generate_turn(
            [{"role": "user", "content": "submit"}],
            temperature=0.0,
            max_tokens=8,
            tool_choice="required",
        )


def test_chat_completions_rejects_responses_provider_state() -> None:
    provider, _ = _provider(upstream_provider="PinnedProvider")

    with pytest.raises(ValueError, match="cannot be sent through Chat Completions"):
        provider.generate_turn(
            [
                {"role": "user", "content": "test"},
                {
                    "role": "assistant",
                    "content": None,
                    "provider_state": [{"type": "reasoning", "encrypted_content": "opaque"}],
                },
            ],
            temperature=0.0,
            max_tokens=8,
        )


def test_openrouter_pin_rejects_returned_provider_drift() -> None:
    provider, _ = _provider(upstream_provider="FallbackProvider")

    with pytest.raises(RuntimeError, match="violated the requested upstream-provider pin"):
        provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)


def test_retry_returns_exact_attempt_and_backoff_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    def request() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise APIConnectionError(request=httpx.Request("POST", "https://provider.invalid"))
        return "response"

    monkeypatch.setattr("trialagentbench_harness.adapters.llm_providers.time.sleep", sleeps.append)

    response, telemetry = _retry(request, max_attempts=5, base_delay=2.0)

    assert response == "response"
    assert telemetry.request_attempts == 3
    assert telemetry.transient_failure_count == 2
    assert telemetry.backoff_seconds == 6.0
    assert sleeps == [2.0, 4.0]


def test_provider_retries_empty_completion_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    valid_create = completions.create
    attempts = 0
    sleeps: list[float] = []

    def create(**kwargs: object) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return SimpleNamespace(choices=None)
        return valid_create(**kwargs)

    completions.create = create  # type: ignore[method-assign]
    monkeypatch.setattr("trialagentbench_harness.adapters.llm_providers.time.sleep", sleeps.append)

    response = provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert attempts == 2
    assert response.metadata.request_attempts == 2
    assert response.metadata.transient_failure_count == 1
    assert response.metadata.backoff_seconds == 2.0
    assert sleeps == [2.0]


def test_provider_retries_completion_inference_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    valid_create = completions.create
    attempts = 0
    sleeps: list[float] = []

    def create(**kwargs: object) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            response = valid_create(**kwargs)
            response.choices[0].finish_reason = "error"
            return response
        return valid_create(**kwargs)

    completions.create = create  # type: ignore[method-assign]
    monkeypatch.setattr("trialagentbench_harness.adapters.llm_providers.time.sleep", sleeps.append)

    response = provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert attempts == 2
    assert response.metadata.request_attempts == 2
    assert response.metadata.transient_failure_count == 1
    assert response.metadata.backoff_seconds == 2.0
    assert sleeps == [2.0]


def test_provider_attempt_timeout_is_capped_to_remaining_deadline() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")

    provider.generate_turn(
        [{"role": "user", "content": "test"}],
        temperature=0.0,
        max_tokens=8,
        timeout_seconds=12.0,
    )

    assert completions.kwargs is not None
    assert 0.0 < cast(float, completions.kwargs["timeout"]) <= 12.0


def test_retry_exhausts_on_empty_provider_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trialagentbench_harness.adapters.llm_providers.time.sleep", lambda _: None)

    with pytest.raises(TransientProviderResponseError, match="no completion choices") as raised:
        _retry(
            lambda: (_ for _ in ()).throw(TransientProviderResponseError("Provider returned no completion choices.")),
            max_attempts=2,
            base_delay=0.0,
        )
    telemetry = raised.value.retry_telemetry
    assert telemetry.request_attempts == 2
    assert telemetry.transient_failure_count == 2
    assert telemetry.backoff_seconds == 0.0


def test_retry_caps_backoff_to_remaining_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = 10.0
    sleeps: list[float] = []

    def monotonic() -> float:
        return clock

    def sleep(seconds: float) -> None:
        nonlocal clock
        sleeps.append(seconds)
        clock += seconds

    monkeypatch.setattr("trialagentbench_harness.adapters.llm_providers.time.monotonic", monotonic)
    monkeypatch.setattr("trialagentbench_harness.adapters.llm_providers.time.sleep", sleep)

    with pytest.raises(TimeoutError, match="retry backoff") as raised:
        _retry(
            lambda: (_ for _ in ()).throw(
                APIConnectionError(request=httpx.Request("POST", "https://provider.invalid"))
            ),
            max_attempts=5,
            base_delay=2.0,
            timeout_seconds=1.25,
        )

    assert sleeps == [1.25]
    assert raised.value.retry_telemetry.request_attempts == 1
    assert raised.value.retry_telemetry.transient_failure_count == 1
    assert raised.value.retry_telemetry.backoff_seconds == 1.25


def _bad_request(*, body: object) -> BadRequestError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return BadRequestError(
        "bad request",
        response=httpx.Response(400, request=request),
        body=body,
    )


def test_openrouter_upstream_error_requires_exact_provider_attribution() -> None:
    attributed = _bad_request(
        body={
            "message": "Provider returned error",
            "metadata": {"provider_name": "PinnedProvider"},
        }
    )
    local_error = _bad_request(body={"message": "Invalid model identifier"})
    wrong_provider = _bad_request(
        body={
            "message": "Provider returned error",
            "metadata": {"provider_name": "OtherProvider"},
        }
    )

    assert _is_attributed_openrouter_upstream_error(attributed, expected_provider="PinnedProvider")
    assert not _is_attributed_openrouter_upstream_error(local_error, expected_provider="PinnedProvider")
    assert not _is_attributed_openrouter_upstream_error(wrong_provider, expected_provider="PinnedProvider")
    assert not _is_attributed_openrouter_upstream_error(attributed, expected_provider=None)


def test_provider_retries_error_attributed_to_pinned_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    valid_create = completions.create
    attempts = 0
    sleeps: list[float] = []

    def create(**kwargs: object) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _bad_request(
                body={
                    "message": "Provider returned error",
                    "metadata": {"provider_name": "PinnedProvider"},
                }
            )
        return valid_create(**kwargs)

    completions.create = create  # type: ignore[method-assign]
    monkeypatch.setattr("trialagentbench_harness.adapters.llm_providers.time.sleep", sleeps.append)

    response = provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert attempts == 2
    assert response.metadata.request_attempts == 2
    assert response.metadata.transient_failure_count == 1
    assert sleeps == [2.0]


def test_provider_does_not_retry_unattributed_bad_request(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    attempts = 0

    def create(**_: object) -> Any:
        nonlocal attempts
        attempts += 1
        raise _bad_request(body={"message": "Invalid model identifier"})

    completions.create = create  # type: ignore[method-assign]
    monkeypatch.setattr("trialagentbench_harness.adapters.llm_providers.time.sleep", lambda _: None)

    with pytest.raises(BadRequestError):
        provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)
    assert attempts == 1


def test_provider_disables_sdk_retries_and_applies_explicit_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", _Client)

    ChatCompletionsProvider(
        model="model-1",
        routing=ProviderRouting(provider="openai"),
        send_temperature=False,
        timeout_s=123.0,
    )

    assert captured["timeout"] == 123.0
    assert captured["max_retries"] == 0


@pytest.mark.parametrize("decoding_seed", [True, -1])
def test_provider_rejects_invalid_decoding_seed(
    monkeypatch: pytest.MonkeyPatch,
    decoding_seed: object,
) -> None:
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace())

    with pytest.raises(ValueError, match="decoding seed"):
        ChatCompletionsProvider(
            model="model-1",
            routing=ProviderRouting(provider="openai"),
            send_temperature=False,
            timeout_s=123.0,
            decoding_seed=cast(Any, decoding_seed),
        )


def test_responses_provider_rejects_unsupported_decoding_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace())

    with pytest.raises(ValueError, match="does not support a decoding seed"):
        ResponsesProvider(
            model="model-1",
            routing=ProviderRouting(provider="openai_responses"),
            send_temperature=False,
            timeout_s=123.0,
            decoding_seed=17,
        )


def test_provider_factory_selects_direct_responses_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace())

    provider = get_provider(
        "model-1",
        routing=ProviderRouting(provider="openai_responses"),
        send_temperature=False,
        timeout_s=123.0,
    )

    assert isinstance(provider, ResponsesProvider)
    assert provider.telemetry_route == "openai_responses"
