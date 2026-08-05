"""LLM provider adapters for explicit OpenAI-compatible transports.

These are *adapters*, not ports. Import from CLI entrypoints only.
Offline grading and deterministic analysis workflows must not import this module.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, TypeVar, cast

from trialagentbench_harness.contracts.core.config import ReasoningEffortV1
from trialagentbench_harness.ports import (
    LLMProvider,
    LLMResponse,
    LLMResponseMetadata,
    RetryTelemetry,
    ToolCall,
)
from trialagentbench_harness.ports.llm_provider import JsonObject, JsonValue

R = TypeVar("R")


class TransientProviderResponseError(RuntimeError):
    """Provider response was successful at transport level but unusable."""


def _is_attributed_openrouter_upstream_error(
    error: object,
    *,
    expected_provider: str | None,
) -> bool:
    """Return whether OpenRouter attributes an error to the pinned upstream."""
    if expected_provider is None:
        return False
    body = getattr(error, "body", None)
    if not isinstance(body, dict) or body.get("message") != "Provider returned error":
        return False
    metadata = body.get("metadata")
    return isinstance(metadata, dict) and metadata.get("provider_name") == expected_provider


def _retry(
    fn: Callable[[], R],
    *,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    timeout_seconds: float | None = None,
) -> tuple[R, RetryTelemetry]:
    """Run fn() with exponential backoff on transient HTTP/provider errors."""
    # Keep exception surface explicit: do not swallow unexpected errors.
    #
    # This adapter is imported only for opt-in network-backed workflows, so it
    # is acceptable to import provider exceptions lazily at call time.
    from openai import (  # type: ignore
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        RateLimitError,
    )

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    if base_delay < 0.0:
        raise ValueError("base_delay must be non-negative.")
    if timeout_seconds is not None and timeout_seconds <= 0.0:
        raise TimeoutError("Provider request deadline was exhausted before the first attempt.")
    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    last_exc: Exception | None = None
    backoff_seconds = 0.0
    for attempt in range(max_attempts):
        if deadline is not None and time.monotonic() >= deadline:
            error = TimeoutError("Provider request deadline was exhausted before a retry attempt.")
            _attach_retry_telemetry(
                error,
                request_attempts=attempt,
                transient_failure_count=attempt,
                backoff_seconds=backoff_seconds,
            )
            raise error from last_exc
        try:
            return (
                fn(),
                RetryTelemetry(
                    request_attempts=attempt + 1,
                    transient_failure_count=attempt,
                    backoff_seconds=backoff_seconds,
                ),
            )
        except (APIConnectionError, APITimeoutError, RateLimitError, TransientProviderResponseError) as e:
            if attempt == max_attempts - 1:
                _attach_retry_telemetry(
                    e,
                    request_attempts=attempt + 1,
                    transient_failure_count=attempt + 1,
                    backoff_seconds=backoff_seconds,
                )
                raise
            last_exc = e
            # Deterministic backoff: no jitter. This keeps reruns predictable and
            # avoids introducing non-reproducible timing into provider-backed
            # workflows.
            delay = base_delay * (2**attempt)
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                if delay >= remaining:
                    if remaining:
                        time.sleep(remaining)
                        backoff_seconds += remaining
                    error = TimeoutError("Provider request deadline was exhausted during retry backoff.")
                    _attach_retry_telemetry(
                        error,
                        request_attempts=attempt + 1,
                        transient_failure_count=attempt + 1,
                        backoff_seconds=backoff_seconds,
                    )
                    raise error from e
            backoff_seconds += delay
            time.sleep(delay)
        except APIStatusError as e:
            # Status-code driven retry policy. Do not retry client errors:
            # those are almost always permanent (bad request, invalid model id,
            # auth failure, etc.). Retry rate limits and transient server
            # failures only.
            status = getattr(e, "status_code", None)
            retriable = status in {429, 500, 502, 503, 504}
            if (not retriable) or attempt == max_attempts - 1:
                _attach_retry_telemetry(
                    e,
                    request_attempts=attempt + 1,
                    transient_failure_count=(attempt + 1 if retriable else attempt),
                    backoff_seconds=backoff_seconds,
                )
                raise
            last_exc = e
            delay = base_delay * (2**attempt)
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                if delay >= remaining:
                    if remaining:
                        time.sleep(remaining)
                        backoff_seconds += remaining
                    error = TimeoutError("Provider request deadline was exhausted during retry backoff.")
                    _attach_retry_telemetry(
                        error,
                        request_attempts=attempt + 1,
                        transient_failure_count=attempt + 1,
                        backoff_seconds=backoff_seconds,
                    )
                    raise error from e
            backoff_seconds += delay
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry loop exhausted without returning or raising")


def _attach_retry_telemetry(
    error: BaseException,
    *,
    request_attempts: int,
    transient_failure_count: int,
    backoff_seconds: float,
) -> None:
    """Attach typed observations to the exact exception propagated to the caller."""

    error.retry_telemetry = RetryTelemetry(  # type: ignore[attr-defined]
        request_attempts=request_attempts,
        transient_failure_count=transient_failure_count,
        backoff_seconds=backoff_seconds,
    )


@dataclass
class ProviderRouting:
    """Explicit API transport and OpenRouter upstream routing pin."""

    provider: Literal["openai", "openai_responses", "openrouter"]
    openrouter_provider: str | None = None


def validate_provider_configuration(
    *,
    model: str,
    routing: ProviderRouting,
) -> None:
    """Validate provider routing before a live request is sent.

    Parameters
    ----------
    model:
        Exact model identifier sent to the selected provider.
    routing:
        Exact provider transport and upstream route.

    Raises
    ------
    ValueError
        If routing is inconsistent or cannot identify the OpenRouter upstream.
    """
    if not model.strip():
        raise ValueError("model must be a non-empty exact provider model identifier")
    if routing.openrouter_provider and routing.provider != "openrouter":
        raise ValueError("openrouter_provider can only be set when provider is 'openrouter'")
    if routing.provider == "openrouter" and not routing.openrouter_provider:
        raise ValueError("OpenRouter runs require an explicit upstream provider pin")


def _validate_decoding_seed(decoding_seed: int | None) -> None:
    """Reject values that cannot be transported as provider decoding seeds."""

    if isinstance(decoding_seed, bool) or (
        decoding_seed is not None and (not isinstance(decoding_seed, int) or decoding_seed < 0)
    ):
        raise ValueError("Provider decoding seed must be a non-negative integer or None.")


class ChatCompletionsProvider:
    """OpenAI Chat Completions-compatible provider (OpenAI or OpenRouter)."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        routing: ProviderRouting,
        send_temperature: bool,
        timeout_s: float,
        decoding_seed: int | None = None,
        reasoning_effort: ReasoningEffortV1 | None = None,
        exclude_reasoning: bool = True,
    ) -> None:
        _validate_decoding_seed(decoding_seed)
        if reasoning_effort is not None and routing.provider != "openrouter":
            raise ValueError("Explicit reasoning effort is supported only through the OpenRouter transport.")
        # Import lazily so offline paths can import the harness without the
        # `providers` extra installed.
        from openai import BadRequestError, OpenAI

        self.model = model
        self._routing = routing
        self.telemetry_route = (
            f"openrouter:{routing.openrouter_provider}"
            if routing.openrouter_provider is not None
            else routing.provider
        )
        self._send_temperature = send_temperature
        self._decoding_seed = decoding_seed
        self._reasoning_effort = reasoning_effort
        self._exclude_reasoning = exclude_reasoning
        self._bad_request_error_type = BadRequestError
        if timeout_s <= 0.0:
            raise ValueError("Provider request timeout must be positive.")
        self._request_timeout_s = float(timeout_s)
        client_kwargs: dict[str, Any] = {
            "timeout": self._request_timeout_s,
            # The harness owns and records retries. SDK retries would be
            # invisible in the per-response provenance.
            "max_retries": 0,
        }
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = OpenAI(**client_kwargs)

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> LLMResponse:
        if tool_choice == "required" and not tools:
            raise ValueError("tool_choice='required' requires at least one provider tool.")
        if any(message.get("provider_state") is not None for message in messages):
            raise ValueError("Responses provider state cannot be sent through Chat Completions.")
        request_deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_completion_tokens": int(max_tokens),
        }
        if self._send_temperature:
            kwargs["temperature"] = float(temperature)
        if self._decoding_seed is not None:
            kwargs["seed"] = self._decoding_seed
        if tools:
            kwargs["tools"] = list(tools)
            if tool_choice == "required":
                kwargs["tool_choice"] = "required"

        extra_body: dict[str, Any] = {}
        if self._routing.openrouter_provider:
            extra_body["provider"] = {
                "order": [self._routing.openrouter_provider],
                "allow_fallbacks": False,
            }
        reasoning_effort = getattr(self, "_reasoning_effort", None)
        if reasoning_effort is not None:
            extra_body["reasoning"] = {
                "effort": reasoning_effort,
                "exclude": getattr(self, "_exclude_reasoning", True),
            }
        if extra_body:
            kwargs["extra_body"] = extra_body

        def request() -> Any:
            try:
                request_kwargs = dict(kwargs)
                if request_deadline is not None:
                    remaining = request_deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError("Provider request deadline was exhausted before transport.")
                    request_kwargs["timeout"] = min(self._request_timeout_s, remaining)
                response = self._client.chat.completions.create(**request_kwargs)
            except self._bad_request_error_type as error:
                if not _is_attributed_openrouter_upstream_error(
                    error,
                    expected_provider=self._routing.openrouter_provider,
                ):
                    raise
                raise TransientProviderResponseError(
                    "OpenRouter attributed a failed completion to the pinned upstream provider."
                ) from error
            choices = getattr(response, "choices", None)
            if not isinstance(choices, list) or not choices:
                raise TransientProviderResponseError(
                    "Provider returned no completion choices for a successful chat request."
                )
            if getattr(choices[0], "message", None) is None:
                raise TransientProviderResponseError(
                    "Provider returned a completion choice without an assistant message."
                )
            if getattr(choices[0], "finish_reason", None) == "error":
                raise TransientProviderResponseError("Provider returned a completion choice with an inference error.")
            return response

        if timeout_seconds is None:
            response, retry = _retry(request)
        else:
            response, retry = _retry(request, timeout_seconds=timeout_seconds)
        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        raw_response_extra = getattr(response, "model_extra", None)
        response_extra = raw_response_extra if isinstance(raw_response_extra, dict) else {}
        raw_usage_extra = getattr(response.usage, "model_extra", None) if response.usage is not None else None
        usage_extra = raw_usage_extra if isinstance(raw_usage_extra, dict) else {}
        provider_name = response_extra.get("provider")
        reported_cost = usage_extra.get("cost")
        metadata = LLMResponseMetadata(
            response_id=_optional_string(getattr(response, "id", None)),
            returned_model=_optional_string(getattr(response, "model", None)),
            upstream_provider=_optional_string(provider_name),
            finish_reason=_optional_string(getattr(choice, "finish_reason", None)),
            created_unix=_optional_integer(getattr(response, "created", None)),
            reported_cost_usd=_optional_nonnegative_float(reported_cost),
            request_attempts=retry.request_attempts,
            transient_failure_count=retry.transient_failure_count,
            backoff_seconds=retry.backoff_seconds,
        )
        if self._routing.openrouter_provider and metadata.upstream_provider != self._routing.openrouter_provider:
            raise RuntimeError(
                "OpenRouter response violated the requested upstream-provider pin: "
                f"requested={self._routing.openrouter_provider!r}, returned={metadata.upstream_provider!r}."
            )
        raw_message = msg.model_dump(mode="json")
        if not isinstance(raw_message, dict):
            raise TypeError("Provider message serialization must produce a JSON object.")
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            usage=usage,
            metadata=metadata,
            raw=raw_message,
        )


def _responses_tools(tools: Sequence[JsonObject] | None) -> list[dict[str, JsonValue]]:
    """Translate Chat Completions function declarations to Responses tools."""

    translated: list[dict[str, JsonValue]] = []
    for index, tool in enumerate(tools or ()):
        if tool.get("type") != "function":
            raise ValueError(f"Responses tool {index} must be a function declaration.")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"Responses tool {index} is missing its function object.")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Responses tool {index} requires a non-empty function name.")
        if not isinstance(parameters, dict):
            raise ValueError(f"Responses tool {name!r} requires an object-valued parameters schema.")
        declaration: dict[str, JsonValue] = {
            "type": "function",
            "name": name,
            "parameters": parameters,
        }
        description = function.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise ValueError(f"Responses tool {name!r} description must be a string.")
            declaration["description"] = description
        strict = function.get("strict")
        if strict is not None:
            if not isinstance(strict, bool):
                raise ValueError(f"Responses tool {name!r} strict flag must be boolean.")
            declaration["strict"] = strict
        translated.append(declaration)
    return translated


def _responses_input(messages: Sequence[JsonObject]) -> list[dict[str, JsonValue]]:
    """Translate the canonical harness conversation to Responses input items."""

    translated: list[dict[str, JsonValue]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role in {"system", "user"}:
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"Responses {role} message {index} requires string content.")
            translated.append({"role": role, "content": content})
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            output = message.get("content")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"Responses tool message {index} requires a tool_call_id.")
            if not isinstance(output, str):
                raise ValueError(f"Responses tool message {index} requires string content.")
            translated.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            )
            continue
        if role != "assistant":
            raise ValueError(f"Unsupported harness message role for Responses input: {role!r}.")

        provider_state = message.get("provider_state")
        if provider_state is not None:
            if not isinstance(provider_state, list) or not provider_state:
                raise ValueError(f"Responses assistant message {index} has invalid provider_state.")
            for item in provider_state:
                if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                    raise ValueError(f"Responses assistant message {index} contains invalid provider state.")
                translated.append(item)
            continue

        content = message.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise ValueError(f"Responses assistant message {index} content must be a string or null.")
            translated.append({"role": "assistant", "content": content})
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            continue
        if not isinstance(raw_calls, list):
            raise ValueError(f"Responses assistant message {index} tool_calls must be a list.")
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                raise ValueError(f"Responses assistant message {index} contains an invalid tool call.")
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            if not isinstance(call_id, str) or not call_id or not isinstance(function, dict):
                raise ValueError(f"Responses assistant message {index} contains an invalid function call.")
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not name or not isinstance(arguments, str):
                raise ValueError(f"Responses assistant message {index} contains an invalid function payload.")
            translated.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            )
    return translated


class ResponsesProvider:
    """Direct OpenAI Responses provider with manual, auditable state replay."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        routing: ProviderRouting,
        send_temperature: bool,
        timeout_s: float,
        decoding_seed: int | None = None,
        reasoning_effort: ReasoningEffortV1 | None = None,
        exclude_reasoning: bool = True,
    ) -> None:
        _validate_decoding_seed(decoding_seed)
        if decoding_seed is not None:
            raise ValueError("The OpenAI Responses transport does not support a decoding seed.")
        if routing.provider != "openai_responses":
            raise ValueError("ResponsesProvider requires provider='openai_responses'.")
        if reasoning_effort is not None:
            raise ValueError("Explicit reasoning effort is not implemented for the OpenAI Responses transport.")
        if not exclude_reasoning:
            raise ValueError("The Responses transport requires private reasoning to remain excluded.")
        from openai import OpenAI

        self.model = model
        self._routing = routing
        self.telemetry_route = "openai_responses"
        self._send_temperature = send_temperature
        if timeout_s <= 0.0:
            raise ValueError("Provider request timeout must be positive.")
        self._request_timeout_s = float(timeout_s)
        client_kwargs: dict[str, Any] = {
            "timeout": self._request_timeout_s,
            "max_retries": 0,
        }
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = OpenAI(**client_kwargs)

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> LLMResponse:
        """Generate one Responses turn and retain every output item for replay."""

        if tool_choice == "required" and not tools:
            raise ValueError("tool_choice='required' requires at least one provider tool.")
        request_deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": _responses_input(messages),
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": int(max_tokens),
            "store": False,
        }
        translated_tools = _responses_tools(tools)
        if translated_tools:
            kwargs["tools"] = translated_tools
            if tool_choice == "required":
                kwargs["tool_choice"] = "required"
        if self._send_temperature:
            kwargs["temperature"] = float(temperature)

        def request() -> Any:
            request_kwargs = dict(kwargs)
            if request_deadline is not None:
                remaining = request_deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("Provider request deadline was exhausted before transport.")
                request_kwargs["timeout"] = min(self._request_timeout_s, remaining)
            response = self._client.responses.create(**request_kwargs)
            if getattr(response, "error", None) is not None:
                raise TransientProviderResponseError("OpenAI Responses returned an inference error.")
            status = getattr(response, "status", None)
            if status in {"failed", "cancelled"}:
                raise TransientProviderResponseError(f"OpenAI Responses returned terminal status {status!r}.")
            output = getattr(response, "output", None)
            if not isinstance(output, list) or not output:
                raise TransientProviderResponseError("OpenAI Responses returned no output items.")
            return response

        if timeout_seconds is None:
            response, retry = _retry(request)
        else:
            response, retry = _retry(request, timeout_seconds=timeout_seconds)

        provider_state: list[dict[str, JsonValue]] = []
        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        for item in response.output:
            raw_item = item.model_dump(mode="json")
            if not isinstance(raw_item, dict):
                raise TypeError("Responses output-item serialization must produce a JSON object.")
            provider_state.append(raw_item)
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                call_id = getattr(item, "call_id", None)
                name = getattr(item, "name", None)
                arguments = getattr(item, "arguments", None)
                if not all(isinstance(value, str) and value for value in (call_id, name)):
                    raise TypeError("Responses function calls require non-empty call_id and name values.")
                if not isinstance(arguments, str):
                    raise TypeError("Responses function-call arguments must be a JSON string.")
                tool_calls.append(
                    ToolCall(
                        id=cast(str, call_id),
                        name=cast(str, name),
                        arguments=arguments,
                    )
                )
            elif item_type == "message":
                for part in getattr(item, "content", ()):
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        text = getattr(part, "text", None)
                        if not isinstance(text, str):
                            raise TypeError("Responses output text must be a string.")
                        text_parts.append(text)
                    elif part_type == "refusal":
                        refusal = getattr(part, "refusal", None)
                        if not isinstance(refusal, str):
                            raise TypeError("Responses refusal text must be a string.")
                        text_parts.append(refusal)
                    else:
                        raise TypeError(f"Unsupported Responses message content type: {part_type!r}.")
            elif item_type == "reasoning":
                if raw_item.get("encrypted_content") is None:
                    raise RuntimeError(
                        "Responses reasoning output omitted encrypted continuation state while store=false."
                    )
            else:
                raise TypeError(f"Unsupported Responses output item type: {item_type!r}.")

        content = "\n".join(part for part in text_parts if part) or None
        usage = None
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        created_at = getattr(response, "created_at", None)
        created_unix = None
        if created_at is not None:
            if not isinstance(created_at, (int, float)) or not isfinite(float(created_at)) or created_at < 0:
                raise TypeError("Responses created_at must be a non-negative numeric timestamp.")
            created_unix = int(created_at)
        status = _optional_string(getattr(response, "status", None))
        incomplete = getattr(response, "incomplete_details", None)
        if incomplete is not None:
            reason = getattr(incomplete, "reason", None)
            if isinstance(reason, str) and reason:
                status = f"{status}:{reason}" if status is not None else reason
        metadata = LLMResponseMetadata(
            response_id=_optional_string(getattr(response, "id", None)),
            returned_model=_optional_string(getattr(response, "model", None)),
            finish_reason=status,
            created_unix=created_unix,
            request_attempts=retry.request_attempts,
            transient_failure_count=retry.transient_failure_count,
            backoff_seconds=retry.backoff_seconds,
        )
        raw_message: dict[str, JsonValue] = {
            "role": "assistant",
            "content": content,
            "provider_state": cast(JsonValue, provider_state),
        }
        if tool_calls:
            raw_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments or "{}"},
                }
                for call in tool_calls
            ]
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            metadata=metadata,
            raw=raw_message,
        )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"Provider response metadata must be a non-empty string, observed {value!r}.")
    return value


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"Provider response timestamp must be a non-negative integer, observed {value!r}.")
    return value


def _optional_nonnegative_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Provider-reported cost must be numeric, observed {value!r}.")
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"Provider-reported cost must be non-negative, observed {numeric!r}.")
    return numeric


def get_provider(
    model: str,
    *,
    routing: ProviderRouting,
    send_temperature: bool,
    timeout_s: float,
    decoding_seed: int | None = None,
    reasoning_effort: ReasoningEffortV1 | None = None,
    exclude_reasoning: bool = True,
) -> LLMProvider:
    """Return the explicitly selected OpenAI-compatible provider."""
    _validate_decoding_seed(decoding_seed)
    validate_provider_configuration(model=model, routing=routing)
    if routing.provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        return ChatCompletionsProvider(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            routing=routing,
            send_temperature=send_temperature,
            decoding_seed=decoding_seed,
            reasoning_effort=reasoning_effort,
            exclude_reasoning=exclude_reasoning,
            timeout_s=timeout_s,
        )
    if routing.provider == "openai_responses":
        return ResponsesProvider(
            model=model,
            routing=routing,
            send_temperature=send_temperature,
            decoding_seed=decoding_seed,
            reasoning_effort=reasoning_effort,
            exclude_reasoning=exclude_reasoning,
            timeout_s=timeout_s,
        )
    # OpenAI direct
    return ChatCompletionsProvider(
        model=model,
        routing=routing,
        send_temperature=send_temperature,
        decoding_seed=decoding_seed,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=exclude_reasoning,
        timeout_s=timeout_s,
    )
