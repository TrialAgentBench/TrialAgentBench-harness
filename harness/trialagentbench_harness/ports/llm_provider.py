"""LLM provider port (hexagonal boundary).

Offline workflows must not import provider implementations. This module defines
the minimal provider protocol used by agent loops and optional eval-time LLM
features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = Mapping[str, JsonValue]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string


@dataclass(frozen=True)
class RetryTelemetry:
    """Observed transport attempts and cumulative retry delay."""

    request_attempts: int
    transient_failure_count: int
    backoff_seconds: float


@dataclass(frozen=True)
class LLMResponseMetadata:
    """Provider response identity, routing metadata, and retry observations."""

    response_id: str | None = None
    returned_model: str | None = None
    upstream_provider: str | None = None
    finish_reason: str | None = None
    created_unix: int | None = None
    reported_cost_usd: float | None = None
    request_attempts: int = 1
    transient_failure_count: int = 0
    backoff_seconds: float = 0.0


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: JsonObject | None = None
    metadata: LLMResponseMetadata = field(default_factory=LLMResponseMetadata)
    raw: object = None


class LLMProvider(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def telemetry_route(self) -> str: ...

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> LLMResponse: ...
