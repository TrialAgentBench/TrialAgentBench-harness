"""Append-only provider request telemetry and strict aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from trialagentbench_harness.contracts.core.runs import (
    ProviderRequestEventV1,
    ProviderTelemetrySummaryV1,
    RunCoverageV1,
)
from trialagentbench_harness.io.json import append_jsonl_model, write_json_model
from trialagentbench_harness.ports import LLMResponse, RetryTelemetry

ProviderFailureType = Literal["timeout", "provider_error", "cancelled"]


@dataclass(frozen=True)
class ProviderRequestHandle:
    """Identity shared by the start and terminal events for one request."""

    path: Path
    request_id: str
    benchmark: Literal["trialeval", "trialdev"]
    unit_id: str
    phase_id: str
    step_id: str
    turn_index: int
    requested_model: str
    provider_route: str


def start_provider_request_v1(
    *,
    path: Path,
    benchmark: Literal["trialeval", "trialdev"],
    unit_id: str,
    phase_id: str,
    step_id: str,
    turn_index: int,
    requested_model: str,
    provider_route: str,
) -> ProviderRequestHandle:
    """Persist request identity and route before provider transport starts."""

    request_id = f"{Path(path).name}:{benchmark}:{unit_id}:{phase_id}:{step_id}:{turn_index}"
    handle = ProviderRequestHandle(
        path=Path(path),
        request_id=request_id,
        benchmark=benchmark,
        unit_id=unit_id,
        phase_id=phase_id,
        step_id=step_id,
        turn_index=turn_index,
        requested_model=requested_model,
        provider_route=provider_route,
    )
    append_jsonl_model(
        handle.path,
        _request_event(
            handle,
            status="started",
            elapsed_seconds=None,
            usage_status="not_applicable",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            request_attempts=0,
            transient_failure_count=0,
            backoff_seconds=0.0,
        ),
    )
    return handle


def succeed_provider_request_v1(
    handle: ProviderRequestHandle,
    *,
    elapsed_seconds: float,
    response: LLMResponse,
) -> ProviderRequestEventV1:
    """Append the successful terminal event for a started request."""

    usage = response.usage
    usage_status: Literal["reported", "not_reported"] = "reported" if usage is not None else "not_reported"
    prompt_tokens = _required_usage_integer(usage, "prompt_tokens") if usage is not None else 0
    completion_tokens = _required_usage_integer(usage, "completion_tokens") if usage is not None else 0
    total_tokens = _required_usage_integer(usage, "total_tokens") if usage is not None else 0
    event = _request_event(
        handle,
        status="succeeded",
        elapsed_seconds=elapsed_seconds,
        usage_status=usage_status,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        response=response,
        request_attempts=response.metadata.request_attempts,
        transient_failure_count=response.metadata.transient_failure_count,
        backoff_seconds=response.metadata.backoff_seconds,
    )
    append_jsonl_model(handle.path, event)
    return event


def fail_provider_request_v1(
    handle: ProviderRequestHandle,
    *,
    elapsed_seconds: float,
    failure_type: ProviderFailureType,
    error: BaseException | None = None,
) -> ProviderRequestEventV1:
    """Append a typed failed terminal event for a started request."""

    retry = retry_telemetry_from_error_v1(error)
    event = _request_event(
        handle,
        status="failed",
        elapsed_seconds=elapsed_seconds,
        usage_status="not_applicable",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        request_attempts=retry.request_attempts,
        transient_failure_count=retry.transient_failure_count,
        backoff_seconds=retry.backoff_seconds,
        failure_type=failure_type,
        error=error,
    )
    append_jsonl_model(handle.path, event)
    return event


def provider_failure_type_v1(error: BaseException | None) -> ProviderFailureType:
    """Map a propagated provider-call failure to the public failure taxonomy."""

    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "cancelled"
    return "provider_error"


def retry_telemetry_from_error_v1(error: BaseException | None) -> RetryTelemetry:
    """Return exact retry observations attached by a live provider adapter."""

    telemetry = getattr(error, "retry_telemetry", None)
    if telemetry is None:
        return RetryTelemetry(
            request_attempts=1,
            transient_failure_count=0,
            backoff_seconds=0.0,
        )
    if not isinstance(telemetry, RetryTelemetry):
        raise TypeError("Provider exception retry_telemetry must use the typed RetryTelemetry contract.")
    return telemetry


def summarize_provider_telemetry_v1(
    *,
    run_root: Path,
    coverage: RunCoverageV1,
) -> ProviderTelemetrySummaryV1:
    """Validate request joins and aggregate terminal provider events."""

    root = Path(run_root)
    all_paths = sorted(root.rglob("*provider_responses.jsonl"), key=lambda path: path.as_posix())
    paths = [path for path in all_paths if _is_authoritative_provider_log(path, root)]
    archived_paths = [path for path in all_paths if path not in paths]
    events: list[ProviderRequestEventV1] = []
    for path in paths:
        events.extend(_read_provider_events(path))

    terminal_events = _validated_terminal_events(events)
    archived_terminal_events: list[ProviderRequestEventV1] = []
    for path in archived_paths:
        archived_terminal_events.extend(_validated_terminal_events(_read_provider_events(path)))
    succeeded = [event for event in terminal_events if event.status == "succeeded"]
    failed = [event for event in terminal_events if event.status == "failed"]
    archived_failed = [event for event in archived_terminal_events if event.status == "failed"]
    summary = ProviderTelemetrySummaryV1(
        run_identity_sha256=coverage.run_identity_sha256,
        schedule_sha256=coverage.schedule_sha256,
        unit_ids=coverage.unit_ids,
        completed_unit_ids=coverage.completed_unit_ids,
        response_count=len(succeeded),
        failed_request_count=len(failed),
        failure_type_counts=_failure_type_counts(failed),
        responses_with_usage=sum(event.usage_status == "reported" for event in succeeded),
        responses_with_response_id=sum(event.response_id is not None for event in succeeded),
        responses_with_reported_cost=sum(event.reported_cost_usd is not None for event in succeeded),
        prompt_tokens=sum(event.prompt_tokens for event in succeeded),
        completion_tokens=sum(event.completion_tokens for event in succeeded),
        total_tokens=sum(event.total_tokens for event in succeeded),
        reported_cost_usd=sum(event.reported_cost_usd or 0.0 for event in succeeded),
        elapsed_seconds=sum(event.elapsed_seconds or 0.0 for event in terminal_events),
        request_attempt_count=sum(event.request_attempts for event in terminal_events),
        transient_failure_count=sum(event.transient_failure_count for event in terminal_events),
        backoff_seconds=sum(event.backoff_seconds for event in terminal_events),
        requested_models=sorted({event.requested_model for event in terminal_events}),
        returned_models=sorted({event.returned_model for event in succeeded if event.returned_model is not None}),
        upstream_providers=sorted(
            {event.upstream_provider for event in succeeded if event.upstream_provider is not None}
        ),
        source_files=[path.relative_to(root).as_posix() for path in paths],
        archived_request_count=len(archived_terminal_events),
        archived_failed_request_count=len(archived_failed),
        archived_failure_type_counts=_failure_type_counts(archived_failed),
        archived_request_attempt_count=sum(event.request_attempts for event in archived_terminal_events),
        archived_transient_failure_count=sum(event.transient_failure_count for event in archived_terminal_events),
        archived_backoff_seconds=sum(event.backoff_seconds for event in archived_terminal_events),
        archived_elapsed_seconds=sum(event.elapsed_seconds or 0.0 for event in archived_terminal_events),
        archived_source_files=[path.relative_to(root).as_posix() for path in archived_paths],
    )
    write_json_model(root / "provider_telemetry_summary.json", summary)
    return summary


def read_provider_terminal_events_v1(path: Path) -> tuple[ProviderRequestEventV1, ...]:
    """Read and validate the terminal provider events from one lifecycle log."""

    return tuple(_validated_terminal_events(_read_provider_events(Path(path))))


def _is_authoritative_provider_log(path: Path, root: Path) -> bool:
    """Exclude retained failed-attempt archives from authoritative run totals."""

    relative_parts = path.relative_to(root).parts[:-1]
    return not any(
        part.lower().startswith(("archive", "failed_attempt", "failed-attempt"))
        or part.lower() in {"attempts", "superseded"}
        for part in relative_parts
    )


def _request_event(
    handle: ProviderRequestHandle,
    *,
    status: Literal["started", "succeeded", "failed"],
    elapsed_seconds: float | None,
    usage_status: Literal["not_applicable", "reported", "not_reported"],
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    request_attempts: int,
    transient_failure_count: int,
    backoff_seconds: float,
    response: LLMResponse | None = None,
    failure_type: ProviderFailureType | None = None,
    error: BaseException | None = None,
) -> ProviderRequestEventV1:
    metadata = response.metadata if response is not None else None
    return ProviderRequestEventV1(
        request_id=handle.request_id,
        status=status,
        benchmark=handle.benchmark,
        unit_id=handle.unit_id,
        phase_id=handle.phase_id,
        step_id=handle.step_id,
        turn_index=handle.turn_index,
        elapsed_seconds=elapsed_seconds,
        requested_model=handle.requested_model,
        provider_route=handle.provider_route,
        response_id=metadata.response_id if metadata is not None else None,
        returned_model=metadata.returned_model if metadata is not None else None,
        upstream_provider=metadata.upstream_provider if metadata is not None else None,
        finish_reason=metadata.finish_reason if metadata is not None else None,
        created_unix=metadata.created_unix if metadata is not None else None,
        usage_status=usage_status,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reported_cost_usd=metadata.reported_cost_usd if metadata is not None else None,
        request_attempts=request_attempts,
        transient_failure_count=transient_failure_count,
        backoff_seconds=backoff_seconds,
        failure_type=failure_type,
        exception_type=type(error).__name__ if error is not None else None,
        http_status_code=_http_status_code(error),
    )


def _http_status_code(error: BaseException | None) -> int | None:
    """Return a bounded integer HTTP status exposed by a provider exception."""

    if error is None:
        return None
    value = getattr(error, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    return None


def _failure_type_counts(
    events: list[ProviderRequestEventV1],
) -> dict[ProviderFailureType, int]:
    """Return complete typed counts for failed terminal events."""

    counts: dict[ProviderFailureType, int] = {
        "timeout": 0,
        "provider_error": 0,
        "cancelled": 0,
    }
    for event in events:
        if event.status != "failed" or event.failure_type is None:
            raise ValueError("Failure aggregation received a non-failed provider event.")
        counts[event.failure_type] += 1
    return counts


def _read_provider_events(path: Path) -> list[ProviderRequestEventV1]:
    """Read one append-only provider lifecycle log."""

    events: list[ProviderRequestEventV1] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(ProviderRequestEventV1.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid provider telemetry at {path}:{line_number}: {exc}") from exc
    return events


def _required_usage_integer(usage: object, key: str) -> int:
    if not isinstance(usage, Mapping):
        raise TypeError("Provider usage must be a mapping.")
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Provider usage {key} must be a non-negative integer, observed {value!r}.")
    return value


def _validated_terminal_events(
    events: list[ProviderRequestEventV1],
) -> list[ProviderRequestEventV1]:
    starts: dict[str, ProviderRequestEventV1] = {}
    terminals: dict[str, ProviderRequestEventV1] = {}
    positions: dict[tuple[str, str], int] = {}
    for position, event in enumerate(events):
        target = starts if event.status == "started" else terminals
        if event.request_id in target:
            raise ValueError(f"Duplicate provider request {event.status} event: {event.request_id}.")
        target[event.request_id] = event
        positions[(event.request_id, "started" if event.status == "started" else "terminal")] = position
    if starts.keys() != terminals.keys():
        missing_terminal = sorted(starts.keys() - terminals.keys())
        missing_start = sorted(terminals.keys() - starts.keys())
        raise ValueError(
            "Provider request telemetry has missing joins: "
            f"missing_terminal={missing_terminal}, missing_start={missing_start}."
        )
    identity_fields = (
        "benchmark",
        "unit_id",
        "phase_id",
        "step_id",
        "turn_index",
        "requested_model",
        "provider_route",
    )
    for request_id, start in starts.items():
        terminal = terminals[request_id]
        if positions[(request_id, "started")] >= positions[(request_id, "terminal")]:
            raise ValueError(f"Provider request terminal precedes its start event: {request_id}.")
        if any(getattr(start, field) != getattr(terminal, field) for field in identity_fields):
            raise ValueError(f"Provider request identity or route mismatch for {request_id}.")
    return [event for event in events if event.status != "started"]


__all__ = [
    "ProviderRequestHandle",
    "fail_provider_request_v1",
    "provider_failure_type_v1",
    "read_provider_terminal_events_v1",
    "retry_telemetry_from_error_v1",
    "start_provider_request_v1",
    "succeed_provider_request_v1",
    "summarize_provider_telemetry_v1",
]
