from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from trialagentbench.contracts.core.runs import ProviderRequestEventV1, RunCoverageV1
from trialagentbench.ports import LLMResponse, LLMResponseMetadata, RetryTelemetry
from trialagentbench.ports.llm_provider import JsonObject
from trialagentbench.util.provider_telemetry import (
    fail_provider_request_v1,
    generate_provider_turn_v1,
    start_provider_request_v1,
    succeed_provider_request_v1,
    summarize_provider_telemetry_v1,
)


class _Provider:
    model = "test/model"
    telemetry_route = "test"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

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
        if self.error is not None:
            raise self.error
        return _response()


def _response() -> LLMResponse:
    return LLMResponse(
        content="ok",
        usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        metadata=LLMResponseMetadata(
            response_id="response-1",
            returned_model="deepseek/deepseek-v4-flash",
            upstream_provider="DeepSeek",
            finish_reason="stop",
            created_unix=123,
            reported_cost_usd=0.001,
            request_attempts=3,
            transient_failure_count=2,
            backoff_seconds=6.0,
        ),
    )


def _start(tmp_path: Path):
    return start_provider_request_v1(
        path=tmp_path / "logs" / "TASK1_provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="TASK1",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
        requested_model="deepseek/deepseek-v4-flash",
        provider_route="openrouter:DeepSeek",
    )


def _coverage() -> RunCoverageV1:
    return RunCoverageV1(
        run_identity_sha256="r" * 64,
        schedule_sha256="s" * 64,
        unit_ids=("TASK1",),
        completed_unit_ids=("TASK1",),
    )


def test_resumed_turn_retains_cancelled_and_successful_requests(tmp_path: Path) -> None:
    cancelled = _start(tmp_path)
    fail_provider_request_v1(cancelled, elapsed_seconds=0.1, failure_type="cancelled")
    resumed = _start(tmp_path)
    succeed_provider_request_v1(resumed, elapsed_seconds=0.5, response=_response())

    assert cancelled.request_id != resumed.request_id
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())
    assert summary.failed_request_count == 1
    assert summary.response_count == 1


def test_shared_provider_turn_records_one_complete_lifecycle(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "TASK1_provider_responses.jsonl"

    result = generate_provider_turn_v1(
        provider=_Provider(),
        messages=({"role": "user", "content": "analyse"},),
        tools=None,
        temperature=0.0,
        max_tokens=100,
        timeout_seconds=30.0,
        tool_choice="auto",
        log_path=log_path,
        benchmark="trialeval",
        unit_id="TASK1",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
    )

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert result.response.content == "ok"
    assert result.elapsed_seconds >= 0.0
    assert [event["status"] for event in events] == ["started", "succeeded"]
    assert events[0]["context_management"] == "local_complete"
    assert events[0]["source_context_messages"] == 1
    assert events[0]["transmitted_context_messages"] == 1
    assert events[0]["requested_completion_tokens"] == 100
    assert events[1]["completion_limit_reached"] is False
    assert events[0]["provider_capability"] is None


def test_raw_provider_payload_is_retained_without_changing_scored_telemetry(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    response = _response()
    response.raw_response = {
        "id": "response-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {"reasoning_details": [{"type": "reasoning.text", "text": "Returned text."}]},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 2}},
    }
    succeed_provider_request_v1(handle, elapsed_seconds=1.5, response=response)
    payload_path = tmp_path / "logs" / "TASK1_provider_payloads.raw.jsonl"
    payload = json.loads(payload_path.read_text())
    assert payload["request_id"] == handle.request_id
    assert payload["elapsed_seconds"] == 1.5
    assert payload["response"] == response.raw_response
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())
    assert summary.response_count == 1
    assert summary.total_tokens == 13


def test_reasoning_only_failure_retains_returned_payload_for_diagnosis(tmp_path: Path) -> None:
    from trialagentbench.adapters.llm_providers import TransientProviderResponseError

    handle = _start(tmp_path)
    payload = {"choices": [{"finish_reason": "length", "message": {"content": None, "reasoning": "Returned text."}}]}
    error = TransientProviderResponseError("No public answer", raw_response=payload)
    fail_provider_request_v1(handle, elapsed_seconds=2, failure_type="provider_error", error=error)
    saved = json.loads((tmp_path / "logs" / "TASK1_provider_payloads.raw.jsonl").read_text())
    assert saved["response"] == payload
    events = [json.loads(line) for line in handle.path.read_text().splitlines()]
    assert events[-1]["status"] == "failed"


@pytest.mark.parametrize("requested", (100, 101))
def test_source_bound_provider_capacity_is_recorded_and_completion_enforced(tmp_path: Path, requested: int) -> None:
    """Character accounting cannot substitute for a declared token capacity."""
    from datetime import UTC, datetime

    from trialagentbench.contracts.core.config import ProviderModelCapabilityV1

    capability = ProviderModelCapabilityV1(
        provider_transport="openai",
        model_id="test/model",
        context_window_tokens=4096,
        maximum_completion_tokens=100,
        source_url="https://example.test/model-capacity",
        source_retrieved_utc=datetime(2026, 9, 7, tzinfo=UTC),
        source_payload_sha256="a" * 64,
    )
    log = tmp_path / "provider.jsonl"

    def call() -> None:
        generate_provider_turn_v1(
            provider=_Provider(),
            messages=({"role": "user", "content": "analyse"},),
            tools=None,
            temperature=0,
            max_tokens=requested,
            timeout_seconds=30,
            tool_choice="auto",
            log_path=log,
            benchmark="trialeval",
            unit_id="task",
            phase_id="analysis",
            step_id="1",
            turn_index=1,
            provider_capability=capability,
        )

    if requested == 100:
        call()
    else:
        with pytest.raises(ValueError, match="exceeds the source-bound"):
            call()
    events = [ProviderRequestEventV1.model_validate_json(line) for line in log.read_text().splitlines()]
    assert [e.status for e in events] == ["started", "succeeded" if requested == 100 else "failed"]
    assert all(e.provider_capability == capability for e in events)
    assert events[0].transmitted_context_characters != capability.context_window_tokens


def test_shared_provider_turn_records_timeout_and_propagates(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "TASK1_provider_responses.jsonl"

    with pytest.raises(TimeoutError, match="deadline"):
        generate_provider_turn_v1(
            provider=_Provider(TimeoutError("deadline")),
            messages=({"role": "user", "content": "analyse"},),
            tools=None,
            temperature=0.0,
            max_tokens=100,
            timeout_seconds=30.0,
            tool_choice="auto",
            log_path=log_path,
            benchmark="trialdev",
            unit_id="TASK1",
            phase_id="portfolio_management",
            step_id="decision-001",
            turn_index=1,
        )

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in events] == ["started", "failed"]
    assert events[-1]["failure_type"] == "timeout"


def test_provider_request_is_persisted_before_terminal_success_and_aggregates(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    path = handle.path
    started = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in started] == ["started"]

    succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=_response())
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in events] == ["started", "succeeded"]
    assert events[0]["request_id"] == events[1]["request_id"]
    assert events[1]["usage_status"] == "reported"
    assert summary.response_count == 1
    assert summary.total_tokens == 13
    assert summary.reported_cost_usd == pytest.approx(0.001)
    assert summary.upstream_providers == ["DeepSeek"]
    assert summary.source_files == ["logs/TASK1_provider_responses.jsonl"]
    assert summary.request_attempt_count == 3
    assert summary.transient_failure_count == 2
    assert summary.backoff_seconds == 6.0


def test_provider_telemetry_separates_compaction_usage(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    response = _response()
    response.usage = {
        "prompt_tokens": 20,
        "completion_tokens": 7,
        "total_tokens": 27,
        "compaction_requests": 1,
        "compaction_input_tokens": 10,
        "compaction_output_tokens": 4,
    }

    event = succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=response)
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    assert event.compaction_requests == 1
    assert event.compaction_input_tokens == 10
    assert event.compaction_output_tokens == 4
    assert summary.compaction_requests == 1
    assert summary.compaction_input_tokens == 10
    assert summary.compaction_output_tokens == 4


def test_provider_telemetry_records_archived_context_and_completion_limit(tmp_path: Path) -> None:
    response = _response()
    response.metadata = LLMResponseMetadata(
        response_id="response-1",
        returned_model="deepseek/deepseek-v4-flash",
        upstream_provider="DeepSeek",
        finish_reason="incomplete:max_output_tokens",
    )
    handle = start_provider_request_v1(
        path=tmp_path / "logs" / "TASK1_provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="TASK1",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
        requested_model="model",
        provider_route="route",
        context_management="local_archived",
        source_context_messages=8,
        transmitted_context_messages=5,
        source_context_characters=10_000,
        transmitted_context_characters=4_000,
        requested_completion_tokens=512,
    )

    terminal = succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=response)
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    assert terminal.completion_limit_reached is True
    assert terminal.context_management == "local_archived"
    assert summary.locally_archived_request_count == 1
    assert summary.completion_limit_reached_count == 1
    assert summary.source_context_characters == 10_000
    assert summary.transmitted_context_characters == 4_000
    assert summary.requested_completion_tokens == 512


def test_provider_summary_includes_decision_boundary_resume_logs(tmp_path: Path) -> None:
    handle = start_provider_request_v1(
        path=tmp_path / "tasks" / "TASK1" / "provider_responses.resume-002.jsonl",
        benchmark="trialdev",
        unit_id="TASK1",
        phase_id="portfolio_management",
        step_id="decision-002",
        turn_index=1,
        requested_model="analysis-model",
        provider_route="openai",
    )
    succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=_response())

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    assert summary.response_count == 1
    assert summary.reported_cost_usd == pytest.approx(0.001)
    assert summary.source_files == ["tasks/TASK1/provider_responses.resume-002.jsonl"]


def test_provider_request_finalizes_typed_failure_without_usage(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    error = TimeoutError("provider deadline")
    error.status_code = 504  # type: ignore[attr-defined]
    fail_provider_request_v1(handle, elapsed_seconds=0.25, failure_type="timeout", error=error)

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())
    terminal = ProviderRequestEventV1.model_validate_json(handle.path.read_text(encoding="utf-8").splitlines()[-1])

    assert terminal.status == "failed"
    assert terminal.failure_type == "timeout"
    assert terminal.exception_type == "TimeoutError"
    assert terminal.http_status_code == 504
    assert terminal.usage_status == "not_applicable"
    assert summary.response_count == 0
    assert summary.failed_request_count == 1
    assert summary.failure_type_counts == {"timeout": 1, "provider_error": 0, "cancelled": 0}
    assert summary.request_attempt_count == 1


def test_exhausted_retry_failure_retains_actual_retry_observations(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    error = TimeoutError("retry deadline")
    error.retry_telemetry = RetryTelemetry(  # type: ignore[attr-defined]
        request_attempts=4,
        transient_failure_count=4,
        backoff_seconds=7.5,
    )
    fail_provider_request_v1(
        handle,
        elapsed_seconds=8.0,
        failure_type="timeout",
        error=error,
    )

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    assert summary.request_attempt_count == 4
    assert summary.transient_failure_count == 4
    assert summary.backoff_seconds == 7.5


def test_provider_summary_excludes_archived_failed_attempt_logs(tmp_path: Path) -> None:
    authoritative = _start(tmp_path)
    succeed_provider_request_v1(authoritative, elapsed_seconds=0.5, response=_response())
    archived = start_provider_request_v1(
        path=tmp_path / "archived_failed_attempts" / "TASK1_provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="TASK1",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
        requested_model="deepseek/deepseek-v4-flash",
        provider_route="openrouter:DeepSeek",
    )
    fail_provider_request_v1(archived, elapsed_seconds=99.0, failure_type="provider_error")

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    assert summary.response_count == 1
    assert summary.elapsed_seconds == 0.5
    assert summary.source_files == ["logs/TASK1_provider_responses.jsonl"]
    assert summary.archived_request_count == 1
    assert summary.archived_failed_request_count == 1
    assert summary.archived_failure_type_counts == {
        "timeout": 0,
        "provider_error": 1,
        "cancelled": 0,
    }
    assert summary.archived_elapsed_seconds == 99.0
    assert summary.archived_source_files == ["archived_failed_attempts/TASK1_provider_responses.jsonl"]


def test_provider_telemetry_rejects_duplicate_terminal_id_across_files(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    terminal = fail_provider_request_v1(handle, elapsed_seconds=0.1, failure_type="provider_error")
    duplicate_path = tmp_path / "other" / "OTHER_provider_responses.jsonl"
    duplicate_path.parent.mkdir(parents=True)
    duplicate_path.write_text(terminal.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate provider request failed event"):
        summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())


def test_provider_telemetry_rejects_missing_terminal_join(tmp_path: Path) -> None:
    _start(tmp_path)

    with pytest.raises(ValueError, match="missing joins"):
        summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())


def test_provider_telemetry_preserves_cost_for_incomplete_run_coverage(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=_response())
    incomplete = _coverage().model_copy(update={"completed_unit_ids": ()})

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=incomplete)

    assert summary.unit_ids == ("TASK1",)
    assert summary.completed_unit_ids == ()
    assert summary.response_count == 1
    assert summary.reported_cost_usd == 0.001


def test_provider_telemetry_rejects_identity_or_route_join_mismatch(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    terminal = fail_provider_request_v1(handle, elapsed_seconds=0.1, failure_type="provider_error")
    lines = handle.path.read_text(encoding="utf-8").splitlines()
    payload = terminal.model_dump(mode="json")
    payload["provider_route"] = "openai"
    handle.path.write_text(lines[0] + "\n" + json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identity or route mismatch"):
        summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())


@pytest.mark.parametrize("field", ["status", "provider_route", "usage_status", "total_tokens"])
def test_provider_event_rejects_missing_status_route_or_usage(field: str) -> None:
    payload = {
        "request_id": "request-1",
        "status": "succeeded",
        "benchmark": "trialeval",
        "unit_id": "TASK1",
        "phase_id": "task",
        "step_id": "analysis",
        "turn_index": 1,
        "elapsed_seconds": 0.1,
        "requested_model": "model",
        "provider_route": "openai",
        "usage_status": "reported",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "request_attempts": 1,
        "transient_failure_count": 0,
        "backoff_seconds": 0.0,
    }
    del payload[field]

    with pytest.raises(ValidationError):
        ProviderRequestEventV1.model_validate(payload)


def test_provider_telemetry_rejects_malformed_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "provider_responses.jsonl"
    path.write_text('{"schema_id":"wrong"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid provider telemetry"):
        summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())


def test_provider_telemetry_rejects_inconsistent_retry_counts() -> None:
    with pytest.raises(ValidationError, match="transient_failure_count"):
        ProviderRequestEventV1(
            request_id="request-1",
            status="succeeded",
            benchmark="trialeval",
            unit_id="TASK1",
            phase_id="task",
            step_id="analysis",
            turn_index=1,
            elapsed_seconds=0.1,
            requested_model="model",
            provider_route="openai",
            usage_status="not_reported",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            request_attempts=3,
            transient_failure_count=1,
            backoff_seconds=0.0,
        )


def test_provider_failure_records_the_chained_message_and_response_body(tmp_path: Path) -> None:
    from trialagentbench.ports import ProviderRequestError

    upstream = RuntimeError("Error code: 400")
    upstream.body = {"error": {"message": "Invalid schema for function 'start_study'"}}  # type: ignore[attr-defined]
    error = ProviderRequestError(
        "Provider request failed with HTTP status 400.",
        retry_telemetry=RetryTelemetry(request_attempts=1, transient_failure_count=0, backoff_seconds=0.0),
    )
    error.__cause__ = upstream
    handle = _start(tmp_path)
    fail_provider_request_v1(handle, elapsed_seconds=0.1, failure_type="provider_error", error=error)

    terminal = ProviderRequestEventV1.model_validate_json(handle.path.read_text(encoding="utf-8").splitlines()[-1])
    assert terminal.error_detail is not None
    assert "HTTP status 400" in terminal.error_detail and "Invalid schema for function" in terminal.error_detail
    long = RuntimeError("x" * 5000)
    fail_provider_request_v1(_start(tmp_path), elapsed_seconds=0.1, failure_type="provider_error", error=long)
    last = ProviderRequestEventV1.model_validate_json(handle.path.read_text(encoding="utf-8").splitlines()[-1])
    assert last.error_detail is not None and len(last.error_detail) == 2000


def test_provider_success_records_cached_and_reasoning_tokens_and_returned_reasoning(tmp_path: Path) -> None:
    response = _response()
    response.usage = {**response.usage, "cached_tokens": 8, "reasoning_tokens": 2}
    response.reasoning = "Compare the arms first."
    handle = _start(tmp_path)
    event = succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=response)

    assert (event.cached_tokens, event.reasoning_tokens, event.reasoning) == (8, 2, "Compare the arms first.")
    with pytest.raises(ValidationError, match="cannot exceed"):
        ProviderRequestEventV1.model_validate(event.model_dump() | {"cached_tokens": 11})
