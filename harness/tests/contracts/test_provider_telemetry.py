from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.core.runs import ProviderRequestEventV1, RunCoverageV1
from trialagentbench_harness.ports import LLMResponse, LLMResponseMetadata, RetryTelemetry
from trialagentbench_harness.util.provider_telemetry import (
    fail_provider_request_v1,
    start_provider_request_v1,
    succeed_provider_request_v1,
    summarize_provider_telemetry_v1,
)


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
