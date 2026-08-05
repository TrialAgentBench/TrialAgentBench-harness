"""Exact resource outcomes for bounded portfolio runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trialagentbench_harness.contracts.trace.observable import BenchmarkRuntimeTraceEventV1
from trialagentbench_harness.contracts.trialdev.portfolio_submission import (
    TrialDevPortfolioSubmissionAttemptV1,
)
from trialagentbench_harness.io.json import write_json_model
from trialagentbench_harness.ports import LLMResponse, LLMResponseMetadata
from trialagentbench_harness.trialdev.portfolio_runtime import portfolio_resource_outcomes_v1
from trialagentbench_harness.util.provider_telemetry import (
    start_provider_request_v1,
    succeed_provider_request_v1,
)


def _runtime_event(index: int, event_type: str, *, tool_name: str | None = None) -> BenchmarkRuntimeTraceEventV1:
    return BenchmarkRuntimeTraceEventV1(
        event_id=f"event-{index}",
        timestamp=datetime(2026, 8, 3, tzinfo=UTC),
        source_artifact_path="conversation.json",
        source_payload_sha256="a" * 64,
        benchmark="trialdev",
        event_index=index,
        program_id="portfolio-1",
        scenario_id="world-1",
        objective_id="benefit_risk",
        phase_id="observational_review",
        step_id="analysis",
        event_type=event_type,
        tool_name=tool_name,
        execution_status="success" if event_type in {"code_execution", "file_inspection"} else None,
    )


def test_portfolio_resources_reproduce_from_persisted_events(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempts" / "observational_review"
    write_json_model(
        attempt_dir / "attempt_01.json",
        TrialDevPortfolioSubmissionAttemptV1(
            checkpoint_id="observational_review",
            attempt_index=1,
            transport_name="submit_portfolio_checkpoint_file",
            status="contract_rejected",
            submitted_payload={},
            validation_error="missing required evidence",
        ),
    )
    events = (
        _runtime_event(0, "assistant_message"),
        _runtime_event(1, "assistant_message"),
        _runtime_event(2, "code_execution", tool_name="execute_code"),
        _runtime_event(3, "file_inspection", tool_name="inspect_parquet"),
    )
    (tmp_path / "events.jsonl").write_text(
        "".join(f"{event.model_dump_json()}\n" for event in events),
        encoding="utf-8",
    )
    handle = start_provider_request_v1(
        path=tmp_path / "provider_responses.jsonl",
        benchmark="trialdev",
        unit_id="portfolio-1",
        phase_id="observational_review",
        step_id="analysis",
        turn_index=1,
        requested_model="model",
        provider_route="test",
    )
    succeed_provider_request_v1(
        handle,
        elapsed_seconds=1.25,
        response=LLMResponse(
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            metadata=LLMResponseMetadata(reported_cost_usd=0.012),
        ),
    )

    resources = portfolio_resource_outcomes_v1(tmp_path)

    assert resources.submission_attempts == 1
    assert resources.correction_count == 1
    assert resources.agent_turns == 2
    assert resources.execute_code_calls == 1
    assert resources.inspect_data_calls == 1
    assert resources.provider_calls == 1
    assert resources.provider_elapsed_seconds == 1.25
    assert resources.prompt_tokens == 11
    assert resources.completion_tokens == 7
    assert resources.provider_reported_usd == 0.012
