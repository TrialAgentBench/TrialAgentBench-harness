"""Focused runtime-event provenance and streaming-ingestion tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.trace.observable import (
    BenchmarkNameV1,
    BenchmarkRuntimeTraceEventV1,
    ModelActionTraceEventV1,
    runtime_event_source_payload_v1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256
from trialagentbench_harness.trialdev.action_trace import (
    _read_runtime_events as read_trialdev_runtime_events,
)
from trialagentbench_harness.trialdev.action_trace import (
    _require_submission_route_join as require_trialdev_submission_route_join,
)
from trialagentbench_harness.trialdev.action_trace import (
    _validate_runtime_event_link as validate_trialdev_runtime_event_link,
)
from trialagentbench_harness.trialeval.action_trace import (
    _runtime_events as read_trialeval_runtime_events,
)
from trialagentbench_harness.trialeval.action_trace import (
    _validate_runtime_events as validate_trialeval_runtime_events,
)

RuntimeEventReader = Callable[[Path], tuple[BenchmarkRuntimeTraceEventV1, ...]]


def _trialdev_submission_event(*, step_id: str, event_index: int) -> ModelActionTraceEventV1:
    return ModelActionTraceEventV1(
        event_id=f"submission:{step_id}:{event_index}",
        timestamp="2026-07-22T12:00:00Z",
        benchmark="trialdev",
        model_id="model",
        run_id="run",
        program_id="program",
        scenario_id="scenario",
        objective_id="benefit_risk",
        phase_id="phase2",
        event_index=event_index,
        event_type="submission",
        source_path="events.jsonl",
        source_artifact_path="conversation.json",
        source_payload_sha256="a" * 64,
        status="observed",
        step_id=step_id,
    )


def _event(
    *,
    benchmark: BenchmarkNameV1,
    event_index: int,
    event_type: str,
    conversation_message: JsonValue | None = None,
) -> dict[str, object]:
    unit = "TASK1" if benchmark == "trialeval" else "program-1"
    event: dict[str, object] = {
        "event_id": f"{benchmark}:{unit}:{event_index:06d}",
        "timestamp": "2026-07-19T12:00:00Z",
        "source_artifact_path": f"/runs/{unit}/conversation.json",
        "benchmark": benchmark,
        "event_index": event_index,
        "phase_id": "task" if benchmark == "trialeval" else "phase1",
        "step_id": "analysis",
        "event_type": event_type,
    }
    if benchmark == "trialeval":
        event["task_id"] = unit
    else:
        event.update(program_id=unit, scenario_id="s01", objective_id="benefit_risk")
    if event_type == "step_terminal":
        event["terminal_status"] = "completed"
    if conversation_message is not None:
        event["conversation_message_index"] = 0
    source_payload = runtime_event_source_payload_v1(
        benchmark=benchmark,
        task_id=unit if benchmark == "trialeval" else None,
        program_id=unit if benchmark == "trialdev" else None,
        scenario_id="s01" if benchmark == "trialdev" else None,
        objective_id="benefit_risk" if benchmark == "trialdev" else None,
        phase_id=str(event["phase_id"]),
        step_id="analysis",
        event_type=event_type,
        terminal_status=str(event["terminal_status"]) if event.get("terminal_status") is not None else None,
        failure_type=None,
        conversation_message=conversation_message,
    )
    event["source_payload_sha256"] = canonical_payload_sha256(source_payload)
    return event


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_runtime_event_contract_requires_native_provenance() -> None:
    payload = _event(benchmark="trialeval", event_index=0, event_type="step_started")
    del payload["event_id"]

    with pytest.raises(ValidationError, match="event_id"):
        BenchmarkRuntimeTraceEventV1.model_validate(payload)


def test_runtime_event_contract_rejects_non_utc_timestamp() -> None:
    payload = _event(benchmark="trialeval", event_index=0, event_type="step_started")
    payload["timestamp"] = "2026-07-19T13:00:00+01:00"

    with pytest.raises(ValidationError, match="timezone-aware UTC"):
        BenchmarkRuntimeTraceEventV1.model_validate(payload)


@pytest.mark.parametrize(
    ("benchmark", "reader"),
    [
        ("trialeval", read_trialeval_runtime_events),
        ("trialdev", read_trialdev_runtime_events),
    ],
)
def test_runtime_jsonl_ingestion_preserves_native_ids(
    tmp_path: Path,
    benchmark: BenchmarkNameV1,
    reader: RuntimeEventReader,
) -> None:
    path = tmp_path / "events.jsonl"
    rows = [
        _event(benchmark=benchmark, event_index=0, event_type="step_started"),
        _event(benchmark=benchmark, event_index=1, event_type="step_terminal"),
    ]
    _write_jsonl(path, rows)

    events = reader(path)

    assert [event.event_id for event in events] == [row["event_id"] for row in rows]
    assert [event.source_payload_sha256 for event in events] == [row["source_payload_sha256"] for row in rows]


@pytest.mark.parametrize(
    ("benchmark", "reader"),
    [
        ("trialeval", read_trialeval_runtime_events),
        ("trialdev", read_trialdev_runtime_events),
    ],
)
def test_runtime_jsonl_ingestion_rejects_duplicate_native_ids(
    tmp_path: Path,
    benchmark: BenchmarkNameV1,
    reader: RuntimeEventReader,
) -> None:
    path = tmp_path / "events.jsonl"
    rows = [
        _event(benchmark=benchmark, event_index=0, event_type="step_started"),
        _event(benchmark=benchmark, event_index=1, event_type="step_terminal"),
    ]
    rows[1]["event_id"] = rows[0]["event_id"]
    _write_jsonl(path, rows)

    with pytest.raises(ValueError, match="event IDs must be unique"):
        reader(path)


def test_trialeval_ingestion_rejects_tampered_indexed_conversation_payload(tmp_path: Path) -> None:
    conversation_path = tmp_path / "conversation.json"
    original_message: JsonValue = {"role": "user", "content": "original prompt"}
    rows = [
        _event(benchmark="trialeval", event_index=0, event_type="step_started"),
        _event(
            benchmark="trialeval",
            event_index=1,
            event_type="prompt",
            conversation_message=original_message,
        ),
        _event(benchmark="trialeval", event_index=2, event_type="step_terminal"),
    ]
    conversation_path.write_text(
        json.dumps([{"role": "user", "content": "tampered prompt"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conversation payload hash mismatch"):
        validate_trialeval_runtime_events(
            tuple(BenchmarkRuntimeTraceEventV1.model_validate(row) for row in rows),
            task_id="TASK1",
            conversation_path=conversation_path,
        )


def test_trialdev_ingestion_rejects_tampered_indexed_conversation_payload(tmp_path: Path) -> None:
    original_message: JsonValue = {"role": "user", "content": "original prompt"}
    row = _event(
        benchmark="trialdev",
        event_index=0,
        event_type="prompt",
        conversation_message=original_message,
    )
    event = BenchmarkRuntimeTraceEventV1.model_validate(row)

    with pytest.raises(ValueError, match="conversation payload hash mismatch"):
        validate_trialdev_runtime_event_link(
            event,
            messages=[{"role": "user", "content": "tampered prompt"}],
            path=tmp_path / "events.jsonl",
        )


def test_trialdev_submission_join_accepts_retries_for_complete_artifacts() -> None:
    require_trialdev_submission_route_join(
        phase_id="phase2",
        submission_paths=(
            "/run/analysis_submission.json",
            "/run/decision_submission.json",
        ),
        phase_events=[
            _trialdev_submission_event(step_id="trial_design_request", event_index=0),
            _trialdev_submission_event(step_id="trial_analysis", event_index=1),
            _trialdev_submission_event(step_id="trial_analysis", event_index=2),
            _trialdev_submission_event(step_id="phase_decision", event_index=3),
        ],
        identity="run:program:phase2",
    )


def test_trialdev_submission_join_rejects_orphaned_semantic_route() -> None:
    with pytest.raises(ValueError, match="disagree with runner-native semantic routes"):
        require_trialdev_submission_route_join(
            phase_id="phase2",
            submission_paths=("/run/analysis_submission.json",),
            phase_events=[
                _trialdev_submission_event(step_id="trial_analysis", event_index=0),
                _trialdev_submission_event(step_id="phase_decision", event_index=1),
            ],
            identity="run:program:phase2",
        )


def test_trialdev_submission_join_requires_complete_composite_artifacts() -> None:
    with pytest.raises(ValueError, match="observational submission is incomplete"):
        require_trialdev_submission_route_join(
            phase_id="observational_review",
            submission_paths=("/run/obs_review_submission.json",),
            phase_events=[],
            identity="run:program:observational_review",
        )
