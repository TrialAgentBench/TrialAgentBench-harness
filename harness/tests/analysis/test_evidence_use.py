"""Observable evidence-use linkage tests."""

from __future__ import annotations

from datetime import UTC, datetime

from trialagentbench_harness.analysis.evidence_use import evidence_rows_from_events
from trialagentbench_harness.contracts.trace.observable import ModelActionTraceEventV1


def _event(
    *,
    event_type: str,
    tool_call_id: str,
    execution_status: str | None = None,
    file_accessed: str | None = None,
    status: str = "observed",
) -> ModelActionTraceEventV1:
    return ModelActionTraceEventV1.model_validate(
        {
            "event_id": f"{event_type}:{tool_call_id}",
            "timestamp": datetime(2026, 7, 22, tzinfo=UTC),
            "benchmark": "trialeval",
            "model_id": "model",
            "run_id": "run",
            "task_id": "TASK1",
            "assignment_id": "assignment",
            "phase_id": "task",
            "step_id": "analysis",
            "event_index": 0,
            "event_type": event_type,
            "source_path": "events.jsonl",
            "source_artifact_path": "conversation.json",
            "source_payload_sha256": "a" * 64,
            "tool_call_id": tool_call_id,
            "tool_name": "inspect_parquet",
            "file_accessed": file_accessed,
            "execution_status": execution_status,
            "status": status,
        }
    )


def test_evidence_use_requires_successful_file_inspection() -> None:
    events = [
        _event(event_type="file_inspection", tool_call_id="successful", file_accessed="data/ADSL.parquet"),
        _event(
            event_type="file_inspection",
            tool_call_id="failed",
            file_accessed="scratch/missing.json",
            status="invalid",
        ),
    ]

    rows = evidence_rows_from_events(events)

    assert len(rows) == 1
    assert rows[0].artifact_path == "data/ADSL.parquet"
    assert rows[0].assignment_id == "assignment"


def test_evidence_use_does_not_treat_workspace_writes_as_inspection() -> None:
    events = [
        _event(event_type="tool_call", tool_call_id="write", file_accessed="scratch/result.json"),
        _event(event_type="tool_result", tool_call_id="write", execution_status="success"),
    ]

    assert evidence_rows_from_events(events) == []
