"""Black-box tests for the public observable trace bundle."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trialagentbench_harness.contracts.trace.observable import (
    EvidenceUseRowV1,
    FailureCascadeRowV1,
    ModelActionTraceEventV1,
    SemanticActionFeatureRowV1,
    TraceFeatureRowV1,
)
from trialagentbench_harness.tools.build import build_trace_analysis_bundle as builder
from trialagentbench_harness.tools.validate.validate_trace_analysis_bundle import (
    validate_trace_analysis_bundle,
)


def _trialeval_rows(
    *,
    run_id: str = "run-001",
    task_id: str = "task-001",
) -> tuple[
    list[ModelActionTraceEventV1],
    list[TraceFeatureRowV1],
    list[EvidenceUseRowV1],
    list[FailureCascadeRowV1],
    list[SemanticActionFeatureRowV1],
]:
    feature = TraceFeatureRowV1(
        benchmark="trialeval",
        model_id="arbitrary/model-id",
        run_id=run_id,
        task_id=task_id,
        phase_id="task",
        trace_coverage_status="full_conversation_trace",
        inspected_public_data=True,
        executed_code=True,
        submitted_structured_answer=True,
        submitted_answer=True,
        submission_interface="structured",
        submission_transport="direct",
        trace_input_authority="authoritative_structured",
        context_tier="C1",
        data_preparation="analysis_ready",
        analysis_specification="locked_sap",
        procedure_assistance="output_contract_only",
        prompt_condition="neutral",
        semantic_feature_source="structured_field",
        score_link_id=f"score:{run_id}:{task_id}",
        endpoint_valid=True,
        endpoint_state="valid",
    )
    return (
        [
            ModelActionTraceEventV1(
                event_id=f"event:{run_id}:{task_id}",
                timestamp=datetime(2026, 7, 22, tzinfo=UTC),
                benchmark="trialeval",
                model_id=feature.model_id,
                run_id=feature.run_id,
                task_id=feature.task_id,
                phase_id="task",
                event_index=0,
                event_type="file_inspection",
                source_path="items/task-001/events.jsonl",
                source_artifact_path="items/task-001/events.jsonl",
                source_payload_sha256="a" * 64,
                file_accessed="public/adtte.parquet",
                status="observed",
            )
        ],
        [feature],
        [
            EvidenceUseRowV1(
                benchmark="trialeval",
                model_id=feature.model_id,
                run_id=feature.run_id,
                task_id=feature.task_id,
                phase_id="task",
                evidence_category="time_to_event",
                source="tool_call",
                artifact_path="public/adtte.parquet",
                participant_facing=True,
            )
        ],
        [
            FailureCascadeRowV1(
                benchmark="trialeval",
                model_id=feature.model_id,
                run_id=feature.run_id,
                task_id=feature.task_id,
                first_failure_phase="task",
                first_failure_type="none_observed",
                downstream_endpoint_failed=False,
                score_link_id=f"score:{run_id}:{task_id}",
            )
        ],
        [
            SemanticActionFeatureRowV1(
                benchmark="trialeval",
                model_id=feature.model_id,
                run_id=feature.run_id,
                task_id=feature.task_id,
                phase_id="task",
                feature_name="uncertainty_interval_reported",
                feature_present=True,
                evidence_strength="structured_submission_field",
                score_link_id=f"score:{run_id}:{task_id}",
            )
        ],
    )


def test_observable_trace_bundle_is_deterministic_and_model_agnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public bundle is deterministic and accepts arbitrary model IDs."""

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(builder, "discover_trialeval_run_dirs", lambda _root: [runs / "run-001"])
    monkeypatch.setattr(builder, "collect_trialeval_action_trace", lambda _runs: _trialeval_rows())

    first = builder.build_trace_analysis_bundle(out_dir=tmp_path / "first", trialeval_root=runs)
    second = builder.build_trace_analysis_bundle(out_dir=tmp_path / "second", trialeval_root=runs)

    manifest = validate_trace_analysis_bundle(first)
    assert manifest.model_ids == ("arbitrary/model-id",)
    assert manifest.benchmark_suites == ("trialeval",)
    assert {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()} == {
        path.name: path.read_bytes() for path in second.iterdir() if path.is_file()
    }


def test_observable_trace_bundle_fails_on_unknown_or_mutated_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown members and checksum drift fail rather than being ignored."""

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(builder, "discover_trialeval_run_dirs", lambda _root: [runs / "run-001"])
    monkeypatch.setattr(builder, "collect_trialeval_action_trace", lambda _runs: _trialeval_rows())
    bundle = builder.build_trace_analysis_bundle(out_dir=tmp_path / "bundle", trialeval_root=runs)

    (bundle / "author_case_selection.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="member mismatch"):
        validate_trace_analysis_bundle(bundle)
    (bundle / "author_case_selection.json").unlink()
    with (bundle / "unit_features.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_trace_analysis_bundle(bundle)


def test_observable_trace_bundle_requires_completed_runs(tmp_path: Path) -> None:
    """An empty requested root is an invalid analysis input."""

    runs = tmp_path / "runs"
    runs.mkdir()
    with pytest.raises(ValueError, match="contains no completed runs"):
        builder.build_trace_analysis_bundle(out_dir=tmp_path / "bundle", trialeval_root=runs)


def test_observable_trace_bundle_collects_one_run_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace collection is bounded by one persisted run rather than the full panel."""

    runs = tmp_path / "runs"
    runs.mkdir()
    run_dirs = [runs / f"run-{index:03d}" for index in range(3)]
    observed_batches: list[tuple[Path, ...]] = []

    def _collect(
        batch: list[Path],
        *,
        expanded_report: Path | None = None,
    ) -> tuple[
        list[ModelActionTraceEventV1],
        list[TraceFeatureRowV1],
        list[EvidenceUseRowV1],
        list[FailureCascadeRowV1],
        list[SemanticActionFeatureRowV1],
    ]:
        assert expanded_report is None
        observed_batches.append(tuple(batch))
        run_id = batch[0].name
        return _trialeval_rows(run_id=run_id, task_id=f"task:{run_id}")

    monkeypatch.setattr(builder, "discover_trialeval_run_dirs", lambda _root: run_dirs)
    monkeypatch.setattr(builder, "collect_trialeval_action_trace", _collect)

    bundle = builder.build_trace_analysis_bundle(out_dir=tmp_path / "bundle", trialeval_root=runs)
    manifest = validate_trace_analysis_bundle(bundle)

    assert observed_batches == [(run_dir,) for run_dir in run_dirs]
    assert manifest.run_ids == tuple(run_dir.name for run_dir in run_dirs)
    assert next(table for table in manifest.tables if table.path == "unit_features.csv").row_count == 3
