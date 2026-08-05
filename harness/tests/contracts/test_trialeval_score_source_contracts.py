"""Tests for canonical TrialEval grade-row ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trialagentbench_harness.analysis.trialeval_score_rows import (
    TrialEvalScoreSourceError,
    iter_trialeval_score_rows,
)
from trialagentbench_harness.contracts.core.runs import TrialEvalRunConfigV1
from trialagentbench_harness.grading.models import GradeRecordV1


def _run_config(*, task_ids: list[str]) -> TrialEvalRunConfigV1:
    return TrialEvalRunConfigV1.create(
        timestamp_utc="2026-07-15T00:00:00Z",
        model="test-model",
        item_watchdog_seconds=3600,
        participant_dir="participant",
        participant_release_sha256="d" * 64,
        prompt_set_sha256="a" * 64,
        scorer_source_sha256="s" * 64,
        agent_source_sha256="g" * 64,
        experiment_condition={
            "condition_id": "primary",
            "request_replicate_id": "request-1",
            "procedure_assistance": "output_contract_only",
            "maximum_turns_per_step": 25,
            "tool_choice": "auto",
        },
        task_evidence_factors={
            task_id: {
                "context_configuration": "C1",
                "data_preparation": "analysis_ready",
                "analysis_specification": "locked_sap",
            }
            for task_id in task_ids
        },
        task_ids=task_ids,
        n_items=len(task_ids),
        decoding={"temperature": 0.0, "max_tokens": 512, "send_temperature": True},
        routing={"provider": "openai", "request_timeout_seconds": 300.0},
        executor={
            "image_reference": "trialagentbench/executor:test",
            "image_id": f"sha256:{'e' * 64}",
            "python_version": "3.12.10",
            "packages": [
                {"name": "lifelines", "version": "0.30.0"},
                {"name": "numpy", "version": "2.4.4"},
                {"name": "pandas", "version": "2.3.3"},
                {"name": "pyarrow", "version": "23.0.1"},
                {"name": "scipy", "version": "1.17.1"},
                {"name": "statsmodels", "version": "0.14.6"},
            ],
            "limits": {},
        },
    )


def _write_missing_grade(path: Path, *, task_ids: list[str] | None = None) -> None:
    task_ids = task_ids or ["TASK1"]
    grade = GradeRecordV1(
        release_id="release.v1",
        item_id="TASK1",
        usable_primary=False,
        route_match=False,
        obligations_met=False,
        result_match=False,
        passed=False,
        gates=(
            {"gate_id": "submission", "status": "failed", "failure_code": "missing_primary_submission"},
            {"gate_id": "question", "status": "not_reached"},
            {"gate_id": "route", "status": "not_reached"},
            {"gate_id": "evidence", "status": "not_reached"},
            {"gate_id": "integrity", "status": "not_reached"},
            {"gate_id": "result", "status": "not_reached"},
            {"gate_id": "conformance", "status": "not_reached"},
            {"gate_id": "decision", "status": "not_reached"},
        ),
        components=(
            {
                "component_id": "submission",
                "status": "failed",
                "failure_code": "missing_primary_submission",
            },
            {"component_id": "question", "status": "not_evaluable"},
            {"component_id": "method", "status": "not_evaluable"},
            {"component_id": "evidence", "status": "not_evaluable"},
            {"component_id": "integrity", "status": "not_evaluable"},
            {"component_id": "result_structure", "status": "not_evaluable"},
            {"component_id": "route_comparison", "status": "not_evaluable"},
        ),
        first_failure_gate="submission",
        failure_codes=("missing_primary_submission",),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "item_id": "TASK1",
                "timestamp_utc": "2026-07-15T00:00:00Z",
                "run_config": _run_config(task_ids=task_ids).model_dump(mode="json"),
                "agent_output": {
                    "status": "success",
                    "turns_used": 1,
                    "result": None,
                    "condition_provenance": {
                        "procedure_assistance": "output_contract_only",
                        "analysis_specification": "locked_sap",
                        "prompt_condition": "neutral",
                        "submission_interface": "structured",
                        "analysis_surface_sha256": "d" * 64,
                        "max_turns": 25,
                        "prompt_set_sha256": "a" * 64,
                        "rendered_system_prompt_sha256": "b" * 64,
                        "tool_schema_sha256": "c" * 64,
                        "response_contract_sha256": "d" * 64,
                    },
                },
                "scores": {
                    "item_id": "TASK1",
                    "task_id": "TASK1",
                    "trial_name": "Trial 1",
                    "design_tier": "D1",
                    "design_subtype": "individual_randomized",
                    "assumption_tier": "A1",
                    "context_tier": "C1",
                    "model": "test-model",
                    "output_mode": "structured",
                    "turns_used": 1,
                    "agent_status": "success",
                    "credit_eligible_route_count": 2,
                    "grade": grade.model_dump(mode="json"),
                    "planning": {
                        "applicable": False,
                        "submitted": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_reader_requires_fresh_canonical_grade_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_missing_grade(run_dir / "items" / "TASK1.json")

    row = next(iter_trialeval_score_rows(run_dir))

    assert row.task_id == "TASK1"
    assert row.condition_id == "primary"
    assert row.request_replicate_id == "request-1"
    assert row.reasoning_effort is None
    assert row.credit_eligible_route_count == 2
    assert row.grade.failure_codes == ("missing_primary_submission",)
    assert row.grade.usable_primary is False


def test_reader_rejects_historical_or_extra_score_fields(tmp_path: Path) -> None:
    path = tmp_path / "run" / "items" / "TASK1.json"
    _write_missing_grade(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scores"]["primary_score_normalised_max_recoverable"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        list(iter_trialeval_score_rows(path.parents[1]))


def test_reader_rejects_incomplete_scheduled_surface(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_missing_grade(
        run_dir / "items" / "TASK1.json",
        task_ids=["TASK1", "TASK2"],
    )

    with pytest.raises(TrialEvalScoreSourceError, match="missing=.*TASK2"):
        list(iter_trialeval_score_rows(run_dir))


def test_reader_rejects_grade_submission_disagreement(tmp_path: Path) -> None:
    path = tmp_path / "run" / "items" / "TASK1.json"
    _write_missing_grade(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scores"]["grade"].update(
        {
            "usable_primary": True,
            "route_match": False,
            "gates": [
                {"gate_id": "submission", "status": "passed"},
                {
                    "gate_id": "question",
                    "status": "failed",
                    "failure_code": "unrecognized_primary_question",
                },
                {"gate_id": "route", "status": "not_reached"},
                {"gate_id": "evidence", "status": "not_reached"},
                {"gate_id": "integrity", "status": "not_reached"},
                {"gate_id": "result", "status": "not_reached"},
                {"gate_id": "conformance", "status": "not_reached"},
                {"gate_id": "decision", "status": "not_reached"},
            ],
            "components": [
                {"component_id": "submission", "status": "passed"},
                {
                    "component_id": "question",
                    "status": "failed",
                    "failure_code": "unrecognized_primary_question",
                },
                {"component_id": "method", "status": "not_evaluable"},
                {"component_id": "evidence", "status": "not_evaluable"},
                {"component_id": "integrity", "status": "not_evaluable"},
                {"component_id": "result_structure", "status": "not_evaluable"},
                {"component_id": "route_comparison", "status": "not_evaluable"},
            ],
            "first_failure_gate": "question",
            "failure_codes": ["unrecognized_primary_question"],
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrialEvalScoreSourceError, match="usability disagrees"):
        list(iter_trialeval_score_rows(path.parents[1]))
