"""Contracts and fail-fast boundaries for TrialDev checkpoint replay."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from trialagentbench_harness.analysis.experiments import trialdev_checkpoint_replay as checkpoint_analysis
from trialagentbench_harness.contracts.core.runs import (
    TrialDevChainSummaryV1,
    TrialDevMaterializationUsageV1,
)
from trialagentbench_harness.contracts.experiments import (
    TrialDevCheckpointAssignmentV1,
    TrialDevCheckpointQualityV1,
    TrialDevCheckpointScheduleV1,
    TrialDevCheckpointScoreRowV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevGradeRecordV1,
    TrialDevTrajectoryGradeV1,
)
from trialagentbench_harness.experiments.trialdev_checkpoint_replay import _safe_source_path
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevDesignEfficiencyV1,
    TrialDevelopmentAnalysisQualityV1,
)


def _assignment(condition: str, *, source: str = "endogenous") -> TrialDevCheckpointAssignmentV1:
    return TrialDevCheckpointAssignmentV1(
        assignment_id=f"assignment-{condition}",
        block_id="block-1",
        program_id="scenario-1__benefit_risk",
        scenario_id="scenario-1",
        objective_id="benefit_risk",
        replicate_id="replicate-1",
        decoding_seed=17,
        condition=condition,
        source_program_relative_path=f"{source}/programs/scenario-1__benefit_risk",
        source_checkpoint_relative_path=(f"{source}/programs/scenario-1__benefit_risk/checkpoints/00000002.json"),
        source_checkpoint_sha256=("a" if source == "endogenous" else "b") * 64,
        source_run_identity_sha256=("c" if source == "endogenous" else "d") * 64,
        checkpoint_phase_id="phase2",
        checkpoint_step_id="trial_design_request",
        canonical_reference_id=("reference-1" if condition == "canonical_state" else None),
    )


def test_checkpoint_schedule_requires_matched_three_condition_blocks() -> None:
    schedule = TrialDevCheckpointScheduleV1(
        experiment_id="checkpoint-pilot",
        participant_release_sha256="1" * 64,
        checkpoint_source_sha256="2" * 64,
        assignments=(
            _assignment("endogenous"),
            _assignment("context_reset"),
            _assignment("canonical_state", source="canonical"),
        ),
    )

    assert schedule.checksum is not None
    assert {row.condition for row in schedule.assignments} == {
        "endogenous",
        "context_reset",
        "canonical_state",
    }


def test_checkpoint_schedule_rejects_different_context_reset_source() -> None:
    with pytest.raises(ValidationError, match="reuse one endogenous source"):
        TrialDevCheckpointScheduleV1(
            experiment_id="checkpoint-pilot",
            participant_release_sha256="1" * 64,
            checkpoint_source_sha256="2" * 64,
            assignments=(
                _assignment("endogenous"),
                _assignment("context_reset", source="other"),
                _assignment("canonical_state", source="canonical"),
            ),
        )


def test_checkpoint_schedule_rejects_endogenous_source_relabelled_as_canonical() -> None:
    canonical = _assignment("canonical_state", source="canonical").model_copy(
        update={
            "source_program_relative_path": _assignment("endogenous").source_program_relative_path,
            "source_checkpoint_relative_path": _assignment("endogenous").source_checkpoint_relative_path,
            "source_checkpoint_sha256": _assignment("endogenous").source_checkpoint_sha256,
            "source_run_identity_sha256": _assignment("endogenous").source_run_identity_sha256,
        }
    )
    with pytest.raises(ValidationError, match="distinct canonical public-state source"):
        TrialDevCheckpointScheduleV1(
            experiment_id="checkpoint-pilot",
            participant_release_sha256="1" * 64,
            checkpoint_source_sha256="2" * 64,
            assignments=(
                _assignment("endogenous"),
                _assignment("context_reset"),
                canonical,
            ),
        )


def test_checkpoint_assignment_rejects_unsafe_or_unlabeled_canonical_source() -> None:
    payload = _assignment("endogenous").model_dump(mode="json")
    payload["source_program_relative_path"] = "../outside"
    with pytest.raises(ValidationError, match="safe relative paths"):
        TrialDevCheckpointAssignmentV1.model_validate(payload)

    payload = _assignment("canonical_state", source="canonical").model_dump(mode="json")
    payload["canonical_reference_id"] = None
    with pytest.raises(ValidationError, match="canonical_reference_id"):
        TrialDevCheckpointAssignmentV1.model_validate(payload)


def test_checkpoint_source_resolver_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        _safe_source_path(root, "../outside")


def test_checkpoint_analysis_excludes_reference_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignment = _assignment("canonical_state", source="canonical")
    chain = TrialDevChainSummaryV1(
        program_id=assignment.program_id,
        scenario_id=assignment.scenario_id,
        objective_id=assignment.objective_id,
        materialization_usage=TrialDevMaterializationUsageV1(),
        execution_status="completed",
    )
    reports = [
        _phase_report("phase1", primary_score=1.0, design_valid=True, effect_score=None, safety_score=1.0),
        _phase_report("phase2", primary_score=0.4, design_valid=False, effect_score=0.5, safety_score=0.25),
        _phase_report("phase3", primary_score=0.8, design_valid=True, effect_score=0.75, safety_score=1.0),
    ]
    grade = TrialDevTrajectoryGradeV1.model_construct(
        phase_reports=reports,
        decision_regret_by_phase={"phase1": 0.0, "phase2": 1.0, "phase3": 0.0},
    )

    def read_model(model: object, path: Path) -> object:
        del path
        if model is TrialDevChainSummaryV1:
            return chain
        if model is TrialDevTrajectoryGradeV1:
            return grade
        raise AssertionError(model)

    monkeypatch.setattr(checkpoint_analysis, "read_json_model", read_model)
    row = checkpoint_analysis._score_assignment(root=tmp_path, assignment=assignment)

    assert row.checkpoint_primary_score == pytest.approx(0.4)
    assert row.checkpoint_decision_correct == 0.0
    assert row.downstream_primary_score == pytest.approx(0.6)
    assert row.downstream_decision_score == pytest.approx(0.5)
    assert row.downstream_phase_count == 2
    assert row.checkpoint_quality.design_validity == 0.0
    assert row.checkpoint_quality.primary_effect_point_agreement == pytest.approx(0.5)
    assert row.downstream_quality.design_validity == pytest.approx(0.5)
    assert row.downstream_quality.primary_effect_interval_agreement == pytest.approx(0.625)
    assert row.downstream_quality.safety_evidence_agreement == pytest.approx(0.625)


def _phase_report(
    phase_id: str,
    *,
    primary_score: float,
    design_valid: bool,
    effect_score: float | None,
    safety_score: float,
) -> TrialDevGradeRecordV1:
    quality = TrialDevelopmentAnalysisQualityV1.model_construct(
        randomized_primary_effect_eligible=effect_score is not None,
        randomized_primary_effect_point_agreement=effect_score,
        randomized_primary_effect_interval_agreement=effect_score,
        safety_evidence_eligible=True,
        safety_evidence_agreement=safety_score,
        phase_evaluation_valid=True,
    )
    design = TrialDevDesignEfficiencyV1.model_construct(design_valid=design_valid)
    return TrialDevGradeRecordV1.model_construct(
        phase_id=phase_id,
        primary_score=primary_score,
        analysis_quality=quality,
        design_efficiency=design,
    )


def _score_row(condition: str, *, checkpoint_primary_score: float) -> TrialDevCheckpointScoreRowV1:
    quality = TrialDevCheckpointQualityV1(
        design_validity=1.0,
        phase_evaluation_validity=1.0,
        primary_effect_point_agreement=1.0,
        primary_effect_interval_agreement=1.0,
        safety_evidence_agreement=1.0,
    )
    return TrialDevCheckpointScoreRowV1(
        assignment_id=f"assignment-{condition}",
        block_id="block-1",
        program_id="scenario-1__benefit_risk",
        scenario_id="scenario-1",
        objective_id="benefit_risk",
        replicate_id="replicate-1",
        condition=condition,
        checkpoint_phase_id="phase2",
        checkpoint_primary_score=checkpoint_primary_score,
        checkpoint_decision_correct=1.0,
        downstream_primary_score=checkpoint_primary_score,
        downstream_decision_score=1.0,
        downstream_phase_count=2,
        checkpoint_quality=quality,
        downstream_quality=quality,
        completed=True,
    )


def test_single_block_checkpoint_pilot_is_descriptive_not_inferential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = TrialDevCheckpointScheduleV1(
        experiment_id="checkpoint-pilot",
        participant_release_sha256="1" * 64,
        checkpoint_source_sha256="2" * 64,
        assignments=(
            _assignment("endogenous"),
            _assignment("context_reset"),
            _assignment("canonical_state", source="canonical"),
        ),
    )
    rows = {
        "endogenous": _score_row("endogenous", checkpoint_primary_score=0.25),
        "context_reset": _score_row("context_reset", checkpoint_primary_score=0.5),
        "canonical_state": _score_row("canonical_state", checkpoint_primary_score=0.75),
    }

    def read_model(model: object, path: Path) -> object:
        del path
        if model is TrialDevCheckpointScheduleV1:
            return schedule
        raise AssertionError(model)

    monkeypatch.setattr(checkpoint_analysis, "read_json_model", read_model)
    monkeypatch.setattr(
        checkpoint_analysis,
        "_score_assignment",
        lambda *, root, assignment: rows[assignment.condition],
    )

    descriptive = checkpoint_analysis.summarise_trialdev_checkpoint_replay_v1(graded_root=tmp_path)

    observed = {(row.contrast_id, row.metric): row.estimate for row in descriptive.observed_contrasts}
    assert observed[("context_reset_minus_endogenous", "checkpoint_primary_score")] == pytest.approx(0.25)
    assert observed[("canonical_state_minus_endogenous", "checkpoint_primary_score")] == pytest.approx(0.5)
    assert observed[("canonical_state_minus_endogenous", "checkpoint_design_validity")] == 0.0
    assert observed[("context_reset_minus_endogenous", "downstream_safety_evidence_agreement")] == 0.0
    with pytest.raises(ValueError, match="requires 2"):
        checkpoint_analysis.analyse_trialdev_checkpoint_replay_v1(
            graded_root=tmp_path,
            confidence_level=0.95,
            bootstrap_resamples=1000,
            bootstrap_seed=1,
        )
