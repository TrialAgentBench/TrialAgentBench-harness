"""Tests for verified TrialDev checkpoint schedule compilation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.experiments import (
    TrialDevCanonicalCheckpointSourceV1,
    TrialDevCheckpointAssignmentV1,
    TrialDevCheckpointBlockPlanV1,
    TrialDevCheckpointSchedulePlanV1,
    TrialDevEndogenousCheckpointSourceV1,
)
from trialagentbench_harness.experiments import build_trialdev_checkpoint_schedule as builder


def _block(**updates: object) -> TrialDevCheckpointBlockPlanV1:
    payload: dict[str, object] = {
        "block_id": "block-1",
        "program_id": "scenario-1__benefit_risk",
        "scenario_id": "scenario-1",
        "objective_id": "benefit_risk",
        "replicate_id": "replicate-1",
        "decoding_seed": 17,
        "checkpoint_phase_id": "phase2",
        "checkpoint_step_id": "trial_design_request",
        "endogenous_program_relative_path": "endogenous/programs/scenario-1__benefit_risk",
        "canonical_program_relative_path": "canonical/programs/scenario-1__benefit_risk",
        "canonical_reference_id": "canonical-reference-1",
    }
    payload.update(updates)
    return TrialDevCheckpointBlockPlanV1.model_validate(payload)


def _assignment(condition: str) -> TrialDevCheckpointAssignmentV1:
    canonical = condition == "canonical_state"
    source = "canonical" if canonical else "endogenous"
    return TrialDevCheckpointAssignmentV1(
        assignment_id=f"block-1--{condition}",
        block_id="block-1",
        program_id="scenario-1__benefit_risk",
        scenario_id="scenario-1",
        objective_id="benefit_risk",
        replicate_id="replicate-1",
        decoding_seed=17,
        condition=condition,
        source_program_relative_path=f"{source}/programs/scenario-1__benefit_risk",
        source_checkpoint_relative_path=(f"{source}/programs/scenario-1__benefit_risk/checkpoints/00000001.json"),
        source_checkpoint_sha256=("b" if canonical else "a") * 64,
        source_run_identity_sha256=("d" if canonical else "c") * 64,
        checkpoint_phase_id="phase2",
        checkpoint_step_id="trial_design_request",
        canonical_reference_id="canonical-reference-1" if canonical else None,
    )


def _endogenous_source() -> TrialDevEndogenousCheckpointSourceV1:
    return TrialDevEndogenousCheckpointSourceV1(
        program_id="scenario-1__benefit_risk",
        scenario_id="scenario-1",
        objective_id="benefit_risk",
        replicate_id="replicate-1",
        decoding_seed=17,
        phase_id="phase2",
        step_id="trial_design_request",
        program_relative_path="endogenous/programs/scenario-1__benefit_risk",
        checkpoint_relative_path=("endogenous/programs/scenario-1__benefit_risk/checkpoints/00000001.json"),
        checkpoint_sha256="a" * 64,
        run_identity_sha256="c" * 64,
        provider_model="model",
        provider_route="provider:route",
        procedure_assistance="output_contract_only",
    )


def test_schedule_plan_rejects_duplicate_or_shared_sources() -> None:
    with pytest.raises(ValidationError, match="must be distinct"):
        _block(canonical_program_relative_path="endogenous/programs/scenario-1__benefit_risk")
    with pytest.raises(ValidationError, match="block IDs must be unique"):
        TrialDevCheckpointSchedulePlanV1(
            experiment_id="experiment",
            blocks=(_block(), _block(replicate_id="replicate-2", decoding_seed=18)),
        )


def test_compiler_derives_matched_triads_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participant = tmp_path / "participant"
    evaluator = tmp_path / "evaluator"
    sources = tmp_path / "sources"
    for path in (participant, evaluator, sources):
        path.mkdir()
    plan = TrialDevCheckpointSchedulePlanV1(
        experiment_id="experiment",
        blocks=(_block(),),
    )

    def assignment(**kwargs: object) -> TrialDevCheckpointAssignmentV1:
        return _assignment(str(kwargs["condition"]))

    checked: list[str] = []
    monkeypatch.setattr(builder, "_assignment", assignment)
    monkeypatch.setattr(
        builder,
        "_canonical_receipt_records",
        lambda **kwargs: {
            "canonical-reference-1": TrialDevCanonicalCheckpointSourceV1(
                canonical_reference_id="canonical-reference-1",
                program_id="scenario-1__benefit_risk",
                phase_id="phase2",
                step_id="trial_design_request",
                program_relative_path="canonical/programs/scenario-1__benefit_risk",
                checkpoint_relative_path=("canonical/programs/scenario-1__benefit_risk/checkpoints/00000001.json"),
                checkpoint_sha256="b" * 64,
            )
        },
    )
    monkeypatch.setattr(
        builder,
        "_endogenous_receipt_records",
        lambda **kwargs: {"endogenous/programs/scenario-1__benefit_risk": _endogenous_source()},
    )
    monkeypatch.setattr(
        builder,
        "_validate_canonical_prefix",
        lambda **kwargs: checked.append(kwargs["assignment"].canonical_reference_id),
    )
    monkeypatch.setattr(builder, "sha256_dir_digest", lambda path: "1" * 64 if path == participant else "2" * 64)

    schedule = builder.compile_trialdev_checkpoint_schedule_v1(
        participant_root=participant,
        evaluator_root=evaluator,
        checkpoint_root=sources,
        plan=plan,
    )

    assert [row.condition for row in schedule.assignments] == [
        "endogenous",
        "context_reset",
        "canonical_state",
    ]
    assert schedule.participant_release_sha256 == "1" * 64
    assert schedule.checkpoint_source_sha256 == "2" * 64
    assert checked == ["canonical-reference-1"]


def test_compiler_rejects_canonical_source_outside_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participant = tmp_path / "participant"
    evaluator = tmp_path / "evaluator"
    sources = tmp_path / "sources"
    for path in (participant, evaluator, sources):
        path.mkdir()
    plan = TrialDevCheckpointSchedulePlanV1(
        experiment_id="experiment",
        blocks=(_block(),),
    )
    monkeypatch.setattr(
        builder,
        "_assignment",
        lambda **kwargs: _assignment(str(kwargs["condition"])),
    )
    monkeypatch.setattr(builder, "_canonical_receipt_records", lambda **kwargs: {})
    monkeypatch.setattr(
        builder,
        "_endogenous_receipt_records",
        lambda **kwargs: {"endogenous/programs/scenario-1__benefit_risk": _endogenous_source()},
    )

    with pytest.raises(ValueError, match="absent from the custody receipt"):
        builder.compile_trialdev_checkpoint_schedule_v1(
            participant_root=participant,
            evaluator_root=evaluator,
            checkpoint_root=sources,
            plan=plan,
        )


def test_canonical_prefix_requires_perfect_public_evidence_grades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = tmp_path / "program"
    (program / "obs_review").mkdir(parents=True)
    (program / "obs_review" / "obs_review_submission.json").write_text("{}\n", encoding="utf-8")
    (program / "agent_workdir").mkdir()
    assignment = _assignment("canonical_state")
    checkpoint = SimpleNamespace(
        payload=SimpleNamespace(
            continuation=SimpleNamespace(payload=SimpleNamespace(violations=(), completed_phase_summaries=())),
            completed_phases=(SimpleNamespace(phase_id="phase1"),),
        )
    )
    monkeypatch.setattr(
        builder,
        "_validate_source_assignment",
        lambda *args, **kwargs: (program, checkpoint),
    )
    monkeypatch.setattr(builder, "scenario_root", lambda *args, **kwargs: tmp_path / "scenario")
    monkeypatch.setattr(
        builder.trialdev_upstream,
        "grade_item",
        lambda **kwargs: SimpleNamespace(
            primary_score=1.0,
            analysis_quality=SimpleNamespace(phase_evaluation_valid=True),
        ),
    )
    monkeypatch.setattr(
        builder.trialdev_upstream,
        "grade_trajectory",
        lambda **kwargs: SimpleNamespace(
            phase_reports=(
                SimpleNamespace(
                    phase_id="phase1",
                    primary_score=0.5,
                    analysis_quality=SimpleNamespace(phase_evaluation_valid=True),
                ),
            ),
            decision_regret_by_phase={"phase1": 0.0},
        ),
    )

    with pytest.raises(ValueError, match="not fully valid"):
        builder._validate_canonical_prefix(
            assignment=assignment,
            checkpoint_root=tmp_path,
            evaluator_root=tmp_path,
        )


def test_canonical_prefix_rejects_evaluator_artifacts(tmp_path: Path) -> None:
    program = tmp_path / "program"
    forbidden = program / "agent_workdir" / "hidden" / "truth.json"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="evaluator-only material"):
        builder._reject_evaluator_artifacts(program)
