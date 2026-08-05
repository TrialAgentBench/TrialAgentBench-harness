"""Tests for the TrialDev evaluation-target register contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.trialdev.trialdev_evaluation_target_register import (
    TrialDevEvaluationTargetRegisterRecordV1,
    load_trialdev_evaluation_target_register_records,
    required_trialdev_evaluation_contexts_v1,
    validate_trialdev_evaluation_target_register,
)
from trialagentbench_harness.contracts.trialdev.trialdev_recoverability import (
    TrialDevRecoverabilityManifestV1,
    TrialDevRecoverabilityRecordV1,
    required_trialdev_recoverability_keys_v1,
)


def _recoverability_record(phase_id: str, objective_id: str) -> TrialDevRecoverabilityRecordV1:
    return TrialDevRecoverabilityRecordV1(
        scenario_id="s1",
        phase_id=phase_id,
        objective_id=objective_id,
        method_route_id="unit_observational_method" if phase_id == "observational_review" else None,
        policy="no_recoverability_relaxation",
        public_evidence_basis=("public/eval_contract.json",),
        rationale="No relaxation in unit fixture.",
    )


def test_trialdev_recoverability_manifest_requires_complete_contexts() -> None:
    """Recoverability manifests fail closed when any phase/objective context is absent."""

    with pytest.raises(ValidationError, match="missing recoverability contexts"):
        TrialDevRecoverabilityManifestV1(
            scenario_id="s1",
            records=(_recoverability_record("observational_review", "benefit_risk"),),
        )

    manifest = TrialDevRecoverabilityManifestV1(
        scenario_id="s1",
        records=tuple(
            _recoverability_record(phase, objective) for phase, objective in required_trialdev_recoverability_keys_v1()
        ),
    )

    assert len(manifest.records) == 13


def test_trialdev_recoverability_rejects_no_relaxation_with_alternatives() -> None:
    """No-relaxation records must not smuggle acceptable alternatives."""

    with pytest.raises(ValidationError, match="cannot declare acceptable alternatives"):
        TrialDevRecoverabilityRecordV1(
            scenario_id="s1",
            phase_id="phase2",
            objective_id="benefit_risk",
            policy="no_recoverability_relaxation",
            acceptable_action_set=("advance",),
            public_evidence_basis=("public/eval_contract.json",),
            rationale="Invalid fixture.",
        )


def _evaluation_target(
    *,
    phase_id: str,
    program_objective_id: str,
    phase_scoring_objective_id: str,
    lane_id: str,
) -> TrialDevEvaluationTargetRegisterRecordV1:
    return TrialDevEvaluationTargetRegisterRecordV1(
        scenario_id="s1",
        phase_id=phase_id,
        program_objective_id=program_objective_id,
        phase_scoring_objective_id=phase_scoring_objective_id,
        lane_id=lane_id,
        scoring_policy_id="unit_policy",
        public_evidence_basis=("public/eval_contract.json",),
        evaluator_evidence_basis=("public/eval_contract.json",),
        reference_target_ids=("target",),
        recoverability_policy_id="no_recoverability_relaxation",
    )


def test_trialdev_evaluation_target_register_rejects_hidden_public_evidence() -> None:
    """Reference records must not expose hidden files as public evidence."""

    with pytest.raises(ValidationError, match="public evidence basis cannot reference"):
        TrialDevEvaluationTargetRegisterRecordV1(
            scenario_id="s1",
            phase_id="phase2",
            program_objective_id="benefit_risk",
            phase_scoring_objective_id="benefit_risk",
            lane_id="decision_action",
            scoring_policy_id="unit_policy",
            public_evidence_basis=("hidden/evaluation_reference_manifest.json",),
            evaluator_evidence_basis=("public/eval_contract.json",),
            reference_target_ids=("target",),
            recoverability_policy_id="no_recoverability_relaxation",
        )


def test_trialdev_evaluation_target_register_coverage_gate_detects_missing_contexts(tmp_path) -> None:
    """Coverage validation reports missing required trajectory reference contexts."""

    (tmp_path / "public").mkdir()
    (tmp_path / "grader").mkdir()
    (tmp_path / "public" / "eval_contract.json").write_text("{}", encoding="utf-8")

    issues = validate_trialdev_evaluation_target_register(
        scenario_root=tmp_path,
        records=(
            _evaluation_target(
                phase_id="phase2",
                program_objective_id="benefit_risk",
                phase_scoring_objective_id="benefit_risk",
                lane_id="decision_action",
            ),
        ),
    )

    assert any("missing required evaluation-target register contexts" in issue for issue in issues)


def test_trialdev_evaluation_target_register_accepts_complete_context_set(tmp_path) -> None:
    """The required context set can be represented without duplicate ambiguity."""

    (tmp_path / "public").mkdir()
    (tmp_path / "grader").mkdir()
    (tmp_path / "public" / "eval_contract.json").write_text("{}", encoding="utf-8")
    records = tuple(
        _evaluation_target(
            phase_id=phase_id,
            program_objective_id=program_objective_id,
            phase_scoring_objective_id=phase_scoring_objective_id,
            lane_id=lane_id,
        )
        for phase_id, program_objective_id, phase_scoring_objective_id, lane_id in required_trialdev_evaluation_contexts_v1()
    )

    assert validate_trialdev_evaluation_target_register(scenario_root=tmp_path, records=records) == ()


def test_trialdev_evaluation_target_register_artifact_requires_explicit_checksum(tmp_path: Path) -> None:
    """Serialized scoring rows cannot acquire custody checksums while loading."""

    record = _evaluation_target(
        phase_id="phase2",
        program_objective_id="benefit_risk",
        phase_scoring_objective_id="benefit_risk",
        lane_id="decision_action",
    )
    path = tmp_path / "evaluation_target_register.jsonl"
    path.write_text(
        json.dumps(record.model_dump(mode="json", exclude={"checksum"})) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicit checksum"):
        load_trialdev_evaluation_target_register_records(path)
