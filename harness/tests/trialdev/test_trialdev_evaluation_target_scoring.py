"""Tests for TrialDev evaluation-target scoring utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevAnalysisQualityEndpointV1,
    TrialDevGradeRecordV1,
    TrialDevLaneScoreRecordV1,
    TrialDevProgrammeAnalysisQualityV1,
)
from trialagentbench_harness.trialdev.aggregate import _lane_score_rows
from trialagentbench_harness.trialdev.grade_wrappers import trajectory_metrics_from_grade, wrap_grade_record
from trialagentbench_harness.trialdev.grading.evaluation_target_register import (
    TrialDevEvaluationTargetRegisterIndex,
    TrialDevEvaluationTargetRegisterRecordV1,
    score_evaluation_target,
)
from trialagentbench_harness.trialdev.grading.models import (
    TRIALDEV_GRADE_GATE_ORDER_V1,
    TrialDevelopmentAnalysisQualityV1,
    TrialDevelopmentGradeGateRecordV1,
)
from trialagentbench_harness.trialdev.grading.sequential import final_decision_lane_scores_from_trajectory


def _analysis_quality() -> TrialDevelopmentAnalysisQualityV1:
    return TrialDevelopmentAnalysisQualityV1(
        observational_analysis_eligible=True,
        observational_analysis_valid=False,
        observational_analysis_score=0.0,
        randomized_primary_effect_eligible=False,
        randomized_primary_effect_valid=None,
        safety_evidence_eligible=False,
        safety_evidence_valid=None,
        phase_evaluation_valid=False,
    )


def _grade_gates() -> tuple[TrialDevelopmentGradeGateRecordV1, ...]:
    return tuple(
        TrialDevelopmentGradeGateRecordV1(
            gate_id=gate_id,
            status="not_applicable" if gate_id == "integrity" else "passed",
        )
        for gate_id in TRIALDEV_GRADE_GATE_ORDER_V1
    )


def _programme_quality() -> TrialDevProgrammeAnalysisQualityV1:
    return TrialDevProgrammeAnalysisQualityV1(
        observational_analysis_validity=TrialDevAnalysisQualityEndpointV1(eligible_units=1, value=1.0),
        observational_analysis_score=TrialDevAnalysisQualityEndpointV1(eligible_units=1, value=1.0),
        randomized_primary_effect_point_agreement=TrialDevAnalysisQualityEndpointV1(eligible_units=0),
        randomized_primary_effect_interval_agreement=TrialDevAnalysisQualityEndpointV1(eligible_units=0),
        safety_evidence_agreement=TrialDevAnalysisQualityEndpointV1(eligible_units=0),
        phase_evaluation_validity=TrialDevAnalysisQualityEndpointV1(eligible_units=1, value=1.0),
    )


def _evaluation_target(*, lane_id: str = "asset_nomination") -> TrialDevEvaluationTargetRegisterRecordV1:
    return TrialDevEvaluationTargetRegisterRecordV1.model_validate(
        {
            "schema_id": "trialdev_evaluation_target_register_record_v1",
            "scenario_id": "s1",
            "phase_id": "observational_review",
            "program_objective_id": "benefit_risk",
            "phase_scoring_objective_id": "benefit_risk",
            "lane_id": lane_id,
            "scoring_policy_id": "candidate_policy_v1",
            "public_evidence_basis": ["public/eval_contract.json"],
            "evaluator_evidence_basis": ["grader/drug_ranking_reference_manifest.json"],
            "reference_target_ids": ["drug_a"],
            "credit_eligible_target_ids": ["drug_b"],
            "recoverability_policy_id": "acceptable_candidate_set",
            "value_payload": {},
            "checksum": "a" * 64,
        }
    )


def test_credit_eligible_alternative_receives_full_credit() -> None:
    """Every prequalified valid alternative receives full credit."""

    row = score_evaluation_target(
        scenario_id="s1",
        phase_id="observational_review",
        program_objective_id="benefit_risk",
        phase_scoring_objective_id="benefit_risk",
        lane_id="asset_nomination",
        submitted_target_id="drug_b",
        evaluation_target=_evaluation_target(),
        artifact_status="present",
    )

    assert row.status == "credit_eligible_alternative"
    assert row.score == 1.0
    assert row.score_derivation == "literal_target"


def test_register_scoring_rejects_different_s_prefixed_scenario() -> None:
    with pytest.raises(ValueError, match="scenario mismatch"):
        score_evaluation_target(
            scenario_id="s2",
            phase_id="observational_review",
            program_objective_id="benefit_risk",
            phase_scoring_objective_id="benefit_risk",
            lane_id="asset_nomination",
            submitted_target_id="drug_a",
            evaluation_target=_evaluation_target(),
            artifact_status="present",
        )


def test_public_evidence_action_can_replace_static_diagnostic_reference_target() -> None:
    """Realized public evidence, not a static oracle route, owns action credit."""

    truth = _evaluation_target(lane_id="decision_action").model_copy(
        update={
            "phase_id": "phase2",
            "reference_target_ids": ["stop_for_futility"],
            "credit_eligible_target_ids": [],
            "target_resolution": "realized_public_evidence",
        }
    )
    row = score_evaluation_target(
        scenario_id="s1",
        phase_id="phase2",
        program_objective_id="benefit_risk",
        phase_scoring_objective_id="benefit_risk",
        lane_id="decision_action",
        submitted_target_id="advance_to_confirmation",
        evaluation_target=truth,
        artifact_status="present",
        score_override=1.0,
        score_derivation="public_evidence_action",
    )

    assert row.status == "credit_eligible_alternative"
    assert row.score == 1.0
    assert row.score_derivation == "public_evidence_action"


def test_realized_public_evidence_target_rejects_literal_scoring() -> None:
    """A runtime-derived decision row cannot fall back to its register marker."""

    truth = _evaluation_target(lane_id="decision_action").model_copy(
        update={
            "phase_id": "phase2",
            "reference_target_ids": ["derived_from_realized_public_evidence"],
            "credit_eligible_target_ids": [],
            "target_resolution": "realized_public_evidence",
        }
    )
    with pytest.raises(ValueError, match="require public_evidence_action"):
        score_evaluation_target(
            scenario_id="s1",
            phase_id="phase2",
            program_objective_id="benefit_risk",
            phase_scoring_objective_id="benefit_risk",
            lane_id="decision_action",
            submitted_target_id="derived_from_realized_public_evidence",
            evaluation_target=truth,
            artifact_status="present",
        )


def test_public_evidence_action_is_restricted_to_runtime_action_lanes() -> None:
    """Dynamic action evidence cannot override unrelated scoring lanes."""

    with pytest.raises(ValueError, match="restricted to decision-action, safety-gate, and route-timing"):
        score_evaluation_target(
            scenario_id="s1",
            phase_id="observational_review",
            program_objective_id="benefit_risk",
            phase_scoring_objective_id="benefit_risk",
            lane_id="asset_nomination",
            submitted_target_id="drug_z",
            evaluation_target=_evaluation_target(),
            artifact_status="present",
            score_override=1.0,
            score_derivation="public_evidence_action",
        )


def test_public_evidence_action_scores_runtime_safety_lane() -> None:
    """Safety credit is derived from the realized public evidence, not hidden action labels."""

    truth = TrialDevEvaluationTargetRegisterRecordV1.model_validate(
        {
            **_evaluation_target().model_dump(),
            "phase_id": "phase1",
            "lane_id": "safety_gate",
            "reference_target_ids": ["derived_from_realized_public_evidence"],
            "credit_eligible_target_ids": [],
            "target_resolution": "realized_public_evidence",
        }
    )
    score = score_evaluation_target(
        scenario_id="s1",
        phase_id="phase1",
        program_objective_id="benefit_risk",
        phase_scoring_objective_id="benefit_risk",
        lane_id="safety_gate",
        submitted_target_id="stop_development",
        evaluation_target=truth,
        artifact_status="present",
        score_override=1.0,
        score_derivation="public_evidence_action",
    )

    assert score.score == 1.0
    assert score.status == "credit_eligible_alternative"


def test_register_scoring_rejects_context_mismatch() -> None:
    """The scoring primitive validates the full evaluation-target register context."""

    with pytest.raises(ValueError, match="context mismatch"):
        score_evaluation_target(
            scenario_id="s1",
            phase_id="phase1",
            program_objective_id="benefit_risk",
            phase_scoring_objective_id="benefit_risk",
            lane_id="asset_nomination",
            submitted_target_id="drug_a",
            evaluation_target=_evaluation_target(),
            artifact_status="present",
        )


def test_register_scoring_rejects_wrong_target_with_score_override() -> None:
    """Score overrides cannot assign credit to unaccepted categorical targets."""

    with pytest.raises(ValueError, match="unaccepted target"):
        score_evaluation_target(
            scenario_id="s1",
            phase_id="observational_review",
            program_objective_id="benefit_risk",
            phase_scoring_objective_id="benefit_risk",
            lane_id="phase_analysis",
            submitted_target_id="drug_z",
            evaluation_target=_evaluation_target(lane_id="phase_analysis"),
            artifact_status="present",
            score_override=0.9,
            score_derivation="numeric_diagnostic",
        )


def test_register_scoring_rejects_score_override_on_categorical_lane() -> None:
    """Categorical lanes cannot use numeric score overrides."""

    with pytest.raises(ValueError, match="not allowed for categorical lane"):
        score_evaluation_target(
            scenario_id="s1",
            phase_id="observational_review",
            program_objective_id="benefit_risk",
            phase_scoring_objective_id="benefit_risk",
            lane_id="asset_nomination",
            submitted_target_id="drug_a",
            evaluation_target=_evaluation_target(),
            artifact_status="present",
            score_override=0.9,
            score_derivation="numeric_diagnostic",
        )


def test_missing_evaluation_target_register_context_fails_loudly() -> None:
    """A scoreable lane cannot fall back to a raw manifest when the register is missing."""

    index = TrialDevEvaluationTargetRegisterIndex((_evaluation_target(),))

    with pytest.raises(ValueError, match="missing TrialDev evaluation-target register scoring context"):
        index.require(
            phase_id="phase1",
            program_objective_id="benefit_risk",
            phase_scoring_objective_id="benefit_risk",
            lane_id="decision_action",
        )


def test_invalid_register_json_row_reports_line_number(tmp_path: Path) -> None:
    """Malformed register JSON is rejected with a row-local error."""

    path = tmp_path / "evaluation_target_register.jsonl"
    path.write_text(
        json.dumps({"schema_id": "trialdev_evaluation_target_register_record_v1"}) + "\n", encoding="utf-8"
    )

    from trialagentbench_harness.trialdev.grading.evaluation_target_register import load_evaluation_target_index

    scenario = tmp_path / "scenario"
    (scenario / "grader").mkdir(parents=True)
    path.rename(scenario / "grader" / "evaluation_target_register.jsonl")

    with pytest.raises(ValueError, match="row 1"):
        load_evaluation_target_index(scenario)


def test_grade_wrapper_rejects_malformed_lane_score_entries() -> None:
    """Malformed lane-score entries must fail instead of being silently skipped."""

    with pytest.raises(ValueError, match="lane_scores entries must be objects"):
        wrap_grade_record(
            {
                "primary_score": 0.0,
                "design_score": 0.0,
                "evaluation_score": 0.0,
                "program_score": 0.0,
                "ranking_score": 0.0,
                "lane_scores": ["not-a-record"],
                "payload": {},
            }
        )


def test_grade_wrapper_rejects_removed_audit_gate_field() -> None:
    """Old audit-gate fields must fail instead of being passed silently."""

    with pytest.raises(ValidationError, match="diagnostic_compatibility_score"):
        wrap_grade_record(
            {
                "primary_score": 0.0,
                "design_score": 0.0,
                "evaluation_score": 0.0,
                "program_score": 0.0,
                "ranking_score": 0.0,
                "audit_gates": {
                    "diagnostic_compatibility_score": 0.0,
                    "gates_triggered": [],
                },
                "payload": {},
            }
        )


def test_grade_wrapper_accepts_diagnostic_alignment_score() -> None:
    """Current audit-gate score field is preserved in the canonical wrapper."""

    record = wrap_grade_record(
        {
            "primary_score": 0.25,
            "design_score": 0.0,
            "evaluation_score": 0.0,
            "program_score": 0.0,
            "ranking_score": 0.0,
            "analysis_quality": _analysis_quality().model_dump(mode="json"),
            "gates": [gate.model_dump(mode="json") for gate in _grade_gates()],
            "validity": {"valid": True},
            "audit_gates": {
                "diagnostic_alignment_score": 0.25,
                "gates_triggered": ["low_evaluation_quality"],
            },
            "payload": {},
        }
    )

    assert record.audit_gates is not None
    assert record.audit_gates.diagnostic_alignment_score == 0.25
    assert record.audit_gates.gates_triggered == ["low_evaluation_quality"]


@pytest.mark.parametrize(
    "missing_field",
    [
        "primary_score",
        "design_score",
        "evaluation_score",
        "program_score",
        "ranking_score",
    ],
)
def test_grade_wrapper_rejects_missing_headline_scores(missing_field: str) -> None:
    payload = {
        "primary_score": 0.0,
        "design_score": 0.0,
        "evaluation_score": 0.0,
        "program_score": 0.0,
        "ranking_score": 0.0,
    }
    payload.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        wrap_grade_record(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("active_lane_scores", {"trial_design": "1.0"}),
        ("lane_breakdown", []),
        ("lane_status", {"trial_design": 1}),
        ("feasibility_failures", "not-a-list"),
        ("policy_reference_regret", "0.5"),
    ],
)
def test_grade_wrapper_rejects_coerced_or_malformed_fields(field: str, value: object) -> None:
    """The upstream JSON boundary must not coerce malformed grade fields."""

    payload: dict[str, object] = {
        "primary_score": 0.0,
        "design_score": 0.0,
        "evaluation_score": 0.0,
        "program_score": 0.0,
        "ranking_score": 0.0,
        "analysis_quality": _analysis_quality().model_dump(mode="json"),
        "gates": [gate.model_dump(mode="json") for gate in _grade_gates()],
        "validity": {"valid": True},
        field: value,
    }

    with pytest.raises((ValueError, ValidationError), match=field):
        wrap_grade_record(payload)


def test_grade_wrapper_rejects_non_json_payload_values() -> None:
    """Audit payloads must remain portable JSON rather than Python objects."""

    payload: dict[str, object] = {
        "primary_score": 0.0,
        "design_score": 0.0,
        "evaluation_score": 0.0,
        "program_score": 0.0,
        "ranking_score": 0.0,
        "unexpected": object(),
    }

    with pytest.raises(ValidationError, match="unexpected"):
        wrap_grade_record(payload)


def _lane_record(*, checksum: str = "b" * 64, scenario_id: str = "s01") -> TrialDevLaneScoreRecordV1:
    return TrialDevLaneScoreRecordV1(
        scenario_id=scenario_id,
        phase_id="observational_review",
        program_objective_id="benefit_risk",
        phase_scoring_objective_id="benefit_risk",
        lane_id="asset_nomination",
        evaluation_target_checksum=checksum,
        scoring_policy_id="candidate_policy_v1",
        recoverability_policy_id="acceptable_candidate_set",
        submitted_target_id="drug_a",
        reference_target_ids=("drug_a",),
        credit_eligible_target_ids=(),
        score=1.0,
        status="scored",
        artifact_status="present",
    )


@pytest.mark.parametrize("policy_id", ["unique", "set_identified", "safety_determined"])
def test_lane_record_accepts_realized_decision_recoverability_classes(policy_id: str) -> None:
    """Decision lanes retain the evidence-derived recoverability class."""

    payload = _lane_record().model_dump()
    payload["recoverability_policy_id"] = policy_id
    assert TrialDevLaneScoreRecordV1.model_validate(payload).recoverability_policy_id == policy_id


def test_supported_observational_stop_is_a_complete_programme() -> None:
    """Qualified non-nomination makes later randomized phases structural, not missing."""

    nomination = _lane_record().model_copy(
        update={
            "submitted_target_id": "withhold_nomination",
            "reference_target_ids": ("withhold_nomination",),
        }
    )
    analysis = _lane_record(checksum="c" * 64).model_copy(
        update={"lane_id": "phase_analysis", "submitted_target_id": "qualified_nonidentification"}
    )
    report = TrialDevGradeRecordV1(
        primary_score=1.0,
        design_score=0.0,
        evaluation_score=1.0,
        program_score=1.0,
        ranking_score=1.0,
        analysis_quality=_analysis_quality(),
        gates=_grade_gates(),
        validity={"valid": True},
        phase_id="observational_review",
        lane_scores=[nomination, analysis],
        payload={"validity": {"valid": True}},
    )

    metrics = trajectory_metrics_from_grade(
        trajectory_grade=None,
        observational_report=report,
        phase_policy_modes={"phase1": "required", "phase2": "required", "phase3": "optional"},
        analysis_quality=_programme_quality(),
    )

    outcomes = {record.phase_id: record for record in metrics.checkpoint_outcomes}
    assert metrics.programme_primary_score == 1.0
    assert metrics.trajectory_decision_score == 1.0
    assert outcomes["phase1"].status == "structural_not_reached"
    assert outcomes["phase2"].status == "structural_not_reached"
    assert outcomes["phase3"].status == "structural_not_reached"
    assert outcomes["final_decision"].status == "reached"
    assert metrics.resource_summary is not None
    assert metrics.resource_summary.phase_count == 0


def test_observational_only_nomination_is_not_mislabelled_as_a_supported_stop() -> None:
    nomination = _lane_record()
    analysis = _lane_record(checksum="c" * 64).model_copy(update={"lane_id": "phase_analysis"})
    report = TrialDevGradeRecordV1(
        primary_score=1.0,
        design_score=0.0,
        evaluation_score=1.0,
        program_score=1.0,
        ranking_score=1.0,
        analysis_quality=_analysis_quality(),
        gates=_grade_gates(),
        validity={"valid": True},
        phase_id="observational_review",
        lane_scores=[nomination, analysis],
        payload={"validity": {"valid": True}},
    )

    metrics = trajectory_metrics_from_grade(
        trajectory_grade=None,
        observational_report=report,
        phase_policy_modes={"phase1": "required", "phase2": "required", "phase3": "optional"},
        analysis_quality=_programme_quality(),
    )

    outcomes = {record.phase_id: record for record in metrics.checkpoint_outcomes}
    assert metrics.programme_primary_score == 0.0
    assert outcomes["phase1"].status == "not_reached_after_invalid"
    assert outcomes["final_decision"].status == "missing_or_invalid"


def test_lane_export_records_scenario_key_and_semantic_identity() -> None:
    """Lane exports must make short scenario keys and semantic truth ids explicit."""

    report = TrialDevGradeRecordV1(
        primary_score=1.0,
        design_score=0.0,
        evaluation_score=0.0,
        program_score=0.0,
        ranking_score=0.0,
        analysis_quality=_analysis_quality(),
        gates=_grade_gates(),
        validity={"valid": True},
        lane_scores=[_lane_record()],
        payload={},
    )

    rows = _lane_score_rows(
        report=report,
        program_id="s01__benefit_risk",
        scenario_key="s01",
        objective_id="benefit_risk",
        reference_scenario_by_checksum={"b" * 64: "no_progression"},
        source="unit",
    )

    assert len(rows) == 1
    assert rows[0].scenario_id == "no_progression"
    assert rows[0].scenario_key == "s01"
    assert rows[0].scenario_semantic_id == "no_progression"


def test_lane_export_rejects_unknown_truth_checksum() -> None:
    """Lane exports cannot silently retain rows that do not map to bundle truth."""

    report = TrialDevGradeRecordV1(
        primary_score=1.0,
        design_score=0.0,
        evaluation_score=0.0,
        program_score=0.0,
        ranking_score=0.0,
        analysis_quality=_analysis_quality(),
        gates=_grade_gates(),
        validity={"valid": True},
        lane_scores=[_lane_record(checksum="c" * 64)],
        payload={},
    )

    with pytest.raises(ValueError, match="unknown evaluation_target_checksum"):
        _lane_score_rows(
            report=report,
            program_id="s01__benefit_risk",
            scenario_key="s01",
            objective_id="benefit_risk",
            reference_scenario_by_checksum={"b" * 64: "no_progression"},
            source="unit",
        )


def test_lane_export_rejects_unregistered_public_evidence_witness() -> None:
    """Runtime evidence cannot replace the immutable register identity."""

    payload = _lane_record(checksum="c" * 64).model_dump()
    payload.update(
        {
            "phase_id": "phase2",
            "lane_id": "decision_action",
            "score_derivation": "public_evidence_action",
            "recoverability_policy_id": "set_identified",
        }
    )
    report = TrialDevGradeRecordV1(
        primary_score=1.0,
        design_score=0.0,
        evaluation_score=0.0,
        program_score=0.0,
        ranking_score=0.0,
        analysis_quality=_analysis_quality(),
        gates=_grade_gates(),
        validity={"valid": True},
        lane_scores=[TrialDevLaneScoreRecordV1.model_validate(payload)],
        payload={},
    )

    with pytest.raises(ValueError, match="unknown evaluation_target_checksum"):
        _lane_score_rows(
            report=report,
            program_id="s01__benefit_risk",
            scenario_key="s01",
            objective_id="benefit_risk",
            reference_scenario_by_checksum={"b" * 64: "s01"},
            source="unit",
        )


def _write_final_decision_truth_bundle(root: Path) -> None:
    grader = root / "grader"
    grader.mkdir(parents=True)
    rows = []
    for index, lane_id in enumerate(("route_timing", "final_recommendation"), start=1):
        rows.append(
            {
                "schema_id": "trialdev_evaluation_target_register_record_v1",
                "scenario_id": "no_progression",
                "phase_id": "final_decision",
                "program_objective_id": "benefit_risk",
                "phase_scoring_objective_id": "benefit_risk",
                "lane_id": lane_id,
                "scoring_policy_id": "trajectory_policy_v1",
                "public_evidence_basis": ["public/program_loop_manifest.json"],
                "evaluator_evidence_basis": ["grader/trajectory_truth.json"],
                "reference_target_ids": ["declare_success"],
                "credit_eligible_target_ids": [],
                "recoverability_policy_id": "no_recoverability_relaxation",
                "value_payload": {},
                "checksum": f"{index}" * 64,
            }
        )
    (grader / "evaluation_target_register.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_runtime_final_decision_truth_bundle(root: Path) -> None:
    grader = root / "grader"
    grader.mkdir(parents=True)
    rows = []
    for index, lane_id in enumerate(("route_timing", "final_recommendation"), start=1):
        rows.append(
            {
                "schema_id": "trialdev_evaluation_target_register_record_v1",
                "scenario_id": "no_progression",
                "phase_id": "final_decision",
                "program_objective_id": "benefit_risk",
                "phase_scoring_objective_id": "benefit_risk",
                "lane_id": lane_id,
                "scoring_policy_id": "trajectory_policy_v1",
                "public_evidence_basis": ["public/program_loop_manifest.json"],
                "evaluator_evidence_basis": ["grader/trajectory_truth.json"],
                "reference_target_ids": ["derived_from_realized_trajectory"],
                "credit_eligible_target_ids": [],
                "recoverability_policy_id": "no_recoverability_relaxation",
                "target_resolution": "realized_trajectory",
                "value_payload": {},
                "checksum": f"{index}" * 64,
            }
        )
    (grader / "evaluation_target_register.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_final_decision_lanes_score_terminal_trajectory_metrics(tmp_path: Path) -> None:
    """Terminal TrialDev trajectories emit final-decision register rows."""

    scenario_root = tmp_path / "scenario_s01"
    _write_final_decision_truth_bundle(scenario_root)

    rows = final_decision_lane_scores_from_trajectory(
        scenario_root=scenario_root,
        scenario_id="no_progression",
        program_objective_id="benefit_risk",
        terminal_action="declare_success",
        terminal_recommendation_score=0.6,
        trajectory_decision_score=0.7,
        artifact_status="present",
        failure_reason=None,
    )

    assert [row.lane_id for row in rows] == ["route_timing", "final_recommendation"]
    assert [row.score for row in rows] == [0.7, 0.6]
    assert all(row.phase_id == "final_decision" for row in rows)
    assert all(row.score_derivation == "numeric_diagnostic" for row in rows)
    assert all(row.derived_from_trajectory_metric for row in rows)
    assert all(row.terminal_action_observed == "declare_success" for row in rows)


def test_final_decision_runtime_targets_score_realized_trajectory_metrics(
    tmp_path: Path,
) -> None:
    """Runtime target markers do not suppress realized trajectory scores."""

    scenario_root = tmp_path / "scenario_s01"
    _write_runtime_final_decision_truth_bundle(scenario_root)

    rows = final_decision_lane_scores_from_trajectory(
        scenario_root=scenario_root,
        scenario_id="no_progression",
        program_objective_id="benefit_risk",
        terminal_action="declare_failure",
        terminal_recommendation_score=0.6,
        trajectory_decision_score=0.7,
        artifact_status="present",
        failure_reason=None,
    )

    assert [row.submitted_target_id for row in rows] == [
        "declare_failure",
        "declare_failure",
    ]
    assert [row.score for row in rows] == [0.7, 0.6]
    assert all(row.derived_from_trajectory_metric for row in rows)


def test_final_decision_lanes_reject_missing_scores_for_present_artifact(tmp_path: Path) -> None:
    """A present final trajectory cannot receive implicit full credit for missing metrics."""

    scenario_root = tmp_path / "scenario_s01"
    _write_final_decision_truth_bundle(scenario_root)

    with pytest.raises(ValueError, match="require scores"):
        final_decision_lane_scores_from_trajectory(
            scenario_root=scenario_root,
            scenario_id="no_progression",
            program_objective_id="benefit_risk",
            terminal_action="declare_success",
            terminal_recommendation_score=None,
            trajectory_decision_score=0.7,
            artifact_status="present",
            failure_reason=None,
        )


def test_final_decision_lanes_never_substitute_reference_targets(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenario_s01"
    _write_final_decision_truth_bundle(scenario_root)

    rows = final_decision_lane_scores_from_trajectory(
        scenario_root=scenario_root,
        scenario_id="no_progression",
        program_objective_id="benefit_risk",
        terminal_action="declare_failure",
        terminal_recommendation_score=1.0,
        trajectory_decision_score=1.0,
        artifact_status="present",
        failure_reason=None,
    )

    assert [row.submitted_target_id for row in rows] == [
        "declare_failure",
        "declare_failure",
    ]
    assert [row.score for row in rows] == [0.0, 0.0]
    assert all(not row.derived_from_trajectory_metric for row in rows)
