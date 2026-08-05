"""Canonical wrappers for upstream TrialDev grader artifacts.

The upstream grader emits plain JSON objects. For benchmark-grade auditability,
the harness stores schema-bearing wrappers on disk so consumers can validate
contracts strictly and avoid untyped JSON boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from pydantic import TypeAdapter
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.core.runs import (
    TrialDevCheckpointOutcomeV1,
    TrialDevProgrammeResourceSummaryV1,
    TrialDevTrajectoryMetricsV1,
)
from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TrialDevScientificAssessmentV1,
    unassessed_scientific_assessment_v1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_evaluation_target_register import (
    required_trialdev_lanes_v1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevAnalysisQualityEndpointV1,
    TrialDevAuditGatesV1,
    TrialDevGradeRecordV1,
    TrialDevLaneScoreRecordV1,
    TrialDevProgrammeAnalysisQualityV1,
    TrialDevTerminalSummaryV1,
    TrialDevTrajectoryGradeV1,
)
from trialagentbench_harness.trialdev.grading.design_frontier import (
    derive_programme_resource_consequence_v1,
)
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevDesignEfficiencyV1,
    TrialDevelopmentAnalysisQualityV1,
    TrialDevelopmentGradeGateIdV1,
    TrialDevelopmentGradeGateRecordV1,
    TrialDevelopmentValidityReportV1,
    TrialDevProgrammeResourceConsequenceV1,
)
from trialagentbench_harness.trialdev.share.sequential import TrialDevelopmentProgramLoopManifestV1

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


def phase_policy_modes_from_manifest(path: Path) -> dict[str, str]:
    """Load the phase availability policy from one public programme manifest."""

    manifest = TrialDevelopmentProgramLoopManifestV1.model_validate_json(Path(path).read_text(encoding="utf-8"))
    return {str(phase_id): str(mode) for phase_id, mode in manifest.phase_policy_modes.items()}


def _validated_payload(raw: Mapping[str, object]) -> dict[str, JsonValue]:
    """Return the grader payload after strict JSON-value validation."""

    return _JSON_OBJECT.validate_python(dict(raw), strict=True)


def _required_score(raw: Mapping[str, object], key: str) -> float:
    """Return one required finite unit-interval score."""

    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"TrialDev grade report requires numeric {key}.")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"TrialDev grade report {key} must be finite and in [0, 1].")
    return score


def _required_count(raw: Mapping[str, object], key: str) -> int:
    """Return one required non-negative integer count."""

    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"TrialDev trajectory report requires non-negative integer {key}.")
    return value


def _optional_finite_number(raw: Mapping[str, object], key: str) -> float | None:
    """Return an optional finite numeric field without string coercion."""

    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"TrialDev grade report {key} must be numeric when present.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"TrialDev grade report {key} must be finite when present.")
    return number


def _optional_string(raw: Mapping[str, object], key: str) -> str | None:
    """Return an optional non-empty string field without coercion."""

    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"TrialDev grade report {key} must be a non-empty string when present.")
    return value


def _mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return an optional object-valued field, defaulting only on absence."""

    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"TrialDev grade report {key} must be an object.")
    return value


def _finite_number_mapping(raw: Mapping[str, object], key: str) -> dict[str, float]:
    """Return a string-keyed mapping of finite numeric values."""

    output: dict[str, float] = {}
    for name, value in _mapping(raw, key).items():
        if not isinstance(name, str):
            raise ValueError(f"TrialDev grade report {key} keys must be strings.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"TrialDev grade report {key}.{name} must be numeric.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"TrialDev grade report {key}.{name} must be finite.")
        output[name] = number
    return output


def _string_mapping(raw: Mapping[str, object], key: str) -> dict[str, str]:
    """Return a string-to-string mapping without coercion."""

    output: dict[str, str] = {}
    for name, value in _mapping(raw, key).items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError(f"TrialDev grade report {key} must contain only string keys and values.")
        output[name] = value
    return output


def _sequence(raw: Mapping[str, object], key: str) -> Sequence[object]:
    """Return an optional list-valued field, defaulting only on absence."""

    value = raw.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"TrialDev grade report {key} must be a list.")
    return value


def _string_list(raw: Mapping[str, object], key: str) -> list[str]:
    """Return a list containing only strings."""

    values = _sequence(raw, key)
    if any(not isinstance(value, str) for value in values):
        raise ValueError(f"TrialDev grade report {key} must contain only strings.")
    return [value for value in values if isinstance(value, str)]


def wrap_grade_record(raw: Mapping[str, object]) -> TrialDevGradeRecordV1:
    """Wrap an upstream `grade_item_v1`-shaped report."""

    payload = _validated_payload(raw)
    audit = raw.get("audit_gates")
    audit_model = None
    if audit is not None:
        if not isinstance(audit, Mapping):
            raise ValueError("TrialDev grade report audit_gates must be an object.")
        audit_model = TrialDevAuditGatesV1.model_validate(audit)
    design_efficiency = raw.get("design_efficiency")
    design_efficiency_model = (
        None if design_efficiency is None else TrialDevDesignEfficiencyV1.model_validate(design_efficiency)
    )
    lane_scores = []
    for entry in _sequence(raw, "lane_scores"):
        if not isinstance(entry, Mapping):
            raise ValueError("lane_scores entries must be objects.")
        lane_scores.append(wrap_lane_score_record(entry))
    return TrialDevGradeRecordV1(
        primary_score=_required_score(raw, "primary_score"),
        design_score=_required_score(raw, "design_score"),
        evaluation_score=_required_score(raw, "evaluation_score"),
        program_score=_required_score(raw, "program_score"),
        ranking_score=_required_score(raw, "ranking_score"),
        analysis_quality=TrialDevelopmentAnalysisQualityV1.model_validate(raw.get("analysis_quality")),
        scientific_assessment=(
            TrialDevScientificAssessmentV1.model_validate(raw["scientific_assessment"])
            if "scientific_assessment" in raw
            else unassessed_scientific_assessment_v1()
        ),
        gates=tuple(TrialDevelopmentGradeGateRecordV1.model_validate(entry) for entry in _sequence(raw, "gates")),
        first_failure_gate=cast(
            TrialDevelopmentGradeGateIdV1 | None,
            _optional_string(raw, "first_failure_gate"),
        ),
        validity=TrialDevelopmentValidityReportV1.model_validate(raw.get("validity")),
        policy_reference_regret=_optional_finite_number(raw, "policy_reference_regret"),
        in_set_regret=_optional_finite_number(raw, "in_set_regret"),
        active_lane_scores=_finite_number_mapping(raw, "active_lane_scores"),
        lane_breakdown=_finite_number_mapping(raw, "lane_breakdown"),
        lane_status=_string_mapping(raw, "lane_status"),
        audit_gates=audit_model,
        design_efficiency=design_efficiency_model,
        selected_winner_drug_id=_optional_string(raw, "selected_winner_drug_id"),
        best_candidate_drug_id=_optional_string(raw, "best_candidate_drug_id"),
        feasibility_failures=_string_list(raw, "feasibility_failures"),
        phase_id=_optional_string(raw, "phase_id"),
        scenario_id=_optional_string(raw, "scenario_id"),
        objective_id=_optional_string(raw, "objective_id"),
        program_objective_id=_optional_string(raw, "program_objective_id"),
        phase_scoring_objective_id=_optional_string(raw, "phase_scoring_objective_id"),
        checksum=_optional_string(raw, "checksum"),
        lane_scores=lane_scores,
        payload=payload,
    )


def wrap_trajectory_grade(raw: Mapping[str, object]) -> TrialDevTrajectoryGradeV1:
    """Wrap an upstream `grade_trajectory_v1` report."""

    payload = _validated_payload(raw)
    phase_reports = []
    for entry in _sequence(raw, "phase_reports"):
        if isinstance(entry, Mapping):
            phase_reports.append(wrap_grade_record(entry))
        else:
            raise ValueError("phase_reports entries must be objects.")
    final_lane_scores = []
    for entry in _sequence(raw, "final_lane_scores"):
        if not isinstance(entry, Mapping):
            raise ValueError("final_lane_scores entries must be objects.")
        final_lane_scores.append(wrap_lane_score_record(entry))
    invalid_reasons: list[str] = []
    for invalid_attempt in _sequence(raw, "invalid_attempts"):
        if not isinstance(invalid_attempt, Mapping):
            raise ValueError("invalid_attempts entries must be objects.")
        reason = invalid_attempt.get("reason_code")
        if reason is not None:
            if not isinstance(reason, str):
                raise ValueError("invalid_attempts reason_code must be a string when present.")
            invalid_reasons.append(reason)
    terminal_summary = TrialDevTerminalSummaryV1.model_validate(raw.get("terminal_summary"))
    resource_consequence = TrialDevProgrammeResourceConsequenceV1.model_validate(raw.get("resource_consequence"))
    return TrialDevTrajectoryGradeV1(
        checksum=_optional_string(raw, "checksum"),
        terminal_status=_optional_string(raw, "terminal_status"),
        program_objective_id=_optional_string(raw, "program_objective_id"),
        phase_scoring_objectives=_string_mapping(raw, "phase_scoring_objectives"),
        trajectory_primary_score=_required_score(raw, "trajectory_primary_score"),
        trajectory_decision_score=_required_score(raw, "trajectory_decision_score"),
        decision_regret_by_phase=_finite_number_mapping(raw, "decision_regret_by_phase"),
        mean_scores=_finite_number_mapping(raw, "mean_scores"),
        n_invalid_attempts=_required_count(raw, "n_invalid_attempts"),
        n_phase_submissions=_required_count(raw, "n_phase_submissions"),
        invalid_attempt_reasons=invalid_reasons,
        terminal_summary=terminal_summary,
        resource_consequence=resource_consequence,
        phase_reports=phase_reports,
        final_lane_scores=final_lane_scores,
        payload=payload,
    )


def wrap_lane_score_record(raw: Mapping[str, object]) -> TrialDevLaneScoreRecordV1:
    """Wrap one upstream lane-score record in the harness contract."""

    payload = _validated_payload(raw)
    payload["schema_id"] = "trialagentbench_trialdev_lane_score_record_v1"
    return TrialDevLaneScoreRecordV1.model_validate(payload)


def _quality_endpoint(values: list[float]) -> TrialDevAnalysisQualityEndpointV1:
    """Reduce eligible phase values conjunctively."""

    return TrialDevAnalysisQualityEndpointV1(
        eligible_units=len(values),
        value=min(values) if values else None,
    )


def summarise_programme_analysis_quality(
    *,
    observational_report: TrialDevGradeRecordV1 | None,
    phase_reports: Sequence[TrialDevGradeRecordV1],
    attempted_phase_ids: set[str],
) -> TrialDevProgrammeAnalysisQualityV1:
    """Summarise typed phase quality without scoring structurally absent phases."""

    if observational_report is None:
        observational_validity = 0.0
        observational_score = 0.0
        observational_phase_validity = 0.0
    else:
        quality = observational_report.analysis_quality
        if (
            observational_report.phase_id != "observational_review"
            or not quality.observational_analysis_eligible
            or quality.observational_analysis_valid is None
            or quality.observational_analysis_score is None
        ):
            raise ValueError("TrialDev observational grade has inconsistent typed quality applicability.")
        observational_validity = float(quality.observational_analysis_valid)
        observational_score = float(quality.observational_analysis_score)
        observational_phase_validity = float(quality.phase_evaluation_valid)

    randomized_phases = {"phase1", "phase2", "phase3"}
    primary_effect_phases = {"phase2", "phase3"}
    reports_by_phase: dict[str, TrialDevGradeRecordV1] = {}
    for candidate_report in phase_reports:
        phase_id = str(candidate_report.phase_id or "")
        if phase_id not in randomized_phases:
            raise ValueError(f"TrialDev trajectory contains an unexpected phase quality report: {phase_id!r}.")
        if phase_id in reports_by_phase:
            raise ValueError(f"TrialDev trajectory duplicates phase quality report: {phase_id!r}.")
        reports_by_phase[phase_id] = candidate_report
    entered_phases = {phase for phase in attempted_phase_ids if phase in randomized_phases}
    entered_phases.update(reports_by_phase)

    effect_point_values: list[float] = []
    effect_interval_values: list[float] = []
    safety_values: list[float] = []
    phase_validity_values = [observational_phase_validity]
    for phase_id in sorted(entered_phases):
        phase_report = reports_by_phase.get(phase_id)
        phase_quality = None if phase_report is None else phase_report.analysis_quality
        if phase_id in primary_effect_phases:
            if phase_quality is None:
                effect_point_values.append(0.0)
                effect_interval_values.append(0.0)
            else:
                if (
                    not phase_quality.randomized_primary_effect_eligible
                    or phase_quality.randomized_primary_effect_point_agreement is None
                    or phase_quality.randomized_primary_effect_interval_agreement is None
                ):
                    raise ValueError(f"TrialDev phase lacks eligible typed efficacy components: {phase_id!r}.")
                effect_point_values.append(float(phase_quality.randomized_primary_effect_point_agreement))
                effect_interval_values.append(float(phase_quality.randomized_primary_effect_interval_agreement))
        elif phase_quality is not None and phase_quality.randomized_primary_effect_eligible:
            raise ValueError("TrialDev phase1 must not be eligible for a randomized primary-effect endpoint.")

        if phase_quality is None:
            safety_values.append(0.0)
            phase_validity_values.append(0.0)
        else:
            if not phase_quality.safety_evidence_eligible or phase_quality.safety_evidence_agreement is None:
                raise ValueError(f"TrialDev randomized phase lacks eligible typed safety evidence: {phase_id!r}.")
            safety_values.append(float(phase_quality.safety_evidence_agreement))
            phase_validity_values.append(float(phase_quality.phase_evaluation_valid))

    return TrialDevProgrammeAnalysisQualityV1(
        observational_analysis_validity=_quality_endpoint([observational_validity]),
        observational_analysis_score=_quality_endpoint([observational_score]),
        randomized_primary_effect_point_agreement=_quality_endpoint(effect_point_values),
        randomized_primary_effect_interval_agreement=_quality_endpoint(effect_interval_values),
        safety_evidence_agreement=_quality_endpoint(safety_values),
        phase_evaluation_validity=_quality_endpoint(phase_validity_values),
    )


def _report_is_invalid(report: TrialDevGradeRecordV1) -> bool:
    """Return whether a reached checkpoint has an invalid grading artifact."""

    payload = report.payload
    if not isinstance(payload, dict):
        raise ValueError("TrialDev grade payload must be an object.")
    validity = payload.get("validity")
    if not isinstance(validity, dict) or not isinstance(validity.get("valid"), bool):
        raise ValueError("TrialDev grade payload requires typed validity state.")
    return not validity["valid"]


def _observational_terminal_stop(report: TrialDevGradeRecordV1 | None) -> bool:
    """Return whether reached observational evidence supports non-nomination."""

    if report is None or _report_is_invalid(report):
        return False
    matches = [
        record
        for record in report.lane_scores
        if record.phase_id == "observational_review" and record.lane_id == "asset_nomination"
    ]
    if len(matches) != 1:
        raise ValueError("TrialDev observational grade requires one asset-nomination lane record.")
    return bool(matches[0].submitted_target_id == "withhold_nomination" and matches[0].score == 1.0)


def _checkpoint_outcomes(
    *,
    observational_report: TrialDevGradeRecordV1 | None,
    trajectory_grade: TrialDevTrajectoryGradeV1 | None,
    phase_policy_modes: Mapping[str, str],
) -> tuple[TrialDevCheckpointOutcomeV1, ...]:
    """Project reached, failed, and structurally absent programme checkpoints."""

    phase_reports = () if trajectory_grade is None else tuple(trajectory_grade.phase_reports)
    reports_by_phase = {str(report.phase_id): report for report in phase_reports}
    if len(reports_by_phase) != len(phase_reports):
        raise ValueError("TrialDev trajectory contains duplicate phase reports.")
    cumulative: float | None = None
    outcomes: list[TrialDevCheckpointOutcomeV1] = []

    def append_score(phase_id: str, report: TrialDevGradeRecordV1 | None) -> None:
        nonlocal cumulative
        invalid = report is None or _report_is_invalid(report)
        conditional = 0.0 if report is None else float(report.primary_score)
        cumulative = conditional if cumulative is None else min(cumulative, conditional)
        outcomes.append(
            TrialDevCheckpointOutcomeV1(
                phase_id=phase_id,
                status="missing_or_invalid" if invalid else "reached",
                required_lane_ids=required_trialdev_lanes_v1(phase_id),
                conditional_score=conditional,
                cumulative_score=cumulative,
            )
        )

    append_score("observational_review", observational_report)
    invalid_reached = observational_report is None or _report_is_invalid(observational_report)
    observational_stop = trajectory_grade is None and _observational_terminal_stop(observational_report)
    if trajectory_grade is None and not observational_stop:
        invalid_reached = True
    for phase_id in ("phase1", "phase2", "phase3"):
        report = reports_by_phase.get(phase_id)
        if report is not None:
            append_score(phase_id, report)
            invalid_reached = invalid_reached or _report_is_invalid(report)
            continue
        mode = str(phase_policy_modes.get(phase_id, ""))
        if mode == "not_available":
            status = "not_scheduled"
        elif invalid_reached or (trajectory_grade is not None and trajectory_grade.terminal_status == "invalid"):
            status = "not_reached_after_invalid"
        else:
            status = "structural_not_reached"
        outcomes.append(
            TrialDevCheckpointOutcomeV1(
                phase_id=phase_id,
                status=status,
                required_lane_ids=required_trialdev_lanes_v1(phase_id),
            )
        )

    if trajectory_grade is None:
        final_score = 0.0 if observational_report is None else float(observational_report.program_score)
        final_invalid = not observational_stop
    else:
        final_score = min((float(record.score) for record in trajectory_grade.final_lane_scores), default=0.0)
        final_invalid = trajectory_grade.terminal_status == "invalid" or len(trajectory_grade.final_lane_scores) != 2
    cumulative = final_score if cumulative is None else min(cumulative, final_score)
    outcomes.append(
        TrialDevCheckpointOutcomeV1(
            phase_id="final_decision",
            status="missing_or_invalid" if final_invalid else "reached",
            required_lane_ids=required_trialdev_lanes_v1("final_decision"),
            conditional_score=0.0 if final_invalid else final_score,
            cumulative_score=0.0 if final_invalid else cumulative,
        )
    )
    return tuple(outcomes)


def trajectory_metrics_from_grade(
    *,
    trajectory_grade: TrialDevTrajectoryGradeV1 | None,
    observational_report: TrialDevGradeRecordV1 | None,
    phase_policy_modes: Mapping[str, str],
    analysis_quality: TrialDevProgrammeAnalysisQualityV1,
) -> TrialDevTrajectoryMetricsV1:
    """Project one complete programme grade into its run-summary metrics."""

    checkpoint_outcomes = _checkpoint_outcomes(
        observational_report=observational_report,
        trajectory_grade=trajectory_grade,
        phase_policy_modes=phase_policy_modes,
    )
    resources = (
        derive_programme_resource_consequence_v1(tuple())
        if trajectory_grade is None
        else trajectory_grade.resource_consequence
    )
    resource_summary = TrialDevProgrammeResourceSummaryV1(
        phase_count=len(resources.phases),
        total_participants=resources.total_participants,
        total_protocol_follow_up_days=resources.total_protocol_follow_up_days,
        total_enrollment_window_days=resources.total_enrollment_window_days,
        total_site_phase_budget=resources.total_site_phase_budget,
        total_planned_phase_duration_days=resources.total_planned_phase_duration_days,
        total_participant_follow_up_days=resources.total_participant_follow_up_days,
        participant_excess_vs_minimum=resources.participant_excess_vs_minimum,
        participant_shortage_vs_minimum=resources.participant_shortage_vs_minimum,
        follow_up_excess_days_vs_minimum=resources.follow_up_excess_days_vs_minimum,
        follow_up_shortage_days_vs_minimum=resources.follow_up_shortage_days_vs_minimum,
        statistically_inadequate_phases=resources.statistically_inadequate_phases,
        operationally_infeasible_phases=resources.operationally_infeasible_phases,
        dominated_phases=resources.dominated_phases,
        design_avoidable_participants_min=resources.design_avoidable_participants_min,
        design_avoidable_participants_max=resources.design_avoidable_participants_max,
        design_avoidable_follow_up_days_min=resources.design_avoidable_follow_up_days_min,
        design_avoidable_follow_up_days_max=resources.design_avoidable_follow_up_days_max,
        design_avoidable_participant_follow_up_days_min=(resources.design_avoidable_participant_follow_up_days_min),
        design_avoidable_participant_follow_up_days_max=(resources.design_avoidable_participant_follow_up_days_max),
        late_continuation_participants=resources.late_continuation_participants,
        late_continuation_protocol_follow_up_days=resources.late_continuation_protocol_follow_up_days,
        late_continuation_enrollment_window_days=resources.late_continuation_enrollment_window_days,
        late_continuation_site_phase_budget=resources.late_continuation_site_phase_budget,
        late_continuation_participant_follow_up_days=resources.late_continuation_participant_follow_up_days,
        cost_status=resources.cost_status,
    )
    return TrialDevTrajectoryMetricsV1(
        trajectory_primary_score=(None if trajectory_grade is None else trajectory_grade.trajectory_primary_score),
        programme_primary_score=min(
            record.conditional_score for record in checkpoint_outcomes if record.conditional_score is not None
        ),
        checkpoint_outcomes=checkpoint_outcomes,
        trajectory_decision_score=(
            (None if observational_report is None else observational_report.program_score)
            if trajectory_grade is None
            else trajectory_grade.trajectory_decision_score
        ),
        decision_regret_by_phase=({} if trajectory_grade is None else dict(trajectory_grade.decision_regret_by_phase)),
        n_invalid_attempts=0 if trajectory_grade is None else trajectory_grade.n_invalid_attempts,
        invalid_attempt_reasons=([] if trajectory_grade is None else list(trajectory_grade.invalid_attempt_reasons)),
        resource_summary=resource_summary,
        analysis_quality=analysis_quality,
    )


__all__ = [
    "phase_policy_modes_from_manifest",
    "summarise_programme_analysis_quality",
    "trajectory_metrics_from_grade",
    "wrap_grade_record",
    "wrap_lane_score_record",
    "wrap_trajectory_grade",
]
