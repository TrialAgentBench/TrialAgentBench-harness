"""Submission and grading models for the TrialDev grader."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TrialDevScientificAssessmentV1,
    unassessed_scientific_assessment_v1,
)
from trialagentbench_harness.trialdev.grading.hashing import compute_sha256_hex
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentAnalysisDiagnosticV1,
    TrialDevelopmentCandidateUtilityEstimateV1,
    TrialDevelopmentEffectEstimateV1,
    TrialDevelopmentIdentificationEvidenceV1,
    TrialDevelopmentSafetyEstimateV1,
)

__all__ = [
    "TrialDevelopmentAnalysisDiagnosticV1",
    "TrialDevelopmentAnalysisQualityV1",
    "TrialDevelopmentAnalysisReportV1",
    "TrialDevelopmentAuditGateReportV1",
    "TrialDevelopmentCandidateEligibilityRecordV1",
    "TrialDevDesignEfficiencyV1",
    "TrialDevDesignFrontierPointV1",
    "TrialDevPhaseResourceConsequenceV1",
    "TrialDevProgrammeResourceConsequenceV1",
    "TrialDevelopmentGradeReportV1",
    "TrialDevelopmentGradeGateRecordV1",
    "TrialDevelopmentInvalidAttemptReportV1",
    "TrialDevelopmentLaneScoreRecordV1",
    "TrialDevelopmentProgramDecisionV1",
    "TrialDevelopmentRequestV1",
    "TrialDevelopmentScoringContextV1",
    "TrialDevelopmentSubmissionV1",
    "TrialDevelopmentTrajectoryReplayReportV1",
    "TrialDevelopmentValidityReportV1",
]

TrialDevelopmentGradeGateIdV1 = Literal[
    "submission",
    "question",
    "route",
    "evidence",
    "integrity",
    "result",
    "conformance",
    "decision",
]
TRIALDEV_GRADE_GATE_ORDER_V1: tuple[TrialDevelopmentGradeGateIdV1, ...] = (
    "submission",
    "question",
    "route",
    "evidence",
    "integrity",
    "result",
    "conformance",
    "decision",
)


class TrialDevelopmentGradeGateRecordV1(BaseModel):
    """One step in the ordered TrialDev grading cascade."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_id: TrialDevelopmentGradeGateIdV1
    status: Literal["passed", "failed", "not_reached", "not_applicable"]
    failure_code: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_failure(self) -> TrialDevelopmentGradeGateRecordV1:
        """Require a failure code exactly for a failed gate."""

        if (self.status == "failed") != (self.failure_code is not None):
            raise ValueError("failure_code is required exactly for a failed TrialDev grade gate.")
        return self


class TrialDevelopmentAnalysisReportV1(BaseModel):
    """Structured evaluation output consumed by the grader."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_winner_drug_id: str | None = None
    response_branch: Literal["estimable", "qualified_non_nomination"] | None = None
    primary_resolution_evidence_class: (
        Literal[
            "empirical_diagnosis",
            "design_or_provenance_reasoning",
            "evidence_insufficient",
        ]
        | None
    ) = None
    ranked_drug_ids: tuple[str, ...] = Field(default_factory=tuple)
    candidate_utility_estimates: tuple[TrialDevelopmentCandidateUtilityEstimateV1, ...] = Field(default_factory=tuple)
    identification_evidence: tuple[TrialDevelopmentIdentificationEvidenceV1, ...] = Field(default_factory=tuple)
    primary_effect: TrialDevelopmentEffectEstimateV1 | None = None
    safety_estimate: TrialDevelopmentSafetyEstimateV1 | None = None
    claimed_subgroup_variables: tuple[str, ...] = Field(default_factory=tuple)
    evidence_summary: str | None = None
    diagnostic_artifacts: tuple[TrialDevelopmentAnalysisDiagnosticV1, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_evidence_ids(self) -> TrialDevelopmentAnalysisReportV1:
        evidence_ids = {artifact.artifact_id for artifact in self.diagnostic_artifacts}
        declared_count = len(evidence_ids)
        for evidence in self.identification_evidence:
            evidence_ids.add(evidence.evidence_id)
            declared_count += 1
        utility_candidates: set[str] = set()
        diagnostic_ids = {artifact.artifact_id for artifact in self.diagnostic_artifacts}
        for estimate in self.candidate_utility_estimates:
            evidence_ids.add(estimate.evidence_id)
            declared_count += 1
            utility_candidates.add(estimate.candidate_drug_id)
            missing = sorted(set(estimate.diagnostic_evidence_ids) - diagnostic_ids)
            if missing:
                raise ValueError(f"candidate utility estimate references unknown diagnostic evidence: {missing!r}")
        if len(utility_candidates) != len(self.candidate_utility_estimates):
            raise ValueError("candidate utility estimates must be unique by candidate_drug_id")
        for value in (
            None if self.primary_effect is None else self.primary_effect.evidence_id,
            None if self.safety_estimate is None else self.safety_estimate.evidence_id,
        ):
            if value is not None:
                evidence_ids.add(value)
                declared_count += 1
        if len(evidence_ids) != declared_count:
            raise ValueError("analysis evidence IDs must be unique")
        if self.primary_effect is not None:
            diagnostic_ids = {artifact.artifact_id for artifact in self.diagnostic_artifacts}
            missing = sorted(set(self.primary_effect.diagnostic_evidence_ids) - diagnostic_ids)
            if missing:
                raise ValueError(f"primary_effect references unknown diagnostic evidence: {missing!r}")
        return self

    def evidence_ids(self) -> frozenset[str]:
        """Return scoreable evidence IDs declared by this analysis."""

        values = {artifact.artifact_id for artifact in self.diagnostic_artifacts}
        values.update(evidence.evidence_id for evidence in self.identification_evidence)
        values.update(estimate.evidence_id for estimate in self.candidate_utility_estimates)
        if self.primary_effect is not None:
            values.add(self.primary_effect.evidence_id)
        if self.safety_estimate is not None:
            values.add(self.safety_estimate.evidence_id)
        return frozenset(values)


class TrialDevelopmentProgramDecisionV1(BaseModel):
    """Program-level decision output consumed by the grader."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective_id: str | None = None
    decision_action: str | None = None
    recommended_drug_id: str | None = None
    supporting_evidence_ids: tuple[str, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_supporting_evidence_ids(self) -> TrialDevelopmentProgramDecisionV1:
        if len(set(self.supporting_evidence_ids)) != len(self.supporting_evidence_ids):
            raise ValueError("supporting_evidence_ids must be unique")
        return self


class TrialDevelopmentSubmissionV1(BaseModel):
    """Complete structured submission passed to the standalone grader."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    request: TrialDevelopmentRequestV1
    analysis_report: TrialDevelopmentAnalysisReportV1
    program_decision: TrialDevelopmentProgramDecisionV1

    @model_validator(mode="after")
    def _validate_decision_evidence(self) -> TrialDevelopmentSubmissionV1:
        phase_id = str(self.request.phase_id)
        estimates = {
            str(estimate.candidate_drug_id): estimate for estimate in self.analysis_report.candidate_utility_estimates
        }
        request_candidates = {str(value) for value in self.request.candidate_drug_ids}
        if phase_id == "observational_review":
            branch = self.analysis_report.response_branch
            resolution_class = self.analysis_report.primary_resolution_evidence_class
            if branch is None or resolution_class is None:
                raise ValueError("observational review requires a response branch and resolution evidence class")
            ranking = tuple(str(value) for value in self.analysis_report.ranked_drug_ids)
            action = str(self.program_decision.decision_action or "")
            selected = self.program_decision.recommended_drug_id
            if branch == "estimable":
                if resolution_class != "empirical_diagnosis":
                    raise ValueError("estimable observational responses require empirical_diagnosis")
                if set(estimates) != request_candidates:
                    raise ValueError("observational candidate utility estimates must cover every requested candidate")
                if len(ranking) != len(set(ranking)) or set(ranking) != request_candidates:
                    raise ValueError("observational ranking must be a complete candidate permutation")
                if self.analysis_report.identification_evidence:
                    raise ValueError("estimable observational responses cannot include non-nomination evidence")
                if action == "nominate_for_early_study":
                    if selected is None or str(selected) not in request_candidates:
                        raise ValueError("observational nomination must recommend one requested candidate")
                elif action == "withhold_nomination":
                    if selected is not None:
                        raise ValueError("observational stop must not recommend a candidate")
                else:
                    raise ValueError(
                        "observational decision_action must be nominate_for_early_study or withhold_nomination"
                    )
                cited_evidence = {
                    estimate.evidence_id for estimate in self.analysis_report.candidate_utility_estimates
                }
            else:
                if estimates or ranking:
                    raise ValueError("qualified non-nomination cannot contain candidate estimates or a ranking")
                if selected is not None or action != "withhold_nomination":
                    raise ValueError("qualified non-nomination must decline nomination without a candidate")
                if not self.analysis_report.identification_evidence:
                    raise ValueError("qualified non-nomination requires identification or support evidence")
                cited_evidence = {evidence.evidence_id for evidence in self.analysis_report.identification_evidence}
            analysis_selected = self.analysis_report.selected_winner_drug_id
            if analysis_selected is not None and str(analysis_selected) != str(selected):
                raise ValueError("analysis and program decision selected candidates must agree")
            if not set(self.program_decision.supporting_evidence_ids) & cited_evidence:
                raise ValueError("observational decision must cite evidence from its response branch")
        elif estimates:
            raise ValueError("candidate utility estimates are only valid for observational_review")
        unknown = sorted(set(self.program_decision.supporting_evidence_ids) - set(self.analysis_report.evidence_ids()))
        if unknown:
            raise ValueError(f"program decision references unknown analysis evidence: {unknown!r}")
        return self


class TrialDevelopmentValidityReportV1(BaseModel):
    """Validity status for a grading report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool = True
    invalid_reasons: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class TrialDevelopmentAuditGateReportV1(BaseModel):
    """Diagnostic-only alignment score.

    The score is not an primary gate and does not alter ``primary_score``.
    It measures whether a scored trajectory is supported by sufficient
    evaluation/evidence detail.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    gates_triggered: tuple[str, ...] = Field(default_factory=tuple)
    diagnostic_alignment_score: float | None = Field(default=None, ge=0.0, le=1.0)


class TrialDevelopmentAnalysisQualityV1(BaseModel):
    """Typed, noncompensatory analysis-quality cascade for one phase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_analysis_quality_v1"] = "trialdev_analysis_quality_v1"
    observational_analysis_eligible: bool
    observational_analysis_valid: bool | None
    observational_analysis_score: float | None = Field(default=None, ge=0.0, le=1.0)
    randomized_primary_effect_eligible: bool
    randomized_primary_effect_valid: bool | None
    randomized_primary_effect_point_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    randomized_primary_effect_interval_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    safety_evidence_eligible: bool
    safety_evidence_valid: bool | None
    safety_evidence_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    phase_evaluation_valid: bool

    @model_validator(mode="after")
    def _validate_applicability(self) -> TrialDevelopmentAnalysisQualityV1:
        groups = (
            (
                "observational_analysis",
                self.observational_analysis_eligible,
                (
                    self.observational_analysis_valid,
                    self.observational_analysis_score,
                ),
            ),
            (
                "randomized_primary_effect",
                self.randomized_primary_effect_eligible,
                (
                    self.randomized_primary_effect_valid,
                    self.randomized_primary_effect_point_agreement,
                    self.randomized_primary_effect_interval_agreement,
                ),
            ),
            (
                "safety_evidence",
                self.safety_evidence_eligible,
                (
                    self.safety_evidence_valid,
                    self.safety_evidence_agreement,
                ),
            ),
        )
        for name, eligible, values in groups:
            if eligible and any(value is None for value in values):
                raise ValueError(f"{name} requires validity and score fields when eligible.")
            if not eligible and any(value is not None for value in values):
                raise ValueError(f"{name} fields must be null when structurally not applicable.")
        required_validity = tuple(bool(values[0]) for _, eligible, values in groups if eligible)
        if self.phase_evaluation_valid != all(required_validity):
            raise ValueError("phase_evaluation_valid must equal the conjunction of eligible component-validity gates.")
        return self


class TrialDevDesignFrontierPointV1(BaseModel):
    """One nondominated, publicly reproducible statistically adequate design."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_sample_size: int = Field(..., ge=1)
    follow_up_days: int = Field(..., ge=1)
    allocation_ratio: str
    achieved_power: float | None = Field(default=None, ge=0.0, le=1.0)
    achieved_safety_absolute_risk_power: float = Field(..., ge=0.0, le=1.0)
    achieved_safety_excess_risk_power: float = Field(..., ge=0.0, le=1.0)


class TrialDevDesignEfficiencyV1(BaseModel):
    """Non-compensatory comparison with the public statistical frontier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_scope: Literal["statistical_frontier_with_separate_public_operational_feasibility"] = (
        "statistical_frontier_with_separate_public_operational_feasibility"
    )
    statistically_adequate: bool
    operationally_feasible: bool
    design_valid: bool
    on_frontier: bool
    dominated_by_frontier: bool
    operational_support: int = Field(..., ge=0)
    operational_headroom: int = Field(..., ge=0)
    operational_shortage: int = Field(..., ge=0)
    minimum_frontier_participants: int = Field(..., ge=1)
    minimum_frontier_follow_up_days: int = Field(..., ge=1)
    participant_excess_vs_minimum: int = Field(..., ge=0)
    participant_shortage_vs_minimum: int = Field(..., ge=0)
    follow_up_excess_days_vs_minimum: int = Field(..., ge=0)
    follow_up_shortage_days_vs_minimum: int = Field(..., ge=0)
    achieved_power: float | None = Field(default=None, ge=0.0, le=1.0)
    target_power: float | None = Field(default=None, ge=0.0, le=1.0)
    achieved_safety_absolute_risk_power: float = Field(..., ge=0.0, le=1.0)
    achieved_safety_excess_risk_power: float = Field(..., ge=0.0, le=1.0)
    target_safety_decision_power: float = Field(..., ge=0.0, le=1.0)
    frontier: tuple[TrialDevDesignFrontierPointV1, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_decomposition(self) -> TrialDevDesignEfficiencyV1:
        """Require canonical order and mutually exclusive signed deviations."""

        keys = tuple(
            (point.target_sample_size, point.follow_up_days, point.allocation_ratio) for point in self.frontier
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("Design frontier points must be unique and canonically ordered.")
        if self.participant_excess_vs_minimum and self.participant_shortage_vs_minimum:
            raise ValueError("Participant excess and shortage cannot both be positive.")
        if self.follow_up_excess_days_vs_minimum and self.follow_up_shortage_days_vs_minimum:
            raise ValueError("Follow-up excess and shortage cannot both be positive.")
        if self.operational_headroom and self.operational_shortage:
            raise ValueError("Operational headroom and shortage cannot both be positive.")
        if self.design_valid != (self.statistically_adequate and self.operationally_feasible):
            raise ValueError("Design validity must require statistical adequacy and operational feasibility.")
        if not self.design_valid and (self.on_frontier or self.dominated_by_frontier):
            raise ValueError("Invalid designs cannot receive frontier or dominance credit.")
        if (self.achieved_power is None) != (self.target_power is None):
            raise ValueError("Achieved and target efficacy power must be present together.")
        return self


class TrialDevPhaseResourceConsequenceV1(BaseModel):
    """Vector-valued consequence of one materialized randomized-phase design."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase_id: Literal["phase1", "phase2", "phase3"]
    request_checksum: str = Field(..., min_length=64, max_length=64)
    target_sample_size: int = Field(..., ge=1)
    follow_up_days: int = Field(..., ge=1)
    enrollment_window_days: int = Field(..., ge=1)
    site_count_budget: int = Field(..., ge=1)
    participant_follow_up_days: int = Field(..., ge=1)
    statistically_adequate: bool
    operationally_feasible: bool
    design_status: Literal[
        "statistically_inadequate",
        "operationally_infeasible",
        "valid_dominated",
        "valid_frontier",
        "valid_nondominated",
    ]
    operational_support: int = Field(..., ge=0)
    operational_headroom: int = Field(..., ge=0)
    operational_shortage: int = Field(..., ge=0)
    achieved_power: float | None = Field(default=None, ge=0.0, le=1.0)
    target_power: float | None = Field(default=None, ge=0.0, le=1.0)
    achieved_safety_absolute_risk_power: float = Field(..., ge=0.0, le=1.0)
    achieved_safety_excess_risk_power: float = Field(..., ge=0.0, le=1.0)
    target_safety_decision_power: float = Field(..., ge=0.0, le=1.0)
    participant_excess_vs_minimum: int = Field(..., ge=0)
    participant_shortage_vs_minimum: int = Field(..., ge=0)
    follow_up_excess_days_vs_minimum: int = Field(..., ge=0)
    follow_up_shortage_days_vs_minimum: int = Field(..., ge=0)
    dominating_frontier: tuple[TrialDevDesignFrontierPointV1, ...] = ()
    avoidable_participants_min: int = Field(..., ge=0)
    avoidable_participants_max: int = Field(..., ge=0)
    avoidable_follow_up_days_min: int = Field(..., ge=0)
    avoidable_follow_up_days_max: int = Field(..., ge=0)
    avoidable_participant_follow_up_days_min: int = Field(..., ge=0)
    avoidable_participant_follow_up_days_max: int = Field(..., ge=0)
    entered_after_unsupported_advance: bool

    @model_validator(mode="after")
    def validate_resource_decomposition(self) -> TrialDevPhaseResourceConsequenceV1:
        """Require exact products, ordered ranges, and dominance semantics."""

        if self.participant_follow_up_days != self.target_sample_size * self.follow_up_days:
            raise ValueError("Phase participant-follow-up days must equal N multiplied by follow-up.")
        if (self.achieved_power is None) != (self.target_power is None):
            raise ValueError("Phase achieved and target efficacy power must be present together.")
        if self.participant_excess_vs_minimum and self.participant_shortage_vs_minimum:
            raise ValueError("Phase participant excess and shortage cannot both be positive.")
        if self.follow_up_excess_days_vs_minimum and self.follow_up_shortage_days_vs_minimum:
            raise ValueError("Phase follow-up excess and shortage cannot both be positive.")
        if self.operational_headroom and self.operational_shortage:
            raise ValueError("Phase operational headroom and shortage cannot both be positive.")
        if self.operational_headroom != max(0, self.operational_support - self.target_sample_size):
            raise ValueError("Phase operational headroom must equal support minus requested participants.")
        if self.operational_shortage != max(0, self.target_sample_size - self.operational_support):
            raise ValueError("Phase operational shortage must equal requested participants minus support.")
        if self.operationally_feasible != (self.operational_shortage == 0):
            raise ValueError("Phase operational feasibility must match the exact public support shortage.")
        if not self.statistically_adequate and self.design_status != "statistically_inadequate":
            raise ValueError("Statistically inadequate phases require the corresponding design status.")
        if (
            self.statistically_adequate
            and not self.operationally_feasible
            and self.design_status != "operationally_infeasible"
        ):
            raise ValueError("Operationally infeasible status requires statistical adequacy and a support shortage.")
        if (
            self.statistically_adequate
            and self.operationally_feasible
            and self.design_status
            in {
                "statistically_inadequate",
                "operationally_infeasible",
            }
        ):
            raise ValueError("Adequate feasible phases require a valid frontier status.")
        ranges = (
            (self.avoidable_participants_min, self.avoidable_participants_max),
            (self.avoidable_follow_up_days_min, self.avoidable_follow_up_days_max),
            (
                self.avoidable_participant_follow_up_days_min,
                self.avoidable_participant_follow_up_days_max,
            ),
        )
        if any(lower > upper for lower, upper in ranges):
            raise ValueError("Phase avoidable-resource ranges must be ordered.")
        has_dominator = bool(self.dominating_frontier)
        if (self.design_status == "valid_dominated") != has_dominator:
            raise ValueError("Only valid dominated designs may declare a dominating frontier.")
        if not has_dominator and any(value for pair in ranges for value in pair):
            raise ValueError("Nondominated or inadequate designs cannot claim avoidable design resources.")
        return self


class TrialDevProgrammeResourceConsequenceV1(BaseModel):
    """Cumulative observed and defensibly avoidable programme resource vectors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phases: tuple[TrialDevPhaseResourceConsequenceV1, ...] = ()
    total_participants: int = Field(..., ge=0)
    total_protocol_follow_up_days: int = Field(..., ge=0)
    total_enrollment_window_days: int = Field(..., ge=0)
    total_site_phase_budget: int = Field(..., ge=0)
    total_planned_phase_duration_days: int = Field(..., ge=0)
    total_participant_follow_up_days: int = Field(..., ge=0)
    participant_excess_vs_minimum: int = Field(..., ge=0)
    participant_shortage_vs_minimum: int = Field(..., ge=0)
    follow_up_excess_days_vs_minimum: int = Field(..., ge=0)
    follow_up_shortage_days_vs_minimum: int = Field(..., ge=0)
    statistically_inadequate_phases: int = Field(..., ge=0)
    operationally_infeasible_phases: int = Field(..., ge=0)
    dominated_phases: int = Field(..., ge=0)
    design_avoidable_participants_min: int = Field(..., ge=0)
    design_avoidable_participants_max: int = Field(..., ge=0)
    design_avoidable_follow_up_days_min: int = Field(..., ge=0)
    design_avoidable_follow_up_days_max: int = Field(..., ge=0)
    design_avoidable_participant_follow_up_days_min: int = Field(..., ge=0)
    design_avoidable_participant_follow_up_days_max: int = Field(..., ge=0)
    late_continuation_participants: int = Field(..., ge=0)
    late_continuation_protocol_follow_up_days: int = Field(..., ge=0)
    late_continuation_enrollment_window_days: int = Field(..., ge=0)
    late_continuation_site_phase_budget: int = Field(..., ge=0)
    late_continuation_participant_follow_up_days: int = Field(..., ge=0)
    cost_status: Literal["not_available_without_public_cost_schedule"] = "not_available_without_public_cost_schedule"

    @model_validator(mode="after")
    def validate_programme_totals(self) -> TrialDevProgrammeResourceConsequenceV1:
        """Require every programme total to replay exactly from phase rows."""

        phases = self.phases
        expected = {
            "total_participants": sum(row.target_sample_size for row in phases),
            "total_protocol_follow_up_days": sum(row.follow_up_days for row in phases),
            "total_enrollment_window_days": sum(row.enrollment_window_days for row in phases),
            "total_site_phase_budget": sum(row.site_count_budget for row in phases),
            "total_planned_phase_duration_days": sum(
                row.enrollment_window_days + row.follow_up_days for row in phases
            ),
            "total_participant_follow_up_days": sum(row.participant_follow_up_days for row in phases),
            "participant_excess_vs_minimum": sum(row.participant_excess_vs_minimum for row in phases),
            "participant_shortage_vs_minimum": sum(row.participant_shortage_vs_minimum for row in phases),
            "follow_up_excess_days_vs_minimum": sum(row.follow_up_excess_days_vs_minimum for row in phases),
            "follow_up_shortage_days_vs_minimum": sum(row.follow_up_shortage_days_vs_minimum for row in phases),
            "statistically_inadequate_phases": sum(not row.statistically_adequate for row in phases),
            "operationally_infeasible_phases": sum(not row.operationally_feasible for row in phases),
            "dominated_phases": sum(row.design_status == "valid_dominated" for row in phases),
            "design_avoidable_participants_min": sum(row.avoidable_participants_min for row in phases),
            "design_avoidable_participants_max": sum(row.avoidable_participants_max for row in phases),
            "design_avoidable_follow_up_days_min": sum(row.avoidable_follow_up_days_min for row in phases),
            "design_avoidable_follow_up_days_max": sum(row.avoidable_follow_up_days_max for row in phases),
            "design_avoidable_participant_follow_up_days_min": sum(
                row.avoidable_participant_follow_up_days_min for row in phases
            ),
            "design_avoidable_participant_follow_up_days_max": sum(
                row.avoidable_participant_follow_up_days_max for row in phases
            ),
            "late_continuation_participants": sum(
                row.target_sample_size for row in phases if row.entered_after_unsupported_advance
            ),
            "late_continuation_protocol_follow_up_days": sum(
                row.follow_up_days for row in phases if row.entered_after_unsupported_advance
            ),
            "late_continuation_enrollment_window_days": sum(
                row.enrollment_window_days for row in phases if row.entered_after_unsupported_advance
            ),
            "late_continuation_site_phase_budget": sum(
                row.site_count_budget for row in phases if row.entered_after_unsupported_advance
            ),
            "late_continuation_participant_follow_up_days": sum(
                row.participant_follow_up_days for row in phases if row.entered_after_unsupported_advance
            ),
        }
        observed = self.model_dump(mode="python", include=set(expected))
        if observed != expected:
            raise ValueError("Programme resource totals do not replay from their phase rows.")
        return self


class TrialDevelopmentCandidateEligibilityRecordV1(BaseModel):
    """Public candidate-membership and phase-eligibility result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_candidate_eligibility_record_v1"] = Field("trialdev_candidate_eligibility_record_v1")
    schema_version: Literal[1] = 1
    scenario_id: str = Field(..., min_length=1)
    phase_id: str = Field(..., min_length=1)
    candidate_drug_id: str = Field(..., min_length=1)
    catalog_member: bool
    sequentially_eligible: bool
    eligibility_source: Literal[
        "public_candidate_catalog",
        "eval_contract_current_state",
        "materialized_program_state",
    ]
    failure_reason: Literal[
        "not_applicable",
        "not_in_public_catalog",
        "not_in_current_eligible_set",
        "program_already_terminal",
    ] = "not_applicable"

    @property
    def failure_reason_detail(self) -> str | None:
        """Return the concrete score invalidation reason, if any."""

        if self.failure_reason == "not_in_public_catalog":
            return f"unknown_candidate:{self.candidate_drug_id}"
        if self.failure_reason == "not_in_current_eligible_set":
            return f"ineligible_candidate:{self.candidate_drug_id}"
        if self.failure_reason == "program_already_terminal":
            return f"program_already_terminal:{self.phase_id}"
        return None


class TrialDevelopmentLaneScoreRecordV1(BaseModel):
    """One scoreable evaluation-target register lane row for TrialDev grading."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_lane_score_record_v1"] = Field("trialdev_lane_score_record_v1")
    schema_version: Literal[1] = 1
    scenario_id: str = Field(..., min_length=1)
    program_id: str | None = None
    phase_id: str = Field(..., min_length=1)
    program_objective_id: str = Field(..., min_length=1)
    phase_scoring_objective_id: str = Field(..., min_length=1)
    lane_id: Literal[
        "asset_nomination",
        "phase_design",
        "phase_analysis",
        "decision_action",
        "route_timing",
        "final_recommendation",
        "safety_gate",
    ]
    evaluation_target_checksum: str = Field(..., min_length=64, max_length=64)
    scoring_policy_id: str = Field(..., min_length=1)
    recoverability_policy_id: str = Field(..., min_length=1)
    submitted_target_id: str | None = None
    reference_target_ids: tuple[str, ...] = Field(..., min_length=1)
    credit_eligible_target_ids: tuple[str, ...] = Field(default_factory=tuple)
    score: float = Field(..., ge=0.0, le=1.0)
    score_derivation: Literal[
        "literal_target",
        "numeric_diagnostic",
        "public_evidence_action",
    ] = "literal_target"
    derived_from_trajectory_metric: bool = False
    terminal_action_observed: str | None = None
    terminal_asset_observed: str | None = None
    terminal_phase_observed: str | None = None
    status: Literal[
        "scored",
        "credit_eligible_alternative",
        "invalid_submission_zeroed",
        "missing_submission_zeroed",
        "not_applicable",
    ]
    artifact_status: Literal["present", "missing", "invalid"]
    missing_reason: str | None = None
    failure_reason: str | None = None
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash(self) -> TrialDevelopmentLaneScoreRecordV1:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialDevelopmentScoringContextV1(BaseModel):
    """Evaluator-held scoring context for one TrialDev program trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    scenario_id: str = Field(..., min_length=1)
    program_id: str = Field(..., min_length=1)
    program_objective_id: str = Field(..., min_length=1)
    phase_scoring_objectives: dict[str, str] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_context(self) -> TrialDevelopmentScoringContextV1:
        phase_objectives = {str(k): str(v) for k, v in self.phase_scoring_objectives.items() if str(k) and str(v)}
        for phase_id in ("phase1", "phase2"):
            if phase_id not in phase_objectives:
                raise ValueError(f"scoring context missing required phase objective: {phase_id}")
        object.__setattr__(self, "phase_scoring_objectives", phase_objectives)
        return self


class TrialDevelopmentGradeReportV1(BaseModel):
    """Deterministic grading report for one submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    schema_id: Literal["trialdev_grade_report_v1"] = Field("trialdev_grade_report_v1")
    scenario_id: str = Field(..., min_length=1)
    phase_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    program_objective_id: str = Field(..., min_length=1)
    phase_scoring_objective_id: str = Field(..., min_length=1)
    primary_score: float = Field(..., ge=0.0, le=1.0)
    design_score: float = Field(..., ge=0.0, le=1.0)
    evaluation_score: float = Field(..., ge=0.0, le=1.0)
    program_score: float = Field(..., ge=0.0, le=1.0)
    ranking_score: float = Field(..., ge=0.0, le=1.0)
    analysis_quality: TrialDevelopmentAnalysisQualityV1
    scientific_assessment: TrialDevScientificAssessmentV1 = Field(default_factory=unassessed_scientific_assessment_v1)
    lane_status: dict[str, Literal["active", "not_applicable", "invalid"]] = Field(default_factory=dict)
    active_lane_scores: dict[str, float] = Field(default_factory=dict)
    gates: tuple[TrialDevelopmentGradeGateRecordV1, ...] = Field(min_length=8, max_length=8)
    first_failure_gate: TrialDevelopmentGradeGateIdV1 | None = None
    validity: TrialDevelopmentValidityReportV1 = Field(default_factory=TrialDevelopmentValidityReportV1)
    audit_gates: TrialDevelopmentAuditGateReportV1 = Field(default_factory=TrialDevelopmentAuditGateReportV1)
    design_efficiency: TrialDevDesignEfficiencyV1 | None = None
    policy_reference_regret: float | None = None
    in_set_regret: float | None = None
    selected_winner_drug_id: str | None = None
    best_candidate_drug_id: str | None = None
    feasibility_failures: tuple[str, ...] = Field(default_factory=tuple)
    lane_breakdown: dict[str, float] = Field(default_factory=dict)
    lane_scores: tuple[TrialDevelopmentLaneScoreRecordV1, ...] = Field(default_factory=tuple)
    payload: dict[str, object] = Field(default_factory=dict)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash(self) -> TrialDevelopmentGradeReportV1:
        if tuple(record.gate_id for record in self.gates) != TRIALDEV_GRADE_GATE_ORDER_V1:
            raise ValueError("TrialDev grade gates must use the canonical ordered eight-gate cascade.")
        failures = tuple(record for record in self.gates if record.status == "failed")
        if len(failures) > 1:
            raise ValueError("Only the first failed TrialDev grade gate may be marked failed.")
        expected_first = None if not failures else failures[0].gate_id
        if self.first_failure_gate != expected_first:
            raise ValueError("first_failure_gate must identify the sole failed TrialDev gate.")
        if failures:
            failure_index = TRIALDEV_GRADE_GATE_ORDER_V1.index(failures[0].gate_id)
            if any(record.status != "not_reached" for record in self.gates[failure_index + 1 :]):
                raise ValueError("Every TrialDev gate after the first failure must be not_reached.")
        elif any(record.status == "not_reached" for record in self.gates):
            raise ValueError("A complete TrialDev cascade cannot contain not_reached gates.")
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["lane_breakdown"] = {str(k): float(v) for k, v in sorted(payload.get("lane_breakdown", {}).items())}
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialDevelopmentInvalidAttemptReportV1(BaseModel):
    """Structured report for an invalid sequential workflow attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    schema_id: Literal["trialdev_invalid_attempt_report_v1"] = Field("trialdev_invalid_attempt_report_v1")
    scenario_id: str = Field(..., min_length=1)
    attempted_phase_id: str | None = None
    reason_code: Literal[
        "phase_not_current",
        "phase_not_available",
        "invalid_action",
        "invalid_analysis",
        "invalid_request",
        "missing_materialized_output",
        "checksum_mismatch",
        "duplicate_phase_submission",
        "post_terminal_submission",
        "unknown_candidate",
        "ineligible_candidate",
        "missing_required_phase",
        "unsupported_submission_format",
    ]
    message: str = Field(..., min_length=1)
    primary_score: float = Field(0.0, ge=0.0, le=0.0)
    validity: TrialDevelopmentValidityReportV1
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash(self) -> TrialDevelopmentInvalidAttemptReportV1:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self


class TrialDevelopmentTerminalSummaryV1(BaseModel):
    """Terminal programme state derived from the model's last phase decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(..., min_length=1)
    terminal_status: Literal["active", "stopped", "completed", "invalid"]
    terminal_action: str | None = None
    final_program_success: bool
    recommended_drug_id: str | None = None


class TrialDevelopmentTrajectoryReplayReportV1(BaseModel):
    """Deterministic replay-grade report for a sequential program trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = Field("v1")
    schema_id: Literal["trialdev_trajectory_replay_report_v1"] = Field("trialdev_trajectory_replay_report_v1")
    scenario_id: str = Field(..., min_length=1)
    program_objective_id: str = Field(..., min_length=1)
    phase_scoring_objectives: dict[str, str] = Field(default_factory=dict)
    terminal_status: Literal["active", "stopped", "completed", "invalid"]
    n_phase_submissions: int = Field(..., ge=0)
    n_invalid_attempts: int = Field(..., ge=0)
    trajectory_primary_score: float = Field(..., ge=0.0, le=1.0)
    trajectory_decision_score: float = Field(..., ge=0.0, le=1.0)
    decision_regret_by_phase: dict[str, float] = Field(default_factory=dict)
    mean_scores: dict[str, float] = Field(default_factory=dict)
    terminal_summary: TrialDevelopmentTerminalSummaryV1
    resource_consequence: TrialDevProgrammeResourceConsequenceV1
    phase_reports: tuple[dict[str, object], ...] = Field(default_factory=tuple)
    final_lane_scores: tuple[TrialDevelopmentLaneScoreRecordV1, ...] = Field(default_factory=tuple)
    invalid_attempts: tuple[TrialDevelopmentInvalidAttemptReportV1, ...] = Field(default_factory=tuple)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash(self) -> TrialDevelopmentTrajectoryReplayReportV1:
        if self.trajectory_decision_score not in {0.0, 1.0}:
            raise ValueError("trajectory_decision_score must be an exact binary action-validity score.")
        invalid_regrets = {
            phase_id: value for phase_id, value in self.decision_regret_by_phase.items() if value not in {0.0, 1.0}
        }
        if invalid_regrets:
            raise ValueError(f"decision_regret_by_phase must contain exact binary regrets: {invalid_regrets!r}")
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        payload["decision_regret_by_phase"] = {
            str(k): float(v) for k, v in sorted(payload.get("decision_regret_by_phase", {}).items())
        }
        payload["mean_scores"] = {str(k): float(v) for k, v in sorted(payload.get("mean_scores", {}).items())}
        object.__setattr__(self, "checksum", compute_sha256_hex(payload))
        return self
