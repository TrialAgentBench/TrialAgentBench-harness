"""Contracts for TrialDevBench grader artifacts written under program directories."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TrialDevScientificAssessmentV1,
    unassessed_scientific_assessment_v1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_evaluation_target_register import (
    TrialDevEvaluationLaneV1,
    TrialDevLaneScoreStatusV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_recoverability import (
    TrialDevLaneRecoverabilityPolicyV1,
    TrialDevObjectiveIdV1,
)
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevDesignEfficiencyV1,
    TrialDevelopmentAnalysisQualityV1,
    TrialDevelopmentGradeGateIdV1,
    TrialDevelopmentGradeGateRecordV1,
    TrialDevelopmentValidityReportV1,
    TrialDevProgrammeResourceConsequenceV1,
)


class TrialDevAuditGatesV1(BaseModel):
    """Diagnostic alignment surface from the upstream grader."""

    model_config = ConfigDict(extra="forbid")
    diagnostic_alignment_score: float | None = Field(default=None, ge=0.0, le=1.0)
    gates_triggered: list[str] = Field(default_factory=list)


class TrialDevLaneScoreRecordV1(BaseModel):
    """One scoreable TrialDev evaluation-target register lane row emitted by the grader."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench_trialdev_lane_score_record_v1"] = (
        "trialagentbench_trialdev_lane_score_record_v1"
    )
    schema_version: Literal[1] = 1
    source_run_id: str | None = None
    model: str | None = None
    seed_label: str | None = None
    scenario_id: str
    program_id: str | None = None
    phase_id: str
    program_objective_id: str
    phase_scoring_objective_id: str
    lane_id: TrialDevEvaluationLaneV1
    evaluation_target_checksum: str
    scoring_policy_id: str
    recoverability_policy_id: TrialDevLaneRecoverabilityPolicyV1
    submitted_target_id: str | None = None
    reference_target_ids: tuple[str, ...]
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
    status: TrialDevLaneScoreStatusV1
    artifact_status: Literal["present", "missing", "invalid"]
    missing_reason: str | None = None
    failure_reason: str | None = None
    checksum: str | None = None


class TrialDevGradeRecordV1(BaseModel):
    """Schema-bearing wrapper for a single grade report (obs_review or one phase)."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench_trialdev_grade_record_v1"] = "trialagentbench_trialdev_grade_record_v1"
    schema_version: Literal[1] = 1

    # Stable headline fields used by aggregation/reporting.
    primary_score: float = Field(..., ge=0.0, le=1.0)
    design_score: float = Field(..., ge=0.0, le=1.0)
    evaluation_score: float = Field(..., ge=0.0, le=1.0)
    program_score: float = Field(..., ge=0.0, le=1.0)
    ranking_score: float = Field(..., ge=0.0, le=1.0)
    analysis_quality: TrialDevelopmentAnalysisQualityV1
    scientific_assessment: TrialDevScientificAssessmentV1 = Field(default_factory=unassessed_scientific_assessment_v1)
    gates: tuple[TrialDevelopmentGradeGateRecordV1, ...]
    first_failure_gate: TrialDevelopmentGradeGateIdV1 | None = None
    validity: TrialDevelopmentValidityReportV1

    policy_reference_regret: float | None = None
    in_set_regret: float | None = None

    active_lane_scores: dict[str, float] = Field(default_factory=dict)
    lane_breakdown: dict[str, float] = Field(default_factory=dict)
    lane_status: dict[str, str] = Field(default_factory=dict)
    audit_gates: TrialDevAuditGatesV1 | None = None
    design_efficiency: TrialDevDesignEfficiencyV1 | None = None

    selected_winner_drug_id: str | None = None
    best_candidate_drug_id: str | None = None
    feasibility_failures: list[str] = Field(default_factory=list)

    # Identification fields (present on many, not all, upstream reports).
    phase_id: str | None = None
    scenario_id: str | None = None
    objective_id: str | None = None
    program_objective_id: str | None = None
    phase_scoring_objective_id: str | None = None
    checksum: str | None = None
    lane_scores: list[TrialDevLaneScoreRecordV1] = Field(default_factory=list)

    # Full upstream report payload for audit/replay.
    payload: JsonValue

    @model_validator(mode="after")
    def validate_gate_cascade(self) -> TrialDevGradeRecordV1:
        """Require the public wrapper to preserve the grader's gate result."""

        failures = tuple(gate for gate in self.gates if gate.status == "failed")
        expected = None if not failures else failures[0].gate_id
        if self.first_failure_gate != expected:
            raise ValueError("TrialDev grade first_failure_gate does not match its gate records.")
        return self


class TrialDevAnalysisQualityEndpointV1(BaseModel):
    """One programme analysis endpoint with its phase eligibility denominator."""

    model_config = ConfigDict(extra="forbid")

    eligible_units: int = Field(..., ge=0)
    value: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_denominator(self) -> TrialDevAnalysisQualityEndpointV1:
        """Require a value exactly when at least one phase unit is eligible."""

        if self.eligible_units == 0 and self.value is not None:
            raise ValueError("An ineligible TrialDev endpoint must not carry a value.")
        if self.eligible_units > 0 and self.value is None:
            raise ValueError("An eligible TrialDev endpoint requires a value.")
        return self


class TrialDevProgrammeAnalysisQualityV1(BaseModel):
    """Noncompensatory analysis-quality endpoints for one programme."""

    model_config = ConfigDict(extra="forbid")

    observational_analysis_validity: TrialDevAnalysisQualityEndpointV1
    observational_analysis_score: TrialDevAnalysisQualityEndpointV1
    randomized_primary_effect_point_agreement: TrialDevAnalysisQualityEndpointV1
    randomized_primary_effect_interval_agreement: TrialDevAnalysisQualityEndpointV1
    safety_evidence_agreement: TrialDevAnalysisQualityEndpointV1
    phase_evaluation_validity: TrialDevAnalysisQualityEndpointV1


class TrialDevProgrammeAnalysisRowV1(BaseModel):
    """Canonical programme row derived from persisted run and grader contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_programme_analysis_row/v1"] = (
        "trialagentbench.trialdev_programme_analysis_row/v1"
    )
    model_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    bundle_sha256: str = Field(..., min_length=64, max_length=64)
    scorer_source_sha256: str = Field(..., min_length=64, max_length=64)
    runner_source_sha256: str = Field(..., min_length=64, max_length=64)
    prompt_interface_sha256: str = Field(..., min_length=64, max_length=64)
    staging_source_sha256: str = Field(..., min_length=64, max_length=64)
    seed_variants: int = Field(..., ge=1)
    condition_id: str = Field(..., min_length=1)
    replicate_id: str = Field(..., min_length=1)
    reasoning_effort: str | None = None
    maximum_turns_per_step: int = Field(..., ge=1)
    maximum_submission_attempts: int = Field(..., ge=1)
    tool_choice: Literal["auto", "required"] = "auto"
    task_materialization_seed: int
    program_id: str = Field(..., min_length=1)
    scenario_id: str = Field(..., min_length=1)
    objective_id: TrialDevObjectiveIdV1
    pairing_sha256: str = Field(..., min_length=64, max_length=64)
    procedure_assistance: Literal[
        "output_contract_only",
        "unordered_checklist",
        "ordered_sop",
    ]
    execution_status: Literal["completed", "model_turn_limit", "model_invalid_submission"]
    completed: bool
    trajectory_primary_score: float = Field(..., ge=0.0, le=1.0)
    trajectory_decision_score: float = Field(..., ge=0.0, le=1.0)
    observational_analysis_validity: TrialDevAnalysisQualityEndpointV1
    observational_analysis_score: TrialDevAnalysisQualityEndpointV1
    randomized_primary_effect_point_agreement: TrialDevAnalysisQualityEndpointV1
    randomized_primary_effect_interval_agreement: TrialDevAnalysisQualityEndpointV1
    safety_evidence_agreement: TrialDevAnalysisQualityEndpointV1
    phase_evaluation_validity: TrialDevAnalysisQualityEndpointV1
    programme_design_validity: bool
    programme_design_nondominance: bool
    randomized_phase_count: int = Field(..., ge=0)
    minimum_randomized_efficacy_power: float | None = Field(default=None, ge=0.0, le=1.0)
    maximum_randomized_efficacy_power_shortfall: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_randomized_safety_power: float | None = Field(default=None, ge=0.0, le=1.0)
    maximum_randomized_safety_power_shortfall: float | None = Field(default=None, ge=0.0, le=1.0)
    total_agent_turns: int = Field(..., ge=0)
    total_execute_code_calls: int = Field(..., ge=0)
    total_inspect_parquet_calls: int = Field(..., ge=0)
    provider_response_count: int = Field(..., ge=0)
    provider_responses_with_usage: int = Field(..., ge=0)
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    provider_elapsed_seconds: float = Field(..., ge=0.0)
    wall_seconds_total: float = Field(..., ge=0.0)
    invalid_submission_attempts: int = Field(..., ge=0)
    phase_materialization_calls: int = Field(..., ge=0)
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
    avoidable_participants_min: int = Field(..., ge=0)
    avoidable_participants_max: int = Field(..., ge=0)
    avoidable_follow_up_days_min: int = Field(..., ge=0)
    avoidable_follow_up_days_max: int = Field(..., ge=0)
    avoidable_participant_follow_up_days_min: int = Field(..., ge=0)
    avoidable_participant_follow_up_days_max: int = Field(..., ge=0)
    late_continuation_participants: int = Field(..., ge=0)
    late_continuation_protocol_follow_up_days: int = Field(..., ge=0)
    late_continuation_enrollment_window_days: int = Field(..., ge=0)
    late_continuation_site_phase_budget: int = Field(..., ge=0)
    late_continuation_participant_follow_up_days: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_completion(self) -> TrialDevProgrammeAnalysisRowV1:
        """Require coherent denominator, design, and resource outcomes."""

        if self.completed != (self.execution_status == "completed"):
            raise ValueError("TrialDev completion must agree with execution_status.")
        if not self.completed and (self.trajectory_primary_score != 0.0 or self.trajectory_decision_score != 0.0):
            raise ValueError("TrialDev model noncompletion must retain zero-valued programme endpoints.")
        if self.avoidable_participants_min > self.avoidable_participants_max:
            raise ValueError("TrialDev avoidable-participant bounds must be ordered.")
        if self.avoidable_follow_up_days_min > self.avoidable_follow_up_days_max:
            raise ValueError("TrialDev avoidable follow-up bounds must be ordered.")
        if self.avoidable_participant_follow_up_days_min > self.avoidable_participant_follow_up_days_max:
            raise ValueError("TrialDev avoidable participant-follow-up bounds must be ordered.")
        if self.programme_design_nondominance and not self.programme_design_validity:
            raise ValueError("A programme cannot be design-nondominated unless all entered designs are valid.")
        if self.randomized_phase_count == 0 and (self.programme_design_validity or self.programme_design_nondominance):
            raise ValueError("TrialDev programmes without randomized phases cannot report design success.")
        efficacy = (
            self.minimum_randomized_efficacy_power,
            self.maximum_randomized_efficacy_power_shortfall,
        )
        safety = (
            self.minimum_randomized_safety_power,
            self.maximum_randomized_safety_power_shortfall,
        )
        if (efficacy[0] is None) != (efficacy[1] is None):
            raise ValueError("TrialDev efficacy power and shortfall summaries must be present together.")
        if (safety[0] is None) != (safety[1] is None):
            raise ValueError("TrialDev safety power and shortfall summaries must be present together.")
        if self.randomized_phase_count == 0 and any(value is not None for value in (*efficacy, *safety)):
            raise ValueError("TrialDev programmes without randomized phases cannot report phase power.")
        if self.randomized_phase_count > 0 and any(value is None for value in safety):
            raise ValueError("Every entered randomized phase requires a safety-power summary.")
        if self.provider_responses_with_usage > self.provider_response_count:
            raise ValueError("Provider responses with usage cannot exceed total provider responses.")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("TrialDev total_tokens must equal prompt_tokens plus completion_tokens.")
        if self.provider_response_count == 0 and any(
            (
                self.provider_responses_with_usage,
                self.prompt_tokens,
                self.completion_tokens,
                self.total_tokens,
                self.provider_elapsed_seconds,
            )
        ):
            raise ValueError("TrialDev programmes without provider responses require zero provider telemetry.")
        return self


class TrialDevTerminalSummaryV1(BaseModel):
    """Terminal state derived from the final evidence-linked phase decision."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    terminal_status: Literal["active", "stopped", "completed", "invalid"]
    terminal_action: str | None = None
    final_program_success: bool
    recommended_drug_id: str | None = None


class TrialDevTrajectoryGradeV1(BaseModel):
    """Schema-bearing wrapper for `trajectory_grade.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench_trialdev_trajectory_grade_v1"] = "trialagentbench_trialdev_trajectory_grade_v1"
    schema_version: Literal[1] = 1

    checksum: str | None = None
    terminal_status: str | None = None
    program_objective_id: str | None = None
    phase_scoring_objectives: dict[str, str] = Field(default_factory=dict)

    # Stable headline metrics written by the upstream grader.
    trajectory_primary_score: float = Field(..., ge=0.0, le=1.0)
    trajectory_decision_score: float = Field(..., ge=0.0, le=1.0)
    decision_regret_by_phase: dict[str, float] = Field(default_factory=dict)
    mean_scores: dict[str, float] = Field(default_factory=dict)
    n_invalid_attempts: int = 0
    n_phase_submissions: int = 0
    invalid_attempt_reasons: list[str] = Field(default_factory=list)
    terminal_summary: TrialDevTerminalSummaryV1
    resource_consequence: TrialDevProgrammeResourceConsequenceV1

    phase_reports: list[TrialDevGradeRecordV1] = Field(default_factory=list)
    final_lane_scores: list[TrialDevLaneScoreRecordV1] = Field(default_factory=list)

    # Full upstream report payload for audit/replay.
    payload: JsonValue

    @model_validator(mode="after")
    def validate_decision_scores(self) -> TrialDevTrajectoryGradeV1:
        """Require exact categorical action validity at the public boundary."""

        if self.trajectory_decision_score not in {0.0, 1.0}:
            raise ValueError("trajectory_decision_score must be an exact binary score")
        invalid = {key: value for key, value in self.decision_regret_by_phase.items() if value not in {0.0, 1.0}}
        if invalid:
            raise ValueError(f"decision_regret_by_phase must contain exact binary regrets: {invalid!r}")
        return self


__all__ = [
    "TrialDevAnalysisQualityEndpointV1",
    "TrialDevAuditGatesV1",
    "TrialDevGradeRecordV1",
    "TrialDevLaneScoreRecordV1",
    "TrialDevProgrammeAnalysisQualityV1",
    "TrialDevProgrammeAnalysisRowV1",
    "TrialDevTerminalSummaryV1",
    "TrialDevTrajectoryGradeV1",
]
