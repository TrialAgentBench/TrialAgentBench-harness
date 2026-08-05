"""Denominator-preserving TrialDev decision-performance contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.core.config import ReasoningEffortV1
from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevCheckpointIdV1,
    TrialDevCheckpointOutcomeV1,
    TrialDevStreamIdV1,
)
from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TRIALDEV_ANALYSIS_CLASSIFICATIONS_V1,
    TrialDevAnalysisClassificationV1,
    TrialDevScientificAssessmentV1,
)

TrialDevCapabilityIdV1 = Literal[
    "evidence_validity",
    "identification_and_uncertainty",
    "policy_compatibility",
    "safety",
    "temporal_coherence",
    "workflow_execution",
    "action_admissibility",
]
TrialDevMetricLaneIdV1 = Literal[
    "asset_nomination",
    "phase_design",
    "phase_analysis",
    "safety_gate",
    "decision_action",
    "route_timing",
    "final_recommendation",
    "portfolio_allocation",
    "resource_feasibility",
]
TrialDevCheckpointMetricStatusV1 = Literal[
    "reached",
    "structural_nonreach",
    "model_noncompletion",
    "infrastructure_failure",
]
TrialDevLaneOutcomeV1 = Literal["accepted", "invalid", "missing"]
TrialDevCapabilityOutcomeV1 = Literal["passed", "failed", "not_applicable"]
TrialDevCapabilityCheckIdV1 = Literal[
    "evidence_integrity",
    "method_eligibility",
    "identification_status",
    "uncertainty_qualification",
    "policy_conclusion_compatibility",
    "safety_evidence",
    "transition_legality",
    "history_immutability",
    "required_output_presence",
    "workflow_completion",
    "selected_action_membership",
]
TrialDevProgrammeExecutionStatusV1 = Literal[
    "completed",
    "model_noncompletion",
    "infrastructure_failure",
]
TrialDevSwitchTimingV1 = Literal["none", "early", "late"]

TRIALDEV_CAPABILITY_IDS_V1: tuple[TrialDevCapabilityIdV1, ...] = (
    "evidence_validity",
    "identification_and_uncertainty",
    "policy_compatibility",
    "safety",
    "temporal_coherence",
    "workflow_execution",
    "action_admissibility",
)
TRIALDEV_CAPABILITY_CHECKS_V1: dict[TrialDevCapabilityIdV1, tuple[TrialDevCapabilityCheckIdV1, ...]] = {
    "evidence_validity": ("evidence_integrity", "method_eligibility"),
    "identification_and_uncertainty": (
        "identification_status",
        "uncertainty_qualification",
    ),
    "policy_compatibility": ("policy_conclusion_compatibility",),
    "safety": ("safety_evidence",),
    "temporal_coherence": ("transition_legality", "history_immutability"),
    "workflow_execution": ("required_output_presence", "workflow_completion"),
    "action_admissibility": ("selected_action_membership",),
}
TRIALDEV_CHECKPOINT_INVENTORY_V1: dict[TrialDevStreamIdV1, tuple[TrialDevCheckpointIdV1, ...]] = {
    "single_asset_development": (
        "observational_review",
        "early_safety_study",
        "proof_of_concept",
        "confirmation",
    ),
    "bounded_portfolio_reallocation": (
        "observational_review",
        "joint_early_study_review",
        "lead_proof_of_concept_review",
        "promoted_reserve_proof_of_concept_review",
        "confirmation",
    ),
}
TRIALDEV_REQUIRED_LANES_V1: dict[
    tuple[TrialDevStreamIdV1, TrialDevCheckpointIdV1], tuple[TrialDevMetricLaneIdV1, ...]
] = {
    ("single_asset_development", "observational_review"): (
        "asset_nomination",
        "phase_analysis",
    ),
    ("single_asset_development", "early_safety_study"): (
        "phase_design",
        "phase_analysis",
        "safety_gate",
        "decision_action",
    ),
    ("single_asset_development", "proof_of_concept"): (
        "phase_design",
        "phase_analysis",
        "decision_action",
    ),
    ("single_asset_development", "confirmation"): (
        "phase_design",
        "phase_analysis",
        "decision_action",
    ),
    ("bounded_portfolio_reallocation", "observational_review"): (
        "asset_nomination",
        "phase_analysis",
        "portfolio_allocation",
        "resource_feasibility",
    ),
    ("bounded_portfolio_reallocation", "joint_early_study_review"): (
        "phase_design",
        "phase_analysis",
        "safety_gate",
        "decision_action",
        "portfolio_allocation",
        "resource_feasibility",
    ),
    ("bounded_portfolio_reallocation", "lead_proof_of_concept_review"): (
        "phase_design",
        "phase_analysis",
        "decision_action",
        "portfolio_allocation",
        "resource_feasibility",
    ),
    ("bounded_portfolio_reallocation", "promoted_reserve_proof_of_concept_review"): (
        "phase_design",
        "phase_analysis",
        "decision_action",
        "portfolio_allocation",
        "resource_feasibility",
    ),
    ("bounded_portfolio_reallocation", "confirmation"): (
        "phase_design",
        "phase_analysis",
        "decision_action",
        "portfolio_allocation",
        "resource_feasibility",
    ),
}
TRIALDEV_TERMINAL_LANES_V1: tuple[TrialDevMetricLaneIdV1, ...] = (
    "route_timing",
    "final_recommendation",
)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevCapabilityCheckV1(_StrictFrozenModel):
    """One atomic, independently reconstructable capability check."""

    check_id: TrialDevCapabilityCheckIdV1
    passed: bool
    source_record_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class TrialDevCapabilityAssessmentV1(_StrictFrozenModel):
    """One capability derived from its complete prespecified check set."""

    capability_id: TrialDevCapabilityIdV1
    outcome: TrialDevCapabilityOutcomeV1
    checks: tuple[TrialDevCapabilityCheckV1, ...] = ()

    @model_validator(mode="after")
    def validate_derivation(self) -> Self:
        """Require applicable outcomes to equal the conjunction of canonical checks."""

        expected = TRIALDEV_CAPABILITY_CHECKS_V1[self.capability_id]
        observed = tuple(item.check_id for item in self.checks)
        if self.outcome == "not_applicable":
            if self.checks:
                raise ValueError("A non-applicable capability cannot carry check results.")
            return self
        if len(observed) != len(set(observed)) or set(observed) != set(expected):
            raise ValueError("An applicable capability requires its complete canonical check set.")
        derived = "passed" if all(item.passed for item in self.checks) else "failed"
        if self.outcome != derived:
            raise ValueError("Capability outcome must equal the conjunction of its check results.")
        return self


class TrialDevLaneAssessmentV1(_StrictFrozenModel):
    """One required lane outcome at a reached checkpoint."""

    lane_id: TrialDevMetricLaneIdV1
    outcome: TrialDevLaneOutcomeV1
    source_record_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class TrialDevCheckpointAssessmentV1(_StrictFrozenModel):
    """One checkpoint's reach, submission, capability, and terminal record."""

    checkpoint_id: TrialDevCheckpointIdV1
    outcome: TrialDevCheckpointOutcomeV1
    lanes: tuple[TrialDevLaneAssessmentV1, ...] = ()
    capabilities: tuple[TrialDevCapabilityAssessmentV1, ...] = ()
    scientific_assessment: TrialDevScientificAssessmentV1 | None = None
    terminal_record_valid: bool | None = None

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Self:
        """Require complete responsibilities exactly at reached checkpoints."""

        lane_ids = tuple(item.lane_id for item in self.lanes)
        capability_ids = tuple(item.capability_id for item in self.capabilities)
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("TrialDev checkpoint lane assessments must be unique.")
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("TrialDev checkpoint capability assessments must be unique.")
        if self.outcome.reach_status == "reached" and self.outcome.execution_status == "completed":
            if not self.lanes:
                raise ValueError("A completed reached TrialDev checkpoint requires lane assessments.")
            if set(capability_ids) != set(TRIALDEV_CAPABILITY_IDS_V1):
                raise ValueError("A completed reached TrialDev checkpoint requires the complete capability vector.")
            if self.scientific_assessment is None:
                raise ValueError("A completed reached TrialDev checkpoint requires its scientific assessment.")
            if self.scientific_assessment.execution == "not_assessed":
                raise ValueError("A completed reached TrialDev checkpoint requires an assessed execution outcome.")
        elif self.outcome.execution_status == "model_noncompletion":
            if not self.lanes or any(lane.outcome != "missing" for lane in self.lanes):
                raise ValueError("Model noncompletion requires explicit missing lane assessments.")
            if self.capabilities or self.scientific_assessment is not None or self.terminal_record_valid is not None:
                raise ValueError("Model noncompletion cannot carry capability or terminal outcomes.")
        elif (
            self.lanes
            or self.capabilities
            or self.scientific_assessment is not None
            or self.terminal_record_valid is not None
        ):
            raise ValueError("An uncompleted checkpoint cannot carry grade or terminal outcomes.")
        return self

    @property
    def status(self) -> TrialDevCheckpointMetricStatusV1:
        """Return the legacy summary projection from the orthogonal outcome axes."""

        if self.outcome.execution_status == "model_noncompletion":
            return "model_noncompletion"
        if self.outcome.execution_status == "infrastructure_failure":
            return "infrastructure_failure"
        if self.outcome.reach_status == "structural_nonreach":
            return "structural_nonreach"
        return "reached"


class TrialDevSecondaryOutcomesV1(_StrictFrozenModel):
    """Named costs and consequences that cannot alter primary grades."""

    elapsed_seconds: float = Field(0.0, ge=0.0)
    provider_calls: int = Field(0, ge=0)
    agent_turns: int = Field(0, ge=0)
    correction_count: int = Field(0, ge=0)
    execute_code_calls: int = Field(0, ge=0)
    inspect_data_calls: int = Field(0, ge=0)
    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    provider_reported_usd: float | None = Field(default=None, ge=0.0)
    programme_resource_units: int | None = Field(default=None, ge=0)
    switch_count: int = Field(0, ge=0, le=1)
    switch_timing: TrialDevSwitchTimingV1 = "none"
    downstream_consequence: float | None = None
    policy_value: float | None = None
    policy_regret: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_secondary_outcomes(self) -> Self:
        """Require token and switch accounting to be internally coherent."""

        if self.switch_count == 0 and self.switch_timing != "none":
            raise ValueError("A programme without a switch must use switch_timing='none'.")
        if self.switch_count == 1 and self.switch_timing == "none":
            raise ValueError("A programme with a switch requires early or late timing.")
        return self


class TrialDevProgrammeAssessmentV1(_StrictFrozenModel):
    """Complete evaluation record for one model and programme-policy view."""

    schema_id: Literal["trialagentbench.trialdev_programme_assessment/v1"] = (
        "trialagentbench.trialdev_programme_assessment/v1"
    )
    model_id: str = Field(..., min_length=1)
    condition_id: str = Field(..., min_length=1)
    request_replicate_id: str = Field(..., min_length=1)
    reasoning_effort: ReasoningEffortV1 | None = None
    procedure_assistance: Literal[
        "output_contract_only",
        "unordered_checklist",
        "ordered_sop",
    ]
    maximum_turns_per_step: int = Field(..., ge=1)
    maximum_submission_attempts: int = Field(..., ge=1)
    tool_choice: Literal["auto", "required"] = "auto"
    task_materialization_seed: int
    release_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    grader_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    evaluation_unit_id: str = Field(..., min_length=1)
    programme_id: str = Field(..., min_length=1)
    scenario_family_id: str = Field(..., min_length=1)
    objective_variant_id: str = Field(..., min_length=1)
    policy_variant_id: str = Field(..., min_length=1)
    stream_id: TrialDevStreamIdV1
    execution_status: TrialDevProgrammeExecutionStatusV1
    checkpoints: tuple[TrialDevCheckpointAssessmentV1, ...] = Field(..., min_length=1)
    secondary_outcomes: TrialDevSecondaryOutcomesV1 = Field(default_factory=TrialDevSecondaryOutcomesV1)

    @model_validator(mode="after")
    def validate_programme(self) -> Self:
        """Require one coherent terminal or typed failure trajectory."""

        checkpoint_ids = tuple(item.checkpoint_id for item in self.checkpoints)
        if checkpoint_ids != TRIALDEV_CHECKPOINT_INVENTORY_V1[self.stream_id]:
            raise ValueError("TrialDev programme assessments require the complete ordered checkpoint inventory.")
        if self.checkpoints[0].status == "structural_nonreach":
            raise ValueError("The initial TrialDev checkpoint cannot be structurally unreached.")
        model_failures = sum(item.status == "model_noncompletion" for item in self.checkpoints)
        infrastructure_failures = sum(item.status == "infrastructure_failure" for item in self.checkpoints)
        terminal_records = tuple(
            item.terminal_record_valid for item in self.checkpoints if item.terminal_record_valid is not None
        )
        expected_model_failures = int(self.execution_status == "model_noncompletion")
        expected_infrastructure_failures = int(self.execution_status == "infrastructure_failure")
        if model_failures != expected_model_failures or infrastructure_failures != expected_infrastructure_failures:
            raise ValueError("Programme execution status must match exactly one typed checkpoint failure.")
        if self.execution_status == "completed" and len(terminal_records) != 1:
            raise ValueError("A completed programme requires exactly one terminal record assessment.")
        if self.execution_status != "completed" and terminal_records:
            raise ValueError("A noncompleted programme cannot report a terminal record assessment.")
        reached_indices = tuple(index for index, item in enumerate(self.checkpoints) if item.status == "reached")
        terminal_indices = tuple(
            index for index, item in enumerate(self.checkpoints) if item.terminal_record_valid is not None
        )
        if terminal_indices and terminal_indices[0] != reached_indices[-1]:
            raise ValueError("A terminal record assessment belongs on the last reached checkpoint.")
        failure_indices = tuple(
            index
            for index, item in enumerate(self.checkpoints)
            if item.status in {"model_noncompletion", "infrastructure_failure"}
        )
        if failure_indices and any(
            item.status != "structural_nonreach" for item in self.checkpoints[failure_indices[0] + 1 :]
        ):
            raise ValueError("Every checkpoint after an execution failure must be structurally unreached.")
        for checkpoint in self.checkpoints:
            if checkpoint.outcome.execution_status not in {"completed", "model_noncompletion"}:
                continue
            required_lanes = set(TRIALDEV_REQUIRED_LANES_V1[(self.stream_id, checkpoint.checkpoint_id)])
            if checkpoint.terminal_record_valid is not None:
                required_lanes.update(TRIALDEV_TERMINAL_LANES_V1)
            if {lane.lane_id for lane in checkpoint.lanes} != required_lanes:
                raise ValueError("A reached checkpoint requires its exact canonical lane set.")
        if self.stream_id == "single_asset_development" and any(
            lane.lane_id in {"portfolio_allocation", "resource_feasibility"}
            for checkpoint in self.checkpoints
            for lane in checkpoint.lanes
        ):
            raise ValueError("Single-asset programmes cannot contain portfolio-only lanes.")
        return self


class TrialDevAssessmentPortfolioV1(_StrictFrozenModel):
    """Input portfolio for one or more model evaluations."""

    schema_id: Literal["trialagentbench.trialdev_assessment_portfolio/v1"] = (
        "trialagentbench.trialdev_assessment_portfolio/v1"
    )
    programmes: tuple[TrialDevProgrammeAssessmentV1, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_unique_views(self) -> Self:
        """Reject duplicate model and evaluation-view records."""

        keys = tuple(
            (
                item.condition_id,
                item.request_replicate_id,
                item.stream_id,
                item.evaluation_unit_id,
                item.objective_variant_id,
                item.policy_variant_id,
            )
            for item in self.programmes
        )
        if len(keys) != len(set(keys)):
            raise ValueError("TrialDev assessment portfolio contains duplicate evaluation views.")
        return self


class TrialDevClusterIntervalV1(_StrictFrozenModel):
    """Scenario-family bootstrap interval for one finite estimate."""

    confidence_level: float = Field(..., gt=0.5, lt=1.0)
    resamples: int = Field(..., ge=1)
    seed: int = Field(..., ge=0)
    cluster_count: int = Field(..., ge=2)
    lower: float
    upper: float


class TrialDevRateMetricV1(_StrictFrozenModel):
    """Exact finite count and optional scenario-cluster uncertainty."""

    numerator: int = Field(..., ge=0)
    denominator: int = Field(..., ge=0)
    finite_estimate: float | None = Field(default=None, ge=0.0, le=1.0)
    cluster_interval: TrialDevClusterIntervalV1 | None = None

    @model_validator(mode="after")
    def validate_rate(self) -> Self:
        """Bind the estimate exactly to its declared counts."""

        if self.numerator > self.denominator:
            raise ValueError("TrialDev rate numerator cannot exceed its denominator.")
        expected = None if self.denominator == 0 else self.numerator / self.denominator
        if self.finite_estimate != expected:
            raise ValueError("TrialDev finite estimate must equal numerator divided by denominator.")
        if self.cluster_interval is not None and self.denominator == 0:
            raise ValueError("An inestimable TrialDev rate cannot carry an interval.")
        return self


class TrialDevDenominatorCountsV1(_StrictFrozenModel):
    """Non-overlapping checkpoint, lane, and programme accounting."""

    programmes: int = Field(..., ge=0)
    scheduled: int = Field(..., ge=0)
    reached: int = Field(..., ge=0)
    structural_nonreach: int = Field(..., ge=0)
    submitted: int = Field(..., ge=0)
    accepted: int = Field(..., ge=0)
    invalid: int = Field(..., ge=0)
    missing: int = Field(..., ge=0)
    model_noncompletion: int = Field(..., ge=0)
    infrastructure_failure: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Require exhaustive scheduled and lane-submission partitions."""

        if self.scheduled != self.reached + self.model_noncompletion + self.infrastructure_failure:
            raise ValueError("Scheduled checkpoints must partition into reached and typed failures.")
        if self.submitted != self.accepted + self.invalid:
            raise ValueError("Submitted lanes must partition into accepted and invalid outcomes.")
        return self


class TrialDevNamedRateV1(_StrictFrozenModel):
    """One named lane or capability estimate."""

    metric_id: str = Field(..., min_length=1)
    estimate: TrialDevRateMetricV1


class TrialDevAnalysisClassificationCountV1(_StrictFrozenModel):
    """Frequency of one analysis class among scientifically assessed checkpoints."""

    classification: TrialDevAnalysisClassificationV1
    count: int = Field(..., ge=0)
    denominator: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_count(self) -> Self:
        """Keep each classification count within its shared denominator."""

        if self.count > self.denominator:
            raise ValueError("Analysis-classification count cannot exceed its denominator.")
        return self


class TrialDevSecondarySummaryV1(_StrictFrozenModel):
    """Descriptive programme costs and consequences with explicit availability."""

    programme_count: int = Field(..., ge=0)
    elapsed_seconds_mean: float | None = Field(default=None, ge=0.0)
    provider_calls_mean: float | None = Field(default=None, ge=0.0)
    agent_turns_mean: float | None = Field(default=None, ge=0.0)
    correction_count_mean: float | None = Field(default=None, ge=0.0)
    execute_code_calls_mean: float | None = Field(default=None, ge=0.0)
    inspect_data_calls_mean: float | None = Field(default=None, ge=0.0)
    total_tokens_mean: float | None = Field(default=None, ge=0.0)
    provider_reported_usd_available: int = Field(..., ge=0)
    provider_reported_usd_mean: float | None = Field(default=None, ge=0.0)
    programme_resource_units_available: int = Field(..., ge=0)
    programme_resource_units_mean: float | None = Field(default=None, ge=0.0)
    switch_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    early_switch_count: int = Field(..., ge=0)
    late_switch_count: int = Field(..., ge=0)
    downstream_consequence_available: int = Field(..., ge=0)
    downstream_consequence_mean: float | None = None
    policy_value_available: int = Field(..., ge=0)
    policy_value_mean: float | None = None
    policy_regret_available: int = Field(..., ge=0)
    policy_regret_mean: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        """Bind each optional mean to its explicit availability denominator."""

        optional_measures = (
            (self.provider_reported_usd_available, self.provider_reported_usd_mean),
            (self.programme_resource_units_available, self.programme_resource_units_mean),
            (self.downstream_consequence_available, self.downstream_consequence_mean),
            (self.policy_value_available, self.policy_value_mean),
            (self.policy_regret_available, self.policy_regret_mean),
        )
        if any(available > self.programme_count for available, _ in optional_measures):
            raise ValueError("Secondary outcome availability cannot exceed the programme count.")
        if any((available == 0) != (mean is None) for available, mean in optional_measures):
            raise ValueError("A secondary outcome mean exists exactly when observations are available.")
        if self.early_switch_count + self.late_switch_count > self.programme_count:
            raise ValueError("Switch timing counts cannot exceed the programme count.")
        return self


class TrialDevStreamMetricSummaryV1(_StrictFrozenModel):
    """Separated TrialDev metrics for one model and stream."""

    model_id: str = Field(..., min_length=1)
    condition_id: str = Field(..., min_length=1)
    request_replicate_id: str = Field(..., min_length=1)
    reasoning_effort: ReasoningEffortV1 | None = None
    procedure_assistance: Literal[
        "output_contract_only",
        "unordered_checklist",
        "ordered_sop",
    ]
    maximum_turns_per_step: int = Field(..., ge=1)
    maximum_submission_attempts: int = Field(..., ge=1)
    tool_choice: Literal["auto", "required"] = "auto"
    task_materialization_seed: int
    stream_id: TrialDevStreamIdV1
    denominators: TrialDevDenominatorCountsV1
    capabilities: tuple[TrialDevNamedRateV1, ...]
    scientific_responsibilities: tuple[TrialDevNamedRateV1, ...]
    analysis_classifications: tuple[TrialDevAnalysisClassificationCountV1, ...]
    lanes: tuple[TrialDevNamedRateV1, ...]
    checkpoint_success: TrialDevRateMetricV1
    complete_chain_success: TrialDevRateMetricV1
    execution_completion: TrialDevRateMetricV1
    secondary: TrialDevSecondarySummaryV1

    @model_validator(mode="after")
    def validate_named_metrics(self) -> Self:
        """Require complete capability identities and unique lane identities."""

        capability_ids = tuple(item.metric_id for item in self.capabilities)
        lane_ids = tuple(item.metric_id for item in self.lanes)
        responsibility_ids = tuple(item.metric_id for item in self.scientific_responsibilities)
        if set(capability_ids) != set(TRIALDEV_CAPABILITY_IDS_V1):
            raise ValueError("TrialDev stream summary requires the complete capability vector.")
        if (
            len(capability_ids) != len(set(capability_ids))
            or len(lane_ids) != len(set(lane_ids))
            or len(responsibility_ids) != len(set(responsibility_ids))
        ):
            raise ValueError("TrialDev named metrics must be unique.")
        classifications = tuple(item.classification for item in self.analysis_classifications)
        if classifications != TRIALDEV_ANALYSIS_CLASSIFICATIONS_V1:
            raise ValueError("TrialDev analysis classifications require the complete canonical order.")
        denominators = {item.denominator for item in self.analysis_classifications}
        if len(denominators) != 1 or sum(item.count for item in self.analysis_classifications) != next(
            iter(denominators)
        ):
            raise ValueError("TrialDev analysis classifications must partition their denominator.")
        return self


class TrialDevMetricPortfolioV1(_StrictFrozenModel):
    """Public TrialDev result surface with streams kept separate."""

    schema_id: Literal["trialagentbench.trialdev_metric_portfolio/v1"] = "trialagentbench.trialdev_metric_portfolio/v1"
    streams: tuple[TrialDevStreamMetricSummaryV1, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_streams(self) -> Self:
        """Reject duplicate model-stream summaries and an overall scalar."""

        keys = tuple(
            (item.condition_id, item.request_replicate_id, item.task_materialization_seed, item.stream_id)
            for item in self.streams
        )
        if len(keys) != len(set(keys)):
            raise ValueError("TrialDev metric portfolio contains duplicate model-stream summaries.")
        return self


class TrialDevPairedDifferenceV1(_StrictFrozenModel):
    """Paired model difference clustered by scenario family."""

    metric_id: str = Field(..., min_length=1)
    pair_count: int = Field(..., ge=1)
    scenario_family_count: int = Field(..., ge=1)
    reference_mean: float = Field(..., ge=0.0, le=1.0)
    intervention_mean: float = Field(..., ge=0.0, le=1.0)
    paired_difference: float = Field(..., ge=-1.0, le=1.0)
    cluster_interval: TrialDevClusterIntervalV1 | None = None


class TrialDevStreamComparisonV1(_StrictFrozenModel):
    """Paired capability and complete-chain differences for one stream."""

    stream_id: TrialDevStreamIdV1
    metrics: tuple[TrialDevPairedDifferenceV1, ...] = Field(..., min_length=1)


class TrialDevConditionComparisonV1(_StrictFrozenModel):
    """Condition comparison that preserves shared scenario-family dependence."""

    schema_id: Literal["trialagentbench.trialdev_condition_comparison/v1"] = (
        "trialagentbench.trialdev_condition_comparison/v1"
    )
    reference_condition_id: str = Field(..., min_length=1)
    intervention_condition_id: str = Field(..., min_length=1)
    streams: tuple[TrialDevStreamComparisonV1, ...] = Field(..., min_length=1)


class TrialDevCalibrationArmV1(_StrictFrozenModel):
    """One paired turn-and-assistance calibration arm."""

    condition_id: str = Field(..., min_length=1)
    procedure_assistance: Literal[
        "output_contract_only",
        "unordered_checklist",
        "ordered_sop",
    ]
    maximum_turns_per_step: int = Field(..., ge=1)
    maximum_submission_attempts: int = Field(..., ge=1)
    tool_choice: Literal["auto", "required"] = "auto"
    programme_count: int = Field(..., ge=1)
    completed_programmes: int = Field(..., ge=0)
    complete_chain_success_rate: float = Field(..., ge=0.0, le=1.0)
    scheduled_checkpoint_count: int = Field(..., ge=1)
    checkpoint_success_rate: float = Field(..., ge=0.0, le=1.0)
    correction_count_mean: float = Field(..., ge=0.0)
    agent_turns_mean: float = Field(..., ge=0.0)
    elapsed_seconds_mean: float = Field(..., ge=0.0)
    total_tokens_mean: float = Field(..., ge=0.0)
    provider_reported_usd_mean: float = Field(..., ge=0.0)
    dominated_by_condition_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Require completed programmes to remain within the paired denominator."""

        if self.completed_programmes > self.programme_count:
            raise ValueError("Completed calibration programmes cannot exceed the arm denominator.")
        return self


class TrialDevCalibrationSelectionV1(_StrictFrozenModel):
    """Pareto and lexicographic disposition of a paired calibration panel."""

    schema_id: Literal["trialagentbench.trialdev_calibration_selection/v1"] = (
        "trialagentbench.trialdev_calibration_selection/v1"
    )
    model_id: str = Field(..., min_length=1)
    reasoning_effort: ReasoningEffortV1 | None = None
    request_replicate_id: str = Field(..., min_length=1)
    task_materialization_seed: int
    stream_id: TrialDevStreamIdV1
    evaluation_unit_ids: tuple[str, ...] = Field(..., min_length=1)
    arms: tuple[TrialDevCalibrationArmV1, ...] = Field(..., min_length=2)
    pareto_condition_ids: tuple[str, ...] = Field(..., min_length=1)
    selected_condition_id: str = Field(..., min_length=1)
    selection_rule: Literal[
        "select_clean_auto_condition_then_maximize_quality_then_minimize_turn_ceiling_and_observed_resources"
    ] = "select_clean_auto_condition_then_maximize_quality_then_minimize_turn_ceiling_and_observed_resources"

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Bind the selected and Pareto identities to unique observed arms."""

        condition_ids = tuple(arm.condition_id for arm in self.arms)
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("Calibration condition identities must be unique.")
        if not set(self.pareto_condition_ids).issubset(condition_ids):
            raise ValueError("Every Pareto calibration condition must identify an observed arm.")
        if self.selected_condition_id not in self.pareto_condition_ids:
            raise ValueError("The selected calibration condition must belong to the Pareto set.")
        selected = next(arm for arm in self.arms if arm.condition_id == self.selected_condition_id)
        if selected.procedure_assistance != "output_contract_only" or selected.tool_choice != "auto":
            raise ValueError("Calibration selection must retain the clean automatic-tool condition.")
        if len(self.evaluation_unit_ids) != len(set(self.evaluation_unit_ids)):
            raise ValueError("Calibration evaluation-unit identities must be unique.")
        return self


__all__ = [
    "TRIALDEV_CAPABILITY_IDS_V1",
    "TRIALDEV_CAPABILITY_CHECKS_V1",
    "TRIALDEV_CHECKPOINT_INVENTORY_V1",
    "TRIALDEV_REQUIRED_LANES_V1",
    "TRIALDEV_TERMINAL_LANES_V1",
    "TrialDevAssessmentPortfolioV1",
    "TrialDevAnalysisClassificationCountV1",
    "TrialDevCapabilityAssessmentV1",
    "TrialDevCapabilityCheckV1",
    "TrialDevCapabilityIdV1",
    "TrialDevCalibrationArmV1",
    "TrialDevCalibrationSelectionV1",
    "TrialDevCheckpointAssessmentV1",
    "TrialDevClusterIntervalV1",
    "TrialDevDenominatorCountsV1",
    "TrialDevLaneAssessmentV1",
    "TrialDevMetricLaneIdV1",
    "TrialDevMetricPortfolioV1",
    "TrialDevConditionComparisonV1",
    "TrialDevNamedRateV1",
    "TrialDevPairedDifferenceV1",
    "TrialDevProgrammeAssessmentV1",
    "TrialDevRateMetricV1",
    "TrialDevSecondaryOutcomesV1",
    "TrialDevSecondarySummaryV1",
    "TrialDevStreamComparisonV1",
    "TrialDevStreamMetricSummaryV1",
]
