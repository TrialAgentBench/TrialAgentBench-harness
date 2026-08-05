"""Contracts for observable benchmark events and evidence-linked features."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.core.trialeval_factors import (
    TrialEvalAnalysisSpecificationV1,
    TrialEvalContextConfigurationV1,
    TrialEvalDataPreparationV1,
    TrialEvalEvidenceFactorsV1,
)
from trialagentbench_harness.contracts.experiments.procedure_assistance import (
    ProcedureAssistanceV1,
    TrialEvalPromptConditionV1,
)
from trialagentbench_harness.contracts.submission import TrialEvalSubmissionV1

BenchmarkNameV1 = Literal["trialeval", "trialdev"]
TraceCoverageStatusV1 = Literal[
    "full_conversation_trace",
    "submission_only_trace",
    "scorer_only_trace",
    "missing_trace",
]
TrialEvalSubmissionTransportV1 = Literal["direct", "file", "not_observed"]
TrialEvalTraceAuthorityV1 = Literal[
    "authoritative_structured",
    "non_authoritative_narrative",
]
TraceEndpointStateV1 = Literal[
    "valid",
    "failed",
    "not_reached_after_stop",
    "not_attempted_noncompletion",
    "score_export_absent",
    "not_scored_by_design",
    "submission_absent",
    "submission_present_score_absent",
    "score_present_submission_absent",
    "source_inconsistent_requires_adjudication",
    "not_scoreable_trace_only",
]
EvidenceCategoryV1 = Literal[
    "observational_extract",
    "baseline_covariates",
    "time_to_event",
    "safety_events",
    "longitudinal_markers",
    "randomization_or_treatment_assignment",
    "missingness_or_data_quality",
    "public_catalog",
    "data_dictionary_or_schema",
    "protocol_or_program_contract",
    "trial_design_request",
    "trial_population_table",
    "trial_output_summary",
    "analysis_or_submission_workfile",
    "cost_or_budget",
    "phase_state_or_prior_decision",
    "simulator_output_public_summary",
    "scratch_or_diagnostic_file",
    "scratch_schema_dump",
    "scratch_summary_table",
    "scratch_model_result",
    "scratch_survival_result",
    "scratch_ci_or_uncertainty_result",
    "scratch_required_fields_or_contract_copy",
    "unclassified_public_scratch",
    "absolute_or_run_internal_path",
    "shell_literal_or_pseudo_path",
    "protected_reference_or_grader_artifact",
]
SemanticActionFeatureNameV1 = Literal[
    "confounding_mentioned",
    "confounding_adjustment_performed",
    "balance_or_overlap_reported",
    "ph_assumption_mentioned",
    "ph_diagnostic_performed",
    "censoring_mentioned",
    "censoring_adjustment_performed",
    "missingness_mentioned",
    "missingness_handling_performed",
    "uncertainty_interval_reported",
    "sensitivity_analysis_mentioned",
    "sensitivity_analysis_performed",
    "safety_tradeoff_reported",
    "cost_tradeoff_reported",
    "objective_alignment_reported",
]
SemanticActionFeatureEvidenceV1 = Literal[
    "structured_submission_field",
    "executed_code_path",
    "not_observed",
]


class ModelActionTraceEventV1(BaseModel):
    """One observable model or harness action in a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.action_trace_event/v1"] = "trialagentbench.action_trace_event/v1"
    event_id: str = Field(min_length=1)
    timestamp: datetime
    benchmark: BenchmarkNameV1
    model_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    assignment_id: str | None = None
    program_id: str | None = None
    scenario_id: str | None = None
    objective_id: str | None = None
    phase_id: str | None = None
    step_id: str | None = None
    turn_index: int | None = None
    event_index: int = Field(ge=0)
    event_type: Literal[
        "prompt",
        "assistant_message",
        "tool_call",
        "tool_result",
        "code_execution",
        "file_inspection",
        "submission",
        "grade_link",
        "state_transition",
        "missing_artifact",
    ]
    source_path: str
    source_artifact_path: str = Field(min_length=1)
    source_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_call_id: str | None = None
    tool_name: str | None = None
    file_accessed: str | None = None
    elapsed_sec: float | None = None
    execution_status: Literal["success", "execution_error", "timeout", "session_terminated"] | None = None
    output_truncated: bool | None = None
    content_chars: int | None = None
    status: Literal["observed", "missing", "invalid", "not_available"]
    failure_code: str | None = None

    @field_validator("timestamp")
    @classmethod
    def _require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("Trace event timestamps must be timezone-aware UTC values.")
        return value

    @model_validator(mode="after")
    def _require_trialdev_context(self) -> ModelActionTraceEventV1:
        if self.benchmark == "trialdev" and not all((self.program_id, self.scenario_id, self.objective_id)):
            raise ValueError("TrialDev action events require program_id, scenario_id, and objective_id.")
        return self


class BenchmarkRuntimeTraceEventV1(BaseModel):
    """One runner-native event linked to an immutable conversation payload."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.runtime_trace_event/v1"] = "trialagentbench.runtime_trace_event/v1"
    event_id: str = Field(min_length=1)
    timestamp: datetime
    source_artifact_path: str = Field(min_length=1)
    source_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark: BenchmarkNameV1
    event_index: int = Field(ge=0)
    task_id: str | None = Field(default=None, min_length=1)
    program_id: str | None = Field(default=None, min_length=1)
    scenario_id: str | None = Field(default=None, min_length=1)
    objective_id: str | None = Field(default=None, min_length=1)
    phase_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    event_type: Literal[
        "step_started",
        "prompt",
        "assistant_message",
        "tool_call",
        "tool_result",
        "code_execution",
        "file_inspection",
        "submission",
        "step_terminal",
    ]
    conversation_message_index: int | None = Field(default=None, ge=0)
    tool_call_index: int | None = Field(default=None, ge=0)
    tool_call_id: str | None = None
    tool_name: str | None = None
    file_accessed: str | None = None
    status: Literal["observed", "invalid"] = "observed"
    execution_status: Literal["success", "execution_error", "timeout", "session_terminated"] | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0.0)
    output_truncated: bool | None = None
    terminal_status: Literal["completed", "failed"] | None = None
    failure_type: str | None = None

    @field_validator("timestamp")
    @classmethod
    def _require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("Runtime event timestamps must be timezone-aware UTC values.")
        return value

    @model_validator(mode="after")
    def _validate_event(self) -> BenchmarkRuntimeTraceEventV1:
        execution_event = self.event_type in {"code_execution", "file_inspection", "tool_result"}
        if self.execution_status is not None and not execution_event:
            raise ValueError("Execution status is only valid for execution and tool-result events.")
        if (self.elapsed_seconds is not None or self.output_truncated is not None) and self.execution_status is None:
            raise ValueError("Execution timing and truncation require an execution status.")
        if self.benchmark == "trialdev" and not all((self.program_id, self.scenario_id, self.objective_id)):
            raise ValueError("TrialDev runtime events require program_id, scenario_id, and objective_id.")
        if self.benchmark == "trialeval" and not self.task_id:
            raise ValueError("TrialEval runtime events require task_id.")
        if self.event_type != "step_terminal":
            if self.terminal_status is not None or self.failure_type is not None:
                raise ValueError("Terminal status is only valid for step_terminal events.")
            return self
        if self.conversation_message_index is not None:
            raise ValueError("step_terminal must not reference a conversation message.")
        if self.terminal_status is None:
            raise ValueError("step_terminal requires terminal_status.")
        if self.terminal_status == "failed" and not self.failure_type:
            raise ValueError("Failed step_terminal events require failure_type.")
        if self.terminal_status == "completed" and self.failure_type is not None:
            raise ValueError("Completed step_terminal events cannot contain failure_type.")
        return self


class TraceExplorerDataAssetV1(BaseModel):
    """One public trace-explorer data asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    row_count: int = Field(ge=0)
    source_table: str = Field(min_length=1)


class TraceExplorerManifestV1(BaseModel):
    """Manifest for a public observable-trace explorer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trace_explorer_manifest/v1"] = "trialagentbench.trace_explorer_manifest/v1"
    bundle_root: str = Field(min_length=1)
    explorer_root: str = Field(min_length=1)
    public_only: bool
    data_assets: tuple[TraceExplorerDataAssetV1, ...] = Field(min_length=1)
    pages: tuple[str, ...] = Field(min_length=1)
    vega_specs: tuple[str, ...]


def runtime_event_source_payload_v1(
    *,
    benchmark: BenchmarkNameV1,
    task_id: str | None,
    program_id: str | None,
    scenario_id: str | None,
    objective_id: str | None,
    phase_id: str,
    step_id: str,
    event_type: str,
    terminal_status: str | None,
    failure_type: str | None,
    conversation_message: JsonValue | None,
) -> JsonValue:
    """Return the immutable payload hashed by one runner-native event."""

    if conversation_message is not None:
        return conversation_message
    if event_type not in {"step_started", "step_terminal"}:
        raise ValueError("Conversation-linked runtime events require their exact indexed message payload.")
    return {
        "benchmark": benchmark,
        "task_id": task_id,
        "program_id": program_id,
        "scenario_id": scenario_id,
        "objective_id": objective_id,
        "phase_id": phase_id,
        "step_id": step_id,
        "event_type": event_type,
        "terminal_status": terminal_status,
        "failure_type": failure_type,
    }


class TraceFeatureRowV1(BaseModel):
    """One benchmark unit summarized as observable action features."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trace_feature_row/v1"] = "trialagentbench.trace_feature_row/v1"
    benchmark: BenchmarkNameV1
    model_id: str
    run_id: str
    task_id: str | None = None
    assignment_id: str | None = None
    program_id: str | None = None
    scenario_id: str | None = None
    objective_id: str | None = None
    phase_id: str | None = None
    trace_coverage_status: TraceCoverageStatusV1
    inspected_public_data: bool
    executed_code: bool
    submitted_structured_answer: bool
    submitted_answer: bool = False
    submission_interface: Literal["structured", "narrative"] | None = None
    submission_transport: TrialEvalSubmissionTransportV1 | None = None
    trace_input_authority: TrialEvalTraceAuthorityV1 | None = None
    context_tier: TrialEvalContextConfigurationV1 | None = None
    data_preparation: TrialEvalDataPreparationV1 | None = None
    analysis_specification: TrialEvalAnalysisSpecificationV1 | None = None
    procedure_assistance: ProcedureAssistanceV1
    prompt_condition: TrialEvalPromptConditionV1 | None = None
    submitted_rationale: bool = False
    checked_confounding: bool | None = None
    checked_ph_assumption: bool | None = None
    checked_missingness: bool | None = None
    checked_censoring: bool | None = None
    quantified_uncertainty: bool | None = None
    used_sensitivity_analysis: bool | None = None
    considered_safety: bool | None = None
    considered_cost: bool | None = None
    objective_aligned_rationale: bool | None = None
    semantic_feature_source: Literal["structured_field", "deterministic_text_rule", "not_available"]
    score_link_id: str | None = None
    endpoint_valid: bool | None = None
    endpoint_state: TraceEndpointStateV1 = "not_scoreable_trace_only"
    submission_artifact_paths: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _require_trialdev_context(self) -> TraceFeatureRowV1:
        if self.benchmark == "trialdev" and not all((self.program_id, self.scenario_id, self.objective_id)):
            raise ValueError("TrialDev trace features require program_id, scenario_id, and objective_id.")
        if self.benchmark == "trialdev" and any(
            value is not None
            for value in (self.context_tier, self.data_preparation, self.analysis_specification, self.prompt_condition)
        ):
            raise ValueError("TrialDev trace features cannot carry TrialEval evidence factors.")
        if self.benchmark == "trialeval":
            if self.submission_interface is None or self.prompt_condition is None:
                raise ValueError("TrialEval trace features require interface and prompt-condition provenance.")
            TrialEvalEvidenceFactorsV1(
                context_configuration=self.context_tier,
                data_preparation=self.data_preparation,
                analysis_specification=self.analysis_specification,
            )
        return self


class TrialEvalTraceInputV1(BaseModel):
    """One typed participant output before trace features are derived."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval_trace_input/v1"] = "trialagentbench.trialeval_trace_input/v1"
    run_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    assignment_id: str | None = Field(default=None, min_length=1)
    context_tier: TrialEvalContextConfigurationV1
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    procedure_assistance: ProcedureAssistanceV1
    prompt_condition: TrialEvalPromptConditionV1
    submission_interface: Literal["structured", "narrative"]
    submission_transport: TrialEvalSubmissionTransportV1
    authority: TrialEvalTraceAuthorityV1
    source_path: str = Field(min_length=1)
    submission: TrialEvalSubmissionV1 | None = None
    narrative_report: str | None = None

    @model_validator(mode="after")
    def _interface_payload_is_consistent(self) -> TrialEvalTraceInputV1:
        TrialEvalEvidenceFactorsV1(
            context_configuration=self.context_tier,
            data_preparation=self.data_preparation,
            analysis_specification=self.analysis_specification,
        )
        if self.submission_interface == "structured":
            if self.authority != "authoritative_structured":
                raise ValueError("Structured trace inputs must be authoritative_structured.")
            if self.narrative_report is not None:
                raise ValueError("Structured trace inputs cannot contain a narrative report.")
        else:
            if self.authority != "non_authoritative_narrative":
                raise ValueError("Raw narrative trace inputs must remain non_authoritative_narrative.")
            if self.submission is not None:
                raise ValueError("Raw narrative trace inputs cannot contain a canonical submission.")
        return self

    @property
    def answer_present(self) -> bool:
        """Return whether the participant supplied the declared output form."""

        if self.submission_interface == "structured":
            return self.submission is not None
        return bool(self.narrative_report and self.narrative_report.strip())


class EvidenceUseRowV1(BaseModel):
    """One evidence category credited or rejected for an observable unit."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.evidence_use_row/v1"] = "trialagentbench.evidence_use_row/v1"
    benchmark: BenchmarkNameV1
    model_id: str
    run_id: str
    task_id: str | None = None
    assignment_id: str | None = None
    program_id: str | None = None
    phase_id: str | None = None
    evidence_category: EvidenceCategoryV1
    source: Literal["tool_call", "code_path", "structured_field", "text_citation", "validator"]
    artifact_path: str
    participant_facing: bool
    leakage_violation: bool = False


class SemanticActionFeatureRowV1(BaseModel):
    """One semantic feature supported by explicit observable evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.semantic_action_feature_row/v1"] = (
        "trialagentbench.semantic_action_feature_row/v1"
    )
    benchmark: BenchmarkNameV1
    model_id: str
    run_id: str
    task_id: str | None = None
    assignment_id: str | None = None
    program_id: str | None = None
    phase_id: str | None = None
    feature_name: SemanticActionFeatureNameV1
    feature_present: bool
    evidence_strength: SemanticActionFeatureEvidenceV1
    evidence_basis: tuple[str, ...] = ()
    score_link_id: str


class TrialDevPhaseOutcomeRowV1(BaseModel):
    """One TrialDev phase outcome derived from public run and score records."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialdev_phase_outcome_row/v1"] = (
        "trialagentbench.trialdev_phase_outcome_row/v1"
    )
    benchmark: Literal["trialdev"] = "trialdev"
    model_id: str
    run_id: str
    scenario_id: str
    program_id: str
    objective_id: str
    phase_id: Literal["observational_review", "phase1", "phase2", "phase3"]
    phase_attempted: bool
    phase_reached: bool
    stopped_at_phase: str | None = None
    decision_action: str | None = None
    advance: bool | None = None
    candidate_drug_id: str | None = None
    endpoint_id: str | None = None
    design_signature_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    target_sample_size: int | None = Field(default=None, ge=1)
    follow_up_days: int | None = Field(default=None, ge=1)
    allocation_ratio: str | None = None
    allocation_weights: tuple[float, ...] = ()
    design_status: (
        Literal[
            "statistically_inadequate",
            "operationally_infeasible",
            "valid_dominated",
            "valid_frontier",
            "valid_nondominated",
        ]
        | None
    ) = None
    statistically_adequate: bool | None = None
    operationally_feasible: bool | None = None
    operational_support: int | None = Field(default=None, ge=0)
    operational_headroom: int | None = Field(default=None, ge=0)
    operational_shortage: int | None = Field(default=None, ge=0)
    participant_excess_vs_minimum: int | None = Field(default=None, ge=0)
    participant_shortage_vs_minimum: int | None = Field(default=None, ge=0)
    avoidable_participant_follow_up_days_min: int | None = Field(default=None, ge=0)
    avoidable_participant_follow_up_days_max: int | None = Field(default=None, ge=0)
    n_materializations: int = Field(default=0, ge=0)
    execute_code_calls: int = Field(default=0, ge=0)
    inspect_parquet_calls: int = Field(default=0, ge=0)
    turns: int = Field(default=0, ge=0)
    matched_item_id: str | None = None
    violations_n: int = Field(default=0, ge=0)
    violation_kinds: tuple[str, ...] = ()
    invalid_attempt_reasons: tuple[str, ...] = ()
    feasibility_failures: tuple[str, ...] = ()
    lane_failure_reasons: tuple[str, ...] = ()
    candidate_eligibility_records: tuple[dict[str, object], ...] = ()
    trajectory_decision_score: float | None = None
    trajectory_primary_score: float | None = None
    decision_regret: float | None = None
    selected_winner_drug_id: str | None = None
    best_candidate_drug_id: str | None = None
    ranking_score: float | None = None
    program_score: float | None = None
    policy_reference_regret: float | None = None
    in_set_regret: float | None = None
    trialdev_result_source: Literal["results_full_score_export", "not_available"] = "not_available"
    endpoint_state: TraceEndpointStateV1
    score_link_id: str

    @model_validator(mode="after")
    def validate_design_observation(self) -> TrialDevPhaseOutcomeRowV1:
        """Require complete design identity and ordered resource consequences."""

        identity = (
            self.design_signature_sha256,
            self.target_sample_size,
            self.follow_up_days,
        )
        if any(value is not None for value in identity) and not all(value is not None for value in identity):
            raise ValueError("A TrialDev phase design identity must be complete when present.")
        has_design = self.design_signature_sha256 is not None
        if has_design and (self.allocation_ratio is None) == (not self.allocation_weights):
            raise ValueError("A TrialDev phase design requires exactly one allocation representation.")
        if not has_design and (self.allocation_ratio is not None or self.allocation_weights):
            raise ValueError("A TrialDev phase allocation requires its design identity.")
        resource = (
            self.design_status,
            self.participant_excess_vs_minimum,
            self.participant_shortage_vs_minimum,
            self.avoidable_participant_follow_up_days_min,
            self.avoidable_participant_follow_up_days_max,
        )
        if any(value is not None for value in resource) and not all(value is not None for value in resource):
            raise ValueError("A TrialDev phase resource consequence must be complete when present.")
        if any(value is not None for value in resource) and self.design_signature_sha256 is None:
            raise ValueError("A TrialDev phase resource consequence requires its design identity.")
        if self.phase_id == "observational_review" and (
            any(value is not None for value in (*identity, *resource)) or self.allocation_weights
        ):
            raise ValueError("Observational review cannot carry a randomized design observation.")
        if (
            self.avoidable_participant_follow_up_days_min is not None
            and self.avoidable_participant_follow_up_days_max is not None
            and self.avoidable_participant_follow_up_days_min > self.avoidable_participant_follow_up_days_max
        ):
            raise ValueError("Avoidable participant-follow-up bounds must be ordered.")
        return self


class FailureCascadeRowV1(BaseModel):
    """First observable failure and downstream outcome for one unit."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.failure_cascade_row/v1"] = "trialagentbench.failure_cascade_row/v1"
    benchmark: BenchmarkNameV1
    model_id: str
    run_id: str
    task_id: str | None = None
    assignment_id: str | None = None
    program_id: str | None = None
    first_failure_phase: str | None = None
    first_failure_type: Literal[
        "missing_output",
        "no_public_evidence_use",
        "endpoint_invalid_or_unusable",
        "assumption_check_omitted",
        "confounding_check_omitted",
        "wrong_asset",
        "wrong_endpoint",
        "request_or_materialization_violation_observed",
        "unsupported_stop_or_advance",
        "rationale_action_mismatch",
        "tool_or_runtime_failure",
        "none_observed",
    ]
    downstream_endpoint_failed: bool
    score_link_id: str | None = None


class ProgramFailureCascadeRowV1(BaseModel):
    """Program-level TrialDev failure cascade linked to observable evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.program_failure_cascade_row/v1"] = (
        "trialagentbench.program_failure_cascade_row/v1"
    )
    benchmark: Literal["trialdev"] = "trialdev"
    model_id: str
    run_id: str
    scenario_id: str
    program_id: str
    objective_id: str
    first_failure_phase: str | None = None
    first_failure_type: Literal[
        "missing_output",
        "tool_or_runtime_failure",
        "hidden_or_grader_access",
        "wrong_asset",
        "wrong_endpoint",
        "request_or_materialization_violation_observed",
        "assumption_check_omitted",
        "confounding_check_omitted",
        "unsupported_stop_or_advance",
        "rationale_action_mismatch",
        "not_reached_after_stop",
        "none_observed",
    ]
    first_failure_evidence: tuple[str, ...]
    terminal_phase: str | None = None
    terminal_decision_action: str | None = None
    terminal_success: bool | None = None
    downstream_endpoint_failed: bool
    trajectory_primary_score: float | None = None
    trajectory_decision_score: float | None = None


__all__ = [
    "BenchmarkNameV1",
    "BenchmarkRuntimeTraceEventV1",
    "EvidenceCategoryV1",
    "EvidenceUseRowV1",
    "FailureCascadeRowV1",
    "ModelActionTraceEventV1",
    "ProgramFailureCascadeRowV1",
    "SemanticActionFeatureNameV1",
    "SemanticActionFeatureRowV1",
    "TraceEndpointStateV1",
    "TraceFeatureRowV1",
    "TrialDevPhaseOutcomeRowV1",
    "TrialEvalTraceInputV1",
    "runtime_event_source_payload_v1",
]
