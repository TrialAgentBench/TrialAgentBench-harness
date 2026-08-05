"""Run- and result-artifact contracts (TrialEval/TrialDev harness outputs)."""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from trialagentbench_harness.contracts.core.config import (
    DecodingConfigV1,
    ExperimentConditionV1,
    RoutingConfigV1,
)
from trialagentbench_harness.contracts.core.trialeval_factors import TrialEvalEvidenceFactorsV1
from trialagentbench_harness.contracts.experiments import (
    ProcedureAssistanceV1,
    TrialEvalAblationAssignmentV1,
    TrialEvalAnalysisSpecificationV1,
    TrialEvalPromptConditionV1,
    TrialEvalSubmissionInterfaceV1,
    procedure_assistance_exposure_v1,
)
from trialagentbench_harness.contracts.scoring.trialeval_scores import TrialEvalItemScoresV1
from trialagentbench_harness.contracts.submission import TrialEvalSubmissionV1
from trialagentbench_harness.contracts.trialdev.trialdev_evaluation_target_register import (
    TrialDevEvaluationLaneV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevProgrammeAnalysisQualityV1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256
from trialagentbench_harness.ports.code_execution import CodeExecutionLimitsV1


class ExecutorPackageV1(BaseModel):
    """One installed distribution in the isolated analysis environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)


class ExecutorEnvironmentV1(BaseModel):
    """Immutable identity and package inventory for model-code execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.executor_environment/v1"] = "trialagentbench.executor_environment/v1"
    image_reference: str = Field(..., min_length=1)
    image_id: str = Field(..., pattern=r"^sha256:[0-9a-f]{64}$")
    python_version: str = Field(..., min_length=1)
    packages: tuple[ExecutorPackageV1, ...] = Field(..., min_length=1)
    limits: CodeExecutionLimitsV1

    @model_validator(mode="after")
    def validate_package_inventory(self) -> ExecutorEnvironmentV1:
        """Require a complete, uniquely named package inventory."""

        normalized = [package.name.casefold() for package in self.packages]
        if normalized != sorted(normalized):
            raise ValueError("Executor packages must be sorted by normalized name.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Executor package names must be unique.")
        return self


class _AuthoritativeRunConfigV1(BaseModel):
    """Base for persisted run configs with a required canonical self-identity."""

    run_identity_sha256: str = Field(..., min_length=64, max_length=64)
    _identity_excluded_fields: ClassVar[set[str]]

    @classmethod
    def create(cls, **data: object) -> Self:
        """Normalize fields, compute the authoritative identity, and revalidate."""

        if "run_identity_sha256" in data:
            raise ValueError("Run identity is computed by the authoritative config factory.")
        provisional = cls.model_validate(
            {**data, "run_identity_sha256": "0" * 64},
            context={"skip_run_identity_validation": True},
        )
        payload = provisional.model_dump(
            mode="json",
            exclude=cls._identity_excluded_fields | {"run_identity_sha256"},
        )
        return cls.model_validate(
            {
                **provisional.model_dump(mode="python"),
                "run_identity_sha256": canonical_payload_sha256(payload),
            }
        )

    @model_validator(mode="after")
    def validate_authoritative_identity(self, info: ValidationInfo) -> Self:
        """Reject persisted configs whose identity does not match their content."""

        if info.context and info.context.get("skip_run_identity_validation") is True:
            return self
        payload = self.model_dump(
            mode="json",
            exclude=self._identity_excluded_fields | {"run_identity_sha256"},
        )
        if self.run_identity_sha256 != canonical_payload_sha256(payload):
            raise ValueError("Run identity does not match its authoritative configuration.")
        return self


class TrialEvalRunConfigV1(_AuthoritativeRunConfigV1):
    """TrialEvalBench live-run configuration persisted to `run_config.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench_trialeval_run_config_v1"] = "trialagentbench_trialeval_run_config_v1"
    schema_version: Literal[1] = 1
    timestamp_utc: datetime

    model: str
    output_mode: Literal["structured"] = "structured"
    item_watchdog_seconds: int = Field(..., ge=1)

    participant_dir: str
    participant_release_sha256: str = Field(..., min_length=64, max_length=64)
    prompt_set_sha256: str = Field(..., min_length=64, max_length=64)
    scorer_source_sha256: str = Field(..., min_length=64, max_length=64)
    agent_source_sha256: str = Field(..., min_length=64, max_length=64)
    experiment_condition: ExperimentConditionV1
    prompt_condition: Literal["neutral"] = "neutral"
    submission_interface: Literal["structured"] = "structured"
    task_ids: list[str] = Field(..., min_length=1)
    task_evidence_factors: dict[str, TrialEvalEvidenceFactorsV1]
    data_format: Literal["trialagentbench_v1"] = "trialagentbench_v1"
    data_version: Literal["trialagentbench_v1"] = "trialagentbench_v1"

    decoding: DecodingConfigV1
    routing: RoutingConfigV1
    executor: ExecutorEnvironmentV1

    workers: int = Field(default=1, ge=1)
    n_items: int = Field(..., ge=1)
    _identity_excluded_fields: ClassVar[set[str]] = {
        "timestamp_utc",
        "participant_dir",
        "workers",
    }

    @model_validator(mode="after")
    def validate_task_denominator(self) -> TrialEvalRunConfigV1:
        """Require a unique task denominator consistent with the run size."""

        condition = self.experiment_condition
        if condition.procedure_assistance != "output_contract_only":
            raise ValueError("TrialEval uses the output-contract-only participant condition.")
        if condition.maximum_submission_attempts is not None:
            raise ValueError("TrialEval has no separate submission-attempt limit.")
        if condition.tool_choice != "auto":
            raise ValueError("TrialEval owns tool selection within the agent loop.")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique.")
        if any(not task_id.strip() for task_id in self.task_ids):
            raise ValueError("task_ids must not contain empty values.")
        if self.n_items != len(self.task_ids):
            raise ValueError("n_items must equal the number of task_ids.")
        if set(self.task_evidence_factors) != set(self.task_ids):
            raise ValueError("task_evidence_factors must contain exactly one entry for every task_id.")
        return self


class RunCoverageV1(BaseModel):
    """Prospective immutable schedule and atomically maintained completion set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.run_coverage/v1"] = "trialagentbench.run_coverage/v1"
    run_identity_sha256: str = Field(..., min_length=64, max_length=64)
    schedule_sha256: str = Field(..., min_length=64, max_length=64)
    unit_ids: tuple[str, ...] = Field(..., min_length=1)
    completed_unit_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_denominator(self) -> RunCoverageV1:
        """Reject duplicate, reordered, or out-of-schedule completion identities."""

        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("Run coverage unit_ids must be unique.")
        if len(set(self.completed_unit_ids)) != len(self.completed_unit_ids):
            raise ValueError("Run coverage completed_unit_ids must be unique.")
        expected_order = tuple(unit_id for unit_id in self.unit_ids if unit_id in self.completed_unit_ids)
        if self.completed_unit_ids != expected_order:
            raise ValueError("Completed units must be an ordered subset of the prospective schedule.")
        return self


class TrialDevRunStopV1(BaseModel):
    """Typed custody for a deliberately stopped incomplete TrialDev schedule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_run_stop/v1"] = "trialagentbench.trialdev_run_stop/v1"
    run_identity_sha256: str = Field(..., min_length=64, max_length=64)
    reason: Literal["reported_cost_threshold", "reported_cost_unavailable", "keyboard_interrupt"]
    unit_ids: tuple[str, ...] = Field(..., min_length=1)
    completed_unit_ids: tuple[str, ...] = ()
    interrupted_unit_ids: tuple[str, ...] = ()
    not_started_unit_ids: tuple[str, ...] = ()
    reported_cost_threshold_usd: float | None = Field(default=None, gt=0.0)
    observed_reported_cost_usd: float = Field(..., ge=0.0)
    reported_cost_overshoot_usd: float = Field(..., ge=0.0)
    provider_response_count: int = Field(..., ge=0)
    cost_complete: bool

    @model_validator(mode="after")
    def validate_partition(self) -> TrialDevRunStopV1:
        """Require exact, ordered, disjoint custody for every scheduled unit."""

        groups = (self.completed_unit_ids, self.interrupted_unit_ids, self.not_started_unit_ids)
        flattened = tuple(item for group in groups for item in group)
        if len(flattened) != len(set(flattened)):
            raise ValueError("Stopped TrialDev unit disposition groups must be disjoint.")
        if set(flattened) != set(self.unit_ids):
            raise ValueError("Stopped TrialDev custody must account for every scheduled unit exactly once.")
        for group in groups:
            expected = tuple(item for item in self.unit_ids if item in group)
            if group != expected:
                raise ValueError("Stopped TrialDev unit dispositions must preserve prospective schedule order.")
        if self.reason == "reported_cost_threshold":
            if self.reported_cost_threshold_usd is None:
                raise ValueError("A reported-cost stop requires its threshold.")
            expected_overshoot = max(
                0.0,
                self.observed_reported_cost_usd - self.reported_cost_threshold_usd,
            )
            if abs(self.reported_cost_overshoot_usd - expected_overshoot) > 1e-12:
                raise ValueError("Reported-cost overshoot does not reproduce from observed cost and threshold.")
        elif self.reported_cost_overshoot_usd != 0.0:
            raise ValueError("Only a reported-cost stop can record cost overshoot.")
        return self


class ProviderRequestEventV1(BaseModel):
    """One append-only lifecycle event for a provider request."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.provider_request_event/v1"] = "trialagentbench.provider_request_event/v1"
    request_id: str = Field(..., min_length=1)
    status: Literal["started", "succeeded", "failed"]
    benchmark: Literal["trialeval", "trialdev"]
    unit_id: str = Field(..., min_length=1)
    phase_id: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    turn_index: int = Field(..., ge=1)
    elapsed_seconds: float | None = Field(..., ge=0.0)
    requested_model: str = Field(..., min_length=1)
    provider_route: str = Field(..., min_length=1)
    response_id: str | None = None
    returned_model: str | None = None
    upstream_provider: str | None = None
    finish_reason: str | None = None
    created_unix: int | None = Field(default=None, ge=0)
    usage_status: Literal["not_applicable", "reported", "not_reported"]
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    reported_cost_usd: float | None = Field(default=None, ge=0.0)
    request_attempts: int = Field(..., ge=0)
    transient_failure_count: int = Field(..., ge=0)
    backoff_seconds: float = Field(..., ge=0.0)
    failure_type: (
        Literal[
            "timeout",
            "provider_error",
            "cancelled",
        ]
        | None
    ) = None
    exception_type: str | None = Field(default=None, min_length=1)
    http_status_code: int | None = Field(default=None, ge=100, le=599)

    @model_validator(mode="after")
    def validate_lifecycle_state(self) -> ProviderRequestEventV1:
        """Require complete, status-consistent route and usage telemetry."""

        if self.status == "started":
            if self.elapsed_seconds is not None or self.request_attempts != 0:
                raise ValueError("Started provider requests cannot contain terminal timing or attempts.")
            if self.usage_status != "not_applicable" or any(
                (self.prompt_tokens, self.completion_tokens, self.total_tokens)
            ):
                raise ValueError("Started provider requests require explicit non-applicable zero usage.")
            if self.failure_type is not None:
                raise ValueError("Started provider requests cannot contain a failure_type.")
            if self.exception_type is not None or self.http_status_code is not None:
                raise ValueError("Started provider requests cannot contain failure diagnostics.")
            return self
        if self.elapsed_seconds is None or self.request_attempts < 1:
            raise ValueError("Terminal provider requests require elapsed_seconds and request_attempts.")
        if self.status == "failed":
            if self.failure_type is None:
                raise ValueError("Failed provider requests require a typed failure_type.")
            if self.usage_status != "not_applicable" or any(
                (self.prompt_tokens, self.completion_tokens, self.total_tokens)
            ):
                raise ValueError("Failed provider requests require explicit non-applicable zero usage.")
            if self.transient_failure_count > self.request_attempts:
                raise ValueError("Failed request transient_failure_count cannot exceed request_attempts.")
            return self
        if self.failure_type is not None or self.exception_type is not None or self.http_status_code is not None:
            raise ValueError("Succeeded provider requests cannot contain failure diagnostics.")
        if self.usage_status == "not_applicable":
            raise ValueError("Succeeded provider requests require reported or not_reported usage status.")
        if self.usage_status == "not_reported" and any(
            (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        ):
            raise ValueError("Unreported provider usage must use explicit zero token counts.")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens.")
        if self.request_attempts != self.transient_failure_count + 1:
            raise ValueError("request_attempts must equal transient_failure_count + 1.")
        return self


class ProviderTelemetrySummaryV1(BaseModel):
    """Run-level provider telemetry totals derived from response records."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.provider_telemetry_summary/v1"] = (
        "trialagentbench.provider_telemetry_summary/v1"
    )
    run_identity_sha256: str = Field(..., min_length=64, max_length=64)
    schedule_sha256: str = Field(..., min_length=64, max_length=64)
    unit_ids: tuple[str, ...] = Field(..., min_length=1)
    completed_unit_ids: tuple[str, ...]
    response_count: int = Field(..., ge=0)
    failed_request_count: int = Field(..., ge=0)
    failure_type_counts: dict[Literal["timeout", "provider_error", "cancelled"], int]
    responses_with_usage: int = Field(..., ge=0)
    responses_with_response_id: int = Field(..., ge=0)
    responses_with_reported_cost: int = Field(..., ge=0)
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    reported_cost_usd: float = Field(..., ge=0.0)
    elapsed_seconds: float = Field(..., ge=0.0)
    request_attempt_count: int = Field(..., ge=0)
    transient_failure_count: int = Field(..., ge=0)
    backoff_seconds: float = Field(..., ge=0.0)
    requested_models: list[str]
    returned_models: list[str]
    upstream_providers: list[str]
    source_files: list[str]
    archived_request_count: int = Field(..., ge=0)
    archived_failed_request_count: int = Field(..., ge=0)
    archived_failure_type_counts: dict[Literal["timeout", "provider_error", "cancelled"], int]
    archived_request_attempt_count: int = Field(..., ge=0)
    archived_transient_failure_count: int = Field(..., ge=0)
    archived_backoff_seconds: float = Field(..., ge=0.0)
    archived_elapsed_seconds: float = Field(..., ge=0.0)
    archived_source_files: list[str]

    @model_validator(mode="after")
    def validate_failure_totals(self) -> ProviderTelemetrySummaryV1:
        """Require typed failure counts to agree with their totals."""

        expected_order = tuple(unit_id for unit_id in self.unit_ids if unit_id in self.completed_unit_ids)
        if self.completed_unit_ids != expected_order:
            raise ValueError("Provider telemetry completed units must be an ordered schedule subset.")
        if sum(self.failure_type_counts.values()) != self.failed_request_count:
            raise ValueError("Provider failure_type_counts do not equal failed_request_count.")
        if sum(self.archived_failure_type_counts.values()) != self.archived_failed_request_count:
            raise ValueError("Archived failure_type_counts do not equal archived_failed_request_count.")
        return self


class TrialEvalConditionProvenanceV1(BaseModel):
    """Exact participant-facing prompt and interface provenance for one item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    procedure_assistance: ProcedureAssistanceV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    analysis_surface_sha256: str = Field(..., min_length=64, max_length=64)
    prompt_condition: TrialEvalPromptConditionV1
    submission_interface: TrialEvalSubmissionInterfaceV1
    max_turns: int = Field(..., ge=1)
    prompt_set_sha256: str = Field(..., min_length=64, max_length=64)
    rendered_system_prompt_sha256: str = Field(..., min_length=64, max_length=64)
    tool_schema_sha256: str = Field(..., min_length=64, max_length=64)
    response_contract_sha256: str = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_trialeval_assistance(self) -> TrialEvalConditionProvenanceV1:
        """Reject assistance levels that are defined only for TrialDev."""

        procedure_assistance_exposure_v1(
            suite="trialeval",
            procedure_assistance=self.procedure_assistance,
        )
        return self


class TrialEvalAgentOutputV1(BaseModel):
    """Minimal agent output persisted alongside scoring results."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench_trialeval_agent_output_v1"] = "trialagentbench_trialeval_agent_output_v1"
    schema_version: Literal[1] = 1
    status: str | None = None
    turns_used: int | None = None
    report: str | None = None
    result: TrialEvalSubmissionV1 | None = None
    condition_provenance: TrialEvalConditionProvenanceV1


class TrialEvalItemResultV1(BaseModel):
    """Per-item TrialEvalBench artifact persisted under `items/<item_id>.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench_trialeval_item_result_v1"] = "trialagentbench_trialeval_item_result_v1"
    schema_version: Literal[1] = 1
    item_id: str
    timestamp_utc: datetime

    run_config: TrialEvalRunConfigV1
    agent_output: TrialEvalAgentOutputV1
    scores: TrialEvalItemScoresV1 | None = None

    @model_validator(mode="after")
    def validate_condition_provenance(self) -> TrialEvalItemResultV1:
        """Bind the participant-facing condition to the immutable run config."""

        provenance = self.agent_output.condition_provenance
        expected = {
            "procedure_assistance": self.run_config.experiment_condition.procedure_assistance,
            "analysis_specification": self.run_config.task_evidence_factors[self.item_id].analysis_specification,
            "prompt_condition": self.run_config.prompt_condition,
            "submission_interface": self.run_config.submission_interface,
            "max_turns": self.run_config.experiment_condition.maximum_turns_per_step,
            "prompt_set_sha256": self.run_config.prompt_set_sha256,
        }
        observed = {
            "procedure_assistance": provenance.procedure_assistance,
            "analysis_specification": provenance.analysis_specification,
            "prompt_condition": provenance.prompt_condition,
            "submission_interface": provenance.submission_interface,
            "max_turns": provenance.max_turns,
            "prompt_set_sha256": provenance.prompt_set_sha256,
        }
        if observed != expected:
            raise ValueError("TrialEval condition provenance does not match the immutable run configuration.")
        return self


class TrialEvalAblationRunConfigV1(_AuthoritativeRunConfigV1):
    """Immutable runtime configuration for one TrialEval ablation schedule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval_ablation_run/v1"] = "trialagentbench.trialeval_ablation_run/v1"
    timestamp_utc: datetime
    experiment_id: str = Field(..., min_length=1)
    schedule_checksum: str = Field(..., min_length=64, max_length=64)
    participant_release_sha256: str = Field(..., min_length=64, max_length=64)
    prompt_set_sha256: str = Field(..., min_length=64, max_length=64)
    scorer_source_sha256: str = Field(..., min_length=64, max_length=64)
    agent_source_sha256: str = Field(..., min_length=64, max_length=64)
    model: str = Field(..., min_length=1)
    max_turns: int | None = Field(default=None, ge=1)
    max_context_characters: int = Field(..., ge=1)
    item_watchdog_seconds: int = Field(..., ge=1)
    decoding: DecodingConfigV1
    routing: RoutingConfigV1
    executor: ExecutorEnvironmentV1
    workers: int = Field(..., ge=1)
    n_assignments: int = Field(..., ge=1)
    _identity_excluded_fields: ClassVar[set[str]] = {"timestamp_utc", "workers"}


class TrialEvalAblationItemResultV1(BaseModel):
    """One immutable response collected under a precommitted assignment."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval_ablation_item_result/v1"] = (
        "trialagentbench.trialeval_ablation_item_result/v1"
    )
    timestamp_utc: datetime
    assignment: TrialEvalAblationAssignmentV1
    run_config: TrialEvalAblationRunConfigV1
    agent_output: TrialEvalAgentOutputV1

    @model_validator(mode="after")
    def validate_assignment_provenance(self) -> TrialEvalAblationItemResultV1:
        """Bind the persisted response to its scheduled treatment cell."""

        provenance = self.agent_output.condition_provenance
        if provenance.procedure_assistance != self.assignment.procedure_assistance:
            raise ValueError("Ablation response assistance condition does not match its assignment.")
        if provenance.analysis_specification != self.assignment.analysis_specification:
            raise ValueError("Ablation response analysis specification does not match its assignment.")
        if provenance.analysis_surface_sha256 != self.assignment.analysis_surface_sha256:
            raise ValueError("Ablation response analysis surface does not match its assignment.")
        if provenance.prompt_condition != self.assignment.prompt_condition:
            raise ValueError("Ablation response prompt condition does not match its assignment.")
        if provenance.submission_interface != self.assignment.submission_interface:
            raise ValueError("Ablation response interface does not match its assignment.")
        if provenance.prompt_set_sha256 != self.run_config.prompt_set_sha256:
            raise ValueError("Ablation response prompt-set hash does not match its run.")
        if self.run_config.max_turns is not None and provenance.max_turns != self.run_config.max_turns:
            raise ValueError("Ablation response turn budget does not match its run.")
        if self.agent_output.result is not None:
            if self.assignment.submission_interface != "structured":
                raise ValueError("Narrative assignments cannot persist a structured result.")
            if self.agent_output.result.task_id != self.assignment.task_id:
                raise ValueError("Ablation structured result task_id does not match its assignment.")
        if self.agent_output.report is not None and self.assignment.submission_interface != "narrative":
            raise ValueError("Structured assignments cannot persist a narrative report.")
        return self


class TrialDevRunConfigV1(_AuthoritativeRunConfigV1):
    """TrialDevBench live-run configuration persisted to `run_config.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench_trialdev_run_config_v1"] = "trialagentbench_trialdev_run_config_v1"
    schema_version: Literal[1] = 1
    timestamp_utc: datetime

    bundle: str
    bundle_sha256: str = Field(..., min_length=64, max_length=64)
    scorer_source_sha256: str = Field(..., min_length=64, max_length=64)
    runner_source_sha256: str = Field(..., min_length=64, max_length=64)
    prompt_interface_sha256: str = Field(..., min_length=64, max_length=64)
    staging_source_sha256: str = Field(..., min_length=64, max_length=64)
    procedure_assistance: ProcedureAssistanceV1
    model: str
    experiment_condition: ExperimentConditionV1
    master_seed: int
    seed_variants: int
    workers: int
    reported_cost_stop_usd: float | None = Field(default=None, gt=0.0)

    decoding: DecodingConfigV1
    routing: RoutingConfigV1
    executor: ExecutorEnvironmentV1

    # Execution knobs (recorded for reproducibility)
    max_turns_per_step: int = Field(..., ge=1)
    max_context_characters: int = Field(..., ge=1)
    max_phase_retries: int = Field(..., ge=1)
    max_submission_attempts: int = Field(default=10, ge=1)
    program_watchdog_seconds: int = Field(..., ge=1)
    programs_filter: list[str] = Field(default_factory=list)
    selected_program_ids: list[str] = Field(..., min_length=1)
    n_programs_selected: int = Field(..., ge=1)
    label: str | None = None
    _identity_excluded_fields: ClassVar[set[str]] = {
        "timestamp_utc",
        "bundle",
        "workers",
        "programs_filter",
        "n_programs_selected",
        "label",
    }

    @model_validator(mode="after")
    def validate_selected_population(self) -> TrialDevRunConfigV1:
        """Require an immutable, unique selected-program denominator when declared."""

        if len(set(self.selected_program_ids)) != len(self.selected_program_ids):
            raise ValueError("selected_program_ids must be unique.")
        if self.n_programs_selected != len(self.selected_program_ids):
            raise ValueError("n_programs_selected must equal the selected_program_ids denominator.")
        if self.procedure_assistance != self.experiment_condition.procedure_assistance:
            raise ValueError("procedure_assistance must match the experimental condition.")
        if self.max_turns_per_step != self.experiment_condition.maximum_turns_per_step:
            raise ValueError("max_turns_per_step must match the experimental condition.")
        if self.max_submission_attempts != self.experiment_condition.maximum_submission_attempts:
            raise ValueError("max_submission_attempts must match the experimental condition.")
        return self


class TrialDevPhaseAttemptSummaryV1(BaseModel):
    """One phase attempt entry stored in `chain_summary.json`."""

    schema_id: Literal["trialagentbench_trialdev_phase_attempt_summary_v1"] = (
        "trialagentbench_trialdev_phase_attempt_summary_v1"
    )
    schema_version: Literal[1] = 1
    phase_id: str
    matched_item_id: str | None = None
    decision_action: str | None = None
    advance: bool | None = None
    candidate_drug_id: str | None = None
    n_materializations: int = 0
    turns: int = 0
    execute_code_calls: int = 0
    inspect_parquet_calls: int = 0


class TrialDevMaterializationUsageV1(BaseModel):
    """Observed materialization usage stored in `chain_summary.json`."""

    schema_id: Literal["trialagentbench_trialdev_materialization_usage_v1"] = (
        "trialagentbench_trialdev_materialization_usage_v1"
    )
    schema_version: Literal[1] = 1
    materialize_calls_by_phase: dict[str, int] = Field(default_factory=dict)


class TrialDevProgrammeResourceSummaryV1(BaseModel):
    """Analysis-facing cumulative TrialDev resource vector."""

    schema_id: Literal["trialagentbench_trialdev_programme_resource_summary_v1"] = (
        "trialagentbench_trialdev_programme_resource_summary_v1"
    )
    schema_version: Literal[1] = 1
    phase_count: int = Field(..., ge=0)
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
    cost_status: Literal["not_available_without_public_cost_schedule"]


class TrialDevCheckpointOutcomeV1(BaseModel):
    """One checkpoint's conditional and cumulative programme outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench_trialdev_checkpoint_outcome_v1"] = (
        "trialagentbench_trialdev_checkpoint_outcome_v1"
    )
    schema_version: Literal[1] = 1
    phase_id: Literal["observational_review", "phase1", "phase2", "phase3", "final_decision"]
    status: Literal[
        "reached",
        "missing_or_invalid",
        "structural_not_reached",
        "not_reached_after_invalid",
        "not_scheduled",
    ]
    required_lane_ids: tuple[TrialDevEvaluationLaneV1, ...]
    conditional_score: float | None = Field(default=None, ge=0.0, le=1.0)
    cumulative_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_score_applicability(self) -> TrialDevCheckpointOutcomeV1:
        """Require scores exactly for reached or failed required checkpoints."""

        score_bearing = self.status in {"reached", "missing_or_invalid"}
        if score_bearing != (self.conditional_score is not None and self.cumulative_score is not None):
            raise ValueError("TrialDev checkpoint scores are required exactly for score-bearing checkpoints.")
        if self.status == "missing_or_invalid" and self.conditional_score != 0.0:
            raise ValueError("A missing or invalid TrialDev checkpoint must have conditional score zero.")
        return self


class TrialDevTrajectoryMetricsV1(BaseModel):
    """Selected trajectory metrics summarized into `chain_summary.json`."""

    schema_id: Literal["trialagentbench_trialdev_trajectory_metrics_v1"] = (
        "trialagentbench_trialdev_trajectory_metrics_v1"
    )
    schema_version: Literal[1] = 1
    trajectory_primary_score: float | None = None
    programme_primary_score: float | None = Field(default=None, ge=0.0, le=1.0)
    checkpoint_outcomes: tuple[TrialDevCheckpointOutcomeV1, ...] = Field(default_factory=tuple)
    trajectory_decision_score: float | None = None
    decision_regret_by_phase: dict[str, float] = Field(default_factory=dict)
    n_invalid_attempts: int = 0
    invalid_attempt_reasons: list[str] = Field(default_factory=list)
    resource_summary: TrialDevProgrammeResourceSummaryV1 | None = None
    analysis_quality: TrialDevProgrammeAnalysisQualityV1 | None = None

    @model_validator(mode="after")
    def validate_programme_score(self) -> TrialDevTrajectoryMetricsV1:
        """Bind the programme score to all required reached checkpoints."""

        if not self.checkpoint_outcomes:
            if self.programme_primary_score is not None:
                raise ValueError("TrialDev programme score requires checkpoint outcomes.")
            return self
        phase_ids = tuple(record.phase_id for record in self.checkpoint_outcomes)
        expected = ("observational_review", "phase1", "phase2", "phase3", "final_decision")
        if phase_ids != expected:
            raise ValueError("TrialDev checkpoint outcomes must use the canonical programme order.")
        score_bearing = [
            record.conditional_score for record in self.checkpoint_outcomes if record.conditional_score is not None
        ]
        expected_score = min(score_bearing) if score_bearing else None
        if self.programme_primary_score != expected_score:
            raise ValueError("TrialDev programme_primary_score must be the minimum score-bearing checkpoint.")
        return self


class TrialDevPathStatsV1(BaseModel):
    """Observed turns and local-tool calls for one TrialDev step."""

    model_config = ConfigDict(extra="forbid")

    turns: int = Field(default=0, ge=0)
    execute_code: int = Field(default=0, ge=0)
    inspect_parquet: int = Field(default=0, ge=0)


class TrialDevChainSummaryV1(BaseModel):
    """Per-program chain summary artifact written by the TrialDev runner."""

    schema_id: Literal["trialagentbench_trialdev_chain_summary_v1"] = "trialagentbench_trialdev_chain_summary_v1"
    schema_version: Literal[1] = 1
    program_id: str
    scenario_id: str
    objective_id: str
    stopped_at_phase: str | None = None
    started_at_utc: str | None = None
    ended_at_utc: str | None = None
    wall_seconds_total: float | None = None

    phases_attempted: list[TrialDevPhaseAttemptSummaryV1] = Field(default_factory=list)
    obs_review_path_stats: TrialDevPathStatsV1 = Field(default_factory=TrialDevPathStatsV1)
    materialization_usage: TrialDevMaterializationUsageV1

    obs_review_grade_path: str | None = None
    trajectory_grade_path: str | None = None
    trajectory_metrics: TrialDevTrajectoryMetricsV1 = Field(default_factory=TrialDevTrajectoryMetricsV1)
    execution_status: Literal[
        "completed",
        "model_turn_limit",
        "model_invalid_submission",
        "infrastructure_timeout",
        "infrastructure_error",
    ]
    error: str | None = None
    violations_n: int = 0
    violations: list[dict[str, str]] = Field(default_factory=list)


class TrialDevPhaseRequestSummaryV1(BaseModel):
    """Schema-bearing summary of a phase request.

    This is a *harness-owned* sidecar that enables schema-first offline analysis
    without modifying upstream `request.json` files (which are consumed by the
    upstream grader/state machine).
    """

    schema_id: Literal["trialagentbench_trialdev_phase_request_summary_v1"] = (
        "trialagentbench_trialdev_phase_request_summary_v1"
    )
    schema_version: Literal[1] = 1

    phase_id: str
    endpoint_id: str | None = None
    selection_objective: str | None = None
    target_sample_size: int = Field(..., ge=1)
    follow_up_days: int = Field(..., ge=1)
    allocation_ratio: str = Field(..., min_length=1)
    site_count_budget: int = Field(..., ge=1)
    enrollment_window_days: int = Field(..., ge=1)


class TrialDevPhaseAnalysisSummaryV1(BaseModel):
    """Schema-bearing summary of a phase analysis submission."""

    schema_id: Literal["trialagentbench_trialdev_phase_analysis_summary_v1"] = (
        "trialagentbench_trialdev_phase_analysis_summary_v1"
    )
    schema_version: Literal[1] = 1

    phase_id: str
    ranked_drug_ids: list[str] = Field(default_factory=list)
    selected_winner_drug_id: str | None = None


class TrialDevPhaseDecisionSummaryV1(BaseModel):
    """Schema-bearing summary of a phase decision submission."""

    schema_id: Literal["trialagentbench_trialdev_phase_decision_summary_v1"] = (
        "trialagentbench_trialdev_phase_decision_summary_v1"
    )
    schema_version: Literal[1] = 1

    phase_id: str
    decision_action: str | None = None
    candidate_drug_id: str | None = None


class TrialDevPhaseStepSummaryV1(BaseModel):
    """Per-phase sidecar summarizing request/analysis/decision artifacts.

    Written by the runner as part of the canonical run contract. Offline
    aggregation and derived metrics consume this schema-bearing surface plus
    schema-wrapped grader outputs rather than untyped raw JSON.
    """

    schema_id: Literal["trialagentbench_trialdev_phase_step_summary_v1"] = (
        "trialagentbench_trialdev_phase_step_summary_v1"
    )
    schema_version: Literal[1] = 1

    program_id: str
    scenario_id: str
    objective_id: str
    phase_id: str

    request: TrialDevPhaseRequestSummaryV1 | None = None
    analysis: TrialDevPhaseAnalysisSummaryV1 | None = None
    decision: TrialDevPhaseDecisionSummaryV1 | None = None


class TrialDevObsReviewSummaryV1(BaseModel):
    """Schema-bearing summary for the observational_review submission."""

    schema_id: Literal["trialagentbench_trialdev_obs_review_summary_v1"] = (
        "trialagentbench_trialdev_obs_review_summary_v1"
    )
    schema_version: Literal[1] = 1

    program_id: str
    scenario_id: str
    objective_id: str

    method_route_id: str | None = None
    ranked_drug_ids: list[str] = Field(default_factory=list)
    recommended_drug_id: str | None = None
