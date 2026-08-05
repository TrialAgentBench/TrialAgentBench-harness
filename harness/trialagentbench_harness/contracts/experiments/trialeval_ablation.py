"""Precommitted TrialEval prompt and response-interface experiments."""

from __future__ import annotations

import random
from typing import Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.core.trialeval_factors import (
    TrialEvalDataPreparationV1,
    TrialEvalEvidenceFactorsV1,
)
from trialagentbench_harness.contracts.experiments.procedure_assistance import (
    ProcedureAssistanceV1,
    TrialEvalAnalysisSpecificationV1,
    TrialEvalPromptConditionV1,
    TrialEvalSubmissionInterfaceV1,
)
from trialagentbench_harness.contracts.experiments.resource_attainment import (
    BenchmarkResourceAttainmentPointV1,
    validate_resource_attainment_v1,
)
from trialagentbench_harness.contracts.experiments.trialeval_design import TrialEvalExperimentProtocolV1
from trialagentbench_harness.contracts.release.trialeval_runtime_surface import (
    TrialEvalSemanticDeliverableV1,
)
from trialagentbench_harness.contracts.submission import TrialEvalSubmissionV1
from trialagentbench_harness.grading.models import FailureCode
from trialagentbench_harness.io.checksums import canonical_payload_sha256

TrialEvalAblationMetricV1: TypeAlias = Literal[  # noqa: UP040
    "usable_primary",
    "route_match",
    "obligations_met",
    "numeric_result_available",
    "primary_uncertainty_valid",
    "primary_interval_agreement",
    "result_match",
    "numeric_absolute_error",
    "numeric_tolerance_ratio",
    "primary_analysis_conforms",
    "planning_valid",
    "planning_usable_with_primary",
    "planning_achieved_power",
    "planning_power_shortfall",
    "planning_underpowered",
    "planning_proportional_participant_deviation",
    "planning_log_sample_size_ratio",
    "planning_event_shortage",
    "planning_excess_events",
    "planning_excess_participants",
    "planning_participant_shortage",
]
TrialEvalMetricPopulationV1: TypeAlias = Literal[  # noqa: UP040
    "all_scheduled_assignments",
    "uncertainty_applicable_assignments",
    "matched_interval_evaluable_assignments",
    "matched_numeric_result_comparisons",
    "planning_eligible_assignments",
    "planning_consequence_evaluable_assignments",
]
TrialEvalObservableMetricV1: TypeAlias = Literal[  # noqa: UP040
    "answer_submitted",
    "public_data_inspected",
    "code_executed_successfully",
    "assistant_turns",
    "tool_calls",
    "file_inspections",
    "code_executions",
    "failed_code_executions",
    "truncated_outputs",
    "events_until_first_data_inspection",
    "events_until_first_successful_code_execution",
    "events_until_submission",
    "execution_elapsed_seconds",
    "declared_diagnostics",
    "declared_sensitivity_analyses",
    "declared_uncertainty",
]
TrialEvalNormalizationStatusV1: TypeAlias = Literal[  # noqa: UP040
    "not_applicable",
    "complete",
    "abstain",
]

_FACTORIAL_CELLS = frozenset(
    {
        ("output_contract_only", "neutral", "structured"),
        ("output_contract_only", "neutral", "narrative"),
        ("unordered_checklist", "neutral", "structured"),
        ("unordered_checklist", "neutral", "narrative"),
        ("ordered_sop", "neutral", "structured"),
        ("ordered_sop", "neutral", "narrative"),
    }
)
_TARGETED_CONDITIONS = frozenset(
    {
        "targeted_covariate_structure",
        "targeted_survival_assumptions",
        "targeted_design_structure",
        "targeted_data_integrity",
        "placebo_deliberation",
    }
)
_TARGETED_CELLS = frozenset(
    {("output_contract_only", condition, "structured") for condition in _TARGETED_CONDITIONS | {"neutral"}}
)
_CAPABILITY_CONDITIONS = frozenset(_TARGETED_CONDITIONS - {"placebo_deliberation"})
EXPERIMENT_IDENTIFIER_PATTERN_V1 = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
_TASK_ID_PATTERN = r"^TASK[A-Z0-9]+$"


def trialeval_metric_population_v1(metric: TrialEvalAblationMetricV1) -> TrialEvalMetricPopulationV1:
    """Return the scientific analysis population for one TrialEval metric."""

    if metric in {
        "usable_primary",
        "route_match",
        "obligations_met",
        "numeric_result_available",
        "primary_analysis_conforms",
        "result_match",
    }:
        return "all_scheduled_assignments"
    if metric == "primary_uncertainty_valid":
        return "uncertainty_applicable_assignments"
    if metric == "primary_interval_agreement":
        return "matched_interval_evaluable_assignments"
    if metric in {"numeric_absolute_error", "numeric_tolerance_ratio"}:
        return "matched_numeric_result_comparisons"
    if metric in {"planning_valid", "planning_usable_with_primary"}:
        return "planning_eligible_assignments"
    return "planning_consequence_evaluable_assignments"


def trialeval_ablation_cells_v1(
    design: Literal["factorial_interface", "targeted_control"],
) -> tuple[
    tuple[
        ProcedureAssistanceV1,
        TrialEvalPromptConditionV1,
        TrialEvalSubmissionInterfaceV1,
    ],
    ...,
]:
    """Return the complete immutable condition/interface cells for a design."""

    cells = _FACTORIAL_CELLS if design == "factorial_interface" else _TARGETED_CELLS
    return tuple(
        sorted(
            cast(
                tuple[
                    ProcedureAssistanceV1,
                    TrialEvalPromptConditionV1,
                    TrialEvalSubmissionInterfaceV1,
                ],
                cell,
            )
            for cell in cells
        )
    )


def trialeval_capability_prompt_conditions_v1() -> tuple[TrialEvalPromptConditionV1, ...]:
    """Return the exact targeted capability prompts requiring applicability labels."""

    return tuple(sorted(cast(TrialEvalPromptConditionV1, condition) for condition in _CAPABILITY_CONDITIONS))


class _ExperimentModelV1(BaseModel):
    """Strict immutable base for experiment artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialEvalAblationAssignmentV1(_ExperimentModelV1):
    """One participant-visible intervention assigned to one task replicate."""

    assignment_id: str = Field(..., pattern=EXPERIMENT_IDENTIFIER_PATTERN_V1)
    task_id: str = Field(..., pattern=_TASK_ID_PATTERN)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    analysis_surface_sha256: str = Field(..., min_length=64, max_length=64)
    replicate_id: str = Field(..., pattern=EXPERIMENT_IDENTIFIER_PATTERN_V1)
    decoding_seed: int = Field(..., strict=True, ge=0)
    procedure_assistance: ProcedureAssistanceV1
    prompt_condition: TrialEvalPromptConditionV1
    submission_interface: TrialEvalSubmissionInterfaceV1

    @model_validator(mode="after")
    def _restrict_targeted_controls(self) -> TrialEvalAblationAssignmentV1:
        TrialEvalEvidenceFactorsV1(
            context_configuration=self.context_tier,
            data_preparation=self.data_preparation,
            analysis_specification=self.analysis_specification,
        )
        if self.prompt_condition in _TARGETED_CONDITIONS and self.submission_interface != "structured":
            raise ValueError("Targeted and placebo controls must use the structured submission interface.")
        if self.prompt_condition in _TARGETED_CONDITIONS and self.procedure_assistance != "output_contract_only":
            raise ValueError("Targeted and placebo controls must use output_contract_only procedure assistance.")
        return self


class TrialEvalAblationScheduleV1(_ExperimentModelV1):
    """Reference-independent schedule for one TrialEval ablation experiment."""

    schema_id: Literal["trialagentbench.trialeval_ablation_schedule/v1"] = (
        "trialagentbench.trialeval_ablation_schedule/v1"
    )
    experiment_id: str = Field(..., min_length=1)
    design: Literal["factorial_interface", "targeted_control"]
    execution_scope: Literal["canary", "pilot", "publication"]
    experiment_design_sha256: str = Field(..., min_length=64, max_length=64)
    participant_release_sha256: str = Field(..., min_length=64, max_length=64)
    prompt_set_sha256: str = Field(..., min_length=64, max_length=64)
    analysis_config_sha256: str = Field(..., min_length=64, max_length=64)
    factorial_task_sample_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    randomization_seed: int = Field(..., strict=True, ge=0)
    assignments: tuple[TrialEvalAblationAssignmentV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_design_and_hash(self) -> TrialEvalAblationScheduleV1:
        if self.execution_scope == "publication" and self.factorial_task_sample_sha256 is None:
            raise ValueError("Publication ablation schedules require a frozen factorial task sample.")
        if len({row.assignment_id for row in self.assignments}) != len(self.assignments):
            raise ValueError("Ablation assignment_id values must be unique.")
        keys = [
            (
                row.task_id,
                row.replicate_id,
                row.analysis_specification,
                row.procedure_assistance,
                row.prompt_condition,
                row.submission_interface,
            )
            for row in self.assignments
        ]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Ablation schedule contains a duplicate task/replicate/assistance/condition/interface cell."
            )

        blocks: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
        block_decoding_seeds: dict[tuple[str, str], int] = {}
        replicate_decoding_seeds: dict[str, int] = {}
        task_factors: dict[str, tuple[str, str, str]] = {}
        for row in self.assignments:
            factors = (row.context_tier, row.data_preparation, row.analysis_specification)
            prior_factors = task_factors.setdefault(row.task_id, factors)
            if prior_factors != factors:
                raise ValueError(f"Task {row.task_id!r} cannot be assigned more than one evidence surface.")
            block = (row.task_id, row.replicate_id)
            blocks.setdefault(block, set()).add(
                (
                    row.procedure_assistance,
                    row.prompt_condition,
                    row.submission_interface,
                )
            )
            prior_block_seed = block_decoding_seeds.setdefault(block, row.decoding_seed)
            if prior_block_seed != row.decoding_seed:
                raise ValueError(f"Ablation block {block!r} must use one decoding seed across all cells.")
            prior_replicate_seed = replicate_decoding_seeds.setdefault(
                row.replicate_id,
                row.decoding_seed,
            )
            if prior_replicate_seed != row.decoding_seed:
                raise ValueError(f"Replicate {row.replicate_id!r} must use one decoding seed across all tasks.")
        if len(set(replicate_decoding_seeds.values())) != len(replicate_decoding_seeds):
            raise ValueError("Ablation replicate decoding seeds must be unique.")
        expected = set(trialeval_ablation_cells_v1(self.design))
        for block, observed in blocks.items():
            if observed != expected:
                raise ValueError(
                    f"Ablation block {block!r} must contain exactly {sorted(expected)!r}; "
                    f"observed {sorted(observed)!r}."
                )

        expected_order = sorted(self.assignments, key=lambda row: row.assignment_id)
        random.Random(self.randomization_seed).shuffle(expected_order)
        if tuple(row.assignment_id for row in self.assignments) != tuple(row.assignment_id for row in expected_order):
            raise ValueError("Ablation assignment order does not match randomization_seed.")

        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        digest = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Ablation schedule checksum does not match its canonical payload.")
        object.__setattr__(self, "checksum", digest)
        return self


class TrialEvalNarrativeSourceSpanV1(_ExperimentModelV1):
    """Exact character span supporting one normalized narrative claim."""

    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)
    text: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_span(self) -> TrialEvalNarrativeSourceSpanV1:
        if self.end <= self.start:
            raise ValueError("Narrative source span end must exceed start.")
        if len(self.text) != self.end - self.start:
            raise ValueError("Narrative source span length does not match its offsets.")
        return self


class TrialEvalNarrativeClaimV1(_ExperimentModelV1):
    """One source-grounded claim extracted from a frozen narrative report."""

    claim_id: str = Field(..., pattern=EXPERIMENT_IDENTIFIER_PATTERN_V1)
    field_path: Literal[
        "primary_analysis.estimand",
        "primary_analysis.estimator",
        "primary_analysis.result_kind",
        "primary_analysis.result",
        "primary_analysis.favorable_direction",
        "primary_analysis.evidence_ids",
        "evidence",
        "limitations",
        "planning",
        "reconstruction",
        "data_integrity_record",
    ]
    claim_role: Literal[
        "primary",
        "sensitivity",
        "secondary",
        "rejected",
        "hypothetical",
        "limitation",
        "ambiguous",
    ]
    evidence_level: Literal["absent", "mentioned", "declared", "executed", "substantiated"]
    spans: tuple[TrialEvalNarrativeSourceSpanV1, ...] = Field(..., min_length=1)
    raw_value: str = Field(..., min_length=1)
    parsed_value: JsonValue
    unit: str | None = Field(default=None, min_length=1)
    orientation: Literal["higher", "lower", "neither"] | None = None
    result_shape: (
        Literal[
            "scalar",
            "identified_interval",
            "vector",
            "test",
            "non_identification",
            "diagnostic_test",
            "diagnostic_summary",
        ]
        | None
    ) = None
    conflict: bool = False
    conflict_group_id: str | None = Field(default=None, pattern=EXPERIMENT_IDENTIFIER_PATTERN_V1)
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_claim(self) -> TrialEvalNarrativeClaimV1:
        if self.evidence_level == "absent":
            raise ValueError("A narrative claim cannot cite absent evidence.")
        if self.raw_value != "\n".join(span.text for span in self.spans):
            raise ValueError("Narrative claim raw_value must equal its ordered source-span text.")
        if self.conflict != (self.conflict_group_id is not None):
            raise ValueError("Narrative claim conflict and conflict_group_id must be declared together.")
        if self.field_path == "primary_analysis.result":
            if not isinstance(self.parsed_value, dict):
                raise ValueError("Primary result claim must contain an object parsed_value.")
            if self.result_shape != self.parsed_value.get("kind"):
                raise ValueError("Primary result claim shape does not match its parsed value.")
            if self.unit != self.parsed_value.get("unit"):
                raise ValueError("Primary result claim unit does not match its parsed value.")
        elif self.result_shape is not None or self.unit is not None:
            raise ValueError("Result shape and unit are valid only for the primary result claim.")
        if self.field_path == "primary_analysis.favorable_direction":
            if self.orientation != self.parsed_value:
                raise ValueError("Favorable-direction claim orientation does not match its parsed value.")
        elif self.orientation is not None:
            raise ValueError("Orientation is valid only for the favorable-direction claim.")
        return self


def _qualifying_narrative_claim_v1(
    claims: tuple[TrialEvalNarrativeClaimV1, ...],
    *,
    field_path: str,
    roles: frozenset[str],
    levels: frozenset[str],
    required: bool,
) -> JsonValue | None:
    eligible = tuple(
        claim
        for claim in claims
        if claim.field_path == field_path
        and claim.claim_role in roles
        and claim.evidence_level in levels
        and not claim.conflict
    )
    if len(eligible) > 1:
        raise ValueError(f"Multiple qualifying narrative claims target {field_path!r}.")
    if not eligible:
        if required:
            raise ValueError(f"Narrative response lacks a qualifying claim for {field_path!r}.")
        return None
    return eligible[0].parsed_value


def _primary_narrative_levels_v1(field_path: str) -> frozenset[str]:
    """Return the evidence levels required by one primary submission field."""

    if field_path in {
        "primary_analysis.estimator",
        "primary_analysis.result_kind",
        "primary_analysis.result",
    }:
        return frozenset({"executed", "substantiated"})
    if field_path in {"primary_analysis.estimand", "primary_analysis.favorable_direction"}:
        return frozenset({"declared", "executed", "substantiated"})
    raise ValueError(f"Unknown primary narrative field path: {field_path!r}.")


def submission_from_narrative_claims_v1(
    *,
    task_id: str,
    claims: tuple[TrialEvalNarrativeClaimV1, ...],
) -> TrialEvalSubmissionV1:
    """Build one canonical submission solely from eligible narrative claims."""

    if any(claim.conflict for claim in claims):
        raise ValueError("Conflicting narrative claims cannot produce a canonical submission.")
    primary_roles = frozenset({"primary"})
    executed = frozenset({"executed", "substantiated"})
    primary_analysis: dict[str, JsonValue] = {
        "declared_primary": True,
        "estimand": _qualifying_narrative_claim_v1(
            claims,
            field_path="primary_analysis.estimand",
            roles=primary_roles,
            levels=_primary_narrative_levels_v1("primary_analysis.estimand"),
            required=True,
        ),
        "estimator": _qualifying_narrative_claim_v1(
            claims,
            field_path="primary_analysis.estimator",
            roles=primary_roles,
            levels=_primary_narrative_levels_v1("primary_analysis.estimator"),
            required=True,
        ),
        "result_kind": _qualifying_narrative_claim_v1(
            claims,
            field_path="primary_analysis.result_kind",
            roles=primary_roles,
            levels=_primary_narrative_levels_v1("primary_analysis.result_kind"),
            required=True,
        ),
        "result": _qualifying_narrative_claim_v1(
            claims,
            field_path="primary_analysis.result",
            roles=primary_roles,
            levels=_primary_narrative_levels_v1("primary_analysis.result"),
            required=True,
        ),
        "favorable_direction": _qualifying_narrative_claim_v1(
            claims,
            field_path="primary_analysis.favorable_direction",
            roles=primary_roles,
            levels=_primary_narrative_levels_v1("primary_analysis.favorable_direction"),
            required=True,
        ),
        "evidence_ids": _qualifying_narrative_claim_v1(
            claims,
            field_path="primary_analysis.evidence_ids",
            roles=primary_roles,
            levels=executed,
            required=False,
        )
        or [],
    }
    limitations = _qualifying_narrative_claim_v1(
        claims,
        field_path="limitations",
        roles=frozenset({"limitation"}),
        levels=frozenset({"declared", "executed", "substantiated"}),
        required=True,
    )
    payload: dict[str, JsonValue] = {
        "schema_id": "trialagentbench.trialeval_submission/v1",
        "task_id": task_id,
        "primary_analysis": primary_analysis,
        "limitations": limitations,
    }
    optional = {
        "evidence": (frozenset({"primary", "sensitivity", "secondary"}), executed),
        "planning": (primary_roles, executed),
        "reconstruction": (primary_roles, executed),
        "data_integrity_record": (primary_roles, executed),
    }
    for field_path, (roles, levels) in optional.items():
        value = _qualifying_narrative_claim_v1(
            claims,
            field_path=field_path,
            roles=roles,
            levels=levels,
            required=False,
        )
        if value is not None:
            payload[field_path] = value
    return TrialEvalSubmissionV1.model_validate(payload)


class TrialEvalNarrativeTranscriptionV1(_ExperimentModelV1):
    """Source-grounded canonical transcription of one frozen narrative report."""

    schema_id: Literal["trialagentbench.trialeval_narrative_transcription/v1"] = (
        "trialagentbench.trialeval_narrative_transcription/v1"
    )
    assignment_id: str = Field(..., min_length=1)
    report_sha256: str = Field(..., min_length=64, max_length=64)
    source: Literal["manual_masked", "automated_importer"]
    source_identity: str = Field(..., min_length=1)
    transcriber_identities: tuple[str, ...] = ()
    transcription_disposition: Literal["independent_exact_agreement", "adjudicated_resolution"] | None = None
    blinded_to_model_identity: bool
    blinded_to_evaluator_reference: bool
    importer_prompt_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    importer_schema_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    importer_response_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    status: Literal["complete", "abstain"]
    submission: TrialEvalSubmissionV1 | None = None
    claims: tuple[TrialEvalNarrativeClaimV1, ...] = ()
    abstention_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_transcription_role(self) -> TrialEvalNarrativeTranscriptionV1:
        if self.source == "manual_masked":
            if not self.blinded_to_model_identity or not self.blinded_to_evaluator_reference:
                raise ValueError("Manual transcription must be blinded to model identity and evaluator reference.")
            if self.importer_prompt_sha256 is not None:
                raise ValueError("Manual transcription cannot declare an importer prompt hash.")
            if self.importer_schema_sha256 is not None or self.importer_response_sha256 is not None:
                raise ValueError("Manual transcription cannot declare automated-importer provenance.")
            normalized_transcribers = tuple(identity.strip() for identity in self.transcriber_identities)
            if len(normalized_transcribers) < 2 or len(set(normalized_transcribers)) != len(normalized_transcribers):
                raise ValueError("Manual transcription requires at least two unique transcriber identities.")
            if any(not identity for identity in normalized_transcribers):
                raise ValueError("Manual transcription transcriber identities cannot be blank.")
            if self.transcription_disposition is None:
                raise ValueError("Manual transcription requires an agreement or adjudication disposition.")
        elif (
            self.importer_prompt_sha256 is None
            or self.importer_schema_sha256 is None
            or self.importer_response_sha256 is None
        ):
            raise ValueError("Automated transcription requires immutable prompt, schema, and response hashes.")
        elif self.transcriber_identities or self.transcription_disposition is not None:
            raise ValueError("Automated transcription cannot declare manual-transcription metadata.")

        if self.status == "abstain":
            if self.submission is not None:
                raise ValueError("Abstaining transcription cannot contain a submission.")
            if self.abstention_reason is None:
                raise ValueError("Abstaining transcription requires abstention_reason.")
            return self

        if self.submission is None:
            raise ValueError("Complete transcription requires a canonical submission.")
        if self.abstention_reason is not None:
            raise ValueError("Complete transcription cannot contain abstention_reason.")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Narrative transcription claim IDs must be unique.")
        if any(claim.conflict for claim in self.claims):
            raise ValueError("Complete transcription cannot contain conflicting claims.")
        required = {
            "primary_analysis.estimand",
            "primary_analysis.estimator",
            "primary_analysis.result",
            "primary_analysis.favorable_direction",
            "limitations",
        }
        primary_required = required - {"limitations"}
        qualifying_paths = {
            claim.field_path
            for claim in self.claims
            if claim.field_path in primary_required
            and claim.claim_role == "primary"
            and claim.evidence_level in _primary_narrative_levels_v1(claim.field_path)
        }
        missing = sorted(required.difference(qualifying_paths | {"limitations"}))
        if not any(
            claim.field_path == "limitations"
            and claim.claim_role == "limitation"
            and claim.evidence_level in {"declared", "executed", "substantiated"}
            for claim in self.claims
        ):
            missing.append("limitations")
        if missing:
            raise ValueError(f"Complete transcription lacks source-grounded claims for fields: {missing!r}.")
        for path in primary_required:
            count = sum(
                claim.field_path == path
                and claim.claim_role == "primary"
                and claim.evidence_level in _primary_narrative_levels_v1(path)
                for claim in self.claims
            )
            if count != 1:
                raise ValueError(f"Complete transcription requires exactly one qualifying claim for {path!r}.")
        derived = submission_from_narrative_claims_v1(task_id=self.submission.task_id, claims=self.claims)
        if self.submission != derived:
            raise ValueError("Transcribed submission differs from its source-grounded narrative claims.")
        return self


def _validate_normalization_disposition_v1(
    *,
    submission_interface: TrialEvalSubmissionInterfaceV1,
    normalization_status: TrialEvalNormalizationStatusV1,
    normalization_failure_reason: str | None,
    omitted_required_deliverables: tuple[TrialEvalSemanticDeliverableV1, ...],
    primary_failure_code: FailureCode | None,
    primary_analysis_conforms: bool,
) -> None:
    """Validate one endpoint's response and normalization disposition."""

    if submission_interface == "structured" and normalization_status != "not_applicable":
        raise ValueError("Structured responses require not_applicable normalization status.")
    if submission_interface == "narrative" and normalization_status == "not_applicable":
        raise ValueError("Narrative responses require complete or abstain normalization status.")
    if (normalization_status == "abstain") != (normalization_failure_reason is not None):
        raise ValueError("A normalization failure reason is present exactly when normalization abstains.")
    if tuple(sorted(set(omitted_required_deliverables))) != omitted_required_deliverables:
        raise ValueError("Omitted required deliverables must be unique and sorted.")
    if primary_analysis_conforms != (primary_failure_code is None):
        raise ValueError("A primary failure code is present exactly when the primary analysis does not conform.")
    if normalization_status == "abstain" and primary_failure_code != "missing_primary_submission":
        raise ValueError("Normalization abstention requires a missing-primary failure.")
    if omitted_required_deliverables and primary_failure_code not in {
        "missing_primary_submission",
        "missing_required_deliverable",
    }:
        raise ValueError("Omitted required deliverables require a submission-completeness failure.")
    if primary_failure_code == "missing_required_deliverable" and not omitted_required_deliverables:
        raise ValueError("A missing-deliverable failure must identify at least one omitted deliverable.")


class TrialEvalAblationGradeRowV1(_ExperimentModelV1):
    """One grader-derived endpoint row joined to an immutable assignment."""

    assignment_id: str = Field(..., min_length=1)
    task_id: str = Field(..., pattern=_TASK_ID_PATTERN)
    base_trial_id: str = Field(..., min_length=1)
    regime_cell_id: str = Field(..., min_length=1)
    evaluation_series_id: str = Field(..., min_length=1)
    design_tier: Literal["D1", "D2", "D3", "D4"]
    design_subtype: Literal[
        "individual_randomized",
        "pragmatic",
        "covariate_structure",
        "endpoint_ascertainment",
        "cluster_parallel",
        "stepped_wedge",
        "group_sequential",
    ]
    assumption_tier: Literal["A1", "A2", "A3", "A4"]
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    model_id: str = Field(..., min_length=1)
    replicate_id: str = Field(..., min_length=1)
    procedure_assistance: ProcedureAssistanceV1
    prompt_condition: TrialEvalPromptConditionV1
    submission_interface: TrialEvalSubmissionInterfaceV1
    normalization_source: Literal["direct_structured", "manual_masked", "automated_importer"]
    normalization_status: TrialEvalNormalizationStatusV1 = "not_applicable"
    normalization_failure_reason: str | None = Field(default=None, min_length=1)
    omitted_required_deliverables: tuple[TrialEvalSemanticDeliverableV1, ...] = ()
    primary_failure_code: FailureCode | None = None
    targeted_applicability: Literal["applicable", "mismatched", "inapplicable"] | None = None
    usable_primary: bool
    route_match: bool
    obligations_met: bool
    credit_eligible_route_count: int = Field(ge=1)
    numeric_result_available: bool
    primary_uncertainty_valid: bool | None = None
    primary_interval_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    result_match: bool
    numeric_absolute_error: float | None = Field(default=None, ge=0.0)
    numeric_tolerance_ratio: float | None = Field(default=None, ge=0.0)
    primary_analysis_conforms: bool
    planning_applicable: bool
    planning_valid: bool | None = None
    planning_usable_with_primary: bool | None = None
    planning_achieved_power: float | None = Field(default=None, ge=0.0, le=1.0)
    planning_power_shortfall: float | None = Field(default=None, ge=0.0, le=1.0)
    planning_underpowered: bool | None = None
    planning_proportional_participant_deviation: float | None = None
    planning_log_sample_size_ratio: float | None = None
    planning_event_shortage: int | None = Field(default=None, ge=0)
    planning_excess_events: int | None = Field(default=None, ge=0)
    planning_excess_participants: int | None = Field(default=None, ge=0)
    planning_participant_shortage: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_analysis_surface(self) -> TrialEvalAblationGradeRowV1:
        TrialEvalEvidenceFactorsV1(
            context_configuration=self.context_tier,
            data_preparation=self.data_preparation,
            analysis_specification=self.analysis_specification,
        )
        if self.submission_interface == "structured" and self.normalization_source != "direct_structured":
            raise ValueError("Structured responses require direct_structured normalization_source.")
        if self.submission_interface == "narrative" and self.normalization_source == "direct_structured":
            raise ValueError("Narrative responses require manual or automated normalization.")
        _validate_normalization_disposition_v1(
            submission_interface=self.submission_interface,
            normalization_status=self.normalization_status,
            normalization_failure_reason=self.normalization_failure_reason,
            omitted_required_deliverables=self.omitted_required_deliverables,
            primary_failure_code=self.primary_failure_code,
            primary_analysis_conforms=self.primary_analysis_conforms,
        )
        is_targeted = self.prompt_condition in _CAPABILITY_CONDITIONS
        if is_targeted != (self.targeted_applicability is not None):
            raise ValueError("Applicability is required exactly for targeted capability conditions.")
        if self.primary_interval_agreement is not None and (
            not self.route_match or self.primary_uncertainty_valid is not True
        ):
            raise ValueError("Interval agreement requires a matched method route and valid uncertainty contract.")
        if (self.numeric_absolute_error is None) != (self.numeric_tolerance_ratio is None):
            raise ValueError("Numeric absolute error and tolerance ratio must be reported together.")
        if self.numeric_absolute_error is not None and not self.route_match:
            raise ValueError("Numeric error requires a matched analysis route.")
        if self.route_match and not self.usable_primary:
            raise ValueError("Route match requires a usable primary submission.")
        if self.obligations_met and not self.route_match:
            raise ValueError("Diagnostic obligations can be met only for a matched route.")
        if self.result_match and not self.obligations_met:
            raise ValueError("Result match requires a matched route with satisfied obligations.")
        if self.primary_analysis_conforms and not (
            self.usable_primary and self.route_match and self.obligations_met and self.result_match
        ):
            raise ValueError("Credit-eligible primary results require the complete noncompensatory cascade.")
        self._validate_planning_consequences()
        return self

    def _validate_planning_consequences(self) -> None:
        values = (
            self.planning_valid,
            self.planning_usable_with_primary,
            self.planning_achieved_power,
            self.planning_power_shortfall,
            self.planning_underpowered,
            self.planning_proportional_participant_deviation,
            self.planning_log_sample_size_ratio,
            self.planning_event_shortage,
            self.planning_excess_events,
            self.planning_excess_participants,
            self.planning_participant_shortage,
        )
        if not self.planning_applicable and any(value is not None for value in values):
            raise ValueError("Planning consequences must be absent when planning is not applicable.")
        if self.planning_applicable and self.planning_valid is None:
            raise ValueError("Applicable planning requires an explicit planning_valid result.")


class TrialEvalAblationEndpointRowV1(_ExperimentModelV1):
    """One scored endpoint before evaluator-only applicability is joined."""

    assignment_id: str = Field(..., min_length=1)
    task_id: str = Field(..., pattern=_TASK_ID_PATTERN)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    model_id: str = Field(..., min_length=1)
    replicate_id: str = Field(..., min_length=1)
    procedure_assistance: ProcedureAssistanceV1
    prompt_condition: TrialEvalPromptConditionV1
    submission_interface: TrialEvalSubmissionInterfaceV1
    normalization_source: Literal["direct_structured", "manual_masked", "automated_importer"]
    normalization_status: TrialEvalNormalizationStatusV1 = "not_applicable"
    normalization_failure_reason: str | None = Field(default=None, min_length=1)
    omitted_required_deliverables: tuple[TrialEvalSemanticDeliverableV1, ...] = ()
    primary_failure_code: FailureCode | None = None
    usable_primary: bool
    route_match: bool
    obligations_met: bool
    credit_eligible_route_count: int = Field(ge=1)
    numeric_result_available: bool
    primary_uncertainty_valid: bool | None = None
    primary_interval_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    result_match: bool
    numeric_absolute_error: float | None = Field(default=None, ge=0.0)
    numeric_tolerance_ratio: float | None = Field(default=None, ge=0.0)
    primary_analysis_conforms: bool
    planning_applicable: bool
    planning_valid: bool | None = None
    planning_usable_with_primary: bool | None = None
    planning_achieved_power: float | None = Field(default=None, ge=0.0, le=1.0)
    planning_power_shortfall: float | None = Field(default=None, ge=0.0, le=1.0)
    planning_underpowered: bool | None = None
    planning_proportional_participant_deviation: float | None = None
    planning_log_sample_size_ratio: float | None = None
    planning_event_shortage: int | None = Field(default=None, ge=0)
    planning_excess_events: int | None = Field(default=None, ge=0)
    planning_excess_participants: int | None = Field(default=None, ge=0)
    planning_participant_shortage: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_normalization_source(self) -> TrialEvalAblationEndpointRowV1:
        TrialEvalEvidenceFactorsV1(
            context_configuration=self.context_tier,
            data_preparation=self.data_preparation,
            analysis_specification=self.analysis_specification,
        )
        if self.submission_interface == "structured" and self.normalization_source != "direct_structured":
            raise ValueError("Structured endpoints require direct_structured normalization_source.")
        if self.submission_interface == "narrative" and self.normalization_source == "direct_structured":
            raise ValueError("Narrative endpoints require manual or automated normalization.")
        _validate_normalization_disposition_v1(
            submission_interface=self.submission_interface,
            normalization_status=self.normalization_status,
            normalization_failure_reason=self.normalization_failure_reason,
            omitted_required_deliverables=self.omitted_required_deliverables,
            primary_failure_code=self.primary_failure_code,
            primary_analysis_conforms=self.primary_analysis_conforms,
        )
        if self.primary_interval_agreement is not None and (
            not self.route_match or self.primary_uncertainty_valid is not True
        ):
            raise ValueError("Interval agreement requires a matched method route and valid uncertainty contract.")
        if (self.numeric_absolute_error is None) != (self.numeric_tolerance_ratio is None):
            raise ValueError("Numeric absolute error and tolerance ratio must be reported together.")
        if self.numeric_absolute_error is not None and not self.route_match:
            raise ValueError("Numeric error requires a matched analysis route.")
        if self.route_match and not self.usable_primary:
            raise ValueError("Route match requires a usable primary submission.")
        if self.obligations_met and not self.route_match:
            raise ValueError("Diagnostic obligations can be met only for a matched route.")
        if self.result_match and not self.obligations_met:
            raise ValueError("Result match requires a matched route with satisfied obligations.")
        if self.primary_analysis_conforms and not (
            self.usable_primary and self.route_match and self.obligations_met and self.result_match
        ):
            raise ValueError("Credit-eligible primary results require the complete noncompensatory cascade.")
        values = (
            self.planning_valid,
            self.planning_usable_with_primary,
            self.planning_achieved_power,
            self.planning_power_shortfall,
            self.planning_underpowered,
            self.planning_proportional_participant_deviation,
            self.planning_log_sample_size_ratio,
            self.planning_event_shortage,
            self.planning_excess_events,
            self.planning_excess_participants,
            self.planning_participant_shortage,
        )
        if not self.planning_applicable and any(value is not None for value in values):
            raise ValueError("Planning consequences must be absent when planning is not applicable.")
        if self.planning_applicable and self.planning_valid is None:
            raise ValueError("Applicable planning requires an explicit planning_valid result.")
        return self


class TrialEvalAblationEndpointSetV1(_ExperimentModelV1):
    """Checksummed scored endpoints from one frozen ablation run."""

    schema_id: Literal["trialagentbench.trialeval_ablation_endpoint_set/v1"] = (
        "trialagentbench.trialeval_ablation_endpoint_set/v1"
    )
    experiment_id: str = Field(..., min_length=1)
    schedule_checksum: str = Field(..., min_length=64, max_length=64)
    evaluator_release_sha256: str = Field(..., min_length=64, max_length=64)
    scoring_implementation_sha256: str = Field(..., min_length=64, max_length=64)
    endpoints: tuple[TrialEvalAblationEndpointRowV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_endpoint_set(self) -> TrialEvalAblationEndpointSetV1:
        ordered = tuple(sorted(self.endpoints, key=lambda row: (row.assignment_id, row.normalization_source)))
        keys = [(row.assignment_id, row.normalization_source) for row in ordered]
        if len(keys) != len(set(keys)):
            raise ValueError("Ablation endpoint set contains duplicate assignment/normalization rows.")
        object.__setattr__(self, "endpoints", ordered)
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        digest = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Ablation endpoint-set checksum does not match its canonical payload.")
        object.__setattr__(self, "checksum", digest)
        return self


class TrialEvalAblationObservableRowV1(_ExperimentModelV1):
    """Runner-observed process measures for one randomized assignment."""

    assignment_id: str = Field(..., pattern=EXPERIMENT_IDENTIFIER_PATTERN_V1)
    task_id: str = Field(..., pattern=_TASK_ID_PATTERN)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    model_id: str = Field(..., min_length=1)
    replicate_id: str = Field(..., pattern=EXPERIMENT_IDENTIFIER_PATTERN_V1)
    procedure_assistance: ProcedureAssistanceV1
    prompt_condition: TrialEvalPromptConditionV1
    submission_interface: TrialEvalSubmissionInterfaceV1
    answer_submitted: bool
    public_data_inspected: bool
    code_executed_successfully: bool
    assistant_turns: int = Field(..., ge=0)
    tool_calls: int = Field(..., ge=0)
    file_inspections: int = Field(..., ge=0)
    code_executions: int = Field(..., ge=0)
    failed_code_executions: int = Field(..., ge=0)
    truncated_outputs: int = Field(..., ge=0)
    events_until_first_data_inspection: int = Field(..., gt=0)
    events_until_first_successful_code_execution: int = Field(..., gt=0)
    events_until_submission: int = Field(..., gt=0)
    execution_elapsed_seconds: float = Field(..., ge=0.0)
    provider_response_count: int = Field(..., ge=0)
    provider_responses_with_usage: int = Field(..., ge=0)
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    provider_elapsed_seconds: float = Field(..., ge=0.0)
    declared_diagnostics: int | None = Field(default=None, ge=0)
    declared_sensitivity_analyses: int | None = Field(default=None, ge=0)
    declared_uncertainty: bool | None = None

    @model_validator(mode="after")
    def _restrict_declared_content_to_structured_output(self) -> TrialEvalAblationObservableRowV1:
        TrialEvalEvidenceFactorsV1(
            context_configuration=self.context_tier,
            data_preparation=self.data_preparation,
            analysis_specification=self.analysis_specification,
        )
        if self.provider_responses_with_usage > self.provider_response_count:
            raise ValueError("Provider responses with usage cannot exceed total provider responses.")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("TrialEval total_tokens must equal prompt_tokens plus completion_tokens.")
        if self.provider_response_count != self.assistant_turns:
            raise ValueError("TrialEval provider responses must equal observed assistant turns.")
        declared = (
            self.declared_diagnostics,
            self.declared_sensitivity_analyses,
            self.declared_uncertainty,
        )
        if self.submission_interface == "narrative" and any(value is not None for value in declared):
            raise ValueError("Narrative process rows require masked normalization before content claims.")
        if self.submission_interface == "structured" and self.answer_submitted != all(
            value is not None for value in declared
        ):
            raise ValueError("Structured content measures are present exactly when an answer was submitted.")
        return self


class TrialEvalAblationObservableContrastV1(_ExperimentModelV1):
    """One paired process contrast over exact randomized blocks."""

    model_id: str = Field(..., min_length=1)
    metric: TrialEvalObservableMetricV1
    contrast_id: Literal[
        "unordered_checklist_minus_output_contract_only",
        "ordered_sop_minus_unordered_checklist",
        "ordered_sop_minus_output_contract_only",
    ]
    analysis_specification: TrialEvalAnalysisSpecificationV1
    submission_interface: TrialEvalSubmissionInterfaceV1
    n_blocks: int = Field(..., gt=0)
    n_base_trial_clusters: int = Field(..., gt=1)
    n_decoding_replicates: int = Field(..., gt=1)
    estimate: float
    interval_low: float
    interval_high: float
    confidence_level: float = Field(..., gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _ordered_interval(self) -> TrialEvalAblationObservableContrastV1:
        if self.interval_low > self.interval_high:
            raise ValueError("Observable contrast interval_low exceeds interval_high.")
        return self


class TrialEvalAblationArmSummaryV1(_ExperimentModelV1):
    """Absolute performance in one TrialEval assistance/interface arm."""

    model_id: str = Field(..., min_length=1)
    metric: TrialEvalAblationMetricV1
    analysis_population: TrialEvalMetricPopulationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    procedure_assistance: ProcedureAssistanceV1
    submission_interface: TrialEvalSubmissionInterfaceV1
    n_assignments: int = Field(..., gt=0)
    n_base_trial_clusters: int = Field(..., gt=1)
    n_decoding_replicates: int = Field(..., gt=1)
    estimate: float
    interval_low: float
    interval_high: float
    confidence_level: float = Field(..., gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _ordered_interval(self) -> TrialEvalAblationArmSummaryV1:
        if self.interval_low > self.interval_high:
            raise ValueError("TrialEval arm-summary interval_low exceeds interval_high.")
        expected = trialeval_metric_population_v1(self.metric)
        if self.analysis_population != expected:
            raise ValueError(
                f"Metric {self.metric!r} requires analysis_population={expected!r}, not {self.analysis_population!r}."
            )
        return self


class TrialEvalAblationTaskIdentityV1(_ExperimentModelV1):
    """Evaluator-owned scientific strata for one opaque participant task."""

    task_id: str = Field(..., pattern=_TASK_ID_PATTERN)
    base_trial_id: str = Field(..., min_length=1)
    regime_cell_id: str = Field(..., min_length=1)
    evaluation_series_id: str = Field(..., min_length=1)
    design_tier: Literal["D1", "D2", "D3", "D4"]
    design_subtype: Literal[
        "individual_randomized",
        "pragmatic",
        "covariate_structure",
        "endpoint_ascertainment",
        "cluster_parallel",
        "stepped_wedge",
        "group_sequential",
    ]
    assumption_tier: Literal["A1", "A2", "A3", "A4"]
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1

    @model_validator(mode="after")
    def _validate_evidence_factors(self) -> TrialEvalAblationTaskIdentityV1:
        TrialEvalEvidenceFactorsV1(
            context_configuration=self.context_tier,
            data_preparation=self.data_preparation,
            analysis_specification=self.analysis_specification,
        )
        return self


class TrialEvalTargetedApplicabilityLabelV1(_ExperimentModelV1):
    """Evaluator-only applicability label for one task and capability prompt."""

    task_id: str = Field(..., pattern=_TASK_ID_PATTERN)
    prompt_condition: Literal[
        "targeted_covariate_structure",
        "targeted_survival_assumptions",
        "targeted_design_structure",
        "targeted_data_integrity",
    ]
    applicability: Literal["applicable", "mismatched", "inapplicable"]
    evidence_basis: tuple[str, ...] = Field(..., min_length=1)


class TrialEvalAblationEvaluatorLabelsV1(_ExperimentModelV1):
    """Checksummed evaluator identities and optional targeted applicability labels."""

    schema_id: Literal["trialagentbench.trialeval_ablation_evaluator_labels/v1"] = (
        "trialagentbench.trialeval_ablation_evaluator_labels/v1"
    )
    evaluator_release_sha256: str = Field(..., min_length=64, max_length=64)
    task_identities: tuple[TrialEvalAblationTaskIdentityV1, ...] = Field(..., min_length=1)
    labels: tuple[TrialEvalTargetedApplicabilityLabelV1, ...] = ()
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_complete_task_blocks(self) -> TrialEvalAblationEvaluatorLabelsV1:
        identities = tuple(sorted(self.task_identities, key=lambda row: row.task_id))
        identity_task_ids = [row.task_id for row in identities]
        if len(identity_task_ids) != len(set(identity_task_ids)):
            raise ValueError("Ablation evaluator task identities contain duplicate task_id values.")
        identity_task_id_set = set(identity_task_ids)
        ordered = tuple(sorted(self.labels, key=lambda row: (row.task_id, row.prompt_condition)))
        keys = [(row.task_id, row.prompt_condition) for row in ordered]
        if len(keys) != len(set(keys)):
            raise ValueError("Ablation evaluator labels contain duplicate task/condition rows.")
        by_task: dict[str, set[str]] = {}
        for row in ordered:
            if row.task_id not in identity_task_id_set:
                raise ValueError(f"Applicability label refers to unknown task {row.task_id!r}.")
            by_task.setdefault(row.task_id, set()).add(row.prompt_condition)
        for task_id, conditions in by_task.items():
            if conditions != _CAPABILITY_CONDITIONS:
                raise ValueError(
                    f"Task {task_id!r} requires all capability conditions; observed {sorted(conditions)!r}."
                )
        object.__setattr__(self, "task_identities", identities)
        object.__setattr__(self, "labels", ordered)
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        digest = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Ablation evaluator-label checksum does not match its canonical payload.")
        object.__setattr__(self, "checksum", digest)
        return self


class TrialEvalRepresentationFixtureV1(_ExperimentModelV1):
    """One fixed semantic answer rendered through both response interfaces."""

    fixture_id: str = Field(..., min_length=1)
    task_id: str = Field(..., pattern=_TASK_ID_PATTERN)
    structured_submission: TrialEvalSubmissionV1
    narrative_report: str = Field(..., min_length=1)
    manual_transcription: TrialEvalNarrativeTranscriptionV1
    automated_transcription: TrialEvalNarrativeTranscriptionV1 | None = None

    @model_validator(mode="after")
    def _validate_fixture_identity(self) -> TrialEvalRepresentationFixtureV1:
        if self.structured_submission.task_id != self.task_id:
            raise ValueError("Representation fixture structured submission has the wrong task_id.")
        if self.manual_transcription.source != "manual_masked":
            raise ValueError("Representation fixture manual_transcription must be manual_masked.")
        if self.automated_transcription is not None and self.automated_transcription.source != "automated_importer":
            raise ValueError("Representation fixture automated_transcription must use automated_importer source.")
        return self


class TrialEvalRepresentationFidelityRowV1(_ExperimentModelV1):
    """Exact semantic parity result for one fixed-answer representation fixture."""

    fixture_id: str
    task_id: str = Field(..., pattern=_TASK_ID_PATTERN)
    manual_exact_match: Literal[True]
    automated_status: Literal["not_run", "abstain", "complete"]
    automated_exact_match: bool | None = None

    @model_validator(mode="after")
    def _validate_automated_result(self) -> TrialEvalRepresentationFidelityRowV1:
        if (self.automated_status == "complete") != (self.automated_exact_match is not None):
            raise ValueError("Automated exact-match result is present exactly when automated_status is complete.")
        return self


class TrialEvalRouteMultiplicitySummaryV1(_ExperimentModelV1):
    """Accepted-primary performance by prospective route multiplicity."""

    model_id: str = Field(min_length=1)
    analysis_specification: TrialEvalAnalysisSpecificationV1
    procedure_assistance: ProcedureAssistanceV1
    submission_interface: TrialEvalSubmissionInterfaceV1
    normalization_source: Literal["direct_structured", "manual_masked", "automated_importer"]
    route_multiplicity: Literal["single_route", "plural_route"]
    n_assignments: int = Field(gt=0)
    n_conforming: int = Field(ge=0)
    accepted_rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _coherent_rate(self) -> TrialEvalRouteMultiplicitySummaryV1:
        if self.n_conforming > self.n_assignments:
            raise ValueError("Accepted count cannot exceed the route-multiplicity denominator.")
        if abs(self.accepted_rate - self.n_conforming / self.n_assignments) > 1e-12:
            raise ValueError("Route-multiplicity rate does not match its counts.")
        return self


class TrialEvalToleranceSensitivitySummaryV1(_ExperimentModelV1):
    """Numerical-equivalence sensitivity among matched numeric comparisons."""

    model_id: str = Field(min_length=1)
    analysis_specification: TrialEvalAnalysisSpecificationV1
    procedure_assistance: ProcedureAssistanceV1
    submission_interface: TrialEvalSubmissionInterfaceV1
    normalization_source: Literal["direct_structured", "manual_masked", "automated_importer"]
    tolerance_policy: Literal[
        "half_qualified_envelope",
        "qualified_envelope",
        "double_qualified_envelope",
    ]
    n_matched_numeric_comparisons: int = Field(gt=0)
    n_within_envelope: int = Field(ge=0)
    within_envelope_rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _coherent_rate(self) -> TrialEvalToleranceSensitivitySummaryV1:
        if self.n_within_envelope > self.n_matched_numeric_comparisons:
            raise ValueError("Numerically equivalent count cannot exceed its denominator.")
        expected = self.n_within_envelope / self.n_matched_numeric_comparisons
        if abs(self.within_envelope_rate - expected) > 1e-12:
            raise ValueError("Tolerance-sensitivity rate does not match its counts.")
        return self


class TrialEvalAblationAnalysisReportV1(_ExperimentModelV1):
    """Checksummed output from one prospective ablation analysis."""

    schema_id: Literal["trialagentbench.trialeval_ablation_analysis/v1"] = (
        "trialagentbench.trialeval_ablation_analysis/v1"
    )
    experiment_id: str = Field(..., min_length=1)
    design: Literal["factorial_interface", "targeted_control"]
    schedule_checksum: str = Field(..., min_length=64, max_length=64)
    endpoint_set_checksum: str = Field(..., min_length=64, max_length=64)
    evaluator_labels_checksum: str = Field(..., min_length=64, max_length=64)
    analysis_config: TrialEvalAblationAnalysisConfigV1
    grade_rows: tuple[TrialEvalAblationGradeRowV1, ...] = Field(..., min_length=1)
    arm_summaries: tuple[TrialEvalAblationArmSummaryV1, ...] = ()
    contrasts: tuple[TrialEvalAblationContrastV1, ...] = Field(..., min_length=1)
    observable_rows: tuple[TrialEvalAblationObservableRowV1, ...] = Field(..., min_length=1)
    observable_contrasts: tuple[TrialEvalAblationObservableContrastV1, ...] = ()
    resource_attainment: tuple[BenchmarkResourceAttainmentPointV1, ...] = Field(..., min_length=1)
    route_multiplicity: tuple[TrialEvalRouteMultiplicitySummaryV1, ...] = Field(
        ...,
        min_length=1,
    )
    tolerance_sensitivity: tuple[TrialEvalToleranceSensitivitySummaryV1, ...] = ()
    representation_fidelity: tuple[TrialEvalRepresentationFidelityRowV1, ...] = ()
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _canonicalize_and_hash(self) -> TrialEvalAblationAnalysisReportV1:
        object.__setattr__(
            self,
            "grade_rows",
            tuple(sorted(self.grade_rows, key=lambda row: (row.assignment_id, row.normalization_source))),
        )
        object.__setattr__(
            self,
            "arm_summaries",
            tuple(
                sorted(
                    self.arm_summaries,
                    key=lambda row: (
                        row.model_id,
                        row.metric,
                        row.analysis_specification,
                        row.procedure_assistance,
                        row.submission_interface,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "contrasts",
            tuple(
                sorted(
                    self.contrasts,
                    key=lambda row: (
                        row.model_id,
                        row.metric,
                        row.contrast_id,
                        row.analysis_specification or "",
                        row.prompt_condition or "",
                        row.applicability or "",
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "representation_fidelity",
            tuple(sorted(self.representation_fidelity, key=lambda row: row.fixture_id)),
        )
        object.__setattr__(
            self,
            "observable_rows",
            tuple(sorted(self.observable_rows, key=lambda row: (row.model_id, row.assignment_id))),
        )
        object.__setattr__(
            self,
            "observable_contrasts",
            tuple(
                sorted(
                    self.observable_contrasts,
                    key=lambda row: (
                        row.model_id,
                        row.analysis_specification,
                        row.submission_interface,
                        row.contrast_id,
                        row.metric,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "resource_attainment",
            validate_resource_attainment_v1(self.resource_attainment),
        )
        if any(row.benchmark != "trialeval" for row in self.resource_attainment):
            raise ValueError("TrialEval analysis cannot contain non-TrialEval resource attainment.")
        expected_resource_strata = {
            (
                row.model_id,
                row.analysis_specification,
                row.procedure_assistance,
                row.prompt_condition,
                row.submission_interface,
            )
            for row in self.grade_rows
        }
        observed_resource_strata = {
            (
                row.model_id,
                row.analysis_specification,
                row.procedure_assistance,
                row.prompt_condition,
                row.submission_interface,
            )
            for row in self.resource_attainment
        }
        if observed_resource_strata != expected_resource_strata:
            raise ValueError("TrialEval resource attainment does not cover the exact scored treatment strata.")
        if self.design != self.analysis_config.design:
            raise ValueError("Ablation report design does not match its prospective analysis contract.")
        grade_assignments = {row.assignment_id for row in self.grade_rows}
        observable_assignments = {row.assignment_id for row in self.observable_rows}
        if grade_assignments != observable_assignments:
            raise ValueError("Ablation process rows must cover exactly the scored assignment denominator.")
        if self.design == "factorial_interface" and not self.observable_contrasts:
            raise ValueError("Factorial ablation reports require paired observable-process contrasts.")
        if self.design == "targeted_control" and self.observable_contrasts:
            raise ValueError("Targeted-control reports cannot contain assistance-dose process contrasts.")
        summary_keys = [
            (
                row.model_id,
                row.metric,
                row.analysis_specification,
                row.procedure_assistance,
                row.submission_interface,
            )
            for row in self.arm_summaries
        ]
        if len(summary_keys) != len(set(summary_keys)):
            raise ValueError("TrialEval ablation report contains duplicate arm summaries.")
        primary = self.analysis_config.primary_estimand
        model_ids = {row.model_id for row in self.grade_rows}
        primary_rows = [
            row
            for row in self.contrasts
            if row.metric == primary.metric
            and row.contrast_id == primary.contrast_id
            and row.analysis_specification == primary.analysis_specification
            and row.prompt_condition == primary.prompt_condition
            and row.applicability == primary.applicability
        ]
        if {row.model_id for row in primary_rows} != model_ids or len(primary_rows) != len(model_ids):
            raise ValueError("Ablation report requires exactly one prospective primary contrast per model.")
        if self.design == "factorial_interface":
            primary_arm_keys = {
                (
                    row.model_id,
                    row.analysis_specification,
                    row.procedure_assistance,
                    row.submission_interface,
                )
                for row in self.arm_summaries
                if row.metric == primary.metric
            }
            specifications = {row.analysis_specification for row in self.grade_rows}
            expected_primary_arm_keys = {
                (model_id, specification, assistance, interface)
                for model_id in model_ids
                for specification in specifications
                for assistance in ("output_contract_only", "unordered_checklist", "ordered_sop")
                for interface in ("structured", "narrative")
            }
            if primary_arm_keys != expected_primary_arm_keys:
                raise ValueError(
                    "Factorial report requires every primary specification/assistance/interface arm summary per model."
                )
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        digest = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Ablation analysis checksum does not match its canonical payload.")
        object.__setattr__(self, "checksum", digest)
        return self


class TrialEvalAblationPrimaryEstimandV1(_ExperimentModelV1):
    """One prospectively selected assistance estimand."""

    metric: TrialEvalAblationMetricV1
    contrast_id: str = Field(..., min_length=1)
    analysis_specification: TrialEvalAnalysisSpecificationV1
    prompt_condition: TrialEvalPromptConditionV1 | None = None
    applicability: Literal["applicable", "mismatched", "inapplicable"] | None = None


class TrialEvalAblationAnalysisConfigV1(_ExperimentModelV1):
    """Prospective uncertainty contract for paired ablation analysis."""

    schema_id: Literal["trialagentbench.trialeval_ablation_analysis_config/v1"] = (
        "trialagentbench.trialeval_ablation_analysis_config/v1"
    )
    design: Literal["factorial_interface", "targeted_control"]
    execution_scope: Literal["canary", "pilot", "publication"]
    experiment_design_sha256: str = Field(..., min_length=64, max_length=64)
    primary_estimand: TrialEvalAblationPrimaryEstimandV1
    supporting_metrics: tuple[TrialEvalAblationMetricV1, ...] = ()
    denominator_policy: Literal["all_scheduled_assignments_noncompletion_is_failure"] = (
        "all_scheduled_assignments_noncompletion_is_failure"
    )
    multiplicity_policy: Literal["single_primary_estimand_supporting_descriptive"] = (
        "single_primary_estimand_supporting_descriptive"
    )
    confidence_level: float = Field(..., gt=0.0, lt=1.0)
    bootstrap_resamples: int = Field(..., ge=1000)
    bootstrap_seed: int = Field(..., ge=0)
    min_base_trial_clusters: int = Field(..., ge=1)
    min_decoding_replicates: int = Field(..., ge=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_estimand_and_hash(self) -> TrialEvalAblationAnalysisConfigV1:
        if self.execution_scope != "canary" and (self.min_base_trial_clusters < 2 or self.min_decoding_replicates < 2):
            raise ValueError("Pilot and publication analyses require at least two trial and decoding clusters.")
        if len(self.supporting_metrics) != len(set(self.supporting_metrics)):
            raise ValueError("Ablation supporting metrics must be unique.")
        if self.primary_estimand.metric in self.supporting_metrics:
            raise ValueError("The primary ablation metric cannot be repeated as a supporting metric.")
        if self.design == "factorial_interface":
            if self.primary_estimand.prompt_condition is not None or self.primary_estimand.applicability is not None:
                raise ValueError("Factorial primary estimands cannot declare targeted-prompt strata.")
        elif self.primary_estimand.prompt_condition not in _CAPABILITY_CONDITIONS:
            raise ValueError("Targeted-control primary estimands require a targeted capability prompt.")
        elif self.primary_estimand.applicability is None:
            raise ValueError("Targeted-control primary estimands require an applicability stratum.")
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        digest = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Ablation analysis-config checksum does not match its canonical payload.")
        object.__setattr__(self, "checksum", digest)
        return self

    @property
    def metrics(self) -> tuple[TrialEvalAblationMetricV1, ...]:
        """Return the primary metric followed by its descriptive cascade."""

        return (self.primary_estimand.metric, *self.supporting_metrics)


def trialeval_publication_analysis_config_v1(
    experiment_design: TrialEvalExperimentProtocolV1,
) -> TrialEvalAblationAnalysisConfigV1:
    """Derive the publication analysis contract from the frozen design."""

    return TrialEvalAblationAnalysisConfigV1(
        design="factorial_interface",
        execution_scope="publication",
        experiment_design_sha256=experiment_design.checksum,
        primary_estimand=TrialEvalAblationPrimaryEstimandV1(
            metric="primary_analysis_conforms",
            contrast_id="P2-P0",
            analysis_specification="protocol_only",
        ),
        supporting_metrics=(
            "usable_primary",
            "route_match",
            "obligations_met",
            "numeric_result_available",
            "result_match",
            "numeric_absolute_error",
            "numeric_tolerance_ratio",
            "primary_uncertainty_valid",
            "primary_interval_agreement",
            "planning_valid",
            "planning_usable_with_primary",
            "planning_achieved_power",
            "planning_power_shortfall",
            "planning_underpowered",
            "planning_proportional_participant_deviation",
            "planning_log_sample_size_ratio",
            "planning_event_shortage",
            "planning_excess_events",
            "planning_excess_participants",
            "planning_participant_shortage",
        ),
        confidence_level=experiment_design.precision.confidence_level,
        bootstrap_resamples=10_000,
        bootstrap_seed=8_675_309,
        min_base_trial_clusters=experiment_design.precision.retained_independent_base_trials,
        min_decoding_replicates=experiment_design.compute_envelope.decoding_replicates,
    )


class TrialEvalAblationContrastV1(_ExperimentModelV1):
    """One paired contrast with crossed trial and decoding-seed uncertainty."""

    model_id: str
    metric: TrialEvalAblationMetricV1
    analysis_population: TrialEvalMetricPopulationV1
    contrast_id: str
    analysis_specification: TrialEvalAnalysisSpecificationV1 | None = None
    prompt_condition: TrialEvalPromptConditionV1 | None = None
    applicability: Literal["applicable", "mismatched", "inapplicable"] | None = None
    n_blocks: int = Field(..., gt=0)
    n_base_trial_clusters: int = Field(..., gt=1)
    n_decoding_replicates: int = Field(..., gt=1)
    estimate: float
    interval_low: float
    interval_high: float
    confidence_level: float = Field(..., gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _ordered_interval(self) -> TrialEvalAblationContrastV1:
        if self.interval_low > self.interval_high:
            raise ValueError("Ablation contrast interval_low exceeds interval_high.")
        expected = trialeval_metric_population_v1(self.metric)
        if self.analysis_population != expected:
            raise ValueError(
                f"Metric {self.metric!r} requires analysis_population={expected!r}, not {self.analysis_population!r}."
            )
        return self


class TrialEvalInterfaceAffordanceV1(_ExperimentModelV1):
    """Fields and enum vocabulary exposed by one response interface."""

    submission_interface: TrialEvalSubmissionInterfaceV1
    tool_schema_sha256: str = Field(..., min_length=64, max_length=64)
    response_contract_sha256: str = Field(..., min_length=64, max_length=64)
    field_paths: tuple[str, ...]
    enum_vocabulary: tuple[str, ...]


class TrialEvalSchemaAffordanceInventoryV1(_ExperimentModelV1):
    """Checksummed comparison of structured and narrative interface affordance."""

    schema_id: Literal["trialagentbench.trialeval_schema_affordance_inventory/v1"] = (
        "trialagentbench.trialeval_schema_affordance_inventory/v1"
    )
    shared_scientific_vocabulary: tuple[str, ...] = Field(min_length=1)
    interfaces: tuple[TrialEvalInterfaceAffordanceV1, ...]
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _canonicalize_and_hash(self) -> TrialEvalSchemaAffordanceInventoryV1:
        vocabulary = tuple(sorted(set(self.shared_scientific_vocabulary)))
        object.__setattr__(self, "shared_scientific_vocabulary", vocabulary)
        ordered = tuple(sorted(self.interfaces, key=lambda row: row.submission_interface))
        if {row.submission_interface for row in ordered} != {"structured", "narrative"}:
            raise ValueError("Schema affordance inventory requires structured and narrative interfaces.")
        object.__setattr__(self, "interfaces", ordered)
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        digest = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Schema affordance inventory checksum does not match its canonical payload.")
        object.__setattr__(self, "checksum", digest)
        return self


__all__ = [
    "EXPERIMENT_IDENTIFIER_PATTERN_V1",
    "TrialEvalAblationAssignmentV1",
    "TrialEvalAblationAnalysisConfigV1",
    "TrialEvalAblationAnalysisReportV1",
    "TrialEvalAblationArmSummaryV1",
    "TrialEvalAblationContrastV1",
    "TrialEvalAblationEndpointRowV1",
    "TrialEvalAblationEndpointSetV1",
    "TrialEvalAblationEvaluatorLabelsV1",
    "TrialEvalAblationGradeRowV1",
    "TrialEvalAblationMetricV1",
    "TrialEvalMetricPopulationV1",
    "TrialEvalAblationObservableContrastV1",
    "TrialEvalAblationObservableRowV1",
    "TrialEvalAblationPrimaryEstimandV1",
    "TrialEvalAblationScheduleV1",
    "TrialEvalAnalysisSpecificationV1",
    "TrialEvalNarrativeClaimV1",
    "TrialEvalNarrativeSourceSpanV1",
    "TrialEvalNarrativeTranscriptionV1",
    "TrialEvalNormalizationStatusV1",
    "TrialEvalObservableMetricV1",
    "TrialEvalPromptConditionV1",
    "TrialEvalInterfaceAffordanceV1",
    "TrialEvalSchemaAffordanceInventoryV1",
    "TrialEvalSubmissionInterfaceV1",
    "TrialEvalRepresentationFidelityRowV1",
    "TrialEvalRepresentationFixtureV1",
    "TrialEvalTargetedApplicabilityLabelV1",
    "trialeval_publication_analysis_config_v1",
    "trialeval_ablation_cells_v1",
    "trialeval_metric_population_v1",
    "submission_from_narrative_claims_v1",
]
