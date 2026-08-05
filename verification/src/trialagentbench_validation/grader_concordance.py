"""Independent TrialEval grader reconstruction and separate-process concordance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AcceptanceEnvelopeV1(_Contract):
    schema_id: Literal["trialagentbench.numerical_acceptance_envelope/v1"]
    reporting_decimal_places: int = Field(ge=0, le=15)
    independent_max_abs_difference: float = Field(ge=0)
    public_verification_id: str
    independent_verification_ids: tuple[str, ...]

    @property
    def absolute_tolerance(self) -> float:
        """Return the prespecified rounding and implementation envelope."""

        return (
            0.5 * 10.0 ** (-self.reporting_decimal_places)
            + self.independent_max_abs_difference
        )


class RouteSignatureV1(_Contract):
    analysis_population_id: str
    estimand_id: str
    intercurrent_event_strategy_ids: tuple[str, ...] = ()
    assessment_horizon_days: float | None = None
    treatment_id: str
    comparator_id: str
    endpoint_id: str
    effect_scale: str
    analysis_method_id: str

    @model_validator(mode="after")
    def _canonical_question(self) -> RouteSignatureV1:
        if (
            tuple(sorted(set(self.intercurrent_event_strategy_ids)))
            != self.intercurrent_event_strategy_ids
        ):
            raise ValueError(
                "intercurrent_event_strategy_ids must be sorted and unique"
            )
        return self


class NamedValueV1(_Contract):
    name: str
    value: float


class NumericPointTargetV1(_Contract):
    kind: Literal["numeric_point"]
    value: float
    result_unit: str
    acceptance_envelope: AcceptanceEnvelopeV1
    require_confidence_interval: bool
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None

    @model_validator(mode="after")
    def _complete_interval(self) -> NumericPointTargetV1:
        endpoints = (self.confidence_interval_lower, self.confidence_interval_upper)
        if (endpoints[0] is None) != (endpoints[1] is None):
            raise ValueError(
                "target confidence interval endpoints must be supplied together"
            )
        if self.require_confidence_interval != (endpoints[0] is not None):
            raise ValueError(
                "require_confidence_interval must equal the presence of target confidence interval endpoints"
            )
        if (
            endpoints[0] is not None
            and endpoints[1] is not None
            and endpoints[0] > endpoints[1]
        ):
            raise ValueError(
                "target confidence interval lower endpoint cannot exceed upper endpoint"
            )
        return self


class NumericIntervalTargetV1(_Contract):
    kind: Literal["numeric_interval"]
    lower: float
    upper: float
    result_unit: str
    acceptance_envelope: AcceptanceEnvelopeV1


class NumericVectorTargetV1(_Contract):
    kind: Literal["numeric_vector"]
    components: tuple[NamedValueV1, ...]
    result_unit: str
    acceptance_envelope: AcceptanceEnvelopeV1


class StatisticalTestTargetV1(_Contract):
    kind: Literal["statistical_test"]
    p_value: float
    reject_null: bool
    acceptance_envelope: AcceptanceEnvelopeV1


class CategoricalTargetV1(_Contract):
    kind: Literal["categorical"]
    credit_eligible_codes: tuple[str, ...]


TargetV1 = Annotated[
    NumericPointTargetV1
    | NumericIntervalTargetV1
    | NumericVectorTargetV1
    | StatisticalTestTargetV1
    | CategoricalTargetV1,
    Field(discriminator="kind"),
]


class AnalysisMethodBindingV1(_Contract):
    analysis_method_id: str
    estimator_family: str
    result_kind: Literal[
        "numeric_point",
        "numeric_interval",
        "numeric_vector",
        "statistical_test",
        "sensitivity_set",
        "identification_bound",
        "limitation",
        "abstention",
        "decision",
    ]
    uncertainty_method: str
    sensitivity_parameters: tuple[float, ...] = ()
    design_modifiers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _canonical_method(self) -> AnalysisMethodBindingV1:
        if tuple(sorted(set(self.design_modifiers))) != self.design_modifiers:
            raise ValueError("design_modifiers must be sorted and unique")
        if (
            tuple(sorted(set(self.sensitivity_parameters)))
            != self.sensitivity_parameters
        ):
            raise ValueError("sensitivity_parameters must be sorted and unique")
        if bool(self.sensitivity_parameters) != (self.result_kind == "sensitivity_set"):
            raise ValueError(
                "sensitivity parameters are required exactly for sensitivity-set methods"
            )
        return self


class RouteV1(_Contract):
    route_id: str
    signature: RouteSignatureV1
    method: AnalysisMethodBindingV1
    required_identification_assumptions: tuple[str, ...]
    required_diagnostics: tuple[str, ...] = ()
    planning_calculator_id: str | None = None
    target: TargetV1

    @model_validator(mode="after")
    def _coherent_method_and_duties(self) -> RouteV1:
        if self.signature.analysis_method_id != self.method.analysis_method_id:
            raise ValueError(
                "route signature and intrinsic method record must use one analysis method ID"
            )
        for name, values in (
            (
                "required_identification_assumptions",
                self.required_identification_assumptions,
            ),
            ("required_diagnostics", self.required_diagnostics),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        compatible_targets = {
            "numeric_point": {"numeric_point"},
            "numeric_interval": {"numeric_interval"},
            "numeric_vector": {"numeric_vector"},
            "statistical_test": {"statistical_test"},
            "sensitivity_set": {"numeric_interval", "numeric_vector"},
            "identification_bound": {"numeric_interval"},
            "limitation": {"categorical"},
            "abstention": {"categorical"},
            "decision": {"categorical"},
        }
        if self.target.kind not in compatible_targets[self.method.result_kind]:
            raise ValueError(
                "intrinsic method result kind is incompatible with the scoring target"
            )
        return self


class IntegrityRecordV1(_Contract):
    condition_id: Literal["exact_transport_row_duplication_v1"]
    affected_domain: str
    compound_key_fields: tuple[str, ...]
    observed_duplicate_group_count: int
    observed_extra_row_count: int
    repair_action: Literal["remove_one_exact_duplicate_copy"]
    repair_status: Literal["repaired", "unexpected_data_integrity_state"]
    post_repair_data_checksum: str
    analysis_input_data_checksum: str


class IntegrityTargetV1(_Contract):
    condition_id: Literal["exact_transport_row_duplication_v1"]
    affected_domain: str
    compound_key_fields: tuple[str, ...]
    observed_duplicate_group_count: int
    observed_extra_row_count: int
    repair_action: Literal["remove_one_exact_duplicate_copy"]
    repair_status: Literal["repaired"]
    post_repair_data_checksum: str
    analysis_input_data_checksum: str


class ScoringKeyV1(_Contract):
    schema_id: Literal["trialagentbench.scoring_key/v1"]
    release_id: str
    item_id: str
    question_id: str
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    credit_eligible_routes: tuple[RouteV1, ...]
    data_integrity_target: IntegrityTargetV1 | None = None

    @model_validator(mode="after")
    def _coherent_scoring_key(self) -> ScoringKeyV1:
        route_ids = tuple(route.route_id for route in self.credit_eligible_routes)
        if not route_ids:
            raise ValueError(
                "a scoring key must contain at least one credit-eligible route"
            )
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("credit-eligible route IDs must be unique")
        signatures = tuple(route.signature for route in self.credit_eligible_routes)
        if len(set(signatures)) != len(signatures):
            raise ValueError("credit-eligible route signatures must be unique")
        if (self.context_tier == "C5") != (self.data_integrity_target is not None):
            raise ValueError("exactly C5 scoring keys require a data-integrity target")
        return self


class NumericPointSubmissionV1(_Contract):
    kind: Literal["numeric_point"]
    value: float
    result_unit: str
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None


class NumericIntervalSubmissionV1(_Contract):
    kind: Literal["numeric_interval"]
    lower: float
    upper: float
    result_unit: str


class NumericVectorSubmissionV1(_Contract):
    kind: Literal["numeric_vector"]
    components: tuple[NamedValueV1, ...]
    result_unit: str


class StatisticalTestSubmissionV1(_Contract):
    kind: Literal["statistical_test"]
    p_value: float
    reject_null: bool


class CategoricalSubmissionV1(_Contract):
    kind: Literal["categorical"]
    code: str


SubmissionResultV1 = Annotated[
    NumericPointSubmissionV1
    | NumericIntervalSubmissionV1
    | NumericVectorSubmissionV1
    | StatisticalTestSubmissionV1
    | CategoricalSubmissionV1,
    Field(discriminator="kind"),
]


class CanonicalSubmissionV1(_Contract):
    schema_id: Literal["trialagentbench.canonical_submission/v1"]
    item_id: str
    primary: RouteSignatureV1
    diagnostic_ids: tuple[str, ...] = ()
    data_integrity_record: IntegrityRecordV1 | None = None
    result: SubmissionResultV1


class CanonicalTrialEvalRouteWitnessV1(_Contract):
    schema_id: Literal["trialagentbench.trialeval_route_witness/v1"] = (
        "trialagentbench.trialeval_route_witness/v1"
    )
    witness_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    submission: CanonicalSubmissionV1

    @model_validator(mode="after")
    def _identity_is_bound(self) -> CanonicalTrialEvalRouteWitnessV1:
        if self.submission.item_id != self.item_id:
            raise ValueError(
                "route-witness item identity disagrees with its submission"
            )
        return self


class GradeGateRecordV1(_Contract):
    gate_id: Literal[
        "submission",
        "question",
        "route",
        "evidence",
        "integrity",
        "result",
        "conformance",
        "decision",
    ]
    status: Literal["passed", "failed", "not_reached", "not_applicable"]
    failure_code: str | None = None


class GradeComponentRecordV1(_Contract):
    component_id: Literal[
        "submission",
        "question",
        "method",
        "evidence",
        "integrity",
        "result_structure",
        "route_comparison",
    ]
    status: Literal["passed", "failed", "not_evaluable", "not_applicable"]
    failure_code: str | None = None


class GradeRecordV1(_Contract):
    schema_id: Literal["trialagentbench.grade_record/v1"] = (
        "trialagentbench.grade_record/v1"
    )
    release_id: str
    item_id: str
    usable_primary: bool
    route_match: bool
    obligations_met: bool
    result_match: bool
    passed: bool
    gates: tuple[GradeGateRecordV1, ...]
    components: tuple[GradeComponentRecordV1, ...]
    first_failure_gate: str | None = None
    matched_route_id: str | None = None
    failure_codes: tuple[str, ...] = ()
    absolute_error: float | None = None
    tolerance_ratio: float | None = None


class CanonicalTrialEvalRouteWitnessGradeV1(_Contract):
    schema_id: Literal["trialagentbench.trialeval_route_witness_grade/v1"] = (
        "trialagentbench.trialeval_route_witness_grade/v1"
    )
    witness_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    grade: GradeRecordV1

    @model_validator(mode="after")
    def _grade_is_bound(self) -> CanonicalTrialEvalRouteWitnessGradeV1:
        if self.grade.item_id != self.item_id:
            raise ValueError("route-witness grade item identity mismatch")
        if self.grade.matched_route_id != self.route_id:
            raise ValueError(
                "positive route witness did not grade through its declared route"
            )
        return self


TrialDevLaneIdV1 = Literal[
    "asset_nomination",
    "phase_design",
    "phase_analysis",
    "decision_action",
    "route_timing",
    "final_recommendation",
    "safety_gate",
]


class TrialDevEvaluationTargetV1(_Contract):
    schema_id: Literal["trialdev_evaluation_target_register_record_v1"]
    scenario_id: str
    phase_id: str
    program_objective_id: str
    phase_scoring_objective_id: str
    lane_id: TrialDevLaneIdV1
    scoring_policy_id: str
    public_evidence_basis: tuple[str, ...]
    evaluator_evidence_basis: tuple[str, ...]
    reference_target_ids: tuple[str, ...]
    credit_eligible_target_ids: tuple[str, ...] = ()
    rejected_shortcut_ids: tuple[str, ...] = ()
    recoverability_policy_id: str
    target_resolution: Literal[
        "release_static",
        "submitted_method_public_evidence",
        "realized_public_evidence",
        "realized_trajectory",
    ]
    value_payload: dict[str, JsonValue] = {}
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class CanonicalTrialDevLaneSubmissionV1(_Contract):
    schema_id: Literal["trialagentbench.trialdev_canonical_lane_submission/v1"]
    evaluation_target_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_id: str
    phase_id: str
    program_objective_id: str
    phase_scoring_objective_id: str
    lane_id: TrialDevLaneIdV1
    submitted_target_id: str | None = None
    artifact_status: Literal["present", "missing", "invalid"]
    failure_reason: str | None = None
    score_override: float | None = Field(default=None, ge=0.0, le=1.0)
    score_derivation: (
        Literal[
            "literal_target",
            "numeric_diagnostic",
            "public_evidence_action",
        ]
        | None
    ) = None
    derived_from_trajectory_metric: bool = False
    terminal_action_observed: str | None = None
    terminal_asset_observed: str | None = None
    terminal_phase_observed: str | None = None


class TrialDevLaneGradeV1(_Contract):
    schema_id: Literal["trialdev_lane_score_record_v1"] = (
        "trialdev_lane_score_record_v1"
    )
    schema_version: Literal[1] = 1
    scenario_id: str
    phase_id: str
    program_objective_id: str
    phase_scoring_objective_id: str
    lane_id: TrialDevLaneIdV1
    evaluation_target_checksum: str
    scoring_policy_id: str
    recoverability_policy_id: str
    submitted_target_id: str | None = None
    reference_target_ids: tuple[str, ...]
    credit_eligible_target_ids: tuple[str, ...] = ()
    score: float
    score_derivation: Literal[
        "literal_target",
        "numeric_diagnostic",
        "public_evidence_action",
    ]
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
    checksum: str


class GraderConcordanceReportV1(_Contract):
    schema_id: Literal["trialagentbench.grader_concordance_report/v1"] = (
        "trialagentbench.grader_concordance_report/v1"
    )
    release_id: str
    trialeval_item_count: int = Field(ge=0)
    trialeval_required_count: int = Field(ge=0)
    trialdev_required_count: int = Field(ge=0)
    raw_projection_required_count: int = Field(ge=0)
    independently_projected_raw_count: int = Field(ge=0)
    harness_projected_raw_count: int = Field(ge=0)
    raw_projection_mismatch_count: int = Field(ge=0)
    trialeval_mutation_required_count: int = Field(ge=0)
    trialeval_mutation_independently_graded_count: int = Field(ge=0)
    trialeval_mutation_public_graded_count: int = Field(ge=0)
    trialeval_mutation_mismatch_count: int = Field(ge=0)
    trialeval_mutation_behavior_failure_count: int = Field(ge=0)
    trialeval_mutation_crashed_count: int = Field(ge=0)
    trialdev_mutation_required_count: int = Field(ge=0)
    trialdev_mutation_independently_graded_count: int = Field(ge=0)
    trialdev_mutation_public_graded_count: int = Field(ge=0)
    trialdev_mutation_mismatch_count: int = Field(ge=0)
    trialdev_mutation_behavior_failure_count: int = Field(ge=0)
    trialdev_mutation_crashed_count: int = Field(ge=0)
    required_count: int = Field(ge=1)
    independently_graded_count: int = Field(ge=0)
    public_grader_count: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)
    unsupported_count: int = Field(ge=0)
    crashed_count: int = Field(ge=0)
    independent_raw_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    harness_raw_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    independent_mutation_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_mutation_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    independent_trialdev_mutation_projection_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    public_trialdev_mutation_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    independent_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mismatched_item_ids: tuple[str, ...] = ()
    public_grader_command: tuple[str, ...]
    independent_projection_frozen_before_public_grader: Literal[True] = True
    verifier_imported_public_grader: Literal[False] = False
    passed: bool

    @model_validator(mode="after")
    def _status_matches_counts(self) -> GraderConcordanceReportV1:
        expected = (
            self.raw_projection_required_count == self.trialeval_required_count
            and self.independent_raw_projection_sha256
            == self.harness_raw_projection_sha256
            and self.independent_mutation_projection_sha256
            == self.public_mutation_projection_sha256
            and self.independent_trialdev_mutation_projection_sha256
            == self.public_trialdev_mutation_projection_sha256
            and self.independent_projection_sha256 == self.public_projection_sha256
            and self.raw_projection_required_count
            == self.independently_projected_raw_count
            == self.harness_projected_raw_count
            and self.trialeval_mutation_required_count
            == self.trialeval_mutation_independently_graded_count
            == self.trialeval_mutation_public_graded_count
            and self.trialdev_mutation_required_count
            == self.trialdev_mutation_independently_graded_count
            == self.trialdev_mutation_public_graded_count
            and self.required_count
            == self.independently_graded_count
            == self.public_grader_count
            and self.raw_projection_mismatch_count
            == self.trialeval_mutation_mismatch_count
            == self.trialeval_mutation_behavior_failure_count
            == self.trialeval_mutation_crashed_count
            == self.trialdev_mutation_mismatch_count
            == self.trialdev_mutation_behavior_failure_count
            == self.trialdev_mutation_crashed_count
            == self.mismatch_count
            == self.unsupported_count
            == self.crashed_count
            == 0
        )
        if self.passed != expected:
            raise ValueError("grader-concordance status does not match its census")
        return self


_GRADE_GATE_ORDER = (
    "submission",
    "question",
    "route",
    "evidence",
    "integrity",
    "result",
    "conformance",
    "decision",
)
_GRADE_COMPONENT_ORDER = (
    "submission",
    "question",
    "method",
    "evidence",
    "integrity",
    "result_structure",
    "route_comparison",
)


def _gate_records(
    *,
    failure_gate: str | None,
    failure_code: str | None,
    integrity_applicable: bool,
) -> tuple[GradeGateRecordV1, ...]:
    failure_index = (
        None if failure_gate is None else _GRADE_GATE_ORDER.index(failure_gate)
    )
    rows: list[GradeGateRecordV1] = []
    for index, gate_id in enumerate(_GRADE_GATE_ORDER):
        if failure_index is not None and index > failure_index:
            status = "not_reached"
        elif gate_id == failure_gate:
            status = "failed"
        elif gate_id == "integrity" and not integrity_applicable:
            status = "not_applicable"
        elif gate_id == "decision":
            status = "not_applicable"
        else:
            status = "passed"
        rows.append(
            GradeGateRecordV1(
                gate_id=gate_id,
                status=status,
                failure_code=failure_code if gate_id == failure_gate else None,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class _ResultAssessment:
    structure_failure: str | None = None
    comparison_failure: str | None = None
    absolute_error: float | None = None
    tolerance_ratio: float | None = None


@dataclass(frozen=True)
class _GradeAssessment:
    components: tuple[GradeComponentRecordV1, ...]
    first_failure_gate: str | None
    first_failure_code: str | None
    matched_route_id: str | None = None
    absolute_error: float | None = None
    tolerance_ratio: float | None = None


def _component(
    component_id: str,
    failure_code: str | None = None,
    *,
    status: str | None = None,
) -> GradeComponentRecordV1:
    return GradeComponentRecordV1.model_validate(
        {
            "component_id": component_id,
            "status": status or ("failed" if failure_code is not None else "passed"),
            "failure_code": failure_code,
        }
    )


def _grade_from_assessment(
    key: ScoringKeyV1, assessment: _GradeAssessment
) -> GradeRecordV1:
    gates = _gate_records(
        failure_gate=assessment.first_failure_gate,
        failure_code=assessment.first_failure_code,
        integrity_applicable=key.data_integrity_target is not None,
    )
    statuses = {record.gate_id: record.status for record in gates}
    payload: dict[str, object] = {
        "release_id": key.release_id,
        "item_id": key.item_id,
        "usable_primary": statuses["submission"] == "passed",
        "route_match": statuses["question"] == statuses["route"] == "passed",
        "obligations_met": (
            statuses["evidence"] == "passed"
            and statuses["integrity"] in {"passed", "not_applicable"}
        ),
        "result_match": statuses["result"] == statuses["conformance"] == "passed",
        "passed": assessment.first_failure_gate is None,
        "gates": gates,
        "components": assessment.components,
        "first_failure_gate": assessment.first_failure_gate,
        "matched_route_id": assessment.matched_route_id,
        "failure_codes": (
            ()
            if assessment.first_failure_code is None
            else (assessment.first_failure_code,)
        ),
        "absolute_error": assessment.absolute_error,
        "tolerance_ratio": assessment.tolerance_ratio,
    }
    return GradeRecordV1.model_validate(payload)


def _integrity_failure(
    target: IntegrityTargetV1 | None, observed: IntegrityRecordV1 | None
) -> str | None:
    if target is None:
        return None if observed is None else "unexpected_data_integrity_record"
    if observed is None:
        return "missing_data_integrity_record"
    comparisons = (
        ("condition_id", "data_integrity_condition_mismatch"),
        ("affected_domain", "data_integrity_domain_mismatch"),
        ("compound_key_fields", "data_integrity_compound_key_mismatch"),
    )
    for field, failure in comparisons:
        if getattr(observed, field) != getattr(target, field):
            return failure
    if (
        observed.observed_duplicate_group_count != target.observed_duplicate_group_count
        or observed.observed_extra_row_count != target.observed_extra_row_count
    ):
        return "data_integrity_counts_mismatch"
    if (
        observed.repair_action != target.repair_action
        or observed.repair_status != target.repair_status
    ):
        return "data_integrity_repair_mismatch"
    if (
        observed.post_repair_data_checksum != target.post_repair_data_checksum
        or observed.analysis_input_data_checksum != target.analysis_input_data_checksum
    ):
        return "data_integrity_checksum_mismatch"
    return None


def _question_binding(signature: RouteSignatureV1) -> tuple[object, ...]:
    return (
        signature.analysis_population_id,
        signature.estimand_id,
        signature.intercurrent_event_strategy_ids,
        signature.assessment_horizon_days,
        signature.treatment_id,
        signature.comparator_id,
        signature.endpoint_id,
        signature.effect_scale,
    )


def grade_trialeval_independently(
    key: ScoringKeyV1, submission: CanonicalSubmissionV1
) -> GradeRecordV1:
    """Reconstruct the narrow TrialEval grade without importing the public grader."""

    return _grade_from_assessment(key, _assess_trialeval(key, submission))


def _assess_trialeval(
    key: ScoringKeyV1, submission: CanonicalSubmissionV1
) -> _GradeAssessment:
    components: dict[str, GradeComponentRecordV1] = {}
    item_matches = submission.item_id == key.item_id
    components["submission"] = _component(
        "submission", None if item_matches else "item_mismatch"
    )
    route = None
    result_assessment: _ResultAssessment | None = None
    if not item_matches:
        for component_id in _GRADE_COMPONENT_ORDER[1:]:
            components[component_id] = _component(component_id, status="not_evaluable")
    else:
        question_routes = tuple(
            candidate
            for candidate in key.credit_eligible_routes
            if _question_binding(candidate.signature)
            == _question_binding(submission.primary)
        )
        if not question_routes:
            components["question"] = _component(
                "question", "unrecognized_primary_question"
            )
            components["method"] = _component("method", status="not_evaluable")
        else:
            components["question"] = _component("question")
            route = next(
                (
                    candidate
                    for candidate in question_routes
                    if candidate.signature == submission.primary
                ),
                None,
            )
            components["method"] = _component(
                "method",
                None if route is not None else "unrecognized_primary_route",
            )
        if route is None:
            components["evidence"] = _component("evidence", status="not_evaluable")
            components["result_structure"] = _component(
                "result_structure", status="not_evaluable"
            )
            components["route_comparison"] = _component(
                "route_comparison", status="not_evaluable"
            )
        else:
            missing_diagnostics = set(route.required_diagnostics) - set(
                submission.diagnostic_ids
            )
            components["evidence"] = _component(
                "evidence",
                "missing_required_diagnostic" if missing_diagnostics else None,
            )
            result_assessment = _assess_result_independently(
                route.target, submission.result
            )
            components["result_structure"] = _component(
                "result_structure",
                result_assessment.structure_failure,
            )
            components["route_comparison"] = (
                _component("route_comparison", status="not_evaluable")
                if result_assessment.structure_failure is not None
                else _component(
                    "route_comparison", result_assessment.comparison_failure
                )
            )
        integrity_failure = _integrity_failure(
            key.data_integrity_target, submission.data_integrity_record
        )
        if (
            key.data_integrity_target is None
            and submission.data_integrity_record is None
        ):
            components["integrity"] = _component("integrity", status="not_applicable")
        else:
            components["integrity"] = _component("integrity", integrity_failure)
    gate_components = (
        ("submission", "submission"),
        ("question", "question"),
        ("route", "method"),
        ("evidence", "evidence"),
        ("integrity", "integrity"),
        ("result", "result_structure"),
        ("conformance", "route_comparison"),
    )
    first_failure_gate = None
    first_failure_code = None
    for gate_id, component_id in gate_components:
        if components[component_id].status == "failed":
            first_failure_gate = gate_id
            first_failure_code = components[component_id].failure_code
            break
    comparison_reached = first_failure_gate in {None, "conformance"}
    return _GradeAssessment(
        components=tuple(
            components[component_id] for component_id in _GRADE_COMPONENT_ORDER
        ),
        first_failure_gate=first_failure_gate,
        first_failure_code=first_failure_code,
        matched_route_id=None if route is None else route.route_id,
        absolute_error=(
            None
            if result_assessment is None or not comparison_reached
            else result_assessment.absolute_error
        ),
        tolerance_ratio=(
            None
            if result_assessment is None or not comparison_reached
            else result_assessment.tolerance_ratio
        ),
    )


def _assess_result_independently(target: object, result: object) -> _ResultAssessment:
    if isinstance(target, NumericPointTargetV1) and isinstance(
        result, NumericPointSubmissionV1
    ):
        if result.result_unit != target.result_unit:
            return _ResultAssessment(structure_failure="result_unit_mismatch")
        if target.require_confidence_interval and (
            result.confidence_interval_lower is None
            or result.confidence_interval_upper is None
        ):
            return _ResultAssessment(structure_failure="missing_confidence_interval")
        error = abs(result.value - target.value)
        tolerance = target.acceptance_envelope.absolute_tolerance
        failure = "numeric_result_outside_tolerance" if error > tolerance else None
        if target.require_confidence_interval:
            if (
                target.confidence_interval_lower is None
                or target.confidence_interval_upper is None
            ):
                raise ValueError(
                    "validated scoring target is missing required confidence interval endpoints"
                )
            if (
                result.confidence_interval_lower is None
                or result.confidence_interval_upper is None
            ):
                raise ValueError(
                    "validated submission is missing required confidence interval endpoints"
                )
            interval_error = max(
                abs(
                    result.confidence_interval_lower - target.confidence_interval_lower
                ),
                abs(
                    result.confidence_interval_upper - target.confidence_interval_upper
                ),
            )
            if interval_error > tolerance and failure is None:
                failure = "confidence_interval_outside_tolerance"
            error = max(error, interval_error)
        return _ResultAssessment(
            comparison_failure=failure,
            absolute_error=error,
            tolerance_ratio=error / tolerance,
        )
    if isinstance(target, NumericIntervalTargetV1) and isinstance(
        result, NumericIntervalSubmissionV1
    ):
        if result.result_unit != target.result_unit:
            return _ResultAssessment(structure_failure="result_unit_mismatch")
        error = max(abs(result.lower - target.lower), abs(result.upper - target.upper))
        tolerance = target.acceptance_envelope.absolute_tolerance
        return _ResultAssessment(
            comparison_failure=(
                "interval_result_outside_tolerance" if error > tolerance else None
            ),
            absolute_error=error,
            tolerance_ratio=error / tolerance,
        )
    if isinstance(target, NumericVectorTargetV1) and isinstance(
        result, NumericVectorSubmissionV1
    ):
        if result.result_unit != target.result_unit:
            return _ResultAssessment(structure_failure="result_unit_mismatch")
        expected = {value.name: value.value for value in target.components}
        observed = {value.name: value.value for value in result.components}
        if set(expected) != set(observed):
            return _ResultAssessment(structure_failure="vector_components_mismatch")
        error = max(abs(observed[name] - value) for name, value in expected.items())
        tolerance = target.acceptance_envelope.absolute_tolerance
        return _ResultAssessment(
            comparison_failure=(
                "vector_result_outside_tolerance" if error > tolerance else None
            ),
            absolute_error=error,
            tolerance_ratio=error / tolerance,
        )
    if isinstance(target, StatisticalTestTargetV1) and isinstance(
        result, StatisticalTestSubmissionV1
    ):
        error = abs(result.p_value - target.p_value)
        tolerance = target.acceptance_envelope.absolute_tolerance
        if result.reject_null != target.reject_null:
            failure = "test_decision_mismatch"
        elif error > tolerance:
            failure = "test_p_value_outside_tolerance"
        else:
            failure = None
        return _ResultAssessment(
            comparison_failure=failure,
            absolute_error=error,
            tolerance_ratio=error / tolerance,
        )
    if isinstance(target, CategoricalTargetV1) and isinstance(
        result, CategoricalSubmissionV1
    ):
        return _ResultAssessment(
            comparison_failure=(
                None
                if result.code in target.credit_eligible_codes
                else "categorical_result_not_credit_eligible"
            )
        )
    return _ResultAssessment(structure_failure="result_kind_mismatch")


def grade_trialdev_lane_independently(
    target: TrialDevEvaluationTargetV1,
    submission: CanonicalTrialDevLaneSubmissionV1,
) -> TrialDevLaneGradeV1:
    """Reconstruct one TrialDev lane grade from its public scoring contract."""

    observed_context = (
        submission.scenario_id,
        submission.phase_id,
        submission.program_objective_id,
        submission.phase_scoring_objective_id,
        submission.lane_id,
    )
    expected_context = (
        target.scenario_id,
        target.phase_id,
        target.program_objective_id,
        target.phase_scoring_objective_id,
        target.lane_id,
    )
    if (
        observed_context != expected_context
        or submission.evaluation_target_checksum != target.checksum
    ):
        raise ValueError(
            f"TrialDev canonical lane context mismatch: observed={observed_context!r}, expected={expected_context!r}."
        )
    reference_targets = set(target.reference_target_ids)
    alternatives = set(target.credit_eligible_target_ids)
    if reference_targets & alternatives:
        raise ValueError(
            "TrialDev reference targets and credit-eligible alternatives overlap."
        )
    derivation = submission.score_derivation or (
        "numeric_diagnostic"
        if submission.score_override is not None
        else "literal_target"
    )
    if (
        target.target_resolution == "realized_public_evidence"
        and derivation != "public_evidence_action"
    ):
        raise ValueError(
            "Realized public-evidence targets require public-evidence action scoring."
        )
    if submission.score_override is not None:
        numeric_context = (None, submission.lane_id) in {
            (None, "phase_design"),
            (None, "phase_analysis"),
            (None, "route_timing"),
            ("final_decision", "final_recommendation"),
        } or (submission.phase_id, submission.lane_id) in {
            (None, "phase_design"),
            (None, "phase_analysis"),
            (None, "route_timing"),
            ("final_decision", "final_recommendation"),
        }
        if derivation == "literal_target":
            raise ValueError(
                "TrialDev score override requires a non-literal derivation."
            )
        if derivation == "numeric_diagnostic" and not numeric_context:
            raise ValueError(
                "TrialDev numeric score override is not allowed for this lane."
            )
        if derivation == "public_evidence_action" and submission.lane_id not in {
            "decision_action",
            "safety_gate",
            "route_timing",
        }:
            raise ValueError(
                "TrialDev public-evidence action scoring is not allowed for this lane."
            )
    submitted = submission.submitted_target_id
    missing_reason: str | None = None
    if submission.artifact_status == "missing":
        status = "missing_submission_zeroed"
        score = 0.0
        missing_reason = submission.failure_reason or "missing_submission"
    elif submission.artifact_status == "invalid":
        status = "invalid_submission_zeroed"
        score = 0.0
    elif submitted is None:
        status = "missing_submission_zeroed"
        score = 0.0
        missing_reason = submission.failure_reason or "missing_target"
    elif submitted in reference_targets:
        status = "scored"
        score = 1.0 if submission.score_override is None else submission.score_override
    elif submitted in alternatives:
        status = "credit_eligible_alternative"
        score = 1.0 if submission.score_override is None else submission.score_override
    elif submission.score_override is None:
        status = "scored"
        score = 0.0
    else:
        if (
            derivation != "public_evidence_action"
            and target.target_resolution != "realized_trajectory"
        ):
            raise ValueError(
                "TrialDev score override cannot credit an unaccepted target."
            )
        status = (
            "scored"
            if target.target_resolution == "realized_trajectory"
            else "credit_eligible_alternative"
        )
        score = submission.score_override
    payload: dict[str, object] = {
        "scenario_id": submission.scenario_id,
        "phase_id": submission.phase_id,
        "program_objective_id": submission.program_objective_id,
        "phase_scoring_objective_id": submission.phase_scoring_objective_id,
        "lane_id": submission.lane_id,
        "evaluation_target_checksum": target.checksum,
        "scoring_policy_id": target.scoring_policy_id,
        "recoverability_policy_id": target.recoverability_policy_id,
        "submitted_target_id": submitted,
        "reference_target_ids": target.reference_target_ids,
        "credit_eligible_target_ids": target.credit_eligible_target_ids,
        "score": max(0.0, min(1.0, float(score))),
        "score_derivation": derivation,
        "derived_from_trajectory_metric": submission.derived_from_trajectory_metric,
        "terminal_action_observed": submission.terminal_action_observed,
        "terminal_asset_observed": submission.terminal_asset_observed,
        "terminal_phase_observed": submission.terminal_phase_observed,
        "status": status,
        "artifact_status": submission.artifact_status,
        "missing_reason": missing_reason,
        "failure_reason": submission.failure_reason,
    }
    checksum_payload = {
        key: value for key, value in payload.items() if value is not None
    }
    checksum_payload["schema_id"] = "trialdev_lane_score_record_v1"
    checksum_payload["schema_version"] = 1
    encoded = json.dumps(
        checksum_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    payload["checksum"] = hashlib.sha256(encoded).hexdigest()
    return TrialDevLaneGradeV1.model_validate(payload)


def _read_jsonl(path: Path, model: type[_Contract]) -> tuple[_Contract, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )
    if not records:
        raise ValueError(f"JSON Lines census is empty: {path}")
    return records


def _write_records(path: Path, records: tuple[_Contract, ...]) -> str:
    body = "".join(
        json.dumps(
            record.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _run_grader_concordance_materialized(
    *,
    release_root: Path,
    canonical_submissions: Path,
    output_dir: Path,
    harness_executable: str = "trialagentbench",
) -> GraderConcordanceReportV1:
    """Freeze an independent census, invoke the harness, and compare exact records."""

    scoring_key_paths = tuple(Path(release_root).rglob("grader/scoring_keys.jsonl"))
    if len(scoring_key_paths) != 1:
        raise ValueError(
            "Grader concordance requires exactly one extracted TrialEval scoring-key census; "
            f"observed={len(scoring_key_paths)}."
        )
    evaluator_root = scoring_key_paths[0].parent.parent
    witnesses_path = Path(canonical_submissions) / "trialeval_route_witnesses.jsonl"
    keys = tuple(_read_jsonl(scoring_key_paths[0], ScoringKeyV1))
    witnesses = tuple(_read_jsonl(witnesses_path, CanonicalTrialEvalRouteWitnessV1))
    key_by_item = {
        str(record.item_id): record
        for record in keys
        if isinstance(record, ScoringKeyV1)
    }
    witness_by_id = {
        str(record.witness_id): record
        for record in witnesses
        if isinstance(record, CanonicalTrialEvalRouteWitnessV1)
    }
    if len(key_by_item) != len(keys):
        raise ValueError(
            "Grader-concordance scoring keys contain duplicate TrialEval item IDs"
        )
    if len(witness_by_id) != len(witnesses):
        raise ValueError(
            "Grader-concordance inputs contain duplicate TrialEval witness IDs"
        )
    expected_pairs = {
        (key.item_id, route.route_id)
        for key in key_by_item.values()
        for route in key.credit_eligible_routes
    }
    observed_pairs = {
        (witness.item_id, witness.route_id) for witness in witness_by_id.values()
    }
    if expected_pairs != observed_pairs:
        raise ValueError(
            "Grader-concordance route-witness denominator mismatch: "
            f"missing={sorted(expected_pairs - observed_pairs)!r}, "
            f"extra={sorted(observed_pairs - expected_pairs)!r}."
        )
    ordered_ids = tuple(
        record.item_id for record in keys if isinstance(record, ScoringKeyV1)
    )
    ordered_witnesses = tuple(
        record
        for record in witnesses
        if isinstance(record, CanonicalTrialEvalRouteWitnessV1)
    )
    raw_witnesses_path = (
        Path(canonical_submissions) / "trialeval_raw_route_witnesses.jsonl"
    )
    route_by_identity = {
        (key.item_id, route.route_id): route
        for key in key_by_item.values()
        for route in key.credit_eligible_routes
    }
    participant_candidates = tuple(
        path.parent
        for path in Path(release_root).rglob("public/items")
        if path.is_dir()
    )
    if len(participant_candidates) != 1:
        raise ValueError(
            "Grader concordance requires exactly one materialized TrialEval participant root; "
            f"observed={len(participant_candidates)}."
        )
    participant_root = participant_candidates[0]
    from trialagentbench_validation.raw_projection import (
        project_raw_witnesses_independently,
    )

    independent_raw = project_raw_witnesses_independently(
        raw_witnesses_path=raw_witnesses_path,
        participant_root=participant_root,
        evaluator_root=evaluator_root,
        route_by_identity=route_by_identity,
    )
    independent_raw_by_id = dict(independent_raw)
    independent_projected_witnesses = tuple(
        CanonicalTrialEvalRouteWitnessV1(
            witness_id=witness.witness_id,
            item_id=witness.item_id,
            route_id=witness.route_id,
            context_tier=witness.context_tier,
            submission=independent_raw_by_id[witness.witness_id],
        )
        for witness in ordered_witnesses
    )
    independent_trialeval_records = tuple(
        CanonicalTrialEvalRouteWitnessGradeV1(
            witness_id=witness.witness_id,
            item_id=witness.item_id,
            route_id=witness.route_id,
            grade=grade_trialeval_independently(
                key_by_item[witness.item_id],
                witness.submission,
            ),
        )
        for witness in ordered_witnesses
    )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite grader-concordance output: {output}"
        )
    output.mkdir(parents=True)
    independent_raw_path = output / "independent_trialeval_route_projections.jsonl"
    independent_raw_sha256 = _write_records(
        independent_raw_path,
        independent_projected_witnesses,
    )
    harness_raw_path = output / "harness_trialeval_route_projections.jsonl"
    raw_projection_command = (
        harness_executable,
        "grade",
        "project-trialeval-witnesses",
        "--participant-root",
        str(participant_root),
        "--evaluator-root",
        str(evaluator_root),
        "--witnesses",
        str(raw_witnesses_path),
        "--output",
        str(harness_raw_path),
    )
    raw_projection_completed = subprocess.run(
        raw_projection_command,
        check=False,
        text=True,
        capture_output=True,
    )
    if raw_projection_completed.returncode != 0:
        raise RuntimeError(
            "Public TrialEval raw projection subprocess failed during concordance.\n"
            f"stdout:\n{raw_projection_completed.stdout}\n"
            f"stderr:\n{raw_projection_completed.stderr}"
        )
    harness_projected_witnesses = tuple(
        _read_jsonl(harness_raw_path, CanonicalTrialEvalRouteWitnessV1)
    )
    harness_raw_sha256 = hashlib.sha256(harness_raw_path.read_bytes()).hexdigest()
    harness_raw_by_id = {
        record.witness_id: record
        for record in harness_projected_witnesses
        if isinstance(record, CanonicalTrialEvalRouteWitnessV1)
    }
    expected_raw_by_id = {record.witness_id: record for record in ordered_witnesses}
    independent_projected_by_id = {
        record.witness_id: record for record in independent_projected_witnesses
    }
    raw_projection_mismatched = tuple(
        witness.witness_id
        for witness in ordered_witnesses
        if harness_raw_by_id.get(witness.witness_id)
        != expected_raw_by_id[witness.witness_id]
        or independent_projected_by_id.get(witness.witness_id)
        != expected_raw_by_id[witness.witness_id]
    )
    trialdev_target_paths = tuple(
        sorted(Path(release_root).rglob("grader/evaluation_target_register.jsonl"))
    )
    if not trialdev_target_paths:
        raise ValueError(
            "Grader concordance requires extracted TrialDev evaluation-target registers."
        )
    trialdev_targets = tuple(
        record
        for path in trialdev_target_paths
        for record in _read_jsonl(path, TrialDevEvaluationTargetV1)
        if isinstance(record, TrialDevEvaluationTargetV1)
    )
    target_by_checksum = {record.checksum: record for record in trialdev_targets}
    if len(target_by_checksum) != len(trialdev_targets):
        raise ValueError(
            "TrialDev evaluation-target census contains duplicate checksums."
        )
    trialdev_submissions_path = (
        Path(canonical_submissions) / "trialdev_canonical_lane_submissions.jsonl"
    )
    trialdev_submissions = tuple(
        record
        for record in _read_jsonl(
            trialdev_submissions_path, CanonicalTrialDevLaneSubmissionV1
        )
        if isinstance(record, CanonicalTrialDevLaneSubmissionV1)
    )
    trialdev_submission_by_checksum = {
        record.evaluation_target_checksum: record for record in trialdev_submissions
    }
    if len(trialdev_submission_by_checksum) != len(trialdev_submissions):
        raise ValueError(
            "TrialDev canonical lane census contains duplicate evaluation-target checksums."
        )
    if set(target_by_checksum) != set(trialdev_submission_by_checksum):
        raise ValueError(
            "TrialDev canonical lane census does not cover every released evaluation target."
        )
    trialdev_checksums = tuple(record.checksum for record in trialdev_targets)
    independent_trialdev_records = tuple(
        grade_trialdev_lane_independently(
            target_by_checksum[checksum],
            trialdev_submission_by_checksum[checksum],
        )
        for checksum in trialdev_checksums
    )
    independent_trialeval_path = output / "independent_trialeval_grade_records.jsonl"
    independent_trialdev_path = output / "independent_trialdev_lane_records.jsonl"
    _write_records(independent_trialeval_path, independent_trialeval_records)
    _write_records(independent_trialdev_path, independent_trialdev_records)
    independent_body = (
        independent_trialeval_path.read_bytes() + independent_trialdev_path.read_bytes()
    )
    independent_sha256 = hashlib.sha256(independent_body).hexdigest()
    public_trialeval_path = output / "public_trialeval_grade_records.jsonl"
    trialeval_command = (
        harness_executable,
        "grade",
        "canonical-trialeval-witnesses",
        "--evaluator-root",
        str(evaluator_root),
        "--witnesses",
        str(witnesses_path),
        "--output",
        str(public_trialeval_path),
    )
    trialeval_completed = subprocess.run(
        trialeval_command, check=False, text=True, capture_output=True
    )
    if trialeval_completed.returncode != 0:
        raise RuntimeError(
            "Public TrialEval grader subprocess failed during concordance.\n"
            f"stdout:\n{trialeval_completed.stdout}\nstderr:\n{trialeval_completed.stderr}"
        )
    public_trialdev_path = output / "public_trialdev_lane_records.jsonl"
    trialdev_command = (
        harness_executable,
        "grade",
        "canonical-trialdev",
        "--release-root",
        str(release_root),
        "--submissions",
        str(trialdev_submissions_path),
        "--output",
        str(public_trialdev_path),
    )
    trialdev_completed = subprocess.run(
        trialdev_command, check=False, text=True, capture_output=True
    )
    if trialdev_completed.returncode != 0:
        raise RuntimeError(
            "Public TrialDev grader subprocess failed during concordance.\n"
            f"stdout:\n{trialdev_completed.stdout}\nstderr:\n{trialdev_completed.stderr}"
        )
    public_trialeval_records = tuple(
        _read_jsonl(public_trialeval_path, CanonicalTrialEvalRouteWitnessGradeV1)
    )
    public_trialdev_records = tuple(
        _read_jsonl(public_trialdev_path, TrialDevLaneGradeV1)
    )
    public_body = public_trialeval_path.read_bytes() + public_trialdev_path.read_bytes()
    public_sha256 = hashlib.sha256(public_body).hexdigest()
    independent_by_witness = {
        record.witness_id: record for record in independent_trialeval_records
    }
    public_by_witness = {
        record.witness_id: record
        for record in public_trialeval_records
        if isinstance(record, CanonicalTrialEvalRouteWitnessGradeV1)
    }
    trialeval_mismatched = tuple(
        witness.witness_id
        for witness in ordered_witnesses
        if witness.witness_id not in public_by_witness
        or public_by_witness[witness.witness_id]
        != independent_by_witness[witness.witness_id]
    )
    independent_trialdev_by_checksum = {
        record.evaluation_target_checksum: record
        for record in independent_trialdev_records
    }
    public_trialdev_by_checksum = {
        record.evaluation_target_checksum: record
        for record in public_trialdev_records
        if isinstance(record, TrialDevLaneGradeV1)
    }
    trialdev_mismatched = tuple(
        checksum
        for checksum in trialdev_checksums
        if checksum not in public_trialdev_by_checksum
        or public_trialdev_by_checksum[checksum]
        != independent_trialdev_by_checksum[checksum]
    )
    from trialagentbench_validation.grader_stress import (
        run_trialeval_grader_stress,
    )

    mutation_report = run_trialeval_grader_stress(
        witnesses=ordered_witnesses,
        key_by_item=key_by_item,
        raw_witnesses_path=raw_witnesses_path,
        participant_root=participant_root,
        route_by_identity=route_by_identity,
        evaluator_root=evaluator_root,
        output_dir=output / "trialeval_grader_stress",
        harness_executable=harness_executable,
    )
    from trialagentbench_validation.trialdev_grader_stress import (
        run_trialdev_grader_stress,
    )

    trialdev_mutation_report = run_trialdev_grader_stress(
        targets=trialdev_targets,
        submission_by_checksum=trialdev_submission_by_checksum,
        release_root=release_root,
        output_dir=output / "trialdev_grader_stress",
        harness_executable=harness_executable,
    )
    mismatched = tuple(
        dict.fromkeys(
            [
                *raw_projection_mismatched,
                *trialeval_mismatched,
                *trialdev_mismatched,
            ]
        )
    )
    required_count = len(ordered_witnesses) + len(trialdev_checksums)
    public_count = len(public_trialeval_records) + len(public_trialdev_records)
    report = GraderConcordanceReportV1(
        release_id=key_by_item[ordered_ids[0]].release_id,
        trialeval_item_count=len(ordered_ids),
        trialeval_required_count=len(ordered_witnesses),
        trialdev_required_count=len(trialdev_checksums),
        raw_projection_required_count=len(ordered_witnesses),
        independently_projected_raw_count=len(independent_projected_witnesses),
        harness_projected_raw_count=len(harness_projected_witnesses),
        raw_projection_mismatch_count=len(raw_projection_mismatched),
        trialeval_mutation_required_count=mutation_report.required_mutation_count,
        trialeval_mutation_independently_graded_count=(
            mutation_report.independently_graded_count
        ),
        trialeval_mutation_public_graded_count=mutation_report.public_graded_count,
        trialeval_mutation_mismatch_count=mutation_report.mismatch_count,
        trialeval_mutation_behavior_failure_count=(
            mutation_report.expected_behavior_failure_count
        ),
        trialeval_mutation_crashed_count=mutation_report.crashed_count,
        trialdev_mutation_required_count=(
            trialdev_mutation_report.required_mutation_count
        ),
        trialdev_mutation_independently_graded_count=(
            trialdev_mutation_report.independently_graded_count
        ),
        trialdev_mutation_public_graded_count=(
            trialdev_mutation_report.public_graded_count
        ),
        trialdev_mutation_mismatch_count=trialdev_mutation_report.mismatch_count,
        trialdev_mutation_behavior_failure_count=(
            trialdev_mutation_report.expected_behavior_failure_count
        ),
        trialdev_mutation_crashed_count=trialdev_mutation_report.crashed_count,
        required_count=required_count,
        independently_graded_count=len(independent_trialeval_records)
        + len(independent_trialdev_records),
        public_grader_count=public_count,
        mismatch_count=len(mismatched),
        unsupported_count=0,
        crashed_count=0,
        independent_raw_projection_sha256=independent_raw_sha256,
        harness_raw_projection_sha256=harness_raw_sha256,
        independent_mutation_projection_sha256=mutation_report.independent_sha256,
        public_mutation_projection_sha256=mutation_report.public_sha256,
        independent_trialdev_mutation_projection_sha256=(
            trialdev_mutation_report.independently_graded_sha256
        ),
        public_trialdev_mutation_projection_sha256=(
            trialdev_mutation_report.publicly_graded_sha256
        ),
        independent_projection_sha256=independent_sha256,
        public_projection_sha256=public_sha256,
        mismatched_item_ids=mismatched,
        public_grader_command=tuple(
            [
                *raw_projection_command,
                "&&",
                *trialeval_command,
                "&&",
                *trialdev_command,
                "&&",
                *trialdev_mutation_report.public_command,
            ]
        ),
        passed=(
            not mismatched
            and mutation_report.status == "pass"
            and trialdev_mutation_report.status == "pass"
            and len(independent_projected_witnesses)
            == len(harness_projected_witnesses)
            == len(ordered_witnesses)
            and public_count == required_count
        ),
    )
    (output / "grader_concordance_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _safe_extract(archive_path: Path, output_root: Path) -> None:
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"Unsafe grader-concordance archive member: {member.filename!r}"
                )
        archive.extractall(output_root)


def run_grader_concordance(
    *,
    release_root: Path,
    canonical_submissions: Path,
    output_dir: Path,
    harness_executable: str = "trialagentbench",
) -> GraderConcordanceReportV1:
    """Run concordance from an expanded release or its role-separated evaluator archives."""

    root = Path(release_root)
    if tuple(root.rglob("grader/scoring_keys.jsonl")):
        return _run_grader_concordance_materialized(
            release_root=root,
            canonical_submissions=canonical_submissions,
            output_dir=output_dir,
            harness_executable=harness_executable,
        )
    trialeval_archives = tuple(root.rglob("*TrialEvalBench_evaluator.zip"))
    trialeval_participant_archives = tuple(
        root.rglob("*TrialEvalBench_participant.zip")
    )
    trialeval_verification_archives = tuple(
        root.rglob("*TrialEvalBench_verification.zip")
    )
    trialdev_archives = tuple(root.rglob("*TrialDevBench_evaluator.zip"))
    if (
        len(trialeval_archives) != 1
        or len(trialeval_participant_archives) != 1
        or len(trialeval_verification_archives) != 1
        or len(trialdev_archives) != 1
    ):
        raise ValueError(
            "Grader concordance requires one TrialEval participant, evaluator, and verification "
            "archive and one TrialDev evaluator archive when expanded roots are absent."
        )
    with tempfile.TemporaryDirectory(
        prefix="trialagentbench-grader-concordance-"
    ) as temporary:
        materialized = Path(temporary)
        _safe_extract(trialeval_archives[0], materialized / "trialeval")
        _safe_extract(trialeval_verification_archives[0], materialized / "trialeval")
        _safe_extract(
            trialeval_participant_archives[0],
            materialized / "trialeval" / "public",
        )
        _safe_extract(trialdev_archives[0], materialized / "trialdev")
        return _run_grader_concordance_materialized(
            release_root=materialized,
            canonical_submissions=canonical_submissions,
            output_dir=output_dir,
            harness_executable=harness_executable,
        )


__all__ = [
    "CanonicalSubmissionV1",
    "CanonicalTrialEvalRouteWitnessGradeV1",
    "CanonicalTrialEvalRouteWitnessV1",
    "CanonicalTrialDevLaneSubmissionV1",
    "GradeRecordV1",
    "GraderConcordanceReportV1",
    "ScoringKeyV1",
    "TrialDevEvaluationTargetV1",
    "TrialDevLaneGradeV1",
    "grade_trialdev_lane_independently",
    "grade_trialeval_independently",
    "run_grader_concordance",
]
