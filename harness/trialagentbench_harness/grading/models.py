"""Typed inputs and outputs for the narrow deterministic grader."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AnalysisMethodResultKindV1 = Literal[
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


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NumericalAcceptanceEnvelopeV1(_ContractModel):
    """Prospective numerical equivalence bound for an accepted route."""

    schema_id: Literal["trialagentbench.numerical_acceptance_envelope/v1"]
    reporting_decimal_places: int = Field(ge=0, le=15)
    independent_max_abs_difference: float = Field(ge=0)
    public_verification_id: str = Field(min_length=1)
    independent_verification_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _independent_evidence(self) -> NumericalAcceptanceEnvelopeV1:
        if len(set(self.independent_verification_ids)) != len(self.independent_verification_ids):
            raise ValueError("independent_verification_ids must be unique")
        return self

    @property
    def absolute_tolerance(self) -> float:
        """Return rounding error plus independently measured implementation disagreement."""

        return 0.5 * 10.0 ** (-self.reporting_decimal_places) + self.independent_max_abs_difference


class RouteSignatureV1(_ContractModel):
    """Exact question and complete method identity declared for grading."""

    analysis_population_id: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    intercurrent_event_strategy_ids: tuple[str, ...] = ()
    assessment_horizon_days: float | None = Field(default=None, gt=0)
    treatment_id: str = Field(min_length=1)
    comparator_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    effect_scale: str = Field(min_length=1)
    analysis_method_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_semantics(self) -> RouteSignatureV1:
        for name, values in (("intercurrent_event_strategy_ids", self.intercurrent_event_strategy_ids),):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        return self


class NumericPointTargetV1(_ContractModel):
    kind: Literal["numeric_point"]
    value: float
    result_unit: str = Field(min_length=1)
    acceptance_envelope: NumericalAcceptanceEnvelopeV1
    require_confidence_interval: bool
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None

    @model_validator(mode="after")
    def _complete_interval(self) -> NumericPointTargetV1:
        endpoints = (self.confidence_interval_lower, self.confidence_interval_upper)
        if (endpoints[0] is None) != (endpoints[1] is None):
            raise ValueError("target confidence interval endpoints must be supplied together")
        if self.require_confidence_interval != (endpoints[0] is not None):
            raise ValueError(
                "require_confidence_interval must equal the presence of target confidence interval endpoints"
            )
        if endpoints[0] is not None and endpoints[1] is not None and endpoints[0] > endpoints[1]:
            raise ValueError("target confidence interval lower endpoint cannot exceed upper endpoint")
        return self


class NumericIntervalTargetV1(_ContractModel):
    kind: Literal["numeric_interval"]
    lower: float
    upper: float
    result_unit: str = Field(min_length=1)
    acceptance_envelope: NumericalAcceptanceEnvelopeV1

    @model_validator(mode="after")
    def _ordered(self) -> NumericIntervalTargetV1:
        if self.lower > self.upper:
            raise ValueError("target interval lower endpoint cannot exceed upper endpoint")
        return self


class NamedNumericValueV1(_ContractModel):
    name: str = Field(min_length=1)
    value: float


class NumericVectorTargetV1(_ContractModel):
    kind: Literal["numeric_vector"]
    components: tuple[NamedNumericValueV1, ...] = Field(min_length=1)
    result_unit: str = Field(min_length=1)
    acceptance_envelope: NumericalAcceptanceEnvelopeV1

    @model_validator(mode="after")
    def _unique_components(self) -> NumericVectorTargetV1:
        names = tuple(component.name for component in self.components)
        if len(set(names)) != len(names):
            raise ValueError("vector component names must be unique")
        return self


class StatisticalTestTargetV1(_ContractModel):
    kind: Literal["statistical_test"]
    p_value: float = Field(ge=0, le=1)
    reject_null: bool
    acceptance_envelope: NumericalAcceptanceEnvelopeV1


class CategoricalTargetV1(_ContractModel):
    kind: Literal["categorical"]
    credit_eligible_codes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_codes(self) -> CategoricalTargetV1:
        if len(set(self.credit_eligible_codes)) != len(self.credit_eligible_codes):
            raise ValueError("credit_eligible_codes must be unique")
        return self


ScoringTargetV1 = Annotated[
    NumericPointTargetV1
    | NumericIntervalTargetV1
    | NumericVectorTargetV1
    | StatisticalTestTargetV1
    | CategoricalTargetV1,
    Field(discriminator="kind"),
]


class AnalysisMethodBindingV1(_ContractModel):
    """Intrinsic properties selected by one canonical analysis method ID."""

    analysis_method_id: str = Field(min_length=1)
    estimator_family: str = Field(min_length=1)
    result_kind: AnalysisMethodResultKindV1
    uncertainty_method: str = Field(min_length=1)
    sensitivity_parameters: tuple[float, ...] = ()
    design_modifiers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _canonical_method(self) -> AnalysisMethodBindingV1:
        if tuple(sorted(set(self.design_modifiers))) != self.design_modifiers:
            raise ValueError("design_modifiers must be sorted and unique")
        if tuple(sorted(set(self.sensitivity_parameters))) != self.sensitivity_parameters:
            raise ValueError("sensitivity_parameters must be sorted and unique")
        if bool(self.sensitivity_parameters) != (self.result_kind == "sensitivity_set"):
            raise ValueError("sensitivity parameters are required exactly for sensitivity-set methods")
        return self


class CreditEligibleScoringRouteV1(_ContractModel):
    route_id: str = Field(min_length=1)
    signature: RouteSignatureV1
    method: AnalysisMethodBindingV1
    required_identification_assumptions: tuple[str, ...] = Field(min_length=1)
    required_diagnostics: tuple[str, ...] = ()
    planning_calculator_id: str | None = Field(default=None, min_length=1)
    target: ScoringTargetV1

    @model_validator(mode="after")
    def _canonical_diagnostics(self) -> CreditEligibleScoringRouteV1:
        if self.signature.analysis_method_id != self.method.analysis_method_id:
            raise ValueError("route signature and intrinsic method record must use one analysis method ID")
        for name, values in (
            ("required_identification_assumptions", self.required_identification_assumptions),
            ("required_diagnostics", self.required_diagnostics),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        if not _method_accepts_target(self.method.result_kind, self.target.kind):
            raise ValueError("intrinsic method result kind is incompatible with the scoring target")
        return self


def _method_accepts_target(method_kind: AnalysisMethodResultKindV1, target_kind: str) -> bool:
    """Return whether a complete method record can produce the stored target shape."""

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
    return target_kind in compatible_targets[method_kind]


class DataIntegrityTargetV1(_ContractModel):
    """Exact noncompensatory C5 repair target."""

    condition_id: Literal["exact_transport_row_duplication_v1"]
    affected_domain: str = Field(min_length=1)
    compound_key_fields: tuple[str, ...] = Field(min_length=1)
    observed_duplicate_group_count: int = Field(ge=1)
    observed_extra_row_count: int = Field(ge=1)
    repair_action: Literal["remove_one_exact_duplicate_copy"]
    repair_status: Literal["repaired"]
    post_repair_data_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_input_data_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _valid_target(self) -> DataIntegrityTargetV1:
        if len(set(self.compound_key_fields)) != len(self.compound_key_fields):
            raise ValueError("data-integrity compound-key fields must be unique")
        if self.analysis_input_data_checksum != self.post_repair_data_checksum:
            raise ValueError("C5 analysis input must equal the repaired domain content")
        return self


class CanonicalDataIntegrityRecordV1(_ContractModel):
    """Canonical participant report for the C5 data-integrity obligation."""

    condition_id: Literal["exact_transport_row_duplication_v1"]
    affected_domain: str = Field(min_length=1)
    compound_key_fields: tuple[str, ...] = Field(min_length=1)
    observed_duplicate_group_count: int = Field(ge=1)
    observed_extra_row_count: int = Field(ge=1)
    repair_action: Literal["remove_one_exact_duplicate_copy"]
    repair_status: Literal["repaired", "unexpected_data_integrity_state"]
    post_repair_data_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_input_data_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class ValidatedScoringKeyV1(_ContractModel):
    """Portable evaluator key produced by benchmark construction."""

    schema_id: Literal["trialagentbench.scoring_key/v1"]
    release_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    credit_eligible_routes: tuple[CreditEligibleScoringRouteV1, ...] = Field(min_length=1)
    data_integrity_target: DataIntegrityTargetV1 | None = None

    @model_validator(mode="after")
    def _unambiguous_routes(self) -> ValidatedScoringKeyV1:
        route_ids = tuple(route.route_id for route in self.credit_eligible_routes)
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("credit-eligible route IDs must be unique")
        signatures = tuple(route.signature for route in self.credit_eligible_routes)
        if len(set(signatures)) != len(signatures):
            raise ValueError("credit-eligible routes must have unique primary-binding signatures")
        question_bindings = {
            (
                signature.analysis_population_id,
                signature.estimand_id,
                signature.intercurrent_event_strategy_ids,
                signature.assessment_horizon_days,
                signature.treatment_id,
                signature.comparator_id,
                signature.endpoint_id,
                signature.effect_scale,
            )
            for signature in signatures
        }
        if len(question_bindings) != 1:
            raise ValueError("credit-eligible routes must answer one precise estimand on one effect scale")
        if (self.context_tier == "C5") != (self.data_integrity_target is not None):
            raise ValueError("exactly C5 scoring keys require a data-integrity target")
        return self


class NumericPointSubmissionV1(_ContractModel):
    kind: Literal["numeric_point"]
    value: float
    result_unit: str = Field(min_length=1)
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None

    @model_validator(mode="after")
    def _complete_interval(self) -> NumericPointSubmissionV1:
        endpoints = (self.confidence_interval_lower, self.confidence_interval_upper)
        if (endpoints[0] is None) != (endpoints[1] is None):
            raise ValueError("confidence interval endpoints must be supplied together")
        if endpoints[0] is not None and endpoints[1] is not None and endpoints[0] > endpoints[1]:
            raise ValueError("confidence interval lower endpoint cannot exceed upper endpoint")
        return self


class NumericIntervalSubmissionV1(_ContractModel):
    kind: Literal["numeric_interval"]
    lower: float
    upper: float
    result_unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> NumericIntervalSubmissionV1:
        if self.lower > self.upper:
            raise ValueError("submitted interval lower endpoint cannot exceed upper endpoint")
        return self


class NumericVectorSubmissionV1(_ContractModel):
    kind: Literal["numeric_vector"]
    components: tuple[NamedNumericValueV1, ...] = Field(min_length=1)
    result_unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_components(self) -> NumericVectorSubmissionV1:
        names = tuple(component.name for component in self.components)
        if len(set(names)) != len(names):
            raise ValueError("vector component names must be unique")
        return self


class StatisticalTestSubmissionV1(_ContractModel):
    kind: Literal["statistical_test"]
    p_value: float = Field(ge=0, le=1)
    reject_null: bool


class CategoricalSubmissionV1(_ContractModel):
    kind: Literal["categorical"]
    code: str = Field(min_length=1)


PrimaryResultV1 = Annotated[
    NumericPointSubmissionV1
    | NumericIntervalSubmissionV1
    | NumericVectorSubmissionV1
    | StatisticalTestSubmissionV1
    | CategoricalSubmissionV1,
    Field(discriminator="kind"),
]


class CanonicalSubmissionV1(_ContractModel):
    """One explicit primary analysis normalized before grading."""

    schema_id: Literal["trialagentbench.canonical_submission/v1"] = "trialagentbench.canonical_submission/v1"
    item_id: str = Field(min_length=1)
    primary: RouteSignatureV1
    diagnostic_ids: tuple[str, ...] = ()
    data_integrity_record: CanonicalDataIntegrityRecordV1 | None = None
    result: PrimaryResultV1

    @model_validator(mode="after")
    def _canonical_diagnostics(self) -> CanonicalSubmissionV1:
        if tuple(sorted(set(self.diagnostic_ids))) != self.diagnostic_ids:
            raise ValueError("diagnostic_ids must be sorted and unique")
        return self


FailureCode = Literal[
    "missing_primary_submission",
    "missing_required_deliverable",
    "item_mismatch",
    "unrecognized_primary_question",
    "unrecognized_primary_route",
    "missing_required_diagnostic",
    "missing_data_integrity_record",
    "unexpected_data_integrity_record",
    "data_integrity_condition_mismatch",
    "data_integrity_domain_mismatch",
    "data_integrity_compound_key_mismatch",
    "data_integrity_counts_mismatch",
    "data_integrity_repair_mismatch",
    "data_integrity_checksum_mismatch",
    "result_kind_mismatch",
    "result_unit_mismatch",
    "missing_confidence_interval",
    "confidence_interval_outside_tolerance",
    "numeric_result_outside_tolerance",
    "interval_result_outside_tolerance",
    "vector_components_mismatch",
    "vector_result_outside_tolerance",
    "test_decision_mismatch",
    "test_p_value_outside_tolerance",
    "categorical_result_not_credit_eligible",
]

GradeGateIdV1 = Literal[
    "submission",
    "question",
    "route",
    "evidence",
    "integrity",
    "result",
    "conformance",
    "decision",
]
GradeGateStatusV1 = Literal["passed", "failed", "not_reached", "not_applicable"]
GRADE_GATE_ORDER_V1: tuple[GradeGateIdV1, ...] = (
    "submission",
    "question",
    "route",
    "evidence",
    "integrity",
    "result",
    "conformance",
    "decision",
)

GradeComponentIdV1 = Literal[
    "submission",
    "question",
    "method",
    "evidence",
    "integrity",
    "result_structure",
    "route_comparison",
]
GradeComponentStatusV1 = Literal["passed", "failed", "not_evaluable", "not_applicable"]
GRADE_COMPONENT_ORDER_V1: tuple[GradeComponentIdV1, ...] = (
    "submission",
    "question",
    "method",
    "evidence",
    "integrity",
    "result_structure",
    "route_comparison",
)


class GradeGateRecordV1(_ContractModel):
    """Outcome of one ordered, noncompensatory grading gate."""

    gate_id: GradeGateIdV1
    status: GradeGateStatusV1
    failure_code: FailureCode | None = None

    @model_validator(mode="after")
    def _failure_is_explicit(self) -> GradeGateRecordV1:
        if (self.status == "failed") != (self.failure_code is not None):
            raise ValueError("failure_code is present exactly for a failed gate")
        return self


class GradeComponentRecordV1(_ContractModel):
    """Descriptive outcome of one grading component evaluated without route search."""

    component_id: GradeComponentIdV1
    status: GradeComponentStatusV1
    failure_code: FailureCode | None = None

    @model_validator(mode="after")
    def _failure_is_explicit(self) -> GradeComponentRecordV1:
        if (self.status == "failed") != (self.failure_code is not None):
            raise ValueError("failure_code is present exactly for a failed component")
        return self


class GradeRecordV1(_ContractModel):
    """Noncompensatory grade for one canonical primary submission."""

    schema_id: Literal["trialagentbench.grade_record/v1"] = "trialagentbench.grade_record/v1"
    release_id: str
    item_id: str
    usable_primary: bool
    route_match: bool
    obligations_met: bool
    result_match: bool
    passed: bool
    gates: tuple[GradeGateRecordV1, ...] = Field(min_length=8, max_length=8)
    components: tuple[GradeComponentRecordV1, ...] = Field(min_length=7, max_length=7)
    first_failure_gate: GradeGateIdV1 | None = None
    matched_route_id: str | None = None
    failure_codes: tuple[FailureCode, ...] = ()
    absolute_error: float | None = Field(default=None, ge=0)
    tolerance_ratio: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _noncompensatory_grade(self) -> GradeRecordV1:
        if tuple(record.gate_id for record in self.gates) != GRADE_GATE_ORDER_V1:
            raise ValueError("Grade gates must contain the canonical ordered eight-gate cascade.")
        if tuple(record.component_id for record in self.components) != GRADE_COMPONENT_ORDER_V1:
            raise ValueError("Grade components must contain the canonical seven-component record.")
        failed = tuple(record for record in self.gates if record.status == "failed")
        if len(failed) > 1:
            raise ValueError("Only the first failed grade gate may be marked failed.")
        if self.first_failure_gate != (None if not failed else failed[0].gate_id):
            raise ValueError("first_failure_gate must identify the sole failed gate.")
        if failed:
            failure_index = GRADE_GATE_ORDER_V1.index(failed[0].gate_id)
            if any(record.status != "not_reached" for record in self.gates[failure_index + 1 :]):
                raise ValueError("Every gate after the first failure must be not_reached.")
        elif any(record.status == "not_reached" for record in self.gates):
            raise ValueError("A complete grade cannot contain not_reached gates.")
        gate_status = {record.gate_id: record.status for record in self.gates}
        expected_usable = gate_status["submission"] == "passed"
        expected_route_match = gate_status["question"] == gate_status["route"] == "passed"
        expected_obligations = bool(
            expected_route_match
            and gate_status["evidence"] == "passed"
            and gate_status["integrity"] in {"passed", "not_applicable"}
        )
        expected_result_match = bool(
            expected_obligations and gate_status["result"] == "passed" and gate_status["conformance"] == "passed"
        )
        if self.usable_primary != expected_usable:
            raise ValueError("usable_primary must project the submission gate.")
        if self.route_match != expected_route_match:
            raise ValueError("route_match must project the question and route gates.")
        if self.obligations_met != expected_obligations:
            raise ValueError("obligations_met must project evidence and integrity gates.")
        if self.result_match != expected_result_match:
            raise ValueError("result_match must project result and conformance gates.")
        if self.route_match != (self.matched_route_id is not None):
            raise ValueError("Matched route identity must be present exactly when route_match is true.")
        if self.route_match and not self.usable_primary:
            raise ValueError("Route match requires a usable primary submission.")
        if self.obligations_met and not self.route_match:
            raise ValueError("Satisfied obligations require a matched route.")
        if self.result_match and not self.obligations_met:
            raise ValueError("Result match requires satisfied route obligations.")
        if (self.absolute_error is None) != (self.tolerance_ratio is None):
            raise ValueError("Numeric error and tolerance ratio must be reported together.")
        if self.absolute_error is not None and not self.obligations_met:
            raise ValueError("Numeric comparison requires a matched route with satisfied obligations.")
        complete = bool(not failed and all(record.status in {"passed", "not_applicable"} for record in self.gates))
        if self.passed != complete:
            raise ValueError("Conformance must equal the complete noncompensatory cascade.")
        if self.passed == bool(self.failure_codes):
            raise ValueError("Passed grades have no failure codes; rejected grades require one.")
        component_failures = tuple(record for record in self.components if record.status == "failed")
        if self.passed and component_failures:
            raise ValueError("Passed grades cannot contain a failed component.")
        if self.passed and any(record.status == "not_evaluable" for record in self.components):
            raise ValueError("Passed grades require every applicable component to be evaluable.")
        if failed:
            component_for_gate = {
                "submission": "submission",
                "question": "question",
                "route": "method",
                "evidence": "evidence",
                "integrity": "integrity",
                "result": "result_structure",
                "conformance": "route_comparison",
            }.get(failed[0].gate_id)
            if component_for_gate is not None:
                component = next(row for row in self.components if row.component_id == component_for_gate)
                if component.status != "failed" or component.failure_code != failed[0].failure_code:
                    raise ValueError("The primary failed gate must agree with its independently evaluated component.")
        return self


__all__ = [
    "AnalysisMethodBindingV1",
    "CreditEligibleScoringRouteV1",
    "CanonicalDataIntegrityRecordV1",
    "CanonicalSubmissionV1",
    "CategoricalSubmissionV1",
    "CategoricalTargetV1",
    "DataIntegrityTargetV1",
    "GRADE_GATE_ORDER_V1",
    "GRADE_COMPONENT_ORDER_V1",
    "GradeComponentRecordV1",
    "GradeGateRecordV1",
    "GradeRecordV1",
    "NamedNumericValueV1",
    "NumericalAcceptanceEnvelopeV1",
    "NumericIntervalSubmissionV1",
    "NumericIntervalTargetV1",
    "NumericPointSubmissionV1",
    "NumericPointTargetV1",
    "NumericVectorSubmissionV1",
    "NumericVectorTargetV1",
    "RouteSignatureV1",
    "StatisticalTestSubmissionV1",
    "StatisticalTestTargetV1",
    "ValidatedScoringKeyV1",
]
