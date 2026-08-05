"""Method-neutral contracts for benchmark submissions."""

from __future__ import annotations

from collections.abc import Collection
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_BOUNDED_DEVIATION_GRID_V1 = (0.05, 0.10, 0.20)

EstimatorFamilyV1 = Literal[
    "km",
    "coxph_binary",
    "coxph_time_interaction",
    "piecewise_cox",
    "weighted_logrank",
    "aft_parametric",
    "milestone_risk",
    "km_ipcw",
    "coxph_ipcw",
    "validated_endpoint",
    "cluster_parallel_participant_weighted",
    "stepped_wedge_period_adjusted",
    "group_sequential_adjustment",
    "competing_risks",
    "bounds",
    "qualified_nonidentification",
    "standardized_cox_g_computation",
    "other",
]
EffectScaleV1 = Literal[
    "log_hr",
    "hazard_ratio",
    "time_varying_log_hr",
    "piecewise_log_hr_vector",
    "weighted_logrank_test",
    "log_time_ratio",
    "cif_difference_tau",
    "rmst_difference_tau",
    "standardized_risk_difference_tau_reference",
    "standardized_risk_difference",
    "risk_difference",
    "risk_difference_tau",
    "milestone_risk_difference_tau",
    "risk_ratio",
    "odds_ratio",
    "rate_ratio",
    "bounds_interval",
    "survival_probability",
    "diagnostic_summary",
    "mean_difference",
    "not_applicable",
]
PrimaryResultKindV1 = Literal[
    "numeric_point",
    "numeric_vector",
    "statistical_test",
    "sensitivity_set",
    "identification_bound",
    "limitation",
    "abstention",
]
MethodModifierV1 = Literal[
    "ipcw_adjusted",
    "cluster_robust_inference",
    "participant_population_target",
    "stepped_wedge_period_adjusted",
    "group_sequential_adjustment",
    "misclassification_corrected",
    "reference_standardization",
    "flexible_model_form",
    "ph_robust_fixed_horizon",
]
DiagnosticIdV1 = Literal[
    "censoring_followup_public",
    "proportional_hazards_public",
    "randomization_integrity_public",
    "model_form_public",
    "cluster_structure_public",
    "secular_trend_public",
    "endpoint_ascertainment_public",
    "sequential_design_adjustment_public",
]
UncertaintyMethodV1 = Literal[
    "cluster_level_sandwich_delta",
    "coxph_information",
    "delete_group_jackknife_20",
    "group_sequential_adjusted_interval",
    "identified_set",
    "km_delta_greenwood",
    "leave_one_cluster_jackknife",
    "misclassification_delta",
    "participant_bootstrap",
    "sensitivity_envelope",
    "stratified_participant_bootstrap_1000",
    "wald_model_based",
    "greenwood",
    "not_applicable",
]
IdentificationAssumptionV1 = Literal[
    "bounded_unmeasured_deviation",
    "censoring_ignorability",
    "cluster_structure",
    "consistency",
    "endpoint_ascertainment",
    "measured_conditional_exchangeability",
    "independent_censoring",
    "model_form",
    "positivity",
    "proportional_hazards",
    "randomization_exchangeability",
    "randomization_integrity",
    "secular_trend",
    "sequential_design_adjustment",
]
ScientificIdV1 = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]


class StrictSubmissionModel(BaseModel):
    """Base configuration for public submission records."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class QuantityV1(StrictSubmissionModel):
    """A finite quantity with an explicit unit."""

    value: float
    unit: str = Field(min_length=1)


class SensitivityParameterV1(StrictSubmissionModel):
    """A bounded probability-scale sensitivity-model parameter."""

    value: float = Field(ge=0.0, le=1.0)
    unit: Literal["probability"]


class TimeHorizonV1(StrictSubmissionModel):
    """Assessment horizon for a time-indexed estimand."""

    value: float = Field(gt=0)
    unit: Literal["days"]


class EstimandDeclarationV1(StrictSubmissionModel):
    """Scientific question targeted by one submitted primary analysis.

    Declare exactly one of ``horizon`` and
    ``horizon_not_applicable_reason``. Intercurrent-event strategy identifiers
    must be unique.
    """

    estimand_id: ScientificIdV1
    population_id: ScientificIdV1
    treatment_id: ScientificIdV1
    comparator_id: ScientificIdV1
    endpoint_id: ScientificIdV1
    intercurrent_event_strategy_ids: tuple[ScientificIdV1, ...] = ()
    horizon: TimeHorizonV1 | None = None
    horizon_not_applicable_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _horizon_is_declared_or_explained(self) -> Self:
        if (self.horizon is None) == (self.horizon_not_applicable_reason is None):
            raise ValueError("declare a horizon or explain why the estimand is not time-indexed")
        if tuple(sorted(set(self.intercurrent_event_strategy_ids))) != self.intercurrent_event_strategy_ids:
            raise ValueError("intercurrent-event strategy identifiers must be sorted and unique")
        return self


class ConfidenceIntervalV1(StrictSubmissionModel):
    """Two-sided confidence interval with lower bound no greater than upper."""

    lower: float
    upper: float
    confidence_level: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("confidence interval lower bound exceeds upper bound")
        return self


class ScalarEstimateV1(StrictSubmissionModel):
    """A point estimate contained by its interval on one declared scale.

    Hazard-ratio estimates and interval limits must be strictly positive.
    """

    kind: Literal["scalar"] = "scalar"
    value: float
    effect_scale: EffectScaleV1
    unit: str = Field(min_length=1)
    interval: ConfidenceIntervalV1

    @model_validator(mode="after")
    def _estimate_is_inside_interval(self) -> Self:
        if not self.interval.lower <= self.value <= self.interval.upper:
            raise ValueError("point estimate must lie within its confidence interval")
        if self.effect_scale == "hazard_ratio" and self.interval.lower <= 0.0:
            raise ValueError("hazard-ratio estimates and interval limits must be positive")
        return self


class IdentifiedIntervalV1(StrictSubmissionModel):
    """An ordered identified interval without a point claim."""

    kind: Literal["identified_interval"] = "identified_interval"
    lower: float
    upper: float
    effect_scale: EffectScaleV1
    unit: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("identified interval lower bound exceeds upper bound")
        return self


class VectorPointV1(StrictSubmissionModel):
    """One coordinate of a vector- or curve-valued estimate."""

    component_id: ScientificIdV1
    index: float
    value: float


class VectorEstimateV1(StrictSubmissionModel):
    """A vector or curve whose point indices are strictly increasing."""

    kind: Literal["vector"] = "vector"
    points: tuple[VectorPointV1, ...] = Field(min_length=2)
    index_unit: str = Field(min_length=1)
    effect_scale: EffectScaleV1
    unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def _indices_are_strictly_increasing(self) -> Self:
        indices = [point.index for point in self.points]
        if any(left >= right for left, right in zip(indices, indices[1:], strict=False)):
            raise ValueError("vector indices must be strictly increasing")
        component_ids = tuple(point.component_id for point in self.points)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("vector component identifiers must be unique")
        return self


class TestResultV1(StrictSubmissionModel):
    """A statistical-test result with explicit direction and parameters."""

    kind: Literal["statistical_test"] = "statistical_test"
    statistic: float
    p_value: float = Field(ge=0, le=1)
    reject_null: bool
    effect_scale: Literal["weighted_logrank_test"]
    unit: str = Field(min_length=1)
    alternative: Literal["two_sided", "greater", "less"] = "two_sided"
    rho: float = Field(ge=0)
    gamma: float = Field(ge=0)


class DiagnosticMeasureV1(StrictSubmissionModel):
    """One explicitly rounded diagnostic quantity."""

    metric_id: ScientificIdV1 = Field(
        description=(
            "Participant-chosen scientific label for the reported quantity. "
            "Diagnostic credit is based on exact value, unit, diagnostic identity, and public-source replay."
        )
    )
    value: float
    unit: str = Field(min_length=1)
    decimal_places: int = Field(ge=0, le=12)


class ProbabilityMeasureV1(StrictSubmissionModel):
    """One explicitly rounded probability in the closed unit interval."""

    metric_id: ScientificIdV1
    value: float = Field(ge=0.0, le=1.0)
    unit: Literal["probability"]
    decimal_places: int = Field(ge=0, le=12)


class DiagnosticTestResultV1(StrictSubmissionModel):
    """A diagnostic test statistic and its reported probability."""

    kind: Literal["diagnostic_test"] = "diagnostic_test"
    statistic: DiagnosticMeasureV1
    p_value: ProbabilityMeasureV1
    alternative: Literal["two_sided", "greater", "less"] = "two_sided"


class DiagnosticSummaryResultV1(StrictSubmissionModel):
    """A descriptive diagnostic whose finite measure identifiers are unique."""

    kind: Literal["diagnostic_summary"] = "diagnostic_summary"
    measures: tuple[DiagnosticMeasureV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _measure_ids_are_unique(self) -> Self:
        metric_ids = tuple(measure.metric_id for measure in self.measures)
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("diagnostic summary metric IDs must be unique")
        return self


class FactualPremiseResultV1(StrictSubmissionModel):
    """A conclusion about one premise that is directly stated in a public artifact."""

    kind: Literal["factual_premise"] = "factual_premise"
    premise_id: Literal[
        "randomized_assignment_declared",
        "cluster_randomization_declared",
        "group_sequential_plan_declared",
        "unmeasured_prognostic_censoring_factor",
    ]
    conclusion: Literal["supported", "not_supported", "unresolved"]


class QualifiedNonIdentificationV1(StrictSubmissionModel):
    """A justified conclusion that the requested effect is not identified."""

    kind: Literal["non_identification"] = "non_identification"
    conclusion_code: ScientificIdV1
    effect_scale: EffectScaleV1
    unit: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    identified_set: IdentifiedIntervalV1 | None = None
    additional_assumption_required: str = Field(min_length=1)


PrimaryResultV1 = Annotated[
    ScalarEstimateV1 | IdentifiedIntervalV1 | VectorEstimateV1 | TestResultV1 | QualifiedNonIdentificationV1,
    Field(discriminator="kind"),
]
EvidenceResultV1 = Annotated[
    ScalarEstimateV1
    | IdentifiedIntervalV1
    | VectorEstimateV1
    | TestResultV1
    | QualifiedNonIdentificationV1
    | DiagnosticTestResultV1
    | DiagnosticSummaryResultV1
    | FactualPremiseResultV1,
    Field(discriminator="kind"),
]


class EstimatorDeclarationV1(StrictSubmissionModel):
    """One complete analysis method selected from the public dictionary."""

    analysis_method_id: ScientificIdV1 = Field(
        description="Exact method identifier from the participant method dictionary."
    )
    implementation: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional audit note describing the implementation. It does not change method identity or credit."
        ),
    )
    qualifications: tuple[str, ...] = Field(
        default=(),
        description=(
            "Optional audit-only assumptions or limitations. Additional qualifications do not alter the selected "
            "method; required item evidence must still be supplied as typed evidence records."
        ),
    )


class EvidenceRecordV1(StrictSubmissionModel):
    """One executed evidence record with fields determined by its evidence type.

    Diagnostic and validity records require ``diagnostic_id``, a diagnostic
    or factual-premise result, and no estimator. Sensitivity and supporting-analysis records
    require an estimator and a non-diagnostic result. Data-quality records use
    a non-diagnostic result and no estimator. ``sensitivity_parameter`` is
    permitted only for sensitivity records.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"evidence_type": {"enum": ["diagnostic", "validity"]}},
                        "required": ["evidence_type"],
                    },
                    "then": {
                        "required": ["diagnostic_id"],
                        "properties": {
                            "diagnostic_id": {"not": {"type": "null"}},
                            "estimator": {"type": "null"},
                            "sensitivity_parameter": {"type": "null"},
                            "result": {
                                "properties": {
                                    "kind": {
                                        "enum": [
                                            "diagnostic_test",
                                            "diagnostic_summary",
                                            "factual_premise",
                                        ]
                                    }
                                },
                                "required": ["kind"],
                            },
                        },
                    },
                    "else": {
                        "properties": {
                            "diagnostic_id": {"type": "null"},
                            "result": {
                                "not": {
                                    "properties": {
                                        "kind": {
                                            "enum": [
                                                "diagnostic_test",
                                                "diagnostic_summary",
                                                "factual_premise",
                                            ]
                                        }
                                    },
                                    "required": ["kind"],
                                }
                            },
                        }
                    },
                },
                {
                    "if": {
                        "properties": {"evidence_type": {"enum": ["sensitivity", "supporting_analysis"]}},
                        "required": ["evidence_type"],
                    },
                    "then": {
                        "required": ["estimator"],
                        "properties": {"estimator": {"not": {"type": "null"}}},
                    },
                    "else": {"properties": {"estimator": {"type": "null"}}},
                },
                {
                    "if": {
                        "properties": {"evidence_type": {"const": "sensitivity"}},
                        "required": ["evidence_type"],
                    },
                    "else": {"properties": {"sensitivity_parameter": {"type": "null"}}},
                },
            ]
        }
    )

    evidence_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    evidence_type: Literal["diagnostic", "sensitivity", "validity", "supporting_analysis", "data_quality"]
    principle: Literal[
        "confounding",
        "proportional_hazards",
        "censoring",
        "missingness",
        "uncertainty",
        "sensitivity",
        "data_quality",
        "design_validity",
        "safety",
        "cost",
        "objective_alignment",
    ]
    operation: Literal[
        "assessment",
        "adjustment",
        "estimation",
        "sensitivity_analysis",
        "data_validation",
        "tradeoff_analysis",
    ]
    diagnostic_id: DiagnosticIdV1 | None = Field(
        default=None,
        description="Required only for diagnostic and validity evidence; omit for every other evidence type.",
    )
    sensitivity_parameter: SensitivityParameterV1 | None = Field(
        default=None,
        description="Optional only for sensitivity evidence; omit for every other evidence type.",
    )
    estimator: EstimatorDeclarationV1 | None = Field(
        default=None,
        description=(
            "Required for sensitivity and supporting-analysis evidence; omit for diagnostic, validity, and "
            "data-quality evidence."
        ),
    )
    target: str = Field(min_length=1)
    result: EvidenceResultV1 = Field(
        description=(
            "Diagnostic and validity evidence require diagnostic_test, diagnostic_summary, or factual_premise; "
            "every other evidence type requires a non-diagnostic result shape."
        )
    )
    interpretation: str = Field(min_length=1)
    source_artifacts: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _diagnostic_identity_matches_evidence_type(self) -> Self:
        requires_id = self.evidence_type in {"diagnostic", "validity"}
        if requires_id != (self.diagnostic_id is not None):
            raise ValueError("diagnostic and validity evidence require diagnostic_id; other evidence must omit it")
        if self.sensitivity_parameter is not None and self.evidence_type != "sensitivity":
            raise ValueError("sensitivity_parameter is valid only for sensitivity evidence")
        diagnostic_result = self.result.kind in {
            "diagnostic_test",
            "diagnostic_summary",
            "factual_premise",
        }
        if requires_id != diagnostic_result:
            raise ValueError("diagnostic and validity evidence require a diagnostic result shape")
        requires_estimator = self.evidence_type in {"sensitivity", "supporting_analysis"}
        if requires_estimator != (self.estimator is not None):
            raise ValueError(
                "sensitivity and supporting analyses require estimator; diagnostic, validity, and data-quality "
                "evidence must omit it"
            )
        return self


class PrimaryAnalysisV1(StrictSubmissionModel):
    """The sole analysis submitted for primary deterministic credit."""

    declared_primary: Literal[True]
    estimand: EstimandDeclarationV1
    estimator: EstimatorDeclarationV1
    result_kind: PrimaryResultKindV1
    result: PrimaryResultV1
    favorable_direction: Literal["higher", "lower", "neither"]
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _result_shape_matches_declared_kind(self) -> Self:
        if isinstance(self.result, ScalarEstimateV1):
            compatible = {"numeric_point"}
        elif isinstance(self.result, IdentifiedIntervalV1):
            compatible = {
                "identification_bound",
            }
        elif isinstance(self.result, VectorEstimateV1):
            compatible = {"numeric_vector", "sensitivity_set"}
        elif isinstance(self.result, TestResultV1):
            compatible = {"statistical_test"}
        else:
            compatible = {"limitation", "abstention"}
        if self.result_kind not in compatible:
            raise ValueError(f"result_kind {self.result_kind!r} is incompatible with {self.result.kind!r}")
        return self


class PlanningSensitivityResultV1(StrictSubmissionModel):
    """Sample size under one event-probability sensitivity assumption."""

    event_probability: float = Field(gt=0.0, le=1.0)
    target_sample_size: int = Field(gt=0)


class PlanningResultV1(StrictSubmissionModel):
    """Optional event-driven plan with unique sensitivity probabilities."""

    method_id: Literal["schoenfeld_logrank_v1"]
    estimand_id: ScientificIdV1
    alpha_two_sided: float = Field(gt=0.0, lt=1.0)
    power: float = Field(gt=0.0, lt=1.0)
    treated_allocation_fraction: float = Field(gt=0.0, lt=1.0)
    event_probability: float = Field(gt=0.0, le=1.0)
    followup_horizon_dy: float = Field(gt=0.0)
    multiplicity_adjustment: Literal["none"]
    required_events: int = Field(gt=0)
    target_sample_size: int = Field(gt=0)
    sensitivity: tuple[PlanningSensitivityResultV1, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _sensitivity_probabilities_are_unique(self) -> Self:
        probabilities = tuple(row.event_probability for row in self.sensitivity)
        if len(probabilities) != len(set(probabilities)):
            raise ValueError("planning sensitivity event probabilities must be unique")
        return self


class ReconstructionSummaryV1(StrictSubmissionModel):
    """Declared analysis-table reconstruction counts and executed checks."""

    n_subjects: int = Field(ge=0)
    n_primary_population: int = Field(ge=0)
    n_events: int = Field(ge=0)
    n_censored: int = Field(ge=0)
    checks_performed: tuple[str, ...] = ()
    notes: str = Field(min_length=1)
    source_artifacts: tuple[str, ...] = Field(min_length=1)


class DataIntegrityRecordV1(StrictSubmissionModel):
    """Observed C5 condition, exact repair, and analysis-input identity."""

    condition_id: Literal["exact_transport_row_duplication_v1"]
    affected_domain: str = Field(min_length=1)
    compound_key_fields: tuple[str, ...] = Field(min_length=1)
    observed_duplicate_group_count: int = Field(ge=1)
    observed_extra_row_count: int = Field(ge=1)
    repair_action: Literal["remove_one_exact_duplicate_copy"]
    repair_status: Literal["repaired", "unexpected_data_integrity_state"]
    post_repair_data_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_input_data_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_integrity_record(self) -> Self:
        if len(set(self.compound_key_fields)) != len(self.compound_key_fields):
            raise ValueError("data-integrity compound-key fields must be unique")
        if self.repair_status == "repaired" and (self.analysis_input_data_checksum != self.post_repair_data_checksum):
            raise ValueError("repaired C5 analysis must use the post-repair data content")
        return self


class TrialEvalSubmissionV1(StrictSubmissionModel):
    """Canonical TrialEval submission with linked analysis references.

    Evidence and data-resolution identifiers must be unique, and every
    evidence record must be linked once from the primary analysis. A planning
    result must target the primary estimand and is valid only for a scalar
    log-hazard-ratio primary result.
    """

    schema_id: Literal["trialagentbench.trialeval_submission/v1"] = "trialagentbench.trialeval_submission/v1"
    task_id: str = Field(min_length=1)
    primary_analysis: PrimaryAnalysisV1
    evidence: tuple[EvidenceRecordV1, ...] = ()
    planning: PlanningResultV1 | None = None
    reconstruction: ReconstructionSummaryV1 | None = None
    data_integrity_record: DataIntegrityRecordV1 | None = None
    limitations: tuple[str, ...] = Field(
        default=(),
        description=(
            "Optional audit-only free-text limitations. A score-bearing non-identification conclusion belongs "
            "in primary_analysis.result as a typed qualified_non_identification result."
        ),
    )

    @model_validator(mode="after")
    def _evidence_links_resolve(self) -> Self:
        identifiers = [record.evidence_id for record in self.evidence]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evidence IDs must be unique")
        unknown = sorted(set(self.primary_analysis.evidence_ids) - set(identifiers))
        if unknown:
            raise ValueError(f"primary analysis references unknown evidence IDs: {unknown}")
        unlinked = sorted(set(identifiers) - set(self.primary_analysis.evidence_ids))
        if unlinked:
            raise ValueError(f"TrialEval evidence records must be linked from the primary analysis: {unlinked}")
        if self.planning is not None:
            if self.planning.estimand_id != self.primary_analysis.estimand.estimand_id:
                raise ValueError("planning estimand_id must match the declared primary estimand")
            primary_result = self.primary_analysis.result
            if not isinstance(primary_result, ScalarEstimateV1) or primary_result.effect_scale != "log_hr":
                raise ValueError("Schoenfeld planning requires a scalar log_hr primary result")
        return self


def missing_trialeval_required_deliverables_v1(
    submission: TrialEvalSubmissionV1,
    *,
    required_deliverables: Collection[str],
) -> tuple[str, ...]:
    """Return required semantic deliverables absent from one typed submission."""

    presence = {
        "primary_analysis": True,
        "evidence": bool(submission.evidence),
        "limitations": bool(submission.limitations),
        "reconstruction": submission.reconstruction is not None,
        "data_integrity_record": submission.data_integrity_record is not None,
    }
    unknown = sorted(set(required_deliverables) - set(presence))
    if unknown:
        raise ValueError(f"Unknown semantic submission deliverables: {unknown!r}")
    return tuple(sorted(name for name in required_deliverables if not presence[name]))


def validate_trialeval_required_deliverables_v1(
    submission: TrialEvalSubmissionV1,
    *,
    required_deliverables: Collection[str],
) -> TrialEvalSubmissionV1:
    """Reject a typed submission that omits participant-declared obligations."""

    missing = missing_trialeval_required_deliverables_v1(
        submission,
        required_deliverables=required_deliverables,
    )
    if missing:
        raise ValueError(f"Submission omits required semantic deliverables: {missing!r}")
    return submission
