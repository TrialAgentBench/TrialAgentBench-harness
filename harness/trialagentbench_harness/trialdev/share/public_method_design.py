"""Strict public TrialDev method and design cell contracts."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.trialdev.share.hashing import compute_sha256_hex

RandomizedPhaseIdV1 = Literal["phase1", "phase2", "phase3"]
TrialDevObjectiveIdV1 = Literal[
    "benefit_risk",
    "cost_effective_best",
    "net_clinical_value_under_budget",
    "pure_efficacy",
]


class TrialDevPublicEfficacyEndpointV1(BaseModel):
    """Method-neutral time-to-event efficacy endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint_id: str = Field(..., min_length=1)
    time_column: str = Field(..., min_length=1)
    event_column: str = Field(..., min_length=1)
    competing_time_column: str = Field(..., min_length=1)
    competing_event_column: str = Field(..., min_length=1)
    horizon_days: int = Field(..., ge=1)
    estimator_id: Literal["standardized_aalen_johansen_cumulative_incidence"]
    effect_scale_id: Literal["risk_difference_control_minus_candidate"]
    effect_orientation_id: Literal["positive_values_favour_candidate"]


class TrialDevPublicUtilityEventV1(BaseModel):
    """One adverse-event contrast used in a participant-visible utility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_source: Literal["serious_safety", "discontinuation", "ltfu"]
    endpoint_id: str = Field(..., min_length=1)
    time_column: str = Field(..., min_length=1)
    event_column: str = Field(..., min_length=1)
    competing_time_column: str = Field(..., min_length=1)
    competing_event_column: str = Field(..., min_length=1)
    horizon_days: int = Field(..., ge=1)
    estimator_id: Literal["standardized_aalen_johansen_cumulative_incidence"]
    effect_scale_id: Literal["risk_difference_candidate_minus_control"]
    effect_orientation_id: Literal["positive_values_favour_control"]


class TrialDevPublicUtilityComponentV1(BaseModel):
    """One declared component of a participant-visible objective."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str = Field(..., min_length=1)
    source: Literal["efficacy_gain", "serious_safety", "discontinuation", "ltfu", "candidate_cost"]
    weight: float = Field(..., gt=0.0)
    direction: Literal["benefit", "penalty"]


class TrialDevPublicObjectiveSpecV1(BaseModel):
    """One method-neutral participant-visible clinical objective."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_public_objective_spec_v1"]
    objective_id: TrialDevObjectiveIdV1
    efficacy_endpoints: tuple[TrialDevPublicEfficacyEndpointV1, ...] = Field(..., min_length=1)
    utility_event_definitions: tuple[TrialDevPublicUtilityEventV1, ...] = Field(default_factory=tuple)
    utility_components: tuple[TrialDevPublicUtilityComponentV1, ...] = Field(..., min_length=1)
    candidate_costs: dict[str, float] = Field(default_factory=dict)
    indifference_margin: float = Field(..., ge=0.0)
    sensitivity_indifference_margins: tuple[float, ...] = Field(..., min_length=3)
    penalty_weight_sensitivity_multipliers: tuple[float, ...] = Field(..., min_length=3)
    utility_unit: Literal["dimensionless_declared_net_benefit"]
    policy_basis: Literal["scenario_declared_target_product_profile"]
    decision_direction: Literal["maximize"]
    public_evidence_basis: tuple[str, ...] = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    checksum: str = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_objective(self) -> TrialDevPublicObjectiveSpecV1:
        """Require coherent cost and sensitivity semantics plus exact checksum."""

        uses_cost = any(component.source == "candidate_cost" for component in self.utility_components)
        if uses_cost != bool(self.candidate_costs):
            raise ValueError("candidate_costs must be present exactly for candidate-cost objectives.")
        required_events = {
            component.source
            for component in self.utility_components
            if component.source in {"serious_safety", "discontinuation", "ltfu"}
        }
        declared_events = {definition.component_source for definition in self.utility_event_definitions}
        if len(declared_events) != len(self.utility_event_definitions) or declared_events != required_events:
            raise ValueError("utility event definitions must cover each event-based utility component exactly once.")
        margins = tuple(float(value) for value in self.sensitivity_indifference_margins)
        if tuple(sorted(set(margins))) != margins or not any(
            abs(value - float(self.indifference_margin)) <= 1e-12 for value in margins
        ):
            raise ValueError("objective sensitivity margins must be ordered, unique, and include the primary.")
        multipliers = tuple(float(value) for value in self.penalty_weight_sensitivity_multipliers)
        if (
            tuple(sorted(set(multipliers))) != multipliers
            or any(value <= 0.0 for value in multipliers)
            or not any(abs(value - 1.0) <= 1e-12 for value in multipliers)
        ):
            raise ValueError("penalty multipliers must be positive, ordered, unique, and include one.")
        payload = self.model_dump(mode="json", exclude={"checksum"}, exclude_none=True)
        if self.checksum != compute_sha256_hex(payload):
            raise ValueError("TrialDevPublicObjectiveSpecV1 checksum mismatch.")
        return self


class _TrialDevPublicObservationalAnalysisBaseV1(BaseModel):
    """Shared participant-visible observational method fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_public_observational_analysis_spec_v1"]
    phase_id: Literal["observational_review"]
    adjustment_covariates: tuple[str, ...] = Field(..., min_length=1)
    analysis_population: Literal["complete_on_declared_adjustment_covariates"]
    categorical_encoding: Literal["reference_level_one_hot"]
    sensitivity_estimator_ids: tuple[Literal["raw_observed"], ...]
    uncertainty_estimator_id: Literal["refitted_nuisance_participant_nonparametric_bootstrap"]
    uncertainty_kind: Literal["two_sided_confidence_interval"]
    confidence_level: float = Field(..., gt=0.5, lt=1.0)
    effect_scale_id: Literal["dimensionless_declared_net_benefit"]
    horizon_source: Literal["objective.efficacy_endpoints[].horizon_days"]
    bootstrap_replicates: int = Field(..., ge=20)
    bootstrap_seed: int = Field(
        ...,
        ge=0,
        description="Seed for the declared participant-level bootstrap resamples.",
    )
    bootstrap_rng_id: Literal["numpy_default_rng_pcg64"]
    bootstrap_standard_error_ddof: Literal[1]
    confidence_interval_id: Literal["normal_critical_value_times_bootstrap_standard_error"]
    identification_assumptions: tuple[str, ...] = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_shared_analysis_spec(self) -> _TrialDevPublicObservationalAnalysisBaseV1:
        """Require unique adjustment and sensitivity declarations."""

        if len(set(self.adjustment_covariates)) != len(self.adjustment_covariates):
            raise ValueError("adjustment_covariates must be unique.")
        if len(set(self.sensitivity_estimator_ids)) != len(self.sensitivity_estimator_ids):
            raise ValueError("sensitivity_estimator_ids must be unique.")
        return self


class TrialDevPublicObservationalAnalysisSpecV1(_TrialDevPublicObservationalAnalysisBaseV1):
    """Multinomial-propensity weighted, stratum-standardized cumulative incidence."""

    method_route_id: Literal["trialdev.observational.multinomial_propensity_weighted_stratified_aalen_johansen.v1"]
    calculator_id: Literal["public_observational_ipw_utility_v1"]
    primary_estimator_id: Literal["multinomial_propensity_weighted_stratified_aalen_johansen"]
    exact_stratification_covariates: tuple[str, ...] = Field(default_factory=tuple)
    quantile_stratification_bins: dict[str, int] = Field(default_factory=dict)
    propensity_solver_id: Literal["deterministic_multinomial_logit_lbfgs"]
    propensity_max_iterations: int = Field(..., ge=1)
    propensity_tolerance: float = Field(..., gt=0.0, lt=1.0)
    propensity_l2_penalty: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def validate_stratification(self) -> TrialDevPublicObservationalAnalysisSpecV1:
        """Require nested, non-overlapping standardization strata."""

        exact = self.exact_stratification_covariates
        quantile = tuple(self.quantile_stratification_bins)
        if len(set(exact)) != len(exact) or set(exact) & set(quantile):
            raise ValueError("stratification covariates must be unique across exact and quantile rules.")
        if not (set(exact) | set(quantile)) <= set(self.adjustment_covariates):
            raise ValueError("stratification covariates must belong to the declared adjustment set.")
        if any(not covariate or bins < 2 for covariate, bins in self.quantile_stratification_bins.items()):
            raise ValueError("quantile stratification requires named covariates and at least two bins.")
        return self


class TrialDevPublicEntropyBalancedAnalysisSpecV1(_TrialDevPublicObservationalAnalysisBaseV1):
    """Entropy-balanced, directly standardized cumulative incidence."""

    method_route_id: Literal["trialdev.observational.entropy_balanced_standardized_aalen_johansen.v1"]
    calculator_id: Literal["public_observational_entropy_balance_utility_v1"]
    primary_estimator_id: Literal["entropy_balanced_standardized_aalen_johansen"]
    calibration_solver_id: Literal["entropy_balancing_dual_lbfgs"]
    calibration_max_iterations: int = Field(..., ge=1)
    calibration_tolerance: float = Field(..., gt=0.0, lt=1.0)
    maximum_mean_balance_error: float = Field(..., gt=0.0, lt=0.1)


TrialDevPublicObservationalMethodSpecV1 = Annotated[
    TrialDevPublicObservationalAnalysisSpecV1 | TrialDevPublicEntropyBalancedAnalysisSpecV1,
    Field(discriminator="primary_estimator_id"),
]


class TrialDevPublicObjectiveCharterV1(BaseModel):
    """Participant-visible clinical objective policy without analysis methods."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_public_objective_charter_v1"]
    version: Literal["v1"]
    scenario_id: str = Field(..., min_length=1)
    confidence_level: float = Field(..., gt=0.5, lt=1.0)
    numeric_reporting_decimal_places: int = Field(..., ge=2, le=6)
    decision_charter_checksum: str = Field(..., min_length=64, max_length=64)
    objectives: tuple[TrialDevPublicObjectiveSpecV1, ...] = Field(..., min_length=1)
    checksum: str = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_manifest(self) -> TrialDevPublicObjectiveCharterV1:
        """Validate the objective-charter checksum."""

        _require_checksum(self)
        return self


class TrialDevPublicAssignmentPrognosticFactorV1(BaseModel):
    """Factual measurement status for a pretreatment assignment factor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_id: str = Field(..., min_length=1)
    used_in_treatment_assignment: bool
    prognostic_for_primary_endpoint: bool
    recorded_in_observational_extract: bool
    released_column_id: str | None = Field(default=None, min_length=1)
    provenance_statement: str = Field(..., min_length=20)

    @model_validator(mode="after")
    def validate_measurement_status(self) -> TrialDevPublicAssignmentPrognosticFactorV1:
        """Bind released column identity exactly to recorded status."""

        if self.recorded_in_observational_extract != (self.released_column_id is not None):
            raise ValueError("released_column_id is required exactly when the factor was recorded.")
        return self


class TrialDevPublicObservationalMethodCatalogV1(BaseModel):
    """Participant-visible methods and factual observational provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_public_observational_method_catalog_v1"]
    version: Literal["v1"]
    scenario_id: str = Field(..., min_length=1)
    assignment_prognostic_factors: tuple[TrialDevPublicAssignmentPrognosticFactorV1, ...] = Field(
        ...,
        min_length=1,
    )
    confidence_level: float = Field(..., gt=0.5, lt=1.0)
    decision_charter_checksum: str = Field(..., min_length=64, max_length=64)
    methods: tuple[TrialDevPublicObservationalMethodSpecV1, ...] = Field(..., min_length=2)
    checksum: str = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_catalog(self) -> TrialDevPublicObservationalMethodCatalogV1:
        """Require unique cells sharing the charter confidence level."""

        factor_ids = tuple(factor.factor_id for factor in self.assignment_prognostic_factors)
        if len(factor_ids) != len(set(factor_ids)):
            raise ValueError("Observational provenance factors must be unique.")
        cell_ids = tuple(method.method_route_id for method in self.methods)
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Observational method-route ids must be unique.")
        calculator_ids = tuple(method.calculator_id for method in self.methods)
        if len(calculator_ids) != len(set(calculator_ids)):
            raise ValueError("Observational calculator ids must be unique.")
        estimator_ids = tuple(method.primary_estimator_id for method in self.methods)
        if len(estimator_ids) != len(set(estimator_ids)):
            raise ValueError("Observational estimator ids must be unique.")
        if any(abs(method.confidence_level - self.confidence_level) > 1e-12 for method in self.methods):
            raise ValueError("Observational method confidence differs from the catalog.")
        _require_checksum(self)
        return self

    def method_for_calculator(self, calculator_id: str) -> TrialDevPublicObservationalMethodSpecV1:
        """Return the unique method executed by a named public calculator."""

        matches = tuple(method for method in self.methods if method.calculator_id == calculator_id)
        if len(matches) != 1:
            raise ValueError(f"Observational catalog has no unique method for calculator {calculator_id!r}.")
        return matches[0]

    def method_by_cell_id(self, method_route_id: str) -> TrialDevPublicObservationalMethodSpecV1:
        """Return the unique method with a declared cell identifier."""

        matches = tuple(method for method in self.methods if method.method_route_id == method_route_id)
        if len(matches) != 1:
            raise ValueError(f"Observational catalog has no unique method route {method_route_id!r}.")
        return matches[0]


class TrialDevPhaseAnalysisMethodRouteV1(BaseModel):
    """One accepted randomized-phase analysis cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_route_id: str = Field(..., min_length=1)
    phase_id: RandomizedPhaseIdV1
    calculator_id: Literal[
        "aalen_johansen_safety_bundle_v1",
        "aalen_johansen_efficacy_safety_bundle_v1",
    ]
    estimator_id: Literal["observed:aalen_johansen_cif_tau"]
    efficacy_estimand_id_template: str | None = Field(default=None, min_length=1)
    efficacy_effect_scale_id: Literal["risk_difference_control_minus_treatment"] | None = None
    efficacy_orientation_id: Literal["positive_values_favour_treatment"] | None = None
    safety_estimand_id_template: Literal["{safety_component_id}:cumulative_incidence_at_horizon"]
    safety_absolute_risk_scale_id: Literal["absolute_risk"]
    safety_excess_risk_scale_id: Literal["risk_difference_treatment_minus_control"]
    safety_reported_measure_ids: tuple[
        Literal[
            "treatment_absolute_risk",
            "control_absolute_risk",
            "risk_difference_treatment_minus_control",
        ],
        ...,
    ]
    safety_uncertainty_scope_id: Literal["two_sided_confidence_interval_per_safety_component_and_measure"]
    safety_orientation_id: Literal["absolute_risk_higher_is_worse_and_risk_difference_positive_is_treatment_harm"]
    result_shape: Literal["safety_component_bundle", "efficacy_safety_bundle"]
    uncertainty_kind: Literal["two_sided_confidence_interval"]
    confidence_level: float = Field(..., gt=0.5, lt=1.0)
    horizon_source: Literal["request.follow_up_days"]
    analysis_population: Literal["all_randomized_participants"]
    censoring_assumption_id: Literal["independent_censoring_conditional_on_randomized_arm"]
    loss_to_follow_up_construction_id: Literal["arm_conditional_random_permutation_v1"]
    safety_component_ids: tuple[Literal["serious_ae", "discontinuation"], ...]

    @model_validator(mode="after")
    def validate_phase_semantics(self) -> TrialDevPhaseAnalysisMethodRouteV1:
        """Reject phase-swapped or incomplete method routes."""

        expected = {
            "phase1": (
                "trialdev.phase1.aalen_johansen_safety_bundle.v1",
                "aalen_johansen_safety_bundle_v1",
                "safety_component_bundle",
                None,
                None,
                None,
            ),
            "phase2": (
                "trialdev.phase2.aalen_johansen_efficacy_safety.v1",
                "aalen_johansen_efficacy_safety_bundle_v1",
                "efficacy_safety_bundle",
                "{treatment_discontinuation_strategy}:cumulative_incidence_at_horizon",
                "risk_difference_control_minus_treatment",
                "positive_values_favour_treatment",
            ),
            "phase3": (
                "trialdev.phase3.aalen_johansen_efficacy_safety.v1",
                "aalen_johansen_efficacy_safety_bundle_v1",
                "efficacy_safety_bundle",
                "{treatment_discontinuation_strategy}:cumulative_incidence_at_horizon",
                "risk_difference_control_minus_treatment",
                "positive_values_favour_treatment",
            ),
        }[self.phase_id]
        observed = (
            self.method_route_id,
            self.calculator_id,
            self.result_shape,
            self.efficacy_estimand_id_template,
            self.efficacy_effect_scale_id,
            self.efficacy_orientation_id,
        )
        if observed != expected:
            raise ValueError(f"Analysis method route semantics drift for {self.phase_id}.")
        if self.safety_component_ids != ("serious_ae", "discontinuation"):
            raise ValueError("Analysis method route requires both declared safety components.")
        if self.safety_reported_measure_ids != (
            "treatment_absolute_risk",
            "control_absolute_risk",
            "risk_difference_treatment_minus_control",
        ):
            raise ValueError("Safety method route requires treated risk, control risk, and excess risk.")
        return self


class TrialDevPhaseAnalysisMethodCatalogV1(BaseModel):
    """Participant-visible randomized method catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_phase_analysis_method_catalog_v1"]
    version: Literal["v1"]
    scenario_id: str = Field(..., min_length=1)
    confidence_level: float = Field(..., gt=0.5, lt=1.0)
    methods: tuple[TrialDevPhaseAnalysisMethodRouteV1, ...] = Field(..., min_length=3, max_length=3)
    checksum: str = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_catalog(self) -> TrialDevPhaseAnalysisMethodCatalogV1:
        """Require complete unique phase coverage and checksum integrity."""

        if {method.phase_id for method in self.methods} != {"phase1", "phase2", "phase3"}:
            raise ValueError("Analysis method catalog must cover every randomized phase once.")
        ids = tuple(method.method_route_id for method in self.methods)
        if len(set(ids)) != len(ids):
            raise ValueError("Analysis method route IDs must be unique.")
        if any(abs(method.confidence_level - self.confidence_level) > 1e-12 for method in self.methods):
            raise ValueError("Analysis method confidence differs from the catalog.")
        _require_checksum(self)
        return self

    def method_for_phase(self, phase_id: str) -> TrialDevPhaseAnalysisMethodRouteV1:
        """Return the unique method route for a randomized phase."""

        matches = tuple(method for method in self.methods if method.phase_id == phase_id)
        if len(matches) != 1:
            raise ValueError(f"Analysis catalog has no unique method for {phase_id!r}.")
        return matches[0]


class TrialDevPhaseDesignCellV1(BaseModel):
    """One randomized-phase binary adequacy policy, not an optimal design claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    design_cell_id: str = Field(..., min_length=1)
    phase_id: RandomizedPhaseIdV1
    calculator_id: Literal["prospective_fixed_final_operating_characteristics_v1"]
    primary_endpoint_id: str | None = Field(default=None, min_length=1)
    planning_alternative_benefit: float | None = Field(default=None, gt=0.0, lt=1.0)
    target_power: float | None = Field(default=None, gt=0.5, lt=1.0)
    supported_interim_policy: Literal["fixed_final"]
    confidence_level: float = Field(..., gt=0.5, lt=1.0)
    evaluation_horizon_days: int = Field(..., ge=1)
    serious_ae_unacceptable_absolute_risk: float = Field(..., gt=0.0, lt=1.0)
    serious_ae_unacceptable_excess_risk: float = Field(..., gt=0.0, lt=1.0)
    planning_safety_control_risk: float = Field(..., ge=0.0, lt=1.0)
    planning_safety_absolute_treatment_risk: float = Field(..., gt=0.0, lt=1.0)
    planning_safety_excess_risk: float = Field(..., gt=0.0, lt=1.0)
    planning_safety_excess_treatment_risk: float = Field(..., gt=0.0, lt=1.0)
    target_safety_decision_power: float = Field(..., gt=0.5, lt=1.0)
    safety_power_adequacy_rule: Literal["minimum_achieved_power_across_absolute_and_excess_hard_gates"]
    planning_safety_estimator_id: Literal["multinomial_propensity_weighted_aalen_johansen_any_serious_ae"]
    planning_safety_analysis_population: Literal["complete_on_declared_adjustment_covariates"]
    planning_safety_control_support_count: int = Field(..., ge=1)
    planning_safety_min_observed_propensity: float = Field(..., gt=0.0, le=1.0)
    planning_safety_max_inverse_propensity_weight: float = Field(..., ge=1.0)
    planning_safety_weighted_effective_sample_size: float = Field(..., gt=0.0)
    planning_information_estimator_id: Literal["one_minus_multinomial_propensity_weighted_aalen_johansen_ltfu_cif"]
    planning_information_fraction_by_drug_id: dict[str, float] = Field(..., min_length=1)
    planning_information_support_count_by_drug_id: dict[str, int] = Field(..., min_length=1)
    planning_information_weighted_effective_sample_size_by_drug_id: dict[str, float] = Field(..., min_length=1)
    planning_control_risk: float | None = Field(default=None, gt=0.0, lt=1.0)
    planning_treatment_risk: float | None = Field(default=None, gt=0.0, lt=1.0)
    planning_estimator_id: Literal["multinomial_propensity_weighted_aalen_johansen"] | None = None
    planning_analysis_population: Literal["complete_on_declared_adjustment_covariates"] | None = None
    planning_control_support_count: int | None = Field(default=None, ge=1)
    planning_min_observed_propensity: float | None = Field(default=None, gt=0.0, le=1.0)
    planning_max_inverse_propensity_weight: float | None = Field(default=None, ge=1.0)
    planning_weighted_effective_sample_size: float | None = Field(default=None, gt=0.0)
    rationale: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_phase_semantics(self) -> TrialDevPhaseDesignCellV1:
        """Reject phase-swapped or incomplete design cells."""

        expected_id = f"trialdev.{self.phase_id}.fixed_final_operating_characteristics.v1"
        if self.design_cell_id != expected_id:
            raise ValueError(f"Design cell identity drift for {self.phase_id}.")
        efficacy = (
            self.primary_endpoint_id,
            self.planning_alternative_benefit,
            self.target_power,
            self.planning_control_risk,
            self.planning_treatment_risk,
            self.planning_estimator_id,
            self.planning_analysis_population,
            self.planning_control_support_count,
            self.planning_min_observed_propensity,
            self.planning_max_inverse_propensity_weight,
            self.planning_weighted_effective_sample_size,
        )
        if self.phase_id == "phase1" and any(value is not None for value in efficacy):
            raise ValueError("Phase1 design cell cannot declare efficacy planning.")
        if self.phase_id != "phase1" and any(value is None for value in efficacy):
            raise ValueError(f"{self.phase_id} design cell requires complete efficacy planning.")
        drug_ids = set(self.planning_information_fraction_by_drug_id)
        if drug_ids != set(self.planning_information_support_count_by_drug_id):
            raise ValueError("Design planning support map has inconsistent drug coverage.")
        if drug_ids != set(self.planning_information_weighted_effective_sample_size_by_drug_id):
            raise ValueError("Design planning effective-sample-size map has inconsistent drug coverage.")
        if self.planning_safety_absolute_treatment_risk <= self.serious_ae_unacceptable_absolute_risk:
            raise ValueError("Authored absolute-risk planning alternative must exceed its charter hard limit.")
        if self.planning_safety_excess_risk <= self.serious_ae_unacceptable_excess_risk:
            raise ValueError("Authored excess-risk planning alternative must exceed its charter hard limit.")
        if not math.isclose(
            self.planning_safety_excess_treatment_risk,
            self.planning_safety_control_risk + self.planning_safety_excess_risk,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Derived treatment risk must equal public control risk plus the authored excess-risk alternative."
            )
        if self.planning_safety_excess_treatment_risk >= 1.0:
            raise ValueError("Public control risk plus the excess-risk alternative must remain below one.")
        return self


class TrialDevPhaseDesignPolicyV1(BaseModel):
    """Participant-visible prospective design policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_phase_design_policy_v1"]
    version: Literal["v1"]
    scenario_id: str = Field(..., min_length=1)
    decision_charter_checksum: str = Field(..., min_length=64, max_length=64)
    confidence_level: float = Field(..., gt=0.5, lt=1.0)
    efficacy_test: Literal["two_sided_normal_approximation_risk_difference"]
    safety_assurance: Literal["minimum_power_across_absolute_and_excess_serious_ae_hard_gates"]
    source_artifact_checksums: dict[str, str] = Field(..., min_length=1)
    phase_rules: tuple[TrialDevPhaseDesignCellV1, ...] = Field(..., min_length=3, max_length=3)
    checksum: str = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_policy(self) -> TrialDevPhaseDesignPolicyV1:
        """Require complete unique phase coverage and checksum integrity."""

        if {rule.phase_id for rule in self.phase_rules} != {"phase1", "phase2", "phase3"}:
            raise ValueError("Design policy must cover every randomized phase once.")
        ids = tuple(rule.design_cell_id for rule in self.phase_rules)
        if len(set(ids)) != len(ids):
            raise ValueError("Design cell IDs must be unique.")
        if any(abs(rule.confidence_level - self.confidence_level) > 1e-12 for rule in self.phase_rules):
            raise ValueError("Design-cell confidence differs from the policy.")
        if any(len(checksum) != 64 for checksum in self.source_artifact_checksums.values()):
            raise ValueError("Design source checksums must be SHA-256 values.")
        _require_checksum(self)
        return self

    def rule_for_phase(self, phase_id: str) -> TrialDevPhaseDesignCellV1:
        """Return the unique design cell for a randomized phase."""

        matches = tuple(rule for rule in self.phase_rules if rule.phase_id == phase_id)
        if len(matches) != 1:
            raise ValueError(f"Design policy has no unique cell for {phase_id!r}.")
        return matches[0]


def _require_checksum(
    model: (
        TrialDevPublicObjectiveCharterV1
        | TrialDevPublicObservationalMethodCatalogV1
        | TrialDevPhaseAnalysisMethodCatalogV1
        | TrialDevPhaseDesignPolicyV1
    ),
) -> None:
    omit_nulls = isinstance(
        model,
        TrialDevPublicObjectiveCharterV1 | TrialDevPublicObservationalMethodCatalogV1,
    )
    payload = model.model_dump(
        mode="json",
        exclude={"checksum"},
        exclude_none=omit_nulls,
    )
    if model.checksum != compute_sha256_hex(payload):
        raise ValueError(f"{model.__class__.__name__} checksum mismatch.")


__all__ = [
    "TrialDevPhaseAnalysisMethodCatalogV1",
    "TrialDevPhaseAnalysisMethodRouteV1",
    "TrialDevPhaseDesignCellV1",
    "TrialDevPhaseDesignPolicyV1",
    "TrialDevPublicAssignmentPrognosticFactorV1",
    "TrialDevPublicEntropyBalancedAnalysisSpecV1",
    "TrialDevPublicEfficacyEndpointV1",
    "TrialDevPublicObjectiveCharterV1",
    "TrialDevPublicObjectiveSpecV1",
    "TrialDevPublicObservationalAnalysisSpecV1",
    "TrialDevPublicObservationalMethodCatalogV1",
    "TrialDevPublicObservationalMethodSpecV1",
    "TrialDevPublicUtilityComponentV1",
    "TrialDevPublicUtilityEventV1",
]
