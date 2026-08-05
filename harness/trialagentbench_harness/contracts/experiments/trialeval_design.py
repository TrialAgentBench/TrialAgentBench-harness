"""Public contracts for the prospective TrialEval experiment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.io.checksums import canonical_payload_sha256

TrialEvalPrimaryMetricV1 = Literal[
    "usable_primary",
    "route_match",
    "primary_analysis_conforms",
    "planning_usable_with_primary",
    "planning_achieved_power",
    "planning_power_shortfall",
    "planning_underpowered",
]


class _ProtocolModelV1(BaseModel):
    """Strict immutable base for public experiment records."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TrialEvalEstimableContrastV1(_ProtocolModelV1):
    """One prespecified contrast with its dependence and denominator semantics."""

    contrast_id: str = Field(..., pattern=r"^[A-Za-z0-9_.:-]+$")
    contrast_kind: Literal[
        "context",
        "procedure_assistance",
        "response_elicitation",
        "normalization_measurement",
        "operational_pipeline",
        "factor_interaction",
        "prompt_condition",
    ]
    minuend: str = Field(..., min_length=1)
    subtrahend: str = Field(..., min_length=1)
    changed_factors: tuple[str, ...] = Field(..., min_length=1)
    held_fixed: tuple[str, ...] = Field(..., min_length=1)
    pairing_fields: tuple[str, ...] = Field(..., min_length=1)
    independent_unit: Literal["base_trial", "narrative_report"]
    denominator_policy: Literal[
        "all_scheduled_pairs_noncompletion_is_failure",
        "all_sampled_reports",
    ]
    effect_measure: Literal[
        "mean_paired_difference",
        "difference_in_paired_differences",
        "field_disagreement_probability",
    ]
    inferential_status: Literal["interval_estimate", "qualification", "descriptive"]
    metrics: tuple[TrialEvalPrimaryMetricV1, ...] = ()
    interpretation: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_estimand(self) -> TrialEvalEstimableContrastV1:
        for label, values in (
            ("changed_factors", self.changed_factors),
            ("held_fixed", self.held_fixed),
            ("pairing_fields", self.pairing_fields),
            ("metrics", self.metrics),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must not contain duplicate values.")
        if self.contrast_kind == "normalization_measurement":
            expected = (
                self.independent_unit == "narrative_report"
                and self.denominator_policy == "all_sampled_reports"
                and self.effect_measure == "field_disagreement_probability"
                and self.inferential_status == "qualification"
                and not self.metrics
            )
            if not expected:
                raise ValueError("Normalization measurement has invalid unit, denominator, effect, or metrics.")
        elif (
            self.independent_unit != "base_trial"
            or self.denominator_policy != "all_scheduled_pairs_noncompletion_is_failure"
            or not self.metrics
        ):
            raise ValueError("Agent-performance contrasts require base-trial pairing and bounded metrics.")
        return self


class TrialEvalPrecisionDesignV1(_ProtocolModelV1):
    """Prospective precision target for bounded paired endpoints."""

    method: Literal["worst_case_bounded_mean_normal_planning"]
    confidence_level: float = Field(..., gt=0.0, lt=1.0)
    endpoint_lower_bound: float
    endpoint_upper_bound: float
    paired_difference_lower_bound: float
    paired_difference_upper_bound: float
    target_half_width: float = Field(..., gt=0.0, lt=1.0)
    minimum_independent_base_trials: int = Field(..., ge=2)
    retained_independent_base_trials: int = Field(..., ge=2)
    regime_cell_count: int = Field(..., ge=1)
    base_trial_replicates_per_regime_cell: int = Field(..., ge=1)
    achieved_worst_case_half_width: float = Field(..., gt=0.0)
    rationale: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_precision(self) -> TrialEvalPrecisionDesignV1:
        if self.endpoint_lower_bound >= self.endpoint_upper_bound:
            raise ValueError("Endpoint bounds must be ordered.")
        if (self.paired_difference_lower_bound, self.paired_difference_upper_bound) != (-1.0, 1.0):
            raise ValueError("Paired differences of bounded endpoints must retain their exact [-1, 1] bounds.")
        expected = self.regime_cell_count * self.base_trial_replicates_per_regime_cell
        if self.retained_independent_base_trials != expected:
            raise ValueError("Retained base-trial count does not match regime-cell replication.")
        if self.retained_independent_base_trials < self.minimum_independent_base_trials:
            raise ValueError("Retained matrix does not meet the precision design.")
        if self.achieved_worst_case_half_width > self.target_half_width:
            raise ValueError("Retained matrix exceeds the target worst-case half-width.")
        return self


class TrialEvalNormalizerQualificationDesignV1(_ProtocolModelV1):
    """Exact-binomial qualification design for narrative normalization."""

    independent_unit: Literal["one_narrative_report_per_base_trial"]
    error_event: Literal["any_score_relevant_field_disagrees_with_masked_human_reference"]
    acceptable_error_probability: float = Field(..., ge=0.0, lt=1.0)
    unacceptable_error_probability: float = Field(..., gt=0.0, le=1.0)
    type_i_error: float = Field(..., gt=0.0, lt=1.0)
    type_ii_error: float = Field(..., gt=0.0, lt=1.0)
    exact_minimum_sample_size: int = Field(..., ge=1)
    inclusion_strata: tuple[Literal["evaluation_series_id", "assumption_tier", "context_configuration"], ...]
    minimum_reports_per_stratum: int = Field(..., ge=1)
    retained_sample_size: int = Field(..., ge=1)
    maximum_accepted_errors: int = Field(..., ge=0)
    realized_type_i_error: float = Field(..., ge=0.0, le=1.0)
    realized_power: float = Field(..., ge=0.0, le=1.0)
    secondary_uncertainty_method: Literal["stratified_cluster_bootstrap_with_weighted_hoeffding_envelope"]
    secondary_confidence_level: float = Field(..., gt=0.0, lt=1.0)
    secondary_bootstrap_replicates: int = Field(..., ge=1000)
    secondary_bootstrap_seed: int = Field(..., ge=0)
    secondary_uncertainty_rationale: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_qualification(self) -> TrialEvalNormalizerQualificationDesignV1:
        if self.acceptable_error_probability >= self.unacceptable_error_probability:
            raise ValueError("Acceptable normalizer error must be below the unacceptable rate.")
        if self.retained_sample_size < self.exact_minimum_sample_size:
            raise ValueError("Retained normalizer sample is below the exact minimum.")
        if self.maximum_accepted_errors >= self.retained_sample_size:
            raise ValueError("Normalizer acceptance threshold must be below the sample size.")
        if self.realized_type_i_error > self.type_i_error or self.realized_power < 1.0 - self.type_ii_error:
            raise ValueError("Normalizer qualification does not meet declared error control.")
        return self


class TrialEvalComputeEnvelopeV1(_ProtocolModelV1):
    """Exact run counts implied by the public experiment protocol."""

    participant_context_count: int = Field(..., ge=1)
    participant_item_count: int = Field(..., ge=1)
    primary_assignments_per_model: int = Field(..., ge=1)
    factorial_sampling_unit: Literal["one_context_view_per_base_trial"]
    factorial_selection_method: Literal["stratified_regime_cell_omission_v1"]
    factorial_item_count: int = Field(..., ge=1)
    factorial_context_allocation: tuple[int, int, int, int, int]
    factorial_cells_per_item: int = Field(..., ge=1)
    decoding_replicates: int = Field(..., ge=2)
    factorial_assignments_per_model: int = Field(..., ge=1)

    @model_validator(mode="after")
    def _validate_counts(self) -> TrialEvalComputeEnvelopeV1:
        if self.primary_assignments_per_model != self.participant_item_count:
            raise ValueError("Primary assignment count differs from participant item count.")
        if any(count < 1 for count in self.factorial_context_allocation):
            raise ValueError("Factorial context allocation must retain every context stratum.")
        if sum(self.factorial_context_allocation) != self.factorial_item_count:
            raise ValueError("Factorial context allocation differs from the factorial item count.")
        expected = self.factorial_item_count * self.factorial_cells_per_item * self.decoding_replicates
        if self.factorial_assignments_per_model != expected:
            raise ValueError("Factorial assignment count is internally inconsistent.")
        return self


class TrialEvalExperimentProtocolV1(_ProtocolModelV1):
    """Checksummed public protocol for TrialEval execution and analysis."""

    schema_id: Literal["trialagentbench.trialeval_experiment_protocol/v1"]
    design_id: str = Field(..., min_length=1)
    frozen_before_model_outputs: Literal[True]
    contrasts: tuple[TrialEvalEstimableContrastV1, ...] = Field(..., min_length=1)
    precision: TrialEvalPrecisionDesignV1
    normalizer_qualification: TrialEvalNormalizerQualificationDesignV1
    compute_envelope: TrialEvalComputeEnvelopeV1
    analysis_constraints: tuple[str, ...] = Field(..., min_length=1)
    checksum: str = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_protocol(self) -> TrialEvalExperimentProtocolV1:
        contrast_ids = [row.contrast_id for row in self.contrasts]
        if len(contrast_ids) != len(set(contrast_ids)):
            raise ValueError("Experiment contrast IDs must be unique.")
        required_context = {"C1-C2", "C3-C4", "C3-C1", "C4-C2", "C5-C4"}
        observed_context = {row.contrast_id for row in self.contrasts if row.contrast_kind == "context"}
        if observed_context != required_context:
            raise ValueError("Experiment protocol requires exactly the five prespecified context contrasts.")
        precision = self.precision
        compute = self.compute_envelope
        expected_items = precision.retained_independent_base_trials * compute.participant_context_count
        if compute.participant_item_count != expected_items:
            raise ValueError("Participant item count differs from independent trials times context views.")
        if compute.factorial_item_count != precision.retained_independent_base_trials:
            raise ValueError("Factorial sampling must retain one view per independent base trial.")
        if precision.base_trial_replicates_per_regime_cell != compute.participant_context_count - 1:
            raise ValueError(
                "Stratified factorial sampling requires one fewer replicate per regime cell than context tiers."
            )
        if precision.regime_cell_count % compute.participant_context_count:
            raise ValueError("Stratified factorial sampling requires equal context omissions across regime cells.")
        if len(set(compute.factorial_context_allocation)) != 1:
            raise ValueError("Stratified factorial sampling requires equal overall context allocation.")
        payload = self.model_dump(mode="json", exclude_none=True)
        checksum = payload.pop("checksum")
        if checksum != canonical_payload_sha256(payload):
            raise ValueError("TrialEval experiment-protocol checksum does not match its payload.")
        return self


__all__ = [
    "TrialEvalComputeEnvelopeV1",
    "TrialEvalEstimableContrastV1",
    "TrialEvalExperimentProtocolV1",
    "TrialEvalNormalizerQualificationDesignV1",
    "TrialEvalPrecisionDesignV1",
]
