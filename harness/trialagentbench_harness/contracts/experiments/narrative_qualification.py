"""Contracts for prospective TrialEval narrative-normalizer qualification."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.core.trialeval_factors import TrialEvalDataPreparationV1
from trialagentbench_harness.contracts.experiments.procedure_assistance import TrialEvalAnalysisSpecificationV1
from trialagentbench_harness.contracts.experiments.trialeval_ablation import (
    TrialEvalAblationEndpointRowV1,
    TrialEvalNarrativeTranscriptionV1,
)
from trialagentbench_harness.contracts.experiments.trialeval_design import (
    TrialEvalNormalizerQualificationDesignV1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256


class _QualificationModelV1(BaseModel):
    """Strict immutable base for normalizer qualification artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialEvalNormalizerFrameUnitV1(_QualificationModelV1):
    """Outcome-blind narrative-report metadata eligible for sampling."""

    unit_id: str = Field(..., pattern=r"^[A-Za-z0-9._-]+$")
    run_identity_sha256: str = Field(..., min_length=64, max_length=64)
    assignment_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    base_trial_id: str = Field(..., min_length=1)
    report_sha256: str = Field(..., min_length=64, max_length=64)
    regime_cell_id: str = Field(..., min_length=1)
    design_tier: Literal["D1", "D2", "D3", "D4"]
    assumption_tier: Literal["A1", "A2", "A3", "A4"]
    context_configuration: Literal["C1", "C2", "C3", "C4", "C5"]
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    result_shape: Literal["scalar", "identified_interval", "vector", "test", "non_identification", "mixed"]
    model_id: str = Field(..., min_length=1)


class TrialEvalNormalizerFrameV1(_QualificationModelV1):
    """Checksummed outcome-blind report population for one frozen campaign."""

    schema_id: Literal["trialagentbench.trialeval_normalizer_frame/v1"] = (
        "trialagentbench.trialeval_normalizer_frame/v1"
    )
    evaluator_release_sha256: str = Field(..., min_length=64, max_length=64)
    participant_release_sha256: str = Field(..., min_length=64, max_length=64)
    schedule_sha256: str = Field(..., min_length=64, max_length=64)
    run_identity_sha256s: tuple[str, ...] = Field(..., min_length=1)
    units: tuple[TrialEvalNormalizerFrameUnitV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_frame(self) -> TrialEvalNormalizerFrameV1:
        """Require exact run coverage, canonical units, and checksum integrity."""

        if self.run_identity_sha256s != tuple(sorted(set(self.run_identity_sha256s))):
            raise ValueError("Normalizer frame run identities must be unique and canonically ordered.")
        ordered = tuple(sorted(self.units, key=lambda row: row.unit_id))
        unit_ids = tuple(row.unit_id for row in self.units)
        assignment_keys = tuple((row.run_identity_sha256, row.assignment_id) for row in self.units)
        if self.units != ordered or len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Normalizer frame units must be unique and canonically ordered.")
        if len(assignment_keys) != len(set(assignment_keys)):
            raise ValueError("Normalizer frame contains duplicate run/assignment units.")
        if {row.run_identity_sha256 for row in self.units} != set(self.run_identity_sha256s):
            raise ValueError("Normalizer frame units do not cover exactly the declared run identities.")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Normalizer frame checksum mismatch.")
        return self

    def with_checksum(self) -> TrialEvalNormalizerFrameV1:
        """Return this frame with its canonical checksum."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


class TrialEvalNormalizerSampleUnitV1(TrialEvalNormalizerFrameUnitV1):
    """Selected probability-sample unit and its design weight."""

    stratum_id: str = Field(..., min_length=1)
    frame_base_trial_count: int = Field(..., ge=1)
    sampled_base_trial_count: int = Field(..., ge=1)
    base_trial_candidate_report_count: int = Field(..., ge=1)
    base_trial_inclusion_probability: float = Field(..., gt=0.0, le=1.0)
    within_base_report_inclusion_probability: float = Field(..., gt=0.0, le=1.0)
    inclusion_probability: float = Field(..., gt=0.0, le=1.0)
    selected_without_normalizer_or_score_outcomes: Literal[True]

    @model_validator(mode="after")
    def validate_probability(self) -> TrialEvalNormalizerSampleUnitV1:
        """Require the declared equal-within-stratum inclusion probability."""

        if self.sampled_base_trial_count > self.frame_base_trial_count:
            raise ValueError("Normalizer sample stratum exceeds its frame stratum.")
        expected_base = self.sampled_base_trial_count / self.frame_base_trial_count
        expected_report = 1.0 / self.base_trial_candidate_report_count
        expected_overall = expected_base * expected_report
        for label, observed, expected in (
            ("base-trial", self.base_trial_inclusion_probability, expected_base),
            ("within-base report", self.within_base_report_inclusion_probability, expected_report),
            ("overall", self.inclusion_probability, expected_overall),
        ):
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"Normalizer sample {label} inclusion probability is incorrect.")
        return self


class TrialEvalNormalizerSampleV1(_QualificationModelV1):
    """Checksummed outcome-blind qualification sample selected from one frame."""

    schema_id: Literal["trialagentbench.trialeval_normalizer_sample/v1"] = (
        "trialagentbench.trialeval_normalizer_sample/v1"
    )
    experiment_design_checksum: str = Field(..., min_length=64, max_length=64)
    frame_checksum: str = Field(..., min_length=64, max_length=64)
    selection_method: Literal["stratified_base_trial_then_within_base_hash_rank_v1"]
    selection_seed: int = Field(..., ge=0)
    units: tuple[TrialEvalNormalizerSampleUnitV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_sample(self) -> TrialEvalNormalizerSampleV1:
        """Require unique canonical units and internally consistent stratum declarations."""

        ordered = tuple(sorted(self.units, key=lambda row: row.unit_id))
        unit_ids = tuple(row.unit_id for row in self.units)
        assignment_keys = tuple((row.model_id, row.assignment_id) for row in self.units)
        if self.units != ordered or len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Normalizer sample units must be unique and canonically ordered.")
        if len(assignment_keys) != len(set(assignment_keys)):
            raise ValueError("Normalizer sample contains duplicate model/assignment units.")
        base_trial_ids = tuple(row.base_trial_id for row in self.units)
        if len(base_trial_ids) != len(set(base_trial_ids)):
            raise ValueError("Normalizer sample may contain only one narrative report per base trial.")
        strata: dict[str, tuple[int, int, float]] = {}
        counts: dict[str, int] = {}
        for unit in self.units:
            declaration = (
                unit.frame_base_trial_count,
                unit.sampled_base_trial_count,
                unit.base_trial_inclusion_probability,
            )
            if strata.setdefault(unit.stratum_id, declaration) != declaration:
                raise ValueError(f"Normalizer sample stratum declaration drift: {unit.stratum_id!r}.")
            counts[unit.stratum_id] = counts.get(unit.stratum_id, 0) + 1
        if any(counts[stratum] != declaration[1] for stratum, declaration in strata.items()):
            raise ValueError("Normalizer sample does not contain its declared stratum allocation.")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Normalizer sample checksum mismatch.")
        return self

    def with_checksum(self) -> TrialEvalNormalizerSampleV1:
        """Return this sample with its canonical checksum."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


class TrialEvalNormalizerQualificationObservationV1(_QualificationModelV1):
    """Paired masked-human and repeated automated normalization evidence."""

    sample_unit: TrialEvalNormalizerSampleUnitV1
    masked_human_reference: TrialEvalNarrativeTranscriptionV1
    automated_repeats: tuple[TrialEvalNarrativeTranscriptionV1, ...] = Field(..., min_length=2)
    masked_human_endpoint: TrialEvalAblationEndpointRowV1
    automated_endpoint: TrialEvalAblationEndpointRowV1

    @model_validator(mode="after")
    def validate_pairing(self) -> TrialEvalNormalizerQualificationObservationV1:
        """Bind every normalization and score to the sampled report and assignment."""

        unit = self.sample_unit
        human = self.masked_human_reference
        if human.source != "manual_masked" or human.assignment_id != unit.assignment_id:
            raise ValueError("Normalizer qualification human reference is not bound to the sampled assignment.")
        if human.report_sha256 != unit.report_sha256:
            raise ValueError("Normalizer qualification human reference report hash drift.")
        if human.submission is not None and human.submission.task_id != unit.task_id:
            raise ValueError("Normalizer qualification human submission task identity drift.")
        prompt_hashes: set[str | None] = set()
        schema_hashes: set[str | None] = set()
        source_identities: set[str] = set()
        for automated in self.automated_repeats:
            if automated.source != "automated_importer" or automated.assignment_id != unit.assignment_id:
                raise ValueError("Automated normalization repeat is not bound to the sampled assignment.")
            if automated.report_sha256 != unit.report_sha256:
                raise ValueError("Automated normalization repeat report hash drift.")
            if automated.submission is not None and automated.submission.task_id != unit.task_id:
                raise ValueError("Automated normalization repeat task identity drift.")
            prompt_hashes.add(automated.importer_prompt_sha256)
            schema_hashes.add(automated.importer_schema_sha256)
            source_identities.add(automated.source_identity)
        if len(prompt_hashes) != 1 or len(schema_hashes) != 1 or len(source_identities) != 1:
            raise ValueError("Automated normalization repeats must use one frozen importer identity.")
        human_endpoint = self.masked_human_endpoint
        automated_endpoint = self.automated_endpoint
        if (
            human_endpoint.assignment_id != unit.assignment_id
            or automated_endpoint.assignment_id != unit.assignment_id
        ):
            raise ValueError("Normalizer qualification endpoints are not bound to the sampled assignment.")
        if human_endpoint.normalization_source != "manual_masked":
            raise ValueError("Human qualification endpoint must use manual_masked normalization.")
        if automated_endpoint.normalization_source != "automated_importer":
            raise ValueError("Automated qualification endpoint must use automated_importer normalization.")
        for endpoint in (human_endpoint, automated_endpoint):
            if endpoint.task_id != unit.task_id:
                raise ValueError("Normalizer qualification endpoint task identity drift.")
            if endpoint.context_tier != unit.context_configuration:
                raise ValueError("Normalizer qualification endpoint context drift.")
            if endpoint.data_preparation != unit.data_preparation:
                raise ValueError("Normalizer qualification endpoint data-preparation drift.")
            if endpoint.analysis_specification != unit.analysis_specification:
                raise ValueError("Normalizer qualification endpoint analysis-specification drift.")
            if endpoint.model_id != unit.model_id:
                raise ValueError("Normalizer qualification endpoint model identity drift.")
        return self


class TrialEvalNormalizerQualificationObservationSetV1(_QualificationModelV1):
    """Checksummed paired human, automated, and scorer evidence for the frozen sample."""

    schema_id: Literal["trialagentbench.trialeval_normalizer_observations/v1"] = (
        "trialagentbench.trialeval_normalizer_observations/v1"
    )
    sample_checksum: str = Field(..., min_length=64, max_length=64)
    packet_set_checksum: str = Field(..., min_length=64, max_length=64)
    normalization_batch_checksum: str = Field(..., min_length=64, max_length=64)
    evaluator_release_sha256: str = Field(..., min_length=64, max_length=64)
    scoring_implementation_sha256: str = Field(..., min_length=64, max_length=64)
    observations: tuple[TrialEvalNormalizerQualificationObservationV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_observations(self) -> TrialEvalNormalizerQualificationObservationSetV1:
        """Require canonical unique sampled units and checksum integrity."""

        ordered = tuple(sorted(self.observations, key=lambda row: row.sample_unit.unit_id))
        unit_ids = tuple(row.sample_unit.unit_id for row in self.observations)
        if self.observations != ordered or len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Normalizer qualification observations must be unique and canonically ordered.")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Normalizer qualification observation-set checksum mismatch.")
        return self

    def with_checksum(self) -> TrialEvalNormalizerQualificationObservationSetV1:
        """Return this observation set with its canonical checksum."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


class TrialEvalNormalizerQualificationUnitResultV1(_QualificationModelV1):
    """Per-report normalization error and score-perturbation measurements."""

    unit_id: str = Field(..., pattern=r"^[A-Za-z0-9._-]+$")
    model_id: str
    regime_cell_id: str
    base_trial_id: str
    design_tier: Literal["D1", "D2", "D3", "D4"]
    assumption_tier: Literal["A1", "A2", "A3", "A4"]
    context_configuration: Literal["C1", "C2", "C3", "C4", "C5"]
    result_shape: str
    inclusion_probability: float = Field(..., gt=0.0, le=1.0)
    score_relevant_error: bool
    claim_true_positive: int = Field(..., ge=0)
    claim_false_positive: int = Field(..., ge=0)
    claim_false_negative: int = Field(..., ge=0)
    primary_role_false_positive: bool
    result_shape_agreement: bool | None
    result_unit_agreement: bool | None
    numeric_value_agreement: bool | None
    abstention_agreement: bool
    stable_across_repeats: bool
    usable_primary_disagreement: bool
    route_match_disagreement: bool
    numeric_availability_disagreement: bool
    primary_analysis_conformance_disagreement: bool
    planning_valid_disagreement: bool | None


class TrialEvalNormalizerQualificationMetricV1(_QualificationModelV1):
    """One inclusion-weighted normalizer error or agreement estimate."""

    metric: Literal[
        "score_relevant_error_rate",
        "claim_precision",
        "claim_recall",
        "primary_role_false_positive_rate",
        "result_shape_agreement_rate",
        "result_unit_agreement_rate",
        "numeric_value_agreement_rate",
        "abstention_agreement_rate",
        "instability_rate",
        "usable_primary_disagreement_rate",
        "route_match_disagreement_rate",
        "numeric_availability_disagreement_rate",
        "primary_analysis_conformance_disagreement_rate",
        "planning_valid_disagreement_rate",
    ]
    subgroup_dimension: Literal["overall", "model_id", "design_tier", "assumption_tier", "context", "result_shape"]
    subgroup_value: str
    sampled_units: int = Field(..., ge=1)
    weighted_numerator: float = Field(..., ge=0.0)
    weighted_denominator: float = Field(..., gt=0.0)
    estimate: float = Field(..., ge=0.0, le=1.0)
    uncertainty_method: Literal["stratified_cluster_bootstrap_with_weighted_hoeffding_envelope"]
    confidence_level: float = Field(..., gt=0.0, lt=1.0)
    confidence_lower: float = Field(..., ge=0.0, le=1.0)
    confidence_upper: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_interval(self) -> TrialEvalNormalizerQualificationMetricV1:
        """Require the point estimate to lie inside its uncertainty interval."""

        if not self.confidence_lower <= self.estimate <= self.confidence_upper:
            raise ValueError("Normalizer qualification estimate lies outside its confidence interval.")
        return self


class TrialEvalNormalizerQualificationReportV1(_QualificationModelV1):
    """Checksummed qualification evidence for one frozen narrative normalizer."""

    schema_id: Literal["trialagentbench.trialeval_normalizer_qualification/v1"] = (
        "trialagentbench.trialeval_normalizer_qualification/v1"
    )
    sample_checksum: str = Field(..., min_length=64, max_length=64)
    qualification_design: TrialEvalNormalizerQualificationDesignV1
    confidence_level: float = Field(..., gt=0.0, lt=1.0)
    bootstrap_replicates: int = Field(..., ge=1000)
    bootstrap_seed: int = Field(..., ge=0)
    exact_binomial_eligible: bool
    observed_error_count: int = Field(..., ge=0)
    qualified: bool
    unit_results: tuple[TrialEvalNormalizerQualificationUnitResultV1, ...] = Field(..., min_length=1)
    metrics: tuple[TrialEvalNormalizerQualificationMetricV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_design_binding(self) -> TrialEvalNormalizerQualificationReportV1:
        """Bind uncertainty and qualification decisions to the frozen design."""

        design = self.qualification_design
        if (
            self.confidence_level != design.secondary_confidence_level
            or self.bootstrap_replicates != design.secondary_bootstrap_replicates
            or self.bootstrap_seed != design.secondary_bootstrap_seed
        ):
            raise ValueError("Normalizer qualification uncertainty differs from the frozen design.")
        if len(self.unit_results) != design.retained_sample_size:
            raise ValueError("Normalizer qualification result denominator differs from the frozen design.")
        if self.observed_error_count != sum(row.score_relevant_error for row in self.unit_results):
            raise ValueError("Normalizer qualification error count differs from its unit results.")
        metric_keys = tuple((row.metric, row.subgroup_dimension, row.subgroup_value) for row in self.metrics)
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError("Normalizer qualification contains duplicate metric/subgroup rows.")
        expected_qualified = (
            self.exact_binomial_eligible
            and len(self.unit_results) == design.retained_sample_size
            and self.observed_error_count <= design.maximum_accepted_errors
        )
        if self.qualified != expected_qualified:
            raise ValueError("Normalizer qualification disposition differs from the frozen acceptance rule.")
        return self

    @model_validator(mode="after")
    def validate_report(self) -> TrialEvalNormalizerQualificationReportV1:
        """Require denominator-complete evidence and derive no permissive qualification state."""

        ordered_units = tuple(sorted(self.unit_results, key=lambda row: row.unit_id))
        if self.unit_results != ordered_units or len({row.unit_id for row in ordered_units}) != len(ordered_units):
            raise ValueError("Normalizer qualification unit results must be unique and ordered.")
        observed_errors = sum(row.score_relevant_error for row in ordered_units)
        if observed_errors != self.observed_error_count:
            raise ValueError("Normalizer qualification error count differs from unit results.")
        expected_qualified = (
            self.exact_binomial_eligible
            and len(ordered_units) >= self.qualification_design.retained_sample_size
            and observed_errors <= self.qualification_design.maximum_accepted_errors
        )
        if self.qualified != expected_qualified:
            raise ValueError("Normalizer qualification disposition differs from the prospective design.")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Normalizer qualification report checksum mismatch.")
        return self

    def with_checksum(self) -> TrialEvalNormalizerQualificationReportV1:
        """Return this report with its canonical checksum."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


__all__ = [
    "TrialEvalNormalizerFrameUnitV1",
    "TrialEvalNormalizerQualificationMetricV1",
    "TrialEvalNormalizerQualificationObservationV1",
    "TrialEvalNormalizerQualificationReportV1",
    "TrialEvalNormalizerQualificationUnitResultV1",
    "TrialEvalNormalizerSampleUnitV1",
    "TrialEvalNormalizerSampleV1",
]
