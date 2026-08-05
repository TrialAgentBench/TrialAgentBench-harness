"""Contracts for external fitting and held-out concordance."""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceRole(str, Enum):
    """Permitted role of an external source."""

    FIT = "fit"
    HELD_OUT = "held_out"
    SECONDARY_VALIDATION = "secondary_validation"


class LicenseStatus(str, Enum):
    """Redistribution status established for a source."""

    REDISTRIBUTABLE = "redistributable"
    ACQUISITION_ONLY = "acquisition_only"


class ExternalSourceV1(_Contract):
    """Immutable identity and local verification rule for one source."""

    source_id: str = Field(min_length=1)
    source_type: Literal["aact", "rct_bench"]
    canonical_url: str = Field(min_length=1)
    snapshot_identity: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: str = Field(min_length=1)
    license_status: LicenseStatus
    redistribution_rationale: str = Field(min_length=1)
    role: SourceRole


class ExternalSourceManifestV1(_Contract):
    """Frozen external-source portfolio."""

    schema_id: Literal["trialagentbench.external_sources/v1"] = (
        "trialagentbench.external_sources/v1"
    )
    sources: tuple[ExternalSourceV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_sources(self) -> ExternalSourceManifestV1:
        ids = tuple(row.source_id for row in self.sources)
        if len(ids) != len(set(ids)):
            raise ValueError("source_id values must be unique")
        return self


class ConstructKind(str, Enum):
    """Supported observable construct type."""

    ENROLLMENT = "enrollment"
    ARM_COUNT = "arm_count"
    BASELINE_COVARIATE_COUNT = "baseline_covariate_count"
    BASELINE_MISSING_FRACTION = "baseline_missing_fraction"
    PRIMARY_OUTCOME_MISSING_FRACTION = "primary_outcome_missing_fraction"
    EVENT_FRACTION = "event_fraction"
    FOLLOW_UP_TIME = "follow_up_time"
    AGE_MEAN = "age_mean"
    AGE_SD = "age_sd"
    BMI_MEAN = "bmi_mean"
    BMI_SD = "bmi_sd"


ConstructionParameterPath = Literal[
    "baseline.age.location_years",
    "baseline.age.scale_years",
    "baseline.bmi.location_kg_m2",
    "baseline.bmi.scale_kg_m2",
]

_CONSTRUCTION_PATH_BY_KIND: dict[ConstructKind, ConstructionParameterPath] = {
    ConstructKind.AGE_MEAN: "baseline.age.location_years",
    ConstructKind.AGE_SD: "baseline.age.scale_years",
    ConstructKind.BMI_MEAN: "baseline.bmi.location_kg_m2",
    ConstructKind.BMI_SD: "baseline.bmi.scale_kg_m2",
}


class ConstructDefinitionV1(_Contract):
    """Prospective semantic match between external and benchmark constructs."""

    construct_id: str = Field(min_length=1)
    kind: ConstructKind
    source_id: str = Field(min_length=1)
    population: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    transformation: Literal["identity", "log1p"]
    minimum_studies: int = Field(ge=2)
    minimum_synthetic_trials: int = Field(ge=2)
    equivalence_margin: float | None = Field(default=None, gt=0)
    margin_rationale: str | None = None
    compatibility_limits: str = Field(min_length=1)
    construction_parameter: ConstructionParameterPath | None = None

    @model_validator(mode="after")
    def _margin_has_rationale(self) -> ConstructDefinitionV1:
        if (self.equivalence_margin is None) != (self.margin_rationale is None):
            raise ValueError(
                "equivalence_margin and margin_rationale must be declared together"
            )
        expected_path = _CONSTRUCTION_PATH_BY_KIND.get(self.kind)
        if (
            self.construction_parameter is not None
            and self.construction_parameter != expected_path
        ):
            raise ValueError(
                "construction_parameter is incompatible with its observable construct kind"
            )
        return self


class ConstructMapV1(_Contract):
    """Frozen set of construct comparisons."""

    schema_id: Literal["trialagentbench.external_construct_map/v1"] = (
        "trialagentbench.external_construct_map/v1"
    )
    constructs: tuple[ConstructDefinitionV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_constructs(self) -> ConstructMapV1:
        ids = tuple(row.construct_id for row in self.constructs)
        if len(ids) != len(set(ids)):
            raise ValueError("construct_id values must be unique")
        return self


class StudySummaryV1(_Contract):
    """Observable study-level summaries extracted from one external trial."""

    study_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    enrollment: int = Field(ge=1)
    observation_count: int = Field(ge=1)
    arm_count: int = Field(ge=1)
    primary_outcome_type: str = Field(min_length=1)
    baseline_covariate_count: int = Field(ge=0)
    baseline_missing_fraction: float | None = Field(default=None, ge=0, le=1)
    primary_outcome_missing_fraction: float | None = Field(default=None, ge=0, le=1)
    event_fraction: float | None = Field(default=None, ge=0, le=1)
    follow_up_time_median: float | None = Field(default=None, ge=0)
    follow_up_time_unit: str | None = None
    age_mean: float | None = Field(default=None, ge=0)
    age_sd: float | None = Field(default=None, gt=0)
    bmi_mean: float | None = Field(default=None, gt=0)
    bmi_sd: float | None = Field(default=None, gt=0)
    observable_exclusions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _follow_up_unit(self) -> StudySummaryV1:
        if (self.follow_up_time_median is None) != (self.follow_up_time_unit is None):
            raise ValueError(
                "follow_up_time_median and follow_up_time_unit must be declared together"
            )
        return self


class AACTInclusionV1(_Contract):
    """Prospective AACT study-design inclusion contract."""

    overall_statuses: tuple[str, ...] = Field(min_length=1)
    phases: tuple[str, ...] = Field(min_length=1)
    allocations: tuple[str, ...] = Field(min_length=1)
    intervention_models: tuple[str, ...] = Field(min_length=1)
    minimum_enrollment: int = Field(ge=1)
    maximum_enrollment: int = Field(ge=1)
    minimum_arms: int = Field(ge=1)
    maximum_arms: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> AACTInclusionV1:
        if self.minimum_enrollment > self.maximum_enrollment:
            raise ValueError("AACT enrollment bounds must be ordered")
        if self.minimum_arms > self.maximum_arms:
            raise ValueError("AACT arm-count bounds must be ordered")
        return self


class StudyPartitionV1(_Contract):
    """Deterministic study-level calibration and held-out split."""

    schema_id: Literal["trialagentbench.external_partition/v1"] = (
        "trialagentbench.external_partition/v1"
    )
    seed: int = Field(ge=0)
    calibration_study_ids: tuple[str, ...] = Field(min_length=1)
    held_out_study_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _disjoint(self) -> StudyPartitionV1:
        calibration = set(self.calibration_study_ids)
        held_out = set(self.held_out_study_ids)
        overlap = sorted(calibration & held_out)
        if overlap:
            raise ValueError(f"study split overlaps: {overlap}")
        return self


class DistributionProfileV1(_Contract):
    """Calibration-study empirical profile for one construct."""

    construct_id: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    transformation: Literal["identity", "log1p"]
    n_studies: int = Field(ge=2)
    median: float
    q25: float
    q75: float
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def _ordered(self) -> DistributionProfileV1:
        if not self.minimum <= self.q25 <= self.median <= self.q75 <= self.maximum:
            raise ValueError("profile quantiles must be ordered")
        return self


class FittedConstructionParameterV1(_Contract):
    """One source-derived nuisance parameter selected for construction."""

    parameter_path: ConstructionParameterPath
    source_construct_id: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    selection_statistic: Literal["calibration_study_median"] = (
        "calibration_study_median"
    )
    calibration_study_values: tuple[CalibrationParameterObservationV1, ...] = Field(
        min_length=2
    )

    @model_validator(mode="after")
    def _unique_studies(self) -> FittedConstructionParameterV1:
        study_ids = tuple(row.study_id for row in self.calibration_study_values)
        if len(study_ids) != len(set(study_ids)):
            raise ValueError(
                "construction parameter observations must have unique studies"
            )
        return self


class CalibrationParameterObservationV1(_Contract):
    """One study-level nuisance observation retained for deterministic sampling."""

    study_id: str = Field(min_length=1)
    value: float


class SelectedObservableProfileV1(_Contract):
    """Externally fitted observable profile exported to construction."""

    schema_id: Literal["trialagentbench.selected_observable_profile/v1"] = (
        "trialagentbench.selected_observable_profile/v1"
    )
    profile_id: str = Field(min_length=1)
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    construct_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    partition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    distributions: tuple[DistributionProfileV1, ...] = Field(min_length=1)
    construction_parameters: tuple[FittedConstructionParameterV1, ...]
    forbidden_uses: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_construction_parameters(self) -> SelectedObservableProfileV1:
        paths = tuple(row.parameter_path for row in self.construction_parameters)
        sources = tuple(row.source_construct_id for row in self.construction_parameters)
        if len(paths) != len(set(paths)):
            raise ValueError("construction parameter paths must be unique")
        if len(sources) != len(set(sources)):
            raise ValueError("construction parameter source constructs must be unique")
        distribution_ids = {row.construct_id for row in self.distributions}
        missing = sorted(set(sources) - distribution_ids)
        if missing:
            raise ValueError(
                f"construction parameters lack fitted distributions: {missing}"
            )
        return self


class ConstructConcordanceV1(_Contract):
    """Held-out distance and study-clustered uncertainty for one construct."""

    construct_id: str = Field(min_length=1)
    n_calibration_studies: int = Field(ge=0)
    n_held_out_studies: int = Field(ge=0)
    status: Literal["supported", "unsupported"]
    wasserstein_distance: float | None = Field(default=None, ge=0)
    bootstrap_ci_low: float | None = Field(default=None, ge=0)
    bootstrap_ci_high: float | None = Field(default=None, ge=0)
    equivalence_margin: float | None = Field(default=None, gt=0)
    equivalent: bool | None = None
    interpretation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent(self) -> ConstructConcordanceV1:
        numeric = (
            self.wasserstein_distance,
            self.bootstrap_ci_low,
            self.bootstrap_ci_high,
        )
        if self.status == "supported":
            if self.n_calibration_studies < 2 or self.n_held_out_studies < 2:
                raise ValueError(
                    "supported concordance requires at least two studies in each split"
                )
            if any(value is None for value in numeric):
                raise ValueError("supported concordance requires distance and interval")
            if self.bootstrap_ci_low > self.bootstrap_ci_high:  # type: ignore[operator]
                raise ValueError("bootstrap interval must be ordered")
            expected = (
                None if self.equivalence_margin is None else bool(self.bootstrap_ci_high <= self.equivalence_margin)  # type: ignore[operator]
            )
            if self.equivalent != expected:
                raise ValueError("equivalent must follow the upper confidence bound")
        elif any(value is not None for value in (*numeric, self.equivalent)):
            raise ValueError(
                "unsupported concordance cannot report inferential results"
            )
        return self


class ExternalValidationReportV1(_Contract):
    """Construct-specific held-out validation report."""

    schema_id: Literal["trialagentbench.external_validation_report/v1"] = (
        "trialagentbench.external_validation_report/v1"
    )
    profile_id: str = Field(min_length=1)
    results: tuple[ConstructConcordanceV1, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)


class SyntheticConstructRole(str, Enum):
    """Scientific role of one synthetic-to-external comparison."""

    EXTERNALLY_FITTED = "externally_fitted"
    QUALIFICATION_CONTROL = "qualification_control"
    PROTOCOL_CONTROL = "protocol_control"
    DESCRIPTIVE_ONLY = "descriptive_only"


class SyntheticConstructConcordanceV1(_Contract):
    """Held-out external comparison for one public synthetic construct."""

    construct_id: str = Field(min_length=1)
    role: SyntheticConstructRole
    n_synthetic_trials: int = Field(ge=0)
    n_held_out_studies: int = Field(ge=0)
    status: Literal["supported", "unsupported"]
    wasserstein_distance: float | None = Field(default=None, ge=0)
    bootstrap_ci_low: float | None = Field(default=None, ge=0)
    bootstrap_ci_high: float | None = Field(default=None, ge=0)
    equivalence_margin: float | None = Field(default=None, gt=0)
    equivalent: bool | None = None
    calibration_reference_p95: float | None = Field(default=None, gt=0)
    calibration_reference_tail_probability: float | None = Field(
        default=None, ge=0, le=1
    )
    within_calibration_reference: bool | None = None
    interpretation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent(self) -> SyntheticConstructConcordanceV1:
        numeric = (
            self.wasserstein_distance,
            self.bootstrap_ci_low,
            self.bootstrap_ci_high,
        )
        if self.status == "supported":
            if any(value is None for value in numeric):
                raise ValueError(
                    "supported synthetic concordance requires distance and interval"
                )
            if self.bootstrap_ci_low > self.bootstrap_ci_high:  # type: ignore[operator]
                raise ValueError("synthetic concordance interval must be ordered")
            expected = (
                None if self.equivalence_margin is None else bool(self.bootstrap_ci_high <= self.equivalence_margin)  # type: ignore[operator]
            )
            if self.equivalent != expected:
                raise ValueError("equivalent must follow the upper confidence bound")
            expected_reference = (
                None
                if self.calibration_reference_p95 is None
                else bool(self.wasserstein_distance <= self.calibration_reference_p95)  # type: ignore[operator]
            )
            if self.within_calibration_reference != expected_reference:
                raise ValueError(
                    "within_calibration_reference must compare the observed distance with the null envelope"
                )
            if (self.calibration_reference_p95 is None) != (
                self.calibration_reference_tail_probability is None
            ):
                raise ValueError(
                    "calibration reference envelope and tail probability must be reported together"
                )
        elif any(
            value is not None
            for value in (
                *numeric,
                self.equivalent,
                self.calibration_reference_p95,
                self.calibration_reference_tail_probability,
                self.within_calibration_reference,
            )
        ):
            raise ValueError(
                "unsupported synthetic concordance cannot report inferential results"
            )
        return self


class SyntheticConcordanceReportV1(_Contract):
    """Independent comparison of public synthetic trials with held-out studies."""

    schema_id: Literal["trialagentbench.synthetic_concordance/v1"] = (
        "trialagentbench.synthetic_concordance/v1"
    )
    reference_profile_id: str = Field(min_length=1)
    reference_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    applied_profile_id: str | None = Field(default=None, min_length=1)
    applied_profile_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    participant_release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_task_count: int = Field(ge=1)
    independent_synthetic_trial_count: int = Field(ge=1)
    synthetic_trial_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    results: tuple[SyntheticConstructConcordanceV1, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _paired_applied_profile(self) -> SyntheticConcordanceReportV1:
        if (self.applied_profile_id is None) != (self.applied_profile_sha256 is None):
            raise ValueError(
                "applied profile identity and checksum must be supplied together"
            )
        return self


class SyntheticConcordanceDifferenceV1(_Contract):
    """Matched pre-fit versus selected-profile distance for one construct."""

    construct_id: str = Field(min_length=1)
    role: SyntheticConstructRole
    status: Literal["supported", "unsupported"]
    prefit: SyntheticConstructConcordanceV1
    selected: SyntheticConstructConcordanceV1
    selected_minus_prefit_distance: float | None = None

    @model_validator(mode="after")
    def _coherent(self) -> SyntheticConcordanceDifferenceV1:
        for phase, row in (("pre-fit", self.prefit), ("selected", self.selected)):
            if (
                row.construct_id != self.construct_id
                or row.role != self.role
                or row.status != self.status
            ):
                raise ValueError(
                    f"{phase} construct record does not match paired metadata"
                )
        if self.status == "unsupported":
            if self.selected_minus_prefit_distance is not None:
                raise ValueError(
                    "unsupported comparison cannot report a distance change"
                )
            return self
        if self.selected_minus_prefit_distance is None:
            raise ValueError("supported comparison requires a distance change")
        assert self.prefit.wasserstein_distance is not None
        assert self.selected.wasserstein_distance is not None
        assert self.selected_minus_prefit_distance is not None
        expected = self.selected.wasserstein_distance - self.prefit.wasserstein_distance
        if not math.isclose(
            self.selected_minus_prefit_distance,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "selected_minus_prefit_distance must equal selected minus pre-fit"
            )
        return self


class PairedSyntheticConcordanceReportV1(_Contract):
    """Matched comparison of pre-fit and selected-profile public releases."""

    schema_id: Literal["trialagentbench.paired_synthetic_concordance/v1"] = (
        "trialagentbench.paired_synthetic_concordance/v1"
    )
    reference_profile_id: str = Field(min_length=1)
    reference_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    synthetic_trial_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_task_count: int = Field(ge=1)
    independent_synthetic_trial_count: int = Field(ge=1)
    prefit_participant_release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_participant_release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_applied_profile_id: str = Field(min_length=1)
    selected_applied_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    results: tuple[SyntheticConcordanceDifferenceV1, ...] = Field(min_length=1)
    interpretation: str = Field(min_length=1)


class ExternalValidationDesignV1(_Contract):
    """Frozen execution design for external fitting and validation."""

    schema_id: Literal["trialagentbench.external_validation_design/v1"] = (
        "trialagentbench.external_validation_design/v1"
    )
    profile_id: str = Field(min_length=1)
    split_seed: int = Field(ge=0)
    held_out_fraction: float = Field(gt=0, lt=1)
    bootstrap_seed: int = Field(ge=0)
    bootstrap_replicates: int = Field(ge=200)
    aact_inclusion: AACTInclusionV1
