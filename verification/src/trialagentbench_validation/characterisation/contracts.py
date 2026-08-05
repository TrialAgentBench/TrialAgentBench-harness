"""Typed inputs and results for clinical-trial characterisation."""

from __future__ import annotations

import math
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, Tag, model_validator

DesignFamily = Literal[
    "individual_randomized",
    "pragmatic_randomized",
    "covariate_subdesign",
    "ascertainment_subdesign",
    "cluster_parallel",
    "stepped_wedge",
    "group_sequential",
]
MissingnessDisposition = Literal["complete_case", "explicit_level", "not_applicable"]
EvidenceLevel = Literal["participant_distribution", "trial", "programme", "portfolio"]
AssumptionSeriesId = Literal[
    "TE-S01",
    "TE-S02",
    "TE-S03",
    "TE-S04",
    "TE-S05",
    "TE-S06",
    "TE-S07",
    "TE-S08",
    "TE-S09",
]
AssumptionTier = Literal["A1", "A2", "A3", "A4"]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContinuousVariableSpec(_Contract):
    """One continuous baseline variable."""

    variable_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    column: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    role: Literal["baseline", "observation"] = "baseline"
    missingness: Literal["complete_case"] = "complete_case"


class CategoricalVariableSpec(_Contract):
    """One categorical baseline variable."""

    variable_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    column: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    role: Literal["baseline", "observation"] = "baseline"
    missingness: Literal["complete_case", "explicit_level"] = "complete_case"
    categories: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _categories_are_unique(self) -> CategoricalVariableSpec:
        if self.categories is not None:
            if not self.categories:
                raise ValueError("categorical variable categories must not be empty")
            if len(self.categories) != len(set(self.categories)):
                raise ValueError("categorical variable categories must be unique")
        return self


class DependenceSpec(_Contract):
    """One pairwise rank-dependence estimate."""

    dependence_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    left_column: str = Field(min_length=1)
    right_column: str = Field(min_length=1)
    stratify_by_arm: bool = True
    missingness: Literal["complete_case"] = "complete_case"

    @model_validator(mode="after")
    def _different_columns(self) -> DependenceSpec:
        if self.left_column == self.right_column:
            raise ValueError("dependence columns must be different")
        return self


class _OutcomeSpec(_Contract):
    outcome_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    table: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    participant_id_column: str = Field(min_length=1)


class BinaryOutcomeSpec(_OutcomeSpec):
    """Binary outcome observed once per participant."""

    kind: Literal["binary"] = "binary"
    value_column: str = Field(min_length=1)
    event_value: int | str | bool
    unit: str = "proportion"
    missingness: Literal["complete_case"] = "complete_case"


class ContinuousOutcomeSpec(_OutcomeSpec):
    """Continuous outcome observed once per participant."""

    kind: Literal["continuous"] = "continuous"
    value_column: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    missingness: Literal["complete_case"] = "complete_case"


class OrdinalOutcomeSpec(_OutcomeSpec):
    """Ordered categorical outcome observed once per participant."""

    kind: Literal["ordinal"] = "ordinal"
    value_column: str = Field(min_length=1)
    categories: tuple[str, ...] = Field(min_length=2)
    unit: str = "proportion"
    missingness: Literal["complete_case"] = "complete_case"

    @model_validator(mode="after")
    def _ordered_categories_are_unique(self) -> OrdinalOutcomeSpec:
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("ordinal outcome categories must be unique")
        return self


class SurvivalOutcomeSpec(_OutcomeSpec):
    """Right-censored time-to-event outcome."""

    kind: Literal["survival"] = "survival"
    duration_column: str = Field(min_length=1)
    event_column: str = Field(min_length=1)
    horizons: tuple[float, ...] = Field(min_length=1)
    unit: str = Field(min_length=1)
    missingness: Literal["complete_case"] = "complete_case"

    @model_validator(mode="after")
    def _valid_horizons(self) -> SurvivalOutcomeSpec:
        _validate_horizons(self.horizons)
        return self


class LongitudinalOutcomeSpec(_OutcomeSpec):
    """Continuous repeated outcome at declared visits."""

    kind: Literal["longitudinal"] = "longitudinal"
    time_column: str = Field(min_length=1)
    value_column: str = Field(min_length=1)
    scheduled_times: tuple[float, ...] = Field(min_length=1)
    time_unit: str = Field(min_length=1)
    value_unit: str = Field(min_length=1)
    missingness: Literal["complete_case"] = "complete_case"

    @model_validator(mode="after")
    def _valid_times(self) -> LongitudinalOutcomeSpec:
        _validate_horizons(self.scheduled_times)
        return self


class RecurrentEventOutcomeSpec(_OutcomeSpec):
    """Recurrent-event process observed through declared horizons."""

    kind: Literal["recurrent_event"] = "recurrent_event"
    event_time_column: str = Field(min_length=1)
    horizons: tuple[float, ...] = Field(min_length=1)
    unit: str = Field(min_length=1)
    missingness: Literal["not_applicable"] = "not_applicable"

    @model_validator(mode="after")
    def _valid_horizons(self) -> RecurrentEventOutcomeSpec:
        _validate_horizons(self.horizons)
        return self


class CompetingRiskOutcomeSpec(_OutcomeSpec):
    """First-event outcome with a primary and competing cause."""

    kind: Literal["competing_risk"] = "competing_risk"
    duration_column: str = Field(min_length=1)
    event_type_column: str = Field(min_length=1)
    primary_event_code: int = Field(ge=1)
    competing_event_codes: tuple[int, ...] = Field(min_length=1)
    horizons: tuple[float, ...] = Field(min_length=1)
    unit: str = Field(min_length=1)
    missingness: Literal["complete_case"] = "complete_case"

    @model_validator(mode="after")
    def _valid_event_codes_and_horizons(self) -> CompetingRiskOutcomeSpec:
        if self.primary_event_code in self.competing_event_codes:
            raise ValueError("primary and competing event codes must be different")
        if len(self.competing_event_codes) != len(set(self.competing_event_codes)):
            raise ValueError("competing event codes must be unique")
        _validate_horizons(self.horizons)
        return self


OutcomeSpec = Annotated[
    Annotated[BinaryOutcomeSpec, Tag("binary")]
    | Annotated[ContinuousOutcomeSpec, Tag("continuous")]
    | Annotated[OrdinalOutcomeSpec, Tag("ordinal")]
    | Annotated[SurvivalOutcomeSpec, Tag("survival")]
    | Annotated[LongitudinalOutcomeSpec, Tag("longitudinal")]
    | Annotated[RecurrentEventOutcomeSpec, Tag("recurrent_event")]
    | Annotated[CompetingRiskOutcomeSpec, Tag("competing_risk")],
    Field(discriminator="kind"),
]


class TrialCharacterisationSpec(_Contract):
    """Complete characterisation request for one clinical trial."""

    schema_id: Literal["trialagentbench.trial_characterisation_spec/v1"] = (
        "trialagentbench.trial_characterisation_spec/v1"
    )
    trial_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    programme_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    design_profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    design_family: DesignFamily
    participant_id_column: str = Field(min_length=1)
    arm_column: str = Field(min_length=1)
    cluster_id_column: str | None = Field(default=None, min_length=1)
    continuous_variables: tuple[ContinuousVariableSpec, ...] = ()
    categorical_variables: tuple[CategoricalVariableSpec, ...] = ()
    dependence: tuple[DependenceSpec, ...] = ()
    outcomes: tuple[OutcomeSpec, ...] = Field(min_length=1)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    bootstrap_replicates: int = Field(default=2_000, ge=200)
    seed: int = Field(default=20260801, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def _identifiers_are_unique(self) -> TrialCharacterisationSpec:
        variable_ids = tuple(
            row.variable_id for row in self.continuous_variables
        ) + tuple(row.variable_id for row in self.categorical_variables)
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("baseline variable IDs must be unique")
        dependence_ids = tuple(row.dependence_id for row in self.dependence)
        if len(dependence_ids) != len(set(dependence_ids)):
            raise ValueError("dependence IDs must be unique")
        outcome_ids = tuple(row.outcome_id for row in self.outcomes)
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("outcome IDs must be unique")
        requires_cluster = self.design_family in {"cluster_parallel", "stepped_wedge"}
        if requires_cluster and self.cluster_id_column is None:
            raise ValueError(
                f"{self.design_family} characterisation requires cluster_id_column"
            )
        return self


class TrialData(BaseModel):
    """Participant and observation tables supplied to characterisation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    participants: pd.DataFrame
    observation_tables: dict[str, pd.DataFrame] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _participants_are_present(self) -> TrialData:
        if self.participants.empty:
            raise ValueError("participant table must not be empty")
        invalid = sorted(
            name
            for name, frame in self.observation_tables.items()
            if not name
            or not name.replace("_", "").isalnum()
            or not name[0].isalpha()
            or frame.empty
        )
        if invalid:
            raise ValueError(
                f"observation tables require safe names and non-empty data: {invalid!r}"
            )
        return self


class TidyEstimate(_Contract):
    """One fully described estimate emitted by characterisation."""

    schema_id: Literal["trialagentbench.tidy_estimate/v1"] = (
        "trialagentbench.tidy_estimate/v1"
    )
    trial_id: str = Field(min_length=1)
    programme_id: str = Field(min_length=1)
    evidence_level: EvidenceLevel
    property_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    group: str = Field(min_length=1)
    time: float | None = None
    estimate: float
    interval_low: float | None = None
    interval_high: float | None = None
    unit: str = Field(min_length=1)
    independent_unit: str = Field(min_length=1)
    estimator: str = Field(min_length=1)
    uncertainty_method: str = Field(min_length=1)
    denominator: int = Field(ge=1)
    observed: int = Field(ge=0)
    missing: int = Field(ge=0)
    missingness_disposition: MissingnessDisposition

    @model_validator(mode="after")
    def _valid_numerics(self) -> TidyEstimate:
        values = (self.estimate, self.interval_low, self.interval_high, self.time)
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("characterisation estimates and times must be finite")
        if (self.interval_low is None) != (self.interval_high is None):
            raise ValueError("characterisation intervals require both endpoints")
        if self.interval_low is not None and self.interval_high is not None:
            if not self.interval_low <= self.estimate <= self.interval_high:
                raise ValueError(
                    "characterisation estimate must lie within its interval"
                )
        if self.observed + self.missing != self.denominator:
            raise ValueError("observed and missing counts must equal the denominator")
        if (
            self.uncertainty_method == "none_descriptive"
            and self.interval_low is not None
        ):
            raise ValueError("descriptive estimates cannot carry inferential intervals")
        return self


class TrialCharacterisation(_Contract):
    """Characterisation results for one trial."""

    schema_id: Literal["trialagentbench.trial_characterisation/v1"] = (
        "trialagentbench.trial_characterisation/v1"
    )
    trial_id: str
    programme_id: str
    design_profile_id: str
    design_family: DesignFamily
    estimates: tuple[TidyEstimate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent_and_unique(self) -> TrialCharacterisation:
        if any(
            row.trial_id != self.trial_id or row.programme_id != self.programme_id
            for row in self.estimates
        ):
            raise ValueError(
                "trial estimates must identify their parent trial and programme"
            )
        keys = tuple((row.property_id, row.group, row.time) for row in self.estimates)
        if len(keys) != len(set(keys)):
            raise ValueError("trial characterisation estimate keys must be unique")
        return self


class CharacterisationCollection(_Contract):
    """Trial, programme, and portfolio characterisation results."""

    schema_id: Literal["trialagentbench.characterisation_collection/v1"] = (
        "trialagentbench.characterisation_collection/v1"
    )
    trials: tuple[TrialCharacterisation, ...] = Field(min_length=1)
    programme_estimates: tuple[TidyEstimate, ...] = Field(min_length=1)
    portfolio_estimates: tuple[TidyEstimate, ...] = Field(min_length=1)


class TrialProfile(_Contract):
    """One independent released trial and its public characterisation."""

    task_id: str = Field(pattern=r"^TASK[0-9A-F]{32}$")
    matched_set_id: str = Field(min_length=1)
    independence_unit_id: str = Field(min_length=1)
    context_id: Literal["C1"]
    design_profile_id: Literal[
        "TE-DP01",
        "TE-DP02",
        "TE-DP03",
        "TE-DP04",
        "TE-DP05",
        "TE-DP06",
        "TE-DP07",
    ]
    design_tier: Literal["D1", "D2", "D3", "D4"]
    assumption_tier: Literal["A1", "A2", "A3", "A4"]
    design_subtype: Literal[
        "individual_randomized",
        "pragmatic",
        "covariate_structure",
        "endpoint_ascertainment",
        "cluster_parallel",
        "stepped_wedge",
        "group_sequential",
    ]
    participant_count: int = Field(ge=1)
    follow_up_horizon_days: float = Field(gt=0)
    primary_paramcd: str = Field(min_length=1)
    primary_estimand_id: str = Field(min_length=1)
    primary_effect_scale: str = Field(min_length=1)
    primary_result_unit: str = Field(min_length=1)
    primary_method_id: str = Field(min_length=1)
    characterisation: TrialCharacterisation

    @model_validator(mode="after")
    def _profile_matches_characterisation(self) -> TrialProfile:
        if self.characterisation.trial_id != self.independence_unit_id:
            raise ValueError(
                "trial profile and characterisation identify different independent trials"
            )
        if self.characterisation.design_profile_id != self.design_profile_id:
            raise ValueError(
                "trial profile and characterisation identify different design profiles"
            )
        participant_counts = [
            row.estimate
            for row in self.characterisation.estimates
            if row.property_id == "trial.participant_count" and row.group == "overall"
        ]
        if len(participant_counts) != 1:
            raise ValueError(
                "trial profile requires one overall participant-count estimate"
            )
        if self.participant_count != round(participant_counts[0]):
            raise ValueError(
                "trial profile participant count disagrees with characterisation"
            )
        return self


class WorkedTrialLineage(_Contract):
    """Participant-to-analysis lineage for the selected worked trial."""

    case_id: str = Field(min_length=1)
    task_id: str = Field(pattern=r"^TASK[0-9A-F]{32}$")
    participant_table_path: str = Field(min_length=1)
    endpoint_table_path: str = Field(min_length=1)
    task_path: str = Field(min_length=1)
    protocol_path: str = Field(min_length=1)
    analysis_plan_path: str = Field(min_length=1)
    participant_rows: int = Field(ge=1)
    endpoint_rows: int = Field(ge=1)
    linked_rows: int = Field(ge=1)
    analysis_population: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    effect_scale: str = Field(min_length=1)
    estimator_method: str = Field(min_length=1)
    estimate: float
    interval_low: float
    interval_high: float
    unit: str = Field(min_length=1)
    uncertainty_method: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_lineage(self) -> WorkedTrialLineage:
        if (
            self.participant_rows != self.endpoint_rows
            or self.participant_rows != self.linked_rows
        ):
            raise ValueError(
                "worked-trial participant and endpoint linkage must be one-to-one"
            )
        if not self.interval_low <= self.estimate <= self.interval_high:
            raise ValueError("worked-trial estimate must lie within its interval")
        return self


class ReleaseCharacterisation(_Contract):
    """Complete C1 round robin and selected worked-trial lineage."""

    schema_id: Literal["trialagentbench.release_characterisation/v1"] = (
        "trialagentbench.release_characterisation/v1"
    )
    release_id: str = Field(min_length=1)
    participant_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalogue_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_view_count: int = Field(ge=1)
    independent_trial_count: int = Field(ge=1)
    profiles: tuple[TrialProfile, ...] = Field(min_length=1)
    worked_trial: WorkedTrialLineage

    @model_validator(mode="after")
    def _complete_release(self) -> ReleaseCharacterisation:
        if self.independent_trial_count != len(self.profiles):
            raise ValueError("independent trial count must equal the profile census")
        task_ids = tuple(row.task_id for row in self.profiles)
        independence_ids = tuple(row.independence_unit_id for row in self.profiles)
        if len(task_ids) != len(set(task_ids)) or len(independence_ids) != len(
            set(independence_ids)
        ):
            raise ValueError(
                "release profiles must be unique by task and independence unit"
            )
        if self.worked_trial.task_id not in set(task_ids):
            raise ValueError("worked trial must belong to the C1 profile census")
        expected_profiles = {f"TE-DP0{index}" for index in range(1, 8)}
        if {row.design_profile_id for row in self.profiles} != expected_profiles:
            raise ValueError(
                "release characterisation must cover all seven design profiles"
            )
        return self


class DesignProperty(_Contract):
    """One design property measured from public trial records."""

    task_id: str = Field(pattern=r"^TASK[0-9A-F]{32}$")
    independence_unit_id: str = Field(min_length=1)
    matched_set_id: str = Field(min_length=1)
    design_profile_id: Literal[
        "TE-DP01",
        "TE-DP02",
        "TE-DP03",
        "TE-DP04",
        "TE-DP05",
        "TE-DP06",
        "TE-DP07",
    ]
    design_tier: Literal["D1", "D2", "D3", "D4"]
    assumption_tier: AssumptionTier
    property_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    estimate: float
    unit: str = Field(min_length=1)
    source: Literal["participant_table", "endpoint_table", "protocol"]
    definition: str = Field(min_length=1)

    @model_validator(mode="after")
    def _finite_estimate(self) -> DesignProperty:
        if not math.isfinite(self.estimate):
            raise ValueError("design-property estimates must be finite")
        return self


class DesignAnalysisComparison(_Contract):
    """Design-aware and design-naive analyses of one estimand."""

    task_id: str = Field(pattern=r"^TASK[0-9A-F]{32}$")
    independence_unit_id: str = Field(min_length=1)
    matched_set_id: str = Field(min_length=1)
    design_profile_id: Literal["TE-DP03", "TE-DP04", "TE-DP05", "TE-DP06", "TE-DP07"]
    design_tier: Literal["D2", "D3", "D4"]
    assumption_tier: AssumptionTier
    comparison_id: Literal[
        "covariate_adjusted_vs_unadjusted",
        "validation_corrected_vs_observed",
        "cluster_robust_vs_independent",
        "period_adjusted_vs_unadjusted",
        "group_sequential_vs_fixed",
    ]
    qualified_method: str = Field(min_length=1)
    qualified_estimate: float
    qualified_interval_low: float
    qualified_interval_high: float
    naive_method: str = Field(min_length=1)
    naive_estimate: float
    naive_interval_low: float
    naive_interval_high: float
    unit: str = Field(min_length=1)
    independent_unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_intervals(self) -> DesignAnalysisComparison:
        values = (
            self.qualified_estimate,
            self.qualified_interval_low,
            self.qualified_interval_high,
            self.naive_estimate,
            self.naive_interval_low,
            self.naive_interval_high,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("design-comparison values must be finite")
        if (
            not self.qualified_interval_low
            <= self.qualified_estimate
            <= self.qualified_interval_high
        ):
            raise ValueError("qualified estimate must lie inside its interval")
        if (
            not self.naive_interval_low
            <= self.naive_estimate
            <= self.naive_interval_high
        ):
            raise ValueError("naive estimate must lie inside its interval")
        return self


class DesignReleaseCharacterisation(_Contract):
    """Complete public-data characterisation of the release Design axis."""

    schema_id: Literal["trialagentbench.design_release_characterisation/v1"] = (
        "trialagentbench.design_release_characterisation/v1"
    )
    release_id: str = Field(min_length=1)
    participant_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalogue_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    independent_trial_count: int = Field(ge=1)
    properties: tuple[DesignProperty, ...] = Field(min_length=1)
    comparisons: tuple[DesignAnalysisComparison, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _complete_design_axis(self) -> DesignReleaseCharacterisation:
        trial_ids = {row.independence_unit_id for row in self.properties}
        if len(trial_ids) != self.independent_trial_count:
            raise ValueError("design properties must cover every independent trial")
        expected_profiles = {f"TE-DP0{index}" for index in range(1, 8)}
        if {row.design_profile_id for row in self.properties} != expected_profiles:
            raise ValueError("design properties must cover all seven profiles")
        property_keys = tuple((row.task_id, row.property_id) for row in self.properties)
        if len(property_keys) != len(set(property_keys)):
            raise ValueError("design properties must be unique within each task")
        comparison_keys = tuple(
            (row.task_id, row.comparison_id) for row in self.comparisons
        )
        if len(comparison_keys) != len(set(comparison_keys)):
            raise ValueError("design comparisons must be unique within each task")
        expected_comparisons = {"TE-DP03", "TE-DP04", "TE-DP05", "TE-DP06", "TE-DP07"}
        if {row.design_profile_id for row in self.comparisons} != expected_comparisons:
            raise ValueError(
                "design comparisons must cover every analysis-relevant subdesign"
            )
        return self


class AssumptionSeriesIdentity(_Contract):
    """Matched scientific identity for one series replicate."""

    series_id: AssumptionSeriesId
    replicate_index: int = Field(ge=1)
    assumption_tiers: tuple[AssumptionTier, ...] = Field(min_length=1)
    task_ids: dict[AssumptionTier, str] = Field(min_length=1)
    random_stream_id: str = Field(min_length=1)
    design_profile_id: str = Field(min_length=1)
    population: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    effect_scale: str = Field(min_length=1)
    default_method: str = Field(min_length=1)
    participant_count: int = Field(ge=1)
    follow_up_horizon_days: float = Field(gt=0)
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _complete_identity(self) -> AssumptionSeriesIdentity:
        if set(self.task_ids) != set(self.assumption_tiers):
            raise ValueError(
                "matched identity task IDs must cover its assumption tiers"
            )
        if len(set(self.task_ids.values())) != len(self.task_ids):
            raise ValueError("matched identity task IDs must be unique")
        return self


class MatchedAssumptionDesign(_Contract):
    """Public pairing design for an Assumption-axis response experiment."""

    schema_id: Literal["trialagentbench.matched_assumption_design/v1"] = (
        "trialagentbench.matched_assumption_design/v1"
    )
    release_id: str = Field(min_length=1)
    analysis_count: int = Field(ge=1)
    identities: tuple[AssumptionSeriesIdentity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _complete_design(self) -> MatchedAssumptionDesign:
        keys = tuple((row.series_id, row.replicate_index) for row in self.identities)
        if len(keys) != len(set(keys)):
            raise ValueError("matched assumption identities must be unique")
        task_ids = tuple(
            task_id for row in self.identities for task_id in row.task_ids.values()
        )
        if len(task_ids) != self.analysis_count:
            raise ValueError(
                "matched assumption analysis count must equal the task identity census"
            )
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("matched assumption task IDs must be unique")
        if len({row.random_stream_id for row in self.identities}) != len(
            self.identities
        ):
            raise ValueError("random stream IDs must be unique by series replicate")
        return self


class AssumptionAnalysisBridge(_Contract):
    """Observable mechanism and same-estimand analysis response for one released trial."""

    task_id: str = Field(pattern=r"^TASK[0-9A-F]{32}$")
    independence_unit_id: str = Field(min_length=1)
    series_id: AssumptionSeriesId
    replicate_index: int = Field(ge=1)
    assumption_tier: AssumptionTier
    design_profile_id: str = Field(pattern=r"^TE-DP0[1-7]$")
    participant_count: int = Field(ge=1)
    follow_up_horizon_days: float = Field(gt=0)
    endpoint_id: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    effect_scale: str = Field(min_length=1)
    mechanism_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    mechanism_label: str = Field(min_length=1)
    mechanism_value: float | None = None
    mechanism_unit: str | None = Field(default=None, min_length=1)
    mechanism_band: str = Field(min_length=1)
    diagnostic_status: str = Field(min_length=1)
    default_method: str = Field(min_length=1)
    default_status: Literal["estimated", "incompatible"]
    default_value: float | None = None
    default_standard_error: float | None = Field(default=None, ge=0)
    default_interval_low: float | None = None
    default_interval_high: float | None = None
    qualified_method: str = Field(min_length=1)
    qualified_shape: Literal["point", "bound"]
    qualified_value: float
    qualified_standard_error: float | None = Field(default=None, ge=0)
    qualified_interval_low: float
    qualified_interval_high: float
    result_unit: str = Field(min_length=1)
    absolute_analysis_difference: float | None = Field(default=None, ge=0)
    default_rejects_null: bool | None = None
    qualified_rejects_null: bool | None = None
    qualified_replay_abs_error: float = Field(ge=0)
    analysis_failure: bool = False

    @model_validator(mode="after")
    def _valid_analysis_response(self) -> AssumptionAnalysisBridge:
        values = (
            self.mechanism_value,
            self.default_value,
            self.default_standard_error,
            self.default_interval_low,
            self.default_interval_high,
            self.qualified_value,
            self.qualified_standard_error,
            self.qualified_interval_low,
            self.qualified_interval_high,
            self.absolute_analysis_difference,
            self.qualified_replay_abs_error,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("assumption-bridge values must be finite")
        if (self.mechanism_value is None) != (self.mechanism_unit is None):
            raise ValueError("numeric mechanisms require a unit")
        default_fields = (
            self.default_value,
            self.default_standard_error,
            self.default_interval_low,
            self.default_interval_high,
        )
        if self.default_status == "estimated" and any(
            value is None for value in default_fields
        ):
            raise ValueError(
                "estimated default analyses require a point, standard error, and interval"
            )
        if self.default_status == "incompatible" and any(
            value is not None for value in default_fields
        ):
            raise ValueError(
                "incompatible default analyses cannot provide numeric results"
            )
        if (
            self.default_status == "incompatible"
            and self.absolute_analysis_difference is not None
        ):
            raise ValueError(
                "incompatible defaults cannot provide an analysis difference"
            )
        if (
            not self.qualified_interval_low
            <= self.qualified_value
            <= self.qualified_interval_high
        ):
            raise ValueError("qualified result must lie within its interval")
        if self.qualified_shape == "point":
            if self.qualified_standard_error is None:
                raise ValueError("point responses require a standard error")
        elif (
            self.qualified_standard_error is not None
            or self.qualified_rejects_null is not None
        ):
            raise ValueError(
                "identification bounds cannot carry sampling standard errors or null tests"
            )
        if self.default_status == "estimated":
            assert self.default_value is not None
            assert self.default_interval_low is not None
            assert self.default_interval_high is not None
            if (
                not self.default_interval_low
                <= self.default_value
                <= self.default_interval_high
            ):
                raise ValueError("default result must lie within its interval")
            expected = abs(self.default_value - self.qualified_value)
            if self.absolute_analysis_difference is None or not math.isclose(
                self.absolute_analysis_difference,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "analysis difference must equal the absolute point-estimate contrast"
                )
        return self


class AssumptionIdentificationResult(_Contract):
    """One identified risk-difference range under an explicit A4 assumption."""

    task_id: str = Field(pattern=r"^TASK[0-9A-F]{32}$")
    series_id: Literal["TE-S04", "TE-S06"]
    replicate_index: int = Field(ge=1)
    model: Literal["dependent_censoring", "endpoint_validation_transport"]
    assumption: Literal["bounded_deviation", "worst_case"]
    sensitivity_parameter: float | None = Field(default=None, gt=0.0, lt=1.0)
    lower: float
    upper: float
    midpoint: float
    width: float = Field(gt=0.0)
    result_unit: Literal["risk difference"] = "risk difference"
    reference_role: Literal[
        "required_primary",
        "sensitivity_only",
        "credit_eligible_primary_alternative",
    ]
    replay_absolute_error: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _valid_identified_set(self) -> AssumptionIdentificationResult:
        values = (
            self.lower,
            self.upper,
            self.midpoint,
            self.width,
            self.replay_absolute_error,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("identified-set results must be finite")
        if not self.lower <= self.midpoint <= self.upper:
            raise ValueError("identified-set midpoint must lie within its bounds")
        if not math.isclose(
            self.midpoint, (self.lower + self.upper) / 2.0, abs_tol=1e-12
        ):
            raise ValueError("identified-set midpoint must equal the bound midpoint")
        if not math.isclose(self.width, self.upper - self.lower, abs_tol=1e-12):
            raise ValueError("identified-set width must equal upper minus lower")
        if (self.assumption == "bounded_deviation") != (
            self.sensitivity_parameter is not None
        ):
            raise ValueError(
                "bounded-deviation results require a sensitivity parameter"
            )
        return self


class AssumptionTierSummary(_Contract):
    """Finite-release summary for one assumption-series cell."""

    series_id: AssumptionSeriesId
    assumption_tier: AssumptionTier
    trial_count: int = Field(ge=1)
    mechanism_value_mean: float | None = None
    mechanism_value_interval_low: float | None = None
    mechanism_value_interval_high: float | None = None
    mechanism_unit: str | None = Field(default=None, min_length=1)
    mean_absolute_analysis_difference: float | None = Field(default=None, ge=0)
    difference_interval_low: float | None = Field(default=None, ge=0)
    difference_interval_high: float | None = Field(default=None, ge=0)
    result_unit: str = Field(min_length=1)
    default_rejection_fraction: float | None = Field(default=None, ge=0, le=1)
    qualified_rejection_fraction: float | None = Field(default=None, ge=0, le=1)
    analysis_failure_count: int = Field(ge=0)
    uncertainty_method: Literal[
        "t_interval_across_independent_trials", "not_applicable"
    ]

    @model_validator(mode="after")
    def _valid_summary(self) -> AssumptionTierSummary:
        mechanism = (
            self.mechanism_value_mean,
            self.mechanism_value_interval_low,
            self.mechanism_value_interval_high,
            self.mechanism_unit,
        )
        if any(value is None for value in mechanism) != all(
            value is None for value in mechanism
        ):
            raise ValueError(
                "mechanism summaries require a mean, interval, and unit together"
            )
        difference = (
            self.mean_absolute_analysis_difference,
            self.difference_interval_low,
            self.difference_interval_high,
        )
        if any(value is None for value in difference) != all(
            value is None for value in difference
        ):
            raise ValueError(
                "analysis-difference summaries require a mean and interval together"
            )
        if self.mechanism_value_interval_low is not None:
            assert self.mechanism_value_mean is not None
            assert self.mechanism_value_interval_high is not None
            if (
                not self.mechanism_value_interval_low
                <= self.mechanism_value_mean
                <= self.mechanism_value_interval_high
            ):
                raise ValueError("mechanism mean must lie within its interval")
        if self.difference_interval_low is not None:
            assert self.mean_absolute_analysis_difference is not None
            assert self.difference_interval_high is not None
            if (
                not self.difference_interval_low
                <= self.mean_absolute_analysis_difference
                <= self.difference_interval_high
            ):
                raise ValueError("difference mean must lie within its interval")
        if self.analysis_failure_count > self.trial_count:
            raise ValueError("analysis failures cannot exceed the trial count")
        return self


class AssumptionPairContrast(_Contract):
    """Paired mechanism and analysis-consequence change between two tiers."""

    series_id: AssumptionSeriesId
    lower_tier: Literal["A1", "A2"]
    upper_tier: Literal["A2", "A3"]
    trial_pair_count: int = Field(ge=2)
    mechanism_change_mean: float
    mechanism_change_interval_low: float
    mechanism_change_interval_high: float
    mechanism_unit: str = Field(min_length=1)
    consequence_change_mean: float
    consequence_change_interval_low: float
    consequence_change_interval_high: float
    default_value_change_mean: float
    default_value_change_interval_low: float
    default_value_change_interval_high: float
    default_magnitude_change_mean: float
    default_magnitude_change_interval_low: float
    default_magnitude_change_interval_high: float
    result_unit: str = Field(min_length=1)
    uncertainty_method: Literal["paired_t_interval_across_random_streams"] = (
        "paired_t_interval_across_random_streams"
    )

    @model_validator(mode="after")
    def _valid_contrast(self) -> AssumptionPairContrast:
        if self.lower_tier >= self.upper_tier:
            raise ValueError("paired tier contrast must follow A1, A2, A3 order")
        if not (
            self.mechanism_change_interval_low
            <= self.mechanism_change_mean
            <= self.mechanism_change_interval_high
        ):
            raise ValueError("mechanism change must lie within its interval")
        if not (
            self.consequence_change_interval_low
            <= self.consequence_change_mean
            <= self.consequence_change_interval_high
        ):
            raise ValueError("analysis consequence must lie within its interval")
        if not (
            self.default_value_change_interval_low
            <= self.default_value_change_mean
            <= self.default_value_change_interval_high
        ):
            raise ValueError("default-value change must lie within its interval")
        if not (
            self.default_magnitude_change_interval_low
            <= self.default_magnitude_change_mean
            <= self.default_magnitude_change_interval_high
        ):
            raise ValueError("default-magnitude change must lie within its interval")
        return self


class AssumptionResponseFigureRow(_Contract):
    """One mechanism and treatment-result response point for display."""

    series_id: AssumptionSeriesId
    assumption_tier: Literal["A1", "A2", "A3"]
    trial_count: int = Field(ge=2)
    mechanism_value_mean: float
    mechanism_value_interval_low: float
    mechanism_value_interval_high: float
    mechanism_label: str = Field(min_length=1)
    mechanism_unit: str = Field(min_length=1)
    consequence_value_mean: float
    consequence_interval_low: float
    consequence_interval_high: float
    consequence_unit: str = Field(min_length=1)
    consequence_label: str = Field(min_length=1)
    uncertainty_method: Literal["t_interval_across_independent_trials"] = (
        "t_interval_across_independent_trials"
    )

    @model_validator(mode="after")
    def _valid_intervals(self) -> AssumptionResponseFigureRow:
        if not (
            self.mechanism_value_interval_low
            <= self.mechanism_value_mean
            <= self.mechanism_value_interval_high
        ):
            raise ValueError("mechanism response mean must lie within its interval")
        if not (
            self.consequence_interval_low
            <= self.consequence_value_mean
            <= self.consequence_interval_high
        ):
            raise ValueError("analysis-consequence mean must lie within its interval")
        return self


class AssumptionReleaseCharacterisation(_Contract):
    """Characterisation of a finite release or matched Assumption response."""

    schema_id: Literal["trialagentbench.assumption_release_characterisation/v1"] = (
        "trialagentbench.assumption_release_characterisation/v1"
    )
    release_id: str = Field(min_length=1)
    analysis_scope: Literal["finite_release", "matched_response"] = "finite_release"
    participant_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_count: int = Field(ge=1)
    identities: tuple[AssumptionSeriesIdentity, ...] = ()
    bridges: tuple[AssumptionAnalysisBridge, ...] = Field(min_length=1)
    identification_results: tuple[AssumptionIdentificationResult, ...] = ()
    summaries: tuple[AssumptionTierSummary, ...] = Field(min_length=1)
    paired_contrasts: tuple[AssumptionPairContrast, ...] = ()

    @model_validator(mode="after")
    def _complete_assumption_axis(self) -> AssumptionReleaseCharacterisation:
        finite_tiers: dict[str, tuple[str, ...]] = {
            "TE-S01": ("A1", "A2", "A3"),
            "TE-S02": ("A1", "A2", "A3"),
            "TE-S03": ("A1", "A2"),
            "TE-S04": ("A1", "A2", "A3", "A4"),
            "TE-S05": ("A1", "A2", "A3"),
            "TE-S06": ("A1", "A2", "A3", "A4"),
            "TE-S07": ("A1", "A2", "A3"),
            "TE-S08": ("A1", "A2"),
            "TE-S09": ("A4",),
        }
        matched_tiers = {
            series_id: tuple(tier for tier in tiers if tier != "A4")
            for series_id, tiers in finite_tiers.items()
            if any(tier != "A4" for tier in tiers)
        }
        expected_tiers = (
            finite_tiers if self.analysis_scope == "finite_release" else matched_tiers
        )
        if len(self.bridges) != self.analysis_count:
            raise ValueError("assumption bridges must cover every analysis")
        if self.analysis_scope == "finite_release":
            if self.identities or self.paired_contrasts:
                raise ValueError(
                    "finite-release summaries cannot claim pair-matched results"
                )
            a4_tasks = {
                row.task_id
                for row in self.bridges
                if row.assumption_tier == "A4" and row.series_id in {"TE-S04", "TE-S06"}
            }
            if {row.task_id for row in self.identification_results} != a4_tasks:
                raise ValueError(
                    "A4 identified-set results must match the partially identified trial census"
                )
            response_by_task = {
                task_id: tuple(
                    row for row in self.identification_results if row.task_id == task_id
                )
                for task_id in a4_tasks
            }
            if set(response_by_task) != a4_tasks or any(
                len(rows) != 4 for rows in response_by_task.values()
            ):
                raise ValueError(
                    "each partially identified A4 trial requires three deviations and one worst-case range"
                )
            for task_id, rows in response_by_task.items():
                deltas = tuple(
                    sorted(
                        row.sensitivity_parameter
                        for row in rows
                        if row.assumption == "bounded_deviation"
                        and row.sensitivity_parameter is not None
                    )
                )
                if deltas != (0.05, 0.10, 0.20):
                    raise ValueError(
                        f"{task_id} A4 sensitivity curve must use deviations 0.05, 0.10, and 0.20"
                    )
                if sum(row.assumption == "worst_case" for row in rows) != 1:
                    raise ValueError(
                        f"{task_id} A4 sensitivity curve requires one worst-case range"
                    )
        else:
            if self.identification_results:
                raise ValueError(
                    "matched A1-A3 response experiments cannot contain A4 identified sets"
                )
            expected_identity_keys = {
                (row.series_id, row.replicate_index) for row in self.bridges
            }
            identity_keys = {
                (row.series_id, row.replicate_index) for row in self.identities
            }
            if identity_keys != expected_identity_keys or len(identity_keys) != len(
                self.identities
            ):
                raise ValueError(
                    "matched-series identities must cover every series replicate exactly once"
                )
            expected_contrasts = {
                (series_id, lower, upper)
                for series_id, tiers in expected_tiers.items()
                for lower, upper in zip(tiers, tiers[1:], strict=False)
            }
            observed_contrasts = {
                (row.series_id, row.lower_tier, row.upper_tier)
                for row in self.paired_contrasts
            }
            if observed_contrasts != expected_contrasts or len(
                observed_contrasts
            ) != len(self.paired_contrasts):
                raise ValueError(
                    "paired contrasts must cover every adjacent matched tier"
                )
        keys = tuple(
            (row.series_id, row.assumption_tier, row.replicate_index)
            for row in self.bridges
        )
        if len(keys) != len(set(keys)):
            raise ValueError(
                "assumption bridges must be unique by series, tier, and replicate"
            )
        for series_id, tiers in expected_tiers.items():
            observed = tuple(
                sorted(
                    {
                        row.assumption_tier
                        for row in self.bridges
                        if row.series_id == series_id
                    }
                )
            )
            if observed != tiers:
                raise ValueError(
                    f"{series_id} assumption tiers do not match the canonical series"
                )
        if self.analysis_scope == "finite_release":
            if any(
                len(
                    [
                        row
                        for row in self.bridges
                        if row.series_id == series_id and row.assumption_tier == tier
                    ]
                )
                != 4
                for series_id, tiers in expected_tiers.items()
                for tier in tiers
            ):
                raise ValueError(
                    "every released assumption cell must contain four independent trials"
                )
        else:
            replicate_sets = {
                series_id: {
                    row.replicate_index
                    for row in self.identities
                    if row.series_id == series_id
                }
                for series_id in expected_tiers
            }
            if any(len(replicates) < 4 for replicates in replicate_sets.values()):
                raise ValueError(
                    "every matched assumption series requires at least four trial replicates"
                )
            for series_id, tiers in expected_tiers.items():
                for tier in tiers:
                    observed_replicates = {
                        row.replicate_index
                        for row in self.bridges
                        if row.series_id == series_id and row.assumption_tier == tier
                    }
                    if observed_replicates != replicate_sets[series_id]:
                        raise ValueError(
                            "every matched assumption tier must cover its declared trial replicates"
                        )
        summary_keys = {(row.series_id, row.assumption_tier) for row in self.summaries}
        expected_summary_keys = {
            (series_id, tier)
            for series_id, tiers in expected_tiers.items()
            for tier in tiers
        }
        if summary_keys != expected_summary_keys:
            raise ValueError(
                "assumption summaries must cover every released series cell"
            )
        return self


def _validate_horizons(values: tuple[float, ...]) -> None:
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("outcome times must be finite and nonnegative")
    if tuple(sorted(set(values))) != values:
        raise ValueError("outcome times must be strictly increasing")


__all__ = [
    "BinaryOutcomeSpec",
    "AssumptionAnalysisBridge",
    "AssumptionIdentificationResult",
    "AssumptionPairContrast",
    "AssumptionResponseFigureRow",
    "AssumptionReleaseCharacterisation",
    "AssumptionSeriesId",
    "AssumptionSeriesIdentity",
    "MatchedAssumptionDesign",
    "AssumptionTier",
    "AssumptionTierSummary",
    "CategoricalVariableSpec",
    "CharacterisationCollection",
    "CompetingRiskOutcomeSpec",
    "ContinuousOutcomeSpec",
    "ContinuousVariableSpec",
    "DependenceSpec",
    "DesignAnalysisComparison",
    "DesignFamily",
    "DesignProperty",
    "DesignReleaseCharacterisation",
    "EvidenceLevel",
    "LongitudinalOutcomeSpec",
    "MissingnessDisposition",
    "OrdinalOutcomeSpec",
    "OutcomeSpec",
    "RecurrentEventOutcomeSpec",
    "ReleaseCharacterisation",
    "SurvivalOutcomeSpec",
    "TidyEstimate",
    "TrialCharacterisation",
    "TrialCharacterisationSpec",
    "TrialData",
    "TrialProfile",
    "WorkedTrialLineage",
]
