"""Contracts for independent TrialDevBench observational replay."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrialDevCandidateReplayV1(BaseModel):
    """Independent point and uncertainty comparison for one candidate and objective."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective_id: str = Field(min_length=1)
    candidate_drug_id: str = Field(min_length=1)
    expected_utility: float
    replayed_utility: float
    utility_absolute_error: float = Field(ge=0.0)
    expected_efficacy_gain: float
    replayed_efficacy_gain: float
    efficacy_gain_absolute_error: float = Field(ge=0.0)
    expected_utility_standard_error: float = Field(ge=0.0)
    replayed_utility_standard_error: float = Field(ge=0.0)
    utility_standard_error_absolute_error: float = Field(ge=0.0)
    expected_efficacy_gain_standard_error: float = Field(ge=0.0)
    replayed_efficacy_gain_standard_error: float = Field(ge=0.0)
    efficacy_gain_standard_error_absolute_error: float = Field(ge=0.0)
    expected_utility_interval: tuple[float, float]
    replayed_utility_interval: tuple[float, float]
    expected_efficacy_gain_interval: tuple[float, float]
    replayed_efficacy_gain_interval: tuple[float, float]
    maximum_interval_endpoint_absolute_error: float = Field(ge=0.0)
    within_tolerance: bool


class TrialDevMethodReplayV1(BaseModel):
    """Independent replay result for one declared observational method."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_route_id: str = Field(min_length=1)
    estimator_id: str = Field(min_length=1)
    uncertainty_estimator_id: str = Field(min_length=1)
    bootstrap_replicates: int = Field(ge=2)
    confidence_level: float = Field(gt=0.5, lt=1.0)
    result_form: Literal["point_estimates", "qualified_non_nomination"] = (
        "point_estimates"
    )
    expected_non_estimability_reason: (
        Literal[
            "empirical_positivity_violation",
            "empty_standardization_cell",
            "residual_unmeasured_confounding",
        ]
        | None
    ) = None
    replayed_non_estimability_reason: (
        Literal[
            "empirical_positivity_violation",
            "empty_standardization_cell",
            "residual_unmeasured_confounding",
        ]
        | None
    ) = None
    non_estimability_match: bool = True
    candidate_results: tuple[TrialDevCandidateReplayV1, ...] = ()
    expected_rankings: dict[str, tuple[str, ...]]
    replayed_rankings: dict[str, tuple[str, ...]]
    expected_actions: dict[str, tuple[str, ...]]
    replayed_actions: dict[str, tuple[str, ...]]
    expected_acceptable_utility_sets: dict[str, tuple[str, ...]]
    replayed_acceptable_utility_sets: dict[str, tuple[str, ...]]
    expected_definitely_qualified_sets: dict[str, tuple[str, ...]]
    replayed_definitely_qualified_sets: dict[str, tuple[str, ...]]
    expected_possibly_qualified_sets: dict[str, tuple[str, ...]]
    replayed_possibly_qualified_sets: dict[str, tuple[str, ...]]
    expected_pairwise_contrast_half_widths: dict[str, dict[str, float]]
    replayed_pairwise_contrast_half_widths: dict[str, dict[str, float]]
    maximum_utility_absolute_error: float = Field(ge=0.0)
    maximum_efficacy_gain_absolute_error: float = Field(ge=0.0)
    maximum_standard_error_absolute_error: float = Field(ge=0.0)
    maximum_interval_endpoint_absolute_error: float = Field(ge=0.0)
    maximum_pairwise_contrast_absolute_error: float = Field(ge=0.0)
    ranking_match: bool
    action_match: bool
    uncertainty_policy_match: bool
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> TrialDevMethodReplayV1:
        """Require status to summarize all method-level checks."""

        if self.result_form == "point_estimates":
            if (
                not self.candidate_results
                or self.expected_non_estimability_reason is not None
                or self.replayed_non_estimability_reason is not None
            ):
                raise ValueError(
                    "point-estimate replay requires numeric results and no non-estimability reason."
                )
        elif self.candidate_results or (
            self.expected_non_estimability_reason is None
            and self.replayed_non_estimability_reason is None
        ):
            raise ValueError(
                "qualified non-nomination replay requires a non-estimability reason and no point results."
            )
        expected = (
            all(row.within_tolerance for row in self.candidate_results)
            and self.ranking_match
            and self.action_match
            and self.uncertainty_policy_match
            and self.non_estimability_match
        )
        if (self.status == "pass") != expected:
            raise ValueError(
                "method replay status disagrees with its point, ranking, or action checks."
            )
        return self


class TrialDevObservationalReplayReportV1(BaseModel):
    """Independent participant-evidence replay report for one TrialDev scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_observational_replay/v1"] = (
        "trialagentbench.trialdev_observational_replay/v1"
    )
    scenario_id: str = Field(min_length=1)
    absolute_tolerance: float = Field(gt=0.0)
    public_input_checksums_match: bool
    methods: tuple[TrialDevMethodReplayV1, ...] = Field(min_length=1)
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> TrialDevObservationalReplayReportV1:
        """Require report status to summarize checksum and method results."""

        expected = self.public_input_checksums_match and all(
            row.status == "pass" for row in self.methods
        )
        if (self.status == "pass") != expected:
            raise ValueError(
                "replay report status disagrees with checksum or method results."
            )
        return self


__all__ = [
    "TrialDevCandidateReplayV1",
    "TrialDevMethodReplayV1",
    "TrialDevObservationalReplayReportV1",
]
