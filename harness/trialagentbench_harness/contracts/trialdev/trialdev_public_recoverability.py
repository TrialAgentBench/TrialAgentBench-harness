"""TrialDev public-surface recoverability contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.trialdev.trialdev_recoverability import (
    TrialDevObjectiveIdV1,
    TrialDevRecoverabilityPolicyV1,
)
from trialagentbench_harness.io.json import read_json_model
from trialagentbench_harness.trialdev.share.public_method_design import (
    TrialDevPublicEfficacyEndpointV1 as TrialDevPublicEfficacyEndpointV1,
)
from trialagentbench_harness.trialdev.share.public_method_design import (
    TrialDevPublicEntropyBalancedAnalysisSpecV1,
    TrialDevPublicObjectiveCharterV1,
    TrialDevPublicObjectiveSpecV1,
    TrialDevPublicObservationalAnalysisSpecV1,
    TrialDevPublicObservationalMethodSpecV1,
    TrialDevPublicUtilityComponentV1,
    TrialDevPublicUtilityEventV1,
)

WITHHOLD_NOMINATION_TARGET_ID = "withhold_nomination"


def _stable_checksum(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TrialDevPublicInputChecksumV1(BaseModel):
    """Checksum for one public input used by the recoverability solver."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=64, max_length=64)


class TrialDevPublicObjectiveAnalysisManifestV1(TrialDevPublicObjectiveCharterV1):
    """Typed analysis projection of the canonical public objective charter."""

    @property
    def objective_specs(self) -> tuple[TrialDevPublicObjectiveSpecV1, ...]:
        """Return the strictly parsed public objective contracts."""

        return self.objectives


class TrialDevPublicCandidateScoreV1(BaseModel):
    """Candidate-level public-surface score and diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_public_candidate_score_v1"] = "trialdev_public_candidate_score_v1"
    scenario_id: str = Field(..., min_length=1)
    objective_id: TrialDevObjectiveIdV1
    candidate_drug_id: str = Field(..., min_length=1)
    adjusted_utility: float | None = None
    utility_se: float | None = Field(default=None, ge=0.0)
    ci_low: float | None = None
    ci_high: float | None = None
    efficacy_gain: float | None = None
    efficacy_gain_se: float | None = Field(default=None, ge=0.0)
    efficacy_gain_ci_low: float | None = None
    efficacy_gain_ci_high: float | None = None
    rank: int | None = Field(default=None, ge=1)
    utility_margin_to_best: float | None = Field(default=None, ge=0.0)
    support_count: int = Field(..., ge=0)
    min_stratum_count: int = Field(..., ge=0)
    max_abs_unadjusted_smd_vs_target: float = Field(..., ge=0.0)
    max_abs_adjusted_smd_vs_target: float | None = Field(default=None, ge=0.0)
    point_estimable: bool
    inference_estimable: bool
    score_state: Literal["scoreable", "near_tie", "insufficient_recoverability"]

    @model_validator(mode="after")
    def validate_estimability(self) -> TrialDevPublicCandidateScoreV1:
        """Require point and uncertainty fields exactly when their estimands are available."""

        numeric = (self.adjusted_utility, self.rank, self.utility_margin_to_best)
        uncertainty = (
            self.utility_se,
            self.ci_low,
            self.ci_high,
            self.efficacy_gain,
            self.efficacy_gain_se,
            self.efficacy_gain_ci_low,
            self.efficacy_gain_ci_high,
        )
        if self.inference_estimable and not self.point_estimable:
            raise ValueError("candidate inference requires an estimable point utility.")
        if self.point_estimable != all(value is not None for value in numeric):
            raise ValueError("point_estimable must agree with the point score fields.")
        if self.inference_estimable != (
            all(value is not None for value in uncertainty) and self.max_abs_adjusted_smd_vs_target is not None
        ):
            raise ValueError("inference_estimable must agree with the uncertainty fields.")
        return self


class TrialDevPublicObservationalActionPolicyV1(BaseModel):
    """Publicly recoverable nominate-or-decline policy for observational review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective_id: TrialDevObjectiveIdV1
    minimum_efficacy_gain: float
    reference_target_ids: tuple[str, ...] = Field(..., min_length=1)
    credit_eligible_target_ids: tuple[str, ...] = Field(..., min_length=1)
    definitely_qualified_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    possibly_qualified_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    utility_contrast_half_widths: dict[str, float] = Field(default_factory=dict)
    pairwise_utility_contrast_half_widths: dict[str, float] = Field(default_factory=dict)
    rationale: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_action_policy(self) -> TrialDevPublicObservationalActionPolicyV1:
        """Require nested qualification sets and acceptable reference targets."""

        definite = set(self.definitely_qualified_candidate_ids)
        possible = set(self.possibly_qualified_candidate_ids)
        if not definite <= possible:
            raise ValueError("definitely qualified candidates must be possibly qualified.")
        if any(
            len(values) != len(set(values))
            for values in (
                self.definitely_qualified_candidate_ids,
                self.possibly_qualified_candidate_ids,
            )
        ):
            raise ValueError("observational qualification candidate sets must be unique.")
        if set(self.utility_contrast_half_widths) != possible:
            raise ValueError("utility contrast half-widths must cover exactly the possibly qualified candidates.")
        if any(not math.isfinite(value) or value < 0.0 for value in self.utility_contrast_half_widths.values()):
            raise ValueError("utility contrast half-widths must be finite and non-negative.")
        candidate_ids = tuple(sorted(self.possibly_qualified_candidate_ids))
        expected_pairs = {
            f"{first}|{second}" for index, first in enumerate(candidate_ids) for second in candidate_ids[index + 1 :]
        }
        observed_pairs = set(self.pairwise_utility_contrast_half_widths)
        if not expected_pairs <= observed_pairs:
            raise ValueError(
                "pairwise utility contrast half-widths must cover each possibly qualified candidate pair."
            )
        if any(len(parts := key.split("|")) != 2 or not all(parts) or parts[0] >= parts[1] for key in observed_pairs):
            raise ValueError("pairwise utility contrast keys must use ascending distinct asset identifiers.")
        if any(
            not math.isfinite(value) or value < 0.0 for value in self.pairwise_utility_contrast_half_widths.values()
        ):
            raise ValueError("pairwise utility contrast half-widths must be finite and non-negative.")
        if not set(self.reference_target_ids) <= set(self.credit_eligible_target_ids):
            raise ValueError("official observational targets must be acceptable.")
        return self


class TrialDevPublicEstimatorCandidateUtilityV1(BaseModel):
    """Candidate utility from one public-surface comparator estimator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_drug_id: str = Field(..., min_length=1)
    utility: float
    rank: int = Field(..., ge=1)


class TrialDevPublicEstimatorComparisonV1(BaseModel):
    """Comparator-estimator audit for one public objective."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective_id: TrialDevObjectiveIdV1
    estimator_id: Literal[
        "multinomial_propensity_weighted_stratified_aalen_johansen",
        "entropy_balanced_standardized_aalen_johansen",
        "raw_observed",
    ]
    status: Literal["estimated", "not_estimable"] = "estimated"
    failure_reason: (
        Literal[
            "empirical_positivity_violation",
            "empty_standardization_cell",
            "residual_unmeasured_confounding",
        ]
        | None
    ) = None
    reference_top_candidate_id: str | None = Field(default=None, min_length=1)
    top_candidate_id: str | None = Field(default=None, min_length=1)
    top_utility: float | None = None
    utility_margin_to_reference_top: float | None = Field(default=None, ge=0.0)
    agrees_with_reference_top: bool | None = None
    rank_order: tuple[str, ...] = Field(default_factory=tuple)
    candidate_utilities: tuple[TrialDevPublicEstimatorCandidateUtilityV1, ...] = Field(default_factory=tuple)
    policy_signal: Literal[
        "supports_unique_best",
        "supports_near_tie_or_acceptable_set",
        "conflicts_materially",
        "diagnostic_only",
        "not_estimable",
    ]

    @model_validator(mode="after")
    def validate_estimator_status(self) -> TrialDevPublicEstimatorComparisonV1:
        """Require complete results only when the estimator is estimable."""

        result_fields = (
            self.top_candidate_id,
            self.top_utility,
            self.utility_margin_to_reference_top,
            self.agrees_with_reference_top,
        )
        if self.status == "estimated":
            if (
                self.reference_top_candidate_id is None
                or self.failure_reason is not None
                or any(value is None for value in result_fields)
            ):
                raise ValueError("estimated comparator requires complete results and no failure reason.")
            if not self.rank_order or not self.candidate_utilities or self.policy_signal == "not_estimable":
                raise ValueError("estimated comparator requires ranked candidate results.")
        elif (
            self.failure_reason is None
            or self.reference_top_candidate_id is not None
            or any(value is not None for value in result_fields)
            or self.rank_order
            or self.candidate_utilities
            or self.policy_signal != "not_estimable"
        ):
            raise ValueError("non-estimable comparator may contain only its typed failure reason.")
        return self


class TrialDevPublicRecoverabilityPolicyV1(BaseModel):
    """Objective-level public-surface recoverability policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective_id: TrialDevObjectiveIdV1
    policy: TrialDevRecoverabilityPolicyV1
    reference_target_ids: tuple[str, ...] = Field(default_factory=tuple)
    acceptable_candidate_set: tuple[str, ...] = Field(default_factory=tuple)
    near_tie_threshold: float | None = Field(default=None, ge=0.0)
    indifference_sensitivity_sets: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    preference_sensitivity_sets: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    rationale: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_policy(self) -> TrialDevPublicRecoverabilityPolicyV1:
        """Validate target-set constraints."""

        if self.policy in {"unique_best", "near_tie_set", "acceptable_candidate_set"}:
            if not self.reference_target_ids:
                raise ValueError(f"{self.policy} requires reference_target_ids.")
            if not self.acceptable_candidate_set:
                raise ValueError(f"{self.policy} requires acceptable_candidate_set.")
            for label, sets in (
                ("indifference", self.indifference_sensitivity_sets),
                ("preference", self.preference_sensitivity_sets),
            ):
                if not sets or any(not candidates for candidates in sets.values()):
                    raise ValueError(f"{self.policy} requires non-empty {label} sensitivity sets.")
        if self.policy == "insufficient_recoverability" and (
            self.reference_target_ids
            or self.acceptable_candidate_set
            or self.near_tie_threshold is not None
            or self.indifference_sensitivity_sets
            or self.preference_sensitivity_sets
        ):
            raise ValueError("insufficient_recoverability cannot declare utility targets or thresholds.")
        return self


class TrialDevPublicRecoverabilityDiagnosticsV1(BaseModel):
    """Method-specific public-surface solver diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_route_id: str = Field(..., min_length=1)
    confounding_regime: Literal["measured_with_overlap", "residual_unmeasured"]
    estimator_id: Literal[
        "multinomial_propensity_weighted_stratified_aalen_johansen",
        "entropy_balanced_standardized_aalen_johansen",
    ]
    baseline_covariates: tuple[str, ...]
    stratum_count: int = Field(..., ge=1)
    candidate_count: int = Field(..., ge=1)
    source_row_count: int = Field(..., ge=1)
    analysis_row_count: int = Field(..., ge=1)
    excluded_missing_covariate_count: int = Field(..., ge=0)
    min_support_count: int = Field(..., ge=0)
    min_observed_treatment_probability: float | None = Field(default=None, gt=0.0, le=1.0)
    maximum_analysis_weight: float = Field(..., gt=0.0)
    min_effective_sample_size: float = Field(..., gt=0.0)
    maximum_calibration_mean_error: float = Field(..., ge=0.0)
    max_abs_unadjusted_smd_vs_target: float = Field(..., ge=0.0)
    max_abs_adjusted_smd_vs_target: float | None = Field(default=None, ge=0.0)
    solver_policy: Literal[
        "multinomial_propensity_weighted_stratified_aalen_johansen_v1",
        "entropy_balanced_standardized_aalen_johansen_v1",
    ]

    @model_validator(mode="after")
    def validate_diagnostics(self) -> TrialDevPublicRecoverabilityDiagnosticsV1:
        """Require complete accounting and method-compatible diagnostics."""

        if self.analysis_row_count + self.excluded_missing_covariate_count != self.source_row_count:
            raise ValueError("analysis and excluded row counts must reconcile to source_row_count.")
        is_propensity = self.estimator_id == "multinomial_propensity_weighted_stratified_aalen_johansen"
        should_report_probability = self.confounding_regime == "measured_with_overlap" and is_propensity
        if should_report_probability != (self.min_observed_treatment_probability is not None):
            raise ValueError(
                "Observed treatment probability is required only for an estimable measured-confounding propensity route."
            )
        expected_solver = {
            "multinomial_propensity_weighted_stratified_aalen_johansen": (
                "multinomial_propensity_weighted_stratified_aalen_johansen_v1"
            ),
            "entropy_balanced_standardized_aalen_johansen": ("entropy_balanced_standardized_aalen_johansen_v1"),
        }[self.estimator_id]
        if self.solver_policy != expected_solver:
            raise ValueError("recoverability diagnostics solver policy disagrees with estimator_id.")
        return self


class TrialDevPublicObservationalMethodResultV1(BaseModel):
    """Publicly reproducible reference result and policy for one observational method route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_route_id: str = Field(..., min_length=1)
    estimator_id: Literal[
        "multinomial_propensity_weighted_stratified_aalen_johansen",
        "entropy_balanced_standardized_aalen_johansen",
    ]
    candidate_scores: tuple[TrialDevPublicCandidateScoreV1, ...] = Field(..., min_length=1)
    estimator_comparisons: tuple[TrialDevPublicEstimatorComparisonV1, ...] = Field(..., min_length=1)
    objective_policies: tuple[TrialDevPublicRecoverabilityPolicyV1, ...] = Field(..., min_length=1)
    observational_action_policies: tuple[TrialDevPublicObservationalActionPolicyV1, ...] = Field(..., min_length=1)
    diagnostics: TrialDevPublicRecoverabilityDiagnosticsV1

    @model_validator(mode="after")
    def validate_method_result(self) -> TrialDevPublicObservationalMethodResultV1:
        """Validate one complete, internally coherent method-specific reference result."""

        if (
            self.diagnostics.method_route_id != self.method_route_id
            or self.diagnostics.estimator_id != self.estimator_id
        ):
            raise ValueError("method result identity disagrees with its diagnostics.")

        policy_objectives = {policy.objective_id for policy in self.objective_policies}
        action_objectives = {policy.objective_id for policy in self.observational_action_policies}
        if action_objectives != policy_objectives or len(action_objectives) != len(self.observational_action_policies):
            raise ValueError("observational action policies must uniquely cover every public objective.")
        score_objectives = {score.objective_id for score in self.candidate_scores}
        missing_scores = sorted(str(objective) for objective in policy_objectives - score_objectives)
        if missing_scores:
            raise ValueError(f"public recoverability policies lack candidate scores: {missing_scores!r}.")
        score_keys = {(score.objective_id, score.candidate_drug_id) for score in self.candidate_scores}
        if len(score_keys) != len(self.candidate_scores):
            raise ValueError("public candidate scores must be unique by objective and candidate.")
        candidate_universes = {
            frozenset(
                str(score.candidate_drug_id) for score in self.candidate_scores if score.objective_id == objective_id
            )
            for objective_id in policy_objectives
        }
        if len(candidate_universes) != 1 or len(next(iter(candidate_universes))) != self.diagnostics.candidate_count:
            raise ValueError("public candidate scores must cover one complete candidate universe per objective.")
        comparison_objectives = {comparison.objective_id for comparison in self.estimator_comparisons}
        missing_comparisons = sorted(str(objective) for objective in policy_objectives - comparison_objectives)
        if missing_comparisons:
            raise ValueError(f"public recoverability policies lack estimator comparisons: {missing_comparisons!r}.")
        for objective_id in policy_objectives:
            objective_scores = tuple(score for score in self.candidate_scores if score.objective_id == objective_id)
            utility_policy = next(policy for policy in self.objective_policies if policy.objective_id == objective_id)
            action_policy = next(
                policy for policy in self.observational_action_policies if policy.objective_id == objective_id
            )
            observed_estimators = {
                comparison.estimator_id
                for comparison in self.estimator_comparisons
                if comparison.objective_id == objective_id
            }
            missing_estimators = sorted({self.estimator_id, "raw_observed"} - observed_estimators)
            if missing_estimators:
                raise ValueError(
                    f"public recoverability objective {objective_id} lacks estimator comparisons: "
                    f"{missing_estimators!r}."
                )
            candidate_universe = next(iter(candidate_universes))
            for comparison in (
                row
                for row in self.estimator_comparisons
                if row.objective_id == objective_id and row.status == "estimated"
            ):
                if {str(row.candidate_drug_id) for row in comparison.candidate_utilities} != candidate_universe:
                    raise ValueError("estimated comparator must cover the complete candidate universe.")
            if not all(score.inference_estimable for score in objective_scores):
                insufficient = ("withhold_nomination",)
                if (
                    action_policy.reference_target_ids != insufficient
                    or action_policy.credit_eligible_target_ids != insufficient
                ):
                    raise ValueError("Non-estimable objectives require qualified non-nomination.")
                continue
            threshold = float(action_policy.minimum_efficacy_gain)
            definite = {
                str(score.candidate_drug_id)
                for score in objective_scores
                if cast(float, score.efficacy_gain_ci_low) >= threshold
            }
            possible = {
                str(score.candidate_drug_id)
                for score in objective_scores
                if cast(float, score.efficacy_gain_ci_high) >= threshold
            }
            if definite != set(action_policy.definitely_qualified_candidate_ids) or possible != set(
                action_policy.possibly_qualified_candidate_ids
            ):
                raise ValueError("observational action qualification sets drift from candidate intervals.")
            stop_target = "withhold_nomination"
            if utility_policy.near_tie_threshold is None:
                raise ValueError("scoreable utility policies require a declared indifference margin.")
            utilities = {
                str(score.candidate_drug_id): cast(float, score.adjusted_utility) for score in objective_scores
            }
            best_eligible_utility = max((utilities[candidate] for candidate in possible), default=None)
            expected_acceptable = {
                candidate
                for candidate in possible
                if cast(float, best_eligible_utility) - utilities[candidate]
                <= max(
                    utility_policy.near_tie_threshold,
                    action_policy.utility_contrast_half_widths[candidate],
                )
            }
            if not definite:
                expected_acceptable.add(stop_target)
            if set(action_policy.credit_eligible_target_ids) != expected_acceptable:
                raise ValueError("observational acceptable actions drift from efficacy and utility policy.")
            expected_reference = stop_target
            if definite:
                expected_reference = min(
                    expected_acceptable,
                    key=lambda candidate: (-utilities[candidate], candidate),
                )
            if action_policy.reference_target_ids != (expected_reference,):
                raise ValueError("observational reference action drifts from qualification and utility policy.")
        return self


class TrialDevPublicRecoverabilityReportV1(BaseModel):
    """Public-surface recoverability proof report for one TrialDev scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_public_recoverability_report_v1"] = "trialdev_public_recoverability_report_v1"
    version: Literal["v1"] = "v1"
    scenario_id: str = Field(..., min_length=1)
    solver_version: Literal["trialdev_public_multi_method_recoverability_v1"] = (
        "trialdev_public_multi_method_recoverability_v1"
    )
    public_input_checksums: tuple[TrialDevPublicInputChecksumV1, ...] = Field(..., min_length=1)
    method_results: tuple[TrialDevPublicObservationalMethodResultV1, ...] = Field(..., min_length=2)
    method_union_objective_sensitivity: tuple[TrialDevPublicRecoverabilityPolicyV1, ...] = Field(
        ...,
        min_length=1,
    )
    method_union_action_sensitivity: tuple[TrialDevPublicObservationalActionPolicyV1, ...] = Field(
        ...,
        min_length=1,
    )
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_report(self) -> TrialDevPublicRecoverabilityReportV1:
        """Validate method coverage, sensitivity scope, and checksum."""

        method_ids = tuple(result.method_route_id for result in self.method_results)
        estimator_ids = tuple(result.estimator_id for result in self.method_results)
        if len(method_ids) != len(set(method_ids)) or len(estimator_ids) != len(set(estimator_ids)):
            raise ValueError("public recoverability requires unique method routes and estimators.")
        scenario_ids = {score.scenario_id for result in self.method_results for score in result.candidate_scores}
        if scenario_ids != {self.scenario_id}:
            raise ValueError("method-specific candidate scores disagree with report scenario_id.")
        method_objectives = {
            frozenset(policy.objective_id for policy in result.objective_policies) for result in self.method_results
        }
        if len(method_objectives) != 1:
            raise ValueError("all public observational methods must cover the same objective set.")
        objective_ids = next(iter(method_objectives))
        sensitivity_objectives = {policy.objective_id for policy in self.method_union_objective_sensitivity}
        sensitivity_actions = {policy.objective_id for policy in self.method_union_action_sensitivity}
        if (
            sensitivity_objectives != objective_ids
            or sensitivity_actions != objective_ids
            or len(sensitivity_objectives) != len(self.method_union_objective_sensitivity)
            or len(sensitivity_actions) != len(self.method_union_action_sensitivity)
        ):
            raise ValueError("cross-method sensitivity summaries must cover every objective exactly once.")
        for objective_id in objective_ids:
            method_policies = tuple(
                next(policy for policy in result.objective_policies if policy.objective_id == objective_id)
                for result in self.method_results
            )
            estimable_policies = tuple(
                policy for policy in method_policies if policy.policy != "insufficient_recoverability"
            )
            sensitivity_policy = next(
                policy for policy in self.method_union_objective_sensitivity if policy.objective_id == objective_id
            )
            if not estimable_policies:
                if sensitivity_policy.policy != "insufficient_recoverability":
                    raise ValueError("method-union utility sensitivity must report insufficient recoverability.")
            else:
                expected_reference = {
                    target for policy in estimable_policies for target in policy.reference_target_ids
                }
                expected_acceptable = {
                    target for policy in estimable_policies for target in policy.acceptable_candidate_set
                }
                expected_kind = "unique_best" if len(expected_acceptable) == 1 else "acceptable_candidate_set"
                expected_threshold = max(cast(float, policy.near_tie_threshold) for policy in estimable_policies)
                if (
                    sensitivity_policy.policy != expected_kind
                    or set(sensitivity_policy.reference_target_ids) != expected_reference
                    or set(sensitivity_policy.acceptable_candidate_set) != expected_acceptable
                    or not (
                        sensitivity_policy.near_tie_threshold is not None
                        and abs(sensitivity_policy.near_tie_threshold - expected_threshold) <= 1e-12
                    )
                ):
                    raise ValueError("method-union utility sensitivity drifts from method-specific policies.")
                for attribute in ("indifference_sensitivity_sets", "preference_sensitivity_sets"):
                    method_sets = tuple(getattr(policy, attribute) for policy in estimable_policies)
                    shared_keys = set.intersection(*(set(values) for values in method_sets))
                    expected_sets = {
                        key: tuple(sorted({candidate for values in method_sets for candidate in values[key]}))
                        for key in sorted(shared_keys)
                    }
                    if getattr(sensitivity_policy, attribute) != expected_sets:
                        raise ValueError("method-union utility sensitivity sets drift from method-specific policies.")

            method_actions = tuple(
                next(policy for policy in result.observational_action_policies if policy.objective_id == objective_id)
                for result in self.method_results
            )
            estimable_actions = tuple(
                action
                for result, action in zip(self.method_results, method_actions, strict=True)
                if next(policy for policy in result.objective_policies if policy.objective_id == objective_id).policy
                != "insufficient_recoverability"
            )
            sensitivity_action = next(
                policy for policy in self.method_union_action_sensitivity if policy.objective_id == objective_id
            )
            if not estimable_actions:
                insufficient = ("withhold_nomination",)
                if (
                    sensitivity_action.reference_target_ids != insufficient
                    or sensitivity_action.credit_eligible_target_ids != insufficient
                ):
                    raise ValueError("method-union action sensitivity must report insufficient recoverability.")
            else:
                thresholds = {float(policy.minimum_efficacy_gain) for policy in estimable_actions}
                if len(thresholds) != 1:
                    raise ValueError("method-specific actions disagree on the minimum efficacy threshold.")
                expected_reference = {target for policy in estimable_actions for target in policy.reference_target_ids}
                expected_acceptable = {
                    target for policy in estimable_actions for target in policy.credit_eligible_target_ids
                }
                expected_definite = set.intersection(
                    *(set(policy.definitely_qualified_candidate_ids) for policy in estimable_actions)
                )
                expected_possible = set().union(
                    *(set(policy.possibly_qualified_candidate_ids) for policy in estimable_actions)
                )
                if (
                    abs(sensitivity_action.minimum_efficacy_gain - next(iter(thresholds))) > 1e-12
                    or set(sensitivity_action.reference_target_ids) != expected_reference
                    or set(sensitivity_action.credit_eligible_target_ids) != expected_acceptable
                    or set(sensitivity_action.definitely_qualified_candidate_ids) != expected_definite
                    or set(sensitivity_action.possibly_qualified_candidate_ids) != expected_possible
                ):
                    raise ValueError("method-union action sensitivity drifts from method-specific policies.")
        if self.checksum is None:
            object.__setattr__(self, "checksum", _stable_checksum(self.model_dump(mode="json", exclude={"checksum"})))
        return self


def load_trialdev_public_recoverability_report(path: Path) -> TrialDevPublicRecoverabilityReportV1:
    """Load and validate a public recoverability report."""

    return cast(
        TrialDevPublicRecoverabilityReportV1,
        read_json_model(TrialDevPublicRecoverabilityReportV1, Path(path)),
    )


__all__ = [
    "WITHHOLD_NOMINATION_TARGET_ID",
    "TrialDevPublicCandidateScoreV1",
    "TrialDevPublicEntropyBalancedAnalysisSpecV1",
    "TrialDevPublicEstimatorCandidateUtilityV1",
    "TrialDevPublicEstimatorComparisonV1",
    "TrialDevPublicInputChecksumV1",
    "TrialDevPublicObjectiveAnalysisManifestV1",
    "TrialDevPublicObjectiveSpecV1",
    "TrialDevPublicObservationalActionPolicyV1",
    "TrialDevPublicObservationalAnalysisSpecV1",
    "TrialDevPublicObservationalMethodResultV1",
    "TrialDevPublicObservationalMethodSpecV1",
    "TrialDevPublicRecoverabilityDiagnosticsV1",
    "TrialDevPublicRecoverabilityPolicyV1",
    "TrialDevPublicRecoverabilityReportV1",
    "TrialDevPublicUtilityComponentV1",
    "TrialDevPublicUtilityEventV1",
    "load_trialdev_public_recoverability_report",
]
