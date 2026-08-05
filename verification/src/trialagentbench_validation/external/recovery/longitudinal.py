"""Independent verification of source-fitted longitudinal trial worlds."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import t as student_t

from trialagentbench_validation.external.realism.longitudinal import (
    LongitudinalTrialFingerprintV1,
    fingerprint_longitudinal_trial,
)
from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.statistics import proportion_interval

LongitudinalMode = Literal[
    "whole_subject",
    "source_anchored",
    "linkage_75",
    "linkage_50",
    "linkage_25",
    "independent_marginal",
]

_LINKAGE_RETENTION: dict[LongitudinalMode, float] = {
    "independent_marginal": 0.0,
    "linkage_25": 0.25,
    "linkage_50": 0.5,
    "linkage_75": 0.75,
    "source_anchored": 1.0,
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LongitudinalArmEffectV1(_FrozenModel):
    """One arm contrast in a fitted longitudinal model."""

    arm_id: str = Field(min_length=1)
    visit_shifts: tuple[float, ...] = Field(min_length=3)


class LongitudinalFittedModelV1(_FrozenModel):
    """Source-fitted fixed-schedule Gaussian repeated-measures model."""

    control_mean_values: tuple[float, ...] = Field(min_length=3)
    source_covariance: tuple[tuple[float, ...], ...] = Field(min_length=3)
    latent_correlation: tuple[tuple[float, ...], ...] = Field(min_length=3)
    marginal_probabilities: tuple[float, ...] = Field(min_length=5)
    marginal_residual_values: tuple[tuple[float, ...], ...] = Field(min_length=3)
    arm_effects: tuple[LongitudinalArmEffectV1, ...]
    dropout_logit_intercepts: tuple[float, ...] | None = Field(
        default=None, min_length=1
    )
    dropout_treatment_arm: str | None = Field(default=None, min_length=1)
    dropout_treatment_coefficient: float | None = Field(
        default=None, allow_inf_nan=False
    )
    measurement_probabilities: tuple[float, ...] = Field(min_length=3)


class LongitudinalQualificationTrialV1(_FrozenModel):
    """Public design for one source-fitted longitudinal trial."""

    trial_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participants: int = Field(ge=20)
    worlds: int = Field(ge=2)
    time_values: tuple[float, ...] = Field(min_length=3)
    arm_ids: tuple[str, ...] = Field(min_length=2)
    control_arm_id: str = Field(min_length=1)
    fitted_model: LongitudinalFittedModelV1
    source_fingerprint: LongitudinalTrialFingerprintV1

    @model_validator(mode="after")
    def _complete(self) -> LongitudinalQualificationTrialV1:
        if (
            len(set(self.arm_ids)) != len(self.arm_ids)
            or self.control_arm_id not in self.arm_ids
        ):
            raise ValueError(
                "Longitudinal qualification arms must be unique and include control."
            )
        effect_arms = {effect.arm_id for effect in self.fitted_model.arm_effects}
        if effect_arms != set(self.arm_ids) - {self.control_arm_id}:
            raise ValueError(
                "Longitudinal fitted arm effects must cover every non-control arm."
            )
        if tuple(sorted(self.time_values)) != self.time_values or len(
            set(self.time_values)
        ) != len(self.time_values):
            raise ValueError(
                "Longitudinal qualification times must be sorted and unique."
            )
        visits = len(self.time_values)
        if len(self.fitted_model.control_mean_values) != visits:
            raise ValueError(
                "Longitudinal fitted control means must match the visit schedule."
            )
        if (
            any(len(row) != visits for row in self.fitted_model.source_covariance)
            or len(self.fitted_model.source_covariance) != visits
        ):
            raise ValueError(
                "Longitudinal fitted source covariance must be square by visit."
            )
        if (
            any(len(row) != visits for row in self.fitted_model.latent_correlation)
            or len(self.fitted_model.latent_correlation) != visits
        ):
            raise ValueError(
                "Longitudinal fitted latent correlation must be square by visit."
            )
        quantiles = len(self.fitted_model.marginal_probabilities)
        if len(self.fitted_model.marginal_residual_values) != visits or any(
            len(row) != quantiles for row in self.fitted_model.marginal_residual_values
        ):
            raise ValueError(
                "Longitudinal fitted residual marginals must be rectangular by visit."
            )
        if any(
            len(effect.visit_shifts) != visits
            for effect in self.fitted_model.arm_effects
        ):
            raise ValueError(
                "Longitudinal fitted arm shifts must match the visit schedule."
            )
        dropout_intercepts = self.fitted_model.dropout_logit_intercepts
        if dropout_intercepts is not None and len(dropout_intercepts) != visits - 1:
            raise ValueError(
                "Longitudinal dropout intercepts must cover every post-baseline visit."
            )
        treatment_arm = self.fitted_model.dropout_treatment_arm
        treatment_coefficient = self.fitted_model.dropout_treatment_coefficient
        if (treatment_arm is None) != (treatment_coefficient is None):
            raise ValueError(
                "Longitudinal dropout treatment arm and coefficient must be declared together."
            )
        if treatment_arm is not None and treatment_arm not in effect_arms:
            raise ValueError(
                "Longitudinal dropout treatment arm must be a declared non-control arm."
            )
        if len(self.fitted_model.measurement_probabilities) != visits or any(
            not 0.0 < probability <= 1.0
            for probability in self.fitted_model.measurement_probabilities
        ):
            raise ValueError(
                "Longitudinal measurement probabilities must cover visits and lie in (0, 1]."
            )
        if self.source_fingerprint.trial_id != self.trial_id:
            raise ValueError(
                "Longitudinal source fingerprint has the wrong trial identity."
            )
        return self


class LongitudinalQualificationDesignV1(_FrozenModel):
    """Path-free longitudinal qualification design."""

    schema_id: Literal["trialagentbench.longitudinal_qualification_design/v1"]
    seed: int = Field(ge=0, le=2**32 - 1)
    trials: tuple[LongitudinalQualificationTrialV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_trials(self) -> LongitudinalQualificationDesignV1:
        ids = [trial.trial_id for trial in self.trials]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Longitudinal qualification trial identities must be unique."
            )
        return self


class LongitudinalWorldReceiptV1(_FrozenModel):
    """Checksum-bound identity of one released longitudinal world."""

    world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    trial_id: str = Field(min_length=1)
    world_index: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    analysis_path: str = Field(pattern=r"^worlds/world_[0-9a-f]{20}\.parquet$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ground_truth_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LongitudinalQualificationReceiptV1(_FrozenModel):
    """Complete inventory of released longitudinal worlds."""

    schema_id: Literal["trialagentbench.longitudinal_qualification_receipt/v1"]
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[LongitudinalWorldReceiptV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_worlds(self) -> LongitudinalQualificationReceiptV1:
        identities = [(row.trial_id, row.world_index) for row in self.worlds]
        if len(identities) != len(set(identities)):
            raise ValueError("Longitudinal world identities must be unique.")
        return self


class LongitudinalWorldEstimateV1(_FrozenModel):
    """One independently estimated arm-specific trajectory."""

    world_id: str
    trial_id: str
    world_index: int
    mode: LongitudinalMode
    arm_id: str
    intercept: float = Field(allow_inf_nan=False)
    intercept_standard_error: float = Field(gt=0, allow_inf_nan=False)
    slope_per_day: float = Field(allow_inf_nan=False)
    slope_standard_error: float = Field(gt=0, allow_inf_nan=False)
    treatment_intercept_contrast: float | None = Field(
        default=None, allow_inf_nan=False
    )
    treatment_intercept_standard_error: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    treatment_slope_contrast: float | None = Field(default=None, allow_inf_nan=False)
    treatment_slope_standard_error: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    fingerprint: LongitudinalTrialFingerprintV1

    @model_validator(mode="after")
    def _complete_treatment_contrast(self) -> LongitudinalWorldEstimateV1:
        values = (
            self.treatment_intercept_contrast,
            self.treatment_intercept_standard_error,
            self.treatment_slope_contrast,
            self.treatment_slope_standard_error,
        )
        if any(value is None for value in values) and not all(
            value is None for value in values
        ):
            raise ValueError(
                "Longitudinal treatment contrast fields must be all present or all absent."
            )
        return self


class LongitudinalModeSummaryV1(_FrozenModel):
    """Across-world fidelity of one longitudinal generation mode."""

    trial_id: str
    mode: LongitudinalMode
    linkage_retention: float | None = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )
    worlds: int = Field(ge=2)
    failures: int = Field(ge=0)
    fingerprint_standardized_error_mean: float = Field(ge=0, allow_inf_nan=False)
    marginal_standardized_error_mean: float = Field(ge=0, allow_inf_nan=False)
    marginal_standardized_error_ci_low: float = Field(ge=0, allow_inf_nan=False)
    marginal_standardized_error_ci_high: float = Field(ge=0, allow_inf_nan=False)
    linkage_absolute_error_mean: float = Field(ge=0, le=2, allow_inf_nan=False)
    linkage_absolute_error_ci_low: float = Field(ge=0, le=2, allow_inf_nan=False)
    linkage_absolute_error_ci_high: float = Field(ge=0, le=2, allow_inf_nan=False)
    adjacent_correlation_mean: float = Field(ge=-1, le=1, allow_inf_nan=False)
    adjacent_correlation_ci_low: float = Field(ge=-1, le=1, allow_inf_nan=False)
    adjacent_correlation_ci_high: float = Field(ge=-1, le=1, allow_inf_nan=False)
    within_stratum_adjacent_correlation_mean: float = Field(
        ge=-1, le=1, allow_inf_nan=False
    )
    within_stratum_adjacent_correlation_ci_low: float = Field(
        ge=-1, le=1, allow_inf_nan=False
    )
    within_stratum_adjacent_correlation_ci_high: float = Field(
        ge=-1, le=1, allow_inf_nan=False
    )
    baseline_final_correlation_mean: float = Field(ge=-1, le=1, allow_inf_nan=False)
    within_stratum_baseline_final_correlation_mean: float = Field(
        ge=-1, le=1, allow_inf_nan=False
    )
    baseline_final_change_mean: float = Field(allow_inf_nan=False)


class LongitudinalArmRecoveryV1(_FrozenModel):
    """Known-truth recovery for one source-anchored arm trajectory."""

    trial_id: str
    arm_id: str
    worlds: int = Field(ge=2)
    truth_intercept: float = Field(allow_inf_nan=False)
    mean_intercept: float = Field(allow_inf_nan=False)
    intercept_bias: float = Field(allow_inf_nan=False)
    intercept_bias_ci_low: float = Field(allow_inf_nan=False)
    intercept_bias_ci_high: float = Field(allow_inf_nan=False)
    intercept_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    intercept_coverage_ci_low: float = Field(ge=0, le=1, allow_inf_nan=False)
    intercept_coverage_ci_high: float = Field(ge=0, le=1, allow_inf_nan=False)
    truth_slope_per_day: float = Field(allow_inf_nan=False)
    mean_slope_per_day: float = Field(allow_inf_nan=False)
    slope_bias: float = Field(allow_inf_nan=False)
    slope_bias_ci_low: float = Field(allow_inf_nan=False)
    slope_bias_ci_high: float = Field(allow_inf_nan=False)
    slope_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    slope_coverage_ci_low: float = Field(ge=0, le=1, allow_inf_nan=False)
    slope_coverage_ci_high: float = Field(ge=0, le=1, allow_inf_nan=False)


class LongitudinalTreatmentRecoveryV1(_FrozenModel):
    """Known-truth recovery for one randomized longitudinal arm contrast."""

    trial_id: str
    arm_id: str
    worlds: int = Field(ge=2)
    truth_intercept_contrast: float = Field(allow_inf_nan=False)
    mean_intercept_contrast: float = Field(allow_inf_nan=False)
    intercept_contrast_bias: float = Field(allow_inf_nan=False)
    intercept_contrast_bias_ci_low: float = Field(allow_inf_nan=False)
    intercept_contrast_bias_ci_high: float = Field(allow_inf_nan=False)
    intercept_contrast_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    intercept_contrast_coverage_ci_low: float = Field(ge=0, le=1, allow_inf_nan=False)
    intercept_contrast_coverage_ci_high: float = Field(ge=0, le=1, allow_inf_nan=False)
    truth_slope_contrast: float = Field(allow_inf_nan=False)
    mean_slope_contrast: float = Field(allow_inf_nan=False)
    slope_contrast_bias: float = Field(allow_inf_nan=False)
    slope_contrast_bias_ci_low: float = Field(allow_inf_nan=False)
    slope_contrast_bias_ci_high: float = Field(allow_inf_nan=False)
    slope_contrast_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    slope_contrast_coverage_ci_low: float = Field(ge=0, le=1, allow_inf_nan=False)
    slope_contrast_coverage_ci_high: float = Field(ge=0, le=1, allow_inf_nan=False)


class LongitudinalLinkageDoseResponseV1(_FrozenModel):
    """Across-world response of within-person correlation to retained linkage."""

    trial_id: str
    worlds: int = Field(ge=2)
    retention_levels: tuple[float, ...] = Field(min_length=5)
    mean_correlations: tuple[float, ...] = Field(min_length=5)
    mean_slope: float = Field(allow_inf_nan=False)
    slope_ci_low: float = Field(allow_inf_nan=False)
    slope_ci_high: float = Field(allow_inf_nan=False)
    positive_slope_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    positive_slope_fraction_ci_low: float = Field(ge=0, le=1, allow_inf_nan=False)
    positive_slope_fraction_ci_high: float = Field(ge=0, le=1, allow_inf_nan=False)
    endpoint_contrast_mean: float = Field(allow_inf_nan=False)
    endpoint_contrast_ci_low: float = Field(allow_inf_nan=False)
    endpoint_contrast_ci_high: float = Field(allow_inf_nan=False)


class LongitudinalMarginalPredictiveV1(_FrozenModel):
    """Native-scale predictive distribution for one arm and visit."""

    trial_id: str = Field(min_length=1)
    mode: LongitudinalMode
    arm_id: str = Field(min_length=1)
    time: float = Field(allow_inf_nan=False)
    worlds: int = Field(ge=2)
    source_observations: int = Field(ge=3)
    source_mean: float = Field(allow_inf_nan=False)
    source_standard_deviation: float = Field(gt=0, allow_inf_nan=False)
    mean_median: float = Field(allow_inf_nan=False)
    mean_interval_50_low: float = Field(allow_inf_nan=False)
    mean_interval_50_high: float = Field(allow_inf_nan=False)
    mean_interval_95_low: float = Field(allow_inf_nan=False)
    mean_interval_95_high: float = Field(allow_inf_nan=False)
    standard_deviation_median: float = Field(gt=0, allow_inf_nan=False)
    standard_deviation_interval_95_low: float = Field(gt=0, allow_inf_nan=False)
    standard_deviation_interval_95_high: float = Field(gt=0, allow_inf_nan=False)
    observations_median: float = Field(ge=0, allow_inf_nan=False)
    observations_interval_95_low: float = Field(ge=0, allow_inf_nan=False)
    observations_interval_95_high: float = Field(ge=0, allow_inf_nan=False)
    source_predictive_rank: float = Field(ge=0, le=1, allow_inf_nan=False)


class LongitudinalQualificationReportV1(_FrozenModel):
    """Independent longitudinal fidelity and recovery report."""

    schema_id: Literal["trialagentbench.longitudinal_qualification_report/v1"] = (
        "trialagentbench.longitudinal_qualification_report/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimates: tuple[LongitudinalWorldEstimateV1, ...]
    modes: tuple[LongitudinalModeSummaryV1, ...]
    source_anchored_recovery: tuple[LongitudinalArmRecoveryV1, ...]
    treatment_recovery: tuple[LongitudinalTreatmentRecoveryV1, ...]
    linkage_dose_response: tuple[LongitudinalLinkageDoseResponseV1, ...]
    marginal_predictive: tuple[LongitudinalMarginalPredictiveV1, ...] = ()


def evaluate_longitudinal_qualification(
    *,
    release_dir: Path,
    minimum_worlds_per_trial: int = 100,
) -> LongitudinalQualificationReportV1:
    """Verify and independently analyze a longitudinal qualification release."""

    if minimum_worlds_per_trial < 2:
        raise ValueError("minimum_worlds_per_trial must be at least two.")
    design_path = release_dir / "design.json"
    receipt_path = release_dir / "receipt.json"
    design_payload = json.loads(design_path.read_text(encoding="utf-8"))
    design = LongitudinalQualificationDesignV1.model_validate(design_payload)
    design_sha256 = _json_sha(design_payload)
    receipt = LongitudinalQualificationReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    if receipt.design_sha256 != design_sha256:
        raise ValueError("Longitudinal receipt does not match its design.")
    trials = {trial.trial_id: trial for trial in design.trials}
    expected = {
        (trial.trial_id, world_index)
        for trial in design.trials
        for world_index in range(trial.worlds)
    }
    observed = {(world.trial_id, world.world_index) for world in receipt.worlds}
    if observed != expected:
        raise ValueError("Longitudinal receipt does not contain the complete design.")
    if short := {
        trial.trial_id: trial.worlds
        for trial in design.trials
        if trial.worlds < minimum_worlds_per_trial
    }:
        raise ValueError(
            f"Longitudinal trials do not meet the replication floor: {short!r}."
        )

    estimates = []
    for world in receipt.worlds:
        trial = trials.get(world.trial_id)
        if trial is None:
            raise ValueError(
                f"Unknown longitudinal trial in receipt: {world.trial_id!r}."
            )
        expected_seed = _world_seed(design.seed, world.trial_id, world.world_index)
        expected_id = _world_id(design_sha256, world.trial_id, world.world_index)
        if world.seed != expected_seed or world.world_id != expected_id:
            raise ValueError(f"Longitudinal world identity mismatch: {world.world_id}.")
        path = _release_path(release_dir, world.analysis_path)
        if sha256_file(path) != world.analysis_sha256:
            raise ValueError(f"Longitudinal world checksum mismatch: {world.world_id}.")
        frame = _validate_world(pd.read_parquet(path), world=world, trial=trial)
        for mode in cast(
            tuple[LongitudinalMode, ...], tuple(sorted(frame["mode"].unique()))
        ):
            mode_frame = frame.loc[frame["mode"].eq(mode)].copy()
            fingerprint = fingerprint_longitudinal_trial(
                mode_frame,
                trial_id=trial.trial_id,
                source=mode,
                measurement=trial.source_fingerprint.measurement,
                measurement_unit=trial.source_fingerprint.measurement_unit,
                time_unit=trial.source_fingerprint.time_unit,
            )
            estimates.extend(
                _fit_arm_trajectories(
                    mode_frame,
                    world=world,
                    mode=mode,
                    trial=trial,
                    fingerprint=fingerprint,
                )
            )
    mode_summaries = _summarize_modes(estimates, trials)
    recovery = _summarize_source_anchored_recovery(estimates, trials)
    treatment_recovery = _summarize_treatment_recovery(estimates, trials)
    dose_response = _summarize_linkage_dose_response(estimates, trials)
    marginal_predictive = _summarize_marginal_predictive(estimates, trials)
    return LongitudinalQualificationReportV1(
        design_sha256=design_sha256,
        receipt_sha256=sha256_file(receipt_path),
        estimates=tuple(estimates),
        modes=tuple(mode_summaries),
        source_anchored_recovery=tuple(recovery),
        treatment_recovery=tuple(treatment_recovery),
        linkage_dose_response=tuple(dose_response),
        marginal_predictive=tuple(marginal_predictive),
    )


def _summarize_marginal_predictive(
    estimates: list[LongitudinalWorldEstimateV1],
    trials: dict[str, LongitudinalQualificationTrialV1],
) -> list[LongitudinalMarginalPredictiveV1]:
    fingerprints = {
        (row.trial_id, row.world_index, row.mode): row.fingerprint for row in estimates
    }
    grouped: dict[
        tuple[str, LongitudinalMode, str, float],
        list[tuple[int, float, float]],
    ] = defaultdict(list)
    for (trial_id, _, mode), fingerprint in fingerprints.items():
        for moment in fingerprint.marginal_moments:
            grouped[(trial_id, mode, moment.arm_id, moment.time)].append(
                (moment.observations, moment.mean, moment.standard_deviation)
            )
    source = {
        (trial_id, moment.arm_id, moment.time): moment
        for trial_id, trial in trials.items()
        for moment in trial.source_fingerprint.marginal_moments
    }

    output = []
    for (trial_id, mode, arm_id, time), rows in sorted(grouped.items()):
        reference = source[(trial_id, arm_id, time)]
        values = np.asarray(rows, dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(
                "Longitudinal marginal prediction requires two finite worlds."
            )
        means = values[:, 1]
        output.append(
            LongitudinalMarginalPredictiveV1(
                trial_id=trial_id,
                mode=mode,
                arm_id=arm_id,
                time=time,
                worlds=len(values),
                source_observations=reference.observations,
                source_mean=reference.mean,
                source_standard_deviation=reference.standard_deviation,
                mean_median=float(np.median(means)),
                mean_interval_50_low=float(np.quantile(means, 0.25)),
                mean_interval_50_high=float(np.quantile(means, 0.75)),
                mean_interval_95_low=float(np.quantile(means, 0.025)),
                mean_interval_95_high=float(np.quantile(means, 0.975)),
                standard_deviation_median=float(np.median(values[:, 2])),
                standard_deviation_interval_95_low=float(
                    np.quantile(values[:, 2], 0.025)
                ),
                standard_deviation_interval_95_high=float(
                    np.quantile(values[:, 2], 0.975)
                ),
                observations_median=float(np.median(values[:, 0])),
                observations_interval_95_low=float(np.quantile(values[:, 0], 0.025)),
                observations_interval_95_high=float(np.quantile(values[:, 0], 0.975)),
                source_predictive_rank=float(
                    (
                        np.count_nonzero(means < reference.mean)
                        + 0.5 * np.count_nonzero(means == reference.mean)
                    )
                    / len(means)
                ),
            )
        )
    return output


def _validate_world(
    frame: pd.DataFrame,
    *,
    world: LongitudinalWorldReceiptV1,
    trial: LongitudinalQualificationTrialV1,
) -> pd.DataFrame:
    required = {
        "world_id",
        "trial_id",
        "mode",
        "participant_id",
        "arm",
        "time",
        "value",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Longitudinal world is missing columns: {missing!r}.")
    values = frame.loc[:, sorted(required)].copy()
    if set(values["world_id"].astype("string")) != {world.world_id}:
        raise ValueError("Longitudinal world rows have inconsistent world identity.")
    if set(values["trial_id"].astype("string")) != {world.trial_id}:
        raise ValueError("Longitudinal world rows have inconsistent trial identity.")
    expected_modes = {
        "whole_subject",
        "source_anchored",
        "linkage_75",
        "linkage_50",
        "linkage_25",
        "independent_marginal",
    }
    if set(values["mode"].astype("string")) != expected_modes:
        raise ValueError(
            "Longitudinal world must contain the complete comparison modes."
        )
    values["time"] = pd.to_numeric(values["time"], errors="raise").astype("float64")
    values["value"] = pd.to_numeric(values["value"], errors="raise").astype("float64")
    if not np.isfinite(values["time"]).all() or np.isinf(values["value"]).any():
        raise ValueError(
            "Longitudinal world times must be finite and measurements cannot be infinite."
        )
    if set(values["time"]) != set(trial.time_values) or not set(
        values["arm"].astype("string")
    ).issubset(set(trial.arm_ids)):
        raise ValueError("Longitudinal world time or arm support differs from design.")
    for mode, group in values.groupby("mode", sort=False):
        participants = group["participant_id"].astype("string").nunique()
        if participants != trial.participants:
            raise ValueError(
                f"Longitudinal mode {mode!r} has {participants} participants; expected {trial.participants}."
            )
        if group.duplicated(["participant_id", "time"]).any():
            raise ValueError(
                f"Longitudinal mode {mode!r} contains duplicate participant-time rows."
            )
        if len(group) != trial.participants * len(trial.time_values):
            raise ValueError(
                f"Longitudinal mode {mode!r} must contain the complete planned visit grid."
            )
    return values


def _fit_arm_trajectories(
    frame: pd.DataFrame,
    *,
    world: LongitudinalWorldReceiptV1,
    mode: LongitudinalMode,
    trial: LongitudinalQualificationTrialV1,
    fingerprint: LongitudinalTrialFingerprintV1,
) -> list[LongitudinalWorldEstimateV1]:
    frame = frame.dropna(subset=["value"]).copy()
    arm_ids = list(trial.arm_ids)
    control = trial.control_arm_id
    time_values = np.asarray(trial.time_values, dtype=float)
    design = pd.DataFrame({"intercept": np.ones(len(frame))})
    for time in time_values[1:]:
        design[f"visit:{time}"] = frame["time"].eq(time).to_numpy(dtype=float)
    for arm in arm_ids:
        if arm == control:
            continue
        indicator = frame["arm"].astype("string").eq(arm).to_numpy(dtype=float)
        design[f"arm:{arm}"] = indicator
        for time in time_values[1:]:
            design[f"arm_visit:{arm}:{time}"] = indicator * design[f"visit:{time}"]
    fit = sm.GEE(
        frame["value"].to_numpy(dtype=float),
        design,
        groups=frame["participant_id"].astype("string").to_numpy(),
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit(cov_type="bias_reduced")
    covariance = np.asarray(fit.cov_params(), dtype=float)
    parameters = np.asarray(fit.params, dtype=float)
    if not np.isfinite(parameters).all() or not np.isfinite(covariance).all():
        raise ValueError(
            f"Non-finite longitudinal estimate for {world.world_id} in mode {mode!r}."
        )
    names = list(design.columns)
    projection = np.linalg.pinv(
        np.column_stack([np.ones(len(time_values)), time_values])
    )

    def mean_vectors(arm: str) -> npt.NDArray[np.float64]:
        vectors = np.zeros((len(time_values), len(names)), dtype=float)
        vectors[:, names.index("intercept")] = 1.0
        for time_index, time in enumerate(time_values[1:], start=1):
            vectors[time_index, names.index(f"visit:{time}")] = 1.0
        if arm != control:
            vectors[:, names.index(f"arm:{arm}")] = 1.0
            for time_index, time in enumerate(time_values[1:], start=1):
                vectors[time_index, names.index(f"arm_visit:{arm}:{time}")] = 1.0
        return vectors

    control_vectors = mean_vectors(control)
    output = []
    for arm in arm_ids:
        arm_vectors = mean_vectors(arm)
        intercept_vector = projection[0] @ arm_vectors
        slope_vector = projection[1] @ arm_vectors
        intercept_variance = float(intercept_vector @ covariance @ intercept_vector)
        slope_variance = float(slope_vector @ covariance @ slope_vector)
        if intercept_variance < 0.0 or slope_variance < 0.0:
            raise ValueError(
                f"Negative longitudinal contrast variance for {world.world_id} in arm {arm!r}."
            )
        treatment_intercept_vector = projection[0] @ (arm_vectors - control_vectors)
        treatment_slope_vector = projection[1] @ (arm_vectors - control_vectors)
        treatment_intercept_variance = float(
            treatment_intercept_vector @ covariance @ treatment_intercept_vector
        )
        treatment_slope_variance = float(
            treatment_slope_vector @ covariance @ treatment_slope_vector
        )
        if treatment_intercept_variance < 0.0 or treatment_slope_variance < 0.0:
            raise ValueError(
                f"Negative longitudinal treatment variance for {world.world_id} in arm {arm!r}."
            )
        output.append(
            LongitudinalWorldEstimateV1(
                world_id=world.world_id,
                trial_id=world.trial_id,
                world_index=world.world_index,
                mode=mode,
                arm_id=arm,
                intercept=float(intercept_vector @ parameters),
                intercept_standard_error=float(np.sqrt(intercept_variance)),
                slope_per_day=float(slope_vector @ parameters),
                slope_standard_error=float(np.sqrt(slope_variance)),
                treatment_intercept_contrast=(
                    None
                    if arm == control
                    else float(treatment_intercept_vector @ parameters)
                ),
                treatment_intercept_standard_error=(
                    None
                    if arm == control
                    else float(np.sqrt(treatment_intercept_variance))
                ),
                treatment_slope_contrast=(
                    None
                    if arm == control
                    else float(treatment_slope_vector @ parameters)
                ),
                treatment_slope_standard_error=(
                    None if arm == control else float(np.sqrt(treatment_slope_variance))
                ),
                fingerprint=fingerprint,
            )
        )
    return output


def _summarize_modes(
    estimates: list[LongitudinalWorldEstimateV1],
    trials: dict[str, LongitudinalQualificationTrialV1],
) -> list[LongitudinalModeSummaryV1]:
    grouped: dict[tuple[str, LongitudinalMode], list[LongitudinalWorldEstimateV1]] = (
        defaultdict(list)
    )
    for estimate in estimates:
        grouped[(estimate.trial_id, estimate.mode)].append(estimate)
    output = []
    for (trial_id, mode), rows in sorted(grouped.items()):
        by_world = {row.world_id: row.fingerprint for row in rows}
        fingerprints = list(by_world.values())
        reference = trials[trial_id].source_fingerprint
        adjacent = np.asarray(
            [row.adjacent_measurement_correlation for row in fingerprints], dtype=float
        )
        centered_adjacent = np.asarray(
            [row.within_stratum_adjacent_correlation for row in fingerprints],
            dtype=float,
        )
        baseline_final = np.asarray(
            [row.baseline_final_correlation for row in fingerprints], dtype=float
        )
        centered_baseline_final = np.asarray(
            [row.within_stratum_baseline_final_correlation for row in fingerprints],
            dtype=float,
        )
        changes = np.asarray(
            [row.baseline_final_change_mean for row in fingerprints], dtype=float
        )
        marginal_errors = np.asarray(
            [_marginal_error(row, reference) for row in fingerprints],
            dtype=float,
        )
        linkage_errors = np.asarray(
            [
                (
                    abs(
                        row.within_stratum_adjacent_correlation
                        - reference.within_stratum_adjacent_correlation
                    )
                    + abs(
                        row.within_stratum_baseline_final_correlation
                        - reference.within_stratum_baseline_final_correlation
                    )
                )
                / 2.0
                for row in fingerprints
            ],
            dtype=float,
        )
        standardized_errors = np.asarray(
            [_fingerprint_error(row, reference) for row in fingerprints],
            dtype=float,
        )
        adjacent_mean, adjacent_low, adjacent_high = _mean_ci(adjacent)
        centered_adjacent_mean, centered_adjacent_low, centered_adjacent_high = (
            _mean_ci(centered_adjacent)
        )
        marginal_mean, marginal_low, marginal_high = _mean_ci(marginal_errors)
        linkage_mean, linkage_low, linkage_high = _mean_ci(linkage_errors)
        output.append(
            LongitudinalModeSummaryV1(
                trial_id=trial_id,
                mode=mode,
                linkage_retention=_LINKAGE_RETENTION.get(mode),
                worlds=len(fingerprints),
                failures=0,
                fingerprint_standardized_error_mean=float(standardized_errors.mean()),
                marginal_standardized_error_mean=marginal_mean,
                marginal_standardized_error_ci_low=max(0.0, marginal_low),
                marginal_standardized_error_ci_high=marginal_high,
                linkage_absolute_error_mean=linkage_mean,
                linkage_absolute_error_ci_low=max(0.0, linkage_low),
                linkage_absolute_error_ci_high=min(2.0, linkage_high),
                adjacent_correlation_mean=adjacent_mean,
                adjacent_correlation_ci_low=max(-1.0, adjacent_low),
                adjacent_correlation_ci_high=min(1.0, adjacent_high),
                within_stratum_adjacent_correlation_mean=centered_adjacent_mean,
                within_stratum_adjacent_correlation_ci_low=max(
                    -1.0, centered_adjacent_low
                ),
                within_stratum_adjacent_correlation_ci_high=min(
                    1.0, centered_adjacent_high
                ),
                baseline_final_correlation_mean=float(baseline_final.mean()),
                within_stratum_baseline_final_correlation_mean=float(
                    centered_baseline_final.mean()
                ),
                baseline_final_change_mean=float(changes.mean()),
            )
        )
    return output


def _summarize_linkage_dose_response(
    estimates: list[LongitudinalWorldEstimateV1],
    trials: dict[str, LongitudinalQualificationTrialV1],
) -> list[LongitudinalLinkageDoseResponseV1]:
    levels = np.asarray(sorted(set(_LINKAGE_RETENTION.values())), dtype=float)
    output = []
    for trial_id in sorted(trials):
        fingerprints = {
            (row.world_id, row.mode): row.fingerprint
            for row in estimates
            if row.trial_id == trial_id and row.mode in _LINKAGE_RETENTION
        }
        world_ids = sorted({world_id for world_id, _ in fingerprints})
        slopes = []
        endpoint_contrasts = []
        correlations = []
        for world_id in world_ids:
            by_level = {
                retention: fingerprints[
                    (world_id, mode)
                ].within_stratum_adjacent_correlation
                for mode, retention in _LINKAGE_RETENTION.items()
            }
            response = np.asarray([by_level[level] for level in levels], dtype=float)
            correlations.append(response)
            slopes.append(float(np.polyfit(levels, response, deg=1)[0]))
            endpoint_contrasts.append(float(response[-1] - response[0]))
        slope_values = np.asarray(slopes, dtype=float)
        contrast_values = np.asarray(endpoint_contrasts, dtype=float)
        slope_mean, slope_low, slope_high = _mean_ci(slope_values)
        contrast_mean, contrast_low, contrast_high = _mean_ci(contrast_values)
        positive_interval = proportion_interval(
            int(np.sum(slope_values > 0.0)),
            len(slope_values),
        )
        output.append(
            LongitudinalLinkageDoseResponseV1(
                trial_id=trial_id,
                worlds=len(world_ids),
                retention_levels=tuple(levels),
                mean_correlations=tuple(np.mean(np.vstack(correlations), axis=0)),
                mean_slope=slope_mean,
                slope_ci_low=slope_low,
                slope_ci_high=slope_high,
                positive_slope_fraction=float(np.mean(slope_values > 0.0)),
                positive_slope_fraction_ci_low=positive_interval[0],
                positive_slope_fraction_ci_high=positive_interval[1],
                endpoint_contrast_mean=contrast_mean,
                endpoint_contrast_ci_low=contrast_low,
                endpoint_contrast_ci_high=contrast_high,
            )
        )
    return output


def _summarize_treatment_recovery(
    estimates: list[LongitudinalWorldEstimateV1],
    trials: dict[str, LongitudinalQualificationTrialV1],
) -> list[LongitudinalTreatmentRecoveryV1]:
    output = []
    for trial_id, trial in sorted(trials.items()):
        rows = [
            row
            for row in estimates
            if row.trial_id == trial_id and row.mode == "source_anchored"
        ]
        for effect in trial.fitted_model.arm_effects:
            arm_rows = [row for row in rows if row.arm_id == effect.arm_id]
            truth_intercept, truth_slope = _linear_projection(
                np.asarray(trial.time_values, dtype=float),
                np.asarray(effect.visit_shifts, dtype=float),
            )
            intercepts = np.asarray(
                [row.treatment_intercept_contrast for row in arm_rows], dtype=float
            )
            intercept_ses = np.asarray(
                [row.treatment_intercept_standard_error for row in arm_rows],
                dtype=float,
            )
            slopes = np.asarray(
                [row.treatment_slope_contrast for row in arm_rows], dtype=float
            )
            slope_ses = np.asarray(
                [row.treatment_slope_standard_error for row in arm_rows], dtype=float
            )
            intercept_biases = intercepts - truth_intercept
            slope_biases = slopes - truth_slope
            _, intercept_bias_low, intercept_bias_high = _mean_ci(intercept_biases)
            _, slope_bias_low, slope_bias_high = _mean_ci(slope_biases)
            intercept_covered = np.abs(intercept_biases) <= 1.96 * intercept_ses
            slope_covered = np.abs(slope_biases) <= 1.96 * slope_ses
            intercept_coverage_low, intercept_coverage_high = proportion_interval(
                int(intercept_covered.sum()), len(intercept_covered)
            )
            slope_coverage_low, slope_coverage_high = proportion_interval(
                int(slope_covered.sum()), len(slope_covered)
            )
            output.append(
                LongitudinalTreatmentRecoveryV1(
                    trial_id=trial_id,
                    arm_id=effect.arm_id,
                    worlds=len(arm_rows),
                    truth_intercept_contrast=truth_intercept,
                    mean_intercept_contrast=float(intercepts.mean()),
                    intercept_contrast_bias=float(intercepts.mean() - truth_intercept),
                    intercept_contrast_bias_ci_low=intercept_bias_low,
                    intercept_contrast_bias_ci_high=intercept_bias_high,
                    intercept_contrast_coverage=float(np.mean(intercept_covered)),
                    intercept_contrast_coverage_ci_low=intercept_coverage_low,
                    intercept_contrast_coverage_ci_high=intercept_coverage_high,
                    truth_slope_contrast=truth_slope,
                    mean_slope_contrast=float(slopes.mean()),
                    slope_contrast_bias=float(slopes.mean() - truth_slope),
                    slope_contrast_bias_ci_low=slope_bias_low,
                    slope_contrast_bias_ci_high=slope_bias_high,
                    slope_contrast_coverage=float(np.mean(slope_covered)),
                    slope_contrast_coverage_ci_low=slope_coverage_low,
                    slope_contrast_coverage_ci_high=slope_coverage_high,
                )
            )
    return output


def _summarize_source_anchored_recovery(
    estimates: list[LongitudinalWorldEstimateV1],
    trials: dict[str, LongitudinalQualificationTrialV1],
) -> list[LongitudinalArmRecoveryV1]:
    output = []
    for trial_id, trial in sorted(trials.items()):
        effects = {effect.arm_id: effect for effect in trial.fitted_model.arm_effects}
        rows = [
            row
            for row in estimates
            if row.trial_id == trial_id and row.mode == "source_anchored"
        ]
        for arm in trial.arm_ids:
            arm_rows = [row for row in rows if row.arm_id == arm]
            effect = effects.get(arm)
            means = np.asarray(trial.fitted_model.control_mean_values, dtype=float)
            if effect is not None:
                means += np.asarray(effect.visit_shifts, dtype=float)
            truth_intercept, truth_slope = _linear_projection(
                np.asarray(trial.time_values, dtype=float),
                means,
            )
            intercepts = np.asarray([row.intercept for row in arm_rows], dtype=float)
            slopes = np.asarray([row.slope_per_day for row in arm_rows], dtype=float)
            intercept_biases = intercepts - truth_intercept
            slope_biases = slopes - truth_slope
            _, intercept_bias_low, intercept_bias_high = _mean_ci(intercept_biases)
            _, slope_bias_low, slope_bias_high = _mean_ci(slope_biases)
            intercept_covered = np.asarray(
                [
                    abs(row.intercept - truth_intercept)
                    <= 1.96 * row.intercept_standard_error
                    for row in arm_rows
                ],
                dtype=bool,
            )
            slope_covered = np.asarray(
                [
                    abs(row.slope_per_day - truth_slope)
                    <= 1.96 * row.slope_standard_error
                    for row in arm_rows
                ],
                dtype=bool,
            )
            intercept_coverage_low, intercept_coverage_high = proportion_interval(
                int(intercept_covered.sum()), len(intercept_covered)
            )
            slope_coverage_low, slope_coverage_high = proportion_interval(
                int(slope_covered.sum()), len(slope_covered)
            )
            output.append(
                LongitudinalArmRecoveryV1(
                    trial_id=trial_id,
                    arm_id=arm,
                    worlds=len(arm_rows),
                    truth_intercept=truth_intercept,
                    mean_intercept=float(intercepts.mean()),
                    intercept_bias=float(intercepts.mean() - truth_intercept),
                    intercept_bias_ci_low=intercept_bias_low,
                    intercept_bias_ci_high=intercept_bias_high,
                    intercept_coverage=float(np.mean(intercept_covered)),
                    intercept_coverage_ci_low=intercept_coverage_low,
                    intercept_coverage_ci_high=intercept_coverage_high,
                    truth_slope_per_day=truth_slope,
                    mean_slope_per_day=float(slopes.mean()),
                    slope_bias=float(slopes.mean() - truth_slope),
                    slope_bias_ci_low=slope_bias_low,
                    slope_bias_ci_high=slope_bias_high,
                    slope_coverage=float(np.mean(slope_covered)),
                    slope_coverage_ci_low=slope_coverage_low,
                    slope_coverage_ci_high=slope_coverage_high,
                )
            )
    return output


def _linear_projection(
    time: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(time), dtype=float), time])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def _fingerprint_error(
    observed: LongitudinalTrialFingerprintV1,
    reference: LongitudinalTrialFingerprintV1,
) -> float:
    scales = {
        "adjacent_measurement_correlation": 0.25,
        "within_stratum_adjacent_correlation": 0.25,
        "baseline_final_correlation": 0.25,
        "within_stratum_baseline_final_correlation": 0.25,
        "baseline_final_change_mean": max(reference.baseline_final_change_sd, 1e-8),
        "baseline_final_change_sd": max(reference.baseline_final_change_sd, 1e-8),
        "observed_timepoints_mean": max(reference.observed_timepoints_mean, 1.0),
        "observation_fraction": 0.1,
    }
    return float(
        np.mean(
            [
                abs(float(getattr(observed, key)) - float(getattr(reference, key)))
                / scale
                for key, scale in scales.items()
            ]
        )
    )


def _marginal_error(
    observed: LongitudinalTrialFingerprintV1,
    reference: LongitudinalTrialFingerprintV1,
) -> float:
    reference_rows = {(row.arm_id, row.time): row for row in reference.marginal_moments}
    observed_rows = {(row.arm_id, row.time): row for row in observed.marginal_moments}
    if set(observed_rows) != set(reference_rows):
        raise ValueError(
            "Longitudinal marginal support differs from the source fingerprint."
        )
    errors = []
    for key, expected in reference_rows.items():
        actual = observed_rows[key]
        scale = expected.standard_deviation
        errors.extend(
            [
                abs(actual.mean - expected.mean) / scale,
                abs(actual.standard_deviation - expected.standard_deviation) / scale,
                abs(actual.q10 - expected.q10) / scale,
                abs(actual.median - expected.median) / scale,
                abs(actual.q90 - expected.q90) / scale,
            ]
        )
    return float(np.mean(errors))


def _mean_ci(values: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    if len(values) < 2:
        raise ValueError("Longitudinal summary requires at least two worlds.")
    mean = float(values.mean())
    half_width = float(student_t.ppf(0.975, df=len(values) - 1)) * float(
        values.std(ddof=1) / np.sqrt(len(values))
    )
    return mean, mean - half_width, mean + half_width


def _release_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Unsafe or missing longitudinal release path: {relative!r}.")
    return path


def _json_sha(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _world_seed(seed: int, trial_id: str, world_index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{trial_id}:{world_index}".encode()).digest()[:4],
        byteorder="big",
    )


def _world_id(design_sha256: str, trial_id: str, world_index: int) -> str:
    digest = hashlib.sha256(
        f"{design_sha256}:{trial_id}:{world_index}".encode()
    ).hexdigest()
    return f"world_{digest[:20]}"


__all__ = [
    "LongitudinalArmEffectV1",
    "LongitudinalFittedModelV1",
    "LongitudinalMarginalPredictiveV1",
    "LongitudinalQualificationDesignV1",
    "LongitudinalQualificationReportV1",
    "LongitudinalQualificationTrialV1",
    "LongitudinalWorldReceiptV1",
    "evaluate_longitudinal_qualification",
]
