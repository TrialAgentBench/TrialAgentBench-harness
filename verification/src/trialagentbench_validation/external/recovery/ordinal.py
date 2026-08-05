"""Independent qualification of source-fitted ordinal trial worlds."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import t as student_t
from statsmodels.miscmodels.ordinal_model import OrderedModel

from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.statistics import proportion_interval


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OrdinalArmReferenceV1(_FrozenModel):
    """Source ordinal distribution and missingness for one arm."""

    arm_id: str = Field(min_length=1)
    participants: int = Field(ge=20)
    missing_outcomes: int = Field(ge=0)
    category_probabilities: tuple[float, ...] = Field(min_length=3)


class OrdinalDoseDistributionV1(_FrozenModel):
    """Source-fitted category probabilities for one arm and effect dose."""

    dose_multiplier: float = Field(ge=0)
    arm_id: str = Field(min_length=1)
    category_probabilities: tuple[float, ...] = Field(min_length=3)


class OrdinalSafetyReferenceV1(_FrozenModel):
    """Source prevalence for one binary safety endpoint and arm."""

    endpoint: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    arm_id: str = Field(min_length=1)
    observed_participants: int = Field(ge=20)
    event_probability: float = Field(gt=0, lt=1)


class OrdinalQualificationDesignV1(_FrozenModel):
    """Path-free public design for an ordinal qualification campaign."""

    schema_id: Literal["trialagentbench.ordinal_qualification_design/v1"] = (
        "trialagentbench.ordinal_qualification_design/v1"
    )
    trial_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participants: int = Field(ge=100)
    worlds: int = Field(ge=2)
    seed: int = Field(ge=0, le=2**32 - 1)
    categories: tuple[int, ...] = Field(min_length=3)
    control_arm_id: str = Field(min_length=1)
    treatment_arm_id: str = Field(min_length=1)
    source_log_common_odds_ratio: float = Field(allow_inf_nan=False)
    dose_multipliers: tuple[float, ...] = Field(min_length=3)
    arms: tuple[OrdinalArmReferenceV1, OrdinalArmReferenceV1]
    fitted_distributions: tuple[OrdinalDoseDistributionV1, ...] = Field(min_length=6)
    safety_references: tuple[OrdinalSafetyReferenceV1, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _complete(self) -> OrdinalQualificationDesignV1:
        if tuple(range(len(self.categories))) != self.categories:
            raise ValueError(
                "Ordinal categories must be consecutive integers from zero"
            )
        if self.control_arm_id == self.treatment_arm_id:
            raise ValueError("Ordinal qualification arms must be distinct")
        if {arm.arm_id for arm in self.arms} != {
            self.control_arm_id,
            self.treatment_arm_id,
        }:
            raise ValueError("Ordinal references must cover control and treatment")
        if sum(arm.participants for arm in self.arms) != self.participants:
            raise ValueError("Ordinal arm counts must sum to participants")
        if tuple(sorted(self.dose_multipliers)) != self.dose_multipliers:
            raise ValueError("Ordinal doses must be sorted")
        if 0.0 not in self.dose_multipliers or 1.0 not in self.dose_multipliers:
            raise ValueError(
                "Ordinal dose response requires null and source-fitted doses"
            )
        expected = {
            (dose, arm)
            for dose in self.dose_multipliers
            for arm in (self.control_arm_id, self.treatment_arm_id)
        }
        if {
            (row.dose_multiplier, row.arm_id) for row in self.fitted_distributions
        } != expected:
            raise ValueError(
                "Ordinal fitted distributions do not cover the dose-by-arm grid"
            )
        safety_endpoints = {row.endpoint for row in self.safety_references}
        if {(row.endpoint, row.arm_id) for row in self.safety_references} != {
            (endpoint, arm)
            for endpoint in safety_endpoints
            for arm in (self.control_arm_id, self.treatment_arm_id)
        }:
            raise ValueError("Ordinal safety references must cover both arms")
        probability_rows = [row.category_probabilities for row in self.arms] + [
            row.category_probabilities for row in self.fitted_distributions
        ]
        for category_probabilities in probability_rows:
            probabilities = np.asarray(category_probabilities, dtype=np.float64)
            if len(probabilities) != len(self.categories):
                raise ValueError("Ordinal probabilities must match category support")
            if np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1.0):
                raise ValueError(
                    "Ordinal probabilities must be positive and sum to one"
                )
        return self


class OrdinalWorldReceiptV1(_FrozenModel):
    """Checksum-bound identity of one released ordinal world."""

    world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    world_index: int = Field(ge=0)
    analysis_path: str = Field(pattern=r"^worlds/world_[0-9a-f]{20}\.parquet$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OrdinalQualificationReceiptV1(_FrozenModel):
    """Complete inventory of released ordinal worlds."""

    schema_id: Literal["trialagentbench.ordinal_qualification_receipt/v1"] = (
        "trialagentbench.ordinal_qualification_receipt/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[OrdinalWorldReceiptV1, ...] = Field(min_length=2)


class OrdinalCategoryResultV1(_FrozenModel):
    """Monte Carlo category probability at one dose and arm."""

    dose_multiplier: float = Field(ge=0)
    arm_id: str = Field(min_length=1)
    category: int = Field(ge=0)
    source_probability: float | None = Field(default=None, ge=0, le=1)
    fitted_probability: float = Field(ge=0, le=1)
    mean_probability: float = Field(ge=0, le=1)
    interval_95_low: float = Field(ge=0, le=1)
    interval_95_high: float = Field(ge=0, le=1)
    predictive_50_low: float = Field(ge=0, le=1)
    predictive_50_high: float = Field(ge=0, le=1)
    predictive_95_low: float = Field(ge=0, le=1)
    predictive_95_high: float = Field(ge=0, le=1)


class OrdinalSafetyResultV1(_FrozenModel):
    """Monte Carlo safety-event probability for one endpoint and arm."""

    endpoint: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    arm_id: str = Field(min_length=1)
    source_probability: float = Field(gt=0, lt=1)
    mean_probability: float = Field(ge=0, le=1)
    bias: float = Field(allow_inf_nan=False)
    bias_ci_low: float = Field(allow_inf_nan=False)
    bias_ci_high: float = Field(allow_inf_nan=False)
    predictive_50_low: float = Field(ge=0, le=1)
    predictive_50_high: float = Field(ge=0, le=1)
    predictive_95_low: float = Field(ge=0, le=1)
    predictive_95_high: float = Field(ge=0, le=1)


class OrdinalDoseRecoveryV1(_FrozenModel):
    """Proportional-odds recovery at one effect dose."""

    dose_multiplier: float = Field(ge=0)
    truth_log_common_odds_ratio: float = Field(allow_inf_nan=False)
    worlds: int = Field(ge=2)
    mean_log_common_odds_ratio: float = Field(allow_inf_nan=False)
    bias: float = Field(allow_inf_nan=False)
    bias_ci_low: float = Field(allow_inf_nan=False)
    bias_ci_high: float = Field(allow_inf_nan=False)
    predictive_50_low: float = Field(allow_inf_nan=False)
    predictive_50_high: float = Field(allow_inf_nan=False)
    predictive_95_low: float = Field(allow_inf_nan=False)
    predictive_95_high: float = Field(allow_inf_nan=False)
    coverage: float = Field(ge=0, le=1)
    coverage_ci_low: float = Field(ge=0, le=1)
    coverage_ci_high: float = Field(ge=0, le=1)
    mean_mortality_probability: float = Field(ge=0, le=1)
    mean_missing_fraction: float = Field(ge=0, le=1)


class OrdinalQualificationReportV1(_FrozenModel):
    """Independent native-scale ordinal qualification report."""

    schema_id: Literal["trialagentbench.ordinal_qualification_report/v1"] = (
        "trialagentbench.ordinal_qualification_report/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: int = Field(ge=2)
    categories: tuple[OrdinalCategoryResultV1, ...] = Field(min_length=1)
    safety: tuple[OrdinalSafetyResultV1, ...] = Field(min_length=2)
    recovery: tuple[OrdinalDoseRecoveryV1, ...] = Field(min_length=3)
    source_dose_category_mae: float = Field(ge=0, le=1)
    source_dose_cumulative_mae: float = Field(ge=0, le=1)
    dose_response_slope: float = Field(allow_inf_nan=False)
    dose_response_slope_ci_low: float = Field(allow_inf_nan=False)
    dose_response_slope_ci_high: float = Field(allow_inf_nan=False)
    monotone_world_fraction: float = Field(ge=0, le=1)
    monotone_world_fraction_ci_low: float = Field(ge=0, le=1)
    monotone_world_fraction_ci_high: float = Field(ge=0, le=1)
    safety_probability_mae: float = Field(ge=0, le=1)
    intact_high_dose_log_odds: float = Field(allow_inf_nan=False)
    intact_high_dose_log_odds_ci_low: float = Field(allow_inf_nan=False)
    intact_high_dose_log_odds_ci_high: float = Field(allow_inf_nan=False)
    broken_arm_linkage_log_odds: float = Field(allow_inf_nan=False)
    broken_arm_linkage_log_odds_ci_low: float = Field(allow_inf_nan=False)
    broken_arm_linkage_log_odds_ci_high: float = Field(allow_inf_nan=False)
    intact_high_dose_category_mae: float = Field(ge=0, le=1)
    intact_high_dose_category_mae_ci_low: float = Field(ge=0, le=1)
    intact_high_dose_category_mae_ci_high: float = Field(ge=0, le=1)
    broken_arm_category_mae: float = Field(ge=0, le=1)
    broken_arm_category_mae_ci_low: float = Field(ge=0, le=1)
    broken_arm_category_mae_ci_high: float = Field(ge=0, le=1)
    broken_minus_intact_category_mae: float = Field(ge=-1, le=1)
    broken_minus_intact_category_mae_ci_low: float = Field(allow_inf_nan=False)
    broken_minus_intact_category_mae_ci_high: float = Field(allow_inf_nan=False)


def evaluate_ordinal_qualification(
    *,
    release_dir: Path,
    minimum_worlds: int = 100,
) -> OrdinalQualificationReportV1:
    """Verify release identity and recompute ordinal evidence."""

    design_path = release_dir / "design.json"
    receipt_path = release_dir / "receipt.json"
    design = OrdinalQualificationDesignV1.model_validate_json(
        design_path.read_text(encoding="utf-8")
    )
    receipt = OrdinalQualificationReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    design_sha = _json_sha(json.loads(design_path.read_text(encoding="utf-8")))
    if receipt.design_sha256 != design_sha:
        raise ValueError("Ordinal receipt does not match its design")
    if len(receipt.worlds) != design.worlds or design.worlds < minimum_worlds:
        raise ValueError("Ordinal release has insufficient or inconsistent worlds")
    expected_paths = {item.analysis_path for item in receipt.worlds}
    actual_paths = {
        path.relative_to(release_dir).as_posix()
        for path in (release_dir / "worlds").glob("*.parquet")
    }
    if actual_paths != expected_paths:
        raise ValueError("Ordinal world inventory does not match the receipt")
    frames = []
    for item in receipt.worlds:
        path = release_dir / item.analysis_path
        if sha256_file(path) != item.analysis_sha256:
            raise ValueError(f"Ordinal world checksum mismatch: {item.world_id}")
        frame = pd.read_parquet(path)
        _validate_world(frame, design=design, world_id=item.world_id)
        frames.append(frame)
    values = pd.concat(frames, ignore_index=True)
    category_results = _category_results(values, design=design)
    safety_results = _safety_results(values, design=design)
    recovery, estimates = _recovery_results(values, design=design)
    source_rows = [row for row in category_results if row.dose_multiplier == 1.0]
    category_mae = float(
        np.mean(
            [
                abs(row.mean_probability - cast(float, row.source_probability))
                for row in source_rows
            ]
        )
    )
    cumulative_errors = []
    for arm in (design.control_arm_id, design.treatment_arm_id):
        rows = sorted(
            (row for row in source_rows if row.arm_id == arm),
            key=lambda row: row.category,
        )
        observed = np.cumsum([row.mean_probability for row in rows])
        source = np.cumsum([cast(float, row.source_probability) for row in rows])
        cumulative_errors.extend(np.abs(observed[:-1] - source[:-1]))
    world_slopes = [
        float(np.polyfit(world["dose_multiplier"], world["estimate"], deg=1)[0])
        for _, world in estimates.groupby("world_id", sort=True)
    ]
    monotone = np.sign(world_slopes) == np.sign(design.source_log_common_odds_ratio)
    monotone_interval = proportion_interval(int(np.sum(monotone)), len(monotone))
    slope, slope_low, slope_high = _mean_ci(np.asarray(world_slopes, dtype=float))
    intact_effects, broken_effects, intact_errors, broken_errors = _arm_linkage_errors(
        values,
        design=design,
    )
    intact_effect, intact_effect_low, intact_effect_high = _mean_ci(intact_effects)
    broken_effect, broken_effect_low, broken_effect_high = _mean_ci(broken_effects)
    intact_category_mae, intact_category_low, intact_category_high = _mean_ci(
        intact_errors
    )
    broken_category_mae, broken_category_low, broken_category_high = _mean_ci(
        broken_errors
    )
    category_difference, difference_low, difference_high = _mean_ci(
        broken_errors - intact_errors
    )
    return OrdinalQualificationReportV1(
        design_sha256=design_sha,
        receipt_sha256=sha256_file(receipt_path),
        worlds=design.worlds,
        categories=tuple(category_results),
        safety=tuple(safety_results),
        recovery=tuple(recovery),
        source_dose_category_mae=category_mae,
        source_dose_cumulative_mae=float(np.mean(cumulative_errors)),
        dose_response_slope=slope,
        dose_response_slope_ci_low=slope_low,
        dose_response_slope_ci_high=slope_high,
        monotone_world_fraction=float(np.mean(monotone)),
        monotone_world_fraction_ci_low=monotone_interval[0],
        monotone_world_fraction_ci_high=monotone_interval[1],
        safety_probability_mae=float(
            np.mean([abs(row.bias) for row in safety_results])
        ),
        intact_high_dose_log_odds=intact_effect,
        intact_high_dose_log_odds_ci_low=intact_effect_low,
        intact_high_dose_log_odds_ci_high=intact_effect_high,
        broken_arm_linkage_log_odds=broken_effect,
        broken_arm_linkage_log_odds_ci_low=broken_effect_low,
        broken_arm_linkage_log_odds_ci_high=broken_effect_high,
        intact_high_dose_category_mae=intact_category_mae,
        intact_high_dose_category_mae_ci_low=intact_category_low,
        intact_high_dose_category_mae_ci_high=intact_category_high,
        broken_arm_category_mae=broken_category_mae,
        broken_arm_category_mae_ci_low=broken_category_low,
        broken_arm_category_mae_ci_high=broken_category_high,
        broken_minus_intact_category_mae=category_difference,
        broken_minus_intact_category_mae_ci_low=difference_low,
        broken_minus_intact_category_mae_ci_high=difference_high,
    )


def _validate_world(
    frame: pd.DataFrame,
    *,
    design: OrdinalQualificationDesignV1,
    world_id: str,
) -> None:
    required = {
        "world_id",
        "dose_multiplier",
        "participant_id",
        "arm",
        "mrs_90d",
        "observed",
        *(row.endpoint for row in design.safety_references),
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Ordinal world is missing columns: {missing}")
    if set(frame["world_id"].astype(str)) != {world_id}:
        raise ValueError("Ordinal world identity mismatch")
    if set(frame["dose_multiplier"].astype(float)) != set(design.dose_multipliers):
        raise ValueError("Ordinal world has an incomplete dose surface")
    for _, dose in frame.groupby("dose_multiplier", observed=True):
        if (
            len(dose) != design.participants
            or dose["participant_id"].duplicated().any()
        ):
            raise ValueError("Ordinal dose must contain one row per participant")
        if set(dose["arm"].astype(str)) != {
            design.control_arm_id,
            design.treatment_arm_id,
        }:
            raise ValueError("Ordinal dose has invalid randomized arms")
    observed = frame["observed"].astype(bool)
    values = frame.loc[observed, "mrs_90d"].astype(int)
    if (
        not values.isin(design.categories).all()
        or frame.loc[~observed, "mrs_90d"].notna().any()
    ):
        raise ValueError("Ordinal observations violate category or missingness support")
    for endpoint in {row.endpoint for row in design.safety_references}:
        if not frame[endpoint].isin([0, 1]).all():
            raise ValueError(f"Ordinal safety endpoint is not binary: {endpoint}")


def _category_results(
    values: pd.DataFrame,
    *,
    design: OrdinalQualificationDesignV1,
) -> list[OrdinalCategoryResultV1]:
    fitted = {
        (row.dose_multiplier, row.arm_id): row.category_probabilities
        for row in design.fitted_distributions
    }
    source = {arm.arm_id: arm.category_probabilities for arm in design.arms}
    samples: dict[tuple[float, str, int], list[float]] = {}
    for (_, dose, arm), group in values.groupby(
        ["world_id", "dose_multiplier", "arm"], observed=True, sort=True
    ):
        observed = group.loc[group["observed"].astype(bool), "mrs_90d"].astype(int)
        probabilities = observed.value_counts(normalize=True).reindex(
            design.categories, fill_value=0.0
        )
        dose_value = cast(float, dose)
        for category in design.categories:
            samples.setdefault((dose_value, str(arm), category), []).append(
                float(probabilities.loc[category])
            )
    output = []
    for (dose, arm, category), estimates in sorted(samples.items()):
        sample_array = np.asarray(estimates, dtype=float)
        mean, low, high = _mean_ci(sample_array)
        predictive = np.quantile(sample_array, [0.025, 0.25, 0.75, 0.975])
        output.append(
            OrdinalCategoryResultV1(
                dose_multiplier=dose,
                arm_id=arm,
                category=category,
                source_probability=source[arm][category] if dose == 1.0 else None,
                fitted_probability=fitted[(dose, arm)][category],
                mean_probability=mean,
                interval_95_low=max(0.0, low),
                interval_95_high=min(1.0, high),
                predictive_50_low=float(predictive[1]),
                predictive_50_high=float(predictive[2]),
                predictive_95_low=float(predictive[0]),
                predictive_95_high=float(predictive[3]),
            )
        )
    return output


def _safety_results(
    values: pd.DataFrame,
    *,
    design: OrdinalQualificationDesignV1,
) -> list[OrdinalSafetyResultV1]:
    source = {
        (row.endpoint, row.arm_id): row.event_probability
        for row in design.safety_references
    }
    source_dose = values.loc[values["dose_multiplier"].eq(1.0)]
    samples: dict[tuple[str, str], list[float]] = {}
    for (_, arm), group in source_dose.groupby(
        ["world_id", "arm"],
        observed=True,
        sort=True,
    ):
        for endpoint in {row.endpoint for row in design.safety_references}:
            samples.setdefault((endpoint, str(arm)), []).append(
                float(group[endpoint].mean())
            )
    output = []
    for (endpoint, arm), estimates in sorted(samples.items()):
        sample_array = np.asarray(estimates, dtype=float)
        reference = source[(endpoint, arm)]
        bias, low, high = _mean_ci(sample_array - reference)
        predictive = np.quantile(sample_array, [0.025, 0.25, 0.75, 0.975])
        output.append(
            OrdinalSafetyResultV1(
                endpoint=endpoint,
                arm_id=arm,
                source_probability=reference,
                mean_probability=float(np.mean(sample_array)),
                bias=bias,
                bias_ci_low=low,
                bias_ci_high=high,
                predictive_50_low=float(predictive[1]),
                predictive_50_high=float(predictive[2]),
                predictive_95_low=float(predictive[0]),
                predictive_95_high=float(predictive[3]),
            )
        )
    return output


def _arm_linkage_errors(
    values: pd.DataFrame,
    *,
    design: OrdinalQualificationDesignV1,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Return paired world-level estimates and errors under intact and broken linkage."""

    dose = max(design.dose_multipliers)
    dose_values = values.loc[values["dose_multiplier"].eq(dose)]
    fitted = {
        row.arm_id: row.category_probabilities
        for row in design.fitted_distributions
        if row.dose_multiplier == dose
    }
    intact_effects: list[float] = []
    broken_effects: list[float] = []
    intact_world_errors: list[float] = []
    broken_world_errors: list[float] = []
    for world_id, group in dose_values.groupby("world_id", sort=True):
        complete = group.loc[group["observed"].astype(bool)].copy()
        arm = complete["arm"].astype(str).to_numpy(dtype=str)
        outcome = complete["mrs_90d"].astype(int).to_numpy(dtype=np.int64)
        intact_effects.append(_fit_log_common_odds(outcome, arm, design=design))
        digest = hashlib.sha256(f"{world_id}:broken-arm-linkage".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        broken_arm = rng.permutation(arm)
        broken_effects.append(_fit_log_common_odds(outcome, broken_arm, design=design))
        intact_errors: list[float] = []
        broken_errors: list[float] = []
        for arm_id in (design.control_arm_id, design.treatment_arm_id):
            reference = np.asarray(fitted[arm_id], dtype=float)
            intact_probabilities = (
                pd.Series(outcome[arm == arm_id])
                .value_counts(normalize=True)
                .reindex(design.categories, fill_value=0.0)
                .to_numpy(dtype=float)
            )
            broken_probabilities = (
                pd.Series(outcome[broken_arm == arm_id])
                .value_counts(normalize=True)
                .reindex(design.categories, fill_value=0.0)
                .to_numpy(dtype=float)
            )
            intact_errors.extend(np.abs(intact_probabilities - reference))
            broken_errors.extend(np.abs(broken_probabilities - reference))
        intact_world_errors.append(float(np.mean(intact_errors)))
        broken_world_errors.append(float(np.mean(broken_errors)))
    return (
        np.asarray(intact_effects, dtype=float),
        np.asarray(broken_effects, dtype=float),
        np.asarray(intact_world_errors, dtype=float),
        np.asarray(broken_world_errors, dtype=float),
    )


def _fit_log_common_odds(
    outcome: npt.ArrayLike,
    arm: npt.ArrayLike,
    *,
    design: OrdinalQualificationDesignV1,
) -> float:
    outcome_array = np.asarray(outcome, dtype=np.int64)
    arm_array = np.asarray(arm, dtype=str)
    treatment = (arm_array == design.treatment_arm_id).astype(float)
    fitted = OrderedModel(
        outcome_array,
        treatment[:, np.newaxis],
        distr="logit",
    ).fit(method="bfgs", disp=False)
    estimate = float(fitted.params[0])
    if not math.isfinite(estimate):
        raise ValueError("Broken-linkage proportional-odds fit is non-finite")
    return estimate


def _recovery_results(
    values: pd.DataFrame,
    *,
    design: OrdinalQualificationDesignV1,
) -> tuple[list[OrdinalDoseRecoveryV1], pd.DataFrame]:
    estimates = []
    for (world_id, dose), group in values.groupby(
        ["world_id", "dose_multiplier"], observed=True, sort=True
    ):
        complete = group.loc[group["observed"].astype(bool)].copy()
        treatment = (
            complete["arm"].astype(str).eq(design.treatment_arm_id).astype(float)
        )
        fitted = OrderedModel(
            complete["mrs_90d"].astype(int).to_numpy(),
            treatment.to_numpy()[:, np.newaxis],
            distr="logit",
        ).fit(method="bfgs", disp=False)
        estimate = float(fitted.params[0])
        standard_error = float(fitted.bse[0])
        if (
            not math.isfinite(estimate)
            or not math.isfinite(standard_error)
            or standard_error <= 0
        ):
            raise ValueError("Proportional-odds analysis produced an invalid estimate")
        estimates.append(
            {
                "world_id": str(world_id),
                "dose_multiplier": cast(float, dose),
                "estimate": estimate,
                "standard_error": standard_error,
                "mortality_probability": float(
                    complete.loc[
                        complete["arm"].astype(str).eq(design.treatment_arm_id),
                        "mrs_90d",
                    ]
                    .astype(int)
                    .eq(design.categories[-1])
                    .mean()
                ),
                "missing_fraction": float(1.0 - group["observed"].astype(bool).mean()),
            }
        )
    frame = pd.DataFrame(estimates)
    output = []
    for dose, rows in frame.groupby("dose_multiplier", observed=True, sort=True):
        dose_value = cast(float, dose)
        truth = dose_value * design.source_log_common_odds_ratio
        bias_samples = rows["estimate"].to_numpy(dtype=float) - truth
        bias, low, high = _mean_ci(bias_samples)
        predictive = np.quantile(
            rows["estimate"].to_numpy(dtype=float),
            [0.025, 0.25, 0.75, 0.975],
        )
        covered = (rows["estimate"] - 1.96 * rows["standard_error"] <= truth) & (
            rows["estimate"] + 1.96 * rows["standard_error"] >= truth
        )
        covered_count = int(covered.sum())
        coverage_interval = proportion_interval(covered_count, len(rows))
        output.append(
            OrdinalDoseRecoveryV1(
                dose_multiplier=dose_value,
                truth_log_common_odds_ratio=truth,
                worlds=len(rows),
                mean_log_common_odds_ratio=cast(float, rows["estimate"].mean()),
                bias=bias,
                bias_ci_low=low,
                bias_ci_high=high,
                predictive_50_low=float(predictive[1]),
                predictive_50_high=float(predictive[2]),
                predictive_95_low=float(predictive[0]),
                predictive_95_high=float(predictive[3]),
                coverage=covered_count / len(rows),
                coverage_ci_low=coverage_interval[0],
                coverage_ci_high=coverage_interval[1],
                mean_mortality_probability=cast(
                    float, rows["mortality_probability"].mean()
                ),
                mean_missing_fraction=cast(float, rows["missing_fraction"].mean()),
            )
        )
    return output, frame


def _mean_ci(values: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Monte Carlo interval requires at least two finite values")
    mean = float(np.mean(values))
    half = float(
        student_t.ppf(0.975, df=len(values) - 1)
        * np.std(values, ddof=1)
        / np.sqrt(len(values))
    )
    return mean, mean - half, mean + half


def _json_sha(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "OrdinalArmReferenceV1",
    "OrdinalDoseDistributionV1",
    "OrdinalQualificationDesignV1",
    "OrdinalQualificationReceiptV1",
    "OrdinalQualificationReportV1",
    "OrdinalSafetyReferenceV1",
    "OrdinalSafetyResultV1",
    "OrdinalWorldReceiptV1",
    "evaluate_ordinal_qualification",
]
