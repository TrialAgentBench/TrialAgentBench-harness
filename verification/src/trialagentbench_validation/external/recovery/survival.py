"""Independent qualification of source-fitted survival trial worlds."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import t as student_t

from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.statistics import proportion_interval


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SurvivalArmReferenceV1(_FrozenModel):
    """Source event process for one randomized arm."""

    arm_id: str = Field(min_length=1)
    participants: int = Field(ge=20)
    events: int = Field(ge=1)
    early_censoring: int = Field(ge=0)
    survival_at_grid: tuple[float, ...] = Field(min_length=3)
    at_risk_at_grid: tuple[int, ...] = Field(min_length=3)
    rmst_at_horizon: float = Field(gt=0)


class SurvivalQualificationDesignV1(_FrozenModel):
    """Path-free public design for a source-fitted survival campaign."""

    schema_id: Literal["trialagentbench.survival_qualification_design/v1"] = (
        "trialagentbench.survival_qualification_design/v1"
    )
    trial_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participants: int = Field(ge=100)
    worlds: int = Field(ge=2)
    seed: int = Field(ge=0, le=2**32 - 1)
    horizon: float = Field(gt=0)
    time_grid: tuple[float, ...] = Field(min_length=3)
    control_arm_id: str = Field(min_length=1)
    treatment_arm_id: str = Field(min_length=1)
    treatment_prevalence: float = Field(gt=0, lt=1)
    source_log_hazard_ratio: float = Field(allow_inf_nan=False)
    dose_multipliers: tuple[float, ...] = Field(min_length=3)
    random_censoring_rate: float = Field(ge=0, allow_inf_nan=False)
    arms: tuple[SurvivalArmReferenceV1, SurvivalArmReferenceV1]

    @model_validator(mode="after")
    def _complete(self) -> SurvivalQualificationDesignV1:
        if self.control_arm_id == self.treatment_arm_id:
            raise ValueError("Survival qualification arms must be distinct")
        if {arm.arm_id for arm in self.arms} != {
            self.control_arm_id,
            self.treatment_arm_id,
        }:
            raise ValueError("Survival arm references must cover control and treatment")
        if sum(arm.participants for arm in self.arms) != self.participants:
            raise ValueError("Survival arm counts must sum to participants")
        if (
            tuple(sorted(self.time_grid)) != self.time_grid
            or self.time_grid[-1] != self.horizon
        ):
            raise ValueError("Survival time grid must be sorted and end at the horizon")
        if tuple(sorted(self.dose_multipliers)) != self.dose_multipliers:
            raise ValueError("Survival doses must be sorted")
        if 0.0 not in self.dose_multipliers or 1.0 not in self.dose_multipliers:
            raise ValueError(
                "Survival dose response requires null and source-fitted doses"
            )
        if any(
            len(arm.survival_at_grid) != len(self.time_grid)
            or len(arm.at_risk_at_grid) != len(self.time_grid)
            for arm in self.arms
        ):
            raise ValueError(
                "Survival source curves and risk sets must match the time grid"
            )
        if any(
            arm.rmst_at_horizon > self.horizon
            or arm.at_risk_at_grid[0] > arm.participants
            or any(
                later > earlier
                for earlier, later in zip(
                    arm.at_risk_at_grid,
                    arm.at_risk_at_grid[1:],
                    strict=False,
                )
            )
            for arm in self.arms
        ):
            raise ValueError("Survival source risk sets or RMST are invalid")
        return self


class SurvivalWorldReceiptV1(_FrozenModel):
    """Checksum-bound identity of one released survival world."""

    world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    world_index: int = Field(ge=0)
    analysis_path: str = Field(pattern=r"^worlds/world_[0-9a-f]{20}\.parquet$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    truth_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SurvivalQualificationReceiptV1(_FrozenModel):
    """Complete inventory of released survival worlds."""

    schema_id: Literal["trialagentbench.survival_qualification_receipt/v1"] = (
        "trialagentbench.survival_qualification_receipt/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[SurvivalWorldReceiptV1, ...] = Field(min_length=2)


class SurvivalCurveResultV1(_FrozenModel):
    """Monte Carlo survival estimate for one dose, arm, and time."""

    dose_multiplier: float = Field(ge=0)
    arm_id: str = Field(min_length=1)
    time: float = Field(gt=0)
    source_survival: float | None = Field(default=None, ge=0, le=1)
    mean_survival: float = Field(ge=0, le=1)
    mean_cumulative_hazard: float = Field(ge=0, allow_inf_nan=False)
    interval_95_low: float = Field(ge=0, le=1)
    interval_95_high: float = Field(ge=0, le=1)
    predictive_50_low: float = Field(ge=0, le=1)
    predictive_50_high: float = Field(ge=0, le=1)
    predictive_95_low: float = Field(ge=0, le=1)
    predictive_95_high: float = Field(ge=0, le=1)


class SurvivalRiskSetResultV1(_FrozenModel):
    """Monte Carlo risk-set size for one dose, arm, and time."""

    dose_multiplier: float = Field(ge=0)
    arm_id: str = Field(min_length=1)
    time: float = Field(gt=0)
    source_at_risk: int | None = Field(default=None, ge=0)
    mean_at_risk: float = Field(ge=0, allow_inf_nan=False)
    predictive_50_low: float = Field(ge=0, allow_inf_nan=False)
    predictive_50_high: float = Field(ge=0, allow_inf_nan=False)
    predictive_95_low: float = Field(ge=0, allow_inf_nan=False)
    predictive_95_high: float = Field(ge=0, allow_inf_nan=False)


class SurvivalRmstResultV1(_FrozenModel):
    """Monte Carlo restricted mean survival time for one dose and arm."""

    dose_multiplier: float = Field(ge=0)
    arm_id: str = Field(min_length=1)
    source_rmst: float | None = Field(default=None, gt=0)
    mean_rmst: float = Field(gt=0, allow_inf_nan=False)
    bias: float | None = Field(default=None, allow_inf_nan=False)
    bias_ci_low: float | None = Field(default=None, allow_inf_nan=False)
    bias_ci_high: float | None = Field(default=None, allow_inf_nan=False)
    predictive_50_low: float = Field(gt=0, allow_inf_nan=False)
    predictive_50_high: float = Field(gt=0, allow_inf_nan=False)
    predictive_95_low: float = Field(gt=0, allow_inf_nan=False)
    predictive_95_high: float = Field(gt=0, allow_inf_nan=False)


class SurvivalDoseRecoveryV1(_FrozenModel):
    """Cox recovery at one treatment-effect dose."""

    dose_multiplier: float = Field(ge=0)
    truth_log_hazard_ratio: float = Field(allow_inf_nan=False)
    worlds: int = Field(ge=2)
    mean_log_hazard_ratio: float = Field(allow_inf_nan=False)
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
    mean_event_rate: float = Field(ge=0, le=1)
    mean_early_censoring_rate: float = Field(ge=0, le=1)


class SurvivalQualificationReportV1(_FrozenModel):
    """Independent native-scale survival qualification report."""

    schema_id: Literal["trialagentbench.survival_qualification_report/v1"] = (
        "trialagentbench.survival_qualification_report/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: int = Field(ge=2)
    curves: tuple[SurvivalCurveResultV1, ...] = Field(min_length=1)
    risk_sets: tuple[SurvivalRiskSetResultV1, ...] = Field(min_length=1)
    rmst: tuple[SurvivalRmstResultV1, ...] = Field(min_length=1)
    recovery: tuple[SurvivalDoseRecoveryV1, ...] = Field(min_length=3)
    source_dose_curve_mae: float = Field(ge=0, le=1)
    source_dose_risk_set_mae: float = Field(ge=0, allow_inf_nan=False)
    source_dose_rmst_mae: float = Field(ge=0, allow_inf_nan=False)
    intact_event_time_curve_mae: float = Field(ge=0, le=1)
    intact_event_time_curve_mae_ci_low: float = Field(ge=0, le=1)
    intact_event_time_curve_mae_ci_high: float = Field(ge=0, le=1)
    broken_event_time_curve_mae: float = Field(ge=0, le=1)
    broken_event_time_curve_mae_ci_low: float = Field(ge=0, le=1)
    broken_event_time_curve_mae_ci_high: float = Field(ge=0, le=1)
    broken_minus_intact_curve_mae: float = Field(ge=-1, le=1)
    broken_minus_intact_curve_mae_ci_low: float = Field(allow_inf_nan=False)
    broken_minus_intact_curve_mae_ci_high: float = Field(allow_inf_nan=False)
    dose_response_slope: float = Field(allow_inf_nan=False)
    dose_response_slope_ci_low: float = Field(allow_inf_nan=False)
    dose_response_slope_ci_high: float = Field(allow_inf_nan=False)
    monotone_world_fraction: float = Field(ge=0, le=1)
    monotone_world_fraction_ci_low: float = Field(ge=0, le=1)
    monotone_world_fraction_ci_high: float = Field(ge=0, le=1)


def evaluate_survival_qualification(
    *,
    release_dir: Path,
    minimum_worlds: int = 100,
) -> SurvivalQualificationReportV1:
    """Verify release identity and recompute survival evidence."""

    design_path = release_dir / "design.json"
    receipt_path = release_dir / "receipt.json"
    design = SurvivalQualificationDesignV1.model_validate_json(
        design_path.read_text(encoding="utf-8")
    )
    receipt = SurvivalQualificationReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    design_sha = _json_sha(json.loads(design_path.read_text(encoding="utf-8")))
    if receipt.design_sha256 != design_sha:
        raise ValueError("Survival receipt does not match its design")
    if len(receipt.worlds) != design.worlds or design.worlds < minimum_worlds:
        raise ValueError("Survival release has insufficient or inconsistent worlds")
    expected_paths = {item.analysis_path for item in receipt.worlds}
    actual_paths = {
        path.relative_to(release_dir).as_posix()
        for path in (release_dir / "worlds").glob("*.parquet")
    }
    if actual_paths != expected_paths:
        raise ValueError("Survival world inventory does not match the receipt")

    frames = []
    for item in receipt.worlds:
        path = release_dir / item.analysis_path
        if sha256_file(path) != item.analysis_sha256:
            raise ValueError(f"Survival world checksum mismatch: {item.world_id}")
        frame = pd.read_parquet(path)
        _validate_world(frame, design=design, world_id=item.world_id)
        frames.append(frame)
    values = pd.concat(frames, ignore_index=True)
    curves, risk_sets, rmst = _process_results(values, design=design)
    recovery, estimates = _recovery_results(values, design=design)
    source_curves = {
        (arm.arm_id, time): survival
        for arm in design.arms
        for time, survival in zip(design.time_grid, arm.survival_at_grid, strict=True)
    }
    source_dose = [row for row in curves if row.dose_multiplier == 1.0]
    source_mae = float(
        np.mean(
            [
                abs(row.mean_survival - source_curves[(row.arm_id, row.time)])
                for row in source_dose
            ]
        )
    )
    source_risk_mae = float(
        np.mean(
            [
                abs(row.mean_at_risk - cast(int, row.source_at_risk))
                for row in risk_sets
                if row.dose_multiplier == 1.0
            ]
        )
    )
    source_rmst_mae = float(
        np.mean(
            [
                abs(row.mean_rmst - cast(float, row.source_rmst))
                for row in rmst
                if row.dose_multiplier == 1.0
            ]
        )
    )
    intact_errors, broken_errors = _event_time_linkage_errors(values, design=design)
    intact_mae, intact_low, intact_high = _mean_ci(intact_errors)
    broken_mae, broken_low, broken_high = _mean_ci(broken_errors)
    error_difference, difference_low, difference_high = _mean_ci(
        broken_errors - intact_errors
    )
    world_slopes = []
    for _, world in estimates.groupby("world_id", sort=True):
        world_slopes.append(
            float(np.polyfit(world["dose_multiplier"], world["estimate"], deg=1)[0])
        )
    slope_mean, slope_low, slope_high = _mean_ci(np.asarray(world_slopes))
    expected_sign = np.sign(design.source_log_hazard_ratio)
    monotone = np.sign(world_slopes) == expected_sign
    monotone_fraction = float(np.mean(monotone))
    monotone_interval = proportion_interval(int(np.sum(monotone)), len(monotone))
    return SurvivalQualificationReportV1(
        design_sha256=design_sha,
        receipt_sha256=sha256_file(receipt_path),
        worlds=design.worlds,
        curves=tuple(curves),
        risk_sets=tuple(risk_sets),
        rmst=tuple(rmst),
        recovery=tuple(recovery),
        source_dose_curve_mae=source_mae,
        source_dose_risk_set_mae=source_risk_mae,
        source_dose_rmst_mae=source_rmst_mae,
        intact_event_time_curve_mae=intact_mae,
        intact_event_time_curve_mae_ci_low=intact_low,
        intact_event_time_curve_mae_ci_high=intact_high,
        broken_event_time_curve_mae=broken_mae,
        broken_event_time_curve_mae_ci_low=broken_low,
        broken_event_time_curve_mae_ci_high=broken_high,
        broken_minus_intact_curve_mae=error_difference,
        broken_minus_intact_curve_mae_ci_low=difference_low,
        broken_minus_intact_curve_mae_ci_high=difference_high,
        dose_response_slope=slope_mean,
        dose_response_slope_ci_low=slope_low,
        dose_response_slope_ci_high=slope_high,
        monotone_world_fraction=monotone_fraction,
        monotone_world_fraction_ci_low=monotone_interval[0],
        monotone_world_fraction_ci_high=monotone_interval[1],
    )


def _validate_world(
    frame: pd.DataFrame,
    *,
    design: SurvivalQualificationDesignV1,
    world_id: str,
) -> None:
    required = {"world_id", "dose_multiplier", "participant_id", "arm", "time", "event"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Survival world is missing columns: {missing}")
    if set(frame["world_id"].astype(str)) != {world_id}:
        raise ValueError("Survival world identity mismatch")
    if set(frame["dose_multiplier"].astype(float)) != set(design.dose_multipliers):
        raise ValueError("Survival world has an incomplete dose surface")
    for _, dose in frame.groupby("dose_multiplier", observed=True):
        if (
            len(dose) != design.participants
            or dose["participant_id"].duplicated().any()
        ):
            raise ValueError("Survival dose must contain one row per participant")
        if set(dose["arm"].astype(str)) != {
            design.control_arm_id,
            design.treatment_arm_id,
        }:
            raise ValueError("Survival dose has invalid randomized arms")
    time = frame["time"].to_numpy(dtype=float)
    event = frame["event"].to_numpy(dtype=int)
    if not np.isfinite(time).all() or np.any((time <= 0) | (time > design.horizon)):
        raise ValueError("Survival observations lie outside the analysis horizon")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("Survival event indicator must be binary")


def _process_results(
    values: pd.DataFrame,
    *,
    design: SurvivalQualificationDesignV1,
) -> tuple[
    list[SurvivalCurveResultV1],
    list[SurvivalRiskSetResultV1],
    list[SurvivalRmstResultV1],
]:
    source = {
        (arm.arm_id, time): survival
        for arm in design.arms
        for time, survival in zip(design.time_grid, arm.survival_at_grid, strict=True)
    }
    source_risk = {
        (arm.arm_id, time): at_risk
        for arm in design.arms
        for time, at_risk in zip(
            design.time_grid,
            arm.at_risk_at_grid,
            strict=True,
        )
    }
    source_rmst = {arm.arm_id: arm.rmst_at_horizon for arm in design.arms}
    estimates: dict[tuple[float, str, float], list[float]] = {}
    risk_estimates: dict[tuple[float, str, float], list[float]] = {}
    rmst_estimates: dict[tuple[float, str], list[float]] = {}
    for (_, dose, arm), group in values.groupby(
        ["world_id", "dose_multiplier", "arm"], observed=True, sort=True
    ):
        fit = KaplanMeierFitter().fit(group["time"], event_observed=group["event"])
        predicted = fit.predict(list(design.time_grid)).to_numpy(dtype=float)
        dose_value = cast(float, dose)
        for time, value in zip(design.time_grid, predicted, strict=True):
            estimates.setdefault((dose_value, str(arm), time), []).append(float(value))
            risk_estimates.setdefault((dose_value, str(arm), time), []).append(
                float(group["time"].ge(time).sum())
            )
        rmst_estimates.setdefault((dose_value, str(arm)), []).append(
            float(restricted_mean_survival_time(fit, t=design.horizon))
        )
    curves = []
    for (dose, arm, time), samples in sorted(estimates.items()):
        sample_array = np.asarray(samples)
        mean, low, high = _mean_ci(sample_array)
        predictive = np.quantile(sample_array, [0.025, 0.25, 0.75, 0.975])
        curves.append(
            SurvivalCurveResultV1(
                dose_multiplier=dose,
                arm_id=arm,
                time=time,
                source_survival=source[(arm, time)] if dose == 1.0 else None,
                mean_survival=mean,
                mean_cumulative_hazard=float(-np.log(max(mean, np.finfo(float).tiny))),
                interval_95_low=max(0.0, low),
                interval_95_high=min(1.0, high),
                predictive_50_low=float(predictive[1]),
                predictive_50_high=float(predictive[2]),
                predictive_95_low=float(predictive[0]),
                predictive_95_high=float(predictive[3]),
            )
        )
    risk_sets = []
    for (dose, arm, time), samples in sorted(risk_estimates.items()):
        sample_array = np.asarray(samples)
        predictive = np.quantile(sample_array, [0.025, 0.25, 0.75, 0.975])
        risk_sets.append(
            SurvivalRiskSetResultV1(
                dose_multiplier=dose,
                arm_id=arm,
                time=time,
                source_at_risk=source_risk[(arm, time)] if dose == 1.0 else None,
                mean_at_risk=float(np.mean(sample_array)),
                predictive_50_low=float(predictive[1]),
                predictive_50_high=float(predictive[2]),
                predictive_95_low=float(predictive[0]),
                predictive_95_high=float(predictive[3]),
            )
        )
    rmst = []
    for (dose, arm), samples in sorted(rmst_estimates.items()):
        sample_array = np.asarray(samples)
        predictive = np.quantile(sample_array, [0.025, 0.25, 0.75, 0.975])
        reference = source_rmst[arm] if dose == 1.0 else None
        interval: tuple[float | None, float | None, float | None] = (
            _mean_ci(sample_array - reference)
            if reference is not None
            else (None, None, None)
        )
        rmst_bias, rmst_low, rmst_high = interval
        rmst.append(
            SurvivalRmstResultV1(
                dose_multiplier=dose,
                arm_id=arm,
                source_rmst=reference,
                mean_rmst=float(np.mean(sample_array)),
                bias=rmst_bias,
                bias_ci_low=rmst_low,
                bias_ci_high=rmst_high,
                predictive_50_low=float(predictive[1]),
                predictive_50_high=float(predictive[2]),
                predictive_95_low=float(predictive[0]),
                predictive_95_high=float(predictive[3]),
            )
        )
    return curves, risk_sets, rmst


def _event_time_linkage_errors(
    values: pd.DataFrame,
    *,
    design: SurvivalQualificationDesignV1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return paired world-level curve errors before and after breaking linkage."""

    source = {
        (arm.arm_id, time): survival
        for arm in design.arms
        for time, survival in zip(
            design.time_grid,
            arm.survival_at_grid,
            strict=True,
        )
    }
    intact_world_errors: list[float] = []
    broken_world_errors: list[float] = []
    source_dose = values.loc[values["dose_multiplier"].eq(1.0)]
    for world_id, world in source_dose.groupby("world_id", observed=True, sort=True):
        intact_errors: list[float] = []
        broken_errors: list[float] = []
        for arm, group in world.groupby("arm", observed=True, sort=True):
            arm_id = str(arm)
            intact_fit = KaplanMeierFitter().fit(
                group["time"],
                event_observed=group["event"],
            )
            digest = hashlib.sha256(
                f"{world_id}:{arm_id}:broken-event-time".encode()
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            broken_fit = KaplanMeierFitter().fit(
                group["time"],
                event_observed=rng.permutation(group["event"].to_numpy()),
            )
            for time, intact, broken in zip(
                design.time_grid,
                intact_fit.predict(list(design.time_grid)).to_numpy(dtype=float),
                broken_fit.predict(list(design.time_grid)).to_numpy(dtype=float),
                strict=True,
            ):
                reference = source[(arm_id, time)]
                intact_errors.append(abs(float(intact) - reference))
                broken_errors.append(abs(float(broken) - reference))
        intact_world_errors.append(float(np.mean(intact_errors)))
        broken_world_errors.append(float(np.mean(broken_errors)))
    return (
        np.asarray(intact_world_errors, dtype=float),
        np.asarray(broken_world_errors, dtype=float),
    )


def _recovery_results(
    values: pd.DataFrame,
    *,
    design: SurvivalQualificationDesignV1,
) -> tuple[list[SurvivalDoseRecoveryV1], pd.DataFrame]:
    estimates = []
    for (world_id, dose), group in values.groupby(
        ["world_id", "dose_multiplier"], observed=True, sort=True
    ):
        analysis = group.assign(
            treatment=group["arm"].astype(str).eq(design.treatment_arm_id).astype(int)
        )[["time", "event", "treatment"]]
        fitted = CoxPHFitter().fit(
            analysis,
            duration_col="time",
            event_col="event",
            formula="treatment",
            robust=True,
        )
        estimate = float(fitted.params_.loc["treatment"])
        standard_error = float(fitted.standard_errors_.loc["treatment"])
        if (
            not math.isfinite(estimate)
            or not math.isfinite(standard_error)
            or standard_error <= 0
        ):
            raise ValueError("Cox analysis produced an invalid estimate")
        estimates.append(
            {
                "world_id": str(world_id),
                "dose_multiplier": cast(float, dose),
                "estimate": estimate,
                "standard_error": standard_error,
                "event_rate": float(group["event"].mean()),
                "early_censoring_rate": float(
                    ((group["event"] == 0) & (group["time"] < design.horizon)).mean()
                ),
            }
        )
    frame = pd.DataFrame(estimates)
    output = []
    for dose, rows in frame.groupby("dose_multiplier", observed=True, sort=True):
        dose_value = cast(float, dose)
        truth = dose_value * design.source_log_hazard_ratio
        bias_samples = rows["estimate"].to_numpy(dtype=float) - truth
        bias, bias_low, bias_high = _mean_ci(bias_samples)
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
            SurvivalDoseRecoveryV1(
                dose_multiplier=dose_value,
                truth_log_hazard_ratio=truth,
                worlds=len(rows),
                mean_log_hazard_ratio=cast(float, rows["estimate"].mean()),
                bias=bias,
                bias_ci_low=bias_low,
                bias_ci_high=bias_high,
                predictive_50_low=float(predictive[1]),
                predictive_50_high=float(predictive[2]),
                predictive_95_low=float(predictive[0]),
                predictive_95_high=float(predictive[3]),
                coverage=covered_count / len(rows),
                coverage_ci_low=coverage_interval[0],
                coverage_ci_high=coverage_interval[1],
                mean_event_rate=cast(float, rows["event_rate"].mean()),
                mean_early_censoring_rate=cast(
                    float, rows["early_censoring_rate"].mean()
                ),
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
    "SurvivalArmReferenceV1",
    "SurvivalQualificationDesignV1",
    "SurvivalQualificationReceiptV1",
    "SurvivalQualificationReportV1",
    "SurvivalRiskSetResultV1",
    "SurvivalRmstResultV1",
    "SurvivalWorldReceiptV1",
    "evaluate_survival_qualification",
]
