"""Source-neutral fingerprints for repeated-measure clinical trial panels."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class LongitudinalMarginalMomentV1(BaseModel):
    """Aggregate distribution of one arm at one scheduled time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str = Field(min_length=1)
    time: float = Field(allow_inf_nan=False)
    observations: int = Field(ge=3)
    mean: float = Field(allow_inf_nan=False)
    standard_deviation: float = Field(gt=0, allow_inf_nan=False)
    q10: float = Field(allow_inf_nan=False)
    median: float = Field(allow_inf_nan=False)
    q90: float = Field(allow_inf_nan=False)


class LongitudinalTrialFingerprintV1(BaseModel):
    """Aggregate structure of one participant-level longitudinal trial."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.longitudinal_trial_fingerprint/v1"] = (
        "trialagentbench.longitudinal_trial_fingerprint/v1"
    )
    trial_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    measurement: str = Field(min_length=1)
    measurement_unit: str = Field(min_length=1)
    time_unit: str = Field(min_length=1)
    participants: int = Field(ge=20)
    arms: int = Field(ge=2)
    timepoints: int = Field(ge=3)
    potential_observations: int = Field(ge=60)
    observed_measurements: int = Field(ge=1)
    observation_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    observed_timepoints_mean: float = Field(gt=0, allow_inf_nan=False)
    observed_timepoints_sd: float = Field(ge=0, allow_inf_nan=False)
    observed_timepoints_min: int = Field(ge=0)
    observed_timepoints_max: int = Field(ge=1)
    followup_mean: float = Field(ge=0, allow_inf_nan=False)
    followup_sd: float = Field(ge=0, allow_inf_nan=False)
    adjacent_measurement_correlation: float = Field(ge=-1, le=1, allow_inf_nan=False)
    within_stratum_adjacent_correlation: float = Field(ge=-1, le=1, allow_inf_nan=False)
    adjacent_pairs: int = Field(ge=3)
    baseline_final_correlation: float = Field(ge=-1, le=1, allow_inf_nan=False)
    within_stratum_baseline_final_correlation: float = Field(
        ge=-1, le=1, allow_inf_nan=False
    )
    baseline_final_pairs: int = Field(ge=3)
    baseline_final_change_mean: float = Field(allow_inf_nan=False)
    baseline_final_change_sd: float = Field(ge=0, allow_inf_nan=False)
    marginal_moments: tuple[LongitudinalMarginalMomentV1, ...] = Field(min_length=6)


def fingerprint_longitudinal_trial(
    frame: pd.DataFrame,
    *,
    trial_id: str,
    source: str,
    measurement: str,
    measurement_unit: str,
    time_unit: str,
) -> LongitudinalTrialFingerprintV1:
    """
    Summarize a standardized participant-by-time repeated-measure panel.

    Parameters
    ----------
    frame
        Table with ``participant_id``, ``arm``, ``time``, and ``value``.
        Missing measurements remain explicit rows with a missing ``value``.
    trial_id
        Stable public trial identifier.
    source
        Stable public source-system identifier.
    measurement
        Clinical construct represented by ``value``.
    measurement_unit
        Unit or scale for ``value``.
    time_unit
        Unit or declared ordering scale for ``time``.

    Returns
    -------
    LongitudinalTrialFingerprintV1
        Aggregate trial-process and repeated-measure fingerprint.

    Raises
    ------
    ValueError
        If participant identity, arm assignment, timing, or repeated-measure
        support is invalid or insufficient.
    """

    required = {"participant_id", "arm", "time", "value"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Longitudinal frame is missing columns: {missing}")
    values = frame.loc[:, ["participant_id", "arm", "time", "value"]].copy()
    if values[["participant_id", "arm", "time"]].isna().any(axis=None):
        raise ValueError("Longitudinal identity, arm, and time must be complete")
    values["participant_id"] = values["participant_id"].astype("string")
    values["arm"] = values["arm"].astype("string")
    values["time"] = pd.to_numeric(values["time"], errors="raise").astype(float)
    values["value"] = pd.to_numeric(values["value"], errors="coerce").astype(float)
    if not np.isfinite(values["time"]).all():
        raise ValueError("Longitudinal time values must be finite")
    if values.duplicated(["participant_id", "time"]).any():
        raise ValueError(
            "Longitudinal frame must contain at most one row per participant and time"
        )
    arm_counts = values.groupby("participant_id", sort=False)["arm"].nunique()
    if not arm_counts.eq(1).all():
        raise ValueError("Each longitudinal participant must have exactly one arm")

    participants = int(values["participant_id"].nunique())
    arms = int(values["arm"].nunique())
    times = np.sort(values["time"].unique())
    if participants < 20 or arms < 2 or len(times) < 3:
        raise ValueError(
            "Longitudinal fingerprint requires 20 participants, two arms, and three timepoints"
        )
    observed = values.dropna(subset=["value"]).sort_values(
        ["participant_id", "time"],
        kind="mergesort",
    )
    if observed.empty:
        raise ValueError("Longitudinal frame contains no observed measurements")
    observed_counts = (
        observed.groupby("participant_id", sort=False)
        .size()
        .reindex(
            values["participant_id"].drop_duplicates(),
            fill_value=0,
        )
    )
    followup = (
        observed.groupby("participant_id", sort=False)["time"].max()
        - observed.groupby("participant_id", sort=False)["time"].min()
    ).reindex(values["participant_id"].drop_duplicates(), fill_value=0.0)

    adjacent = observed.copy()
    adjacent["next_value"] = adjacent.groupby("participant_id", sort=False)[
        "value"
    ].shift(-1)
    adjacent = adjacent.dropna(subset=["next_value"])
    if len(adjacent) < 3:
        raise ValueError(
            "Longitudinal frame has fewer than three adjacent measurement pairs"
        )

    pivot = observed.pivot(index="participant_id", columns="time", values="value")
    paired = pivot.loc[:, [times[0], times[-1]]].dropna()
    if len(paired) < 3:
        raise ValueError("Longitudinal frame has fewer than three baseline-final pairs")
    change = paired[times[-1]] - paired[times[0]]

    adjacent_correlation = _correlation(
        adjacent["value"].to_numpy(dtype=float),
        adjacent["next_value"].to_numpy(dtype=float),
        label="adjacent measurements",
    )
    baseline_final_correlation = _correlation(
        paired[times[0]].to_numpy(dtype=float),
        paired[times[-1]].to_numpy(dtype=float),
        label="baseline-final measurements",
    )
    observed["_centered_value"] = observed["value"] - observed.groupby(
        ["arm", "time"],
        observed=True,
    )["value"].transform("mean")
    observed["_next_centered_value"] = observed.groupby("participant_id", sort=False)[
        "_centered_value"
    ].shift(-1)
    centered_adjacent = observed.dropna(subset=["_next_centered_value"])
    within_stratum_adjacent_correlation = _correlation(
        centered_adjacent["_centered_value"].to_numpy(dtype=float),
        centered_adjacent["_next_centered_value"].to_numpy(dtype=float),
        label="within-stratum adjacent measurements",
    )
    centered_pivot = observed.pivot(
        index="participant_id", columns="time", values="_centered_value"
    )
    centered_paired = centered_pivot.loc[:, [times[0], times[-1]]].dropna()
    within_stratum_baseline_final_correlation = _correlation(
        centered_paired[times[0]].to_numpy(dtype=float),
        centered_paired[times[-1]].to_numpy(dtype=float),
        label="within-stratum baseline-final measurements",
    )
    marginal_moments = tuple(
        LongitudinalMarginalMomentV1(
            arm_id=str(arm_id),
            time=float(str(time)),
            observations=len(group),
            mean=float(group["value"].mean()),
            standard_deviation=float(group["value"].std(ddof=1)),
            q10=float(group["value"].quantile(0.1)),
            median=float(group["value"].median()),
            q90=float(group["value"].quantile(0.9)),
        )
        for (arm_id, time), group in observed.groupby(
            ["arm", "time"], observed=True, sort=True
        )
    )
    potential = participants * len(times)
    return LongitudinalTrialFingerprintV1(
        trial_id=trial_id,
        source=source,
        measurement=measurement,
        measurement_unit=measurement_unit,
        time_unit=time_unit,
        participants=participants,
        arms=arms,
        timepoints=len(times),
        potential_observations=potential,
        observed_measurements=len(observed),
        observation_fraction=len(observed) / potential,
        observed_timepoints_mean=float(observed_counts.mean()),
        observed_timepoints_sd=float(observed_counts.std(ddof=0)),
        observed_timepoints_min=int(observed_counts.min()),
        observed_timepoints_max=int(observed_counts.max()),
        followup_mean=float(followup.mean()),
        followup_sd=float(followup.std(ddof=0)),
        adjacent_measurement_correlation=adjacent_correlation,
        within_stratum_adjacent_correlation=within_stratum_adjacent_correlation,
        adjacent_pairs=len(adjacent),
        baseline_final_correlation=baseline_final_correlation,
        within_stratum_baseline_final_correlation=within_stratum_baseline_final_correlation,
        baseline_final_pairs=len(paired),
        baseline_final_change_mean=float(change.mean()),
        baseline_final_change_sd=float(change.std(ddof=0)),
        marginal_moments=marginal_moments,
    )


def _correlation(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
    *,
    label: str,
) -> float:
    if float(np.std(left, ddof=0)) <= 0 or float(np.std(right, ddof=0)) <= 0:
        raise ValueError(f"{label} must vary on both margins")
    result = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(result):
        raise ValueError(f"{label} correlation is not finite")
    return result


__all__ = [
    "LongitudinalMarginalMomentV1",
    "LongitudinalTrialFingerprintV1",
    "fingerprint_longitudinal_trial",
]
