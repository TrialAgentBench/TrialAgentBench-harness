"""Measure period-adjusted inference under graded stepped-wedge trends."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy import stats

from trialagentbench_validation.characterisation.cluster_response import (
    SOURCE_ANCHORED_SETTING,
)
from trialagentbench_validation.statistics import proportion_interval


@dataclass(frozen=True, slots=True)
class SteppedWedgeResponseSetting:
    """One interpretable end-to-start secular hazard ratio."""

    setting: str
    secular_hazard_ratio_period_4_to_1: float

    @property
    def secular_log_hazard_range(self) -> float:
        """Return the end-to-start secular log-hazard range."""

        return math.log(self.secular_hazard_ratio_period_4_to_1)


SETTINGS = (
    SteppedWedgeResponseSetting("zero", 1.0),
    SteppedWedgeResponseSetting("low", 1.2),
    SteppedWedgeResponseSetting("benchmark", 1.65),
    SteppedWedgeResponseSetting("moderate", 2.0),
    SteppedWedgeResponseSetting("strong", 3.0),
)


def simulate_stepped_wedge_response(
    *,
    world_count: int = 1_000,
    cluster_count: int = 80,
    participants_per_cluster_period: int = 30,
    event_probability: float = 0.08,
    treatment_hazard_ratio: float = math.exp(-0.35),
    cluster_hazard_ratio_90_to_10: float = SOURCE_ANCHORED_SETTING.hazard_ratio_90_to_10,
    seed: int = 918_220,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate period-adjusted and period-omitting treatment inference."""

    if world_count < 3:
        raise ValueError("stepped-wedge response requires at least three worlds")
    if cluster_count < 8 or cluster_count % 4:
        raise ValueError("stepped-wedge response requires a multiple of four clusters")
    if participants_per_cluster_period < 2:
        raise ValueError(
            "stepped-wedge response requires at least two participants per cluster-period"
        )
    if not math.isfinite(event_probability) or not 0 < event_probability < 1:
        raise ValueError(
            "stepped-wedge event probability must lie strictly between zero and one"
        )
    if not math.isfinite(treatment_hazard_ratio) or treatment_hazard_ratio <= 0:
        raise ValueError(
            "stepped-wedge treatment hazard ratio must be positive and finite"
        )
    if (
        not math.isfinite(cluster_hazard_ratio_90_to_10)
        or cluster_hazard_ratio_90_to_10 < 1
    ):
        raise ValueError(
            "stepped-wedge cluster hazard ratio must be finite and at least one"
        )

    periods = np.tile(np.arange(4), cluster_count)
    cluster_ids = np.repeat(np.arange(cluster_count), 4)
    treatment_log_hazard_ratio = math.log(treatment_hazard_ratio)
    cluster_log_hazard_sd = math.log(cluster_hazard_ratio_90_to_10) / (
        2.0 * stats.norm.ppf(0.90)
    )
    baseline_log_hazard = math.log(-math.log1p(-event_probability))
    critical = float(stats.t.ppf(0.975, cluster_count - 1))
    records: list[dict[str, int | float | str]] = []
    for world_id in range(1, world_count + 1):
        rng = np.random.default_rng(seed + world_id)
        adoption_period = rng.permutation(np.repeat(np.arange(4), cluster_count // 4))
        treated = (periods >= np.repeat(adoption_period, 4)).astype(float)
        cluster_effect = np.repeat(
            rng.normal(scale=cluster_log_hazard_sd, size=cluster_count), 4
        )
        uniforms = rng.uniform(
            size=(cluster_count * 4, participants_per_cluster_period)
        )
        for setting in SETTINGS:
            period_effect = periods * setting.secular_log_hazard_range / 3.0
            event_risk = 1.0 - np.exp(
                -np.exp(
                    baseline_log_hazard
                    + cluster_effect
                    + period_effect
                    + treatment_log_hazard_ratio * treated
                )
            )
            event_count = np.sum(uniforms < event_risk[:, None], axis=1)
            event_fraction = event_count / participants_per_cluster_period
            period_indicators = pd.get_dummies(
                periods, drop_first=True, dtype=float
            ).to_numpy()
            adjusted_design = np.column_stack(
                (np.ones(len(treated)), treated, period_indicators)
            )
            naive_design = np.column_stack((np.ones(len(treated)), treated))
            adjusted = _fit_clustered_risk_difference(
                event_fraction=event_fraction,
                design=adjusted_design,
                standardization_design=period_indicators,
                cluster_ids=cluster_ids,
                participants_per_row=participants_per_cluster_period,
            )
            naive = _fit_clustered_risk_difference(
                event_fraction=event_fraction,
                design=naive_design,
                standardization_design=np.empty((1, 0), dtype=np.float64),
                cluster_ids=cluster_ids,
                participants_per_row=participants_per_cluster_period,
            )
            control_risk = 1.0 - np.exp(
                -np.exp(baseline_log_hazard + cluster_effect + period_effect)
            )
            treated_risk = 1.0 - np.exp(
                -np.exp(
                    baseline_log_hazard
                    + cluster_effect
                    + period_effect
                    + treatment_log_hazard_ratio
                )
            )
            true_risk_difference = float(np.mean(treated_risk - control_risk))
            records.append(
                {
                    "setting": setting.setting,
                    "secular_log_hazard_range": setting.secular_log_hazard_range,
                    "secular_hazard_ratio_period_4_to_1": setting.secular_hazard_ratio_period_4_to_1,
                    "world_id": world_id,
                    "cluster_count": cluster_count,
                    "participants_per_cluster_period": participants_per_cluster_period,
                    "participant_count": cluster_count
                    * 4
                    * participants_per_cluster_period,
                    "treatment_log_hazard_ratio": treatment_log_hazard_ratio,
                    "true_risk_difference": true_risk_difference,
                    "cluster_log_hazard_sd": cluster_log_hazard_sd,
                    "cluster_hazard_ratio_90_to_10": cluster_hazard_ratio_90_to_10,
                    "event_rate": float(np.mean(event_fraction)),
                    "period_adjusted_estimate": adjusted[0],
                    "period_adjusted_bias": adjusted[0] - true_risk_difference,
                    "period_adjusted_interval_width": 2.0 * critical * adjusted[1],
                    "period_adjusted_covered": int(
                        abs(adjusted[0] - true_risk_difference)
                        <= critical * adjusted[1]
                    ),
                    "period_omitting_estimate": naive[0],
                    "period_omitting_bias": naive[0] - true_risk_difference,
                    "period_omitting_interval_width": 2.0 * critical * naive[1],
                    "period_omitting_covered": int(
                        abs(naive[0] - true_risk_difference) <= critical * naive[1]
                    ),
                }
            )
    worlds = pd.DataFrame.from_records(records)
    summaries: list[dict[str, int | float | str]] = []
    for keys, group in worlds.groupby(
        ["setting", "secular_log_hazard_range", "secular_hazard_ratio_period_4_to_1"],
        sort=False,
    ):
        setting_name, log_range, hazard_ratio = keys
        for measure, unit in (
            ("event_rate", "proportion"),
            ("true_risk_difference", "probability difference"),
            ("period_adjusted_estimate", "probability difference"),
            ("period_adjusted_bias", "probability difference"),
            ("period_adjusted_interval_width", "probability difference"),
            ("period_adjusted_covered", "proportion"),
            ("period_omitting_estimate", "probability difference"),
            ("period_omitting_bias", "probability difference"),
            ("period_omitting_interval_width", "probability difference"),
            ("period_omitting_covered", "proportion"),
        ):
            values = group[measure].to_numpy(dtype=float)
            mean = float(np.mean(values))
            if measure.endswith("_covered"):
                successes = int(np.sum(values))
                interval_low, interval_high = proportion_interval(
                    successes, len(values)
                )
                interval_low = 0.0 if successes == 0 else max(0.0, interval_low)
                interval_high = (
                    1.0 if successes == len(values) else min(1.0, interval_high)
                )
                standard_error = math.sqrt(mean * (1.0 - mean) / len(values))
            else:
                standard_error = float(stats.sem(values))
                margin = float(stats.t.ppf(0.975, len(values) - 1)) * standard_error
                interval_low, interval_high = mean - margin, mean + margin
            summaries.append(
                {
                    "setting": str(setting_name),
                    "secular_log_hazard_range": cast(float, log_range),
                    "secular_hazard_ratio_period_4_to_1": cast(float, hazard_ratio),
                    "measure": measure,
                    "unit": unit,
                    "world_count": len(values),
                    "participant_count": cluster_count
                    * 4
                    * participants_per_cluster_period,
                    "cluster_count": cluster_count,
                    "cluster_log_hazard_sd": cluster_log_hazard_sd,
                    "cluster_hazard_ratio_90_to_10": cluster_hazard_ratio_90_to_10,
                    "mean": mean,
                    "standard_error": standard_error,
                    "ci_low": interval_low,
                    "ci_high": interval_high,
                    "median": float(np.median(values)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q95": float(np.quantile(values, 0.95)),
                    "failure_count": 0,
                }
            )
    return worlds, pd.DataFrame.from_records(summaries)


def _fit_clustered_risk_difference(
    *,
    event_fraction: NDArray[np.float64],
    design: NDArray[np.float64],
    standardization_design: NDArray[np.float64],
    cluster_ids: NDArray[np.int64],
    participants_per_row: int,
) -> tuple[float, float]:
    """Fit a complementary-log-log model and standardize the event-risk difference."""

    fit = sm.GLM(
        np.column_stack(
            (
                np.rint(event_fraction * participants_per_row),
                participants_per_row - np.rint(event_fraction * participants_per_row),
            )
        ),
        design,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_ids, "use_correction": True},
        use_t=True,
    )
    coefficients = np.asarray(fit.params, dtype=np.float64)
    covariance = np.asarray(fit.cov_params(), dtype=np.float64)
    expected_columns = standardization_design.shape[1] + 2
    if coefficients.shape != (expected_columns,) or covariance.shape != (
        expected_columns,
        expected_columns,
    ):
        raise ValueError("stepped-wedge analysis produced an invalid model shape")
    control_design = np.column_stack(
        (
            np.ones(len(standardization_design), dtype=np.float64),
            np.zeros(len(standardization_design), dtype=np.float64),
            standardization_design,
        )
    )
    treated_design = control_design.copy()
    treated_design[:, 1] = 1.0
    control_linear_predictor = control_design @ coefficients
    treated_linear_predictor = treated_design @ coefficients
    control_hazard = np.exp(control_linear_predictor)
    treated_hazard = np.exp(treated_linear_predictor)
    control_risk = 1.0 - np.exp(-control_hazard)
    treated_risk = 1.0 - np.exp(-treated_hazard)
    estimate = float(np.mean(treated_risk - control_risk))
    gradient = np.mean(
        treated_design * (treated_hazard * np.exp(-treated_hazard))[:, None]
        - control_design * (control_hazard * np.exp(-control_hazard))[:, None],
        axis=0,
    )
    variance = float(gradient @ covariance @ gradient)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    if (
        not math.isfinite(estimate)
        or not math.isfinite(standard_error)
        or standard_error <= 0
    ):
        raise ValueError(
            "stepped-wedge analysis produced an invalid standardized treatment effect"
        )
    return estimate, standard_error


def write_stepped_wedge_response(output_dir: Path) -> tuple[Path, Path]:
    """Write complete matched worlds and their summary table."""

    output_dir.mkdir(parents=True, exist_ok=False)
    worlds, summary = simulate_stepped_wedge_response()
    worlds_path = output_dir / "stepped_wedge_response_worlds.csv"
    summary_path = output_dir / "stepped_wedge_response_summary.csv"
    worlds.to_csv(worlds_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    return worlds_path, summary_path


def main() -> None:
    """Run the deterministic stepped-wedge response experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    write_stepped_wedge_response(args.output_dir)


if __name__ == "__main__":
    main()


__all__ = [
    "SETTINGS",
    "SteppedWedgeResponseSetting",
    "simulate_stepped_wedge_response",
    "write_stepped_wedge_response",
]
