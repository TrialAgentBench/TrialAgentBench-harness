"""Reproduce the event-scale response to increasing cluster heterogeneity."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from trialagentbench_validation.characterisation.cluster_statistics import (
    one_way_cluster_information,
)
from trialagentbench_validation.statistics import proportion_interval

_NORMAL_90_TO_10_SPAN = 2.0 * NormalDist().inv_cdf(0.90)
_TARGET_EVENT_PROBABILITY = 0.035
_TARGET_EVENT_ICC = 0.011


@dataclass(frozen=True, slots=True)
class ClusterResponseSetting:
    """One interpretable level of cluster heterogeneity."""

    setting: str
    hazard_ratio_90_to_10: float

    @property
    def log_hazard_sd(self) -> float:
        """Return the normal cluster-effect standard deviation."""

        return math.log(self.hazard_ratio_90_to_10) / _NORMAL_90_TO_10_SPAN


_SOURCE_ANCHORED_LOG_HAZARD_SD = math.sqrt(
    math.log1p(
        _TARGET_EVENT_ICC
        * (1.0 - _TARGET_EVENT_PROBABILITY)
        / _TARGET_EVENT_PROBABILITY
    )
)
SOURCE_ANCHORED_SETTING = ClusterResponseSetting(
    "source_anchored",
    math.exp(_NORMAL_90_TO_10_SPAN * _SOURCE_ANCHORED_LOG_HAZARD_SD),
)
SETTINGS = (
    ClusterResponseSetting("zero", 1.0),
    ClusterResponseSetting("low", 2.0),
    SOURCE_ANCHORED_SETTING,
    ClusterResponseSetting("moderate", 5.0),
    ClusterResponseSetting("strong", 7.5),
)


def simulate_cluster_response(
    *,
    world_count: int = 1_000,
    cluster_count: int = 80,
    cluster_size: int = 85,
    follow_up_days: float = 280.0,
    event_probability: float = _TARGET_EVENT_PROBABILITY,
    treatment_hazard_ratio: float = 0.8,
    seed: int = 718_220,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate cluster dependence and risk-difference interval coverage."""

    if world_count < 3:
        raise ValueError("cluster response requires at least three worlds")
    if cluster_count < 4 or cluster_count % 2:
        raise ValueError(
            "cluster response requires an even number of at least four clusters"
        )
    if cluster_size < 2:
        raise ValueError(
            "cluster response requires at least two participants per cluster"
        )
    if not math.isfinite(follow_up_days) or follow_up_days <= 0:
        raise ValueError("cluster response follow-up must be positive and finite")
    if not math.isfinite(event_probability) or not 0 < event_probability < 1:
        raise ValueError(
            "cluster response event probability must lie strictly between zero and one"
        )
    if not math.isfinite(treatment_hazard_ratio) or treatment_hazard_ratio <= 0:
        raise ValueError(
            "cluster response treatment hazard ratio must be positive and finite"
        )

    participant_count = cluster_count * cluster_size
    cluster_ids = np.repeat(np.arange(cluster_count), cluster_size)
    baseline_hazard = -math.log1p(-event_probability) / follow_up_days
    treatment_log_hazard_ratio = math.log(treatment_hazard_ratio)
    cluster_critical = float(stats.t.ppf(0.975, cluster_count - 1))
    participant_critical = float(stats.t.ppf(0.975, participant_count - 2))
    records: list[dict[str, int | float | str]] = []
    for world_id in range(1, world_count + 1):
        rng = np.random.default_rng(seed + world_id)
        cluster_z = rng.normal(size=cluster_count)
        event_uniform = rng.uniform(size=participant_count)
        randomized_arm = np.repeat(
            rng.permutation(np.repeat((0, 1), cluster_count // 2)),
            cluster_size,
        )
        for setting in SETTINGS:
            cluster_effect = np.repeat(cluster_z * setting.log_hazard_sd, cluster_size)
            control_risk = 1.0 - np.exp(
                -follow_up_days * baseline_hazard * np.exp(cluster_effect)
            )
            treated_risk = 1.0 - np.exp(
                -follow_up_days
                * baseline_hazard
                * np.exp(cluster_effect + treatment_log_hazard_ratio)
            )
            truth = float(np.mean(treated_risk - control_risk))
            event_risk = np.where(randomized_arm == 1, treated_risk, control_risk)
            events = (event_uniform < event_risk).astype(float)
            information = one_way_cluster_information(
                pd.Series(events),
                pd.Series(cluster_ids),
                strata=pd.Series(randomized_arm),
            )
            model = sm.OLS(events, sm.add_constant(randomized_arm)).fit()
            cluster_model = model.get_robustcov_results(
                cov_type="cluster",
                groups=cluster_ids,
                use_correction=True,
                df_correction=True,
                use_t=True,
            )
            estimate = float(model.params[1])
            cluster_standard_error = float(cluster_model.bse[1])
            participant_standard_error = float(model.bse[1])
            cluster_half_width = cluster_critical * cluster_standard_error
            participant_half_width = participant_critical * participant_standard_error
            records.append(
                {
                    "setting": setting.setting,
                    "cluster_log_hazard_sd": setting.log_hazard_sd,
                    "hazard_ratio_90_to_10": setting.hazard_ratio_90_to_10,
                    "world_id": world_id,
                    "participant_count": participant_count,
                    "cluster_count": cluster_count,
                    "treatment_hazard_ratio": treatment_hazard_ratio,
                    "event_rate": float(np.mean(events)),
                    "event_intraclass_correlation": information.intraclass_correlation,
                    "event_variance_inflation": information.variance_inflation,
                    "true_risk_difference": truth,
                    "estimated_risk_difference": estimate,
                    "risk_difference_bias": estimate - truth,
                    "cluster_robust_interval_width": 2.0 * cluster_half_width,
                    "participant_independent_interval_width": 2.0
                    * participant_half_width,
                    "cluster_robust_covered": int(
                        abs(estimate - truth) <= cluster_half_width
                    ),
                    "participant_independent_covered": int(
                        abs(estimate - truth) <= participant_half_width
                    ),
                }
            )
    worlds = pd.DataFrame.from_records(records)
    summaries: list[dict[str, int | float | str]] = []
    for (setting_value, log_sd_value, hazard_ratio_value), group in worlds.groupby(
        ["setting", "cluster_log_hazard_sd", "hazard_ratio_90_to_10"],
        sort=False,
    ):
        setting_name = str(setting_value)
        log_hazard_sd = cast(float, log_sd_value)
        hazard_ratio_span = cast(float, hazard_ratio_value)
        for measure, unit in (
            ("event_rate", "proportion"),
            ("event_intraclass_correlation", "correlation coefficient"),
            ("event_variance_inflation", "variance ratio"),
            ("risk_difference_bias", "probability difference"),
            ("cluster_robust_interval_width", "probability difference"),
            ("participant_independent_interval_width", "probability difference"),
            ("cluster_robust_covered", "proportion"),
            ("participant_independent_covered", "proportion"),
        ):
            values = group[measure].to_numpy(dtype=float)
            mean = float(np.mean(values))
            if measure.endswith("_covered"):
                interval_low, interval_high = proportion_interval(
                    int(np.sum(values)), len(values)
                )
                standard_error = math.sqrt(mean * (1.0 - mean) / len(values))
            else:
                standard_error = float(stats.sem(values))
                critical = float(stats.t.ppf(0.975, len(values) - 1))
                interval_low = mean - critical * standard_error
                interval_high = mean + critical * standard_error
            summaries.append(
                {
                    "setting": setting_name,
                    "cluster_log_hazard_sd": log_hazard_sd,
                    "hazard_ratio_90_to_10": hazard_ratio_span,
                    "measure": measure,
                    "unit": unit,
                    "world_count": len(values),
                    "participant_count": participant_count,
                    "cluster_count": cluster_count,
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


def write_cluster_response(output_dir: Path) -> tuple[Path, Path]:
    """Write the complete matched worlds and summary table."""

    output_dir.mkdir(parents=True, exist_ok=False)
    worlds, summary = simulate_cluster_response()
    worlds_path = output_dir / "cluster_response_worlds.csv"
    summary_path = output_dir / "cluster_response_summary.csv"
    worlds.to_csv(worlds_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    return worlds_path, summary_path


def main() -> None:
    """Run the deterministic cluster-response experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    write_cluster_response(args.output_dir)


if __name__ == "__main__":
    main()


__all__ = [
    "ClusterResponseSetting",
    "SOURCE_ANCHORED_SETTING",
    "SETTINGS",
    "simulate_cluster_response",
    "write_cluster_response",
]
