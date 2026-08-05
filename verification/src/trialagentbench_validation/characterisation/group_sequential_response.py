"""Evaluate a released group-sequential monitoring plan."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from trialagentbench_validation.statistics import proportion_interval


def simulate_group_sequential_response(
    *,
    information_fractions: tuple[float, ...],
    efficacy_boundaries: tuple[float, ...],
    signal_to_final_boundary: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
    world_count: int = 500_000,
    seed: int = 825_451,
) -> pd.DataFrame:
    """Estimate rejection and stopping probabilities for one monitoring plan."""

    response, _ = _simulate_group_sequential_plan(
        information_fractions=information_fractions,
        efficacy_boundaries=efficacy_boundaries,
        signal_to_final_boundary=signal_to_final_boundary,
        world_count=world_count,
        seed=seed,
    )
    return response


def simulate_group_sequential_operating_characteristics(
    *,
    information_fractions: tuple[float, ...],
    efficacy_boundaries: tuple[float, ...],
    signal_to_final_boundary: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
    world_count: int = 500_000,
    seed: int = 825_451,
) -> pd.DataFrame:
    """Estimate realized-look inference for one monitoring plan."""

    _, operating_characteristics = _simulate_group_sequential_plan(
        information_fractions=information_fractions,
        efficacy_boundaries=efficacy_boundaries,
        signal_to_final_boundary=signal_to_final_boundary,
        world_count=world_count,
        seed=seed,
    )
    return operating_characteristics


def _simulate_group_sequential_plan(
    *,
    information_fractions: tuple[float, ...],
    efficacy_boundaries: tuple[float, ...],
    signal_to_final_boundary: tuple[float, ...],
    world_count: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    information = np.asarray(information_fractions, dtype=float)
    boundaries = np.asarray(efficacy_boundaries, dtype=float)
    signals = np.asarray(signal_to_final_boundary, dtype=float)
    if (
        information.ndim != 1
        or len(information) < 2
        or len(boundaries) != len(information)
        or not np.isfinite(information).all()
        or not np.isfinite(boundaries).all()
    ):
        raise ValueError(
            "group-sequential response requires matching finite look fractions and boundaries"
        )
    if (
        not np.all(np.diff(information) > 0)
        or not math.isclose(float(information[-1]), 1.0)
        or np.any((information <= 0.0) | (information > 1.0))
        or np.any(boundaries <= 0.0)
    ):
        raise ValueError(
            "information fractions must increase to one and boundaries must be positive"
        )
    if (
        signals.ndim != 1
        or len(signals) < 2
        or not np.isfinite(signals).all()
        or np.any(signals < 0.0)
        or not np.all(np.diff(signals) > 0.0)
        or signals[0] != 0.0
    ):
        raise ValueError(
            "signal levels must be finite, nonnegative, increasing, and begin at zero"
        )
    if world_count < 10_000:
        raise ValueError("group-sequential response requires at least 10,000 worlds")

    rng = np.random.default_rng(seed)
    increments = rng.normal(
        scale=np.sqrt(np.diff(np.concatenate(([0.0], information)))),
        size=(world_count, len(information)),
    )
    standardized_brownian_motion = np.cumsum(increments, axis=1) / np.sqrt(information)
    records: list[dict[str, float | int]] = []
    operating_records: list[dict[str, float | int]] = []
    for signal in signals:
        final_mean_z = float(signal * boundaries[-1])
        statistics = standardized_brownian_motion + final_mean_z * np.sqrt(information)
        crossing = np.abs(statistics) >= boundaries
        has_crossing = crossing.any(axis=1)
        first_crossing = np.where(has_crossing, crossing.argmax(axis=1), -1)
        cumulative = np.zeros(world_count, dtype=bool)
        for look_index, (fraction, boundary) in enumerate(
            zip(information, boundaries, strict=True),
        ):
            stopped = first_crossing == look_index
            cumulative |= stopped
            stop_count = int(np.count_nonzero(stopped))
            cumulative_count = int(np.count_nonzero(cumulative))
            stop_interval = proportion_interval(stop_count, world_count)
            cumulative_interval = proportion_interval(cumulative_count, world_count)
            records.append(
                {
                    "signal_to_final_boundary": float(signal),
                    "final_mean_z": final_mean_z,
                    "look": look_index + 1,
                    "information_fraction": float(fraction),
                    "efficacy_boundary_z": float(boundary),
                    "world_count": world_count,
                    "stopping_probability": stop_count / world_count,
                    "stopping_probability_ci_low": stop_interval[0],
                    "stopping_probability_ci_high": stop_interval[1],
                    "cumulative_rejection_probability": cumulative_count / world_count,
                    "cumulative_rejection_probability_ci_low": cumulative_interval[0],
                    "cumulative_rejection_probability_ci_high": cumulative_interval[1],
                }
            )
        realized_look = np.where(has_crossing, first_crossing, len(information) - 1)
        world_indices = np.arange(world_count)
        realized_information = information[realized_look]
        realized_boundary = boundaries[realized_look]
        estimates = statistics[world_indices, realized_look] / np.sqrt(
            realized_information
        )
        standard_errors = 1.0 / np.sqrt(realized_information)
        bias = estimates - final_mean_z
        bias_mean = float(np.mean(bias))
        bias_standard_error = float(stats.sem(bias))
        critical = float(stats.t.ppf(0.975, world_count - 1))
        repeated_covered = np.abs(bias) <= realized_boundary * standard_errors
        ordinary_covered = np.abs(bias) <= 1.96 * standard_errors
        repeated_interval = proportion_interval(
            int(np.count_nonzero(repeated_covered)), world_count
        )
        ordinary_interval = proportion_interval(
            int(np.count_nonzero(ordinary_covered)), world_count
        )
        rejection_interval = proportion_interval(
            int(np.count_nonzero(has_crossing)), world_count
        )
        early_stop = has_crossing & (first_crossing < len(information) - 1)
        early_stop_interval = proportion_interval(
            int(np.count_nonzero(early_stop)), world_count
        )
        operating_records.append(
            {
                "signal_to_final_boundary": float(signal),
                "truth_final_mean_z": final_mean_z,
                "world_count": world_count,
                "mean_realized_estimate": float(np.mean(estimates)),
                "mean_bias": bias_mean,
                "mean_bias_ci_low": bias_mean - critical * bias_standard_error,
                "mean_bias_ci_high": bias_mean + critical * bias_standard_error,
                "repeated_interval_coverage": float(np.mean(repeated_covered)),
                "repeated_interval_coverage_ci_low": repeated_interval[0],
                "repeated_interval_coverage_ci_high": repeated_interval[1],
                "ordinary_interval_coverage": float(np.mean(ordinary_covered)),
                "ordinary_interval_coverage_ci_low": ordinary_interval[0],
                "ordinary_interval_coverage_ci_high": ordinary_interval[1],
                "rejection_probability": float(np.mean(has_crossing)),
                "rejection_probability_ci_low": rejection_interval[0],
                "rejection_probability_ci_high": rejection_interval[1],
                "early_stop_probability": float(np.mean(early_stop)),
                "early_stop_probability_ci_low": early_stop_interval[0],
                "early_stop_probability_ci_high": early_stop_interval[1],
                "mean_information_fraction": float(np.mean(realized_information)),
                "failure_count": 0,
            }
        )
    return (
        pd.DataFrame.from_records(records),
        pd.DataFrame.from_records(operating_records),
    )


def write_group_sequential_response(
    *,
    output_path: Path,
    information_fractions: tuple[float, ...],
    efficacy_boundaries: tuple[float, ...],
) -> Path:
    """Write the group-sequential operating-characteristic response."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    result = simulate_group_sequential_response(
        information_fractions=information_fractions,
        efficacy_boundaries=efficacy_boundaries,
    )
    result.to_csv(output_path, index=False, lineterminator="\n")
    return output_path


def write_group_sequential_operating_characteristics(
    *,
    output_path: Path,
    information_fractions: tuple[float, ...],
    efficacy_boundaries: tuple[float, ...],
) -> Path:
    """Write realized-look group-sequential inference."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    result = simulate_group_sequential_operating_characteristics(
        information_fractions=information_fractions,
        efficacy_boundaries=efficacy_boundaries,
    )
    result.to_csv(output_path, index=False, lineterminator="\n")
    return output_path


def main() -> None:
    """Run a group-sequential response experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operating-characteristics-output", type=Path)
    parser.add_argument("--information-fractions", required=True, type=float, nargs="+")
    parser.add_argument("--efficacy-boundaries", required=True, type=float, nargs="+")
    args = parser.parse_args()
    write_group_sequential_response(
        output_path=args.output,
        information_fractions=tuple(args.information_fractions),
        efficacy_boundaries=tuple(args.efficacy_boundaries),
    )
    if args.operating_characteristics_output is not None:
        write_group_sequential_operating_characteristics(
            output_path=args.operating_characteristics_output,
            information_fractions=tuple(args.information_fractions),
            efficacy_boundaries=tuple(args.efficacy_boundaries),
        )


if __name__ == "__main__":
    main()


__all__ = [
    "simulate_group_sequential_operating_characteristics",
    "simulate_group_sequential_response",
    "write_group_sequential_operating_characteristics",
    "write_group_sequential_response",
]
