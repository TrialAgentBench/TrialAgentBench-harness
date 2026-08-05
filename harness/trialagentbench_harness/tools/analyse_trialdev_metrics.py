"""Summarize typed TrialDev programme assessments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from trialagentbench_harness.contracts.trialdev.metrics import TrialDevAssessmentPortfolioV1
from trialagentbench_harness.io.json import read_json_model, write_json_model
from trialagentbench_harness.trialdev.metrics import (
    TRIALDEV_METRIC_BOOTSTRAP_RESAMPLES_V1,
    TRIALDEV_METRIC_BOOTSTRAP_SEED_V1,
    compare_trialdev_conditions_v1,
    select_trialdev_calibration_v1,
    summarize_trialdev_metrics_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more typed assessment portfolios to analyse together.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-condition")
    parser.add_argument("--intervention-condition")
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--calibration-conditions", nargs="+")
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=TRIALDEV_METRIC_BOOTSTRAP_RESAMPLES_V1,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=TRIALDEV_METRIC_BOOTSTRAP_SEED_V1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write stream-specific metrics and an optional paired comparison."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    portfolios = tuple(read_json_model(TrialDevAssessmentPortfolioV1, path) for path in args.input)
    portfolio = TrialDevAssessmentPortfolioV1(
        programmes=tuple(programme for item in portfolios for programme in item.programmes)
    )
    summary = summarize_trialdev_metrics_v1(
        portfolio.programmes,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json_model(args.output, summary)
    comparison_requested = any(
        value is not None for value in (args.reference_condition, args.intervention_condition, args.comparison_output)
    )
    comparison_complete = all(
        value is not None for value in (args.reference_condition, args.intervention_condition, args.comparison_output)
    )
    if comparison_requested and not comparison_complete:
        raise ValueError(
            "A TrialDev comparison requires --reference-condition, --intervention-condition, and --comparison-output."
        )
    if comparison_complete:
        comparison = compare_trialdev_conditions_v1(
            portfolio.programmes,
            reference_condition_id=cast(str, args.reference_condition),
            intervention_condition_id=cast(str, args.intervention_condition),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        write_json_model(cast(Path, args.comparison_output), comparison)
    calibration_requested = args.calibration_conditions is not None or args.calibration_output is not None
    calibration_complete = args.calibration_conditions is not None and args.calibration_output is not None
    if calibration_requested and not calibration_complete:
        raise ValueError("Calibration selection requires --calibration-conditions and --calibration-output.")
    if calibration_complete:
        calibration = select_trialdev_calibration_v1(
            portfolio.programmes,
            condition_ids=cast(list[str], args.calibration_conditions),
        )
        write_json_model(cast(Path, args.calibration_output), calibration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
