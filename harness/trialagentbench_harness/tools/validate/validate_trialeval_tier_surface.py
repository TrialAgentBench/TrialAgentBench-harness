"""Validate TrialEvalBench C-tier public-surface compatibility."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.analysis.tier_surface_compatibility import (
    write_trialeval_tier_surface_compatibility_artifacts_v1,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TrialEvalBench C-tier surface compatibility validator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-zip", required=True, help="Path to TrialEvalBench public zip.")
    parser.add_argument("--evaluator-zip", required=True, help="Path to TrialEvalBench evaluator zip.")
    parser.add_argument("--out-dir", required=True, help="Output directory for JSON and Markdown reports.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = write_trialeval_tier_surface_compatibility_artifacts_v1(
        public_zip=Path(args.public_zip),
        evaluator_zip=Path(args.evaluator_zip),
        out_dir=Path(args.out_dir),
    )
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
