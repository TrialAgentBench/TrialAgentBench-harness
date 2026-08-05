"""Validate TrialEvalBench public diagnostic proof surfaces."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.analysis.trialeval_diagnostic_proof_artifacts import (
    write_trialeval_diagnostic_proof_surface_artifacts_v1,
)
from trialagentbench_harness.execution_policy import TRIALEVAL_DIAGNOSTIC_PROOF_DEFAULT_WORKERS


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TrialEvalBench diagnostic proof-surface validator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-zip", required=True, help="Path to TrialEvalBench participant/public zip.")
    parser.add_argument("--evaluator-zip", required=True, help="Path to TrialEvalBench evaluator zip.")
    parser.add_argument("--out-dir", required=True, help="Output directory for JSON, CSV, SVG, and Markdown reports.")
    parser.add_argument(
        "--workers",
        type=int,
        default=TRIALEVAL_DIAGNOSTIC_PROOF_DEFAULT_WORKERS,
        help="Concurrent participant-diagnostic batches.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = write_trialeval_diagnostic_proof_surface_artifacts_v1(
        public_zip=Path(args.public_zip),
        evaluator_zip=Path(args.evaluator_zip),
        out_dir=Path(args.out_dir),
        workers=args.workers,
    )
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
