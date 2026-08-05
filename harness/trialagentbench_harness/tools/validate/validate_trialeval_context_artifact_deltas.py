"""Validate TrialEvalBench matched-context artifact semantics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.analysis.context_artifact_deltas import (
    write_trialeval_context_artifact_deltas_v1,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TrialEvalBench matched-context artifact validator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-zip", required=True, help="Path to TrialEvalBench participant/public zip.")
    parser.add_argument("--evaluator-zip", required=True, help="Path to TrialEvalBench evaluator zip.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON report path; a human-readable Markdown report is written beside it.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = write_trialeval_context_artifact_deltas_v1(
        public_zip=Path(args.public_zip),
        evaluator_zip=Path(args.evaluator_zip),
        output_path=Path(args.output),
    )
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
