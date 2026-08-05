"""Validate TrialEvalBench participant-context sufficiency."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.analysis.context_sufficiency import (
    write_trialeval_context_sufficiency_artifacts_v1,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TrialEvalBench context-sufficiency validator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-zip", required=True, help="Path to TrialEvalBench participant/public zip.")
    parser.add_argument("--evaluator-zip", required=True, help="Path to TrialEvalBench evaluator zip.")
    parser.add_argument("--out-dir", required=True, help="Output directory for JSON and Markdown reports.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = write_trialeval_context_sufficiency_artifacts_v1(
        public_zip=Path(args.public_zip),
        evaluator_zip=Path(args.evaluator_zip),
        out_dir=Path(args.out_dir),
    )
    return 0 if report.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
