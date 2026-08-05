"""Export reproducible TrialDev worked programmes for scientific inspection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.trialdev.worked_programmes import build_trialdev_worked_programmes_v1


def main(argv: Sequence[str] | None = None) -> int:
    """Write one non-score-bearing worked programme for each TrialDev stream."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-identity", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    build_trialdev_worked_programmes_v1(
        output_dir=args.output,
        source_identity=args.source_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
