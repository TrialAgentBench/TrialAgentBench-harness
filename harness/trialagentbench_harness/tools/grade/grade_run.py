"""Grade one canonical TrialAgentBench run against its evaluator release."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from trialagentbench_harness.tools.grade.grade_trialdev import main as grade_trialdev_run
from trialagentbench_harness.tools.grade.grade_trialeval import grade_trialeval_run


def main(argv: Sequence[str] | None = None) -> int:
    """Grade a canonical TrialAgentBench run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("trialeval", "trialdev"), help="Benchmark suite to grade.")
    parser.add_argument("suite_args", nargs=argparse.REMAINDER, help="Arguments passed to the suite scorer.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.suite == "trialeval":
        return int(grade_trialeval_run(args.suite_args))
    return int(grade_trialdev_run(args.suite_args))


if __name__ == "__main__":
    raise SystemExit(main())
