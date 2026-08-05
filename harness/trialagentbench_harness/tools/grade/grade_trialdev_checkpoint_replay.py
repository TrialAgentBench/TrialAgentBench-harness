"""Grade a TrialDev checkpoint-replay experiment without model calls."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.contracts.experiments import TrialDevCheckpointScheduleV1
from trialagentbench_harness.io import read_json_model, staged_directory
from trialagentbench_harness.tools.grade.grade_trialdev import grade_program
from trialagentbench_harness.tools.grade.release_pair import materialized_trialdev_release_root


def main(argv: Sequence[str] | None = None) -> int:
    """Copy and grade every completed checkpoint assignment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    source = Path(args.run_root).resolve()
    schedule = read_json_model(TrialDevCheckpointScheduleV1, source / "schedule.json")
    with staged_directory(Path(args.out_dir).resolve()) as output:
        shutil.copytree(source, output, dirs_exist_ok=True)
        with materialized_trialdev_release_root(Path(args.bundle)) as bundle:
            for assignment in schedule.assignments:
                program_dir = output / "assignments" / assignment.assignment_id / "programs" / assignment.program_id
                _, trajectory_graded = grade_program(program_dir, bundle=bundle)
                if not trajectory_graded:
                    raise ValueError(
                        f"Checkpoint assignment {assignment.assignment_id} lacks a randomized-phase grade."
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
