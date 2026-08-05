"""Execute one isolated participant-diagnostic batch."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.analysis.trialeval_diagnostic_proof_surface import (
    _build_participant_diagnostic_batch_v1,
)
from trialagentbench_harness.execution_policy import TRIALEVAL_DIAGNOSTIC_WORKER_INVALID_INPUT_EXIT_CODE


def main(argv: Sequence[str] | None = None) -> int:
    """Write one diagnostic batch as validated JSON records."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-zip", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    task_ids = tuple(str(task_id) for task_id in args.task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Participant diagnostic worker task IDs must be unique.")
    try:
        rows = _build_participant_diagnostic_batch_v1(
            public_zip=Path(args.public_zip),
            task_ids=task_ids,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return TRIALEVAL_DIAGNOSTIC_WORKER_INVALID_INPUT_EXIT_CODE
    json.dump(
        [row.model_dump(mode="json") for row in rows],
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
