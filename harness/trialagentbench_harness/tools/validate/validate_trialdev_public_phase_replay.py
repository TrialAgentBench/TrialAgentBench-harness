"""Replay TrialDev randomized phases into public decision and design evidence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.contracts.trialdev.trialdev_public_phase_replay import (
    TrialDevPhaseReplayCaseV1,
)
from trialagentbench_harness.trialdev.public_phase_replay import (
    replay_trialdev_public_phases_v1,
)


def _load_cases(path: Path) -> tuple[TrialDevPhaseReplayCaseV1, ...]:
    cases: list[TrialDevPhaseReplayCaseV1] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(TrialDevPhaseReplayCaseV1.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid public phase-replay case row {line_number} in {path}: {exc}") from exc
    if not cases:
        raise ValueError("Public phase-replay case JSONL is empty.")
    return tuple(cases)


def main(argv: Sequence[str] | None = None) -> int:
    """Replay randomized phases and write public-evidence JSONL."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--trial-seed", type=int, action="append", required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    records = replay_trialdev_public_phases_v1(
        bundle_root=args.bundle_root,
        materialized_root=args.materialized_root,
        cases=_load_cases(args.cases),
        trial_seeds=tuple(args.trial_seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
