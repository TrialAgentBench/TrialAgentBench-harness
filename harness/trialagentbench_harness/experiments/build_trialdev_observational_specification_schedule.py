"""Build the paired TrialDev observational specification schedule."""

from __future__ import annotations

import argparse
import hashlib
import random
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from trialagentbench_harness.adapters.trialdev_share import (
    TrialDevPublicObservationalMethodCatalogV1,
)
from trialagentbench_harness.contracts.experiments import (
    TrialDevObservationalSpecificationAssignmentV1,
    TrialDevObservationalSpecificationConditionV1,
    TrialDevObservationalSpecificationScheduleV1,
)
from trialagentbench_harness.io import read_json_model, sha256_dir_digest, write_json_model
from trialagentbench_harness.trialdev.data import discover_programs, scenario_root

_CONDITIONS: tuple[TrialDevObservationalSpecificationConditionV1, ...] = (
    "open_selection",
    "prespecified_execution",
)


def build_trialdev_observational_specification_schedule_v1(
    *,
    participant_root: Path,
    experiment_id: str,
    replicate_seeds: dict[str, int],
    randomization_seed: int,
    program_ids: tuple[str, ...] | None = None,
) -> TrialDevObservationalSpecificationScheduleV1:
    """Build complete method-stratified pairs from participant files only."""

    root = Path(participant_root)
    if not replicate_seeds:
        raise ValueError("replicate_seeds must be non-empty.")
    if len(set(replicate_seeds.values())) != len(replicate_seeds):
        raise ValueError("replicate_seeds values must be unique.")
    if any(not name or isinstance(seed, bool) or seed < 0 for name, seed in replicate_seeds.items()):
        raise ValueError("replicate_seeds requires non-empty IDs and nonnegative integer seeds.")
    programs = {program.program_id: program for program in discover_programs(root)}
    selected_ids = tuple(sorted(program_ids if program_ids is not None else programs))
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("program_ids must be non-empty and unique.")
    unknown = sorted(set(selected_ids) - set(programs))
    if unknown:
        raise ValueError(f"Unknown TrialDev program IDs: {unknown!r}.")

    assignments: list[TrialDevObservationalSpecificationAssignmentV1] = []
    for program_id in selected_ids:
        program = programs[program_id]
        if not program.has_obs_review:
            raise ValueError(f"TrialDev observational experiment program lacks an observational item: {program_id}")
        catalog = read_json_model(
            TrialDevPublicObservationalMethodCatalogV1,
            scenario_root(root, program.scenario_id) / "public" / "observational_method_catalog.json",
        )
        for replicate_id, decoding_seed in sorted(replicate_seeds.items()):
            for method in catalog.methods:
                pair_key = f"{experiment_id}\0{program_id}\0{replicate_id}\0{method.method_route_id}"
                pair_id = f"P{hashlib.sha256(pair_key.encode()).hexdigest()[:32]}"
                for condition in _CONDITIONS:
                    assignment_key = f"{pair_id}\0{condition}"
                    assignments.append(
                        TrialDevObservationalSpecificationAssignmentV1(
                            assignment_id=f"A{hashlib.sha256(assignment_key.encode()).hexdigest()[:32]}",
                            pair_id=pair_id,
                            program_id=program.program_id,
                            scenario_id=program.scenario_id,
                            objective_id=program.objective_id,
                            replicate_id=replicate_id,
                            decoding_seed=decoding_seed,
                            condition=condition,
                            method_catalog_checksum=cast(str, catalog.checksum),
                            method_specification=method,
                        )
                    )
    random.Random(randomization_seed).shuffle(assignments)
    return TrialDevObservationalSpecificationScheduleV1(
        experiment_id=experiment_id,
        participant_release_sha256=sha256_dir_digest(root),
        randomization_seed=randomization_seed,
        assignments=tuple(assignments),
    )


def _parse_replicates(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        name, separator, raw_seed = value.partition("=")
        if not separator or not name or not raw_seed.isascii() or not raw_seed.isdecimal():
            raise ValueError("--replicate must use ID=SEED with a nonnegative decimal seed.")
        if name in result:
            raise ValueError(f"Duplicate replicate ID: {name!r}.")
        result[name] = int(raw_seed)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Write one immutable observational specification schedule."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--randomization-seed", required=True, type=int)
    parser.add_argument("--replicate", action="append", required=True)
    parser.add_argument("--program-id", action="append")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite schedule: {output}")
    schedule = build_trialdev_observational_specification_schedule_v1(
        participant_root=Path(args.participant_dir),
        experiment_id=str(args.experiment_id),
        replicate_seeds=_parse_replicates(tuple(args.replicate)),
        randomization_seed=int(args.randomization_seed),
        program_ids=tuple(args.program_id) if args.program_id else None,
    )
    write_json_model(output, schedule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
