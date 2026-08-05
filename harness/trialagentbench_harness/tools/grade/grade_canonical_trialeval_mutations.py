"""Grade generated canonical TrialEval single-fault mutations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_harness.grading import (
    CanonicalSubmissionV1,
    GradeRecordV1,
    ScoringKeyStoreV1,
    grade,
)


class TrialEvalMutationWitnessV1(BaseModel):
    """One schema-valid single-coordinate mutation of a positive witness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval_mutation_witness/v1"]
    mutation_id: str = Field(min_length=1)
    base_witness_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    mutated_coordinate: str = Field(min_length=1)
    expected_first_gate: str | None = None
    expected_failure_code: str | None = None
    submission: CanonicalSubmissionV1


class TrialEvalMutationGradeV1(BaseModel):
    """One public grade bound to the generated mutation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval_mutation_grade/v1"] = "trialagentbench.trialeval_mutation_grade/v1"
    mutation_id: str
    base_witness_id: str
    item_id: str
    route_id: str
    mutated_coordinate: str
    expected_first_gate: str | None = None
    expected_failure_code: str | None = None
    grade: GradeRecordV1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--mutations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Grade each mutation against its original item's scoring key."""

    args = _parser().parse_args(argv)
    mutations = tuple(
        TrialEvalMutationWitnessV1.model_validate_json(line)
        for line in args.mutations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not mutations:
        raise ValueError("TrialEval mutation census cannot be empty")
    mutation_ids = tuple(row.mutation_id for row in mutations)
    if len(mutation_ids) != len(set(mutation_ids)):
        raise ValueError("TrialEval mutation census contains duplicate mutation IDs")
    item_ids = tuple(sorted({row.item_id for row in mutations}))
    store = ScoringKeyStoreV1.from_release(args.evaluator_root, expected_item_ids=item_ids)
    grades = tuple(
        TrialEvalMutationGradeV1(
            mutation_id=row.mutation_id,
            base_witness_id=row.base_witness_id,
            item_id=row.item_id,
            route_id=row.route_id,
            mutated_coordinate=row.mutated_coordinate,
            expected_first_gate=row.expected_first_gate,
            expected_failure_code=row.expected_failure_code,
            grade=grade(store.for_item(row.item_id), row.submission),
        )
        for row in mutations
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite TrialEval mutation grades: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(
                row.model_dump(mode="json", exclude_none=True),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in grades
        ),
        encoding="utf-8",
    )
    return 0


__all__ = ["TrialEvalMutationGradeV1", "TrialEvalMutationWitnessV1", "main"]
