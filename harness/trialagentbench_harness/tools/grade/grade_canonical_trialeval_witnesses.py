"""Grade every canonical TrialEval item-route witness."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.grading import (
    CanonicalSubmissionV1,
    GradeRecordV1,
    ScoringKeyStoreV1,
    grade,
)


class CanonicalTrialEvalRouteWitnessV1(BaseModel):
    """One positive canonical submission bound to an eligible route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval_route_witness/v1"]
    witness_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    submission: CanonicalSubmissionV1

    @model_validator(mode="after")
    def _identity_is_bound(self) -> CanonicalTrialEvalRouteWitnessV1:
        if self.submission.item_id != self.item_id:
            raise ValueError("route-witness item identity disagrees with its submission")
        return self


class CanonicalTrialEvalRouteWitnessGradeV1(BaseModel):
    """One public grade bound to its route-witness identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval_route_witness_grade/v1"] = (
        "trialagentbench.trialeval_route_witness_grade/v1"
    )
    witness_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    grade: GradeRecordV1

    @model_validator(mode="after")
    def _grade_is_bound(self) -> CanonicalTrialEvalRouteWitnessGradeV1:
        if self.grade.item_id != self.item_id:
            raise ValueError("route-witness grade item identity mismatch")
        if self.grade.matched_route_id != self.route_id:
            raise ValueError("positive route witness did not grade through its declared route")
        return self


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--witnesses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_witnesses(path: Path) -> tuple[CanonicalTrialEvalRouteWitnessV1, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical TrialEval route-witness census is missing: {path}")
    witnesses = tuple(
        CanonicalTrialEvalRouteWitnessV1.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not witnesses:
        raise ValueError("Canonical TrialEval route-witness census cannot be empty")
    identities = tuple(witness.witness_id for witness in witnesses)
    if len(identities) != len(set(identities)):
        raise ValueError("Canonical TrialEval route-witness census contains duplicate witness IDs")
    return witnesses


def main(argv: Sequence[str] | None = None) -> int:
    """Grade each witness without collapsing multiple routes for one item."""

    args = _parser().parse_args(argv)
    witnesses = _read_witnesses(args.witnesses)
    item_ids = tuple(sorted({witness.item_id for witness in witnesses}))
    store = ScoringKeyStoreV1.from_release(args.evaluator_root, expected_item_ids=item_ids)
    records = tuple(
        CanonicalTrialEvalRouteWitnessGradeV1(
            witness_id=witness.witness_id,
            item_id=witness.item_id,
            route_id=witness.route_id,
            grade=grade(store.for_item(witness.item_id), witness.submission),
        )
        for witness in witnesses
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite canonical route-witness grades: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(
                record.model_dump(mode="json", exclude_none=True),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return 0


__all__ = [
    "CanonicalTrialEvalRouteWitnessGradeV1",
    "CanonicalTrialEvalRouteWitnessV1",
    "main",
]
