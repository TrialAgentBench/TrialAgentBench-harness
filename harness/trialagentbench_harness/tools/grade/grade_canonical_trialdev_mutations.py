"""Grade generated canonical TrialDev target mutations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_harness.contracts.trialdev.trialdev_evaluation_target_register import (
    TrialDevEvaluationTargetRegisterRecordV1,
    load_trialdev_evaluation_target_register_records,
)
from trialagentbench_harness.tools.grade.grade_canonical_trialdev import (
    CanonicalTrialDevLaneSubmissionV1,
)
from trialagentbench_harness.trialdev.grading.evaluation_target_register import (
    score_evaluation_target,
)
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevelopmentLaneScoreRecordV1,
)


class TrialDevMutationWitnessV1(BaseModel):
    """One schema-valid TrialDev target mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_mutation_witness/v1"]
    mutation_id: str = Field(min_length=1)
    mutated_coordinate: Literal[
        "submitted_target_id",
        "artifact_status.missing",
        "artifact_status.invalid",
    ]
    expected_status: Literal[
        "scored",
        "missing_submission_zeroed",
        "invalid_submission_zeroed",
    ]
    expected_score: Literal[0]
    submission: CanonicalTrialDevLaneSubmissionV1


class TrialDevMutationGradeV1(BaseModel):
    """One public TrialDev grade bound to its mutation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_mutation_grade/v1"] = "trialagentbench.trialdev_mutation_grade/v1"
    mutation_id: str
    mutated_coordinate: str
    expected_status: str
    expected_score: float
    grade: TrialDevelopmentLaneScoreRecordV1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--mutations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Grade every mutation without imposing a one-row-per-target census."""

    args = _parser().parse_args(argv)
    target_paths = tuple(sorted(args.release_root.rglob("grader/evaluation_target_register.jsonl")))
    if not target_paths:
        raise FileNotFoundError("TrialDev release contains no evaluation-target registers.")
    targets = tuple(
        target for path in target_paths for target in load_trialdev_evaluation_target_register_records(path)
    )
    target_by_checksum = {str(target.checksum): target for target in targets}
    if len(target_by_checksum) != len(targets):
        raise ValueError("TrialDev release contains duplicate evaluation-target checksums.")
    mutations = tuple(
        TrialDevMutationWitnessV1.model_validate_json(line)
        for line in args.mutations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not mutations:
        raise ValueError("TrialDev mutation census cannot be empty")
    mutation_ids = tuple(row.mutation_id for row in mutations)
    if len(mutation_ids) != len(set(mutation_ids)):
        raise ValueError("TrialDev mutation census contains duplicate mutation IDs")
    grades: list[TrialDevMutationGradeV1] = []
    for mutation in mutations:
        submission = mutation.submission
        target = target_by_checksum.get(submission.evaluation_target_checksum)
        if target is None:
            raise ValueError("TrialDev mutation names an unknown evaluation-target checksum")
        grade = score_evaluation_target(
            scenario_id=submission.scenario_id,
            phase_id=submission.phase_id,
            program_objective_id=submission.program_objective_id,
            phase_scoring_objective_id=submission.phase_scoring_objective_id,
            lane_id=submission.lane_id,
            submitted_target_id=submission.submitted_target_id,
            evaluation_target=TrialDevEvaluationTargetRegisterRecordV1.model_validate(target),
            artifact_status=submission.artifact_status,
            failure_reason=submission.failure_reason,
            score_override=submission.score_override,
            score_derivation=submission.score_derivation,
            derived_from_trajectory_metric=submission.derived_from_trajectory_metric,
            terminal_action_observed=submission.terminal_action_observed,
            terminal_asset_observed=submission.terminal_asset_observed,
            terminal_phase_observed=submission.terminal_phase_observed,
        )
        grades.append(
            TrialDevMutationGradeV1(
                mutation_id=mutation.mutation_id,
                mutated_coordinate=mutation.mutated_coordinate,
                expected_status=mutation.expected_status,
                expected_score=mutation.expected_score,
                grade=grade,
            )
        )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite TrialDev mutation grades: {output}")
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


__all__ = ["TrialDevMutationGradeV1", "TrialDevMutationWitnessV1", "main"]
