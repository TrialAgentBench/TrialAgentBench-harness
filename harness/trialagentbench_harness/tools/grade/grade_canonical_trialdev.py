"""Grade a complete canonical TrialDev evaluation-lane census."""

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
from trialagentbench_harness.trialdev.grading.evaluation_target_register import (
    score_evaluation_target,
)


class CanonicalTrialDevLaneSubmissionV1(BaseModel):
    """One fully resolved TrialDev lane input for deterministic scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_canonical_lane_submission/v1"] = (
        "trialagentbench.trialdev_canonical_lane_submission/v1"
    )
    evaluation_target_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_id: str
    phase_id: str
    program_objective_id: str
    phase_scoring_objective_id: str
    lane_id: str
    submitted_target_id: str | None = None
    artifact_status: Literal["present", "missing", "invalid"]
    failure_reason: str | None = None
    score_override: float | None = Field(default=None, ge=0.0, le=1.0)
    score_derivation: (
        Literal[
            "literal_target",
            "numeric_diagnostic",
            "public_evidence_action",
        ]
        | None
    ) = None
    derived_from_trajectory_metric: bool = False
    terminal_action_observed: str | None = None
    terminal_asset_observed: str | None = None
    terminal_phase_observed: str | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Grade every resolved TrialDev lane and write stable JSON Lines records."""

    args = _parser().parse_args(argv)
    target_paths = tuple(sorted(args.release_root.rglob("grader/evaluation_target_register.jsonl")))
    if not target_paths:
        raise FileNotFoundError("TrialDev release contains no evaluation-target registers.")
    targets = tuple(
        target for path in target_paths for target in load_trialdev_evaluation_target_register_records(path)
    )
    by_checksum = {str(target.checksum): target for target in targets}
    if len(by_checksum) != len(targets):
        raise ValueError("TrialDev release contains duplicate evaluation-target checksums.")
    submissions = tuple(
        CanonicalTrialDevLaneSubmissionV1.model_validate_json(line)
        for line in args.submissions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not submissions:
        raise ValueError("Canonical TrialDev lane submission census cannot be empty.")
    observed_checksums = tuple(submission.evaluation_target_checksum for submission in submissions)
    if len(set(observed_checksums)) != len(observed_checksums):
        raise ValueError("Canonical TrialDev lane submission census contains duplicate targets.")
    if set(observed_checksums) != set(by_checksum):
        raise ValueError(
            "Canonical TrialDev lane census is incomplete: "
            f"missing={sorted(set(by_checksum) - set(observed_checksums))!r}, "
            f"extra={sorted(set(observed_checksums) - set(by_checksum))!r}."
        )
    submission_by_checksum = {submission.evaluation_target_checksum: submission for submission in submissions}
    records = []
    for target in targets:
        checksum = str(target.checksum)
        submission = submission_by_checksum[checksum]
        records.append(
            score_evaluation_target(
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
        )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite canonical TrialDev grade census: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(record.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return 0


__all__ = ["CanonicalTrialDevLaneSubmissionV1", "main"]
