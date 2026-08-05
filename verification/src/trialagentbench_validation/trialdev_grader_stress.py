"""Generated target-level stress census for the TrialDev grader."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.grader_concordance import (
    CanonicalTrialDevLaneSubmissionV1,
    TrialDevEvaluationTargetV1,
    TrialDevLaneGradeV1,
    grade_trialdev_lane_independently,
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevMutationWitnessV1(_Contract):
    """One schema-valid target-level TrialDev mutation."""

    schema_id: Literal["trialagentbench.trialdev_mutation_witness/v1"] = (
        "trialagentbench.trialdev_mutation_witness/v1"
    )
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
    expected_score: Literal[0] = 0
    submission: CanonicalTrialDevLaneSubmissionV1


class TrialDevMutationGradeV1(_Contract):
    """One TrialDev grade bound to a generated mutation."""

    schema_id: Literal["trialagentbench.trialdev_mutation_grade/v1"] = (
        "trialagentbench.trialdev_mutation_grade/v1"
    )
    mutation_id: str
    mutated_coordinate: str
    expected_status: str
    expected_score: float
    grade: TrialDevLaneGradeV1


class TrialDevGraderStressReportV1(_Contract):
    """Exact public/independent agreement over TrialDev target mutations."""

    schema_id: Literal["trialagentbench.trialdev_grader_stress/v1"] = (
        "trialagentbench.trialdev_grader_stress/v1"
    )
    positive_target_count: int = Field(ge=1)
    required_mutation_count: int = Field(ge=1)
    independently_graded_count: int = Field(ge=0)
    public_graded_count: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)
    expected_behavior_failure_count: int = Field(ge=0)
    crashed_count: int = Field(ge=0)
    mutation_counts_by_coordinate: dict[str, int]
    independently_graded_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    publicly_graded_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_command: tuple[str, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def _status_matches_census(self) -> TrialDevGraderStressReportV1:
        passed = (
            self.required_mutation_count
            == self.independently_graded_count
            == self.public_graded_count
            and self.mismatch_count
            == self.expected_behavior_failure_count
            == self.crashed_count
            == 0
        )
        if (self.status == "pass") != passed:
            raise ValueError("TrialDev grader-stress status does not match its census")
        return self


def _mutate_submission(
    submission: CanonicalTrialDevLaneSubmissionV1,
    mutate: Callable[..., object],
) -> CanonicalTrialDevLaneSubmissionV1:
    payload = copy.deepcopy(submission.model_dump(mode="json", exclude_none=True))
    mutate(payload)
    return CanonicalTrialDevLaneSubmissionV1.model_validate(payload)


def generate_trialdev_mutations(
    *,
    targets: tuple[TrialDevEvaluationTargetV1, ...],
    submission_by_checksum: dict[str, CanonicalTrialDevLaneSubmissionV1],
) -> tuple[TrialDevMutationWitnessV1, ...]:
    """Generate all applicable target/action and artifact-state mutations."""

    mutations: list[TrialDevMutationWitnessV1] = []
    for target in targets:
        submission = submission_by_checksum[target.checksum]
        prefix = (
            f"{target.scenario_id}::{target.phase_id}::{target.lane_id}::"
            f"{target.checksum[:12]}"
        )
        if submission.artifact_status != "present":
            raise ValueError(
                "positive TrialDev witness must be present before stress mutation"
            )
        for artifact_status, expected_status in (
            ("missing", "missing_submission_zeroed"),
            ("invalid", "invalid_submission_zeroed"),
        ):
            coordinate = f"artifact_status.{artifact_status}"
            mutations.append(
                TrialDevMutationWitnessV1(
                    mutation_id=f"{prefix}::{coordinate}",
                    mutated_coordinate=coordinate,
                    expected_status=expected_status,
                    submission=_mutate_submission(
                        submission,
                        lambda payload, replacement=artifact_status: payload.update(
                            {"artifact_status": replacement}
                        ),
                    ),
                )
            )
        if (
            submission.score_override is None
            and target.target_resolution != "realized_public_evidence"
        ):
            mutations.append(
                TrialDevMutationWitnessV1(
                    mutation_id=f"{prefix}::submitted_target_id",
                    mutated_coordinate="submitted_target_id",
                    expected_status="scored",
                    submission=_mutate_submission(
                        submission,
                        lambda payload: payload.update(
                            {"submitted_target_id": "__unsupported_target__"}
                        ),
                    ),
                )
            )
    mutation_ids = tuple(row.mutation_id for row in mutations)
    if not mutation_ids:
        raise ValueError("TrialDev release has no target mutation to exercise")
    if len(mutation_ids) != len(set(mutation_ids)):
        raise ValueError("TrialDev mutation IDs are not unique")
    return tuple(mutations)


def _write_jsonl(path: Path, rows: Sequence[BaseModel]) -> str:
    body = "".join(
        json.dumps(
            row.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _matches_expected_behavior(row: TrialDevMutationGradeV1) -> bool:
    return (
        row.grade.status == row.expected_status
        and row.grade.score == row.expected_score
    )


def run_trialdev_grader_stress(
    *,
    targets: tuple[TrialDevEvaluationTargetV1, ...],
    submission_by_checksum: dict[str, CanonicalTrialDevLaneSubmissionV1],
    release_root: Path,
    output_dir: Path,
    harness_executable: str,
) -> TrialDevGraderStressReportV1:
    """Generate, grade twice, and reconcile the TrialDev mutation census."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    target_by_checksum = {target.checksum: target for target in targets}
    mutations = generate_trialdev_mutations(
        targets=targets,
        submission_by_checksum=submission_by_checksum,
    )
    mutation_path = output / "trialdev_mutations.jsonl"
    _write_jsonl(mutation_path, mutations)
    independent = tuple(
        TrialDevMutationGradeV1(
            mutation_id=row.mutation_id,
            mutated_coordinate=row.mutated_coordinate,
            expected_status=row.expected_status,
            expected_score=row.expected_score,
            grade=grade_trialdev_lane_independently(
                target_by_checksum[row.submission.evaluation_target_checksum],
                row.submission,
            ),
        )
        for row in mutations
    )
    independent_path = output / "independent_mutation_grades.jsonl"
    independent_sha256 = _write_jsonl(independent_path, independent)
    public_path = output / "public_mutation_grades.jsonl"
    command = (
        harness_executable,
        "grade",
        "canonical-trialdev-mutations",
        "--release-root",
        str(release_root),
        "--mutations",
        str(mutation_path),
        "--output",
        str(public_path),
    )
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    crashed = int(completed.returncode != 0)
    public = (
        ()
        if crashed
        else tuple(
            TrialDevMutationGradeV1.model_validate_json(line)
            for line in public_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )
    public_sha256 = hashlib.sha256(
        public_path.read_bytes() if public_path.is_file() else b""
    ).hexdigest()
    independent_by_id = {row.mutation_id: row for row in independent}
    public_by_id = {row.mutation_id: row for row in public}
    mismatches = sum(
        public_by_id.get(row.mutation_id) != independent_by_id[row.mutation_id]
        for row in mutations
    )
    behavior_failures = sum(not _matches_expected_behavior(row) for row in independent)
    counts = Counter(row.mutated_coordinate for row in mutations)
    report = TrialDevGraderStressReportV1(
        positive_target_count=len(targets),
        required_mutation_count=len(mutations),
        independently_graded_count=len(independent),
        public_graded_count=len(public),
        mismatch_count=mismatches,
        expected_behavior_failure_count=behavior_failures,
        crashed_count=crashed,
        mutation_counts_by_coordinate=dict(sorted(counts.items())),
        independently_graded_sha256=independent_sha256,
        publicly_graded_sha256=public_sha256,
        public_command=command,
        status=(
            "pass"
            if not crashed
            and not mismatches
            and not behavior_failures
            and len(independent) == len(public) == len(mutations)
            else "fail"
        ),
    )
    (output / "trialdev_grader_stress_report.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "TrialDevGraderStressReportV1",
    "TrialDevMutationGradeV1",
    "TrialDevMutationWitnessV1",
    "generate_trialdev_mutations",
    "run_trialdev_grader_stress",
]
