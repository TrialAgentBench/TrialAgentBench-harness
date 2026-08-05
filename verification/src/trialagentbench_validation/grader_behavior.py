"""Audit public-grader behavior across canonical acceptance and failure classes."""

from __future__ import annotations

import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from trialagentbench_validation.grader_concordance import (
    CanonicalSubmissionV1,
    GradeRecordV1,
    ScoringKeyV1,
    grade_trialeval_independently,
)
from trialagentbench_validation.io import write_model


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


BehaviorClass = Literal[
    "accepted",
    "defensible_alternative",
    "rejected",
    "abstaining",
    "malformed",
    "qualified_nonidentification",
]


class _BehaviorCaseV1(_Contract):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    behavior_class: BehaviorClass
    relative_path: str = Field(min_length=1)
    focus_item_id: str = Field(min_length=1)
    expected_public_exit_code: int = Field(ge=0)
    expected_focus_passed: bool | None
    expected_matched_route_id: str | None = None


class _BehaviorManifestV1(_Contract):
    schema_id: Literal["trialagentbench.canonical_grader_behavior_manifest/v1"]
    cases: tuple[_BehaviorCaseV1, ...] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def _complete_classes(self) -> _BehaviorManifestV1:
        classes = tuple(row.behavior_class for row in self.cases)
        expected = (
            "accepted",
            "abstaining",
            "defensible_alternative",
            "malformed",
            "qualified_nonidentification",
            "rejected",
        )
        if classes != expected:
            raise ValueError(
                "grader behavior manifest must contain all six classes in canonical order"
            )
        return self


class GraderBehaviorCaseResultV1(_Contract):
    """Independent/public result for one complete-census behavior class."""

    case_id: str
    behavior_class: BehaviorClass
    focus_item_id: str
    submission_count: int = Field(ge=1)
    independent_rejected_malformed: bool
    public_exit_code: int = Field(ge=0)
    compared_grade_count: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)
    focus_passed: bool | None
    matched_route_id: str | None
    status: Literal["pass", "fail"]


class GraderBehaviorReportV1(_Contract):
    """Complete canonical grader behavior census."""

    schema_id: Literal["trialagentbench.grader_behavior_report/v1"] = (
        "trialagentbench.grader_behavior_report/v1"
    )
    release_id: str = Field(min_length=1)
    required_class_count: Literal[6] = 6
    passed_class_count: int = Field(ge=0, le=6)
    cases: tuple[GraderBehaviorCaseResultV1, ...] = Field(min_length=6, max_length=6)
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def _status_matches_cases(self) -> GraderBehaviorReportV1:
        passed = sum(row.status == "pass" for row in self.cases)
        if self.passed_class_count != passed or (self.status == "pass") != (
            passed == 6
        ):
            raise ValueError("grader behavior report status disagrees with its cases")
        return self


def _safe_extract(archive_path: Path, output_root: Path) -> None:
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
                raise ValueError(
                    f"unsafe evaluator archive member: {member.filename!r}"
                )
        archive.extractall(output_root)


def _read_keys(evaluator_root: Path) -> tuple[ScoringKeyV1, ...]:
    paths = tuple(evaluator_root.rglob("grader/scoring_keys.jsonl"))
    if len(paths) != 1:
        raise ValueError(
            "grader behavior requires exactly one TrialEval scoring-key census"
        )
    records = tuple(
        ScoringKeyV1.model_validate_json(line)
        for line in paths[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records or len({row.item_id for row in records}) != len(records):
        raise ValueError("grader behavior scoring-key census is empty or duplicated")
    return records


def _read_submissions(path: Path) -> tuple[CanonicalSubmissionV1, ...]:
    return tuple(
        CanonicalSubmissionV1.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _public_records(path: Path) -> tuple[GradeRecordV1, ...]:
    return tuple(
        GradeRecordV1.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _valid_case_result(
    *,
    case: _BehaviorCaseV1,
    case_path: Path,
    keys: tuple[ScoringKeyV1, ...],
    evaluator_root: Path,
    output_root: Path,
    harness_executable: str,
) -> GraderBehaviorCaseResultV1:
    submissions = _read_submissions(case_path)
    key_by_item = {row.item_id: row for row in keys}
    submission_by_item = {row.item_id: row for row in submissions}
    if set(submission_by_item) != set(key_by_item) or len(submission_by_item) != len(
        submissions
    ):
        raise ValueError(
            f"grader behavior case is not a complete unique census: {case.case_id}"
        )
    ordered_ids = tuple(row.item_id for row in keys)
    independent = tuple(
        grade_trialeval_independently(key_by_item[item_id], submission_by_item[item_id])
        for item_id in ordered_ids
    )
    public_path = output_root / f"{case.case_id}.jsonl"
    completed = subprocess.run(
        (
            harness_executable,
            "grade",
            "canonical-trialeval",
            "--evaluator-root",
            str(evaluator_root),
            "--submissions",
            str(case_path),
            "--output",
            str(public_path),
        ),
        check=False,
        text=True,
        capture_output=True,
    )
    public = (
        _public_records(public_path)
        if completed.returncode == 0 and public_path.is_file()
        else ()
    )
    public_by_item = {row.item_id: row for row in public}
    independent_by_item = {row.item_id: row for row in independent}
    mismatches = sum(
        public_by_item.get(item_id) != independent_by_item[item_id]
        for item_id in ordered_ids
    )
    focus = independent_by_item[case.focus_item_id]
    semantic_match = (
        completed.returncode == case.expected_public_exit_code
        and focus.passed == case.expected_focus_passed
        and (
            case.expected_matched_route_id is None
            or focus.matched_route_id == case.expected_matched_route_id
        )
    )
    passed = semantic_match and len(public) == len(independent) and mismatches == 0
    return GraderBehaviorCaseResultV1(
        case_id=case.case_id,
        behavior_class=case.behavior_class,
        focus_item_id=case.focus_item_id,
        submission_count=len(submissions),
        independent_rejected_malformed=False,
        public_exit_code=completed.returncode,
        compared_grade_count=len(independent),
        mismatch_count=mismatches,
        focus_passed=focus.passed,
        matched_route_id=focus.matched_route_id,
        status="pass" if passed else "fail",
    )


def _malformed_case_result(
    *,
    case: _BehaviorCaseV1,
    case_path: Path,
    evaluator_root: Path,
    output_root: Path,
    harness_executable: str,
) -> GraderBehaviorCaseResultV1:
    lines = tuple(
        line
        for line in case_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    independently_rejected = False
    try:
        _read_submissions(case_path)
    except ValidationError:
        independently_rejected = True
    public_path = output_root / f"{case.case_id}.jsonl"
    completed = subprocess.run(
        (
            harness_executable,
            "grade",
            "canonical-trialeval",
            "--evaluator-root",
            str(evaluator_root),
            "--submissions",
            str(case_path),
            "--output",
            str(public_path),
        ),
        check=False,
        text=True,
        capture_output=True,
    )
    passed = (
        independently_rejected
        and completed.returncode != 0
        and not public_path.exists()
    )
    return GraderBehaviorCaseResultV1(
        case_id=case.case_id,
        behavior_class=case.behavior_class,
        focus_item_id=case.focus_item_id,
        submission_count=len(lines),
        independent_rejected_malformed=independently_rejected,
        public_exit_code=completed.returncode,
        compared_grade_count=0,
        mismatch_count=0,
        focus_passed=None,
        matched_route_id=None,
        status="pass" if passed else "fail",
    )


def run_grader_behavior_census(
    *,
    release_id: str,
    release_root: Path,
    canonical_submissions: Path,
    output_dir: Path,
    harness_executable: str = "trialagentbench",
) -> GraderBehaviorReportV1:
    """Compare six complete behavioral classes with the public grader."""

    root = release_root.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace grader behavior output: {output}")
    output.mkdir(parents=True)
    manifest = _BehaviorManifestV1.model_validate_json(
        (canonical_submissions / "grader_behavior" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expanded_paths = tuple(root.rglob("grader/scoring_keys.jsonl"))
    with tempfile.TemporaryDirectory(
        prefix="trialagentbench-grader-behavior-"
    ) as temporary:
        if expanded_paths:
            evaluator_root = expanded_paths[0].parent.parent
        else:
            archives = tuple(root.rglob("*TrialEvalBench_evaluator.zip"))
            if len(archives) != 1:
                raise ValueError(
                    "grader behavior requires one TrialEval evaluator archive"
                )
            evaluator_root = Path(temporary) / "evaluator"
            _safe_extract(archives[0], evaluator_root)
        keys = _read_keys(evaluator_root)
        results = []
        for case in manifest.cases:
            case_path = (canonical_submissions / case.relative_path).resolve()
            if (
                not case_path.is_relative_to(canonical_submissions.resolve())
                or not case_path.is_file()
            ):
                raise ValueError(
                    f"grader behavior case path is unsafe or missing: {case.relative_path}"
                )
            if case.behavior_class == "malformed":
                result = _malformed_case_result(
                    case=case,
                    case_path=case_path,
                    evaluator_root=evaluator_root,
                    output_root=output,
                    harness_executable=harness_executable,
                )
            else:
                result = _valid_case_result(
                    case=case,
                    case_path=case_path,
                    keys=keys,
                    evaluator_root=evaluator_root,
                    output_root=output,
                    harness_executable=harness_executable,
                )
            results.append(result)
    cases = tuple(sorted(results, key=lambda row: row.behavior_class))
    report = GraderBehaviorReportV1(
        release_id=release_id,
        passed_class_count=sum(row.status == "pass" for row in cases),
        cases=cases,
        status="pass" if all(row.status == "pass" for row in cases) else "fail",
    )
    write_model(output / "grader_behavior_report.json", report)
    return report


__all__ = [
    "GraderBehaviorCaseResultV1",
    "GraderBehaviorReportV1",
    "run_grader_behavior_census",
]
