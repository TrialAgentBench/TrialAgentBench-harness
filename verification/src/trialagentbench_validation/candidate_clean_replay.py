"""Verify candidate evidence reproduced by independently installed public wheels."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.grader_behavior import GraderBehaviorReportV1
from trialagentbench_validation.grader_concordance import GraderConcordanceReportV1
from trialagentbench_validation.io import sha256_file, write_model
from trialagentbench_validation.recovery import RecoverabilityReportV1
from trialagentbench_validation.trialeval.integrity import (
    C5IntegrityRecoveryReportV1,
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CleanWheelIdentityV1(_Contract):
    """Identity of one wheel installed for candidate replay."""

    package: Literal["trial-agent-bench", "trialagentbench-validation"]
    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CandidateReplayComparisonV1(_Contract):
    """Structural and numerical comparison for one replayed evidence family."""

    evidence_family: Literal[
        "trialeval_recovery",
        "trialdev_recovery",
        "trialeval_c5_integrity",
        "grader_concordance",
        "grader_behavior",
        "role_boundary",
    ]
    compared_numeric_value_count: int = Field(ge=0)
    maximum_absolute_difference: float = Field(ge=0)
    structural_difference_count: int = Field(ge=0)
    nonnumeric_difference_count: int = Field(ge=0)
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def _status_matches_differences(self) -> CandidateReplayComparisonV1:
        passed = (
            self.structural_difference_count == 0
            and self.nonnumeric_difference_count == 0
        )
        if (self.status == "pass") != passed:
            raise ValueError(
                "clean-wheel comparison status disagrees with its structural results"
            )
        return self


class CandidateCleanWheelReplayV1(_Contract):
    """Receipt for repository-independent replay of one finite candidate."""

    schema_id: Literal["trialagentbench.candidate_clean_wheel_replay/v1"] = (
        "trialagentbench.candidate_clean_wheel_replay/v1"
    )
    release_id: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    python_implementation: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    installed_environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    installation_constraints_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    absolute_tolerance: float = Field(gt=0)
    isolated_install: Literal[True]
    network_blocked_during_replay: Literal[True]
    repository_on_import_path: Literal[False]
    prohibited_import_count: Literal[0]
    wheels: tuple[CleanWheelIdentityV1, ...] = Field(min_length=2, max_length=2)
    comparisons: tuple[CandidateReplayComparisonV1, ...] = Field(
        min_length=6, max_length=6
    )
    status: Literal["pass"]

    @model_validator(mode="after")
    def _complete(self) -> CandidateCleanWheelReplayV1:
        packages = tuple(row.package for row in self.wheels)
        if packages != ("trial-agent-bench", "trialagentbench-validation"):
            raise ValueError(
                "clean replay must bind the harness and validation wheels in canonical order"
            )
        families = tuple(row.evidence_family for row in self.comparisons)
        expected = (
            "grader_behavior",
            "grader_concordance",
            "role_boundary",
            "trialdev_recovery",
            "trialeval_c5_integrity",
            "trialeval_recovery",
        )
        if families != expected:
            raise ValueError(
                "clean replay comparison families are incomplete or not canonical"
            )
        failures = tuple(
            row
            for row in self.comparisons
            if row.status != "pass"
            or row.maximum_absolute_difference > self.absolute_tolerance
        )
        if failures:
            details = ", ".join(
                f"{row.evidence_family}("
                f"status={row.status}, "
                f"maximum_absolute_difference={row.maximum_absolute_difference:.17g}, "
                f"structural_difference_count={row.structural_difference_count}, "
                f"nonnumeric_difference_count={row.nonnumeric_difference_count})"
                for row in failures
            )
            raise ValueError(
                f"clean replay contains failed or out-of-tolerance comparisons: {details}"
            )
        return self


def _compare_values(
    reference: object,
    replay: object,
    *,
    tolerance: float,
) -> tuple[int, float, int, int]:
    """Return numeric count, maximum difference, structural count, and text count."""

    if isinstance(reference, bool) or isinstance(replay, bool):
        return (0, 0.0, 0, int(reference != replay))
    if isinstance(reference, int | float) and isinstance(replay, int | float):
        difference = abs(float(reference) - float(replay))
        return (1, difference, 0, int(difference > tolerance))
    if isinstance(reference, dict) and isinstance(replay, dict):
        reference_keys = set(reference)
        replay_keys = set(replay)
        structural = len(reference_keys ^ replay_keys)
        counts = [
            _compare_values(reference[key], replay[key], tolerance=tolerance)
            for key in sorted(reference_keys & replay_keys)
        ]
    elif isinstance(reference, list) and isinstance(replay, list):
        structural = abs(len(reference) - len(replay))
        counts = [
            _compare_values(left, right, tolerance=tolerance)
            for left, right in zip(reference, replay, strict=False)
        ]
    else:
        return (
            0,
            0.0,
            int(type(reference) is not type(replay)),
            int(reference != replay),
        )
    return (
        sum(row[0] for row in counts),
        max((row[1] for row in counts), default=0.0),
        structural + sum(row[2] for row in counts),
        sum(row[3] for row in counts),
    )


def _comparison(
    *,
    family: Literal[
        "trialeval_recovery",
        "trialdev_recovery",
        "trialeval_c5_integrity",
        "grader_concordance",
        "grader_behavior",
    ],
    reference: BaseModel,
    replay: BaseModel,
    tolerance: float,
) -> CandidateReplayComparisonV1:
    reference_payload = reference.model_dump(mode="json")
    replay_payload = replay.model_dump(mode="json")
    if family in {"trialeval_recovery", "trialdev_recovery"}:
        for field in (
            "participant_release",
            "evaluator_release",
            "verification_release",
        ):
            reference_payload[field] = Path(str(reference_payload[field])).name
            replay_payload[field] = Path(str(replay_payload[field])).name
    elif family == "grader_concordance":
        reference_payload["public_grader_command"] = ["<installed-harness>"]
        replay_payload["public_grader_command"] = ["<installed-harness>"]
    numeric, maximum, structural, nonnumeric = _compare_values(
        reference_payload,
        replay_payload,
        tolerance=tolerance,
    )
    return CandidateReplayComparisonV1(
        evidence_family=family,
        compared_numeric_value_count=numeric,
        maximum_absolute_difference=maximum,
        structural_difference_count=structural,
        nonnumeric_difference_count=nonnumeric,
        status="pass" if structural == 0 and nonnumeric == 0 else "fail",
    )


def _read_import_audits(paths: tuple[Path, ...]) -> tuple[bool, int]:
    repository_path_detected = False
    prohibited_imports: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        repository_path_detected = repository_path_detected or bool(
            payload["repository_path_detected"]
        )
        prohibited_imports.update(str(value) for value in payload["prohibited_imports"])
    return repository_path_detected, len(prohibited_imports)


def compare_candidate_clean_replay(
    *,
    release_root: Path,
    replay_root: Path,
    validation_wheel: Path,
    harness_wheel: Path,
    import_audits: tuple[Path, ...],
    installed_environment: Path,
    installation_constraints: Path,
    transcript: Path,
    output: Path,
    absolute_tolerance: float,
) -> CandidateCleanWheelReplayV1:
    """Compare installed-wheel replay receipts with the immutable candidate."""

    release = release_root.resolve()
    replay = replay_root.resolve()
    manifest = json.loads(
        (release / "RELEASE_MANIFEST.json").read_text(encoding="utf-8")
    )
    reference_trialeval = RecoverabilityReportV1.model_validate_json(
        (release / "recovery" / "trialeval" / "recoverability_report.json").read_text(
            encoding="utf-8"
        )
    )
    replay_trialeval = RecoverabilityReportV1.model_validate_json(
        (replay / "trialeval" / "recoverability_report.json").read_text(
            encoding="utf-8"
        )
    )
    reference_trialdev = RecoverabilityReportV1.model_validate_json(
        (release / "recovery" / "trialdev" / "recoverability_report.json").read_text(
            encoding="utf-8"
        )
    )
    replay_trialdev = RecoverabilityReportV1.model_validate_json(
        (replay / "trialdev" / "recoverability_report.json").read_text(encoding="utf-8")
    )
    reference_c5 = C5IntegrityRecoveryReportV1.model_validate_json(
        (release / "recovery" / "trialeval" / "c5_integrity_recovery.json").read_text(
            encoding="utf-8"
        )
    )
    replay_c5 = C5IntegrityRecoveryReportV1.model_validate_json(
        (replay / "trialeval" / "c5_integrity_recovery.json").read_text(
            encoding="utf-8"
        )
    )
    reference_grader = GraderConcordanceReportV1.model_validate_json(
        (
            release
            / "recovery"
            / "grader_concordance"
            / "grader_concordance_report.json"
        ).read_text(encoding="utf-8")
    )
    replay_grader = GraderConcordanceReportV1.model_validate_json(
        (replay / "grader_concordance" / "grader_concordance_report.json").read_text(
            encoding="utf-8"
        )
    )
    reference_behavior = GraderBehaviorReportV1.model_validate_json(
        (
            release / "recovery" / "grader_behavior" / "grader_behavior_report.json"
        ).read_text(encoding="utf-8")
    )
    replay_behavior = GraderBehaviorReportV1.model_validate_json(
        (replay / "grader_behavior" / "grader_behavior_report.json").read_text(
            encoding="utf-8"
        )
    )
    clean_room = json.loads(
        (replay / "clean_room" / "clean_room_workflow_report.json").read_text(
            encoding="utf-8"
        )
    )
    boundary_passed = clean_room.get("status") == "pass" and not clean_room.get(
        "findings"
    )
    repository_detected, prohibited_count = _read_import_audits(import_audits)
    comparisons = tuple(
        sorted(
            (
                _comparison(
                    family="trialeval_recovery",
                    reference=reference_trialeval,
                    replay=replay_trialeval,
                    tolerance=absolute_tolerance,
                ),
                _comparison(
                    family="trialdev_recovery",
                    reference=reference_trialdev,
                    replay=replay_trialdev,
                    tolerance=absolute_tolerance,
                ),
                _comparison(
                    family="trialeval_c5_integrity",
                    reference=reference_c5,
                    replay=replay_c5,
                    tolerance=absolute_tolerance,
                ),
                _comparison(
                    family="grader_concordance",
                    reference=reference_grader,
                    replay=replay_grader,
                    tolerance=absolute_tolerance,
                ),
                _comparison(
                    family="grader_behavior",
                    reference=reference_behavior,
                    replay=replay_behavior,
                    tolerance=absolute_tolerance,
                ),
                CandidateReplayComparisonV1(
                    evidence_family="role_boundary",
                    compared_numeric_value_count=0,
                    maximum_absolute_difference=0.0,
                    structural_difference_count=0 if boundary_passed else 1,
                    nonnumeric_difference_count=0,
                    status="pass" if boundary_passed else "fail",
                ),
            ),
            key=lambda row: row.evidence_family,
        )
    )
    receipt = CandidateCleanWheelReplayV1(
        release_id=str(manifest["release_id"]),
        source_commit=str(manifest["source_commit"]),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        installed_environment_sha256=sha256_file(installed_environment),
        installation_constraints_sha256=sha256_file(installation_constraints),
        transcript_sha256=sha256_file(transcript),
        absolute_tolerance=absolute_tolerance,
        isolated_install=True,
        network_blocked_during_replay=True,
        repository_on_import_path=repository_detected,
        prohibited_import_count=prohibited_count,
        wheels=(
            CleanWheelIdentityV1(
                package="trial-agent-bench",
                filename=harness_wheel.name,
                sha256=sha256_file(harness_wheel),
            ),
            CleanWheelIdentityV1(
                package="trialagentbench-validation",
                filename=validation_wheel.name,
                sha256=sha256_file(validation_wheel),
            ),
        ),
        comparisons=comparisons,
        status="pass",
    )
    write_model(output, receipt)
    return receipt


__all__ = [
    "CandidateCleanWheelReplayV1",
    "CandidateReplayComparisonV1",
    "CleanWheelIdentityV1",
    "compare_candidate_clean_replay",
]
