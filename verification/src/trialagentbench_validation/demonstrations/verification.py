"""Verify and export public worked-example records from release archives."""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

from trialagentbench_validation.contracts.v1_scope import (
    RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
)
from trialagentbench_validation.demonstrations.contracts import (
    DemonstrationCaseV1,
    DemonstrationIndexV1,
    DemonstrationVerificationReportV1,
)
from trialagentbench_validation.recovery import recover_release


def _member_sha256(archive: ZipFile, path: str) -> str:
    digest = hashlib.sha256()
    with archive.open(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_case(
    verification: ZipFile,
    *,
    case_path: str,
    expected_sha256: str,
    expected_checksum: str,
) -> DemonstrationCaseV1:
    body = verification.read(case_path)
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError(
            f"demonstration case bytes do not match the index: {case_path}"
        )
    case = DemonstrationCaseV1.model_validate_json(body)
    if case.checksum != expected_checksum:
        raise ValueError(f"demonstration case contract checksum drifted: {case_path}")
    return case


def verify_worked_examples(
    *,
    participant_release: Path,
    evaluator_release: Path,
    verification_release: Path,
    case_ids: tuple[str, ...] | None,
    output_dir: Path,
    workers: int = 1,
    absolute_tolerance: float = RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
) -> DemonstrationVerificationReportV1:
    """Verify release-bound cases and export their canonical records."""

    recovery = recover_release(
        participant_release=participant_release,
        evaluator_release=evaluator_release,
        verification_release=verification_release,
        workers=workers,
        absolute_tolerance=absolute_tolerance,
    )
    with (
        ZipFile(participant_release) as participant,
        ZipFile(evaluator_release) as evaluator,
        ZipFile(verification_release) as verification,
    ):
        index = DemonstrationIndexV1.model_validate_json(
            verification.read("demonstrations/index.json")
        )
        entries = {row.case_id: row for row in index.cases}
        requested = tuple(entries) if case_ids is None else case_ids
        unknown = sorted(set(requested) - set(entries))
        if unknown:
            raise ValueError(f"unknown demonstration case IDs: {unknown!r}")
        if len(set(requested)) != len(requested):
            raise ValueError("requested demonstration case IDs must be unique")
        cases = tuple(
            _read_case(
                verification,
                case_path=entries[case_id].case_path,
                expected_sha256=entries[case_id].case_sha256,
                expected_checksum=entries[case_id].case_checksum,
            )
            for case_id in requested
        )
        if any(case.release_id != index.release_id for case in cases):
            raise ValueError(
                "demonstration cases and index identify different releases"
            )
        archives = {
            "participant": participant,
            "evaluator": evaluator,
            "verification": verification,
        }
        for case in cases:
            for artifact in case.artifacts:
                archive = archives[artifact.role]
                if _member_sha256(archive, artifact.path) != artifact.sha256:
                    raise ValueError(
                        f"demonstration artifact checksum failed: "
                        f"{artifact.role}/{artifact.path}"
                    )

    passed_routes = {
        (row.unit_id, row.route_id) for row in recovery.routes if row.status == "pass"
    }
    for case in cases:
        missing = sorted(
            {(row.unit_id, row.route_id) for row in case.required_recoverability_routes}
            - passed_routes
        )
        if missing:
            raise ValueError(
                f"demonstration references unrecovered routes: "
                f"{case.case_id}/{missing!r}"
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    for case in cases:
        case_root = output_dir / case.case_id
        case_root.mkdir()
        (case_root / "case.json").write_text(
            case.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    report = DemonstrationVerificationReportV1(
        release_id=index.release_id,
        requested_case_ids=requested,
        verified_case_ids=tuple(case.case_id for case in cases),
        recoverability_status=recovery.status,
        status="pass" if recovery.status == "pass" else "fail",
    )
    (output_dir / "demonstration_verification.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["verify_worked_examples"]
