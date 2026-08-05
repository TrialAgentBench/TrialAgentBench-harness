"""Worked-example release verification tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from pydantic import ValidationError

from trialagentbench_validation.demonstrations import contracts
from trialagentbench_validation.demonstrations import (
    verification as demonstration_verification,
)
from trialagentbench_validation.recovery import (
    RecoverabilityReportV1,
    RecoverabilityRouteV1,
)


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _write_roles(
    tmp_path: Path,
    *,
    artifact_sha256: str | None = None,
) -> tuple[Path, Path, Path]:
    participant_body = b'{"question":"Estimate the treatment effect."}'
    evaluator_body = b'{"route":"route-1"}'
    verification_body = b'{"replay":"matched"}'
    case = contracts.DemonstrationCaseV1(
        release_id="release-1",
        case_id="case-1",
        suite="trialeval",
        disclosure_status="demonstration_only",
        title="Parallel-group survival example",
        question="Estimate the treatment effect at one year.",
        evidence_boundary="Only participant-visible files are used.",
        trial_structure=("Individually randomized trial.",),
        artifacts=(
            contracts.DemonstrationArtifactV1(
                role="participant",
                path="items/demo/task.json",
                sha256=artifact_sha256 or _sha256(participant_body),
            ),
            contracts.DemonstrationArtifactV1(
                role="evaluator",
                path="grader/route.json",
                sha256=_sha256(evaluator_body),
            ),
            contracts.DemonstrationArtifactV1(
                role="verification",
                path="replay/route.json",
                sha256=_sha256(verification_body),
            ),
        ),
        diagnostics=(
            contracts.DemonstrationDiagnosticV1(
                diagnostic_id="ph-check",
                operation="Inspect the released proportional-hazards diagnostic",
                value=0.42,
                unit="p-value",
                interpretation="The released evidence does not reject proportional hazards.",
                evidence_paths=("items/demo/task.json",),
            ),
        ),
        routes=(
            contracts.DemonstrationRouteV1(
                unit_id="ITEM1",
                route_id="route-1",
                disposition="credit_eligible",
                estimator_family="cox_ph",
                effect_scale="log_hr",
                rationale="The estimand and released diagnostic support this route.",
                result_summary="log HR -0.50",
            ),
        ),
        required_recoverability_routes=(
            contracts.DemonstrationRecoverabilityKeyV1(
                unit_id="ITEM1",
                route_id="route-1",
            ),
        ),
        consequence="The credit-eligible route determines the planning effect.",
        limitations=("This demonstration is not a scored evaluation item.",),
    )
    case_body = (case.model_dump_json(indent=2) + "\n").encode()
    index = contracts.DemonstrationIndexV1(
        release_id="release-1",
        cases=(
            contracts.DemonstrationIndexEntryV1(
                case_id=case.case_id,
                suite=case.suite,
                disclosure_status=case.disclosure_status,
                case_path="demonstrations/cases/case-1.json",
                case_sha256=_sha256(case_body),
                case_checksum=case.checksum or "",
            ),
        ),
    )

    participant = tmp_path / "participant.zip"
    evaluator = tmp_path / "evaluator.zip"
    verification = tmp_path / "verification.zip"
    with ZipFile(participant, "w") as archive:
        archive.writestr("items/demo/task.json", participant_body)
    with ZipFile(evaluator, "w") as archive:
        archive.writestr("grader/route.json", evaluator_body)
    with ZipFile(verification, "w") as archive:
        archive.writestr("replay/route.json", verification_body)
        archive.writestr("demonstrations/cases/case-1.json", case_body)
        archive.writestr("demonstrations/index.json", index.model_dump_json())
    return participant, evaluator, verification


def _recovery(*, status: str = "pass") -> RecoverabilityReportV1:
    route = RecoverabilityRouteV1(
        suite="trialeval",
        unit_id="ITEM1",
        context_or_checkpoint_id="C1",
        route_id="route-1",
        estimator_family="cox_ph",
        effect_scale="log_hr",
        result_kind="numeric_point",
        comparison_denominator=1,
        maximum_absolute_difference=0.0,
        declared_absolute_tolerance=1e-6,
        difference_to_tolerance_ratio=0.0,
        comparison_rule="numeric_envelope",
        recovery_path="direct_analysis_ready",
        public_input_paths=("items/ITEM1/data/ADSL.parquet",),
        expected_summary="absolute_tolerance=1e-06",
        reproduced_summary="maximum_absolute_difference=0",
        status=status,
    )
    return RecoverabilityReportV1(
        suite="trialeval",
        participant_release="participant.zip",
        evaluator_release="evaluator.zip",
        verification_release="verification.zip",
        required_route_count=1,
        replayed_route_count=1,
        failed_route_count=0 if status == "pass" else 1,
        maximum_absolute_difference=0.0,
        routes=(route,),
        status=status,
    )


def test_verification_exports_canonical_record_without_review_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participant, evaluator, verification = _write_roles(tmp_path)
    monkeypatch.setattr(
        demonstration_verification,
        "recover_release",
        lambda **_: _recovery(),
    )

    report = demonstration_verification.verify_worked_examples(
        participant_release=participant,
        evaluator_release=evaluator,
        verification_release=verification,
        case_ids=("case-1",),
        output_dir=tmp_path / "output",
    )

    assert report.status == "pass"
    assert (tmp_path / "output" / "case-1" / "case.json").is_file()
    assert not (tmp_path / "output" / "case-1" / "casebook.md").exists()
    assert not (tmp_path / "output" / "case-1" / "proof_card.md").exists()


def test_verification_rejects_artifact_checksum_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participant, evaluator, verification = _write_roles(
        tmp_path,
        artifact_sha256="0" * 64,
    )
    monkeypatch.setattr(
        demonstration_verification,
        "recover_release",
        lambda **_: _recovery(),
    )

    with pytest.raises(ValueError, match="artifact checksum failed"):
        demonstration_verification.verify_worked_examples(
            participant_release=participant,
            evaluator_release=evaluator,
            verification_release=verification,
            case_ids=None,
            output_dir=tmp_path / "output",
        )


def test_verification_requires_exact_unit_route_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participant, evaluator, verification = _write_roles(tmp_path)
    wrong_unit = _recovery().model_copy(
        update={
            "routes": (_recovery().routes[0].model_copy(update={"unit_id": "OTHER"}),)
        }
    )
    monkeypatch.setattr(
        demonstration_verification,
        "recover_release",
        lambda **_: wrong_unit,
    )

    with pytest.raises(ValueError, match="unrecovered routes"):
        demonstration_verification.verify_worked_examples(
            participant_release=participant,
            evaluator_release=evaluator,
            verification_release=verification,
            case_ids=None,
            output_dir=tmp_path / "output",
        )


@pytest.mark.parametrize("path", ("../secret", "/absolute", "a//b", r"a\b"))
def test_demonstration_artifact_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="safe relative paths"):
        contracts.DemonstrationArtifactV1(
            role="participant",
            path=path,
            sha256="0" * 64,
        )


def test_case_checksum_rejects_mutation() -> None:
    payload = {
        "release_id": "release-1",
        "case_id": "case-1",
        "suite": "trialeval",
        "disclosure_status": "demonstration_only",
        "title": "Example",
        "question": "Question",
        "evidence_boundary": "Public evidence",
        "trial_structure": ["RCT"],
        "artifacts": [
            {
                "role": "participant",
                "path": "task.json",
                "sha256": "0" * 64,
            }
        ],
        "routes": [
            {
                "unit_id": "ITEM1",
                "route_id": "route-1",
                "disposition": "credit_eligible",
                "estimator_family": "cox_ph",
                "effect_scale": "log_hr",
                "rationale": "Supported",
            }
        ],
        "required_recoverability_routes": [{"unit_id": "ITEM1", "route_id": "route-1"}],
        "consequence": "Planning",
        "limitations": ["Illustrative"],
        "checksum": "0" * 64,
    }

    with pytest.raises(ValidationError, match="checksum is invalid"):
        contracts.DemonstrationCaseV1.model_validate_json(json.dumps(payload))
