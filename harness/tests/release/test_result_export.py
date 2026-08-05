"""Tests for the release-bound collaborator result exporter."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from trialagentbench_harness.contracts.release.result_export import (
    ResultExportBundleManifestV1,
)
from trialagentbench_harness.tools.export_results import export_results_v1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _release(root: Path) -> Path:
    contract_path = root / "metadata" / "export_results_manifest.json"
    _write_json(root / "metadata" / "canonical_result.schema.json", {"type": "object"})
    (root / "metadata" / "simulation_properties.jsonl").write_text("{}\n", encoding="utf-8")
    _write_json(
        contract_path,
        {
            "schema_id": "trialagentbench.export_results_manifest/v1",
            "release_id": "release-v1",
            "status": "awaiting_model_runs",
            "canonical_result_schema_path": "metadata/canonical_result.schema.json",
            "simulation_properties_path": "metadata/simulation_properties.jsonl",
            "expected_analysis_unit_count": 550,
            "join_key": "canonical_result_join_key",
            "denominator_policy": "retain_every_scheduled_assignment_with_typed_status",
            "result_export_command": "trialagentbench export results",
            "required_artifact_kinds": ["run", "grade", "verification"],
        },
    )
    _write_json(
        root / "RELEASE_MANIFEST.json",
        {
            "release_id": "release-v1",
            "release_stage": "collaborator_single_seed",
            "result_export_manifest": {
                "path": "metadata/export_results_manifest.json",
                "sha256": _sha256(contract_path),
            },
        },
    )
    return root


def _artifacts(root: Path) -> tuple[Path, Path, Path, Path]:
    paths = tuple(root / name for name in ("run", "grade", "verification", "analysis"))
    for index, path in enumerate(paths):
        (path / "nested").mkdir(parents=True)
        (path / "nested" / "record.json").write_text(f'{{"value": {index}}}\n', encoding="utf-8")
    return paths


def test_result_export_is_deterministic_and_self_verifying(tmp_path: Path) -> None:
    """Equivalent inputs must produce an identical portable archive."""

    release = _release(tmp_path / "release")
    run, grade, verification, analysis = _artifacts(tmp_path / "artifacts")
    first = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "first",
        run_roots=[run],
        grade_roots=[grade],
        verification_roots=[verification],
        analysis_roots=[analysis],
    )
    second = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "second",
        run_roots=[run],
        grade_roots=[grade],
        verification_roots=[verification],
        analysis_roots=[analysis],
    )

    archive_name = "release-v1-results.zip"
    first_archive = first / archive_name
    second_archive = second / archive_name
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert (first / f"{archive_name}.sha256").read_text(encoding="utf-8") == (
        f"{_sha256(first_archive)}  {archive_name}\n"
    )

    receipt = ResultExportBundleManifestV1.model_validate_json(
        (first / "result_export_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt.release_id == "release-v1"
    assert {source.kind for source in receipt.sources} == {
        "run",
        "grade",
        "verification",
        "analysis",
    }
    serialized = json.dumps(receipt.model_dump(mode="json"))
    assert str(tmp_path) not in serialized

    with zipfile.ZipFile(first_archive) as archive:
        assert archive.read("manifest.json") == (first / "result_export_receipt.json").read_bytes()
        checksums = {
            line.split("  ", maxsplit=1)[1]: line.split("  ", maxsplit=1)[0]
            for line in archive.read("SHA256SUMS").decode().splitlines()
        }
        assert checksums["manifest.json"] == hashlib.sha256(archive.read("manifest.json")).hexdigest()
        for member in receipt.members:
            assert checksums[member.path] == member.sha256
            assert hashlib.sha256(archive.read(member.path)).hexdigest() == member.sha256


def test_result_export_rejects_release_contract_mutation(tmp_path: Path) -> None:
    """The outer release identity must bind the executable export contract."""

    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts")
    (release / "metadata" / "export_results_manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contract checksum"):
        export_results_v1(
            release_root=release,
            output_dir=tmp_path / "output",
            run_roots=[run],
            grade_roots=[grade],
            verification_roots=[verification],
        )


@pytest.mark.parametrize("unsafe_name", [".env", "private.pem", "cached.pyc"])
def test_result_export_rejects_credentials_and_transient_files(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    """Credential-shaped and transient files must never enter result bundles."""

    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts")
    (run / unsafe_name).write_text("not-a-secret\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden credential or transient"):
        export_results_v1(
            release_root=release,
            output_dir=tmp_path / "output",
            run_roots=[run],
            grade_roots=[grade],
            verification_roots=[verification],
        )


def test_result_export_preserves_immutable_release_boundary(tmp_path: Path) -> None:
    """Neither caller results nor export output may be placed in the release."""

    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts")
    release_result = release / "caller-result"
    release_result.mkdir()
    (release_result / "record.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the immutable release"):
        export_results_v1(
            release_root=release,
            output_dir=tmp_path / "output",
            run_roots=[release_result],
            grade_roots=[grade],
            verification_roots=[verification],
        )
    with pytest.raises(ValueError, match="outside the immutable release"):
        export_results_v1(
            release_root=release,
            output_dir=release / "output",
            run_roots=[run],
            grade_roots=[grade],
            verification_roots=[verification],
        )
