"""Tests for clinical-process source custody and identity resolution."""

from __future__ import annotations

import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from trialagentbench_validation.external.sources.custody import (
    CanonicalTrialIdentityV1,
    ClinicalProcessSourceManifestV1,
    ClinicalProcessSourceReceiptV1,
    EligibilityStatus,
    EvidenceVisibility,
    ExternalCorpusCoverageReportV1,
    LocalProcessSourceBindingV1,
    PermittedDerivedOutput,
    ProcessSourceRole,
    ProcessSourceType,
    SourceEligibilityDecisionV1,
    TrialIdentityTableV1,
    TrialOccurrenceV1,
    credential_preflight,
    inventory_immport,
    verify_process_source_manifest,
    write_safe_evidence_bundle,
)

_TABLE_ROWS = {
    "subject.txt": ("subject_accession", ("SUB1", "SUB2")),
    "arm_or_cohort.txt": ("arm_accession", ("ARM1", "ARM2")),
    "planned_visit.txt": ("planned_visit_accession", ("VIS1",)),
    "assessment_component.txt": ("assessment_accession", ("ASSESS1", "ASSESS2")),
    "lab_test.txt": ("lab_test_accession", ("LAB1",)),
    "adverse_event.txt": ("adverse_event_accession", ("AE1",)),
    "intervention.txt": ("intervention_accession", ("INT1",)),
}


def _archive(
    vault_root: Path,
    *,
    accession: str,
    version: str = "DR58",
    registry_id: str,
    omitted_tables: tuple[str, ...] = (),
) -> Path:
    root = f"{accession}-{version}_Tab"
    path = vault_root / "immport" / "tabular" / accession / f"{root}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/", "")
        archive.writestr(f"{root}/Tab/", "")
        archive.writestr(f"{root}/Tab/study.txt", f"study_accession\n{accession}\n")
        for name, (header, rows) in _TABLE_ROWS.items():
            if name in omitted_tables:
                continue
            archive.writestr(
                f"{root}/Tab/{name}",
                "\n".join((header, *rows)) + "\n",
            )
        archive.writestr(
            f"{root}/Tab/study_link.txt",
            f"study_accession\tlink\n{accession}\thttps://clinicaltrials.gov/{registry_id}\n",
        )
    return path


def _inventory(
    vault_root: Path,
) -> tuple[
    ClinicalProcessSourceManifestV1,
    TrialIdentityTableV1,
    ExternalCorpusCoverageReportV1,
    tuple[SourceEligibilityDecisionV1, ...],
    tuple[LocalProcessSourceBindingV1, ...],
]:
    return inventory_immport(
        vault_root=vault_root,
        observed_at_utc=datetime(2026, 7, 24, 12, tzinfo=UTC),
        imported_source_manifest_relative_path="evidence/external/source_manifest.json",
        imported_source_manifest_sha256="a" * 64,
    )


def _receipt(**updates: object) -> ClinicalProcessSourceReceiptV1:
    payload = {
        "source_object_id": "immport:SDY1:DR58",
        "source_type": "immport",
        "canonical_accession": "SDY1",
        "immutable_version": "DR58",
        "object_name": "SDY1-DR58_Tab.zip",
        "media_type": "application/zip",
        "byte_size": 100,
        "sha256": "a" * 64,
        "observed_at_utc": "2026-07-24T12:00:00Z",
        "acquisition_locator": "immport:SDY1/DR58",
        "retrieval_method": "verified local package",
        "role": "structural_screening_only",
        "visibility": "not_applicable",
        "eligibility_status": "pending_human_review",
        "license_evidence": "Review required.",
        "credential_class": "credentialed",
        "raw_redistribution_permitted": False,
        "permitted_derived_output": "structural_counts",
        "human_review_required": True,
    }
    payload.update(updates)
    return ClinicalProcessSourceReceiptV1.model_validate(payload)


def _occurrence(
    occurrence_id: str,
    *,
    role: ProcessSourceRole,
    visibility: EvidenceVisibility,
) -> TrialOccurrenceV1:
    return TrialOccurrenceV1(
        occurrence_id=occurrence_id,
        source_object_id=occurrence_id,
        canonical_accession=occurrence_id,
        registry_identifiers=("NCT00000001",),
        relationship="same_trial",
        resolution_basis="exact_registry_id",
        role=role,
        visibility=visibility,
    )


def test_immport_inventory_resolves_fifteen_packages_to_fourteen_trials(
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "external"
    for index in range(15):
        registry_index = 13 if index == 14 else index
        _archive(
            vault_root,
            accession=f"SDY{index + 1}",
            version="DR60" if index == 6 else "DR58",
            registry_id=f"NCT{registry_index:08d}",
        )

    manifest, identities, coverage, eligibility, bindings = _inventory(vault_root)

    assert len(manifest.receipts) == 15
    assert len(identities.trials) == 14
    assert len(bindings) == 15
    assert coverage.source_object_count == 15
    assert coverage.canonical_trial_count == 14
    assert coverage.real_distribution_eligible_trial_count == 0
    assert sum(row.subject_records for row in coverage.rows) == 30
    assert all(
        decision.status == EligibilityStatus.PENDING_HUMAN_REVIEW
        for decision in eligibility
    )
    duplicate = next(
        trial for trial in identities.trials if len(trial.occurrences) == 2
    )
    assert {row.relationship for row in duplicate.occurrences} == {
        "same_trial",
        "companion",
    }


def test_immport_inventory_can_bind_an_exact_source_set(tmp_path: Path) -> None:
    vault_root = tmp_path / "external"
    _archive(vault_root, accession="SDY1", registry_id="NCT00000001")
    _archive(vault_root, accession="SDY2", registry_id="NCT00000002")

    manifest, identities, coverage, _, _ = inventory_immport(
        vault_root=vault_root,
        observed_at_utc=datetime(2026, 7, 24, 12, tzinfo=UTC),
        imported_source_manifest_relative_path="evidence/external/source_manifest.json",
        imported_source_manifest_sha256="a" * 64,
        source_object_ids=("immport:SDY2:DR58",),
    )

    assert [row.source_object_id for row in manifest.receipts] == ["immport:SDY2:DR58"]
    assert len(identities.trials) == 1
    assert coverage.source_object_count == 1

    with pytest.raises(FileNotFoundError, match="source objects are unavailable"):
        inventory_immport(
            vault_root=vault_root,
            observed_at_utc=datetime(2026, 7, 24, 12, tzinfo=UTC),
            imported_source_manifest_relative_path="evidence/external/source_manifest.json",
            imported_source_manifest_sha256="a" * 64,
            source_object_ids=("immport:SDY3:DR58",),
        )


def test_manifest_verification_detects_checksum_drift(tmp_path: Path) -> None:
    vault_root = tmp_path / "external"
    archive = _archive(vault_root, accession="SDY1", registry_id="NCT00000001")
    manifest, _, _, _, bindings = _inventory(vault_root)
    archive.write_bytes(archive.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="byte-size drift"):
        verify_process_source_manifest(manifest, bindings=bindings)


def test_absent_optional_process_table_is_zero_coverage(tmp_path: Path) -> None:
    vault_root = tmp_path / "external"
    _archive(
        vault_root,
        accession="SDY1",
        registry_id="NCT00000001",
        omitted_tables=("lab_test.txt", "intervention.txt"),
    )

    _, _, coverage, _, _ = _inventory(vault_root)

    assert coverage.rows[0].laboratory_records == 0
    assert coverage.rows[0].intervention_records == 0


def test_manifest_requires_exact_binding_set(tmp_path: Path) -> None:
    vault_root = tmp_path / "external"
    _archive(vault_root, accession="SDY1", registry_id="NCT00000001")
    manifest, _, _, _, _ = _inventory(vault_root)

    with pytest.raises(ValueError, match="exactly match"):
        verify_process_source_manifest(manifest, bindings=())


def test_pro_act_is_rejected_by_source_contract() -> None:
    with pytest.raises(ValidationError, match="PRO-ACT is prohibited"):
        _receipt(source_type=ProcessSourceType.PRO_ACT)


def test_cdisc_artificial_data_cannot_enter_distribution_role() -> None:
    with pytest.raises(ValidationError, match="format/workflow control"):
        _receipt(
            source_type=ProcessSourceType.CDISC_ARTIFICIAL,
            role=ProcessSourceRole.CALIBRATION,
        )


def test_signed_or_local_acquisition_locators_are_rejected() -> None:
    with pytest.raises(ValidationError, match="signed URL"):
        _receipt(acquisition_locator="https://example.test/data?token=secret")
    with pytest.raises(ValidationError, match="absolute local path"):
        _receipt(acquisition_locator="/private/vault/data.zip")


def test_canonical_trial_rejects_cross_role_alias() -> None:
    with pytest.raises(ValidationError, match="crosses evidence roles"):
        CanonicalTrialIdentityV1(
            canonical_trial_id="clinicaltrials.gov:NCT00000001",
            occurrences=(
                _occurrence(
                    "source:a",
                    role=ProcessSourceRole.CALIBRATION,
                    visibility=EvidenceVisibility.OPENED,
                ),
                _occurrence(
                    "source:b",
                    role=ProcessSourceRole.HELD_OUT,
                    visibility=EvidenceVisibility.UNOPENED,
                ),
            ),
        )


def test_previously_inspected_trial_cannot_reenter_unopened() -> None:
    with pytest.raises(ValidationError, match="re-enters as unopened"):
        CanonicalTrialIdentityV1(
            canonical_trial_id="clinicaltrials.gov:NCT00000001",
            occurrences=(
                _occurrence(
                    "source:a",
                    role=ProcessSourceRole.IMPORTED_PREVIOUSLY_INSPECTED,
                    visibility=EvidenceVisibility.PREVIOUSLY_INSPECTED,
                ),
                _occurrence(
                    "source:b",
                    role=ProcessSourceRole.HELD_OUT,
                    visibility=EvidenceVisibility.UNOPENED,
                ),
            ),
        )


def test_credential_preflight_requires_environment_and_private_permissions(
    tmp_path: Path,
) -> None:
    credential = tmp_path / "credential.json"
    credential.write_text('{"credential": "not-emitted"}\n', encoding="utf-8")
    credential.chmod(0o600)

    report = credential_preflight({"IMMPORT_KEY_FILE": credential.as_posix()})

    assert report.byte_size == credential.stat().st_size
    assert "not-emitted" not in report.model_dump_json()
    credential.chmod(0o640)
    with pytest.raises(PermissionError, match="group or other"):
        credential_preflight({"IMMPORT_KEY_FILE": credential.as_posix()})
    with pytest.raises(ValueError, match="IMMPORT_KEY_FILE is required"):
        credential_preflight({})
    credential.write_bytes(b"x" * (64 * 1024 + 1))
    credential.chmod(0o600)
    with pytest.raises(ValueError, match="size limit"):
        credential_preflight({"IMMPORT_KEY_FILE": credential.as_posix()})


def test_safe_evidence_bundle_contains_no_raw_rows_or_local_paths(
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "private" / "external"
    _archive(vault_root, accession="SDY1", registry_id="NCT00000001")
    manifest, identities, coverage, eligibility, _ = _inventory(vault_root)
    output_root = tmp_path / "safe"

    write_safe_evidence_bundle(
        output_root=output_root,
        manifest=manifest,
        identities=identities,
        coverage=coverage,
        eligibility=eligibility,
    )

    output_text = "\n".join(
        path.read_text(encoding="utf-8") for path in output_root.iterdir()
    )
    assert vault_root.as_posix() not in output_text
    assert "SUB1" not in output_text
    assert "not-emitted" not in output_text
    assert {
        "source_manifest.json",
        "source_receipts.jsonl",
        "license_and_redistribution.csv",
        "trial_identity_table.json",
        "identity_collision_report.json",
        "source_eligibility.jsonl",
        "source_coverage.csv",
        "excluded_sources.jsonl",
        "acquisition_verification_report.json",
        "artifact_manifest.json",
    } == {path.name for path in output_root.iterdir()}


def test_source_receipts_do_not_serialize_bindings(tmp_path: Path) -> None:
    binding = LocalProcessSourceBindingV1(
        source_object_id="immport:SDY1:DR58",
        local_path=tmp_path / "private.zip",
    )
    manifest = ClinicalProcessSourceManifestV1(
        imported_source_manifest_relative_path="evidence/source_manifest.json",
        imported_source_manifest_sha256="a" * 64,
        receipts=(_receipt(),),
    )

    assert "local_path" not in manifest.model_dump_json()
    assert os.fspath(binding.local_path) not in manifest.model_dump_json()
    assert (
        manifest.receipts[0].permitted_derived_output
        == PermittedDerivedOutput.STRUCTURAL_COUNTS
    )
