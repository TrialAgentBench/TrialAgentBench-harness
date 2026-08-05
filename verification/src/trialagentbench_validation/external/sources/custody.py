"""Offline custody and identity verification for clinical-process sources."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import stat
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trialagentbench_validation.io import sha256_file, write_model

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IMM_PORT_OBJECT_PATTERN = re.compile(r"^(SDY[0-9]+)-(DR[0-9]+)_Tab\.zip$")
_REGISTRY_ID_PATTERN = re.compile(rb"NCT[0-9]{8}")
_SENSITIVE_OUTPUT_TOKENS = (
    '"api_key"',
    '"token"',
    '"secret"',
    '"subject_id"',
    '"participant_id"',
    '"signed_url"',
    "x-amz-signature",
    "x-goog-signature",
    "file://",
    "/home/",
)
_STRUCTURAL_TABLES = {
    "subject_records": "subject.txt",
    "arms": "arm_or_cohort.txt",
    "planned_visits": "planned_visit.txt",
    "assessment_components": "assessment_component.txt",
    "laboratory_records": "lab_test.txt",
    "adverse_event_records": "adverse_event.txt",
    "intervention_records": "intervention.txt",
}


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProcessSourceType(str, Enum):
    """Supported source families at the custody boundary."""

    IMMPORT = "immport"
    CDISC_ARTIFICIAL = "cdisc_artificial"
    DRYAD = "dryad"
    ZENODO = "zenodo"
    IMPORTED_AACT = "imported_aact"
    IMPORTED_RCT_BENCH = "imported_rct_bench"
    PRO_ACT = "pro_act"


class ProcessSourceRole(str, Enum):
    """Permitted evidence role for one source object."""

    STRUCTURAL_SCREENING_ONLY = "structural_screening_only"
    CALIBRATION = "calibration"
    HELD_OUT = "held_out"
    SECONDARY_VALIDATION = "secondary_validation"
    FORMAT_AND_WORKFLOW_CONTROL = "format_and_workflow_control"
    IMPORTED_PREVIOUSLY_INSPECTED = "imported_previously_inspected"


class EvidenceVisibility(str, Enum):
    """Whether evidence was inspected before the additive-source freeze."""

    NOT_APPLICABLE = "not_applicable"
    PREVIOUSLY_INSPECTED = "previously_inspected"
    UNOPENED = "unopened"
    OPENED = "opened"


class EligibilityStatus(str, Enum):
    """Current eligibility disposition for one exact object."""

    ELIGIBLE = "eligible"
    EXCLUDED = "excluded"
    PENDING_HUMAN_REVIEW = "pending_human_review"


class PermittedDerivedOutput(str, Enum):
    """Highest output class permitted by the current disposition."""

    NONE = "none"
    STRUCTURAL_COUNTS = "structural_counts"
    AGGREGATE_STATISTICS = "aggregate_statistics"
    REDISTRIBUTABLE_DERIVATIVES = "redistributable_derivatives"


class ClinicalProcessSourceReceiptV1(_Contract):
    """Immutable, nonsecret identity of one exact source object."""

    schema_id: Literal["trialagentbench.clinical_process_source_receipt/v1"] = (
        "trialagentbench.clinical_process_source_receipt/v1"
    )
    source_object_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]+$")
    source_type: ProcessSourceType
    canonical_accession: str = Field(min_length=1)
    immutable_version: str = Field(min_length=1)
    object_name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    byte_size: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    observed_at_utc: datetime
    acquisition_locator: str = Field(min_length=1)
    retrieval_method: str = Field(min_length=1)
    role: ProcessSourceRole
    visibility: EvidenceVisibility
    eligibility_status: EligibilityStatus
    license_evidence: str = Field(min_length=1)
    credential_class: str = Field(min_length=1)
    raw_redistribution_permitted: bool
    permitted_derived_output: PermittedDerivedOutput
    human_review_required: bool
    exclusion_reason: str | None = Field(default=None, min_length=1)

    @field_validator("acquisition_locator")
    @classmethod
    def _locator_is_safe(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme == "file" or parsed.query or parsed.fragment:
            raise ValueError(
                "acquisition_locator must not contain local or signed URL details"
            )
        if Path(value).is_absolute():
            raise ValueError(
                "acquisition_locator must not contain an absolute local path"
            )
        return value

    @model_validator(mode="after")
    def _source_policy_is_consistent(self) -> ClinicalProcessSourceReceiptV1:
        if self.source_type == ProcessSourceType.PRO_ACT:
            raise ValueError("PRO-ACT is prohibited from this source portfolio")
        if self.source_type == ProcessSourceType.CDISC_ARTIFICIAL:
            if self.role != ProcessSourceRole.FORMAT_AND_WORKFLOW_CONTROL:
                raise ValueError(
                    "CDISC artificial data may only be a format/workflow control"
                )
        if (
            self.eligibility_status == EligibilityStatus.EXCLUDED
            and self.exclusion_reason is None
        ):
            raise ValueError("excluded sources require an exclusion_reason")
        if (
            self.eligibility_status != EligibilityStatus.EXCLUDED
            and self.exclusion_reason is not None
        ):
            raise ValueError("only excluded sources may declare an exclusion_reason")
        if (
            self.human_review_required
            and self.eligibility_status == EligibilityStatus.ELIGIBLE
        ):
            raise ValueError("human review must close before an object is eligible")
        if self.raw_redistribution_permitted and self.human_review_required:
            raise ValueError(
                "raw redistribution cannot be permitted while human review is pending"
            )
        if (
            self.visibility == EvidenceVisibility.UNOPENED
            and self.role != ProcessSourceRole.HELD_OUT
        ):
            raise ValueError("only held-out sources may have unopened visibility")
        return self


class ClinicalProcessSourceManifestV1(_Contract):
    """Supplemental source portfolio bound to the imported 4_5 authority."""

    schema_id: Literal["trialagentbench.clinical_process_source_manifest/v1"] = (
        "trialagentbench.clinical_process_source_manifest/v1"
    )
    imported_source_manifest_relative_path: str
    imported_source_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    receipts: tuple[ClinicalProcessSourceReceiptV1, ...]

    @field_validator("imported_source_manifest_relative_path")
    @classmethod
    def _imported_path_is_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "imported source manifest path must be repository relative"
            )
        return value

    @model_validator(mode="after")
    def _receipt_ids_are_unique(self) -> ClinicalProcessSourceManifestV1:
        ids = [receipt.source_object_id for receipt in self.receipts]
        if len(ids) != len(set(ids)):
            raise ValueError("source_object_id values must be unique")
        return self


class SourceEligibilityDecisionV1(_Contract):
    """Explicit eligibility and legal-use disposition for one object."""

    schema_id: Literal["trialagentbench.source_eligibility_decision/v1"] = (
        "trialagentbench.source_eligibility_decision/v1"
    )
    source_object_id: str = Field(min_length=1)
    status: EligibilityStatus
    decision_basis: str = Field(min_length=1)
    approved_by_human: bool
    usable_process_families: tuple[str, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def _approval_matches_status(self) -> SourceEligibilityDecisionV1:
        if self.status == EligibilityStatus.ELIGIBLE and not self.approved_by_human:
            raise ValueError("eligible source decisions require human approval")
        return self


class TrialOccurrenceV1(_Contract):
    """One source occurrence linked to a conservative canonical trial."""

    occurrence_id: str = Field(min_length=1)
    source_object_id: str = Field(min_length=1)
    canonical_accession: str = Field(min_length=1)
    registry_identifiers: tuple[str, ...] = Field(min_length=1)
    publication_dois: tuple[str, ...] = ()
    relationship: Literal[
        "same_trial",
        "substudy",
        "companion",
        "publication_only",
        "independent",
    ]
    resolution_basis: Literal["exact_registry_id", "doi_link", "manual_review"]
    role: ProcessSourceRole
    visibility: EvidenceVisibility


class CanonicalTrialIdentityV1(_Contract):
    """All known source occurrences for one underlying trial."""

    canonical_trial_id: str = Field(min_length=1)
    occurrences: tuple[TrialOccurrenceV1, ...] = Field(min_length=1)
    manual_disposition: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _roles_do_not_conflict(self) -> CanonicalTrialIdentityV1:
        evidence_roles = {
            occurrence.role
            for occurrence in self.occurrences
            if occurrence.role
            in {
                ProcessSourceRole.CALIBRATION,
                ProcessSourceRole.HELD_OUT,
                ProcessSourceRole.SECONDARY_VALIDATION,
            }
        }
        if len(evidence_roles) > 1:
            raise ValueError(
                f"canonical trial {self.canonical_trial_id} crosses evidence roles"
            )
        visibility = {occurrence.visibility for occurrence in self.occurrences}
        if (
            EvidenceVisibility.PREVIOUSLY_INSPECTED in visibility
            and EvidenceVisibility.UNOPENED in visibility
        ):
            raise ValueError(
                f"canonical trial {self.canonical_trial_id} re-enters as unopened"
            )
        return self


class TrialIdentityTableV1(_Contract):
    """Conservative cross-source canonical trial identity table."""

    schema_id: Literal["trialagentbench.trial_identity_table/v1"] = (
        "trialagentbench.trial_identity_table/v1"
    )
    trials: tuple[CanonicalTrialIdentityV1, ...]

    @model_validator(mode="after")
    def _occurrences_are_unique(self) -> TrialIdentityTableV1:
        trial_ids = [trial.canonical_trial_id for trial in self.trials]
        occurrences = [
            occurrence.occurrence_id
            for trial in self.trials
            for occurrence in trial.occurrences
        ]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("canonical_trial_id values must be unique")
        if len(occurrences) != len(set(occurrences)):
            raise ValueError("occurrence_id values must be unique")
        return self


class SourceCoverageRowV1(_Contract):
    """Structural counts for one object without participant uniqueness claims."""

    source_object_id: str = Field(min_length=1)
    canonical_trial_id: str = Field(min_length=1)
    source_role: ProcessSourceRole
    eligibility_status: EligibilityStatus
    subject_records: int = Field(ge=0)
    arms: int = Field(ge=0)
    planned_visits: int = Field(ge=0)
    assessment_components: int = Field(ge=0)
    laboratory_records: int = Field(ge=0)
    adverse_event_records: int = Field(ge=0)
    intervention_records: int = Field(ge=0)


class ExternalCorpusCoverageReportV1(_Contract):
    """Coverage denominators with artificial controls kept separate."""

    schema_id: Literal["trialagentbench.external_corpus_coverage/v1"] = (
        "trialagentbench.external_corpus_coverage/v1"
    )
    source_object_count: int = Field(ge=0)
    canonical_trial_count: int = Field(ge=0)
    real_distribution_eligible_trial_count: int = Field(ge=0)
    format_control_object_count: int = Field(ge=0)
    rows: tuple[SourceCoverageRowV1, ...]
    denominator_note: Literal[
        "Subject records are source rows, not deduplicated unique participants."
    ] = "Subject records are source rows, not deduplicated unique participants."

    @model_validator(mode="after")
    def _counts_match_rows(self) -> ExternalCorpusCoverageReportV1:
        if self.source_object_count != len(self.rows):
            raise ValueError("source_object_count does not match coverage rows")
        format_controls = sum(
            row.source_role == ProcessSourceRole.FORMAT_AND_WORKFLOW_CONTROL
            for row in self.rows
        )
        if self.format_control_object_count != format_controls:
            raise ValueError("format_control_object_count does not match coverage rows")
        return self


class LocalProcessSourceBindingV1(_Contract):
    """Local-only binding between a receipt and source bytes."""

    source_object_id: str = Field(min_length=1)
    local_path: Path


class CredentialPreflightV1(_Contract):
    """Nonsecret result of checking the configured ImmPort credential."""

    schema_id: Literal["trialagentbench.immport_credential_preflight/v1"] = (
        "trialagentbench.immport_credential_preflight/v1"
    )
    configured: Literal[True] = True
    restrictive_permissions: Literal[True] = True
    json_object: Literal[True] = True
    byte_size: int = Field(gt=0)


class AcquisitionVerificationReportV1(_Contract):
    """Outcome of offline checksum and archive verification."""

    schema_id: Literal["trialagentbench.acquisition_verification_report/v1"] = (
        "trialagentbench.acquisition_verification_report/v1"
    )
    status: Literal["pass", "fail"]
    verified_object_count: int = Field(ge=0)
    canonical_trial_count: int = Field(ge=0)
    findings: tuple[str, ...] = ()


class ArtifactRecordV1(_Contract):
    """Identity of one safe generated evidence artifact."""

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_size: int = Field(gt=0)


class ArtifactManifestV1(_Contract):
    """Checksums for all safe custody evidence outputs."""

    schema_id: Literal["trialagentbench.process_source_artifact_manifest/v1"] = (
        "trialagentbench.process_source_artifact_manifest/v1"
    )
    artifacts: tuple[ArtifactRecordV1, ...]


def credential_preflight(
    environment: dict[str, str] | None = None,
) -> CredentialPreflightV1:
    """Validate the configured credential without emitting its path or content."""

    environ = os.environ if environment is None else environment
    configured_path = environ.get("IMMPORT_KEY_FILE")
    if not configured_path:
        raise ValueError("IMMPORT_KEY_FILE is required")
    path = Path(configured_path)
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("IMMPORT_KEY_FILE must reference a regular file")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PermissionError(
            "IMMPORT_KEY_FILE must not be accessible by group or other"
        )
    if metadata.st_size > 64 * 1024:
        raise ValueError("IMMPORT_KEY_FILE exceeds the credential size limit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("IMMPORT_KEY_FILE must contain a non-empty JSON object")
    return CredentialPreflightV1(byte_size=metadata.st_size)


def verify_process_source_manifest(
    manifest: ClinicalProcessSourceManifestV1,
    *,
    bindings: tuple[LocalProcessSourceBindingV1, ...],
) -> None:
    """Verify every source byte and its format without network access."""

    binding_by_id = {binding.source_object_id: binding for binding in bindings}
    receipt_by_id = {receipt.source_object_id: receipt for receipt in manifest.receipts}
    if len(binding_by_id) != len(bindings):
        raise ValueError("local bindings contain duplicate source_object_id values")
    if set(binding_by_id) != set(receipt_by_id):
        raise ValueError("local bindings must exactly match source receipts")
    for source_object_id, receipt in receipt_by_id.items():
        path = binding_by_id[source_object_id].local_path
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != receipt.byte_size:
            raise ValueError(f"byte-size drift for {source_object_id}")
        observed = sha256_file(path)
        if observed != receipt.sha256:
            raise ValueError(f"checksum drift for {source_object_id}")
        if receipt.source_type == ProcessSourceType.IMMPORT:
            _verify_immport_archive(
                path,
                accession=receipt.canonical_accession,
                version=receipt.immutable_version,
            )


def _verify_immport_archive(path: Path, *, accession: str, version: str) -> None:
    expected_name = f"{accession}-{version}_Tab.zip"
    if path.name != expected_name:
        raise ValueError(f"ImmPort object name mismatch: expected {expected_name}")
    expected_root = f"{accession}-{version}_Tab/Tab/"
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ImmPort ZIP CRC failure: {path.name}")
        names = tuple(info.filename for info in archive.infolist())
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate ZIP member in {path.name}")
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe ZIP member in {path.name}")
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted ZIP member in {path.name}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"symbolic-link ZIP member in {path.name}")
        files = tuple(name for name in names if not name.endswith("/"))
        if not files or any(not name.startswith(expected_root) for name in files):
            raise ValueError(f"unexpected ImmPort package root in {path.name}")
        required = {
            expected_root + "study.txt",
            expected_root + "study_link.txt",
        }
        missing = required - set(names)
        if missing:
            raise ValueError(
                f"ImmPort package {path.name} lacks required structural tables: {sorted(missing)}"
            )


def inventory_immport(
    *,
    vault_root: Path,
    observed_at_utc: datetime,
    imported_source_manifest_relative_path: str,
    imported_source_manifest_sha256: str,
    source_object_ids: tuple[str, ...] | None = None,
) -> tuple[
    ClinicalProcessSourceManifestV1,
    TrialIdentityTableV1,
    ExternalCorpusCoverageReportV1,
    tuple[SourceEligibilityDecisionV1, ...],
    tuple[LocalProcessSourceBindingV1, ...],
]:
    """Inventory verified ImmPort archives without loading participant tables."""

    tabular_root = Path(vault_root) / "immport" / "tabular"
    archives = tuple(sorted(tabular_root.glob("SDY*/*.zip")))
    if not archives:
        raise FileNotFoundError(
            f"No ImmPort tabular archives found under {tabular_root}"
        )
    if source_object_ids is not None:
        if not source_object_ids:
            raise ValueError("source_object_ids cannot be empty when supplied")
        if len(source_object_ids) != len(set(source_object_ids)):
            raise ValueError("source_object_ids must be unique")
        requested = set(source_object_ids)
        by_id = {_immport_source_object_id(path): path for path in archives}
        missing = sorted(requested - set(by_id))
        if missing:
            raise FileNotFoundError(
                f"Requested ImmPort source objects are unavailable: {missing}"
            )
        archives = tuple(
            by_id[source_object_id] for source_object_id in sorted(requested)
        )
    receipts = []
    bindings = []
    eligibility = []
    coverage_by_occurrence: dict[str, dict[str, int]] = {}
    registry_by_occurrence: dict[str, tuple[str, ...]] = {}
    for path in archives:
        source_object_id = _immport_source_object_id(path)
        _, accession, version = source_object_id.split(":")
        _verify_immport_archive(path, accession=accession, version=version)
        receipt = ClinicalProcessSourceReceiptV1(
            source_object_id=source_object_id,
            source_type=ProcessSourceType.IMMPORT,
            canonical_accession=accession,
            immutable_version=version,
            object_name=path.name,
            media_type="application/zip",
            byte_size=path.stat().st_size,
            sha256=sha256_file(path),
            observed_at_utc=observed_at_utc,
            acquisition_locator=f"immport:{accession}/{version}",
            retrieval_method="verified existing ImmPort tabular package",
            role=ProcessSourceRole.STRUCTURAL_SCREENING_ONLY,
            visibility=EvidenceVisibility.NOT_APPLICABLE,
            eligibility_status=EligibilityStatus.PENDING_HUMAN_REVIEW,
            license_evidence="ImmPort access terms require explicit local-use and output review.",
            credential_class="credentialed ImmPort shared data",
            raw_redistribution_permitted=False,
            permitted_derived_output=PermittedDerivedOutput.STRUCTURAL_COUNTS,
            human_review_required=True,
        )
        receipts.append(receipt)
        bindings.append(
            LocalProcessSourceBindingV1(
                source_object_id=source_object_id,
                local_path=path,
            )
        )
        eligibility.append(
            SourceEligibilityDecisionV1(
                source_object_id=source_object_id,
                status=EligibilityStatus.PENDING_HUMAN_REVIEW,
                decision_basis=(
                    "Archive bytes and structure verify; scientific use and derived-output "
                    "interpretation await human license review."
                ),
                approved_by_human=False,
                usable_process_families=(
                    "visits",
                    "assessments",
                    "laboratories",
                    "adverse_events",
                    "interventions",
                ),
                limitations=(
                    "Structural screening does not establish construct compatibility.",
                    "Subject records are not deduplicated unique participants.",
                ),
            )
        )
        registry_by_occurrence[source_object_id] = _registry_identifiers(path)
        coverage_by_occurrence[source_object_id] = _structural_counts(path)
    manifest = ClinicalProcessSourceManifestV1(
        imported_source_manifest_relative_path=imported_source_manifest_relative_path,
        imported_source_manifest_sha256=imported_source_manifest_sha256,
        receipts=tuple(receipts),
    )
    verify_process_source_manifest(manifest, bindings=tuple(bindings))
    identities = _identity_table(tuple(receipts), registry_by_occurrence)
    canonical_by_occurrence = {
        occurrence.source_object_id: trial.canonical_trial_id
        for trial in identities.trials
        for occurrence in trial.occurrences
    }
    coverage_rows = tuple(
        SourceCoverageRowV1(
            source_object_id=receipt.source_object_id,
            canonical_trial_id=canonical_by_occurrence[receipt.source_object_id],
            source_role=receipt.role,
            eligibility_status=receipt.eligibility_status,
            **coverage_by_occurrence[receipt.source_object_id],
        )
        for receipt in receipts
    )
    coverage = ExternalCorpusCoverageReportV1(
        source_object_count=len(coverage_rows),
        canonical_trial_count=len(identities.trials),
        real_distribution_eligible_trial_count=0,
        format_control_object_count=0,
        rows=coverage_rows,
    )
    return manifest, identities, coverage, tuple(eligibility), tuple(bindings)


def _immport_source_object_id(path: Path) -> str:
    match = _IMM_PORT_OBJECT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected ImmPort object name: {path.name}")
    accession, version = match.groups()
    if path.parent.name != accession:
        raise ValueError(f"ImmPort accession directory mismatch: {path}")
    return f"immport:{accession}:{version}"


def _registry_identifiers(path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path) as archive:
        member = next(
            info
            for info in archive.infolist()
            if info.filename.endswith("/study_link.txt")
        )
        identifiers: set[str] = set()
        with archive.open(member) as handle:
            for line in handle:
                identifiers.update(
                    match.decode("ascii")
                    for match in _REGISTRY_ID_PATTERN.findall(line)
                )
        values = tuple(sorted(identifiers))
    if not values:
        raise ValueError(f"No exact NCT identifier found in {path.name}")
    return values


def _structural_counts(path: Path) -> dict[str, int]:
    counts = {}
    with zipfile.ZipFile(path) as archive:
        by_name = {
            PurePosixPath(info.filename).name: info for info in archive.infolist()
        }
        for field_name, table_name in _STRUCTURAL_TABLES.items():
            info = by_name.get(table_name)
            if info is None:
                counts[field_name] = 0
                continue
            with archive.open(info) as handle:
                line_count = sum(1 for _ in handle)
            counts[field_name] = max(0, line_count - 1)
    return counts


def _identity_table(
    receipts: tuple[ClinicalProcessSourceReceiptV1, ...],
    registry_by_occurrence: dict[str, tuple[str, ...]],
) -> TrialIdentityTableV1:
    grouped: dict[str, list[ClinicalProcessSourceReceiptV1]] = defaultdict(list)
    for receipt in receipts:
        registry_ids = registry_by_occurrence[receipt.source_object_id]
        if len(registry_ids) != 1:
            raise ValueError(
                f"ImmPort occurrence {receipt.source_object_id} has ambiguous registry IDs"
            )
        grouped[registry_ids[0]].append(receipt)
    trials = []
    for registry_id, group in sorted(grouped.items()):
        occurrences = []
        for index, receipt in enumerate(
            sorted(group, key=lambda item: item.canonical_accession)
        ):
            occurrences.append(
                TrialOccurrenceV1(
                    occurrence_id=receipt.source_object_id,
                    source_object_id=receipt.source_object_id,
                    canonical_accession=receipt.canonical_accession,
                    registry_identifiers=(registry_id,),
                    relationship="same_trial" if index == 0 else "companion",
                    resolution_basis="exact_registry_id",
                    role=receipt.role,
                    visibility=receipt.visibility,
                )
            )
        trials.append(
            CanonicalTrialIdentityV1(
                canonical_trial_id=f"clinicaltrials.gov:{registry_id}",
                occurrences=tuple(occurrences),
            )
        )
    return TrialIdentityTableV1(trials=tuple(trials))


def write_safe_evidence_bundle(
    *,
    output_root: Path,
    manifest: ClinicalProcessSourceManifestV1,
    identities: TrialIdentityTableV1,
    coverage: ExternalCorpusCoverageReportV1,
    eligibility: tuple[SourceEligibilityDecisionV1, ...],
) -> None:
    """Write the safe receipt, identity, eligibility, and coverage artifacts."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_model(output_root / "source_manifest.json", manifest)
    _write_jsonl(output_root / "source_receipts.jsonl", manifest.receipts)
    _write_license_csv(
        output_root / "license_and_redistribution.csv", manifest.receipts
    )
    write_model(output_root / "trial_identity_table.json", identities)
    write_model(
        output_root / "identity_collision_report.json",
        AcquisitionVerificationReportV1(
            status="pass",
            verified_object_count=len(manifest.receipts),
            canonical_trial_count=len(identities.trials),
        ),
    )
    _write_jsonl(output_root / "source_eligibility.jsonl", eligibility)
    _write_coverage_csv(output_root / "source_coverage.csv", coverage.rows)
    _write_jsonl(
        output_root / "excluded_sources.jsonl",
        (
            SourceEligibilityDecisionV1(
                source_object_id="pro_act",
                status=EligibilityStatus.EXCLUDED,
                decision_basis=(
                    "PRO-ACT prohibits the consumer-LLM interaction required by this workflow."
                ),
                approved_by_human=False,
                usable_process_families=(),
                limitations=("The source is excluded from this workflow.",),
            ),
        ),
    )
    write_model(
        output_root / "acquisition_verification_report.json",
        AcquisitionVerificationReportV1(
            status="pass",
            verified_object_count=len(manifest.receipts),
            canonical_trial_count=len(identities.trials),
            findings=(
                "All retained ImmPort ZIP bytes, CRCs, roots, required tables, and exact NCT identities verified.",
                "All objects remain structural-screening-only pending human license and derived-output review.",
            ),
        ),
    )
    _assert_safe_outputs(output_root)
    artifact_paths = tuple(
        sorted(
            path
            for path in output_root.iterdir()
            if path.is_file() and path.name != "artifact_manifest.json"
        )
    )
    artifact_manifest = ArtifactManifestV1(
        artifacts=tuple(
            ArtifactRecordV1(
                relative_path=path.name,
                sha256=sha256_file(path),
                byte_size=path.stat().st_size,
            )
            for path in artifact_paths
        )
    )
    write_model(output_root / "artifact_manifest.json", artifact_manifest)
    _assert_safe_outputs(output_root)


def _write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _write_license_csv(
    path: Path, receipts: tuple[ClinicalProcessSourceReceiptV1, ...]
) -> None:
    fields = (
        "source_object_id",
        "source_type",
        "eligibility_status",
        "license_evidence",
        "credential_class",
        "raw_redistribution_permitted",
        "permitted_derived_output",
        "human_review_required",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for receipt in receipts:
            payload = receipt.model_dump(mode="json")
            writer.writerow({field: payload[field] for field in fields})


def _write_coverage_csv(path: Path, rows: tuple[SourceCoverageRowV1, ...]) -> None:
    fields = tuple(SourceCoverageRowV1.model_fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _assert_safe_outputs(output_root: Path) -> None:
    for path in output_root.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").lower()
        matches = [token for token in _SENSITIVE_OUTPUT_TOKENS if token in text]
        if matches:
            raise ValueError(f"Sensitive content detected in {path.name}: {matches}")


def _bindings_from_manifest(
    manifest: ClinicalProcessSourceManifestV1, vault_root: Path
) -> tuple[LocalProcessSourceBindingV1, ...]:
    bindings = []
    for receipt in manifest.receipts:
        if receipt.source_type != ProcessSourceType.IMMPORT:
            raise ValueError(
                "automatic local binding currently supports retained ImmPort archives only"
            )
        path = (
            Path(vault_root)
            / "immport"
            / "tabular"
            / receipt.canonical_accession
            / receipt.object_name
        )
        bindings.append(
            LocalProcessSourceBindingV1(
                source_object_id=receipt.source_object_id,
                local_path=path,
            )
        )
    return tuple(bindings)


def _parse_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return timestamp


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("credential-preflight")
    preflight.add_argument("--output", type=Path)
    inventory = subparsers.add_parser("inventory-immport")
    inventory.add_argument("--vault-root", type=Path, required=True)
    inventory.add_argument("--observed-at-utc", type=_parse_timestamp, required=True)
    inventory.add_argument("--imported-source-manifest-relative-path", required=True)
    inventory.add_argument("--imported-source-manifest-sha256", required=True)
    inventory.add_argument(
        "--source-object-id",
        action="append",
        dest="source_object_ids",
    )
    inventory.add_argument("--output-root", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--vault-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run credential preflight, inventory, or offline verification."""

    args = _parser().parse_args(argv)
    if args.command == "credential-preflight":
        report = credential_preflight()
        if args.output is not None:
            write_model(args.output, report)
        return 0
    if args.command == "inventory-immport":
        manifest, identities, coverage, eligibility, _ = inventory_immport(
            vault_root=args.vault_root,
            observed_at_utc=args.observed_at_utc,
            imported_source_manifest_relative_path=(
                args.imported_source_manifest_relative_path
            ),
            imported_source_manifest_sha256=args.imported_source_manifest_sha256,
            source_object_ids=(
                tuple(args.source_object_ids)
                if args.source_object_ids is not None
                else None
            ),
        )
        write_safe_evidence_bundle(
            output_root=args.output_root,
            manifest=manifest,
            identities=identities,
            coverage=coverage,
            eligibility=eligibility,
        )
        return 0
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = ClinicalProcessSourceManifestV1.model_validate(payload)
    verify_process_source_manifest(
        manifest,
        bindings=_bindings_from_manifest(manifest, args.vault_root),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AcquisitionVerificationReportV1",
    "CanonicalTrialIdentityV1",
    "ClinicalProcessSourceManifestV1",
    "ClinicalProcessSourceReceiptV1",
    "CredentialPreflightV1",
    "EligibilityStatus",
    "EvidenceVisibility",
    "ExternalCorpusCoverageReportV1",
    "LocalProcessSourceBindingV1",
    "PermittedDerivedOutput",
    "ProcessSourceRole",
    "ProcessSourceType",
    "SourceCoverageRowV1",
    "SourceEligibilityDecisionV1",
    "TrialIdentityTableV1",
    "TrialOccurrenceV1",
    "credential_preflight",
    "inventory_immport",
    "verify_process_source_manifest",
    "write_safe_evidence_bundle",
]
