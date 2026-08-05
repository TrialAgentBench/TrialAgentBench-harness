"""Replay TrialEval scoreable reference from public release evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal, TypeVar
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_validation.contracts.scoring.evaluation_target_register import (
    EvaluationTargetRegisterEntryV1,
)
from trialagentbench_validation.contracts.scoring.public_estimand import (
    PublicEstimandContractV1,
)
from trialagentbench_validation.contracts.scoring.public_reference_sources import (
    PublicReferenceSourceManifestV1,
    PublicReferenceSourceRecordV1,
)
from trialagentbench_validation.contracts.scoring.route_reference_inputs import (
    RouteReferenceInputManifestV1,
    RouteReferenceInputRecordV1,
)
from trialagentbench_validation.contracts.scoring.route_references import (
    RouteReferenceRecordV1,
)
from trialagentbench_validation.io.json import write_json_model
from trialagentbench_validation.trialeval.public_archive import (
    participant_semantic_member_names_v1,
)
from trialagentbench_validation.trialeval.references.io import (
    resolve_public_member_v1,
)

PublicEvidenceReplayStatusV1 = Literal["pass", "fail"]
PublicEvidenceReferenceClassV1 = Literal[
    "public_surface_mirror",
    "canonical_analysis_public_evidence",
    "reconstruction_public_evidence",
    "public_surface_gap",
]
PublicEvidenceDriftStatusV1 = Literal[
    "identical",
    "public_surface_gap",
    "contract_defect",
]
PublicEvidenceRouteReferenceExposureV1 = Literal[
    "public_contract_and_scoreable_input",
    "scoreable_input_only",
    "public_surface_gap",
    "contract_defect",
]
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class PublicEvidenceReferenceReplayRecordV1(BaseModel):
    """One public-evidence replay record for a scoreable reference-input bundle."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.public_evidence_reference_replay_record/v1"] = (
        "trialagentbench.public_evidence_reference_replay_record/v1"
    )
    task_id: str = Field(..., min_length=1)
    input_bundle_id: str = Field(..., min_length=1)
    estimator_method_id: str = Field(..., min_length=1)
    effect_scale: str = Field(..., min_length=1)
    lane_ids: tuple[str, ...] = Field(..., min_length=1)
    route_reference_ids: tuple[str, ...] = Field(..., min_length=1)
    source_role: str = Field(..., min_length=1)
    evaluation_class: PublicEvidenceReferenceClassV1
    drift_status: PublicEvidenceDriftStatusV1
    public_input_hashes: dict[str, str] = Field(default_factory=dict)
    missing_public_inputs: tuple[str, ...] = Field(default_factory=tuple)
    checksum_mismatched_inputs: tuple[str, ...] = Field(default_factory=tuple)


class PublicEvidenceRouteReferenceReplayRecordV1(BaseModel):
    """One official route reference classified against public replay evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal[
        "trialagentbench.public_evidence_route_reference_replay_record/v1"
    ] = "trialagentbench.public_evidence_route_reference_replay_record/v1"
    task_id: str = Field(..., min_length=1)
    route_reference_id: str = Field(..., min_length=1)
    variant_role: str = Field(..., min_length=1)
    route_family: str = Field(..., min_length=1)
    effect_scale: str = Field(..., min_length=1)
    answer_shape: str = Field(..., min_length=1)
    identification_class: Literal["point_identified", "partially_identified"]
    exposure_class: PublicEvidenceRouteReferenceExposureV1
    drift_status: PublicEvidenceDriftStatusV1
    scoreable_input_bundle_ids: tuple[str, ...] = Field(default_factory=tuple)
    public_contract_bound: bool
    public_evidence_basis_missing: tuple[str, ...] = Field(default_factory=tuple)


class PublicEvidenceReferenceReplayReportV1(BaseModel):
    """Public-evidence reference replay report for TrialEvalBench."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.public_evidence_reference_replay_report/v1"] = (
        "trialagentbench.public_evidence_reference_replay_report/v1"
    )
    evaluator_zip: str
    public_zip: str
    status: PublicEvidenceReplayStatusV1
    task_count: int
    replay_record_count: int
    route_reference_input_rows: int
    route_reference_rows: int
    public_reference_source_rows: int
    missing_public_reference_source_count: int
    orphan_public_reference_source_count: int
    official_route_reference_count: int
    evaluation_target_register_rows: int
    evaluator_public_estimand_contract_count: int
    public_estimand_contract_count: int
    public_contract_route_reference_count: int
    public_contract_bound_route_reference_count: int
    scoreable_input_only_variant_count: int
    public_surface_gap_variant_count: int
    contract_defect_variant_count: int
    public_input_count: int
    missing_public_input_count: int
    checksum_mismatch_count: int
    public_contract_missing_count: int
    public_contract_mismatch_count: int
    route_reference_contract_mismatch_count: int
    public_evidence_basis_missing_count: int
    route_reference_without_replay_count: int
    task_without_replay_count: int
    drift_status_counts: dict[str, int]
    evaluation_class_counts: dict[str, int]
    route_reference_exposure_counts: dict[str, int]
    findings: tuple[str, ...]
    records: tuple[PublicEvidenceReferenceReplayRecordV1, ...]
    route_reference_records: tuple[PublicEvidenceRouteReferenceReplayRecordV1, ...]


def replay_trialeval_public_evidence_reference_v1(
    *,
    evaluator_zip: Path,
    public_zip: Path,
) -> PublicEvidenceReferenceReplayReportV1:
    """Replay scoreable reference-input provenance from public release evidence.

    This replay proves that the tables declared as scoreable-reference inputs are
    present on the participant-facing public surface with matching bytes. It
    deliberately does not use non-public construction inputs or hidden DGP
    metadata.
    """

    findings: list[str] = []
    with ZipFile(evaluator_zip) as evaluator, ZipFile(public_zip) as public:
        public_names = set(participant_semantic_member_names_v1(public))
        item_index = json.loads(_member_bytes(evaluator, "grader/item_index.json"))
        task_ids = {
            str(entry["task_id"])
            for entry in item_index.get("entries", [])
            if isinstance(entry, dict) and str(entry.get("task_id") or "")
        }
        evaluation_target_rows = _read_jsonl_models_from_zip(
            evaluator,
            EvaluationTargetRegisterEntryV1,
            "grader/domains/evaluation_target_register.jsonl",
        )
        route_references = _read_jsonl_models_from_zip(
            evaluator,
            RouteReferenceRecordV1,
            "grader/domains/route_references.jsonl",
        )
        public_reference_sources = _read_jsonl_models_from_zip(
            evaluator,
            PublicReferenceSourceRecordV1,
            "grader/domains/public_reference_sources.jsonl",
        )
        public_reference_source_manifest = _read_json_model_from_zip(
            evaluator,
            PublicReferenceSourceManifestV1,
            "grader/domains/public_reference_sources_manifest.json",
        )
        route_reference_ids = {row.route_reference_id for row in route_references}
        public_reference_source_ids = {
            row.route_reference_id for row in public_reference_sources
        }
        if len(route_reference_ids) != len(route_references):
            findings.append("route_references_duplicate_id")
        if len(public_reference_source_ids) != len(public_reference_sources):
            findings.append("public_reference_sources_duplicate_id")
        missing_public_reference_source_ids = (
            route_reference_ids - public_reference_source_ids
        )
        orphan_public_reference_source_ids = (
            public_reference_source_ids - route_reference_ids
        )
        if missing_public_reference_source_ids:
            findings.append("route_references_missing_public_reference_source")
        if orphan_public_reference_source_ids:
            findings.append("public_reference_sources_without_route_reference")
        if public_reference_source_manifest.row_count != len(public_reference_sources):
            findings.append("public_reference_sources_manifest_row_count_mismatch")
        if (
            public_reference_source_manifest.route_references_sha256
            != _zip_member_sha256(evaluator, "grader/domains/route_references.jsonl")
        ):
            findings.append(
                "public_reference_sources_manifest_route_references_checksum_mismatch"
            )
        if (
            public_reference_source_manifest.public_reference_sources_jsonl_sha256
            != _zip_member_sha256(
                evaluator, "grader/domains/public_reference_sources.jsonl"
            )
        ):
            findings.append("public_reference_sources_manifest_checksum_mismatch")
        evaluator_contracts = _read_public_estimand_contracts_from_evaluator(evaluator)
        participant_contracts = _read_public_estimand_contracts_from_public(public)
        missing_evaluator_contracts = task_ids - set(evaluator_contracts)
        if missing_evaluator_contracts:
            findings.append("evaluator_public_estimand_contract_missing")
        if participant_contracts:
            findings.append("participant_public_estimand_contract_leakage")
        reference_inputs = _read_jsonl_models_from_zip(
            evaluator,
            RouteReferenceInputRecordV1,
            "grader/domains/route_reference_inputs.jsonl",
        )
        reference_input_manifest = _read_json_model_from_zip(
            evaluator,
            RouteReferenceInputManifestV1,
            "grader/domains/route_reference_inputs_manifest.json",
        )

        if reference_input_manifest.row_count != len(reference_inputs):
            findings.append("route_reference_inputs_manifest_row_count_mismatch")
        if not reference_inputs:
            findings.append("route_reference_inputs_empty")

        official_route_reference_ids = {
            row.route_reference_id
            for row in route_references
            if row.support_status == "official_supported"
        }
        public_contract_required_route_reference_ids = {
            row.route_reference_id
            for row in route_references
            if row.support_status == "official_supported"
            and row.variant_role
            in {"required_primary", "credit_eligible_primary_alternative"}
        }
        public_contract_route_references = {
            str(variant.route_reference_id)
            for contract in evaluator_contracts.values()
            for variant in contract.variants
            if variant.route_reference_id
        }
        missing_public_contract_route_references = (
            public_contract_required_route_reference_ids
            - public_contract_route_references
        )
        if missing_public_contract_route_references:
            findings.append(
                "official_route_references_missing_evaluator_public_estimand_contract"
            )
        route_reference_contract_mismatches = (
            _route_reference_public_contract_mismatches(
                route_references=route_references,
                public_contracts=evaluator_contracts,
            )
        )
        if route_reference_contract_mismatches:
            findings.append("route_reference_public_contract_field_mismatch")
        missing_public_evidence_basis = _missing_public_evidence_basis(
            route_references=route_references,
            public_contracts=evaluator_contracts,
            public_names=public_names,
        )
        if missing_public_evidence_basis:
            findings.append("public_evidence_basis_missing_from_public_zip")

        replayed_route_reference_ids: set[str] = set()
        records: list[PublicEvidenceReferenceReplayRecordV1] = []
        for reference_input in reference_inputs:
            public_input_hashes: dict[str, str] = {}
            missing_public_inputs: list[str] = []
            checksum_mismatched_inputs: list[str] = []
            for table_ref in reference_input.required_table_refs:
                public_rel_path = _public_rel_path_for_scoreable_ref(table_ref.rel_path)
                if public_rel_path not in public_names:
                    missing_public_inputs.append(public_rel_path)
                    continue
                public_sha256 = _zip_member_sha256(
                    public, resolve_public_member_v1(public, public_rel_path)
                )
                public_input_hashes[public_rel_path] = public_sha256
                if public_sha256 != table_ref.sha256:
                    checksum_mismatched_inputs.append(public_rel_path)

            replayed_route_reference_ids.update(reference_input.route_reference_ids)
            evaluation_class = _evaluation_class_for_input(
                source_role=reference_input.source_role,
                missing_count=len(missing_public_inputs),
                mismatch_count=len(checksum_mismatched_inputs),
            )
            drift_status: PublicEvidenceDriftStatusV1 = (
                "identical"
                if not missing_public_inputs and not checksum_mismatched_inputs
                else "public_surface_gap"
            )
            records.append(
                PublicEvidenceReferenceReplayRecordV1(
                    task_id=reference_input.task_id,
                    input_bundle_id=reference_input.input_bundle_id,
                    estimator_method_id=reference_input.estimator_method_id,
                    effect_scale=reference_input.effect_scale,
                    lane_ids=reference_input.lane_ids,
                    route_reference_ids=reference_input.route_reference_ids,
                    source_role=reference_input.source_role,
                    evaluation_class=evaluation_class,
                    drift_status=drift_status,
                    public_input_hashes=public_input_hashes,
                    missing_public_inputs=tuple(sorted(missing_public_inputs)),
                    checksum_mismatched_inputs=tuple(
                        sorted(checksum_mismatched_inputs)
                    ),
                )
            )

        missing_route_references = (
            official_route_reference_ids - replayed_route_reference_ids
        )
        if missing_route_references:
            findings.append("official_route_references_without_public_evidence_replay")
        replay_task_ids = {record.task_id for record in records}
        task_without_replay = task_ids - replay_task_ids
        if task_without_replay:
            findings.append("tasks_without_public_evidence_replay")
        if any(record.missing_public_inputs for record in records):
            findings.append("route_reference_inputs_missing_public_sources")
        if any(record.checksum_mismatched_inputs for record in records):
            findings.append("route_reference_inputs_public_checksum_mismatch")
        route_reference_records = _route_reference_replay_records(
            route_references=route_references,
            replay_records=tuple(records),
            public_contract_required_route_reference_ids=public_contract_required_route_reference_ids,
            public_contract_route_references=public_contract_route_references,
            route_reference_contract_mismatches=route_reference_contract_mismatches,
            missing_public_evidence_basis=missing_public_evidence_basis,
        )

    drift_status_counts = Counter(record.drift_status for record in records)
    evaluation_class_counts = Counter(record.evaluation_class for record in records)
    route_reference_exposure_counts = Counter(
        record.exposure_class for record in route_reference_records
    )
    public_inputs = {path for record in records for path in record.public_input_hashes}
    missing_public_input_count = sum(
        len(record.missing_public_inputs) for record in records
    )
    checksum_mismatch_count = sum(
        len(record.checksum_mismatched_inputs) for record in records
    )
    return PublicEvidenceReferenceReplayReportV1(
        evaluator_zip=evaluator_zip.as_posix(),
        public_zip=public_zip.as_posix(),
        status="fail" if findings else "pass",
        task_count=len(task_ids),
        replay_record_count=len(records),
        route_reference_input_rows=len(reference_inputs),
        route_reference_rows=len(route_references),
        public_reference_source_rows=len(public_reference_sources),
        missing_public_reference_source_count=len(missing_public_reference_source_ids),
        orphan_public_reference_source_count=len(orphan_public_reference_source_ids),
        official_route_reference_count=len(official_route_reference_ids),
        evaluation_target_register_rows=len(evaluation_target_rows),
        public_input_count=len(public_inputs),
        missing_public_input_count=missing_public_input_count,
        checksum_mismatch_count=checksum_mismatch_count,
        evaluator_public_estimand_contract_count=len(evaluator_contracts),
        public_estimand_contract_count=len(participant_contracts),
        public_contract_route_reference_count=len(public_contract_route_references),
        public_contract_bound_route_reference_count=route_reference_exposure_counts.get(
            "public_contract_and_scoreable_input", 0
        ),
        scoreable_input_only_variant_count=route_reference_exposure_counts.get(
            "scoreable_input_only", 0
        ),
        public_surface_gap_variant_count=route_reference_exposure_counts.get(
            "public_surface_gap", 0
        ),
        contract_defect_variant_count=route_reference_exposure_counts.get(
            "contract_defect", 0
        ),
        public_contract_missing_count=len(missing_evaluator_contracts),
        public_contract_mismatch_count=len(participant_contracts),
        route_reference_contract_mismatch_count=len(
            route_reference_contract_mismatches
        ),
        public_evidence_basis_missing_count=len(missing_public_evidence_basis),
        route_reference_without_replay_count=len(missing_route_references),
        task_without_replay_count=len(task_without_replay),
        drift_status_counts=dict(sorted(drift_status_counts.items())),
        evaluation_class_counts=dict(sorted(evaluation_class_counts.items())),
        route_reference_exposure_counts=dict(
            sorted(route_reference_exposure_counts.items())
        ),
        findings=tuple(sorted(set(findings))),
        records=tuple(records),
        route_reference_records=tuple(route_reference_records),
    )


def render_public_evidence_reference_replay_report_v1(
    report: PublicEvidenceReferenceReplayReportV1,
) -> str:
    """Render a public-evidence reference replay report as Markdown."""

    lines = [
        "# Public-Evidence Reference Replay Report",
        "",
        f"- Evaluator zip: `{report.evaluator_zip}`",
        f"- Public zip: `{report.public_zip}`",
        f"- Status: `{report.status}`",
        f"- Tasks: `{report.task_count}`",
        f"- Replay records: `{report.replay_record_count}`",
        f"- Scoreable reference-input rows: `{report.route_reference_input_rows}`",
        f"- Route references: `{report.route_reference_rows}`",
        f"- Public reference sources: `{report.public_reference_source_rows}`",
        f"- Route references missing public sources: `{report.missing_public_reference_source_count}`",
        f"- Orphan public reference sources: `{report.orphan_public_reference_source_count}`",
        f"- Official route references: `{report.official_route_reference_count}`",
        f"- Evaluation-target register rows: `{report.evaluation_target_register_rows}`",
        f"- Evaluator public estimand contracts: `{report.evaluator_public_estimand_contract_count}`",
        f"- Public estimand contracts: `{report.public_estimand_contract_count}`",
        f"- Public-contract route references: `{report.public_contract_route_reference_count}`",
        f"- Public-contract-bound variants: `{report.public_contract_bound_route_reference_count}`",
        f"- Scoreable-input-only variants: `{report.scoreable_input_only_variant_count}`",
        f"- Public-surface-gap variants: `{report.public_surface_gap_variant_count}`",
        f"- Contract-defect variants: `{report.contract_defect_variant_count}`",
        f"- Public input tables: `{report.public_input_count}`",
        f"- Missing public inputs: `{report.missing_public_input_count}`",
        f"- Checksum mismatches: `{report.checksum_mismatch_count}`",
        f"- Missing public contracts: `{report.public_contract_missing_count}`",
        f"- Public contract mismatches: `{report.public_contract_mismatch_count}`",
        f"- Route-reference contract mismatches: `{report.route_reference_contract_mismatch_count}`",
        f"- Missing public evidence-basis files: `{report.public_evidence_basis_missing_count}`",
        f"- Official route references without replay: `{report.route_reference_without_replay_count}`",
        f"- Tasks without replay: `{report.task_without_replay_count}`",
        "",
        "## Drift Status Counts",
        "",
    ]
    lines.extend(
        f"- `{key}`: `{value}`"
        for key, value in sorted(report.drift_status_counts.items())
    )
    lines.extend(["", "## Reference Class Counts", ""])
    lines.extend(
        f"- `{key}`: `{value}`"
        for key, value in sorted(report.evaluation_class_counts.items())
    )
    lines.extend(["", "## Variant Exposure Counts", ""])
    lines.extend(
        f"- `{key}`: `{value}`"
        for key, value in sorted(report.route_reference_exposure_counts.items())
    )
    if report.findings:
        lines.extend(["", "## Findings", ""])
        lines.extend(f"- `{finding}`" for finding in report.findings)
    return "\n".join(lines) + "\n"


def write_public_evidence_reference_replay_artifacts_v1(
    *,
    evaluator_zip: Path,
    public_zip: Path,
    out_dir: Path,
) -> PublicEvidenceReferenceReplayReportV1:
    """Replay public-evidence reference and write JSON/JSONL/Markdown artifacts."""

    report = replay_trialeval_public_evidence_reference_v1(
        evaluator_zip=evaluator_zip, public_zip=public_zip
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_model(out_dir / "public_evidence_reference_replay_report.json", report)
    records_path = out_dir / "public_evidence_reference_replay_records.jsonl"
    records_path.write_text(
        "".join(
            record.model_dump_json(by_alias=False, exclude_none=True) + "\n"
            for record in report.records
        ),
        encoding="utf-8",
    )
    route_reference_records_path = (
        out_dir / "public_evidence_route_reference_replay_records.jsonl"
    )
    route_reference_records_path.write_text(
        "".join(
            record.model_dump_json(by_alias=False, exclude_none=True) + "\n"
            for record in report.route_reference_records
        ),
        encoding="utf-8",
    )
    (out_dir / "public_evidence_reference_replay_report.md").write_text(
        render_public_evidence_reference_replay_report_v1(report),
        encoding="utf-8",
    )
    return report


def _public_rel_path_for_scoreable_ref(scoreable_rel_path: str) -> str:
    if (
        not scoreable_rel_path.startswith("items/")
        or "/data/" not in scoreable_rel_path
    ):
        raise ValueError(
            f"Scoreable reference ref is not a participant item data path: {scoreable_rel_path}"
        )
    return scoreable_rel_path


def _evaluation_class_for_input(
    *, source_role: str, missing_count: int, mismatch_count: int
) -> PublicEvidenceReferenceClassV1:
    if missing_count or mismatch_count:
        return "public_surface_gap"
    if source_role == "public_surface_mirror":
        return "public_surface_mirror"
    if source_role == "canonical_analysis":
        return "canonical_analysis_public_evidence"
    if source_role == "reconstruction_reference":
        return "reconstruction_public_evidence"
    raise ValueError(f"Unknown scoreable reference source_role: {source_role}")


def _read_public_estimand_contracts_from_evaluator(
    zf: ZipFile,
) -> dict[str, PublicEstimandContractV1]:
    contracts: dict[str, PublicEstimandContractV1] = {}
    member = "grader/domains/public_estimand_contract.jsonl"
    for line_number, line in enumerate(
        _member_bytes(zf, member).decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSONL row at {member}:{line_number}.")
        row_payload = payload.get("payload")
        if not isinstance(row_payload, dict):
            raise ValueError(
                f"Public estimand row missing payload at {member}:{line_number}."
            )
        contract_payload = row_payload.get("contract")
        if not isinstance(contract_payload, dict):
            raise ValueError(
                f"Public estimand row missing payload.contract at {member}:{line_number}."
            )
        contract = PublicEstimandContractV1.model_validate(contract_payload)
        if contract.task_id in contracts:
            raise ValueError(
                f"Duplicate evaluator public estimand contract for task_id={contract.task_id!r}."
            )
        contracts[contract.task_id] = contract
    return contracts


def _read_public_estimand_contracts_from_public(
    zf: ZipFile,
) -> dict[str, PublicEstimandContractV1]:
    contracts: dict[str, PublicEstimandContractV1] = {}
    members = sorted(
        name
        for name in zf.namelist()
        if name.removeprefix("public/").startswith("public_estimand_contracts/")
        and name.endswith(".json")
        and not name.endswith("/")
    )
    for member in members:
        contract = PublicEstimandContractV1.model_validate(
            json.loads(_member_bytes(zf, member))
        )
        if contract.task_id in contracts:
            raise ValueError(
                f"Duplicate participant public estimand contract for task_id={contract.task_id!r}."
            )
        contracts[contract.task_id] = contract
    return contracts


def _route_reference_public_contract_mismatches(
    *,
    route_references: tuple[RouteReferenceRecordV1, ...],
    public_contracts: dict[str, PublicEstimandContractV1],
) -> tuple[str, ...]:
    mismatches: list[str] = []
    for route_reference in route_references:
        if (
            route_reference.support_status != "official_supported"
            or route_reference.variant_role
            not in {
                "required_primary",
                "credit_eligible_primary_alternative",
            }
        ):
            continue
        contract = public_contracts.get(route_reference.task_id)
        if contract is None:
            continue
        matching_variants = tuple(
            variant
            for variant in contract.variants
            if variant.route_reference_id == route_reference.route_reference_id
        )
        if len(matching_variants) != 1:
            mismatches.append(route_reference.route_reference_id)
            continue
        public_variant = matching_variants[0]
        expected_shapes = _public_answer_shapes_for_route_reference(
            route_reference.answer_shape
        )
        if (
            public_variant.route_family != route_reference.route_family
            or public_variant.effect_scale != route_reference.effect_scale
            or not expected_shapes & set(public_variant.answer_shapes)
            or not set(route_reference.public_evidence_basis)
            <= set(public_variant.public_evidence_basis)
        ):
            mismatches.append(route_reference.route_reference_id)
    return tuple(sorted(mismatches))


def _public_answer_shapes_for_route_reference(answer_shape: str) -> set[str]:
    if answer_shape == "point":
        return {"numeric_point"}
    if answer_shape == "bound":
        return {"bounds_interval"}
    if answer_shape == "vector":
        return {"numeric_vector"}
    if answer_shape == "test":
        return {"statistical_test"}
    return {answer_shape}


def _missing_public_evidence_basis(
    *,
    route_references: tuple[RouteReferenceRecordV1, ...],
    public_contracts: dict[str, PublicEstimandContractV1],
    public_names: set[str],
) -> tuple[str, ...]:
    paths: set[str] = set()
    for route_reference in route_references:
        if route_reference.support_status == "official_supported":
            paths.update(str(path) for path in route_reference.public_evidence_basis)
    for contract in public_contracts.values():
        paths.update(str(path) for path in contract.public_evidence_basis)
        for variant in contract.variants:
            paths.update(str(path) for path in variant.public_evidence_basis)
            for modifier in variant.modifier_evidence_basis:
                paths.update(str(path) for path in modifier.public_rel_paths)
    return tuple(
        sorted(
            path
            for path in paths
            if not path.startswith("assumption_evidence:") and path not in public_names
        )
    )


def _route_reference_replay_records(
    *,
    route_references: tuple[RouteReferenceRecordV1, ...],
    replay_records: tuple[PublicEvidenceReferenceReplayRecordV1, ...],
    public_contract_required_route_reference_ids: set[str],
    public_contract_route_references: set[str],
    route_reference_contract_mismatches: tuple[str, ...],
    missing_public_evidence_basis: tuple[str, ...],
) -> tuple[PublicEvidenceRouteReferenceReplayRecordV1, ...]:
    replay_by_reference_id: dict[str, list[PublicEvidenceReferenceReplayRecordV1]] = {}
    for replay_record in replay_records:
        for route_reference_id in replay_record.route_reference_ids:
            replay_by_reference_id.setdefault(route_reference_id, []).append(
                replay_record
            )
    contract_mismatch_set = set(route_reference_contract_mismatches)
    missing_evidence = set(missing_public_evidence_basis)
    records: list[PublicEvidenceRouteReferenceReplayRecordV1] = []
    for route_reference in route_references:
        if route_reference.support_status != "official_supported":
            continue
        variant_replays = tuple(
            replay_by_reference_id.get(route_reference.route_reference_id, ())
        )
        input_gap = (
            not variant_replays
            or any(record.drift_status != "identical" for record in variant_replays)
            or bool(missing_evidence & set(route_reference.public_evidence_basis))
        )
        public_contract_required = (
            route_reference.route_reference_id
            in public_contract_required_route_reference_ids
        )
        public_contract_bound = (
            route_reference.route_reference_id in public_contract_route_references
        )
        contract_defect = public_contract_required and (
            not public_contract_bound
            or route_reference.route_reference_id in contract_mismatch_set
        )
        if input_gap:
            exposure_class: PublicEvidenceRouteReferenceExposureV1 = (
                "public_surface_gap"
            )
            drift_status: PublicEvidenceDriftStatusV1 = "public_surface_gap"
        elif contract_defect:
            exposure_class = "contract_defect"
            drift_status = "contract_defect"
        elif public_contract_bound:
            exposure_class = "public_contract_and_scoreable_input"
            drift_status = "identical"
        else:
            exposure_class = "scoreable_input_only"
            drift_status = "identical"
        records.append(
            PublicEvidenceRouteReferenceReplayRecordV1(
                task_id=route_reference.task_id,
                route_reference_id=route_reference.route_reference_id,
                variant_role=route_reference.variant_role,
                route_family=route_reference.route_family,
                effect_scale=route_reference.effect_scale,
                answer_shape=route_reference.answer_shape,
                identification_class=route_reference.identification_class,
                exposure_class=exposure_class,
                drift_status=drift_status,
                scoreable_input_bundle_ids=tuple(
                    sorted(record.input_bundle_id for record in variant_replays)
                ),
                public_contract_bound=public_contract_bound,
                public_evidence_basis_missing=tuple(
                    sorted(
                        missing_evidence & set(route_reference.public_evidence_basis)
                    )
                ),
            )
        )
    return tuple(records)


def _member_bytes(zf: ZipFile, member: str) -> bytes:
    try:
        return zf.read(member)
    except KeyError as exc:
        raise FileNotFoundError(f"Missing evaluator bundle member: {member}") from exc


def _zip_member_sha256(zf: ZipFile, member: str) -> str:
    return hashlib.sha256(_member_bytes(zf, member)).hexdigest()


def _read_json_model_from_zip(
    zf: ZipFile, model: type[_ModelT], member: str
) -> _ModelT:
    payload = json.loads(_member_bytes(zf, member))
    return model.model_validate(payload)


def _read_jsonl_models_from_zip(
    zf: ZipFile, model: type[_ModelT], member: str
) -> tuple[_ModelT, ...]:
    records: list[_ModelT] = []
    for line_number, line in enumerate(
        _member_bytes(zf, member).decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSONL row at {member}:{line_number}.")
        records.append(model.model_validate(payload))
    return tuple(records)


__all__ = [
    "PublicEvidenceReferenceReplayRecordV1",
    "PublicEvidenceReferenceReplayReportV1",
    "PublicEvidenceRouteReferenceReplayRecordV1",
    "replay_trialeval_public_evidence_reference_v1",
    "render_public_evidence_reference_replay_report_v1",
    "write_public_evidence_reference_replay_artifacts_v1",
]
