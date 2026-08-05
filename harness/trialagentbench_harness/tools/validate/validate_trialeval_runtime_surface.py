"""Validate TrialEvalBench model-facing runtime-surface parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from trialagentbench_harness.contracts.core.trialeval_factors import TrialEvalEvidenceFactorsV1
from trialagentbench_harness.contracts.release import (
    TrialAgentBenchCharterV1,
    TrialEvalParticipantManifestV1,
    TrialEvalPublicIntegrityPolicyV1,
    render_benchmark_map_markdown,
)
from trialagentbench_harness.contracts.release.artifacts import (
    TRIALEVAL_EVALUATOR_ARCHIVE,
    TRIALEVAL_PARTICIPANT_ARCHIVE,
    TRIALEVAL_PARTICIPANT_ROOT_MEMBERS,
)
from trialagentbench_harness.contracts.release.trialeval_runtime_surface import (
    TRIALEVAL_AGENT_FORBIDDEN_ZIP_FILENAMES,
    TRIALEVAL_AGENT_FORBIDDEN_ZIP_PARTS,
    TrialEvalParticipantTaskV1,
    TrialEvalSemanticSubmissionContractV1,
    classify_trialeval_item_member,
    required_trialeval_participant_members,
    trialeval_agent_excluded_files,
    trialeval_agent_forbidden_json_key_paths,
    trialeval_agent_semantic_leakage_paths,
)


@dataclass(frozen=True)
class TrialEvalRuntimeSurfaceFinding:
    """One model-facing runtime-surface parity finding."""

    path: str
    message: str


def _zip_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {name for name in archive.namelist() if not name.endswith("/")}


def _load_item_evidence_factors(evaluator_zip: Path) -> dict[str, TrialEvalEvidenceFactorsV1]:
    with zipfile.ZipFile(evaluator_zip) as archive:
        try:
            payload = json.loads(archive.read("grader/item_index.json").decode("utf-8"))
        except KeyError as exc:
            raise ValueError(f"TrialEval evaluator ZIP is missing grader/item_index.json: {evaluator_zip}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"TrialEval evaluator item index is invalid JSON: {evaluator_zip}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"TrialEval evaluator item index is not a JSON object: {evaluator_zip}")
    factors_by_task: dict[str, TrialEvalEvidenceFactorsV1] = {}
    for entry in payload.get("entries", []):
        if not isinstance(entry, dict):
            raise ValueError("TrialEval evaluator item index entries must be JSON objects.")
        task_id = entry.get("task_id")
        if isinstance(task_id, str) and task_id:
            factors = entry.get("factors")
            if not isinstance(factors, dict):
                raise ValueError(f"TrialEval evaluator task {task_id!r} lacks explicit factors.")
            factors_by_task[task_id] = TrialEvalEvidenceFactorsV1.model_validate(
                {
                    "context_configuration": factors.get("context_configuration"),
                    "data_preparation": factors.get("data_preparation"),
                    "analysis_specification": factors.get("analysis_specification"),
                }
            )
    return factors_by_task


def _task_id_from_public_member(name: str) -> str | None:
    parts = Path(name).parts
    if len(parts) >= 3 and parts[0] == "items":
        return parts[1]
    return None


def _read_public_json(archive: zipfile.ZipFile, name: str) -> object:
    return json.loads(archive.read(name).decode("utf-8"))


def _read_charter(archive: zipfile.ZipFile) -> TrialAgentBenchCharterV1:
    """Read the canonical charter from one TrialEval archive."""

    return TrialAgentBenchCharterV1.model_validate_json(archive.read("benchmark_charter.json").decode("utf-8"))


def _task_file_reference_findings(*, task_id: str, payload: object, names: set[str]) -> list[str]:
    if not isinstance(payload, dict):
        return []
    findings: list[str] = []
    for key, value in payload.items():
        if not key.endswith("_file") or not isinstance(value, str):
            continue
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
            findings.append(f"{key} contains an unsafe item-relative path: {value!r}")
            continue
        expected = f"items/{task_id}/{value}"
        if expected not in names:
            findings.append(f"{key} references missing participant member: {value!r}")
    return findings


def validate_trialeval_runtime_surface(
    *,
    public_zip: Path,
    evaluator_zip: Path,
) -> list[TrialEvalRuntimeSurfaceFinding]:
    """Validate prompt/filesystem parity for TrialEvalBench participant artifacts."""

    public_zip = Path(public_zip)
    evaluator_zip = Path(evaluator_zip)
    findings: list[TrialEvalRuntimeSurfaceFinding] = []
    if not public_zip.is_file():
        return [TrialEvalRuntimeSurfaceFinding(public_zip.as_posix(), "TrialEval participant ZIP is missing")]
    if not evaluator_zip.is_file():
        return [TrialEvalRuntimeSurfaceFinding(evaluator_zip.as_posix(), "TrialEval evaluator ZIP is missing")]

    try:
        factors_by_task = _load_item_evidence_factors(evaluator_zip)
    except ValueError as exc:
        return [TrialEvalRuntimeSurfaceFinding(evaluator_zip.as_posix(), str(exc))]
    names = _zip_names(public_zip)
    with zipfile.ZipFile(public_zip) as archive, zipfile.ZipFile(evaluator_zip) as evaluator_archive:
        try:
            public_charter = _read_charter(archive)
            evaluator_charter = _read_charter(evaluator_archive)
            if public_charter != evaluator_charter:
                findings.append(
                    TrialEvalRuntimeSurfaceFinding(
                        "benchmark_charter.json",
                        "participant and evaluator benchmark charters disagree",
                    )
                )
            expected_map = render_benchmark_map_markdown(public_charter)
            if (
                archive.read("benchmark_map.md").decode("utf-8") != expected_map
                or evaluator_archive.read("benchmark_map.md").decode("utf-8") != expected_map
            ):
                findings.append(
                    TrialEvalRuntimeSurfaceFinding(
                        "benchmark_map.md",
                        "participant or evaluator benchmark map disagrees with the canonical charter",
                    )
                )
        except KeyError as exc:
            missing_member = str(exc.args[0])
            findings.append(
                TrialEvalRuntimeSurfaceFinding(
                    missing_member,
                    f"participant or evaluator {missing_member} is missing",
                )
            )
        except (UnicodeDecodeError, ValueError) as exc:
            findings.append(
                TrialEvalRuntimeSurfaceFinding(
                    "benchmark_charter.json",
                    f"participant or evaluator benchmark charter is invalid: {exc}",
                )
            )
        try:
            participant_manifest = TrialEvalParticipantManifestV1.model_validate_json(
                archive.read("manifest.json").decode("utf-8")
            )
        except KeyError:
            findings.append(TrialEvalRuntimeSurfaceFinding("manifest.json", "participant manifest is missing"))
            participant_manifest = None
        except (UnicodeDecodeError, ValueError) as exc:
            findings.append(TrialEvalRuntimeSurfaceFinding("manifest.json", f"participant manifest is invalid: {exc}"))
            participant_manifest = None
        if participant_manifest is not None:
            if participant_manifest.task_evidence_factors != factors_by_task:
                findings.append(
                    TrialEvalRuntimeSurfaceFinding(
                        "manifest.json",
                        "participant evidence factors disagree with the evaluator index",
                    )
                )
            manifest_artifacts = {row.rel_path: row for row in participant_manifest.artifacts}
            expected_members = names - {"manifest.json"}
            if set(manifest_artifacts) != expected_members:
                findings.append(
                    TrialEvalRuntimeSurfaceFinding(
                        "manifest.json",
                        "participant artifact inventory differs from ZIP members",
                    )
                )
            else:
                for member_name, record in manifest_artifacts.items():
                    member_bytes = archive.read(member_name)
                    if (
                        len(member_bytes) != record.size_bytes
                        or hashlib.sha256(member_bytes).hexdigest() != record.sha256
                    ):
                        findings.append(
                            TrialEvalRuntimeSurfaceFinding(
                                member_name,
                                "participant artifact digest or size differs from manifest",
                            )
                        )
        for name in sorted(names):
            if name in TRIALEVAL_PARTICIPANT_ROOT_MEMBERS:
                continue
            member_path = Path(name)
            task_id = _task_id_from_public_member(name)
            factors = factors_by_task.get(task_id or "")
            forbidden_parts = set(member_path.parts) & TRIALEVAL_AGENT_FORBIDDEN_ZIP_PARTS
            if forbidden_parts:
                findings.append(
                    TrialEvalRuntimeSurfaceFinding(
                        name,
                        f"participant filesystem exposes evaluator-only path part(s): {sorted(forbidden_parts)}",
                    )
                )
            if task_id is None:
                findings.append(TrialEvalRuntimeSurfaceFinding(name, "participant member is outside items/<task_id>/"))
                continue
            if factors is None:
                findings.append(
                    TrialEvalRuntimeSurfaceFinding(name, "participant task is absent from evaluator index")
                )
                continue
            item_relative_path = PurePosixPath(*member_path.parts[2:]).as_posix()
            try:
                classify_trialeval_item_member(
                    item_relative_path=item_relative_path,
                    data_preparation=factors.data_preparation,
                )
            except ValueError as exc:
                findings.append(TrialEvalRuntimeSurfaceFinding(name, str(exc)))
            if member_path.name in TRIALEVAL_AGENT_FORBIDDEN_ZIP_FILENAMES:
                findings.append(
                    TrialEvalRuntimeSurfaceFinding(
                        name,
                        f"participant filesystem exposes evaluator-only file: {member_path.name}",
                    )
                )
            if member_path.name in trialeval_agent_excluded_files():
                findings.append(
                    TrialEvalRuntimeSurfaceFinding(
                        name,
                        "participant filesystem exposes an evaluator-only file",
                    )
                )
            if member_path.suffix.lower() != ".json":
                continue
            try:
                payload = _read_public_json(archive, name)
            except json.JSONDecodeError as exc:
                findings.append(TrialEvalRuntimeSurfaceFinding(name, f"participant JSON is invalid: {exc.msg}"))
                continue
            forbidden_keys = trialeval_agent_forbidden_json_key_paths(payload)
            if forbidden_keys:
                findings.append(
                    TrialEvalRuntimeSurfaceFinding(
                        name,
                        f"participant JSON exposes target-bearing key(s): {forbidden_keys[:10]!r}",
                    )
                )
            semantic_leaks = trialeval_agent_semantic_leakage_paths(payload)
            if semantic_leaks:
                findings.append(
                    TrialEvalRuntimeSurfaceFinding(
                        name,
                        f"participant JSON contains answer-bearing language at: {semantic_leaks[:10]!r}",
                    )
                )
            if member_path.name == "task.json" and task_id is not None:
                for message in _task_file_reference_findings(task_id=task_id, payload=payload, names=names):
                    findings.append(TrialEvalRuntimeSurfaceFinding(name, message))
                try:
                    TrialEvalParticipantTaskV1.model_validate(payload)
                except ValueError as exc:
                    findings.append(TrialEvalRuntimeSurfaceFinding(name, f"participant task is invalid: {exc}"))
                    continue
            elif member_path.name == "submission_contract.json" and task_id is not None:
                try:
                    output_contract = TrialEvalSemanticSubmissionContractV1.model_validate(payload)
                except ValueError as exc:
                    findings.append(
                        TrialEvalRuntimeSurfaceFinding(name, f"participant output contract is invalid: {exc}")
                    )
                else:
                    if output_contract.task_id != task_id:
                        findings.append(
                            TrialEvalRuntimeSurfaceFinding(name, "participant output-contract/task identity mismatch")
                        )
                    try:
                        output_contract.validate_data_preparation(factors.data_preparation)
                    except ValueError as exc:
                        findings.append(
                            TrialEvalRuntimeSurfaceFinding(
                                name,
                                f"participant output obligations are invalid: {exc}",
                            )
                        )
            elif member_path.name == "data_integrity_policy.json" and task_id is not None:
                try:
                    integrity_policy = TrialEvalPublicIntegrityPolicyV1.model_validate(payload)
                except ValueError as exc:
                    findings.append(
                        TrialEvalRuntimeSurfaceFinding(name, f"participant data-integrity policy is invalid: {exc}")
                    )
                else:
                    if integrity_policy.task_id != task_id:
                        findings.append(
                            TrialEvalRuntimeSurfaceFinding(
                                name,
                                "participant data-integrity policy/task identity mismatch",
                            )
                        )
        for task_id, factors in sorted(factors_by_task.items()):
            prefix = f"items/{task_id}/"
            item_members = {name.removeprefix(prefix) for name in names if name.startswith(prefix)}
            missing = sorted(
                required_trialeval_participant_members(
                    data_preparation=factors.data_preparation,
                    analysis_specification=factors.analysis_specification,
                )
                - item_members
            )
            if missing:
                findings.append(
                    TrialEvalRuntimeSurfaceFinding(
                        prefix,
                        f"{factors.context_configuration} participant item is missing required members: {missing}",
                    )
                )
            if factors.data_preparation == "analysis_ready":
                prepared = {name for name in item_members if name.startswith("data/") and name.endswith(".parquet")}
                if not {"data/ADSL.parquet", "data/ADTTE.parquet"}.issubset(prepared):
                    findings.append(
                        TrialEvalRuntimeSurfaceFinding(
                            prefix,
                            "Analysis-ready item requires data/ADSL.parquet and data/ADTTE.parquet",
                        )
                    )
            else:
                raw = {name for name in item_members if name.startswith("data/raw/") and name.endswith(".parquet")}
                if not raw:
                    findings.append(TrialEvalRuntimeSurfaceFinding(prefix, "Raw-domain item has no source domains"))
    return findings


def validate_trialeval_runtime_surface_package(package_root: Path) -> list[TrialEvalRuntimeSurfaceFinding]:
    """Validate the TrialEvalBench runtime surface inside an HF-style package root."""

    root = Path(package_root)
    return validate_trialeval_runtime_surface(
        public_zip=root / TRIALEVAL_PARTICIPANT_ARCHIVE,
        evaluator_zip=root / TRIALEVAL_EVALUATOR_ARCHIVE,
    )


def main(argv: list[str] | None = None) -> int:
    """Validate TrialEvalBench runtime-surface parity from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", help="HF-style TrialAgentBench package root.")
    args = parser.parse_args(argv)
    findings = validate_trialeval_runtime_surface_package(Path(args.package_root))
    for finding in findings:
        print(f"{finding.path}: {finding.message}")
    return 1 if findings else 0


__all__ = [
    "TrialEvalRuntimeSurfaceFinding",
    "validate_trialeval_runtime_surface",
    "validate_trialeval_runtime_surface_package",
    "main",
]
