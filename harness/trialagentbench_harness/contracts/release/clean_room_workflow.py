"""Clean-room validation for TrialAgentBench release-role boundaries."""

from __future__ import annotations

import json
import re
import tomllib
import zipfile
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trialagentbench_harness.contracts.release.archive_safety import inspect_release_zip
from trialagentbench_harness.contracts.release.artifacts import (
    TRIALDEV_EVALUATOR_ARCHIVE,
    TRIALDEV_FIXED_TRAJECTORY_REPLICATE_MEMBERS,
    TRIALDEV_PARTICIPANT_ARCHIVE,
    TRIALDEV_SCORER_RELATIVE_MEMBERS,
    TRIALDEV_VERIFICATION_ARCHIVE,
    TRIALDEV_VERIFICATION_ROOT_MEMBERS,
    TRIALEVAL_EVALUATOR_ARCHIVE,
    TRIALEVAL_PARTICIPANT_ARCHIVE,
    TRIALEVAL_VERIFICATION_ARCHIVE,
)
from trialagentbench_harness.contracts.release.trialdev_runtime_surface import (
    TRIALDEV_FIXED_OBSERVATIONAL_COLUMNS_V1,
    TrialDevPublicDataDictionaryV1,
    classify_trialdev_participant_archive_member,
    required_trialdev_public_members,
)
from trialagentbench_harness.contracts.release.trialeval_runtime_surface import (
    TRIALEVAL_AGENT_FORBIDDEN_ZIP_FILENAMES,
    TRIALEVAL_AGENT_FORBIDDEN_ZIP_PARTS,
    trialeval_agent_forbidden_json_key_paths,
    trialeval_agent_semantic_leakage_paths,
)
from trialagentbench_harness.contracts.trialdev.trialdev_public_phase_replay import (
    TrialDevPhaseReplayCaseV1,
)
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentBenchmarkSuiteManifestV1

REQUIRED_PARTICIPANT_ARTIFACTS: tuple[str, ...] = (
    TRIALEVAL_PARTICIPANT_ARCHIVE,
    TRIALDEV_PARTICIPANT_ARCHIVE,
)
REQUIRED_EVALUATOR_ARTIFACTS: tuple[str, ...] = (
    TRIALEVAL_EVALUATOR_ARCHIVE,
    TRIALDEV_EVALUATOR_ARCHIVE,
)
REQUIRED_VERIFICATION_ARTIFACTS: tuple[str, ...] = (
    TRIALEVAL_VERIFICATION_ARCHIVE,
    TRIALDEV_VERIFICATION_ARCHIVE,
)
REQUIRED_AUDIT_REPORT_NAMES: tuple[str, ...] = (
    "trialeval_context_artifact_delta_report.json",
    "trialeval_context_sufficiency_report.json",
    "recoverability_report.json",
)
REQUIRED_PUBLIC_HARNESS_ENTRYPOINTS: tuple[str, ...] = ("trialagentbench",)
FORBIDDEN_HARNESS_PATH_MARKERS: tuple[str, ...] = (
    "collaborator_" + "packages/",
    "release_" + "evidence/",
    "manu" + "scripts" + "/",
    "tick" + "ets/",
)
FORBIDDEN_FIXED_PANEL_DOC_MARKERS: tuple[str, ...] = (
    "current trace " "examples",
    "exactly the same " "numbers",
    "six" + "-model",
    "6 " + "models",
    "six " + "evaluated",
    "6 " + "evaluated",
)
MACHINE_LOCAL_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile("/" + "home" + r"/[^\s\"',)]+"),
    re.compile("/" + "Users" + r"/[^\s\"',)]+"),
    re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\\\[^\s\"',)]+"),
)
PUBLIC_TEXT_SUFFIXES: frozenset[str] = frozenset(
    {".cfg", ".csv", ".json", ".jsonl", ".md", ".py", ".rst", ".toml", ".tsv", ".txt", ".yaml", ".yml"}
)
TRIALDEV_PARTICIPANT_FORBIDDEN_PATH_PARTS: frozenset[str] = frozenset(
    {"grader", "hidden", "decision_surface", "oracle", "evaluation_target_register"}
)
TRIALDEV_PARTICIPANT_FORBIDDEN_TEXT_MARKERS: tuple[str, ...] = (
    "counterfactual",
    "evaluation_target_register",
    "diagnostic_reference_route",
    "accepted_answer",
    "decision_surface",
    '"confounding_regime"',
    '"supported_primary_result_kind"',
    '"known_limitations"',
    "residual_unmeasured_confounding",
    "point causal ranking is not identified",
)
PARTICIPANT_BINARY_FORBIDDEN_COLUMN_PREFIXES: tuple[str, ...] = (
    "COUNTERFACTUAL__",
    "DGP__",
    "LATENT__",
    "ORACLE__",
    "TRUE__",
)
PARTICIPANT_BINARY_FORBIDDEN_COLUMNS: frozenset[str] = frozenset({"CANDIDATE_DRUG_ID"})
ALLOWED_PARTICIPANT_PARQUET_METADATA_KEYS: frozenset[bytes] = frozenset({b"ARROW:schema", b"pandas"})


class CleanRoomWorkflowFindingV1(BaseModel):
    """One clean-room release-boundary finding."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.clean_room_workflow_finding/v1"] = (
        "trialagentbench.clean_room_workflow_finding/v1"
    )
    severity: Literal["error", "warning"] = "error"
    code: str = Field(min_length=1)
    path: str = Field(min_length=1)
    message: str = Field(min_length=1)


class CleanRoomSurfaceSummaryV1(BaseModel):
    """Summary of one validated release role surface."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["participant", "evaluator", "verification", "audit", "harness"]
    root: str
    artifact_count: int = Field(ge=0)
    finding_count: int = Field(ge=0)


class CleanRoomWorkflowReportV1(BaseModel):
    """Clean-room release workflow validation report."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.clean_room_workflow_report/v1"] = (
        "trialagentbench.clean_room_workflow_report/v1"
    )
    status: Literal["pass", "fail"]
    package_root: str
    harness_root: str
    audit_roots: tuple[str, ...] = ()
    participant_artifacts: tuple[str, ...]
    evaluator_artifacts: tuple[str, ...]
    verification_artifacts: tuple[str, ...]
    audit_artifacts: tuple[str, ...]
    surfaces: tuple[CleanRoomSurfaceSummaryV1, ...]
    findings: tuple[CleanRoomWorkflowFindingV1, ...]


def _zip_names(path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path) as archive:
        return tuple(sorted(name for name in archive.namelist() if not name.endswith("/")))


def _read_zip_text(archive: zipfile.ZipFile, name: str) -> str:
    return archive.read(name).decode("utf-8", errors="replace")


def _read_zip_parquet_schema(archive: zipfile.ZipFile, name: str) -> pa.Schema:
    parquet = pq.ParquetFile(BytesIO(archive.read(name)))
    schema = parquet.schema_arrow
    file_metadata = parquet.metadata.metadata or {}
    unknown_metadata = sorted(set(file_metadata) - ALLOWED_PARTICIPANT_PARQUET_METADATA_KEYS)
    if unknown_metadata:
        raise ValueError(
            "participant Parquet contains undeclared file metadata keys: "
            f"{[key.decode('utf-8', errors='replace') for key in unknown_metadata]!r}"
        )
    schema_metadata = schema.metadata or {}
    unknown_schema_metadata = sorted(set(schema_metadata) - {b"pandas"})
    if unknown_schema_metadata:
        raise ValueError(
            "participant Parquet contains undeclared Arrow schema metadata keys: "
            f"{[key.decode('utf-8', errors='replace') for key in unknown_schema_metadata]!r}"
        )
    fields_with_metadata = sorted(field.name for field in schema if field.metadata)
    if fields_with_metadata:
        raise ValueError(f"participant Parquet contains undeclared field metadata: {fields_with_metadata!r}")
    return schema


def _validate_zip_archive(
    path: Path,
    *,
    role: Literal["participant", "evaluator", "verification"],
) -> list[CleanRoomWorkflowFindingV1]:
    if not path.is_file():
        return []
    return [
        _finding(
            f"{role}_archive_{issue.code}",
            f"{path}:{issue.member}" if issue.member is not None else path,
            issue.message,
        )
        for issue in inspect_release_zip(path)
    ]


def _validate_release_tree(
    root: Path,
    *,
    role: Literal["package", "harness", "audit"],
) -> list[CleanRoomWorkflowFindingV1]:
    if not root.exists():
        return []
    findings: list[CleanRoomWorkflowFindingV1] = []
    candidates = (root, *sorted(root.rglob("*")))
    for path in candidates:
        if path.is_symlink():
            findings.append(
                _finding(
                    f"{role}_filesystem_link",
                    path,
                    "clean-room release surfaces must not contain symbolic links",
                )
            )
        elif not path.is_file() and not path.is_dir():
            findings.append(
                _finding(
                    f"{role}_filesystem_special_file",
                    path,
                    "clean-room release surfaces may contain only regular files and directories",
                )
            )
    return findings


def _forbidden_participant_binary_columns(schema: pa.Schema) -> list[str]:
    return sorted(
        field.name
        for field in schema
        if field.name.upper() in PARTICIPANT_BINARY_FORBIDDEN_COLUMNS
        or any(field.name.upper().startswith(prefix) for prefix in PARTICIPANT_BINARY_FORBIDDEN_COLUMN_PREFIXES)
    )


def _declared_dtype_matches(*, expected: str, observed: pa.DataType) -> bool:
    normalized = expected.strip().lower()
    if normalized in {"bool", "boolean"}:
        return pa.types.is_boolean(observed)
    if normalized in {"string", "str", "object"}:
        return (
            pa.types.is_string(observed)
            or pa.types.is_large_string(observed)
            or (pa.types.is_dictionary(observed) and pa.types.is_string(observed.value_type))
        )
    primitive = {
        "float32": pa.float32(),
        "float64": pa.float64(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "uint64": pa.uint64(),
    }.get(normalized)
    if primitive is not None:
        return observed.equals(primitive)
    if normalized.startswith("datetime64"):
        return pa.types.is_timestamp(observed)
    return str(observed).lower() == normalized


def _validate_trialeval_binary_schemas(
    archive: zipfile.ZipFile,
    *,
    archive_path: Path,
    names: tuple[str, ...],
) -> list[CleanRoomWorkflowFindingV1]:
    findings: list[CleanRoomWorkflowFindingV1] = []
    for name in (member for member in names if member.endswith(".parquet")):
        parts = Path(name).parts
        if len(parts) < 4 or parts[0] != "items":
            findings.append(
                _finding(
                    "participant_binary_unknown_member",
                    f"{archive_path}:{name}",
                    "TrialEval Parquet member is outside an item data surface",
                )
            )
            continue
        dictionary_name = f"items/{parts[1]}/data_dictionary.json"
        if dictionary_name not in names:
            findings.append(
                _finding(
                    "participant_binary_schema_undeclared",
                    f"{archive_path}:{name}",
                    "TrialEval item has no public data dictionary",
                )
            )
            continue
        try:
            dictionary = json.loads(_read_zip_text(archive, dictionary_name))
            semantic_columns = dictionary["semantic_columns"]
            if not isinstance(semantic_columns, list):
                raise TypeError("semantic_columns must be a list")
            item_relative = Path(*parts[2:]).as_posix()
            declared_rows = [
                row for row in semantic_columns if isinstance(row, dict) and row.get("table") == item_relative
            ]
            schema = _read_zip_parquet_schema(archive, name)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            findings.append(
                _finding(
                    "participant_binary_schema_invalid",
                    f"{archive_path}:{name}",
                    f"Cannot validate TrialEval binary schema: {exc}",
                )
            )
            continue
        declared_by_column = {
            str(row.get("column")): str(row.get("dtype"))
            for row in declared_rows
            if row.get("column") and row.get("dtype")
        }
        if len(declared_by_column) != len(declared_rows):
            findings.append(
                _finding(
                    "participant_binary_schema_invalid",
                    f"{archive_path}:{dictionary_name}",
                    f"TrialEval data dictionary has incomplete or duplicate declarations for {item_relative}",
                )
            )
        observed_by_column = {field.name: field.type for field in schema}
        if len(observed_by_column) != len(schema):
            findings.append(
                _finding(
                    "participant_binary_duplicate_column",
                    f"{archive_path}:{name}",
                    "TrialEval Parquet schema contains duplicate columns",
                )
            )
        if set(observed_by_column) != set(declared_by_column):
            findings.append(
                _finding(
                    "participant_binary_schema_mismatch",
                    f"{archive_path}:{name}",
                    f"TrialEval Parquet columns differ from the public dictionary: missing={sorted(set(declared_by_column) - set(observed_by_column))!r} undeclared={sorted(set(observed_by_column) - set(declared_by_column))!r}",
                )
            )
        incompatible = sorted(
            column
            for column in set(observed_by_column) & set(declared_by_column)
            if not _declared_dtype_matches(expected=declared_by_column[column], observed=observed_by_column[column])
        )
        if incompatible:
            findings.append(
                _finding(
                    "participant_binary_type_mismatch",
                    f"{archive_path}:{name}",
                    f"TrialEval Parquet types differ from the public dictionary: {incompatible!r}",
                )
            )
        forbidden = _forbidden_participant_binary_columns(schema)
        if forbidden:
            findings.append(
                _finding(
                    "participant_binary_private_column",
                    f"{archive_path}:{name}",
                    f"TrialEval participant data exposes construction-only columns: {forbidden!r}",
                )
            )
    return findings


def _validate_trialdev_binary_schemas(
    archive: zipfile.ZipFile,
    *,
    archive_path: Path,
    names: tuple[str, ...],
) -> list[CleanRoomWorkflowFindingV1]:
    findings: list[CleanRoomWorkflowFindingV1] = []
    for name in names:
        if not name.startswith("fixed_trajectories/materialized/") or not name.endswith(".parquet"):
            continue
        try:
            trajectory_schema = _read_zip_parquet_schema(archive, name)
        except (TypeError, ValueError, OSError) as exc:
            findings.append(
                _finding(
                    "participant_fixed_trajectory_schema_invalid",
                    f"{archive_path}:{name}",
                    f"Cannot read fixed trajectory schema: {exc}",
                )
            )
            continue
        forbidden = _forbidden_participant_binary_columns(trajectory_schema)
        if forbidden:
            findings.append(
                _finding(
                    "participant_binary_private_column",
                    f"{archive_path}:{name}",
                    f"TrialDev fixed trajectory exposes construction-only columns: {forbidden!r}",
                )
            )
    scenarios = sorted({Path(name).parts[0] for name in names if Path(name).parts[0].startswith("scenario_")})
    for scenario_id in scenarios:
        dictionary_name = f"{scenario_id}/public/data_dictionary.json"
        parquet_name = f"{scenario_id}/public/observational_extract.parquet"
        if dictionary_name not in names or parquet_name not in names:
            continue
        try:
            dictionary = TrialDevPublicDataDictionaryV1.model_validate_json(_read_zip_text(archive, dictionary_name))
            schema = _read_zip_parquet_schema(archive, parquet_name)
            catalog_columns = _trialdev_catalog_observational_columns(
                archive,
                scenario_id=scenario_id,
                names=names,
            )
        except (TypeError, ValueError, OSError) as exc:
            findings.append(
                _finding(
                    "participant_binary_schema_invalid",
                    f"{archive_path}:{scenario_id}",
                    f"Cannot validate TrialDev binary schema: {exc}",
                )
            )
            continue
        declared = tuple((field.column, field.arrow_type, field.nullable) for field in dictionary.observational_schema)
        observed = tuple((field.name, str(field.type), field.nullable) for field in schema)
        if observed != declared:
            findings.append(
                _finding(
                    "participant_binary_schema_mismatch",
                    f"{archive_path}:{parquet_name}",
                    "TrialDev observational schema differs from its checksummed public dictionary",
                )
            )
        observed_columns = {field.name for field in schema}
        if observed_columns != catalog_columns:
            findings.append(
                _finding(
                    "participant_binary_unknown_column",
                    f"{archive_path}:{parquet_name}",
                    "TrialDev observational columns differ from the independent public catalogs: "
                    f"missing={sorted(catalog_columns - observed_columns)!r} "
                    f"unknown={sorted(observed_columns - catalog_columns)!r}",
                )
            )
        forbidden = _forbidden_participant_binary_columns(schema)
        if forbidden:
            findings.append(
                _finding(
                    "participant_binary_private_column",
                    f"{archive_path}:{parquet_name}",
                    f"TrialDev participant data exposes construction-only columns: {forbidden!r}",
                )
            )
    return findings


def _trialdev_catalog_observational_columns(
    archive: zipfile.ZipFile,
    *,
    scenario_id: str,
    names: tuple[str, ...],
) -> set[str]:
    public_prefix = f"{scenario_id}/public/"
    required = {
        "variable_catalog": f"{public_prefix}variable_catalog.json",
        "endpoint_catalog": f"{public_prefix}endpoint_catalog.json",
        "safety_policy": f"{public_prefix}safety_decision_policy.json",
    }
    missing = sorted(set(required.values()) - set(names))
    if missing:
        raise ValueError(f"TrialDev public catalogs are missing: {missing!r}")
    variable_catalog = json.loads(_read_zip_text(archive, required["variable_catalog"]))
    endpoint_catalog = json.loads(_read_zip_text(archive, required["endpoint_catalog"]))
    safety_policy = json.loads(_read_zip_text(archive, required["safety_policy"]))
    if not isinstance(variable_catalog, dict) or not isinstance(variable_catalog.get("variables"), list):
        raise ValueError("TrialDev variable catalog must contain a variables array")
    if not isinstance(endpoint_catalog, dict) or not isinstance(endpoint_catalog.get("endpoints"), list):
        raise ValueError("TrialDev endpoint catalog must contain an endpoints array")
    if not isinstance(safety_policy, dict) or not isinstance(safety_policy.get("serious_event_definitions"), list):
        raise ValueError("TrialDev safety policy must contain serious_event_definitions")

    columns = set(TRIALDEV_FIXED_OBSERVATIONAL_COLUMNS_V1)
    for row in variable_catalog["variables"]:
        if not isinstance(row, dict) or not str(row.get("variable_id", "")).strip():
            raise ValueError("TrialDev variable catalog entries require variable_id")
        columns.add(str(row["variable_id"]))
    for row in endpoint_catalog["endpoints"]:
        if not isinstance(row, dict) or not str(row.get("endpoint_id", "")).strip():
            raise ValueError("TrialDev endpoint catalog entries require endpoint_id")
        outcome_id = f"EFF_{row['endpoint_id']}"
        columns.update({f"{outcome_id}_T", f"{outcome_id}_E"})
    safety_keys = ("event_column", "time_column", "seriousness_column", "severity_column")
    for row in safety_policy["serious_event_definitions"]:
        if not isinstance(row, dict) or any(not str(row.get(key, "")).strip() for key in safety_keys):
            raise ValueError("TrialDev serious-event definitions require all data columns")
        columns.update(str(row[key]) for key in safety_keys)
    return columns


def _iter_text_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if (
            "tests" not in path.relative_to(root).parts
            and path.is_file()
            and path.suffix.lower() in PUBLIC_TEXT_SUFFIXES
        ):
            yield path


def _load_pyproject_scripts(pyproject_path: Path) -> dict[str, str]:
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = payload.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        return {}
    return {str(key): str(value) for key, value in scripts.items()}


def _finding(code: str, path: str | Path, message: str) -> CleanRoomWorkflowFindingV1:
    return CleanRoomWorkflowFindingV1(code=code, path=Path(path).as_posix(), message=message)


def _validate_required_files(root: Path, required: Iterable[str], *, code: str) -> list[CleanRoomWorkflowFindingV1]:
    findings: list[CleanRoomWorkflowFindingV1] = []
    for relative in required:
        if not (root / relative).is_file():
            findings.append(_finding(code, relative, "required clean-room release artifact is missing"))
    return findings


def _validate_trialeval_participant_zip(path: Path) -> list[CleanRoomWorkflowFindingV1]:
    findings: list[CleanRoomWorkflowFindingV1] = []
    if not path.is_file():
        return findings
    with zipfile.ZipFile(path) as archive:
        names = _zip_names(path)
        for name in names:
            member = Path(name)
            forbidden_parts = sorted(set(member.parts) & TRIALEVAL_AGENT_FORBIDDEN_ZIP_PARTS)
            if forbidden_parts:
                findings.append(
                    _finding(
                        "participant_evaluator_path_leak",
                        f"{path}:{name}",
                        f"TrialEval participant member exposes evaluator-only path part(s): {forbidden_parts}",
                    )
                )
            if member.name in TRIALEVAL_AGENT_FORBIDDEN_ZIP_FILENAMES:
                findings.append(
                    _finding(
                        "participant_evaluator_file_leak",
                        f"{path}:{name}",
                        f"TrialEval participant member exposes evaluator-only file {member.name!r}",
                    )
                )
            if member.suffix.lower() != ".json":
                continue
            try:
                payload = json.loads(_read_zip_text(archive, name))
            except json.JSONDecodeError:
                continue
            forbidden_keys = trialeval_agent_forbidden_json_key_paths(payload)
            if forbidden_keys:
                findings.append(
                    _finding(
                        "participant_target_field_leak",
                        f"{path}:{name}",
                        f"TrialEval participant JSON exposes target-bearing keys: {forbidden_keys[:10]!r}",
                    )
                )
            semantic_leaks = trialeval_agent_semantic_leakage_paths(payload)
            if semantic_leaks:
                findings.append(
                    _finding(
                        "participant_answer_language_leak",
                        f"{path}:{name}",
                        f"TrialEval participant JSON contains answer-bearing language: {semantic_leaks[:10]!r}",
                    )
                )
        findings.extend(_validate_trialeval_binary_schemas(archive, archive_path=path, names=names))
    return findings


def _validate_trialdev_fixed_trajectory_index(
    archive: zipfile.ZipFile,
    *,
    archive_path: Path,
    names: tuple[str, ...],
) -> list[CleanRoomWorkflowFindingV1]:
    """Require every declared fixed trajectory to identify one released replicate."""

    index_name = "fixed_trajectories/cases.jsonl"
    if index_name not in names:
        return []
    findings: list[CleanRoomWorkflowFindingV1] = []
    try:
        lines = tuple(line for line in _read_zip_text(archive, index_name).splitlines() if line.strip())
        cases = tuple(TrialDevPhaseReplayCaseV1.model_validate_json(line) for line in lines)
    except (UnicodeDecodeError, ValidationError, ValueError) as exc:
        return [
            _finding(
                "participant_fixed_trajectory_index_invalid",
                f"{archive_path}:{index_name}",
                f"Fixed trajectory index violates its public contract: {exc}",
            )
        ]
    if not cases:
        return [
            _finding(
                "participant_fixed_trajectory_index_empty",
                f"{archive_path}:{index_name}",
                "Fixed trajectory index contains no replay cases.",
            )
        ]
    case_keys = tuple(
        (
            case.request.scenario_id,
            str(case.request.phase_id),
            tuple(case.request.candidate_drug_ids),
        )
        for case in cases
    )
    if len(case_keys) != len(set(case_keys)):
        findings.append(
            _finding(
                "participant_fixed_trajectory_case_ambiguous",
                f"{archive_path}:{index_name}",
                "Fixed trajectories must be unique by scenario, phase, and nominated asset.",
            )
        )
    materialized_names = tuple(name for name in names if name.startswith("fixed_trajectories/materialized/"))
    expected_request_roots: set[str] = set()
    for case in cases:
        request_root = "fixed_trajectories/materialized/" f"world_{case.world_seed}/request_{case.request.checksum()}"
        expected_request_roots.add(request_root)
        replicate_roots = {
            "/".join(Path(name).parts[:5])
            for name in materialized_names
            if name.startswith(f"{request_root}/trial_seed_") and len(Path(name).parts) >= 6
        }
        if len(replicate_roots) != 1:
            findings.append(
                _finding(
                    "participant_fixed_trajectory_replicate_count_invalid",
                    f"{archive_path}:{request_root}",
                    "Each fixed trajectory case must identify exactly one released evidence replicate.",
                )
            )
            continue
        replicate_root = next(iter(replicate_roots))
        replicate_members = {
            name.removeprefix(f"{replicate_root}/")
            for name in materialized_names
            if name.startswith(f"{replicate_root}/")
        }
        missing_members = sorted(TRIALDEV_FIXED_TRAJECTORY_REPLICATE_MEMBERS - replicate_members)
        if missing_members:
            findings.append(
                _finding(
                    "participant_fixed_trajectory_incomplete",
                    f"{archive_path}:{replicate_root}",
                    f"Fixed trajectory evidence is missing runtime inputs: {missing_members!r}.",
                )
            )
    observed_request_roots = {
        "/".join(Path(name).parts[:4]) for name in materialized_names if len(Path(name).parts) >= 6
    }
    unindexed = sorted(observed_request_roots - expected_request_roots)
    if unindexed:
        findings.append(
            _finding(
                "participant_fixed_trajectory_unindexed",
                str(archive_path),
                f"Fixed trajectory data are not declared by the public index: {unindexed!r}",
            )
        )
    return findings


def _validate_trialdev_participant_zip(path: Path) -> list[CleanRoomWorkflowFindingV1]:
    findings: list[CleanRoomWorkflowFindingV1] = []
    if not path.is_file():
        return findings
    with zipfile.ZipFile(path) as archive:
        names = _zip_names(path)
        if "benchmark_suite_manifest.json" not in names:
            findings.append(
                _finding(
                    "participant_suite_manifest_missing",
                    str(path),
                    "TrialDev participant archive is not runnable without benchmark_suite_manifest.json.",
                )
            )
        if "fixed_trajectories/cases.jsonl" not in names:
            findings.append(
                _finding(
                    "participant_fixed_trajectory_index_missing",
                    str(path),
                    "TrialDev participant archive lacks its fixed trajectory index.",
                )
            )
        if not any(name.startswith("fixed_trajectories/materialized/") for name in names):
            findings.append(
                _finding(
                    "participant_fixed_trajectory_data_missing",
                    str(path),
                    "TrialDev participant archive lacks fixed randomized-phase evidence.",
                )
            )
        findings.extend(
            _validate_trialdev_fixed_trajectory_index(
                archive,
                archive_path=path,
                names=names,
            )
        )
        for name in names:
            member = Path(name)
            forbidden_parts = sorted(set(member.parts) & TRIALDEV_PARTICIPANT_FORBIDDEN_PATH_PARTS)
            if forbidden_parts:
                findings.append(
                    _finding(
                        "participant_evaluator_path_leak",
                        f"{path}:{name}",
                        f"TrialDev participant member exposes evaluator-only path part(s): {forbidden_parts}",
                    )
                )
            try:
                classify_trialdev_participant_archive_member(name)
            except ValueError as exc:
                findings.append(_finding("participant_unknown_member", f"{path}:{name}", str(exc)))
            if member.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
                continue
            text = _read_zip_text(archive, name)
            markers = sorted(marker for marker in TRIALDEV_PARTICIPANT_FORBIDDEN_TEXT_MARKERS if marker in text)
            if markers:
                findings.append(
                    _finding(
                        "participant_decision_reference_text_leak",
                        f"{path}:{name}",
                        f"TrialDev participant member exposes evaluator-only marker(s): {markers}",
                    )
                )
        findings.extend(_validate_trialdev_binary_schemas(archive, archive_path=path, names=names))
        scenario_ids = sorted({Path(name).parts[0] for name in names if Path(name).parts[0].startswith("scenario_")})
        if "benchmark_suite_manifest.json" in names:
            try:
                raw_manifest = json.loads(archive.read("benchmark_suite_manifest.json"))
                manifest = TrialDevelopmentBenchmarkSuiteManifestV1.model_validate(raw_manifest)
            except (json.JSONDecodeError, ValidationError) as exc:
                findings.append(
                    _finding(
                        "participant_suite_manifest_invalid",
                        f"{path}:benchmark_suite_manifest.json",
                        f"TrialDev participant suite manifest is invalid: {exc}",
                    )
                )
            else:
                if not isinstance(raw_manifest, dict) or raw_manifest.get("checksum") != manifest.checksum:
                    findings.append(
                        _finding(
                            "participant_suite_manifest_checksum",
                            f"{path}:benchmark_suite_manifest.json",
                            "TrialDev participant suite manifest checksum does not match its inventory.",
                        )
                    )
                declared_scenarios = {item.scenario_id for item in manifest.items}
                archive_scenarios = {scenario_id.removeprefix("scenario_") for scenario_id in scenario_ids}
                if declared_scenarios != archive_scenarios:
                    findings.append(
                        _finding(
                            "participant_suite_manifest_inventory",
                            f"{path}:benchmark_suite_manifest.json",
                            "TrialDev participant suite manifest scenarios differ from archive scenarios.",
                        )
                    )
        for scenario_id in scenario_ids:
            public_prefix = f"{scenario_id}/public/"
            public_members = {name.removeprefix(public_prefix) for name in names if name.startswith(public_prefix)}
            missing = sorted(required_trialdev_public_members() - public_members)
            if missing:
                findings.append(
                    _finding(
                        "participant_required_member_missing",
                        f"{path}:{public_prefix}",
                        f"TrialDev participant scenario is missing required public members: {missing}",
                    )
                )
    return findings


def _scenario_relative_members(names: Iterable[str]) -> dict[str, set[str]]:
    members: dict[str, set[str]] = {}
    for name in names:
        parts = Path(name).parts
        if len(parts) < 2 or not parts[0].startswith("scenario_"):
            continue
        members.setdefault(parts[0], set()).add(Path(*parts[1:]).as_posix())
    return members


def _validate_trialdev_evaluator_zip(path: Path) -> list[CleanRoomWorkflowFindingV1]:
    findings: list[CleanRoomWorkflowFindingV1] = []
    if not path.is_file():
        return findings
    names = _zip_names(path)
    scenarios = _scenario_relative_members(names)
    if not scenarios:
        return [_finding("evaluator_scenario_missing", path, "TrialDev evaluator contains no scenario scorer state")]
    unscoped = sorted(name for name in names if not Path(name).parts[0].startswith("scenario_"))
    for name in unscoped:
        findings.append(_finding("evaluator_unknown_member", f"{path}:{name}", "unscoped evaluator member"))
    for scenario_id, observed in sorted(scenarios.items()):
        expected = set(TRIALDEV_SCORER_RELATIVE_MEMBERS)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing:
            findings.append(
                _finding(
                    "evaluator_required_member_missing",
                    f"{path}:{scenario_id}",
                    f"TrialDev scorer projection is missing members: {missing}",
                )
            )
        for member in unexpected:
            findings.append(
                _finding(
                    "evaluator_unknown_member",
                    f"{path}:{scenario_id}/{member}",
                    "artifact is not a deterministic score-time dependency",
                )
            )
    return findings


def _validate_trialdev_verification_zip(path: Path) -> list[CleanRoomWorkflowFindingV1]:
    findings: list[CleanRoomWorkflowFindingV1] = []
    if not path.is_file():
        return findings
    names = set(_zip_names(path))
    required = {
        "phase_replay/cases.jsonl",
        "phase_replay/records.jsonl",
        *TRIALDEV_VERIFICATION_ROOT_MEMBERS,
    }
    for member in sorted(required - names):
        findings.append(
            _finding(
                "verification_required_member_missing",
                f"{path}:{member}",
                "TrialDev verification is missing a phase-replay contract member",
            )
        )
    if not any(name.startswith("phase_replay/materialized/") for name in names):
        findings.append(
            _finding(
                "verification_materialized_tables_missing",
                path,
                "TrialDev verification contains no retained randomized-phase tables",
            )
        )
    forbidden_parts = {"hidden", "grader", "oracle", "counterfactual_pool.parquet"}
    for name in sorted(names):
        if not name.startswith("phase_replay/") and name not in TRIALDEV_VERIFICATION_ROOT_MEMBERS:
            findings.append(
                _finding(
                    "verification_unknown_member",
                    f"{path}:{name}",
                    "TrialDev verification members must be scoped under phase_replay/",
                )
            )
        if forbidden_parts.intersection(Path(name).parts):
            findings.append(
                _finding(
                    "verification_private_state",
                    f"{path}:{name}",
                    "TrialDev verification must not contain generating or score-time state",
                )
            )
    return findings


def _validate_harness_root(harness_root: Path) -> list[CleanRoomWorkflowFindingV1]:
    findings: list[CleanRoomWorkflowFindingV1] = []
    pyproject = harness_root / "pyproject.toml"
    if not pyproject.is_file():
        findings.append(_finding("harness_pyproject_missing", pyproject, "public harness pyproject is missing"))
    else:
        scripts = _load_pyproject_scripts(pyproject)
        for script in REQUIRED_PUBLIC_HARNESS_ENTRYPOINTS:
            if script not in scripts:
                findings.append(
                    _finding(
                        "harness_required_cli_missing",
                        "pyproject.toml",
                        f"public harness does not expose required CLI {script!r}",
                    )
                )
    evaluator_judge_root = harness_root / "eval_judge"
    if evaluator_judge_root.exists():
        findings.append(
            _finding(
                "harness_agent_run_surface_exposed",
                "eval_judge",
                "public clean-room harness must not expose evaluator-judge surfaces",
            )
        )
    for path in _iter_text_files(harness_root):
        rel = path.relative_to(harness_root)
        text = path.read_text(encoding="utf-8", errors="replace")
        markers = sorted(marker for marker in FORBIDDEN_HARNESS_PATH_MARKERS if marker in text)
        if markers:
            findings.append(
                _finding(
                    "harness_local_artifact_reference",
                    rel,
                    f"public harness text references non-release path marker(s): {markers}",
                )
            )
        fixed_panel_markers = sorted(marker for marker in FORBIDDEN_FIXED_PANEL_DOC_MARKERS if marker in text)
        if fixed_panel_markers:
            findings.append(
                _finding(
                    "harness_fixed_panel_assumption",
                    rel,
                    f"public harness text implies fixed author-run model/trace assumptions: {fixed_panel_markers}",
                )
            )
        if any(pattern.search(text) for pattern in MACHINE_LOCAL_PATH_PATTERNS):
            findings.append(
                _finding("harness_machine_local_path", rel, "public harness text contains a machine-local path")
            )
    return findings


def _validate_audit_root(path: Path) -> list[CleanRoomWorkflowFindingV1]:
    findings: list[CleanRoomWorkflowFindingV1] = []
    if not path.is_dir():
        return [_finding("audit_root_missing", path, "explicit audit/witness root does not exist")]
    report_names = {item.name for item in path.rglob("*.json")}
    if not any(name in report_names for name in REQUIRED_AUDIT_REPORT_NAMES):
        findings.append(
            _finding(
                "audit_root_has_no_known_witness_report",
                path,
                "audit root contains no recognized context, trace-contamination, or decision-witness report",
            )
        )
    return findings


def _artifact_count(root: Path, relatives: Iterable[str]) -> int:
    return sum(1 for relative in relatives if (root / relative).is_file())


def validate_clean_room_workflow(
    *,
    package_root: Path,
    harness_root: Path,
    audit_roots: Iterable[Path] = (),
) -> CleanRoomWorkflowReportV1:
    """Validate clean-room role separation from unpacked release artifacts."""

    package_root = Path(package_root)
    harness_root = Path(harness_root)
    audit_roots_tuple = tuple(Path(root) for root in audit_roots)
    findings: list[CleanRoomWorkflowFindingV1] = []
    if not package_root.is_dir():
        findings.append(_finding("package_root_missing", package_root, "HF-style package root does not exist"))
    if not harness_root.is_dir():
        findings.append(_finding("harness_root_missing", harness_root, "GitHub-style harness root does not exist"))
    if package_root.is_dir():
        findings.extend(_validate_release_tree(package_root, role="package"))
        findings.extend(
            _validate_required_files(package_root, REQUIRED_PARTICIPANT_ARTIFACTS, code="participant_artifact_missing")
        )
        findings.extend(
            _validate_required_files(package_root, REQUIRED_EVALUATOR_ARTIFACTS, code="evaluator_artifact_missing")
        )
        findings.extend(
            _validate_required_files(
                package_root,
                REQUIRED_VERIFICATION_ARTIFACTS,
                code="verification_artifact_missing",
            )
        )
        archive_roles: dict[str, Literal["participant", "evaluator", "verification"]] = {
            TRIALEVAL_PARTICIPANT_ARCHIVE: "participant",
            TRIALDEV_PARTICIPANT_ARCHIVE: "participant",
            TRIALEVAL_EVALUATOR_ARCHIVE: "evaluator",
            TRIALDEV_EVALUATOR_ARCHIVE: "evaluator",
            TRIALEVAL_VERIFICATION_ARCHIVE: "verification",
            TRIALDEV_VERIFICATION_ARCHIVE: "verification",
        }
        unsafe_archives: set[str] = set()
        for relative, role in archive_roles.items():
            archive_findings = _validate_zip_archive(package_root / relative, role=role)
            findings.extend(archive_findings)
            if archive_findings:
                unsafe_archives.add(relative)
        if TRIALEVAL_PARTICIPANT_ARCHIVE not in unsafe_archives:
            findings.extend(_validate_trialeval_participant_zip(package_root / TRIALEVAL_PARTICIPANT_ARCHIVE))
        if TRIALDEV_PARTICIPANT_ARCHIVE not in unsafe_archives:
            findings.extend(_validate_trialdev_participant_zip(package_root / TRIALDEV_PARTICIPANT_ARCHIVE))
        if TRIALDEV_EVALUATOR_ARCHIVE not in unsafe_archives:
            findings.extend(_validate_trialdev_evaluator_zip(package_root / TRIALDEV_EVALUATOR_ARCHIVE))
        if TRIALDEV_VERIFICATION_ARCHIVE not in unsafe_archives:
            findings.extend(_validate_trialdev_verification_zip(package_root / TRIALDEV_VERIFICATION_ARCHIVE))
    if harness_root.is_dir():
        findings.extend(_validate_release_tree(harness_root, role="harness"))
        findings.extend(_validate_harness_root(harness_root))
    for audit_root in audit_roots_tuple:
        findings.extend(_validate_release_tree(audit_root, role="audit"))
        findings.extend(_validate_audit_root(audit_root))
    participant_finding_count = sum(1 for finding in findings if finding.code.startswith("participant_"))
    evaluator_finding_count = sum(1 for finding in findings if finding.code.startswith("evaluator_"))
    verification_finding_count = sum(1 for finding in findings if finding.code.startswith("verification_"))
    audit_finding_count = sum(1 for finding in findings if finding.code.startswith("audit_"))
    harness_finding_count = sum(1 for finding in findings if finding.code.startswith("harness_"))
    verification_artifacts = tuple(
        relative
        for relative in REQUIRED_VERIFICATION_ARTIFACTS
        if package_root.is_dir() and (package_root / relative).is_file()
    )
    audit_artifacts = tuple(root.as_posix() for root in audit_roots_tuple if root.is_dir())
    surfaces = (
        CleanRoomSurfaceSummaryV1(
            role="participant",
            root=package_root.as_posix(),
            artifact_count=(
                _artifact_count(package_root, REQUIRED_PARTICIPANT_ARTIFACTS) if package_root.is_dir() else 0
            ),
            finding_count=participant_finding_count,
        ),
        CleanRoomSurfaceSummaryV1(
            role="evaluator",
            root=package_root.as_posix(),
            artifact_count=_artifact_count(package_root, REQUIRED_EVALUATOR_ARTIFACTS) if package_root.is_dir() else 0,
            finding_count=evaluator_finding_count,
        ),
        CleanRoomSurfaceSummaryV1(
            role="verification",
            root=package_root.as_posix(),
            artifact_count=len(verification_artifacts),
            finding_count=verification_finding_count,
        ),
        CleanRoomSurfaceSummaryV1(
            role="audit",
            root=package_root.as_posix(),
            artifact_count=len(audit_artifacts),
            finding_count=audit_finding_count,
        ),
        CleanRoomSurfaceSummaryV1(
            role="harness",
            root=harness_root.as_posix(),
            artifact_count=(
                sum(1 for _ in harness_root.rglob("*") if harness_root.is_dir()) if harness_root.is_dir() else 0
            ),
            finding_count=harness_finding_count,
        ),
    )
    return CleanRoomWorkflowReportV1(
        status="fail" if findings else "pass",
        package_root=package_root.as_posix(),
        harness_root=harness_root.as_posix(),
        audit_roots=tuple(root.as_posix() for root in audit_roots_tuple),
        participant_artifacts=tuple(
            relative
            for relative in REQUIRED_PARTICIPANT_ARTIFACTS
            if package_root.is_dir() and (package_root / relative).is_file()
        ),
        evaluator_artifacts=tuple(
            relative
            for relative in REQUIRED_EVALUATOR_ARTIFACTS
            if package_root.is_dir() and (package_root / relative).is_file()
        ),
        verification_artifacts=verification_artifacts,
        audit_artifacts=audit_artifacts,
        surfaces=surfaces,
        findings=tuple(findings),
    )


def clean_room_workflow_markdown(report: CleanRoomWorkflowReportV1) -> str:
    """Render a compact Markdown clean-room workflow report."""

    lines = [
        "# TrialAgentBench Clean-Room Workflow Validation",
        "",
        f"- Status: `{report.status}`",
        f"- Package root: `{report.package_root}`",
        f"- Harness root: `{report.harness_root}`",
        f"- Findings: `{len(report.findings)}`",
        "",
        "## Role Surfaces",
        "",
        "| Role | Artifact count | Finding count |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {surface.role} | {surface.artifact_count} | {surface.finding_count} |" for surface in report.surfaces
    )
    if report.findings:
        lines.extend(["", "## Findings", ""])
        lines.extend(f"- `{finding.code}` `{finding.path}`: {finding.message}" for finding in report.findings)
    return "\n".join(lines) + "\n"


__all__ = [
    "CleanRoomSurfaceSummaryV1",
    "CleanRoomWorkflowFindingV1",
    "CleanRoomWorkflowReportV1",
    "REQUIRED_AUDIT_REPORT_NAMES",
    "REQUIRED_EVALUATOR_ARTIFACTS",
    "REQUIRED_PARTICIPANT_ARTIFACTS",
    "REQUIRED_PUBLIC_HARNESS_ENTRYPOINTS",
    "clean_room_workflow_markdown",
    "validate_clean_room_workflow",
]
