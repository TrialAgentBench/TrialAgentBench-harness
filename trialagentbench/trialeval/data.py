"""Current-release TrialEval item discovery and participant context."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from pydantic import JsonValue as PydanticJsonValue
from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1

from trialagentbench.contracts.core.trialeval_factors import (
    TrialEvalDataPreparationV1,
    TrialEvalEvidenceFactorsV1,
)
from trialagentbench.contracts.release import (
    TrialEvalParticipantDiagnosticDictionaryV1,
    TrialEvalParticipantManifestV1,
)
from trialagentbench.contracts.release.trialeval_integrity import TrialEvalPublicIntegrityPolicyV1
from trialagentbench.contracts.release.trialeval_runtime_surface import (
    TRIALEVAL_PARTICIPANT_TEXT_FILENAMES,
    JsonObject,
    JsonValue,
    TrialEvalParticipantTaskV1,
    TrialEvalSemanticSubmissionContractV1,
    canonical_trialeval_workspace_path,
    sanitize_trialeval_agent_analysis_plan_payload,
    sanitize_trialeval_agent_endpoint_definition_payload,
    sanitize_trialeval_agent_json_text,
    sanitize_trialeval_agent_task_payload,
    trialeval_agent_allows_item_member,
)
from trialagentbench.contracts.trialeval_diagnostics import require_installed_diagnostic_semantics_v1
from trialagentbench.io import canonical_payload_sha256
from trialagentbench.release import ParticipantTask, load_participant_release
from trialagentbench.trialeval.schema import BenchmarkItem


def _participant_member(participant_root: Path, path: Path) -> Path:
    """Return a resolved participant member after enforcing filesystem isolation."""

    root = Path(participant_root).absolute()
    candidate = Path(path).absolute()
    if root.is_symlink():
        raise ValueError(f"TrialEval participant path must not be a symlink: {root}")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"TrialEval participant path escapes its release root: {candidate}") from exc
    if ".." in relative.parts:
        raise ValueError(f"TrialEval participant path escapes its release root: {candidate}")

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"TrialEval participant path must not be a symlink: {current}")

    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"TrialEval participant path escapes its release root: {candidate}") from exc
    return resolved


def _validate_participant_tree(participant_root: Path, subtree: Path) -> Path:
    """Validate all existing files and directories below a participant subtree."""

    root = Path(participant_root)
    validated_subtree = _participant_member(root, subtree)
    if not validated_subtree.is_dir():
        raise NotADirectoryError(f"TrialEval participant directory is missing: {subtree}")
    for path in validated_subtree.rglob("*"):
        _participant_member(root, path)
    return validated_subtree


def _item_participant_root(item: BenchmarkItem) -> Path:
    """Return the participant isolation root associated with an item."""

    return item.suite_dir if item.suite_dir is not None else item.visible_dir


def discover_items(
    suite_dir: Path,
    *,
    design_tiers: list[str] | None = None,
    assumption_tiers: list[str] | None = None,
    context_tiers: list[str] | None = None,
    trial_names: list[str] | None = None,
) -> list[BenchmarkItem]:
    """Discover participant items using only the evaluator's compact item index."""
    index_path = suite_dir / "grader" / "item_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"TrialEval evaluator item index is missing: {index_path}")
    index = json.loads(index_path.read_text())
    if not isinstance(index, dict) or not isinstance(index.get("entries"), list):
        raise ValueError("TrialEval evaluator item index requires an entries array.")
    if not index["entries"]:
        raise ValueError("TrialEval evaluator item index cannot be empty.")

    items: list[BenchmarkItem] = []
    for entry in index["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("TrialEval evaluator item index entries must be objects.")
        task_id = entry["task_id"]
        item_name = entry["item_id"]
        base_case_id = entry["base_case_id"]
        variant_id = entry.get("variant_id", "")
        factor_payload = entry.get("factors")
        if not isinstance(factor_payload, dict):
            raise ValueError(f"TrialEval evaluator item index entry lacks factors: task_id={task_id!r}.")
        dt = factor_payload["design_archetype"]
        design_subtype = factor_payload["design_subtype"]
        at = factor_payload["assumption_regime"]
        ct = factor_payload["context_configuration"]
        factors = TrialEvalEvidenceFactorsV1.model_validate(
            {
                "context_configuration": ct,
                "data_preparation": factor_payload["data_preparation"],
                "analysis_specification": factor_payload["analysis_specification"],
            }
        )

        if trial_names and (base_case_id not in trial_names and item_name not in trial_names):
            continue
        if design_tiers and dt not in [d.upper() for d in design_tiers]:
            continue
        if assumption_tiers and at not in [a.upper() for a in assumption_tiers]:
            continue
        if context_tiers and ct not in [c.upper() for c in context_tiers]:
            continue

        # Evaluator bundles use public/items; a paired participant archive may
        # be extracted directly beside the grader root as items/.
        visible_pub = suite_dir / "public" / "items" / task_id
        visible_flat = suite_dir / "items" / task_id
        if visible_pub.is_dir():
            visible_dir = visible_pub
        elif visible_flat.is_dir():
            visible_dir = visible_flat
        else:
            raise FileNotFoundError(f"Indexed TrialEval participant item is missing: task_id={task_id!r}.")
        task_path = visible_dir / "task.json"
        if not task_path.is_file():
            raise FileNotFoundError(f"Indexed TrialEval task contract is missing: task_id={task_id!r}.")
        task = json.loads(task_path.read_text())
        validated_task = TrialEvalParticipantTaskV1.model_validate(task)
        if validated_task.design_subtype != design_subtype:
            raise ValueError(f"TrialEval task/index design_subtype mismatch: task_id={task_id!r}.")

        # Data directory — C3/C4/C5 have raw subdir; C1/C2 have flat data/
        data_dir = visible_dir / "data"
        raw_data_dir = data_dir / "raw"
        has_raw = raw_data_dir.exists() and any(raw_data_dir.glob("*.parquet"))

        sc_path = visible_dir / "submission_contract.json"
        validated_submission_contract = TrialEvalSemanticSubmissionContractV1.model_validate_json(
            sc_path.read_text(encoding="utf-8")
        ).validate_data_preparation(factors.data_preparation)
        if validated_submission_contract.task_id != task_id:
            raise ValueError(f"TrialEval output-contract/task identity mismatch: task_id={task_id!r}.")
        submission_contract = validated_submission_contract.model_dump(mode="json")

        reconstruction_task = None
        rt_path = visible_dir / "reconstruction_task.json"
        if rt_path.exists():
            reconstruction_task = json.loads(rt_path.read_text())

        items.append(
            BenchmarkItem(
                item_id=f"{item_name}__{ct}",
                trial_name=base_case_id,
                design_tier=dt,
                design_subtype=design_subtype,
                assumption_tier=at,
                context_tier=ct,
                visible_dir=visible_dir,
                data_dir=raw_data_dir if has_raw else data_dir,
                task=task,
                estimand_mode=validated_task.estimand_mode or "",
                data_preparation=factors.data_preparation,
                analysis_specification=factors.analysis_specification,
                data_version="trialagentbench_v1",
                submission_contract=submission_contract,
                reconstruction_task=reconstruction_task,
                raw_data_dir=raw_data_dir if has_raw else None,
                task_id=task_id,
                variant_id=variant_id,
                suite_dir=suite_dir,
            )
        )

    return items


def discover_participant_items(
    participant_root: Path,
    *,
    task_ids: tuple[str, ...] | None = None,
) -> dict[str, BenchmarkItem]:
    """Load scheduled items without reading or requiring evaluator artifacts."""

    root = Path(participant_root)
    if _is_canonical_participant_release(root):
        return _discover_canonical_participant_items(root, task_ids=task_ids)
    load_participant_diagnostic_dictionary(root)
    declared_task_ids, declared_factors = participant_task_factors(root)
    selected_task_ids = tuple(task_ids if task_ids is not None else declared_task_ids)
    if not selected_task_ids or len(selected_task_ids) != len(set(selected_task_ids)):
        raise ValueError("TrialEval participant task selection must be non-empty and unique.")
    missing = sorted(set(selected_task_ids).difference(declared_task_ids))
    if missing:
        raise ValueError(f"Task selection references tasks absent from the participant release: {missing!r}.")

    items: dict[str, BenchmarkItem] = {}
    _validate_participant_tree(root, root / "items")
    for task_id in sorted(selected_task_ids):
        factors = declared_factors[task_id]
        visible_dir = _participant_member(root, root / "items" / task_id)
        if not visible_dir.is_dir():
            raise FileNotFoundError(f"Scheduled participant item is missing: {visible_dir}")
        task_path = _participant_member(root, visible_dir / "task.json")
        task = json.loads(task_path.read_text(encoding="utf-8"))
        validated_task = TrialEvalParticipantTaskV1.model_validate(task)
        data_root = visible_dir / "data"
        raw_root = data_root / "raw"
        has_raw = raw_root.is_dir() and any(raw_root.glob("*.parquet"))
        reconstruction_path = visible_dir / "reconstruction_task.json"
        reconstruction_task = (
            json.loads(_participant_member(root, reconstruction_path).read_text(encoding="utf-8"))
            if reconstruction_path.is_file()
            else None
        )
        submission_contract_path = _participant_member(root, visible_dir / "submission_contract.json")
        validated_submission_contract = TrialEvalSemanticSubmissionContractV1.model_validate_json(
            submission_contract_path.read_text(encoding="utf-8")
        ).validate_data_preparation(factors.data_preparation)
        if validated_submission_contract.task_id != task_id:
            raise ValueError(f"TrialEval output-contract/task identity mismatch: task_id={task_id!r}.")
        submission_contract = validated_submission_contract.model_dump(mode="json")
        items[task_id] = BenchmarkItem(
            item_id=task_id,
            trial_name=task_id,
            design_tier="undisclosed",
            design_subtype=validated_task.design_subtype,
            assumption_tier="undisclosed",
            context_tier=factors.context_configuration,
            visible_dir=visible_dir,
            data_dir=raw_root if has_raw else data_root,
            task=task,
            estimand_mode=validated_task.estimand_mode or "",
            data_preparation=factors.data_preparation,
            analysis_specification=factors.analysis_specification,
            submission_contract=submission_contract,
            reconstruction_task=reconstruction_task,
            raw_data_dir=raw_root if has_raw else None,
            task_id=task_id,
            suite_dir=root,
        )
    return items


def participant_task_factors(
    participant_root: Path,
) -> tuple[list[str], dict[str, TrialEvalEvidenceFactorsV1]]:
    """Return public task order and explicit evidence factors."""

    root = Path(participant_root)
    if _is_canonical_participant_release(root):
        release = load_participant_release(root)
        factors = {task.task_id: canonical_task_factors(task) for task in release.tasks}
        return [task.task_id for task in release.tasks], factors
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError(f"TrialEval participant path must not be a symlink: {manifest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"TrialEval participant manifest is missing: {manifest_path}")
    manifest_path = _participant_member(root, manifest_path)
    manifest = TrialEvalParticipantManifestV1.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    return list(manifest.task_ids), dict(manifest.task_evidence_factors)


def _is_canonical_participant_release(root: Path) -> bool:
    """Return whether a release declares the canonical role manifest."""

    path = root / "manifest.json"
    if not path.is_file() or path.is_symlink():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return isinstance(payload, dict) and payload.get("schema_id") == "ariadne.benchmark_role_manifest/v1"


def canonical_task_factors(task: ParticipantTask) -> TrialEvalEvidenceFactorsV1:
    """Recover declared context factors from one canonical evidence surface."""

    identifiers = {material.artifact_id for material in task.available_materials}
    paths = tuple(material.relative_path for material in task.available_materials)
    has_analysis = "analysis_data" in identifiers and any(path.endswith("/analysis.parquet") for path in paths)
    has_raw = {"participant_records", "event_history"}.issubset(identifiers)
    has_integrity = "data_integrity_policy" in identifiers
    has_sap = "analysis_plan" in identifiers
    has_protocol = "protocol" in identifiers
    if has_analysis == has_raw or not has_protocol:
        raise ValueError(f"TrialEval task has an ambiguous evidence configuration: {task.task_id}")
    if has_integrity and (not has_raw or has_sap):
        raise ValueError(f"TrialEval integrity task has an incompatible evidence configuration: {task.task_id}")
    context = (
        "C5"
        if has_integrity
        else "C1" if has_analysis and has_sap else "C2" if has_analysis else "C3" if has_sap else "C4"
    )
    data_preparation = (
        "raw_domains_declared_defect" if has_integrity else "analysis_ready" if has_analysis else "raw_domains"
    )
    return TrialEvalEvidenceFactorsV1.model_validate(
        {
            "context_configuration": context,
            "data_preparation": data_preparation,
            "analysis_specification": "locked_sap" if has_sap else "protocol_only",
        }
    )


def _projected_canonical_task(task: ParticipantTask) -> dict[str, object]:
    """Return task metadata whose material paths match the mounted workspace."""

    payload = task.model_dump(mode="json")
    payload["available_materials"] = [
        {
            **material.model_dump(mode="json"),
            "relative_path": canonical_trialeval_workspace_path(material.relative_path),
        }
        for material in task.available_materials
    ]
    return payload


def _canonical_submission_contract(
    task_id: str,
    factors: TrialEvalEvidenceFactorsV1,
) -> TrialEvalSemanticSubmissionContractV1:
    """Create minimal response obligations implied by released evidence."""

    required = {"primary_analysis", "limitations"}
    if factors.data_preparation != "analysis_ready":
        required.add("reconstruction")
    payload: dict[str, object] = {
        "schema_id": "ariadne.trialeval_semantic_submission_contract/v1",
        "task_id": task_id,
        "submission_semantics_id": "ariadne.trialeval_submission/v1",
        "required_deliverables": sorted(required),
    }
    return TrialEvalSemanticSubmissionContractV1.model_validate(
        {**payload, "checksum": canonical_payload_sha256(cast(PydanticJsonValue, payload))}
    ).validate_data_preparation(factors.data_preparation)


def _discover_canonical_participant_items(
    root: Path,
    *,
    task_ids: tuple[str, ...] | None,
) -> dict[str, BenchmarkItem]:
    """Load canonical role-separated tasks through their release contract."""

    release = load_participant_release(root)
    task_by_id = {task.task_id: task for task in release.tasks}
    selected = tuple(task_ids) if task_ids is not None else tuple(task_by_id)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("TrialEval participant task selection must be non-empty and unique.")
    missing = sorted(set(selected).difference(task_by_id))
    if missing:
        raise ValueError(f"Task selection references tasks absent from the participant release: {missing!r}.")
    items: dict[str, BenchmarkItem] = {}
    for task_id in sorted(selected):
        task = task_by_id[task_id]
        factors = canonical_task_factors(task)
        data_materials = tuple(
            material for material in task.available_materials if "data" in PurePosixPath(material.relative_path).parts
        )
        data_parents = {(root / material.relative_path).parent.resolve(strict=True) for material in data_materials}
        if not data_parents:
            raise ValueError(f"TrialEval task has no data material: {task_id}")
        data_dir = sorted(data_parents, key=lambda path: path.as_posix())[0]
        raw_data_dir = data_dir if factors.data_preparation != "analysis_ready" else None
        contract = _canonical_submission_contract(task_id, factors)
        items[task_id] = BenchmarkItem(
            item_id=task_id,
            trial_name=task_id,
            design_tier="undisclosed",
            design_subtype="undisclosed",
            assumption_tier="undisclosed",
            context_tier=factors.context_configuration,
            visible_dir=root / "tasks" / task_id,
            data_dir=data_dir,
            task=_projected_canonical_task(task),
            data_preparation=factors.data_preparation,
            analysis_specification=factors.analysis_specification,
            submission_contract=contract.model_dump(mode="json"),
            reconstruction_task={} if factors.data_preparation != "analysis_ready" else None,
            raw_data_dir=raw_data_dir,
            workspace_data_paths=tuple(
                sorted(canonical_trialeval_workspace_path(material.relative_path) for material in data_materials)
            ),
            task_id=task_id,
            suite_dir=root,
        )
    return items


_PROTOCOL_EXCLUDED_FILES = frozenset({"analysis_plan.json", "analysis_tasks.md"})
_ON_DEMAND_PROMPT_FILES = frozenset({"data_dictionary.json"})


def load_visible_context(
    item: BenchmarkItem,
) -> str:
    """Assemble the immutable participant-visible context."""
    _validate_participant_tree(_item_participant_root(item), item.visible_dir)
    return _assemble_visible_context(item)


def stage_participant_evidence(
    item: BenchmarkItem,
    destination: Path,
) -> Path:
    """Materialize the exact sanitized evidence tree mounted for model code."""

    participant_root = _item_participant_root(item)
    _validate_participant_tree(participant_root, item.visible_dir)
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Participant evidence destination already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    members = _projected_participant_members(item)
    for relative, payload in members:
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    return target


def participant_evidence_artifacts(item: BenchmarkItem) -> tuple[ReplayInputArtifactV1, ...]:
    """Bind replay inputs to the exact projection of the authenticated release."""
    return tuple(
        ReplayInputArtifactV1(
            relative_path=relative, sha256=hashlib.sha256(payload).hexdigest(), byte_length=len(payload)
        )
        for relative, payload in _projected_participant_members(item)
    )


def participant_analysis_surface_sha256(
    item: BenchmarkItem,
) -> str:
    """Return the digest of the exact projected participant evidence surface."""

    return _analysis_surface_sha256(_projected_participant_members(item))


def _analysis_surface_sha256(members: tuple[tuple[str, bytes], ...]) -> str:
    """Hash one ordered projected participant evidence surface."""

    digest = hashlib.sha256()
    for relative, payload in members:
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _redact_task_json_for_agent(
    task: Mapping[str, object],
) -> dict[str, JsonValue]:
    """Return participant-safe task metadata without changing its scientific contract."""

    return dict(
        sanitize_trialeval_agent_task_payload(
            cast(JsonObject, task),
        )
    )


def _assemble_visible_context(
    item: BenchmarkItem,
) -> str:
    """Load the participant-visible context for an item.

    - submission_contract.json is included as the participant-facing output
      contract; release validation forbids grader lanes and credit-eligible methods.
    - Large reference documents remain mounted and listed by the system prompt
      but are read on demand instead of repeated in every provider request.
    - Context tiers differ through prepared versus source-domain data and
      reconstruction evidence, not by hiding the requested estimand.
    """
    participant_root = _item_participant_root(item)
    if _is_canonical_participant_release(participant_root):
        members = dict(_projected_canonical_members(item))
        document_order = (
            "analysis_request.md",
            "protocol.md",
            "statistical_analysis_plan.md",
            "data_specification.md",
            "data_integrity_policy.md",
            "report_instructions.md",
        )
        canonical_parts = [
            f"=== {name} ===\n{members[name].decode('utf-8')}" for name in document_order if name in members
        ]
        if "analysis_request.md" not in members:
            raise ValueError("Canonical TrialEval tasks require an analyst-facing analysis_request.md")
        return "\n\n".join(canonical_parts)

    parts: list[str] = []

    # Redacted task.json
    task_redacted = _redact_task_json_for_agent(item.task)
    parts.append("=== task.json ===\n" + json.dumps(task_redacted, indent=2))

    for name in TRIALEVAL_PARTICIPANT_TEXT_FILENAMES:
        if name == "task.json":
            continue
        if item.analysis_specification == "protocol_only" and name in _PROTOCOL_EXCLUDED_FILES:
            continue
        if name in _ON_DEMAND_PROMPT_FILES:
            continue
        p = item.visible_dir / name
        if not p.exists():
            continue
        p = _participant_member(_item_participant_root(item), p)
        txt = p.read_text()
        if name == "endpoint_definition.json":
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    obj = sanitize_trialeval_agent_endpoint_definition_payload(
                        obj,
                    )
                txt = json.dumps(obj, indent=2)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid endpoint_definition.json for public context assembly: {p}") from exc
        elif name == "analysis_plan.json":
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    obj = sanitize_trialeval_agent_analysis_plan_payload(
                        obj,
                    )
                txt = json.dumps(obj, indent=2)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid analysis_plan.json for public context assembly: {p}") from exc
        parts.append(f"=== {name} ===\n{txt}")

    return "\n\n".join(parts)


def _projected_participant_members(
    item: BenchmarkItem,
) -> tuple[tuple[str, bytes], ...]:
    """Return deterministic path/content pairs for one participant view."""

    participant_root = _item_participant_root(item)
    if _is_canonical_participant_release(participant_root):
        return _projected_canonical_members(item)
    source = _validate_participant_tree(participant_root, item.visible_dir)
    candidates = tuple(path for path in sorted(source.rglob("*")) if path.is_file())
    allowed_paths = frozenset(
        path.relative_to(source).as_posix()
        for path in candidates
        if trialeval_agent_allows_item_member(
            item_relative_path=path.relative_to(source).as_posix(),
            data_preparation=cast(TrialEvalDataPreparationV1, item.data_preparation),
        )
        and not (
            item.analysis_specification == "protocol_only"
            and path.relative_to(source).as_posix() in _PROTOCOL_EXCLUDED_FILES
        )
    )
    members: list[tuple[str, bytes]] = []
    for path in candidates:
        relative = path.relative_to(source).as_posix()
        if relative not in allowed_paths:
            continue
        resolved = _participant_member(participant_root, path)
        if resolved.suffix.lower() != ".json":
            payload = resolved.read_bytes()
        elif relative == "task.json":
            task_payload = json.loads(resolved.read_text(encoding="utf-8"))
            if not isinstance(task_payload, dict):
                raise ValueError(f"TrialEval task contract must be a JSON object: {resolved}")
            projected = _redact_task_json_for_agent(task_payload)
            payload = (json.dumps(projected, indent=2, sort_keys=True) + "\n").encode("utf-8")
        else:
            payload = sanitize_trialeval_agent_json_text(
                archive_name=relative,
                text=resolved.read_text(encoding="utf-8"),
                available_item_members=allowed_paths,
            ).encode("utf-8")
        members.append((relative, payload))
    return tuple(sorted(members, key=lambda member: member[0]))


def _projected_canonical_members(item: BenchmarkItem) -> tuple[tuple[str, bytes], ...]:
    """Return the task-specific projection of a verified canonical release."""

    root = _item_participant_root(item)
    release = load_participant_release(root)
    task_by_id = {task.task_id: task for task in release.tasks}
    task_id = item.task_id or item.item_id
    try:
        task = task_by_id[task_id]
    except KeyError as exc:
        raise ValueError(f"TrialEval task is absent from its canonical participant release: {task_id}") from exc
    return _projected_canonical_members_for_task(item, task)


def _projected_canonical_members_for_task(
    item: BenchmarkItem,
    task: ParticipantTask,
) -> tuple[tuple[str, bytes], ...]:
    """Project one task from an already verified canonical release."""

    root = _item_participant_root(item)
    task_id = item.task_id or item.item_id
    if task.task_id != task_id:
        raise ValueError(f"TrialEval task identity differs from its participant item: {task_id}")
    members: dict[str, bytes] = {}
    for material in task.available_materials:
        relative = canonical_trialeval_workspace_path(material.relative_path)
        if relative in members:
            raise ValueError(f"TrialEval task materials collide after workspace projection: {relative}")
        source = _participant_member(root, root / material.relative_path)
        if source.suffix.lower() == ".json":
            payload = sanitize_trialeval_agent_json_text(
                archive_name=relative,
                text=source.read_text(encoding="utf-8"),
                available_item_members=frozenset(
                    canonical_trialeval_workspace_path(row.relative_path) for row in task.available_materials
                ),
            ).encode("utf-8")
        else:
            payload = source.read_bytes()
        members[relative] = payload
    integrity_materials = tuple(
        material for material in task.available_materials if material.artifact_id == "data_integrity_policy"
    )
    if integrity_materials:
        if len(integrity_materials) != 1:
            raise ValueError(f"TrialEval task declares multiple data-integrity policies: {task_id}")
        policy_path = root / PurePosixPath(integrity_materials[0].relative_path).with_suffix(".json")
        policy = TrialEvalPublicIntegrityPolicyV1.model_validate_json(
            _participant_member(root, policy_path).read_text(encoding="utf-8")
        )
        if policy.task_id != task_id:
            raise ValueError("TrialEval data-integrity policy does not match the selected task")
        members["data_integrity_policy.json"] = (
            json.dumps(policy.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    return tuple(sorted(members.items()))


def _participant_root_contract_path(participant_root: Path, filename: str) -> Path:
    """Resolve one required public-root contract from either release layout."""

    root = Path(participant_root)
    candidates = (root / filename, root / "public" / filename)
    existing = tuple(path for path in candidates if path.is_file())
    if len(existing) != 1:
        raise FileNotFoundError(
            f"TrialEval participant release must contain exactly one {filename} at its public root."
        )
    return _participant_member(root, existing[0])


def load_participant_diagnostic_dictionary(
    participant_root: Path,
) -> tuple[Path, TrialEvalParticipantDiagnosticDictionaryV1]:
    """Load the task-general diagnostic dictionary from either release layout."""

    path = _participant_root_contract_path(participant_root, "diagnostic_dictionary.json")
    dictionary = TrialEvalParticipantDiagnosticDictionaryV1.model_validate_json(path.read_text(encoding="utf-8"))
    require_installed_diagnostic_semantics_v1(dictionary)
    return path, dictionary
