"""Translation between agent JSON tool-call payloads and the upstream pydantic models.

The agent submits raw JSON via its tool-calling interface; the upstream
state machine expects strict, frozen pydantic models with checksum chains.
This module is the contract surface — every model that flows in either
direction passes through here for validation and checksum computation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from trialagentbench_harness.adapters.trialdev_share import (
    TrialDevelopmentPhaseActionPolicyV1,
    TrialDevelopmentPhaseAnalysisSubmissionV1,
    TrialDevelopmentPhaseDecisionSubmissionV1,
    TrialDevelopmentRequestV1,
    TrialDevelopmentTrialOutputManifestV1,
    sha256_file_hex,
)
from trialagentbench_harness.contracts.trialdev.programme import TRIALDEV_PROGRAMME_STATE_ADAPTER_V1
from trialagentbench_harness.trialdev.participant_submission import (
    build_phase_analysis_v1,
    build_phase_decision_v1,
    build_phase_request_v1,
)

# ---------------------------------------------------------------------------
# Validation helpers — return (model, error_msg) so the agent loop can re-prompt
# ---------------------------------------------------------------------------


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def parse_request(
    payload: dict[str, Any],
    *,
    scenario_id: str,
    phase_id: str,
) -> tuple[TrialDevelopmentRequestV1 | None, str | None]:
    """Validate an analyst's design and bind it to the current phase."""
    payload = dict(payload)  # don't mutate caller's dict
    # Strip our agent-only rationale field — upstream pydantic forbids extras.
    payload.pop("request_rationale", None)
    try:
        request = build_phase_request_v1(payload, scenario_id=scenario_id, phase_id=phase_id)
    except (ValidationError, ValueError) as exc:
        if not isinstance(exc, ValidationError):
            return None, f"Submission validation failed: {exc}"
        return None, _format_validation_error(exc)
    return request, None


def parse_phase_analysis(
    payload: dict[str, Any],
    *,
    scenario_id: str,
    phase_id: str,
    request_checksum: str,
    trial_output_checksum: str,
    effect_source_artifact_checksums: Mapping[str, str],
    safety_source_artifact_checksums: Mapping[str, str],
) -> tuple[TrialDevelopmentPhaseAnalysisSubmissionV1 | None, str | None]:
    payload = dict(payload)
    payload.pop("analysis_rationale", None)  # agent-only field; upstream forbids extras
    try:
        analysis = build_phase_analysis_v1(
            payload,
            scenario_id=scenario_id,
            phase_id=phase_id,
            request_checksum=request_checksum,
            trial_output_checksum=trial_output_checksum,
            effect_source_artifact_checksums=effect_source_artifact_checksums,
            safety_source_artifact_checksums=safety_source_artifact_checksums,
        )
    except (ValidationError, ValueError) as exc:
        if not isinstance(exc, ValidationError):
            return None, f"Submission validation failed: {exc}"
        return None, _format_validation_error(exc)
    return analysis, None


def write_rationale_sidecar(
    *,
    phase_dir: Path,
    request_payload: dict[str, Any] | None = None,
    analysis_payload: dict[str, Any] | None = None,
) -> None:
    """Persist agent-only rationale fields next to submission files.

    The function does not write a sidecar when neither payload has rationale
    text.
    """
    out: dict[str, Any] = {}
    if request_payload and request_payload.get("request_rationale"):
        out["request_rationale"] = str(request_payload["request_rationale"])
    if analysis_payload and analysis_payload.get("analysis_rationale"):
        out["analysis_rationale"] = str(analysis_payload["analysis_rationale"])
    if not out:
        return
    p = Path(phase_dir) / "agent_rationale.json"
    existing = {}
    if p.is_file():
        try:
            existing = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid rationale sidecar JSON: {p}") from exc
    existing.update(out)
    p.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


def parse_phase_decision(
    payload: dict[str, Any],
    *,
    action_policy: TrialDevelopmentPhaseActionPolicyV1 | None = None,
    scenario_id: str,
    phase_id: str,
    request_checksum: str,
    analysis_checksum: str,
) -> tuple[TrialDevelopmentPhaseDecisionSubmissionV1 | None, str | None]:
    """Validate an agent decision against its scenario-and-phase action policy."""
    try:
        decision = build_phase_decision_v1(
            payload,
            scenario_id=scenario_id,
            phase_id=phase_id,
            request_checksum=request_checksum,
            analysis_checksum=analysis_checksum,
        )
    except (ValidationError, ValueError) as exc:
        if not isinstance(exc, ValidationError):
            return None, f"Submission validation failed: {exc}"
        return None, _format_validation_error(exc)
    if action_policy is not None:
        err = _check_action_policy(decision, action_policy)
        if err is not None:
            return None, err
    return decision, None


def _check_action_policy(
    decision: TrialDevelopmentPhaseDecisionSubmissionV1,
    action_policy: TrialDevelopmentPhaseActionPolicyV1,
) -> str | None:
    """Reject decisions whose action isn't allowed by the scenario's policy."""
    try:
        spec = action_policy.action_spec(str(decision.phase_id))
    except ValueError:
        return (
            "Schema validation failed:\n"
            f"  - phase_id={decision.phase_id!r} is not present in this scenario's "
            f"phase_action_policy. Submitting a decision for an unreachable phase."
        )
    allowed = list(spec.allowed_action_ids)
    if decision.decision_action not in allowed:
        return (
            "Schema validation failed:\n"
            f"  - decision_action={decision.decision_action!r} is not legal for "
            f"phase_id={decision.phase_id!r}. Allowed: {allowed}."
        )
    requires_drug = list(spec.requires_candidate_drug_id)
    if decision.decision_action in requires_drug and not decision.candidate_drug_id:
        return (
            "Schema validation failed:\n"
            f"  - decision_action={decision.decision_action!r} requires candidate_drug_id "
            f"to be set."
        )
    return None


def load_action_policy(scenario_root: Path) -> TrialDevelopmentPhaseActionPolicyV1:
    """Read ``phase_action_policy.json`` from a scenario's ``public/`` dir."""
    path = Path(scenario_root) / "public" / "phase_action_policy.json"
    return TrialDevelopmentPhaseActionPolicyV1.model_validate(_read_json_object(path))


def _format_validation_error(exc: ValidationError) -> str:
    """Compress pydantic errors into agent-friendly text."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "")
        typ = err.get("type", "")
        lines.append(f"  - {loc}: {msg} ({typ})")
    return "Schema validation failed:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def write_request(request: TrialDevelopmentRequestV1, path: Path) -> str:
    """Persist a request to disk and return the model's canonical checksum.

    The upstream state machine compares ``analysis.request_checksum`` against
    ``request.checksum()`` — which uses the model's own canonical
    ``model_dump(mode='json', exclude_none=True)`` *after* validator defaults
    have been applied (e.g. ``treatment_discontinuation_strategy`` defaulting to
    ``treatment_policy``). Using ``request.checksum()`` here keeps the
    chain consistent regardless of what fields the agent omitted.
    """
    payload = request.model_dump(mode="json", exclude_none=True)
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return str(request.checksum())


def write_phase_submission(
    submission: TrialDevelopmentPhaseAnalysisSubmissionV1 | TrialDevelopmentPhaseDecisionSubmissionV1,
    path: Path,
) -> str:
    """Persist a phase submission and return its file-hash checksum.

    We hash the file bytes (not the canonical payload) because the upstream
    state machine validates ``decision.analysis_checksum == sha256_file_hex(analysis_path)``.
    """
    payload = submission.model_dump(mode="json", exclude_none=True)
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return str(sha256_file_hex(path))


def read_trial_output_manifest(trial_output_root: Path) -> TrialDevelopmentTrialOutputManifestV1:
    payload = _read_json_object(trial_output_root / "trial_output_manifest.json")
    return TrialDevelopmentTrialOutputManifestV1.model_validate(payload)


def observational_source_artifact_checksums(
    scenario_root: Path,
    *,
    objective_id: str,
) -> dict[str, str]:
    """Return the exact participant-visible provenance map for an observational objective."""

    root = Path(scenario_root)
    charter = _read_json_object(root / "public" / "objective_charter.json")
    objectives = charter.get("objectives")
    if not isinstance(objectives, list):
        raise ValueError("objective_charter.json must declare an objectives array.")
    matching = [
        row for row in objectives if isinstance(row, dict) and str(row.get("objective_id")) == str(objective_id)
    ]
    if len(matching) != 1:
        raise ValueError(f"objective_charter.json requires one objective_id={objective_id!r}.")
    evidence_basis = matching[0].get("public_evidence_basis")
    if not isinstance(evidence_basis, list) or not evidence_basis:
        raise ValueError("The observational objective requires a non-empty public_evidence_basis.")
    paths = tuple(str(value) for value in evidence_basis)
    if len(paths) != len(set(paths)) or any(not path.startswith("public/") for path in paths):
        raise ValueError("Observational public_evidence_basis paths must be unique and scoped under public/.")
    checksums: dict[str, str] = {}
    for relative in paths:
        artifact = root / relative
        if not artifact.is_file():
            raise FileNotFoundError(f"Observational provenance artifact is missing: {artifact}")
        checksums[relative] = sha256_file_hex(artifact)
    return checksums


def observational_identification_artifact_checksums(scenario_root: Path) -> dict[str, str]:
    """Return exact hashes accepted for observational identification evidence."""

    root = Path(scenario_root)
    paths = (
        "public/observational_extract.parquet",
        "public/observational_method_catalog.json",
    )
    checksums: dict[str, str] = {}
    for relative in paths:
        artifact = root / relative
        if not artifact.is_file():
            raise FileNotFoundError(f"Observational identification artifact is missing: {artifact}")
        checksums[relative] = sha256_file_hex(artifact)
    return checksums


# ---------------------------------------------------------------------------
# Agent-readable summaries (no hidden data leaks)
# ---------------------------------------------------------------------------


def summarize_trial_output_for_agent(trial_output_root: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Return a JSON-friendly summary of a materialized trial bundle.

    The summary references parquet paths (the agent reads them via
    ``inspect_parquet`` / ``execute_code``), the public phase-summary file,
    and arm-mapping metadata — but *not* the audit, world spec, or any other
    surface that could leak counterfactual or oracle information.

    If ``relative_to`` is supplied (typically the agent's CWD), a
    ``trial_output_relpath`` field is included so the agent can use a
    short relative path instead of the absolute one.
    """
    manifest = read_trial_output_manifest(trial_output_root)
    arm_mapping = _read_json_object(trial_output_root / "arm_mapping.json")
    phase_summary = _read_json_object(trial_output_root / "phase_summary_public.json")
    summary: dict[str, Any] = {
        "trial_output_checksum": str(manifest.checksum),
        "request_checksum": str(manifest.request_checksum),
        "evidence_request_checksum": str(manifest.evidence_request_checksum),
        "n_participants": int(manifest.n_participants),
        "table_files": list(manifest.table_files),
        "table_checksums": dict(manifest.table_checksums),
        "arm_mapping": arm_mapping,
        "phase_summary_public": phase_summary,
    }
    if relative_to is not None:
        resolved_output = Path(trial_output_root).resolve()
        resolved_base = Path(relative_to).resolve()
        if resolved_output.is_relative_to(resolved_base):
            summary["trial_output_relpath"] = str(resolved_output.relative_to(resolved_base))
        public_policy_files = (
            "phase_action_policy.json",
            "phase_decision_evidence_policy.json",
            "safety_decision_policy.json",
        )
        effect_output_files = ("arm_mapping.json", "endpoints.parquet", "request.json")
        effect_source_checksums = {
            f"trial_output/{name}": sha256_file_hex(resolved_output / name) for name in effect_output_files
        }
        summary["effect_source_artifact_checksums"] = effect_source_checksums
        summary["safety_source_artifact_checksums"] = {
            **{f"public/{name}": sha256_file_hex(resolved_base / name) for name in public_policy_files},
            **effect_source_checksums,
            "trial_output/safety.parquet": sha256_file_hex(resolved_output / "safety.parquet"),
        }
    return summary


def summarize_program_state_for_agent(state_path: Path) -> dict[str, Any]:
    """Return the participant-visible programme state needed for the next design."""
    state = TRIALDEV_PROGRAMME_STATE_ADAPTER_V1.validate_python(_read_json_object(state_path))
    return {
        "programme_id": state.programme_id,
        "stream_id": state.stream_id,
        "current_checkpoint_id": state.current_checkpoint_id,
        "terminal_disposition": state.terminal_disposition,
        "active_asset_id": state.active_asset_id,
        "retired_asset_ids": list(state.retired_asset_ids),
        "completed_checkpoint_ids": [entry.checkpoint_id for entry in state.history],
        "state_index": len(state.history),
    }


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def derive_phase_seed(master_seed: int, program_id: str, phase_id: str) -> int:
    """Deterministic per-(master_seed, program, phase) materialization seed."""
    digest = hashlib.sha256(f"{int(master_seed)}|{program_id}|{phase_id}".encode()).hexdigest()
    return int(digest[:8], 16)


__all__ = [
    "parse_request",
    "parse_phase_analysis",
    "parse_phase_decision",
    "load_action_policy",
    "observational_identification_artifact_checksums",
    "observational_source_artifact_checksums",
    "write_request",
    "write_phase_submission",
    "write_rationale_sidecar",
    "read_trial_output_manifest",
    "summarize_trial_output_for_agent",
    "summarize_program_state_for_agent",
    "derive_phase_seed",
]
