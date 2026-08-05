"""Public safety decision policy helpers for TrialDevBench participants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trialagentbench_harness.trialdev.share.hashing import compute_sha256_hex
from trialagentbench_harness.trialdev.share.io import read_json
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentSeriousEventDefinitionV1


def load_safety_policy_v1(*, scenario_root: Path) -> dict[str, Any]:
    """Load and checksum-validate the public safety policy for one scenario."""

    payload = read_json(Path(scenario_root) / "public" / "safety_decision_policy.json")
    if not isinstance(payload, dict):
        raise ValueError("safety_decision_policy.json must be a JSON object.")
    if payload.get("schema_id") != "trialdev_safety_decision_policy_v1":
        raise ValueError("safety_decision_policy.json has unsupported schema_id.")
    checksum = payload.get("checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("safety_decision_policy.json requires a SHA-256 checksum.")
    checksum_payload = dict(payload)
    checksum_payload.pop("checksum", None)
    if compute_sha256_hex(checksum_payload) != checksum:
        raise ValueError("safety_decision_policy.json checksum mismatch.")
    return payload


def serious_event_definitions_v1(*, scenario_root: Path) -> tuple[TrialDevelopmentSeriousEventDefinitionV1, ...]:
    """Return unique, typed adverse-event column identities from public policy."""

    policy = load_safety_policy_v1(scenario_root=Path(scenario_root))
    raw_definitions = policy.get("serious_event_definitions")
    if not isinstance(raw_definitions, list) or not raw_definitions:
        raise ValueError("safety_decision_policy.json requires serious_event_definitions.")
    definitions = tuple(
        TrialDevelopmentSeriousEventDefinitionV1.model_validate(definition) for definition in raw_definitions
    )
    endpoint_ids = tuple(definition.endpoint_id for definition in definitions)
    columns = tuple(
        column
        for definition in definitions
        for column in (
            definition.event_column,
            definition.time_column,
            definition.seriousness_column,
            definition.severity_column,
        )
    )
    if len(set(endpoint_ids)) != len(endpoint_ids) or len(set(columns)) != len(columns):
        raise ValueError("Serious-event definitions require unique endpoint and column identities.")
    return definitions


def safety_thresholds_for_phase_v1(*, scenario_root: Path, phase_id: str) -> tuple[dict[str, Any], ...]:
    """Return public safety thresholds for one phase."""

    policy = load_safety_policy_v1(scenario_root=Path(scenario_root))
    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, list):
        raise ValueError("safety_decision_policy.thresholds must be an array.")
    return tuple(item for item in thresholds if isinstance(item, dict) and str(item.get("phase_id")) == str(phase_id))


__all__ = [
    "load_safety_policy_v1",
    "safety_thresholds_for_phase_v1",
    "serious_event_definitions_v1",
]
