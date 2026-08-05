"""Build custody-bound TrialDev records from analyst-facing submissions."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

from pydantic import BaseModel

from trialagentbench_harness.adapters.trialdev_share import (
    TrialDevelopmentObservationalReviewSubmissionV1,
    TrialDevelopmentPhaseAnalysisSubmissionV1,
    TrialDevelopmentPhaseDecisionSubmissionV1,
    TrialDevelopmentRequestV1,
)
from trialagentbench_harness.contracts.trialdev.portfolio_submission import (
    TrialDevPortfolioCheckpointSubmissionV1,
)
from trialagentbench_harness.contracts.trialdev.programme import TrialDevPortfolioProgrammeStateV1

_MECHANICAL_FIELDS = frozenset(
    {
        "checksum",
        "schema_id",
        "state_checksum",
        "request_checksum",
        "trial_output_checksum",
        "analysis_checksum",
        "source_artifact_checksums",
        "public_artifact_sha256",
        "evidence_reference_checksums",
        "identification_evidence_reference_checksums",
        "proposed_trial_plan_checksum",
        "generation_seed",
        "source_family_id",
        "world_id",
    }
)


def _is_mechanical_field(field_name: str) -> bool:
    """Return whether a field is owned by runtime custody rather than the analyst."""

    return (
        field_name in _MECHANICAL_FIELDS
        or field_name.endswith("_checksum")
        or field_name.endswith("_checksums")
        or field_name.endswith("_sha256")
    )


def participant_payload_v1(
    model: BaseModel,
    *,
    root_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Project a current programme record onto participant-owned content."""

    def project(value: object) -> object:
        if isinstance(value, dict):
            return {key: project(child) for key, child in value.items() if not _is_mechanical_field(key)}
        if isinstance(value, list):
            return [project(child) for child in value]
        return value

    projected = project(model.model_dump(mode="json", exclude_none=True))
    if not isinstance(projected, dict):
        raise TypeError("Participant projection requires an object-valued model.")
    for field_name in root_fields:
        projected.pop(field_name, None)
    return cast(dict[str, Any], projected)


def _drop_schema_fields(schema: dict[str, Any], field_names: frozenset[str]) -> None:
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for field_name in tuple(properties):
            if not (_is_mechanical_field(field_name) or field_name in field_names):
                continue
            properties.pop(field_name, None)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [field for field in required if not (_is_mechanical_field(field) or field in field_names)]
    discriminator = schema.get("discriminator")
    if isinstance(discriminator, dict) and discriminator.get("propertyName") in field_names:
        schema.pop("discriminator")
    for value in schema.values():
        if isinstance(value, dict):
            _drop_schema_fields(value, field_names)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _drop_schema_fields(item, field_names)


def participant_schema_v1(
    model: type[BaseModel] | Mapping[str, Any],
    *,
    root_fields: frozenset[str] = frozenset(),
    definition_fields: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Project one internal record schema onto analyst-owned fields."""

    schema = copy.deepcopy(model.model_json_schema() if isinstance(model, type) else dict(model))
    _drop_schema_fields(schema, _MECHANICAL_FIELDS)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for field_name in root_fields:
            properties.pop(field_name, None)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [field for field in required if field not in root_fields]
    definitions = schema.get("$defs")
    if isinstance(definitions, dict):
        for definition_name, fields in (definition_fields or {}).items():
            definition = definitions.get(definition_name)
            if isinstance(definition, dict):
                properties = definition.get("properties")
                if isinstance(properties, dict):
                    for field_name in fields:
                        properties.pop(field_name, None)
                required = definition.get("required")
                if isinstance(required, list):
                    definition["required"] = [field for field in required if field not in fields]
    return cast(dict[str, Any], schema)


def _copy_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(dict(payload))
    _reject_mechanical_fields(copied)
    return copied


def _reject_mechanical_fields(value: object, *, location: str = "submission") -> None:
    if isinstance(value, dict):
        present = sorted(_MECHANICAL_FIELDS.intersection(value))
        if present:
            raise ValueError(f"{location} contains harness-owned fields: {present!r}.")
        for key, child in value.items():
            _reject_mechanical_fields(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_mechanical_fields(child, location=f"{location}[{index}]")


def build_phase_request_v1(
    payload: Mapping[str, Any],
    *,
    scenario_id: str,
    phase_id: str,
) -> TrialDevelopmentRequestV1:
    """Validate an analyst's proposed design and bind its current phase."""

    enriched = _copy_payload(payload)
    for field in ("scenario_id", "phase_id", "version"):
        if field in enriched:
            raise ValueError(f"{field} is supplied by the current programme state.")
    enriched.update(version="v1", scenario_id=scenario_id, phase_id=phase_id)
    return TrialDevelopmentRequestV1.model_validate(enriched)


def build_observational_review_v1(
    payload: Mapping[str, Any],
    *,
    source_artifact_checksums: Mapping[str, str],
    identification_artifact_checksums: Mapping[str, str],
) -> TrialDevelopmentObservationalReviewSubmissionV1:
    """Validate an observational review and attach immutable source custody."""

    enriched = _copy_payload(payload)
    for estimate in enriched.get("candidate_utility_estimates", []):
        if not isinstance(estimate, dict):
            raise ValueError("candidate_utility_estimates must contain JSON objects.")
        estimate["source_artifact_checksums"] = dict(source_artifact_checksums)
    for evidence in enriched.get("identification_evidence", []):
        if not isinstance(evidence, dict):
            raise ValueError("identification_evidence must contain JSON objects.")
        path = str(evidence.get("public_artifact_path", ""))
        checksum = identification_artifact_checksums.get(path)
        if checksum is None:
            raise ValueError(f"Identification evidence does not name a current public artifact: {path!r}.")
        evidence["public_artifact_sha256"] = checksum
    return TrialDevelopmentObservationalReviewSubmissionV1.model_validate(enriched)


def build_phase_analysis_v1(
    payload: Mapping[str, Any],
    *,
    scenario_id: str,
    phase_id: str,
    request_checksum: str,
    trial_output_checksum: str,
    effect_source_artifact_checksums: Mapping[str, str],
    safety_source_artifact_checksums: Mapping[str, str],
) -> TrialDevelopmentPhaseAnalysisSubmissionV1:
    """Validate a randomized analysis and bind it to the completed study."""

    enriched = _copy_payload(payload)
    for field in ("scenario_id", "phase_id", "version"):
        if field in enriched:
            raise ValueError(f"{field} is supplied by the current programme state.")
    enriched.update(
        version="v1",
        scenario_id=scenario_id,
        phase_id=phase_id,
        request_checksum=request_checksum,
        trial_output_checksum=trial_output_checksum,
    )
    primary_effect = enriched.get("primary_effect")
    if isinstance(primary_effect, dict):
        primary_effect["source_artifact_checksums"] = dict(effect_source_artifact_checksums)
    safety_estimate = enriched.get("safety_estimate")
    if isinstance(safety_estimate, dict):
        safety_estimate["source_artifact_checksums"] = dict(safety_source_artifact_checksums)
    return TrialDevelopmentPhaseAnalysisSubmissionV1.model_validate(enriched)


def build_phase_decision_v1(
    payload: Mapping[str, Any],
    *,
    scenario_id: str,
    phase_id: str,
    request_checksum: str,
    analysis_checksum: str,
) -> TrialDevelopmentPhaseDecisionSubmissionV1:
    """Validate a phase decision and bind it to its analysis."""

    enriched = _copy_payload(payload)
    for field in ("scenario_id", "phase_id", "version"):
        if field in enriched:
            raise ValueError(f"{field} is supplied by the current programme state.")
    enriched.update(
        version="v1",
        scenario_id=scenario_id,
        phase_id=phase_id,
        request_checksum=request_checksum,
        analysis_checksum=analysis_checksum,
    )
    return TrialDevelopmentPhaseDecisionSubmissionV1.model_validate(enriched)


def portfolio_participant_schema_v1() -> dict[str, Any]:
    """Return the portfolio schema containing only analyst-owned fields."""

    schema = participant_schema_v1(
        TrialDevPortfolioCheckpointSubmissionV1,
        root_fields=frozenset({"state_checksum"}),
        definition_fields={
            "TrialDevPortfolioActionSelectionV1": frozenset(
                {"checkpoint_id", "analysis_method_id", "supporting_evidence_ids"}
            ),
        },
    )
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise ValueError("Portfolio participant schema requires model definitions.")
    for definition_name in (
        "TrialDevObservationalCandidateEvidenceV1",
        "TrialDevDecisionRuleEvidenceV1",
    ):
        definition = definitions.get(definition_name)
        if not isinstance(definition, dict) or not isinstance(definition.get("properties"), dict):
            raise ValueError(f"Portfolio participant schema is missing {definition_name}.")
        definition["properties"]["evidence_ids"] = {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "uniqueItems": True,
            "description": "Current evidence records used for this estimate or decision rule.",
        }
        definition.setdefault("required", []).append("evidence_ids")
    observational = definitions.get("TrialDevObservationalDecisionEvidenceV1")
    if not isinstance(observational, dict) or not isinstance(observational.get("properties"), dict):
        raise ValueError("Portfolio participant schema is missing observational decision evidence.")
    observational["properties"]["identification_evidence_ids"] = {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "uniqueItems": True,
        "default": [],
        "description": "Current evidence records supporting a conclusion that the comparison is not identified.",
    }
    return schema


def build_portfolio_checkpoint_v1(
    payload: Mapping[str, Any],
    *,
    state: TrialDevPortfolioProgrammeStateV1,
) -> TrialDevPortfolioCheckpointSubmissionV1:
    """Validate portfolio science and attach current evidence and state custody."""

    enriched = _copy_payload(payload)
    evidence = enriched.get("decision_evidence")
    action = enriched.get("selected_action")
    if not isinstance(evidence, dict) or not isinstance(action, dict):
        raise ValueError("decision_evidence and selected_action must be JSON objects.")
    _normalize_implicit_action_assets_v1(action=action, state=state)
    analysis_method_id = evidence.get("analysis_method_id")
    if not isinstance(analysis_method_id, str) or not analysis_method_id:
        raise ValueError("decision_evidence.analysis_method_id is required.")
    current_evidence = tuple(item for item in state.evidence if item.checkpoint_id == state.current_checkpoint_id)
    if not current_evidence:
        raise ValueError("The current programme state has no evidence for this checkpoint.")
    evidence_ids = tuple(sorted(item.evidence_id for item in current_evidence))
    evidence_by_id = {item.evidence_id: str(item.checksum) for item in current_evidence}

    def resolve_evidence_ids(record: dict[str, Any], field_name: str = "evidence_ids") -> tuple[str, ...]:
        identifiers = record.pop(field_name, None)
        if not isinstance(identifiers, list) or not identifiers:
            raise ValueError(f"{field_name} must name at least one current evidence record.")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{field_name} must be unique.")
        unknown = sorted(str(identifier) for identifier in identifiers if str(identifier) not in evidence_by_id)
        if unknown:
            raise ValueError(f"{field_name} names evidence outside the current checkpoint: {unknown!r}.")
        return tuple(sorted(evidence_by_id[str(identifier)] for identifier in identifiers))

    state_checksum = str(state.checksum)
    if state.current_checkpoint_id == "observational_review":
        evidence["schema_id"] = "trialagentbench.trialdev_observational_decision_evidence/v1"
        evidence["identification_evidence_reference_checksums"] = (
            resolve_evidence_ids(evidence, "identification_evidence_ids")
            if evidence.get("identification_status") == "not_identified"
            else ()
        )
        evidence.pop("identification_evidence_ids", None)
        for candidate in evidence.get("candidates", []):
            if not isinstance(candidate, dict):
                raise ValueError("decision_evidence.candidates must contain JSON objects.")
            candidate["evidence_reference_checksums"] = resolve_evidence_ids(candidate)
    else:
        evidence["schema_id"] = "trialagentbench.trialdev_randomized_decision_evidence/v1"
        for rule in evidence.get("rules", []):
            if not isinstance(rule, dict):
                raise ValueError("decision_evidence.rules must contain JSON objects.")
            rule["evidence_reference_checksums"] = resolve_evidence_ids(rule)
    evidence["state_checksum"] = state_checksum
    action.update(
        state_checksum=state_checksum,
        checkpoint_id=state.current_checkpoint_id,
        analysis_method_id=analysis_method_id,
        supporting_evidence_ids=evidence_ids,
    )
    enriched["state_checksum"] = state_checksum
    return TrialDevPortfolioCheckpointSubmissionV1.model_validate(enriched)


def _normalize_implicit_action_assets_v1(
    *,
    action: dict[str, Any],
    state: TrialDevPortfolioProgrammeStateV1,
) -> None:
    """Remove a redundant asset name when the selected action already implies it."""

    action_id = action.get("action_id")
    if not isinstance(action_id, str):
        return
    implied_asset = {
        "advance_lead_to_proof_of_concept": state.lead_asset_id,
        "promote_reserve_to_proof_of_concept": state.reserve_asset_id,
        "advance_active_to_confirmation": state.active_asset_id,
    }.get(action_id)
    if action_id not in {
        "advance_lead_to_proof_of_concept",
        "promote_reserve_to_proof_of_concept",
        "advance_active_to_confirmation",
    }:
        return
    if implied_asset is None:
        raise ValueError(f"{action_id} requires an asset assigned by the current programme state.")
    supplied_assets = {
        str(value)
        for field_name in ("target_asset_id", "reserve_asset_id")
        if (value := action.get(field_name)) is not None
    }
    if supplied_assets and supplied_assets != {implied_asset}:
        raise ValueError(f"{action_id} may name only the current implied asset {implied_asset!r}.")
    action.pop("target_asset_id", None)
    action.pop("reserve_asset_id", None)


__all__ = [
    "build_observational_review_v1",
    "build_phase_analysis_v1",
    "build_phase_decision_v1",
    "build_phase_request_v1",
    "build_portfolio_checkpoint_v1",
    "participant_schema_v1",
    "participant_payload_v1",
    "portfolio_participant_schema_v1",
]
