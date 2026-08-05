"""Contracts for TrialDevBench decision-surface artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.trialdev.trialdev_evaluation_target_register import (
    TrialDevEvaluationLaneV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_public_recoverability import (
    WITHHOLD_NOMINATION_TARGET_ID,
)
from trialagentbench_harness.contracts.trialdev.trialdev_recoverability import (
    TrialDevObjectiveIdV1,
    TrialDevPhaseIdV1,
)
from trialagentbench_harness.trialdev.route import is_trialdev_action_lane, trialdev_route_sort_key

TrialDevLaneKindV1 = Literal["action", "diagnostic", "numeric", "report"]
TrialDevTargetSemanticsV1 = Literal[
    "asset",
    "stop_advance",
    "final_recommendation",
    "design_consistency",
    "analysis_consistency",
    "safety_gate",
    "route_timing",
]
TrialDevMarginKindV1 = Literal["utility", "regret", "score", "not_applicable"]
TrialDevDecisionTargetResolutionV1 = Literal[
    "release_static",
    "submitted_method_public_evidence",
    "realized_public_evidence",
    "realized_trajectory",
]
TrialDevDecisionScoringRoleV1 = Literal["reference_static", "runtime_context"]


def _stable_checksum(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TrialDevDiagnosticReferenceRouteStepV1(BaseModel):
    """One phase-level step on a diagnostic-reference TrialDev route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase_id: TrialDevPhaseIdV1
    lane_id: TrialDevEvaluationLaneV1
    action_id: str = Field(..., min_length=1)
    asset_id: str | None = None
    utility: float | None = None
    regret: float | None = Field(default=None, ge=0.0)
    margin_to_next_best: float | None = Field(default=None, ge=0.0)


class TrialDevDecisionSurfaceRecordV1(BaseModel):
    """One scoreable decision-surface context for a TrialDev program."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_decision_surface_record_v1"] = "trialdev_decision_surface_record_v1"
    schema_version: Literal[1] = 1
    scenario_id: str = Field(..., min_length=1)
    scenario_key: str = Field(..., min_length=1)
    program_id: str = Field(..., min_length=1)
    objective_id: TrialDevObjectiveIdV1
    phase_id: TrialDevPhaseIdV1
    lane_id: TrialDevEvaluationLaneV1
    lane_kind: TrialDevLaneKindV1
    target_semantics: TrialDevTargetSemanticsV1
    decision_context_id: str = Field(..., min_length=1)
    diagnostic_reference_target_ids: tuple[str, ...] = Field(..., min_length=1)
    credit_eligible_target_ids: tuple[str, ...] = Field(default_factory=tuple)
    rejected_shortcut_ids: tuple[str, ...] = Field(default_factory=tuple)
    diagnostic_reference_asset_id: str | None = None
    diagnostic_reference_action_id: str | None = None
    diagnostic_reference_route_id: str = Field(..., min_length=1)
    target_resolution: TrialDevDecisionTargetResolutionV1 = "release_static"
    scoring_role: TrialDevDecisionScoringRoleV1 = "reference_static"
    margin_kind: TrialDevMarginKindV1 = "not_applicable"
    selected_route_margin: float | None = Field(default=None, ge=0.0)
    utility_payload: dict[str, JsonValue] = Field(default_factory=dict)
    public_evidence_basis: tuple[str, ...] = Field(..., min_length=1)
    evaluator_evidence_basis: tuple[str, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_record(self) -> TrialDevDecisionSurfaceRecordV1:
        """Validate target disjointness, evidence scope, and checksum."""

        reference = {str(value) for value in self.diagnostic_reference_target_ids}
        acceptable = {str(value) for value in self.credit_eligible_target_ids}
        overlap = sorted(reference & acceptable)
        if overlap:
            raise ValueError(f"diagnostic_reference_target_ids and credit_eligible_target_ids overlap: {overlap!r}")
        action_lane = is_trialdev_action_lane(str(self.lane_id))
        if action_lane and self.lane_kind != "action":
            raise ValueError(f"action-bearing lane {self.lane_id!r} must use lane_kind='action'.")
        if not action_lane and self.lane_kind == "action":
            raise ValueError(f"non-action lane {self.lane_id!r} cannot use lane_kind='action'.")
        runtime_target = self.target_resolution != "release_static"
        expected_runtime_target = {
            "submitted_method_public_evidence": "derived_from_submitted_method_public_evidence",
            "realized_public_evidence": "derived_from_realized_public_evidence",
            "realized_trajectory": "derived_from_realized_trajectory",
        }.get(self.target_resolution)
        if expected_runtime_target is not None and (reference != {expected_runtime_target} or acceptable):
            raise ValueError(
                "runtime-resolved contexts require their sole declared derivation target and no static alternatives."
            )
        if runtime_target and self.scoring_role != "runtime_context":
            raise ValueError("runtime-resolved targets must use scoring_role='runtime_context'.")
        if not runtime_target and self.scoring_role != "reference_static":
            raise ValueError("release-static targets must use scoring_role='reference_static'.")
        if self.target_resolution == "realized_public_evidence" and self.lane_id not in {
            "decision_action",
            "safety_gate",
            "route_timing",
        }:
            raise ValueError(
                "realized_public_evidence is restricted to decision-action, safety-gate, and route-timing lanes."
            )
        if self.target_resolution == "realized_trajectory" and self.lane_id not in {
            "route_timing",
            "final_recommendation",
        }:
            raise ValueError("realized_trajectory is restricted to route-timing and final-recommendation lanes.")
        if action_lane and not runtime_target and not self.diagnostic_reference_action_id:
            raise ValueError(f"action-bearing lane {self.lane_id!r} requires diagnostic_reference_action_id.")
        if runtime_target and self.diagnostic_reference_action_id is not None:
            raise ValueError("runtime-resolved action contexts cannot declare a diagnostic_reference_action_id.")
        if not action_lane and self.diagnostic_reference_action_id is not None:
            raise ValueError(f"non-action lane {self.lane_id!r} cannot declare diagnostic_reference_action_id.")
        if self.margin_kind == "not_applicable":
            if self.selected_route_margin is not None:
                raise ValueError("not_applicable margin records cannot declare selected_route_margin.")
        elif self.selected_route_margin is None:
            raise ValueError("scoreable margin records require selected_route_margin.")
        if WITHHOLD_NOMINATION_TARGET_ID in reference:
            if self.diagnostic_reference_asset_id is not None:
                raise ValueError("A no-nomination action cannot identify an asset.")
            if self.diagnostic_reference_action_id != WITHHOLD_NOMINATION_TARGET_ID:
                raise ValueError("A static no-nomination target must retain its action identity.")
            if self.margin_kind != "not_applicable":
                raise ValueError("A no-nomination action cannot declare a candidate utility margin.")
        for path in self.public_evidence_basis:
            parts = set(str(path).split("/"))
            if parts & {"hidden", "grader"}:
                raise ValueError(f"public evidence basis cannot reference evaluator-only material: {path}")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        expected_checksum = _stable_checksum(payload)
        if self.checksum is None:
            object.__setattr__(self, "checksum", expected_checksum)
        elif self.checksum != expected_checksum:
            raise ValueError("checksum does not match decision-surface record payload.")
        return self


class TrialDevDiagnosticReferenceRouteRecordV1(BaseModel):
    """One diagnostic-reference route through a TrialDev program."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_diagnostic_reference_route_record_v1"] = (
        "trialdev_diagnostic_reference_route_record_v1"
    )
    schema_version: Literal[1] = 1
    scoring_role: Literal["diagnostic_reference"] = "diagnostic_reference"
    scenario_id: str = Field(..., min_length=1)
    scenario_key: str = Field(..., min_length=1)
    program_id: str = Field(..., min_length=1)
    objective_id: TrialDevObjectiveIdV1
    route_steps: tuple[TrialDevDiagnosticReferenceRouteStepV1, ...] = Field(..., min_length=1)
    terminal_action_id: str = Field(..., min_length=1)
    terminal_recommendation_target_id: str | None = None
    terminal_asset_id: str | None = None
    total_utility: float
    regret_tolerance: float = Field(..., ge=0.0)
    public_evidence_basis: tuple[str, ...] = Field(..., min_length=1)
    evaluator_evidence_basis: tuple[str, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_record(self) -> TrialDevDiagnosticReferenceRouteRecordV1:
        """Validate public evidence scope and checksum."""

        previous_key: tuple[int, int] | None = None
        final_step_count = 0
        has_observational_review = False
        for step in self.route_steps:
            key = trialdev_route_sort_key(str(step.phase_id), str(step.lane_id))
            if previous_key is not None and key <= previous_key:
                raise ValueError("diagnostic-reference route steps must follow declared TrialDev phase/lane order.")
            previous_key = key
            if step.phase_id == "observational_review":
                has_observational_review = True
            if step.phase_id == "final_decision":
                final_step_count += 1
        if not has_observational_review:
            raise ValueError("diagnostic-reference route must contain an observational_review step.")
        if final_step_count != 1:
            raise ValueError("diagnostic-reference route must contain exactly one final_decision step.")
        final_step = self.route_steps[-1]
        if final_step.phase_id != "final_decision":
            raise ValueError("diagnostic-reference route must end in final_decision.")
        if self.terminal_action_id != final_step.action_id:
            raise ValueError("terminal_action_id must match the final route step action_id.")
        for path in self.public_evidence_basis:
            parts = set(str(path).split("/"))
            if parts & {"hidden", "grader"}:
                raise ValueError(f"public evidence basis cannot reference evaluator-only material: {path}")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        expected_checksum = _stable_checksum(payload)
        if self.checksum is None:
            object.__setattr__(self, "checksum", expected_checksum)
        elif self.checksum != expected_checksum:
            raise ValueError("checksum does not match diagnostic-reference-route record payload.")
        return self


class TrialDevUtilitySensitivityProfileV1(BaseModel):
    """One declared TrialDev utility-weight sensitivity profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_utility_sensitivity_profile_v1"] = "trialdev_utility_sensitivity_profile_v1"
    schema_version: Literal[1] = 1
    profile_id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    objective_id: TrialDevObjectiveIdV1
    weights: dict[str, float] = Field(..., min_length=1)
    official: bool = False
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_profile(self) -> TrialDevUtilitySensitivityProfileV1:
        """Validate positive utility weights and checksum."""

        if any(float(value) < 0.0 for value in self.weights.values()):
            raise ValueError("utility sensitivity weights must be non-negative.")
        if sum(float(value) for value in self.weights.values()) <= 0.0:
            raise ValueError("utility sensitivity weights must have positive total weight.")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        expected_checksum = _stable_checksum(payload)
        if self.checksum is None:
            object.__setattr__(self, "checksum", expected_checksum)
        elif self.checksum != expected_checksum:
            raise ValueError("checksum does not match utility sensitivity profile payload.")
        return self


class TrialDevDecisionSurfaceManifestV1(BaseModel):
    """Manifest for decision-surface artifacts in one TrialDev release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_decision_surface_manifest_v1"] = "trialdev_decision_surface_manifest_v1"
    schema_version: Literal[1] = 1
    release_id: str = Field(..., min_length=1)
    scenario_count: int = Field(..., ge=1)
    decision_surface_record_count: int = Field(..., ge=1)
    diagnostic_reference_route_record_count: int = Field(..., ge=1)
    sensitivity_profile_count: int = Field(..., ge=1)
    diagnostic_reference_route_scoring_role: Literal["diagnostic_reference"] = "diagnostic_reference"
    source_checksums: dict[str, str] = Field(default_factory=dict)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_manifest(self) -> TrialDevDecisionSurfaceManifestV1:
        """Assign the manifest checksum."""

        payload = self.model_dump(mode="json", exclude={"checksum"})
        expected_checksum = _stable_checksum(payload)
        if self.checksum is None:
            object.__setattr__(self, "checksum", expected_checksum)
        elif self.checksum != expected_checksum:
            raise ValueError("checksum does not match decision-surface manifest payload.")
        return self


__all__ = [
    "TrialDevDecisionSurfaceManifestV1",
    "TrialDevDecisionSurfaceRecordV1",
    "TrialDevDecisionScoringRoleV1",
    "TrialDevDecisionTargetResolutionV1",
    "TrialDevLaneKindV1",
    "TrialDevMarginKindV1",
    "TrialDevDiagnosticReferenceRouteRecordV1",
    "TrialDevDiagnosticReferenceRouteStepV1",
    "TrialDevTargetSemanticsV1",
    "TrialDevUtilitySensitivityProfileV1",
]
