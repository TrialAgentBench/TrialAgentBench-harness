"""TrialDevBench recoverability contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.io.json import read_json_model

TrialDevPhaseIdV1 = Literal["observational_review", "phase1", "phase2", "phase3", "final_decision"]
TrialDevObjectiveIdV1 = Literal[
    "benefit_risk",
    "cost_effective_best",
    "net_clinical_value_under_budget",
    "pure_efficacy",
]
TrialDevRecoverabilityPolicyV1 = Literal[
    "unique_best",
    "acceptable_candidate_set",
    "near_tie_set",
    "insufficient_recoverability",
    "acceptable_action_set",
    "no_recoverability_relaxation",
]
TrialDevDecisionRecoverabilityClassV1 = Literal[
    "unique",
    "set_identified",
    "safety_determined",
]
TrialDevLaneRecoverabilityPolicyV1 = TrialDevRecoverabilityPolicyV1 | TrialDevDecisionRecoverabilityClassV1

TRIALDEV_OBJECTIVES_V1: tuple[TrialDevObjectiveIdV1, ...] = (
    "benefit_risk",
    "cost_effective_best",
    "net_clinical_value_under_budget",
    "pure_efficacy",
)
TRIALDEV_PHASES_V1: tuple[TrialDevPhaseIdV1, ...] = ("observational_review", "phase1", "phase2", "phase3")


def required_trialdev_recoverability_keys_v1() -> set[tuple[str, str]]:
    """Return required phase/objective recoverability contexts."""

    keys: set[tuple[str, str]] = set()
    for phase_id in TRIALDEV_PHASES_V1:
        objectives = ("benefit_risk",) if phase_id == "phase1" else TRIALDEV_OBJECTIVES_V1
        for objective_id in objectives:
            keys.add((phase_id, objective_id))
    return keys


def _stable_checksum(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TrialDevCandidateRecoverabilityRecordV1(BaseModel):
    """One candidate-level recoverability record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_drug_id: str = Field(..., min_length=1)
    method_route_id: str | None = Field(default=None, min_length=1)
    acceptable_candidate: bool
    released_data_reference_rank: int | None = Field(default=None, ge=1)
    released_data_reference_utility: float | None = None
    policy_reference_regret: float | None = None


class TrialDevRecoverabilityRecordV1(BaseModel):
    """One recoverability policy row for a TrialDev phase/objective context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_recoverability_policy_record_v1"] = "trialdev_recoverability_policy_record_v1"
    scenario_id: str = Field(..., min_length=1)
    phase_id: TrialDevPhaseIdV1
    objective_id: TrialDevObjectiveIdV1
    method_route_id: str | None = Field(default=None, min_length=1)
    policy: TrialDevRecoverabilityPolicyV1
    acceptable_candidate_set: tuple[str, ...] = Field(default_factory=tuple)
    acceptable_action_set: tuple[str, ...] = Field(default_factory=tuple)
    candidate_records: tuple[TrialDevCandidateRecoverabilityRecordV1, ...] = Field(default_factory=tuple)
    near_tie_threshold: float | None = Field(default=None, ge=0.0)
    public_evidence_basis: tuple[str, ...] = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_policy_payload(self) -> TrialDevRecoverabilityRecordV1:
        """Validate policy-specific payload constraints and assign checksum."""

        if (self.phase_id == "observational_review") != (self.method_route_id is not None):
            raise ValueError("method_route_id is required exactly for observational-review recoverability records.")
        candidate_set = tuple(sorted(set(str(value) for value in self.acceptable_candidate_set)))
        action_set = tuple(sorted(set(str(value) for value in self.acceptable_action_set)))
        object.__setattr__(self, "acceptable_candidate_set", candidate_set)
        object.__setattr__(self, "acceptable_action_set", action_set)
        if self.policy in {"no_recoverability_relaxation", "insufficient_recoverability"} and (
            candidate_set or action_set or self.candidate_records or self.near_tie_threshold is not None
        ):
            raise ValueError(f"{self.policy} records cannot declare acceptable alternatives.")
        if self.policy in {"acceptable_candidate_set", "near_tie_set", "unique_best"} and not candidate_set:
            raise ValueError(f"{self.policy} records require at least one acceptable candidate.")
        if self.policy == "acceptable_action_set" and not action_set:
            raise ValueError("acceptable_action_set records require at least one acceptable action.")
        if self.checksum is None:
            payload = self.model_dump(mode="json", exclude={"checksum"})
            object.__setattr__(self, "checksum", _stable_checksum(payload))
        return self


class TrialDevRecoverabilityManifestV1(BaseModel):
    """Recoverability manifest for one TrialDev scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_recoverability_manifest_v1"] = "trialdev_recoverability_manifest_v1"
    version: Literal["v1"] = "v1"
    scenario_id: str = Field(..., min_length=1)
    records: tuple[TrialDevRecoverabilityRecordV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_manifest(self) -> TrialDevRecoverabilityManifestV1:
        """Validate required coverage and assign a manifest checksum."""

        observed: set[tuple[str, str, str | None]] = set()
        for record in self.records:
            if record.scenario_id != self.scenario_id:
                raise ValueError("recoverability record scenario_id must match manifest scenario_id.")
            key = (str(record.phase_id), str(record.objective_id), record.method_route_id)
            if key in observed:
                raise ValueError(f"duplicate recoverability record for {key}.")
            observed.add(key)
        covered_contexts = {(phase_id, objective_id) for phase_id, objective_id, _ in observed}
        missing = sorted(required_trialdev_recoverability_keys_v1() - covered_contexts)
        if missing:
            raise ValueError(f"missing recoverability contexts: {missing!r}.")
        if self.checksum is None:
            payload = self.model_dump(mode="json", exclude={"checksum"})
            object.__setattr__(self, "checksum", _stable_checksum(payload))
        return self


def load_trialdev_recoverability_manifest(path: Path) -> TrialDevRecoverabilityManifestV1:
    """Load and validate a TrialDev recoverability manifest."""

    return read_json_model(TrialDevRecoverabilityManifestV1, Path(path))


__all__ = [
    "TRIALDEV_OBJECTIVES_V1",
    "TRIALDEV_PHASES_V1",
    "TrialDevCandidateRecoverabilityRecordV1",
    "TrialDevDecisionRecoverabilityClassV1",
    "TrialDevLaneRecoverabilityPolicyV1",
    "TrialDevObjectiveIdV1",
    "TrialDevPhaseIdV1",
    "TrialDevRecoverabilityManifestV1",
    "TrialDevRecoverabilityPolicyV1",
    "TrialDevRecoverabilityRecordV1",
    "load_trialdev_recoverability_manifest",
    "required_trialdev_recoverability_keys_v1",
]
