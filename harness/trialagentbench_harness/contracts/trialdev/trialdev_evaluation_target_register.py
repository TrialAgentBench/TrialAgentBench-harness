"""TrialDev trajectory evaluation-target register contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.trialdev.trialdev_recoverability import (
    TRIALDEV_OBJECTIVES_V1,
    TrialDevObjectiveIdV1,
    TrialDevPhaseIdV1,
    TrialDevRecoverabilityPolicyV1,
)
from trialagentbench_harness.io.json import read_json_model

METHOD_CONDITIONED_PUBLIC_EVIDENCE_TARGET_ID = "derived_from_submitted_method_public_evidence"

TrialDevEvaluationLaneV1 = Literal[
    "asset_nomination",
    "phase_design",
    "phase_analysis",
    "decision_action",
    "route_timing",
    "final_recommendation",
    "safety_gate",
]

TrialDevLaneScoreStatusV1 = Literal[
    "scored",
    "credit_eligible_alternative",
    "invalid_submission_zeroed",
    "missing_submission_zeroed",
    "not_applicable",
]


def required_trialdev_lanes_v1(phase_id: str) -> tuple[TrialDevEvaluationLaneV1, ...]:
    """Return the score-bearing lanes required at one reached checkpoint."""

    required: dict[str, tuple[TrialDevEvaluationLaneV1, ...]] = {
        "observational_review": ("asset_nomination", "phase_analysis"),
        "phase1": ("phase_design", "phase_analysis", "safety_gate", "decision_action"),
        "phase2": ("phase_design", "phase_analysis", "decision_action"),
        "phase3": ("phase_design", "phase_analysis", "decision_action"),
        "final_decision": ("route_timing", "final_recommendation"),
    }
    try:
        return required[str(phase_id)]
    except KeyError as exc:
        raise ValueError(f"Unsupported TrialDev checkpoint: {phase_id!r}.") from exc


def _stable_checksum(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TrialDevEvaluationTargetRegisterRecordV1(BaseModel):
    """One evaluation-target register row for a TrialDev trajectory scoring lane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_evaluation_target_register_record_v1"] = (
        "trialdev_evaluation_target_register_record_v1"
    )
    scenario_id: str = Field(..., min_length=1)
    phase_id: TrialDevPhaseIdV1
    program_objective_id: TrialDevObjectiveIdV1
    phase_scoring_objective_id: TrialDevObjectiveIdV1
    lane_id: TrialDevEvaluationLaneV1
    scoring_policy_id: str = Field(..., min_length=1)
    public_evidence_basis: tuple[str, ...] = Field(..., min_length=1)
    evaluator_evidence_basis: tuple[str, ...] = Field(..., min_length=1)
    reference_target_ids: tuple[str, ...] = Field(..., min_length=1)
    credit_eligible_target_ids: tuple[str, ...] = Field(default_factory=tuple)
    rejected_shortcut_ids: tuple[str, ...] = Field(default_factory=tuple)
    recoverability_policy_id: TrialDevRecoverabilityPolicyV1
    target_resolution: Literal[
        "release_static",
        "submitted_method_public_evidence",
        "realized_public_evidence",
        "realized_trajectory",
    ] = "release_static"
    value_payload: dict[str, JsonValue] = Field(default_factory=dict)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_record(self) -> TrialDevEvaluationTargetRegisterRecordV1:
        """Normalize tuple fields and assign checksum."""

        expected_runtime_target = {
            "submitted_method_public_evidence": METHOD_CONDITIONED_PUBLIC_EVIDENCE_TARGET_ID,
            "realized_public_evidence": "derived_from_realized_public_evidence",
            "realized_trajectory": "derived_from_realized_trajectory",
        }.get(self.target_resolution)
        if expected_runtime_target is not None and (
            set(self.reference_target_ids) != {expected_runtime_target} or self.credit_eligible_target_ids
        ):
            raise ValueError(
                "A runtime-resolved target requires one declared derivation target and no static alternatives."
            )
        if self.phase_id == "phase1" and self.phase_scoring_objective_id != "benefit_risk":
            raise ValueError(
                "phase1 evaluation-target register records must use benefit_risk as phase_scoring_objective_id."
            )
        if self.target_resolution == "realized_public_evidence" and self.lane_id not in {
            "decision_action",
            "safety_gate",
            "route_timing",
        }:
            raise ValueError(
                "realized_public_evidence is restricted to decision-action, safety-gate, and route-timing lanes."
            )
        if self.target_resolution == "submitted_method_public_evidence" and self.lane_id != "asset_nomination":
            raise ValueError("submitted_method_public_evidence is restricted to the asset-nomination lane.")
        if self.target_resolution == "realized_trajectory" and self.lane_id not in {
            "route_timing",
            "final_recommendation",
        }:
            raise ValueError(
                "realized_trajectory target resolution is restricted to route-timing and final-recommendation lanes."
            )
        for path in self.public_evidence_basis:
            parts = set(Path(path).parts)
            if parts & {"hidden", "grader"}:
                raise ValueError(f"public evidence basis cannot reference evaluator-only material: {path}")
        object.__setattr__(self, "reference_target_ids", tuple(sorted(set(self.reference_target_ids))))
        object.__setattr__(self, "credit_eligible_target_ids", tuple(sorted(set(self.credit_eligible_target_ids))))
        object.__setattr__(self, "rejected_shortcut_ids", tuple(sorted(set(self.rejected_shortcut_ids))))
        overlap = sorted(set(self.reference_target_ids) & set(self.credit_eligible_target_ids))
        if overlap:
            raise ValueError(f"reference_target_ids and credit_eligible_target_ids overlap: {overlap!r}")
        if self.checksum is None:
            payload = self.model_dump(mode="json", exclude={"checksum"})
            object.__setattr__(self, "checksum", _stable_checksum(payload))
        return self


class TrialDevEvaluationTargetRegisterManifestV1(BaseModel):
    """Manifest for one scenario's TrialDev evaluation-target register."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_evaluation_target_register_manifest_v1"] = (
        "trialdev_evaluation_target_register_manifest_v1"
    )
    version: Literal["v1"] = "v1"
    scenario_id: str = Field(..., min_length=1)
    row_count: int = Field(..., ge=1)
    register_jsonl_sha256: str = Field(..., min_length=64, max_length=64)
    required_context_count: int = Field(..., ge=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def assign_checksum(self) -> TrialDevEvaluationTargetRegisterManifestV1:
        """Assign a manifest checksum."""

        if self.checksum is None:
            payload = self.model_dump(mode="json", exclude={"checksum"})
            object.__setattr__(self, "checksum", _stable_checksum(payload))
        return self


class TrialDevEvaluationTargetRegisterGateReportV1(BaseModel):
    """Validation report for a TrialDev evaluation-target register."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialdev_evaluation_target_register_gate_report_v1"] = (
        "trialdev_evaluation_target_register_gate_report_v1"
    )
    version: Literal["v1"] = "v1"
    scenario_id: str = Field(..., min_length=1)
    passed: bool
    row_count: int = Field(..., ge=0)
    issue_count: int = Field(..., ge=0)
    issues: tuple[str, ...] = Field(default_factory=tuple)


TrialDevEvaluationContextV1 = tuple[
    TrialDevPhaseIdV1, TrialDevObjectiveIdV1, TrialDevObjectiveIdV1, TrialDevEvaluationLaneV1
]


def required_trialdev_evaluation_contexts_v1() -> set[TrialDevEvaluationContextV1]:
    """Return required scenario-local phase, objective, and scoring contexts."""

    contexts: set[TrialDevEvaluationContextV1] = set()
    for objective in TRIALDEV_OBJECTIVES_V1:
        contexts.add(("observational_review", objective, objective, "asset_nomination"))
        contexts.add(("observational_review", objective, objective, "phase_analysis"))
        contexts.add(("phase1", objective, "benefit_risk", "phase_design"))
        contexts.add(("phase1", objective, "benefit_risk", "phase_analysis"))
        contexts.add(("phase1", objective, "benefit_risk", "safety_gate"))
        contexts.add(("phase1", objective, "benefit_risk", "decision_action"))
        contexts.add(("phase2", objective, objective, "phase_design"))
        contexts.add(("phase2", objective, objective, "phase_analysis"))
        contexts.add(("phase2", objective, objective, "decision_action"))
        contexts.add(("phase3", objective, objective, "phase_design"))
        contexts.add(("phase3", objective, objective, "phase_analysis"))
        contexts.add(("phase3", objective, objective, "decision_action"))
        contexts.add(("final_decision", objective, objective, "route_timing"))
        contexts.add(("final_decision", objective, objective, "final_recommendation"))
    return contexts


def load_trialdev_evaluation_target_register_records(
    path: Path,
) -> tuple[TrialDevEvaluationTargetRegisterRecordV1, ...]:
    """Load evaluation-target register records from JSONL."""

    rows: list[TrialDevEvaluationTargetRegisterRecordV1] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict) or not isinstance(payload.get("checksum"), str):
                raise ValueError("serialized scoring row requires an explicit checksum")
            rows.append(TrialDevEvaluationTargetRegisterRecordV1.model_validate(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid evaluation-target register JSONL row {line_number}: {exc}") from exc
    return tuple(rows)


def validate_trialdev_evaluation_target_register(
    *,
    scenario_root: Path,
    records: tuple[TrialDevEvaluationTargetRegisterRecordV1, ...],
) -> tuple[str, ...]:
    """Return validation issues for a scenario evaluation-target register."""

    issues: list[str] = []
    observed: set[TrialDevEvaluationContextV1] = set()
    scenario_id = None
    for record in records:
        scenario_id = scenario_id or record.scenario_id
        if record.scenario_id != scenario_id:
            issues.append("all evaluation-target register records in a scenario file must share scenario_id")
        key = (
            record.phase_id,
            record.program_objective_id,
            record.phase_scoring_objective_id,
            record.lane_id,
        )
        if key in observed:
            issues.append(f"duplicate evaluation-target register context: {key!r}")
        observed.add(key)
        for public_path in record.public_evidence_basis:
            if not (scenario_root / public_path).exists():
                issues.append(f"missing public evidence path: {public_path}")
        for evidence_path in record.evaluator_evidence_basis:
            if not (scenario_root / evidence_path).exists():
                issues.append(f"missing evaluator evidence path: {evidence_path}")
    missing = sorted(required_trialdev_evaluation_contexts_v1() - observed)
    if missing:
        issues.append(f"missing required evaluation-target register contexts: {missing!r}")
    return tuple(issues)


def load_trialdev_evaluation_target_register_manifest(path: Path) -> TrialDevEvaluationTargetRegisterManifestV1:
    """Load and validate a TrialDev evaluation-target register manifest."""

    return cast(
        TrialDevEvaluationTargetRegisterManifestV1,
        read_json_model(TrialDevEvaluationTargetRegisterManifestV1, Path(path)),
    )


def load_trialdev_evaluation_target_register_gate_report(path: Path) -> TrialDevEvaluationTargetRegisterGateReportV1:
    """Load and validate a TrialDev evaluation-target register gate report."""

    return cast(
        TrialDevEvaluationTargetRegisterGateReportV1,
        read_json_model(TrialDevEvaluationTargetRegisterGateReportV1, Path(path)),
    )


__all__ = [
    "METHOD_CONDITIONED_PUBLIC_EVIDENCE_TARGET_ID",
    "TrialDevEvaluationLaneV1",
    "TrialDevLaneScoreStatusV1",
    "TrialDevEvaluationTargetRegisterGateReportV1",
    "TrialDevEvaluationTargetRegisterManifestV1",
    "TrialDevEvaluationTargetRegisterRecordV1",
    "load_trialdev_evaluation_target_register_gate_report",
    "load_trialdev_evaluation_target_register_manifest",
    "load_trialdev_evaluation_target_register_records",
    "required_trialdev_lanes_v1",
    "required_trialdev_evaluation_contexts_v1",
    "validate_trialdev_evaluation_target_register",
]
