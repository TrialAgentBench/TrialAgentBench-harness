"""Independent validation of the TrialDevBench scientific construction map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.contracts.scientific_sources import (
    ScientificSourceRegistryV1,
)


class _ReleaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevScientificConstructionRowV1(_ReleaseModel):
    """One independently readable TrialDev grading-lane declaration."""

    scenario_id: str = Field(min_length=1)
    generation_seed: int = Field(ge=1, le=2**31 - 2)
    phase_id: Literal[
        "observational_review", "phase1", "phase2", "phase3", "final_decision"
    ]
    program_objective_id: str = Field(min_length=1)
    phase_scoring_objective_id: str = Field(min_length=1)
    lane_id: Literal[
        "asset_nomination",
        "phase_design",
        "phase_analysis",
        "decision_action",
        "route_timing",
        "final_recommendation",
        "safety_gate",
    ]
    development_purpose: Literal[
        "observational_candidate_prioritization_exercise",
        "randomized_early_safety_and_feasibility_exercise",
        "randomized_exploratory_evidence_exercise",
        "randomized_confirmatory_style_evidence_exercise",
        "sequential_program_decision_exercise",
    ]
    allocation_structure: Literal[
        "nonrandomized_comparative_cohort",
        "participant_randomized_with_concurrent_control",
        "realized_multiphase_trajectory",
    ]
    identification_class: Literal[
        "point_identified_under_declared_measured_confounding_assumptions",
        "qualified_nonidentification_under_residual_unmeasured_confounding",
        "randomized_descriptive_risk_under_arm_conditional_independent_censoring",
        "randomized_comparative_risk_under_arm_conditional_independent_censoring",
        "trajectory_conditioned_decision",
    ]
    identification_assumptions: tuple[str, ...] = Field(min_length=1)
    analysis_method_route_ids: tuple[str, ...] = ()
    design_cell_ids: tuple[str, ...] = ()
    permitted_intercurrent_event_bindings: tuple[str, ...] = ()
    competing_event_handling_id: str = Field(min_length=1)
    treatment_discontinuation_handling_id: str = Field(min_length=1)
    loss_to_follow_up_handling_id: str = Field(min_length=1)
    missing_observation_handling_id: str = Field(min_length=1)
    scoring_policy_id: str = Field(min_length=1)
    target_resolution: Literal[
        "release_static",
        "submitted_method_public_evidence",
        "realized_public_evidence",
        "realized_trajectory",
    ]
    reference_target_ids: tuple[str, ...] = Field(min_length=1)
    credit_eligible_target_ids: tuple[str, ...] = ()
    rejected_shortcut_ids: tuple[str, ...] = ()
    recoverability_policy_id: str = Field(min_length=1)
    public_evidence_basis: tuple[str, ...] = Field(min_length=1)
    evaluator_evidence_basis: tuple[str, ...] = Field(min_length=1)
    normative_source_ids: tuple[str, ...] = Field(min_length=1)
    method_source_ids: tuple[str, ...] = Field(min_length=1)
    precedent_source_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _canonical_row(self) -> TrialDevScientificConstructionRowV1:
        fields = (
            "identification_assumptions",
            "analysis_method_route_ids",
            "design_cell_ids",
            "permitted_intercurrent_event_bindings",
            "reference_target_ids",
            "credit_eligible_target_ids",
            "rejected_shortcut_ids",
            "public_evidence_basis",
            "evaluator_evidence_basis",
            "normative_source_ids",
            "method_source_ids",
            "precedent_source_ids",
        )
        for name in fields:
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        return self


class TrialDevScientificConstructionInventoryV1(_ReleaseModel):
    """Checksum-bound independent representation of the TrialDev scientific map."""

    schema_id: Literal["trialagentbench.trialdev.scientific_construction_inventory/v1"]
    release_id: str = Field(min_length=1)
    suite_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_registry_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: tuple[TrialDevScientificConstructionRowV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_and_checksummed(self) -> TrialDevScientificConstructionInventoryV1:
        identities = tuple(_row_identity(row) for row in self.rows)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(
            identities
        ):
            raise ValueError("TrialDev scientific rows must be sorted and unique")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError("TrialDev scientific inventory checksum mismatch")
        return self


def _row_identity(
    row: TrialDevScientificConstructionRowV1,
) -> tuple[str, str, str, str, str]:
    return (
        row.scenario_id,
        row.phase_id,
        row.program_objective_id,
        row.phase_scoring_objective_id,
        row.lane_id,
    )


def _evaluation_target_rows(
    bundle_root: Path,
) -> dict[tuple[str, str, str, str, str], dict[str, object]]:
    rows: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for scenario_root in sorted(
        path for path in Path(bundle_root).glob("scenario_s*") if path.is_dir()
    ):
        path = scenario_root / "grader" / "evaluation_target_register.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"TrialDev evaluation-target row is not an object: {path}"
                )
            identity = (
                str(payload["scenario_id"]),
                str(payload["phase_id"]),
                str(payload["program_objective_id"]),
                str(payload["phase_scoring_objective_id"]),
                str(payload["lane_id"]),
            )
            if identity in rows:
                raise ValueError(
                    f"duplicate TrialDev evaluation-target identity: {identity!r}"
                )
            rows[identity] = payload
    if not rows:
        raise ValueError(
            "TrialDev evaluator contains no evaluation-target register rows"
        )
    return rows


def validate_trialdev_scientific_inventory(
    *,
    inventory: TrialDevScientificConstructionInventoryV1,
    registry: ScientificSourceRegistryV1,
    bundle_root: Path,
) -> None:
    """Require the scientific map to equal the released suite and evaluation-target register."""

    suite_payload = json.loads(
        (Path(bundle_root) / "benchmark_suite_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(
        suite_payload, dict
    ) or inventory.suite_manifest_checksum != suite_payload.get("checksum"):
        raise ValueError(
            "TrialDev scientific inventory does not identify the released suite manifest"
        )
    if inventory.source_registry_checksum != registry.checksum:
        raise ValueError(
            "TrialDev scientific inventory does not identify the released source registry"
        )
    declared = {_row_identity(row): row for row in inventory.rows}
    evaluation_targets = _evaluation_target_rows(bundle_root)
    if set(declared) != set(evaluation_targets):
        raise ValueError(
            "TrialDev scientific inventory must cover exactly every evaluation-target register row"
        )
    compared_fields = (
        "scoring_policy_id",
        "target_resolution",
        "reference_target_ids",
        "credit_eligible_target_ids",
        "rejected_shortcut_ids",
        "recoverability_policy_id",
        "public_evidence_basis",
        "evaluator_evidence_basis",
    )
    for identity, row in declared.items():
        source = evaluation_targets[identity]
        for field in compared_fields:
            observed = getattr(row, field)
            expected = source[field]
            if isinstance(observed, tuple):
                if not isinstance(expected, list) or not all(
                    isinstance(value, str) for value in expected
                ):
                    raise ValueError(
                        f"TrialDev evaluation-target field {field!r} is not a string array"
                    )
                expected = tuple(sorted(set(expected)))
            if observed != expected:
                raise ValueError(
                    f"TrialDev scientific field {field!r} drifts for {identity!r}"
                )
    known = {source.source_id for source in registry.sources}
    used = {
        source_id
        for row in inventory.rows
        for source_id in (
            *row.normative_source_ids,
            *row.method_source_ids,
            *row.precedent_source_ids,
        )
    }
    missing = sorted(used - known)
    if missing:
        raise ValueError(
            f"TrialDev scientific inventory cites unresolved sources: {missing!r}"
        )


__all__ = [
    "TrialDevScientificConstructionInventoryV1",
    "TrialDevScientificConstructionRowV1",
    "validate_trialdev_scientific_inventory",
]
