"""Evaluation-target register contracts for TrialEval release graders."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.contracts.scoring.modifier_evidence import (
    ModifierEvidenceBasisV1,
)
from trialagentbench_validation.io.checksums import sha256_file
from trialagentbench_validation.io.json import read_json_model

EstimatorRouteFamilyRoleV1 = Literal[
    "estimand_family",
    "design_modifier",
    "procedure_modifier",
    "diagnostic_only",
    "unsupported",
]
EstimatorFamilyExclusionReasonV1 = Literal[
    "different_estimand",
    "failed_or_missing_qualification",
    "regime_invalid_diagnostic_only",
]
ESTIMATOR_FAMILY_VOCABULARY_V1 = frozenset(
    {
        "km",
        "coxph_binary",
        "coxph_time_interaction",
        "piecewise_cox",
        "weighted_logrank",
        "aft_parametric",
        "milestone_risk",
        "km_ipcw",
        "coxph_ipcw",
        "validated_endpoint",
        "cluster_parallel_participant_weighted",
        "stepped_wedge_period_adjusted",
        "group_sequential_adjustment",
        "competing_risks",
        "bounds",
        "other",
    }
)
ScoreProfileIdV1 = Literal[
    "credit_eligible_family_v1",
    "strict_method_id_v1",
    "diagnostic_recognition_v1",
]
GroundReferenceContestednessV1 = Literal["construction_determined", "underdetermined"]
ROUTE_FAMILY_MODIFIER_TOKENS_V1 = frozenset(
    {
        "ipcw_adjusted",
        "cluster_robust_inference",
        "participant_population_target",
        "stepped_wedge_period_adjusted",
        "group_sequential_adjustment",
        "misclassification_corrected",
        "reference_standardization",
        "flexible_model_form",
        "ph_robust_fixed_horizon",
    }
)

_ROUTE_FAMILY_BY_EFFECT_SCALE_V1 = {
    "risk_difference_tau": "risk_difference",
    "standardized_risk_difference_tau_reference": "standardized_risk",
    "rmst_difference_tau": "rmst_contrast",
    "log_hr": "global_cox_ph",
    "hr": "global_cox_ph",
    "hazard_ratio": "global_cox_ph",
    "time_varying_log_hr": "time_varying_cox",
    "piecewise_log_hr_vector": "piecewise_cox",
    "weighted_logrank_test": "weighted_logrank",
    "log_time_ratio": "aft_parametric",
    "cif_difference_tau": "competing_risk",
    "milestone_risk_difference_tau": "milestone_risk",
    "bounds_interval": "partial_identification",
    "non_identification": "qualified_limitation",
}


def route_family_for_effect_scale_v1(effect_scale: str) -> str:
    """Return the official route family represented by an effect scale."""

    try:
        return _ROUTE_FAMILY_BY_EFFECT_SCALE_V1[effect_scale]
    except KeyError as error:
        raise ValueError(
            f"Unsupported TrialEval effect scale: {effect_scale!r}."
        ) from error


class EstimatorRouteFamilyMapEntryV1(BaseModel):
    """One estimator-family to route-family mapping row."""

    model_config = ConfigDict(extra="forbid")

    estimator_family: str = Field(..., min_length=1)
    route_family: str | None = Field(default=None, min_length=1)
    role: EstimatorRouteFamilyRoleV1
    requires_proportional_hazards_diagnostic: bool = False
    requires_limitation: bool = False
    exclusion_reason: EstimatorFamilyExclusionReasonV1 | None = None
    credit_eligible_when: str = Field(..., min_length=1)
    rejected_when: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_role_payload(self) -> EstimatorRouteFamilyMapEntryV1:
        """Validate that route-family mappings are explicit for estimand rows."""

        if self.role == "estimand_family" and self.route_family is None:
            raise ValueError("estimand_family rows require route_family.")
        if self.role != "estimand_family" and self.route_family is not None:
            raise ValueError("Only estimand_family rows may declare route_family.")
        if self.role == "unsupported" and self.exclusion_reason is None:
            raise ValueError(
                "Unsupported estimator families require a typed exclusion_reason."
            )
        if self.role != "unsupported" and self.exclusion_reason is not None:
            raise ValueError(
                "Only unsupported estimator families may declare exclusion_reason."
            )
        if self.route_family in ROUTE_FAMILY_MODIFIER_TOKENS_V1:
            raise ValueError(
                f"route_family cannot be a method modifier: {self.route_family}"
            )
        return self


class EstimatorRouteFamilyMapV1(BaseModel):
    """Versioned estimator route-family map shipped with a release."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["v1"] = "v1"
    schema_id: Literal["trialagentbench.trialeval.estimator_route_family_map/v1"]
    entries: tuple[EstimatorRouteFamilyMapEntryV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_entries(self) -> EstimatorRouteFamilyMapV1:
        """Validate uniqueness and checksum stability."""

        families = [entry.estimator_family for entry in self.entries]
        if len(families) != len(set(families)):
            raise ValueError(
                "Duplicate estimator_family in estimator route-family map."
            )
        if set(families) != ESTIMATOR_FAMILY_VOCABULARY_V1:
            raise ValueError(
                "Estimator route-family map must partition the complete submission vocabulary: "
                f"missing={sorted(ESTIMATOR_FAMILY_VOCABULARY_V1 - set(families))!r} "
                f"unexpected={sorted(set(families) - ESTIMATOR_FAMILY_VOCABULARY_V1)!r}."
            )
        if self.checksum is not None and self.checksum != _payload_checksum(
            self.model_dump(mode="json")
        ):
            raise ValueError("Estimator route-family map checksum mismatch.")
        return self


class EvaluationTargetRegisterEntryV1(BaseModel):
    """One score-lane reference-policy row."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval.evaluation_target_register_entry/v1"]
    task_id: str = Field(..., min_length=1)
    item_id: str = Field(..., min_length=1)
    lane_id: str = Field(..., min_length=1)
    lane_class: str = Field(..., min_length=1)
    evaluation_class: str = Field(..., min_length=1)
    scoring_reference_set: str = Field(..., min_length=1)
    allowed_answer_shapes: tuple[str, ...] = Field(..., min_length=1)
    identification_class: Literal["point_identified", "partially_identified"]
    contestedness: GroundReferenceContestednessV1
    estimand_mode: Literal["fixed_declared_estimand"]
    declared_primary_effect_scale: str = Field(..., min_length=1)
    credit_eligible_primary_effect_scales: tuple[str, ...] = Field(..., min_length=1)
    primary_route_family: str = Field(..., min_length=1)
    credit_eligible_route_families: tuple[str, ...] = Field(..., min_length=1)
    rejected_shortcut_families: tuple[str, ...] = Field(default_factory=tuple)
    expected_effect_scale: str | None = Field(default=None, min_length=1)
    expected_method_id: str | None = Field(default=None, min_length=1)
    expected_route_family: str | None = Field(default=None, min_length=1)
    required_diagnostic_keys: tuple[str, ...] = Field(default_factory=tuple)
    required_limitation_keys: tuple[str, ...] = Field(default_factory=tuple)
    public_evidence_basis: tuple[str, ...] = Field(default_factory=tuple)
    modifier_evidence_basis: tuple[ModifierEvidenceBasisV1, ...] = Field(
        default_factory=tuple
    )
    score_profile_eligibility: tuple[ScoreProfileIdV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_entry(self) -> EvaluationTargetRegisterEntryV1:
        """Validate lane-level evaluation-target invariants."""

        accepted = set(self.credit_eligible_route_families)
        accepted_scales = tuple(
            dict.fromkeys(self.credit_eligible_primary_effect_scales)
        )
        rejected = set(self.rejected_shortcut_families)
        if self.primary_route_family not in accepted:
            raise ValueError("primary_route_family must be credit-eligible.")
        if self.declared_primary_effect_scale not in set(accepted_scales):
            raise ValueError(
                "declared_primary_effect_scale must be eligible for primary credit."
            )
        if accepted_scales != (self.declared_primary_effect_scale,):
            raise ValueError(
                "fixed_declared_estimand requires exactly the declared primary effect scale."
            )
        projected_families = {
            route_family_for_effect_scale_v1(scale) for scale in accepted_scales
        }
        if self.lane_id == "primary_numeric.v1" and accepted != projected_families:
            raise ValueError(
                "credit_eligible_route_families must exactly match the credit-eligible primary effect-scale projection."
            )
        if accepted & rejected:
            raise ValueError(
                "credit_eligible_route_families and rejected_shortcut_families overlap."
            )
        family_values = {self.primary_route_family, *accepted, *rejected}
        if self.expected_route_family is not None:
            family_values.add(self.expected_route_family)
        modifier_families = sorted(family_values & ROUTE_FAMILY_MODIFIER_TOKENS_V1)
        if modifier_families:
            raise ValueError(
                f"Evaluation-target register route-family fields cannot contain modifiers: {modifier_families}"
            )
        numeric_shapes = {
            "numeric_point",
            "numeric_interval",
            "bounds_interval",
            "curve",
        }
        if (
            self.lane_id == "primary_numeric.v1"
            and bool(set(self.allowed_answer_shapes) & numeric_shapes)
            and accepted == {"qualified_limitation"}
        ):
            raise ValueError(
                "primary numeric lanes cannot be qualified-limitation-only."
            )
        if not self.public_evidence_basis:
            raise ValueError(
                "Evaluation-target register entries require public_evidence_basis."
            )
        for evidence_row in self.modifier_evidence_basis:
            if not set(evidence_row.public_rel_paths) <= set(
                self.public_evidence_basis
            ):
                raise ValueError(
                    "modifier_evidence_basis paths must be included in public_evidence_basis."
                )
        if self.checksum is not None and self.checksum != _payload_checksum(
            self.model_dump(mode="json")
        ):
            raise ValueError("Evaluation-target register entry checksum mismatch.")
        return self


class EvaluationTargetRegisterManifestV1(BaseModel):
    """Manifest for a evaluation-target register JSONL file."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["v1"]
    schema_id: Literal[
        "trialagentbench.trialeval.evaluation_target_register_manifest/v1"
    ]
    release_root: str
    generated_at_utc: datetime | None = None
    row_count: int = Field(..., ge=0)
    task_count: int = Field(..., ge=0)
    register_jsonl_sha256: str = Field(..., min_length=64, max_length=64)
    estimator_route_family_map_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    checksum: str | None = Field(default=None, min_length=64, max_length=64)


def _payload_checksum(payload: dict[str, object]) -> str:
    payload = cast(dict[str, object], _drop_none(dict(payload)))
    payload.pop("checksum", None)
    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _drop_none(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_drop_none(v) for v in value)
    return value


def read_jsonl_register(path: Path) -> tuple[EvaluationTargetRegisterEntryV1, ...]:
    """Read and validate a evaluation-target register JSONL file."""

    entries: list[EvaluationTargetRegisterEntryV1] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid evaluation-target row at {path}:{line_number}.")
        entries.append(EvaluationTargetRegisterEntryV1.model_validate(payload))
    return tuple(entries)


def read_ground_reference_domains(
    *,
    release_root: Path,
) -> tuple[
    EstimatorRouteFamilyMapV1,
    tuple[EvaluationTargetRegisterEntryV1, ...],
    EvaluationTargetRegisterManifestV1,
]:
    """Read the three canonical ground-reference domains from a release root."""

    domains = Path(release_root) / "grader" / "domains"
    family_map = read_json_model(
        EstimatorRouteFamilyMapV1, domains / "estimator_route_family_map.json"
    )
    evaluation_targets = read_jsonl_register(
        domains / "evaluation_target_register.jsonl"
    )
    manifest = read_json_model(
        EvaluationTargetRegisterManifestV1,
        domains / "evaluation_target_register_manifest.json",
    )
    if manifest.row_count != len(evaluation_targets):
        raise ValueError(
            "Evaluation-target register row_count does not match JSONL row count."
        )
    if manifest.register_jsonl_sha256 != sha256_file(
        domains / "evaluation_target_register.jsonl"
    ):
        raise ValueError(
            "Evaluation-target register manifest SHA-256 does not match JSONL file."
        )
    if manifest.estimator_route_family_map_sha256 is not None:
        observed = sha256_file(domains / "estimator_route_family_map.json")
        if manifest.estimator_route_family_map_sha256 != observed:
            raise ValueError(
                "Estimator route-family map SHA-256 does not match manifest."
            )
    return family_map, evaluation_targets, manifest


__all__ = [
    "EstimatorRouteFamilyMapEntryV1",
    "EstimatorRouteFamilyMapV1",
    "EvaluationTargetRegisterEntryV1",
    "EvaluationTargetRegisterManifestV1",
    "read_ground_reference_domains",
    "read_jsonl_register",
]
