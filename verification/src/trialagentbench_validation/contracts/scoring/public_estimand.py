"""Public estimand contract models for TrialEvalBench releases."""

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


class IntercurrentEventStrategyV1(BaseModel):
    """One declared intercurrent-event strategy."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(..., min_length=1)
    event_label: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    strategy: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class CausalIdentificationV1(BaseModel):
    """Target-trial and identification attributes for the treatment contrast."""

    model_config = ConfigDict(extra="forbid")

    eligibility: str = Field(..., min_length=1)
    treatment_strategies: tuple[str, ...] = Field(..., min_length=2)
    assignment_mechanism: str = Field(..., min_length=1)
    time_zero: str = Field(..., min_length=1)
    follow_up: str = Field(..., min_length=1)
    outcome: str = Field(..., min_length=1)
    causal_contrast: str = Field(..., min_length=1)
    adjustment_set: tuple[str, ...] = Field(default_factory=tuple)
    positivity_support: str = Field(..., min_length=1)
    identifying_assumptions: tuple[str, ...] = Field(..., min_length=1)


class EstimandAttributesV1(BaseModel):
    """Clinical and SAP attributes shared by scoreable primary variants."""

    model_config = ConfigDict(extra="forbid")

    estimand_id: str = Field(..., min_length=1)
    treated_condition: str = Field(..., min_length=1)
    control_condition: str = Field(..., min_length=1)
    population: str = Field(..., min_length=1)
    analysis_set: str = Field(..., min_length=1)
    endpoint_id: str = Field(..., min_length=1)
    endpoint_label: str = Field(..., min_length=1)
    assessment_time_days: float = Field(..., gt=0.0)
    intercurrent_event_strategies: tuple[IntercurrentEventStrategyV1, ...] = Field(
        default_factory=tuple
    )
    objective: Literal["estimation", "superiority", "non_inferiority", "equivalence"]
    direction: Literal["two_sided", "benefit_positive", "benefit_negative"]
    alpha: float = Field(..., gt=0.0, lt=1.0)
    multiplicity_strategy: str = Field(..., min_length=1)
    multiplicity_family: str = Field(..., min_length=1)
    missing_data_strategy: str = Field(..., min_length=1)
    censoring_strategy: str = Field(..., min_length=1)
    design_family: str = Field(..., min_length=1)
    randomization_unit: Literal["participant", "cluster"]
    stratification_factors: tuple[str, ...] = Field(default_factory=tuple)
    design_adjustments: tuple[str, ...] = Field(default_factory=tuple)
    causal_identification: CausalIdentificationV1
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_estimand(self) -> EstimandAttributesV1:
        """Validate event uniqueness and randomization-unit consistency."""

        strategies = tuple(
            sorted(self.intercurrent_event_strategies, key=lambda row: row.event_id)
        )
        event_ids = tuple(row.event_id for row in strategies)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError(
                "Intercurrent-event strategy event_id values must be unique."
            )
        cluster_design = (
            "cluster" in self.design_family or "stepped_wedge" in self.design_family
        )
        if cluster_design != (self.randomization_unit == "cluster"):
            raise ValueError("randomization_unit must agree with design_family.")
        object.__setattr__(self, "intercurrent_event_strategies", strategies)
        object.__setattr__(
            self,
            "stratification_factors",
            tuple(sorted(set(self.stratification_factors))),
        )
        object.__setattr__(
            self, "design_adjustments", tuple(sorted(set(self.design_adjustments)))
        )
        object.__setattr__(
            self, "checksum", _payload_checksum(self.model_dump(mode="json"))
        )
        return self


class PublicEstimandVariantV1(BaseModel):
    """One scoreable public estimand variant."""

    model_config = ConfigDict(extra="forbid")

    variant_id: str = Field(..., min_length=1)
    estimand_id: str = Field(..., min_length=1)
    route_family: str = Field(..., min_length=1)
    estimator_families: tuple[str, ...] = Field(default_factory=tuple)
    effect_scale: str = Field(..., min_length=1)
    answer_shapes: tuple[str, ...] = Field(..., min_length=1)
    required_modifiers: tuple[str, ...] = Field(default_factory=tuple)
    eligibility_class: str | None = Field(default=None, min_length=1)
    route_reference_id: str | None = Field(default=None, min_length=1)
    public_evidence_basis: tuple[str, ...] = Field(..., min_length=1)
    modifier_evidence_basis: tuple[ModifierEvidenceBasisV1, ...] = Field(
        default_factory=tuple
    )
    acceptance_rationale: str | None = Field(default=None, min_length=1)
    rationale: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_variant(self) -> PublicEstimandVariantV1:
        """Validate that public variants separate families from modifiers."""

        object.__setattr__(
            self, "answer_shapes", tuple(sorted(set(self.answer_shapes)))
        )
        object.__setattr__(
            self, "estimator_families", tuple(sorted(set(self.estimator_families)))
        )
        object.__setattr__(
            self, "required_modifiers", tuple(sorted(set(self.required_modifiers)))
        )
        object.__setattr__(
            self,
            "public_evidence_basis",
            tuple(sorted(set(self.public_evidence_basis))),
        )
        modifier_evidence = tuple(
            sorted(self.modifier_evidence_basis, key=lambda entry: str(entry.modifier))
        )
        object.__setattr__(self, "modifier_evidence_basis", modifier_evidence)
        if self.route_family in ROUTE_FAMILY_MODIFIER_TOKENS_V1:
            raise ValueError(
                f"variant route_family cannot be a method modifier: {self.route_family}"
            )
        modifier_evidence_keys = tuple(
            str(entry.modifier) for entry in modifier_evidence
        )
        if len(modifier_evidence_keys) != len(set(modifier_evidence_keys)):
            raise ValueError(
                "modifier_evidence_basis cannot contain duplicate modifiers."
            )
        if set(modifier_evidence_keys) != set(self.required_modifiers):
            raise ValueError(
                "Public estimand variant modifier evidence must exactly match required_modifiers."
            )
        for evidence_row in modifier_evidence:
            if not set(evidence_row.public_rel_paths) <= set(
                self.public_evidence_basis
            ):
                raise ValueError(
                    "Public estimand modifier evidence paths must be included in public_evidence_basis."
                )
        return self


class PublicEstimandContractV1(BaseModel):
    """Participant-facing primary estimand contract."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval.public_estimand_contract/v1"]
    task_id: str = Field(..., min_length=1)
    item_id: str = Field(..., min_length=1)
    lane_id: Literal["primary_numeric.v1"]
    estimand: EstimandAttributesV1
    mode: Literal["fixed_declared_estimand",]
    declared_primary_effect_scale: str | None = Field(default=None, min_length=1)
    primary_route_family: str = Field(..., min_length=1)
    credit_eligible_route_families: tuple[str, ...] = Field(..., min_length=1)
    variants: tuple[PublicEstimandVariantV1, ...] = Field(..., min_length=1)
    public_evidence_basis: tuple[str, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_contract(self) -> PublicEstimandContractV1:
        """Validate contract consistency and optional checksum."""

        accepted_values = tuple(sorted(set(self.credit_eligible_route_families)))
        public_evidence_basis = tuple(sorted(set(self.public_evidence_basis)))
        object.__setattr__(self, "credit_eligible_route_families", accepted_values)
        object.__setattr__(self, "public_evidence_basis", public_evidence_basis)
        accepted = set(accepted_values)
        if self.primary_route_family not in accepted:
            raise ValueError("primary_route_family must be credit-eligible.")
        modifier_families = sorted(
            {self.primary_route_family, *accepted} & ROUTE_FAMILY_MODIFIER_TOKENS_V1
        )
        if modifier_families:
            raise ValueError(
                f"Public estimand contract route-family fields cannot contain modifiers: {modifier_families}"
            )
        for variant in self.variants:
            if variant.route_family not in accepted:
                raise ValueError(
                    "Every public estimand variant route_family must be credit-eligible."
                )
        variant_ids = tuple(variant.variant_id for variant in self.variants)
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError(
                "Public estimand variant IDs must be unique within a task."
            )
        estimand_by_scale: dict[str, str] = {}
        scale_by_estimand: dict[str, str] = {}
        for variant in self.variants:
            scale = str(variant.effect_scale)
            estimand_id = str(variant.estimand_id)
            if scale in estimand_by_scale and estimand_by_scale[scale] != estimand_id:
                raise ValueError(
                    "One effect scale cannot map to multiple estimand identities."
                )
            if (
                estimand_id in scale_by_estimand
                and scale_by_estimand[estimand_id] != scale
            ):
                raise ValueError(
                    "Distinct effect scales require distinct estimand identities."
                )
            estimand_by_scale[scale] = estimand_id
            scale_by_estimand[estimand_id] = scale
        if self.declared_primary_effect_scale is None:
            raise ValueError(
                "fixed_declared_estimand requires declared_primary_effect_scale."
            )
        scales = {variant.effect_scale for variant in self.variants}
        if scales != {self.declared_primary_effect_scale}:
            raise ValueError(
                "fixed_declared_estimand variants must all use the declared primary scale."
            )
        if set(scale_by_estimand) != {str(self.estimand.estimand_id)}:
            raise ValueError(
                "fixed_declared_estimand variants must use the declared estimand identity."
            )
        object.__setattr__(
            self, "checksum", _payload_checksum(self.model_dump(mode="json"))
        )
        return self


class PublicEstimandContractManifestV1(BaseModel):
    """Manifest binding public estimand contracts to their scorer reference."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["v1"]
    schema_id: Literal["trialagentbench.trialeval.public_estimand_contract_manifest/v1"]
    release_root: str = Field(..., min_length=1)
    generated_at_utc: datetime | None = None
    row_count: int = Field(..., ge=0)
    task_count: int = Field(..., ge=0)
    public_estimand_contract_jsonl_sha256: str = Field(
        ..., min_length=64, max_length=64
    )
    route_references_sha256: str = Field(..., min_length=64, max_length=64)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_checksum(self) -> PublicEstimandContractManifestV1:
        """Validate the optional canonical payload checksum."""

        object.__setattr__(
            self, "checksum", _payload_checksum(self.model_dump(mode="json"))
        )
        return self


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


def read_jsonl_public_estimand_contracts(
    path: Path,
) -> tuple[PublicEstimandContractV1, ...]:
    """Read public estimand contracts from a grader-domain JSONL file."""

    contracts: list[PublicEstimandContractV1] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Invalid public estimand row at {path}:{line_number}.")
        payload_obj = row.get("payload")
        if not isinstance(payload_obj, dict) or not isinstance(
            payload_obj.get("contract"), dict
        ):
            raise ValueError(
                f"Public estimand row missing payload.contract at {path}:{line_number}."
            )
        contract_obj = payload_obj.get("contract")
        contracts.append(PublicEstimandContractV1.model_validate(contract_obj))
    return tuple(contracts)


def read_public_estimand_domains(
    *, release_root: Path
) -> tuple[tuple[PublicEstimandContractV1, ...], PublicEstimandContractManifestV1]:
    """Read public estimand contracts and verify their packaged dependencies."""

    domains = Path(release_root) / "grader" / "domains"
    path = domains / "public_estimand_contract.jsonl"
    contracts = read_jsonl_public_estimand_contracts(path)
    manifest = read_json_model(
        PublicEstimandContractManifestV1,
        domains / "public_estimand_contract_manifest.json",
    )
    if manifest.row_count != len(contracts) or manifest.task_count != len(
        {contract.task_id for contract in contracts}
    ):
        raise ValueError(
            "Public-estimand manifest counts do not match its JSONL domain."
        )
    if manifest.public_estimand_contract_jsonl_sha256 != sha256_file(path):
        raise ValueError(
            "Public-estimand manifest SHA-256 does not match its JSONL domain."
        )
    if manifest.route_references_sha256 != sha256_file(
        domains / "route_references.jsonl"
    ):
        raise ValueError("Public-estimand manifest route-reference SHA-256 mismatch.")
    return contracts, manifest


__all__ = [
    "CausalIdentificationV1",
    "EstimandAttributesV1",
    "IntercurrentEventStrategyV1",
    "PublicEstimandContractV1",
    "PublicEstimandContractManifestV1",
    "PublicEstimandVariantV1",
    "read_jsonl_public_estimand_contracts",
    "read_public_estimand_domains",
]
