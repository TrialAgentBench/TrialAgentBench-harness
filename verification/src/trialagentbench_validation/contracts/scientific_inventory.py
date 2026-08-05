"""Independent schema and checks for the TrialEval scientific inventory."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.contracts.scoring_keys import (
    CategoricalTargetV1,
    ScoringKeyManifestV1,
    ValidatedScoringKeyV1,
)

_BOUNDED_DEVIATION_GRID_V1 = (0.05, 0.10, 0.20)


class _ReleaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialEvalScientificConstructionRowV1(_ReleaseModel):
    """One independently inspectable scientific route declaration."""

    generation_unit_id: str = Field(min_length=1)
    generation_seed: int = Field(ge=1, le=2**31 - 2)
    item_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    design_tier: Literal["D1", "D2", "D3", "D4"]
    design_subtype: str = Field(min_length=1)
    assumption_tier: Literal["A1", "A2", "A3", "A4"]
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    objective: Literal["estimation", "hypothesis_testing", "identification", "decision"]
    analysis_role: Literal[
        "main", "sensitivity", "supplementary", "diagnostic", "limitation"
    ]
    scoring_role: Literal[
        "main_credit", "required_support", "optional_report", "not_scored"
    ]
    target_population_id: str = Field(min_length=1)
    analysis_population_id: str = Field(min_length=1)
    treatment_id: str = Field(min_length=1)
    comparator_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    assessment_horizon_days: float | None = Field(default=None, gt=0)
    time_origin_id: Literal["randomization"]
    intercurrent_event_bindings: tuple[str, ...] = ()
    safety_handling_id: Literal[
        "safety_endpoints_reported_separately_from_the_primary_efficacy_estimand"
    ]
    competing_event_handling_id: Literal[
        "no_competing_event_component_in_declared_primary_endpoint"
    ]
    censoring_handling_id: Literal[
        "unweighted_time_to_event_under_declared_censoring_assumption",
        "baseline_covariate_ipcw",
    ]
    missing_observation_handling_id: Literal[
        "observed_event_time_and_censoring_indicator_without_endpoint_imputation"
    ]
    interpretation_constraints: tuple[str, ...] = Field(min_length=1)
    identification_class: Literal[
        "point_identified",
        "design_adjusted",
        "sensitivity_identified",
        "partially_identified",
        "decision_identified",
    ]
    identification_assumptions: tuple[str, ...] = Field(min_length=1)
    analysis_method_id: str = Field(min_length=1)
    estimator_family: str = Field(min_length=1)
    effect_scale: str = Field(min_length=1)
    result_kind: str = Field(min_length=1)
    uncertainty_method: str = Field(min_length=1)
    sensitivity_parameters: tuple[float, ...] = ()
    design_obligations: tuple[str, ...] = ()
    required_diagnostics: tuple[str, ...] = ()
    participant_evidence_paths: tuple[str, ...] = Field(min_length=1)
    evaluator_reference_kind: Literal[
        "numeric_point",
        "numeric_interval",
        "numeric_vector",
        "statistical_test",
        "categorical",
    ]
    comparison_rule: Literal["numeric_envelope", "categorical_code_membership"]
    reporting_decimal_places: int | None = Field(default=None, ge=0, le=15)
    independent_max_abs_difference: float | None = Field(default=None, ge=0)
    public_verification_id: str = Field(min_length=1)
    independent_verification_ids: tuple[str, ...] = Field(min_length=1)
    verification_record_paths: tuple[str, ...] = Field(min_length=1)
    normative_source_ids: tuple[str, ...] = Field(min_length=1)
    method_source_ids: tuple[str, ...] = Field(min_length=1)
    precedent_source_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _canonical_route(self) -> TrialEvalScientificConstructionRowV1:
        canonical_fields = (
            "intercurrent_event_bindings",
            "interpretation_constraints",
            "identification_assumptions",
            "design_obligations",
            "required_diagnostics",
            "participant_evidence_paths",
            "independent_verification_ids",
            "verification_record_paths",
            "normative_source_ids",
            "method_source_ids",
            "precedent_source_ids",
        )
        for name in canonical_fields:
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        if (
            tuple(sorted(set(self.sensitivity_parameters)))
            != self.sensitivity_parameters
        ):
            raise ValueError("sensitivity_parameters must be sorted and unique")
        if bool(self.sensitivity_parameters) != (self.result_kind == "sensitivity_set"):
            raise ValueError(
                "sensitivity parameters are required exactly for sensitivity-set routes"
            )
        if (
            self.result_kind == "sensitivity_set"
            and self.sensitivity_parameters != _BOUNDED_DEVIATION_GRID_V1
        ):
            raise ValueError(
                "sensitivity-set routes require the complete fixed parameter grid"
            )
        if any(":" not in binding for binding in self.intercurrent_event_bindings):
            raise ValueError(
                "intercurrent-event bindings must encode event_id:strategy"
            )
        if (
            self.identification_class == "partially_identified"
            and self.result_kind
            not in {
                "sensitivity_set",
                "identification_bound",
                "limitation",
                "abstention",
            }
        ):
            raise ValueError(
                "partially identified routes require a set, bound, limitation, or abstention"
            )
        if (
            self.analysis_role not in {"main", "limitation"}
            or self.scoring_role != "main_credit"
        ):
            raise ValueError(
                "credit-eligible scientific inventory rows require main or limitation/main-credit roles"
            )
        return self


class TrialEvalScientificConstructionInventoryV1(_ReleaseModel):
    """Checksum-bound inventory independently read from the verification split."""

    schema_id: Literal["trialagentbench.trialeval.scientific_construction_inventory/v1"]
    release_id: str = Field(min_length=1)
    specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: tuple[TrialEvalScientificConstructionRowV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_and_checksummed(self) -> TrialEvalScientificConstructionInventoryV1:
        identities = tuple((row.item_id, row.route_id) for row in self.rows)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(
            identities
        ):
            raise ValueError("scientific-construction rows must be sorted and unique")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError("scientific-construction inventory checksum mismatch")
        return self


def validate_scientific_inventory(
    *,
    inventory: TrialEvalScientificConstructionInventoryV1,
    scoring_manifest: ScoringKeyManifestV1,
    scoring_keys: tuple[ValidatedScoringKeyV1, ...],
    context_tiers: tuple[Literal["C1", "C2", "C3", "C4", "C5"], ...] | None = None,
) -> None:
    """Require the scientific declaration to equal every score-bearing route."""

    if inventory.release_id != scoring_manifest.release_id:
        raise ValueError(
            "scientific inventory and scoring keys identify different releases"
        )
    if inventory.specification_sha256 != scoring_manifest.specification_sha256:
        raise ValueError(
            "scientific inventory and scoring keys identify different specifications"
        )
    allowed_contexts = None if context_tiers is None else set(context_tiers)
    rows = {
        (row.item_id, row.route_id): row
        for row in inventory.rows
        if allowed_contexts is None or row.context_tier in allowed_contexts
    }
    routes = {
        (key.item_id, route.route_id): (key, route)
        for key in scoring_keys
        for route in key.credit_eligible_routes
    }
    if set(rows) != set(routes):
        raise ValueError(
            "scientific inventory must cover exactly every credit-eligible scoring route"
        )
    for identity, row in rows.items():
        key, route = routes[identity]
        signature = route.signature
        expected = {
            "question_id": key.question_id,
            "context_tier": key.context_tier,
            "analysis_population_id": signature.analysis_population_id,
            "treatment_id": signature.treatment_id,
            "comparator_id": signature.comparator_id,
            "endpoint_id": signature.endpoint_id,
            "estimand_id": signature.estimand_id,
            "assessment_horizon_days": signature.assessment_horizon_days,
            "intercurrent_event_bindings": signature.intercurrent_event_strategy_ids,
            "identification_assumptions": route.required_identification_assumptions,
            "analysis_method_id": signature.analysis_method_id,
            "estimator_family": route.method.estimator_family,
            "effect_scale": signature.effect_scale,
            "result_kind": route.method.result_kind,
            "uncertainty_method": route.method.uncertainty_method,
            "sensitivity_parameters": route.method.sensitivity_parameters,
            "design_obligations": route.method.design_modifiers,
            "required_diagnostics": route.required_diagnostics,
        }
        for field, value in expected.items():
            if getattr(row, field) != value:
                raise ValueError(
                    f"scientific inventory disagrees with scoring route {identity[0]}/{identity[1]}: {field}"
                )
        target = route.target
        if isinstance(target, CategoricalTargetV1):
            target_expected: dict[str, object] = {
                "evaluator_reference_kind": "categorical",
                "comparison_rule": "categorical_code_membership",
                "reporting_decimal_places": None,
                "independent_max_abs_difference": None,
            }
        else:
            envelope = target.acceptance_envelope
            target_expected = {
                "evaluator_reference_kind": target.kind,
                "comparison_rule": "numeric_envelope",
                "reporting_decimal_places": envelope.reporting_decimal_places,
                "independent_max_abs_difference": envelope.independent_max_abs_difference,
                "public_verification_id": envelope.public_verification_id,
                "independent_verification_ids": envelope.independent_verification_ids,
            }
        for target_field, target_value in target_expected.items():
            if getattr(row, target_field) != target_value:
                raise ValueError(
                    "scientific inventory disagrees with scoring target "
                    f"{identity[0]}/{identity[1]}: {target_field}"
                )


__all__ = [
    "TrialEvalScientificConstructionInventoryV1",
    "TrialEvalScientificConstructionRowV1",
    "validate_scientific_inventory",
]
