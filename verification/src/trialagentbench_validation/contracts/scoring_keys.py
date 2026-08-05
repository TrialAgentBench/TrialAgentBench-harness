"""Released TrialEval scoring-key schemas used by independent validation."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ReleaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NumericalAcceptanceEnvelopeV1(_ReleaseModel):
    """Prospective numerical tolerance backed by independent replay."""

    schema_id: Literal["trialagentbench.numerical_acceptance_envelope/v1"]
    reporting_decimal_places: int = Field(ge=0, le=15)
    independent_max_abs_difference: float = Field(ge=0)
    public_verification_id: str = Field(min_length=1)
    independent_verification_ids: tuple[str, ...] = Field(min_length=1)


class RouteSignatureV1(_ReleaseModel):
    """Exact question and complete method identity of one accepted route."""

    analysis_population_id: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    intercurrent_event_strategy_ids: tuple[str, ...] = ()
    assessment_horizon_days: float | None = Field(default=None, gt=0)
    treatment_id: str = Field(min_length=1)
    comparator_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    effect_scale: str = Field(min_length=1)
    analysis_method_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_question(self) -> RouteSignatureV1:
        if (
            tuple(sorted(set(self.intercurrent_event_strategy_ids)))
            != self.intercurrent_event_strategy_ids
        ):
            raise ValueError(
                "intercurrent_event_strategy_ids must be sorted and unique"
            )
        return self


class _AcceptanceTarget(_ReleaseModel):
    acceptance_envelope: NumericalAcceptanceEnvelopeV1


class NumericPointTargetV1(_AcceptanceTarget):
    kind: Literal["numeric_point"]
    value: float
    result_unit: str = Field(min_length=1)
    require_confidence_interval: bool
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None

    @model_validator(mode="after")
    def _complete_interval(self) -> NumericPointTargetV1:
        endpoints = (self.confidence_interval_lower, self.confidence_interval_upper)
        if (endpoints[0] is None) != (endpoints[1] is None):
            raise ValueError(
                "target confidence interval endpoints must be supplied together"
            )
        if self.require_confidence_interval != (endpoints[0] is not None):
            raise ValueError(
                "require_confidence_interval must equal the presence of target confidence interval endpoints"
            )
        if (
            endpoints[0] is not None
            and endpoints[1] is not None
            and endpoints[0] > endpoints[1]
        ):
            raise ValueError(
                "target confidence interval lower endpoint cannot exceed upper endpoint"
            )
        return self


class NumericIntervalTargetV1(_AcceptanceTarget):
    kind: Literal["numeric_interval"]
    lower: float
    upper: float
    result_unit: str = Field(min_length=1)


class NamedNumericValueV1(_ReleaseModel):
    name: str = Field(min_length=1)
    value: float


class NumericVectorTargetV1(_AcceptanceTarget):
    kind: Literal["numeric_vector"]
    components: tuple[NamedNumericValueV1, ...] = Field(min_length=1)
    result_unit: str = Field(min_length=1)


class StatisticalTestTargetV1(_AcceptanceTarget):
    kind: Literal["statistical_test"]
    p_value: float = Field(ge=0, le=1)
    reject_null: bool


class CategoricalTargetV1(_ReleaseModel):
    kind: Literal["categorical"]
    credit_eligible_codes: tuple[str, ...] = Field(min_length=1)


ScoringTargetV1 = Annotated[
    NumericPointTargetV1
    | NumericIntervalTargetV1
    | NumericVectorTargetV1
    | StatisticalTestTargetV1
    | CategoricalTargetV1,
    Field(discriminator="kind"),
]


class AnalysisMethodBindingV1(_ReleaseModel):
    """Intrinsic properties selected by one canonical analysis method ID."""

    analysis_method_id: str = Field(min_length=1)
    estimator_family: str = Field(min_length=1)
    result_kind: Literal[
        "numeric_point",
        "numeric_interval",
        "numeric_vector",
        "statistical_test",
        "sensitivity_set",
        "identification_bound",
        "limitation",
        "abstention",
        "decision",
    ]
    uncertainty_method: str = Field(min_length=1)
    sensitivity_parameters: tuple[float, ...] = ()
    design_modifiers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _canonical_method(self) -> AnalysisMethodBindingV1:
        if tuple(sorted(set(self.design_modifiers))) != self.design_modifiers:
            raise ValueError("design_modifiers must be sorted and unique")
        if (
            tuple(sorted(set(self.sensitivity_parameters)))
            != self.sensitivity_parameters
        ):
            raise ValueError("sensitivity_parameters must be sorted and unique")
        if bool(self.sensitivity_parameters) != (self.result_kind == "sensitivity_set"):
            raise ValueError(
                "sensitivity parameters are required exactly for sensitivity-set methods"
            )
        return self


class CreditEligibleScoringRouteV1(_ReleaseModel):
    """One admissible route and its independently qualified target."""

    route_id: str = Field(min_length=1)
    signature: RouteSignatureV1
    method: AnalysisMethodBindingV1
    required_identification_assumptions: tuple[str, ...] = Field(min_length=1)
    required_diagnostics: tuple[str, ...] = ()
    planning_calculator_id: str | None = Field(default=None, min_length=1)
    target: ScoringTargetV1

    @model_validator(mode="after")
    def _coherent_method_and_duties(self) -> CreditEligibleScoringRouteV1:
        if self.signature.analysis_method_id != self.method.analysis_method_id:
            raise ValueError(
                "route signature and intrinsic method record must use one analysis method ID"
            )
        for name, values in (
            (
                "required_identification_assumptions",
                self.required_identification_assumptions,
            ),
            ("required_diagnostics", self.required_diagnostics),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        compatible_targets = {
            "numeric_point": {"numeric_point"},
            "numeric_interval": {"numeric_interval"},
            "numeric_vector": {"numeric_vector"},
            "statistical_test": {"statistical_test"},
            "sensitivity_set": {"numeric_interval", "numeric_vector"},
            "identification_bound": {"numeric_interval"},
            "limitation": {"categorical"},
            "abstention": {"categorical"},
            "decision": {"categorical"},
        }
        if self.target.kind not in compatible_targets[self.method.result_kind]:
            raise ValueError(
                "intrinsic method result kind is incompatible with the scoring target"
            )
        return self


class DataIntegrityTargetV1(_ReleaseModel):
    """Exact noncompensatory C5 repair target."""

    condition_id: Literal["exact_transport_row_duplication_v1"]
    affected_domain: str = Field(min_length=1)
    compound_key_fields: tuple[str, ...] = Field(min_length=1)
    observed_duplicate_group_count: int = Field(ge=1)
    observed_extra_row_count: int = Field(ge=1)
    repair_action: Literal["remove_one_exact_duplicate_copy"]
    repair_status: Literal["repaired"]
    post_repair_data_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_input_data_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _valid_target(self) -> DataIntegrityTargetV1:
        if len(set(self.compound_key_fields)) != len(self.compound_key_fields):
            raise ValueError("data-integrity compound-key fields must be unique")
        if self.analysis_input_data_checksum != self.post_repair_data_checksum:
            raise ValueError("C5 analysis input must equal the repaired domain content")
        return self


class ValidatedScoringKeyV1(_ReleaseModel):
    """Portable evaluator key for one TrialEval task."""

    schema_id: Literal["trialagentbench.scoring_key/v1"]
    release_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    credit_eligible_routes: tuple[CreditEligibleScoringRouteV1, ...] = Field(
        min_length=1
    )
    data_integrity_target: DataIntegrityTargetV1 | None = None

    @model_validator(mode="after")
    def _unique_routes(self) -> ValidatedScoringKeyV1:
        route_ids = tuple(route.route_id for route in self.credit_eligible_routes)
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("credit-eligible route IDs must be unique")
        signatures = tuple(route.signature for route in self.credit_eligible_routes)
        if len(set(signatures)) != len(signatures):
            raise ValueError("credit-eligible route signatures must be unique")
        if (self.context_tier == "C5") != (self.data_integrity_target is not None):
            raise ValueError("exactly C5 scoring keys require a data-integrity target")
        return self


class ScoringKeyManifestV1(_ReleaseModel):
    """Checksum-bound scoring-key inventory."""

    schema_id: Literal["trialagentbench.scoring_key_manifest/v1"]
    release_id: str = Field(min_length=1)
    specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scoring_keys_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_ids: tuple[str, ...] = Field(min_length=1)


def read_scoring_keys(
    archive: ZipFile,
    *,
    expected_item_ids: tuple[str, ...],
) -> tuple[ValidatedScoringKeyV1, ...]:
    """Read checksum-bound scoring keys and require exact item coverage."""

    body = archive.read("grader/scoring_keys.jsonl")
    manifest = ScoringKeyManifestV1.model_validate_json(
        archive.read("grader/scoring_key_manifest.json")
    )
    if hashlib.sha256(body).hexdigest() != manifest.scoring_keys_sha256:
        raise ValueError("scoring-key checksum does not match its manifest")
    keys = tuple(
        ValidatedScoringKeyV1.model_validate(json.loads(line))
        for line in body.decode("utf-8").splitlines()
        if line.strip()
    )
    item_ids = tuple(key.item_id for key in keys)
    if item_ids != manifest.item_ids:
        raise ValueError("scoring-key rows do not match manifest order")
    if tuple(sorted(expected_item_ids)) != tuple(sorted(item_ids)):
        raise ValueError("scoring-key coverage must equal the item-index denominator")
    if {key.release_id for key in keys} != {manifest.release_id}:
        raise ValueError("scoring keys and manifest must identify one release")
    return keys


__all__ = [
    "CreditEligibleScoringRouteV1",
    "ScoringKeyManifestV1",
    "ValidatedScoringKeyV1",
    "read_scoring_keys",
]
