"""Independent validation of the canonical TrialEval component scope."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.contracts.scoring_keys import ScoringKeyManifestV1
from trialagentbench_validation.contracts.trialeval_release import ItemIndexV1
from trialagentbench_validation.contracts.v1_scope import (
    TRIALEVAL_BASE_TRIAL_COUNT_V1,
    TRIALEVAL_BASE_TRIAL_REPLICATES_PER_CELL_V1,
    TRIALEVAL_DESIGN_PROFILE_COUNT_V1,
    TRIALEVAL_EVALUATION_SERIES_COUNT_V1,
    TRIALEVAL_ITEM_COUNT_V1,
    TRIALEVAL_REGIME_CELL_COUNT_V1,
    TRIALEVAL_ROUTE_ARCHETYPE_COUNT_V1,
)


class _ReleaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialEvalDesignProfileV1(_ReleaseModel):
    design_profile_id: str = Field(pattern=r"^TE-DP0[1-7]$")
    design_tier: Literal["D1", "D2", "D3", "D4"]
    study_format: str = Field(min_length=1)
    allocation_unit: Literal["participant", "cluster"]
    intervention_model: Literal[
        "parallel_group", "stepped_wedge", "group_sequential_parallel"
    ]
    analysis_defining_subdesign: str = Field(min_length=1)
    evaluation_series_ids: tuple[str, ...] = Field(min_length=1)


class TrialEvalEvaluationSeriesV1(_ReleaseModel):
    evaluation_series_id: str = Field(pattern=r"^TE-S0[1-9]$")
    design_profile_id: str = Field(pattern=r"^TE-DP0[1-7]$")
    label: str = Field(min_length=1)
    public_question: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    target_population: Literal["all_randomized_participants"]
    endpoint: Literal["death_by_tau", "survival_time_through_tau"]
    contrast_direction: Literal["treated_minus_control"]
    intercurrent_event_bindings: tuple[str, ...]
    competing_event_handling_id: Literal[
        "no_competing_event_component_in_primary_endpoint"
    ]
    censoring_handling_id: Literal[
        "administrative_and_recorded_loss_to_follow_up_censoring"
    ]
    missing_observation_handling_id: Literal["no_endpoint_imputation"]
    safety_handling_id: Literal["safety_reported_separately_from_primary_efficacy"]
    default_route_id: str = Field(min_length=1)
    assumption_tiers: tuple[Literal["A1", "A2", "A3", "A4"], ...] = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)


class TrialEvalRegimeCellV1(_ReleaseModel):
    regime_cell_id: str = Field(pattern=r"^TE-S0[1-9]-A[1-4]$")
    evaluation_series_id: str = Field(pattern=r"^TE-S0[1-9]$")
    assumption_tier: Literal["A1", "A2", "A3", "A4"]
    controlled_condition: str = Field(min_length=1)
    default_response: Literal[
        "compatible",
        "detectable_not_material",
        "material_failure_with_alternative",
        "structurally_broken",
    ]
    identification_class: Literal["point_identified", "partially_identified"]
    eligible_route_ids: tuple[str, ...] = Field(min_length=1)
    excluded_default_route_id: str | None = None
    excluded_default_reason: str | None = None
    required_response: str = Field(min_length=1)
    context_ids: tuple[Literal["C1", "C2", "C3", "C4", "C5"], ...]
    base_trial_replicates: int = Field(ge=1)

    @model_validator(mode="after")
    def _fixed_replicates(self) -> TrialEvalRegimeCellV1:
        if self.base_trial_replicates != TRIALEVAL_BASE_TRIAL_REPLICATES_PER_CELL_V1:
            raise ValueError(
                "every TrialEval regime cell requires exactly "
                f"{TRIALEVAL_BASE_TRIAL_REPLICATES_PER_CELL_V1} base-trial replicates"
            )
        return self


class TrialEvalRouteArchetypeV1(_ReleaseModel):
    route_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    eligible_series_ids: tuple[str, ...] = Field(min_length=1)
    estimator_family: str = Field(min_length=1)
    result_kind: Literal[
        "point_with_interval", "identified_set", "limitation_or_abstention"
    ]
    effect_scale: Literal[
        "death_risk_difference_tau",
        "rmst_difference_tau",
        "validated_endpoint_death_risk_difference_tau",
        "not_applicable",
    ]
    uncertainty_method: str = Field(min_length=1)
    design_obligations: tuple[str, ...]
    diagnostic_requirements: tuple[str, ...]
    implementation_summary: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)


class TrialEvalCanonicalComponentInventoryV1(_ReleaseModel):
    schema_id: Literal["trialagentbench.trialeval.canonical_components/v1"]
    statistical_methods_contract: Literal[
        "trialagentbench.trialeval.statistical_methods/v1"
    ]
    design_profiles: tuple[TrialEvalDesignProfileV1, ...]
    evaluation_series: tuple[TrialEvalEvaluationSeriesV1, ...]
    regime_cells: tuple[TrialEvalRegimeCellV1, ...]
    route_archetypes: tuple[TrialEvalRouteArchetypeV1, ...]
    base_trial_count: int = Field(ge=1)
    participant_context_item_count: int = Field(ge=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _complete_and_checksummed(self) -> TrialEvalCanonicalComponentInventoryV1:
        if (
            len(self.design_profiles) != TRIALEVAL_DESIGN_PROFILE_COUNT_V1
            or len(self.evaluation_series) != TRIALEVAL_EVALUATION_SERIES_COUNT_V1
            or len(self.regime_cells) != TRIALEVAL_REGIME_CELL_COUNT_V1
            or len(self.route_archetypes) != TRIALEVAL_ROUTE_ARCHETYPE_COUNT_V1
        ):
            raise ValueError(
                "canonical TrialEval scope must contain "
                f"{TRIALEVAL_DESIGN_PROFILE_COUNT_V1} profiles, "
                f"{TRIALEVAL_EVALUATION_SERIES_COUNT_V1} series, "
                f"{TRIALEVAL_REGIME_CELL_COUNT_V1} cells, and "
                f"{TRIALEVAL_ROUTE_ARCHETYPE_COUNT_V1} routes"
            )
        if self.base_trial_count != TRIALEVAL_BASE_TRIAL_COUNT_V1:
            raise ValueError(
                f"TrialEval requires exactly {TRIALEVAL_BASE_TRIAL_COUNT_V1} base trials"
            )
        if self.participant_context_item_count != TRIALEVAL_ITEM_COUNT_V1:
            raise ValueError(
                f"TrialEval requires exactly {TRIALEVAL_ITEM_COUNT_V1} participant-context items"
            )
        profile_ids = tuple(row.design_profile_id for row in self.design_profiles)
        series_ids = tuple(row.evaluation_series_id for row in self.evaluation_series)
        cell_ids = tuple(row.regime_cell_id for row in self.regime_cells)
        route_ids = tuple(row.route_id for row in self.route_archetypes)
        if profile_ids != tuple(sorted(set(profile_ids))):
            raise ValueError("design profiles must be sorted and unique")
        if series_ids != tuple(sorted(set(series_ids))):
            raise ValueError("evaluation series must be sorted and unique")
        if cell_ids != tuple(sorted(set(cell_ids))):
            raise ValueError("regime cells must be sorted and unique")
        if route_ids != tuple(sorted(set(route_ids))):
            raise ValueError("route archetypes must be sorted and unique")
        eligible_routes = {
            route_id
            for cell in self.regime_cells
            for route_id in cell.eligible_route_ids
        }
        if eligible_routes != set(route_ids):
            raise ValueError(
                "route archetypes must equal the eligible-route vocabulary"
            )
        expected_items = sum(
            row.base_trial_replicates * len(row.context_ids)
            for row in self.regime_cells
        )
        if expected_items != self.participant_context_item_count:
            raise ValueError(
                "regime cells do not generate the declared participant-context item count"
            )
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError("canonical-component inventory checksum mismatch")
        return self


class TrialEvalReleaseScopeV1(_ReleaseModel):
    """Checksum-bound component census from the verification split."""

    schema_id: Literal["trialagentbench.trialeval.release_scope/v1"]
    release_id: str = Field(min_length=1)
    components: TrialEvalCanonicalComponentInventoryV1
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_and_checksummed(self) -> TrialEvalReleaseScopeV1:
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError("release-scope checksum mismatch")
        return self


def validate_release_scope(
    *,
    scope: TrialEvalReleaseScopeV1,
    item_index: ItemIndexV1,
    scoring_manifest: ScoringKeyManifestV1,
) -> None:
    """Require the released item census to match every canonical regime cell."""

    if scope.release_id != scoring_manifest.release_id:
        raise ValueError("release scope and scoring keys identify different releases")
    cells = {row.regime_cell_id: row for row in scope.components.regime_cells}
    observed_counts = Counter(entry.base_case_id for entry in item_index.entries)
    if set(observed_counts) != set(cells):
        raise ValueError(
            "released item cells must equal the canonical regime-cell scope"
        )
    if len(item_index.entries) != scope.components.participant_context_item_count:
        raise ValueError(
            "released item count does not match the canonical component scope"
        )
    for entry in item_index.entries:
        cell = cells[entry.base_case_id]
        if entry.assumption_tier != cell.assumption_tier:
            raise ValueError(
                "item assumption tier disagrees with its canonical regime cell"
            )
        if entry.context_tier not in cell.context_ids:
            raise ValueError(
                "item context is unavailable for its canonical regime cell"
            )
    expected_counts = {
        cell_id: cell.base_trial_replicates * len(cell.context_ids)
        for cell_id, cell in cells.items()
    }
    if dict(observed_counts) != expected_counts:
        raise ValueError(
            "released item multiplicity disagrees with canonical replication and context rules"
        )


__all__ = [
    "TrialEvalCanonicalComponentInventoryV1",
    "TrialEvalDesignProfileV1",
    "TrialEvalEvaluationSeriesV1",
    "TrialEvalRegimeCellV1",
    "TrialEvalRouteArchetypeV1",
    "TrialEvalReleaseScopeV1",
    "validate_release_scope",
]
