"""Independent validation contract for canonical TrialEval component evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.contracts.release_scope import TrialEvalReleaseScopeV1
from trialagentbench_validation.contracts.scientific_inventory import (
    TrialEvalScientificConstructionInventoryV1,
)
from trialagentbench_validation.contracts.scientific_sources import (
    ScientificSourceRegistryV1,
)
from trialagentbench_validation.contracts.v1_scope import (
    TRIALEVAL_EVALUATION_SERIES_COUNT_V1,
    TRIALEVAL_REGIME_CELL_COUNT_V1,
)


class _ReleaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialEvalSourceApplicationV1(_ReleaseModel):
    source_id: str = Field(pattern=r"^TAB-SRC-[0-9]{3}$")
    exact_locator: str = Field(min_length=1)


class TrialEvalRegimeCellEvidenceV1(_ReleaseModel):
    """Evidence boundary and credit-eligible methods for one regime cell."""

    regime_cell_id: str = Field(pattern=r"^TE-S0[1-9]-A[1-4]$")
    eligible_route_ids: tuple[str, ...] = Field(min_length=1)
    excluded_default_route_id: str | None = Field(default=None, min_length=1)
    excluded_default_reason: str | None = Field(default=None, min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical(self) -> TrialEvalRegimeCellEvidenceV1:
        for field_name in ("eligible_route_ids", "source_ids"):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be sorted and unique")
        if (self.excluded_default_route_id is None) != (
            self.excluded_default_reason is None
        ):
            raise ValueError(
                "default-route exclusion and reason must be declared together"
            )
        return self


class TrialEvalEvaluationSeriesEvidenceV1(_ReleaseModel):
    """Evidence coverage for one fixed statistical question."""

    evaluation_series_id: str = Field(pattern=r"^TE-S0[1-9]$")
    design_profile_id: str = Field(pattern=r"^TE-DP0[1-7]$")
    public_question: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    source_applications: tuple[TrialEvalSourceApplicationV1, ...] = Field(min_length=1)
    cells: tuple[TrialEvalRegimeCellEvidenceV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_cells(self) -> TrialEvalEvaluationSeriesEvidenceV1:
        identities = tuple(cell.regime_cell_id for cell in self.cells)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("series evidence cells must be sorted and unique")
        if any(
            not identity.startswith(f"{self.evaluation_series_id}-")
            for identity in identities
        ):
            raise ValueError("series evidence contains a cell from another series")
        source_ids = tuple(row.source_id for row in self.source_applications)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("series source applications must be sorted and unique")
        if any(cell.source_ids != source_ids for cell in self.cells):
            raise ValueError(
                "every cell must cite the complete series source application set"
            )
        return self


class TrialEvalComponentEvidenceInventoryV1(_ReleaseModel):
    """Checksum-bound evidence authority for all canonical components."""

    schema_id: Literal["trialagentbench.trialeval_component_evidence/v1"]
    release_id: str = Field(min_length=1)
    component_inventory_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_registry_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_series: tuple[TrialEvalEvaluationSeriesEvidenceV1, ...] = Field(
        min_length=1
    )
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _complete_and_checksummed(self) -> TrialEvalComponentEvidenceInventoryV1:
        series_ids = tuple(row.evaluation_series_id for row in self.evaluation_series)
        if series_ids != tuple(
            f"TE-S{index:02d}"
            for index in range(1, TRIALEVAL_EVALUATION_SERIES_COUNT_V1 + 1)
        ):
            raise ValueError(
                "component evidence must contain the nine ordered evaluation series"
            )
        cells = tuple(cell for row in self.evaluation_series for cell in row.cells)
        if (
            len(cells) != TRIALEVAL_REGIME_CELL_COUNT_V1
            or len({cell.regime_cell_id for cell in cells})
            != TRIALEVAL_REGIME_CELL_COUNT_V1
        ):
            raise ValueError("component evidence must contain exactly 25 regime cells")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError("component-evidence checksum mismatch")
        return self


def validate_trialeval_component_evidence(
    *,
    inventory: TrialEvalComponentEvidenceInventoryV1,
    release_scope: TrialEvalReleaseScopeV1,
    source_registry: ScientificSourceRegistryV1,
) -> None:
    """Verify complete component coverage and verified evidence sources."""

    if inventory.release_id != release_scope.release_id:
        raise ValueError("component evidence identifies a different release")
    if inventory.component_inventory_checksum != release_scope.components.checksum:
        raise ValueError(
            "component evidence is not bound to the release component inventory"
        )
    if inventory.source_registry_checksum != source_registry.checksum:
        raise ValueError("component evidence is not bound to its source registry")
    scope_cells = {
        cell.regime_cell_id for cell in release_scope.components.regime_cells
    }
    evidence_cells = {
        cell.regime_cell_id
        for series in inventory.evaluation_series
        for cell in series.cells
    }
    if evidence_cells != scope_cells:
        raise ValueError("component evidence must equal the release regime-cell census")
    verified = {
        source.source_id
        for source in source_registry.sources
        if source.verification_status == "verified"
    }
    cited = {
        source_id
        for series in inventory.evaluation_series
        for cell in series.cells
        for source_id in cell.source_ids
    }
    unresolved = sorted(cited - verified)
    if unresolved:
        raise ValueError(
            f"component evidence cites absent or unverified sources: {unresolved!r}"
        )


def validate_route_component_evidence(
    *,
    component_inventory: TrialEvalComponentEvidenceInventoryV1,
    route_inventory: TrialEvalScientificConstructionInventoryV1,
    item_cell_ids: Mapping[str, str],
) -> None:
    """Require every score-bearing route to preserve its cell evidence boundary."""

    cells_by_id = {
        cell.regime_cell_id: cell
        for series in component_inventory.evaluation_series
        for cell in series.cells
    }
    for route in route_inventory.rows:
        cell_id = item_cell_ids.get(route.item_id)
        if cell_id is None:
            raise ValueError(
                f"scientific route has no item-to-cell join: {route.item_id}"
            )
        cell = cells_by_id.get(cell_id)
        if cell is None:
            raise ValueError(
                f"scientific route resolves to unknown regime cell: {cell_id}"
            )
        if route.assumption_tier != cell_id.rsplit("-", 1)[1]:
            raise ValueError(
                f"scientific route drifts from its assumption tier: {route.route_id}"
            )
        canonical_route_id = _canonical_route_method_id(
            route_id=route.route_id,
            item_id=route.item_id,
            effect_scale=route.effect_scale,
            eligible_route_ids=cell.eligible_route_ids,
        )
        if canonical_route_id not in cell.eligible_route_ids:
            raise ValueError(
                f"scientific route is not eligible for its regime cell: {route.route_id}"
            )
        route_sources = tuple(
            sorted(
                {
                    *route.normative_source_ids,
                    *route.method_source_ids,
                    *route.precedent_source_ids,
                }
            )
        )
        if route_sources != cell.source_ids:
            raise ValueError(
                f"scientific route source IDs drift from component evidence: {route.route_id}"
            )


def _canonical_route_method_id(
    *,
    route_id: str,
    item_id: str,
    effect_scale: str,
    eligible_route_ids: tuple[str, ...],
) -> str:
    """Recover the canonical method from one task-scoped route identity."""

    prefix = f"{item_id}:"
    if not route_id.startswith(prefix):
        raise ValueError(
            f"scientific route has an invalid task-scoped identity: {route_id}"
        )
    route_body = route_id[len(prefix) :]
    route_components = route_body.split(":", maxsplit=2)
    if len(route_components) != 3 or any(
        not component for component in route_components
    ):
        raise ValueError(
            f"scientific route has an invalid task-scoped identity: {route_id}"
        )
    scoped_method_id = route_components[2]
    matches = tuple(
        method_id
        for method_id in eligible_route_ids
        if scoped_method_id in {method_id, f"{method_id}:{effect_scale}"}
    )
    if len(matches) != 1:
        raise ValueError(
            f"scientific route has an invalid or ambiguous canonical method: {route_id}"
        )
    return matches[0]


__all__ = [
    "TrialEvalComponentEvidenceInventoryV1",
    "validate_trialeval_component_evidence",
    "validate_route_component_evidence",
]
