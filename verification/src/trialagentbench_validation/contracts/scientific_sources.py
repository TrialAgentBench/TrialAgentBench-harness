"""Independent contract for the public scientific-source registry."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.contracts.release_scope import TrialEvalReleaseScopeV1
from trialagentbench_validation.contracts.scientific_inventory import (
    TrialEvalScientificConstructionInventoryV1,
)


class _ReleaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScientificSourceV1(_ReleaseModel):
    """One independently readable scientific source record."""

    source_id: str = Field(pattern=r"^TAB-SRC-[0-9]{3}$")
    title: str = Field(min_length=1)
    source_type: Literal[
        "consensus_framework",
        "journal_article",
        "regulatory_guidance",
        "regulatory_review",
        "regulatory_template",
        "reporting_guideline",
        "statistical_analysis_plan",
        "technical_standard",
    ]
    evidence_role: Literal[
        "design_consensus",
        "methods_evidence",
        "normative_principle",
        "operational_exemplar",
        "regulatory_precedent",
    ]
    canonical_id: str = Field(min_length=1)
    canonical_url: str = Field(pattern=r"^https://")
    access_class: Literal[
        "official_public",
        "open_access",
        "citation_only",
        "public_standard_reference",
    ]
    verification_status: Literal["verified", "unresolved"]
    scope_note: str = Field(min_length=1)


class ScientificSourceRegistryV1(_ReleaseModel):
    """Checksum-bound evidence identities from the verification split."""

    schema_id: Literal["trialagentbench.scientific_source_registry/v1"]
    sources: tuple[ScientificSourceV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_and_checksummed(self) -> ScientificSourceRegistryV1:
        source_ids = tuple(source.source_id for source in self.sources)
        if source_ids != tuple(sorted(source_ids)) or len(set(source_ids)) != len(
            source_ids
        ):
            raise ValueError("scientific sources must be sorted and unique")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != self.checksum:
            raise ValueError("scientific-source registry checksum mismatch")
        return self


def validate_scientific_source_coverage(
    *,
    registry: ScientificSourceRegistryV1,
    inventory: TrialEvalScientificConstructionInventoryV1,
    release_scope: TrialEvalReleaseScopeV1,
) -> None:
    """Require every scientific decision to resolve to a public source record."""

    verified = {
        source.source_id
        for source in registry.sources
        if source.verification_status == "verified"
    }
    used = {
        source_id
        for row in inventory.rows
        for source_id in (
            *row.normative_source_ids,
            *row.method_source_ids,
            *row.precedent_source_ids,
        )
    }
    used.update(
        source_id
        for series in release_scope.components.evaluation_series
        for source_id in series.source_ids
    )
    missing = sorted(used - verified)
    if missing:
        raise ValueError(
            f"release decisions cite absent or unverified scientific sources: {missing!r}"
        )


__all__ = [
    "ScientificSourceRegistryV1",
    "ScientificSourceV1",
    "validate_scientific_source_coverage",
]
