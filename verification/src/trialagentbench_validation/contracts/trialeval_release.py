"""Strict schemas for the TrialEval evaluator domains used in validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ReleaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ItemFactorAssignmentV1(_ReleaseModel):
    design_tier: Literal["D1", "D2", "D3", "D4"] = Field(alias="design_archetype")
    design_subtype: str = Field(min_length=1)
    assumption_tier: Literal["A1", "A2", "A3", "A4"] = Field(alias="assumption_regime")
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"] = Field(
        alias="context_configuration"
    )
    data_preparation: Literal[
        "analysis_ready", "raw_domains", "raw_domains_declared_defect"
    ]
    analysis_specification: Literal["locked_sap", "protocol_only"]
    procedure_assistance: Literal[
        "output_contract_only", "unordered_checklist", "ordered_sop"
    ]
    response_interface: Literal["structured", "narrative"]
    regime_cell_id: str = Field(min_length=1)
    evaluation_series_id: str = Field(min_length=1)


class ItemIndexEntryV1(_ReleaseModel):
    task_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    generation_seed: int = Field(ge=1, le=2**31 - 2)
    base_case_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    factors: ItemFactorAssignmentV1
    scoring_row_offset: int = Field(ge=0)
    scoring_row_count: int = Field(ge=0)
    reconstruction_row_offset: int = Field(ge=0)
    reconstruction_row_count: int = Field(ge=0)
    data_integrity_reference_row_offset: int = Field(ge=0)
    data_integrity_reference_row_count: int = Field(ge=0)

    @property
    def design_tier(self) -> str:
        return self.factors.design_tier

    @property
    def assumption_tier(self) -> str:
        return self.factors.assumption_tier

    @property
    def context_tier(self) -> str:
        return self.factors.context_tier


class ItemIndexV1(_ReleaseModel):
    schema_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    checksum: str = Field(min_length=64, max_length=64)
    entries: tuple[ItemIndexEntryV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_tasks(self) -> ItemIndexV1:
        task_ids = tuple(entry.task_id for entry in self.entries)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("item index task IDs must be unique")
        return self


__all__ = [
    "ItemIndexEntryV1",
    "ItemFactorAssignmentV1",
    "ItemIndexV1",
]
