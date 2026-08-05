"""Coverage-report contracts (TrialDevBench run provenance)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrialDevCoverageProgramV1(BaseModel):
    """One declared program in a TrialDevBench run."""

    program_id: str
    scenario_id: str
    objective_id: str


class TrialDevCoverageItemV1(BaseModel):
    """One declared item entry in the TrialDevBench coverage grid."""

    program_id: str
    scenario_id: str
    objective_id: str
    phase_id: str
    endpoint_id: str | None = None
    item_id: str
    task_definition_id: str


class TrialDevCoverageCountsV1(BaseModel):
    """Item count in the coverage report."""

    total_items_present: int = 0


class TrialDevCoverageReportV1(BaseModel):
    """Coverage report written next to TrialDev run outputs."""

    schema_id: Literal["trialagentbench_trialdev_coverage_report_v1"]
    schema_version: Literal[1]
    counts: TrialDevCoverageCountsV1 = Field(default_factory=TrialDevCoverageCountsV1)
    items: list[TrialDevCoverageItemV1] = Field(default_factory=list)
    n_programs: int = 0
    programs: list[TrialDevCoverageProgramV1] = Field(default_factory=list)


__all__ = [
    "TrialDevCoverageProgramV1",
    "TrialDevCoverageItemV1",
    "TrialDevCoverageCountsV1",
    "TrialDevCoverageReportV1",
]
