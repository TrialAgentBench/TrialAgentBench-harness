"""Contracts for the TrialDev observational specification experiment."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.adapters.trialdev_share import (
    TrialDevPublicObservationalMethodSpecV1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256

TrialDevObservationalSpecificationConditionV1: TypeAlias = Literal[  # noqa: UP040
    "open_selection",
    "prespecified_execution",
]


class _FrozenExperimentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevObservationalSpecificationAssignmentV1(_FrozenExperimentModel):
    """One randomized arm within a method-stratified paired block."""

    assignment_id: str = Field(..., min_length=2)
    pair_id: str = Field(..., min_length=2)
    program_id: str = Field(..., min_length=1)
    scenario_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    replicate_id: str = Field(..., min_length=1)
    decoding_seed: int = Field(..., ge=0)
    condition: TrialDevObservationalSpecificationConditionV1
    method_catalog_checksum: str = Field(..., min_length=64, max_length=64)
    method_specification: TrialDevPublicObservationalMethodSpecV1


class TrialDevObservationalSpecificationScheduleV1(_FrozenExperimentModel):
    """Immutable participant-only schedule for analysis selection versus execution."""

    schema_id: Literal["trialagentbench.trialdev_observational_specification_schedule/v1"] = (
        "trialagentbench.trialdev_observational_specification_schedule/v1"
    )
    experiment_id: str = Field(..., min_length=1)
    participant_release_sha256: str = Field(..., min_length=64, max_length=64)
    randomization_seed: int = Field(..., ge=0)
    assignments: tuple[TrialDevObservationalSpecificationAssignmentV1, ...] = Field(..., min_length=2)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_pairs_and_hash(self) -> TrialDevObservationalSpecificationScheduleV1:
        """Require exact paired arms and bind the canonical schedule."""

        assignment_ids = tuple(row.assignment_id for row in self.assignments)
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("TrialDev observational specification assignment IDs must be unique.")
        pairs: dict[str, list[TrialDevObservationalSpecificationAssignmentV1]] = {}
        for row in self.assignments:
            pairs.setdefault(row.pair_id, []).append(row)
        required = {"open_selection", "prespecified_execution"}
        for pair_id, rows in pairs.items():
            if len(rows) != 2 or {row.condition for row in rows} != required:
                raise ValueError(f"TrialDev observational specification pair {pair_id!r} requires both arms.")
            left, right = rows
            fixed = (
                "program_id",
                "scenario_id",
                "objective_id",
                "replicate_id",
                "decoding_seed",
                "method_catalog_checksum",
                "method_specification",
            )
            if any(getattr(left, name) != getattr(right, name) for name in fixed):
                raise ValueError(f"TrialDev observational specification pair {pair_id!r} is not matched.")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        digest = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("TrialDev observational specification schedule checksum mismatch.")
        object.__setattr__(self, "checksum", digest)
        return self


class TrialDevObservationalSpecificationRunConfigV1(_FrozenExperimentModel):
    """Live-run identity for one frozen observational specification schedule."""

    schema_id: Literal["trialagentbench.trialdev_observational_specification_run/v1"] = (
        "trialagentbench.trialdev_observational_specification_run/v1"
    )
    timestamp_utc: datetime
    schedule_checksum: str = Field(..., min_length=64, max_length=64)
    participant_release_sha256: str = Field(..., min_length=64, max_length=64)
    model: str = Field(..., min_length=1)
    provider: Literal["openai", "openai_responses", "openrouter"]
    openrouter_provider: str | None = Field(default=None, min_length=1)
    temperature: float
    max_tokens: int = Field(..., ge=1)
    max_turns_per_step: int = Field(..., ge=1)
    request_timeout_seconds: float = Field(..., gt=0.0, le=900.0)
    program_watchdog_seconds: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_routing(self) -> TrialDevObservationalSpecificationRunConfigV1:
        """Require an exact OpenRouter upstream only for OpenRouter."""

        if (self.provider == "openrouter") != (self.openrouter_provider is not None):
            raise ValueError("OpenRouter observational experiments require one exact upstream-provider pin.")
        return self


class TrialDevObservationalSpecificationScoreRowV1(_FrozenExperimentModel):
    """One method-stratified observational score."""

    assignment_id: str
    pair_id: str
    scenario_id: str
    objective_id: str
    replicate_id: str
    condition: TrialDevObservationalSpecificationConditionV1
    method_route_id: str
    primary_score: float = Field(..., ge=0.0, le=1.0)
    analysis_valid: bool
    analysis_score: float = Field(..., ge=0.0, le=1.0)
    ranking_score: float = Field(..., ge=0.0, le=1.0)


class TrialDevObservationalSpecificationContrastV1(_FrozenExperimentModel):
    """Prespecified-minus-open paired contrast."""

    metric: Literal["primary_score", "analysis_valid", "analysis_score", "ranking_score"]
    method_route_id: str | None = None
    n_pairs: int = Field(..., gt=0)
    n_scenarios: int = Field(..., gt=1)
    n_replicates: int = Field(..., gt=1)
    estimate: float
    interval_low: float
    interval_high: float


class TrialDevObservationalSpecificationAnalysisV1(_FrozenExperimentModel):
    """Checksummed score rows and crossed-cluster paired contrasts."""

    schema_id: Literal["trialagentbench.trialdev_observational_specification_analysis/v1"] = (
        "trialagentbench.trialdev_observational_specification_analysis/v1"
    )
    schedule_checksum: str = Field(..., min_length=64, max_length=64)
    confidence_level: float = Field(..., gt=0.0, lt=1.0)
    bootstrap_resamples: int = Field(..., ge=1000)
    bootstrap_seed: int = Field(..., ge=0)
    rows: tuple[TrialDevObservationalSpecificationScoreRowV1, ...] = Field(..., min_length=1)
    contrasts: tuple[TrialDevObservationalSpecificationContrastV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def hash_analysis(self) -> TrialDevObservationalSpecificationAnalysisV1:
        """Bind the complete analysis artifact."""

        payload = self.model_dump(mode="json", exclude={"checksum"})
        digest = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("TrialDev observational specification analysis checksum mismatch.")
        object.__setattr__(self, "checksum", digest)
        return self


__all__ = [
    "TrialDevObservationalSpecificationAssignmentV1",
    "TrialDevObservationalSpecificationAnalysisV1",
    "TrialDevObservationalSpecificationContrastV1",
    "TrialDevObservationalSpecificationConditionV1",
    "TrialDevObservationalSpecificationRunConfigV1",
    "TrialDevObservationalSpecificationScheduleV1",
    "TrialDevObservationalSpecificationScoreRowV1",
]
