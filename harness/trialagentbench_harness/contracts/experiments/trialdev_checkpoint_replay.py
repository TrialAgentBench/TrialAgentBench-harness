"""Contracts for matched TrialDev checkpoint replay."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.io.checksums import canonical_payload_sha256

TrialDevCheckpointConditionV1: TypeAlias = Literal[  # noqa: UP040
    "endogenous",
    "context_reset",
    "canonical_state",
]
TrialDevCheckpointMetricV1: TypeAlias = Literal[  # noqa: UP040
    "checkpoint_primary_score",
    "checkpoint_decision_correct",
    "downstream_primary_score",
    "downstream_decision_score",
    "checkpoint_design_validity",
    "checkpoint_phase_evaluation_validity",
    "checkpoint_primary_effect_point_agreement",
    "checkpoint_primary_effect_interval_agreement",
    "checkpoint_safety_evidence_agreement",
    "downstream_design_validity",
    "downstream_phase_evaluation_validity",
    "downstream_primary_effect_point_agreement",
    "downstream_primary_effect_interval_agreement",
    "downstream_safety_evidence_agreement",
]
TrialDevCheckpointContrastIdV1: TypeAlias = Literal[  # noqa: UP040
    "context_reset_minus_endogenous",
    "canonical_state_minus_endogenous",
]


class _FrozenCheckpointExperimentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevCheckpointBlockPlanV1(_FrozenCheckpointExperimentModel):
    """One matched checkpoint block before source hashes are resolved."""

    block_id: str = Field(..., min_length=2)
    program_id: str = Field(..., min_length=1)
    scenario_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    replicate_id: str = Field(..., min_length=1)
    decoding_seed: int = Field(..., ge=0)
    checkpoint_phase_id: Literal["phase1", "phase2", "phase3"]
    checkpoint_step_id: str = Field(..., min_length=1)
    endogenous_program_relative_path: str = Field(..., min_length=1)
    canonical_program_relative_path: str = Field(..., min_length=1)
    canonical_reference_id: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> TrialDevCheckpointBlockPlanV1:
        """Require safe, distinct source programmes."""

        paths = (
            self.endogenous_program_relative_path,
            self.canonical_program_relative_path,
        )
        if any(value.startswith("/") or ".." in value.split("/") for value in paths):
            raise ValueError("Checkpoint source paths must be safe relative paths.")
        if paths[0] == paths[1]:
            raise ValueError("Endogenous and canonical checkpoint sources must be distinct.")
        return self


class TrialDevCheckpointSchedulePlanV1(_FrozenCheckpointExperimentModel):
    """Prospective matched-block plan compiled into an immutable schedule."""

    schema_id: Literal["trialagentbench.trialdev_checkpoint_schedule_plan/v1"] = (
        "trialagentbench.trialdev_checkpoint_schedule_plan/v1"
    )
    experiment_id: str = Field(..., min_length=1)
    blocks: tuple[TrialDevCheckpointBlockPlanV1, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_blocks(self) -> TrialDevCheckpointSchedulePlanV1:
        """Require unique blocks and programme-replicate assignments."""

        block_ids = tuple(block.block_id for block in self.blocks)
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("Checkpoint plan block IDs must be unique.")
        units = tuple((block.program_id, block.replicate_id, block.decoding_seed) for block in self.blocks)
        if len(units) != len(set(units)):
            raise ValueError("Checkpoint plan programme-replicate assignments must be unique.")
        return self


class TrialDevCanonicalCheckpointSourceV1(_FrozenCheckpointExperimentModel):
    """One public-reference checkpoint captured through the TrialDev runner."""

    canonical_reference_id: str = Field(..., min_length=1)
    program_id: str = Field(..., min_length=1)
    phase_id: Literal["phase1", "phase2", "phase3"]
    step_id: str = Field(..., min_length=1)
    program_relative_path: str = Field(..., min_length=1)
    checkpoint_relative_path: str = Field(..., min_length=1)
    checkpoint_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_paths(self) -> TrialDevCanonicalCheckpointSourceV1:
        """Require safe source paths with the checkpoint below its programme."""

        for value in (self.program_relative_path, self.checkpoint_relative_path):
            if value.startswith("/") or ".." in value.split("/"):
                raise ValueError("Canonical checkpoint receipt paths must be safe relative paths.")
        prefix = f"{self.program_relative_path}/checkpoints/"
        if not self.checkpoint_relative_path.startswith(prefix):
            raise ValueError("Canonical checkpoint must be below its recorded programme path.")
        return self


class TrialDevCanonicalCheckpointSourcesV1(_FrozenCheckpointExperimentModel):
    """Checksummed custody receipt for canonical public-reference checkpoints."""

    schema_id: Literal["trialagentbench.canonical_checkpoint_sources/v1"] = (
        "trialagentbench.canonical_checkpoint_sources/v1"
    )
    participant_release_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    recorded_reference_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    records: tuple[TrialDevCanonicalCheckpointSourceV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_records_and_hash(self) -> TrialDevCanonicalCheckpointSourcesV1:
        """Require one source per reference and bind the receipt payload."""

        reference_ids = tuple(row.canonical_reference_id for row in self.records)
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("Canonical checkpoint reference IDs must be unique.")
        sources = tuple((row.program_relative_path, row.checkpoint_relative_path) for row in self.records)
        if len(sources) != len(set(sources)):
            raise ValueError("Canonical checkpoint source paths must be unique.")
        digest = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Canonical checkpoint source receipt checksum mismatch.")
        object.__setattr__(self, "checksum", digest)
        return self


class TrialDevEndogenousCheckpointSourceV1(_FrozenCheckpointExperimentModel):
    """One model-produced pre-response checkpoint and its execution identity."""

    program_id: str = Field(..., min_length=1)
    scenario_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    replicate_id: str = Field(..., min_length=1)
    decoding_seed: int = Field(..., ge=0)
    phase_id: Literal["phase1", "phase2", "phase3"]
    step_id: str = Field(..., min_length=1)
    program_relative_path: str = Field(..., min_length=1)
    checkpoint_relative_path: str = Field(..., min_length=1)
    checkpoint_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    run_identity_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    provider_model: str = Field(..., min_length=1)
    provider_route: str = Field(..., min_length=1)
    procedure_assistance: Literal[
        "output_contract_only",
        "unordered_checklist",
        "ordered_sop",
    ]

    @model_validator(mode="after")
    def validate_paths(self) -> TrialDevEndogenousCheckpointSourceV1:
        """Require safe custody paths with the checkpoint below its programme."""

        for value in (self.program_relative_path, self.checkpoint_relative_path):
            if value.startswith("/") or ".." in value.split("/"):
                raise ValueError("Endogenous checkpoint receipt paths must be safe relative paths.")
        if not self.checkpoint_relative_path.startswith(f"{self.program_relative_path}/checkpoints/"):
            raise ValueError("Endogenous checkpoint must be below its recorded programme path.")
        return self


class TrialDevEndogenousCheckpointSourcesV1(_FrozenCheckpointExperimentModel):
    """Checksummed custody receipt for model-produced checkpoint sources."""

    schema_id: Literal["trialagentbench.endogenous_checkpoint_sources/v1"] = (
        "trialagentbench.endogenous_checkpoint_sources/v1"
    )
    participant_release_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    records: tuple[TrialDevEndogenousCheckpointSourceV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_records_and_hash(self) -> TrialDevEndogenousCheckpointSourcesV1:
        """Require unique planned units and bind the receipt payload."""

        units = tuple(
            (
                row.program_id,
                row.replicate_id,
                row.decoding_seed,
                row.phase_id,
                row.step_id,
            )
            for row in self.records
        )
        if len(units) != len(set(units)):
            raise ValueError("Endogenous checkpoint source units must be unique.")
        paths = tuple(row.program_relative_path for row in self.records)
        if len(paths) != len(set(paths)):
            raise ValueError("Endogenous checkpoint programme paths must be unique.")
        digest = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Endogenous checkpoint source receipt checksum mismatch.")
        object.__setattr__(self, "checksum", digest)
        return self


class TrialDevCheckpointAssignmentV1(_FrozenCheckpointExperimentModel):
    """One condition in a matched checkpoint-replay block."""

    assignment_id: str = Field(..., min_length=2)
    block_id: str = Field(..., min_length=2)
    program_id: str = Field(..., min_length=1)
    scenario_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    replicate_id: str = Field(..., min_length=1)
    decoding_seed: int = Field(..., ge=0)
    condition: TrialDevCheckpointConditionV1
    source_program_relative_path: str = Field(..., min_length=1)
    source_checkpoint_relative_path: str = Field(..., min_length=1)
    source_checkpoint_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    source_run_identity_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    checkpoint_phase_id: Literal["phase1", "phase2", "phase3"]
    checkpoint_step_id: str = Field(..., min_length=1)
    canonical_reference_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_condition(self) -> TrialDevCheckpointAssignmentV1:
        """Require a reference identity only for the canonical-state arm."""

        if (self.condition == "canonical_state") != (self.canonical_reference_id is not None):
            raise ValueError("canonical_reference_id is required exactly for canonical_state.")
        for value in (
            self.source_program_relative_path,
            self.source_checkpoint_relative_path,
        ):
            if value.startswith("/") or ".." in value.split("/"):
                raise ValueError("Checkpoint source paths must be safe relative paths.")
        return self


class TrialDevCheckpointScheduleV1(_FrozenCheckpointExperimentModel):
    """Immutable matched schedule for long-horizon checkpoint contrasts."""

    schema_id: Literal["trialagentbench.trialdev_checkpoint_schedule/v1"] = (
        "trialagentbench.trialdev_checkpoint_schedule/v1"
    )
    experiment_id: str = Field(..., min_length=1)
    participant_release_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    checkpoint_source_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    assignments: tuple[TrialDevCheckpointAssignmentV1, ...] = Field(..., min_length=3)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_blocks_and_hash(self) -> TrialDevCheckpointScheduleV1:
        """Require complete matched triads and bind the schedule."""

        assignment_ids = tuple(row.assignment_id for row in self.assignments)
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("TrialDev checkpoint assignment IDs must be unique.")
        blocks: dict[str, list[TrialDevCheckpointAssignmentV1]] = {}
        for row in self.assignments:
            blocks.setdefault(row.block_id, []).append(row)
        required = {"endogenous", "context_reset", "canonical_state"}
        for block_id, rows in blocks.items():
            if len(rows) != 3 or {row.condition for row in rows} != required:
                raise ValueError(f"Checkpoint block {block_id!r} requires all three conditions.")
            fixed = (
                "program_id",
                "scenario_id",
                "objective_id",
                "replicate_id",
                "decoding_seed",
                "checkpoint_phase_id",
                "checkpoint_step_id",
            )
            if any(len({getattr(row, name) for row in rows}) != 1 for name in fixed):
                raise ValueError(f"Checkpoint block {block_id!r} is not matched.")
            endogenous = next(row for row in rows if row.condition == "endogenous")
            context_reset = next(row for row in rows if row.condition == "context_reset")
            source_fields = (
                "source_program_relative_path",
                "source_checkpoint_relative_path",
                "source_checkpoint_sha256",
                "source_run_identity_sha256",
            )
            if any(getattr(endogenous, name) != getattr(context_reset, name) for name in source_fields):
                raise ValueError(f"Checkpoint block {block_id!r} must reuse one endogenous source for context reset.")
            canonical = next(row for row in rows if row.condition == "canonical_state")
            if all(getattr(endogenous, name) == getattr(canonical, name) for name in source_fields):
                raise ValueError(f"Checkpoint block {block_id!r} requires a distinct canonical public-state source.")
        reference_sources: dict[str, tuple[str, str, str, str]] = {}
        source_references: dict[tuple[str, str, str, str], str] = {}
        for row in self.assignments:
            if row.condition != "canonical_state":
                continue
            assert row.canonical_reference_id is not None
            source = (
                row.source_program_relative_path,
                row.source_checkpoint_relative_path,
                row.source_checkpoint_sha256,
                row.source_run_identity_sha256,
            )
            prior_source = reference_sources.setdefault(row.canonical_reference_id, source)
            if prior_source != source:
                raise ValueError("One canonical reference ID cannot identify multiple checkpoint sources.")
            prior_reference = source_references.setdefault(source, row.canonical_reference_id)
            if prior_reference != row.canonical_reference_id:
                raise ValueError("One canonical checkpoint source cannot have multiple reference IDs.")
        digest = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("TrialDev checkpoint schedule checksum mismatch.")
        object.__setattr__(self, "checksum", digest)
        return self


class TrialDevCheckpointRunConfigV1(_FrozenCheckpointExperimentModel):
    """Live-run identity for one frozen checkpoint schedule."""

    schema_id: Literal["trialagentbench.trialdev_checkpoint_run/v1"] = "trialagentbench.trialdev_checkpoint_run/v1"
    timestamp_utc: datetime
    schedule_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    participant_release_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    checkpoint_source_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    model: str = Field(..., min_length=1)
    provider: Literal["openai", "openai_responses", "openrouter"]
    openrouter_provider: str | None = Field(default=None, min_length=1)
    temperature: float
    max_tokens: int = Field(..., ge=1)
    max_turns_per_step: int = Field(..., ge=1)
    request_timeout_seconds: float = Field(..., gt=0.0, le=900.0)
    program_watchdog_seconds: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_routing(self) -> TrialDevCheckpointRunConfigV1:
        """Require one exact OpenRouter route when applicable."""

        if (self.provider == "openrouter") != (self.openrouter_provider is not None):
            raise ValueError("OpenRouter checkpoint experiments require one exact upstream-provider pin.")
        return self


class TrialDevCheckpointQualityV1(_FrozenCheckpointExperimentModel):
    """Noncompensatory analysis and design coordinates after one checkpoint."""

    design_validity: float = Field(..., ge=0.0, le=1.0)
    phase_evaluation_validity: float = Field(..., ge=0.0, le=1.0)
    primary_effect_point_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    primary_effect_interval_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    safety_evidence_agreement: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_effect_pair(self) -> TrialDevCheckpointQualityV1:
        """Require point and interval agreement to share applicability."""

        if (self.primary_effect_point_agreement is None) != (self.primary_effect_interval_agreement is None):
            raise ValueError("Checkpoint primary-effect point and interval agreement must be present together.")
        return self


class TrialDevCheckpointScoreRowV1(_FrozenCheckpointExperimentModel):
    """Checkpoint-local and downstream-only outcomes for one assignment."""

    assignment_id: str
    block_id: str
    program_id: str
    scenario_id: str
    objective_id: str
    replicate_id: str
    condition: TrialDevCheckpointConditionV1
    checkpoint_phase_id: Literal["phase1", "phase2", "phase3"]
    checkpoint_primary_score: float = Field(..., ge=0.0, le=1.0)
    checkpoint_decision_correct: float = Field(..., ge=0.0, le=1.0)
    downstream_primary_score: float = Field(..., ge=0.0, le=1.0)
    downstream_decision_score: float = Field(..., ge=0.0, le=1.0)
    downstream_phase_count: int = Field(..., ge=0)
    checkpoint_quality: TrialDevCheckpointQualityV1
    downstream_quality: TrialDevCheckpointQualityV1
    completed: bool


class TrialDevCheckpointObservedContrastV1(_FrozenCheckpointExperimentModel):
    """Observed matched contrast for a bounded pilot without inference."""

    metric: TrialDevCheckpointMetricV1
    contrast_id: TrialDevCheckpointContrastIdV1
    n_blocks: int = Field(..., gt=0)
    estimate: float


class TrialDevCheckpointDescriptiveV1(_FrozenCheckpointExperimentModel):
    """Checksummed checkpoint rows and observed pilot contrasts."""

    schema_id: Literal["trialagentbench.trialdev_checkpoint_descriptive/v1"] = (
        "trialagentbench.trialdev_checkpoint_descriptive/v1"
    )
    schedule_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    rows: tuple[TrialDevCheckpointScoreRowV1, ...] = Field(..., min_length=3)
    observed_contrasts: tuple[TrialDevCheckpointObservedContrastV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_rows_and_hash(self) -> TrialDevCheckpointDescriptiveV1:
        """Require complete triads and bind the descriptive payload."""

        _validate_complete_checkpoint_blocks(self.rows)
        digest = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("TrialDev checkpoint descriptive checksum mismatch.")
        object.__setattr__(self, "checksum", digest)
        return self


class TrialDevCheckpointContrastV1(_FrozenCheckpointExperimentModel):
    """One paired checkpoint contrast with crossed-cluster uncertainty."""

    metric: TrialDevCheckpointMetricV1
    contrast_id: TrialDevCheckpointContrastIdV1
    n_blocks: int = Field(..., gt=0)
    n_scenarios: int = Field(..., gt=1)
    n_replicates: int = Field(..., gt=1)
    estimate: float
    interval_low: float
    interval_high: float


class TrialDevCheckpointAnalysisV1(_FrozenCheckpointExperimentModel):
    """Checksummed checkpoint rows and predeclared paired contrasts."""

    schema_id: Literal["trialagentbench.trialdev_checkpoint_analysis/v1"] = (
        "trialagentbench.trialdev_checkpoint_analysis/v1"
    )
    schedule_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    confidence_level: float = Field(..., gt=0.0, lt=1.0)
    bootstrap_resamples: int = Field(..., ge=1000)
    bootstrap_seed: int = Field(..., ge=0)
    rows: tuple[TrialDevCheckpointScoreRowV1, ...] = Field(..., min_length=3)
    contrasts: tuple[TrialDevCheckpointContrastV1, ...] = Field(..., min_length=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_rows_and_hash(self) -> TrialDevCheckpointAnalysisV1:
        """Require complete blocks and bind the analysis payload."""

        _validate_complete_checkpoint_blocks(self.rows)
        digest = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("TrialDev checkpoint analysis checksum mismatch.")
        object.__setattr__(self, "checksum", digest)
        return self


def _validate_complete_checkpoint_blocks(rows: tuple[TrialDevCheckpointScoreRowV1, ...]) -> None:
    """Require one complete matched triad in every observed block."""

    blocks: dict[str, list[TrialDevCheckpointScoreRowV1]] = {}
    for row in rows:
        blocks.setdefault(row.block_id, []).append(row)
    required = {"endogenous", "context_reset", "canonical_state"}
    if any(len(block) != 3 or {row.condition for row in block} != required for block in blocks.values()):
        raise ValueError("Checkpoint rows must contain complete matched triads.")


__all__ = [
    "TrialDevCanonicalCheckpointSourceV1",
    "TrialDevCanonicalCheckpointSourcesV1",
    "TrialDevCheckpointAnalysisV1",
    "TrialDevCheckpointAssignmentV1",
    "TrialDevCheckpointBlockPlanV1",
    "TrialDevCheckpointConditionV1",
    "TrialDevCheckpointContrastIdV1",
    "TrialDevCheckpointContrastV1",
    "TrialDevCheckpointDescriptiveV1",
    "TrialDevCheckpointMetricV1",
    "TrialDevCheckpointObservedContrastV1",
    "TrialDevCheckpointQualityV1",
    "TrialDevCheckpointRunConfigV1",
    "TrialDevCheckpointScheduleV1",
    "TrialDevCheckpointSchedulePlanV1",
    "TrialDevCheckpointScoreRowV1",
]
