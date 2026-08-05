"""TrialDevBench release program-grid contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.trialdev.trialdev_recoverability import TrialDevObjectiveIdV1


class TrialDevProgramGridEntryV1(BaseModel):
    """One canonical scoreable TrialDev program."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    program_id: str = Field(..., min_length=1)
    scenario_key: str = Field(..., min_length=1)
    scenario_semantic_id: str = Field(..., min_length=1)
    objective_id: TrialDevObjectiveIdV1

    @model_validator(mode="after")
    def validate_program_id(self) -> TrialDevProgramGridEntryV1:
        """Ensure program IDs use the scenario-key/objective convention."""

        expected = f"{self.scenario_key}__{self.objective_id}"
        if self.program_id != expected:
            raise ValueError(f"program_id must equal {expected!r}.")
        return self


class TrialDevProgramGridV1(BaseModel):
    """Release-declared TrialDev program denominator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench_trialdev_program_grid_v1"] = "trialagentbench_trialdev_program_grid_v1"
    schema_version: Literal[1] = 1
    release_id: str = Field(..., min_length=1)
    programs: tuple[TrialDevProgramGridEntryV1, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_grid(self) -> TrialDevProgramGridV1:
        """Reject duplicate program or scenario/objective contexts."""

        program_ids = [program.program_id for program in self.programs]
        if len(program_ids) != len(set(program_ids)):
            raise ValueError("TrialDev program grid contains duplicate program_id values.")
        contexts = [(program.scenario_key, program.objective_id) for program in self.programs]
        if len(contexts) != len(set(contexts)):
            raise ValueError("TrialDev program grid contains duplicate scenario/objective contexts.")
        return self

    @property
    def program_ids(self) -> tuple[str, ...]:
        """Return canonical program IDs in grid order."""

        return tuple(program.program_id for program in self.programs)


__all__ = ["TrialDevProgramGridEntryV1", "TrialDevProgramGridV1"]
