"""Contracts for public non-score-bearing TrialDev worked programmes."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.trialdev.metrics import TrialDevProgrammeAssessmentV1
from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevActionSelectionV1,
    TrialDevObservationalDecisionEvidenceV1,
    TrialDevProgrammeStateRecordV1,
    TrialDevRandomizedDecisionEvidenceV1,
    TrialDevSupportedActionSetV1,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevWorkedCheckpointV1(_StrictModel):
    """One fully inspectable evidence-to-transition step."""

    state_before: TrialDevProgrammeStateRecordV1 = Field(discriminator="stream_id")
    decision_evidence: TrialDevObservationalDecisionEvidenceV1 | TrialDevRandomizedDecisionEvidenceV1
    supported_action_set: TrialDevSupportedActionSetV1
    selected_action: TrialDevActionSelectionV1
    state_after: TrialDevProgrammeStateRecordV1 = Field(discriminator="stream_id")
    selection_accepted: Literal[True] = True

    @model_validator(mode="after")
    def validate_step(self) -> Self:
        """Bind the selected action and adjacent immutable states."""

        if self.state_before.checksum != self.supported_action_set.state_checksum:
            raise ValueError("Worked supported actions must bind the preceding state.")
        if self.selected_action.state_checksum != self.state_before.checksum:
            raise ValueError("Worked selection must bind the preceding state.")
        selected = (
            self.selected_action.action_id,
            self.selected_action.target_asset_id,
            self.selected_action.reserve_asset_id,
        )
        supported = {
            (item.action_id, item.target_asset_id, item.reserve_asset_id)
            for item in self.supported_action_set.supported_actions
        }
        if selected not in supported:
            raise ValueError("Worked selection must belong to the supported action set.")
        if self.state_after.previous_state_checksum != self.state_before.checksum:
            raise ValueError("Worked transition must preserve the preceding state checksum.")
        if len(self.state_after.history) != len(self.state_before.history) + 1:
            raise ValueError("Worked transition must append exactly one history record.")
        return self


class TrialDevWorkedProgrammeV1(_StrictModel):
    """One complete worked programme and its denominator-preserving assessment."""

    programme_id: str = Field(min_length=1)
    stream_id: Literal["single_asset_development", "bounded_portfolio_reallocation"]
    qualification_seed: int = Field(ge=0)
    checkpoints: tuple[TrialDevWorkedCheckpointV1, ...] = Field(min_length=1)
    assessment: TrialDevProgrammeAssessmentV1

    @model_validator(mode="after")
    def validate_programme(self) -> Self:
        """Require a contiguous terminal trajectory in one declared stream."""

        if self.assessment.programme_id != self.programme_id or self.assessment.stream_id != self.stream_id:
            raise ValueError("Worked programme and assessment identities must agree.")
        if any(
            step.state_before.stream_id != self.stream_id or step.state_after.stream_id != self.stream_id
            for step in self.checkpoints
        ):
            raise ValueError("Worked checkpoints must use the programme stream.")
        for previous, current in zip(self.checkpoints, self.checkpoints[1:], strict=False):
            if previous.state_after.checksum != current.state_before.checksum:
                raise ValueError("Worked programme checkpoints must form a contiguous state chain.")
        if self.checkpoints[-1].state_after.terminal_disposition == "active":
            raise ValueError("A worked programme must end in a terminal state.")
        return self


class TrialDevWorkedPackageV1(_StrictModel):
    """Public worked examples for both TrialDev streams."""

    schema_id: Literal["trialagentbench.trialdev_worked_programmes/v1"] = (
        "trialagentbench.trialdev_worked_programmes/v1"
    )
    purpose: Literal["non_score_bearing_scientific_demonstration"] = "non_score_bearing_scientific_demonstration"
    programmes: tuple[TrialDevWorkedProgrammeV1, TrialDevWorkedProgrammeV1]

    @model_validator(mode="after")
    def validate_streams(self) -> Self:
        """Require exactly one example for each public stream."""

        if {item.stream_id for item in self.programmes} != {
            "single_asset_development",
            "bounded_portfolio_reallocation",
        }:
            raise ValueError("Worked package requires one programme for each TrialDev stream.")
        return self


__all__ = [
    "TrialDevWorkedCheckpointV1",
    "TrialDevWorkedPackageV1",
    "TrialDevWorkedProgrammeV1",
]
