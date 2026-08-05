"""Typed TrialEval score artifacts and analysis rows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.contracts.core.config import ReasoningEffortV1
from trialagentbench_harness.grading.models import GradeRecordV1
from trialagentbench_harness.trialeval.planning import PlanningAssessmentV1


class TrialEvalItemScoresV1(BaseModel):
    """Schema-bearing score wrapper persisted with one TrialEval item."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval.item_scores/v1"] = "trialagentbench.trialeval.item_scores/v1"
    schema_version: Literal[1] = 1
    data_format: Literal["trialagentbench_v1"] = "trialagentbench_v1"
    item_id: str
    task_id: str | None = None
    trial_name: str | None = None
    design_tier: str = ""
    design_subtype: str = ""
    assumption_tier: str = ""
    context_tier: str = ""
    estimand_mode: str = ""
    model: str
    output_mode: str
    turns_used: int = 0
    agent_status: str = ""
    credit_eligible_route_count: int = Field(ge=1)
    grade: GradeRecordV1
    planning: PlanningAssessmentV1

    @model_validator(mode="after")
    def _score_matches_item(self) -> TrialEvalItemScoresV1:
        if self.grade.item_id != self.item_id:
            raise ValueError("TrialEval grade item identity must match its score wrapper.")
        if self.planning.applicable and not self.grade.route_match:
            raise ValueError("Applicable planning requires a matched analysis route.")
        return self


class TrialEvalGradedItemRowV1(BaseModel):
    """One canonical graded item used by trace and downstream analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval_graded_item_row/v1"] = "trialagentbench.trialeval_graded_item_row/v1"
    model_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    condition_id: str = Field(min_length=1)
    request_replicate_id: str = Field(min_length=1)
    reasoning_effort: ReasoningEffortV1 | None = None
    reasoning_capability_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    procedure_assistance: Literal["output_contract_only"]
    analysis_specification: Literal["protocol_only", "locked_sap"]
    trial_name: str = ""
    design_tier: str = Field(min_length=1)
    design_subtype: str = Field(min_length=1)
    assumption_tier: str = Field(min_length=1)
    context_tier: str = Field(min_length=1)
    estimand_mode: str = ""
    output_mode: str = ""
    agent_status: str = ""
    turns_used: int = Field(ge=0)
    credit_eligible_route_count: int = Field(ge=1)
    grade: GradeRecordV1
    planning: PlanningAssessmentV1
    source_json_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def _reasoning_identity_is_complete(self) -> TrialEvalGradedItemRowV1:
        if (self.reasoning_effort is None) != (self.reasoning_capability_sha256 is None):
            raise ValueError("Reasoning effort and capability identity must be present together.")
        return self

    @model_validator(mode="after")
    def _grade_matches_item(self) -> TrialEvalGradedItemRowV1:
        if self.grade.item_id != self.task_id:
            raise ValueError("TrialEval grade item identity must match the analysis task.")
        return self


__all__ = ["TrialEvalGradedItemRowV1", "TrialEvalItemScoresV1"]
