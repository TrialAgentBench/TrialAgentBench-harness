"""Participant submission and grade contracts for TrialDev portfolios."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevCheckpointIdV1,
    TrialDevCheckpointOutcomeV1,
    TrialDevObservationalDecisionEvidenceV1,
    TrialDevPortfolioActionSelectionV1,
    TrialDevRandomizedDecisionEvidenceV1,
    TrialDevSupportedActionSetV1,
)
from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TrialDevScientificAssessmentV1,
    unassessed_scientific_assessment_v1,
)

TrialDevPortfolioDecisionEvidenceV1 = Annotated[
    TrialDevObservationalDecisionEvidenceV1 | TrialDevRandomizedDecisionEvidenceV1,
    Field(discriminator="schema_id"),
]


class TrialDevScheduledStudyV1(BaseModel):
    """One next-stage study design selected with a programme action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(..., min_length=1)
    phase_id: Literal["phase1", "phase2", "phase3"]
    design_cell_id: str = Field(..., min_length=1)


class TrialDevPortfolioCheckpointSubmissionV1(BaseModel):
    """Complete analysis, action, and next-study selection at one checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_portfolio_checkpoint_submission/v1"] = (
        "trialagentbench.trialdev_portfolio_checkpoint_submission/v1"
    )
    state_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    decision_evidence: TrialDevPortfolioDecisionEvidenceV1
    selected_action: TrialDevPortfolioActionSelectionV1
    scheduled_studies: tuple[TrialDevScheduledStudyV1, ...] = Field(
        default_factory=tuple,
        description=(
            "Studies initiated by the selected action: two phase-1 studies for lead-reserve selection, "
            "one study for a later advance or reserve promotion, and none for a terminal action."
        ),
    )

    @model_validator(mode="after")
    def validate_submission(self) -> Self:
        """Bind all records and require the action-compatible study shape."""

        if (
            self.decision_evidence.state_checksum != self.state_checksum
            or self.selected_action.state_checksum != self.state_checksum
        ):
            raise ValueError("Portfolio submission records must bind to one current state.")
        if self.selected_action.analysis_method_id != self.decision_evidence.analysis_method_id:
            raise ValueError("Portfolio action and decision evidence must use one analysis method.")
        action = self.selected_action.action_id
        expected_phase = {
            "select_lead_and_reserve": "phase1",
            "advance_lead_to_proof_of_concept": "phase2",
            "promote_reserve_to_proof_of_concept": "phase2",
            "advance_active_to_confirmation": "phase3",
        }.get(action)
        expected_count = 2 if action == "select_lead_and_reserve" else int(expected_phase is not None)
        if len(self.scheduled_studies) != expected_count:
            raise ValueError(f"Portfolio action {action!r} requires {expected_count} scheduled studies.")
        if expected_phase is not None and any(item.phase_id != expected_phase for item in self.scheduled_studies):
            raise ValueError(f"Portfolio action {action!r} must schedule {expected_phase} evidence.")
        assets = tuple(item.asset_id for item in self.scheduled_studies)
        if len(assets) != len(set(assets)):
            raise ValueError("Portfolio scheduled studies must identify unique assets.")
        return self


class TrialDevPortfolioCheckpointGradeV1(BaseModel):
    """Independently reproducible grade for one portfolio checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_portfolio_checkpoint_grade/v1"] = (
        "trialagentbench.trialdev_portfolio_checkpoint_grade/v1"
    )
    checkpoint_id: TrialDevCheckpointIdV1
    state_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    evidence_numeric_agreement: bool
    numeric_disagreement_paths: tuple[str, ...] = ()
    provenance_valid: bool
    supported_action_set: TrialDevSupportedActionSetV1
    selected_action_supported: bool
    scheduled_designs_valid: bool
    scientific_assessment: TrialDevScientificAssessmentV1 = Field(default_factory=unassessed_scientific_assessment_v1)
    outcome: TrialDevCheckpointOutcomeV1

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Require the outcome axes to summarize the independent checks."""

        if self.evidence_numeric_agreement == bool(self.numeric_disagreement_paths):
            raise ValueError("Numeric agreement must be true exactly when no numeric field paths disagree.")
        if len(set(self.numeric_disagreement_paths)) != len(self.numeric_disagreement_paths):
            raise ValueError("Numeric disagreement field paths must be unique.")
        if any(not path.strip() for path in self.numeric_disagreement_paths):
            raise ValueError("Numeric disagreement field paths must not be empty.")
        if self.outcome.reach_status != "reached" or self.outcome.execution_status != "completed":
            raise ValueError("A portfolio checkpoint grade represents one completed reached submission.")
        if self.outcome.submission_status != "accepted":
            raise ValueError("A graded portfolio checkpoint has passed structural submission validation.")
        if self.outcome.analysis_status not in {"estimable", "non_estimable"}:
            raise ValueError("A graded portfolio checkpoint requires its estimability status.")
        return self


class TrialDevPortfolioRunSummaryV1(BaseModel):
    """Immutable inventory and terminal status for one portfolio programme run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_portfolio_run_summary/v1"] = (
        "trialagentbench.trialdev_portfolio_run_summary/v1"
    )
    programme_id: str = Field(..., min_length=1)
    scenario_id: str = Field(..., min_length=1)
    objective_id: str = Field(..., min_length=1)
    resource_budget_units: Literal[8, 10]
    participant_view_checksum: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    release_source_identity: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    execution_status: Literal["completed", "model_noncompletion", "infrastructure_failure", "run_stopped"]
    terminal_disposition: Literal["withheld", "stopped", "success", "failure", "inconclusive"] | None = None
    reached_checkpoint_ids: tuple[TrialDevCheckpointIdV1, ...] = ()
    state_relative_paths: tuple[str, ...] = ()
    submission_relative_paths: tuple[str, ...] = ()
    grade_relative_paths: tuple[str, ...] = ()
    wall_seconds_total: float = Field(..., ge=0.0)
    submission_attempts: int = Field(..., ge=0)
    correction_count: int = Field(..., ge=0)
    agent_turns: int = Field(..., ge=0)
    execute_code_calls: int = Field(..., ge=0)
    inspect_data_calls: int = Field(..., ge=0)
    provider_calls: int = Field(..., ge=0)
    provider_elapsed_seconds: float = Field(..., ge=0.0)
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    provider_reported_usd: float | None = Field(default=None, ge=0.0)
    error: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        """Require aligned accepted artifacts for completed programmes."""

        if len(self.submission_relative_paths) != len(self.grade_relative_paths):
            raise ValueError("Portfolio submissions and grades must be aligned.")
        if self.correction_count > self.submission_attempts:
            raise ValueError("Portfolio corrections cannot exceed submission attempts.")
        state_count = len(self.state_relative_paths)
        grade_count = len(self.grade_relative_paths)
        complete_state_history = state_count == grade_count + 1
        pending_final_transition = state_count == grade_count
        if self.state_relative_paths and not complete_state_history:
            if self.execution_status not in {"infrastructure_failure", "run_stopped"} or not pending_final_transition:
                raise ValueError("Portfolio state history must include initial and post-decision states.")
        if self.execution_status == "completed":
            if self.terminal_disposition is None or self.error is not None:
                raise ValueError("A completed portfolio run requires one terminal disposition and no error.")
            if not self.reached_checkpoint_ids:
                raise ValueError("A completed portfolio run must reach at least one checkpoint.")
        elif self.terminal_disposition is not None or self.error is None:
            raise ValueError("An incomplete portfolio run requires an error and no terminal disposition.")
        return self


class TrialDevPortfolioSubmissionAttemptV1(BaseModel):
    """One exact checkpoint submission attempt and its transport outcome.

    A structurally valid submission is accepted for programme transition even
    when its scientific assessment is incomplete. Scientific disagreement is
    a measured result, not a transport error.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_portfolio_submission_attempt/v1"] = (
        "trialagentbench.trialdev_portfolio_submission_attempt/v1"
    )
    checkpoint_id: TrialDevCheckpointIdV1
    attempt_index: int = Field(..., ge=1)
    transport_name: str = Field(..., min_length=1)
    status: Literal["contract_rejected", "accepted"]
    submitted_payload: JsonValue
    validation_error: str | None = Field(default=None, min_length=1)
    grade: TrialDevPortfolioCheckpointGradeV1 | None = None

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        """Bind each status to exactly the evidence available at that boundary."""

        if self.status == "contract_rejected":
            if self.validation_error is None or self.grade is not None:
                raise ValueError("A contract rejection requires only its validation error.")
        else:
            if self.validation_error is not None or self.grade is None:
                raise ValueError("A graded attempt requires its grade and no validation error.")
        return self


__all__ = [
    "TrialDevPortfolioSubmissionAttemptV1",
    "TrialDevPortfolioCheckpointGradeV1",
    "TrialDevPortfolioCheckpointSubmissionV1",
    "TrialDevPortfolioDecisionEvidenceV1",
    "TrialDevPortfolioRunSummaryV1",
    "TrialDevScheduledStudyV1",
]
