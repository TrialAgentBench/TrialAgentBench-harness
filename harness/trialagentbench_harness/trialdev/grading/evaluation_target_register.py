"""Evaluation-target loading and lane-native scoring utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from trialagentbench_harness.contracts.trialdev.trialdev_evaluation_target_register import (
    TrialDevEvaluationTargetRegisterRecordV1,
    load_trialdev_evaluation_target_register_records,
)
from trialagentbench_harness.trialdev.grading.models import TrialDevelopmentLaneScoreRecordV1

TrialDevScoreDerivation = Literal[
    "literal_target",
    "numeric_diagnostic",
    "public_evidence_action",
]

_NUMERIC_DIAGNOSTIC_OVERRIDE_CONTEXTS: frozenset[tuple[str | None, str]] = frozenset(
    {
        (None, "phase_design"),
        (None, "phase_analysis"),
        (None, "route_timing"),
        ("final_decision", "final_recommendation"),
    }
)


class TrialDevEvaluationTargetRegisterIndex:
    """Strict index over one scenario's evaluation-target register records."""

    def __init__(self, records: tuple[TrialDevEvaluationTargetRegisterRecordV1, ...]) -> None:
        """Create an index and reject duplicate scoring contexts."""

        self._records = records
        self._by_key: dict[tuple[str, str, str, str], TrialDevEvaluationTargetRegisterRecordV1] = {}
        for record in records:
            key = (
                str(record.phase_id),
                str(record.program_objective_id),
                str(record.phase_scoring_objective_id),
                str(record.lane_id),
            )
            if key in self._by_key:
                raise ValueError(f"duplicate TrialDev evaluation-target register context: {key!r}")
            self._by_key[key] = record

    @property
    def records(self) -> tuple[TrialDevEvaluationTargetRegisterRecordV1, ...]:
        """Return indexed records."""

        return self._records

    def require(
        self,
        *,
        phase_id: str,
        program_objective_id: str,
        phase_scoring_objective_id: str,
        lane_id: str,
    ) -> TrialDevEvaluationTargetRegisterRecordV1:
        """Return a register row or fail loudly if the scoreable lane is absent."""

        key = (
            str(phase_id),
            str(program_objective_id),
            str(phase_scoring_objective_id),
            str(lane_id),
        )
        if key not in self._by_key:
            raise ValueError(f"missing TrialDev evaluation-target register scoring context: {key!r}")
        return self._by_key[key]

    def checksums_for(
        self,
        *,
        phase_id: str,
        program_objective_id: str,
        phase_scoring_objective_id: str,
    ) -> dict[str, str]:
        """Return all lane checksums for a phase/objective context."""

        return {
            str(record.lane_id): str(record.checksum)
            for record in self.records
            if str(record.phase_id) == str(phase_id)
            and str(record.program_objective_id) == str(program_objective_id)
            and str(record.phase_scoring_objective_id) == str(phase_scoring_objective_id)
        }


def load_evaluation_target_index(scenario_root: Path) -> TrialDevEvaluationTargetRegisterIndex:
    """Load the scenario evaluation-target register."""

    path = Path(scenario_root) / "grader" / "evaluation_target_register.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"TrialDev evaluation-target register is missing: {path}")
    records = load_trialdev_evaluation_target_register_records(path)
    if not records:
        raise ValueError(f"TrialDev evaluation-target register has no rows: {path}")
    return TrialDevEvaluationTargetRegisterIndex(records)


def score_evaluation_target(
    *,
    scenario_id: str,
    phase_id: str,
    program_objective_id: str,
    phase_scoring_objective_id: str,
    lane_id: str,
    submitted_target_id: str | None,
    evaluation_target: TrialDevEvaluationTargetRegisterRecordV1,
    evaluation_target_register_checksum: str | None = None,
    artifact_status: Literal["present", "missing", "invalid"],
    failure_reason: str | None = None,
    score_override: float | None = None,
    score_derivation: TrialDevScoreDerivation | None = None,
    derived_from_trajectory_metric: bool = False,
    terminal_action_observed: str | None = None,
    terminal_asset_observed: str | None = None,
    terminal_phase_observed: str | None = None,
) -> TrialDevelopmentLaneScoreRecordV1:
    """Score one submitted target against a evaluation-target register row."""

    expected_context = (
        str(evaluation_target.phase_id),
        str(evaluation_target.program_objective_id),
        str(evaluation_target.phase_scoring_objective_id),
        str(evaluation_target.lane_id),
    )
    observed_context = (
        str(phase_id),
        str(program_objective_id),
        str(phase_scoring_objective_id),
        str(lane_id),
    )
    if observed_context != expected_context:
        raise ValueError(
            f"TrialDev evaluation-target register context mismatch: observed={observed_context!r}, expected={expected_context!r}"
        )
    scenario_text = str(scenario_id)
    if scenario_text != str(evaluation_target.scenario_id):
        raise ValueError(
            "TrialDev evaluation-target register scenario mismatch: "
            f"observed={scenario_text!r}, expected={str(evaluation_target.scenario_id)!r}"
        )
    reference_targets = {str(value) for value in evaluation_target.reference_target_ids}
    accepted_targets = {str(value) for value in evaluation_target.credit_eligible_target_ids}
    overlap = sorted(reference_targets & accepted_targets)
    if overlap:
        raise ValueError(f"reference_target_ids and credit_eligible_target_ids overlap: {overlap!r}")
    resolved_derivation: TrialDevScoreDerivation = score_derivation or (
        "numeric_diagnostic" if score_override is not None else "literal_target"
    )
    if (
        evaluation_target.target_resolution == "realized_public_evidence"
        and resolved_derivation != "public_evidence_action"
    ):
        raise ValueError("realized_public_evidence targets require public_evidence_action scoring.")
    if score_override is not None:
        allowed = (None, str(lane_id)) in _NUMERIC_DIAGNOSTIC_OVERRIDE_CONTEXTS or (
            str(phase_id),
            str(lane_id),
        ) in _NUMERIC_DIAGNOSTIC_OVERRIDE_CONTEXTS
        if resolved_derivation == "literal_target":
            raise ValueError("score_override requires non-literal score_derivation.")
        if resolved_derivation == "numeric_diagnostic" and not allowed:
            raise ValueError(f"score_override is not allowed for categorical lane: {phase_id}/{lane_id}")
        if resolved_derivation == "public_evidence_action" and str(lane_id) not in {
            "decision_action",
            "safety_gate",
            "route_timing",
        }:
            raise ValueError(
                "public_evidence_action is restricted to decision-action, safety-gate, and route-timing lanes."
            )
        if float(score_override) < 0.0 or float(score_override) > 1.0:
            raise ValueError("score_override must be bounded in [0, 1].")
    submitted = None if submitted_target_id is None else str(submitted_target_id)
    if artifact_status == "missing":
        status = "missing_submission_zeroed"
        score = 0.0
        missing_reason = failure_reason or "missing_submission"
    elif artifact_status == "invalid":
        status = "invalid_submission_zeroed"
        score = 0.0
        missing_reason = None
    elif submitted is None:
        status = "missing_submission_zeroed"
        score = 0.0
        missing_reason = failure_reason or "missing_target"
    elif submitted in reference_targets:
        status = "scored"
        score = 1.0 if score_override is None else float(score_override)
        missing_reason = None
    elif submitted in accepted_targets:
        status = "credit_eligible_alternative"
        score = 1.0 if score_override is None else float(score_override)
        missing_reason = None
    else:
        if score_override is not None:
            if (
                resolved_derivation != "public_evidence_action"
                and evaluation_target.target_resolution != "realized_trajectory"
            ):
                raise ValueError("score_override cannot assign positive credit to an unaccepted target.")
            status = (
                "scored"
                if evaluation_target.target_resolution == "realized_trajectory"
                else "credit_eligible_alternative"
            )
            score = float(score_override)
        else:
            status = "scored"
            score = 0.0
        missing_reason = None
    return TrialDevelopmentLaneScoreRecordV1(
        scenario_id=scenario_id,
        phase_id=phase_id,
        program_objective_id=program_objective_id,
        phase_scoring_objective_id=phase_scoring_objective_id,
        lane_id=lane_id,
        evaluation_target_checksum=str(evaluation_target_register_checksum or evaluation_target.checksum),
        scoring_policy_id=str(evaluation_target.scoring_policy_id),
        recoverability_policy_id=str(evaluation_target.recoverability_policy_id),
        submitted_target_id=submitted,
        reference_target_ids=evaluation_target.reference_target_ids,
        credit_eligible_target_ids=evaluation_target.credit_eligible_target_ids,
        score=max(0.0, min(1.0, float(score))),
        score_derivation=resolved_derivation,
        derived_from_trajectory_metric=bool(derived_from_trajectory_metric),
        terminal_action_observed=terminal_action_observed,
        terminal_asset_observed=terminal_asset_observed,
        terminal_phase_observed=terminal_phase_observed,
        status=status,
        artifact_status=artifact_status,
        missing_reason=missing_reason,
        failure_reason=failure_reason,
    )


__all__ = [
    "TrialDevEvaluationTargetRegisterIndex",
    "TrialDevEvaluationTargetRegisterRecordV1",
    "load_evaluation_target_index",
    "score_evaluation_target",
]
