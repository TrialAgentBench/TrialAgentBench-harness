"""Ground-reference-linked failure-cascade diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Literal

from trialagentbench_harness.contracts.trace.observable import (
    EvidenceUseRowV1,
    FailureCascadeRowV1,
    ProgramFailureCascadeRowV1,
    TraceFeatureRowV1,
    TrialDevPhaseOutcomeRowV1,
)

PhaseFailureTypeV1 = Literal[
    "missing_output",
    "no_public_evidence_use",
    "endpoint_invalid_or_unusable",
    "assumption_check_omitted",
    "confounding_check_omitted",
    "wrong_asset",
    "wrong_endpoint",
    "request_or_materialization_violation_observed",
    "unsupported_stop_or_advance",
    "rationale_action_mismatch",
    "tool_or_runtime_failure",
    "none_observed",
]
ProgramFailureTypeV1 = Literal[
    "missing_output",
    "tool_or_runtime_failure",
    "hidden_or_grader_access",
    "wrong_asset",
    "wrong_endpoint",
    "request_or_materialization_violation_observed",
    "assumption_check_omitted",
    "confounding_check_omitted",
    "unsupported_stop_or_advance",
    "rationale_action_mismatch",
    "not_reached_after_stop",
    "none_observed",
]


def phase_failure_cascade(feature: TraceFeatureRowV1, *, hidden_access: bool = False) -> FailureCascadeRowV1:
    """Classify the first observable failure for one benchmark phase or item."""
    if feature.score_link_id is None:
        raise ValueError("Trace feature score_link_id is required for phase cascades")
    endpoint_failed = feature.endpoint_state == "failed" or feature.endpoint_valid is False
    if hidden_access:
        failure: PhaseFailureTypeV1 = "tool_or_runtime_failure"
    elif feature.endpoint_state in {
        "score_export_absent",
        "submission_absent",
        "submission_present_score_absent",
        "score_present_submission_absent",
        "source_inconsistent_requires_adjudication",
    }:
        failure = "missing_output"
    elif not feature.submitted_structured_answer and feature.endpoint_state not in {
        "not_reached_after_stop",
        "not_attempted_noncompletion",
        "not_scoreable_trace_only",
    }:
        failure = "missing_output"
    elif feature.benchmark == "trialeval" and feature.endpoint_valid is False:
        failure = "endpoint_invalid_or_unusable"
    elif feature.benchmark == "trialdev" and feature.endpoint_state == "failed":
        failure = "unsupported_stop_or_advance"
    elif feature.endpoint_state == "not_reached_after_stop":
        failure = "none_observed"
    else:
        failure = "none_observed"
    return FailureCascadeRowV1(
        benchmark=feature.benchmark,
        model_id=feature.model_id,
        run_id=feature.run_id,
        task_id=feature.task_id,
        assignment_id=feature.assignment_id,
        program_id=feature.program_id,
        first_failure_phase=feature.phase_id,
        first_failure_type=failure,
        downstream_endpoint_failed=endpoint_failed,
        score_link_id=feature.score_link_id,
    )


def program_failure_cascades(
    outcomes: list[TrialDevPhaseOutcomeRowV1],
    evidence_rows: list[EvidenceUseRowV1],
) -> list[ProgramFailureCascadeRowV1]:
    """Build TrialDev program-level cascades from phase outcomes and evidence."""
    hidden_by_program = {
        (row.model_id, row.run_id, row.program_id)
        for row in evidence_rows
        if row.benchmark == "trialdev" and row.leakage_violation
    }
    grouped: dict[tuple[str, str, str], list[TrialDevPhaseOutcomeRowV1]] = defaultdict(list)
    for row in outcomes:
        grouped[(row.model_id, row.run_id, row.program_id)].append(row)

    cascades: list[ProgramFailureCascadeRowV1] = []
    phase_rank = {"observational_review": 0, "phase1": 1, "phase2": 2, "phase3": 3}
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: phase_rank[row.phase_id])
        first = rows[0]
        first_phase: str | None = None
        failure: ProgramFailureTypeV1 = "none_observed"
        evidence: tuple[str, ...] = ()
        if key in hidden_by_program:
            first_phase = None
            failure = "hidden_or_grader_access"
            evidence = ("evidence_coverage:leakage_violation",)
        else:
            for row in rows:
                if row.endpoint_state in {
                    "score_export_absent",
                    "submission_absent",
                    "submission_present_score_absent",
                    "score_present_submission_absent",
                    "source_inconsistent_requires_adjudication",
                }:
                    first_phase = row.phase_id
                    failure = "missing_output"
                    evidence = (row.score_link_id,)
                    break
                if row.violations_n > 0:
                    first_phase = row.phase_id
                    if any("materialize" in kind or "schema" in kind for kind in row.violation_kinds):
                        failure = "request_or_materialization_violation_observed"
                    else:
                        failure = "tool_or_runtime_failure"
                    evidence = row.violation_kinds
                    break
                if row.phase_id == "observational_review" and row.decision_regret not in (None, 0.0):
                    first_phase = row.phase_id
                    failure = "wrong_asset"
                    evidence = (f"decision_regret={row.decision_regret}",)
                    break
                if row.endpoint_state == "failed":
                    first_phase = row.phase_id
                    failure = "unsupported_stop_or_advance"
                    evidence = (row.score_link_id,)
                    break
                if row.endpoint_state == "not_reached_after_stop":
                    first_phase = row.phase_id
                    failure = "not_reached_after_stop"
                    evidence = (f"stopped_at_phase={row.stopped_at_phase}",)
                    break
        terminal = next((row for row in reversed(rows) if row.phase_reached or row.phase_attempted), rows[-1])
        terminal_success = terminal.endpoint_state in {"valid", "not_reached_after_stop"}
        cascades.append(
            ProgramFailureCascadeRowV1(
                model_id=first.model_id,
                run_id=first.run_id,
                scenario_id=first.scenario_id,
                program_id=first.program_id,
                objective_id=first.objective_id,
                first_failure_phase=first_phase,
                first_failure_type=failure,
                first_failure_evidence=evidence,
                terminal_phase=terminal.phase_id,
                terminal_decision_action=terminal.decision_action,
                terminal_success=terminal_success,
                downstream_endpoint_failed=not terminal_success,
                trajectory_primary_score=first.trajectory_primary_score,
                trajectory_decision_score=first.trajectory_decision_score,
            )
        )
    return cascades


def cascade_summary_counts(
    phase_rows: list[FailureCascadeRowV1],
    program_rows: list[ProgramFailureCascadeRowV1],
) -> dict[str, int]:
    """Return deterministic cascade counts for JSON summaries."""
    counts: dict[str, int] = {}
    for phase_failure_type, value in sorted(Counter(row.first_failure_type for row in phase_rows).items()):
        counts[f"phase:{phase_failure_type}"] = value
    for program_failure_type, value in sorted(Counter(row.first_failure_type for row in program_rows).items()):
        counts[f"program:{program_failure_type}"] = value
    return counts


__all__ = ["cascade_summary_counts", "phase_failure_cascade", "program_failure_cascades"]
