"""Read canonical TrialDev grader reports for aggregation.

Execution and grading are separate workflows. This module exposes small
helpers used by ``aggregate.py`` to extract lane scores from the deterministic
offline grade reports.

Note: ``grade_trajectory_v1`` returns per-phase ``phase_reports``, descriptive
``mean_scores``, and a conjunctive ``trajectory_primary_score``. The phase
primary score is the minimum required lane score; the trajectory primary score
is the minimum over phase scores, including zero for invalid attempts.
"""

from __future__ import annotations

from collections.abc import Iterator

from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevGradeRecordV1,
    TrialDevTrajectoryGradeV1,
)

LANE_KEYS: tuple[str, ...] = (
    "trial_design",
    "trial_evaluation",
    "program_decision",
    "drug_ranking",
)


def extract_grade_record(report: TrialDevGradeRecordV1, *, source: str) -> dict[str, object]:
    """Normalise a ``grade_item_v1``-shaped report into a flat row.

    ``source`` identifies whether the report came from ``grade_item_v1``
    (obs_review) or one of the trajectory's ``phase_reports``.
    """
    row = {
        "source": source,
        "primary_score": float(report.primary_score),
        "design_score": float(report.design_score),
        "evaluation_score": float(report.evaluation_score),
        "program_score": float(report.program_score),
        "ranking_score": float(report.ranking_score),
        "policy_reference_regret": (
            None if report.policy_reference_regret is None else float(report.policy_reference_regret)
        ),
        "in_set_regret": None if report.in_set_regret is None else float(report.in_set_regret),
        "selected_winner_drug_id": report.selected_winner_drug_id,
        "best_candidate_drug_id": report.best_candidate_drug_id,
        "feasibility_failures": list(report.feasibility_failures or ()),
        "phase_id": report.phase_id,
        "scenario_id": report.scenario_id,
        "objective_id": report.objective_id,
    }
    missing_lanes = sorted(set(LANE_KEYS).difference(report.lane_breakdown))
    if missing_lanes:
        raise ValueError(f"TrialDev grade report is missing lane breakdowns: {missing_lanes!r}")
    for lane in LANE_KEYS:
        row[f"lane_raw__{lane}"] = float(report.lane_breakdown[lane])
    return row


def iter_phase_reports(
    trajectory_grade: TrialDevTrajectoryGradeV1,
) -> Iterator[tuple[str | None, TrialDevGradeRecordV1]]:
    """Yield (phase_id, report) for each per-phase report.

    The harness stores a schema-bearing wrapper on disk; each phase report is
    stored as a schema-bearing `TrialDevGradeRecordV1`.
    """
    for report in trajectory_grade.phase_reports:
        yield report.phase_id, report


__all__ = [
    "LANE_KEYS",
    "extract_grade_record",
    "iter_phase_reports",
]
