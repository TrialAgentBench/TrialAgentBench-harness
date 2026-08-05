"""Declared TrialDev route semantics."""

from __future__ import annotations

from typing import cast

from trialagentbench_harness.contracts.trialdev.trialdev_evaluation_target_register import TrialDevEvaluationLaneV1

TRIALDEV_PHASE_ORDER: tuple[str, ...] = (
    "observational_review",
    "phase1",
    "phase2",
    "phase3",
    "final_decision",
)

TRIALDEV_LANE_ORDER: tuple[TrialDevEvaluationLaneV1, ...] = (
    "asset_nomination",
    "safety_gate",
    "decision_action",
    "phase_design",
    "phase_analysis",
    "route_timing",
    "final_recommendation",
)

ACTION_BEARING_LANES: frozenset[str] = frozenset(
    {
        "asset_nomination",
        "safety_gate",
        "decision_action",
        "final_recommendation",
    }
)


def trialdev_route_sort_key(phase_id: str, lane_id: str) -> tuple[int, int]:
    """Return the declared TrialDev route sort key.

    Parameters
    ----------
    phase_id
        TrialDev phase identifier.
    lane_id
        TrialDev scoring lane identifier.

    Returns
    -------
    tuple[int, int]
        Numeric phase and lane order.

    Raises
    ------
    ValueError
        If either identifier is unknown.
    """

    try:
        phase_index = TRIALDEV_PHASE_ORDER.index(phase_id)
    except ValueError as exc:
        raise ValueError(f"Unknown TrialDev phase_id: {phase_id!r}") from exc
    try:
        lane_index = TRIALDEV_LANE_ORDER.index(cast(TrialDevEvaluationLaneV1, lane_id))
    except ValueError as exc:
        raise ValueError(f"Unknown TrialDev lane_id: {lane_id!r}") from exc
    return phase_index, lane_index


def is_trialdev_action_lane(lane_id: str) -> bool:
    """Return whether a TrialDev lane carries an action-like target."""

    return lane_id in ACTION_BEARING_LANES


__all__ = [
    "ACTION_BEARING_LANES",
    "TRIALDEV_LANE_ORDER",
    "TRIALDEV_PHASE_ORDER",
    "is_trialdev_action_lane",
    "trialdev_route_sort_key",
]
