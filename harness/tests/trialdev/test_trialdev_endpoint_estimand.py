"""Tests for TrialDev endpoint construction under intercurrent-event strategies."""

from __future__ import annotations

import numpy as np
import pytest

from trialagentbench_harness.trialdev.share.materialize import apply_endpoint_estimand_v1


def _surface(strategy: str) -> tuple[list[int], list[int], list[float], list[str]]:
    event, competing, time, cause = apply_endpoint_estimand_v1(
        strategy=strategy,
        endpoint_time=np.asarray([10.0, 20.0, 40.0, 50.0]),
        endpoint_event=np.asarray([True, True, True, False]),
        follow_up_days=30.0,
        discontinuation_time=np.asarray([15.0, 5.0, 25.0, 30.0]),
        discontinuation_event=np.asarray([True, True, True, True]),
        ltfu_time=np.asarray([90.0, 90.0, 12.0, 90.0]),
        ltfu_event=np.asarray([False, False, True, False]),
        terminal_time=np.asarray([90.0, 18.0, 35.0, 90.0]),
        terminal_event=np.asarray([False, True, True, False]),
    )
    return event.tolist(), competing.tolist(), time.tolist(), [str(value) for value in cause]


def test_treatment_policy_ignores_discontinuation_for_endpoint_follow_up() -> None:
    """Treatment-policy follow-up preserves outcomes after discontinuation."""

    event, competing, time, cause = _surface("treatment_policy")

    assert event == [1, 0, 0, 0]
    assert competing == [0, 1, 0, 0]
    assert time == [10.0, 18.0, 12.0, 30.0]
    assert cause == ["endpoint_event", "terminal_event", "ltfu", "administrative"]


def test_while_on_treatment_censors_at_discontinuation() -> None:
    """While-on-treatment follow-up censors before later clinical events."""

    event, competing, time, cause = _surface("while_on_treatment")

    assert event == [1, 0, 0, 0]
    assert competing == [0, 0, 0, 0]
    assert time == [10.0, 5.0, 12.0, 30.0]
    assert cause == [
        "endpoint_event",
        "while_on_treatment_discontinuation",
        "ltfu",
        "while_on_treatment_discontinuation",
    ]


def test_composite_strategy_counts_discontinuation_as_an_event() -> None:
    """Composite follow-up uses the earlier endpoint or discontinuation event."""

    event, competing, time, cause = _surface("composite_discontinuation")

    assert event == [1, 1, 0, 1]
    assert competing == [0, 0, 0, 0]
    assert time == [10.0, 5.0, 12.0, 30.0]
    assert cause == ["endpoint_event", "composite_discontinuation", "ltfu", "composite_discontinuation"]


def test_endpoint_estimand_rejects_invalid_event_time() -> None:
    """Malformed endpoint chronology fails before materialization."""

    with pytest.raises(ValueError, match="Endpoint times must be non-negative"):
        apply_endpoint_estimand_v1(
            strategy="treatment_policy",
            endpoint_time=np.asarray([np.nan]),
            endpoint_event=np.asarray([True]),
            follow_up_days=30.0,
            discontinuation_time=None,
            discontinuation_event=None,
            ltfu_time=None,
            ltfu_event=None,
            terminal_time=None,
            terminal_event=None,
        )
