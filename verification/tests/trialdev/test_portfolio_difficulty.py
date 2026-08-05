"""Tests for TrialDev released-view difficulty criteria."""

from __future__ import annotations

from trialagentbench_validation.trialdev.portfolio_difficulty import (
    StrategyClass,
    StrategyId,
    TrialDevShortcutPerformanceV1,
    _difficulty_findings,
)


def _strategy(
    strategy_id: StrategyId,
    strategy_class: StrategyClass,
    supported: int,
) -> TrialDevShortcutPerformanceV1:
    return TrialDevShortcutPerformanceV1(
        strategy_id=strategy_id,
        strategy_class=strategy_class,
        evaluated_view_count=100,
        supported_view_count=supported,
        supported_view_rate=supported / 100,
        complete_statistical_submission=strategy_class == "complete_analysis",
        primary_grade_eligible=strategy_class == "complete_analysis",
    )


def test_difficulty_criteria_preserve_a_gradient_between_baseline_classes() -> None:
    performance = (
        _strategy("evidence_and_policy", "complete_analysis", 100),
        _strategy("always_withhold", "action_only", 50),
        _strategy("alphabetical_pair", "action_only", 25),
        _strategy("adjusted_point_pair", "point_estimate_only", 75),
        _strategy("raw_observed_pair", "point_estimate_only", 30),
    )

    assert _difficulty_findings(performance) == ()


def test_difficulty_criteria_reject_action_and_uncertainty_blind_shortcuts() -> None:
    performance = (
        _strategy("evidence_and_policy", "complete_analysis", 99),
        _strategy("always_withhold", "action_only", 51),
        _strategy("alphabetical_pair", "action_only", 25),
        _strategy("adjusted_point_pair", "point_estimate_only", 76),
        _strategy("raw_observed_pair", "point_estimate_only", 30),
    )

    assert _difficulty_findings(performance) == (
        "action_only_shortcut_exceeds_majority_support_ceiling",
        "evidence_policy_not_universally_supported",
        "point_estimate_shortcut_exceeds_three_quarter_support_ceiling",
    )
