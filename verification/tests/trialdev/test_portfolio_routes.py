"""Tests for exact-release TrialDev portfolio route qualification."""

from typing import Any

from trialagentbench_validation.trialdev.portfolio_routes import (
    _REQUIRED_ACTIONS,
    _REQUIRED_CHECKPOINTS,
    _REQUIRED_TERMINAL_DISPOSITIONS,
    _family_contrast_findings,
    _findings,
)


def _complete_counts() -> dict[str, Any]:
    return {
        "action_ids": set(_REQUIRED_ACTIONS),
        "checkpoint_ids": set(_REQUIRED_CHECKPOINTS),
        "terminal_dispositions": set(_REQUIRED_TERMINAL_DISPOSITIONS),
        "nonidentified_view_count": 1,
        "identified_withholding_only_view_count": 1,
        "multiple_initial_action_view_count": 1,
        "multiple_randomized_action_state_count": 1,
        "safety_exclusion_state_count": 1,
        "joint_safety_stop_state_count": 1,
        "early_reserve_promotion_route_count": 1,
        "late_reserve_promotion_route_count": 1,
        "promoted_reserve_confirmation_route_count": 1,
        "promoted_reserve_stop_route_count": 1,
        "budget_contrasts": {"world:objective"},
    }


def test_complete_route_coverage_has_no_findings() -> None:
    assert _findings(**_complete_counts()) == ()


def test_joint_safety_stop_is_a_required_observed_behavior() -> None:
    values = _complete_counts()
    values["joint_safety_stop_state_count"] = 0

    assert _findings(**values) == ("missing_joint_safety_stop",)


def test_route_coverage_requires_every_terminal_decision() -> None:
    values = _complete_counts()
    values["terminal_dispositions"] = {"withheld", "stopped", "inconclusive"}

    assert _findings(**values) == ("incomplete_terminal_decision_coverage",)


def _complete_family_rows() -> dict[str, list[dict[str, object]]]:
    selected = {("select_lead_and_reserve", "regimen_a", "regimen_b")}
    rows = {
        f"P{index:02d}_family": [
            {
                "selection_supported": True,
                "withholding_only": False,
                "initial_signatures": set(selected),
                "initial_supported_pair_count": 1,
                "identification_statuses": {"identified"},
                "joint_safety_stop_state_count": 0,
                "checkpoint_actions": {},
                "checkpoints": {"joint_early_study_review", "confirmation"},
                "terminals": {"success", "failure", "inconclusive"},
            }
        ]
        for index in range(1, 13)
    }
    rows["P02_family"][0]["initial_signatures"] = {
        *selected,
        ("select_lead_and_reserve", "regimen_c", "regimen_b"),
    }
    rows["P02_family"][0]["initial_supported_pair_count"] = 2
    rows["P03_family"][0]["checkpoint_actions"] = {
        "joint_early_study_review": {
            "advance_lead_to_proof_of_concept",
            "terminate_portfolio",
        }
    }
    rows["P04_family"][0]["joint_safety_stop_state_count"] = 1
    for family_id in ("P05_family", "P06_family"):
        rows[family_id][0].update(
            selection_supported=False,
            withholding_only=True,
            initial_signatures={("withhold_selection", None, None)},
            initial_supported_pair_count=0,
        )
    rows["P07_family"][0]["checkpoint_actions"] = {
        "lead_proof_of_concept_review": {
            "advance_active_to_confirmation",
            "terminate_portfolio",
        }
    }
    rows["P09_family"][0]["checkpoint_actions"] = {
        "promoted_reserve_proof_of_concept_review": {"terminate_portfolio"}
    }
    rows["P11_family"][0].update(
        selection_supported=False,
        withholding_only=True,
        initial_signatures={("withhold_selection", None, None)},
        initial_supported_pair_count=0,
        identification_statuses={"not_identified"},
    )
    return rows


def test_named_family_contrasts_are_required_without_prescribing_one_route() -> None:
    rows = _complete_family_rows()

    assert (
        _family_contrast_findings(
            grouped=rows,
            budget_contrasts={"portfolio-world-08:benefit_risk"},
        )
        == ()
    )

    rows["P04_family"][0]["joint_safety_stop_state_count"] = 0
    assert _family_contrast_findings(
        grouped=rows,
        budget_contrasts={"portfolio-world-08:benefit_risk"},
    ) == ("P04_shared_safety_stop_not_recovered",)


def test_confirmatory_success_control_is_required_from_its_declared_family() -> None:
    rows = _complete_family_rows()
    rows["P10_family"][0]["terminals"] = {"failure", "inconclusive"}

    assert _family_contrast_findings(
        grouped=rows,
        budget_contrasts={"portfolio-world-08:benefit_risk"},
    ) == ("P10_confirmatory_success_not_recovered",)
