"""Tests for deterministic TrialDev scientific-figure rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import get_args

import pandas as pd

from trialagentbench_validation.trialdev.figure_rendering import (
    render_clinical_realism_v1,
    render_decision_difficulty_v1,
    render_failure_decomposition_v1,
    render_identification_recoverability_v1,
    render_operating_characteristics_v1,
    render_paired_effect_forest_v1,
    render_policy_value_v1,
    render_portfolio_routes_v1,
    write_clinical_realism_source_data_v1,
    write_decision_difficulty_source_data_v1,
    write_grader_control_source_data_v1,
    write_observational_replay_source_data_v1,
    write_operating_effect_source_data_v1,
    write_portfolio_route_source_data_v1,
)
from trialagentbench_validation.trialdev.portfolio_grader_controls import ControlId


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_grader_control_source_data_covers_the_public_control_census(
    tmp_path: Path,
) -> None:
    controls = tmp_path / "controls.csv"
    control_ids = get_args(ControlId)
    pd.DataFrame(
        {
            "control_id": control_ids,
            "control_kind": ("positive",) + ("negative",) * (len(control_ids) - 1),
            "detected": (True,) * len(control_ids),
        }
    ).to_csv(controls, index=False)

    output = write_grader_control_source_data_v1(
        controls_csv=controls,
        output_csv=tmp_path / "source.csv",
    )

    source = pd.read_csv(output)
    assert set(source["control_id"]) == set(control_ids)
    assert set(source["expected_behavior_rate"]) == {1.0}
    assert source["responsibility"].str.strip().ne("").all()


def test_trialdev_vector_figures_are_deterministic(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    axes = ("information", "confounding", "overlap")
    pd.DataFrame(
        {
            "experiment_id": tuple(f"{axis}_response" for axis in axes),
            "axis": axes,
            "primary_metric": (
                "indeterminate",
                "withholding_supported",
                "promotion_supported",
            ),
            "world_count": (400, 400, 400),
            "paired_difference": (-0.4, 0.5, 0.6),
            "paired_bootstrap_lower": (-0.5, 0.4, 0.5),
            "paired_bootstrap_upper": (-0.3, 0.6, 0.7),
            "reference_mean": (0.5, 0.4, 0.3),
            "reference_lower": (0.4, 0.3, 0.2),
            "reference_upper": (0.6, 0.5, 0.4),
            "intervention_mean": (0.1, 0.9, 0.9),
            "intervention_lower": (0.05, 0.8, 0.8),
            "intervention_upper": (0.2, 0.95, 0.95),
            "expected_direction": ("decrease", "increase", "increase"),
        }
    ).to_csv(summary, index=False)
    controls = tmp_path / "controls.csv"
    pd.DataFrame(
        (
            {
                "control_id": "valid_submission",
                "responsibility": "numeric_evidence",
                "control_kind": "positive",
                "detected": True,
            },
            {
                "control_id": "numeric_mutation",
                "responsibility": "numeric_evidence",
                "control_kind": "negative",
                "detected": True,
            },
            {
                "control_id": "stale_state",
                "responsibility": "state_custody",
                "control_kind": "negative",
                "detected": True,
            },
        )
    ).to_csv(controls, index=False)
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "episode_realism": [
                    {
                        "phase_id": phase,
                        "episode_id": f"world-{offset}:{phase}",
                        "row_count": count + offset,
                        "follow_up_days": {"phase1": 28, "phase2": 90, "phase3": 365}[
                            phase
                        ],
                        "efficacy_event_rate_treated": (
                            None if phase == "phase1" else 0.45 + 0.01 * offset
                        ),
                        "efficacy_event_rate_control": (
                            None if phase == "phase1" else 0.40 + 0.005 * offset
                        ),
                        "serious_ae_rate_treated": 0.04 + 0.01 * offset,
                        "serious_ae_rate_control": 0.03 + 0.005 * offset,
                        "discontinuation_rate_treated": 0.08 + 0.01 * offset,
                        "loss_to_follow_up_rate": 0.06 + 0.005 * offset,
                    }
                    for phase, count in (
                        ("phase1", 120),
                        ("phase2", 1500),
                        ("phase3", 2000),
                    )
                    for offset in range(3)
                ],
                "observational_realism": [
                    {
                        "world_id": f"world-{offset}",
                        "treatment_counts": {
                            "control": 3000 + offset,
                            "regimen_a": 2900 + offset,
                            "regimen_b": 2800 + offset,
                            ("regimen_c" if offset < 2 else "regimen_g"): 2700 + offset,
                        },
                    }
                    for offset in range(3)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    boundary = tmp_path / "boundary.json"
    boundary.write_text(
        json.dumps(
            {
                "status": "pass",
                "efficacy_threshold": 0.5,
                "safety_threshold": 0.4,
                "cells": [
                    {
                        "axis": axis,
                        "information_size": information,
                        "mechanism_value": value,
                        "world_count": 400,
                        "clear_pass_rate": 0.05 if value <= threshold else 0.80,
                        "clear_fail_rate": 0.80 if value < threshold else 0.05,
                        "indeterminate_rate": 0.90 if value == threshold else 0.15,
                    }
                    for axis, threshold, values in (
                        ("efficacy", 0.5, (0.35, 0.5, 0.65)),
                        ("safety", 0.4, (0.2, 0.4, 0.6)),
                    )
                    for information in (80, 240, 600)
                    for value in values
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    difficulty = tmp_path / "difficulty.json"
    difficulty.write_text(
        json.dumps(
            {
                "status": "pass",
                "maximum_action_only_shortcut_support_rate": 0.5,
                "maximum_point_estimate_shortcut_support_rate": 0.75,
                "strategies": [
                    {
                        "strategy_id": strategy_id,
                        "strategy_class": strategy_class,
                        "supported_view_rate": rate,
                        "evaluated_view_count": 96,
                    }
                    for strategy_id, strategy_class, rate in (
                        ("evidence_and_policy", "complete_analysis", 1.0),
                        ("adjusted_point_pair", "point_estimate_only", 0.70),
                        ("always_withhold", "action_only", 0.42),
                        ("raw_observed_pair", "point_estimate_only", 0.33),
                        ("alphabetical_pair", "action_only", 0.27),
                    )
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = write_decision_difficulty_source_data_v1(
        decision_boundary_json=boundary,
        portfolio_difficulty_json=difficulty,
        output_csv=tmp_path / "decision_difficulty_source.csv",
    )
    realism_source = write_clinical_realism_source_data_v1(
        release_audit_json=audit,
        output_csv=tmp_path / "clinical_realism_source.csv",
    )
    control_source = write_grader_control_source_data_v1(
        controls_csv=controls,
        output_csv=tmp_path / "grader_control_source.csv",
    )
    operating_source = write_operating_effect_source_data_v1(
        summary_csv=summary,
        axes=axes,
        output_csv=tmp_path / "operating_effect_source.csv",
    )
    replay_root = tmp_path / "replays"
    replay_root.mkdir()
    for world_index in range(3):
        methods = []
        for method_index in range(2):
            result_form = (
                "qualified_non_nomination"
                if method_index == 1 and world_index == 2
                else "point_estimates"
            )
            candidate_results = []
            if result_form == "point_estimates":
                candidate_results = [
                    {
                        "objective_id": "benefit_risk",
                        "candidate_drug_id": candidate,
                        "expected_utility": 0.1 * candidate_index,
                        "replayed_utility": 0.1 * candidate_index + 1e-8,
                        "utility_absolute_error": 1e-8,
                        "within_tolerance": True,
                    }
                    for candidate_index, candidate in enumerate(
                        ("regimen_a", "regimen_b", "regimen_c"), start=1
                    )
                ]
            methods.append(
                {
                    "method_route_id": f"method-{method_index + 1}",
                    "result_form": result_form,
                    "candidate_results": candidate_results,
                    "maximum_utility_absolute_error": (
                        1e-8 if candidate_results else 0.0
                    ),
                    "maximum_standard_error_absolute_error": (
                        2e-7 if candidate_results else 0.0
                    ),
                    "maximum_interval_endpoint_absolute_error": (
                        4e-7 if candidate_results else 0.0
                    ),
                    "status": "pass",
                }
            )
        (
            replay_root / f"observational_replay_world_{world_index + 1:02d}.json"
        ).write_text(
            json.dumps(
                {
                    "status": "pass",
                    "scenario_id": f"world-{world_index}",
                    "absolute_tolerance": 1e-4,
                    "methods": methods,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (replay_root / "observational_replay_census.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "requested_world_count": 3,
                "passing_world_count": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replay_source = write_observational_replay_source_data_v1(
        replay_root=replay_root,
        output_csv=tmp_path / "observational_replay_source.csv",
    )
    policy_value = tmp_path / "policy_value.csv"
    pd.DataFrame(
        [
            {
                "scenario_id": scenario_id,
                "information_size": information_size,
                "resource_budget_units": budget,
                "oracle_action_supported_rate": 0.88 + information_size / 12000,
                "oracle_action_supported_lower": 0.84 + information_size / 12000,
                "oracle_action_supported_upper": 0.92 + information_size / 12000,
                "best_supported_regret": 0.01,
                "best_supported_regret_lower": 0.005,
                "best_supported_regret_upper": 0.015,
                "worst_supported_regret": 0.08,
                "worst_supported_regret_lower": 0.06,
                "worst_supported_regret_upper": 0.10,
                "adjusted_point_regret": 0.03,
                "adjusted_point_regret_lower": 0.02,
                "adjusted_point_regret_upper": 0.04,
                "alphabetical_regret": 0.12,
                "alphabetical_regret_lower": 0.10,
                "alphabetical_regret_upper": 0.14,
                "oracle_terminal_success_probability": 0.72 + 0.02 * (budget == 10),
            }
            for scenario_id in ("clear_separation", "near_tie", "threshold_uncertainty")
            for information_size in (80, 240, 600)
            for budget in (8, 10)
        ]
    ).to_csv(policy_value, index=False)
    route_report = tmp_path / "portfolio_routes.json"
    action_ids = (
        "select_lead_and_reserve",
        "withhold_selection",
        "advance_lead_to_proof_of_concept",
        "promote_reserve_to_proof_of_concept",
        "advance_active_to_confirmation",
        "terminate_portfolio",
        "declare_success",
        "declare_failure",
        "declare_inconclusive",
    )
    route_report.write_text(
        json.dumps(
            {
                "status": "pass",
                "families": [
                    {
                        "family_id": f"P{index:02d}_family_{index}",
                        "supported_action_ids": action_ids[: 2 + index % 8],
                        "terminal_route_count_min": 1 + index % 3,
                        "terminal_route_count_max": 4 + index % 5,
                    }
                    for index in range(1, 13)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    route_source = write_portfolio_route_source_data_v1(
        route_report_json=route_report,
        output_csv=tmp_path / "portfolio_routes.csv",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        render_identification_recoverability_v1(
            source_csv=replay_source,
            output_stem=root / "identification",
        )
        render_paired_effect_forest_v1(
            summary_csv=operating_source,
            axes=axes,
            output_stem=root / "paired",
            panel_label="a",
        )
        render_operating_characteristics_v1(
            summary_csv=summary,
            output_stem=root / "operating",
        )
        render_failure_decomposition_v1(
            source_csv=control_source,
            output_stem=root / "controls",
        )
        render_clinical_realism_v1(
            source_csv=realism_source,
            output_stem=root / "realism",
        )
        render_decision_difficulty_v1(
            source_csv=source,
            output_stem=root / "decision_difficulty",
        )
        render_policy_value_v1(
            source_csv=policy_value,
            output_stem=root / "policy_value",
        )
        render_portfolio_routes_v1(
            source_csv=route_source,
            output_stem=root / "portfolio_routes",
        )
    for name in (
        "identification.pdf",
        "identification.svg",
        "operating.pdf",
        "operating.svg",
        "controls.pdf",
        "controls.svg",
        "paired.pdf",
        "paired.svg",
        "realism.pdf",
        "realism.svg",
        "decision_difficulty.pdf",
        "decision_difficulty.svg",
        "policy_value.pdf",
        "policy_value.svg",
        "portfolio_routes.pdf",
        "portfolio_routes.svg",
    ):
        assert _sha256(first / name) == _sha256(second / name)
    realism_svg = (first / "realism.svg").read_text(encoding="utf-8")
    assert "Investigational" in realism_svg
    assert "Regimen g" not in realism_svg
