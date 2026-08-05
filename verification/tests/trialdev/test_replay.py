"""Independent TrialDevBench observational replay tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.cli import main
from trialagentbench_validation.trialdev.replay import (
    _standardized_cumulative_incidence,
    replay_trialdev_observational_reference,
)


def test_independent_replay_uses_the_first_observed_competing_event() -> None:
    """A terminal event removes a participant before a later endpoint event."""

    risk = _standardized_cumulative_incidence(
        candidate_id="regimen_a",
        time=np.asarray([5.0, 2.0]),
        event=np.asarray([1, 1], dtype=np.int64),
        competing_time=np.asarray([1.0, 10.0]),
        competing_event=np.asarray([1, 0], dtype=np.int64),
        horizon=10.0,
        stratum_weights=(("all", 1.0),),
        group_masks={
            ("regimen_a", "all"): np.asarray([True, True]),
        },
        analysis_weights=np.asarray([1.0, 1.0]),
    )

    assert risk == pytest.approx(0.5)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _scenario(root: Path) -> Path:
    public = root / "public"
    grader = root / "grader"
    rows: list[dict[str, object]] = []
    event_counts = {"control": 4, "candidate_a": 2, "candidate_b": 3}
    for treatment, event_count in event_counts.items():
        for x_value in (0.0, 1.0):
            for index in range(10):
                rows.append(
                    {
                        "TREATMENT": treatment,
                        "X": x_value,
                        "TIME": 5.0 if index < event_count else 10.0,
                        "EVENT": 1 if index < event_count else 0,
                        "COMPETING_TIME": 10.0,
                        "COMPETING_EVENT": 0,
                        "SAE": 0.0,
                        "SAE_E": 0,
                        "DISCONTINUATION_E": 0.0,
                        "LTFU_E": 0.0,
                    }
                )
    public.mkdir(parents=True)
    frame_path = public / "observational_extract.parquet"
    pd.DataFrame(rows).to_parquet(frame_path, index=False)
    _write_json(
        public / "candidate_drug_catalog.json",
        {
            "candidate_drugs": [
                {"candidate_drug_id": "control", "role": "control"},
                {"candidate_drug_id": "candidate_a", "role": "investigational"},
                {"candidate_drug_id": "candidate_b", "role": "investigational"},
            ]
        },
    )
    _write_json(
        public / "objective_charter.json",
        {
            "checksum": "1" * 64,
            "confidence_level": 0.95,
            "objectives": [
                {
                    "objective_id": "pure_efficacy",
                    "efficacy_endpoints": [
                        {
                            "endpoint_id": "efficacy",
                            "time_column": "TIME",
                            "event_column": "EVENT",
                            "competing_time_column": "COMPETING_TIME",
                            "competing_event_column": "COMPETING_EVENT",
                            "horizon_days": 10,
                            "estimator_id": "standardized_aalen_johansen_cumulative_incidence",
                            "effect_scale_id": "risk_difference_control_minus_candidate",
                            "effect_orientation_id": "positive_values_favour_candidate",
                        }
                    ],
                    "utility_event_definitions": [],
                    "candidate_costs": {},
                    "indifference_margin": 0.01,
                    "utility_components": [
                        {
                            "component_id": "efficacy_gain",
                            "source": "efficacy_gain",
                            "direction": "benefit",
                            "weight": 1.0,
                        }
                    ],
                }
            ],
        },
    )
    method_ids = (
        (
            "trialdev.observational.multinomial_propensity_weighted_stratified_aalen_johansen.v1",
            "multinomial_propensity_weighted_stratified_aalen_johansen",
        ),
        (
            "trialdev.observational.entropy_balanced_standardized_aalen_johansen.v1",
            "entropy_balanced_standardized_aalen_johansen",
        ),
    )
    methods = [
        {
            "method_route_id": method_id,
            "primary_estimator_id": estimator_id,
            "adjustment_covariates": ["X"],
            "bootstrap_replicates": 40,
            "bootstrap_seed": int("1" * 8, 16),
            "bootstrap_rng_id": "numpy_default_rng_pcg64",
            "bootstrap_standard_error_ddof": 1,
            "confidence_interval_id": "normal_critical_value_times_bootstrap_standard_error",
            "confidence_level": 0.95,
            "uncertainty_estimator_id": (
                "refitted_nuisance_participant_nonparametric_bootstrap"
            ),
            **(
                {
                    "exact_stratification_covariates": [],
                    "quantile_stratification_bins": {},
                    "propensity_max_iterations": 1000,
                    "propensity_tolerance": 1e-10,
                }
                if estimator_id
                == "multinomial_propensity_weighted_stratified_aalen_johansen"
                else {
                    "calibration_max_iterations": 1000,
                    "calibration_tolerance": 1e-10,
                    "maximum_mean_balance_error": 1e-6,
                }
            ),
        }
        for method_id, estimator_id in method_ids
    ]
    _write_json(
        public / "observational_method_catalog.json",
        {
            "assignment_prognostic_factors": [],
            "methods": methods,
        },
    )
    _write_json(
        public / "safety_decision_policy.json",
        {
            "serious_event_definitions": [
                {
                    "endpoint_id": "sae",
                    "event_column": "SAE_E",
                    "seriousness_column": "SAE",
                    "severity_column": "SAE",
                    "time_column": "TIME",
                }
            ]
        },
    )
    _write_json(
        public / "decision_charter.json",
        {
            "efficacy_rules": [
                {
                    "phase_id": "observational_review",
                    "minimum_benefit": 0.05,
                }
            ]
        },
    )
    uncertainty_by_estimator = {
        "multinomial_propensity_weighted_stratified_aalen_johansen": (
            (0.15204804342155803, -0.09800868902603593, 0.49800868902603596),
            (0.14698123799771484, -0.18807793287863106, 0.3880779328786311),
        ),
        "entropy_balanced_standardized_aalen_johansen": (
            (0.15204788470102024, -0.09800837793949829, 0.4980083779394983),
            (0.14698111014911816, -0.18807768229998612, 0.3880776822999862),
        ),
    }

    def candidate_scores(estimator_id: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for rank, (candidate_id, estimate, uncertainty) in enumerate(
            zip(
                ("candidate_a", "candidate_b"),
                (0.2, 0.1),
                uncertainty_by_estimator[estimator_id],
                strict=True,
            ),
            start=1,
        ):
            standard_error, lower, upper = uncertainty
            rows.append(
                {
                    "objective_id": "pure_efficacy",
                    "candidate_drug_id": candidate_id,
                    "point_estimable": True,
                    "adjusted_utility": estimate,
                    "efficacy_gain": estimate,
                    "utility_se": standard_error,
                    "efficacy_gain_se": standard_error,
                    "ci_low": lower,
                    "ci_high": upper,
                    "efficacy_gain_ci_low": lower,
                    "efficacy_gain_ci_high": upper,
                    "rank": rank,
                }
            )
        return rows

    method_results = [
        {
            "method_route_id": method_id,
            "estimator_id": estimator_id,
            "candidate_scores": candidate_scores(estimator_id),
            "estimator_comparisons": [
                {
                    "objective_id": "pure_efficacy",
                    "estimator_id": estimator_id,
                    "status": "estimated",
                    "failure_reason": None,
                }
            ],
            "objective_policies": [
                {
                    "objective_id": "pure_efficacy",
                    "policy": "acceptable_candidate_set",
                    "reference_target_ids": ["candidate_a"],
                    "acceptable_candidate_set": ["candidate_a", "candidate_b"],
                }
            ],
            "observational_action_policies": [
                {
                    "objective_id": "pure_efficacy",
                    "reference_target_ids": ["withhold_nomination"],
                    "definitely_qualified_candidate_ids": [],
                    "possibly_qualified_candidate_ids": [
                        "candidate_a",
                        "candidate_b",
                    ],
                    "pairwise_utility_contrast_half_widths": {
                        "candidate_a|candidate_b": (
                            0.23562002971184012
                            if estimator_id
                            == "multinomial_propensity_weighted_stratified_aalen_johansen"
                            else 0.23561988749126853
                        ),
                    },
                }
            ],
        }
        for method_id, estimator_id in method_ids
    ]
    _write_json(
        grader / "public_recoverability_report.json",
        {
            "scenario_id": "unit",
            "public_input_checksums": [
                {
                    "path": "public/observational_extract.parquet",
                    "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                }
            ],
            "method_results": method_results,
        },
    )
    return root


def test_trialdev_replay_reconstructs_both_public_methods(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)

    report = replay_trialdev_observational_reference(scenario, absolute_tolerance=1e-8)

    assert report.status == "pass"
    assert report.public_input_checksums_match
    assert {row.estimator_id for row in report.methods} == {
        "multinomial_propensity_weighted_stratified_aalen_johansen",
        "entropy_balanced_standardized_aalen_johansen",
    }
    assert all(row.ranking_match and row.action_match for row in report.methods)


def test_trialdev_replay_reconstructs_provenance_based_non_nomination(
    tmp_path: Path,
) -> None:
    scenario = _scenario(tmp_path)
    catalog_path = scenario / "public" / "observational_method_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["assignment_prognostic_factors"] = [
        {
            "factor_id": "latent_severity",
            "used_in_treatment_assignment": True,
            "prognostic_for_primary_endpoint": True,
            "recorded_in_observational_extract": False,
        }
    ]
    _write_json(catalog_path, catalog)
    report_path = scenario / "grader" / "public_recoverability_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    for method in payload["method_results"]:
        method["candidate_scores"] = [
            {
                "objective_id": "pure_efficacy",
                "candidate_drug_id": candidate,
                "point_estimable": False,
            }
            for candidate in ("candidate_a", "candidate_b")
        ]
        method["estimator_comparisons"] = [
            {
                "objective_id": "pure_efficacy",
                "estimator_id": method["estimator_id"],
                "status": "not_estimable",
                "failure_reason": "residual_unmeasured_confounding",
            }
        ]
        method["objective_policies"] = [
            {
                "objective_id": "pure_efficacy",
                "policy": "insufficient_recoverability",
                "reference_target_ids": [],
                "acceptable_candidate_set": [],
            }
        ]
        method["observational_action_policies"] = [
            {
                "objective_id": "pure_efficacy",
                "reference_target_ids": ["withhold_nomination"],
                "definitely_qualified_candidate_ids": [],
                "possibly_qualified_candidate_ids": [],
                "pairwise_utility_contrast_half_widths": {},
            }
        ]
    payload["public_input_checksums"] = [
        {
            "path": "public/observational_method_catalog.json",
            "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        }
    ]
    _write_json(report_path, payload)

    report = replay_trialdev_observational_reference(scenario, absolute_tolerance=1e-8)

    assert report.status == "pass"
    assert all(row.result_form == "qualified_non_nomination" for row in report.methods)
    assert all(
        row.replayed_non_estimability_reason == "residual_unmeasured_confounding"
        and row.replayed_actions == {"pure_efficacy": ("withhold_nomination",)}
        for row in report.methods
    )


def test_trialdev_replay_fails_on_checksum_or_numeric_drift(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    report_path = scenario / "grader" / "public_recoverability_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["public_input_checksums"][0]["sha256"] = "0" * 64
    payload["method_results"][0]["candidate_scores"][0]["adjusted_utility"] = 0.9
    _write_json(report_path, payload)

    report = replay_trialdev_observational_reference(scenario, absolute_tolerance=1e-8)

    assert report.status == "fail"
    assert not report.public_input_checksums_match
    assert report.methods[0].status == "fail"


def test_trialdev_replay_cli_writes_report_and_returns_status(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path / "scenario")
    output = tmp_path / "replay.json"

    status = main(
        [
            "trialdev-replay",
            "--scenario-root",
            str(scenario),
            "--output",
            str(output),
            "--absolute-tolerance",
            "1e-8",
        ]
    )

    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_trialdev_replay_fails_on_uncertainty_or_policy_drift(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    report_path = scenario / "grader" / "public_recoverability_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["method_results"][0]["candidate_scores"][0]["utility_se"] = 0.01
    payload["method_results"][0]["objective_policies"][0][
        "acceptable_candidate_set"
    ] = ["candidate_a"]
    _write_json(report_path, payload)

    report = replay_trialdev_observational_reference(scenario, absolute_tolerance=1e-8)

    assert report.status == "fail"
    assert report.methods[0].maximum_standard_error_absolute_error > 0.1
    assert not report.methods[0].uncertainty_policy_match


def test_trialdev_replay_rejects_unsupported_uncertainty_contract(
    tmp_path: Path,
) -> None:
    scenario = _scenario(tmp_path)
    catalog_path = scenario / "public" / "observational_method_catalog.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    payload["methods"][0]["uncertainty_estimator_id"] = "bootstrap_unspecified"
    _write_json(catalog_path, payload)

    with pytest.raises(ValueError, match="unsupported uncertainty estimator"):
        replay_trialdev_observational_reference(scenario, absolute_tolerance=1e-8)
