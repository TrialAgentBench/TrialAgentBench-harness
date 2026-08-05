"""Tests for evidence-derived TrialDev phase action sets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.stats import beta, binom, norm

from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevEvidenceReferenceV1,
    TrialDevPolicyBindingV1,
    TrialDevSingleAssetProgrammeStateV1,
)
from trialagentbench_harness.trialdev.grading.decision_evidence import (
    derive_phase_decision_witness_v1,
    derive_phase_design_witness_v1,
    harmful_risk_difference_power_v1,
    risk_difference_power_v1,
    safety_decision_power_v1,
)
from trialagentbench_harness.trialdev.share.hashing import compute_sha256_hex
from trialagentbench_harness.trialdev.share.models import (
    PhaseModuleSpecV1,
    TrialDevelopmentEvalContractV1,
    TrialDevelopmentRequestV1,
)
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentPhaseActionSpecV1,
    TrialDevelopmentProgramLoopManifestV1,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _checksummed(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["checksum"] = compute_sha256_hex(result)
    return result


def _request(root: Path) -> TrialDevelopmentRequestV1:
    return TrialDevelopmentRequestV1.model_validate(json.loads((root / "request.json").read_text(encoding="utf-8")))


def _design_rule(phase_id: str) -> dict[str, object]:
    phase1 = phase_id == "phase1"
    rule: dict[str, object] = {
        "design_cell_id": f"trialdev.{phase_id}.fixed_final_operating_characteristics.v1",
        "phase_id": phase_id,
        "calculator_id": "prospective_fixed_final_operating_characteristics_v1",
        "serious_ae_unacceptable_excess_risk": 0.05,
        "planning_safety_control_risk": 0.05,
        "planning_safety_absolute_treatment_risk": 0.30,
        "planning_safety_excess_risk": 0.25,
        "planning_safety_excess_treatment_risk": 0.30,
        "target_safety_decision_power": 0.80,
        "safety_power_adequacy_rule": ("minimum_achieved_power_across_absolute_and_excess_hard_gates"),
        "planning_safety_estimator_id": ("multinomial_propensity_weighted_aalen_johansen_any_serious_ae"),
        "planning_safety_analysis_population": "complete_on_declared_adjustment_covariates",
        "planning_safety_control_support_count": 1000,
        "planning_safety_min_observed_propensity": 0.1,
        "planning_safety_max_inverse_propensity_weight": 10.0,
        "planning_safety_weighted_effective_sample_size": 900.0,
        "supported_interim_policy": "fixed_final",
        "confidence_level": 0.95,
        "evaluation_horizon_days": {"phase1": 28, "phase2": 90, "phase3": 365}[phase_id],
        "serious_ae_unacceptable_absolute_risk": 0.10,
        "planning_information_estimator_id": ("one_minus_multinomial_propensity_weighted_aalen_johansen_ltfu_cif"),
        "planning_information_fraction_by_drug_id": {"control": 1.0, "drug_a": 1.0},
        "planning_information_support_count_by_drug_id": {"control": 1000, "drug_a": 1000},
        "planning_information_weighted_effective_sample_size_by_drug_id": {
            "control": 1000.0,
            "drug_a": 1000.0,
        },
        "rationale": "Test policy.",
    }
    if not phase1:
        rule.update(
            {
                "primary_endpoint_id": ("DISEASE_PROGRESSION" if phase_id == "phase2" else "HARD_ENDPOINT"),
                "planning_alternative_benefit": 0.04 if phase_id == "phase2" else 0.07,
                "target_power": 0.80,
                "planning_control_risk": 0.10 if phase_id == "phase2" else 0.30,
                "planning_treatment_risk": 0.06 if phase_id == "phase2" else 0.23,
                "planning_estimator_id": "multinomial_propensity_weighted_aalen_johansen",
                "planning_analysis_population": "complete_on_declared_adjustment_covariates",
                "planning_control_support_count": 1000,
                "planning_min_observed_propensity": 0.1,
                "planning_max_inverse_propensity_weight": 10.0,
                "planning_weighted_effective_sample_size": 900.0,
            }
        )
    else:
        rule.update(
            {
                "primary_endpoint_id": None,
                "planning_alternative_benefit": None,
                "target_power": None,
                "planning_control_risk": None,
                "planning_treatment_risk": None,
                "planning_estimator_id": None,
                "planning_analysis_population": None,
                "planning_control_support_count": None,
                "planning_min_observed_propensity": None,
                "planning_max_inverse_propensity_weight": None,
                "planning_weighted_effective_sample_size": None,
            }
        )
    return rule


def _action_specs() -> list[dict[str, object]]:
    return [
        {
            "phase_id": "phase1",
            "allowed_action_ids": ["advance_to_proof_of_concept", "stop_development"],
            "stop_action_ids": ["stop_development"],
            "advance_action_ids": ["advance_to_proof_of_concept"],
        },
        {
            "phase_id": "phase2",
            "allowed_action_ids": ["advance_to_confirmation", "stop_development"],
            "stop_action_ids": ["stop_development"],
            "advance_action_ids": ["advance_to_confirmation"],
        },
        {
            "phase_id": "phase3",
            "allowed_action_ids": ["declare_success", "declare_failure", "declare_inconclusive"],
            "stop_action_ids": ["declare_failure", "declare_inconclusive"],
            "advance_action_ids": ["declare_success"],
        },
    ]


def test_phase2_contract_rejects_direct_completion_action() -> None:
    with pytest.raises(ValidationError, match="complete_development_without_phase3"):
        TrialDevelopmentPhaseActionSpecV1.model_validate(
            {
                "phase_id": "phase2",
                "allowed_action_ids": [
                    "advance_to_confirmation",
                    "complete_development_without_phase3",
                    "stop_development",
                ],
                "stop_action_ids": ["stop_development"],
                "advance_action_ids": [
                    "advance_to_confirmation",
                    "complete_development_without_phase3",
                ],
            }
        )


def test_program_state_rejects_future_phase_evidence() -> None:
    with pytest.raises(ValidationError, match="most recent evidence"):
        TrialDevSingleAssetProgrammeStateV1(
            programme_id="programme",
            scenario_id="scenario-1",
            stream_id="single_asset_development",
            current_checkpoint_id="early_safety_study",
            candidate_asset_ids=("drug-a",),
            nominated_asset_id="drug-a",
            active_asset_id="drug-a",
            policy_binding=TrialDevPolicyBindingV1(
                stream_id="single_asset_development",
                objective_id="benefit_risk",
                objective_policy_checksum="a" * 64,
                action_policy_checksum="b" * 64,
                design_menu_checksum="c" * 64,
            ),
            evidence=(
                TrialDevEvidenceReferenceV1(
                    evidence_id="future",
                    evidence_kind="dataset",
                    checkpoint_id="confirmation",
                    asset_id="drug-a",
                    evidence_protocol_id="confirmation",
                    evidence_protocol_checksum="a" * 64,
                    source_family_id="programme",
                    world_id="world",
                    generation_seed=1,
                    relative_path="future/confirmation.json",
                    artifact_sha256="d" * 64,
                ),
            ),
        )


def _normal_standard_error(
    *,
    control_sample_size: int,
    treatment_sample_size: int,
    control_risk: float,
    treatment_risk: float,
) -> float:
    return float(
        (
            control_risk * (1.0 - control_risk) / control_sample_size
            + treatment_risk * (1.0 - treatment_risk) / treatment_sample_size
        )
        ** 0.5
    )


def _absolute_safety_reference(
    *,
    sample_size: int,
    decision_limit: float,
    planning_unacceptable_risk: float,
    confidence_level: float,
) -> tuple[float, int]:
    alpha = 1.0 - confidence_level
    critical_value = norm.ppf(0.5 + confidence_level / 2.0)
    for event_count in range(sample_size + 1):
        estimate = event_count / sample_size
        if event_count == 0:
            lower = 0.0
        elif event_count == sample_size:
            lower = beta.ppf(alpha / 2.0, sample_size, 1)
        else:
            lower = max(
                0.0,
                estimate - critical_value * (estimate * (1.0 - estimate) / sample_size) ** 0.5,
            )
        if lower > decision_limit:
            return float(binom.sf(event_count - 1, sample_size, planning_unacceptable_risk)), event_count
    return 0.0, sample_size + 1


@pytest.mark.parametrize(
    ("control_n", "treatment_n", "control_risk", "treatment_risk", "confidence_level"),
    [
        (2, 2, 0.02, 0.01, 0.500001),
        (17, 31, 0.25, 0.10, 0.90),
        (180, 220, 0.25, 0.18, 0.95),
        (10_000, 9_999, 0.99, 0.98, 0.999),
    ],
)
def test_risk_difference_power_matches_independent_normal_reference(
    control_n: int,
    treatment_n: int,
    control_risk: float,
    treatment_risk: float,
    confidence_level: float,
) -> None:
    standard_error = _normal_standard_error(
        control_sample_size=control_n,
        treatment_sample_size=treatment_n,
        control_risk=control_risk,
        treatment_risk=treatment_risk,
    )
    noncentrality = (control_risk - treatment_risk) / standard_error
    critical_value = norm.ppf(0.5 + confidence_level / 2.0)
    reference = norm.sf(critical_value - noncentrality) + norm.cdf(-critical_value - noncentrality)
    actual = risk_difference_power_v1(
        control_sample_size=control_n,
        treatment_sample_size=treatment_n,
        control_risk=control_risk,
        treatment_risk=treatment_risk,
        alternative_benefit=control_risk - treatment_risk,
        confidence_level=confidence_level,
    )

    assert actual == pytest.approx(reference, abs=1e-14)


@pytest.mark.parametrize("confidence_level", [0.500001, 0.80, 0.95, 0.999])
def test_risk_difference_power_has_declared_two_sided_null_size(confidence_level: float) -> None:
    actual = risk_difference_power_v1(
        control_sample_size=250,
        treatment_sample_size=400,
        control_risk=0.20,
        treatment_risk=0.20,
        alternative_benefit=0.0,
        confidence_level=confidence_level,
    )

    assert actual == pytest.approx(1.0 - confidence_level, abs=1e-14)


@pytest.mark.parametrize(
    ("control_n", "treatment_n", "control_risk", "treatment_risk", "decision_limit", "confidence_level"),
    [
        (2, 3, 0.0, 0.50, 0.10, 0.500001),
        (25, 40, 0.05, 0.30, 0.05, 0.95),
        (350, 275, 0.40, 0.55, 0.10, 0.90),
        (10_000, 9_999, 0.90, 0.99, 0.01, 0.999),
    ],
)
def test_harmful_risk_difference_power_matches_independent_normal_reference(
    control_n: int,
    treatment_n: int,
    control_risk: float,
    treatment_risk: float,
    decision_limit: float,
    confidence_level: float,
) -> None:
    standard_error = _normal_standard_error(
        control_sample_size=control_n,
        treatment_sample_size=treatment_n,
        control_risk=control_risk,
        treatment_risk=treatment_risk,
    )
    reference = norm.cdf(
        ((treatment_risk - control_risk) - decision_limit) / standard_error - norm.ppf(0.5 + confidence_level / 2.0)
    )
    actual = harmful_risk_difference_power_v1(
        control_sample_size=control_n,
        treatment_sample_size=treatment_n,
        control_risk=control_risk,
        treatment_risk=treatment_risk,
        decision_limit=decision_limit,
        confidence_level=confidence_level,
    )

    assert actual == pytest.approx(reference, abs=1e-14)


@pytest.mark.parametrize(
    ("sample_size", "decision_limit", "planning_risk", "confidence_level"),
    [
        (2, 0.10, 0.30, 0.500001),
        (25, 0.10, 0.30, 0.95),
        (200, 0.05, 0.20, 0.90),
        (10_000, 0.10, 0.30, 0.999),
    ],
)
def test_safety_decision_power_matches_independent_binomial_reference(
    sample_size: int,
    decision_limit: float,
    planning_risk: float,
    confidence_level: float,
) -> None:
    reference = _absolute_safety_reference(
        sample_size=sample_size,
        decision_limit=decision_limit,
        planning_unacceptable_risk=planning_risk,
        confidence_level=confidence_level,
    )

    assert safety_decision_power_v1(
        sample_size=sample_size,
        decision_limit=decision_limit,
        planning_unacceptable_risk=planning_risk,
        confidence_level=confidence_level,
    ) == pytest.approx(reference, abs=1e-14)


def test_design_powers_are_monotone_in_information_and_harm() -> None:
    efficacy = [
        risk_difference_power_v1(
            control_sample_size=sample_size,
            treatment_sample_size=sample_size,
            control_risk=0.30,
            treatment_risk=0.20,
            alternative_benefit=0.10,
            confidence_level=0.95,
        )
        for sample_size in (10, 50, 250, 1_000)
    ]
    excess_safety = [
        harmful_risk_difference_power_v1(
            control_sample_size=sample_size,
            treatment_sample_size=sample_size,
            control_risk=0.05,
            treatment_risk=0.30,
            decision_limit=0.05,
            confidence_level=0.95,
        )
        for sample_size in (10, 50, 250, 1_000)
    ]
    absolute_safety = [
        safety_decision_power_v1(
            sample_size=100,
            decision_limit=0.10,
            planning_unacceptable_risk=planning_risk,
            confidence_level=0.95,
        )[0]
        for planning_risk in (0.11, 0.20, 0.30, 0.50)
    ]

    assert efficacy == sorted(efficacy)
    assert excess_safety == sorted(excess_safety)
    assert absolute_safety == sorted(absolute_safety)


def test_power_calculators_preserve_small_sample_and_decision_boundaries() -> None:
    assert (
        risk_difference_power_v1(
            control_sample_size=1,
            treatment_sample_size=20,
            control_risk=0.20,
            treatment_risk=0.10,
            alternative_benefit=0.10,
            confidence_level=0.95,
        )
        == 0.0
    )
    assert (
        harmful_risk_difference_power_v1(
            control_sample_size=20,
            treatment_sample_size=1,
            control_risk=0.05,
            treatment_risk=0.30,
            decision_limit=0.05,
            confidence_level=0.95,
        )
        == 0.0
    )
    assert safety_decision_power_v1(
        sample_size=1,
        decision_limit=0.10,
        planning_unacceptable_risk=0.30,
        confidence_level=0.95,
    ) == (0.0, 2)
    boundary_power = harmful_risk_difference_power_v1(
        control_sample_size=100,
        treatment_sample_size=100,
        control_risk=0.10,
        treatment_risk=0.30,
        decision_limit=0.20 - 1e-12,
        confidence_level=0.95,
    )
    assert boundary_power == pytest.approx(0.025, abs=1e-10)


@pytest.mark.parametrize(
    ("decision_limit", "planning_risk", "confidence_level"),
    [
        (0.0, 0.30, 0.95),
        (0.10, 0.10, 0.95),
        (0.30, 0.10, 0.95),
        (0.10, 1.0, 0.95),
        (0.10, 0.30, 0.5),
        (0.10, 0.30, 1.0),
    ],
)
def test_safety_decision_power_rejects_inputs_outside_declared_range(
    decision_limit: float,
    planning_risk: float,
    confidence_level: float,
) -> None:
    with pytest.raises(ValueError):
        safety_decision_power_v1(
            sample_size=100,
            decision_limit=decision_limit,
            planning_unacceptable_risk=planning_risk,
            confidence_level=confidence_level,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "control_risk": 0.10,
            "treatment_risk": 0.20,
            "alternative_benefit": -0.10,
            "confidence_level": 0.95,
        },
        {
            "control_risk": 0.20,
            "treatment_risk": 0.10,
            "alternative_benefit": 0.05,
            "confidence_level": 0.95,
        },
        {
            "control_risk": 0.20,
            "treatment_risk": 0.10,
            "alternative_benefit": 0.10,
            "confidence_level": 0.5,
        },
        {
            "control_risk": 0.20,
            "treatment_risk": 0.10,
            "alternative_benefit": 0.10,
            "confidence_level": 1.0,
        },
    ],
)
def test_risk_difference_power_rejects_inputs_outside_declared_range(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        risk_difference_power_v1(control_sample_size=100, treatment_sample_size=100, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "control_risk": 0.20,
            "treatment_risk": 0.10,
            "decision_limit": 0.05,
            "confidence_level": 0.95,
        },
        {
            "control_risk": 0.10,
            "treatment_risk": 0.20,
            "decision_limit": 0.10,
            "confidence_level": 0.95,
        },
        {
            "control_risk": 0.10,
            "treatment_risk": 0.20,
            "decision_limit": 0.05,
            "confidence_level": 0.5,
        },
    ],
)
def test_harmful_risk_difference_power_rejects_inputs_outside_declared_range(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        harmful_risk_difference_power_v1(control_sample_size=100, treatment_sample_size=100, **kwargs)


def _phase_modules() -> tuple[PhaseModuleSpecV1, ...]:
    common = {
        "allowed_site_count_budgets": (1,),
        "allowed_allocation_ratios": ("1:1",),
        "max_sample_size": 3000,
        "max_analysis_covariates": 0,
        "max_subgroup_splits": 0,
        "allowed_interim_policies": ("fixed_final",),
        "allowed_site_strategies": ("high_enrolling",),
        "allowed_selection_objectives": ("benefit_risk",),
    }
    return (
        PhaseModuleSpecV1(
            phase_id="phase1",
            allowed_follow_up_days=(28,),
            allowed_enrollment_window_days=(28,),
            **common,
        ),
        PhaseModuleSpecV1(
            phase_id="phase2",
            allowed_endpoint_ids=("DISEASE_PROGRESSION",),
            allowed_follow_up_days=(90,),
            allowed_enrollment_window_days=(90,),
            allowed_treatment_discontinuation_strategies=("treatment_policy",),
            **common,
        ),
        PhaseModuleSpecV1(
            phase_id="phase3",
            allowed_endpoint_ids=("HARD_ENDPOINT",),
            allowed_follow_up_days=(365,),
            allowed_enrollment_window_days=(365,),
            allowed_treatment_discontinuation_strategies=("treatment_policy",),
            **common,
        ),
    )


def _write_scenario(root: Path) -> None:
    modules = _phase_modules()
    _write_json(
        root / "public" / "eval_contract.json",
        TrialDevelopmentEvalContractV1(
            scenario_id="s01",
            phase_modules=modules,
        ).model_dump(mode="json", exclude_none=True),
    )
    _write_json(
        root / "public" / "phase_module_catalog.json",
        {
            "phase_modules": [module.model_dump(mode="json", exclude_none=True) for module in modules],
        },
    )
    _write_json(
        root / "public" / "program_loop_manifest.json",
        TrialDevelopmentProgramLoopManifestV1(
            scenario_id="s01",
            program_archetype="asset_development",
            phase_order=("observational_review", "phase1", "phase2", "phase3"),
            conditionally_materializable_phase_ids=("phase1", "phase2", "phase3"),
            phase_policy_modes={
                "phase1": "required",
                "phase2": "required",
                "phase3": "optional",
            },
            phase1_carryover_consequential=False,
            decision_charter_checksum="0" * 64,
            terminal_statuses=("stopped", "completed"),
            public_state_summary_fields=(
                "scenario_id",
                "current_phase_id",
                "eligible_candidate_drug_ids",
                "completed_phase_ids",
            ),
        ).model_dump(mode="json", exclude_none=True),
    )
    _write_json(
        root / "public" / "candidate_drug_catalog.json",
        {
            "candidate_drugs": [
                {"candidate_drug_id": "control", "role": "control"},
                {"candidate_drug_id": "drug_a", "role": "investigational"},
            ]
        },
    )
    _write_json(
        root / "public" / "objective_charter.json",
        {
            "objectives": [
                {
                    "objective_id": "benefit_risk",
                }
            ]
        },
    )
    _write_json(
        root / "public" / "phase_action_policy.json",
        {
            "action_specs": _action_specs(),
        },
    )
    _write_json(
        root / "public" / "phase_decision_evidence_policy.json",
        {
            "schema_id": "trialdev_phase_decision_evidence_policy_v1",
            "confidence_level": 0.95,
            "phase_rules": [
                {"phase_id": "phase1", "evaluation_horizon_days": 28, "minimum_benefit": None},
                {
                    "phase_id": "phase2",
                    "evaluation_horizon_days": 90,
                    "efficacy_endpoint_column": "EVENT",
                    "time_column": "TIME",
                    "minimum_benefit": 0.02,
                    "sensitivity_minimum_benefits": [0.01, 0.02, 0.03],
                },
                {
                    "phase_id": "phase3",
                    "evaluation_horizon_days": 365,
                    "efficacy_endpoint_column": "EVENT",
                    "time_column": "TIME",
                    "minimum_benefit": 0.015,
                    "sensitivity_minimum_benefits": [0.01, 0.015, 0.02],
                },
            ],
        },
    )
    _write_json(
        root / "public" / "phase_design_policy.json",
        _checksummed(
            {
                "schema_id": "trialdev_phase_design_policy_v1",
                "version": "v1",
                "scenario_id": "s01",
                "decision_charter_checksum": "0" * 64,
                "confidence_level": 0.95,
                "efficacy_test": "two_sided_normal_approximation_risk_difference",
                "safety_assurance": ("minimum_power_across_absolute_and_excess_serious_ae_hard_gates"),
                "source_artifact_checksums": {"public/test.json": "0" * 64},
                "phase_rules": [_design_rule(phase) for phase in ("phase1", "phase2", "phase3")],
            }
        ),
    )
    _write_json(
        root / "public" / "safety_decision_policy.json",
        {
            "serious_event_definitions": [
                {
                    "endpoint_id": "test",
                    "event_column": "AE_TEST_EVENT_E",
                    "time_column": "AE_TEST_EVENT_T",
                    "seriousness_column": "AE_TEST_SERIOUS",
                    "severity_column": "AE_TEST_SEVERITY",
                }
            ],
            "thresholds": [
                {
                    "phase_id": phase,
                    "evaluation_horizon_days": {"phase1": 28, "phase2": 90, "phase3": 365}[phase],
                    "component_id": "serious_ae",
                    "role": "hard_gate",
                    "max_absolute_rate": 0.10,
                    "max_excess_vs_control": 0.035,
                    "sensitivity_max_absolute_rates": {
                        "strict": 0.08,
                        "primary": 0.10,
                        "permissive": 0.12,
                    },
                    "sensitivity_max_excess_vs_control": {
                        "strict": 0.015,
                        "primary": 0.035,
                        "permissive": 0.055,
                    },
                }
                for phase in ("phase1", "phase2", "phase3")
            ],
        },
    )


def _write_trial_output(
    root: Path,
    *,
    n_per_arm: int,
    control_events: int,
    treatment_events: int,
    treatment_serious_events: int = 0,
    treatment_serious_time: float = 30.0,
    request_follow_up_days: int = 90,
    phase_id: str = "phase2",
    endpoint_id: str | None = "DISEASE_PROGRESSION",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    arms = ["CONTROL"] * n_per_arm + ["TREATMENT"] * n_per_arm
    event = (
        [1] * control_events
        + [0] * (n_per_arm - control_events)
        + [1] * treatment_events
        + [0] * (n_per_arm - treatment_events)
    )
    endpoints = pd.DataFrame(
        {
            "USUBJID": [f"P{index:05d}" for index in range(2 * n_per_arm)],
            "ARM": arms,
            "EVENT": event,
            "COMPETING_EVENT": [0] * (2 * n_per_arm),
            "TIME": [30.0 if value else 90.0 for value in event],
            "TERMINAL_EVENT": event,
            "TERMINAL_TIME": [30.0 if value else 90.0 for value in event],
        }
    )
    safety = pd.DataFrame(
        {
            "USUBJID": endpoints["USUBJID"],
            "ARM": arms,
            "AE_TEST_SERIOUS": [0] * n_per_arm
            + [1] * treatment_serious_events
            + [0] * (n_per_arm - treatment_serious_events),
            "AE_TEST_EVENT_E": [0] * n_per_arm
            + [1] * treatment_serious_events
            + [0] * (n_per_arm - treatment_serious_events),
            "AE_TEST_EVENT_T": [90.0] * n_per_arm
            + [treatment_serious_time] * treatment_serious_events
            + [90.0] * (n_per_arm - treatment_serious_events),
            "LTFU_E": [0] * (2 * n_per_arm),
            "LTFU_T": [float(request_follow_up_days)] * (2 * n_per_arm),
            "TERMINAL_EVENT": [0] * (2 * n_per_arm),
            "TERMINAL_TIME": [float(request_follow_up_days)] * (2 * n_per_arm),
        }
    )
    endpoints.to_parquet(root / "endpoints.parquet", index=False)
    safety.to_parquet(root / "safety.parquet", index=False)
    _write_json(
        root / "arm_mapping.json",
        {
            "control_arm_id": "CONTROL",
            "candidate_arm_ids": ["TREATMENT"],
            "drug_id_by_arm": {"CONTROL": "control", "TREATMENT": "drug_a"},
        },
    )
    _write_json(
        root / "request.json",
        {
            "version": "v1",
            "scenario_id": "s01",
            "phase_id": phase_id,
            "candidate_drug_ids": ["drug_a"],
            "target_sample_size": 2 * n_per_arm,
            "follow_up_days": request_follow_up_days,
            "endpoint_id": endpoint_id,
            "enrollment_window_days": request_follow_up_days,
            "site_count_budget": 1,
            "allocation_ratio": "1:1",
            "design_cell_id": f"trialdev.{phase_id}.fixed_final_operating_characteristics.v1",
            **({} if phase_id == "phase1" else {"treatment_discontinuation_strategy": "treatment_policy"}),
            "interim_policy": "fixed_final",
            "site_strategy": "high_enrolling",
            "selection_objective": "benefit_risk",
        },
    )
    _write_json(
        root / "execution_summary.json",
        {
            "payload": {
                "loss_to_follow_up_assignment": "arm_conditional_random_permutation_v1",
            }
        },
    )


def test_phase2_clear_benefit_identifies_advancement(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(output, n_per_arm=1000, control_events=100, treatment_events=40)

    witness = derive_phase_decision_witness_v1(
        scenario_root=scenario,
        trial_output_root=output,
        phase_id="phase2",
    )

    assert witness.recoverability_class == "unique"
    assert witness.acceptable_action_ids == ("advance_to_confirmation",)
    assert witness.safety_state == "acceptable"
    assert set(witness.evidence["source_checksums"]) == {
        "public/phase_action_policy.json",
        "public/phase_decision_evidence_policy.json",
        "public/safety_decision_policy.json",
        "trial_output/arm_mapping.json",
        "trial_output/endpoints.parquet",
        "trial_output/execution_summary.json",
        "trial_output/request.json",
        "trial_output/safety.parquet",
    }
    sensitivity = witness.evidence["efficacy"]["margin_sensitivity_action_sets"]
    assert sensitivity["0.010000"] == ["advance_to_confirmation"]
    assert sensitivity["0.030000"] == ["advance_to_confirmation"]
    assert set(witness.evidence["safety"]["sensitivity_states"]) == {
        "strict",
        "primary",
        "permissive",
    }


def test_phase_decision_uses_declared_safety_columns_without_name_inference(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(output, n_per_arm=1000, control_events=100, treatment_events=40)
    policy_path = scenario / "public" / "safety_decision_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["serious_event_definitions"] = [
        {
            "endpoint_id": "declared_signal",
            "event_column": "OBSERVED_SIGNAL",
            "time_column": "SIGNAL_DAY",
            "seriousness_column": "CLINICALLY_SERIOUS",
            "severity_column": "CLINICAL_SEVERITY",
        }
    ]
    _write_json(policy_path, policy)
    safety = pd.read_parquet(output / "safety.parquet").rename(
        columns={
            "AE_TEST_EVENT_E": "OBSERVED_SIGNAL",
            "AE_TEST_EVENT_T": "SIGNAL_DAY",
            "AE_TEST_SERIOUS": "CLINICALLY_SERIOUS",
        }
    )
    safety.to_parquet(output / "safety.parquet", index=False)

    witness = derive_phase_decision_witness_v1(
        scenario_root=scenario,
        trial_output_root=output,
        phase_id="phase2",
    )

    assert witness.acceptable_action_ids == ("advance_to_confirmation",)


def test_phase_design_witness_distinguishes_adequate_and_underpowered_requests(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    adequate_output = tmp_path / "adequate"
    underpowered_output = tmp_path / "underpowered"
    _write_scenario(scenario)
    _write_trial_output(adequate_output, n_per_arm=1000, control_events=100, treatment_events=60)
    _write_trial_output(underpowered_output, n_per_arm=10, control_events=1, treatment_events=0)

    adequate = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=_request(adequate_output),
        trial_output_root=adequate_output,
        phase_id="phase2",
    )
    underpowered = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=_request(underpowered_output),
        trial_output_root=underpowered_output,
        phase_id="phase2",
    )

    assert adequate.adequate is True
    assert adequate.achieved_power is not None and adequate.achieved_power >= 0.80
    assert underpowered.adequate is False
    assert set(underpowered.failures) == {
        "insufficient_efficacy_power",
        "insufficient_safety_decision_power",
    }


def test_phase_design_witness_rejects_unregistered_submitted_cell(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    trial_output = tmp_path / "trial_output"
    _write_scenario(scenario)
    _write_trial_output(
        trial_output,
        n_per_arm=1000,
        control_events=100,
        treatment_events=60,
    )
    request = _request(trial_output).model_copy(update={"design_cell_id": "trialdev.phase2.unregistered.v1"})

    witness = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=request,
        phase_id="phase2",
    )

    assert witness.adequate is False
    assert "unaccepted_design_cell" in witness.failures


def test_phase1_design_power_matches_the_safety_decision_boundary(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    adequate_output = tmp_path / "adequate"
    underpowered_output = tmp_path / "underpowered"
    _write_scenario(scenario)
    _write_trial_output(
        adequate_output,
        n_per_arm=51,
        control_events=0,
        treatment_events=0,
        request_follow_up_days=28,
        phase_id="phase1",
        endpoint_id=None,
    )
    _write_trial_output(
        underpowered_output,
        n_per_arm=10,
        control_events=0,
        treatment_events=0,
        request_follow_up_days=28,
        phase_id="phase1",
        endpoint_id=None,
    )

    adequate = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=_request(adequate_output),
        trial_output_root=adequate_output,
        phase_id="phase1",
    )
    underpowered = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=_request(underpowered_output),
        trial_output_root=underpowered_output,
        phase_id="phase1",
    )

    assert adequate.adequate is True
    assert adequate.achieved_safety_absolute_risk_power >= 0.80
    assert adequate.achieved_safety_excess_risk_power >= 0.80
    assert underpowered.adequate is False
    assert underpowered.failures == ("insufficient_safety_decision_power",)


def test_phase_design_power_uses_public_followup_information_fraction(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(output, n_per_arm=1000, control_events=100, treatment_events=60)
    policy_path = scenario / "public" / "phase_design_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    phase2 = next(rule for rule in policy["phase_rules"] if rule["phase_id"] == "phase2")
    phase2["planning_information_fraction_by_drug_id"] = {"control": 0.001, "drug_a": 0.001}
    policy.pop("checksum")
    policy["checksum"] = compute_sha256_hex(policy)
    _write_json(policy_path, policy)

    witness = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=_request(output),
        trial_output_root=output,
        phase_id="phase2",
    )

    assert witness.evidence["effective_control_arm_count"] == 1
    assert witness.evidence["effective_treatment_arm_counts"] == {"TREATMENT": 1}
    assert set(witness.failures) == {
        "insufficient_efficacy_power",
        "insufficient_safety_decision_power",
    }


def test_phase1_acceptable_safety_identifies_advancement(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(
        output,
        n_per_arm=1000,
        control_events=0,
        treatment_events=0,
        request_follow_up_days=28,
        phase_id="phase1",
        endpoint_id=None,
    )

    witness = derive_phase_decision_witness_v1(
        scenario_root=scenario,
        trial_output_root=output,
        phase_id="phase1",
    )

    assert witness.safety_state == "acceptable"
    assert witness.recoverability_class == "unique"
    assert witness.acceptable_action_ids == ("advance_to_proof_of_concept",)


def test_phase1_indeterminate_safety_preserves_both_actions(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(
        output,
        n_per_arm=10,
        control_events=0,
        treatment_events=0,
        request_follow_up_days=28,
        phase_id="phase1",
        endpoint_id=None,
    )

    witness = derive_phase_decision_witness_v1(
        scenario_root=scenario,
        trial_output_root=output,
        phase_id="phase1",
    )

    assert witness.safety_state == "indeterminate"
    assert witness.recoverability_class == "set_identified"
    assert set(witness.acceptable_action_ids) == {
        "advance_to_proof_of_concept",
        "stop_development",
    }


def test_phase2_indeterminate_safety_preserves_stop_despite_clear_efficacy(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(
        output,
        n_per_arm=1000,
        control_events=100,
        treatment_events=60,
        treatment_serious_events=30,
    )

    witness = derive_phase_decision_witness_v1(
        scenario_root=scenario,
        trial_output_root=output,
        phase_id="phase2",
    )

    assert witness.safety_state == "indeterminate"
    assert witness.acceptable_action_ids == ("advance_to_confirmation", "stop_development")


def test_phase2_uncertainty_credits_all_non_excluded_actions(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(output, n_per_arm=100, control_events=10, treatment_events=9)

    witness = derive_phase_decision_witness_v1(
        scenario_root=scenario,
        trial_output_root=output,
        phase_id="phase2",
    )

    assert witness.recoverability_class == "set_identified"
    assert set(witness.acceptable_action_ids) == {
        "advance_to_confirmation",
        "stop_development",
    }
    efficacy = witness.evidence["efficacy"]
    lower, upper = efficacy["confidence_interval"]
    assert lower <= efficacy["minimum_benefit"] <= upper


@pytest.mark.parametrize(
    ("control_events", "treatment_events", "expected_action"),
    [
        (300, 230, "declare_success"),
        (300, 285, "declare_inconclusive"),
        (300, 330, "declare_failure"),
    ],
)
def test_phase3_uses_a_unique_confirmatory_conclusion(
    tmp_path: Path,
    control_events: int,
    treatment_events: int,
    expected_action: str,
) -> None:
    """Phase-3 uncertainty supports an inconclusive result, not optional success."""

    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(
        output,
        n_per_arm=1000,
        control_events=control_events,
        treatment_events=treatment_events,
        request_follow_up_days=365,
        phase_id="phase3",
        endpoint_id="HARD_ENDPOINT",
    )

    witness = derive_phase_decision_witness_v1(
        scenario_root=scenario,
        trial_output_root=output,
        phase_id="phase3",
    )

    assert witness.acceptable_action_ids == (expected_action,)


def test_hard_safety_gate_precedes_efficacy(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(
        output,
        n_per_arm=1000,
        control_events=100,
        treatment_events=20,
        treatment_serious_events=200,
    )

    witness = derive_phase_decision_witness_v1(
        scenario_root=scenario,
        trial_output_root=output,
        phase_id="phase2",
    )

    assert witness.recoverability_class == "safety_determined"
    assert witness.acceptable_action_ids == ("stop_development",)
    assert witness.efficacy_action_ids == ("advance_to_confirmation",)


def test_safety_gate_uses_declared_horizon_not_longer_requested_follow_up(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(
        output,
        n_per_arm=1000,
        control_events=100,
        treatment_events=50,
        treatment_serious_events=200,
        treatment_serious_time=120.0,
        request_follow_up_days=150,
    )

    witness = derive_phase_decision_witness_v1(
        scenario_root=scenario,
        trial_output_root=output,
        phase_id="phase2",
    )

    assert witness.safety_state == "acceptable"
    assert "advance_to_confirmation" in witness.acceptable_action_ids
    candidate = witness.evidence["candidates"]["drug_a"]
    assert candidate["safety"]["evaluation_horizon_days"] == 90.0


def test_decision_witness_rejects_follow_up_shorter_than_policy_horizon(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(
        output,
        n_per_arm=100,
        control_events=10,
        treatment_events=5,
        request_follow_up_days=60,
    )

    with pytest.raises(ValueError, match="shorter than its decision horizon"):
        derive_phase_decision_witness_v1(
            scenario_root=scenario,
            trial_output_root=output,
            phase_id="phase2",
        )


def test_decision_witness_rejects_multiple_investigational_regimens(
    tmp_path: Path,
) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(output, n_per_arm=1000, control_events=100, treatment_events=50)
    endpoints = pd.read_parquet(output / "endpoints.parquet")
    safety = pd.read_parquet(output / "safety.parquet")
    endpoints.loc[endpoints["ARM"].eq("TREATMENT"), "ARM"] = "TREATMENT_A"
    safety.loc[safety["ARM"].eq("TREATMENT"), "ARM"] = "TREATMENT_A"
    second_endpoints = endpoints.loc[endpoints["ARM"].eq("TREATMENT_A")].copy()
    second_endpoints["ARM"] = "TREATMENT_B"
    second_endpoints["USUBJID"] = "B_" + second_endpoints["USUBJID"].astype(str)
    second_safety = safety.loc[safety["ARM"].eq("TREATMENT_A")].copy()
    second_safety["ARM"] = "TREATMENT_B"
    second_safety["USUBJID"] = "B_" + second_safety["USUBJID"].astype(str)
    second_safety["AE_TEST_SERIOUS"] = 1
    second_safety["AE_TEST_EVENT_E"] = 1
    second_safety["AE_TEST_EVENT_T"] = 30.0
    pd.concat([endpoints, second_endpoints], ignore_index=True).to_parquet(output / "endpoints.parquet", index=False)
    pd.concat([safety, second_safety], ignore_index=True).to_parquet(output / "safety.parquet", index=False)
    _write_json(
        output / "arm_mapping.json",
        {
            "control_arm_id": "CONTROL",
            "candidate_arm_ids": ["TREATMENT_A", "TREATMENT_B"],
            "drug_id_by_arm": {
                "CONTROL": "control",
                "TREATMENT_A": "drug_a",
                "TREATMENT_B": "drug_b",
            },
        },
    )
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    request["candidate_drug_ids"] = ["drug_a", "drug_b"]
    _write_json(output / "request.json", request)

    with pytest.raises(ValidationError, match="exactly one investigational regimen"):
        derive_phase_decision_witness_v1(
            scenario_root=scenario,
            trial_output_root=output,
            phase_id="phase2",
        )


def test_decision_witness_rejects_candidate_arm_mapping_drift(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(output, n_per_arm=100, control_events=10, treatment_events=5)
    mapping_path = output / "arm_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["drug_id_by_arm"]["TREATMENT"] = "another_drug"
    _write_json(mapping_path, mapping)

    with pytest.raises(ValueError, match="exactly cover"):
        derive_phase_decision_witness_v1(
            scenario_root=scenario,
            trial_output_root=output,
            phase_id="phase2",
        )


def test_missing_public_decision_policy_fails(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    (scenario / "public" / "phase_decision_evidence_policy.json").unlink()
    _write_trial_output(output, n_per_arm=100, control_events=10, treatment_events=9)

    with pytest.raises(FileNotFoundError):
        derive_phase_decision_witness_v1(
            scenario_root=scenario,
            trial_output_root=output,
            phase_id="phase2",
        )


def test_observed_safety_event_with_missing_seriousness_fails(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(output, n_per_arm=100, control_events=10, treatment_events=9)
    safety = pd.read_parquet(output / "safety.parquet")
    safety["AE_TEST_EVENT_E"] = 0
    treated_index = safety.index[safety["ARM"].eq("TREATMENT")][0]
    safety.loc[treated_index, "AE_TEST_EVENT_E"] = 1
    safety.loc[treated_index, "AE_TEST_SERIOUS"] = None
    safety.to_parquet(output / "safety.parquet", index=False)

    with pytest.raises(ValueError, match="Seriousness is missing for an observed safety event"):
        derive_phase_decision_witness_v1(
            scenario_root=scenario,
            trial_output_root=output,
            phase_id="phase2",
        )


def test_safety_event_after_ltfu_fails(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_scenario(scenario)
    _write_trial_output(output, n_per_arm=100, control_events=10, treatment_events=9)
    safety = pd.read_parquet(output / "safety.parquet")
    treated_index = safety.index[safety["ARM"].eq("TREATMENT")][0]
    safety.loc[treated_index, "AE_TEST_EVENT_E"] = 1
    safety.loc[treated_index, "AE_TEST_EVENT_T"] = 60.0
    safety.loc[treated_index, "AE_TEST_SERIOUS"] = 1
    safety.loc[treated_index, "LTFU_E"] = 1
    safety.loc[treated_index, "LTFU_T"] = 30.0
    safety.to_parquet(output / "safety.parquet", index=False)

    with pytest.raises(ValueError, match="after LTFU"):
        derive_phase_decision_witness_v1(
            scenario_root=scenario,
            trial_output_root=output,
            phase_id="phase2",
        )
