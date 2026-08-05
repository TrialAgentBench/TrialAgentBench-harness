"""Adversarial tests for prospective TrialDev method/design contracts."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from trialagentbench_harness.trialdev.grading.method_design import safety_submission_matches_cell_v1
from trialagentbench_harness.trialdev.grading.validate import _recompute_checksum
from trialagentbench_harness.trialdev.share.hashing import compute_sha256_hex
from trialagentbench_harness.trialdev.share.public_method_design import (
    TrialDevPhaseAnalysisMethodCatalogV1,
    TrialDevPhaseDesignCellV1,
    TrialDevPhaseDesignPolicyV1,
)
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentCandidateUtilityEstimateV1,
    TrialDevelopmentEffectEstimateV1,
    TrialDevelopmentSafetyEstimateV1,
)


def _method(phase_id: str) -> dict[str, object]:
    phase1 = phase_id == "phase1"
    return {
        "method_route_id": (
            "trialdev.phase1.aalen_johansen_safety_bundle.v1"
            if phase1
            else f"trialdev.{phase_id}.aalen_johansen_efficacy_safety.v1"
        ),
        "phase_id": phase_id,
        "calculator_id": ("aalen_johansen_safety_bundle_v1" if phase1 else "aalen_johansen_efficacy_safety_bundle_v1"),
        "estimator_id": "observed:aalen_johansen_cif_tau",
        "efficacy_estimand_id_template": (
            None if phase1 else "{treatment_discontinuation_strategy}:cumulative_incidence_at_horizon"
        ),
        "efficacy_effect_scale_id": (None if phase1 else "risk_difference_control_minus_treatment"),
        "efficacy_orientation_id": (None if phase1 else "positive_values_favour_treatment"),
        "safety_estimand_id_template": ("{safety_component_id}:cumulative_incidence_at_horizon"),
        "safety_absolute_risk_scale_id": "absolute_risk",
        "safety_excess_risk_scale_id": "risk_difference_treatment_minus_control",
        "safety_reported_measure_ids": [
            "treatment_absolute_risk",
            "control_absolute_risk",
            "risk_difference_treatment_minus_control",
        ],
        "safety_uncertainty_scope_id": ("two_sided_confidence_interval_per_safety_component_and_measure"),
        "safety_orientation_id": ("absolute_risk_higher_is_worse_and_risk_difference_positive_is_treatment_harm"),
        "result_shape": "safety_component_bundle" if phase1 else "efficacy_safety_bundle",
        "uncertainty_kind": "two_sided_confidence_interval",
        "confidence_level": 0.95,
        "horizon_source": "request.follow_up_days",
        "analysis_population": "all_randomized_participants",
        "censoring_assumption_id": "independent_censoring_conditional_on_randomized_arm",
        "loss_to_follow_up_construction_id": "arm_conditional_random_permutation_v1",
        "safety_component_ids": ["serious_ae", "discontinuation"],
    }


def test_grader_manifest_checksum_orders_distinct_method_routes_canonically() -> None:
    """Standalone validation must reproduce the canonical multi-method checksum."""

    records = [
        {
            "phase_id": "observational_review",
            "lane_id": "drug_ranking",
            "objective_id": "benefit_risk",
            "metric": "objective_score",
            "endpoint_id": None,
            "method_route_id": method_route_id,
            "candidate_drug_ids": [candidate_id],
            "value": value,
        }
        for method_route_id, candidate_id, value in (
            ("method_b", "candidate_a", 0.2),
            ("method_a", "candidate_b", 0.3),
            ("method_a", "candidate_a", 0.4),
        )
    ]
    canonical = {
        "version": "v1",
        "scenario_id": "scenario",
        "domain_id": "drug_ranking_reference",
        "records": sorted(
            records,
            key=lambda record: (
                str(record["phase_id"]),
                str(record["lane_id"]),
                str(record["objective_id"]),
                str(record["metric"]),
                str(record["endpoint_id"] or ""),
                str(record["method_route_id"]),
                tuple(str(value) for value in record["candidate_drug_ids"]),
            ),
        ),
    }
    payload = {**canonical, "records": list(reversed(records)), "checksum": compute_sha256_hex(canonical)}

    assert _recompute_checksum(payload, label="drug_ranking_reference_manifest.json") == payload["checksum"]


def _catalog_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "trialdev_phase_analysis_method_catalog_v1",
        "version": "v1",
        "scenario_id": "s01",
        "confidence_level": 0.95,
        "methods": [_method(phase) for phase in ("phase1", "phase2", "phase3")],
    }
    payload["checksum"] = compute_sha256_hex(payload)
    return payload


def _phase1_design_cell_payload() -> dict[str, object]:
    return {
        "design_cell_id": "trialdev.phase1.fixed_final_operating_characteristics.v1",
        "phase_id": "phase1",
        "calculator_id": "prospective_fixed_final_operating_characteristics_v1",
        "primary_endpoint_id": None,
        "planning_alternative_benefit": None,
        "target_power": None,
        "supported_interim_policy": "fixed_final",
        "confidence_level": 0.95,
        "evaluation_horizon_days": 90,
        "serious_ae_unacceptable_absolute_risk": 0.20,
        "serious_ae_unacceptable_excess_risk": 0.05,
        "planning_safety_control_risk": 0.05,
        "planning_safety_absolute_treatment_risk": 0.25,
        "planning_safety_excess_risk": 0.10,
        "planning_safety_excess_treatment_risk": 0.15,
        "target_safety_decision_power": 0.80,
        "safety_power_adequacy_rule": ("minimum_achieved_power_across_absolute_and_excess_hard_gates"),
        "planning_safety_estimator_id": ("multinomial_propensity_weighted_aalen_johansen_any_serious_ae"),
        "planning_safety_analysis_population": "complete_on_declared_adjustment_covariates",
        "planning_safety_control_support_count": 100,
        "planning_safety_min_observed_propensity": 0.1,
        "planning_safety_max_inverse_propensity_weight": 10.0,
        "planning_safety_weighted_effective_sample_size": 90.0,
        "planning_information_estimator_id": ("one_minus_multinomial_propensity_weighted_aalen_johansen_ltfu_cif"),
        "planning_information_fraction_by_drug_id": {"drug_a": 0.9},
        "planning_information_support_count_by_drug_id": {"drug_a": 100},
        "planning_information_weighted_effective_sample_size_by_drug_id": {"drug_a": 90.0},
        "planning_control_risk": None,
        "planning_treatment_risk": None,
        "planning_estimator_id": None,
        "planning_analysis_population": None,
        "planning_control_support_count": None,
        "planning_min_observed_propensity": None,
        "planning_max_inverse_propensity_weight": None,
        "planning_weighted_effective_sample_size": None,
        "rationale": "Prospective binary adequacy policy.",
    }


def _policy_payload() -> dict[str, object]:
    phase_rules: list[dict[str, object]] = []
    for phase_id in ("phase1", "phase2", "phase3"):
        rule = _phase1_design_cell_payload()
        rule["phase_id"] = phase_id
        rule["design_cell_id"] = f"trialdev.{phase_id}.fixed_final_operating_characteristics.v1"
        if phase_id != "phase1":
            rule.update(
                {
                    "primary_endpoint_id": "E1",
                    "planning_alternative_benefit": 0.1,
                    "target_power": 0.8,
                    "planning_control_risk": 0.4,
                    "planning_treatment_risk": 0.3,
                    "planning_estimator_id": "multinomial_propensity_weighted_aalen_johansen",
                    "planning_analysis_population": "complete_on_declared_adjustment_covariates",
                    "planning_control_support_count": 100,
                    "planning_min_observed_propensity": 0.1,
                    "planning_max_inverse_propensity_weight": 10.0,
                    "planning_weighted_effective_sample_size": 90.0,
                }
            )
        phase_rules.append(rule)
    payload: dict[str, object] = {
        "schema_id": "trialdev_phase_design_policy_v1",
        "version": "v1",
        "scenario_id": "s01",
        "decision_charter_checksum": "a" * 64,
        "confidence_level": 0.95,
        "efficacy_test": "two_sided_normal_approximation_risk_difference",
        "safety_assurance": "minimum_power_across_absolute_and_excess_serious_ae_hard_gates",
        "source_artifact_checksums": {"observational_extract.parquet": "b" * 64},
        "phase_rules": phase_rules,
    }
    payload["checksum"] = compute_sha256_hex(payload)
    return payload


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "planning_safety_absolute_treatment_risk",
            0.20,
            "absolute-risk planning alternative",
        ),
        ("planning_safety_excess_risk", 0.05, "excess-risk planning alternative"),
        (
            "planning_safety_excess_treatment_risk",
            0.14,
            "Derived treatment risk",
        ),
    ],
)
def test_phase_design_rejects_invalid_authored_safety_alternatives(
    field: str,
    value: float,
    message: str,
) -> None:
    payload = _phase1_design_cell_payload()
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        TrialDevPhaseDesignCellV1.model_validate(payload)


def test_phase_design_rejects_removed_sensitivity_margin() -> None:
    payload = _phase1_design_cell_payload()
    payload["planning_safety_sensitivity_margin"] = 0.05

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrialDevPhaseDesignCellV1.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    [
        "planning_safety_absolute_treatment_risk",
        "planning_safety_excess_risk",
        "planning_safety_control_risk",
        "planning_safety_excess_treatment_risk",
    ],
)
def test_phase_design_requires_every_safety_planning_input(field: str) -> None:
    payload = _phase1_design_cell_payload()
    payload.pop(field)

    with pytest.raises(ValidationError, match=field):
        TrialDevPhaseDesignCellV1.model_validate(payload)


def test_phase_design_rejects_control_plus_excess_at_or_above_one() -> None:
    payload = _phase1_design_cell_payload()
    payload.update(
        {
            "planning_safety_control_risk": 0.90,
            "planning_safety_excess_risk": 0.15,
            "planning_safety_excess_treatment_risk": 1.05,
        }
    )

    with pytest.raises(ValidationError, match="less than 1"):
        TrialDevPhaseDesignCellV1.model_validate(payload)


def test_method_catalog_rejects_checksum_mismatch() -> None:
    payload = _catalog_payload()
    payload["checksum"] = "f" * 64

    with pytest.raises(ValidationError, match="checksum mismatch"):
        TrialDevPhaseAnalysisMethodCatalogV1.model_validate(payload)


def test_phase1_design_with_null_efficacy_fields_round_trips() -> None:
    root_payload = _policy_payload()
    phase1_payload = root_payload["phase_rules"][0]

    assert phase1_payload["primary_endpoint_id"] is None
    package_policy = TrialDevPhaseDesignPolicyV1.model_validate(root_payload)
    assert package_policy.checksum == root_payload["checksum"]
    assert package_policy.model_dump(mode="json") == root_payload

    method_payload = _catalog_payload()
    assert {method["censoring_assumption_id"] for method in method_payload["methods"]} == {
        "independent_censoring_conditional_on_randomized_arm"
    }
    assert {method["loss_to_follow_up_construction_id"] for method in method_payload["methods"]} == {
        "arm_conditional_random_permutation_v1"
    }
    package_catalog = TrialDevPhaseAnalysisMethodCatalogV1.model_validate(method_payload)
    assert package_catalog.checksum == method_payload["checksum"]
    assert package_catalog.model_dump(mode="json") == method_payload


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            TrialDevelopmentEffectEstimateV1,
            {
                "evidence_id": "effect",
                "method_route_id": "cell",
                "candidate_drug_id": "drug_a",
                "endpoint_id": "E1",
                "estimand_id": "estimand",
                "estimator_id": "estimator",
                "effect_scale_id": "scale",
                "orientation_id": "orientation",
                "estimate": 0.1,
                "lower": 0.0,
                "upper": 0.2,
                "analysis_population": "population",
                "source_artifact_checksums": {"source": "a" * 64},
            },
        ),
        (
            TrialDevelopmentSafetyEstimateV1,
            {
                "evidence_id": "safety",
                "method_route_id": "cell",
                "candidate_drug_id": "drug_a",
                "estimator_id": "estimator",
                "estimand_ids": ["serious", "discontinuation"],
                "absolute_risk_scale_id": "absolute_risk",
                "excess_risk_scale_id": "risk_difference_treatment_minus_control",
                "orientation_id": ("absolute_risk_higher_is_worse_and_risk_difference_positive_is_treatment_harm"),
                "horizon_days": 90,
                "analysis_population": "population",
                "serious_ae_treatment_rate": 0.1,
                "serious_ae_treatment_lower": 0.0,
                "serious_ae_treatment_upper": 0.2,
                "serious_ae_control_rate": 0.05,
                "serious_ae_control_lower": 0.0,
                "serious_ae_control_upper": 0.1,
                "serious_ae_excess": 0.05,
                "serious_ae_excess_lower": -0.1,
                "serious_ae_excess_upper": 0.2,
                "discontinuation_treatment_rate": 0.1,
                "discontinuation_treatment_lower": 0.0,
                "discontinuation_treatment_upper": 0.2,
                "discontinuation_control_rate": 0.05,
                "discontinuation_control_lower": 0.0,
                "discontinuation_control_upper": 0.1,
                "discontinuation_excess": 0.05,
                "discontinuation_excess_lower": -0.1,
                "discontinuation_excess_upper": 0.2,
                "source_artifact_checksums": {"source": "a" * 64},
            },
        ),
        (
            TrialDevelopmentCandidateUtilityEstimateV1,
            {
                "evidence_id": "utility",
                "method_route_id": "cell",
                "candidate_drug_id": "drug_a",
                "objective_id": "benefit_risk",
                "estimator_id": "estimator",
                "estimate": 0.1,
                "lower": 0.0,
                "upper": 0.2,
                "analysis_covariate_ids": ["age"],
                "source_artifact_checksums": {"source": "a" * 64},
            },
        ),
    ],
)
def test_submitted_estimates_require_explicit_confidence(
    model: type[
        TrialDevelopmentEffectEstimateV1
        | TrialDevelopmentSafetyEstimateV1
        | TrialDevelopmentCandidateUtilityEstimateV1
    ],
    payload: dict[str, object],
) -> None:
    missing_confidence = copy.deepcopy(payload)

    with pytest.raises(ValidationError, match="confidence_level"):
        model.model_validate(missing_confidence)


def test_observational_submission_rejects_duplicate_covariates() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        TrialDevelopmentCandidateUtilityEstimateV1.model_validate(
            {
                "evidence_id": "utility",
                "method_route_id": (
                    "trialdev.observational.multinomial_propensity_weighted_stratified_aalen_johansen.v1"
                ),
                "candidate_drug_id": "drug_a",
                "objective_id": "benefit_risk",
                "estimator_id": "multinomial_propensity_weighted_stratified_aalen_johansen",
                "estimate": 0.1,
                "lower": 0.0,
                "upper": 0.2,
                "confidence_level": 0.95,
                "analysis_covariate_ids": ["AGE", "AGE"],
                "source_artifact_checksums": {"observational_extract.parquet": "a" * 64},
            }
        )


def test_safety_method_membership_is_estimand_order_invariant() -> None:
    catalog = TrialDevPhaseAnalysisMethodCatalogV1.model_validate(_catalog_payload())
    cell = next(method for method in catalog.methods if method.phase_id == "phase2")
    checksums = {"safety.parquet": "a" * 64}
    safety = TrialDevelopmentSafetyEstimateV1.model_validate(
        {
            "evidence_id": "safety",
            "method_route_id": cell.method_route_id,
            "candidate_drug_id": "drug_a",
            "estimator_id": cell.estimator_id,
            "estimand_ids": [
                "discontinuation:cumulative_incidence_at_horizon",
                "serious_ae:cumulative_incidence_at_horizon",
            ],
            "absolute_risk_scale_id": "absolute_risk",
            "excess_risk_scale_id": "risk_difference_treatment_minus_control",
            "orientation_id": cell.safety_orientation_id,
            "horizon_days": 90,
            "analysis_population": cell.analysis_population,
            "serious_ae_treatment_rate": 0.10,
            "serious_ae_treatment_lower": 0.05,
            "serious_ae_treatment_upper": 0.15,
            "serious_ae_control_rate": 0.05,
            "serious_ae_control_lower": 0.02,
            "serious_ae_control_upper": 0.08,
            "serious_ae_excess": 0.05,
            "serious_ae_excess_lower": 0.0,
            "serious_ae_excess_upper": 0.10,
            "discontinuation_treatment_rate": 0.08,
            "discontinuation_treatment_lower": 0.04,
            "discontinuation_treatment_upper": 0.12,
            "discontinuation_control_rate": 0.03,
            "discontinuation_control_lower": 0.01,
            "discontinuation_control_upper": 0.05,
            "discontinuation_excess": 0.05,
            "discontinuation_excess_lower": 0.0,
            "discontinuation_excess_upper": 0.10,
            "confidence_level": 0.95,
            "source_artifact_checksums": checksums,
        }
    )

    assert safety_submission_matches_cell_v1(
        safety=safety,
        cell=cell,
        phase_id="phase2",
        horizon_days=90,
        source_checksums=checksums,
    )


def test_safety_method_membership_resolves_semantics_without_catalog_id() -> None:
    catalog = TrialDevPhaseAnalysisMethodCatalogV1.model_validate(_catalog_payload())
    cell = next(method for method in catalog.methods if method.phase_id == "phase2")
    checksums = {"safety.parquet": "a" * 64}
    payload = {
        "evidence_id": "safety",
        "candidate_drug_id": "drug_a",
        "estimator_id": cell.estimator_id,
        "estimand_ids": [
            "serious_ae:cumulative_incidence_at_horizon",
            "discontinuation:cumulative_incidence_at_horizon",
        ],
        "absolute_risk_scale_id": "absolute_risk",
        "excess_risk_scale_id": "risk_difference_treatment_minus_control",
        "orientation_id": cell.safety_orientation_id,
        "horizon_days": 90,
        "analysis_population": cell.analysis_population,
        "serious_ae_treatment_rate": 0.10,
        "serious_ae_treatment_lower": 0.05,
        "serious_ae_treatment_upper": 0.15,
        "serious_ae_control_rate": 0.05,
        "serious_ae_control_lower": 0.02,
        "serious_ae_control_upper": 0.08,
        "serious_ae_excess": 0.05,
        "serious_ae_excess_lower": 0.0,
        "serious_ae_excess_upper": 0.10,
        "discontinuation_treatment_rate": 0.08,
        "discontinuation_treatment_lower": 0.04,
        "discontinuation_treatment_upper": 0.12,
        "discontinuation_control_rate": 0.03,
        "discontinuation_control_lower": 0.01,
        "discontinuation_control_upper": 0.05,
        "discontinuation_excess": 0.05,
        "discontinuation_excess_lower": 0.0,
        "discontinuation_excess_upper": 0.10,
        "confidence_level": 0.95,
        "source_artifact_checksums": checksums,
    }
    safety = TrialDevelopmentSafetyEstimateV1.model_validate(payload)

    assert safety_submission_matches_cell_v1(
        safety=safety,
        cell=cell,
        phase_id="phase2",
        horizon_days=90,
        source_checksums=checksums,
    )

    contradictory = TrialDevelopmentSafetyEstimateV1.model_validate(
        {**payload, "method_route_id": "trialdev.phase3.aalen_johansen_efficacy_safety.v1"}
    )
    assert not safety_submission_matches_cell_v1(
        safety=contradictory,
        cell=cell,
        phase_id="phase2",
        horizon_days=90,
        source_checksums=checksums,
    )


def test_safety_submission_rejects_incoherent_excess_and_obsolete_treated_names() -> None:
    payload = {
        "evidence_id": "safety",
        "method_route_id": "cell",
        "candidate_drug_id": "drug_a",
        "estimator_id": "estimator",
        "estimand_ids": ["serious", "discontinuation"],
        "absolute_risk_scale_id": "absolute_risk",
        "excess_risk_scale_id": "risk_difference_treatment_minus_control",
        "orientation_id": ("absolute_risk_higher_is_worse_and_risk_difference_positive_is_treatment_harm"),
        "horizon_days": 90,
        "analysis_population": "population",
        "serious_ae_treatment_rate": 0.10,
        "serious_ae_treatment_lower": 0.05,
        "serious_ae_treatment_upper": 0.15,
        "serious_ae_control_rate": 0.05,
        "serious_ae_control_lower": 0.02,
        "serious_ae_control_upper": 0.08,
        "serious_ae_excess": 0.04,
        "serious_ae_excess_lower": 0.0,
        "serious_ae_excess_upper": 0.10,
        "discontinuation_treatment_rate": 0.08,
        "discontinuation_treatment_lower": 0.04,
        "discontinuation_treatment_upper": 0.12,
        "discontinuation_control_rate": 0.03,
        "discontinuation_control_lower": 0.01,
        "discontinuation_control_upper": 0.05,
        "discontinuation_excess": 0.05,
        "discontinuation_excess_lower": 0.0,
        "discontinuation_excess_upper": 0.10,
        "confidence_level": 0.95,
        "source_artifact_checksums": {"safety.parquet": "a" * 64},
    }
    with pytest.raises(ValidationError, match="excess must equal treated rate minus control rate"):
        TrialDevelopmentSafetyEstimateV1.model_validate(payload)

    obsolete_alias_payload = dict(payload)
    obsolete_alias_payload["serious_ae_excess"] = 0.05
    obsolete_alias_payload["serious_ae_rate"] = obsolete_alias_payload.pop("serious_ae_treatment_rate")
    with pytest.raises(ValidationError, match="serious_ae_treatment_rate"):
        TrialDevelopmentSafetyEstimateV1.model_validate(obsolete_alias_payload)
