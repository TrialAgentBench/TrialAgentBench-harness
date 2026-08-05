from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevEvidenceReferenceV1,
    TrialDevPolicyBindingV1,
    TrialDevSingleAssetProgrammeStateV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevAnalysisQualityEndpointV1,
    TrialDevProgrammeAnalysisQualityV1,
)
from trialagentbench_harness.trialdev.grade_wrappers import (
    trajectory_metrics_from_grade,
    wrap_grade_record,
    wrap_trajectory_grade,
)
from trialagentbench_harness.trialdev.grading import reference_submissions
from trialagentbench_harness.trialdev.grading.decision_evidence import derive_phase_design_witness_v1
from trialagentbench_harness.trialdev.grading.design_frontier import (
    build_phase_design_frontiers_v1,
    derive_phase_design_efficiency_v1,
    derive_phase_resource_consequence_v1,
    derive_programme_resource_consequence_v1,
    load_phase_design_frontiers_v1,
)
from trialagentbench_harness.trialdev.grading.evaluation_target_register import (
    TrialDevEvaluationTargetRegisterRecordV1,
)
from trialagentbench_harness.trialdev.grading.grade import (
    PublicCandidateUtilityReferenceV1,
    PublicObservationalReferenceV1,
    _method_conditioned_asset_reference,
    _ranking_concordance,
    _safety_uncertainty_support_v1,
    grade_item_v1,
    grade_report_payload_v1,
)
from trialagentbench_harness.trialdev.grading.hashing import compute_sha256_hex, sha256_file_hex
from trialagentbench_harness.trialdev.grading.io import write_json
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevDesignEfficiencyV1,
    TrialDevDesignFrontierPointV1,
    TrialDevelopmentAnalysisQualityV1,
    TrialDevelopmentAuditGateReportV1,
    TrialDevelopmentGradeReportV1,
    TrialDevelopmentValidityReportV1,
)
from trialagentbench_harness.trialdev.grading.reference_submissions import (
    build_observational_reference_submission_v1,
    build_phase_reference_request_v1,
)
from trialagentbench_harness.trialdev.grading.sequential import _validate_decision_evidence_links, grade_trajectory_v1
from trialagentbench_harness.trialdev.grading.validate import _recompute_checksum, validate_release_v1
from trialagentbench_harness.trialdev.participant_submission import (
    build_phase_analysis_v1,
    build_phase_decision_v1,
)
from trialagentbench_harness.trialdev.share.models import (
    PhaseModuleSpecV1,
    TrialDevelopmentEvalContractV1,
    TrialDevelopmentRequestV1,
)
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentProgramLoopManifestV1,
    TrialDevelopmentSafetyEstimateV1,
    TrialDevelopmentTrialOutputManifestV1,
)
from trialagentbench_harness.trialdev.share.validate import (
    TrialDevelopmentRequestRejectedError,
    _validate_phase_analysis_method_catalog,
    candidate_ids_by_role_v1,
    validate_request_against_scenario_v1,
)


def test_asset_truth_is_bound_to_the_resolved_observational_method() -> None:
    base_payload = {
        "scenario_id": "scenario",
        "phase_id": "observational_review",
        "program_objective_id": "benefit_risk",
        "phase_scoring_objective_id": "benefit_risk",
        "lane_id": "asset_nomination",
        "scoring_policy_id": "static_union_not_scoreable",
        "public_evidence_basis": ("public/observational_extract.parquet",),
        "evaluator_evidence_basis": ("grader/public_recoverability_report.json",),
        "reference_target_ids": ("drug_from_other_method",),
        "recoverability_policy_id": "acceptable_candidate_set",
        "checksum": "a" * 64,
    }
    evaluation_target = TrialDevEvaluationTargetRegisterRecordV1.model_validate(base_payload)
    reference = PublicObservationalReferenceV1(
        method_route_id="declared_method",
        estimator_id="declared_estimator",
        effect_scale_id="utility",
        confidence_level=0.95,
        absolute_tolerance=0.0005,
        scientific_absolute_margin=0.02,
        adjustment_covariates=frozenset({"AGE"}),
        source_artifact_checksums={"public/observational_extract.parquet": "b" * 64},
        candidate_utilities={
            "drug_from_declared_method": PublicCandidateUtilityReferenceV1(
                estimate=1.0,
                lower=0.8,
                upper=1.2,
            )
        },
        reference_target_ids=("drug_from_declared_method",),
        credit_eligible_target_ids=("near_tie_for_declared_method",),
        recoverability_policy_id="acceptable_candidate_set",
    )

    conditioned = _method_conditioned_asset_reference(evaluation_target=evaluation_target, reference=reference)

    assert conditioned.reference_target_ids == ("drug_from_declared_method",)
    assert conditioned.credit_eligible_target_ids == ("near_tie_for_declared_method",)
    assert conditioned.value_payload["method_route_id"] == "declared_method"
    assert conditioned.checksum != evaluation_target.checksum


def test_candidate_catalog_requires_explicit_semantic_roles(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    write_json(
        public / "candidate_drug_catalog.json",
        {
            "candidate_drugs": [
                {"candidate_drug_id": "control"},
                {"candidate_drug_id": "drug_a", "role": "investigational"},
            ]
        },
    )

    with pytest.raises(ValueError, match="roles must be control or investigational"):
        candidate_ids_by_role_v1(scenario_root=tmp_path)


def test_candidate_catalog_roles_do_not_depend_on_identifier_spelling(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    write_json(
        public / "candidate_drug_catalog.json",
        {
            "candidate_drugs": [
                {"candidate_drug_id": "usual_care", "role": "control"},
                {"candidate_drug_id": "control_named_asset", "role": "investigational"},
            ]
        },
    )

    assert candidate_ids_by_role_v1(scenario_root=tmp_path) == {
        "control": ("usual_care",),
        "investigational": ("control_named_asset",),
    }


def test_ranking_concordance_rejects_incomplete_reference_scores() -> None:
    with pytest.raises(ValueError, match="Policy reference ranking is incomplete"):
        _ranking_concordance(
            ranked_drug_ids=("drug_a", "drug_b"),
            scores_by_drug={"drug_a": 0.3},
            candidate_drug_ids=("drug_a", "drug_b"),
        )


def _checked_payload(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["checksum"] = _recompute_checksum(out, label=label)
    return out


def _write_checked(path: Path, label: str, payload: dict[str, Any]) -> None:
    write_json(path, _checked_payload(label, payload))


def _observational_method_result(
    *,
    method_route_id: str,
    estimator_id: str,
    utility_offset: float = 0.0,
) -> dict[str, object]:
    values_by_objective = {
        "benefit_risk": {"drug_a": 0.8 + utility_offset, "drug_b": 0.2 + utility_offset},
        "cost_effective_best": {"drug_a": 0.3 + utility_offset, "drug_b": 0.9 + utility_offset},
    }
    scores: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    policies: list[dict[str, object]] = []
    action_policies: list[dict[str, object]] = []
    for objective_id, values in values_by_objective.items():
        ranked = sorted(values, key=lambda candidate: (-values[candidate], candidate))
        best = values[ranked[0]]
        scores.extend(
            {
                "schema_id": "trialdev_public_candidate_score_v1",
                "scenario_id": "s01",
                "objective_id": objective_id,
                "candidate_drug_id": candidate,
                "adjusted_utility": value,
                "utility_se": 0.05,
                "ci_low": value - 0.1,
                "ci_high": value + 0.1,
                "efficacy_gain": value,
                "efficacy_gain_se": 0.05,
                "efficacy_gain_ci_low": value - 0.1,
                "efficacy_gain_ci_high": value + 0.1,
                "rank": ranked.index(candidate) + 1,
                "utility_margin_to_best": best - value,
                "support_count": 100,
                "min_stratum_count": 20,
                "max_abs_unadjusted_smd_vs_target": 0.2,
                "max_abs_adjusted_smd_vs_target": 0.05,
                "point_estimable": True,
                "inference_estimable": True,
                "score_state": "scoreable",
            }
            for candidate, value in values.items()
        )
        candidate_utilities = [
            {
                "candidate_drug_id": candidate,
                "utility": values[candidate],
                "rank": rank,
            }
            for rank, candidate in enumerate(ranked, start=1)
        ]
        for comparison_estimator in (estimator_id, "raw_observed"):
            comparisons.append(
                {
                    "objective_id": objective_id,
                    "estimator_id": comparison_estimator,
                    "status": "estimated",
                    "reference_top_candidate_id": ranked[0],
                    "top_candidate_id": ranked[0],
                    "top_utility": best,
                    "utility_margin_to_reference_top": 0.0,
                    "agrees_with_reference_top": True,
                    "rank_order": ranked,
                    "candidate_utilities": candidate_utilities,
                    "policy_signal": "supports_unique_best",
                }
            )
        sensitivity_sets = {"0.025": [ranked[0]], "0.05": [ranked[0]], "0.1": [ranked[0]]}
        policies.append(
            {
                "objective_id": objective_id,
                "policy": "unique_best",
                "reference_target_ids": [ranked[0]],
                "acceptable_candidate_set": [ranked[0]],
                "near_tie_threshold": 0.05,
                "indifference_sensitivity_sets": sensitivity_sets,
                "preference_sensitivity_sets": {
                    "0.5": [ranked[0]],
                    "1": [ranked[0]],
                    "2": [ranked[0]],
                },
                "rationale": "The declared utility has one candidate outside the indifference margin.",
            }
        )
        action_policies.append(
            {
                "objective_id": objective_id,
                "minimum_efficacy_gain": 0.05,
                "reference_target_ids": [ranked[0]],
                "credit_eligible_target_ids": [ranked[0]],
                "definitely_qualified_candidate_ids": ["drug_a", "drug_b"],
                "possibly_qualified_candidate_ids": ["drug_a", "drug_b"],
                "utility_contrast_half_widths": {"drug_a": 0.1, "drug_b": 0.1},
                "pairwise_utility_contrast_half_widths": {"drug_a|drug_b": 0.1},
                "rationale": "Both candidates clear the efficacy gate; utility selects the action.",
            }
        )
    return {
        "method_route_id": method_route_id,
        "estimator_id": estimator_id,
        "candidate_scores": scores,
        "estimator_comparisons": comparisons,
        "objective_policies": policies,
        "observational_action_policies": action_policies,
        "diagnostics": {
            "method_route_id": method_route_id,
            "confounding_regime": "measured_with_overlap",
            "estimator_id": estimator_id,
            "baseline_covariates": ["AGE", "BASELINE_SEVERITY"],
            "stratum_count": 1,
            "candidate_count": 2,
            "source_row_count": 300,
            "analysis_row_count": 300,
            "excluded_missing_covariate_count": 0,
            "min_support_count": 100,
            "min_observed_treatment_probability": (
                0.2 if estimator_id == "multinomial_propensity_weighted_stratified_aalen_johansen" else None
            ),
            "maximum_analysis_weight": 5.0,
            "min_effective_sample_size": 80.0,
            "maximum_calibration_mean_error": 1e-8,
            "max_abs_unadjusted_smd_vs_target": 0.2,
            "max_abs_adjusted_smd_vs_target": 0.05,
            "solver_policy": (
                "multinomial_propensity_weighted_stratified_aalen_johansen_v1"
                if estimator_id == "multinomial_propensity_weighted_stratified_aalen_johansen"
                else "entropy_balanced_standardized_aalen_johansen_v1"
            ),
        },
    }


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
            "allowed_action_ids": ["declare_success", "declare_failure"],
            "stop_action_ids": ["declare_failure"],
            "advance_action_ids": ["declare_success"],
        },
    ]


def _phase_method(phase_id: str) -> dict[str, object]:
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


def _phase_design_rule(phase_id: str) -> dict[str, object]:
    phase1 = phase_id == "phase1"
    candidates = {"control": 1.0, "drug_a": 1.0, "drug_b": 1.0}
    rule: dict[str, object] = {
        "design_cell_id": f"trialdev.{phase_id}.fixed_final_operating_characteristics.v1",
        "phase_id": phase_id,
        "calculator_id": "prospective_fixed_final_operating_characteristics_v1",
        "serious_ae_unacceptable_excess_risk": 0.10,
        "planning_safety_control_risk": 0.10,
        "planning_safety_absolute_treatment_risk": 0.40,
        "planning_safety_excess_risk": 0.30,
        "planning_safety_excess_treatment_risk": 0.40,
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
        "serious_ae_unacceptable_absolute_risk": 0.20,
        "planning_information_estimator_id": ("one_minus_multinomial_propensity_weighted_aalen_johansen_ltfu_cif"),
        "planning_information_fraction_by_drug_id": candidates,
        "planning_information_support_count_by_drug_id": {candidate: 1000 for candidate in candidates},
        "planning_information_weighted_effective_sample_size_by_drug_id": {
            candidate: 900.0 for candidate in candidates
        },
        "rationale": "Test prospective design.",
    }
    if not phase1:
        rule.update(
            {
                "primary_endpoint_id": "E1" if phase_id == "phase2" else "HARD_ENDPOINT",
                "planning_alternative_benefit": 0.40,
                "target_power": 0.51,
                "planning_control_risk": 0.50,
                "planning_treatment_risk": 0.10,
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


def _write_minimal_scenario(root: Path) -> Path:
    scenario = root / "scenario_s01"
    grader = scenario / "grader"
    public = scenario / "public"
    hidden = scenario / "hidden"
    public.mkdir(parents=True, exist_ok=True)
    hidden.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "USUBJID": [f"S{index:04d}" for index in range(300)],
            "ENROLLMENT_DAY": [index % 42 + 1 for index in range(300)],
            "SITE_ID": [f"SITE{index % 8:02d}" for index in range(300)],
            "REGION": [f"R{index % 2}" for index in range(300)],
        }
    ).to_parquet(public / "observational_extract.parquet", index=False)
    phase_module = PhaseModuleSpecV1(
        phase_id="phase2",
        allowed_endpoint_ids=("E1",),
        allowed_follow_up_days=(90, 120),
        allowed_enrollment_window_days=(42,),
        allowed_site_count_budgets=(8,),
        allowed_allocation_ratios=("1:1",),
        max_sample_size=200,
        max_analysis_covariates=8,
        max_subgroup_splits=2,
        allowed_treatment_discontinuation_strategies=("treatment_policy",),
        allowed_interim_policies=("fixed_final",),
        allowed_site_strategies=("high_enrolling",),
        allowed_selection_objectives=("benefit_risk", "cost_effective_best"),
    )
    write_json(
        public / "eval_contract.json",
        TrialDevelopmentEvalContractV1(
            scenario_id="s01",
            phase_modules=(phase_module,),
        ).model_dump(mode="json", exclude_none=True),
    )
    write_json(
        public / "phase_module_catalog.json",
        {"phase_modules": [phase_module.model_dump(mode="json", exclude_none=True)]},
    )
    write_json(
        public / "program_loop_manifest.json",
        TrialDevelopmentProgramLoopManifestV1(
            scenario_id="s01",
            program_archetype="asset_development",
            decision_charter_checksum="0" * 64,
            phase_order=("observational_review", "phase1", "phase2", "phase3"),
            conditionally_materializable_phase_ids=("phase2",),
            phase_policy_modes={"phase1": "required", "phase2": "required", "phase3": "not_available"},
            phase1_carryover_consequential=False,
            terminal_statuses=("stopped", "completed"),
            public_state_summary_fields=(
                "scenario_id",
                "current_phase_id",
                "eligible_candidate_drug_ids",
                "completed_phase_ids",
            ),
        ).model_dump(mode="json", exclude_none=True),
    )
    write_json(
        public / "candidate_drug_catalog.json",
        {
            "candidate_drugs": [
                {"candidate_drug_id": "control", "role": "control"},
                {"candidate_drug_id": "drug_a", "role": "investigational"},
                {"candidate_drug_id": "drug_b", "role": "investigational"},
            ]
        },
    )
    charter = _checked_payload(
        "decision_charter.json",
        {
            "schema_id": "trialdev_decision_charter_v1",
            "scenario_id": "s01",
            "confidence_level": 0.95,
            "efficacy_rules": [{"phase_id": "observational_review", "minimum_benefit": 0.05}],
        },
    )
    write_json(public / "decision_charter.json", charter)
    observational_analysis = {
        "schema_id": "trialdev_public_observational_analysis_spec_v1",
        "method_route_id": ("trialdev.observational.multinomial_propensity_weighted_stratified_aalen_johansen.v1"),
        "phase_id": "observational_review",
        "calculator_id": "public_observational_ipw_utility_v1",
        "primary_estimator_id": "multinomial_propensity_weighted_stratified_aalen_johansen",
        "adjustment_covariates": ["AGE", "BASELINE_SEVERITY"],
        "exact_stratification_covariates": [],
        "quantile_stratification_bins": {},
        "analysis_population": "complete_on_declared_adjustment_covariates",
        "categorical_encoding": "reference_level_one_hot",
        "propensity_solver_id": "deterministic_multinomial_logit_lbfgs",
        "propensity_max_iterations": 1000,
        "propensity_tolerance": 1e-8,
        "propensity_l2_penalty": 0.0,
        "sensitivity_estimator_ids": ["raw_observed"],
        "uncertainty_estimator_id": ("refitted_nuisance_participant_nonparametric_bootstrap"),
        "uncertainty_kind": "two_sided_confidence_interval",
        "confidence_level": 0.95,
        "effect_scale_id": "dimensionless_declared_net_benefit",
        "horizon_source": "objective.efficacy_endpoints[].horizon_days",
        "bootstrap_replicates": 500,
        "bootstrap_seed": 286331153,
        "bootstrap_rng_id": "numpy_default_rng_pcg64",
        "bootstrap_standard_error_ddof": 1,
        "confidence_interval_id": "normal_critical_value_times_bootstrap_standard_error",
        "identification_assumptions": ["consistency"],
        "rationale": "Test observational method.",
    }
    entropy_balanced_analysis = {
        "schema_id": "trialdev_public_observational_analysis_spec_v1",
        "method_route_id": "trialdev.observational.entropy_balanced_standardized_aalen_johansen.v1",
        "phase_id": "observational_review",
        "calculator_id": "public_observational_entropy_balance_utility_v1",
        "primary_estimator_id": "entropy_balanced_standardized_aalen_johansen",
        "adjustment_covariates": ["AGE", "BASELINE_SEVERITY"],
        "analysis_population": "complete_on_declared_adjustment_covariates",
        "categorical_encoding": "reference_level_one_hot",
        "calibration_solver_id": "entropy_balancing_dual_lbfgs",
        "calibration_max_iterations": 1000,
        "calibration_tolerance": 1e-10,
        "maximum_mean_balance_error": 1e-4,
        "sensitivity_estimator_ids": ["raw_observed"],
        "uncertainty_estimator_id": "refitted_nuisance_participant_nonparametric_bootstrap",
        "uncertainty_kind": "two_sided_confidence_interval",
        "confidence_level": 0.95,
        "effect_scale_id": "dimensionless_declared_net_benefit",
        "horizon_source": "objective.efficacy_endpoints[].horizon_days",
        "bootstrap_replicates": 500,
        "bootstrap_seed": 286331153,
        "bootstrap_rng_id": "numpy_default_rng_pcg64",
        "bootstrap_standard_error_ddof": 1,
        "confidence_interval_id": "normal_critical_value_times_bootstrap_standard_error",
        "identification_assumptions": [
            "consistency",
            "measured_conditional_exchangeability",
            "positivity",
            "conditional_independent_censoring",
        ],
        "rationale": "Test entropy-balanced observational method.",
    }

    def objective_record(objective_id: str) -> dict[str, object]:
        components: list[dict[str, object]] = [
            {
                "component_id": "efficacy",
                "source": "efficacy_gain",
                "weight": 1.0,
                "direction": "benefit",
            },
            {
                "component_id": "safety",
                "source": "serious_safety",
                "weight": 1.0,
                "direction": "penalty",
            },
        ]
        candidate_costs: dict[str, float] = {}
        if objective_id == "cost_effective_best":
            components.append(
                {
                    "component_id": "cost",
                    "source": "candidate_cost",
                    "weight": 1.0,
                    "direction": "penalty",
                }
            )
            candidate_costs = {"drug_a": 0.1, "drug_b": 0.2}
        payload: dict[str, object] = {
            "schema_id": "trialdev_public_objective_spec_v1",
            "objective_id": objective_id,
            "efficacy_endpoints": [
                {
                    "endpoint_id": "TEST",
                    "time_column": "EFF_TEST_T",
                    "event_column": "EFF_TEST_E",
                    "competing_time_column": "EFF_TERMINAL_T",
                    "competing_event_column": "EFF_TERMINAL_E",
                    "horizon_days": 90,
                    "estimator_id": "standardized_aalen_johansen_cumulative_incidence",
                    "effect_scale_id": "risk_difference_control_minus_candidate",
                    "effect_orientation_id": "positive_values_favour_candidate",
                }
            ],
            "utility_event_definitions": [
                {
                    "component_source": "serious_safety",
                    "endpoint_id": "ANY_SERIOUS_ADVERSE_EVENT",
                    "time_column": "AE_TEST_EVENT_T",
                    "event_column": "AE_TEST_EVENT_E",
                    "competing_time_column": "EFF_TERMINAL_T",
                    "competing_event_column": "EFF_TERMINAL_E",
                    "horizon_days": 90,
                    "estimator_id": "standardized_aalen_johansen_cumulative_incidence",
                    "effect_scale_id": "risk_difference_candidate_minus_control",
                    "effect_orientation_id": "positive_values_favour_control",
                }
            ],
            "utility_components": components,
            "candidate_costs": candidate_costs,
            "indifference_margin": 0.05,
            "sensitivity_indifference_margins": [0.025, 0.05, 0.1],
            "penalty_weight_sensitivity_multipliers": [0.5, 1.0, 2.0],
            "utility_unit": "dimensionless_declared_net_benefit",
            "policy_basis": "scenario_declared_target_product_profile",
            "decision_direction": "maximize",
            "public_evidence_basis": ["public/candidate_drug_catalog.json"],
            "rationale": "Fixture objective defined on participant-visible efficacy and safety evidence.",
        }
        payload["checksum"] = compute_sha256_hex(payload)
        return payload

    objective_payload = _checked_payload(
        "objective_charter.json",
        {
            "schema_id": "trialdev_public_objective_charter_v1",
            "version": "v1",
            "scenario_id": "s01",
            "confidence_level": 0.95,
            "numeric_reporting_decimal_places": 3,
            "decision_charter_checksum": charter["checksum"],
            "objectives": [
                objective_record("benefit_risk"),
                objective_record("cost_effective_best"),
            ],
        },
    )
    write_json(public / "objective_charter.json", objective_payload)
    write_json(
        public / "observational_method_catalog.json",
        _checked_payload(
            "observational_method_catalog.json",
            {
                "schema_id": "trialdev_public_observational_method_catalog_v1",
                "version": "v1",
                "scenario_id": "s01",
                "assignment_prognostic_factors": [
                    {
                        "factor_id": "clinician_assessed_prognosis",
                        "used_in_treatment_assignment": True,
                        "prognostic_for_primary_endpoint": True,
                        "recorded_in_observational_extract": True,
                        "released_column_id": "BASELINE_SEVERITY",
                        "provenance_statement": (
                            "The clinician-assessed prognosis used during treatment assignment is represented by "
                            "BASELINE_SEVERITY in the released observational extract."
                        ),
                    }
                ],
                "confidence_level": 0.95,
                "decision_charter_checksum": charter["checksum"],
                "methods": [observational_analysis, entropy_balanced_analysis],
            },
        ),
    )
    write_json(
        public / "phase_action_policy.json",
        {
            "action_specs": _action_specs(),
        },
    )
    write_json(
        public / "phase_decision_evidence_policy.json",
        {
            "schema_id": "trialdev_phase_decision_evidence_policy_v1",
            "confidence_level": 0.95,
            "phase_rules": [
                {"phase_id": "phase1", "minimum_benefit": None},
                {
                    "phase_id": "phase2",
                    "efficacy_endpoint_column": "EVENT",
                    "time_column": "TIME",
                    "evaluation_horizon_days": 90,
                    "minimum_benefit": 0.02,
                    "sensitivity_minimum_benefits": [0.01, 0.02, 0.03],
                },
                {
                    "phase_id": "phase3",
                    "efficacy_endpoint_column": "EVENT",
                    "time_column": "TIME",
                    "evaluation_horizon_days": 365,
                    "minimum_benefit": 0.015,
                    "sensitivity_minimum_benefits": [0.01, 0.015, 0.02],
                },
            ],
        },
    )
    _write_checked(
        public / "phase_design_policy.json",
        "phase_design_policy.json",
        {
            "schema_id": "trialdev_phase_design_policy_v1",
            "version": "v1",
            "scenario_id": "s01",
            "decision_charter_checksum": charter["checksum"],
            "confidence_level": 0.95,
            "efficacy_test": "two_sided_normal_approximation_risk_difference",
            "safety_assurance": ("minimum_power_across_absolute_and_excess_serious_ae_hard_gates"),
            "source_artifact_checksums": {"public/test.json": "0" * 64},
            "phase_rules": [_phase_design_rule(phase) for phase in ("phase1", "phase2", "phase3")],
        },
    )
    _write_checked(
        public / "phase_analysis_method_catalog.json",
        "phase_analysis_method_catalog.json",
        {
            "schema_id": "trialdev_phase_analysis_method_catalog_v1",
            "version": "v1",
            "scenario_id": "s01",
            "confidence_level": 0.95,
            "methods": [_phase_method(phase) for phase in ("phase1", "phase2", "phase3")],
        },
    )
    write_json(
        public / "safety_decision_policy.json",
        {
            "schema_id": "trialdev_safety_decision_policy_v1",
            "scenario_id": "s01",
            "serious_event_definitions": [
                {
                    "endpoint_id": "test",
                    "event_column": "AE_TEST_EVENT_E",
                    "time_column": "AE_TEST_EVENT_T",
                    "seriousness_column": "AE_TEST_SERIOUS",
                    "severity_column": "AE_TEST_SEVERITY",
                }
            ],
            "checksum": "0" * 64,
            "thresholds": [
                {
                    "phase_id": phase_id,
                    "component_id": component_id,
                    "role": role,
                    "evaluation_horizon_days": {
                        "phase1": 28,
                        "phase2": 90,
                        "phase3": 365,
                    }[phase_id],
                    "max_absolute_rate": max_absolute_rate,
                    "max_excess_vs_control": 0.035,
                    "sensitivity_max_absolute_rates": {
                        "strict": max_absolute_rate - borderline_margin,
                        "primary": max_absolute_rate,
                        "permissive": max_absolute_rate + borderline_margin,
                    },
                    "sensitivity_max_excess_vs_control": {
                        "strict": 0.035 - borderline_margin,
                        "primary": 0.035,
                        "permissive": 0.035 + borderline_margin,
                    },
                }
                for phase_id in ("phase1", "phase2", "phase3")
                for component_id, role, max_absolute_rate, borderline_margin in (
                    ("serious_ae", "hard_gate", 0.12, 0.02),
                    ("discontinuation", "diagnostic_only", 0.18, 0.03),
                )
            ],
            "decision_rules": [
                {
                    "phase_id": "phase2",
                    "objective_id": objective,
                    "state": state,
                    "action_credits": [
                        {
                            "action_id": "advance_to_confirmation",
                            "credit": "full_credit",
                            "score": 1.0,
                        }
                    ],
                }
                for objective in ("benefit_risk", "cost_effective_best")
                for state in ("acceptable", "borderline", "unacceptable", "indeterminate")
            ],
        },
    )

    trial_output = scenario / "trial_output"
    trial_output.mkdir()
    arms = ["CONTROL"] * 100 + ["TREATMENT"] * 100
    events = [1] * 20 + [0] * 80 + [1] * 10 + [0] * 90
    pd.DataFrame(
        {
            "USUBJID": [f"P{index:04d}" for index in range(200)],
            "ARM": arms,
            "EVENT": events,
            "COMPETING_EVENT": [0] * 200,
            "TREATMENT_DISCONTINUATION_STRATEGY": ["treatment_policy"] * 200,
            "TIME": [14.0 if event else 90.0 for event in events],
            "TERMINAL_EVENT": events,
            "TERMINAL_TIME": [14.0 if event else 90.0 for event in events],
        }
    ).to_parquet(trial_output / "endpoints.parquet", index=False)
    pd.DataFrame(
        {
            "USUBJID": [f"P{index:04d}" for index in range(200)],
            "ARM": arms,
            "AE_TEST_SERIOUS": [0] * 200,
            "AE_TEST_EVENT_E": [0] * 200,
            "AE_TEST_EVENT_T": [90.0] * 200,
            "DISCONTINUATION_E": [0] * 200,
            "DISCONTINUATION_T": [90.0] * 200,
            "LTFU_E": [0] * 200,
            "LTFU_T": [90.0] * 200,
            "TERMINAL_EVENT": [0] * 200,
            "TERMINAL_TIME": [90.0] * 200,
        }
    ).to_parquet(trial_output / "safety.parquet", index=False)
    write_json(
        trial_output / "arm_mapping.json",
        {
            "control_arm_id": "CONTROL",
            "candidate_arm_ids": ["TREATMENT"],
            "drug_id_by_arm": {"CONTROL": "control", "TREATMENT": "drug_a"},
        },
    )
    write_json(
        trial_output / "execution_summary.json",
        {
            "payload": {
                "loss_to_follow_up_assignment": "arm_conditional_random_permutation_v1",
            }
        },
    )
    write_json(
        trial_output / "request.json",
        {
            "version": "v1",
            "scenario_id": "s01",
            "phase_id": "phase2",
            "candidate_drug_ids": ["drug_a"],
            "target_sample_size": 200,
            "follow_up_days": 90,
            "endpoint_id": "E1",
            "enrollment_window_days": 42,
            "site_count_budget": 8,
            "allocation_ratio": "1:1",
            "design_cell_id": "trialdev.phase2.fixed_final_operating_characteristics.v1",
            "treatment_discontinuation_strategy": "treatment_policy",
            "interim_policy": "fixed_final",
            "site_strategy": "high_enrolling",
            "selection_objective": "benefit_risk",
        },
    )

    _write_checked(
        grader / "grading_procedure.json",
        "grading_procedure.json",
        {
            "version": "v1",
            "scenario_id": "s01",
            "supported_lanes": [
                "trial_design",
                "trial_evaluation",
                "program_decision",
                "drug_ranking",
            ],
            "supported_objectives": ["benefit_risk", "cost_effective_best"],
        },
    )
    _write_checked(
        grader / "submission_schema.json",
        "submission_schema.json",
        {
            "version": "v1",
            "scenario_id": "s01",
            "required_sections": ["request", "analysis_report", "program_decision"],
        },
    )
    ranking_records = []
    for objective, values in {
        "benefit_risk": {"drug_a": 0.8, "drug_b": 0.2},
        "cost_effective_best": {"drug_a": 0.3, "drug_b": 0.9},
    }.items():
        for drug, value in values.items():
            ranking_records.append(
                {
                    "phase_id": "phase2",
                    "objective_id": objective,
                    "metric": "objective_score",
                    "candidate_drug_ids": [drug],
                    "value": value,
                }
            )
            ranking_records.append(
                {
                    "phase_id": "observational_review",
                    "objective_id": objective,
                    "metric": "objective_score",
                    "candidate_drug_ids": [drug],
                    "value": value,
                }
            )
    _write_checked(
        grader / "drug_ranking_reference_manifest.json",
        "drug_ranking_reference_manifest.json",
        {"records": ranking_records},
    )
    method_results = [
        _observational_method_result(
            method_route_id=observational_analysis["method_route_id"],
            estimator_id=observational_analysis["primary_estimator_id"],
        ),
        _observational_method_result(
            method_route_id=entropy_balanced_analysis["method_route_id"],
            estimator_id=entropy_balanced_analysis["primary_estimator_id"],
            utility_offset=0.04,
        ),
    ]
    method_union_objective_sensitivity = method_results[0]["objective_policies"]
    method_union_action_sensitivity = method_results[0]["observational_action_policies"]
    _write_checked(
        grader / "public_recoverability_report.json",
        "public_recoverability_report.json",
        {
            "schema_id": "trialdev_public_recoverability_report_v1",
            "version": "v1",
            "scenario_id": "s01",
            "solver_version": "trialdev_public_multi_method_recoverability_v1",
            "method_results": method_results,
            "public_input_checksums": [
                {
                    "path": "public/candidate_drug_catalog.json",
                    "sha256": sha256_file_hex(public / "candidate_drug_catalog.json"),
                },
                {
                    "path": "public/observational_method_catalog.json",
                    "sha256": sha256_file_hex(public / "observational_method_catalog.json"),
                },
                {
                    "path": "public/observational_extract.parquet",
                    "sha256": sha256_file_hex(public / "observational_extract.parquet"),
                },
            ],
            "method_union_objective_sensitivity": method_union_objective_sensitivity,
            "method_union_action_sensitivity": method_union_action_sensitivity,
        },
    )

    write_json(
        grader / "safety_alignment_report.json",
        {
            "schema_id": "trialdev_safety_alignment_report_v1",
            "version": "v1",
            "scenario_id": "s01",
            "passed": True,
            "issue_count": 0,
            "issues": [],
            "rows": [],
            "checksum": "0" * 64,
        },
    )

    evaluation_targets = []
    lanes_by_objective = {
        "benefit_risk": ("phase_design", "phase_analysis", "decision_action", "route_timing"),
        "cost_effective_best": ("phase_design", "phase_analysis", "decision_action", "route_timing"),
    }
    for objective, lanes in lanes_by_objective.items():
        observational_target = "drug_a" if objective == "benefit_risk" else "drug_b"
        for lane_id in ("asset_nomination", "phase_analysis"):
            evaluation_targets.append(
                {
                    "schema_id": "trialdev_evaluation_target_register_record_v1",
                    "scenario_id": "s01",
                    "phase_id": "observational_review",
                    "program_objective_id": objective,
                    "phase_scoring_objective_id": objective,
                    "lane_id": lane_id,
                    "scoring_policy_id": "primary",
                    "public_evidence_basis": ["public/candidate_drug_catalog.json"],
                    "evaluator_evidence_basis": ["grader/public_recoverability_report.json"],
                    "reference_target_ids": [
                        (
                            observational_target
                            if lane_id == "asset_nomination"
                            else (
                                "trialdev.observational.multinomial_propensity_weighted_stratified_aalen_johansen.v1"
                            )
                        )
                    ],
                    "credit_eligible_target_ids": (
                        []
                        if lane_id == "asset_nomination"
                        else ["trialdev.observational.entropy_balanced_standardized_aalen_johansen.v1"]
                    ),
                    "recoverability_policy_id": "unique_best",
                    "value_payload": {},
                    "checksum": "0" * 64,
                }
            )
        for lane_id in lanes:
            evaluation_targets.append(
                {
                    "schema_id": "trialdev_evaluation_target_register_record_v1",
                    "scenario_id": "s01",
                    "phase_id": "phase2",
                    "program_objective_id": objective,
                    "phase_scoring_objective_id": objective,
                    "lane_id": lane_id,
                    "scoring_policy_id": "primary",
                    "public_evidence_basis": ["public/phase_action_policy.json"],
                    "evaluator_evidence_basis": ["grader/evaluation_target_register.jsonl"],
                    "reference_target_ids": [
                        (
                            "trialdev.phase2.fixed_final_operating_characteristics.v1"
                            if lane_id == "phase_design"
                            else (
                                "trialdev.phase2.aalen_johansen_efficacy_safety.v1"
                                if lane_id == "phase_analysis"
                                else "advance_to_confirmation"
                            )
                        )
                    ],
                    "credit_eligible_target_ids": [],
                    "recoverability_policy_id": "no_recoverability_relaxation",
                    "value_payload": {},
                    "checksum": "0" * 64,
                }
            )
    (grader / "evaluation_target_register.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in evaluation_targets) + "\n",
        encoding="utf-8",
    )
    write_json(
        grader / "recoverability_manifest.json",
        {
            "schema_id": "trialdev_recoverability_manifest_v1",
            "version": "v1",
            "scenario_id": "s01",
            "records": [
                {
                    "schema_id": "trialdev_recoverability_policy_record_v1",
                    "phase_id": "observational_review",
                    "objective_id": "benefit_risk",
                    "decision_context": "asset_nomination",
                    "recoverability_policy_id": "acceptable_candidate_set",
                    "acceptable_candidate_set": ["drug_b"],
                    "acceptable_action_set": [],
                    "candidate_records": [],
                    "near_tie_threshold": 0.05,
                    "basis": "test",
                }
            ],
        },
    )
    build_phase_design_frontiers_v1(scenario_root=scenario)
    return scenario


def test_public_validation_rejects_method_catalog_confidence_drift(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    public = scenario / "public"
    catalog = json.loads((public / "phase_analysis_method_catalog.json").read_text(encoding="utf-8"))
    catalog["confidence_level"] = 0.9
    for method in catalog["methods"]:
        method["confidence_level"] = 0.9
    catalog.pop("checksum")
    catalog = _checked_payload("phase_analysis_method_catalog.json", catalog)
    write_json(public / "phase_analysis_method_catalog.json", catalog)

    with pytest.raises(
        ValueError,
        match="must match objective_charter, observational_method_catalog, and decision_charter",
    ):
        _validate_phase_analysis_method_catalog(public_dir=public, scenario_id="s01")


def test_program_loop_manifest_requires_complete_execution_methodology() -> None:
    with pytest.raises(ValueError, match="phase_order|Field required"):
        TrialDevelopmentProgramLoopManifestV1.model_validate(
            {
                "scenario_id": "s01",
                "program_archetype": "asset_development",
                "decision_charter_checksum": "0" * 64,
                "phase_policy_modes": {"phase1": "required", "phase2": "required", "phase3": "optional"},
            }
        )


def test_phase_decision_evidence_must_belong_to_selected_candidate() -> None:
    root = Path(__file__).parents[2]
    analysis_payload = json.loads(
        (root / "examples" / "submissions" / "trialdev_phase_analysis.json").read_text(encoding="utf-8")
    )
    analysis_payload.pop("analysis_rationale")
    analysis = build_phase_analysis_v1(
        analysis_payload,
        scenario_id="example_scenario",
        phase_id="phase2",
        request_checksum="a" * 64,
        trial_output_checksum="b" * 64,
        effect_source_artifact_checksums={"trial_output/endpoints.parquet": "c" * 64},
        safety_source_artifact_checksums={"trial_output/safety.parquet": "d" * 64},
    )
    decision_payload = json.loads(
        (root / "examples" / "submissions" / "trialdev_phase_decision.json").read_text(encoding="utf-8")
    )
    decision_payload.update(
        {
            "decision_action": "stop_development",
            "candidate_drug_id": "candidate_b",
        }
    )
    decision = build_phase_decision_v1(
        decision_payload,
        scenario_id="example_scenario",
        phase_id="phase2",
        request_checksum="a" * 64,
        analysis_checksum="d" * 64,
    )

    with pytest.raises(ValueError, match="analysis-selected candidate"):
        _validate_decision_evidence_links(analysis=analysis, decision=decision)
    without_selected_winner = analysis.model_copy(update={"selected_winner_drug_id": None})
    with pytest.raises(ValueError, match="primary-effect evidence"):
        _validate_decision_evidence_links(analysis=without_selected_winner, decision=decision)


def _write_submission(path: Path, *, objective_id: str, selected: str | None = "drug_a") -> None:
    scenario = path.parent / "scenario_s01"
    trial_output = scenario / "trial_output"
    method_route_id = "trialdev.phase2.aalen_johansen_efficacy_safety.v1"
    effect_checksums = {
        f"trial_output/{name}": sha256_file_hex(trial_output / name)
        for name in ("arm_mapping.json", "endpoints.parquet", "request.json")
    }
    safety_checksums = {
        f"public/{name}": sha256_file_hex(scenario / "public" / name)
        for name in (
            "phase_action_policy.json",
            "phase_decision_evidence_policy.json",
            "safety_decision_policy.json",
        )
    }
    safety_checksums.update(
        {
            f"trial_output/{name}": sha256_file_hex(trial_output / name)
            for name in (
                "arm_mapping.json",
                "endpoints.parquet",
                "execution_summary.json",
                "request.json",
                "safety.parquet",
            )
        }
    )
    analysis: dict[str, Any] = {
        "evidence_summary": "analysis complete",
        "ranked_drug_ids": ["drug_a", "drug_b"],
        "primary_effect": {
            "evidence_id": "effect_primary",
            "method_route_id": method_route_id,
            "candidate_drug_id": selected or "drug_a",
            "endpoint_id": "E1",
            "estimand_id": "treatment_policy:cumulative_incidence_at_horizon",
            "estimator_id": "observed:aalen_johansen_cif_tau",
            "effect_scale_id": "risk_difference_control_minus_treatment",
            "orientation_id": "positive_values_favour_treatment",
            "estimate": 0.10,
            "lower": 0.002,
            "upper": 0.198,
            "confidence_level": 0.95,
            "horizon_days": 90,
            "analysis_population": "all_randomized_participants",
            "source_artifact_checksums": effect_checksums,
        },
        "safety_estimate": {
            "evidence_id": "safety_primary",
            "method_route_id": method_route_id,
            "candidate_drug_id": selected or "drug_a",
            "estimator_id": "observed:aalen_johansen_cif_tau",
            "estimand_ids": [
                "serious_ae:cumulative_incidence_at_horizon",
                "discontinuation:cumulative_incidence_at_horizon",
            ],
            "absolute_risk_scale_id": "absolute_risk",
            "excess_risk_scale_id": "risk_difference_treatment_minus_control",
            "orientation_id": ("absolute_risk_higher_is_worse_and_risk_difference_positive_is_treatment_harm"),
            "horizon_days": 90,
            "analysis_population": "all_randomized_participants",
            "serious_ae_treatment_rate": 0.0,
            "serious_ae_treatment_lower": 0.0,
            "serious_ae_treatment_upper": 0.036,
            "serious_ae_control_rate": 0.0,
            "serious_ae_control_lower": 0.0,
            "serious_ae_control_upper": 0.036,
            "serious_ae_excess": 0.0,
            "serious_ae_excess_lower": -0.036,
            "serious_ae_excess_upper": 0.036,
            "discontinuation_treatment_rate": 0.0,
            "discontinuation_treatment_lower": 0.0,
            "discontinuation_treatment_upper": 0.036,
            "discontinuation_control_rate": 0.0,
            "discontinuation_control_lower": 0.0,
            "discontinuation_control_upper": 0.036,
            "discontinuation_excess": 0.0,
            "discontinuation_excess_lower": -0.036,
            "discontinuation_excess_upper": 0.036,
            "confidence_level": 0.95,
            "source_artifact_checksums": safety_checksums,
        },
    }
    if selected is not None:
        analysis["selected_winner_drug_id"] = selected
    write_json(
        path,
        {
            "version": "v1",
            "scenario_id": "s01",
            "request": {
                "scenario_id": "s01",
                "phase_id": "phase2",
                "candidate_drug_ids": ["drug_a"],
                "target_sample_size": 200,
                "endpoint_id": "E1",
                "follow_up_days": 90,
                "enrollment_window_days": 42,
                "site_count_budget": 8,
                "allocation_ratio": "1:1",
                "design_cell_id": "trialdev.phase2.fixed_final_operating_characteristics.v1",
                "treatment_discontinuation_strategy": "treatment_policy",
                "interim_policy": "fixed_final",
                "site_strategy": "high_enrolling",
                "selection_objective": objective_id,
            },
            "analysis_report": analysis,
            "program_decision": {
                "objective_id": objective_id,
                "decision_action": "advance_to_confirmation",
                "recommended_drug_id": selected,
                "supporting_evidence_ids": ["effect_primary", "safety_primary"],
            },
        },
    )


def _write_observational_submission(
    path: Path,
    *,
    scenario: Path,
    objective_id: str = "benefit_risk",
    selected: str | None = "drug_a",
    decision_action: str = "nominate_for_early_study",
    method_index: int = 0,
    include_method_route_id: bool = True,
) -> None:
    report = json.loads((scenario / "grader" / "public_recoverability_report.json").read_text(encoding="utf-8"))
    specification = json.loads((scenario / "public" / "observational_method_catalog.json").read_text(encoding="utf-8"))
    objective_charter = json.loads((scenario / "public" / "objective_charter.json").read_text(encoding="utf-8"))
    observational_method = specification["methods"][method_index]
    method_result = next(
        result
        for result in report["method_results"]
        if result["method_route_id"] == observational_method["method_route_id"]
    )
    scores = [row for row in method_result["candidate_scores"] if row["objective_id"] == objective_id]
    ranked = [
        row["candidate_drug_id"]
        for row in sorted(scores, key=lambda row: (-float(row["adjusted_utility"]), row["candidate_drug_id"]))
    ]
    objective = next(row for row in objective_charter["objectives"] if row["objective_id"] == objective_id)
    evidence_paths = set(objective["public_evidence_basis"])
    source_checksums = {
        row["path"]: row["sha256"] for row in report["public_input_checksums"] if row["path"] in evidence_paths
    }
    estimates = []
    for row in scores:
        estimate = {
            "evidence_id": f"utility_{row['candidate_drug_id']}",
            "candidate_drug_id": row["candidate_drug_id"],
            "objective_id": objective_id,
            "estimator_id": observational_method["primary_estimator_id"],
            "estimate": row["adjusted_utility"],
            "lower": row["ci_low"],
            "upper": row["ci_high"],
            "confidence_level": objective_charter["confidence_level"],
            "analysis_covariate_ids": observational_method["adjustment_covariates"],
            "source_artifact_checksums": source_checksums,
        }
        if include_method_route_id:
            estimate["method_route_id"] = observational_method["method_route_id"]
        estimates.append(estimate)
    write_json(
        path,
        {
            "version": "v1",
            "scenario_id": "s01",
            "request": {
                "version": "v1",
                "scenario_id": "s01",
                "phase_id": "observational_review",
                "candidate_drug_ids": ["drug_a", "drug_b"],
                "selection_objective": objective_id,
            },
            "analysis_report": {
                "response_branch": "estimable",
                "primary_resolution_evidence_class": "empirical_diagnosis",
                "selected_winner_drug_id": selected,
                "ranked_drug_ids": ranked,
                "candidate_utility_estimates": estimates,
            },
            "program_decision": {
                "objective_id": objective_id,
                "decision_action": decision_action,
                "recommended_drug_id": selected,
                "supporting_evidence_ids": [f"utility_{selected or ranked[0]}"],
            },
        },
    )


def test_direct_grade_rejects_objective_override(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk")
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["program_decision"]["objective_id"] = "cost_effective_best"
    write_json(submission, payload)

    with pytest.raises(ValueError, match="cannot override|must match"):
        grade_item_v1(scenario_root=scenario, submission_path=submission)


def test_observational_primary_requires_correct_full_numeric_analysis(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario)

    report = grade_item_v1(scenario_root=scenario, submission_path=submission)

    assert report.validity.valid is True
    assert report.evaluation_score == 1.0
    assert report.program_score == 1.0
    assert report.primary_score == 1.0
    assert report.analysis_quality.observational_analysis_valid is True
    assert report.analysis_quality.observational_analysis_score == 1.0
    assert report.analysis_quality.randomized_primary_effect_eligible is False
    assert report.analysis_quality.safety_evidence_eligible is False
    assert report.analysis_quality.phase_evaluation_valid is True
    registered = [
        json.loads(line)
        for line in (scenario / "grader" / "evaluation_target_register.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registered_asset = next(
        row
        for row in registered
        if row["phase_id"] == "observational_review"
        and row["program_objective_id"] == "benefit_risk"
        and row["lane_id"] == "asset_nomination"
    )
    scored_asset = next(row for row in report.lane_scores if row.lane_id == "asset_nomination")
    assert scored_asset.evaluation_target_checksum == registered_asset["checksum"]
    assert scored_asset.scoring_policy_id == "trialdev_method_conditioned_public_action_v1"


def test_observational_reference_submission_uses_one_complete_method_route(
    tmp_path: Path,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = build_observational_reference_submission_v1(
        scenario_root=scenario,
        objective_id="benefit_risk",
    )
    method_ids = {estimate.method_route_id for estimate in submission.candidate_utility_estimates}
    estimator_ids = {estimate.estimator_id for estimate in submission.candidate_utility_estimates}

    assert method_ids == {"trialdev.observational.entropy_balanced_standardized_aalen_johansen.v1"}
    assert estimator_ids == {"entropy_balanced_standardized_aalen_johansen"}
    assert submission.decision_action == "nominate_for_early_study"
    assert submission.candidate_drug_id == "drug_a"


def test_provenance_qualified_non_nomination_is_graded_without_fabricated_ranking(
    tmp_path: Path,
) -> None:
    measured = _write_minimal_scenario(tmp_path / "measured")
    residual = _write_minimal_scenario(tmp_path / "residual")
    catalog_path = residual / "public" / "observational_method_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    factor = catalog["assignment_prognostic_factors"][0]
    factor.update(
        {
            "recorded_in_observational_extract": False,
            "provenance_statement": (
                "The clinician-assessed prognosis informed treatment assignment and predicted the primary "
                "endpoint, but it was not retained in the released observational extract."
            ),
        },
    )
    factor.pop("released_column_id")
    catalog.pop("checksum")
    catalog["checksum"] = compute_sha256_hex(catalog)
    write_json(catalog_path, catalog)
    recoverability_path = residual / "grader" / "public_recoverability_report.json"
    recoverability = json.loads(recoverability_path.read_text(encoding="utf-8"))
    for checksum in recoverability["public_input_checksums"]:
        if checksum["path"] == "public/observational_method_catalog.json":
            checksum["sha256"] = sha256_file_hex(catalog_path)
    recoverability.pop("checksum")
    _write_checked(
        recoverability_path,
        "public_recoverability_report.json",
        recoverability,
    )
    reference = build_observational_reference_submission_v1(
        scenario_root=residual,
        objective_id="benefit_risk",
    )
    submission_path = tmp_path / "qualified_non_nomination.json"
    write_json(
        submission_path,
        {
            "version": "v1",
            "scenario_id": "s01",
            "request": {
                "version": "v1",
                "scenario_id": "s01",
                "phase_id": "observational_review",
                "candidate_drug_ids": ["drug_a", "drug_b"],
                "selection_objective": "benefit_risk",
            },
            "analysis_report": {
                **reference.model_dump(
                    mode="json",
                    exclude={"supporting_evidence_ids", "candidate_drug_id", "decision_action", "decision_rationale"},
                ),
                "selected_winner_drug_id": None,
            },
            "program_decision": {
                "objective_id": "benefit_risk",
                "decision_action": reference.decision_action,
                "recommended_drug_id": reference.candidate_drug_id,
                "supporting_evidence_ids": list(reference.supporting_evidence_ids),
            },
        },
    )

    accepted = grade_item_v1(scenario_root=residual, submission_path=submission_path)
    rejected = grade_item_v1(scenario_root=measured, submission_path=submission_path)

    assert reference.response_branch == "qualified_non_nomination"
    assert not reference.candidate_utility_estimates
    assert not reference.ranked_drug_ids
    assert accepted.primary_score == 1.0
    assert accepted.program_score == 1.0
    assert accepted.analysis_quality.observational_analysis_valid is True
    assert rejected.primary_score == 0.0
    assert rejected.analysis_quality.observational_analysis_valid is False


def test_empirical_non_estimability_accepts_evidence_linked_non_nomination(
    tmp_path: Path,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    unsupported_point_path = tmp_path / "unsupported_point.json"
    _write_observational_submission(
        unsupported_point_path,
        scenario=scenario,
        method_index=1,
    )
    report_path = scenario / "grader" / "public_recoverability_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    method = sorted(report["method_results"], key=lambda row: row["method_route_id"])[0]
    for score in method["candidate_scores"]:
        for field in (
            "adjusted_utility",
            "utility_se",
            "ci_low",
            "ci_high",
            "efficacy_gain",
            "efficacy_gain_se",
            "efficacy_gain_ci_low",
            "efficacy_gain_ci_high",
            "rank",
            "utility_margin_to_best",
            "max_abs_adjusted_smd_vs_target",
        ):
            score[field] = None
        score.update(
            {
                "point_estimable": False,
                "inference_estimable": False,
                "score_state": "insufficient_recoverability",
            }
        )
    for comparison in method["estimator_comparisons"]:
        comparison.update(
            {
                "status": "not_estimable",
                "failure_reason": "empirical_positivity_violation",
                "reference_top_candidate_id": None,
                "top_candidate_id": None,
                "top_utility": None,
                "utility_margin_to_reference_top": None,
                "agrees_with_reference_top": None,
                "rank_order": [],
                "candidate_utilities": [],
                "policy_signal": "not_estimable",
            }
        )
    for policy in method["objective_policies"]:
        policy.update(
            {
                "policy": "insufficient_recoverability",
                "reference_target_ids": [],
                "acceptable_candidate_set": [],
                "near_tie_threshold": None,
                "indifference_sensitivity_sets": {},
                "preference_sensitivity_sets": {},
                "rationale": "The released observations do not support the declared method.",
            }
        )
    for policy in method["observational_action_policies"]:
        policy.update(
            {
                "reference_target_ids": ["withhold_nomination"],
                "credit_eligible_target_ids": ["withhold_nomination"],
                "definitely_qualified_candidate_ids": [],
                "possibly_qualified_candidate_ids": [],
                "utility_contrast_half_widths": {},
                "pairwise_utility_contrast_half_widths": {},
                "rationale": "No causal candidate ranking is reproducible with the declared method.",
            }
        )
    method["diagnostics"]["max_abs_adjusted_smd_vs_target"] = None
    report.pop("checksum")
    _write_checked(report_path, "public_recoverability_report.json", report)

    reference = build_observational_reference_submission_v1(
        scenario_root=scenario,
        objective_id="benefit_risk",
    )
    submission_path = tmp_path / "empirical_non_nomination.json"
    write_json(
        submission_path,
        {
            "version": "v1",
            "scenario_id": "s01",
            "request": {
                "version": "v1",
                "scenario_id": "s01",
                "phase_id": "observational_review",
                "candidate_drug_ids": ["drug_a", "drug_b"],
                "selection_objective": "benefit_risk",
            },
            "analysis_report": {
                **reference.model_dump(
                    mode="json",
                    exclude={"supporting_evidence_ids", "candidate_drug_id", "decision_action", "decision_rationale"},
                ),
                "selected_winner_drug_id": None,
            },
            "program_decision": {
                "objective_id": "benefit_risk",
                "decision_action": reference.decision_action,
                "recommended_drug_id": reference.candidate_drug_id,
                "supporting_evidence_ids": list(reference.supporting_evidence_ids),
            },
        },
    )

    accepted = grade_item_v1(scenario_root=scenario, submission_path=submission_path)
    unsupported_point = grade_item_v1(
        scenario_root=scenario,
        submission_path=unsupported_point_path,
    )

    assert reference.primary_resolution_evidence_class == "evidence_insufficient"
    assert reference.identification_evidence[0].premise_id == "practical_positivity"
    assert reference.identification_evidence[0].evidence_kind == "empirical_diagnostic"
    assert accepted.primary_score == 1.0
    assert unsupported_point.primary_score == 0.0
    assert unsupported_point.analysis_quality.observational_analysis_valid is False


def test_phase_reference_request_selects_a_public_frontier_point(
    tmp_path: Path,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)

    request = build_phase_reference_request_v1(
        scenario_root=scenario,
        phase_id="phase2",
        candidate_drug_id="drug_a",
        objective_id="benefit_risk",
    )
    witness = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=request,
        phase_id="phase2",
    )
    efficiency = derive_phase_design_efficiency_v1(
        scenario_root=scenario,
        request=request,
        design_witness=witness,
    )

    assert witness.adequate is True
    assert efficiency.on_frontier is True
    assert efficiency.dominated_by_frontier is False
    assert efficiency.operationally_feasible is True


def test_phase_reference_request_rejects_an_unrecruitable_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    artifact = load_phase_design_frontiers_v1(scenario_root=scenario)
    unsupported = artifact.model_copy(
        update={
            "operational_support": tuple(
                record.model_copy(update={"eligible_subject_count": 0}) for record in artifact.operational_support
            )
        }
    )
    monkeypatch.setattr(
        reference_submissions,
        "load_phase_design_frontiers_v1",
        lambda *, scenario_root: unsupported,
    )

    with pytest.raises(ValueError, match="cannot recruit any released statistical frontier point"):
        build_phase_reference_request_v1(
            scenario_root=scenario,
            phase_id="phase2",
            candidate_drug_id="drug_a",
            objective_id="benefit_risk",
        )


def test_observational_grade_file_preserves_analysis_quality_applicability(tmp_path: Path) -> None:
    """Serialized grades retain required nulls for inapplicable components."""

    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    grade_path = tmp_path / "grade.json"
    _write_observational_submission(submission, scenario=scenario)

    grade_item_v1(scenario_root=scenario, submission_path=submission, write_path=grade_path)

    payload = json.loads(grade_path.read_text(encoding="utf-8"))
    quality = TrialDevelopmentAnalysisQualityV1.model_validate(payload["analysis_quality"])
    assert quality.randomized_primary_effect_valid is None
    assert quality.safety_evidence_valid is None


@pytest.mark.parametrize(("method_index", "include_method_route_id"), [(0, True), (1, True), (1, False)])
def test_observational_methods_grade_only_against_their_own_truth(
    tmp_path: Path,
    method_index: int,
    include_method_route_id: bool,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    _write_observational_submission(
        submission,
        scenario=scenario,
        method_index=method_index,
        include_method_route_id=include_method_route_id,
    )

    report = grade_item_v1(scenario_root=scenario, submission_path=submission)

    assert report.primary_score == 1.0
    assert report.analysis_quality.observational_analysis_valid is True


def test_observational_method_cannot_receive_best_of_other_method_truth(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario, method_index=1)
    payload = json.loads(submission.read_text(encoding="utf-8"))
    report = json.loads((scenario / "grader" / "public_recoverability_report.json").read_text(encoding="utf-8"))
    ipw_scores = {
        row["candidate_drug_id"]: row
        for row in report["method_results"][0]["candidate_scores"]
        if row["objective_id"] == "benefit_risk"
    }
    for estimate in payload["analysis_report"]["candidate_utility_estimates"]:
        wrong = ipw_scores[estimate["candidate_drug_id"]]
        estimate["estimate"] = wrong["adjusted_utility"]
        estimate["lower"] = wrong["ci_low"]
        estimate["upper"] = wrong["ci_high"]
    write_json(submission, payload)

    grade = grade_item_v1(scenario_root=scenario, submission_path=submission)

    assert grade.scientific_assessment.exact_reproduction == "failed"
    assert grade.scientific_assessment.scientific_agreement == "passed"
    assert grade.evaluation_score == 1.0
    assert grade.analysis_quality.observational_analysis_valid is True


def test_mixed_observational_methods_are_scored_as_unresolved(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario)
    payload = json.loads(submission.read_text(encoding="utf-8"))
    methods = json.loads((scenario / "public" / "observational_method_catalog.json").read_text(encoding="utf-8"))[
        "methods"
    ]
    mixed = payload["analysis_report"]["candidate_utility_estimates"][1]
    mixed["method_route_id"] = methods[1]["method_route_id"]
    mixed["estimator_id"] = methods[1]["primary_estimator_id"]
    write_json(submission, payload)

    grade = grade_item_v1(scenario_root=scenario, submission_path=submission)

    assert grade.evaluation_score == 0.0
    assert grade.analysis_quality.observational_analysis_valid is False
    assert "observational_method_not_resolved" in grade.feasibility_failures


def test_observational_decline_can_receive_full_credit_without_a_selected_asset(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    recoverability_path = scenario / "grader" / "public_recoverability_report.json"
    recoverability = json.loads(recoverability_path.read_text(encoding="utf-8"))
    for result in recoverability["method_results"]:
        for score in result["candidate_scores"]:
            if score["objective_id"] == "benefit_risk":
                score["efficacy_gain"] = 0.0
                score["efficacy_gain_ci_low"] = 0.0
        action_policy = result["observational_action_policies"][0]
        action_policy["definitely_qualified_candidate_ids"] = []
        action_policy["reference_target_ids"] = ["withhold_nomination"]
        action_policy["credit_eligible_target_ids"] = [
            "withhold_nomination",
            action_policy["credit_eligible_target_ids"][0],
        ]
    sensitivity_action = recoverability["method_union_action_sensitivity"][0]
    sensitivity_action["definitely_qualified_candidate_ids"] = []
    sensitivity_action["reference_target_ids"] = ["withhold_nomination"]
    sensitivity_action["credit_eligible_target_ids"] = [
        "withhold_nomination",
        *sensitivity_action["credit_eligible_target_ids"],
    ]
    _write_checked(recoverability_path, "public_recoverability_report.json", recoverability)
    register_path = scenario / "grader" / "evaluation_target_register.jsonl"
    register = [json.loads(line) for line in register_path.read_text(encoding="utf-8").splitlines()]
    for row in register:
        if (
            row["phase_id"] == "observational_review"
            and row["program_objective_id"] == "benefit_risk"
            and row["lane_id"] == "asset_nomination"
        ):
            row["reference_target_ids"] = ["withhold_nomination"]
            row["credit_eligible_target_ids"] = []
    register_path.write_text("\n".join(json.dumps(row) for row in register) + "\n", encoding="utf-8")
    submission = tmp_path / "observational_decline.json"
    _write_observational_submission(
        submission,
        scenario=scenario,
        selected=None,
        decision_action="withhold_nomination",
    )
    submitted = json.loads(submission.read_text(encoding="utf-8"))
    submitted["program_decision"]["supporting_evidence_ids"] = [
        row["evidence_id"] for row in submitted["analysis_report"]["candidate_utility_estimates"]
    ]
    write_json(submission, submitted)

    report = grade_item_v1(scenario_root=scenario, submission_path=submission)

    assert report.primary_score == 1.0
    assert report.selected_winner_drug_id is None
    assert report.policy_reference_regret is None
    assert report.in_set_regret is None
    assert report.scientific_assessment.action_admissibility == "passed"
    assert report.scientific_assessment.evidential_support == "passed"


def test_observational_decline_requires_evidence_for_every_candidate(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational_decline.json"
    _write_observational_submission(
        submission,
        scenario=scenario,
        selected=None,
        decision_action="withhold_nomination",
    )

    report = grade_item_v1(scenario_root=scenario, submission_path=submission)

    assert report.scientific_assessment.evidential_support == "failed"
    assert "decision_evidence_not_supported" in report.scientific_assessment.failure_reasons


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("estimator_id", "raw_observed", "observational_estimator_mismatch"),
        ("analysis_covariate_ids", ["AGE"], "observational_adjustment_covariates_mismatch"),
        (
            "source_artifact_checksums",
            {"public/candidate_drug_catalog.json": "f" * 64},
            "observational_source_artifact_checksums_mismatch",
        ),
    ],
)
def test_observational_semantic_or_provenance_mismatch_is_lane_local(
    tmp_path: Path,
    field: str,
    value: object,
    reason: str,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario)
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["analysis_report"]["candidate_utility_estimates"][0][field] = value
    write_json(submission, payload)

    report = grade_item_v1(scenario_root=scenario, submission_path=submission)

    assert report.primary_score == 0.0
    assert report.evaluation_score == 0.0
    assert report.validity.valid is True
    assert any(item.startswith(reason) for item in report.feasibility_failures)
    asset_lane = next(row for row in report.lane_scores if row.lane_id == "asset_nomination")
    analysis_lane = next(row for row in report.lane_scores if row.lane_id == "phase_analysis")
    assert asset_lane.score == 1.0
    assert analysis_lane.score == 0.0


def test_observational_covariate_order_is_representation_invariant(
    tmp_path: Path,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario)
    payload = json.loads(submission.read_text(encoding="utf-8"))
    for estimate in payload["analysis_report"]["candidate_utility_estimates"]:
        estimate["analysis_covariate_ids"] = list(reversed(estimate["analysis_covariate_ids"]))
    write_json(submission, payload)

    report = grade_item_v1(scenario_root=scenario, submission_path=submission)

    analysis_lane = next(row for row in report.lane_scores if row.lane_id == "phase_analysis")
    assert report.validity.valid is True
    assert analysis_lane.score == 1.0
    assert report.primary_score == 1.0


def test_observational_correct_asset_without_analysis_is_rejected(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario)
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["analysis_report"]["candidate_utility_estimates"] = []
    write_json(submission, payload)

    with pytest.raises(ValueError, match="candidate utility estimates must cover"):
        grade_item_v1(scenario_root=scenario, submission_path=submission)


def test_unadjusted_observational_analysis_is_descriptive_not_causal_completion(
    tmp_path: Path,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario)
    payload = json.loads(submission.read_text(encoding="utf-8"))
    for estimate in payload["analysis_report"]["candidate_utility_estimates"]:
        estimate["analysis_covariate_ids"] = []
    write_json(submission, payload)

    report = grade_item_v1(scenario_root=scenario, submission_path=submission)

    assert report.scientific_assessment.analysis_classification == "descriptive"
    assert report.scientific_assessment.assumptions == "failed"
    assert report.scientific_assessment.scientific_agreement == "not_assessed"
    assert report.scientific_assessment.decision_complete is False
    assert report.primary_score == 0.0


def test_public_grade_wrapper_preserves_the_canonical_scientific_assessment(
    tmp_path: Path,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario)

    report = grade_item_v1(scenario_root=scenario, submission_path=submission)
    wrapped = wrap_grade_record(grade_report_payload_v1(report=report, report_mode="audit"))

    assert wrapped.scientific_assessment == report.scientific_assessment
    assert wrapped.scientific_assessment.decision_complete is True


def test_material_numeric_error_fails_scientific_agreement_at_its_own_boundary(
    tmp_path: Path,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario)
    payload = json.loads(submission.read_text(encoding="utf-8"))
    for estimate in payload["analysis_report"]["candidate_utility_estimates"]:
        estimate["estimate"] += 0.5
        estimate["lower"] += 0.5
        estimate["upper"] += 0.5
    write_json(submission, payload)

    report = grade_item_v1(scenario_root=scenario, submission_path=submission)

    assert report.scientific_assessment.assumptions == "passed"
    assert report.scientific_assessment.scientific_agreement == "failed"
    assert report.scientific_assessment.exact_reproduction == "failed"
    assert report.scientific_assessment.decision_complete is False
    assert report.evaluation_score == 0.0


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("objective_policies", "near_tie_threshold", "indifference margin drifts"),
        ("observational_action_policies", "minimum_efficacy_gain", "efficacy threshold drifts"),
    ],
)
def test_observational_report_cannot_override_public_decision_policy(
    tmp_path: Path,
    section: str,
    field: str,
    message: str,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    path = scenario / "grader" / "public_recoverability_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["method_results"][0][section][0][field] = 0.04
    _write_checked(path, "public_recoverability_report.json", payload)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario)

    with pytest.raises(ValueError, match=message):
        grade_item_v1(scenario_root=scenario, submission_path=submission)


def test_observational_report_requires_adjusted_balance_for_scoreable_candidates(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    path = scenario / "grader" / "public_recoverability_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["method_results"][0]["candidate_scores"][0]["max_abs_adjusted_smd_vs_target"] = None
    _write_checked(path, "public_recoverability_report.json", payload)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario)

    with pytest.raises(ValueError, match="max_abs_adjusted_smd_vs_target"):
        grade_item_v1(scenario_root=scenario, submission_path=submission)


def test_observational_numeric_error_reduces_analysis_lane_without_invalidating_contract(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "observational.json"
    _write_observational_submission(submission, scenario=scenario)
    payload = json.loads(submission.read_text(encoding="utf-8"))
    estimate = payload["analysis_report"]["candidate_utility_estimates"][0]
    estimate["estimate"] = 1.5
    estimate["lower"] = 1.4
    estimate["upper"] = 1.6
    write_json(submission, payload)

    report = grade_item_v1(scenario_root=scenario, submission_path=submission)

    assert report.validity.valid is True
    assert 0.0 <= report.evaluation_score < 1.0
    assert report.analysis_quality.observational_analysis_valid is True
    assert report.analysis_quality.observational_analysis_score == report.evaluation_score
    assert report.program_score == 1.0
    assert report.primary_score == report.evaluation_score


def test_missing_selected_asset_does_not_use_hidden_best(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected=None)
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["analysis_report"]["ranked_drug_ids"] = []
    payload["program_decision"].pop("recommended_drug_id", None)
    write_json(submission, payload)

    report = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )

    assert report.selected_winner_drug_id == "drug_a"
    assert report.validity.valid is True
    assert report.primary_score > 0.0


def test_phase_candidate_validity_does_not_require_phase_ranking_records(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    baseline = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )
    ranking_path = scenario / "grader" / "drug_ranking_reference_manifest.json"
    ranking_payload = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking_payload["records"] = [row for row in ranking_payload["records"] if str(row.get("phase_id")) != "phase2"]
    _write_checked(ranking_path, "drug_ranking_reference_manifest.json", ranking_payload)
    report = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )

    assert report.validity.valid is True
    assert not any(str(reason).startswith("unknown_candidate:") for reason in report.feasibility_failures)
    assert report.payload["candidate_eligibility"][0]["candidate_drug_id"] == "drug_a"
    assert report.payload["candidate_eligibility"][0]["catalog_member"] is True
    assert report.primary_score == baseline.primary_score
    assert report.evaluation_score == baseline.evaluation_score


def test_malformed_evaluator_reference_value_fails_grading(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    ranking_path = scenario / "grader" / "drug_ranking_reference_manifest.json"
    ranking_payload = json.loads(ranking_path.read_text(encoding="utf-8"))
    objective_record = next(row for row in ranking_payload["records"] if row.get("metric") == "objective_score")
    objective_record["value"] = "not-numeric"
    _write_checked(ranking_path, "drug_ranking_reference_manifest.json", ranking_payload)

    with pytest.raises(ValueError, match="reference record value must be numeric"):
        grade_item_v1(
            scenario_root=scenario,
            submission_path=submission,
            trial_output_root=scenario / "trial_output",
        )


def test_set_identified_action_requires_structured_uncertainty_evidence(tmp_path: Path) -> None:
    """An arbitrary action in a broad evidence set cannot receive full credit."""

    scenario = _write_minimal_scenario(tmp_path)
    endpoints_path = scenario / "trial_output" / "endpoints.parquet"
    endpoints = pd.read_parquet(endpoints_path)
    endpoints.loc[:, "EVENT"] = [1] * 20 + [0] * 80 + [1] * 19 + [0] * 81
    endpoints.to_parquet(endpoints_path, index=False)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["program_decision"]["supporting_evidence_ids"] = ["safety_primary"]
    payload["analysis_report"]["safety_estimate"]["serious_ae_excess_upper"] = 0.03
    write_json(submission, payload)

    report = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )
    decision_lane = next(row for row in report.lane_scores if row.lane_id == "decision_action")

    assert report.payload["decision_action_diagnostics"]["decision_recoverability_class"] == "set_identified"
    assert report.payload["decision_action_diagnostics"]["uncertainty_evidence_present"] is False
    assert decision_lane.score == 0.0


def test_safety_uncertainty_can_be_supported_only_by_excess_interval() -> None:
    payload = {
        "evidence_id": "safety",
        "method_route_id": "trialdev.phase2.aalen_johansen_efficacy_safety.v1",
        "candidate_drug_id": "drug_a",
        "estimator_id": "observed:aalen_johansen_cif_tau",
        "estimand_ids": [
            "serious_ae:cumulative_incidence_at_horizon",
            "discontinuation:cumulative_incidence_at_horizon",
        ],
        "absolute_risk_scale_id": "absolute_risk",
        "excess_risk_scale_id": "risk_difference_treatment_minus_control",
        "orientation_id": ("absolute_risk_higher_is_worse_and_risk_difference_positive_is_treatment_harm"),
        "horizon_days": 90,
        "analysis_population": "all_randomized_participants",
        "serious_ae_treatment_rate": 0.04,
        "serious_ae_treatment_lower": 0.01,
        "serious_ae_treatment_upper": 0.06,
        "serious_ae_control_rate": 0.02,
        "serious_ae_control_lower": 0.0,
        "serious_ae_control_upper": 0.04,
        "serious_ae_excess": 0.02,
        "serious_ae_excess_lower": -0.01,
        "serious_ae_excess_upper": 0.05,
        "discontinuation_treatment_rate": 0.04,
        "discontinuation_treatment_lower": 0.01,
        "discontinuation_treatment_upper": 0.06,
        "discontinuation_control_rate": 0.02,
        "discontinuation_control_lower": 0.0,
        "discontinuation_control_upper": 0.04,
        "discontinuation_excess": 0.02,
        "discontinuation_excess_lower": -0.01,
        "discontinuation_excess_upper": 0.05,
        "confidence_level": 0.95,
        "source_artifact_checksums": {"safety.parquet": "a" * 64},
    }
    estimate = TrialDevelopmentSafetyEstimateV1.model_validate(payload)
    diagnostics = {
        "absolute_limit": 0.12,
        "excess_limit": 0.035,
        "discontinuation": {"role": "diagnostic_only"},
    }

    assert _safety_uncertainty_support_v1(
        safety_estimate=estimate,
        safety_diagnostics=diagnostics,
    )
    estimate_without_crossing = TrialDevelopmentSafetyEstimateV1.model_validate(
        {**payload, "serious_ae_excess_upper": 0.03}
    )
    assert not _safety_uncertainty_support_v1(
        safety_estimate=estimate_without_crossing,
        safety_diagnostics=diagnostics,
    )


def test_self_attested_flags_are_rejected_by_submission_contract(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["analysis_report"]["uncertainty_calibrated"] = True
    write_json(submission, payload)

    with pytest.raises(ValueError, match="uncertainty_calibrated"):
        grade_item_v1(
            scenario_root=scenario,
            submission_path=submission,
            trial_output_root=scenario / "trial_output",
        )


def test_decision_cannot_reference_undeclared_analysis_evidence(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["program_decision"]["supporting_evidence_ids"].append("fabricated_evidence")
    write_json(submission, payload)

    with pytest.raises(ValueError, match="unknown analysis evidence"):
        grade_item_v1(
            scenario_root=scenario,
            submission_path=submission,
            trial_output_root=scenario / "trial_output",
        )


def test_effect_candidate_mismatch_invalidates_submission(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["analysis_report"]["primary_effect"]["candidate_drug_id"] = "drug_b"
    write_json(submission, payload)

    report = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )

    assert report.validity.valid is False
    assert "effect_candidate_mismatch" in report.validity.invalid_reasons
    assert report.primary_score == 0.0


def test_rmst_cannot_support_a_risk_difference_decision_policy(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["analysis_report"]["primary_effect"].update(
        {
            "estimand_id": "treatment_policy:rmst_at_horizon",
            "estimator_id": "observed:km_rmst_tau",
            "effect_scale_id": "rmst_difference_treatment_minus_control",
            "estimate": 1.4,
            "lower": 0.0,
            "upper": 2.8,
        }
    )
    write_json(submission, payload)

    report = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )

    assert report.validity.valid is True
    assert "unaccepted_effect_method_route" in report.feasibility_failures
    assert report.primary_score == 0.0


def test_effect_source_checksum_mismatch_gets_no_analysis_credit(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["analysis_report"]["primary_effect"]["source_artifact_checksums"] = {
        "trial_output/arm_mapping.json": "f" * 64,
        "trial_output/endpoints.parquet": "f" * 64,
        "trial_output/request.json": "f" * 64,
    }
    write_json(submission, payload)

    report = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )

    analysis_lane = next(row for row in report.lane_scores if row.lane_id == "phase_analysis")
    assert analysis_lane.submitted_target_id == ("trialdev.phase2.aalen_johansen_efficacy_safety.v1")
    assert analysis_lane.score == 0.0
    assert report.primary_score == 0.0
    numeric = report.payload["numeric_score_components"]
    assert isinstance(numeric, dict)
    assert float(numeric["score_before_method_and_completeness_gates"]) > 0.0
    assert numeric["score_after_method_and_completeness_gates"] == 0.0
    assert report.analysis_quality.randomized_primary_effect_valid is False
    assert report.analysis_quality.randomized_primary_effect_point_agreement == 0.0
    assert report.analysis_quality.randomized_primary_effect_interval_agreement == 0.0
    assert report.analysis_quality.phase_evaluation_valid is False


def test_actual_method_and_design_cells_require_prospective_numeric_equivalence(
    tmp_path: Path,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["analysis_report"]["primary_effect"].update(
        {
            "estimate": 0.09,
            "lower": -0.01,
            "upper": 0.19,
            "confidence_level": 0.9500000000001,
        }
    )
    payload["analysis_report"]["safety_estimate"]["confidence_level"] = 0.9500000000001
    write_json(submission, payload)

    report = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )

    analysis_lane = next(row for row in report.lane_scores if row.lane_id == "phase_analysis")
    design_lane = next(row for row in report.lane_scores if row.lane_id == "phase_design")
    assert report.validity.valid is True
    assert analysis_lane.submitted_target_id == ("trialdev.phase2.aalen_johansen_efficacy_safety.v1")
    assert design_lane.submitted_target_id == ("trialdev.phase2.fixed_final_operating_characteristics.v1")
    assert analysis_lane.score == 1.0
    assert report.scientific_assessment.scientific_agreement == "passed"
    assert report.scientific_assessment.exact_reproduction == "failed"


def test_unregistered_effect_cell_is_invalid_not_charitably_matched(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["analysis_report"]["primary_effect"].update(
        {
            "estimand_id": "treatment_policy:hazard_ratio",
            "estimator_id": "cox mentioned somewhere in prose",
            "effect_scale_id": "log_hazard_ratio",
        }
    )
    write_json(submission, payload)

    report = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )

    assert report.validity.valid is True
    assert "unaccepted_effect_method_route" in report.feasibility_failures
    assert report.primary_score == 0.0


def test_wrong_analysis_cell_zeros_only_analysis_lane(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["analysis_report"]["primary_effect"]["method_route_id"] = "trialdev.phase2.unregistered.v1"
    payload["analysis_report"]["safety_estimate"]["method_route_id"] = "trialdev.phase2.unregistered.v1"
    write_json(submission, payload)

    report = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )

    analysis_lane = next(row for row in report.lane_scores if row.lane_id == "phase_analysis")
    action_lane = next(row for row in report.lane_scores if row.lane_id == "decision_action")
    assert report.validity.valid is True
    assert analysis_lane.submitted_target_id == "trialdev.phase2.unregistered.v1"
    assert analysis_lane.score == 0.0
    assert action_lane.score == 1.0
    assert report.primary_score == 0.0


def test_wrong_design_cell_zeros_only_design_lane_for_adequate_request(
    tmp_path: Path,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    materialized_request_path = scenario / "trial_output" / "request.json"
    materialized_request = json.loads(materialized_request_path.read_text(encoding="utf-8"))
    materialized_request["design_cell_id"] = "trialdev.phase2.unregistered.v1"
    write_json(materialized_request_path, materialized_request)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["request"]["design_cell_id"] = "trialdev.phase2.unregistered.v1"
    write_json(submission, payload)

    report = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )

    design_lane = next(row for row in report.lane_scores if row.lane_id == "phase_design")
    action_lane = next(row for row in report.lane_scores if row.lane_id == "decision_action")
    assert report.validity.valid is True
    assert design_lane.submitted_target_id == "trialdev.phase2.unregistered.v1"
    assert design_lane.score == 0.0
    assert action_lane.score == 1.0
    assert report.primary_score == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enrollment_window_days", 999),
        ("site_count_budget", 999),
        ("site_strategy", "region_balanced"),
        ("stratification_variables", ["X"]),
        ("analysis_covariates", ["X"]),
        ("subgroup_variables", ["X"]),
    ],
)
def test_design_witness_rejects_materialized_request_identity_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    request_path = scenario / "trial_output" / "request.json"
    materialized_request = json.loads(request_path.read_text(encoding="utf-8"))
    materialized_request[field] = value
    write_json(request_path, materialized_request)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")

    with pytest.raises(ValueError, match="design-defining fields"):
        grade_item_v1(
            scenario_root=scenario,
            submission_path=submission,
            trial_output_root=scenario / "trial_output",
        )


def test_public_design_contract_rejects_missing_required_fields(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    incomplete = tmp_path / "incomplete.json"
    _write_submission(incomplete, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(incomplete.read_text(encoding="utf-8"))
    payload["request"].pop("target_sample_size")
    payload["request"].pop("follow_up_days")
    payload["request"].pop("endpoint_id")
    write_json(incomplete, payload)

    with pytest.raises(ValueError, match="target_sample_size|endpoint_id|follow_up_days"):
        grade_item_v1(
            scenario_root=scenario,
            submission_path=incomplete,
            trial_output_root=scenario / "trial_output",
        )


def test_public_request_contract_rejects_ambiguous_allocation() -> None:
    with pytest.raises(ValidationError, match="exactly one allocation_ratio or allocation_weights"):
        TrialDevelopmentRequestV1(
            scenario_id="s01",
            phase_id="phase2",
            candidate_drug_ids=("drug_a",),
            target_sample_size=80,
            endpoint_id="E1",
            follow_up_days=90,
            enrollment_window_days=42,
            site_count_budget=8,
            allocation_ratio="1:1",
            allocation_weights=(1.0, 1.0),
            design_cell_id="trialdev.phase2.fixed_final_operating_characteristics.v1",
            treatment_discontinuation_strategy="treatment_policy",
            interim_policy="fixed_final",
            site_strategy="high_enrolling",
            selection_objective="benefit_risk",
        )


def test_public_observational_request_accepts_candidate_comparison() -> None:
    request = TrialDevelopmentRequestV1(
        scenario_id="s01",
        phase_id="observational_review",
        candidate_drug_ids=("drug_a", "drug_b"),
        selection_objective="benefit_risk",
    )

    assert request.candidate_drug_ids == ("drug_a", "drug_b")


def test_public_randomized_request_rejects_multiple_investigational_regimens() -> None:
    with pytest.raises(ValidationError, match="exactly one investigational regimen"):
        TrialDevelopmentRequestV1(
            scenario_id="s01",
            phase_id="phase2",
            candidate_drug_ids=("drug_a", "drug_b"),
            target_sample_size=80,
            endpoint_id="E1",
            follow_up_days=90,
            enrollment_window_days=42,
            site_count_budget=8,
            allocation_ratio="1:1",
            design_cell_id="trialdev.phase2.fixed_final_operating_characteristics.v1",
            treatment_discontinuation_strategy="treatment_policy",
            interim_policy="fixed_final",
            site_strategy="high_enrolling",
            selection_objective="benefit_risk",
        )


def test_public_menu_violation_is_a_participant_correctable_rejection(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    request = TrialDevelopmentRequestV1(
        scenario_id="s01",
        phase_id="phase2",
        candidate_drug_ids=("drug_a",),
        target_sample_size=201,
        endpoint_id="E1",
        follow_up_days=90,
        enrollment_window_days=42,
        site_count_budget=8,
        allocation_ratio="1:1",
        design_cell_id="trialdev.phase2.fixed_final_operating_characteristics.v1",
        treatment_discontinuation_strategy="treatment_policy",
        interim_policy="fixed_final",
        site_strategy="high_enrolling",
        selection_objective="benefit_risk",
    )

    with pytest.raises(TrialDevelopmentRequestRejectedError, match="target_sample_size exceeds"):
        validate_request_against_scenario_v1(scenario_root=scenario, request=request)


def test_released_materialization_requires_a_fixed_trajectory_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trialagentbench_harness.trialdev.grading.sequential as seq

    monkeypatch.setattr(seq, "validate_design_request_file_v1", lambda **_: object())
    monkeypatch.setattr(seq, "validate_program_state_file_v1", lambda **_: object())
    monkeypatch.setattr(seq, "_validate_phase_request", lambda **_: None)

    with pytest.raises(FileNotFoundError, match="does not regenerate construction-time trial worlds"):
        seq.materialize_phase_v1(
            scenario_root=tmp_path / "scenario",
            state_path=tmp_path / "state.json",
            request_path=tmp_path / "request.json",
            out_dir=tmp_path / "output",
            seed=1,
        )


def test_missing_fixed_trajectory_index_fails_before_writing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trialagentbench_harness.trialdev.grading.sequential as seq

    monkeypatch.setattr(seq, "validate_design_request_file_v1", lambda **_: object())
    monkeypatch.setattr(seq, "validate_program_state_file_v1", lambda **_: object())
    monkeypatch.setattr(seq, "_validate_phase_request", lambda **_: None)

    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="fixed_trajectories/cases.jsonl"):
        seq.materialize_phase_v1(
            scenario_root=tmp_path / "scenario",
            state_path=tmp_path / "state.json",
            request_path=tmp_path / "request.json",
            out_dir=output,
            seed=1,
        )
    assert not output.exists()


def test_distinct_publicly_feasible_designs_receive_equal_design_credit(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_submission(first, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["request"]["target_sample_size"] = 180
    payload["request"]["follow_up_days"] = 120
    write_json(second, payload)

    first_request = TrialDevelopmentRequestV1.model_validate(json.loads(first.read_text(encoding="utf-8"))["request"])
    second_request = TrialDevelopmentRequestV1.model_validate(payload["request"])
    first_witness = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=first_request,
        phase_id="phase2",
    )
    second_witness = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=second_request,
        phase_id="phase2",
    )

    assert first_witness.adequate is True
    assert second_witness.adequate is True
    first_efficiency = derive_phase_design_efficiency_v1(
        scenario_root=scenario,
        request=first_request,
        design_witness=first_witness,
    )
    second_efficiency = derive_phase_design_efficiency_v1(
        scenario_root=scenario,
        request=second_request,
        design_witness=second_witness,
    )
    assert first_efficiency.design_valid is True
    assert second_efficiency.design_valid is True
    assert first_efficiency.participant_excess_vs_minimum > (second_efficiency.participant_excess_vs_minimum)
    assert first_efficiency.follow_up_excess_days_vs_minimum == 0
    assert second_efficiency.follow_up_excess_days_vs_minimum == 30
    assert first_efficiency.dominated_by_frontier is True
    assert second_efficiency.dominated_by_frontier is True
    first_resource = derive_phase_resource_consequence_v1(
        request=first_request,
        design_efficiency=first_efficiency,
        entered_after_unsupported_advance=False,
    )
    second_resource = derive_phase_resource_consequence_v1(
        request=second_request,
        design_efficiency=second_efficiency,
        entered_after_unsupported_advance=True,
    )
    assert first_resource.design_status == "valid_dominated"
    assert first_resource.dominating_frontier
    assert first_resource.avoidable_participant_follow_up_days_min > 0
    programme = derive_programme_resource_consequence_v1((first_resource, second_resource))
    assert programme.total_participants == 380
    assert programme.late_continuation_participants == 180
    assert programme.cost_status == "not_available_without_public_cost_schedule"


def test_exact_public_frontier_design_is_valid_and_nondominated(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    request = TrialDevelopmentRequestV1.model_validate(json.loads(submission.read_text(encoding="utf-8"))["request"])
    witness = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=request,
        phase_id="phase2",
    )
    initial = derive_phase_design_efficiency_v1(
        scenario_root=scenario,
        request=request,
        design_witness=witness,
    )
    point = initial.frontier[0]
    payload = request.model_dump(mode="json")
    payload.update(
        {
            "target_sample_size": point.target_sample_size,
            "follow_up_days": point.follow_up_days,
            "allocation_ratio": point.allocation_ratio,
            "allocation_weights": [],
        }
    )
    frontier_request = TrialDevelopmentRequestV1.model_validate(payload)
    frontier_witness = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=frontier_request,
        phase_id="phase2",
    )
    efficiency = derive_phase_design_efficiency_v1(
        scenario_root=scenario,
        request=frontier_request,
        design_witness=frontier_witness,
    )

    assert frontier_witness.adequate is True
    assert efficiency.design_valid is True
    assert efficiency.on_frontier is True
    assert efficiency.dominated_by_frontier is False
    resource = derive_phase_resource_consequence_v1(
        request=frontier_request,
        design_efficiency=efficiency,
        entered_after_unsupported_advance=False,
    )
    assert resource.design_status == "valid_frontier"
    assert resource.dominating_frontier == ()
    assert resource.avoidable_participants_max == 0
    assert resource.avoidable_follow_up_days_max == 0


def test_design_frontier_build_to_grade_parity(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    artifact = build_phase_design_frontiers_v1(scenario_root=scenario)
    stratum = artifact.strata[0]
    point = stratum.frontier[0]
    modules = json.loads((scenario / "public" / "phase_module_catalog.json").read_text(encoding="utf-8"))[
        "phase_modules"
    ]
    module = next(value for value in modules if value["phase_id"] == stratum.phase_id)
    request = TrialDevelopmentRequestV1(
        scenario_id=artifact.scenario_id,
        phase_id=stratum.phase_id,
        candidate_drug_ids=stratum.candidate_drug_ids,
        target_sample_size=point.target_sample_size,
        endpoint_id=stratum.endpoint_id,
        follow_up_days=point.follow_up_days,
        enrollment_window_days=min(module["allowed_enrollment_window_days"]),
        site_count_budget=min(module["allowed_site_count_budgets"]),
        allocation_ratio=point.allocation_ratio,
        design_cell_id=stratum.design_cell_id,
        treatment_discontinuation_strategy=stratum.treatment_discontinuation_strategy,
        interim_policy=stratum.interim_policy,
        site_strategy=min(module["allowed_site_strategies"]),
        selection_objective=min(module["allowed_selection_objectives"]),
    )
    witness = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=request,
        phase_id=stratum.phase_id,
    )
    efficiency = derive_phase_design_efficiency_v1(
        scenario_root=scenario,
        request=request,
        design_witness=witness,
    )

    assert efficiency.frontier == stratum.frontier
    assert efficiency.on_frontier is True

    alternate_request = request.model_copy(
        update={
            "enrollment_window_days": max(module["allowed_enrollment_window_days"]),
            "site_count_budget": max(module["allowed_site_count_budgets"]),
            "site_strategy": max(module["allowed_site_strategies"]),
            "selection_objective": max(module["allowed_selection_objectives"]),
        }
    )
    alternate_witness = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=alternate_request,
        phase_id=stratum.phase_id,
    )
    alternate_efficiency = derive_phase_design_efficiency_v1(
        scenario_root=scenario,
        request=alternate_request,
        design_witness=alternate_witness,
    )
    assert alternate_efficiency.frontier == efficiency.frontier


def test_design_frontier_exposes_exact_public_operational_support(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    artifact = load_phase_design_frontiers_v1(scenario_root=scenario)

    assert artifact.operational_support
    assert {record.phase_id for record in artifact.operational_support} == {"phase2"}
    assert max(record.eligible_subject_count for record in artifact.operational_support) == 300
    assert max(point.target_sample_size for stratum in artifact.strata for point in stratum.frontier) <= 300


def test_request_contract_rejects_unplanned_eligibility_population(tmp_path: Path) -> None:
    _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    payload = json.loads(submission.read_text(encoding="utf-8"))["request"]
    payload["eligibility_filters"] = [{"kind": "numeric_range", "variable_id": "AGE", "min_value": 50.0}]

    with pytest.raises(ValidationError, match="eligibility_filters"):
        TrialDevelopmentRequestV1.model_validate(payload)


def test_design_frontier_missing_artifact_fails_without_runtime_recomputation(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    request = TrialDevelopmentRequestV1.model_validate(json.loads(submission.read_text(encoding="utf-8"))["request"])
    witness = derive_phase_design_witness_v1(
        scenario_root=scenario,
        request=request,
        phase_id="phase2",
    )
    (scenario / "public" / "phase_design_frontiers.json").unlink()

    with pytest.raises(FileNotFoundError, match="frontier artifact is missing"):
        derive_phase_design_efficiency_v1(
            scenario_root=scenario,
            request=request,
            design_witness=witness,
        )


def test_design_frontier_stale_source_checksum_fails(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    source = scenario / "public" / "phase_module_catalog.json"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source checksums are stale"):
        load_phase_design_frontiers_v1(scenario_root=scenario)


def test_design_frontier_ambiguous_stratum_fails(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    path = scenario / "public" / "phase_design_frontiers.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strata"].append(dict(payload["strata"][0]))
    payload.pop("checksum")
    payload["checksum"] = compute_sha256_hex(payload)
    write_json(path, payload)

    with pytest.raises(ValidationError, match="strata must be unique"):
        load_phase_design_frontiers_v1(scenario_root=scenario)


def test_release_rejects_public_stratum_without_feasible_design(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    path = scenario / "public" / "phase_design_frontiers.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strata"][0]["frontier"] = []
    payload.pop("checksum")
    payload["checksum"] = compute_sha256_hex(payload)
    write_json(path, payload)

    with pytest.raises(ValueError, match="requires at least one feasible statistical design"):
        validate_release_v1(scenario_root=scenario)


def test_eval_contract_checksum_is_stable_across_json_round_trip() -> None:
    contract = TrialDevelopmentEvalContractV1(
        scenario_id="s01",
        phase_modules=(
            PhaseModuleSpecV1(
                phase_id="phase1",
                allowed_follow_up_days=(28,),
                allowed_enrollment_window_days=(42,),
                allowed_site_count_budgets=(8,),
                allowed_allocation_ratios=("1:1",),
                allowed_interim_policies=("fixed_final",),
                allowed_site_strategies=("high_enrolling",),
                allowed_selection_objectives=("benefit_risk",),
                max_sample_size=100,
                max_analysis_covariates=8,
                max_subgroup_splits=2,
            ),
        ),
    )

    reloaded = TrialDevelopmentEvalContractV1.model_validate(contract.model_dump(mode="json", exclude_none=True))

    assert reloaded.checksum == contract.checksum


def test_monetary_cost_is_absent_without_a_public_cost_schedule(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    benefit_submission = tmp_path / "benefit.json"
    cost_submission = tmp_path / "cost.json"
    _write_submission(benefit_submission, objective_id="benefit_risk", selected="drug_a")
    _write_submission(cost_submission, objective_id="cost_effective_best", selected="drug_a")

    benefit = grade_item_v1(
        scenario_root=scenario,
        submission_path=benefit_submission,
        trial_output_root=scenario / "trial_output",
    )
    cost = grade_item_v1(
        scenario_root=scenario,
        submission_path=cost_submission,
        trial_output_root=scenario / "trial_output",
    )

    assert "cost_effectiveness" not in benefit.lane_status
    assert "cost_effectiveness" not in cost.lane_status
    assert "cost_effectiveness" not in cost.active_lane_scores


def test_hidden_recoverability_manifest_cannot_change_grade(tmp_path: Path) -> None:
    scenario = _write_minimal_scenario(tmp_path)
    submission = tmp_path / "submission.json"
    _write_submission(submission, objective_id="benefit_risk", selected="drug_a")
    baseline = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )
    write_json(
        scenario / "grader" / "recoverability_manifest.json",
        {
            "schema_id": "trialdev_recoverability_manifest_v1",
            "version": "v1",
            "scenario_id": "s01",
            "phase_id": "observational_review",
            "objective_id": "benefit_risk",
            "acceptable_candidate_set": ["drug_b"],
        },
    )
    report = grade_item_v1(
        scenario_root=scenario,
        submission_path=submission,
        trial_output_root=scenario / "trial_output",
    )

    assert report.primary_score == baseline.primary_score
    assert report.lane_scores == baseline.lane_scores


def _write_minimal_trajectory_scenario(root: Path) -> Path:
    scenario = root / "scenario_s01"
    grader = scenario / "grader"
    public = scenario / "public"
    grader.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    write_json(public / "eval_contract.json", {"scenario_id": "s01"})
    write_json(
        public / "candidate_drug_catalog.json",
        {
            "candidate_drugs": [
                {"candidate_drug_id": "control", "role": "control"},
                {"candidate_drug_id": "drug_a", "role": "investigational"},
            ]
        },
    )
    write_json(
        public / "program_loop_manifest.json",
        {
            "scenario_id": "s01",
            "program_archetype": "asset_development",
            "phase_order": ["observational_review", "phase1", "phase2", "phase3"],
            "conditionally_materializable_phase_ids": ["phase1", "phase2", "phase3"],
            "phase1_carryover_consequential": False,
            "terminal_statuses": ["stopped", "completed"],
            "public_state_summary_fields": [
                "scenario_id",
                "current_phase_id",
                "eligible_candidate_drug_ids",
                "completed_phase_ids",
            ],
            "phase_policy_checksum": "0" * 64,
            "decision_charter_checksum": "0" * 64,
            "phase_policy_modes": {"phase1": "required", "phase2": "optional", "phase3": "optional"},
        },
    )
    write_json(
        public / "phase_action_policy.json",
        {
            "version": "v1",
            "scenario_id": "s01",
            "program_archetype": "asset_development",
            "phase_policy_checksum": "0" * 64,
            "decision_charter_checksum": "0" * 64,
            "action_specs": _action_specs(),
        },
    )
    write_json(
        public / "safety_decision_policy.json",
        {
            "schema_id": "trialdev_safety_decision_policy_v1",
            "scenario_id": "s01",
            "serious_event_definitions": [
                {
                    "endpoint_id": "test",
                    "event_column": "AE_TEST_EVENT_E",
                    "time_column": "AE_TEST_EVENT_T",
                    "seriousness_column": "AE_TEST_SERIOUS",
                    "severity_column": "AE_TEST_SEVERITY",
                }
            ],
            "checksum": "0" * 64,
            "thresholds": [
                {
                    "phase_id": phase_id,
                    "component_id": component_id,
                    "role": role,
                    "max_absolute_rate": max_absolute_rate,
                    "max_excess_vs_control": 0.035,
                    "sensitivity_max_absolute_rates": {
                        "strict": max_absolute_rate - borderline_margin,
                        "primary": max_absolute_rate,
                        "permissive": max_absolute_rate + borderline_margin,
                    },
                    "sensitivity_max_excess_vs_control": {
                        "strict": 0.035 - borderline_margin,
                        "primary": 0.035,
                        "permissive": 0.035 + borderline_margin,
                    },
                }
                for phase_id in ("phase1", "phase2", "phase3")
                for component_id, role, max_absolute_rate, borderline_margin in (
                    ("serious_ae", "hard_gate", 0.12, 0.02),
                    ("discontinuation", "diagnostic_only", 0.18, 0.03),
                )
            ],
            "decision_rules": [],
        },
    )
    write_json(
        grader / "safety_alignment_report.json",
        {
            "schema_id": "trialdev_safety_alignment_report_v1",
            "version": "v1",
            "scenario_id": "s01",
            "passed": True,
            "issue_count": 0,
            "issues": [],
            "rows": [],
            "checksum": "0" * 64,
        },
    )
    evaluation_targets = []
    for lane_id in ("phase_design", "phase_analysis", "safety_gate", "decision_action"):
        evaluation_targets.append(
            {
                "schema_id": "trialdev_evaluation_target_register_record_v1",
                "scenario_id": "s01",
                "phase_id": "phase1",
                "program_objective_id": "net_clinical_value_under_budget",
                "phase_scoring_objective_id": "benefit_risk",
                "lane_id": lane_id,
                "scoring_policy_id": "primary",
                "public_evidence_basis": ["public/phase_action_policy.json"],
                "evaluator_evidence_basis": ["grader/evaluation_target_register.jsonl"],
                "reference_target_ids": ["stop_development"],
                "credit_eligible_target_ids": [],
                "recoverability_policy_id": "no_recoverability_relaxation",
                "value_payload": {},
                "checksum": "0" * 64,
            }
        )
    for lane_id in ("route_timing", "final_recommendation"):
        evaluation_targets.append(
            {
                "schema_id": "trialdev_evaluation_target_register_record_v1",
                "scenario_id": "s01",
                "phase_id": "final_decision",
                "program_objective_id": "net_clinical_value_under_budget",
                "phase_scoring_objective_id": "net_clinical_value_under_budget",
                "lane_id": lane_id,
                "scoring_policy_id": "primary",
                "public_evidence_basis": ["public/program_loop_manifest.json"],
                "evaluator_evidence_basis": ["grader/evaluation_target_register.jsonl"],
                "reference_target_ids": ["stop_development"],
                "credit_eligible_target_ids": [],
                "recoverability_policy_id": "no_recoverability_relaxation",
                "value_payload": {},
                "checksum": "0" * 64,
            }
        )
    (grader / "evaluation_target_register.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in evaluation_targets) + "\n",
        encoding="utf-8",
    )
    return scenario


def test_trajectory_scoring_is_report_mode_invariant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trialagentbench_harness.trialdev.grading.sequential as seq

    scenario = _write_minimal_trajectory_scenario(tmp_path)
    workdir = tmp_path / "trajectory"
    initial_state_path = tmp_path / "initial_state.json"
    phase_dir = workdir / "phase_phase1"
    phase_dir.mkdir(parents=True)
    initial_state = TrialDevSingleAssetProgrammeStateV1(
        programme_id="s01__net_clinical_value_under_budget",
        scenario_id="s01",
        stream_id="single_asset_development",
        current_checkpoint_id="early_safety_study",
        candidate_asset_ids=("drug_a",),
        nominated_asset_id="drug_a",
        active_asset_id="drug_a",
        policy_binding=TrialDevPolicyBindingV1(
            stream_id="single_asset_development",
            objective_id="net_clinical_value_under_budget",
            objective_policy_checksum="a" * 64,
            action_policy_checksum="a" * 64,
            design_menu_checksum="a" * 64,
        ),
        evidence=(
            TrialDevEvidenceReferenceV1(
                evidence_id="phase1-protocol",
                evidence_kind="protocol",
                checkpoint_id="early_safety_study",
                asset_id="drug_a",
                evidence_protocol_id="phase1",
                evidence_protocol_checksum="a" * 64,
                source_family_id="s01",
                world_id="s01",
                relative_path="public/phase_module_catalog.json",
                artifact_sha256="a" * 64,
            ),
        ),
    )
    write_json(initial_state_path, initial_state.model_dump(mode="json", exclude_none=True))
    write_json(
        workdir / "scoring_context.json",
        {
            "version": "v1",
            "scenario_id": "s01",
            "program_id": "s01__net_clinical_value_under_budget",
            "program_objective_id": "net_clinical_value_under_budget",
            "phase_scoring_objectives": {
                "phase1": "benefit_risk",
                "phase2": "net_clinical_value_under_budget",
            },
        },
    )

    request = TrialDevelopmentRequestV1(
        scenario_id="s01",
        phase_id="phase1",
        candidate_drug_ids=("drug_a",),
        target_sample_size=40,
        follow_up_days=28,
        enrollment_window_days=42,
        site_count_budget=8,
        allocation_ratio="1:1",
        design_cell_id="trialdev.phase1.fixed_final_operating_characteristics.v1",
        interim_policy="fixed_final",
        site_strategy="high_enrolling",
        selection_objective="net_clinical_value_under_budget",
    )
    request_checksum = request.checksum()
    output_manifest = TrialDevelopmentTrialOutputManifestV1(
        scenario_id="s01",
        phase_id="phase1",
        request_checksum=request_checksum,
        evidence_request_checksum=request_checksum,
        table_checksums={
            "participants.parquet": "1" * 64,
            "endpoints.parquet": "2" * 64,
            "safety.parquet": "3" * 64,
        },
        n_participants=10,
    )
    write_json(phase_dir / "request.json", request.model_dump(mode="json", exclude_none=True))
    trial_output_dir = phase_dir / "trial_output"
    trial_output_dir.mkdir()
    write_json(trial_output_dir / "request.json", request.model_dump(mode="json", exclude_none=True))
    write_json(
        trial_output_dir / "trial_output_manifest.json",
        output_manifest.model_dump(mode="json", exclude_none=True),
    )
    write_json(
        phase_dir / "analysis_submission.json",
        {
            "version": "v1",
            "scenario_id": "s01",
            "phase_id": "phase1",
            "request_checksum": request_checksum,
            "trial_output_checksum": str(output_manifest.checksum),
            "selected_winner_drug_id": "drug_a",
            "ranked_drug_ids": ["drug_a"],
            "diagnostic_artifacts": [
                {
                    "artifact_id": "analysis_summary",
                    "metric_family": "robustness",
                    "primary_value": 10.0,
                }
            ],
        },
    )
    write_json(
        phase_dir / "decision_submission.json",
        {
            "version": "v1",
            "scenario_id": "s01",
            "phase_id": "phase1",
            "request_checksum": request_checksum,
            "analysis_checksum": "a" * 64,
            "decision_action": "stop_development",
            "candidate_drug_id": "drug_a",
            "supporting_evidence_ids": ["analysis_summary"],
        },
    )

    monkeypatch.setattr(seq, "_validate_phase_request", lambda **kwargs: None)
    monkeypatch.setattr(seq, "_resolve_trial_output_root", lambda **kwargs: phase_dir)
    monkeypatch.setattr(seq, "validate_trial_output_bundle_v1", lambda **kwargs: output_manifest)
    monkeypatch.setattr(seq, "sha256_file_hex", lambda path: "a" * 64)
    monkeypatch.setattr(seq, "derive_phase_design_witness_v1", lambda **kwargs: object())
    monkeypatch.setattr(seq, "_validate_decision_evidence_links", lambda **kwargs: None)

    def _grade_stepwise_report(**kwargs: Any) -> TrialDevelopmentGradeReportV1:
        return TrialDevelopmentGradeReportV1(
            scenario_id="s01",
            phase_id="phase1",
            objective_id="benefit_risk",
            program_objective_id="net_clinical_value_under_budget",
            phase_scoring_objective_id="benefit_risk",
            primary_score=0.6,
            design_score=0.6,
            evaluation_score=0.6,
            program_score=0.9,
            ranking_score=1.0,
            analysis_quality=TrialDevelopmentAnalysisQualityV1(
                observational_analysis_eligible=False,
                observational_analysis_valid=None,
                randomized_primary_effect_eligible=False,
                randomized_primary_effect_valid=None,
                safety_evidence_eligible=True,
                safety_evidence_valid=True,
                safety_evidence_agreement=0.6,
                phase_evaluation_valid=True,
            ),
            lane_status={
                "trial_design": "active",
                "trial_evaluation": "active",
                "program_decision": "active",
                "drug_ranking": "not_applicable",
            },
            active_lane_scores={},
            gates=(
                {"gate_id": "submission", "status": "passed"},
                {"gate_id": "question", "status": "passed"},
                {"gate_id": "route", "status": "passed"},
                {"gate_id": "evidence", "status": "passed"},
                {"gate_id": "integrity", "status": "not_applicable"},
                {
                    "gate_id": "result",
                    "status": "failed",
                    "failure_code": "submitted_result_not_equivalent",
                },
                {"gate_id": "conformance", "status": "not_reached"},
                {"gate_id": "decision", "status": "not_reached"},
            ),
            first_failure_gate="result",
            validity=TrialDevelopmentValidityReportV1(),
            audit_gates=TrialDevelopmentAuditGateReportV1(diagnostic_alignment_score=0.6),
            design_efficiency=TrialDevDesignEfficiencyV1(
                statistically_adequate=True,
                operationally_feasible=True,
                design_valid=True,
                on_frontier=True,
                dominated_by_frontier=False,
                operational_support=40,
                operational_headroom=0,
                operational_shortage=0,
                minimum_frontier_participants=40,
                minimum_frontier_follow_up_days=28,
                participant_excess_vs_minimum=0,
                participant_shortage_vs_minimum=0,
                follow_up_excess_days_vs_minimum=0,
                follow_up_shortage_days_vs_minimum=0,
                achieved_power=None,
                target_power=None,
                achieved_safety_absolute_risk_power=0.9,
                achieved_safety_excess_risk_power=0.9,
                target_safety_decision_power=0.8,
                frontier=(
                    TrialDevDesignFrontierPointV1(
                        target_sample_size=40,
                        follow_up_days=28,
                        allocation_ratio="1:1",
                        achieved_power=None,
                        achieved_safety_absolute_risk_power=0.9,
                        achieved_safety_excess_risk_power=0.9,
                    ),
                ),
            ),
            policy_reference_regret=0.0,
            in_set_regret=0.0,
            selected_winner_drug_id="drug_a",
            best_candidate_drug_id="drug_a",
            lane_breakdown={
                "trial_design": 0.6,
                "trial_evaluation": 0.6,
                "program_decision": 0.9,
                "drug_ranking": 1.0,
            },
            payload={
                "decision_action_score": 0.0,
                "analysis_method_diagnostics": {
                    "submitted_method_route_id": "submitted",
                    "required_method_route_id": "required",
                },
                "numeric_score_components": {
                    "score_before_method_and_completeness_gates": 0.8,
                    "score_after_method_and_completeness_gates": 0.6,
                },
            },
        )

    assert _grade_stepwise_report().schema_id == "trialdev_grade_report_v1"
    monkeypatch.setattr(seq, "_grade_stepwise_report", _grade_stepwise_report)
    monkeypatch.setattr(
        seq,
        "derive_phase_design_efficiency_v1",
        lambda **kwargs: _grade_stepwise_report().design_efficiency,
    )

    score_payload = grade_trajectory_v1(
        scenario_root=scenario,
        trajectory_root=workdir,
        initial_state_path=initial_state_path,
        report_mode="score",
    )
    audit_payload = grade_trajectory_v1(
        scenario_root=scenario,
        trajectory_root=workdir,
        initial_state_path=initial_state_path,
        report_mode="audit",
    )

    assert score_payload["trajectory_primary_score"] == audit_payload["trajectory_primary_score"]
    assert score_payload["trajectory_primary_score"] == pytest.approx(0.6)
    assert score_payload["trajectory_decision_score"] == audit_payload["trajectory_decision_score"] == 0.0
    assert score_payload["n_invalid_attempts"] == 0
    assert score_payload["decision_regret_by_phase"] == audit_payload["decision_regret_by_phase"]
    assert score_payload["program_objective_id"] == "net_clinical_value_under_budget"
    assert score_payload["phase_scoring_objectives"]["phase1"] == "benefit_risk"
    assert score_payload["terminal_summary"]["terminal_action"] == "stop_development"
    assert audit_payload["terminal_summary"]["terminal_action"] == "stop_development"
    assert score_payload["phase_reports"][0]["analysis_method_diagnostics"] == {
        "submitted_method_route_id": "submitted",
        "required_method_route_id": "required",
    }
    assert (
        score_payload["phase_reports"][0]["numeric_score_components"]["score_before_method_and_completeness_gates"]
        == 0.8
    )

    phase_report = wrap_grade_record(score_payload["phase_reports"][0])
    observational_report = phase_report.model_copy(update={"phase_id": "observational_review", "primary_score": 0.8})
    trajectory_grade = wrap_trajectory_grade(score_payload)
    programme_quality = TrialDevProgrammeAnalysisQualityV1(
        observational_analysis_validity=TrialDevAnalysisQualityEndpointV1(eligible_units=1, value=1.0),
        observational_analysis_score=TrialDevAnalysisQualityEndpointV1(eligible_units=1, value=0.8),
        randomized_primary_effect_point_agreement=TrialDevAnalysisQualityEndpointV1(eligible_units=0),
        randomized_primary_effect_interval_agreement=TrialDevAnalysisQualityEndpointV1(eligible_units=0),
        safety_evidence_agreement=TrialDevAnalysisQualityEndpointV1(eligible_units=1, value=0.6),
        phase_evaluation_validity=TrialDevAnalysisQualityEndpointV1(eligible_units=2, value=1.0),
    )
    metrics = trajectory_metrics_from_grade(
        trajectory_grade=trajectory_grade,
        observational_report=observational_report,
        phase_policy_modes={"phase1": "required", "phase2": "required", "phase3": "optional"},
        analysis_quality=programme_quality,
    )
    outcomes = {record.phase_id: record for record in metrics.checkpoint_outcomes}
    assert outcomes["observational_review"].conditional_score == pytest.approx(0.8)
    assert outcomes["phase1"].conditional_score == pytest.approx(0.6)
    assert outcomes["phase1"].cumulative_score == pytest.approx(0.6)
    assert outcomes["phase2"].status == "structural_not_reached"
    assert outcomes["phase3"].status == "structural_not_reached"
    assert outcomes["final_decision"].status == "reached"
    assert metrics.programme_primary_score == 0.0

    missing_observation = trajectory_metrics_from_grade(
        trajectory_grade=trajectory_grade,
        observational_report=None,
        phase_policy_modes={"phase1": "required", "phase2": "required", "phase3": "optional"},
        analysis_quality=metrics.analysis_quality,
    )
    missing_outcomes = {record.phase_id: record for record in missing_observation.checkpoint_outcomes}
    assert missing_outcomes["observational_review"].status == "missing_or_invalid"
    assert missing_outcomes["phase2"].status == "not_reached_after_invalid"
    assert missing_observation.programme_primary_score == 0.0

    (phase_dir / "analysis_submission.json").unlink()
    incomplete_payload = grade_trajectory_v1(
        scenario_root=scenario,
        trajectory_root=workdir,
        initial_state_path=initial_state_path,
        report_mode="score",
    )

    assert incomplete_payload["trajectory_primary_score"] == 0.0
    assert incomplete_payload["trajectory_decision_score"] == 0.0
    assert incomplete_payload["n_invalid_attempts"] == 1
    assert incomplete_payload["invalid_attempts"][0]["reason_code"] == "invalid_analysis"
    assert incomplete_payload["resource_consequence"]["total_participants"] == 40
    assert len(incomplete_payload["resource_consequence"]["phases"]) == 1
