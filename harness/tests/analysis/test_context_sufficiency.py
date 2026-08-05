from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from trialagentbench_harness.analysis.context_sufficiency import (
    validate_trialeval_context_sufficiency_v1,
)
from trialagentbench_harness.contracts.trialeval_diagnostics import participant_diagnostic_dictionary_v1
from trialagentbench_harness.contracts.trialeval_methods import (
    TrialEvalParticipantMethodDictionaryV1,
    TrialEvalParticipantMethodV1,
)
from trialagentbench_harness.io.json import read_json
from trialagentbench_harness.tools.validate.validate_trialeval_context_sufficiency import main
from trialagentbench_harness.verification.trialeval.analysis_class_witness import (
    derive_public_analysis_classes_v1,
)


def _locked_sap_payload(*, estimator_family: str, required_modifiers: tuple[str, ...]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "trial_analysis_plan_contract_v1",
        "task_id": "TASK001",
        "item_id": "TASK001",
        "primary_estimand_id": "itt_treatment_policy",
        "primary_endpoint_id": "death",
        "primary_analysis": {
            "effect_scale": "risk_difference_tau",
            "method_id": f"fixture:{estimator_family}",
            "estimator_family": estimator_family,
            "implementation": "Fixture implementation matching the declared method route.",
            "uncertainty_method": "fixture_uncertainty",
            "required_method_modifiers": list(required_modifiers),
            "baseline_covariate_strategy": "unadjusted",
        },
        "diagnostic_requirements": [
            {
                "assumption_id": "censoring_ignorability",
                "diagnostic_methods": ["censoring review"],
            }
        ],
        "sensitivity_analyses": [],
        "familywise_alpha": 0.05,
        "multiplicity_strategy": "none",
        "lane_rules": [
            {
                "lane_id": "primary_analysis.response.v1",
                "estimand_id": "itt_treatment_policy",
                "endpoint_id": "death",
                "effect_scale": "risk_difference_tau",
                "role": "primary",
                "multiplicity_family_id": "family_primary_confirmatory",
                "alpha": 0.05,
                "mandatory": True,
            }
        ],
    }
    payload["checksum"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return payload


def test_context_sufficiency_accepts_complete_stepped_wedge_reconstruction(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C4",
        design_family="stepped_wedge_cluster_rollout",
        protocol_extra={
            "stepped_wedge_plan": {
                "schema_id": "trial_analysis_stepped_wedge_plan_v1",
                "cluster_id_field": "SITEID",
                "baseline_date_field": "RFSTDTC",
                "planned_intervention_start_day_field": "INTERVENTION_START_DY",
                "period_length_dy": 28,
                "source_rel_path": "data/raw/randomization.parquet",
            }
        },
        extra_members=("items/TASK001/data/raw/randomization.parquet", "items/TASK001/data/raw/sites.parquet"),
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "pass"
    assert report.total_items == 1
    assert report.rows[0].disposition == "retain_official"
    assert report.rows[0].credit_eligible_route_ids == ("TASK001::primary",)
    assert report.rows[0].recoverable_route_ids == ("TASK001::primary",)
    assert report.rows[0].unrecoverable_route_ids == ()
    assert report.rows[0].analysis_class_witness.candidate_classes[0].estimator_family == (
        "stepped_wedge_period_adjusted"
    )
    assert "protocol_summary.json:stepped_wedge_plan.cluster_id_field" in report.rows[0].design_adjustment_fields


def test_context_sufficiency_recovers_public_limitation_route(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C4",
        design_family="randomized_trial",
        bounded_deviation_delta_options=(0.05, 0.1),
        method_family_override="bounds",
        include_limitation_route=True,
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "pass"
    assert report.rows[0].recoverable_route_ids == (
        "TASK001::limitation",
        "TASK001::primary",
    )


def test_context_sufficiency_rejects_limitation_without_its_required_public_diagnostic(
    tmp_path: Path,
) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C4",
        design_family="randomized_trial",
        bounded_deviation_delta_options=(0.05, 0.1),
        method_family_override="bounds",
        include_limitation_route=True,
        limitation_possible_diagnostic_ids=("endpoint_ascertainment_public",),
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "fail"
    assert report.rows[0].recoverable_route_ids == ("TASK001::primary",)
    assert report.rows[0].unrecoverable_route_ids == ("TASK001::limitation",)


def test_context_sufficiency_rejects_missing_stepped_wedge_source(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C4",
        design_family="stepped_wedge_cluster_rollout",
        protocol_extra={
            "stepped_wedge_plan": {
                "cluster_id_field": "SITEID",
                "baseline_date_field": "RFSTDTC",
                "planned_intervention_start_day_field": "INTERVENTION_START_DY",
                "period_length_dy": 28,
                "source_rel_path": "data/raw/randomization.parquet",
            }
        },
        extra_members=("items/TASK001/data/raw/sites.parquet",),
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "fail"
    assert report.rows[0].disposition == "block_release"
    assert "stepped_wedge_missing_source_rel_path" in report.rows[0].findings


def test_context_sufficiency_rejects_analysis_ready_context_without_tables(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C2",
        design_family="cluster_parallel_randomized",
        extra_members=("items/TASK001/data/site_summary.parquet",),
        include_analysis_tables=False,
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "fail"
    assert "analysis_ready_surface_missing_analysis_tables" in report.rows[0].findings


def test_context_sufficiency_rejects_missing_endpoint_definition_for_every_configuration(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C2",
        design_family="randomized_trial",
        include_endpoint_definition=False,
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "fail"
    assert "missing_required_participant_member:endpoint_definition.json" in report.rows[0].findings
    assert report.rows[0].has_endpoint_definition is False


def test_context_sufficiency_rejects_factor_configuration_mismatch(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C1",
        design_family="randomized_trial",
        data_preparation="raw_domains",
    )

    with pytest.raises(ValueError, match="C1 requires data_preparation='analysis_ready'"):
        validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)


def test_analysis_class_derivation_rejects_unknown_public_semantics() -> None:
    with pytest.raises(ValueError, match="Unsupported participant design_subtype"):
        derive_public_analysis_classes_v1(
            design_subtype="unknown",
            primary_effect_scale="risk_difference_tau",
            primary_effect_scale_options=("risk_difference_tau",),
            bounded_deviation_delta_options=(),
            ipcw_executable=False,
        )
    with pytest.raises(ValueError, match="must be unique"):
        derive_public_analysis_classes_v1(
            design_subtype="individual_randomized",
            primary_effect_scale=None,
            primary_effect_scale_options=("risk_difference_tau", "risk_difference_tau"),
            bounded_deviation_delta_options=(),
            ipcw_executable=False,
        )


def test_analysis_class_derivation_covers_every_declared_primary_scale() -> None:
    classes = derive_public_analysis_classes_v1(
        design_subtype="individual_randomized",
        primary_effect_scale=None,
        primary_effect_scale_options=("log_hr", "risk_difference_tau", "rmst_difference_tau"),
        bounded_deviation_delta_options=(),
        ipcw_executable=True,
    )

    assert {
        (row.estimator_family, row.effect_scale, row.applicability, row.required_modifiers) for row in classes
    } == {
        ("coxph_binary", "log_hr", "public_candidate", ()),
        ("coxph_ipcw", "log_hr", "diagnostic_contingent", ("ipcw_adjusted",)),
        ("coxph_binary", "risk_difference_tau", "diagnostic_contingent", ()),
        ("km", "risk_difference_tau", "public_candidate", ("ph_robust_fixed_horizon",)),
        ("km_ipcw", "risk_difference_tau", "diagnostic_contingent", ("ipcw_adjusted",)),
        ("km", "rmst_difference_tau", "public_candidate", ("ph_robust_fixed_horizon",)),
        ("km_ipcw", "rmst_difference_tau", "diagnostic_contingent", ("ipcw_adjusted",)),
    }


def test_context_sufficiency_rejects_scorer_class_not_recoverable_from_public_surface(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C2",
        design_family="randomized_trial",
        method_family_override="group_sequential_adjustment",
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "fail"
    assert report.rows[0].recoverable_route_ids == ()
    assert report.rows[0].unrecoverable_route_ids == ("TASK001::primary",)
    assert "credit_eligible_routes_not_recoverable_from_public_surface" in report.rows[0].findings


def test_context_sufficiency_rejects_one_unrecoverable_accepted_alternative(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C2",
        design_family="randomized_trial",
        alternative_method_family="group_sequential_adjustment",
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "fail"
    assert report.rows[0].credit_eligible_route_ids == ("TASK001::alternative", "TASK001::primary")
    assert report.rows[0].recoverable_route_ids == ("TASK001::primary",)
    assert report.rows[0].unrecoverable_route_ids == ("TASK001::alternative",)


def test_context_sufficiency_applies_locked_sap_route_policy(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C1",
        design_family="randomized_trial",
        alternative_method_family="group_sequential_adjustment",
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "pass"
    assert report.rows[0].credit_eligible_route_ids == ("TASK001::primary",)
    assert report.rows[0].recoverable_route_ids == ("TASK001::primary",)
    assert report.rows[0].unrecoverable_route_ids == ()


def test_context_sufficiency_rejects_locked_sap_method_family_mismatch(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C1",
        design_family="randomized_trial",
        sap_method_family_override="coxph_binary",
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "fail"
    assert "locked_sap_primary_analysis_family_mismatch" in report.rows[0].findings


def test_context_sufficiency_recovers_executable_bounds_from_public_result_contract(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C2",
        design_family="randomized_trial",
        method_family_override="bounds",
        bounded_deviation_delta_options=(0.1,),
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "pass"
    assert "bounds" in {
        candidate.estimator_family for candidate in report.rows[0].analysis_class_witness.candidate_classes
    }


def test_context_sufficiency_rejects_corrupt_locked_sap(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C1",
        design_family="randomized_trial",
        corrupt_sap=True,
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "fail"
    assert report.rows[0].sap_integrity_passed is False
    assert any(finding.startswith("locked_sap_invalid:") for finding in report.rows[0].findings)


def test_context_sufficiency_rejects_sap_leak_on_protocol_only_surface(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C2",
        design_family="randomized_trial",
        include_sap_in_protocol_only=True,
    )

    report = validate_trialeval_context_sufficiency_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)

    assert report.status == "fail"
    assert "protocol_only_surface_contains_locked_sap" in report.rows[0].findings


def test_context_sufficiency_cli_writes_artifacts(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_pair(
        tmp_path,
        context_tier="C1",
        design_family="group_sequential_monitoring",
        protocol_extra={
            "group_sequential_plan": {
                "looks": [0.5, 0.75, 1.0],
                "spending_function_id": "obrien_fleming",
                "monitoring_effect_scale": "risk_difference_tau",
                "two_sided_alpha": 0.05,
                "nominal_two_sided_alpha_by_look": [0.0056, 0.0215, 0.0411],
                "z_critical_by_look": [2.7718, 2.2981, 2.0426],
                "analysis_look_index": 2,
                "analysis_information_fraction": 1.0,
                "analysis_horizon_dy": 252.0,
                "information_basis": "planned_complete_subject_fraction",
                "stopped_early": False,
            }
        },
    )
    out_dir = tmp_path / "out"

    rc = main(
        [
            "--public-zip",
            public_zip.as_posix(),
            "--evaluator-zip",
            evaluator_zip.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
        ]
    )

    assert rc == 0
    assert read_json(out_dir / "trialeval_context_sufficiency_report.json")["status"] == "pass"
    assert "# TrialEvalBench Context-Sufficiency Report" in (
        out_dir / "trialeval_context_sufficiency_report.md"
    ).read_text(encoding="utf-8")


def _write_release_pair(
    tmp_path: Path,
    *,
    context_tier: str,
    design_family: str,
    protocol_extra: dict[str, object] | None = None,
    bounded_deviation_delta_options: tuple[float, ...] = (),
    extra_members: tuple[str, ...] = (),
    include_analysis_tables: bool = True,
    include_endpoint_definition: bool = True,
    data_preparation: str | None = None,
    analysis_specification: str | None = None,
    method_family_override: str | None = None,
    sap_method_family_override: str | None = None,
    alternative_method_family: str | None = None,
    corrupt_sap: bool = False,
    include_sap_in_protocol_only: bool = False,
    include_limitation_route: bool = False,
    limitation_possible_diagnostic_ids: tuple[str, ...] = (
        "censoring_followup_public",
        "endpoint_ascertainment_public",
    ),
) -> tuple[Path, Path]:
    context_factors = {
        "C1": ("analysis_ready", "locked_sap"),
        "C2": ("analysis_ready", "protocol_only"),
        "C3": ("raw_domains", "locked_sap"),
        "C4": ("raw_domains", "protocol_only"),
        "C5": ("raw_domains_declared_defect", "protocol_only"),
    }
    default_preparation, default_specification = context_factors[context_tier]
    selected_preparation = data_preparation or default_preparation
    selected_specification = analysis_specification or default_specification
    design_subtype_by_family = {
        "randomized_trial": "individual_randomized",
        "cluster_parallel_randomized": "cluster_parallel",
        "stepped_wedge_cluster_rollout": "stepped_wedge",
        "group_sequential_monitoring": "group_sequential",
    }
    design_subtype = design_subtype_by_family[design_family]
    method_family_by_subtype = {
        "individual_randomized": "km",
        "cluster_parallel": "cluster_parallel_participant_weighted",
        "stepped_wedge": "stepped_wedge_period_adjusted",
        "group_sequential": "km",
    }
    modifiers_by_subtype = {
        "individual_randomized": ("ph_robust_fixed_horizon",),
        "cluster_parallel": ("cluster_robust_inference", "participant_population_target"),
        "stepped_wedge": (
            "cluster_robust_inference",
            "participant_population_target",
            "stepped_wedge_period_adjusted",
        ),
        "group_sequential": (
            "group_sequential_adjustment",
            "ph_robust_fixed_horizon",
        ),
    }
    selected_method_family = method_family_override or method_family_by_subtype[design_subtype]
    is_bounds = selected_method_family == "bounds"
    required_modifiers = () if is_bounds else modifiers_by_subtype[design_subtype]
    selected_method_id = f"fixture:{selected_method_family}:risk_difference_tau"
    public_zip = tmp_path / "public.zip"
    evaluator_zip = tmp_path / "evaluator.zip"
    with ZipFile(evaluator_zip, "w") as zf:
        zf.writestr(
            "grader/item_index.json",
            json.dumps(
                {
                    "schema_id": "trialagentbench.trialeval_item_index/v1",
                    "version": 1,
                    "checksum": "fixture",
                    "entries": [
                        {
                            "task_id": "TASK001",
                            "item_id": "d4_fixture",
                            "variant_id": "d4_fixture__C",
                            "factors": {
                                "design_archetype": "D4",
                                "assumption_regime": "A2",
                                "context_configuration": context_tier,
                                "data_preparation": selected_preparation,
                                "analysis_specification": selected_specification,
                                "procedure_assistance": "output_contract_only",
                                "response_interface": "structured",
                            },
                            "reconstruction_row_count": 10 if selected_preparation != "analysis_ready" else 0,
                        }
                    ],
                }
            ),
        )
        routes = [
            {
                "route_id": "TASK001::primary",
                "signature": {
                    "analysis_population_id": "ITT",
                    "estimand_id": "itt_treatment_policy",
                    "intercurrent_event_strategy_ids": ["treatment_policy"],
                    "assessment_horizon_days": 180,
                    "treatment_id": "treated",
                    "comparator_id": "control",
                    "endpoint_id": "death",
                    "effect_scale": "risk_difference_tau",
                    "analysis_method_id": selected_method_id,
                },
                "method": {
                    "analysis_method_id": selected_method_id,
                    "estimator_family": selected_method_family,
                    "result_kind": "identification_bound" if is_bounds else "numeric_point",
                    "uncertainty_method": "fixture_uncertainty",
                    "design_modifiers": list(required_modifiers),
                },
                "required_identification_assumptions": ["randomization_exchangeability"],
                "required_diagnostics": [],
                "planning_calculator_id": None,
                "target": (
                    {
                        "kind": "numeric_interval",
                        "lower": -0.1,
                        "upper": 0.1,
                        "result_unit": "probability_difference",
                        "acceptance_envelope": {
                            "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
                            "reporting_decimal_places": 2,
                            "independent_max_abs_difference": 0.005,
                            "public_verification_id": "fixture-public-bounds-replay",
                            "independent_verification_ids": ["fixture-independent-bounds-replay"],
                        },
                    }
                    if is_bounds
                    else {
                        "kind": "numeric_point",
                        "value": -0.05,
                        "result_unit": "probability_difference",
                        "acceptance_envelope": {
                            "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
                            "reporting_decimal_places": 2,
                            "independent_max_abs_difference": 0.005,
                            "public_verification_id": "fixture-public-point-replay",
                            "independent_verification_ids": ["fixture-independent-point-replay"],
                        },
                        "require_confidence_interval": True,
                        "confidence_interval_lower": -0.08,
                        "confidence_interval_upper": -0.02,
                    }
                ),
            }
        ]
        if alternative_method_family is not None and selected_specification != "locked_sap":
            alternative_method_id = f"fixture:{alternative_method_family}:risk_difference_tau"
            routes.append(
                {
                    **routes[0],
                    "route_id": "TASK001::alternative",
                    "signature": {
                        **routes[0]["signature"],
                        "analysis_method_id": alternative_method_id,
                    },
                    "method": {
                        **routes[0]["method"],
                        "analysis_method_id": alternative_method_id,
                        "estimator_family": alternative_method_family,
                    },
                }
            )
        if include_limitation_route:
            routes.append(
                {
                    **routes[0],
                    "route_id": "TASK001::limitation",
                    "signature": {
                        **routes[0]["signature"],
                        "analysis_method_id": "qualified_limitation_or_abstention",
                    },
                    "method": {
                        "analysis_method_id": "qualified_limitation_or_abstention",
                        "estimator_family": "qualified_nonidentification",
                        "result_kind": "limitation",
                        "uncertainty_method": "not_applicable",
                        "design_modifiers": [],
                    },
                    "required_diagnostics": ["censoring_followup_public"],
                    "target": {
                        "kind": "categorical",
                        "credit_eligible_codes": [
                            "point_not_identified_due_to_censoring_or_support_failure",
                        ],
                    },
                }
            )
        scoring_key = {
            "schema_id": "trialagentbench.scoring_key/v1",
            "release_id": "fixture_release",
            "item_id": "TASK001",
            "question_id": "primary_analysis",
            "context_tier": context_tier,
            "credit_eligible_routes": routes,
        }
        scoring_key_body = (json.dumps(scoring_key, sort_keys=True) + "\n").encode("utf-8")
        zf.writestr("grader/scoring_keys.jsonl", scoring_key_body)
        zf.writestr(
            "grader/scoring_key_manifest.json",
            json.dumps(
                {
                    "schema_id": "trialagentbench.scoring_key_manifest/v1",
                    "release_id": "fixture_release",
                    "specification_sha256": "a" * 64,
                    "scoring_keys_sha256": hashlib.sha256(scoring_key_body).hexdigest(),
                    "item_ids": ["TASK001"],
                }
            ),
        )
    protocol: dict[str, object] = {
        "schema_id": "trial_analysis_protocol_summary_v1",
        "design_family": design_family,
        "primary_population": "ITT",
        "primary_endpoint_term": "All-cause death",
        "followup_horizon_dy": 180,
    }
    if protocol_extra:
        protocol.update(protocol_extra)
    with ZipFile(public_zip, "w") as zf:
        methods = [
            TrialEvalParticipantMethodV1(
                method_id=selected_method_id,
                estimator_family=selected_method_family,
                objective="estimation",
                result_kind="identification_bound" if is_bounds else "numeric_point",
                effect_scale="risk_difference_tau",
                design_modifiers=required_modifiers,
                uncertainty_method_id="fixture_uncertainty",
                description="Fixture participant method.",
            )
        ]
        if include_limitation_route:
            methods.append(
                TrialEvalParticipantMethodV1(
                    method_id="qualified_limitation_or_abstention",
                    estimator_family="qualified_nonidentification",
                    objective="estimation",
                    result_kind="limitation",
                    effect_scale="risk_difference_tau",
                    uncertainty_method_id="not_applicable",
                    possible_diagnostic_ids=limitation_possible_diagnostic_ids,
                    supported_conclusion_codes=(
                        "point_not_identified_due_to_censoring_or_support_failure",
                        "point_not_identified_due_to_endpoint_validation_failure",
                    ),
                    description="Qualified nonidentification response.",
                )
            )
        zf.writestr(
            "diagnostic_dictionary.json",
            participant_diagnostic_dictionary_v1().model_dump_json(),
        )
        zf.writestr(
            "method_dictionary.json",
            TrialEvalParticipantMethodDictionaryV1(methods=tuple(methods)).model_dump_json(),
        )
        zf.writestr("items/TASK001/README.md", "Participant item\n")
        zf.writestr("items/TASK001/data_dictionary.json", "{}")
        zf.writestr("items/TASK001/study_brief.md", "Study brief\n")
        zf.writestr(
            "items/TASK001/task.json",
            json.dumps(
                {
                    "schema_id": "trial_analysis_task_v1",
                    "task_id": "TASK001",
                    "design_subtype": design_subtype,
                    "primary_endpoint_id": "death",
                    "primary_estimand_id": "itt_treatment_policy",
                    "primary_effect_scale": "risk_difference_tau",
                    "estimand_mode": "fixed_declared_estimand",
                    "primary_effect_scale_options": ["risk_difference_tau"],
                    "bounded_deviation_delta_options": list(bounded_deviation_delta_options),
                    "primary_result_unit": "probability_difference",
                    "primary_population_id": "ITT",
                    "primary_intercurrent_event_strategy_ids": ["treatment_policy"],
                    "primary_tau_dy": 180,
                    "primary_control_arm_id": "control",
                    "primary_treated_arm_id": "treated",
                    "primary_paramcd": "DEATH",
                }
            ),
        )
        zf.writestr("items/TASK001/protocol_summary.json", json.dumps(protocol))
        zf.writestr("items/TASK001/submission_contract.json", json.dumps({"schema_id": "fixture_output_contract"}))
        if include_endpoint_definition:
            zf.writestr("items/TASK001/endpoint_definition.json", json.dumps({"schema_id": "endpoint_v1"}))
        zf.writestr("items/TASK001/intercurrent_event_strategy.json", json.dumps({"schema_id": "ice_v1"}))
        if selected_preparation == "analysis_ready":
            zf.writestr("items/TASK001/analysis_population_guidance.json", "{}")
        if selected_specification == "locked_sap" or include_sap_in_protocol_only:
            sap_payload = _locked_sap_payload(
                estimator_family=sap_method_family_override or selected_method_family,
                required_modifiers=required_modifiers,
            )
            if corrupt_sap:
                sap_payload["checksum"] = "0" * 64
            zf.writestr("items/TASK001/analysis_plan.json", json.dumps(sap_payload))
            zf.writestr("items/TASK001/analysis_tasks.md", "Analysis tasks\n")
        if selected_preparation == "analysis_ready" and include_analysis_tables:
            zf.writestr("items/TASK001/data/ADSL.parquet", b"fixture")
            zf.writestr("items/TASK001/data/ADTTE.parquet", b"fixture")
        if selected_preparation != "analysis_ready":
            zf.writestr("items/TASK001/reconstruction_task.json", json.dumps({"allowed_sources": []}))
            zf.writestr("items/TASK001/data/raw/subjects.parquet", b"fixture")
        if selected_preparation == "raw_domains_declared_defect":
            zf.writestr("items/TASK001/data_integrity_policy.json", json.dumps({"entries": ["fixture"]}))
        for member in extra_members:
            zf.writestr(member, b"fixture")
    return public_zip, evaluator_zip
