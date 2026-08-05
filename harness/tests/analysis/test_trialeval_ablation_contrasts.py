"""Tests for paired, cluster-aware TrialEval ablation contrasts."""

from __future__ import annotations

import pytest

from trialagentbench_harness.analysis.experiments.trialeval_inference import (
    analysis_specification_contrasts_v1,
    factorial_ablation_arm_summaries_v1,
    factorial_ablation_contrasts_v1,
    factorial_observable_contrasts_v1,
    targeted_control_contrasts_v1,
    trialeval_route_multiplicity_summaries_v1,
    trialeval_tolerance_sensitivity_summaries_v1,
)
from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationAnalysisConfigV1,
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalAblationGradeRowV1,
    TrialEvalAblationObservableRowV1,
)


def _config(
    *metrics: str,
    design: str = "factorial_interface",
) -> TrialEvalAblationAnalysisConfigV1:
    if not metrics:
        raise ValueError("Test analysis config requires at least one metric.")
    primary_estimand = {
        "metric": metrics[0],
        "contrast_id": ("P2-P1" if design == "factorial_interface" else "targeted_vs_neutral"),
        "analysis_specification": "locked_sap",
        "prompt_condition": (None if design == "factorial_interface" else "targeted_covariate_structure"),
        "applicability": None if design == "factorial_interface" else "applicable",
    }
    return TrialEvalAblationAnalysisConfigV1.model_validate(
        {
            "design": design,
            "execution_scope": "pilot",
            "experiment_design_sha256": "d" * 64,
            "primary_estimand": primary_estimand,
            "supporting_metrics": metrics[1:],
            "confidence_level": 0.90,
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 71,
            "min_base_trial_clusters": 2,
            "min_decoding_replicates": 2,
        }
    )


def _row(
    *,
    task: str,
    cluster: str,
    condition: str,
    assistance: str = "output_contract_only",
    interface: str = "structured",
    score: float,
    applicability: str | None = None,
    replicate: str = "seed-1",
    specification: str = "locked_sap",
    planning_power_shortfall: float | None = None,
    normalization_source: str | None = None,
    route_match: bool | None = None,
    result_match: bool | None = None,
    credit_eligible_route_count: int = 1,
    numeric_tolerance_ratio: float | None = None,
) -> TrialEvalAblationGradeRowV1:
    context_tier = "C1" if specification == "locked_sap" else "C2"
    matched_route = score > 0.0 if route_match is None else route_match
    matched_result = score > 0.0 if result_match is None else result_match
    return TrialEvalAblationGradeRowV1.model_validate(
        {
            "assignment_id": f"{task}-{replicate}-{assistance}-{condition}-{interface}",
            "task_id": task,
            "base_trial_id": cluster,
            "regime_cell_id": "family-1",
            "evaluation_series_id": "randomized",
            "design_tier": "D1",
            "design_subtype": "individual_randomized",
            "assumption_tier": "A1",
            "context_tier": context_tier,
            "data_preparation": "analysis_ready",
            "analysis_specification": specification,
            "model_id": "model-a",
            "replicate_id": replicate,
            "procedure_assistance": assistance,
            "prompt_condition": condition,
            "submission_interface": interface,
            "normalization_source": normalization_source
            or ("direct_structured" if interface == "structured" else "manual_masked"),
            "normalization_status": "not_applicable" if interface == "structured" else "complete",
            "primary_failure_code": None if score > 0.0 else "numeric_result_outside_tolerance",
            "targeted_applicability": applicability,
            "usable_primary": matched_route or score > 0.0,
            "route_match": matched_route,
            "obligations_met": matched_route,
            "credit_eligible_route_count": credit_eligible_route_count,
            "numeric_result_available": matched_route,
            "result_match": matched_result,
            "numeric_absolute_error": (numeric_tolerance_ratio if numeric_tolerance_ratio is not None else None),
            "numeric_tolerance_ratio": numeric_tolerance_ratio,
            "primary_analysis_conforms": score > 0.0,
            "planning_applicable": planning_power_shortfall is not None,
            "planning_valid": True if planning_power_shortfall is not None else None,
            "planning_usable_with_primary": score > 0.0 if planning_power_shortfall is not None else None,
            "planning_achieved_power": (
                0.8 - planning_power_shortfall if planning_power_shortfall is not None else None
            ),
            "planning_power_shortfall": planning_power_shortfall,
            "planning_underpowered": (
                planning_power_shortfall > 0.0 if planning_power_shortfall is not None else None
            ),
            "planning_excess_participants": 0 if planning_power_shortfall is not None else None,
            "planning_participant_shortage": (
                int(round(100 * planning_power_shortfall)) if planning_power_shortfall is not None else None
            ),
        }
    )


def test_analysis_specification_effect_pairs_base_trial_views_at_each_assistance_dose() -> None:
    rows = []
    for trial_index in range(2):
        for replicate in ("seed-1", "seed-2"):
            for assistance in ("output_contract_only", "unordered_checklist", "ordered_sop"):
                for interface in ("structured", "narrative"):
                    for specification, score in (("protocol_only", 0.0), ("locked_sap", 1.0)):
                        rows.append(
                            _row(
                                task=f"TASK{trial_index}{specification[0]}{assistance[0]}{interface[0]}".upper(),
                                cluster=f"trial-{trial_index}",
                                condition="neutral",
                                assistance=assistance,
                                interface=interface,
                                score=score,
                                replicate=replicate,
                                specification=specification,
                            )
                        )

    contrasts = analysis_specification_contrasts_v1(
        rows=tuple(rows),
        config=_config("primary_analysis_conforms"),
    )
    assert len(contrasts) == 6
    assert all(row.estimate == pytest.approx(1.0) for row in contrasts)


def test_factorial_analysis_estimates_distinct_prompt_interface_and_interaction_effects() -> None:
    rows = []
    for index in range(2):
        task = f"TASK100{index}"
        cluster = f"trial-{index}"
        for replicate in ("seed-1", "seed-2"):
            rows.extend(
                (
                    _row(
                        task=task,
                        cluster=cluster,
                        condition="neutral",
                        assistance="output_contract_only",
                        score=0.0,
                        replicate=replicate,
                    ),
                    _row(
                        task=task,
                        cluster=cluster,
                        condition="neutral",
                        assistance="output_contract_only",
                        interface="narrative",
                        score=0.0,
                        replicate=replicate,
                    ),
                    _row(
                        task=task,
                        cluster=cluster,
                        condition="neutral",
                        assistance="unordered_checklist",
                        score=0.4,
                        replicate=replicate,
                    ),
                    _row(
                        task=task,
                        cluster=cluster,
                        condition="neutral",
                        assistance="unordered_checklist",
                        interface="narrative",
                        score=0.3,
                        replicate=replicate,
                    ),
                    _row(
                        task=task,
                        cluster=cluster,
                        condition="neutral",
                        assistance="ordered_sop",
                        score=1.0,
                        replicate=replicate,
                    ),
                    _row(
                        task=task,
                        cluster=cluster,
                        condition="neutral",
                        assistance="ordered_sop",
                        interface="narrative",
                        score=0.7,
                        replicate=replicate,
                    ),
                    _row(
                        task=task,
                        cluster=cluster,
                        condition="neutral",
                        assistance="output_contract_only",
                        interface="narrative",
                        score=0.0,
                        replicate=replicate,
                        normalization_source="automated_importer",
                    ),
                    _row(
                        task=task,
                        cluster=cluster,
                        condition="neutral",
                        assistance="unordered_checklist",
                        interface="narrative",
                        score=0.2,
                        replicate=replicate,
                        normalization_source="automated_importer",
                    ),
                    _row(
                        task=task,
                        cluster=cluster,
                        condition="neutral",
                        assistance="ordered_sop",
                        interface="narrative",
                        score=0.5,
                        replicate=replicate,
                        normalization_source="automated_importer",
                    ),
                )
            )

    contrasts = factorial_ablation_contrasts_v1(
        rows=tuple(rows),
        config=_config("primary_analysis_conforms"),
    )
    arm_summaries = factorial_ablation_arm_summaries_v1(
        rows=tuple(rows),
        config=_config("primary_analysis_conforms"),
    )
    output_contract_structured = next(
        row
        for row in arm_summaries
        if row.procedure_assistance == "output_contract_only" and row.submission_interface == "structured"
    )
    assert output_contract_structured.estimate == pytest.approx(0.0)
    assert output_contract_structured.n_assignments == 4
    assert output_contract_structured.n_base_trial_clusters == 2
    assert output_contract_structured.n_decoding_replicates == 2
    estimates = {row.contrast_id: row.estimate for row in contrasts}
    assert estimates == pytest.approx(
        {
            "P1-P0": 1.0,
            "P2-P1": 0.0,
            "P2-P0": 1.0,
            "R1-R0": 0.0,
            "O1-O0": 0.0,
            "I.interface_by_P1-P0": 0.0,
            "I.interface_by_P2-P1": 0.0,
        }
    )


def test_factorial_analysis_rejects_partial_automated_normalization() -> None:
    rows = []
    for index in range(2):
        task = f"TASK200{index}"
        for replicate in ("seed-1", "seed-2"):
            for assistance in ("output_contract_only", "unordered_checklist", "ordered_sop"):
                rows.extend(
                    (
                        _row(
                            task=task,
                            cluster=f"trial-{index}",
                            condition="neutral",
                            assistance=assistance,
                            score=0.5,
                            replicate=replicate,
                        ),
                        _row(
                            task=task,
                            cluster=f"trial-{index}",
                            condition="neutral",
                            assistance=assistance,
                            interface="narrative",
                            score=0.5,
                            replicate=replicate,
                        ),
                    )
                )
    rows.append(
        _row(
            task="TASK2000",
            cluster="trial-0",
            condition="neutral",
            interface="narrative",
            score=0.5,
            normalization_source="automated_importer",
        )
    )

    with pytest.raises(ValueError, match="Automated narrative normalization is incomplete"):
        factorial_ablation_contrasts_v1(
            rows=tuple(rows),
            config=_config("primary_analysis_conforms"),
        )


def test_factorial_analysis_never_pools_protocol_and_sap_specifications() -> None:
    rows = []
    effects = {"protocol_only": 0.0, "locked_sap": 1.0}
    for specification, sop_score in effects.items():
        for index in range(2):
            task = f"TASK{specification[0].upper()}{index}01"
            for replicate in ("seed-1", "seed-2"):
                for assistance, score in (
                    ("output_contract_only", 0.0),
                    ("unordered_checklist", 0.0),
                    ("ordered_sop", sop_score),
                ):
                    for interface in ("structured", "narrative"):
                        rows.append(
                            _row(
                                task=task,
                                cluster=f"trial-{specification}-{index}",
                                condition="neutral",
                                assistance=assistance,
                                interface=interface,
                                score=score,
                                replicate=replicate,
                                specification=specification,
                            )
                        )

    contrasts = factorial_ablation_contrasts_v1(
        rows=tuple(rows),
        config=_config("primary_analysis_conforms"),
    )
    incremental = {row.analysis_specification: row.estimate for row in contrasts if row.contrast_id == "P2-P1"}
    assert incremental == pytest.approx(effects)


def _observable_rows() -> tuple[TrialEvalAblationObservableRowV1, ...]:
    rows = []
    for task_index in range(2):
        for replicate in ("seed-1", "seed-2"):
            for specification in ("protocol_only", "locked_sap"):
                task_id = f"TASKOBS{task_index}{'P' if specification == 'protocol_only' else 'S'}"
                for assistance, dose in (
                    ("output_contract_only", 0),
                    ("unordered_checklist", 1),
                    ("ordered_sop", 3),
                ):
                    for interface in ("structured", "narrative"):
                        rows.append(
                            TrialEvalAblationObservableRowV1.model_validate(
                                {
                                    "assignment_id": (
                                        f"{task_id}-{replicate}-{specification}-{assistance}-{interface}"
                                    ),
                                    "task_id": task_id,
                                    "context_tier": ("C1" if specification == "locked_sap" else "C2"),
                                    "data_preparation": "analysis_ready",
                                    "analysis_specification": specification,
                                    "model_id": "model-a",
                                    "replicate_id": replicate,
                                    "procedure_assistance": assistance,
                                    "prompt_condition": "neutral",
                                    "submission_interface": interface,
                                    "answer_submitted": True,
                                    "public_data_inspected": dose > 0,
                                    "code_executed_successfully": dose > 0,
                                    "assistant_turns": dose,
                                    "tool_calls": dose,
                                    "file_inspections": dose,
                                    "code_executions": dose,
                                    "failed_code_executions": 0,
                                    "truncated_outputs": 0,
                                    "events_until_first_data_inspection": 5 - dose,
                                    "events_until_first_successful_code_execution": 5 - dose,
                                    "events_until_submission": 5,
                                    "execution_elapsed_seconds": float(dose),
                                    "provider_response_count": dose,
                                    "provider_responses_with_usage": dose,
                                    "prompt_tokens": dose * 10,
                                    "completion_tokens": dose * 5,
                                    "total_tokens": dose * 15,
                                    "provider_elapsed_seconds": float(dose),
                                    "declared_diagnostics": dose if interface == "structured" else None,
                                    "declared_sensitivity_analyses": (dose if interface == "structured" else None),
                                    "declared_uncertainty": (dose > 0 if interface == "structured" else None),
                                }
                            )
                        )
    return tuple(rows)


def _observable_labels() -> TrialEvalAblationEvaluatorLabelsV1:
    return TrialEvalAblationEvaluatorLabelsV1.model_validate(
        {
            "evaluator_release_sha256": "e" * 64,
            "task_identities": [
                {
                    "task_id": f"TASKOBS{index}{suffix}",
                    "base_trial_id": f"trial-{index}",
                    "regime_cell_id": "family-1",
                    "evaluation_series_id": "randomized",
                    "design_tier": "D1",
                    "design_subtype": "individual_randomized",
                    "assumption_tier": "A1",
                    "context_tier": "C1" if specification == "locked_sap" else "C2",
                    "data_preparation": "analysis_ready",
                    "analysis_specification": specification,
                }
                for index in range(2)
                for specification, suffix in (("protocol_only", "P"), ("locked_sap", "S"))
            ],
        }
    )


def test_observable_assistance_contrasts_use_exact_randomized_blocks() -> None:
    contrasts = factorial_observable_contrasts_v1(
        rows=_observable_rows(),
        evaluator_labels=_observable_labels(),
        config=_config("primary_analysis_conforms"),
    )

    sop_turns = [
        row
        for row in contrasts
        if row.metric == "assistant_turns"
        and row.contrast_id == "ordered_sop_minus_unordered_checklist"
        and row.analysis_specification == "protocol_only"
        and row.submission_interface == "structured"
    ]
    assert len(sop_turns) == 1
    assert sop_turns[0].estimate == pytest.approx(2.0)
    assert sop_turns[0].n_blocks == 4
    assert sop_turns[0].n_base_trial_clusters == 2
    assert sop_turns[0].n_decoding_replicates == 2
    total_sop_turns = [
        row
        for row in contrasts
        if row.metric == "assistant_turns"
        and row.contrast_id == "ordered_sop_minus_output_contract_only"
        and row.analysis_specification == "protocol_only"
        and row.submission_interface == "structured"
    ]
    assert len(total_sop_turns) == 1
    assert total_sop_turns[0].estimate == pytest.approx(3.0)
    assert not any(row.metric.startswith("declared_") and row.submission_interface == "narrative" for row in contrasts)


def test_observable_assistance_contrasts_reject_incomplete_condition_blocks() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        factorial_observable_contrasts_v1(
            rows=_observable_rows()[:-1],
            evaluator_labels=_observable_labels(),
            config=_config("primary_analysis_conforms"),
        )


def test_narrative_observable_rows_cannot_claim_untranscribed_diagnostics() -> None:
    payload = _observable_rows()[1].model_dump(mode="json")
    assert payload["submission_interface"] == "narrative"
    payload["declared_diagnostics"] = 1
    with pytest.raises(ValueError, match="masked normalization"):
        TrialEvalAblationObservableRowV1.model_validate(payload)


def test_targeted_analysis_reports_inducibility_specificity_and_harm_separately() -> None:
    rows = []
    for applicability_index, applicability in enumerate(("applicable", "mismatched", "inapplicable")):
        for cluster_index in range(2):
            task = f"TASK{applicability_index + 1}{cluster_index + 1}01"
            cluster = f"trial-{applicability}-{cluster_index}"
            target_score = 1.0 if applicability == "applicable" else 0.0
            for replicate in ("seed-1", "seed-2"):
                rows.extend(
                    (
                        _row(
                            task=task,
                            cluster=cluster,
                            condition="neutral",
                            score=0.0,
                            replicate=replicate,
                        ),
                        _row(
                            task=task,
                            cluster=cluster,
                            condition="placebo_deliberation",
                            score=0.0,
                            replicate=replicate,
                        ),
                        _row(
                            task=task,
                            cluster=cluster,
                            condition="targeted_covariate_structure",
                            score=target_score,
                            applicability=applicability,
                            replicate=replicate,
                        ),
                    )
                )

    contrasts = targeted_control_contrasts_v1(
        rows=tuple(rows),
        config=_config(
            "primary_analysis_conforms",
            "usable_primary",
            design="targeted_control",
        ),
    )
    indexed = {(row.metric, row.applicability, row.contrast_id): row.estimate for row in contrasts}
    assert indexed[("primary_analysis_conforms", "applicable", "targeted_vs_neutral")] == 1.0
    assert indexed[("primary_analysis_conforms", "inapplicable", "targeted_vs_placebo_deliberation")] == 0.0
    assert indexed[("usable_primary", "inapplicable", "targeted_vs_neutral")] == 0.0
    specificity = {(row.metric, row.contrast_id): row.estimate for row in contrasts if row.applicability is None}
    assert specificity[("primary_analysis_conforms", "specificity_vs_mismatched_adjusted_for_neutral")] == 1.0
    assert (
        specificity[("primary_analysis_conforms", "specificity_vs_inapplicable_adjusted_for_placebo_deliberation")]
        == 1.0
    )


def test_factorial_analysis_fails_instead_of_dropping_missing_cells() -> None:
    rows = (
        _row(
            task="TASK1001",
            cluster="trial-1",
            condition="neutral",
            assistance="output_contract_only",
            score=0.0,
        ),
        _row(
            task="TASK1001",
            cluster="trial-1",
            condition="neutral",
            assistance="ordered_sop",
            score=1.0,
        ),
    )
    with pytest.raises(ValueError, match="incomplete"):
        factorial_ablation_contrasts_v1(rows=rows, config=_config("primary_analysis_conforms"))


def test_factorial_planning_contrasts_use_only_prospectively_applicable_tasks() -> None:
    rows = []
    shortfall_by_assistance = {
        "output_contract_only": 0.2,
        "unordered_checklist": 0.1,
        "ordered_sop": 0.0,
    }
    for index in range(3):
        task = f"TASK20{index}1"
        cluster = f"trial-{index}"
        applicable = index < 2
        for replicate in ("seed-1", "seed-2"):
            for assistance, shortfall in shortfall_by_assistance.items():
                for interface in ("structured", "narrative"):
                    rows.append(
                        _row(
                            task=task,
                            cluster=cluster,
                            condition="neutral",
                            assistance=assistance,
                            interface=interface,
                            score=1.0,
                            replicate=replicate,
                            planning_power_shortfall=shortfall if applicable else None,
                        )
                    )

    contrasts = factorial_ablation_contrasts_v1(
        rows=tuple(rows),
        config=_config("planning_power_shortfall"),
    )
    indexed = {row.contrast_id: row for row in contrasts}
    assert indexed["P1-P0"].estimate == pytest.approx(-0.1)
    assert indexed["P2-P1"].estimate == pytest.approx(-0.1)
    assert indexed["P1-P0"].n_base_trial_clusters == 2
    assert indexed["P2-P1"].analysis_population == "planning_consequence_evaluable_assignments"


def test_route_multiplicity_summary_preserves_distinct_denominators() -> None:
    rows = (
        _row(
            task="TASK3001",
            cluster="trial-1",
            condition="neutral",
            score=1.0,
            credit_eligible_route_count=1,
        ),
        _row(
            task="TASK3002",
            cluster="trial-2",
            condition="neutral",
            score=0.0,
            credit_eligible_route_count=3,
        ),
    )

    summaries = trialeval_route_multiplicity_summaries_v1(rows)
    by_multiplicity = {row.route_multiplicity: row for row in summaries}
    assert by_multiplicity["single_route"].n_assignments == 1
    assert by_multiplicity["single_route"].accepted_rate == 1.0
    assert by_multiplicity["plural_route"].n_assignments == 1
    assert by_multiplicity["plural_route"].accepted_rate == 0.0


def test_tolerance_sensitivity_uses_only_matched_numeric_comparisons() -> None:
    rows = (
        _row(
            task="TASK4001",
            cluster="trial-1",
            condition="neutral",
            score=1.0,
            numeric_tolerance_ratio=0.4,
        ),
        _row(
            task="TASK4002",
            cluster="trial-2",
            condition="neutral",
            score=0.0,
            route_match=True,
            result_match=False,
            numeric_tolerance_ratio=1.5,
        ),
        _row(
            task="TASK4003",
            cluster="trial-3",
            condition="neutral",
            score=0.0,
        ),
    )

    summaries = trialeval_tolerance_sensitivity_summaries_v1(rows)
    by_policy = {row.tolerance_policy: row for row in summaries}
    assert by_policy["half_qualified_envelope"].n_matched_numeric_comparisons == 2
    assert by_policy["half_qualified_envelope"].within_envelope_rate == 0.5
    assert by_policy["qualified_envelope"].within_envelope_rate == 0.5
    assert by_policy["double_qualified_envelope"].within_envelope_rate == 1.0
