"""Tests for denominator-preserving TrialDev metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialagentbench_harness.contracts.trialdev.metrics import (
    TRIALDEV_CAPABILITY_CHECKS_V1,
    TRIALDEV_CAPABILITY_IDS_V1,
    TRIALDEV_CHECKPOINT_INVENTORY_V1,
    TRIALDEV_REQUIRED_LANES_V1,
    TRIALDEV_TERMINAL_LANES_V1,
    TrialDevAssessmentPortfolioV1,
    TrialDevCapabilityAssessmentV1,
    TrialDevCapabilityCheckV1,
    TrialDevCheckpointAssessmentV1,
    TrialDevLaneAssessmentV1,
    TrialDevMetricPortfolioV1,
    TrialDevProgrammeAssessmentV1,
    TrialDevSecondaryOutcomesV1,
)
from trialagentbench_harness.contracts.trialdev.programme import TrialDevCheckpointOutcomeV1
from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TrialDevScientificAssessmentV1,
    TrialDevScientificEnvelopeV1,
)
from trialagentbench_harness.io.json import read_json_model, write_json_model
from trialagentbench_harness.tools.analyse_trialdev_metrics import main
from trialagentbench_harness.trialdev.metrics import (
    compare_trialdev_conditions_v1,
    select_trialdev_calibration_v1,
    summarize_trialdev_metrics_v1,
)


def _capabilities(*, failed: str | None = None) -> tuple[TrialDevCapabilityAssessmentV1, ...]:
    return tuple(
        TrialDevCapabilityAssessmentV1(
            capability_id=capability_id,
            outcome="failed" if capability_id == failed else "passed",
            checks=tuple(
                TrialDevCapabilityCheckV1(
                    check_id=check_id,
                    passed=capability_id != failed,
                    source_record_sha256="a" * 64,
                )
                for check_id in TRIALDEV_CAPABILITY_CHECKS_V1[capability_id]
            ),
        )
        for capability_id in TRIALDEV_CAPABILITY_IDS_V1
    )


def _reached(
    checkpoint_id: str,
    *,
    lane_outcomes: tuple[tuple[str, str], ...],
    failed_capability: str | None = None,
    terminal: bool | None = None,
) -> TrialDevCheckpointAssessmentV1:
    return TrialDevCheckpointAssessmentV1.model_validate(
        {
            "checkpoint_id": checkpoint_id,
            "outcome": TrialDevCheckpointOutcomeV1(
                reach_status="reached",
                submission_status="accepted",
                analysis_status="estimable",
                execution_status="completed",
            ),
            "lanes": [
                TrialDevLaneAssessmentV1.model_validate(
                    {
                        "lane_id": lane_id,
                        "outcome": outcome,
                        "source_record_sha256": "b" * 64,
                    }
                )
                for lane_id, outcome in lane_outcomes
            ],
            "capabilities": _capabilities(failed=failed_capability),
            "scientific_assessment": TrialDevScientificAssessmentV1(
                execution="passed",
                question_estimand="passed",
                design="passed",
                assumptions="passed",
                analysis_classification="uncertainty_qualified",
                scientific_agreement="passed",
                exact_reproduction="passed",
                uncertainty="passed",
                action_admissibility="passed",
                evidential_support="passed",
                sequential_coherence="passed",
                resources="within_budget",
                scientific_envelope=TrialDevScientificEnvelopeV1(
                    envelope_id="test-decision-threshold",
                    basis="declared_decision_thresholds",
                    decision_thresholds=(0.1,),
                    exact_reproduction_tolerance=0.0005,
                ),
                decision_complete=True,
            ),
            "terminal_record_valid": terminal,
        }
    )


def _programme(
    *,
    model_id: str,
    condition_id: str | None = None,
    evaluation_unit_id: str,
    scenario_family_id: str,
    objective_variant_id: str = "benefit_risk",
    stream_id: str = "single_asset_development",
    procedure_assistance: str = "output_contract_only",
    maximum_turns_per_step: int = 90,
    execution_status: str = "completed",
    checkpoints: tuple[TrialDevCheckpointAssessmentV1, ...] | None = None,
    secondary: TrialDevSecondaryOutcomesV1 | None = None,
) -> TrialDevProgrammeAssessmentV1:
    default_checkpoints = (
        _reached(
            "observational_review",
            lane_outcomes=(("asset_nomination", "accepted"), ("phase_analysis", "accepted")),
        ),
        _reached(
            "early_safety_study",
            lane_outcomes=(("safety_gate", "accepted"), ("decision_action", "accepted")),
            terminal=True,
        ),
    )
    supplied = checkpoints or default_checkpoints
    completed_lanes = []
    for checkpoint in supplied:
        if checkpoint.status != "reached":
            completed_lanes.append(checkpoint)
            continue
        required = set(TRIALDEV_REQUIRED_LANES_V1[(stream_id, checkpoint.checkpoint_id)])
        if checkpoint.terminal_record_valid is not None:
            required.update(TRIALDEV_TERMINAL_LANES_V1)
        observed = {lane.lane_id: lane for lane in checkpoint.lanes}
        completed_lanes.append(
            checkpoint.model_copy(
                update={
                    "lanes": tuple(
                        observed.get(
                            lane_id,
                            TrialDevLaneAssessmentV1(
                                lane_id=lane_id,
                                outcome="accepted",
                                source_record_sha256="b" * 64,
                            ),
                        )
                        for lane_id in sorted(required)
                    )
                }
            )
        )
    supplied_by_id = {item.checkpoint_id: item for item in completed_lanes}
    complete_checkpoints = tuple(
        supplied_by_id.get(
            checkpoint_id,
            TrialDevCheckpointAssessmentV1(
                checkpoint_id=checkpoint_id,
                outcome=TrialDevCheckpointOutcomeV1(
                    reach_status="structural_nonreach",
                    submission_status="not_applicable",
                    analysis_status="not_applicable",
                    execution_status="not_applicable",
                ),
            ),
        )
        for checkpoint_id in TRIALDEV_CHECKPOINT_INVENTORY_V1[stream_id]
    )
    return TrialDevProgrammeAssessmentV1.model_validate(
        {
            "model_id": model_id,
            "condition_id": condition_id or model_id,
            "request_replicate_id": "request-1",
            "reasoning_effort": None,
            "procedure_assistance": procedure_assistance,
            "maximum_turns_per_step": maximum_turns_per_step,
            "maximum_submission_attempts": 3,
            "task_materialization_seed": 45560,
            "release_id": "release-v1",
            "run_id": f"run-{model_id}",
            "grader_sha256": "c" * 64,
            "evaluation_unit_id": evaluation_unit_id,
            "programme_id": f"{model_id}-{evaluation_unit_id}-{objective_variant_id}",
            "scenario_family_id": scenario_family_id,
            "objective_variant_id": objective_variant_id,
            "policy_variant_id": "policy-v1",
            "stream_id": stream_id,
            "execution_status": execution_status,
            "checkpoints": complete_checkpoints,
            "secondary_outcomes": secondary or TrialDevSecondaryOutcomesV1(),
        }
    )


def test_summary_preserves_every_denominator_and_strict_chain() -> None:
    complete = _programme(
        model_id="model-a",
        evaluation_unit_id="unit-1",
        scenario_family_id="scenario-1",
    )
    noncomplete = _programme(
        model_id="model-a",
        evaluation_unit_id="unit-2",
        scenario_family_id="scenario-2",
        execution_status="model_noncompletion",
        checkpoints=(
            _reached(
                "observational_review",
                lane_outcomes=(("asset_nomination", "accepted"), ("phase_analysis", "accepted")),
            ),
            TrialDevCheckpointAssessmentV1(
                checkpoint_id="early_safety_study",
                outcome=TrialDevCheckpointOutcomeV1(
                    reach_status="reached",
                    submission_status="missing",
                    analysis_status="missing",
                    execution_status="model_noncompletion",
                ),
                lanes=tuple(
                    TrialDevLaneAssessmentV1(
                        lane_id=lane_id,
                        outcome="missing",
                        source_record_sha256="b" * 64,
                    )
                    for lane_id in TRIALDEV_REQUIRED_LANES_V1[("single_asset_development", "early_safety_study")]
                ),
            ),
            TrialDevCheckpointAssessmentV1(
                checkpoint_id="proof_of_concept",
                outcome=TrialDevCheckpointOutcomeV1(
                    reach_status="structural_nonreach",
                    submission_status="not_applicable",
                    analysis_status="not_applicable",
                    execution_status="not_applicable",
                ),
            ),
        ),
    )

    result = summarize_trialdev_metrics_v1((complete, noncomplete), bootstrap_resamples=200)
    stream = result.streams[0]

    assert stream.denominators.model_dump() == {
        "programmes": 2,
        "scheduled": 4,
        "reached": 3,
        "structural_nonreach": 4,
        "submitted": 10,
        "accepted": 10,
        "invalid": 0,
        "missing": 4,
        "model_noncompletion": 1,
        "infrastructure_failure": 0,
    }
    assert stream.checkpoint_success.finite_estimate == 0.75
    assert stream.complete_chain_success.finite_estimate == 0.5
    assert stream.execution_completion.finite_estimate == 0.5
    assert stream.complete_chain_success.cluster_interval is not None
    assert stream.complete_chain_success.cluster_interval.cluster_count == 2
    scientific = {item.metric_id: item.estimate for item in stream.scientific_responsibilities}
    assert scientific["decision_complete"].denominator == 3
    assert scientific["decision_complete"].finite_estimate == 1.0
    assert scientific["exact_reproduction"].denominator == 3
    classifications = {item.classification: item for item in stream.analysis_classifications}
    assert classifications["uncertainty_qualified"].count == 3
    assert all(item.denominator == 3 for item in classifications.values())


def test_lane_acceptance_uses_submitted_denominator_and_keeps_missing_count() -> None:
    programme = _programme(
        model_id="model-a",
        evaluation_unit_id="unit-1",
        scenario_family_id="scenario-1",
        checkpoints=(
            _reached(
                "observational_review",
                lane_outcomes=(
                    ("asset_nomination", "accepted"),
                    ("phase_analysis", "invalid"),
                    ("final_recommendation", "missing"),
                ),
                terminal=False,
            ),
        ),
    )

    stream = summarize_trialdev_metrics_v1((programme,), bootstrap_resamples=20).streams[0]
    lanes = {item.metric_id: item.estimate for item in stream.lanes}

    assert stream.denominators.submitted == 3
    assert stream.denominators.accepted == 2
    assert stream.denominators.invalid == 1
    assert stream.denominators.missing == 1
    assert lanes["asset_nomination"].finite_estimate == 1.0
    assert lanes["phase_analysis"].finite_estimate == 0.0
    assert lanes["final_recommendation"].denominator == 0
    assert stream.checkpoint_success.finite_estimate == 0.0


def test_safety_and_resources_cannot_be_hidden_by_secondary_outcomes() -> None:
    programme = _programme(
        model_id="model-a",
        evaluation_unit_id="unit-1",
        scenario_family_id="scenario-1",
        stream_id="bounded_portfolio_reallocation",
        checkpoints=(
            _reached(
                "observational_review",
                lane_outcomes=(
                    ("phase_analysis", "accepted"),
                    ("portfolio_allocation", "accepted"),
                ),
            ),
            _reached(
                "joint_early_study_review",
                lane_outcomes=(
                    ("safety_gate", "invalid"),
                    ("portfolio_allocation", "accepted"),
                    ("resource_feasibility", "accepted"),
                ),
                failed_capability="safety",
                terminal=False,
            ),
        ),
        secondary=TrialDevSecondaryOutcomesV1(
            correction_count=2,
            execute_code_calls=3,
            inspect_data_calls=4,
            programme_resource_units=1,
            downstream_consequence=1.0,
            policy_value=100.0,
        ),
    )

    stream = summarize_trialdev_metrics_v1((programme,), bootstrap_resamples=20).streams[0]
    capabilities = {item.metric_id: item.estimate for item in stream.capabilities}

    assert capabilities["safety"].numerator == 1
    assert capabilities["safety"].denominator == 2
    assert stream.checkpoint_success.numerator == 1
    assert stream.checkpoint_success.denominator == 2
    assert stream.complete_chain_success.finite_estimate == 0.0
    assert stream.secondary.policy_value_mean == 100.0
    assert stream.secondary.correction_count_mean == 2.0
    assert stream.secondary.execute_code_calls_mean == 3.0
    assert stream.secondary.inspect_data_calls_mean == 4.0
    assert "overall_score" not in stream.model_dump(mode="json")


def test_checkpoint_and_chain_success_require_applicable_capabilities() -> None:
    programme = _programme(
        model_id="model-a",
        evaluation_unit_id="unit-1",
        scenario_family_id="scenario-1",
        checkpoints=(
            _reached(
                "observational_review",
                lane_outcomes=(("asset_nomination", "accepted"), ("phase_analysis", "accepted")),
                failed_capability="identification_and_uncertainty",
                terminal=True,
            ),
        ),
    )

    stream = summarize_trialdev_metrics_v1((programme,), bootstrap_resamples=20).streams[0]

    assert stream.checkpoint_success.finite_estimate == 0.0
    assert stream.complete_chain_success.finite_estimate == 0.0


def test_model_noncompletion_requires_the_exact_missing_lane_set() -> None:
    with pytest.raises(ValueError, match="exact canonical lane set"):
        _programme(
            model_id="model-a",
            evaluation_unit_id="unit-1",
            scenario_family_id="scenario-1",
            execution_status="model_noncompletion",
            checkpoints=(
                TrialDevCheckpointAssessmentV1(
                    checkpoint_id="observational_review",
                    outcome=TrialDevCheckpointOutcomeV1(
                        reach_status="reached",
                        submission_status="missing",
                        analysis_status="missing",
                        execution_status="model_noncompletion",
                    ),
                    lanes=(
                        TrialDevLaneAssessmentV1(
                            lane_id="asset_nomination",
                            outcome="missing",
                            source_record_sha256="b" * 64,
                        ),
                    ),
                ),
            ),
        )


def test_shared_objectives_are_clustered_at_scenario_family() -> None:
    programmes = tuple(
        _programme(
            model_id="model-a",
            evaluation_unit_id=f"{scenario}-{objective}",
            scenario_family_id=scenario,
            objective_variant_id=objective,
        )
        for scenario in ("scenario-1", "scenario-2")
        for objective in ("benefit_risk", "efficacy_priority")
    )

    stream = summarize_trialdev_metrics_v1(programmes, bootstrap_resamples=200).streams[0]

    assert stream.complete_chain_success.denominator == 4
    assert stream.complete_chain_success.cluster_interval is not None
    assert stream.complete_chain_success.cluster_interval.cluster_count == 2


def test_paired_model_comparison_requires_exact_views_and_clusters_scenarios() -> None:
    programmes = []
    for model_id in ("reference", "intervention"):
        for scenario in ("scenario-1", "scenario-2"):
            programmes.append(
                _programme(
                    model_id=model_id,
                    evaluation_unit_id=scenario,
                    scenario_family_id=scenario,
                )
            )
    comparison = compare_trialdev_conditions_v1(
        programmes,
        reference_condition_id="reference",
        intervention_condition_id="intervention",
        bootstrap_resamples=200,
    )
    metrics = {item.metric_id: item for item in comparison.streams[0].metrics}

    assert metrics["complete_chain_success"].pair_count == 2
    assert metrics["complete_chain_success"].paired_difference == 0.0
    assert metrics["complete_chain_success"].cluster_interval is not None
    assert metrics["complete_chain_success"].cluster_interval.cluster_count == 2

    with pytest.raises(ValueError, match="complete matched evaluation views"):
        compare_trialdev_conditions_v1(
            programmes[:-1],
            reference_condition_id="reference",
            intervention_condition_id="intervention",
            bootstrap_resamples=20,
        )


def test_paired_comparison_preserves_capability_missingness_after_noncompletion() -> None:
    completed = _programme(
        model_id="model-a",
        condition_id="reference",
        evaluation_unit_id="scenario-1",
        scenario_family_id="scenario-1",
    )
    noncompleted = _programme(
        model_id="model-a",
        condition_id="intervention",
        evaluation_unit_id="scenario-1",
        scenario_family_id="scenario-1",
        execution_status="model_noncompletion",
        checkpoints=(
            TrialDevCheckpointAssessmentV1(
                checkpoint_id="observational_review",
                outcome=TrialDevCheckpointOutcomeV1(
                    reach_status="reached",
                    submission_status="missing",
                    analysis_status="missing",
                    execution_status="model_noncompletion",
                ),
                lanes=tuple(
                    TrialDevLaneAssessmentV1(
                        lane_id=lane_id,
                        outcome="missing",
                        source_record_sha256="b" * 64,
                    )
                    for lane_id in TRIALDEV_REQUIRED_LANES_V1[("single_asset_development", "observational_review")]
                ),
            ),
        ),
    )

    comparison = compare_trialdev_conditions_v1(
        (completed, noncompleted),
        reference_condition_id="reference",
        intervention_condition_id="intervention",
        bootstrap_resamples=20,
    )
    metrics = {item.metric_id: item for item in comparison.streams[0].metrics}

    assert metrics["execution_completion"].pair_count == 1
    assert metrics["execution_completion"].paired_difference == -1.0
    assert "evidence_identification" not in metrics


def test_metric_contract_rejects_reached_checkpoint_without_complete_capabilities() -> None:
    with pytest.raises(ValueError, match="complete capability vector"):
        TrialDevCheckpointAssessmentV1(
            checkpoint_id="observational_review",
            outcome=TrialDevCheckpointOutcomeV1(
                reach_status="reached",
                submission_status="accepted",
                analysis_status="estimable",
                execution_status="completed",
            ),
            lanes=(
                TrialDevLaneAssessmentV1(
                    lane_id="asset_nomination",
                    outcome="accepted",
                    source_record_sha256="b" * 64,
                ),
            ),
            capabilities=(),
        )


def test_public_cli_writes_the_same_typed_summary_and_comparison(tmp_path: Path) -> None:
    programmes = tuple(
        _programme(
            model_id=model_id,
            evaluation_unit_id=scenario,
            scenario_family_id=scenario,
        )
        for model_id in ("reference", "intervention")
        for scenario in ("scenario-1", "scenario-2")
    )
    portfolio = TrialDevAssessmentPortfolioV1(programmes=programmes)
    input_path = tmp_path / "assessments.json"
    output_path = tmp_path / "metrics.json"
    comparison_path = tmp_path / "comparison.json"
    write_json_model(input_path, portfolio)

    assert (
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--reference-condition",
                "reference",
                "--intervention-condition",
                "intervention",
                "--comparison-output",
                str(comparison_path),
                "--bootstrap-resamples",
                "200",
            ]
        )
        == 0
    )
    observed = read_json_model(TrialDevMetricPortfolioV1, output_path)
    assert observed == summarize_trialdev_metrics_v1(programmes, bootstrap_resamples=200)
    assert comparison_path.is_file()


def test_public_cli_combines_distinct_condition_portfolios(tmp_path: Path) -> None:
    reference = TrialDevAssessmentPortfolioV1(
        programmes=tuple(
            _programme(
                model_id="model-a",
                condition_id="reference",
                evaluation_unit_id=scenario,
                scenario_family_id=scenario,
            )
            for scenario in ("scenario-1", "scenario-2")
        )
    )
    intervention = TrialDevAssessmentPortfolioV1(
        programmes=tuple(
            _programme(
                model_id="model-a",
                condition_id="intervention",
                evaluation_unit_id=scenario,
                scenario_family_id=scenario,
            )
            for scenario in ("scenario-1", "scenario-2")
        )
    )
    reference_path = tmp_path / "reference.json"
    intervention_path = tmp_path / "intervention.json"
    output_path = tmp_path / "metrics.json"
    comparison_path = tmp_path / "comparison.json"
    write_json_model(reference_path, reference)
    write_json_model(intervention_path, intervention)

    assert (
        main(
            [
                "--input",
                str(reference_path),
                str(intervention_path),
                "--output",
                str(output_path),
                "--reference-condition",
                "reference",
                "--intervention-condition",
                "intervention",
                "--comparison-output",
                str(comparison_path),
                "--bootstrap-resamples",
                "200",
            ]
        )
        == 0
    )
    observed = read_json_model(TrialDevMetricPortfolioV1, output_path)
    assert {item.condition_id for item in observed.streams} == {"reference", "intervention"}
    assert comparison_path.is_file()


def test_calibration_keeps_assisted_conditions_out_of_default_selection() -> None:
    programmes = tuple(
        _programme(
            model_id="model-a",
            condition_id=condition_id,
            evaluation_unit_id=scenario,
            scenario_family_id=scenario,
            procedure_assistance=assistance,
            maximum_turns_per_step=turns,
            secondary=TrialDevSecondaryOutcomesV1(
                elapsed_seconds=elapsed,
                provider_calls=2,
                agent_turns=agent_turns,
                correction_count=corrections,
                prompt_tokens=tokens,
                completion_tokens=10,
                provider_reported_usd=cost,
            ),
        )
        for condition_id, assistance, turns, corrections, agent_turns, elapsed, tokens, cost in (
            ("medium-p0-t90", "output_contract_only", 90, 2, 30, 20.0, 100, 0.02),
            ("medium-p2-t45", "ordered_sop", 45, 0, 12, 10.0, 50, 0.01),
        )
        for scenario in ("scenario-1", "scenario-2")
    )

    selection = select_trialdev_calibration_v1(
        programmes,
        condition_ids=("medium-p0-t90", "medium-p2-t45"),
    )

    assert selection.selected_condition_id == "medium-p0-t90"
    assert selection.pareto_condition_ids == ("medium-p0-t90",)
    assert all(not arm.dominated_by_condition_ids for arm in selection.arms)


def test_calibration_prefers_smaller_clean_turn_ceiling_when_quality_ties() -> None:
    programmes = tuple(
        _programme(
            model_id="model-a",
            condition_id=condition_id,
            evaluation_unit_id=scenario,
            scenario_family_id=scenario,
            procedure_assistance="output_contract_only",
            maximum_turns_per_step=turns,
            secondary=TrialDevSecondaryOutcomesV1(
                elapsed_seconds=elapsed,
                provider_calls=2,
                agent_turns=agent_turns,
                correction_count=2,
                prompt_tokens=tokens,
                completion_tokens=10,
                provider_reported_usd=cost,
            ),
        )
        for condition_id, turns, agent_turns, elapsed, tokens, cost in (
            ("medium-p0-t45", 45, 30, 30.0, 300, 0.03),
            ("medium-p0-t90", 90, 20, 20.0, 200, 0.02),
        )
        for scenario in ("scenario-1", "scenario-2")
    )

    selection = select_trialdev_calibration_v1(
        programmes,
        condition_ids=("medium-p0-t45", "medium-p0-t90"),
    )

    assert selection.selected_condition_id == "medium-p0-t45"
    assert selection.pareto_condition_ids == ("medium-p0-t45", "medium-p0-t90")
