"""End-to-end tests for the frozen TrialEval ablation analysis command."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from trialagentbench_harness.analysis.experiments import trialeval_ablation_cli
from trialagentbench_harness.analysis.experiments.resource_attainment import (
    trialeval_resource_attainment_v1,
)
from trialagentbench_harness.analysis.experiments.trialeval_ablation_cli import main
from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationAnalysisConfigV1,
    TrialEvalAblationAnalysisReportV1,
    TrialEvalAblationEndpointSetV1,
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalAblationObservableRowV1,
    TrialEvalAblationScheduleV1,
)
from trialagentbench_harness.io import read_json_model, sha256_path, write_json_model


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    (evaluator / "release.txt").write_text("frozen evaluator\n", encoding="utf-8")
    evaluator_hash = sha256_path(evaluator)
    endpoints = []
    assignments = []
    task_identities = []
    conditions = (
        ("output_contract_only", "structured", "direct_structured"),
        ("output_contract_only", "narrative", "manual_masked"),
        ("unordered_checklist", "structured", "direct_structured"),
        ("unordered_checklist", "narrative", "manual_masked"),
        ("ordered_sop", "structured", "direct_structured"),
        ("ordered_sop", "narrative", "manual_masked"),
    )
    for index in range(2):
        task_id = f"TASK100{index}"
        task_identities.append(
            {
                "task_id": task_id,
                "base_trial_id": f"trial-{index}",
                "regime_cell_id": "family-1",
                "evaluation_series_id": "randomized",
                "design_tier": "D1",
                "design_subtype": "individual_randomized",
                "assumption_tier": "A1",
                "context_tier": "C1",
                "data_preparation": "analysis_ready",
                "analysis_specification": "locked_sap",
            }
        )
        for replicate_id, decoding_seed in (("seed-1", 101), ("seed-2", 202)):
            for specification in ("locked_sap",):
                for assistance, interface, normalization in conditions:
                    assignment_id = f"a-{index}-{replicate_id}-{specification}-{assistance}-{interface}"
                    assignments.append(
                        {
                            "assignment_id": assignment_id,
                            "task_id": task_id,
                            "context_tier": "C1",
                            "data_preparation": "analysis_ready",
                            "analysis_specification": specification,
                            "analysis_surface_sha256": ("1" if specification == "protocol_only" else "2") * 64,
                            "replicate_id": replicate_id,
                            "decoding_seed": decoding_seed,
                            "procedure_assistance": assistance,
                            "prompt_condition": "neutral",
                            "submission_interface": interface,
                        }
                    )
                    endpoints.append(
                        {
                            "assignment_id": assignment_id,
                            "task_id": task_id,
                            "context_tier": "C1",
                            "data_preparation": "analysis_ready",
                            "analysis_specification": specification,
                            "model_id": "model-a",
                            "replicate_id": replicate_id,
                            "procedure_assistance": assistance,
                            "prompt_condition": "neutral",
                            "submission_interface": interface,
                            "normalization_source": normalization,
                            "normalization_status": ("not_applicable" if interface == "structured" else "complete"),
                            "primary_failure_code": (
                                None if assistance == "ordered_sop" else "missing_primary_submission"
                            ),
                            "usable_primary": assistance == "ordered_sop",
                            "route_match": assistance == "ordered_sop",
                            "obligations_met": assistance == "ordered_sop",
                            "credit_eligible_route_count": 1,
                            "numeric_result_available": assistance == "ordered_sop",
                            "result_match": assistance == "ordered_sop",
                            "primary_analysis_conforms": (1.0 if assistance == "ordered_sop" else 0.0),
                            "planning_applicable": False,
                        }
                    )
                    if interface == "narrative":
                        endpoints.append(
                            {
                                **endpoints[-1],
                                "normalization_source": "automated_importer",
                            }
                        )
    assignments.sort(key=lambda row: row["assignment_id"])
    random.Random(17).shuffle(assignments)
    config = TrialEvalAblationAnalysisConfigV1(
        design="factorial_interface",
        execution_scope="pilot",
        experiment_design_sha256="d" * 64,
        primary_estimand={
            "metric": "usable_primary",
            "contrast_id": "P2-P1",
            "analysis_specification": "locked_sap",
        },
        supporting_metrics=("primary_analysis_conforms",),
        confidence_level=0.9,
        bootstrap_resamples=1000,
        bootstrap_seed=9,
        min_base_trial_clusters=2,
        min_decoding_replicates=2,
    )
    schedule = TrialEvalAblationScheduleV1.model_validate(
        {
            "experiment_id": "experiment-1",
            "design": "factorial_interface",
            "execution_scope": "pilot",
            "experiment_design_sha256": "d" * 64,
            "participant_release_sha256": "p" * 64,
            "prompt_set_sha256": "q" * 64,
            "analysis_config_sha256": config.checksum,
            "randomization_seed": 17,
            "assignments": assignments,
        }
    )
    endpoint_set = TrialEvalAblationEndpointSetV1.model_validate(
        {
            "experiment_id": "experiment-1",
            "schedule_checksum": schedule.checksum,
            "evaluator_release_sha256": evaluator_hash,
            "scoring_implementation_sha256": "g" * 64,
            "endpoints": endpoints,
        }
    )
    evaluator_labels = TrialEvalAblationEvaluatorLabelsV1.model_validate(
        {
            "evaluator_release_sha256": evaluator_hash,
            "task_identities": task_identities,
            "labels": [],
        }
    )
    endpoint_path = tmp_path / "endpoints.json"
    labels_path = tmp_path / "labels.json"
    config_path = tmp_path / "config.json"
    schedule_path = tmp_path / "schedule.json"
    write_json_model(endpoint_path, endpoint_set)
    write_json_model(labels_path, evaluator_labels)
    write_json_model(config_path, config)
    write_json_model(schedule_path, schedule)
    return schedule_path, endpoint_path, labels_path, config_path, evaluator


def _observable_rows(schedule_path: Path) -> tuple[TrialEvalAblationObservableRowV1, ...]:
    schedule = read_json_model(TrialEvalAblationScheduleV1, schedule_path)
    return tuple(
        TrialEvalAblationObservableRowV1(
            assignment_id=assignment.assignment_id,
            task_id=assignment.task_id,
            context_tier=assignment.context_tier,
            data_preparation=assignment.data_preparation,
            analysis_specification=assignment.analysis_specification,
            model_id="model-a",
            replicate_id=assignment.replicate_id,
            procedure_assistance=assignment.procedure_assistance,
            prompt_condition=assignment.prompt_condition,
            submission_interface=assignment.submission_interface,
            answer_submitted=True,
            public_data_inspected=True,
            code_executed_successfully=True,
            assistant_turns=2,
            tool_calls=3,
            file_inspections=1,
            code_executions=1,
            failed_code_executions=0,
            truncated_outputs=0,
            events_until_first_data_inspection=2,
            events_until_first_successful_code_execution=3,
            events_until_submission=5,
            execution_elapsed_seconds=0.2,
            provider_response_count=2,
            provider_responses_with_usage=2,
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
            provider_elapsed_seconds=0.3,
            declared_diagnostics=(1 if assignment.submission_interface == "structured" else None),
            declared_sensitivity_analyses=(0 if assignment.submission_interface == "structured" else None),
            declared_uncertainty=(True if assignment.submission_interface == "structured" else None),
        )
        for assignment in schedule.assignments
    )


def test_ablation_analysis_cli_emits_checksummed_paired_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule_path, endpoint_path, labels_path, config_path, evaluator = _write_inputs(tmp_path)
    monkeypatch.setattr(
        trialeval_ablation_cli,
        "collect_trialeval_ablation_observables",
        lambda _: _observable_rows(schedule_path),
    )
    output = tmp_path / "analysis.json"
    assert (
        main(
            [
                "--endpoint-set",
                str(endpoint_path),
                "--schedule",
                str(schedule_path),
                "--evaluator-labels",
                str(labels_path),
                "--evaluator-release",
                str(evaluator),
                "--analysis-config",
                str(config_path),
                "--run-dir",
                str(tmp_path / "run"),
                "--design",
                "factorial_interface",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    report = read_json_model(TrialEvalAblationAnalysisReportV1, output)
    assert report.checksum is not None
    assert len(report.grade_rows) == 36
    assert len(report.observable_rows) == 24
    assert report.observable_contrasts
    assert report.resource_attainment
    assert report.route_multiplicity
    assert {row.route_multiplicity for row in report.route_multiplicity} == {"single_route"}
    assert {row.resource for row in report.resource_attainment} == {"turns", "tokens"}
    incomplete_usage = list(report.observable_rows)
    incomplete_usage[0] = incomplete_usage[0].model_copy(update={"provider_responses_with_usage": 0})
    with pytest.raises(ValueError, match="reported usage"):
        trialeval_resource_attainment_v1(
            grades=report.grade_rows,
            observables=tuple(incomplete_usage),
        )
    assert {row.contrast_id for row in report.contrasts} == {
        "P1-P0",
        "P2-P1",
        "P2-P0",
        "R1-R0",
        "O1-O0",
        "I.interface_by_P1-P0",
        "I.interface_by_P2-P1",
    }


def test_ablation_analysis_cli_rejects_evaluator_drift(tmp_path: Path) -> None:
    schedule_path, endpoint_path, labels_path, config_path, evaluator = _write_inputs(tmp_path)
    (evaluator / "release.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        main(
            [
                "--endpoint-set",
                str(endpoint_path),
                "--schedule",
                str(schedule_path),
                "--evaluator-labels",
                str(labels_path),
                "--evaluator-release",
                str(evaluator),
                "--analysis-config",
                str(config_path),
                "--run-dir",
                str(tmp_path / "run"),
                "--design",
                "factorial_interface",
                "--out",
                str(tmp_path / "analysis.json"),
            ]
        )


def test_ablation_analysis_cli_rejects_canary_inference(tmp_path: Path) -> None:
    schedule_path, endpoint_path, labels_path, config_path, evaluator = _write_inputs(tmp_path)
    config = read_json_model(TrialEvalAblationAnalysisConfigV1, config_path)
    payload = config.model_dump(mode="python", exclude={"checksum"})
    payload.update(
        execution_scope="canary",
        min_base_trial_clusters=1,
        min_decoding_replicates=1,
    )
    write_json_model(config_path, TrialEvalAblationAnalysisConfigV1.model_validate(payload))

    with pytest.raises(ValueError, match="qualify runtime execution only"):
        main(
            [
                "--endpoint-set",
                str(endpoint_path),
                "--schedule",
                str(schedule_path),
                "--evaluator-labels",
                str(labels_path),
                "--evaluator-release",
                str(evaluator),
                "--analysis-config",
                str(config_path),
                "--run-dir",
                str(tmp_path / "run"),
                "--design",
                "factorial_interface",
                "--out",
                str(tmp_path / "analysis.json"),
            ]
        )


def test_ablation_analysis_cli_rejects_partial_automated_normalization(tmp_path: Path) -> None:
    schedule_path, endpoint_path, labels_path, config_path, evaluator = _write_inputs(tmp_path)
    endpoint_set = read_json_model(TrialEvalAblationEndpointSetV1, endpoint_path)
    automated = tuple(row for row in endpoint_set.endpoints if row.normalization_source == "automated_importer")
    payload = endpoint_set.model_dump(mode="json", exclude={"checksum"})
    payload["endpoints"] = [row.model_dump(mode="json") for row in endpoint_set.endpoints if row != automated[0]]
    reduced = TrialEvalAblationEndpointSetV1.model_validate(payload)
    write_json_model(endpoint_path, reduced)

    with pytest.raises(ValueError, match="must cover every scheduled narrative assignment"):
        main(
            [
                "--endpoint-set",
                str(endpoint_path),
                "--schedule",
                str(schedule_path),
                "--evaluator-labels",
                str(labels_path),
                "--evaluator-release",
                str(evaluator),
                "--analysis-config",
                str(config_path),
                "--run-dir",
                str(tmp_path / "run"),
                "--design",
                "factorial_interface",
                "--out",
                str(tmp_path / "analysis.json"),
            ]
        )


def test_ablation_analysis_cli_rejects_post_schedule_analysis_change(tmp_path: Path) -> None:
    schedule_path, endpoint_path, labels_path, config_path, evaluator = _write_inputs(tmp_path)
    config = read_json_model(TrialEvalAblationAnalysisConfigV1, config_path)
    changed_payload = config.model_dump(mode="python")
    changed_payload["primary_estimand"] = {
        "metric": "primary_analysis_conforms",
        "contrast_id": "P2-P0",
        "analysis_specification": "locked_sap",
    }
    changed_payload["supporting_metrics"] = ("usable_primary",)
    changed_payload["checksum"] = None
    changed = TrialEvalAblationAnalysisConfigV1.model_validate(changed_payload)
    write_json_model(config_path, changed)

    with pytest.raises(ValueError, match="checksum frozen in the schedule"):
        main(
            [
                "--endpoint-set",
                str(endpoint_path),
                "--schedule",
                str(schedule_path),
                "--evaluator-labels",
                str(labels_path),
                "--evaluator-release",
                str(evaluator),
                "--analysis-config",
                str(config_path),
                "--run-dir",
                str(tmp_path / "run"),
                "--design",
                "factorial_interface",
                "--out",
                str(tmp_path / "analysis.json"),
            ]
        )
