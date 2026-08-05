"""Tests for the masked TrialEval narrative packet exporter."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)

from trialagentbench_harness.contracts.core.config import DecodingConfigV1
from trialagentbench_harness.contracts.core.runs import (
    RunCoverageV1,
    TrialEvalAblationItemResultV1,
    TrialEvalAblationRunConfigV1,
    TrialEvalAgentOutputV1,
)
from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalAblationScheduleV1,
    TrialEvalAblationTaskIdentityV1,
    TrialEvalNormalizerSampleUnitV1,
    TrialEvalNormalizerSampleV1,
)
from trialagentbench_harness.experiments import build_trialeval_normalizer_frame as frame_builder
from trialagentbench_harness.experiments import export_trialeval_narrative_packets as exporter
from trialagentbench_harness.experiments import trialeval_run_artifacts as run_artifacts
from trialagentbench_harness.experiments.export_trialeval_normalizer_sample_packets import (
    export_normalizer_sample_packets_v1,
)
from trialagentbench_harness.io import read_json, read_json_model, sha256_dir_digest, write_json_model
from trialagentbench_harness.util.provider_telemetry import summarize_provider_telemetry_v1


def _completed_run(root: Path) -> Path:
    participant = root / "participant"
    item = participant / "items" / "TASK1001"
    (item / "data").mkdir(parents=True)
    (participant / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_public_bundle_manifest/v1",
                "applied_baseline_profile_id": None,
                "applied_baseline_profile_sha256": None,
                "task_ids": ["TASK1001"],
                "task_evidence_factors": {
                    "TASK1001": {
                        "context_configuration": "C1",
                        "data_preparation": "analysis_ready",
                        "analysis_specification": "locked_sap",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (item / "task.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_task_v1",
                "task_id": "TASK1001",
                "design_subtype": "individual_randomized",
                "primary_endpoint_id": "endpoint",
                "primary_paramcd": "endpoint",
                "primary_estimand_id": "estimand",
                "primary_effect_scale": "risk_difference_tau",
                "estimand_mode": "fixed_declared_estimand",
                "primary_effect_scale_options": ["risk_difference_tau"],
                "primary_result_unit": "probability_difference",
                "primary_tau_dy": 365.0,
                "primary_population_id": "itt",
                "primary_intercurrent_event_strategy_ids": ["treatment_policy"],
                "primary_control_arm_id": "control",
                "primary_treated_arm_id": "treated",
            }
        ),
        encoding="utf-8",
    )
    (item / "submission_contract.json").write_text(
        json.dumps(minimal_participant_output_contract("TASK1001")),
        encoding="utf-8",
    )
    write_minimal_trialeval_release_dictionaries(participant)
    participant_release_sha256 = sha256_dir_digest(participant)
    run_dir = root / "run"
    run_dir.mkdir()
    assignments = [
        {
            "assignment_id": f"assignment-{specification}-{index}",
            "task_id": "TASK1001",
            "context_tier": "C1",
            "data_preparation": "analysis_ready",
            "analysis_specification": specification,
            "analysis_surface_sha256": ("1" if specification == "protocol_only" else "2") * 64,
            "replicate_id": "seed-1",
            "decoding_seed": 101,
            "procedure_assistance": assistance,
            "prompt_condition": "neutral",
            "submission_interface": interface,
        }
        for specification in ("locked_sap",)
        for index, (assistance, interface) in enumerate(
            (assistance, interface)
            for assistance in ("output_contract_only", "unordered_checklist", "ordered_sop")
            for interface in ("structured", "narrative")
        )
    ]
    assignments.sort(key=lambda assignment: assignment["assignment_id"])
    random.Random(9).shuffle(assignments)
    schedule = TrialEvalAblationScheduleV1.model_validate(
        {
            "experiment_id": "experiment-1",
            "design": "factorial_interface",
            "execution_scope": "pilot",
            "experiment_design_sha256": "d" * 64,
            "participant_release_sha256": participant_release_sha256,
            "prompt_set_sha256": "q" * 64,
            "analysis_config_sha256": "r" * 64,
            "randomization_seed": 9,
            "assignments": assignments,
        }
    )
    run_config = TrialEvalAblationRunConfigV1.create(
        timestamp_utc=datetime.now(UTC),
        experiment_id=schedule.experiment_id,
        schedule_checksum=str(schedule.checksum),
        participant_release_sha256=schedule.participant_release_sha256,
        prompt_set_sha256=schedule.prompt_set_sha256,
        scorer_source_sha256="s" * 64,
        agent_source_sha256="a" * 64,
        model="secret-model",
        max_context_characters=120_000,
        item_watchdog_seconds=3600,
        decoding=DecodingConfigV1(temperature=0.0, max_tokens=1024, send_temperature=True),
        routing={"provider": "openai", "request_timeout_seconds": 300.0},
        executor={
            "image_reference": "executor:test",
            "image_id": f"sha256:{'e' * 64}",
            "python_version": "3.12",
            "packages": [{"name": "pandas", "version": "2.2"}],
            "limits": {},
        },
        workers=1,
        n_assignments=len(assignments),
    )
    write_json_model(run_dir / "schedule.json", schedule)
    write_json_model(run_dir / "run_config.json", run_config)
    ids = tuple(assignment.assignment_id for assignment in schedule.assignments)
    coverage = RunCoverageV1(
        run_identity_sha256=run_config.run_identity_sha256,
        schedule_sha256=str(schedule.checksum),
        unit_ids=ids,
        completed_unit_ids=ids,
    )
    write_json_model(run_dir / "coverage.json", coverage)
    summarize_provider_telemetry_v1(run_root=run_dir, coverage=coverage)
    for assignment in schedule.assignments:
        report = (
            f"Frozen report for {assignment.assignment_id}.\n"
            if assignment.submission_interface == "narrative"
            else None
        )
        result = TrialEvalAblationItemResultV1(
            timestamp_utc=datetime.now(UTC),
            assignment=assignment,
            run_config=run_config,
            agent_output=TrialEvalAgentOutputV1(
                status="complete",
                turns_used=1,
                report=report,
                condition_provenance={
                    "procedure_assistance": assignment.procedure_assistance,
                    "analysis_specification": assignment.analysis_specification,
                    "analysis_surface_sha256": assignment.analysis_surface_sha256,
                    "prompt_condition": assignment.prompt_condition,
                    "submission_interface": assignment.submission_interface,
                    "max_turns": 1,
                    "prompt_set_sha256": schedule.prompt_set_sha256,
                    "rendered_system_prompt_sha256": "r" * 64,
                    "tool_schema_sha256": "t" * 64,
                    "response_contract_sha256": "u" * 64,
                },
            ),
        )
        write_json_model(run_dir / "assignments" / f"{assignment.assignment_id}.json", result)
    return run_dir


def _participant_root(run_dir: Path) -> Path:
    return run_dir.parent / "participant"


def test_export_emits_only_blinded_source_grounded_narrative_packets(tmp_path: Path) -> None:
    run_dir = _completed_run(tmp_path)
    output = exporter.export_narrative_transcription_packets(
        run_dir,
        _participant_root(run_dir),
        tmp_path / "packets",
    )

    packet_dirs = sorted(path for path in output.iterdir() if path.is_dir())
    assert [path.name for path in packet_dirs] == [
        "masked-narrative-0001",
        "masked-narrative-0002",
        "masked-narrative-0003",
    ]
    export_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert export_manifest["schema_id"] == "trialagentbench.trialeval_narrative_packet_set/v1"
    assert export_manifest["checksum"]
    assert len(export_manifest["packets"]) == 3
    expected_assignment_sources = {
        f"assignments/{assignment.assignment_id}.json"
        for assignment in read_json_model(
            TrialEvalAblationScheduleV1,
            run_dir / "schedule.json",
        ).assignments
    }
    assert set(export_manifest["source_files_sha256"]) == {
        *expected_assignment_sources,
        "coverage.json",
        "provider_telemetry_summary.json",
        "run_config.json",
        "schedule.json",
    }
    for packet_dir in packet_dirs:
        manifest = json.loads((packet_dir / "packet.json").read_text(encoding="utf-8"))
        report_bytes = (packet_dir / "frozen_report.txt").read_bytes()
        template = read_json(packet_dir / "transcription_template.json")
        assert manifest["participant_task_id"] == "TASK1001"
        assert manifest["report_state"] == "present"
        assert manifest["assignment_id"] in {
            assignment.assignment_id
            for assignment in read_json_model(
                TrialEvalAblationScheduleV1,
                run_dir / "schedule.json",
            ).assignments
            if assignment.submission_interface == "narrative"
        }
        assert manifest["report_sha256"] == hashlib.sha256(report_bytes).hexdigest()
        assert (
            manifest["participant_context_sha256"]
            == hashlib.sha256((packet_dir / "participant_context.json").read_bytes()).hexdigest()
        )
        assert template["report_sha256"] == manifest["report_sha256"]
        assert template["source"] == "manual_masked"
        assert template["source_identity"] == ""
        assert template["transcriber_identities"] == []
        assert template["transcription_disposition"] is None
        assert template["blinded_to_model_identity"] is True
        assert template["blinded_to_evaluator_reference"] is True

    exported_text = "".join(path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file())
    assert "secret-model" not in exported_text
    assert not any(path.name in {"assignment-0", "assignment-2"} for path in packet_dirs)
    with pytest.raises(FileExistsError, match="overwrite"):
        exporter.export_narrative_transcription_packets(run_dir, _participant_root(run_dir), output)


@pytest.mark.parametrize("failure", ("incomplete", "duplicate_id", "inside_run"))
def test_export_fails_closed_without_publishing_partial_output(tmp_path: Path, failure: str) -> None:
    run_dir = _completed_run(tmp_path)
    output = tmp_path / "packets"
    if failure == "incomplete":
        coverage = read_json_model(RunCoverageV1, run_dir / "coverage.json")
        write_json_model(
            run_dir / "coverage.json",
            coverage.model_copy(update={"completed_unit_ids": coverage.completed_unit_ids[:-1]}),
        )
    elif failure == "duplicate_id":
        schedule = read_json_model(TrialEvalAblationScheduleV1, run_dir / "schedule.json")
        narrative = next(row for row in schedule.assignments if row.submission_interface == "narrative")
        source = run_dir / "assignments" / f"{narrative.assignment_id}.json"
        (run_dir / "assignments" / "duplicate.json").write_bytes(source.read_bytes())
    else:
        output = run_dir / "packets"

    with pytest.raises((ValueError, FileNotFoundError)):
        exporter.export_narrative_transcription_packets(run_dir, _participant_root(run_dir), output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("report", "expected_state", "expected_reason", "expected_bytes"),
    (
        (None, "absent", "No narrative report was submitted.", b""),
        (" \n\t", "blank", "The submitted narrative report was blank.", b" \n\t"),
    ),
)
def test_export_preserves_narrative_noncompletion_as_abstention_packet(
    tmp_path: Path,
    report: str | None,
    expected_state: str,
    expected_reason: str,
    expected_bytes: bytes,
) -> None:
    run_dir = _completed_run(tmp_path)
    schedule = read_json_model(TrialEvalAblationScheduleV1, run_dir / "schedule.json")
    narrative = next(row for row in schedule.assignments if row.submission_interface == "narrative")
    result_path = run_dir / "assignments" / f"{narrative.assignment_id}.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["agent_output"]["report"] = report
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    output = exporter.export_narrative_transcription_packets(
        run_dir,
        _participant_root(run_dir),
        tmp_path / "packets",
    )
    packet = next(
        path
        for path in output.iterdir()
        if path.is_dir()
        and json.loads((path / "packet.json").read_text(encoding="utf-8"))["assignment_id"] == narrative.assignment_id
    )
    manifest = read_json(packet / "packet.json")
    template = read_json(packet / "transcription_template.json")

    assert manifest["report_state"] == expected_state
    assert (packet / "frozen_report.txt").read_bytes() == expected_bytes
    assert manifest["report_sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    assert template["status"] == "abstain"
    assert template["submission"] is None
    assert template["claims"] == []
    assert template["abstention_reason"] == expected_reason


def test_export_detects_source_drift_before_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _completed_run(tmp_path)
    output = tmp_path / "packets"
    real_sha256_file = run_artifacts.sha256_file
    schedule = read_json_model(TrialEvalAblationScheduleV1, run_dir / "schedule.json")
    narrative = next(row for row in schedule.assignments if row.submission_interface == "narrative")
    assignment_path = run_dir / "assignments" / f"{narrative.assignment_id}.json"
    calls = 0

    def drifting_sha256(path: Path) -> str:
        nonlocal calls
        digest = real_sha256_file(path)
        if path == assignment_path:
            calls += 1
            if calls == 2:
                return "0" * 64
        return digest

    monkeypatch.setattr(run_artifacts, "sha256_file", drifting_sha256)
    with pytest.raises(ValueError, match="drifted"):
        exporter.export_narrative_transcription_packets(run_dir, _participant_root(run_dir), output)
    assert not output.exists()


def test_normalizer_frame_preserves_all_reports_without_reading_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _completed_run(tmp_path)
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    labels = TrialEvalAblationEvaluatorLabelsV1(
        evaluator_release_sha256=sha256_dir_digest(evaluator),
        task_identities=(
            TrialEvalAblationTaskIdentityV1(
                task_id="TASK1001",
                base_trial_id="base-1",
                regime_cell_id="family-1",
                evaluation_series_id="family-1",
                design_tier="D1",
                design_subtype="individual_randomized",
                assumption_tier="A1",
                context_tier="C1",
                data_preparation="analysis_ready",
                analysis_specification="locked_sap",
            ),
        ),
    )
    monkeypatch.setattr(frame_builder, "_planned_result_shapes", lambda _: {"TASK1001": "scalar"})

    frame = frame_builder.build_trialeval_normalizer_frame_v1(
        run_dirs=(run_dir,),
        evaluator_root=evaluator,
        evaluator_labels=labels,
    )

    assert len(frame.units) == 3
    assert {row.base_trial_id for row in frame.units} == {"base-1"}
    assert {row.model_id for row in frame.units} == {"secret-model"}
    assert len({row.assignment_id for row in frame.units}) == 3
    assert {row.result_shape for row in frame.units} == {"scalar"}
    assert frame.checksum


def test_qualification_packet_export_is_exact_and_model_blind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _completed_run(tmp_path)
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    labels = TrialEvalAblationEvaluatorLabelsV1(
        evaluator_release_sha256=sha256_dir_digest(evaluator),
        task_identities=(
            TrialEvalAblationTaskIdentityV1(
                task_id="TASK1001",
                base_trial_id="base-1",
                regime_cell_id="family-1",
                evaluation_series_id="family-1",
                design_tier="D1",
                design_subtype="individual_randomized",
                assumption_tier="A1",
                context_tier="C1",
                data_preparation="analysis_ready",
                analysis_specification="locked_sap",
            ),
        ),
    )
    monkeypatch.setattr(frame_builder, "_planned_result_shapes", lambda _: {"TASK1001": "scalar"})
    frame = frame_builder.build_trialeval_normalizer_frame_v1(
        run_dirs=(run_dir,),
        evaluator_root=evaluator,
        evaluator_labels=labels,
    )
    selected = frame.units[0]
    sample_unit = TrialEvalNormalizerSampleUnitV1(
        **selected.model_dump(mode="python"),
        stratum_id="regime_cell_id=family-1|context_configuration=C1",
        frame_base_trial_count=1,
        sampled_base_trial_count=1,
        base_trial_candidate_report_count=3,
        base_trial_inclusion_probability=1.0,
        within_base_report_inclusion_probability=1.0 / 3.0,
        inclusion_probability=1.0 / 3.0,
        selected_without_normalizer_or_score_outcomes=True,
    )
    sample = TrialEvalNormalizerSampleV1(
        experiment_design_checksum="d" * 64,
        frame_checksum=str(frame.checksum),
        selection_method="stratified_base_trial_then_within_base_hash_rank_v1",
        selection_seed=19,
        units=(sample_unit,),
    ).with_checksum()
    output = tmp_path / "qualification-packets"

    manifest = export_normalizer_sample_packets_v1(
        sample=sample,
        run_dirs=(run_dir,),
        participant_root=_participant_root(run_dir),
        output_dir=output,
    )

    assert len(manifest.packets) == 1
    assert manifest.packets[0].qualification_unit_id == selected.unit_id
    assert manifest.sample_checksum == sample.checksum
    assert b"secret-model" not in b"".join(path.read_bytes() for path in output.rglob("*") if path.is_file())
