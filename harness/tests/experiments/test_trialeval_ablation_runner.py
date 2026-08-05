"""Black-box tests for the participant-only TrialEval ablation runner."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)

from trialagentbench_harness.contracts.core.runs import (
    ExecutorEnvironmentV1,
    ExecutorPackageV1,
)
from trialagentbench_harness.contracts.experiments import TrialEvalAblationScheduleV1
from trialagentbench_harness.contracts.trace.observable import (
    runtime_event_source_payload_v1,
)
from trialagentbench_harness.experiments import trialeval_ablation
from trialagentbench_harness.io import (
    canonical_payload_sha256,
    sha256_dir_digest,
    write_json_model,
)
from trialagentbench_harness.ports import CodeExecutionLimitsV1
from trialagentbench_harness.trialeval.conditions import prompt_set_sha256_v1
from trialagentbench_harness.trialeval.data import (
    discover_participant_items,
    participant_analysis_surface_sha256,
)


@pytest.fixture(autouse=True)
def _use_test_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = ExecutorEnvironmentV1(
        image_reference="executor:test",
        image_id=f"sha256:{'a' * 64}",
        python_version="3.11",
        packages=(ExecutorPackageV1(name="pandas", version="2.3.3"),),
        limits=CodeExecutionLimitsV1(),
    )
    monkeypatch.setattr(trialeval_ablation, "resolve_executor_environment", lambda: executor)


def _participant_release(root: Path) -> Path:
    participant = root / "public"
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
                    },
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
    return participant


def _agent_payload(**kwargs):
    interface = kwargs["submission_interface"]
    source_path = kwargs["conversation_log_path"].as_posix()
    started_payload = runtime_event_source_payload_v1(
        benchmark="trialeval",
        task_id="TASK1001",
        program_id=None,
        scenario_id=None,
        objective_id=None,
        phase_id="task",
        step_id="analysis",
        event_type="step_started",
        terminal_status=None,
        failure_type=None,
        conversation_message=None,
    )
    terminal_payload = runtime_event_source_payload_v1(
        benchmark="trialeval",
        task_id="TASK1001",
        program_id=None,
        scenario_id=None,
        objective_id=None,
        phase_id="task",
        step_id="analysis",
        event_type="step_terminal",
        terminal_status="failed",
        failure_type="turn_limit_no_submission",
        conversation_message=None,
    )
    return {
        "status": "max_turns_reached",
        "result": None,
        "report": None,
        "turns_used": 1,
        "conversation": [],
        "events": [
            {
                "event_id": "trialeval:TASK1001:000000",
                "timestamp": datetime.now(UTC).isoformat(),
                "source_artifact_path": source_path,
                "source_payload_sha256": canonical_payload_sha256(started_payload),
                "benchmark": "trialeval",
                "task_id": "TASK1001",
                "event_index": 0,
                "phase_id": "task",
                "step_id": "analysis",
                "event_type": "step_started",
            },
            {
                "event_id": "trialeval:TASK1001:000001",
                "timestamp": datetime.now(UTC).isoformat(),
                "source_artifact_path": source_path,
                "source_payload_sha256": canonical_payload_sha256(terminal_payload),
                "benchmark": "trialeval",
                "task_id": "TASK1001",
                "event_index": 1,
                "phase_id": "task",
                "step_id": "analysis",
                "event_type": "step_terminal",
                "terminal_status": "failed",
                "failure_type": "turn_limit_no_submission",
            },
        ],
        "condition_provenance": {
            "procedure_assistance": kwargs["procedure_assistance"],
            "analysis_specification": kwargs["analysis_specification"],
            "analysis_surface_sha256": kwargs["analysis_surface_sha256"],
            "prompt_condition": kwargs["prompt_condition"],
            "submission_interface": interface,
            "max_turns": kwargs["max_turns"],
            "prompt_set_sha256": prompt_set_sha256_v1(),
            "rendered_system_prompt_sha256": "b" * 64,
            "tool_schema_sha256": "c" * 64,
            "response_contract_sha256": "d" * 64,
        },
    }


def _schedule(participant: Path) -> TrialEvalAblationScheduleV1:
    randomization_seed = 9
    assignments = []
    item = discover_participant_items(participant, task_ids=("TASK1001",))["TASK1001"]
    for index, (assistance, interface) in enumerate(
        (assistance, interface)
        for assistance in ("output_contract_only", "unordered_checklist", "ordered_sop")
        for interface in ("structured", "narrative")
    ):
        assignments.append(
            {
                "assignment_id": f"assignment-{index}",
                "task_id": "TASK1001",
                "context_tier": "C1",
                "data_preparation": "analysis_ready",
                "analysis_specification": "locked_sap",
                "analysis_surface_sha256": participant_analysis_surface_sha256(item),
                "replicate_id": "seed-1",
                "decoding_seed": 101,
                "procedure_assistance": assistance,
                "prompt_condition": "neutral",
                "submission_interface": interface,
            }
        )
    assignments.sort(key=lambda row: row["assignment_id"])
    random.Random(randomization_seed).shuffle(assignments)
    return TrialEvalAblationScheduleV1.model_validate(
        {
            "experiment_id": "experiment-1",
            "design": "factorial_interface",
            "execution_scope": "pilot",
            "experiment_design_sha256": "d" * 64,
            "participant_release_sha256": sha256_dir_digest(participant),
            "prompt_set_sha256": prompt_set_sha256_v1(),
            "analysis_config_sha256": "d" * 64,
            "randomization_seed": randomization_seed,
            "assignments": assignments,
        }
    )


def _runner_args(participant: Path, schedule_path: Path, output: Path) -> list[str]:
    return [
        "--participant-dir",
        str(participant),
        "--schedule",
        str(schedule_path),
        "--model",
        "fixture/model",
        "--provider",
        "openrouter",
        "--openrouter-provider",
        "GMICloud",
        "--output-dir",
        str(output),
        "--turns",
        "1",
        "--max-tokens",
        "2048",
        "--max-context-characters",
        "40000",
        "--quiet",
    ]


def test_runner_executes_complete_schedule_without_evaluator_access(tmp_path: Path, monkeypatch) -> None:
    participant = _participant_release(tmp_path)
    (tmp_path / "grader").mkdir()
    (tmp_path / "grader" / "item_index.json").write_text("not-json", encoding="utf-8")
    schedule = _schedule(participant)
    schedule_path = tmp_path / "schedule.json"
    write_json_model(schedule_path, schedule)

    provider_seeds: list[int] = []

    def provider(*args, **kwargs):
        del args
        provider_seeds.append(kwargs["decoding_seed"])
        return object()

    monkeypatch.setattr(trialeval_ablation, "get_provider", provider)
    provider_log_paths: list[Path] = []
    executor_limits: list[object] = []
    context_limits: list[int] = []

    def fake_run_agent(item, provider, **kwargs):
        del provider
        provider_log_paths.append(kwargs["provider_log_path"])
        executor_limits.append(kwargs["executor_limits"])
        context_limits.append(kwargs["max_context_chars"])
        return _agent_payload(
            **kwargs,
            analysis_surface_sha256=participant_analysis_surface_sha256(item),
        )

    monkeypatch.setattr(trialeval_ablation, "run_agent", fake_run_agent)
    output = tmp_path / "output"
    exit_code = trialeval_ablation.main(_runner_args(participant, schedule_path, output))

    assert exit_code == 0
    assert len(tuple((output / "assignments").glob("*.json"))) == 6
    assert provider_seeds == [101] * 6
    assert len(provider_log_paths) == 6
    assert len(set(provider_log_paths)) == 6
    assert all(path.parent.name == "logs" for path in provider_log_paths)
    run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["item_watchdog_seconds"] == 3600
    assert run_config["decoding"]["max_tokens"] == 2048
    assert run_config["max_context_characters"] == 40_000
    assert context_limits == [40_000] * 6
    assert all(limit.model_dump(mode="json") == run_config["executor"]["limits"] for limit in executor_limits)
    assert len(tuple((output / "events").glob("*_events.jsonl"))) == 6
    assert (output / "schema_affordance_inventory.json").is_file()
    contrasts = (output / "condition_contrasts.md").read_text(encoding="utf-8")
    assert f"Canonical prompt-set SHA-256: `{prompt_set_sha256_v1()}`." in contrasts
    assert "## P0 versus P1" in contrasts
    assert "## Structured versus narrative interface affordance" in contrasts
    assert (output / "provider_telemetry_summary.json").is_file()
    assert not (output / "traces").exists()


def test_runner_resumes_completed_assignments_and_preserves_partial_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    participant = _participant_release(tmp_path)
    schedule = _schedule(participant)
    schedule_path = tmp_path / "schedule.json"
    write_json_model(schedule_path, schedule)
    monkeypatch.setattr(trialeval_ablation, "get_provider", lambda *args, **kwargs: object())

    calls = 0

    def interrupted_run_agent(item, provider, **kwargs):
        nonlocal calls
        del provider
        calls += 1
        if calls == 2:
            trace_path = kwargs["conversation_log_path"]
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text('{"partial": true}\n', encoding="utf-8")
            workspace = trace_path.parent / f"{trace_path.stem}_workspace"
            workspace.mkdir()
            (workspace / "stale.txt").write_text("must not be reused\n", encoding="utf-8")
            raise RuntimeError("provider interruption")
        return _agent_payload(
            **kwargs,
            analysis_surface_sha256=participant_analysis_surface_sha256(item),
        )

    monkeypatch.setattr(trialeval_ablation, "run_agent", interrupted_run_agent)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="provider interruption"):
        trialeval_ablation.main(_runner_args(participant, schedule_path, output))

    completed_assignment_id = schedule.assignments[0].assignment_id
    partial_assignment_id = schedule.assignments[1].assignment_id
    assert (output / "assignments" / f"{completed_assignment_id}.json").is_file()
    assert not (output / "provider_telemetry_summary.json").exists()

    resumed_calls = 0

    def resumed_run_agent(item, provider, **kwargs):
        nonlocal resumed_calls
        del provider
        resumed_calls += 1
        workspace = kwargs["conversation_log_path"].parent / (f"{kwargs['conversation_log_path'].stem}_workspace")
        assert not workspace.exists()
        return _agent_payload(
            **kwargs,
            analysis_surface_sha256=participant_analysis_surface_sha256(item),
        )

    monkeypatch.setattr(trialeval_ablation, "run_agent", resumed_run_agent)
    assert trialeval_ablation.main([*_runner_args(participant, schedule_path, output), "--resume"]) == 0

    assert resumed_calls == 5
    assert len(tuple((output / "assignments").glob("*.json"))) == 6
    archived = (
        output / "failed_attempts" / partial_assignment_id / "attempt-1" / f"{partial_assignment_id}.archived.json"
    )
    assert archived.is_file()
    archived_workspace = (
        output
        / "failed_attempts"
        / partial_assignment_id
        / "attempt-1"
        / f"{partial_assignment_id}_workspace.archived"
    )
    assert (archived_workspace / "stale.txt").is_file()
    assert (output / "provider_telemetry_summary.json").is_file()
    summary = json.loads((output / "provider_telemetry_summary.json").read_text(encoding="utf-8"))
    assert all("failed_attempts" not in path for path in summary["source_files"])
    with pytest.raises(ValueError, match="completed"):
        trialeval_ablation.main([*_runner_args(participant, schedule_path, output), "--resume"])
