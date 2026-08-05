"""Custody tests for TrialEval interruption and exact append/resume."""

from __future__ import annotations

import json
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
from trialagentbench_harness.ports import CodeExecutionLimitsV1
from trialagentbench_harness.tools.run import trialeval
from trialagentbench_harness.trialeval.conditions import prompt_set_sha256_v1


@pytest.fixture(autouse=True)
def _use_test_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = ExecutorEnvironmentV1(
        image_reference="executor:test",
        image_id=f"sha256:{'a' * 64}",
        python_version="3.11",
        packages=(ExecutorPackageV1(name="pandas", version="2.3.3"),),
        limits=CodeExecutionLimitsV1(),
    )
    monkeypatch.setattr(trialeval, "resolve_executor_environment", lambda: executor)


def _participant_release(root: Path) -> Path:
    participant = root / "participant"
    task_ids = ("TASK1001", "TASK1002")
    (participant / "manifest.json").parent.mkdir(parents=True)
    (participant / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_public_bundle_manifest/v1",
                "applied_baseline_profile_id": None,
                "applied_baseline_profile_sha256": None,
                "task_ids": list(task_ids),
                "task_evidence_factors": {
                    task_id: {
                        "context_configuration": "C1",
                        "data_preparation": "analysis_ready",
                        "analysis_specification": "locked_sap",
                    }
                    for task_id in task_ids
                },
            }
        ),
        encoding="utf-8",
    )
    for task_id in task_ids:
        item = participant / "items" / task_id
        (item / "data").mkdir(parents=True)
        (item / "task.json").write_text(
            json.dumps(
                {
                    "schema_id": "trial_analysis_task_v1",
                    "task_id": task_id,
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
            json.dumps(minimal_participant_output_contract(task_id)),
            encoding="utf-8",
        )
    write_minimal_trialeval_release_dictionaries(participant)
    return participant


def _args(participant: Path, output: Path) -> list[str]:
    return [
        "--participant-dir",
        str(participant),
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
        "--quiet",
    ]


def _agent_payload(item, **kwargs):
    return {
        "status": "max_turns_reached",
        "result": None,
        "report": None,
        "turns_used": 1,
        "conversation": [],
        "events": [],
        "condition_provenance": {
            "procedure_assistance": kwargs["procedure_assistance"],
            "analysis_specification": kwargs["analysis_specification"],
            "prompt_condition": "neutral",
            "submission_interface": "structured",
            "analysis_surface_sha256": "d" * 64,
            "max_turns": kwargs["max_turns"],
            "prompt_set_sha256": prompt_set_sha256_v1(),
            "rendered_system_prompt_sha256": "b" * 64,
            "tool_schema_sha256": "c" * 64,
            "response_contract_sha256": "d" * 64,
        },
    }


def _capability(path: Path, *, model: str = "fixture/model") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_id": "trialagentbench.provider_reasoning_capability/v1",
                "provider_transport": "openrouter",
                "model_id": model,
                "upstream_provider": "GMICloud",
                "supported_efforts": ["low", "medium", "high"],
                "source_url": "https://openrouter.ai/models/fixture/model",
                "source_retrieved_utc": "2026-08-04T00:00:00Z",
                "source_payload_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_trialeval_rejects_reasoning_route_mismatch_before_output(
    tmp_path: Path,
) -> None:
    participant = _participant_release(tmp_path)
    capability = _capability(tmp_path / "capability.json", model="fixture/other")
    output = tmp_path / "runs"

    with pytest.raises(ValueError, match="model does not match"):
        trialeval.main(
            [
                *_args(participant, output),
                "--reasoning-effort",
                "high",
                "--reasoning-capability-snapshot",
                str(capability),
            ]
        )

    assert not output.exists()


def test_trialeval_persists_and_forwards_reasoning_condition(
    tmp_path: Path,
    monkeypatch,
) -> None:
    participant = _participant_release(tmp_path)
    capability = _capability(tmp_path / "capability.json")
    provider_arguments: list[dict[str, object]] = []

    def provider(*args, **kwargs):
        del args
        provider_arguments.append(kwargs)
        return object()

    monkeypatch.setattr(trialeval, "get_provider", provider)
    monkeypatch.setattr(
        trialeval,
        "run_agent",
        lambda item, provider, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    args = [
        *_args(participant, tmp_path / "runs"),
        "--condition-id",
        "luna-high",
        "--request-replicate-id",
        "request-2",
        "--reasoning-effort",
        "high",
        "--reasoning-capability-snapshot",
        str(capability),
    ]

    with pytest.raises(RuntimeError, match="interrupted"):
        trialeval.main(args)

    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    condition = run_config["experiment_condition"]
    assert condition["condition_id"] == "luna-high"
    assert condition["request_replicate_id"] == "request-2"
    assert condition["reasoning"]["effort"] == "high"
    assert len(condition["reasoning"]["capability"]["checksum"]) == 64
    assert provider_arguments[0]["reasoning_effort"] == "high"
    assert provider_arguments[0]["exclude_reasoning"] is True


def test_trialeval_resumes_from_immutable_item_checkpoint(tmp_path: Path, monkeypatch) -> None:
    participant = _participant_release(tmp_path)
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: object())
    calls = 0

    def interrupted(item, provider, **kwargs):
        nonlocal calls
        del provider
        calls += 1
        if calls == 2:
            conversation_path = kwargs["conversation_log_path"]
            workspace = conversation_path.parent / "TASK1002_workspace"
            workspace.mkdir(parents=True)
            (workspace / "stale.txt").write_text("must not be reused\n", encoding="utf-8")
            raise RuntimeError("interrupted")
        return _agent_payload(item, **kwargs)

    monkeypatch.setattr(trialeval, "run_agent", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        trialeval.main(_args(participant, tmp_path / "runs"))
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    checkpoint = run_root / "items" / "TASK1001.json"
    checkpoint_bytes = checkpoint.read_bytes()

    resumed_calls = 0

    def resumed(item, provider, **kwargs):
        nonlocal resumed_calls
        del provider
        resumed_calls += 1
        workspace = kwargs["conversation_log_path"].parent / f"{item.item_id}_workspace"
        assert not workspace.exists()
        return _agent_payload(item, **kwargs)

    monkeypatch.setattr(trialeval, "run_agent", resumed)
    assert trialeval.main([*_args(participant, tmp_path / "unused"), "--resume-run-dir", str(run_root)]) == 0
    assert resumed_calls == 1
    assert checkpoint.read_bytes() == checkpoint_bytes
    coverage = json.loads((run_root / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["unit_ids"] == ["TASK1001", "TASK1002"]
    assert coverage["completed_unit_ids"] == ["TASK1001", "TASK1002"]
    assert (
        run_root / "failed_attempts" / "TASK1002" / "attempt-1" / "TASK1002_workspace.archived" / "stale.txt"
    ).is_file()


def test_trialeval_records_seed_and_rejects_resume_seed_drift(tmp_path: Path, monkeypatch) -> None:
    participant = _participant_release(tmp_path)
    provider_arguments: list[dict[str, object]] = []

    def provider(*args, **kwargs):
        del args
        provider_arguments.append(kwargs)
        return object()

    monkeypatch.setattr(trialeval, "get_provider", provider)
    monkeypatch.setattr(
        trialeval,
        "run_agent",
        lambda item, provider, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    seeded_args = [*_args(participant, tmp_path / "runs"), "--decoding-seed", "29"]
    with pytest.raises(RuntimeError, match="interrupted"):
        trialeval.main(seeded_args)
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))

    assert run_config["decoding"]["decoding_seed"] == 29
    assert provider_arguments == [
        {
            "routing": trialeval.ProviderRouting(provider="openrouter", openrouter_provider="GMICloud"),
            "send_temperature": True,
            "decoding_seed": 29,
            "reasoning_effort": None,
            "exclude_reasoning": True,
            "timeout_s": 300.0,
        }
    ]
    with pytest.raises(ValueError, match="identity"):
        trialeval.main(
            [
                *_args(participant, tmp_path / "unused"),
                "--decoding-seed",
                "30",
                "--resume-run-dir",
                str(run_root),
            ]
        )


def test_trialeval_resume_rejects_release_mutation_and_denominator_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    participant = _participant_release(tmp_path)
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: object())

    def interrupted(item, provider, **kwargs):
        del item, provider, kwargs
        raise RuntimeError("interrupted")

    monkeypatch.setattr(trialeval, "run_agent", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        trialeval.main(_args(participant, tmp_path / "runs"))
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())

    changed_model_args = _args(participant, tmp_path / "unused")
    changed_model_args[changed_model_args.index("fixture/model")] = "fixture/other-model"
    with pytest.raises(ValueError, match="identity"):
        trialeval.main([*changed_model_args, "--resume-run-dir", str(run_root)])

    with pytest.raises(ValueError, match="denominator|identity"):
        trialeval.main(
            [
                *_args(participant, tmp_path / "unused"),
                "--task-id",
                "TASK1001",
                "--resume-run-dir",
                str(run_root),
            ]
        )

    task = participant / "items" / "TASK1001" / "task.json"
    payload = json.loads(task.read_text(encoding="utf-8"))
    payload["primary_endpoint_id"] = "mutated"
    task.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        trialeval.main([*_args(participant, tmp_path / "unused"), "--resume-run-dir", str(run_root)])


def test_trialeval_persists_and_forwards_analysis_contract(tmp_path: Path, monkeypatch) -> None:
    participant = _participant_release(tmp_path)
    observed: list[tuple[str, str]] = []
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: object())

    def run(item, provider, **kwargs):
        del provider
        observed.append((kwargs["procedure_assistance"], kwargs["analysis_specification"]))
        return _agent_payload(item, **kwargs)

    monkeypatch.setattr(trialeval, "run_agent", run)
    assert trialeval.main(_args(participant, tmp_path / "runs")) == 0
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))

    assert run_config["experiment_condition"] == {
        "schema_id": "trialagentbench.experiment_condition/v1",
        "condition_id": "primary",
        "request_replicate_id": "request-1",
        "reasoning": {
            "effort": None,
            "exclude_from_response": True,
            "capability": None,
        },
        "procedure_assistance": "output_contract_only",
        "maximum_turns_per_step": 1,
        "maximum_submission_attempts": None,
        "tool_choice": "auto",
    }
    assert run_config["task_evidence_factors"] == {
        task_id: {
            "context_configuration": "C1",
            "data_preparation": "analysis_ready",
            "analysis_specification": "locked_sap",
        }
        for task_id in ("TASK1001", "TASK1002")
    }
    assert observed == [("output_contract_only", "locked_sap")] * 2

    changed = _args(participant, tmp_path / "unused")
    turns_index = changed.index("--turns") + 1
    changed[turns_index] = "2"
    with pytest.raises(ValueError, match="identity"):
        trialeval.main([*changed, "--resume-run-dir", str(run_root)])
