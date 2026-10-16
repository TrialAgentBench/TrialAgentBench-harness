"""Custody tests for TrialEval interruption and exact append/resume."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import JsonValue
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)

from trialagentbench.contracts.core.runs import (
    ExecutorEnvironmentV1,
    ExecutorPackageV1,
)
from trialagentbench.execution_policy import TRIALEVAL_RELEASE_BUDGET_V1
from trialagentbench.io.checksums import canonical_payload_sha256
from trialagentbench.tools.run import trialeval
from trialagentbench.trialeval.conditions import prompt_set_sha256_v1

EXECUTOR_ENVIRONMENT = ExecutorEnvironmentV1(
    image_reference="trialagentbench/executor:1.1.1",
    image_id=f"sha256:{'0' * 64}",
    python_version="3.12.0",
    packages=(ExecutorPackageV1(name="pandas", version="2.3.0"),),
    limits=TRIALEVAL_RELEASE_BUDGET_V1.code_execution,
)
EXECUTOR_ENVIRONMENT_SHA256 = canonical_payload_sha256(cast(JsonValue, EXECUTOR_ENVIRONMENT.model_dump(mode="json")))


@pytest.fixture(autouse=True)
def _isolated_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep run-custody tests independent of the host Docker inventory."""

    def resolve_trusted_executor_environment(
        *,
        image: str | None,
        expected_image_id: str,
        expected_environment_sha256: str,
        limits: object,
    ) -> ExecutorEnvironmentV1:
        assert image is None
        assert expected_image_id == EXECUTOR_ENVIRONMENT.image_id
        assert expected_environment_sha256 == EXECUTOR_ENVIRONMENT_SHA256
        assert limits == TRIALEVAL_RELEASE_BUDGET_V1.code_execution
        return EXECUTOR_ENVIRONMENT

    monkeypatch.setattr(trialeval, "resolve_trusted_executor_environment", resolve_trusted_executor_environment)
    monkeypatch.setattr(
        trialeval,
        "load_participant_release",
        lambda root: SimpleNamespace(root=root, manifest=SimpleNamespace(checksum="f" * 64)),
    )


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
        (item / "data" / "analysis.parquet").write_bytes(b"fixture-analysis-data")
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


def _participant_archive(participant: Path) -> tuple[Path, str]:
    archive_path = participant.parent / "TrialEvalBench_participant.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(path for path in participant.rglob("*") if path.is_file()):
            member = zipfile.ZipInfo(source.relative_to(participant).as_posix(), (2020, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, source.read_bytes())
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return archive_path, digest


def _args(participant: Path, output: Path) -> list[str]:
    archive, digest = _participant_archive(participant)
    return [
        "--participant-release",
        str(archive),
        "--expected-sha256",
        digest,
        "--model",
        "fixture/model",
        "--expected-executor-image-id",
        EXECUTOR_ENVIRONMENT.image_id,
        "--expected-executor-environment-sha256",
        EXECUTOR_ENVIRONMENT_SHA256,
        "--provider",
        "openrouter",
        "--openrouter-provider",
        "GMICloud",
        "--output-dir",
        str(output),
        "--turns",
        "1",
        "--quiet",
        "--context",
        "all",
    ]


def test_preparation_freezes_identity_without_provider_and_allows_exact_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prespecify a real native schedule before credentials or model outcomes."""
    participant = _participant_release(tmp_path)

    def forbidden_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("preparation contacted a provider")

    monkeypatch.setattr(trialeval, "get_provider", forbidden_provider)
    arguments = _args(participant, tmp_path / "runs")
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    config_path = next((tmp_path / "runs").rglob("run_config.json"))
    before = config_path.read_bytes()
    coverage = json.loads((config_path.parent / "coverage.json").read_bytes())
    assert len(coverage["unit_ids"]) == 2
    assert coverage["completed_unit_ids"] == []
    assert trialeval.main([*arguments, "--prepare-only", "--resume-run-dir", str(config_path.parent)]) == 0
    assert config_path.read_bytes() == before
    with pytest.raises(AssertionError, match="contacted a provider"):
        trialeval.main([*arguments, "--resume-run-dir", str(config_path.parent)])


def test_trialeval_has_one_report_plus_program_interface() -> None:
    """The public runner has no interface-selection factor."""

    args = trialeval._parser().parse_args(
        [
            "--participant-release",
            "participant.zip",
            "--expected-sha256",
            "0" * 64,
            "--model",
            "fixture/model",
            "--expected-executor-image-id",
            EXECUTOR_ENVIRONMENT.image_id,
            "--expected-executor-environment-sha256",
            EXECUTOR_ENVIRONMENT_SHA256,
            "--provider",
            "openai",
        ]
    )

    assert not hasattr(args, "submission_interface")
    assert args.max_tokens == TRIALEVAL_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn


def test_trialeval_exposes_the_completion_budget_as_a_run_condition() -> None:
    """Reasoning-budget experiments can vary tokens without changing source."""

    args = trialeval._parser().parse_args(
        [
            "--participant-release",
            "participant.zip",
            "--expected-sha256",
            "0" * 64,
            "--model",
            "fixture/model",
            "--expected-executor-image-id",
            EXECUTOR_ENVIRONMENT.image_id,
            "--expected-executor-environment-sha256",
            EXECUTOR_ENVIRONMENT_SHA256,
            "--provider",
            "openai",
            "--max-tokens",
            "8192",
        ]
    )

    assert args.max_tokens == 8192


def test_trialeval_rejects_a_nonpositive_completion_budget(tmp_path: Path) -> None:
    """Invalid provider budgets fail before release or credential access."""

    participant = _participant_release(tmp_path)

    with pytest.raises(ValueError, match="--max-tokens must be at least 1"):
        trialeval.main(
            [
                *_args(participant, tmp_path / "runs"),
                "--max-tokens",
                "0",
            ]
        )


def _agent_payload(item, **kwargs):
    return {
        "status": "max_turns_reached",
        "failure_type": "turn_limit_no_submission",
        "report": None,
        "turns_used": 1,
        "conversation": [],
        "events": [],
        "condition_provenance": {
            "procedure_assistance": kwargs["procedure_assistance"],
            "analysis_specification": kwargs["analysis_specification"],
            "analysis_surface_sha256": "d" * 64,
            "max_turns": kwargs["max_turns"],
            "prompt_set_sha256": prompt_set_sha256_v1(),
            "rendered_system_prompt_sha256": "b" * 64,
            "tool_schema_sha256": "c" * 64,
        },
    }


def _capability(path: Path, *, model: str = "fixture/model") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_id": "ariadne.provider_model_capability/v1",
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
                "--provider-capability-snapshot",
                str(capability),
            ]
        )

    assert not output.exists()


@pytest.mark.parametrize("mode", ["default", "exclude", "replay"])
def test_trialeval_persists_and_forwards_reasoning_condition(
    tmp_path: Path,
    monkeypatch,
    mode,
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
        "--provider-capability-snapshot",
        str(capability),
    ]
    if mode != "default":
        args.append(f"--{mode}-returned-reasoning")

    with pytest.raises(RuntimeError, match="interrupted"):
        trialeval.main(args)

    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    condition = run_config["experiment_condition"]
    assert condition["condition_id"] == "luna-high"
    assert condition["request_replicate_id"] == "request-2"
    assert condition["reasoning"]["effort"] == "high"
    assert len(condition["provider_capability"]["checksum"]) == 64
    assert provider_arguments[0]["reasoning_effort"] == "high"
    assert provider_arguments[0]["exclude_reasoning"] is (mode == "exclude")
    assert provider_arguments[0]["replay_reasoning"] is (mode == "replay")
    assert condition["reasoning"]["exclude_from_response"] is (mode == "exclude")
    assert condition["reasoning"].get("replay_in_context", False) is (mode == "replay")


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


def test_trialeval_watchdog_checkpoints_failure_and_continues_the_schedule(tmp_path: Path, monkeypatch) -> None:
    """A replay watchdog is a terminal scientific attempt, not a campaign interruption."""
    from trialagentbench.ports import CodeExecutionResultV1, LLMResponse, ToolCall
    from trialagentbench.trialeval import agent
    from trialagentbench.util.runtime_context import RuntimeDeadlineExceeded

    class Session:
        image_id = "sha256:" + "e" * 64

        def execute(self, code):
            return ""

        def execute_result(self, code):
            return CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.0)

        def close(self):
            pass

    class Provider:
        model = "fixture/model"
        telemetry_route = "test"

        def __init__(self):
            self.calls = 0

        def generate_turn(self, *args, **kwargs):
            self.calls += 1
            call = (
                ToolCall(id="check", name="check_analysis", arguments="{}")
                if self.calls == 1
                else ToolCall(id="submit", name="submit_response", arguments='{"content":"Completed report."}')
            )
            return LLMResponse(tool_calls=[call])

    checked = []

    def check(call, **kwargs):
        task_id = kwargs["context"].checkpoint_id
        checked.append(task_id)
        if task_id == "TASK1001":
            raise RuntimeDeadlineExceeded("TrialEval item exhausted while blocking on analysis replay check.")
        return CodeExecutionResultV1(status="success", output="reproduced", elapsed_seconds=0.0), kwargs["session"]

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: Session())
    monkeypatch.setattr(agent, "check_analysis_tool_v1", check)
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: Provider())
    participant = _participant_release(tmp_path)
    assert trialeval.main([*_args(participant, tmp_path / "runs"), "--turns", "3"]) == 0
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    first = json.loads((run_root / "items" / "TASK1001.json").read_text())["agent_output"]
    second = json.loads((run_root / "items" / "TASK1002.json").read_text())["agent_output"]
    assert first["status"] == "failed"
    assert first["failure_type"] == "timeout"
    assert first["turns_used"] == 1
    assert second["status"] == "success"
    assert second["failure_type"] is None
    assert checked == ["TASK1001", "TASK1002", "TASK1002"]
    coverage = json.loads((run_root / "coverage.json").read_text())
    assert coverage["completed_unit_ids"] == coverage["unit_ids"] == ["TASK1001", "TASK1002"]
    events = [json.loads(line) for line in (run_root / "logs" / "TASK1001_events.jsonl").read_text().splitlines()]
    terminals = [event for event in events if event["event_type"] == "step_terminal"]
    assert len(terminals) == 1
    assert terminals[0]["failure_type"] == "timeout"
    assert (run_root / "provider_telemetry_summary.json").is_file()


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
            "exclude_reasoning": False,
            "replay_reasoning": False,
            "timeout_s": 300.0,
            "retry_policy": trialeval.ProviderRetryPolicyV1(),
            "memory_strategy": "bounded",
            "request_control": None,
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
        "schema_id": "ariadne.experiment_condition/v1",
        "condition_id": "primary",
        "request_replicate_id": "request-1",
        "reasoning": {
            "effort": None,
            "exclude_from_response": False,
        },
        "provider_capability": None,
        "procedure_assistance": "output_contract_only",
        "maximum_turns_per_step": 1,
        "tool_choice": "auto",
        "memory_strategy": "bounded",
    }
    assert run_config["task_evidence_factors"] == {
        task_id: {
            "context_configuration": "C1",
            "data_preparation": "analysis_ready",
            "analysis_specification": "locked_sap",
        }
        for task_id in ("TASK1001", "TASK1002")
    }
    assert run_config["budget_profile_id"] == TRIALEVAL_RELEASE_BUDGET_V1.profile_id
    assert run_config["maximum_context_characters"] == TRIALEVAL_RELEASE_BUDGET_V1.maximum_context_characters
    assert observed == [("output_contract_only", "locked_sap")] * 2

    changed = _args(participant, tmp_path / "unused")
    turns_index = changed.index("--turns") + 1
    changed[turns_index] = "2"
    with pytest.raises(ValueError, match="identity"):
        trialeval.main([*changed, "--resume-run-dir", str(run_root)])


def test_trialeval_runs_the_unordered_checklist_ablation(tmp_path: Path, monkeypatch) -> None:
    """The documented assistance comparison uses the ordinary public runner."""

    participant = _participant_release(tmp_path)
    observed: list[str] = []
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: object())

    def run(item, provider, **kwargs):
        del provider
        observed.append(kwargs["procedure_assistance"])
        return _agent_payload(item, **kwargs)

    monkeypatch.setattr(trialeval, "run_agent", run)
    arguments = [
        *_args(participant, tmp_path / "runs"),
        "--procedure-assistance",
        "unordered_checklist",
    ]

    assert trialeval.main(arguments) == 0

    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["experiment_condition"]["procedure_assistance"] == "unordered_checklist"
    assert observed == ["unordered_checklist", "unordered_checklist"]


def test_canonical_default_rejects_a_release_without_c5_but_explicit_ids_remain_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    participant = _participant_release(tmp_path)
    arguments = _args(participant, tmp_path / "runs")[:-2]

    def forbidden_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("selection and preparation must not contact a provider")

    monkeypatch.setattr(trialeval, "get_provider", forbidden_provider)
    with pytest.raises(ValueError, match="No TrialEval tasks match"):
        trialeval.main([*arguments, "--prepare-only"])
    assert trialeval.main([*arguments, "--task-id", "TASK1001", "--prepare-only"]) == 0
    config = json.loads(next((tmp_path / "runs").rglob("run_config.json")).read_text())
    assert config["task_ids"] == ["TASK1001"]
    with pytest.raises(ValueError, match="conflict"):
        trialeval.main([*arguments, "--task-id", "TASK1001", "--context", "C5", "--prepare-only"])
