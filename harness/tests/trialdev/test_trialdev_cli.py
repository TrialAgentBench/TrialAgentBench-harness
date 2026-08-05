"""TrialDev live-run CLI contracts."""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from trialagentbench_harness.analysis.trialdev_ingestion import (
    trialdev_programme_pairing_sha256,
)
from trialagentbench_harness.contracts.core.config import ProviderReasoningCapabilityV1
from trialagentbench_harness.contracts.core.runs import (
    ExecutorEnvironmentV1,
    ExecutorPackageV1,
    TrialDevChainSummaryV1,
    TrialDevMaterializationUsageV1,
    TrialDevRunConfigV1,
)
from trialagentbench_harness.io.json import write_json_model
from trialagentbench_harness.ports import CodeExecutionLimitsV1
from trialagentbench_harness.tools.run.trialdev import (
    _build_experiment_condition,
    _build_run_config,
    _configure_logging,
    _filter_programs,
    _make_run_root,
    _program_progress_status,
    _programs_to_append,
    _staging_source_digest,
    _trialdev_runtime_source_digest,
    _validate_append_identity,
    main,
    parse_args,
    run_one_master_seed,
)
from trialagentbench_harness.trialdev.agent import DEFAULT_MAX_TURNS_PER_STEP
from trialagentbench_harness.trialdev.schema import Program, ProgramRun


def _executor() -> ExecutorEnvironmentV1:
    return ExecutorEnvironmentV1(
        image_reference="executor:test",
        image_id=f"sha256:{'a' * 64}",
        python_version="3.12",
        packages=(ExecutorPackageV1(name="pandas", version="2.2.0"),),
        limits=CodeExecutionLimitsV1(),
    )


def _args(**overrides: object) -> argparse.Namespace:
    values = vars(parse_args(("--bundle", ".", "--model", "model", "--provider", "openai")))
    values.update(overrides)
    return argparse.Namespace(**values)


def _program() -> Program:
    return Program(
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        items_by_phase={},
    )


def test_cli_has_no_dimensionally_invalid_cost_budget() -> None:
    args = parse_args(("--bundle", "release", "--model", "model", "--provider", "openai"))

    assert "enforce_cost_budget" not in vars(args)
    assert "resume_run_dir" not in vars(args)
    assert args.max_turns_per_step == DEFAULT_MAX_TURNS_PER_STEP == 45
    assert args.max_submission_attempts == 3
    assert args.procedure_assistance == "output_contract_only"
    assert args.tool_choice == "auto"
    assert args.reasoning_effort is None
    assert args.decoding_seed is None


def test_machine_readable_request_uses_clean_defaults_and_relative_paths(tmp_path: Path) -> None:
    config = tmp_path / "experiment.json"
    config.write_text(
        """{
  "schema_id": "trialagentbench.trialdev_execution_request/v1",
  "bundle": "release",
  "model": "provider/model",
  "provider": "openrouter",
  "dotenv": true,
  "openrouter_provider": "Provider",
  "programs": ["programme-1"],
  "output_root": "results"
}
""",
        encoding="utf-8",
    )

    args = parse_args(("--experiment-config", str(config)))

    assert args.bundle == str((tmp_path / "release").resolve())
    assert args.output_root == str((tmp_path / "results").resolve())
    assert args.model == "provider/model"
    assert args.dotenv is True
    assert args.programs == ["programme-1"]
    assert args.max_turns_per_step == 45
    assert args.max_submission_attempts == 3
    assert args.procedure_assistance == "output_contract_only"
    assert args.tool_choice == "auto"


def test_machine_readable_request_rejects_route_errors_before_provider_construction(tmp_path: Path) -> None:
    config = tmp_path / "invalid.json"
    config.write_text(
        """{
  "schema_id": "trialagentbench.trialdev_execution_request/v1",
  "bundle": "release",
  "model": "provider/model",
  "provider": "openrouter"
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="openrouter_provider"):
        parse_args(("--experiment-config", str(config)))


def test_single_asset_filter_accepts_documented_programme_identifiers() -> None:
    programme = _program()

    assert _filter_programs([programme], [programme.program_id]) == [programme]
    assert _filter_programs([programme], ["s01:benefit_risk"]) == [programme]
    assert _filter_programs([programme], ["s01"]) == [programme]


@pytest.mark.parametrize("execution_status", ("model_turn_limit", "model_invalid_submission"))
def test_progress_reports_typed_model_outcome_without_infrastructure_error(
    tmp_path: Path,
    execution_status: Literal["model_turn_limit", "model_invalid_submission"],
) -> None:
    run = ProgramRun(
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        workdir=tmp_path,
        execution_status=execution_status,
        error="A model submission was not accepted.",
    )

    assert _program_progress_status(run) == execution_status


def test_cli_rejects_incomplete_single_asset_release_before_creating_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    output = tmp_path / "runs"
    monkeypatch.setattr(
        "trialagentbench_harness.tools.run.trialdev.discover_programs",
        lambda _: [_program()],
    )

    status = main(
        (
            "--bundle",
            str(bundle),
            "--model",
            "model",
            "--provider",
            "openai",
            "--output-root",
            str(output),
        )
    )

    assert status == 2
    assert "missing fixed_trajectories/cases.jsonl" in capsys.readouterr().err
    assert not output.exists()


def test_cli_records_explicit_model_resource_limits(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    args = _args(
        bundle=str(bundle),
        max_tokens=2048,
        max_context_characters=40_000,
    )

    config = _build_run_config(
        args,
        1,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )

    assert config.decoding.max_tokens == 2048
    assert config.max_context_characters == 40_000


def _capability(tmp_path: Path) -> Path:
    path = tmp_path / "reasoning-capability.json"
    write_json_model(
        path,
        ProviderReasoningCapabilityV1(
            provider_transport="openrouter",
            model_id="openai/gpt-5.6-luna",
            upstream_provider="OpenAI",
            supported_efforts=("low", "medium", "high"),
            source_url="https://openrouter.ai/api/v1/models",
            source_retrieved_utc=datetime(2026, 8, 3, tzinfo=UTC),
            source_payload_sha256="a" * 64,
        ),
    )
    return path


def test_reasoning_condition_is_source_bound_and_enters_run_identity(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    capability = _capability(tmp_path)
    common = {
        "bundle": str(bundle),
        "provider": "openrouter",
        "model": "openai/gpt-5.6-luna",
        "openrouter_provider": "OpenAI",
        "reasoning_capability_snapshot": capability,
    }
    low = _build_run_config(
        _args(**common, condition_id="luna-low", reasoning_effort="low"),
        1,
        45560,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )
    high = _build_run_config(
        _args(**common, condition_id="luna-high", reasoning_effort="high"),
        1,
        45560,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )

    assert low.experiment_condition.reasoning.effort == "low"
    assert low.run_identity_sha256 != high.run_identity_sha256


def test_run_identity_binds_the_complete_trialdev_runtime_tree(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    config = _build_run_config(
        _args(bundle=str(bundle)),
        1,
        45560,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )

    assert config.runner_source_sha256 == _trialdev_runtime_source_digest()


def test_interface_and_correction_controls_enter_condition_identity(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    clean = _build_run_config(
        _args(bundle=str(bundle)),
        1,
        45560,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )
    required = _build_run_config(
        _args(bundle=str(bundle), tool_choice="required"),
        1,
        45560,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )
    more_corrections = _build_run_config(
        _args(bundle=str(bundle), max_submission_attempts=4),
        1,
        45560,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )

    assert clean.experiment_condition.tool_choice == "auto"
    assert clean.experiment_condition.maximum_submission_attempts == 3
    assert len({clean.run_identity_sha256, required.run_identity_sha256, more_corrections.run_identity_sha256}) == 3


def test_reasoning_condition_rejects_unsupported_or_drifted_routes(tmp_path: Path) -> None:
    capability = _capability(tmp_path)
    base = {
        "provider": "openrouter",
        "model": "openai/gpt-5.6-luna",
        "openrouter_provider": "OpenAI",
        "reasoning_effort": "max",
        "reasoning_capability_snapshot": capability,
    }
    with pytest.raises(ValidationError, match="not supported"):
        _build_experiment_condition(_args(**base))

    with pytest.raises(ValueError, match="model does not match"):
        _build_experiment_condition(_args(**(base | {"model": "openai/other", "reasoning_effort": "high"})))


def test_live_run_stops_before_offline_grading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(
        "trialagentbench_harness.tools.run.trialdev.resolve_executor_environment",
        _executor,
    )
    monkeypatch.setattr(
        "trialagentbench_harness.tools.run.trialdev._execute_one",
        lambda program, **kwargs: {
            "program_id": program.program_id,
            "execution_status": "completed",
        },
    )
    monkeypatch.setattr(
        "trialagentbench_harness.tools.run.trialdev.summarize_provider_telemetry_v1",
        lambda **kwargs: None,
    )

    run_root = run_one_master_seed(
        args=_args(bundle=str(bundle), output_root=str(tmp_path), workers=1),
        programs=[_program()],
        master_seed=42,
        output_root=tmp_path,
        label="ungraded",
    )

    assert not (run_root / "results_summary.json").exists()
    assert not (run_root / "results_full.csv").exists()
    assert "Grade this immutable run" in capsys.readouterr().out


def test_run_config_records_optional_decoding_seed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    config = _build_run_config(
        _args(bundle=str(bundle), decoding_seed=23),
        1,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )

    assert config.decoding.decoding_seed == 23
    assert config.staging_source_sha256 == _staging_source_digest()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("scorer_source_sha256", "1" * 64),
        ("runner_source_sha256", "3" * 64),
        ("staging_source_sha256", "2" * 64),
        ("seed_variants", 7),
    ),
)
def test_ablation_pairing_identity_binds_scientific_run_provenance(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = _build_run_config(
        _args(bundle=str(bundle)),
        1,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )

    changed = config.model_copy(update={field: value})

    assert trialdev_programme_pairing_sha256(
        changed,
        program_id="s01__benefit_risk",
    ) != trialdev_programme_pairing_sha256(
        config,
        program_id="s01__benefit_risk",
    )


@pytest.mark.parametrize("decoding_seed", [True, -1])
def test_run_config_rejects_invalid_decoding_seed(
    tmp_path: Path,
    decoding_seed: object,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    with pytest.raises(ValidationError, match="decoding_seed"):
        _build_run_config(
            _args(bundle=str(bundle), decoding_seed=decoding_seed),
            1,
            42,
            executor=_executor(),
            selected_program_ids=["s01__benefit_risk"],
        )


def test_append_identity_rejects_decoding_seed_drift(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    persisted = _build_run_config(
        _args(bundle=str(bundle), decoding_seed=11),
        1,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )
    write_json_model(tmp_path / "run_config.json", persisted)
    requested = _build_run_config(
        _args(bundle=str(bundle), decoding_seed=12),
        1,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )

    with pytest.raises(ValueError, match="decoding|run_identity_sha256"):
        _validate_append_identity(tmp_path, requested)


def test_new_run_root_refuses_existing_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> FixedDatetime:
            return cls(2026, 7, 18, 12, 0, tzinfo=tz)

    monkeypatch.setattr("trialagentbench_harness.tools.run.trialdev.datetime", FixedDatetime)
    _make_run_root(tmp_path, "model", 42, None)

    with pytest.raises(FileExistsError):
        _make_run_root(tmp_path, "model", 42, None)


def test_append_identity_rejects_model_drift(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{"version": 1}', encoding="utf-8")
    persisted = _build_run_config(
        _args(model="model-a", bundle=str(bundle)),
        1,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )
    write_json_model(tmp_path / "run_config.json", persisted)
    requested = _build_run_config(
        _args(model="model-b", bundle=str(bundle)),
        1,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )

    with pytest.raises(ValueError, match="model"):
        _validate_append_identity(tmp_path, requested)


def test_append_identity_rejects_bundle_mutation_at_same_path(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = bundle / "manifest.json"
    manifest.write_text('{"version": 1}', encoding="utf-8")
    persisted = _build_run_config(
        _args(bundle=str(bundle)),
        1,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )
    write_json_model(tmp_path / "run_config.json", persisted)

    manifest.write_text('{"version": 2}', encoding="utf-8")
    requested = _build_run_config(
        _args(bundle=str(bundle)),
        1,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )

    with pytest.raises(ValueError, match="bundle_sha256|run_identity_sha256"):
        _validate_append_identity(tmp_path, requested)


def test_append_identity_rejects_selected_population_drift(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    persisted = _build_run_config(
        _args(bundle=str(bundle)),
        2,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk", "s02__benefit_risk"],
    )
    write_json_model(tmp_path / "run_config.json", persisted)
    requested = _build_run_config(
        _args(bundle=str(bundle)),
        1,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )

    with pytest.raises(ValueError, match="run_identity_sha256|selected_program_ids"):
        _validate_append_identity(tmp_path, requested)


@pytest.mark.parametrize(
    "field",
    (
        "bundle_sha256",
        "scorer_source_sha256",
        "runner_source_sha256",
        "prompt_interface_sha256",
        "staging_source_sha256",
        "run_identity_sha256",
        "selected_program_ids",
        "n_programs_selected",
    ),
)
def test_trialdev_run_contract_rejects_omitted_authoritative_identity(
    tmp_path: Path,
    field: str,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = _build_run_config(
        _args(bundle=str(bundle)),
        1,
        42,
        executor=_executor(),
        selected_program_ids=["s01__benefit_risk"],
    )
    payload = config.model_dump(mode="python")
    del payload[field]

    with pytest.raises(ValidationError, match=field):
        TrialDevRunConfigV1.model_validate(payload)


def test_append_skips_only_exact_completed_program(tmp_path: Path) -> None:
    program = _program()
    program_dir = tmp_path / "programs" / program.program_id
    program_dir.mkdir(parents=True)
    write_json_model(
        program_dir / "chain_summary.json",
        TrialDevChainSummaryV1(
            program_id=program.program_id,
            scenario_id=program.scenario_id,
            objective_id=program.objective_id,
            materialization_usage=TrialDevMaterializationUsageV1(),
            execution_status="completed",
        ),
    )

    assert _programs_to_append(tmp_path, [program]) == ([], [program.program_id])


def test_append_skips_observed_model_turn_limit_outcome(tmp_path: Path) -> None:
    program = _program()
    program_dir = tmp_path / "programs" / program.program_id
    program_dir.mkdir(parents=True)
    write_json_model(
        program_dir / "chain_summary.json",
        TrialDevChainSummaryV1(
            program_id=program.program_id,
            scenario_id=program.scenario_id,
            objective_id=program.objective_id,
            materialization_usage=TrialDevMaterializationUsageV1(),
            execution_status="model_turn_limit",
            error="AgentTurnLimitExceeded: no submission within the declared turn budget",
        ),
    )

    assert _programs_to_append(tmp_path, [program]) == ([], [program.program_id])


def test_append_skips_observed_invalid_submission_outcome(tmp_path: Path) -> None:
    program = _program()
    program_dir = tmp_path / "programs" / program.program_id
    program_dir.mkdir(parents=True)
    write_json_model(
        program_dir / "chain_summary.json",
        TrialDevChainSummaryV1(
            program_id=program.program_id,
            scenario_id=program.scenario_id,
            objective_id=program.objective_id,
            materialization_usage=TrialDevMaterializationUsageV1(),
            execution_status="model_invalid_submission",
            error="TrialMaterializationRejectedError: correction budget exhausted",
        ),
    )

    assert _programs_to_append(tmp_path, [program]) == ([], [program.program_id])


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("infrastructure_error", "provider failed"),
        ("completed", "unexpected persisted error"),
    ],
)
def test_append_rejects_existing_noncomplete_program(
    tmp_path: Path,
    status: Literal["infrastructure_error", "completed"],
    error: str,
) -> None:
    program = _program()
    program_dir = tmp_path / "programs" / program.program_id
    program_dir.mkdir(parents=True)
    write_json_model(
        program_dir / "chain_summary.json",
        TrialDevChainSummaryV1(
            program_id=program.program_id,
            scenario_id=program.scenario_id,
            objective_id=program.objective_id,
            materialization_usage=TrialDevMaterializationUsageV1(),
            execution_status=status,
            error=error,
        ),
    )

    with pytest.raises(FileExistsError, match="no exact continuation checkpoint"):
        _programs_to_append(tmp_path, [program])


def test_verbose_logging_does_not_enable_provider_payload_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    configured: dict[str, object] = {}

    def capture_config(**kwargs: object) -> None:
        configured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", capture_config)
    harness_loggers = tuple(
        logging.getLogger(name) for name in ("trialagentbench_harness.trialdev", "trialagentbench_harness")
    )
    original_levels = tuple(logger.level for logger in harness_loggers)
    try:
        _configure_logging(verbose=True)

        assert configured["level"] == logging.INFO
        assert all(logger.level == logging.DEBUG for logger in harness_loggers)
        assert logging.getLogger("openai").getEffectiveLevel() != logging.DEBUG
    finally:
        for logger, level in zip(harness_loggers, original_levels, strict=True):
            logger.setLevel(level)
