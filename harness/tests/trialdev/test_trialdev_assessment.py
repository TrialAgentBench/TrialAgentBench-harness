"""Tests for source-bound TrialDev assessment production."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from trialagentbench_harness.contracts.core.config import (
    DecodingConfigV1,
    ExperimentConditionV1,
    RoutingConfigV1,
)
from trialagentbench_harness.contracts.core.runs import (
    ExecutorEnvironmentV1,
    ExecutorPackageV1,
    TrialDevChainSummaryV1,
    TrialDevMaterializationUsageV1,
    TrialDevRunConfigV1,
)
from trialagentbench_harness.contracts.trialdev.metrics import (
    TrialDevAssessmentPortfolioV1,
    TrialDevMetricPortfolioV1,
)
from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevEvidenceReferenceV1,
    TrialDevPolicyBindingV1,
    TrialDevSingleAssetProgrammeStateV1,
)
from trialagentbench_harness.io import write_json, write_json_model
from trialagentbench_harness.ports import CodeExecutionLimitsV1
from trialagentbench_harness.tools.grade.grade_trialdev import main as grade_main
from trialagentbench_harness.trialdev.assessment import build_single_asset_programme_assessment_v1
from trialagentbench_harness.trialdev.share.sequential import TrialDevelopmentProgramLoopManifestV1


def _run_config() -> TrialDevRunConfigV1:
    return TrialDevRunConfigV1.create(
        timestamp_utc=datetime.now().astimezone(),
        bundle="release",
        bundle_sha256="a" * 64,
        scorer_source_sha256="b" * 64,
        runner_source_sha256="f" * 64,
        prompt_interface_sha256="c" * 64,
        staging_source_sha256="d" * 64,
        procedure_assistance="output_contract_only",
        model="model-a",
        experiment_condition=ExperimentConditionV1(
            condition_id="primary",
            request_replicate_id="request-1",
            maximum_turns_per_step=10,
            maximum_submission_attempts=3,
        ),
        master_seed=1,
        seed_variants=1,
        workers=1,
        decoding=DecodingConfigV1(temperature=0.0, max_tokens=512, send_temperature=True),
        routing=RoutingConfigV1(provider="openai", request_timeout_seconds=300.0),
        executor=ExecutorEnvironmentV1(
            image_reference="executor:test",
            image_id=f"sha256:{'e' * 64}",
            python_version="3.12",
            packages=(ExecutorPackageV1(name="pandas", version="2.2.0"),),
            limits=CodeExecutionLimitsV1(),
        ),
        max_turns_per_step=10,
        max_context_characters=10_000,
        max_phase_retries=2,
        max_submission_attempts=3,
        program_watchdog_seconds=600,
        selected_program_ids=["s01__benefit_risk"],
        n_programs_selected=1,
    )


def _write_noncompletion(program_dir: Path) -> None:
    state = TrialDevSingleAssetProgrammeStateV1(
        programme_id="s01__benefit_risk",
        scenario_id="s01",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=("drug-a",),
        policy_binding=TrialDevPolicyBindingV1(
            stream_id="single_asset_development",
            objective_id="benefit_risk",
            objective_policy_checksum="1" * 64,
            action_policy_checksum="2" * 64,
            design_menu_checksum="3" * 64,
        ),
        evidence=(
            TrialDevEvidenceReferenceV1(
                evidence_id="public-observational-data",
                evidence_kind="dataset",
                checkpoint_id="observational_review",
                evidence_protocol_id="observational-v1",
                evidence_protocol_checksum="4" * 64,
                source_family_id="s01",
                world_id="world-1",
                generation_seed=1,
                relative_path="public/observational_extract.parquet",
                artifact_sha256="5" * 64,
            ),
        ),
    )
    write_json(program_dir / "states" / "state_initial.json", state.model_dump(mode="json", exclude_none=True))
    write_json_model(
        program_dir / "chain_summary.json",
        TrialDevChainSummaryV1(
            program_id="s01__benefit_risk",
            scenario_id="s01",
            objective_id="benefit_risk",
            materialization_usage=TrialDevMaterializationUsageV1(),
            execution_status="model_turn_limit",
            error="The model did not submit the observational review.",
        ),
    )


def test_model_noncompletion_is_emitted_with_explicit_missing_lanes(tmp_path: Path) -> None:
    program_dir = tmp_path / "program"
    _write_noncompletion(program_dir)

    assessment = build_single_asset_programme_assessment_v1(
        program_dir=program_dir,
        run_config=_run_config(),
    )

    first = assessment.checkpoints[0]
    assert assessment.execution_status == "model_noncompletion"
    assert first.outcome.reach_status == "reached"
    assert first.outcome.execution_status == "model_noncompletion"
    assert {lane.lane_id for lane in first.lanes} == {"asset_nomination", "phase_analysis"}
    assert all(lane.outcome == "missing" for lane in first.lanes)
    assert all(item.outcome.reach_status == "structural_nonreach" for item in assessment.checkpoints[1:])


def test_assessment_rejects_a_mutated_state_checksum(tmp_path: Path) -> None:
    program_dir = tmp_path / "program"
    _write_noncompletion(program_dir)
    path = program_dir / "states" / "state_initial.json"
    payload = path.read_text(encoding="utf-8").replace('"checksum":', '"checksum":"0", "old_checksum":')
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        build_single_asset_programme_assessment_v1(
            program_dir=program_dir,
            run_config=_run_config(),
        )


def test_offline_grade_command_emits_assessments_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    program_dir = run_root / "programs" / "s01__benefit_risk"
    _write_noncompletion(program_dir)
    write_json_model(run_root / "run_config.json", _run_config())
    bundle = tmp_path / "bundle"
    write_json_model(
        bundle / "scenario_s01" / "public" / "program_loop_manifest.json",
        TrialDevelopmentProgramLoopManifestV1(
            scenario_id="s01",
            program_archetype="asset_development",
            decision_charter_checksum="0" * 64,
            phase_order=("observational_review", "phase1", "phase2", "phase3"),
            conditionally_materializable_phase_ids=("phase1", "phase2", "phase3"),
            phase_policy_modes={"phase1": "required", "phase2": "required", "phase3": "optional"},
            phase1_carryover_consequential=False,
            terminal_statuses=("stopped", "completed"),
            public_state_summary_fields=("scenario_id", "current_phase_id"),
        ),
    )
    monkeypatch.setattr(
        "trialagentbench_harness.tools.grade.grade_trialdev.aggregate_run",
        lambda *args, **kwargs: None,
    )
    output = tmp_path / "graded"

    assert grade_main([str(run_root), "--bundle", str(bundle), "--out-dir", str(output)]) == 0

    portfolio = TrialDevAssessmentPortfolioV1.model_validate_json(
        (output / "trialdev_assessments.json").read_text(encoding="utf-8")
    )
    metrics = TrialDevMetricPortfolioV1.model_validate_json(
        (output / "trialdev_metrics.json").read_text(encoding="utf-8")
    )
    assert portfolio.programmes[0].execution_status == "model_noncompletion"
    assert metrics.streams[0].denominators.model_noncompletion == 1
    assert metrics.streams[0].denominators.missing == 2


def test_offline_grade_dispatches_a_direct_portfolio_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    (run_root / "programs" / "portfolio-1").mkdir(parents=True)
    write_json_model(run_root / "run_config.json", _run_config())
    bundle = tmp_path / "portfolio-release"
    write_json(bundle / "participant_catalogue.json", {"release_id": "portfolio-v1"})
    observed: dict[str, Path] = {}

    def grade_portfolio(*, run_root: Path, bundle: Path, run_config: object) -> int:
        del run_config
        observed["run_root"] = run_root
        observed["bundle"] = bundle
        return 1

    monkeypatch.setattr(
        "trialagentbench_harness.tools.grade.grade_trialdev._grade_portfolio_run",
        grade_portfolio,
    )
    output = tmp_path / "graded"

    assert grade_main([str(run_root), "--bundle", str(bundle), "--out-dir", str(output)]) == 0

    assert observed["bundle"] == bundle.resolve()
    assert observed["run_root"].parent == output.parent
    assert observed["run_root"] != output
    assert (output / "GRADE_MANIFEST.json").is_file()
