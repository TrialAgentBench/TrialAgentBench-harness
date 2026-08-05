from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.core.config import DecodingConfigV1
from trialagentbench_harness.contracts.core.runs import (
    TrialDevChainSummaryV1,
    TrialDevMaterializationUsageV1,
    TrialDevPhaseAttemptSummaryV1,
    TrialEvalItemResultV1,
    TrialEvalRunConfigV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevTerminalSummaryV1,
    TrialDevTrajectoryGradeV1,
)
from trialagentbench_harness.grading.models import ValidatedScoringKeyV1
from trialagentbench_harness.io import read_json_model, staged_directory, write_json_model
from trialagentbench_harness.io.checksums import sha256_path
from trialagentbench_harness.tools.grade import grade_trialeval
from trialagentbench_harness.tools.grade.grade_trialdev import grade_program
from trialagentbench_harness.trialdev.grading.models import TrialDevProgrammeResourceConsequenceV1
from trialagentbench_harness.trialdev.share.sequential import TrialDevelopmentProgramLoopManifestV1
from trialagentbench_harness.trialeval.schema import BenchmarkItem


def _run_config_payload() -> dict[str, object]:
    config = TrialEvalRunConfigV1.create(
        timestamp_utc=datetime.now(UTC),
        model="fixture-model",
        output_mode="structured",
        item_watchdog_seconds=3600,
        participant_dir="participant",
        participant_release_sha256="a" * 64,
        prompt_set_sha256="b" * 64,
        scorer_source_sha256="c" * 64,
        agent_source_sha256="d" * 64,
        experiment_condition={
            "condition_id": "primary",
            "request_replicate_id": "request-1",
            "procedure_assistance": "output_contract_only",
            "maximum_turns_per_step": 25,
            "tool_choice": "auto",
        },
        task_evidence_factors={
            "TASK1": {
                "context_configuration": "C1",
                "data_preparation": "analysis_ready",
                "analysis_specification": "locked_sap",
            }
        },
        task_ids=["TASK1"],
        n_items=1,
        data_format="trialagentbench_v1",
        data_version="trialagentbench_v1",
        decoding=DecodingConfigV1(temperature=0.0, max_tokens=512, send_temperature=True),
        routing={"provider": "openai", "request_timeout_seconds": 300.0},
        executor={
            "image_reference": "executor:test",
            "image_id": f"sha256:{'e' * 64}",
            "python_version": "3.12",
            "packages": [{"name": "pandas", "version": "2.2"}],
            "limits": {},
        },
        workers=1,
    )
    return config.model_dump(mode="python")


def _write_program_loop_manifest(bundle: Path) -> None:
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
            public_state_summary_fields=(
                "scenario_id",
                "current_phase_id",
                "eligible_candidate_drug_ids",
                "completed_phase_ids",
            ),
        ),
    )


def test_trial_eval_run_contract_rejects_superseded_identity() -> None:
    payload = _run_config_payload()
    payload["data_version"] = "invalid_release"

    with pytest.raises(ValidationError, match="trialagentbench_v1"):
        TrialEvalRunConfigV1.model_validate(payload)


def test_trial_eval_run_contract_rejects_judge_configuration() -> None:
    payload = _run_config_payload()
    payload["judge_model"] = "judge-model"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrialEvalRunConfigV1.model_validate(payload)


def test_trial_eval_run_contract_requires_positive_item_watchdog() -> None:
    payload = _run_config_payload()
    payload["item_watchdog_seconds"] = 0

    with pytest.raises(ValidationError, match="item_watchdog_seconds"):
        TrialEvalRunConfigV1.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    (
        "prompt_set_sha256",
        "scorer_source_sha256",
        "agent_source_sha256",
        "run_identity_sha256",
    ),
)
def test_trial_eval_run_contract_rejects_omitted_authoritative_identity(field: str) -> None:
    payload = _run_config_payload()
    del payload[field]

    with pytest.raises(ValidationError, match=field):
        TrialEvalRunConfigV1.model_validate(payload)


def test_trial_eval_item_rejects_lossy_extracted_score_authority() -> None:
    """A persisted derived extraction cannot override the canonical submission."""

    payload = {
        "item_id": "TASK1__C1",
        "timestamp_utc": datetime.now(UTC),
        "run_config": _run_config_payload(),
        "agent_output": {
            "status": "failed",
            "result": None,
            "condition_provenance": {
                "procedure_assistance": "output_contract_only",
                "analysis_specification": "locked_sap",
                "prompt_condition": "neutral",
                "submission_interface": "structured",
                "max_turns": 25,
                "prompt_set_sha256": "b" * 64,
                "rendered_system_prompt_sha256": "b" * 64,
                "tool_schema_sha256": "c" * 64,
                "response_contract_sha256": "d" * 64,
            },
        },
        "extracted": {"primary": {"estimator": "coxph_binary"}},
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrialEvalItemResultV1.model_validate(payload)


def test_trial_eval_item_rejects_analysis_specification_drift() -> None:
    payload = {
        "item_id": "TASK1",
        "timestamp_utc": datetime.now(UTC),
        "run_config": _run_config_payload(),
        "agent_output": {
            "status": "failed",
            "result": None,
            "condition_provenance": {
                "procedure_assistance": "output_contract_only",
                "analysis_specification": "protocol_only",
                "prompt_condition": "neutral",
                "submission_interface": "structured",
                "analysis_surface_sha256": "a" * 64,
                "max_turns": 25,
                "prompt_set_sha256": "b" * 64,
                "rendered_system_prompt_sha256": "c" * 64,
                "tool_schema_sha256": "d" * 64,
                "response_contract_sha256": "e" * 64,
            },
        },
    }

    with pytest.raises(ValidationError, match="condition provenance"):
        TrialEvalItemResultV1.model_validate(payload)


def test_trial_eval_offline_grader_uses_portable_key_and_preserves_noncompletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite"
    public = suite / "public"
    public.mkdir(parents=True)
    (suite / "grader").mkdir()
    (public / "manifest.json").write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_config = TrialEvalRunConfigV1.create(
        timestamp_utc=datetime.now(UTC),
        model="fixture-model",
        output_mode="structured",
        item_watchdog_seconds=3600,
        participant_dir=str(public),
        participant_release_sha256=sha256_path(public),
        prompt_set_sha256="b" * 64,
        scorer_source_sha256="c" * 64,
        agent_source_sha256="d" * 64,
        experiment_condition={
            "condition_id": "primary",
            "request_replicate_id": "request-1",
            "procedure_assistance": "output_contract_only",
            "maximum_turns_per_step": 25,
            "tool_choice": "auto",
        },
        task_evidence_factors={
            "TASK1": {
                "context_configuration": "C1",
                "data_preparation": "analysis_ready",
                "analysis_specification": "locked_sap",
            }
        },
        task_ids=["TASK1"],
        n_items=1,
        decoding=DecodingConfigV1(temperature=0.0, max_tokens=512, send_temperature=True),
        routing={"provider": "openai", "request_timeout_seconds": 300.0},
        executor={
            "image_reference": "executor:test",
            "image_id": f"sha256:{'e' * 64}",
            "python_version": "3.12",
            "packages": [{"name": "pandas", "version": "2.2"}],
            "limits": {},
        },
        workers=1,
    )
    write_json_model(run_dir / "run_config.json", run_config)
    write_json_model(
        run_dir / "items" / "TASK1.json",
        TrialEvalItemResultV1(
            item_id="TASK1",
            timestamp_utc=datetime.now(UTC),
            run_config=run_config,
            agent_output={
                "status": "failed",
                "turns_used": 25,
                "result": None,
                "condition_provenance": {
                    "procedure_assistance": "output_contract_only",
                    "analysis_specification": "locked_sap",
                    "analysis_surface_sha256": "a" * 64,
                    "prompt_condition": "neutral",
                    "submission_interface": "structured",
                    "max_turns": 25,
                    "prompt_set_sha256": "b" * 64,
                    "rendered_system_prompt_sha256": "c" * 64,
                    "tool_schema_sha256": "d" * 64,
                    "response_contract_sha256": "e" * 64,
                },
            },
        ),
    )
    item = BenchmarkItem(
        item_id="TASK1",
        task_id="TASK1",
        trial_name="fixture",
        design_tier="D1",
        design_subtype="individual_randomized",
        assumption_tier="A1",
        context_tier="C1",
        data_preparation="analysis_ready",
        analysis_specification="locked_sap",
        visible_dir=public,
        data_dir=public,
        task={},
    )
    key = ValidatedScoringKeyV1.model_validate(
        {
            "schema_id": "trialagentbench.scoring_key/v1",
            "release_id": "fixture-release",
            "item_id": "TASK1",
            "question_id": "question-1",
            "context_tier": "C1",
            "credit_eligible_routes": [
                {
                    "route_id": "route-1",
                    "signature": {
                        "analysis_population_id": "itt",
                        "estimand_id": "estimand",
                        "intercurrent_event_strategy_ids": ["treatment_policy"],
                        "treatment_id": "active",
                        "comparator_id": "control",
                        "endpoint_id": "endpoint",
                        "effect_scale": "scale",
                        "analysis_method_id": "fixture_decision_method",
                    },
                    "method": {
                        "analysis_method_id": "fixture_decision_method",
                        "estimator_family": "estimator",
                        "result_kind": "decision",
                        "uncertainty_method": "not_applicable",
                        "design_modifiers": [],
                    },
                    "required_identification_assumptions": ["randomization"],
                    "target": {
                        "kind": "categorical",
                        "credit_eligible_codes": ["advance"],
                    },
                }
            ],
        }
    )

    class _ScoringKeyStore:
        manifest = SimpleNamespace(scoring_keys_sha256="6" * 64)

        @classmethod
        def from_release(
            cls,
            release: Path,
            *,
            expected_item_ids: tuple[str, ...],
        ) -> _ScoringKeyStore:
            assert (release / "public" / "manifest.json").is_file()
            assert (release / "grader").is_dir()
            assert expected_item_ids == ("TASK1",)
            return cls()

        def for_item(self, item_id: str) -> ValidatedScoringKeyV1:
            assert item_id == "TASK1"
            return key

    monkeypatch.setattr(grade_trialeval, "ScoringKeyStoreV1", _ScoringKeyStore)
    monkeypatch.setattr(grade_trialeval, "discover_items", lambda _: [item])
    monkeypatch.setattr(
        grade_trialeval,
        "read_assumption_evidence_domains",
        lambda **_: {"TASK1": object()},
    )

    graded = tmp_path / "graded"
    assert (
        grade_trialeval.grade_trialeval_run([str(run_dir), "--suite-dir", str(suite), "--out-dir", str(graded)]) == 0
    )
    scored = read_json_model(TrialEvalItemResultV1, graded / "items" / "TASK1.json")
    assert scored.scores is not None
    assert scored.scores.grade.failure_codes == ("missing_primary_submission",)
    assert scored.scores.grade.passed is False
    assert scored.scores.credit_eligible_route_count == 1
    assert scored.scores.planning.applicable is False
    summary = json.loads((graded / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_single_route_items"] == 1
    assert summary["n_plural_route_items"] == 0

    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    with ZipFile(archive_root / "TrialEvalBench_participant.zip", "w") as archive:
        archive.write(public / "manifest.json", "manifest.json")
    with ZipFile(archive_root / "TrialEvalBench_evaluator.zip", "w") as archive:
        archive.writestr("grader/item_index.json", "{}\n")
    graded_from_archives = tmp_path / "graded-from-archives"

    assert (
        grade_trialeval.grade_trialeval_run(
            [
                str(run_dir),
                "--suite-dir",
                str(archive_root / "TrialEvalBench_evaluator.zip"),
                "--out-dir",
                str(graded_from_archives),
            ]
        )
        == 0
    )
    assert (graded_from_archives / "GRADE_MANIFEST.json").is_file()


def test_staged_directory_publishes_complete_output(tmp_path: Path) -> None:
    destination = tmp_path / "graded"

    with staged_directory(destination) as staging:
        (staging / "GRADE_MANIFEST.json").write_text("{}", encoding="utf-8")
        assert not destination.exists()

    assert (destination / "GRADE_MANIFEST.json").is_file()


def test_staged_directory_removes_failed_output(tmp_path: Path) -> None:
    destination = tmp_path / "graded"

    with pytest.raises(RuntimeError, match="scoring failed"):
        with staged_directory(destination) as staging:
            (staging / "partial.json").write_text("{}", encoding="utf-8")
            raise RuntimeError("scoring failed")

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".graded.*"))


def test_staged_directory_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "graded"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        with staged_directory(destination):
            pytest.fail("existing output must not be entered")


def test_trialdev_offline_grader_preserves_zero_credit_noncompletion(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "scenario_s01" / "public").mkdir(parents=True)
    _write_program_loop_manifest(bundle)
    program = tmp_path / "run" / "programs" / "s01__benefit_risk"
    write_json_model(
        program / "chain_summary.json",
        TrialDevChainSummaryV1(
            program_id="s01__benefit_risk",
            scenario_id="s01",
            objective_id="benefit_risk",
            execution_status="model_turn_limit",
            error="AgentTurnLimitExceeded: model did not submit",
            obs_review_path_stats={"turns": 1, "execute_code": 1},
            materialization_usage=TrialDevMaterializationUsageV1(),
        ),
    )

    assert grade_program(program, bundle=bundle) == (False, False)


def test_trialdev_offline_grader_replays_materialized_partial_programme(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trialagentbench_harness.adapters import trialdev_upstream

    bundle = tmp_path / "bundle"
    (bundle / "scenario_s01" / "public").mkdir(parents=True)
    _write_program_loop_manifest(bundle)
    program = tmp_path / "run" / "programs" / "s01__benefit_risk"
    (program / "agent_workdir").mkdir(parents=True)
    write_json_model(
        program / "chain_summary.json",
        TrialDevChainSummaryV1(
            program_id="s01__benefit_risk",
            scenario_id="s01",
            objective_id="benefit_risk",
            execution_status="model_turn_limit",
            error="AgentTurnLimitExceeded: analysis was not submitted",
            phases_attempted=[
                TrialDevPhaseAttemptSummaryV1(
                    phase_id="phase1",
                    n_materializations=1,
                )
            ],
            materialization_usage=TrialDevMaterializationUsageV1(materialize_calls_by_phase={"phase1": 1}),
        ),
    )
    resources = TrialDevProgrammeResourceConsequenceV1(
        total_participants=0,
        total_protocol_follow_up_days=0,
        total_enrollment_window_days=0,
        total_site_phase_budget=0,
        total_planned_phase_duration_days=0,
        total_participant_follow_up_days=0,
        participant_excess_vs_minimum=0,
        participant_shortage_vs_minimum=0,
        follow_up_excess_days_vs_minimum=0,
        follow_up_shortage_days_vs_minimum=0,
        statistically_inadequate_phases=0,
        operationally_infeasible_phases=0,
        dominated_phases=0,
        design_avoidable_participants_min=0,
        design_avoidable_participants_max=0,
        design_avoidable_follow_up_days_min=0,
        design_avoidable_follow_up_days_max=0,
        design_avoidable_participant_follow_up_days_min=0,
        design_avoidable_participant_follow_up_days_max=0,
        late_continuation_participants=0,
        late_continuation_protocol_follow_up_days=0,
        late_continuation_enrollment_window_days=0,
        late_continuation_site_phase_budget=0,
        late_continuation_participant_follow_up_days=0,
    )
    trajectory_grade = TrialDevTrajectoryGradeV1(
        terminal_status="invalid",
        trajectory_primary_score=0.0,
        trajectory_decision_score=0.0,
        n_invalid_attempts=1,
        invalid_attempt_reasons=["invalid_analysis"],
        terminal_summary=TrialDevTerminalSummaryV1(
            scenario_id="s01",
            terminal_status="invalid",
            final_program_success=False,
        ),
        resource_consequence=resources,
        payload={},
    )

    def _grade_trajectory(**kwargs: object) -> TrialDevTrajectoryGradeV1:
        write_json_model(Path(str(kwargs["out_path"])), trajectory_grade)
        return trajectory_grade

    monkeypatch.setattr(trialdev_upstream, "grade_trajectory", _grade_trajectory)

    assert grade_program(program, bundle=bundle) == (False, True)
    chain = read_json_model(TrialDevChainSummaryV1, program / "chain_summary.json")
    assert chain.trajectory_grade_path == "trajectory_grade.json"
    assert chain.trajectory_metrics.n_invalid_attempts == 1
    assert chain.trajectory_metrics.invalid_attempt_reasons == ["invalid_analysis"]
