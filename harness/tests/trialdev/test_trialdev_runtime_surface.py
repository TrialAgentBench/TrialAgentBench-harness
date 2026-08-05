from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.release.trialdev_runtime_surface import (
    TRIALDEV_PUBLIC_FILE_ROLES,
    TrialDevPublicMemberRoleV1,
    classify_trialdev_participant_archive_member,
    classify_trialdev_public_member,
)
from trialagentbench_harness.ports import CodeExecutionResultV1, ToolCall
from trialagentbench_harness.trialdev import prompts as prompts_mod
from trialagentbench_harness.trialdev import runner as runner_mod
from trialagentbench_harness.trialdev.agent import (
    AgentLoop,
    runtime_submission_contracts,
    tools_for_obs_review,
    tools_for_phase_analysis,
    tools_for_phase_request,
    write_runtime_submission_contracts,
)
from trialagentbench_harness.trialdev.data import stage_working_dir
from trialagentbench_harness.trialdev.prompts import (
    build_obs_review_block,
    build_phase_analysis_block,
    build_system_prompt,
    get_phase_module,
)
from trialagentbench_harness.trialdev.runner import RunOptions, _build_obs_review_submission_for_grader
from trialagentbench_harness.trialdev.schema import BenchmarkItem, MaterializationUsage, Program
from trialagentbench_harness.trialdev.share.public_method_design import (
    TrialDevPublicObservationalAnalysisSpecV1,
)
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentObservationalReviewSubmissionV1,
    TrialDevelopmentPhaseActionSpecV1,
    TrialDevelopmentPhaseAnalysisSubmissionV1,
)


def _observational_specification() -> TrialDevPublicObservationalAnalysisSpecV1:
    return TrialDevPublicObservationalAnalysisSpecV1(
        schema_id="trialdev_public_observational_analysis_spec_v1",
        phase_id="observational_review",
        method_route_id=("trialdev.observational.multinomial_propensity_weighted_stratified_aalen_johansen.v1"),
        calculator_id="public_observational_ipw_utility_v1",
        primary_estimator_id="multinomial_propensity_weighted_stratified_aalen_johansen",
        adjustment_covariates=("baseline",),
        analysis_population="complete_on_declared_adjustment_covariates",
        categorical_encoding="reference_level_one_hot",
        sensitivity_estimator_ids=("raw_observed",),
        uncertainty_estimator_id="refitted_nuisance_participant_nonparametric_bootstrap",
        uncertainty_kind="two_sided_confidence_interval",
        confidence_level=0.95,
        effect_scale_id="dimensionless_declared_net_benefit",
        horizon_source="objective.efficacy_endpoints[].horizon_days",
        bootstrap_replicates=100,
        bootstrap_seed=20260802,
        bootstrap_rng_id="numpy_default_rng_pcg64",
        bootstrap_standard_error_ddof=1,
        confidence_interval_id="normal_critical_value_times_bootstrap_standard_error",
        identification_assumptions=("conditional_exchangeability",),
        rationale="Prospective participant-visible method.",
        exact_stratification_covariates=(),
        quantile_stratification_bins={},
        propensity_solver_id="deterministic_multinomial_logit_lbfgs",
        propensity_max_iterations=100,
        propensity_tolerance=1e-8,
        propensity_l2_penalty=0.0,
    )


class _Session:
    def execute(self, code: str) -> str:
        return code

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        return CodeExecutionResultV1(status="success", output=code, elapsed_seconds=0.1)

    def close(self) -> None:
        return None


def _agent_loop_for_tool_test(tmp_path: Path) -> AgentLoop:
    loop = object.__new__(AgentLoop)
    loop.session = _Session()
    loop.workdir = tmp_path
    loop.verbose = False
    loop.max_tool_output_chars = 4000
    return loop


def _scenario(root: Path) -> Path:
    public = root / "scenario_s01" / "public"
    public.mkdir(parents=True)
    for name in TRIALDEV_PUBLIC_FILE_ROLES:
        (public / name).write_text("{}", encoding="utf-8")
    (public / "study_brief.md").write_text("study", encoding="utf-8")
    (public / "trial_request_schema.json").write_text(
        json.dumps({"$defs": {"Request": {"type": "object"}}, "$ref": "#/$defs/Request"}),
        encoding="utf-8",
    )
    (public / "observational_extract.parquet").write_bytes(b"data")
    return public


def _observational_submission_payload() -> dict[str, object]:
    return {
        "response_branch": "estimable",
        "primary_resolution_evidence_class": "empirical_diagnosis",
        "ranked_drug_ids": ["drug_a"],
        "candidate_utility_estimates": [
            {
                "evidence_id": "utility-drug-a",
                "candidate_drug_id": "drug_a",
                "objective_id": "benefit_risk",
                "estimator_id": "entropy_balanced_standardized_aalen_johansen",
                "estimate": 0.2,
                "lower": 0.1,
                "upper": 0.3,
                "confidence_level": 0.95,
                "analysis_covariate_ids": ["AGE"],
                "source_artifact_checksums": {"observational_extract.parquet": "a" * 64},
            }
        ],
        "supporting_evidence_ids": ["utility-drug-a"],
        "candidate_drug_id": "drug_a",
        "decision_action": "nominate_for_early_study",
        "decision_rationale": "The adjusted estimate supports nomination.",
    }


def test_trialdev_member_roles_are_exact() -> None:
    assert (
        classify_trialdev_public_member("observational_extract.parquet")
        == TrialDevPublicMemberRoleV1.OBSERVATIONAL_DATA
    )
    assert classify_trialdev_public_member("phase_design_policy.json") == TrialDevPublicMemberRoleV1.DECISION_POLICY
    for name in (
        "checkpoint_outcome_schema.json",
        "policy_binding_schema.json",
        "portfolio_action_selection_schema.json",
        "portfolio_checkpoint_action_policy_schema.json",
        "portfolio_programme_state_schema.json",
        "resource_schedule_schema.json",
        "single_asset_action_selection_schema.json",
        "single_asset_checkpoint_action_policy_schema.json",
        "single_asset_programme_state_schema.json",
    ):
        assert classify_trialdev_public_member(name) == TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT
    with pytest.raises(ValueError, match="Unknown TrialDev public scenario member"):
        classify_trialdev_public_member("grader_notes.json")
    with pytest.raises(ValueError, match="must be flat"):
        classify_trialdev_public_member("future/phase3.parquet")
    assert (
        classify_trialdev_participant_archive_member("benchmark_suite_manifest.json")
        == TrialDevPublicMemberRoleV1.SUITE_MANIFEST
    )
    assert (
        classify_trialdev_participant_archive_member("distribution_mode_participant_manifest.json")
        == TrialDevPublicMemberRoleV1.DISTRIBUTION_MANIFEST
    )
    assert (
        classify_trialdev_participant_archive_member("docs/QUICKSTART.md") == TrialDevPublicMemberRoleV1.DOCUMENTATION
    )
    with pytest.raises(ValueError, match="Unknown TrialDev participant archive member"):
        classify_trialdev_participant_archive_member("docs/internal_notes.md")


def test_trialdev_staging_preserves_published_schema_bytes(tmp_path: Path) -> None:
    public = _scenario(tmp_path / "bundle")

    staged = stage_working_dir(
        tmp_path / "bundle",
        "s01",
        tmp_path / "work",
        procedure_assistance="unordered_checklist",
    )

    assert (staged / "trial_request_schema.json").read_bytes() == (public / "trial_request_schema.json").read_bytes()


def test_output_contract_only_staging_includes_prospective_method_contracts(tmp_path: Path) -> None:
    public = _scenario(tmp_path / "bundle")

    staged = stage_working_dir(
        tmp_path / "bundle",
        "s01",
        tmp_path / "work",
        procedure_assistance="output_contract_only",
    )

    assert (staged / "phase_analysis_method_catalog.json").is_file()
    assert (staged / "observational_method_catalog.json").is_file()
    assert (staged / "objective_charter.json").read_bytes() == (public / "objective_charter.json").read_bytes()
    assert (staged / "decision_charter.json").is_file()
    assert (staged / "phase_design_policy.json").is_file()


def test_assistance_staging_preserves_identical_participant_files(tmp_path: Path) -> None:
    public = _scenario(tmp_path / "bundle")
    charter = stage_working_dir(
        tmp_path / "bundle",
        "s01",
        tmp_path / "charter",
        procedure_assistance="output_contract_only",
    )
    assisted = stage_working_dir(
        tmp_path / "bundle",
        "s01",
        tmp_path / "assisted",
        procedure_assistance="unordered_checklist",
    )

    assert {path.name for path in assisted.iterdir()} == {path.name for path in charter.iterdir()}
    for name in TRIALDEV_PUBLIC_FILE_ROLES:
        assert (charter / name).read_bytes() == (assisted / name).read_bytes() == (public / name).read_bytes()


def test_trialdev_assistance_prompts_change_only_checklist_ordering(tmp_path: Path) -> None:
    _scenario(tmp_path / "bundle")
    program = Program(
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        items_by_phase={},
    )

    charter = build_system_prompt(
        program,
        stage_working_dir(
            tmp_path / "bundle",
            "s01",
            tmp_path / "charter",
            procedure_assistance="output_contract_only",
        ),
        max_turns_per_step=25,
        procedure_assistance="output_contract_only",
    )
    contract = build_system_prompt(
        program,
        stage_working_dir(
            tmp_path / "bundle",
            "s01",
            tmp_path / "contract",
            procedure_assistance="unordered_checklist",
        ),
        max_turns_per_step=25,
        procedure_assistance="unordered_checklist",
    )
    sop = build_system_prompt(
        program,
        stage_working_dir(
            tmp_path / "bundle",
            "s01",
            tmp_path / "sop",
            procedure_assistance="ordered_sop",
        ),
        max_turns_per_step=25,
        procedure_assistance="ordered_sop",
    )

    assert "You are the statistical lead for one clinical development programme" in charter
    assert "CLINICAL QUESTION" in charter
    assert "EVIDENCE AVAILABLE" in charter
    assert "WORK REQUIRED" in charter
    assert "CONCLUSION" in charter
    assert "Choose and execute a defensible analysis" in charter
    assert "synthetic" not in charter.lower()
    assert "observational_method_catalog.json" in charter
    assert "phase_analysis_method_catalog.json" in charter
    assert "observational_method_catalog.json" in contract
    assert "phase_analysis_method_catalog.json" in contract
    assert "neutral prospective\nmethod catalog" in contract
    assert "operations in any order" in contract
    assert "required order" not in contract
    assert "same operations in this required order" in sop
    assert "do not prescribe\na conclusion or action" in sop
    assert "do not prescribe\na conclusion or action" in contract
    assert "method-route\nidentifier must agree" in contract
    assert "method-route\nidentifier must agree" in sop
    assert all(contract.count(component) == 1 for component in prompts_mod._TRIALDEV_ANALYSIS_COMPONENTS)
    assert all(sop.count(component) == 1 for component in prompts_mod._TRIALDEV_ANALYSIS_COMPONENTS)
    assert "treatment-minus-control excess risk" not in charter
    assert "each with its own confidence" not in charter
    shared_operations = (
        "inspect current state, prior evidence, and the prospective design contract",
        "submit a legal phase design and inspect the participant-level, endpoint, and safety records",
        "define the estimand, assess identification, design structure, data integrity, and relevant model assumptions",
        "submit effect and safety evidence with uncertainty and identify the records used",
        "choose an admissible action supported by that evidence",
    )
    for operation in shared_operations:
        assert operation in contract
        assert operation in sop


def test_observational_specification_prompt_prescribes_method_without_result(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    (public / "observational_analysis_specification.json").write_text("{}", encoding="utf-8")
    program = Program(
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        items_by_phase={},
    )

    prompt = build_system_prompt(
        program,
        public,
        max_turns_per_step=25,
        procedure_assistance="output_contract_only",
        observational_analysis_specification=_observational_specification(),
    )

    assert "execute the complete prospective method" in prompt
    assert "Do not substitute another method" in prompt
    assert "contains no fitted result" in prompt


def test_observational_specification_is_restricted_to_unassisted_bounded_runs(tmp_path: Path) -> None:
    specification = _observational_specification()
    with pytest.raises(ValueError, match="observational_review_only"):
        RunOptions(
            bundle_root=tmp_path,
            output_root=tmp_path,
            model="model",
            observational_analysis_specification=specification,
        )
    with pytest.raises(ValueError, match="output_contract_only"):
        RunOptions(
            bundle_root=tmp_path,
            output_root=tmp_path,
            model="model",
            execution_scope="observational_review_only",
            procedure_assistance="ordered_sop",
            observational_analysis_specification=specification,
        )


@pytest.mark.parametrize(
    ("unsupported_nomination", "expected_status", "expected_advance_calls"),
    [
        (False, "completed", 1),
        (True, "model_invalid_submission", 0),
    ],
)
def test_observational_only_runner_preserves_nomination_without_materializing(
    unsupported_nomination: bool,
    expected_status: str,
    expected_advance_calls: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = Program(
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        items_by_phase={
            "observational_review": (
                BenchmarkItem(
                    item_id="item",
                    scenario_id="s01",
                    phase_id="observational_review",
                    objective_id="benefit_risk",
                    endpoint_id=None,
                    task_definition_id="observational_review",
                ),
            )
        },
    )
    output = tmp_path / "output"
    specification = _observational_specification()

    def stage(*_args: object, **kwargs: object) -> Path:
        destination = Path(_args[2])
        destination.mkdir(parents=True)
        return destination

    class Loop:
        messages: list[object] = []
        session = None

        def close(self) -> None:
            return None

    monkeypatch.setattr(runner_mod, "scenario_root", lambda *_args: tmp_path / "scenario_s01")
    monkeypatch.setattr(runner_mod, "stage_working_dir", stage)
    monkeypatch.setattr(runner_mod.agent_mod, "write_runtime_submission_contracts", lambda *_args: None)
    monkeypatch.setattr(
        runner_mod.bridge,
        "load_action_policy",
        lambda *_args: SimpleNamespace(
            action_specs=[SimpleNamespace(phase_id=phase) for phase in ("phase1", "phase2", "phase3")]
        ),
    )
    monkeypatch.setattr(runner_mod.prompts, "build_system_prompt", lambda *_args, **_kwargs: "prompt")
    loop_options: dict[str, object] = {}

    def build_loop(**kwargs: object) -> Loop:
        loop_options.update(kwargs)
        return Loop()

    monkeypatch.setattr(runner_mod.agent_mod, "AgentLoop", build_loop)
    monkeypatch.setattr(
        runner_mod.trialdev_upstream,
        "build_initial_program_state",
        lambda *, out_path, **_kwargs: Path(out_path).write_text("{}", encoding="utf-8"),
    )
    observational_submission = object()
    monkeypatch.setattr(runner_mod, "_run_obs_review", lambda **_kwargs: observational_submission)
    monkeypatch.setattr(
        runner_mod,
        "_unsupported_observational_nomination",
        lambda **_kwargs: unsupported_nomination,
    )
    persisted_statuses: list[str] = []

    def persist_summary(_program_dir: Path, run: object, _usage: object) -> None:
        persisted_statuses.append(str(run.execution_status))

    monkeypatch.setattr(runner_mod, "_persist_chain_summary", persist_summary)
    monkeypatch.setattr(runner_mod, "_persist_conversation", lambda *_args: None)
    initial_state = object()
    monkeypatch.setattr(runner_mod.trialdev_upstream, "load_program_state", lambda *_args: initial_state)
    advance_calls = 0

    def advance_observational_state(**_kwargs: object) -> SimpleNamespace:
        nonlocal advance_calls
        advance_calls += 1
        return SimpleNamespace(terminal_disposition="active")

    monkeypatch.setattr(
        runner_mod.trialdev_upstream,
        "advance_observational_programme_state",
        advance_observational_state,
    )

    run = runner_mod.run_program(
        program,
        options=RunOptions(
            bundle_root=tmp_path / "bundle",
            output_root=output,
            model="model",
            procedure_assistance="output_contract_only",
            execution_scope="observational_review_only",
            observational_analysis_specification=specification,
        ),
        provider=object(),  # type: ignore[arg-type]
    )

    assert run.stopped_at_phase == "observational_review"
    assert run.phases == []
    assert run.execution_status == expected_status
    assert persisted_statuses == [expected_status]
    assert advance_calls == expected_advance_calls
    assert loop_options["tool_choice"] == "auto"
    persisted = (
        output / "programs" / program.program_id / "agent_workdir" / "observational_analysis_specification.json"
    )
    assert (
        TrialDevPublicObservationalAnalysisSpecV1.model_validate_json(persisted.read_text(encoding="utf-8"))
        == specification
    )


@pytest.mark.parametrize(
    ("decision_action", "candidate_id", "replay_available", "expected"),
    [
        ("withhold_nomination", None, False, False),
        ("nominate_for_early_study", "regimen_a", True, False),
        ("nominate_for_early_study", "regimen_a", False, True),
    ],
)
def test_unsupported_observational_nomination_is_a_model_outcome(
    decision_action: str,
    candidate_id: str | None,
    replay_available: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = SimpleNamespace(
        program_decision=SimpleNamespace(
            decision_action=decision_action,
            recommended_drug_id=candidate_id,
        )
    )
    monkeypatch.setattr(
        runner_mod,
        "_fixed_phase_replay_available",
        lambda **_kwargs: replay_available,
    )

    assert (
        runner_mod._unsupported_observational_nomination(  # type: ignore[arg-type]
            submission=submission,
            scenario_root=Path("scenario_s01"),
        )
        is expected
    )


def test_materialization_correction_exhaustion_is_a_terminal_model_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = BenchmarkItem(
        item_id="s01_phase1_benefit_risk",
        scenario_id="s01",
        phase_id="phase1",
        objective_id="benefit_risk",
        endpoint_id=None,
        task_definition_id="phase1",
    )
    program = Program(
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        items_by_phase={"phase1": (item,)},
    )
    output = tmp_path / "output"

    def stage(*_args: object, **_kwargs: object) -> Path:
        destination = Path(_args[2])
        destination.mkdir(parents=True)
        return destination

    class Loop:
        messages: list[object] = []
        session = None

        def close(self) -> None:
            return None

    monkeypatch.setattr(runner_mod, "scenario_root", lambda *_args: tmp_path / "scenario_s01")
    monkeypatch.setattr(runner_mod, "stage_working_dir", stage)
    monkeypatch.setattr(runner_mod.agent_mod, "write_runtime_submission_contracts", lambda *_args: None)
    monkeypatch.setattr(
        runner_mod.bridge,
        "load_action_policy",
        lambda *_args: SimpleNamespace(
            action_specs=[SimpleNamespace(phase_id=phase) for phase in ("phase1", "phase2", "phase3")]
        ),
    )
    monkeypatch.setattr(runner_mod.prompts, "build_system_prompt", lambda *_args, **_kwargs: "prompt")
    materialization_loop_options: dict[str, object] = {}

    def build_materialization_loop(**kwargs: object) -> Loop:
        materialization_loop_options.update(kwargs)
        return Loop()

    monkeypatch.setattr(runner_mod.agent_mod, "AgentLoop", build_materialization_loop)
    monkeypatch.setattr(
        runner_mod.trialdev_upstream,
        "build_initial_program_state",
        lambda *, out_path, **_kwargs: Path(out_path).write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        runner_mod.trialdev_upstream,
        "load_program_state",
        lambda *_args: SimpleNamespace(
            terminal_disposition="active",
            current_checkpoint_id="early_safety_study",
        ),
    )

    def reject(**_kwargs: object) -> object:
        raise runner_mod.trialdev_upstream.TrialMaterializationRejectedError(
            "No fixed randomized evidence exists for this action"
        )

    monkeypatch.setattr(runner_mod, "_run_one_phase", reject)
    monkeypatch.setattr(runner_mod, "_persist_conversation", lambda *_args: None)

    run = runner_mod.run_program(
        program,
        options=RunOptions(
            bundle_root=tmp_path / "bundle",
            output_root=output,
            model="model",
            tool_choice="required",
        ),
        provider=object(),  # type: ignore[arg-type]
    )

    assert run.execution_status == "model_invalid_submission"
    assert run.error is not None
    assert "TrialMaterializationRejectedError" in run.error
    assert materialization_loop_options["tool_choice"] == "required"
    summary = json.loads((output / "programs" / program.program_id / "chain_summary.json").read_text(encoding="utf-8"))
    assert summary["execution_status"] == "model_invalid_submission"


def test_charter_observational_prompt_defers_scientific_fields_to_runtime_contract() -> None:
    block = build_obs_review_block()

    assert "runtime submission" in block
    assert "submit_obs_review_analysis_and_decision_file" in block
    assert "submit_obs_review_analysis_and_decision``" in block
    assert "exact adjustment covariates" not in block
    assert "diagnostics" not in block
    assert "complete permutation" not in block
    assert "checksum" not in block
    assert "causal ranking" in block


def test_observational_prompt_provenance_map_is_derived_from_the_public_objective(tmp_path: Path) -> None:
    public = tmp_path / "scenario_s01" / "public"
    public.mkdir(parents=True)
    (public / "observational_extract.parquet").write_bytes(b"participant evidence")
    (public / "objective_charter.json").write_text(
        json.dumps(
            {
                "objectives": [
                    {
                        "objective_id": "benefit_risk",
                        "public_evidence_basis": [
                            "public/objective_charter.json",
                            "public/observational_extract.parquet",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    checksums = runner_mod.bridge.observational_source_artifact_checksums(
        tmp_path / "scenario_s01",
        objective_id="benefit_risk",
    )

    assert set(checksums) == {
        "public/objective_charter.json",
        "public/observational_extract.parquet",
    }
    assert all(len(checksum) == 64 for checksum in checksums.values())


def test_trialdev_staging_rejects_unknown_public_members(tmp_path: Path) -> None:
    public = _scenario(tmp_path / "bundle")
    (public / "diagnostic_reference_route.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown TrialDev public scenario member"):
        stage_working_dir(
            tmp_path / "bundle",
            "s01",
            tmp_path / "work",
            procedure_assistance="unordered_checklist",
        )


@pytest.mark.parametrize(
    "tool_call",
    [
        ToolCall(id="1", name="unknown", arguments="{}"),
        ToolCall(id="2", name="execute_code", arguments='{"code": 3}'),
        ToolCall(id="3", name="execute_code", arguments='{"code": "", "purpose": 1}'),
        ToolCall(id="4", name="inspect_parquet", arguments='{"path": "../grader/truth.parquet"}'),
        ToolCall(id="5", name="inspect_parquet", arguments='{"path": "table.csv"}'),
        ToolCall(id="6", name="inspect_parquet", arguments="[]"),
        ToolCall(id="7", name="write_workspace_file", arguments='{"path": "../truth.json", "content": "x"}'),
        ToolCall(id="8", name="read_workspace_file", arguments='{"path": "x", "start_line": 2, "end_line": 1}'),
    ],
)
def test_trialdev_local_tools_reject_invalid_calls(tool_call: ToolCall, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _agent_loop_for_tool_test(tmp_path)._dispatch_local_tool(tool_call)


def test_trialdev_inspect_parquet_accepts_contained_relative_path(tmp_path: Path) -> None:
    output = _agent_loop_for_tool_test(tmp_path)._dispatch_local_tool(
        ToolCall(id="1", name="inspect_parquet", arguments='{"path": "phase_phase2/trial_output/endpoints.parquet"}')
    )

    assert "_pd.read_parquet('phase_phase2/trial_output/endpoints.parquet')" in output.output


def test_phase_analysis_prompt_uses_only_participant_relative_output_path(tmp_path: Path) -> None:
    """Host paths must not enter the participant-visible phase prompt."""

    host_output = tmp_path / "private-run/programs/p1/agent_workdir/phase_phase2/trial_output"
    prompt = build_phase_analysis_block(
        phase_id="phase2",
        trial_output_summary={
            "trial_output_root": str(host_output),
            "trial_output_relpath": "phase_phase2/trial_output",
            "n_participants": 120,
            "request_checksum": "a" * 64,
            "trial_output_checksum": "b" * 64,
            "effect_source_artifact_checksums": {
                "trial_output/endpoints.parquet": "c" * 64,
            },
            "safety_source_artifact_checksums": {
                "trial_output/safety.parquet": "d" * 64,
            },
        },
    )

    assert str(tmp_path) not in prompt
    assert "phase_phase2/trial_output" in prompt
    assert "checksum" not in prompt
    assert "submit_phase_analysis_file" in prompt
    assert "under ``scratch/``" in prompt


def test_phase_analysis_prompt_rejects_missing_relative_output_path(tmp_path: Path) -> None:
    """An absolute host path cannot serve as a participant-path fallback."""

    with pytest.raises(ValueError, match="participant-relative"):
        build_phase_analysis_block(
            phase_id="phase2",
            trial_output_summary={
                "trial_output_root": str(tmp_path / "private-run/phase_phase2/trial_output"),
                "n_participants": 120,
                "request_checksum": "a" * 64,
                "trial_output_checksum": "b" * 64,
            },
        )


@pytest.mark.parametrize(
    "field",
    [
        "max_tokens",
        "max_context_chars",
        "max_turns_per_step",
        "max_phase_retries",
        "program_watchdog_seconds",
    ],
)
def test_trialdev_run_options_reject_nonpositive_execution_budgets(tmp_path: Path, field: str) -> None:
    values = {
        "bundle_root": tmp_path / "bundle",
        "output_root": tmp_path / "output",
        "model": "test-model",
        field: 0,
    }
    with pytest.raises(ValueError, match=field):
        RunOptions(**values)


def test_trialdev_runner_refuses_existing_program_directory(tmp_path: Path) -> None:
    program = Program(
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        items_by_phase={},
    )
    output_root = tmp_path / "output"
    program_dir = output_root / "programs" / program.program_id
    program_dir.mkdir(parents=True)
    sentinel = program_dir / "custody.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner_mod.run_program(
            program,
            options=RunOptions(bundle_root=tmp_path / "bundle", output_root=output_root, model="model"),
            provider=object(),  # type: ignore[arg-type]
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_trialdev_runner_delegates_new_workdir_creation_to_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = Program(
        program_id="s01__benefit_risk",
        scenario_id="s01",
        objective_id="benefit_risk",
        items_by_phase={},
    )
    output_root = tmp_path / "output"

    def assert_uncreated_workdir(
        _bundle_root: Path,
        _scenario_id: str,
        dest_root: Path,
        *,
        procedure_assistance: str,
    ) -> Path:
        assert not dest_root.exists()
        assert procedure_assistance == "output_contract_only"
        raise RuntimeError("staging boundary reached")

    monkeypatch.setattr(runner_mod, "scenario_root", lambda *_: tmp_path / "scenario_s01")
    monkeypatch.setattr(runner_mod, "stage_working_dir", assert_uncreated_workdir)

    with pytest.raises(RuntimeError, match="staging boundary reached"):
        runner_mod.run_program(
            program,
            options=RunOptions(bundle_root=tmp_path / "bundle", output_root=output_root, model="model"),
            provider=object(),  # type: ignore[arg-type]
        )

    assert (output_root / "programs" / program.program_id).is_dir()
    assert not (output_root / "programs" / program.program_id / "agent_workdir").exists()


def test_incomplete_materialization_is_archived_outside_agent_workdir(tmp_path: Path) -> None:
    program_dir = tmp_path / "program"
    trial_output = program_dir / "agent_workdir" / "phase_phase1" / "trial_output"
    trial_output.mkdir(parents=True)
    (trial_output / "execution_summary.json").write_text('{"feasibility_status":"rejected"}', encoding="utf-8")

    archive = runner_mod._archive_incomplete_materialization(
        trial_output_root=trial_output,
        program_dir=program_dir,
        phase_id="phase1",
    )

    assert archive == program_dir / "materialization_attempts" / "phase1" / "attempt-1"
    assert not trial_output.exists()
    assert (archive / "execution_summary.json").read_text(encoding="utf-8") == ('{"feasibility_status":"rejected"}')
    assert "agent_workdir" not in archive.relative_to(program_dir).parts


@pytest.mark.parametrize(
    "artifact_relative_path",
    [
        "/materialization_attempts/attempt-1",
        "../attempt-1",
        "materialization_attempts/../attempt-1",
    ],
)
def test_checkpoint_violation_rejects_unsafe_archive_paths(artifact_relative_path: str) -> None:
    with pytest.raises(ValueError, match="normalized relative paths"):
        runner_mod._runtime_violation(
            {
                "phase_id": "phase1",
                "kind": "materialize_rejection",
                "error": "insufficient_enrollment_window_support",
                "artifact_relative_path": artifact_relative_path,
            }
        )


def test_checkpoint_violation_rejects_archive_for_schema_failure() -> None:
    with pytest.raises(ValueError, match="Only materialization rejections"):
        runner_mod._runtime_violation(
            {
                "phase_id": "phase1",
                "kind": "schema_validation",
                "error": "invalid request",
                "artifact_relative_path": "materialization_attempts/phase1/attempt-1",
            }
        )


def test_materialization_retry_continues_same_design_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Loop:
        def __init__(self) -> None:
            self.begin_calls = 0
            self.user_messages: list[str] = []
            self.run_calls = 0
            self.tool_replies: list[str] = []

        def begin_step(self, *, phase_id: str, step_id: str) -> None:
            assert (phase_id, step_id) == ("phase1", "trial_design_request")
            self.begin_calls += 1

        def append_user_message(self, content: str) -> None:
            self.user_messages.append(content)

        def run_until_submit(self, *, tools: list[dict], submit_tool_names: set[str]) -> SimpleNamespace:
            assert submit_tool_names == {"submit_phase_request", "submit_phase_request_file"}
            self.run_calls += 1
            return SimpleNamespace(payload={}, tool_call_id=f"call-{self.run_calls}", name="submit_phase_request")

        def append_tool_reply(self, tool_call_id: str, content: str, *, tool_name: str) -> None:
            self.tool_replies.append(content)

    request = SimpleNamespace(
        endpoint_id="endpoint",
        selection_objective="objective",
        target_sample_size=40,
        follow_up_days=28,
        allocation_ratio="1:1",
        site_count_budget=4,
        enrollment_window_days=42,
        checksum=lambda: "a" * 64,
    )
    monkeypatch.setattr(runner_mod.agent_mod, "tools_for_phase_request", lambda **_: [])
    monkeypatch.setattr(runner_mod.bridge, "parse_request", lambda _payload, **_context: (request, None))
    monkeypatch.setattr(
        runner_mod.bridge,
        "write_request",
        lambda _request, path: path.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(runner_mod.bridge, "write_rationale_sidecar", lambda **_: None)
    monkeypatch.setattr(runner_mod.prompts, "build_phase_request_block", lambda **_: "design prompt")

    phase_dir = tmp_path / "phase"
    phase_dir.mkdir()
    loop = _Loop()
    common = {
        "phase_id": "phase1",
        "phase_dir": phase_dir,
        "loop": loop,
        "usage": MaterializationUsage(),
        "state_summary": {},
        "phase_module": {},
        "prior_phase_summaries": [],
        "program_id": "program",
        "scenario_id": "scenario",
        "program_objective": "objective",
    }

    runner_mod._drive_phase_request(**common, start_step=True)
    loop.append_user_message("MATERIALIZATION REJECTED")
    runner_mod._drive_phase_request(**common, start_step=False)

    assert loop.begin_calls == 1
    assert loop.user_messages == ["design prompt", "MATERIALIZATION REJECTED"]
    assert loop.run_calls == 2


def test_obs_review_candidate_roles_do_not_depend_on_identifier_spelling(tmp_path: Path) -> None:
    public = tmp_path / "scenario_s01" / "public"
    public.mkdir(parents=True)
    (public / "candidate_drug_catalog.json").write_text(
        json.dumps(
            {
                "candidate_drugs": [
                    {"candidate_drug_id": "usual_care", "role": "control"},
                    {"candidate_drug_id": "control", "role": "investigational"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (public / "objective_charter.json").write_text(
        json.dumps(
            {
                "objectives": [
                    {
                        "objective_id": "benefit_risk",
                        "public_evidence_basis": ["public/observational_extract.parquet"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (public / "observational_extract.parquet").write_bytes(b"participant evidence")
    (public / "observational_method_catalog.json").write_text("{}", encoding="utf-8")
    out_path = tmp_path / "submission.json"

    _build_obs_review_submission_for_grader(
        program=Program(
            program_id="s01__benefit_risk",
            scenario_id="s01",
            objective_id="benefit_risk",
            items_by_phase={},
        ),
        scenario_root=public.parent,
        agent_payload={
            "response_branch": "estimable",
            "primary_resolution_evidence_class": "empirical_diagnosis",
            "ranked_drug_ids": ["control"],
            "candidate_drug_id": "control",
            "decision_action": "nominate_for_early_study",
            "decision_rationale": "The adjusted utility estimate supports nomination.",
            "candidate_utility_estimates": [
                {
                    "evidence_id": "utility-control",
                    "method_route_id": (
                        "trialdev.observational.multinomial_propensity_weighted_stratified_aalen_johansen.v1"
                    ),
                    "candidate_drug_id": "control",
                    "objective_id": "benefit_risk",
                    "estimator_id": "adjusted_outcome_model",
                    "utility_unit": "dimensionless_declared_net_benefit",
                    "estimate": 0.1,
                    "lower": 0.0,
                    "upper": 0.2,
                    "confidence_level": 0.95,
                    "analysis_covariate_ids": ["age"],
                    "diagnostic_evidence_ids": [],
                }
            ],
            "supporting_evidence_ids": ["utility-control"],
        },
        out_path=out_path,
    )

    assert json.loads(out_path.read_text(encoding="utf-8"))["request"]["candidate_drug_ids"] == ["control"]


def test_phase_analysis_contract_exposes_numeric_evidence_not_self_attestation() -> None:
    phase_analysis = runtime_submission_contracts()["phase_analysis"]
    assert isinstance(phase_analysis, dict)
    properties = phase_analysis["properties"]

    for removed in (
        "uncertainty_calibrated",
        "sensitivity_analysis_performed",
        "temporal_reasoning_performed",
        "identifiability_assessment_performed",
        "multiplicity_control_considered",
    ):
        assert removed not in properties
    diagnostic_ref = properties["diagnostic_artifacts"]["items"]["$ref"].removeprefix("#/$defs/")
    assert phase_analysis["$defs"][diagnostic_ref]["required"] == [
        "artifact_id",
        "metric_family",
        "primary_value",
    ]

    submission = TrialDevelopmentPhaseAnalysisSubmissionV1.model_validate(
        {
            "scenario_id": "s01",
            "phase_id": "phase2",
            "request_checksum": "a" * 64,
            "trial_output_checksum": "b" * 64,
            "primary_effect": {
                "evidence_id": "effect_primary",
                "method_route_id": "trialdev.phase2.aalen_johansen_efficacy_safety.v1",
                "candidate_drug_id": "drug_a",
                "endpoint_id": "E1",
                "estimand_id": "treatment_policy_time_to_event",
                "estimator_id": "rmst_difference",
                "effect_scale_id": "days",
                "orientation_id": "positive_values_favour_treatment",
                "estimate": 0.12,
                "lower": 0.03,
                "upper": 0.21,
                "confidence_level": 0.95,
                "analysis_population": "all_randomized_participants",
                "source_artifact_checksums": {"endpoints.parquet": "a" * 64},
                "diagnostic_evidence_ids": ["sensitivity_span"],
            },
            "diagnostic_artifacts": [
                {
                    "artifact_id": "sensitivity_span",
                    "metric_family": "sensitivity",
                    "primary_value": 0.04,
                }
            ],
        }
    )
    assert submission.diagnostic_artifacts[0].primary_value == 0.04


def test_observational_file_contract_requires_complete_numeric_ranking_evidence(tmp_path: Path) -> None:
    contract_path = write_runtime_submission_contracts(tmp_path)
    contracts = json.loads(contract_path.read_text(encoding="utf-8"))
    required = set(contracts["observational_review"]["required"])

    assert {
        "response_branch",
        "primary_resolution_evidence_class",
        "supporting_evidence_ids",
    } <= required
    properties = contracts["observational_review"]["properties"]
    assert "Complete permutation" in properties["ranked_drug_ids"]["description"]
    assert "Exactly one utility estimate" in properties["candidate_utility_estimates"]["description"]
    assert "candidate_drug_id" not in required
    evidence_schema = contracts["observational_review"]["$defs"]["TrialDevelopmentIdentificationEvidenceV1"]
    assert "Exact factor_id" in evidence_schema["properties"]["source_record_id"]["description"]
    tool_names = {item["function"]["name"] for item in tools_for_obs_review()}
    assert "submit_obs_review_analysis_and_decision_file" in tool_names
    assert "submit_obs_review_analysis_and_decision" in tool_names
    phase_tool_names = {item["function"]["name"] for item in tools_for_phase_analysis()}
    assert "submit_phase_analysis_file" in phase_tool_names
    assert "submit_phase_analysis" in phase_tool_names


def test_observational_contract_projects_harness_custody_out_of_participant_schema() -> None:
    contract = runtime_submission_contracts()["observational_review"]

    serialized = json.dumps(contract)
    assert "source_artifact_checksums" not in serialized
    assert "public_artifact_sha256" not in serialized
    assert "candidate_utility_estimates" in contract["properties"]


def test_runtime_contract_does_not_enumerate_scientific_method_answers() -> None:
    """Typed output fields must not disclose passed method identities."""

    contracts = runtime_submission_contracts()
    scientific_fields: list[dict[str, object]] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in {"method_route_id", "estimator_id"}:
                    assert isinstance(nested, dict)
                    scientific_fields.append(nested)
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(contracts)

    assert scientific_fields
    for field in scientific_fields:
        assert "enum" not in field
        assert "const" not in field


def test_observational_contract_rejects_rank_without_numeric_evidence() -> None:
    payload = _observational_submission_payload()
    payload["ranked_drug_ids"] = ["drug_a", "drug_b"]

    with pytest.raises(ValidationError, match="exactly match"):
        TrialDevelopmentObservationalReviewSubmissionV1.model_validate(payload)


def test_observational_contract_rejects_unsubmitted_supporting_evidence() -> None:
    payload = _observational_submission_payload()
    payload["supporting_evidence_ids"] = ["mentioned-only"]

    with pytest.raises(ValidationError, match="reference submitted evidence"):
        TrialDevelopmentObservationalReviewSubmissionV1.model_validate(payload)


def test_observational_contract_rejects_candidate_on_stop_action() -> None:
    payload = _observational_submission_payload()
    payload["decision_action"] = "withhold_nomination"

    with pytest.raises(ValidationError, match="requires a null candidate"):
        TrialDevelopmentObservationalReviewSubmissionV1.model_validate(payload)


def test_randomized_phase_contract_omits_observational_only_utility_evidence() -> None:
    phase_analysis = runtime_submission_contracts()["phase_analysis"]

    assert "candidate_utility_estimates" not in phase_analysis["properties"]
    assert "request_checksum" not in phase_analysis["properties"]
    assert "trial_output_checksum" not in phase_analysis["properties"]


def test_phase_tools_reject_missing_public_contracts() -> None:
    with pytest.raises(ValidationError):
        TrialDevelopmentPhaseActionSpecV1.model_validate({})

    request_tool = next(
        tool
        for tool in tools_for_phase_request(
            {
                "phase_id": "phase2",
                "allowed_endpoint_ids": ["E1"],
                "allowed_follow_up_days": [90],
                "allowed_enrollment_window_days": [120],
                "allowed_site_count_budgets": [16],
                "allowed_allocation_ratios": ["1:1"],
                "allowed_variable_ids": ["AGE"],
                "max_sample_size": 100,
                "max_analysis_covariates": 1,
                "max_subgroup_splits": 1,
                "allowed_treatment_discontinuation_strategies": ["treatment_policy"],
                "allowed_interim_policies": ["fixed_final"],
                "allowed_site_strategies": ["region_balanced"],
                "allowed_selection_objectives": ["benefit_risk"],
            }
        )
        if tool["function"]["name"] == "submit_phase_request"
    )
    parameters = request_tool["function"]["parameters"]
    assert "phase_id" not in parameters["properties"]
    assert "scenario_id" not in parameters["properties"]
    assert {
        "design_cell_id",
        "target_sample_size",
        "endpoint_id",
        "follow_up_days",
        "enrollment_window_days",
        "site_count_budget",
        "allocation_ratio",
        "treatment_discontinuation_strategy",
        "interim_policy",
        "site_strategy",
        "selection_objective",
    } <= set(parameters["required"])

    with pytest.raises(ValueError, match="complete request menus"):
        tools_for_phase_request(
            {
                "phase_id": "phase2",
                "allowed_endpoint_ids": ["E1"],
                "max_sample_size": 100,
            }
        )


def test_phase_tool_exposes_exactly_one_investigational_regimen() -> None:
    request_tool = next(
        tool
        for tool in tools_for_phase_request(
            {
                "phase_id": "phase2",
                "allowed_endpoint_ids": ["E1"],
                "allowed_follow_up_days": [90],
                "allowed_enrollment_window_days": [120],
                "allowed_site_count_budgets": [16],
                "allowed_allocation_ratios": ["1:1"],
                "allowed_variable_ids": ["AGE"],
                "max_sample_size": 300,
                "max_analysis_covariates": 1,
                "max_subgroup_splits": 1,
                "allowed_treatment_discontinuation_strategies": ["treatment_policy"],
                "allowed_interim_policies": ["fixed_final"],
                "allowed_site_strategies": ["region_balanced"],
                "allowed_selection_objectives": ["benefit_risk"],
            }
        )
        if tool["function"]["name"] == "submit_phase_request"
    )
    parameters = request_tool["function"]["parameters"]
    assert "allocation_weights" not in parameters["properties"]
    assert "allocation_ratio" in parameters["required"]
    assert parameters["properties"]["candidate_drug_ids"]["minItems"] == 1
    assert parameters["properties"]["candidate_drug_ids"]["maxItems"] == 1


def test_missing_phase_module_fails_before_agent_execution(tmp_path: Path) -> None:
    (tmp_path / "eval_contract.json").write_text(
        json.dumps({"phase_modules": [{"phase_id": "phase1"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not define phase module 'phase2'"):
        get_phase_module(tmp_path, "phase2")
