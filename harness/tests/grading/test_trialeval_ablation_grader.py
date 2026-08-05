"""Integration tests for canonical grading of TrialEval ablation responses."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest
from trialagentbench_test_helpers import minimal_participant_output_contract, write_minimal_ground_truth_domains

from trialagentbench_harness.analysis.experiments.trialeval_endpoint_scoring import (
    score_trialeval_ablation_submission_v1,
    trialeval_numeric_result_available_v1,
)
from trialagentbench_harness.contracts.core.config import DecodingConfigV1
from trialagentbench_harness.contracts.core.runs import (
    ProviderRequestEventV1,
    RunCoverageV1,
    TrialEvalAblationItemResultV1,
    TrialEvalAblationRunConfigV1,
    TrialEvalAgentOutputV1,
    TrialEvalConditionProvenanceV1,
)
from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationAssignmentV1,
    TrialEvalAblationEndpointSetV1,
    TrialEvalAblationScheduleV1,
)
from trialagentbench_harness.contracts.scoring.assumption_evidence import (
    read_assumption_evidence_domains,
)
from trialagentbench_harness.contracts.submission import TrialEvalSubmissionV1
from trialagentbench_harness.contracts.trace.observable import (
    runtime_event_source_payload_v1,
)
from trialagentbench_harness.grading import ScoringKeyStoreV1
from trialagentbench_harness.io import (
    canonical_payload_sha256,
    read_json_model,
    sha256_path,
    write_json,
    write_json_model,
)
from trialagentbench_harness.tools.grade.grade_trialeval_ablation import main
from trialagentbench_harness.trialeval.action_trace import (
    collect_trialeval_ablation_observables,
    collect_trialeval_action_trace,
    collect_trialeval_trace_inputs,
)
from trialagentbench_harness.trialeval.conditions import prompt_set_sha256_v1
from trialagentbench_harness.trialeval.data import discover_items
from trialagentbench_harness.util.provider_telemetry import summarize_provider_telemetry_v1


def _submission() -> TrialEvalSubmissionV1:
    return TrialEvalSubmissionV1.model_validate(
        {
            "task_id": "TASK0001",
            "primary_analysis": {
                "declared_primary": True,
                "estimand": {
                    "estimand_id": "primary_itt",
                    "population_id": "intention_to_treat",
                    "treatment_id": "active",
                    "comparator_id": "control",
                    "endpoint_id": "primary_endpoint",
                    "intercurrent_event_strategy_ids": ["rescue_therapy:treatment_policy"],
                    "horizon": {"value": 365, "unit": "days"},
                },
                "estimator": {
                    "analysis_method_id": "coxph_binary_wald",
                    "implementation": "Breslow partial likelihood",
                    "qualifications": ["randomization_exchangeability"],
                },
                "result_kind": "numeric_point",
                "result": {
                    "kind": "scalar",
                    "value": -0.5,
                    "effect_scale": "log_hr",
                    "unit": "log_hazard_ratio",
                    "interval": {"lower": -0.7, "upper": -0.3, "confidence_level": 0.95},
                },
                "favorable_direction": "lower",
                "evidence_ids": ["randomization-check"],
            },
            "evidence": [
                {
                    "evidence_id": "randomization-check",
                    "evidence_type": "validity",
                    "principle": "design_validity",
                    "operation": "assessment",
                    "diagnostic_id": "randomization_integrity_public",
                    "target": "randomization integrity",
                    "result": {
                        "kind": "factual_premise",
                        "premise_id": "randomized_assignment_declared",
                        "conclusion": "supported",
                    },
                    "interpretation": "The public protocol declares randomized treatment assignment.",
                    "source_artifacts": ["protocol_summary.json"],
                }
            ],
            "limitations": ["No material limitation for this fixed test fixture."],
        }
    )


def _evaluator_release(root: Path) -> Path:
    write_minimal_ground_truth_domains(root)
    write_json(
        root / "grader" / "item_index.json",
        {
            "entries": [
                {
                    "task_id": "TASK0001",
                    "item_id": "d1a1_rct_clean_01",
                    "base_case_id": "d1a1_rct_clean_01",
                    "variant_id": "base",
                    "factors": {
                        "evaluation_series_id": "randomized",
                        "design_archetype": "D1",
                        "design_subtype": "individual_randomized",
                        "assumption_regime": "A1",
                        "context_configuration": "C1",
                        "data_preparation": "analysis_ready",
                        "analysis_specification": "locked_sap",
                        "procedure_assistance": "output_contract_only",
                        "response_interface": "structured",
                    },
                }
            ]
        },
    )
    task = {
        "schema_id": "trial_analysis_task_v1",
        "task_id": "TASK0001",
        "design_subtype": "individual_randomized",
        "primary_endpoint_id": "primary_endpoint",
        "primary_paramcd": "primary_endpoint",
        "primary_estimand_id": "primary_itt",
        "primary_effect_scale": "log_hr",
        "estimand_mode": "fixed_declared_estimand",
        "primary_effect_scale_options": ["log_hr"],
        "primary_result_unit": "log_hazard_ratio",
        "primary_population_id": "intention_to_treat",
        "primary_intercurrent_event_strategy_ids": ["rescue_therapy:treatment_policy"],
        "primary_control_arm_id": "control",
        "primary_treated_arm_id": "active",
    }
    write_json(root / "public" / "items" / "TASK0001" / "task.json", task)
    write_json(
        root / "public" / "items" / "TASK0001" / "protocol_summary.json",
        {
            "design_family": "parallel_randomized",
            "arms": [
                {"arm_id": "control"},
                {"arm_id": "active"},
            ],
        },
    )
    submission_contract = minimal_participant_output_contract("TASK0001")
    submission_contract.pop("checksum")
    submission_contract["diagnostic_obligations"] = [
        {
            "assumption_id": "randomization_integrity",
            "diagnostic_id": "randomization_integrity_public",
            "evidence_requirement": "design_declaration",
            "primary_credit_policy": "design_modifier",
            "operation": "Verify the randomized unit and allocation declaration.",
            "public_evidence_basis": ["protocol_summary.json"],
            "interpretation": "Design declaration only.",
        }
    ]
    submission_contract["checksum"] = canonical_payload_sha256(submission_contract)
    write_json(
        root / "public" / "items" / "TASK0001" / "submission_contract.json",
        submission_contract,
    )
    return root


def _ablation_run(
    root: Path,
    *,
    participant_release: Path,
    submission: TrialEvalSubmissionV1 | None = None,
) -> Path:
    conditions = (
        "neutral",
        "targeted_covariate_structure",
        "targeted_survival_assumptions",
        "targeted_design_structure",
        "targeted_data_integrity",
        "placebo_deliberation",
    )
    assignments = [
        TrialEvalAblationAssignmentV1(
            assignment_id=f"assignment-{specification}-{index}",
            task_id="TASK0001",
            context_tier="C1",
            data_preparation="analysis_ready",
            analysis_specification=specification,
            analysis_surface_sha256=("1" if specification == "protocol_only" else "2") * 64,
            replicate_id="seed-1",
            decoding_seed=101,
            procedure_assistance="output_contract_only",
            prompt_condition=condition,
            submission_interface="structured",
        )
        for specification in ("locked_sap",)
        for index, condition in enumerate(conditions)
    ]
    assignments.sort(key=lambda assignment: assignment.assignment_id)
    random.Random(17).shuffle(assignments)
    schedule = TrialEvalAblationScheduleV1(
        experiment_id="targeted-smoke",
        design="targeted_control",
        execution_scope="pilot",
        experiment_design_sha256="d" * 64,
        participant_release_sha256=sha256_path(participant_release),
        prompt_set_sha256=prompt_set_sha256_v1(),
        analysis_config_sha256="r" * 64,
        randomization_seed=17,
        assignments=tuple(assignments),
    )
    run_config = TrialEvalAblationRunConfigV1.create(
        timestamp_utc=datetime.now(UTC),
        experiment_id=schedule.experiment_id,
        schedule_checksum=str(schedule.checksum),
        participant_release_sha256=schedule.participant_release_sha256,
        prompt_set_sha256=schedule.prompt_set_sha256,
        scorer_source_sha256="s" * 64,
        agent_source_sha256="a" * 64,
        model="model-a",
        max_context_characters=120_000,
        item_watchdog_seconds=3600,
        decoding=DecodingConfigV1(temperature=0.0, max_tokens=1024, send_temperature=True),
        routing={"provider": "openai", "request_timeout_seconds": 300.0},
        executor={
            "image_reference": "trialagentbench/executor:test",
            "image_id": f"sha256:{'e' * 64}",
            "python_version": "3.12.10",
            "packages": [
                {"name": "lifelines", "version": "0.30.0"},
                {"name": "numpy", "version": "2.4.4"},
                {"name": "pandas", "version": "2.3.3"},
                {"name": "pyarrow", "version": "23.0.1"},
                {"name": "scipy", "version": "1.17.1"},
                {"name": "statsmodels", "version": "0.14.6"},
            ],
            "limits": {},
        },
        workers=1,
        n_assignments=len(assignments),
    )
    write_json_model(root / "schedule.json", schedule)
    write_json_model(root / "run_config.json", run_config)
    assignment_ids = tuple(assignment.assignment_id for assignment in schedule.assignments)
    coverage = RunCoverageV1(
        run_identity_sha256=run_config.run_identity_sha256,
        schedule_sha256=str(schedule.checksum),
        unit_ids=assignment_ids,
        completed_unit_ids=assignment_ids,
    )
    write_json_model(root / "coverage.json", coverage)
    for assignment in assignments:
        provenance = TrialEvalConditionProvenanceV1(
            procedure_assistance=assignment.procedure_assistance,
            analysis_specification=assignment.analysis_specification,
            analysis_surface_sha256=assignment.analysis_surface_sha256,
            prompt_condition=assignment.prompt_condition,
            submission_interface="structured",
            max_turns=25,
            prompt_set_sha256=schedule.prompt_set_sha256,
            rendered_system_prompt_sha256="r" * 64,
            tool_schema_sha256="t" * 64,
            response_contract_sha256="u" * 64,
        )
        result = TrialEvalAblationItemResultV1(
            timestamp_utc=datetime.now(UTC),
            assignment=assignment,
            run_config=run_config,
            agent_output=TrialEvalAgentOutputV1(
                status="success",
                turns_used=1,
                result=submission or _submission(),
                condition_provenance=provenance,
            ),
        )
        write_json_model(root / "assignments" / f"{assignment.assignment_id}.json", result)
    summarize_provider_telemetry_v1(run_root=root, coverage=coverage)
    return root


def test_ablation_grader_reuses_complete_canonical_method_route_scoring(tmp_path: Path) -> None:
    evaluator = _evaluator_release(tmp_path / "evaluator")
    run = _ablation_run(tmp_path / "run", participant_release=evaluator / "public")
    output = tmp_path / "endpoints.json"

    assert main([str(run), "--suite-dir", str(evaluator), "--out", str(output)]) == 0
    endpoint_set = read_json_model(TrialEvalAblationEndpointSetV1, output)
    assert len(endpoint_set.endpoints) == 6
    assert all(row.usable_primary for row in endpoint_set.endpoints)
    assert all(row.obligations_met for row in endpoint_set.endpoints)
    assert all(row.credit_eligible_route_count == 1 for row in endpoint_set.endpoints)
    assert all(row.primary_analysis_conforms == 1.0 for row in endpoint_set.endpoints)
    assert all(row.result_match for row in endpoint_set.endpoints)
    assert all(row.numeric_absolute_error is not None for row in endpoint_set.endpoints)
    assert all(row.numeric_tolerance_ratio is not None for row in endpoint_set.endpoints)
    assert all(row.primary_interval_agreement is None for row in endpoint_set.endpoints)


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("conclusion", "not_supported"),
        ("premise_id", "cluster_randomization_declared"),
        ("source_artifacts", ["task.json"]),
    ),
)
def test_ablation_grader_rejects_fabricated_factual_premise(
    tmp_path: Path,
    mutation: str,
    value: object,
) -> None:
    evaluator = _evaluator_release(tmp_path / "evaluator")
    payload = _submission().model_dump(mode="json")
    evidence = payload["evidence"][0]
    if mutation == "source_artifacts":
        evidence[mutation] = value
    else:
        evidence["result"][mutation] = value
    submission = TrialEvalSubmissionV1.model_validate(payload)
    run = _ablation_run(
        tmp_path / "run",
        participant_release=evaluator / "public",
        submission=submission,
    )
    output = tmp_path / "endpoints.json"

    assert main([str(run), "--suite-dir", str(evaluator), "--out", str(output)]) == 0
    endpoint_set = read_json_model(TrialEvalAblationEndpointSetV1, output)
    assert len(endpoint_set.endpoints) == 6
    assert all(not row.obligations_met for row in endpoint_set.endpoints)
    assert all(row.primary_analysis_conforms == 0.0 for row in endpoint_set.endpoints)


def test_ablation_grader_retains_missing_required_deliverables_as_typed_failure(tmp_path: Path) -> None:
    evaluator = _evaluator_release(tmp_path / "evaluator")
    payload = _submission().model_dump(mode="json")
    payload["primary_analysis"]["evidence_ids"] = []
    payload["evidence"] = []
    incomplete = TrialEvalSubmissionV1.model_validate(payload)
    run = _ablation_run(
        tmp_path / "run",
        participant_release=evaluator / "public",
        submission=incomplete,
    )
    output = tmp_path / "endpoints.json"

    assert main([str(run), "--suite-dir", str(evaluator), "--out", str(output)]) == 0
    endpoint_set = read_json_model(TrialEvalAblationEndpointSetV1, output)
    assert len(endpoint_set.endpoints) == 6
    assert all(row.primary_failure_code == "missing_required_deliverable" for row in endpoint_set.endpoints)
    assert all(row.omitted_required_deliverables == ("evidence",) for row in endpoint_set.endpoints)
    assert all(not row.usable_primary for row in endpoint_set.endpoints)


def test_ablation_endpoint_retains_normalizer_abstention(tmp_path: Path) -> None:
    evaluator = _evaluator_release(tmp_path / "evaluator")
    run = _ablation_run(tmp_path / "run", participant_release=evaluator / "public")
    result_path = sorted((run / "assignments").glob("*.json"))[0]
    structured_result = read_json_model(TrialEvalAblationItemResultV1, result_path)
    narrative_assignment = structured_result.assignment.model_copy(update={"submission_interface": "narrative"})
    provenance = structured_result.agent_output.condition_provenance.model_copy(
        update={"submission_interface": "narrative"}
    )
    narrative_result = TrialEvalAblationItemResultV1(
        timestamp_utc=structured_result.timestamp_utc,
        assignment=narrative_assignment,
        run_config=structured_result.run_config,
        agent_output=TrialEvalAgentOutputV1(
            status="success",
            turns_used=1,
            report="The report does not contain a complete primary analysis.",
            condition_provenance=provenance,
        ),
    )
    item = discover_items(evaluator)[0]
    scoring_key = ScoringKeyStoreV1.from_release(
        evaluator,
        expected_item_ids=(item.task_id,),
    ).for_item(item.task_id)
    assumption_evidence = read_assumption_evidence_domains(release_root=evaluator)[item.task_id]

    endpoint = score_trialeval_ablation_submission_v1(
        scoring_key=scoring_key,
        assumption_evidence=assumption_evidence,
        item=item,
        result=narrative_result,
        submission=None,
        normalization_source="automated_importer",
        normalization_status="abstain",
        normalization_failure_reason="No complete primary analysis could be transcribed.",
    )

    assert endpoint.normalization_status == "abstain"
    assert endpoint.normalization_failure_reason == "No complete primary analysis could be transcribed."
    assert endpoint.primary_failure_code == "missing_primary_submission"
    assert endpoint.omitted_required_deliverables == ("evidence", "limitations", "primary_analysis")
    assert not endpoint.primary_analysis_conforms


def test_ablation_grader_rejects_missing_assignment_without_denominator_drop(tmp_path: Path) -> None:
    evaluator = _evaluator_release(tmp_path / "evaluator")
    run = _ablation_run(tmp_path / "run", participant_release=evaluator / "public")
    schedule = read_json_model(TrialEvalAblationScheduleV1, run / "schedule.json")
    missing_id = schedule.assignments[0].assignment_id
    (run / "assignments" / f"{missing_id}.json").unlink()

    try:
        main([str(run), "--suite-dir", str(evaluator), "--out", str(tmp_path / "endpoints.json")])
    except ValueError as exc:
        assert "denominator mismatch" in str(exc)
    else:
        raise AssertionError("Expected missing ablation assignment to fail grading.")


def test_ablation_grader_rejects_evaluator_public_surface_drift(tmp_path: Path) -> None:
    evaluator = _evaluator_release(tmp_path / "evaluator")
    run = _ablation_run(tmp_path / "run", participant_release=evaluator / "public")
    task_path = evaluator / "public" / "items" / "TASK0001" / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["primary_endpoint_id"] = "mutated_after_run"
    write_json(task_path, task)

    try:
        main([str(run), "--suite-dir", str(evaluator), "--out", str(tmp_path / "endpoints.json")])
    except ValueError as exc:
        assert "public surface" in str(exc)
    else:
        raise AssertionError("Expected evaluator participant-surface drift to fail grading.")


def test_nonidentification_without_identified_set_is_not_a_numeric_result() -> None:
    payload = _submission().model_dump(mode="json")
    payload["primary_analysis"]["result"] = {
        "kind": "non_identification",
        "conclusion_code": "effect_not_identified",
        "effect_scale": "log_hr",
        "unit": "log_hazard_ratio",
        "reason": "The requested point effect is not identified from the released evidence.",
        "identified_set": None,
        "additional_assumption_required": "Conditional exchangeability after treatment selection.",
    }
    payload["primary_analysis"]["result_kind"] = "abstention"
    submission = TrialEvalSubmissionV1.model_validate(payload)

    assert trialeval_numeric_result_available_v1(submission) is False
    payload["primary_analysis"]["result"]["identified_set"] = {
        "kind": "identified_interval",
        "lower": -1.0,
        "upper": 0.0,
        "effect_scale": "log_hr",
        "unit": "log_hazard_ratio",
        "interpretation": "Effects compatible with the released evidence.",
    }
    assert trialeval_numeric_result_available_v1(TrialEvalSubmissionV1.model_validate(payload)) is True


def test_mixed_interface_and_direct_file_outputs_are_typed_trace_inputs(tmp_path: Path) -> None:
    root = tmp_path / "same-suffix"
    assignments = [
        TrialEvalAblationAssignmentV1(
            assignment_id=f"assignment-{specification}-{assistance}-{interface}-{transport}",
            task_id="TASK0001",
            context_tier="C1",
            data_preparation="analysis_ready",
            analysis_specification=specification,
            analysis_surface_sha256=("1" if specification == "protocol_only" else "2") * 64,
            replicate_id="seed-1",
            decoding_seed=101,
            procedure_assistance=assistance,
            prompt_condition="neutral",
            submission_interface=interface,
        )
        for specification in ("locked_sap",)
        for assistance, interface, transport in (
            ("output_contract_only", "structured", "direct"),
            ("output_contract_only", "narrative", "direct"),
            ("unordered_checklist", "structured", "file"),
            ("unordered_checklist", "narrative", "file"),
            ("ordered_sop", "structured", "direct"),
            ("ordered_sop", "narrative", "direct"),
        )
    ]
    assignments.sort(key=lambda assignment: assignment.assignment_id)
    random.Random(17).shuffle(assignments)
    schedule = TrialEvalAblationScheduleV1(
        experiment_id="factorial-trace-ingest",
        design="factorial_interface",
        execution_scope="pilot",
        experiment_design_sha256="d" * 64,
        participant_release_sha256="p" * 64,
        prompt_set_sha256=prompt_set_sha256_v1(),
        analysis_config_sha256="r" * 64,
        randomization_seed=17,
        assignments=tuple(assignments),
    )
    run_config = TrialEvalAblationRunConfigV1.create(
        timestamp_utc=datetime.now(UTC),
        experiment_id=schedule.experiment_id,
        schedule_checksum=str(schedule.checksum),
        participant_release_sha256=schedule.participant_release_sha256,
        prompt_set_sha256=schedule.prompt_set_sha256,
        scorer_source_sha256="s" * 64,
        agent_source_sha256="a" * 64,
        model="model-a",
        max_context_characters=120_000,
        item_watchdog_seconds=3600,
        decoding=DecodingConfigV1(temperature=0.0, max_tokens=1024, send_temperature=True),
        routing={"provider": "openai", "request_timeout_seconds": 300.0},
        executor={
            "image_reference": "trialagentbench/executor:test",
            "image_id": f"sha256:{'e' * 64}",
            "python_version": "3.12.10",
            "packages": [
                {"name": "lifelines", "version": "0.30.0"},
                {"name": "numpy", "version": "2.4.4"},
                {"name": "pandas", "version": "2.3.3"},
                {"name": "pyarrow", "version": "23.0.1"},
                {"name": "scipy", "version": "1.17.1"},
                {"name": "statsmodels", "version": "0.14.6"},
            ],
            "limits": {},
        },
        workers=1,
        n_assignments=len(assignments),
    )
    write_json_model(root / "schedule.json", schedule)
    write_json_model(root / "run_config.json", run_config)
    for assignment in assignments:
        interface = assignment.submission_interface
        transport = "file" if assignment.procedure_assistance == "unordered_checklist" else "direct"
        result = TrialEvalAblationItemResultV1(
            timestamp_utc=datetime.now(UTC),
            assignment=assignment,
            run_config=run_config,
            agent_output=TrialEvalAgentOutputV1(
                status="success",
                turns_used=1,
                report="Participant-authored analysis report." if interface == "narrative" else None,
                result=_submission() if interface == "structured" else None,
                condition_provenance=TrialEvalConditionProvenanceV1(
                    procedure_assistance=assignment.procedure_assistance,
                    analysis_specification=assignment.analysis_specification,
                    analysis_surface_sha256=assignment.analysis_surface_sha256,
                    prompt_condition=assignment.prompt_condition,
                    submission_interface=interface,
                    max_turns=25,
                    prompt_set_sha256=schedule.prompt_set_sha256,
                    rendered_system_prompt_sha256="r" * 64,
                    tool_schema_sha256="t" * 64,
                    response_contract_sha256="u" * 64,
                ),
            ),
        )
        write_json_model(root / "assignments" / f"{assignment.assignment_id}.json", result)
        tool_name = "submit_response_file" if transport == "file" else "submit_response"
        conversation_path = root / "traces" / f"{assignment.assignment_id}.json"
        submission_message = {
            "role": "tool",
            "tool_call_id": f"call-{assignment.assignment_id}",
            "tool": tool_name,
            "output": "submitted",
        }
        conversation = [
            {"role": "user", "content": "Analyse the released trial."},
            {"role": "assistant", "content": "I will submit the result."},
            submission_message,
        ]
        write_json(conversation_path, conversation)
        event_rows = [
            ("step_started", None, None),
            ("prompt", 0, None),
            ("assistant_message", 1, None),
            ("submission", 2, None),
            ("step_terminal", None, "completed"),
        ]
        payloads = []
        for index, (event_type, message_index, terminal_status) in enumerate(event_rows):
            message = conversation[message_index] if message_index is not None else None
            source_payload = runtime_event_source_payload_v1(
                benchmark="trialeval",
                task_id=assignment.task_id,
                program_id=None,
                scenario_id=None,
                objective_id=None,
                phase_id="task",
                step_id="analysis",
                event_type=event_type,
                terminal_status=terminal_status,
                failure_type=None,
                conversation_message=message,
            )
            payload = {
                "schema_id": "trialagentbench.runtime_trace_event/v1",
                "event_id": f"trialeval:{assignment.assignment_id}:{index:06d}",
                "timestamp": datetime.now(UTC).isoformat(),
                "source_artifact_path": conversation_path.as_posix(),
                "source_payload_sha256": canonical_payload_sha256(source_payload),
                "benchmark": "trialeval",
                "event_index": index,
                "task_id": assignment.task_id,
                "phase_id": "task",
                "step_id": "analysis",
                "event_type": event_type,
                "conversation_message_index": message_index,
                "terminal_status": terminal_status,
            }
            if event_type == "submission":
                payload["tool_call_id"] = submission_message["tool_call_id"]
                payload["tool_name"] = tool_name
            payloads.append(payload)
        event_path = root / "events" / f"{assignment.assignment_id}_events.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_text(
            "".join(json.dumps(payload) + "\n" for payload in payloads),
            encoding="utf-8",
        )
        provider_path = root / "logs" / f"{assignment.assignment_id}_provider_responses.jsonl"
        provider_path.parent.mkdir(parents=True, exist_ok=True)
        provider_identity = {
            "request_id": f"request-{assignment.assignment_id}",
            "benchmark": "trialeval",
            "unit_id": assignment.task_id,
            "phase_id": "task",
            "step_id": "analysis",
            "turn_index": 1,
            "requested_model": run_config.model,
            "provider_route": "test",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "transient_failure_count": 0,
            "backoff_seconds": 0.0,
        }
        provider_events = (
            ProviderRequestEventV1(
                **provider_identity,
                status="started",
                elapsed_seconds=None,
                usage_status="not_applicable",
                request_attempts=0,
            ),
            ProviderRequestEventV1(
                **{
                    **provider_identity,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                status="succeeded",
                elapsed_seconds=0.1,
                usage_status="reported",
                request_attempts=1,
            ),
        )
        provider_path.write_text(
            "".join(event.model_dump_json() + "\n" for event in provider_events),
            encoding="utf-8",
        )

    inputs = collect_trialeval_trace_inputs([root])

    assert {(row.submission_interface, row.submission_transport) for row in inputs} == {
        ("structured", "direct"),
        ("structured", "file"),
        ("narrative", "direct"),
        ("narrative", "file"),
    }
    narratives = [row for row in inputs if row.submission_interface == "narrative"]
    assert all(row.answer_present for row in narratives)
    assert all(row.authority == "non_authoritative_narrative" for row in narratives)
    assert all(row.submission is None for row in narratives)

    observables = collect_trialeval_ablation_observables([root])
    assert len(observables) == 6
    assert {
        (
            row.analysis_specification,
            row.procedure_assistance,
            row.submission_interface,
        )
        for row in observables
    } == {
        (specification, assistance, interface)
        for specification in ("locked_sap",)
        for assistance in ("output_contract_only", "unordered_checklist", "ordered_sop")
        for interface in ("structured", "narrative")
    }
    assert all(row.answer_submitted for row in observables)
    assert all(row.events_until_submission == 4 for row in observables)
    assert all(row.events_until_first_data_inspection == 6 for row in observables)
    assert all(row.declared_uncertainty is None for row in observables if row.submission_interface == "narrative")
    assert all(row.declared_uncertainty is True for row in observables if row.submission_interface == "structured")

    events, features, evidence, cascades, semantic = collect_trialeval_action_trace([root])
    assert len(features) == 6
    assert {row.assignment_id for row in features} == {assignment.assignment_id for assignment in schedule.assignments}
    assert all(row.context_tier == "C1" for row in features)
    assert {row.procedure_assistance for row in features} == {
        "output_contract_only",
        "unordered_checklist",
        "ordered_sop",
    }
    assert {row.step_id for row in events} == {"analysis"}
    assert evidence == []
    assert len(cascades) == 6
    assert len(semantic) == 90
