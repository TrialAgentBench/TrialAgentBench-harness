"""Action-trace extraction for TrialEvalBench stored results."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic.types import JsonValue

from trialagentbench_harness.analysis.evidence_use import evidence_rows_from_events
from trialagentbench_harness.analysis.failure_cascade import phase_failure_cascade
from trialagentbench_harness.analysis.run_identity import require_unique_run_ids
from trialagentbench_harness.analysis.semantic_features import structured_feature_flags, structured_feature_rows
from trialagentbench_harness.analysis.trialeval_score_rows import iter_trialeval_score_rows
from trialagentbench_harness.contracts.core.runs import (
    TrialEvalAblationItemResultV1,
    TrialEvalAblationRunConfigV1,
    TrialEvalItemResultV1,
)
from trialagentbench_harness.contracts.experiments import TrialEvalAblationObservableRowV1
from trialagentbench_harness.contracts.trace.observable import (
    BenchmarkRuntimeTraceEventV1,
    EvidenceUseRowV1,
    FailureCascadeRowV1,
    ModelActionTraceEventV1,
    SemanticActionFeatureRowV1,
    TraceFeatureRowV1,
    TrialEvalTraceInputV1,
    runtime_event_source_payload_v1,
)
from trialagentbench_harness.io import read_json_model
from trialagentbench_harness.io.checksums import canonical_payload_sha256, sha256_file
from trialagentbench_harness.util.provider_telemetry import read_provider_terminal_events_v1


def _runtime_events(path: Path) -> tuple[BenchmarkRuntimeTraceEventV1, ...]:
    events: list[BenchmarkRuntimeTraceEventV1] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                events.append(BenchmarkRuntimeTraceEventV1.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid TrialEval runtime event at {path}:{line_number}") from exc
    if [event.event_index for event in events] != list(range(len(events))):
        raise ValueError(f"TrialEval runtime event indices must be contiguous: {path}")
    if len({event.event_id for event in events}) != len(events):
        raise ValueError(f"TrialEval runtime event IDs must be unique: {path}")
    return tuple(events)


def _validate_runtime_events(
    events: tuple[BenchmarkRuntimeTraceEventV1, ...],
    *,
    task_id: str,
    conversation_path: Path,
) -> None:
    if not events:
        raise ValueError(f"TrialEval runtime event stream is empty: {conversation_path}")
    if events[0].event_type != "step_started":
        raise ValueError(f"TrialEval runtime events must start with step_started: {conversation_path}")
    for event in events:
        if event.benchmark != "trialeval" or event.task_id != task_id:
            raise ValueError(f"TrialEval runtime event identity mismatch: {conversation_path}")
    terminals = [event for event in events if event.event_type == "step_terminal"]
    if len(terminals) != 1 or events[-1] is not terminals[0]:
        raise ValueError(f"TrialEval runtime events require exactly one final step_terminal: {conversation_path}")
    messages = json.loads(conversation_path.read_text(encoding="utf-8"))
    if not isinstance(messages, list):
        raise ValueError(f"TrialEval conversation must be a list: {conversation_path}")
    for event in events:
        if event.phase_id != "task" or event.step_id != "analysis":
            raise ValueError(f"TrialEval runtime event route mismatch: {conversation_path}")
        if event.event_type in {"step_started", "step_terminal"}:
            if event.conversation_message_index is not None:
                raise ValueError("TrialEval lifecycle events must not reference a conversation message.")
            source_payload = runtime_event_source_payload_v1(
                benchmark=event.benchmark,
                task_id=event.task_id,
                program_id=event.program_id,
                scenario_id=event.scenario_id,
                objective_id=event.objective_id,
                phase_id=event.phase_id,
                step_id=event.step_id,
                event_type=event.event_type,
                terminal_status=event.terminal_status,
                failure_type=event.failure_type,
                conversation_message=None,
            )
            if canonical_payload_sha256(source_payload) != event.source_payload_sha256:
                raise ValueError(f"TrialEval runtime lifecycle payload hash mismatch: {conversation_path}")
            continue
        index = event.conversation_message_index
        if index is None or index >= len(messages) or not isinstance(messages[index], dict):
            raise ValueError(f"TrialEval runtime event references a missing conversation message: {conversation_path}")
        message = messages[index]
        source_payload = runtime_event_source_payload_v1(
            benchmark=event.benchmark,
            task_id=event.task_id,
            program_id=event.program_id,
            scenario_id=event.scenario_id,
            objective_id=event.objective_id,
            phase_id=event.phase_id,
            step_id=event.step_id,
            event_type=event.event_type,
            terminal_status=event.terminal_status,
            failure_type=event.failure_type,
            conversation_message=cast(JsonValue, message),
        )
        if canonical_payload_sha256(source_payload) != event.source_payload_sha256:
            raise ValueError(f"TrialEval runtime conversation payload hash mismatch: {conversation_path}")
        expected_role = "user" if event.event_type == "prompt" else "assistant"
        if event.event_type in {"tool_call", "tool_result", "code_execution", "file_inspection", "submission"}:
            expected_role = "tool"
            if str(message.get("tool_call_id") or "") != str(event.tool_call_id or ""):
                raise ValueError(f"TrialEval runtime tool id mismatch: {conversation_path}")
            if str(message.get("tool") or "") != str(event.tool_name or ""):
                raise ValueError(f"TrialEval runtime tool name mismatch: {conversation_path}")
        if str(message.get("role") or "") != expected_role:
            raise ValueError(f"TrialEval runtime event role mismatch: {conversation_path}")


def _require_submission_terminal(
    events: tuple[BenchmarkRuntimeTraceEventV1, ...],
    *,
    submission_expected: bool,
    path: Path,
) -> None:
    terminal = events[-1]
    submissions = [event for event in events if event.event_type == "submission" and event.status == "observed"]
    if submission_expected:
        if len(submissions) != 1:
            raise ValueError(f"TrialEval submitted result requires exactly one runner-native submission event: {path}")
        if terminal.terminal_status != "completed":
            raise ValueError(f"TrialEval submitted result requires a completed runner-native terminal event: {path}")
    elif submissions:
        raise ValueError(f"TrialEval scorer state says no submission but runner telemetry contains one: {path}")


def _events_until(
    events: tuple[BenchmarkRuntimeTraceEventV1, ...],
    event_type: str,
    *,
    successful: bool = False,
) -> int:
    """Count events through the first matching event, including nonoccurrence."""

    matched = [
        event.event_index + 1
        for event in events
        if event.event_type == event_type and (not successful or event.execution_status == "success")
    ]
    return min(matched, default=len(events) + 1)


def _submission_transport(run_dir: Path, *unit_ids: str) -> str:
    candidates = tuple(
        dict.fromkeys(
            parent / f"{unit_id}_events.jsonl"
            for unit_id in unit_ids
            for parent in (run_dir / "logs", run_dir / "events")
        )
    )
    observed_paths = [path for path in candidates if path.is_file()]
    if not observed_paths:
        return "not_observed"
    if len(observed_paths) > 1:
        raise ValueError(f"TrialEval unit has multiple runtime event sources: {observed_paths}")
    runtime_path = observed_paths[0]
    submission_tools = {
        event.tool_name
        for event in _runtime_events(runtime_path)
        if event.event_type == "submission" and event.tool_name is not None
    }
    if len(submission_tools) > 1:
        raise ValueError(f"TrialEval unit has conflicting submission transports: {runtime_path}")
    if not submission_tools:
        return "not_observed"
    tool_name = next(iter(submission_tools))
    if tool_name == "submit_response":
        return "direct"
    if tool_name == "submit_response_file":
        return "file"
    raise ValueError(f"TrialEval unit has an unknown submission tool {tool_name!r}: {runtime_path}")


def collect_trialeval_trace_inputs(run_dirs: list[Path]) -> tuple[TrialEvalTraceInputV1, ...]:
    """Load structured and raw narrative outputs without deriving prose semantics."""

    run_ids = require_unique_run_ids(run_dirs, suite="trialeval")
    inputs: list[TrialEvalTraceInputV1] = []
    for raw_run_dir in sorted(run_dirs):
        run_dir = raw_run_dir.resolve()
        run_id = run_ids[run_dir]
        config_path = run_dir / "run_config.json"
        config_document = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        is_ablation = config_document.get("schema_id") == "trialagentbench.trialeval_ablation_run/v1"
        if is_ablation:
            run_config = read_json_model(TrialEvalAblationRunConfigV1, run_dir / "run_config.json")
            paths = sorted((run_dir / "assignments").glob("*.json"))
            for path in paths:
                ablation_record = read_json_model(TrialEvalAblationItemResultV1, path)
                interface = ablation_record.assignment.submission_interface
                if ablation_record.run_config != run_config:
                    raise ValueError(f"TrialEval ablation trace input has configuration drift: {path}")
                inputs.append(
                    TrialEvalTraceInputV1(
                        run_id=run_id,
                        model_id=run_config.model,
                        task_id=ablation_record.assignment.task_id,
                        assignment_id=ablation_record.assignment.assignment_id,
                        context_tier=ablation_record.assignment.context_tier,
                        data_preparation=ablation_record.assignment.data_preparation,
                        analysis_specification=ablation_record.assignment.analysis_specification,
                        procedure_assistance=ablation_record.assignment.procedure_assistance,
                        prompt_condition=ablation_record.assignment.prompt_condition,
                        submission_interface=interface,
                        submission_transport=_submission_transport(run_dir, ablation_record.assignment.assignment_id),
                        authority=(
                            "authoritative_structured" if interface == "structured" else "non_authoritative_narrative"
                        ),
                        source_path=path.as_posix(),
                        submission=(ablation_record.agent_output.result if interface == "structured" else None),
                        narrative_report=(ablation_record.agent_output.report if interface == "narrative" else None),
                    )
                )
            continue
        paths = sorted((run_dir / "items").glob("*.json"))
        for path in paths:
            item_record = read_json_model(TrialEvalItemResultV1, path)
            interface = item_record.agent_output.condition_provenance.submission_interface
            scored_task_id = item_record.scores.task_id if item_record.scores is not None else None
            task_id = scored_task_id or item_record.item_id
            evidence_factors = item_record.run_config.task_evidence_factors[item_record.item_id]
            provenance = item_record.agent_output.condition_provenance
            inputs.append(
                TrialEvalTraceInputV1(
                    run_id=run_id,
                    model_id=item_record.run_config.model,
                    task_id=task_id,
                    context_tier=evidence_factors.context_configuration,
                    data_preparation=evidence_factors.data_preparation,
                    analysis_specification=evidence_factors.analysis_specification,
                    procedure_assistance=provenance.procedure_assistance,
                    prompt_condition=provenance.prompt_condition,
                    submission_interface=interface,
                    submission_transport=_submission_transport(run_dir, task_id, item_record.item_id),
                    authority=(
                        "authoritative_structured" if interface == "structured" else "non_authoritative_narrative"
                    ),
                    source_path=path.as_posix(),
                    submission=item_record.agent_output.result if interface == "structured" else None,
                    narrative_report=item_record.agent_output.report if interface == "narrative" else None,
                )
            )
    return tuple(inputs)


def collect_trialeval_ablation_observables(
    run_dirs: list[Path],
) -> tuple[TrialEvalAblationObservableRowV1, ...]:
    """Derive condition-bound process measures from runner-native ablation events."""

    require_unique_run_ids(run_dirs, suite="trialeval")
    rows: list[TrialEvalAblationObservableRowV1] = []
    for raw_run_dir in sorted(run_dirs):
        run_dir = raw_run_dir.resolve()
        run_config = read_json_model(TrialEvalAblationRunConfigV1, run_dir / "run_config.json")
        trace_inputs = {
            row.assignment_id: row
            for row in collect_trialeval_trace_inputs([run_dir])
            if row.assignment_id is not None
        }
        assignment_paths = sorted((run_dir / "assignments").glob("*.json"))
        if set(trace_inputs) != {path.stem for path in assignment_paths}:
            raise ValueError(f"TrialEval ablation trace denominator mismatch: {run_dir}")
        for path in assignment_paths:
            result = read_json_model(TrialEvalAblationItemResultV1, path)
            assignment = result.assignment
            trace_input = trace_inputs[assignment.assignment_id]
            if result.run_config != run_config:
                raise ValueError(f"TrialEval ablation process row has configuration drift: {path}")
            runtime_path = run_dir / "events" / f"{assignment.assignment_id}_events.jsonl"
            conversation_path = run_dir / "traces" / f"{assignment.assignment_id}.json"
            if not runtime_path.is_file():
                raise FileNotFoundError(f"TrialEval ablation assignment lacks runtime events: {runtime_path}")
            if not conversation_path.is_file():
                raise FileNotFoundError(
                    f"TrialEval ablation runtime events require a conversation: {conversation_path}"
                )
            runtime_events = _runtime_events(runtime_path)
            _validate_runtime_events(
                runtime_events,
                task_id=assignment.task_id,
                conversation_path=conversation_path,
            )
            _require_submission_terminal(
                runtime_events,
                submission_expected=trace_input.answer_present,
                path=runtime_path,
            )
            provider_path = run_dir / "logs" / f"{assignment.assignment_id}_provider_responses.jsonl"
            if not provider_path.is_file():
                raise FileNotFoundError(f"TrialEval ablation assignment lacks provider telemetry: {provider_path}")
            provider_events = read_provider_terminal_events_v1(provider_path)
            if any(
                event.benchmark != "trialeval"
                or event.unit_id != assignment.task_id
                or event.phase_id != "task"
                or event.step_id != "analysis"
                for event in provider_events
            ):
                raise ValueError(f"TrialEval ablation provider telemetry identity mismatch: {provider_path}")
            succeeded_provider_events = tuple(event for event in provider_events if event.status == "succeeded")
            failed_provider_events = tuple(event for event in provider_events if event.status == "failed")
            if failed_provider_events:
                raise ValueError(f"Completed TrialEval assignment contains failed provider requests: {provider_path}")
            code_events = [event for event in runtime_events if event.event_type == "code_execution"]
            execution_events = [
                event for event in runtime_events if event.event_type in {"code_execution", "file_inspection"}
            ]
            submission = trace_input.submission
            diagnostics: int | None = None
            sensitivities: int | None = None
            uncertainty: bool | None = None
            if submission is not None:
                diagnostics = sum(
                    evidence.evidence_type in {"diagnostic", "validity"} for evidence in submission.evidence
                )
                sensitivities = sum(evidence.evidence_type == "sensitivity" for evidence in submission.evidence)
                uncertainty = bool(
                    structured_feature_flags(
                        submission.evidence,
                        primary_interval_reported=submission.primary_analysis.result.kind
                        in {"scalar", "identified_interval"},
                    )["quantified_uncertainty"]
                )
            rows.append(
                TrialEvalAblationObservableRowV1(
                    assignment_id=assignment.assignment_id,
                    task_id=assignment.task_id,
                    context_tier=assignment.context_tier,
                    data_preparation=assignment.data_preparation,
                    analysis_specification=assignment.analysis_specification,
                    model_id=run_config.model,
                    replicate_id=assignment.replicate_id,
                    procedure_assistance=assignment.procedure_assistance,
                    prompt_condition=assignment.prompt_condition,
                    submission_interface=assignment.submission_interface,
                    answer_submitted=trace_input.answer_present,
                    public_data_inspected=any(event.event_type == "file_inspection" for event in runtime_events),
                    code_executed_successfully=any(event.execution_status == "success" for event in code_events),
                    assistant_turns=sum(event.event_type == "assistant_message" for event in runtime_events),
                    tool_calls=sum(event.event_type == "tool_call" for event in runtime_events),
                    file_inspections=sum(event.event_type == "file_inspection" for event in runtime_events),
                    code_executions=len(code_events),
                    failed_code_executions=sum(
                        event.execution_status in {"execution_error", "timeout", "session_terminated"}
                        for event in code_events
                    ),
                    truncated_outputs=sum(event.output_truncated is True for event in execution_events),
                    events_until_first_data_inspection=_events_until(runtime_events, "file_inspection"),
                    events_until_first_successful_code_execution=_events_until(
                        runtime_events,
                        "code_execution",
                        successful=True,
                    ),
                    events_until_submission=_events_until(runtime_events, "submission"),
                    execution_elapsed_seconds=sum(event.elapsed_seconds or 0.0 for event in execution_events),
                    provider_response_count=len(succeeded_provider_events),
                    provider_responses_with_usage=sum(
                        event.usage_status == "reported" for event in succeeded_provider_events
                    ),
                    prompt_tokens=sum(event.prompt_tokens for event in succeeded_provider_events),
                    completion_tokens=sum(event.completion_tokens for event in succeeded_provider_events),
                    total_tokens=sum(event.total_tokens for event in succeeded_provider_events),
                    provider_elapsed_seconds=sum(event.elapsed_seconds or 0.0 for event in succeeded_provider_events),
                    declared_diagnostics=diagnostics,
                    declared_sensitivity_analyses=sensitivities,
                    declared_uncertainty=uncertainty,
                )
            )
    return tuple(sorted(rows, key=lambda row: (row.model_id, row.assignment_id)))


def _collect_trialeval_ablation_action_trace(
    run_dirs: list[Path],
    *,
    run_ids: dict[Path, str],
) -> tuple[
    list[ModelActionTraceEventV1],
    list[TraceFeatureRowV1],
    list[FailureCascadeRowV1],
    list[SemanticActionFeatureRowV1],
]:
    """Project randomized TrialEval assignments into the generic trace contract."""

    events: list[ModelActionTraceEventV1] = []
    features: list[TraceFeatureRowV1] = []
    cascades: list[FailureCascadeRowV1] = []
    semantic_rows: list[SemanticActionFeatureRowV1] = []
    for raw_run_dir in sorted(run_dirs):
        run_dir = raw_run_dir.resolve()
        run_id = run_ids[run_dir]
        run_config = read_json_model(TrialEvalAblationRunConfigV1, run_dir / "run_config.json")
        trace_inputs = {
            row.assignment_id: row
            for row in collect_trialeval_trace_inputs([run_dir])
            if row.assignment_id is not None
        }
        assignment_paths = sorted((run_dir / "assignments").glob("*.json"))
        if set(trace_inputs) != {path.stem for path in assignment_paths}:
            raise ValueError(f"TrialEval ablation trace denominator mismatch: {run_dir}")
        for path in assignment_paths:
            result = read_json_model(TrialEvalAblationItemResultV1, path)
            if result.run_config != run_config:
                raise ValueError(f"TrialEval ablation trace has configuration drift: {path}")
            assignment = result.assignment
            trace_input = trace_inputs[assignment.assignment_id]
            runtime_path = run_dir / "events" / f"{assignment.assignment_id}_events.jsonl"
            conversation_path = run_dir / "traces" / f"{assignment.assignment_id}.json"
            if not runtime_path.is_file():
                raise FileNotFoundError(f"TrialEval ablation assignment lacks runtime events: {runtime_path}")
            if not conversation_path.is_file():
                raise FileNotFoundError(
                    f"TrialEval ablation runtime events require a conversation: {conversation_path}"
                )
            runtime_events = _runtime_events(runtime_path)
            _validate_runtime_events(
                runtime_events,
                task_id=assignment.task_id,
                conversation_path=conversation_path,
            )
            _require_submission_terminal(
                runtime_events,
                submission_expected=trace_input.answer_present,
                path=runtime_path,
            )
            unit_events: list[ModelActionTraceEventV1] = []
            event_type_map = {
                "step_started": "state_transition",
                "prompt": "prompt",
                "assistant_message": "assistant_message",
                "tool_call": "tool_call",
                "tool_result": "tool_result",
                "code_execution": "code_execution",
                "file_inspection": "file_inspection",
                "submission": "submission",
                "step_terminal": "state_transition",
            }
            for runtime_event in runtime_events:
                action_event = ModelActionTraceEventV1(
                    event_id=runtime_event.event_id,
                    timestamp=runtime_event.timestamp,
                    benchmark="trialeval",
                    model_id=run_config.model,
                    run_id=run_id,
                    task_id=assignment.task_id,
                    assignment_id=assignment.assignment_id,
                    phase_id=runtime_event.phase_id,
                    step_id=runtime_event.step_id,
                    turn_index=runtime_event.conversation_message_index,
                    event_index=runtime_event.event_index,
                    event_type=event_type_map[runtime_event.event_type],
                    source_path=runtime_path.as_posix(),
                    source_artifact_path=runtime_event.source_artifact_path,
                    source_payload_sha256=runtime_event.source_payload_sha256,
                    tool_call_id=runtime_event.tool_call_id,
                    tool_name=runtime_event.tool_name,
                    file_accessed=runtime_event.file_accessed,
                    elapsed_sec=runtime_event.elapsed_seconds,
                    execution_status=runtime_event.execution_status,
                    output_truncated=runtime_event.output_truncated,
                    status=runtime_event.status,
                )
                events.append(action_event)
                unit_events.append(action_event)

            submission = trace_input.submission
            submitted_evidence = submission.evidence if submission is not None else ()
            interval_reported = submission is not None and submission.primary_analysis.result.kind in {
                "scalar",
                "identified_interval",
            }
            flags = structured_feature_flags(
                submitted_evidence,
                primary_interval_reported=interval_reported,
            )
            feature = TraceFeatureRowV1(
                benchmark="trialeval",
                model_id=run_config.model,
                run_id=run_id,
                task_id=assignment.task_id,
                assignment_id=assignment.assignment_id,
                phase_id="task",
                trace_coverage_status="full_conversation_trace",
                inspected_public_data=any(event.event_type == "file_inspection" for event in unit_events),
                executed_code=any(event.event_type == "code_execution" for event in unit_events),
                submitted_structured_answer=assignment.submission_interface == "structured" and submission is not None,
                submitted_answer=trace_input.answer_present,
                submission_interface=assignment.submission_interface,
                submission_transport=trace_input.submission_transport,
                trace_input_authority=trace_input.authority,
                context_tier=trace_input.context_tier,
                data_preparation=trace_input.data_preparation,
                analysis_specification=trace_input.analysis_specification,
                procedure_assistance=trace_input.procedure_assistance,
                prompt_condition=trace_input.prompt_condition,
                checked_confounding=flags["checked_confounding"],
                checked_ph_assumption=flags["checked_ph_assumption"],
                checked_missingness=flags["checked_missingness"],
                checked_censoring=flags["checked_censoring"],
                quantified_uncertainty=flags["quantified_uncertainty"],
                used_sensitivity_analysis=flags["used_sensitivity_analysis"],
                considered_safety=None,
                considered_cost=None,
                objective_aligned_rationale=None,
                semantic_feature_source="structured_field" if submission is not None else "not_available",
                score_link_id=f"{run_id}:{assignment.assignment_id}",
                endpoint_valid=None,
                endpoint_state="not_scoreable_trace_only",
            )
            features.append(feature)
            cascades.append(phase_failure_cascade(feature))
            semantic_rows.extend(
                structured_feature_rows(
                    feature,
                    evidence=submitted_evidence,
                    primary_interval_reported=interval_reported,
                    evidence_basis=tuple(
                        sorted({artifact for record in submitted_evidence for artifact in record.source_artifacts})
                    ),
                )
            )
    return events, features, cascades, semantic_rows


def collect_trialeval_action_trace(
    run_dirs: list[Path],
) -> tuple[
    list[ModelActionTraceEventV1],
    list[TraceFeatureRowV1],
    list[EvidenceUseRowV1],
    list[FailureCascadeRowV1],
    list[SemanticActionFeatureRowV1],
]:
    """Collect TrialEval action-trace rows from stored result directories."""
    run_ids = require_unique_run_ids(run_dirs, suite="trialeval")
    ablation_run_dirs: list[Path] = []
    canonical_run_dirs: list[Path] = []
    for raw_run_dir in run_dirs:
        run_dir = raw_run_dir.resolve()
        config_path = run_dir / "run_config.json"
        config_document = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        if config_document.get("schema_id") == "trialagentbench.trialeval_ablation_run/v1":
            ablation_run_dirs.append(run_dir)
        else:
            canonical_run_dirs.append(run_dir)
    trace_inputs = {
        (trace_input.run_id, trace_input.task_id): trace_input
        for trace_input in collect_trialeval_trace_inputs(canonical_run_dirs)
        if trace_input.assignment_id is None
    }
    events: list[ModelActionTraceEventV1] = []
    features: list[TraceFeatureRowV1] = []
    evidence: list[EvidenceUseRowV1] = []
    cascades: list[FailureCascadeRowV1] = []
    semantic_rows: list[SemanticActionFeatureRowV1] = []
    if ablation_run_dirs:
        ablation_events, ablation_features, ablation_cascades, ablation_semantic = (
            _collect_trialeval_ablation_action_trace(ablation_run_dirs, run_ids=run_ids)
        )
        events.extend(ablation_events)
        features.extend(ablation_features)
        cascades.extend(ablation_cascades)
        semantic_rows.extend(ablation_semantic)
    for raw_run_dir in sorted(canonical_run_dirs):
        run_dir = raw_run_dir.resolve()
        run_label = run_ids[run_dir]
        for row in iter_trialeval_score_rows(run_dir):
            task_id = row.task_id
            model_id = row.model_id
            validates = row.grade.usable_primary
            complete_scoreable = row.grade.passed
            endpoint_state = "valid" if complete_scoreable else "failed"
            source_path = row.source_json_path
            source_artifact = Path(source_path)
            if not source_artifact.is_file():
                raise FileNotFoundError(f"TrialEval score source artifact is missing: {source_artifact}")
            event = ModelActionTraceEventV1(
                event_id=f"trialeval:{run_label}:{task_id}:grade",
                timestamp=datetime.fromtimestamp(0, tz=UTC),
                benchmark="trialeval",
                model_id=model_id,
                run_id=run_label,
                task_id=task_id,
                event_index=0,
                event_type="grade_link",
                source_path=source_path,
                source_artifact_path=source_path,
                source_payload_sha256=sha256_file(source_artifact),
                status="observed",
                content_chars=len(str(row)),
            )
            events.append(event)

            trace_input = trace_inputs.get((run_label, task_id))
            if trace_input is None:
                raise ValueError(f"TrialEval score lacks a typed trace input: {run_label}:{task_id}")
            runtime_path = run_dir / "logs" / f"{task_id}_events.jsonl"
            conversation_path = run_dir / "logs" / f"{task_id}_conversation.json"
            if not runtime_path.is_file():
                raise FileNotFoundError(f"TrialEval score row requires runner-native runtime events: {runtime_path}")
            if not conversation_path.is_file():
                raise FileNotFoundError(f"TrialEval runtime events require a conversation: {conversation_path}")
            runtime_events = _runtime_events(runtime_path)
            _validate_runtime_events(runtime_events, task_id=task_id, conversation_path=conversation_path)
            _require_submission_terminal(
                runtime_events,
                submission_expected=trace_input.submission_transport != "not_observed",
                path=runtime_path,
            )
            event_type_map = {
                "step_started": "state_transition",
                "prompt": "prompt",
                "assistant_message": "assistant_message",
                "tool_call": "tool_call",
                "tool_result": "tool_result",
                "code_execution": "code_execution",
                "file_inspection": "file_inspection",
                "submission": "submission",
                "step_terminal": "state_transition",
            }
            for runtime_event in runtime_events:
                events.append(
                    ModelActionTraceEventV1(
                        event_id=runtime_event.event_id,
                        timestamp=runtime_event.timestamp,
                        benchmark="trialeval",
                        model_id=model_id,
                        run_id=run_label,
                        task_id=task_id,
                        phase_id=runtime_event.phase_id,
                        step_id=runtime_event.step_id,
                        turn_index=runtime_event.conversation_message_index,
                        event_index=runtime_event.event_index,
                        event_type=event_type_map[runtime_event.event_type],
                        source_path=runtime_path.as_posix(),
                        source_artifact_path=runtime_event.source_artifact_path,
                        source_payload_sha256=runtime_event.source_payload_sha256,
                        tool_call_id=runtime_event.tool_call_id,
                        tool_name=runtime_event.tool_name,
                        file_accessed=runtime_event.file_accessed,
                        elapsed_sec=runtime_event.elapsed_seconds,
                        execution_status=runtime_event.execution_status,
                        output_truncated=runtime_event.output_truncated,
                        status=runtime_event.status,
                    )
                )

            submission = trace_input.submission
            submitted_evidence = submission.evidence if submission is not None else ()
            interval_reported = submission is not None and submission.primary_analysis.result.kind in {
                "scalar",
                "identified_interval",
            }
            evidence_flags = structured_feature_flags(
                submitted_evidence,
                primary_interval_reported=interval_reported,
            )
            flags = {
                key: evidence_flags[key]
                for key in (
                    "checked_confounding",
                    "checked_ph_assumption",
                    "checked_missingness",
                    "checked_censoring",
                    "quantified_uncertainty",
                    "used_sensitivity_analysis",
                    "considered_safety",
                    "considered_cost",
                    "objective_aligned_rationale",
                )
            }
            unit_events = [e for e in events if e.run_id == run_label and e.task_id == task_id]
            feature = TraceFeatureRowV1(
                benchmark="trialeval",
                model_id=model_id,
                run_id=run_label,
                task_id=task_id,
                phase_id="task",
                trace_coverage_status=(
                    "full_conversation_trace"
                    if runtime_events
                    else ("submission_only_trace" if validates else "scorer_only_trace")
                ),
                inspected_public_data=any(event.event_type == "file_inspection" for event in unit_events),
                executed_code=any(event.event_type == "code_execution" for event in unit_events),
                submitted_structured_answer=(
                    trace_input.submission_interface == "structured"
                    and (trace_input.submission is not None or validates)
                ),
                submitted_answer=trace_input.answer_present or validates,
                submission_interface=trace_input.submission_interface,
                submission_transport=trace_input.submission_transport,
                trace_input_authority=trace_input.authority,
                context_tier=trace_input.context_tier,
                data_preparation=trace_input.data_preparation,
                analysis_specification=trace_input.analysis_specification,
                procedure_assistance=trace_input.procedure_assistance,
                prompt_condition=trace_input.prompt_condition,
                checked_confounding=flags["checked_confounding"],
                checked_ph_assumption=flags["checked_ph_assumption"],
                checked_missingness=flags["checked_missingness"],
                checked_censoring=flags["checked_censoring"],
                quantified_uncertainty=flags["quantified_uncertainty"],
                used_sensitivity_analysis=flags["used_sensitivity_analysis"],
                considered_safety=None,
                considered_cost=None,
                objective_aligned_rationale=None,
                semantic_feature_source="structured_field" if submission is not None else "not_available",
                score_link_id=f"{run_label}:{task_id}",
                endpoint_valid=complete_scoreable,
                endpoint_state=endpoint_state,
            )
            features.append(feature)
            cascades.append(phase_failure_cascade(feature))
            semantic_rows.extend(
                structured_feature_rows(
                    feature,
                    evidence=submitted_evidence,
                    primary_interval_reported=interval_reported,
                    evidence_basis=tuple(
                        sorted({artifact for record in submitted_evidence for artifact in record.source_artifacts})
                    ),
                )
            )
    evidence = evidence_rows_from_events(events)
    return events, features, evidence, cascades, semantic_rows


def discover_trialeval_run_dirs(root: Path) -> list[Path]:
    """Discover persisted canonical and ablation TrialEval run directories."""

    configured = {
        path.parent
        for path in root.rglob("run_config.json")
        if (path.parent / "items").is_dir() or (path.parent / "assignments").is_dir()
    }
    return sorted(configured)


__all__ = [
    "collect_trialeval_ablation_observables",
    "collect_trialeval_action_trace",
    "collect_trialeval_trace_inputs",
    "discover_trialeval_run_dirs",
]
