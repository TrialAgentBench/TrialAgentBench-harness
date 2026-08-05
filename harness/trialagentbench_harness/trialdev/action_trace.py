"""Action-trace extraction for TrialDevBench stored run trees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

from pydantic.types import JsonValue

from trialagentbench_harness.analysis.evidence_use import evidence_rows_from_events
from trialagentbench_harness.analysis.failure_cascade import phase_failure_cascade, program_failure_cascades
from trialagentbench_harness.analysis.run_identity import require_unique_run_ids
from trialagentbench_harness.analysis.semantic_features import structured_feature_flags, structured_feature_rows
from trialagentbench_harness.analysis.trialdev_ingestion import (
    TrialDevAnalysisDataset,
    load_trialdev_analysis_dataset,
)
from trialagentbench_harness.contracts.core.runs import TrialDevRunConfigV1
from trialagentbench_harness.contracts.trace.observable import (
    BenchmarkRuntimeTraceEventV1,
    EvidenceUseRowV1,
    FailureCascadeRowV1,
    ModelActionTraceEventV1,
    ProgramFailureCascadeRowV1,
    SemanticActionFeatureRowV1,
    TraceFeatureRowV1,
    TrialDevPhaseOutcomeRowV1,
    runtime_event_source_payload_v1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256

TrialDevPhaseId = Literal["observational_review", "phase1", "phase2", "phase3"]
PHASE_ORDER: tuple[TrialDevPhaseId, ...] = ("observational_review", "phase1", "phase2", "phase3")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_config(run_dir: Path) -> TrialDevRunConfigV1:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"TrialDev run is missing model provenance: {config_path}")
    return TrialDevRunConfigV1.model_validate(_read_json(config_path))


def _read_text_if_exists(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _read_runtime_events(path: Path) -> tuple[BenchmarkRuntimeTraceEventV1, ...]:
    events: list[BenchmarkRuntimeTraceEventV1] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                events.append(BenchmarkRuntimeTraceEventV1.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid TrialDev runtime event at {path}:{line_number}") from exc
    if [event.event_index for event in events] != list(range(len(events))):
        raise ValueError(f"TrialDev runtime event indices must be contiguous: {path}")
    if len({event.event_id for event in events}) != len(events):
        raise ValueError(f"TrialDev runtime event IDs must be unique: {path}")
    active_route: tuple[str, str] | None = None
    for event in events:
        route = (event.phase_id, event.step_id)
        if event.event_type == "step_started":
            if active_route is not None:
                raise ValueError(f"TrialDev runtime step is missing a terminal event: {path}")
            active_route = route
        elif active_route is None or route != active_route:
            raise ValueError(f"TrialDev runtime event route does not join the active step: {path}")
        elif event.event_type == "step_terminal":
            active_route = None
    if active_route is not None:
        raise ValueError(f"TrialDev runtime step is missing a terminal event: {path}")
    return tuple(events)


def _validate_runtime_event_link(
    event: BenchmarkRuntimeTraceEventV1,
    *,
    messages: list[dict[str, Any]],
    path: Path,
) -> None:
    """Require runtime events to point to the exact saved conversation record."""

    if event.event_type in {"step_started", "step_terminal"}:
        if event.conversation_message_index is not None:
            raise ValueError(f"Runtime lifecycle event must not reference a conversation message: {path}")
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
            raise ValueError(f"TrialDev runtime lifecycle payload hash mismatch: {path}")
        return
    message_index = event.conversation_message_index
    if message_index is None or message_index >= len(messages):
        raise ValueError(f"Runtime event references a missing conversation message: {path}")
    message = messages[message_index]
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
        raise ValueError(f"TrialDev runtime conversation payload hash mismatch: {path}")
    role = str(message.get("role") or "")
    expected_role = {
        "prompt": "user",
        "assistant_message": "assistant",
        "tool_result": "tool",
    }.get(event.event_type)
    if expected_role is not None and role != expected_role:
        raise ValueError(f"Runtime event role mismatch at conversation message {message_index}: {path}")
    if event.event_type == "tool_result":
        if str(message.get("tool_call_id") or "") != str(event.tool_call_id or ""):
            raise ValueError(f"Runtime tool result id mismatch at conversation message {message_index}: {path}")
        return
    if event.event_type not in {"tool_call", "code_execution", "file_inspection", "submission"}:
        return
    if role != "assistant" or event.tool_call_index is None:
        raise ValueError(f"Runtime tool event lacks an assistant tool-call link: {path}")
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list) or event.tool_call_index >= len(tool_calls):
        raise ValueError(f"Runtime event references a missing tool call: {path}")
    tool_call = tool_calls[event.tool_call_index]
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(function, dict):
        raise ValueError(f"Runtime event references a malformed tool call: {path}")
    if str(tool_call.get("id") or "") != str(event.tool_call_id or ""):
        raise ValueError(f"Runtime tool call id mismatch: {path}")
    if str(function.get("name") or "") != str(event.tool_name or ""):
        raise ValueError(f"Runtime tool call name mismatch: {path}")


def trialdev_submission_paths(program_dir: Path, phase_id: str) -> tuple[Path, ...]:
    """Return current-contract submission paths for a TrialDev phase."""
    if phase_id == "observational_review":
        return (
            program_dir / "obs_review" / "obs_review_submission.json",
            program_dir / "obs_review" / "agent_obs_review_payload.json",
        )
    if phase_id in {"phase1", "phase2", "phase3"}:
        phase_dir = program_dir / "agent_workdir" / f"phase_{phase_id}"
        return (
            phase_dir / "analysis_submission.json",
            phase_dir / "decision_submission.json",
        )
    raise ValueError(f"Unsupported TrialDev phase_id: {phase_id}")


def read_trialdev_submission_text(program_dir: Path, phase_id: str) -> tuple[bool, str, tuple[str, ...]]:
    """Read current-contract TrialDev submission artifacts for a phase."""
    paths = trialdev_submission_paths(program_dir, phase_id)
    existing = tuple(path for path in paths if path.is_file())
    texts = [_read_text_if_exists(path) for path in existing]
    return bool(existing), "\n".join(texts), tuple(path.as_posix() for path in existing)


def _require_submission_route_join(
    *,
    phase_id: TrialDevPhaseId,
    submission_paths: tuple[str, ...],
    phase_events: list[ModelActionTraceEventV1],
    identity: str,
) -> None:
    """Require persisted artifacts and successful semantic submission routes to agree."""

    artifact_names = {Path(path).name for path in submission_paths}
    if phase_id == "observational_review":
        expected_names = {"obs_review_submission.json", "agent_obs_review_payload.json"}
        if artifact_names and artifact_names != expected_names:
            raise ValueError(f"TrialDev observational submission is incomplete: {identity}")
        artifact_routes = {"analysis_and_decision"} if artifact_names else set()
        semantic_routes = {"analysis_and_decision"}
    else:
        route_by_artifact = {
            "analysis_submission.json": "trial_analysis",
            "decision_submission.json": "phase_decision",
        }
        unknown_names = artifact_names - set(route_by_artifact)
        if unknown_names:
            raise ValueError(f"TrialDev submission has unknown artifact names {sorted(unknown_names)!r}: {identity}")
        artifact_routes = {route_by_artifact[name] for name in artifact_names}
        semantic_routes = set(route_by_artifact.values())
    native_routes = {
        event.step_id
        for event in phase_events
        if event.event_type == "submission" and event.status == "observed" and event.step_id in semantic_routes
    }
    if native_routes != artifact_routes:
        raise ValueError(
            "TrialDev persisted submission artifacts disagree with runner-native semantic routes: "
            f"{identity}; artifacts={sorted(artifact_routes)!r}, events={sorted(native_routes)!r}"
        )


def _submitted_rationale(paths: tuple[str, ...]) -> bool:
    """Return whether structured submission artifacts contain a rationale."""

    for raw_path in paths:
        payload = _read_json(Path(raw_path))
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("decision_rationale"), str) and payload["decision_rationale"].strip():
            return True
        analysis = payload.get("analysis_report")
        if (
            isinstance(analysis, dict)
            and isinstance(analysis.get("evidence_summary"), str)
            and analysis["evidence_summary"].strip()
        ):
            return True
    return False


def collect_trialdev_phase_outcomes(run_dirs: list[Path]) -> tuple[TrialDevPhaseOutcomeRowV1, ...]:
    """Collect phase outcomes through the canonical typed analysis boundary."""

    return load_trialdev_analysis_dataset(run_dirs).phases


def collect_trialdev_action_trace(
    run_dirs: list[Path],
    *,
    trialdev_release_root: Path | None = None,
    analysis_dataset: TrialDevAnalysisDataset | None = None,
) -> tuple[
    list[ModelActionTraceEventV1],
    list[TraceFeatureRowV1],
    list[EvidenceUseRowV1],
    list[FailureCascadeRowV1],
    list[SemanticActionFeatureRowV1],
    list[TrialDevPhaseOutcomeRowV1],
    list[ProgramFailureCascadeRowV1],
]:
    """Collect TrialDev action-trace rows from stored run directories."""
    run_ids = require_unique_run_ids(run_dirs, suite="trialdev")
    events: list[ModelActionTraceEventV1] = []
    features: list[TraceFeatureRowV1] = []
    cascades: list[FailureCascadeRowV1] = []
    semantic_rows: list[SemanticActionFeatureRowV1] = []
    dataset = analysis_dataset or load_trialdev_analysis_dataset(run_dirs)
    phase_outcomes = list(dataset.phases)
    outcomes_by_key = {(row.model_id, row.run_id, row.program_id, row.phase_id): row for row in phase_outcomes}

    for raw_run_dir in sorted(run_dirs):
        run_dir = raw_run_dir.resolve()
        run_label = run_ids[run_dir]
        run_config = _run_config(run_dir)
        model_id = run_config.model
        programs_root = run_dir / "programs"
        if not programs_root.is_dir():
            continue
        for program_dir in sorted(path for path in programs_root.iterdir() if path.is_dir()):
            program_id = program_dir.name
            program_events: list[ModelActionTraceEventV1] = []
            conversation_path = program_dir / "conversation.json"
            if not conversation_path.is_file():
                raise FileNotFoundError(
                    f"TrialDev analysis requires a runner-native conversation: {conversation_path}"
                )
            payload = _read_json(conversation_path)
            if not isinstance(payload, list) or not all(isinstance(message, dict) for message in payload):
                raise ValueError(f"TrialDev conversation must be a list of message objects: {conversation_path}")
            messages = cast(list[dict[str, Any]], payload)
            runtime_events_path = program_dir / "events.jsonl"
            if not runtime_events_path.is_file():
                raise FileNotFoundError(
                    f"TrialDev analysis requires runner-native runtime events: {runtime_events_path}"
                )
            runtime_events = _read_runtime_events(runtime_events_path)
            if not runtime_events:
                raise ValueError(f"TrialDev runtime event stream is empty: {runtime_events_path}")
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
                if runtime_event.benchmark != "trialdev" or runtime_event.program_id != program_id:
                    raise ValueError(f"TrialDev runtime event identity mismatch: {runtime_events_path}")
                if runtime_event.phase_id not in PHASE_ORDER:
                    raise ValueError(f"TrialDev runtime event has invalid phase_id: {runtime_events_path}")
                runtime_phase = cast(TrialDevPhaseId, runtime_event.phase_id)
                runtime_outcome = outcomes_by_key[(model_id, run_label, program_id, runtime_phase)]
                if (
                    runtime_event.scenario_id != runtime_outcome.scenario_id
                    or runtime_event.objective_id != runtime_outcome.objective_id
                ):
                    raise ValueError(f"TrialDev runtime event context mismatch: {runtime_events_path}")
                _validate_runtime_event_link(runtime_event, messages=messages, path=runtime_events_path)
                message_index = runtime_event.conversation_message_index
                content_chars = None
                if message_index is not None and message_index < len(messages):
                    message = messages[message_index]
                    content = "" if message.get("content") is None else str(message.get("content"))
                    content_chars = len(content)
                action_event = ModelActionTraceEventV1(
                    event_id=runtime_event.event_id,
                    timestamp=runtime_event.timestamp,
                    benchmark="trialdev",
                    model_id=model_id,
                    run_id=run_label,
                    program_id=program_id,
                    scenario_id=runtime_event.scenario_id,
                    objective_id=runtime_event.objective_id,
                    phase_id=runtime_event.phase_id,
                    step_id=runtime_event.step_id,
                    turn_index=message_index,
                    event_index=runtime_event.event_index,
                    event_type=event_type_map[runtime_event.event_type],
                    source_path=runtime_events_path.as_posix(),
                    source_artifact_path=runtime_event.source_artifact_path,
                    source_payload_sha256=runtime_event.source_payload_sha256,
                    tool_call_id=runtime_event.tool_call_id,
                    tool_name=runtime_event.tool_name,
                    file_accessed=runtime_event.file_accessed,
                    elapsed_sec=runtime_event.elapsed_seconds,
                    execution_status=runtime_event.execution_status,
                    output_truncated=runtime_event.output_truncated,
                    status=runtime_event.status,
                    content_chars=content_chars,
                )
                events.append(action_event)
                program_events.append(action_event)
            for phase in PHASE_ORDER:
                submitted, _, submission_paths = read_trialdev_submission_text(program_dir, phase)
                phase_events = [event for event in program_events if event.phase_id == phase]
                outcome = outcomes_by_key[(model_id, run_label, program_id, phase)]
                _require_submission_route_join(
                    phase_id=phase,
                    submission_paths=submission_paths,
                    phase_events=phase_events,
                    identity=f"{run_label}:{program_id}:{phase}",
                )
                evidence_flags = structured_feature_flags(
                    (),
                    primary_interval_reported=False,
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
                feature = TraceFeatureRowV1(
                    benchmark="trialdev",
                    model_id=model_id,
                    run_id=run_label,
                    program_id=program_id,
                    scenario_id=outcome.scenario_id,
                    objective_id=outcome.objective_id,
                    phase_id=phase,
                    trace_coverage_status="full_conversation_trace" if runtime_events else "submission_only_trace",
                    inspected_public_data=any(event.event_type == "file_inspection" for event in phase_events),
                    executed_code=any(event.event_type == "code_execution" for event in phase_events),
                    submitted_structured_answer=submitted,
                    procedure_assistance=run_config.procedure_assistance,
                    submitted_rationale=_submitted_rationale(submission_paths),
                    semantic_feature_source="not_available",
                    endpoint_valid=outcome.endpoint_state == "valid",
                    endpoint_state=outcome.endpoint_state,
                    score_link_id=outcome.score_link_id,
                    submission_artifact_paths=submission_paths,
                    **flags,
                )
                features.append(feature)
                cascades.append(phase_failure_cascade(feature))
                semantic_rows.extend(
                    structured_feature_rows(
                        feature,
                        evidence=(),
                        primary_interval_reported=False,
                    )
                )
    evidence = evidence_rows_from_events(events, trialdev_release_root=trialdev_release_root)
    program_cascades = program_failure_cascades(phase_outcomes, evidence)
    return (
        events,
        features,
        evidence,
        cascades,
        semantic_rows,
        phase_outcomes,
        program_cascades,
    )


def discover_trialdev_run_dirs(root: Path) -> list[Path]:
    """Discover TrialDev run directories containing a `programs` tree."""
    return sorted(
        {
            path.parent
            for path in root.rglob("programs")
            if path.is_dir() and (path.parent / "results_full.csv").is_file()
        }
    )


__all__ = [
    "collect_trialdev_action_trace",
    "collect_trialdev_phase_outcomes",
    "discover_trialdev_run_dirs",
    "read_trialdev_submission_text",
    "trialdev_submission_paths",
]
