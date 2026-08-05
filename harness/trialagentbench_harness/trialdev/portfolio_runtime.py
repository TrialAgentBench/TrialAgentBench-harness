"""Execute bounded TrialDev portfolios over immutable released evidence."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from trialagentbench_harness.contracts.core.config import ToolChoiceV1
from trialagentbench_harness.contracts.experiments import ProcedureAssistanceV1
from trialagentbench_harness.contracts.trace.observable import BenchmarkRuntimeTraceEventV1
from trialagentbench_harness.contracts.trialdev.portfolio_submission import (
    TrialDevPortfolioCheckpointGradeV1,
    TrialDevPortfolioCheckpointSubmissionV1,
    TrialDevPortfolioRunSummaryV1,
    TrialDevPortfolioSubmissionAttemptV1,
)
from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevCheckpointIdV1,
    TrialDevPortfolioCheckpointActionPolicyV1,
    TrialDevPortfolioEvidenceIndexV1,
    TrialDevPortfolioProgrammeStateV1,
)
from trialagentbench_harness.io import read_json_model, write_json, write_json_model
from trialagentbench_harness.ports import CodeExecutionLimitsV1, LLMProvider
from trialagentbench_harness.trialdev.agent import AgentLoop, AgentTurnLimitExceeded
from trialagentbench_harness.trialdev.participant_submission import (
    build_portfolio_checkpoint_v1,
    participant_payload_v1,
    portfolio_participant_schema_v1,
)
from trialagentbench_harness.trialdev.portfolio_grading import (
    PortfolioSubmissionError,
    grade_portfolio_checkpoint_v1,
)
from trialagentbench_harness.trialdev.portfolio_release import (
    initial_portfolio_state_v1,
    portfolio_evaluator_view_v1,
    portfolio_participant_view_v1,
    stage_portfolio_checkpoint_evidence_v1,
    stage_portfolio_public_view_v1,
)
from trialagentbench_harness.trialdev.programme import (
    build_checkpoint_action_policy_v1,
    transition_portfolio_programme_state_v1,
)
from trialagentbench_harness.util.provider_telemetry import read_provider_terminal_events_v1

_SUBMIT_TOOL_NAME = "submit_portfolio_checkpoint"
_SUBMIT_FILE_TOOL_NAME = "submit_portfolio_checkpoint_file"


@dataclass(frozen=True)
class PortfolioResourceOutcomesV1:
    """Exact resource projection from one programme's persisted event custody."""

    submission_attempts: int
    correction_count: int
    agent_turns: int
    execute_code_calls: int
    inspect_data_calls: int
    provider_calls: int
    provider_elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    provider_reported_usd: float | None


def portfolio_resource_outcomes_v1(programme_root: Path) -> PortfolioResourceOutcomesV1:
    attempts = tuple(
        read_json_model(TrialDevPortfolioSubmissionAttemptV1, path)
        for path in sorted((programme_root / "attempts").glob("*/*.json"))
    )
    runtime_events: list[BenchmarkRuntimeTraceEventV1] = []
    for path in sorted(programme_root.glob("events*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                runtime_events.append(BenchmarkRuntimeTraceEventV1.model_validate_json(line))
    provider_events = tuple(
        event
        for path in sorted(programme_root.glob("provider_responses*.jsonl"))
        for event in read_provider_terminal_events_v1(path)
    )
    succeeded = tuple(event for event in provider_events if event.status == "succeeded")
    reported_costs = tuple(event.reported_cost_usd for event in succeeded if event.reported_cost_usd is not None)
    all_costs_reported = bool(succeeded) and len(reported_costs) == len(succeeded)
    return PortfolioResourceOutcomesV1(
        submission_attempts=len(attempts),
        correction_count=sum(attempt.status != "accepted" for attempt in attempts),
        agent_turns=sum(event.event_type == "assistant_message" for event in runtime_events),
        execute_code_calls=sum(event.event_type == "code_execution" for event in runtime_events),
        inspect_data_calls=sum(
            event.event_type == "file_inspection" and event.tool_name == "inspect_parquet" for event in runtime_events
        ),
        provider_calls=len(provider_events),
        provider_elapsed_seconds=sum(float(event.elapsed_seconds or 0.0) for event in provider_events),
        prompt_tokens=sum(event.prompt_tokens for event in provider_events),
        completion_tokens=sum(event.completion_tokens for event in provider_events),
        provider_reported_usd=(sum(cast(tuple[float, ...], reported_costs)) if all_costs_reported else None),
    )


def _require_submission_attempt_available(*, attempts_used: int, maximum_attempts: int, checkpoint: str) -> None:
    """Stop before requesting a submission beyond the declared attempt budget."""

    if attempts_used >= maximum_attempts:
        raise AgentTurnLimitExceeded(f"Checkpoint {checkpoint!r} exceeded {maximum_attempts} submission attempts.")


def _participant_submission_schema() -> dict[str, object]:
    """Return the analyst-owned checkpoint fields."""

    return cast(dict[str, object], portfolio_participant_schema_v1())


def _submission_tools() -> list[dict[str, object]]:
    from trialagentbench_harness.trialdev.agent import tools_for_phase_analysis

    local_tools = [
        tool
        for tool in tools_for_phase_analysis()
        if str(tool.get("function", {}).get("name")) not in {"submit_phase_analysis", "submit_phase_analysis_file"}
    ]
    direct: dict[str, object] = {
        "type": "function",
        "function": {
            "name": _SUBMIT_TOOL_NAME,
            "description": "Submit the complete analysis, programme action, and any next studies for this checkpoint.",
            "parameters": _participant_submission_schema(),
        },
    }
    file_transport: dict[str, object] = {
        "type": "function",
        "function": {
            "name": _SUBMIT_FILE_TOOL_NAME,
            "description": "Submit one complete checkpoint JSON object from a UTF-8 file under scratch/.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to scratch/; do not include the scratch/ prefix.",
                    }
                },
                "required": ["path"],
            },
        },
    }
    return [*local_tools, direct, file_transport]


def _system_prompt(
    *,
    programme_id: str,
    objective_id: str,
    budget: int,
    procedure_assistance: ProcedureAssistanceV1,
) -> str:
    from trialagentbench_harness.trialdev.prompts import procedure_assistance_prompt

    return f"""You are the statistical lead for a clinical development portfolio.

PROGRAMME
* programme_id: {programme_id}
* objective: {objective_id}
* resource budget: {budget} units

CLINICAL QUESTION
At each checkpoint, what does the available evidence show about the candidate
regimens, and which next action does it support under the stated objective,
safety rules, and resource budget?

EVIDENCE AVAILABLE
The working directory contains the current programme files. At each checkpoint,
`programme_state.json` records the current programme state and
`current_action_policy.json` records the feasible actions. Randomized evidence
appears only after the corresponding study is reached. No future or unselected
study data are available.

WORKING APPROACH
{procedure_assistance_prompt(procedure_assistance)}

WORK REQUIRED
Analyse the participant-level records for the stated population, comparison,
outcome, time horizon, safety thresholds, and objective. State the method and
assumptions, and report the estimates and intervals used to reach the decision.
At observational review, a comparison is identified only under the stated
assumptions and the available treatment-assignment information; those
assumptions are not established merely by reporting an adjusted estimate. When
the available information does not support a point comparison, state the
limitation and withhold selection without inventing candidate estimates.

Use `runtime_submission_contracts.json` for the checkpoint schema. A scheduled
study must select the applicable `design_cell_id` from
`phase_design_policy.json`. Submit directly or write and review the JSON under
scratch/ before using `{_SUBMIT_FILE_TOOL_NAME}`. Free text does not complete a
checkpoint.

CONCLUSION
Select any feasible action supported by the analysis. When the evidence
supports more than one action, state the uncertainty; do not force a unique
ranking. Stop an unsafe or unsupported asset, and allocate further studies only
within the disclosed resource budget.
"""


def _checkpoint_prompt(
    *,
    state: TrialDevPortfolioProgrammeStateV1,
    evidence_directories: tuple[Path, ...],
) -> str:
    evidence = [path.relative_to(path.parents[1]).as_posix() for path in evidence_directories]
    return f"""CHECKPOINT: {state.current_checkpoint_id}

Review `programme_state.json`, `current_action_policy.json`, and the public
analysis and design policies. Current randomized evidence directories:
{json.dumps(evidence, indent=2) if evidence else "(observational evidence is at the workspace root)"}

Use only evidence referenced by the current state. Submit one complete typed
checkpoint record. Multiple actions may be defensible when confidence intervals
do not distinguish them; select any action supported by your reported evidence.
"""


def _load_evidence_index(release_root: Path, programme_id: str) -> TrialDevPortfolioEvidenceIndexV1:
    evaluator_view = portfolio_evaluator_view_v1(release_root, programme_id)
    return TrialDevPortfolioEvidenceIndexV1.model_validate(
        read_json_model(
            TrialDevPortfolioEvidenceIndexV1,
            Path(release_root) / evaluator_view.evidence_index_relative_path,
        )
    )


def _resume_state_v1(
    *,
    release_root: Path,
    programme_root: Path,
    summary: TrialDevPortfolioRunSummaryV1,
    index: TrialDevPortfolioEvidenceIndexV1,
) -> tuple[TrialDevPortfolioProgrammeStateV1, tuple[str, ...]]:
    """Revalidate accepted custody and return the first unfinished state."""

    states = tuple(
        read_json_model(TrialDevPortfolioProgrammeStateV1, programme_root / relative_path)
        for relative_path in summary.state_relative_paths
    )
    if not states:
        raise ValueError("Portfolio continuation requires an initial persisted state.")
    state_relative_paths = list(summary.state_relative_paths)
    reached: list[TrialDevCheckpointIdV1] = []
    resumed_state = states[-1]
    for position, (submission_relative, grade_relative) in enumerate(
        zip(summary.submission_relative_paths, summary.grade_relative_paths, strict=True)
    ):
        state = states[position]
        reached.append(state.current_checkpoint_id)
        submission = read_json_model(
            TrialDevPortfolioCheckpointSubmissionV1,
            programme_root / submission_relative,
        )
        persisted_grade = read_json_model(
            TrialDevPortfolioCheckpointGradeV1,
            programme_root / grade_relative,
        )
        reproduced_grade = grade_portfolio_checkpoint_v1(
            release_root=release_root,
            state=state,
            submission=submission,
        )
        if reproduced_grade != persisted_grade:
            raise ValueError(f"Portfolio continuation grade {position} does not reproduce from released evidence.")
        reproduced_state = transition_portfolio_programme_state_v1(
            state=state,
            evidence_index=index,
            action_policy=cast(
                TrialDevPortfolioCheckpointActionPolicyV1,
                build_checkpoint_action_policy_v1(state=state),
            ),
            selection=submission.selected_action,
            outcome=reproduced_grade.outcome,
        )
        resumed_state = reproduced_state
        if position + 1 < len(states):
            if reproduced_state != states[position + 1]:
                raise ValueError(
                    f"Portfolio continuation state {position + 1} does not reproduce from accepted custody."
                )
            continue
        recovered_path = (
            programme_root / "states" / f"{position + 1:03d}_{reproduced_state.current_checkpoint_id}.json"
        )
        write_json_model(recovered_path, reproduced_state)
        state_relative_paths.append(recovered_path.relative_to(programme_root).as_posix())
    state = resumed_state
    if state.terminal_disposition != "active":
        raise ValueError("An incomplete portfolio run cannot resume from a terminal state.")
    expected_reached = tuple(reached)
    if summary.reached_checkpoint_ids != expected_reached:
        raise ValueError("Portfolio continuation checkpoint inventory is inconsistent with its state history.")
    return TrialDevPortfolioProgrammeStateV1.model_validate(state), tuple(state_relative_paths)


def _execute_portfolio_v1(
    *,
    release: Path,
    programme_root: Path,
    provider: LLMProvider,
    state: TrialDevPortfolioProgrammeStateV1,
    index: TrialDevPortfolioEvidenceIndexV1,
    source_identity: str,
    participant_view_checksum: str,
    resource_budget_units: int,
    objective_id: str,
    scenario_id: str,
    state_paths: list[str],
    submission_paths: list[str],
    grade_paths: list[str],
    reached: list[str],
    checkpoint_index: int,
    execution_label: str,
    prior_wall_seconds: float,
    max_turns_per_checkpoint: int,
    max_tokens: int,
    max_context_characters: int,
    watchdog_seconds: int,
    max_submission_attempts: int,
    procedure_assistance: ProcedureAssistanceV1,
    tool_choice: ToolChoiceV1,
    executor_image: str | None,
    executor_limits: CodeExecutionLimitsV1,
    verbose: bool,
) -> TrialDevPortfolioRunSummaryV1:
    """Execute checkpoints from one independently validated programme state."""

    started_monotonic = time.monotonic()
    workspace = programme_root / "agent_workdir"
    states_root = programme_root / "states"
    submissions_root = programme_root / "submissions"
    grades_root = programme_root / "grades"
    deadline = time.monotonic() + watchdog_seconds
    log_suffix = "" if execution_label == "initial" else f"_{execution_label}"
    loop = AgentLoop(
        provider=provider,
        workdir=workspace,
        system_prompt=_system_prompt(
            programme_id=state.programme_id,
            objective_id=objective_id,
            budget=resource_budget_units,
            procedure_assistance=procedure_assistance,
        ),
        max_turns_per_step=max_turns_per_checkpoint,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        max_context_chars=max_context_characters,
        verbose=verbose,
        conversation_log_path=programme_root / f"conversation{log_suffix}.json",
        event_log_path=programme_root / f"events{log_suffix}.jsonl",
        provider_log_path=programme_root / f"provider_responses{log_suffix}.jsonl",
        program_id=state.programme_id,
        scenario_id=scenario_id,
        objective_id=objective_id,
        executor_image=executor_image,
        executor_limits=executor_limits,
        deadline_monotonic=deadline,
    )
    try:
        while state.terminal_disposition == "active":
            checkpoint = state.current_checkpoint_id
            evidence_directories: tuple[Path, ...] = ()
            if checkpoint != "observational_review":
                current_evidence = tuple(item for item in state.evidence if item.checkpoint_id == checkpoint)
                evidence_directories = stage_portfolio_checkpoint_evidence_v1(
                    release_root=release,
                    scenario_public_root=release
                    / portfolio_participant_view_v1(release, state.programme_id).public_scenario_relative_path,
                    evidence=current_evidence,
                    destination=workspace / f"{execution_label}_checkpoint_{checkpoint_index:03d}_{checkpoint}",
                )
            write_json(workspace / "programme_state.json", participant_payload_v1(state))
            action_policy = build_checkpoint_action_policy_v1(state=state)
            write_json(
                workspace / "current_action_policy.json",
                participant_payload_v1(action_policy),
            )
            loop.begin_step(phase_id=checkpoint, step_id="analysis_design_and_decision")
            loop.append_user_message(_checkpoint_prompt(state=state, evidence_directories=evidence_directories))
            submission_attempt_index = 0
            while True:
                _require_submission_attempt_available(
                    attempts_used=submission_attempt_index,
                    maximum_attempts=max_submission_attempts,
                    checkpoint=checkpoint,
                )
                capture = loop.run_until_submit(
                    tools=_submission_tools(),
                    submit_tool_names={_SUBMIT_TOOL_NAME, _SUBMIT_FILE_TOOL_NAME},
                )
                submission_attempt_index += 1
                attempt_path = (
                    programme_root
                    / "attempts"
                    / f"{checkpoint_index:03d}_{checkpoint}"
                    / f"{submission_attempt_index:03d}.json"
                )
                try:
                    submission = build_portfolio_checkpoint_v1(capture.payload, state=state)
                    grade = grade_portfolio_checkpoint_v1(
                        release_root=release,
                        state=state,
                        submission=submission,
                    )
                    next_state = transition_portfolio_programme_state_v1(
                        state=state,
                        evidence_index=index,
                        action_policy=cast(
                            TrialDevPortfolioCheckpointActionPolicyV1,
                            action_policy,
                        ),
                        selection=submission.selected_action,
                        outcome=grade.outcome,
                    )
                except (PortfolioSubmissionError, ValueError) as exc:
                    write_json_model(
                        attempt_path,
                        TrialDevPortfolioSubmissionAttemptV1(
                            checkpoint_id=checkpoint,
                            attempt_index=submission_attempt_index,
                            transport_name=capture.name,
                            status="contract_rejected",
                            submitted_payload=capture.payload,
                            validation_error=str(exc),
                        ),
                    )
                    loop.append_tool_reply(
                        capture.tool_call_id,
                        f"Checkpoint submission rejected: {exc}\nCorrect the evidence or action and resubmit.",
                        tool_name=capture.name,
                        status="invalid",
                    )
                    continue
                write_json_model(
                    attempt_path,
                    TrialDevPortfolioSubmissionAttemptV1(
                        checkpoint_id=checkpoint,
                        attempt_index=submission_attempt_index,
                        transport_name=capture.name,
                        status="accepted",
                        submitted_payload=capture.payload,
                        grade=grade,
                    ),
                )
                loop.append_tool_reply(
                    capture.tool_call_id,
                    json.dumps(
                        {
                            "status": "checkpoint recorded",
                            "checkpoint_id": checkpoint,
                            "selected_action": submission.selected_action.action_id,
                        }
                    ),
                    tool_name=capture.name,
                )
                break
            submission_path = submissions_root / f"{checkpoint_index:03d}_{checkpoint}.json"
            grade_path = grades_root / f"{checkpoint_index:03d}_{checkpoint}.json"
            write_json_model(submission_path, submission)
            write_json_model(grade_path, grade)
            submission_paths.append(submission_path.relative_to(programme_root).as_posix())
            grade_paths.append(grade_path.relative_to(programme_root).as_posix())
            reached.append(checkpoint)
            state = next_state
            checkpoint_index += 1
            state_path = states_root / f"{checkpoint_index:03d}_{state.current_checkpoint_id}.json"
            write_json_model(state_path, state)
            state_paths.append(state_path.relative_to(programme_root).as_posix())
        resources = portfolio_resource_outcomes_v1(programme_root)
        summary = TrialDevPortfolioRunSummaryV1(
            programme_id=state.programme_id,
            scenario_id=scenario_id,
            objective_id=objective_id,
            resource_budget_units=resource_budget_units,
            participant_view_checksum=participant_view_checksum,
            release_source_identity=source_identity,
            execution_status="completed",
            terminal_disposition=state.terminal_disposition,
            reached_checkpoint_ids=tuple(reached),
            state_relative_paths=tuple(state_paths),
            submission_relative_paths=tuple(submission_paths),
            grade_relative_paths=tuple(grade_paths),
            wall_seconds_total=prior_wall_seconds + time.monotonic() - started_monotonic,
            submission_attempts=resources.submission_attempts,
            correction_count=resources.correction_count,
            agent_turns=resources.agent_turns,
            execute_code_calls=resources.execute_code_calls,
            inspect_data_calls=resources.inspect_data_calls,
            provider_calls=resources.provider_calls,
            provider_elapsed_seconds=resources.provider_elapsed_seconds,
            prompt_tokens=resources.prompt_tokens,
            completion_tokens=resources.completion_tokens,
            provider_reported_usd=resources.provider_reported_usd,
        )
    except AgentTurnLimitExceeded as exc:
        resources = portfolio_resource_outcomes_v1(programme_root)
        summary = TrialDevPortfolioRunSummaryV1(
            programme_id=state.programme_id,
            scenario_id=scenario_id,
            objective_id=objective_id,
            resource_budget_units=resource_budget_units,
            participant_view_checksum=participant_view_checksum,
            release_source_identity=source_identity,
            execution_status="model_noncompletion",
            reached_checkpoint_ids=tuple(reached),
            state_relative_paths=tuple(state_paths),
            submission_relative_paths=tuple(submission_paths),
            grade_relative_paths=tuple(grade_paths),
            wall_seconds_total=prior_wall_seconds + time.monotonic() - started_monotonic,
            submission_attempts=resources.submission_attempts,
            correction_count=resources.correction_count,
            agent_turns=resources.agent_turns,
            execute_code_calls=resources.execute_code_calls,
            inspect_data_calls=resources.inspect_data_calls,
            provider_calls=resources.provider_calls,
            provider_elapsed_seconds=resources.provider_elapsed_seconds,
            prompt_tokens=resources.prompt_tokens,
            completion_tokens=resources.completion_tokens,
            provider_reported_usd=resources.provider_reported_usd,
            error=str(exc),
        )
    finally:
        loop.close()
    write_json_model(programme_root / "portfolio_run_summary.json", summary)
    return summary


def run_portfolio_programme_v1(
    *,
    release_root: Path,
    programme_id: str,
    output_root: Path,
    provider: LLMProvider,
    max_turns_per_checkpoint: int,
    max_tokens: int,
    max_context_characters: int,
    watchdog_seconds: int,
    max_submission_attempts: int,
    procedure_assistance: ProcedureAssistanceV1,
    tool_choice: ToolChoiceV1 = "auto",
    executor_image: str | None = None,
    executor_limits: CodeExecutionLimitsV1 | None = None,
    verbose: bool = False,
) -> TrialDevPortfolioRunSummaryV1:
    """Run one portfolio programme until a legal terminal action or noncompletion."""

    from trialagentbench_harness.trialdev.portfolio_release import (
        load_portfolio_catalogue_v1,
        portfolio_participant_view_v1,
    )

    if max_submission_attempts < 1:
        raise ValueError("max_submission_attempts must be at least 1.")
    release = Path(release_root).resolve(strict=True)
    resolved_executor_limits = executor_limits or CodeExecutionLimitsV1()
    catalogue = load_portfolio_catalogue_v1(release)
    view = portfolio_participant_view_v1(release, programme_id)
    programme_root = Path(output_root) / "programs" / programme_id
    workspace = programme_root / "agent_workdir"
    if programme_root.exists():
        raise FileExistsError(f"Portfolio programme output already exists: {programme_root}")
    programme_root.mkdir(parents=True)
    stage_portfolio_public_view_v1(release_root=release, view=view, workdir=workspace)
    write_json(
        workspace / "runtime_submission_contracts.json",
        {
            "schema_id": "trialagentbench.trialdev_portfolio_runtime_submission_contracts/v1",
            "checkpoint": _participant_submission_schema(),
        },
    )
    index = _load_evidence_index(release, programme_id)
    state = initial_portfolio_state_v1(view)
    states_root = programme_root / "states"
    submissions_root = programme_root / "submissions"
    grades_root = programme_root / "grades"
    for path in (states_root, submissions_root, grades_root):
        path.mkdir()
    state_paths: list[str] = []
    submission_paths: list[str] = []
    grade_paths: list[str] = []
    reached: list[str] = []
    initial_state_path = states_root / "000_observational_review.json"
    write_json_model(initial_state_path, state)
    state_paths.append(initial_state_path.relative_to(programme_root).as_posix())
    return _execute_portfolio_v1(
        release=release,
        programme_root=programme_root,
        provider=provider,
        state=state,
        index=index,
        source_identity=catalogue.source_identity,
        participant_view_checksum=str(view.checksum),
        resource_budget_units=view.resource_budget_units,
        objective_id=str(view.objective_id),
        scenario_id=view.scenario_id,
        state_paths=state_paths,
        submission_paths=submission_paths,
        grade_paths=grade_paths,
        reached=reached,
        checkpoint_index=0,
        execution_label="initial",
        prior_wall_seconds=0.0,
        max_turns_per_checkpoint=max_turns_per_checkpoint,
        max_tokens=max_tokens,
        max_context_characters=max_context_characters,
        watchdog_seconds=watchdog_seconds,
        max_submission_attempts=max_submission_attempts,
        procedure_assistance=procedure_assistance,
        tool_choice=tool_choice,
        executor_image=executor_image,
        executor_limits=resolved_executor_limits,
        verbose=verbose,
    )


def resume_portfolio_programme_v1(
    *,
    release_root: Path,
    programme_id: str,
    output_root: Path,
    provider: LLMProvider,
    max_turns_per_checkpoint: int,
    max_tokens: int,
    max_context_characters: int,
    watchdog_seconds: int,
    max_submission_attempts: int,
    procedure_assistance: ProcedureAssistanceV1,
    tool_choice: ToolChoiceV1 = "auto",
    executor_image: str | None = None,
    executor_limits: CodeExecutionLimitsV1 | None = None,
    verbose: bool = False,
) -> TrialDevPortfolioRunSummaryV1:
    """Continue one incomplete portfolio from its last accepted checkpoint."""

    from trialagentbench_harness.trialdev.portfolio_release import load_portfolio_catalogue_v1

    if max_submission_attempts < 1:
        raise ValueError("max_submission_attempts must be at least 1.")
    release = Path(release_root).resolve(strict=True)
    resolved_executor_limits = executor_limits or CodeExecutionLimitsV1()
    catalogue = load_portfolio_catalogue_v1(release)
    view = portfolio_participant_view_v1(release, programme_id)
    programme_root = Path(output_root) / "programs" / programme_id
    if not programme_root.is_dir():
        raise FileNotFoundError(f"Portfolio programme output does not exist: {programme_root}")
    summary = read_json_model(
        TrialDevPortfolioRunSummaryV1,
        programme_root / "portfolio_run_summary.json",
    )
    expected_identity = (
        programme_id,
        view.scenario_id,
        str(view.objective_id),
        str(view.checksum),
        catalogue.source_identity,
    )
    persisted_identity = (
        summary.programme_id,
        summary.scenario_id,
        summary.objective_id,
        summary.participant_view_checksum,
        summary.release_source_identity,
    )
    if persisted_identity != expected_identity:
        raise ValueError("Portfolio continuation identity conflicts with the released programme.")
    if summary.execution_status == "completed":
        raise ValueError("A completed portfolio programme cannot be resumed.")
    index = _load_evidence_index(release, programme_id)
    state, state_paths = _resume_state_v1(
        release_root=release,
        programme_root=programme_root,
        summary=summary,
        index=index,
    )
    resume_number = 1
    while (programme_root / f"conversation_resume_{resume_number:03d}.json").exists():
        resume_number += 1
    return _execute_portfolio_v1(
        release=release,
        programme_root=programme_root,
        provider=provider,
        state=state,
        index=index,
        source_identity=catalogue.source_identity,
        participant_view_checksum=str(view.checksum),
        resource_budget_units=view.resource_budget_units,
        objective_id=str(view.objective_id),
        scenario_id=view.scenario_id,
        state_paths=list(state_paths),
        submission_paths=list(summary.submission_relative_paths),
        grade_paths=list(summary.grade_relative_paths),
        reached=list(summary.reached_checkpoint_ids),
        checkpoint_index=len(summary.grade_relative_paths),
        execution_label=f"resume_{resume_number:03d}",
        prior_wall_seconds=summary.wall_seconds_total,
        max_turns_per_checkpoint=max_turns_per_checkpoint,
        max_tokens=max_tokens,
        max_context_characters=max_context_characters,
        watchdog_seconds=watchdog_seconds,
        max_submission_attempts=max_submission_attempts,
        procedure_assistance=procedure_assistance,
        tool_choice=tool_choice,
        executor_image=executor_image,
        executor_limits=resolved_executor_limits,
        verbose=verbose,
    )


__all__ = [
    "PortfolioResourceOutcomesV1",
    "portfolio_resource_outcomes_v1",
    "resume_portfolio_programme_v1",
    "run_portfolio_programme_v1",
]
