"""Recompute canonical TrialDev grades from authenticated program ingress."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict
from trialagentbench_verifier.execution_contracts import AnalysisMode
from trialagentbench_verifier.io import trialdev_task_run_paths

from trialagentbench.contracts.trialdev.release import TrialDevTaskV1
from trialagentbench.contracts.trialdev.run import TrialDevRunIndexV1, TrialDevRunInvocationV1, TrialDevTaskRunV1
from trialagentbench.io import canonical_payload_sha256, read_json_model, sha256_file, write_json_model
from trialagentbench.ports import CodeExecutionLimitsV1
from trialagentbench.ports.tool_input import ToolInputError, parse_tool_arguments
from trialagentbench.trialdev.actions import TrialDevActionRequest, TrialDevActionSession, trialdev_action_names
from trialagentbench.trialdev.errors import TrialDevActionRejectedError
from trialagentbench.trialdev.ranking import rank_trialdev_run_v1
from trialagentbench.trialdev.release import TrialDevAuthoredProgrammeConfiguration, TrialDevEvaluatorRelease
from trialagentbench.trialdev.resources import TrialDevBudgetLedger
from trialagentbench.trialdev.run import (
    TrialDevAnalysisRun,
    TrialDevProgrammeRun,
    _numerical_artifact_paths,
    evaluate_trialdev_analysis,
    evaluate_trialdev_programme,
    trialdev_task_artifact_v1,
    verify_trialdev_task_run_artifacts_v1,
)
from trialagentbench.trialdev.task_agent import TrialDevAgentCheckpoint
from trialagentbench.util.parallel import run_bounded_tasks
from trialagentbench.util.reported_cost import RunStopSignal


class _Function(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    name: str
    arguments: str


class _Call(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    id: str
    function: _Function


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    role: str
    tool_calls: tuple[_Call, ...] | None = None
    tool_call_id: str | None = None
    content: str | None = None


def replay_trialdev_programme_actions(
    *,
    task_root: Path,
    pool_root: Path,
    configuration: TrialDevAuthoredProgrammeConfiguration,
) -> None:
    """Recreate committed actions in conversation order from the released initial state.

    Parameters
    ----------
    task_root
        Authenticated retained run containing agent state, requests, ledger and raw readouts.
    pool_root, configuration
        Released frozen pool and task-owned resource and recruitment configuration.

    Raises
    ------
    ValueError
        If requests are absent from history, differ from tool calls, or cannot
        reproduce the retained ledger, commitments and raw evidence.

    Notes
    -----
    Uses the canonical offline action engine; no provider, submitted code or
    native data generation is invoked. Independent arithmetic/readout checks are separate.
    """
    checkpoint = read_json_model(TrialDevAgentCheckpoint, task_root / "private/agent_state.json")
    if checkpoint.phase_id != "development" or checkpoint.step_id != "programme":
        raise ValueError("Authored action history identifies a different execution phase")
    original = task_root / "private/commitments"
    requests = {path.stem.removesuffix(".request"): path for path in original.glob("*.request.json")}
    observed: set[str] = set()
    messages = tuple(_Message.model_validate(raw) for raw in checkpoint.messages)
    acknowledged = {message.tool_call_id: message for message in messages if message.role == "tool"}
    pending = {call.id: call for call in checkpoint.pending_calls}
    expected = read_json_model(TrialDevBudgetLedger, task_root / "private/budget.json")
    programme_tools = frozenset(trialdev_action_names())
    with TemporaryDirectory(prefix="trialdev-action-replay-") as temporary:
        root = Path(temporary)
        budget = root / "private/budget.json"
        write_json_model(
            budget,
            TrialDevBudgetLedger(
                budget_usd=configuration.budget_usd,
                pool_manifest_sha256=configuration.pool_manifest_sha256,
                recruitment_seed=configuration.recruitment_seed,
            ),
        )
        replay = TrialDevActionSession(
            pool_root=pool_root,
            budget_path=budget,
            evidence_root=root / "evidence",
            commitment_root=root / "private/commitments",
            recruitment_seed=configuration.recruitment_seed,
        )
        for message in messages:
            if message.role != "assistant":
                continue
            for call in message.tool_calls or ():
                if call.function.name not in programme_tools:
                    continue
                identity = "action-" + canonical_payload_sha256({"step": "programme", "call": call.id})[:24]
                path = requests.get(identity)
                try:
                    raw = parse_tool_arguments(call.function.arguments, tool_name=call.function.name)
                    decoded = replay.decode(call.function.name, raw)
                except (ToolInputError, TrialDevActionRejectedError) as error:
                    if path is not None:
                        raise ValueError("Committed programme action does not decode from its tool call") from error
                    continue
                if decoded.preview:
                    if path is not None:
                        raise ValueError("A study preview cannot commit a programme action")
                    continue
                request = TrialDevActionRequest(request=decoded.action)
                if path is None:
                    reply = acknowledged.get(call.id)
                    if (
                        request.request.action in {"readout", "advance", "stop"}
                        and reply is not None
                        and (reply.content or "").startswith(
                            (
                                "Tool input rejected: Invalid scratch/estimates.csv:",
                                "Tool input rejected: Analysis capture rejected:",
                            )
                        )
                    ):
                        # Result validation can reject a cutoff before the action
                        # owner commits a request or changes programme state.
                        continue
                    if call.id in acknowledged:
                        raise ValueError("Acknowledged programme action lacks its committed request")
                    continue
                if read_json_model(TrialDevActionRequest, path) != request:
                    raise ValueError("Committed programme request differs from its recorded tool call")
                observed.add(identity)
                if call.id not in acknowledged and call.id in pending:
                    queued = pending[call.id]
                    if (
                        queued.name != call.function.name
                        or parse_tool_arguments(queued.arguments, tool_name=queued.name) != raw
                    ):
                        raise ValueError("Pending programme action differs from its recorded request")
                    if read_json_model(TrialDevBudgetLedger, budget) == expected:
                        # An interruption can persist the request before mutating
                        # the ledger. Preserve that request without inventing execution.
                        write_json_model(replay.commitment_root / path.name, request)
                        continue
                try:
                    replay.execute(request.request, action_id=identity)
                except TrialDevActionRejectedError:
                    # Rejected participant requests remain in history; their state
                    # consequences are checked by the final reconstruction below.
                    continue
        if observed != requests.keys():
            raise ValueError("Committed programme requests are absent from the recorded conversation")
        if read_json_model(TrialDevBudgetLedger, budget) != expected:
            raise ValueError("Programme action replay differs from the retained recruitment or resource ledger")
        for source, regenerated in (
            (original, replay.commitment_root),
            (task_root / "workspace/evidence", replay.evidence_root),
        ):
            source_files = {path.relative_to(source).as_posix(): path for path in source.rglob("*.json")}
            replay_files = {path.relative_to(regenerated).as_posix(): path for path in regenerated.rglob("*.json")}
            if source_files.keys() != replay_files.keys() or any(
                source_files[name].read_bytes() != replay_files[name].read_bytes() for name in source_files
            ):
                raise ValueError("Programme action replay differs from retained commitments or readout metadata")
        original_evidence = task_root / "workspace/evidence"
        original_files = {
            path.relative_to(original_evidence).as_posix(): sha256_file(path)
            for path in original_evidence.rglob("*")
            if path.is_file()
        }
        regenerated_files = {
            path.relative_to(replay.evidence_root).as_posix(): sha256_file(path)
            for path in replay.evidence_root.rglob("*")
            if path.is_file()
        }
        if original_files != regenerated_files:
            raise ValueError("Programme action replay differs from retained participant evidence")


def _task(release: TrialDevEvaluatorRelease, task_id: str) -> TrialDevTaskV1:
    matches = tuple(task for task in release.participant.tasks if task.task_id == task_id)
    if len(matches) != 1:
        raise ValueError(f"TrialDev release does not contain one task_id={task_id!r}")
    return matches[0]


def regrade_trialdev_task_directory(
    *,
    release: TrialDevEvaluatorRelease,
    task_root: Path,
    executor_limits: CodeExecutionLimitsV1 | None = None,
    analysis_language: AnalysisMode = "python_r",
    executor_image: str | None = None,
) -> TrialDevTaskRunV1:
    """Regrade one canonical task directory without a provider or construction code."""

    task_root = Path(task_root)
    run = read_json_model(TrialDevTaskRunV1, task_root / "run.json")
    verify_trialdev_task_run_artifacts_v1(task_root=task_root, run=run)
    if (
        run.release_id != release.manifest.release_id
        or run.release_manifest_sha256 != release.participant.release_manifest_sha256
    ):
        raise ValueError("TrialDev run and evaluator release identities differ")
    task = _task(release, run.task_id)
    if task.stream is not run.stream:
        raise ValueError("TrialDev task and run streams differ")
    if run.programme_relative_path is not None:
        if executor_limits is None or executor_image is None:
            raise ValueError("Authored regrading requires the invocation executor image and limits")
        configuration = release.authored_configuration(task)
        if configuration is None:
            raise ValueError("Authored run lacks a released programme configuration")
        budget_path = task_root / "private" / "budget.json"
        ledger = read_json_model(TrialDevBudgetLedger, budget_path)
        if (
            ledger.pool_manifest_sha256 != configuration.pool_manifest_sha256
            or ledger.recruitment_seed != configuration.recruitment_seed
            or ledger.budget_usd != configuration.budget_usd
        ):
            raise ValueError("Authored ledger differs from the release configuration")
        replay_trialdev_programme_actions(
            task_root=task_root,
            pool_root=release.root / "evaluation/worlds" / task.case_id / "pool",
            configuration=configuration,
        )
        previous = read_json_model(TrialDevProgrammeRun, task_root / run.programme_relative_path)
        evaluation = evaluate_trialdev_programme(
            workspace=task_root / "workspace",
            programme=TrialDevActionSession(
                pool_root=release.root / "evaluation" / "worlds" / task.case_id / "pool",
                budget_path=budget_path,
                evidence_root=task_root / "workspace" / "evidence",
                commitment_root=task_root / "private" / "commitments",
                recruitment_seed=configuration.recruitment_seed,
            ),
            preferences=configuration.preferences,
            initial_evidence=configuration.initial_evidence,
            executor_limits=executor_limits,
            analysis_language=analysis_language,
            executor_image=executor_image,
        )
        result = TrialDevProgrammeRun(
            execution_status=run.execution_status,
            error=run.error,
            evaluation=evaluation,
        )
        if previous.execution_status != run.execution_status or previous.error != run.error:
            raise ValueError("Regrading cannot change the recorded model execution outcome")
        write_json_model(task_root / run.programme_relative_path, result)
        payload = run.model_dump(mode="python", exclude={"checksum"})
        payload["artifacts"] = tuple(
            trialdev_task_artifact_v1(task_root, relative)
            for relative in _numerical_artifact_paths(task_root, programme=True)
        )
        updated = TrialDevTaskRunV1.model_validate(payload)
        write_json_model(task_root / "run.json", updated)
        return updated
    if run.analysis_relative_path is not None:
        if executor_limits is None or executor_image is None:
            raise ValueError("Numerical regrading requires the invocation executor image and limits")
        analysis_configuration = release.analysis_configuration(task)
        if analysis_configuration is None:
            raise ValueError("Numerical analysis lacks its released configuration")
        previous_analysis = read_json_model(TrialDevAnalysisRun, task_root / run.analysis_relative_path)
        if previous_analysis.execution_status != run.execution_status or previous_analysis.error != run.error:
            raise ValueError("Regrading cannot change the recorded model execution outcome")
        if previous_analysis.report_relative_path is not None and previous_analysis.report_relative_path not in {
            artifact.relative_path for artifact in run.artifacts
        }:
            raise ValueError("Standalone report is not bound by its result inventory")
        analysis = evaluate_trialdev_analysis(
            workspace=task_root / "workspace",
            configuration=analysis_configuration,
            report_relative_path=previous_analysis.report_relative_path,
            execution_status=previous_analysis.execution_status,
            error=previous_analysis.error,
            executor_limits=executor_limits,
            analysis_language=analysis_language,
            executor_image=executor_image,
        )
        write_json_model(task_root / run.analysis_relative_path, analysis)
        payload = run.model_dump(mode="python", exclude={"checksum"})
        payload["artifacts"] = tuple(
            trialdev_task_artifact_v1(task_root, relative)
            for relative in _numerical_artifact_paths(task_root, programme=False)
        )
        updated = TrialDevTaskRunV1.model_validate(payload)
        write_json_model(task_root / "run.json", updated)
        return updated
    raise ValueError("TrialDev regrading requires a current numerical or authored-programme result")


def regrade_trialdev_run(
    *,
    release: TrialDevEvaluatorRelease,
    run_root: Path,
) -> tuple[TrialDevTaskRunV1, ...]:
    """Regrade tasks at the recorded worker bound, then publish their ordered index."""

    roots = tuple(sorted(path.parent for path in trialdev_task_run_paths(Path(run_root)).values()))
    index_path = run_root / "run_index.json"
    index = read_json_model(TrialDevRunIndexV1, index_path) if index_path.is_file() else None
    invocation = read_json_model(TrialDevRunInvocationV1, run_root / "invocation.json")
    if index is not None:
        if (
            index.invocation_checksum != invocation.checksum
            or index.release_id != invocation.release_id
            or index.model != invocation.model
            or index.scheduled_task_ids != invocation.scheduled_task_ids
        ):
            raise ValueError("TrialDev input run index and invocation differ")
        prior_runs = tuple(read_json_model(TrialDevTaskRunV1, root / "run.json") for root in roots)
        before = {run.task_id: run.checksum for run in prior_runs}
        if len(before) != len(prior_runs):
            raise ValueError("TrialDev input repeats a task run")
        if before != {row.task_id: row.checksum for row in index.results}:
            raise ValueError("TrialDev input run index and task custody differ")

    def regrade(item: tuple[int, Path]) -> tuple[int, TrialDevTaskRunV1]:
        position, root = item
        return position, regrade_trialdev_task_directory(
            release=release,
            task_root=root,
            executor_limits=invocation.executor_limits,
            analysis_language=invocation.experiment_condition.analysis_language,
            executor_image=invocation.executor_image,
        )

    completed: list[tuple[int, TrialDevTaskRunV1]] = []
    run_bounded_tasks(
        enumerate(roots),
        regrade,
        workers=invocation.workers or 1,
        consume=completed.append,
        stop=RunStopSignal(),
    )
    runs = tuple(run for _, run in sorted(completed, key=lambda item: item[0]))
    if index is not None:
        updated_runs = {run.task_id: run for run in runs}
        payload = index.model_dump(mode="python", exclude={"checksum"})
        payload["results"] = tuple(
            row.model_copy(update={"checksum": updated_runs[row.task_id].checksum}) for row in index.results
        )
        write_json_model(index_path, TrialDevRunIndexV1.model_validate(payload))
    write_json_model(
        run_root / "ranking.json", rank_trialdev_run_v1(release=release, run_root=run_root, task_roots=roots)
    )
    return runs


__all__ = ["regrade_trialdev_run", "regrade_trialdev_task_directory"]
