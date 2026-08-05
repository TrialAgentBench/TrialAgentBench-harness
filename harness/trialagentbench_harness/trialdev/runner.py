"""Per-program orchestrator wrapping the upstream sequential state machine.

For each program (one ``(scenario_id, objective_id)``), this module:

1. Stages a per-program working directory containing only the public
   surface (no hidden/grader paths reachable by the agent).
2. Runs the agent through the observational review and commits its nomination
   or withholding decision to the same programme state used thereafter.
3. Loops phase1 -> phase2 -> phase3,
   with the agent submitting request -> (harness materializes) ->
   analysis -> decision at each phase, and ``advance_program_state_v1``
   transitioning state between phases.
4. Persists complete execution custody for deterministic offline grading.

Returns a ``ProgramRun`` populated with paths to every artefact for
downstream grading and aggregation.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC  # type: ignore[attr-defined]
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from trialagentbench_harness.adapters import trialdev_upstream
from trialagentbench_harness.adapters.trialdev_share import (
    TrialDevelopmentPhaseActionPolicyV1,
    TrialDevelopmentPhaseActionSpecV1,
    TrialDevelopmentRequestV1,
    TrialDevProgrammeStateV1,
    TrialDevPublicObservationalMethodSpecV1,
    candidate_ids_by_role_v1,
)
from trialagentbench_harness.contracts.core.config import ToolChoiceV1
from trialagentbench_harness.contracts.core.runs import TrialDevMaterializationUsageV1
from trialagentbench_harness.contracts.experiments import ProcedureAssistanceV1
from trialagentbench_harness.contracts.release.artifacts import (
    TRIALDEV_FIXED_TRAJECTORY_REPLICATE_MEMBERS,
)
from trialagentbench_harness.contracts.trialdev.run_checkpoint import (
    TrialDevRunCheckpointPayloadV1,
    TrialDevRunCheckpointPhaseV1,
    TrialDevRunCheckpointV1,
)
from trialagentbench_harness.contracts.trialdev.runtime_checkpoint import (
    TrialDevCheckpointArtifactV1,
    TrialDevCheckpointPhaseSummaryV1,
    TrialDevCheckpointViolationV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_public_phase_replay import (
    TrialDevPhaseReplayCaseV1,
)
from trialagentbench_harness.execution_policy import TRIALDEV_RELEASE_BUDGET_V1
from trialagentbench_harness.io import read_json_model, sha256_path
from trialagentbench_harness.ports import CodeExecutionLimitsV1, LLMProvider
from trialagentbench_harness.trialdev import agent as agent_mod
from trialagentbench_harness.trialdev import bridge, prompts, transcript
from trialagentbench_harness.trialdev.data import scenario_root, stage_working_dir
from trialagentbench_harness.trialdev.grading.models import TrialDevelopmentSubmissionV1
from trialagentbench_harness.trialdev.participant_submission import build_observational_review_v1
from trialagentbench_harness.trialdev.schema import (
    MaterializationRecord,
    MaterializationUsage,
    PhaseAttempt,
    PhaseId,
    Program,
    ProgramRun,
)

logger = logging.getLogger(__name__)


class CheckpointCaptureComplete(RuntimeError):
    """Signal that a requested pre-response checkpoint was durably captured."""


@dataclass
class RunOptions:
    bundle_root: Path
    output_root: Path
    model: str
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only"
    tool_choice: ToolChoiceV1 = "auto"
    execution_scope: Literal["full_programme", "observational_review_only"] = "full_programme"
    observational_analysis_specification: TrialDevPublicObservationalMethodSpecV1 | None = None
    master_seed: int = 42
    temperature: float = 0.0
    max_tokens: int = TRIALDEV_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn
    max_context_chars: int = TRIALDEV_RELEASE_BUDGET_V1.maximum_context_characters
    request_timeout_seconds: float = TRIALDEV_RELEASE_BUDGET_V1.provider_request_timeout_seconds
    max_turns_per_step: int = agent_mod.DEFAULT_MAX_TURNS_PER_STEP
    max_phase_retries: int = 10  # how many times the materializer can reject before we give up
    program_watchdog_seconds: int = TRIALDEV_RELEASE_BUDGET_V1.wall_time_limit_seconds
    verbose: bool = False
    executor_image: str | None = None
    executor_limits: CodeExecutionLimitsV1 = field(default_factory=CodeExecutionLimitsV1)
    run_identity_sha256: str | None = None
    checkpoint_observer: Callable[[TrialDevRunCheckpointV1], None] | None = None
    checkpoint_context_mode: agent_mod.CheckpointContextMode = "exact"
    continuation_budget_scope: Literal["remaining_program", "checkpoint_local"] = "remaining_program"

    def __post_init__(self) -> None:
        """Reject invalid execution budgets before creating run artifacts."""

        if self.max_turns_per_step < 1:
            raise ValueError("max_turns_per_step must be at least 1.")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")
        if self.max_context_chars < 1:
            raise ValueError("max_context_chars must be at least 1.")
        if not 0.0 < self.request_timeout_seconds <= 900.0:
            raise ValueError("request_timeout_seconds must be in (0, 900].")
        if self.max_phase_retries < 1:
            raise ValueError("max_phase_retries must be at least 1.")
        if self.program_watchdog_seconds < 1:
            raise ValueError("program_watchdog_seconds must be at least 1.")
        if self.checkpoint_context_mode not in {"exact", "active_step_only"}:
            raise ValueError("checkpoint_context_mode must be exact or active_step_only.")
        if self.continuation_budget_scope not in {"remaining_program", "checkpoint_local"}:
            raise ValueError("continuation_budget_scope must be remaining_program or checkpoint_local.")
        if self.observational_analysis_specification is not None:
            if self.execution_scope != "observational_review_only":
                raise ValueError(
                    "An observational analysis specification is only valid for observational_review_only runs."
                )
            if self.procedure_assistance != "output_contract_only":
                raise ValueError(
                    "Prespecified observational execution requires output_contract_only assistance "
                    "so workflow assistance remains held fixed."
                )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def _checkpoint_artifact(path: Path, *, root: Path) -> TrialDevCheckpointArtifactV1:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"TrialDev checkpoint artifact is outside program custody: {path}") from exc
    if resolved.is_file():
        kind = "file"
    elif resolved.is_dir():
        kind = "directory"
    else:
        raise FileNotFoundError(resolved)
    return TrialDevCheckpointArtifactV1(
        relative_path=relative,
        kind=kind,
        sha256=sha256_path(resolved),
    )


def _checkpoint_phase(
    attempt: PhaseAttempt,
    *,
    program_dir: Path,
) -> TrialDevRunCheckpointPhaseV1:
    materialization = attempt.materializations[-1] if attempt.materializations else None
    return TrialDevRunCheckpointPhaseV1(
        phase_id=cast(Literal["phase1", "phase2", "phase3"], attempt.phase_id),
        request=(
            _checkpoint_artifact(attempt.request_path, root=program_dir) if attempt.request_path is not None else None
        ),
        trial_output=(
            _checkpoint_artifact(attempt.trial_output_root, root=program_dir)
            if attempt.trial_output_root is not None
            else None
        ),
        materialization_seed=materialization.seed if materialization is not None else None,
        request_checksum=(materialization.request_checksum if materialization is not None else None),
        trial_output_checksum=(materialization.trial_output_checksum if materialization is not None else None),
        analysis=(
            _checkpoint_artifact(attempt.analysis_path, root=program_dir)
            if attempt.analysis_path is not None
            else None
        ),
        decision=(
            _checkpoint_artifact(attempt.decision_path, root=program_dir)
            if attempt.decision_path is not None
            else None
        ),
        matched_item_id=attempt.matched_item_id,
        decision_action=attempt.decision_action,
        advance=attempt.advance,
        candidate_drug_id=attempt.candidate_drug_id,
    )


def _runtime_phase_summary(summary: dict[str, Any]) -> TrialDevCheckpointPhaseSummaryV1:
    return TrialDevCheckpointPhaseSummaryV1.model_validate(summary)


def _runtime_violation(violation: dict[str, Any]) -> TrialDevCheckpointViolationV1:
    return TrialDevCheckpointViolationV1.model_validate(violation)


def _persist_run_checkpoint(
    *,
    program_dir: Path,
    loop: agent_mod.AgentLoop,
    options: RunOptions,
    current_state_path: Path,
    usage: MaterializationUsage,
    prior_phase_summaries: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    completed_phases: list[PhaseAttempt],
    pending_operation: Literal[
        "observational_review",
        "phase_request",
        "materialize",
        "phase_analysis",
        "phase_decision",
        "advance_state",
    ],
    current_phase: PhaseAttempt | None,
) -> TrialDevRunCheckpointV1:
    if options.run_identity_sha256 is None:
        raise ValueError("TrialDev continuation requires the authoritative run identity.")
    checkpoint_dir = program_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(checkpoint_dir.glob("*.json"))
    sequence = 0
    previous = None
    if existing:
        prior = _load_checkpoint_chain(
            program_dir,
            run_identity_sha256=options.run_identity_sha256,
        )
        sequence = prior.payload.sequence + 1
        previous = prior.payload_sha256
    continuation = loop.create_checkpoint(
        custody_root=program_dir,
        current_state_path=current_state_path,
        materialization_usage=TrialDevMaterializationUsageV1(
            materialize_calls_by_phase=dict(usage.materialize_calls_by_phase)
        ),
        completed_phase_summaries=tuple(_runtime_phase_summary(summary) for summary in prior_phase_summaries),
        violations=tuple(_runtime_violation(violation) for violation in violations),
    )
    payload = TrialDevRunCheckpointPayloadV1(
        sequence=sequence,
        previous_checkpoint_sha256=previous,
        run_identity_sha256=options.run_identity_sha256,
        pending_operation=pending_operation,
        continuation=continuation,
        completed_phases=tuple(_checkpoint_phase(attempt, program_dir=program_dir) for attempt in completed_phases),
        current_phase=(
            _checkpoint_phase(current_phase, program_dir=program_dir) if current_phase is not None else None
        ),
    )
    checkpoint = TrialDevRunCheckpointV1.create(payload)
    path = checkpoint_dir / f"{sequence:08d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            checkpoint.model_dump(mode="json"),
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    if options.checkpoint_observer is not None:
        options.checkpoint_observer(checkpoint)
    return checkpoint


def run_program(program: Program, *, options: RunOptions, provider: LLMProvider) -> ProgramRun:
    """Run one program end-to-end. Returns a ProgramRun describing artefacts."""
    import time as _time
    from datetime import datetime

    started_at_utc = datetime.now(UTC)
    started_monotonic = _time.monotonic()

    program_dir = options.output_root / "programs" / program.program_id
    workdir = program_dir / "agent_workdir"
    if program_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing program directory: {program_dir}")
    program_dir.mkdir(parents=True, exist_ok=True)

    src_root = scenario_root(options.bundle_root, program.scenario_id)
    stage_working_dir(
        options.bundle_root,
        program.scenario_id,
        workdir,
        procedure_assistance=options.procedure_assistance,
    )
    if options.observational_analysis_specification is not None:
        (workdir / "observational_analysis_specification.json").write_text(
            options.observational_analysis_specification.model_dump_json(indent=2),
            encoding="utf-8",
        )
    agent_mod.write_runtime_submission_contracts(workdir)
    # Public files are staged flat inside workdir, so the agent can read them
    # with simple relative paths like ``observational_extract.parquet``.
    public_dir = workdir
    action_policy = bridge.load_action_policy(src_root)
    action_specs_by_phase = {str(spec.phase_id): spec for spec in action_policy.action_specs}
    missing_action_specs = [phase for phase in ("phase1", "phase2", "phase3") if phase not in action_specs_by_phase]
    if missing_action_specs:
        raise ValueError(f"Action policy is missing phase contracts: {', '.join(missing_action_specs)}")

    run = ProgramRun(
        program_id=program.program_id,
        scenario_id=program.scenario_id,
        objective_id=program.objective_id,
        workdir=workdir,
    )

    usage = MaterializationUsage()

    system_prompt = prompts.build_system_prompt(
        program,
        public_dir,
        max_turns_per_step=options.max_turns_per_step,
        procedure_assistance=options.procedure_assistance,
        observational_analysis_specification=options.observational_analysis_specification,
    )
    loop = agent_mod.AgentLoop(
        provider=provider,
        workdir=workdir,
        system_prompt=system_prompt,
        temperature=float(options.temperature),
        max_tokens=int(options.max_tokens),
        max_context_chars=int(options.max_context_chars),
        max_turns_per_step=options.max_turns_per_step,
        tool_choice=options.tool_choice,
        verbose=options.verbose,
        conversation_log_path=program_dir / "conversation.json",
        event_log_path=program_dir / "events.jsonl",
        provider_log_path=program_dir / "provider_responses.jsonl",
        program_id=program.program_id,
        scenario_id=program.scenario_id,
        objective_id=program.objective_id,
        executor_image=options.executor_image,
        executor_limits=options.executor_limits,
        deadline_monotonic=started_monotonic + options.program_watchdog_seconds,
    )
    state_dir = program_dir / "states"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_initial_path = state_dir / "state_initial.json"
    trialdev_upstream.build_initial_program_state(
        scenario_root=src_root,
        programme_id=program.program_id,
        objective_id=program.objective_id,
        out_path=state_initial_path,
    )
    current_state_path = state_initial_path
    prior_phase_summaries: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    run.violations = violations
    interrupted = False

    def checkpoint(
        operation: Literal[
            "observational_review",
            "phase_request",
            "materialize",
            "phase_analysis",
            "phase_decision",
            "advance_state",
        ],
        current_phase: PhaseAttempt | None,
    ) -> None:
        _persist_run_checkpoint(
            program_dir=program_dir,
            loop=loop,
            options=options,
            current_state_path=current_state_path,
            usage=usage,
            prior_phase_summaries=prior_phase_summaries,
            violations=violations,
            completed_phases=run.phases,
            pending_operation=operation,
            current_phase=current_phase,
        )

    try:
        # ── Step 1: observational_review (pre-state-machine) ────────────────
        if program.has_obs_review:
            observational_submission = _run_obs_review(
                program=program,
                program_dir=program_dir,
                loop=loop,
                usage=usage,
                src_root=src_root,
                checkpoint=lambda: checkpoint("observational_review", None),
            )
            if _unsupported_observational_nomination(
                submission=observational_submission,
                scenario_root=src_root,
            ):
                run.execution_status = "model_invalid_submission"
                run.error = (
                    "The nominated asset has no randomized continuation in the released " "programme action space."
                )
                run.stopped_at_phase = "observational_review"
                return run
            observational_state_path = state_dir / "state_after_observational_review.json"
            current_state = trialdev_upstream.advance_observational_programme_state(
                state=trialdev_upstream.load_program_state(current_state_path),
                submission=observational_submission,
                submission_path=program_dir / "obs_review" / "obs_review_submission.json",
                out_path=observational_state_path,
            )
            current_state_path = observational_state_path
            if current_state.terminal_disposition != "active":
                run.stopped_at_phase = "observational_review"
                return run
            if options.execution_scope == "observational_review_only":
                run.stopped_at_phase = "observational_review"
                return run

        if not program.materializing_phases():
            # Some programs (e.g. cost_effective_best public-only) have no
            # phase1+ items. The trajectory grader has nothing to grade in
            # this case — record obs_review only and return.
            return run

        current_state = trialdev_upstream.load_program_state(current_state_path)

        while current_state.terminal_disposition == "active":
            phase_id = _phase_id_for_state(current_state)
            if phase_id not in action_specs_by_phase:
                raise ValueError(f"Programme state reached unsupported phase: {phase_id!r}.")
            # Phase artefacts (request, trial output, submissions) live inside
            # the agent's workdir so the trial parquets have simple relative
            # paths. ``grade_trajectory_v1`` uses rglob to
            # find them so the nesting doesn't matter for grading.
            phase_dir = workdir / f"phase_{phase_id}"
            phase_dir.mkdir(parents=True, exist_ok=True)

            attempt = _run_one_phase(
                phase_id=phase_id,
                public_dir=public_dir,
                phase_dir=phase_dir,
                state_path=current_state_path,
                program=program,
                loop=loop,
                usage=usage,
                options=options,
                src_root=src_root,
                prior_phase_summaries=prior_phase_summaries,
                violations=violations,
                action_policy=action_policy,
                action_spec=action_specs_by_phase[phase_id],
                checkpoint=checkpoint,
            )
            run.phases.append(attempt)
            prior_phase_summaries.append(_summarize_phase_attempt(attempt))

            # Every accepted decision advances state, including stopping and terminal decisions.
            next_state_path = state_dir / f"state_after_{phase_id}.json"
            request_path = _require_path(attempt.request_path, f"{phase_id} request")
            trial_output_root = _require_path(attempt.trial_output_root, f"{phase_id} trial output")
            analysis_path = _require_path(attempt.analysis_path, f"{phase_id} analysis")
            decision_path = _require_path(attempt.decision_path, f"{phase_id} decision")
            trialdev_upstream.advance_program_state(
                scenario_root=src_root,
                state_path=current_state_path,
                request_path=request_path,
                trial_output_root=trial_output_root,
                analysis_path=analysis_path,
                decision_path=decision_path,
                out_path=next_state_path,
            )
            current_state_path = next_state_path
            current_state = trialdev_upstream.load_program_state(current_state_path)
            if not attempt.advance:
                if current_state.terminal_disposition == "active":
                    raise ValueError("A non-advancing decision did not produce a terminal programme state.")
                run.stopped_at_phase = phase_id
                break

        # ── Step 2b: final-program submission ────────────────────────────
    except agent_mod.AgentTurnLimitExceeded as exc:
        run.execution_status = "model_turn_limit"
        run.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.info("Program reached the model turn limit: %s", program.program_id)
    except trialdev_upstream.TrialMaterializationRejectedError as exc:
        run.execution_status = "model_invalid_submission"
        run.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.info("Program exhausted the materialization correction budget: %s", program.program_id)
    except CheckpointCaptureComplete:
        interrupted = True
        raise
    except TimeoutError:
        logger.exception("Program run timed out: %s", program.program_id)
        interrupted = True
        raise
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError):
        logger.exception("Program run failed: %s", program.program_id)
        interrupted = True
        raise
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        raise
    finally:
        if interrupted:
            if loop.session is not None:
                loop.session.close()
                loop.session = None
        else:
            if run.execution_status == "running":
                run.execution_status = "completed"
            ended_at_utc = datetime.now(UTC)
            run.started_at_utc = started_at_utc.isoformat()
            run.ended_at_utc = ended_at_utc.isoformat()
            run.wall_seconds_total = _time.monotonic() - started_monotonic
            _persist_chain_summary(program_dir, run, usage)
            _persist_conversation(program_dir, loop)
            loop.close()

    return run


def _resolve_checkpoint_artifact(
    artifact: TrialDevCheckpointArtifactV1,
    *,
    program_dir: Path,
) -> Path:
    path = (program_dir / str(artifact.relative_path)).resolve()
    try:
        path.relative_to(program_dir.resolve())
    except ValueError as exc:
        raise ValueError("TrialDev checkpoint artifact escaped program custody.") from exc
    if artifact.kind == "file" and not path.is_file():
        raise FileNotFoundError(path)
    if artifact.kind == "directory" and not path.is_dir():
        raise NotADirectoryError(path)
    if sha256_path(path) != artifact.sha256:
        raise ValueError(f"TrialDev checkpoint artifact checksum drift: {artifact.relative_path}")
    return path


def _restore_phase_attempt(
    phase: TrialDevRunCheckpointPhaseV1,
    *,
    program_dir: Path,
) -> PhaseAttempt:
    attempt = PhaseAttempt(
        phase_id=phase.phase_id,
        matched_item_id=phase.matched_item_id,
        decision_action=phase.decision_action,
        advance=phase.advance,
        candidate_drug_id=phase.candidate_drug_id,
    )
    if phase.request is not None:
        attempt.request_path = _resolve_checkpoint_artifact(phase.request, program_dir=program_dir)
    if phase.trial_output is not None:
        attempt.trial_output_root = _resolve_checkpoint_artifact(
            phase.trial_output,
            program_dir=program_dir,
        )
        if (
            phase.materialization_seed is None
            or phase.request_checksum is None
            or phase.trial_output_checksum is None
            or attempt.request_path is None
        ):
            raise ValueError("TrialDev restored materialization custody is incomplete.")
        attempt.materializations.append(
            MaterializationRecord(
                phase_id=phase.phase_id,
                seed=phase.materialization_seed,
                request_path=attempt.request_path,
                request_checksum=phase.request_checksum,
                trial_output_root=attempt.trial_output_root,
                trial_output_checksum=phase.trial_output_checksum,
            )
        )
    if phase.analysis is not None:
        attempt.analysis_path = _resolve_checkpoint_artifact(phase.analysis, program_dir=program_dir)
    if phase.decision is not None:
        attempt.decision_path = _resolve_checkpoint_artifact(phase.decision, program_dir=program_dir)
    return attempt


def _checkpoint_reset_log_paths(
    program_dir: Path,
    *,
    context_mode: agent_mod.CheckpointContextMode,
) -> tuple[Path | None, Path | None, Path | None]:
    """Return isolated log destinations only for active-step context replay."""

    if context_mode == "exact":
        return None, None, None
    return (
        program_dir / "checkpoint_conversation.json",
        program_dir / "checkpoint_events.jsonl",
        program_dir / "checkpoint_provider_responses.jsonl",
    )


def _load_checkpoint_chain(
    program_dir: Path,
    *,
    run_identity_sha256: str,
) -> TrialDevRunCheckpointV1:
    paths = sorted((program_dir / "checkpoints").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"TrialDev partial program has no continuation checkpoints: {program_dir}")
    previous: str | None = None
    latest: TrialDevRunCheckpointV1 | None = None
    for sequence, path in enumerate(paths):
        checkpoint = read_json_model(TrialDevRunCheckpointV1, path)
        payload = checkpoint.payload
        if path.name != f"{sequence:08d}.json" or payload.sequence != sequence:
            raise ValueError("TrialDev checkpoint chain sequence is not contiguous.")
        if payload.previous_checkpoint_sha256 != previous:
            raise ValueError("TrialDev checkpoint predecessor hash mismatch.")
        if payload.run_identity_sha256 != run_identity_sha256:
            raise ValueError("TrialDev checkpoint run identity mismatch.")
        previous = checkpoint.payload_sha256
        latest = checkpoint
    if latest is None:
        raise RuntimeError("TrialDev checkpoint discovery returned no latest checkpoint.")
    return latest


def resume_program(program: Program, *, options: RunOptions, provider: LLMProvider) -> ProgramRun:
    """Continue one partial program from its latest append-only checkpoint."""

    import time as _time
    from datetime import datetime

    if options.run_identity_sha256 is None:
        raise ValueError("TrialDev continuation requires the authoritative run identity.")
    started_at_utc = datetime.now(UTC)
    started_monotonic = _time.monotonic()
    program_dir = options.output_root / "programs" / program.program_id
    workdir = program_dir / "agent_workdir"
    if not program_dir.is_dir():
        raise FileNotFoundError(program_dir)
    specification_path = workdir / "observational_analysis_specification.json"
    if options.observational_analysis_specification is None:
        if specification_path.exists():
            raise ValueError("TrialDev continuation unexpectedly contains an observational analysis specification.")
    else:
        persisted_specification = read_json_model(
            type(options.observational_analysis_specification),
            specification_path,
        )
        if persisted_specification != options.observational_analysis_specification:
            raise ValueError("TrialDev continuation observational analysis specification mismatch.")
    checkpoint_envelope = _load_checkpoint_chain(
        program_dir,
        run_identity_sha256=options.run_identity_sha256,
    )
    custody = checkpoint_envelope.payload
    continuation = custody.continuation
    if (
        continuation.payload.program_id,
        continuation.payload.scenario_id,
        continuation.payload.objective_id,
    ) != (program.program_id, program.scenario_id, program.objective_id):
        raise ValueError("TrialDev checkpoint program identity mismatch.")

    src_root = scenario_root(options.bundle_root, program.scenario_id)
    action_policy = bridge.load_action_policy(src_root)
    action_specs_by_phase = {str(spec.phase_id): spec for spec in action_policy.action_specs}
    missing_action_specs = [phase for phase in ("phase1", "phase2", "phase3") if phase not in action_specs_by_phase]
    if missing_action_specs:
        raise ValueError(f"Action policy is missing phase contracts: {', '.join(missing_action_specs)}")
    system_prompt = prompts.build_system_prompt(
        program,
        workdir,
        max_turns_per_step=options.max_turns_per_step,
        procedure_assistance=options.procedure_assistance,
        observational_analysis_specification=options.observational_analysis_specification,
    )
    reset_conversation_log, reset_event_log, reset_provider_log = _checkpoint_reset_log_paths(
        program_dir,
        context_mode=options.checkpoint_context_mode,
    )
    checkpoint_deadline_monotonic = (
        started_monotonic + options.program_watchdog_seconds
        if options.continuation_budget_scope == "checkpoint_local"
        else None
    )
    loop = agent_mod.AgentLoop.restore_from_checkpoint(
        continuation,
        provider=provider,
        custody_root=program_dir,
        system_prompt=system_prompt,
        verbose=options.verbose,
        conversation_log_path=program_dir / "conversation.json",
        event_log_path=program_dir / "events.jsonl",
        provider_log_path=program_dir / "provider_responses.jsonl",
        context_mode=options.checkpoint_context_mode,
        reset_conversation_log_path=reset_conversation_log,
        reset_event_log_path=reset_event_log,
        reset_provider_log_path=reset_provider_log,
        checkpoint_deadline_monotonic=checkpoint_deadline_monotonic,
    )
    current_state_path = _resolve_checkpoint_artifact(
        continuation.payload.current_state,
        program_dir=program_dir,
    )
    usage = MaterializationUsage(
        materialize_calls_by_phase=dict(continuation.payload.materialization_usage.materialize_calls_by_phase)
    )
    prior_phase_summaries = [
        summary.model_dump(mode="python") for summary in continuation.payload.completed_phase_summaries
    ]
    violations = [violation.model_dump(mode="python") for violation in continuation.payload.violations]
    run = ProgramRun(
        program_id=program.program_id,
        scenario_id=program.scenario_id,
        objective_id=program.objective_id,
        workdir=workdir,
        phases=[_restore_phase_attempt(phase, program_dir=program_dir) for phase in custody.completed_phases],
        violations=violations,
    )
    state_dir = program_dir / "states"
    interrupted = False

    def persist(
        operation: Literal[
            "observational_review",
            "phase_request",
            "materialize",
            "phase_analysis",
            "phase_decision",
            "advance_state",
        ],
        current_phase: PhaseAttempt | None,
    ) -> None:
        _persist_run_checkpoint(
            program_dir=program_dir,
            loop=loop,
            options=options,
            current_state_path=current_state_path,
            usage=usage,
            prior_phase_summaries=prior_phase_summaries,
            violations=violations,
            completed_phases=run.phases,
            pending_operation=operation,
            current_phase=current_phase,
        )

    try:
        operation = custody.pending_operation
        if operation == "observational_review":
            observational_submission = _run_obs_review(
                program=program,
                program_dir=program_dir,
                loop=loop,
                usage=usage,
                src_root=src_root,
                checkpoint=lambda: persist("observational_review", None),
                start_step=False,
            )
            if _unsupported_observational_nomination(
                submission=observational_submission,
                scenario_root=src_root,
            ):
                run.execution_status = "model_invalid_submission"
                run.error = (
                    "The nominated asset has no randomized continuation in the released " "programme action space."
                )
                run.stopped_at_phase = "observational_review"
                return run
            observational_state_path = state_dir / "state_after_observational_review.json"
            current_state = trialdev_upstream.advance_observational_programme_state(
                state=trialdev_upstream.load_program_state(current_state_path),
                submission=observational_submission,
                submission_path=program_dir / "obs_review" / "obs_review_submission.json",
                out_path=observational_state_path,
            )
            current_state_path = observational_state_path
            if current_state.terminal_disposition != "active":
                run.stopped_at_phase = "observational_review"
                return run
            if options.execution_scope == "observational_review_only":
                run.stopped_at_phase = "observational_review"
                return run
            operation = "phase_request"
            restored_attempt = None
        else:
            if custody.current_phase is None:
                raise ValueError("TrialDev phase continuation is missing current phase custody.")
            restored_attempt = _restore_phase_attempt(
                custody.current_phase,
                program_dir=program_dir,
            )

        current_state = trialdev_upstream.load_program_state(current_state_path)
        while current_state.terminal_disposition == "active":
            phase_id = _phase_id_for_state(current_state)
            if phase_id not in action_specs_by_phase:
                raise ValueError(f"Programme state reached unsupported phase: {phase_id!r}.")
            if restored_attempt is not None and restored_attempt.phase_id != phase_id:
                raise ValueError("TrialDev checkpoint current phase does not match the upstream state.")
            phase_dir = workdir / f"phase_{phase_id}"
            phase_dir.mkdir(parents=True, exist_ok=True)
            attempt = _run_one_phase(
                phase_id=phase_id,
                public_dir=workdir,
                phase_dir=phase_dir,
                state_path=current_state_path,
                program=program,
                loop=loop,
                usage=usage,
                options=options,
                src_root=src_root,
                prior_phase_summaries=prior_phase_summaries,
                violations=violations,
                action_policy=action_policy,
                action_spec=action_specs_by_phase[phase_id],
                checkpoint=persist,
                resume_operation=cast(
                    Literal[
                        "phase_request",
                        "materialize",
                        "phase_analysis",
                        "phase_decision",
                        "advance_state",
                    ],
                    operation,
                ),
                restored_attempt=restored_attempt,
            )
            run.phases.append(attempt)
            prior_phase_summaries.append(_summarize_phase_attempt(attempt))
            restored_attempt = None
            operation = "phase_request"
            next_state_path = state_dir / f"state_after_{phase_id}.json"
            if not next_state_path.exists():
                trialdev_upstream.advance_program_state(
                    scenario_root=src_root,
                    state_path=current_state_path,
                    request_path=_require_path(attempt.request_path, f"{phase_id} request"),
                    trial_output_root=_require_path(
                        attempt.trial_output_root,
                        f"{phase_id} trial output",
                    ),
                    analysis_path=_require_path(attempt.analysis_path, f"{phase_id} analysis"),
                    decision_path=_require_path(attempt.decision_path, f"{phase_id} decision"),
                    out_path=next_state_path,
                )
            current_state_path = next_state_path
            current_state = trialdev_upstream.load_program_state(current_state_path)
            if not attempt.advance:
                if current_state.terminal_disposition == "active":
                    raise ValueError("A non-advancing decision did not produce a terminal programme state.")
                run.stopped_at_phase = phase_id
                break
    except agent_mod.AgentTurnLimitExceeded as exc:
        run.execution_status = "model_turn_limit"
        run.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    except trialdev_upstream.TrialMaterializationRejectedError as exc:
        run.execution_status = "model_invalid_submission"
        run.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.info("Resumed program exhausted the materialization correction budget: %s", program.program_id)
    except TimeoutError:
        logger.exception("Resumed program run timed out: %s", program.program_id)
        interrupted = True
        raise
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError):
        logger.exception("Resumed program run failed: %s", program.program_id)
        interrupted = True
        raise
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        raise
    finally:
        if interrupted:
            if loop.session is not None:
                loop.session.close()
                loop.session = None
        else:
            if run.execution_status == "running":
                run.execution_status = "completed"
            run.started_at_utc = started_at_utc.isoformat()
            run.ended_at_utc = datetime.now(UTC).isoformat()
            run.wall_seconds_total = _time.monotonic() - started_monotonic
            _persist_chain_summary(program_dir, run, usage)
            _persist_conversation(program_dir, loop)
            loop.close()
    return run


# ---------------------------------------------------------------------------
# Per-checkpoint helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_PHASE_IDS = {
    "early_safety_study": "phase1",
    "proof_of_concept": "phase2",
    "confirmation": "phase3",
}


def _phase_id_for_state(state: TrialDevProgrammeStateV1) -> PhaseId:
    phase_id = _CHECKPOINT_PHASE_IDS.get(str(state.current_checkpoint_id))
    if phase_id is None:
        raise ValueError(f"Programme checkpoint is not a materializable phase: {state.current_checkpoint_id!r}.")
    return cast(PhaseId, phase_id)


def _require_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise RuntimeError(f"Missing required {label} path")
    return path


def _run_obs_review(
    *,
    program: Program,
    program_dir: Path,
    loop: agent_mod.AgentLoop,
    usage: MaterializationUsage,
    src_root: Path,
    checkpoint: Callable[[], None],
    start_step: bool = True,
) -> TrialDevelopmentSubmissionV1:
    """Drive the obs_review checkpoint and persist the synthesized submission.

    Returns the accepted, typed observational submission.
    """
    obs_dir = program_dir / "obs_review"
    obs_dir.mkdir(parents=True, exist_ok=True)
    if start_step:
        loop.begin_step(phase_id="observational_review", step_id="analysis_and_decision")
        loop.append_user_message(prompts.build_obs_review_block())
        checkpoint()
    while True:
        capture = loop.run_until_submit(
            tools=agent_mod.tools_for_obs_review(),
            submit_tool_names={
                "submit_obs_review_analysis_and_decision",
                "submit_obs_review_analysis_and_decision_file",
            },
        )
        payload = capture.payload

        # Persist the latest raw payload so interrupted runs retain the exact
        # model submission that reached validation.
        raw_path = obs_dir / "agent_obs_review_payload.json"
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        # Build a TrialDevelopmentSubmissionV1-shaped object for grade_item_v1.
        # The filename is intentionally NOT "submission.json" — that name is
        # globbed by grade_trajectory_v1 and would cause double-counting if
        # trajectory_root ever pointed here.
        submission_path = obs_dir / "obs_review_submission.json"
        try:
            submission = _build_obs_review_submission_for_grader(
                program=program,
                scenario_root=src_root,
                agent_payload=payload,
                out_path=submission_path,
            )
        except (ValidationError, ValueError) as exc:
            loop.append_tool_reply(
                capture.tool_call_id,
                f"Observational submission validation failed:\n{exc}\nCorrect the JSON file and resubmit it.",
                tool_name=capture.name,
                status="invalid",
            )
            continue
        break
    # Canonical schema-bearing summary for offline analysis.
    from trialagentbench_harness.contracts.core.runs import TrialDevObsReviewSummaryV1
    from trialagentbench_harness.io.json import write_json_model

    method_route_ids = {
        str(estimate.method_route_id)
        for estimate in submission.analysis_report.candidate_utility_estimates
        if estimate.method_route_id is not None
    }
    write_json_model(
        obs_dir / "obs_review_summary.json",
        TrialDevObsReviewSummaryV1(
            program_id=str(program.program_id),
            scenario_id=str(program.scenario_id),
            objective_id=str(program.objective_id),
            method_route_id=next(iter(method_route_ids)) if len(method_route_ids) == 1 else None,
            ranked_drug_ids=list(submission.analysis_report.ranked_drug_ids),
            recommended_drug_id=submission.program_decision.recommended_drug_id,
        ),
    )
    action = submission.program_decision.decision_action
    if action is None:
        raise ValueError("Validated observational submission is missing decision_action.")
    continuation_available = action == "nominate_for_early_study" and not _unsupported_observational_nomination(
        submission=submission,
        scenario_root=src_root,
    )
    loop.append_tool_reply(
        capture.tool_call_id,
        json.dumps(
            {
                "status": "obs_review recorded",
                "decision_action": action,
                "randomized_continuation_available": continuation_available,
            }
        ),
        tool_name=capture.name,
    )
    return submission


def _unsupported_observational_nomination(
    *,
    submission: TrialDevelopmentSubmissionV1,
    scenario_root: Path,
) -> bool:
    """Return whether a nomination falls outside the released continuation space."""

    decision = submission.program_decision
    if decision.decision_action != "nominate_for_early_study":
        return False
    candidate_id = decision.recommended_drug_id
    if candidate_id is None:
        raise ValueError("Validated nomination is missing recommended_drug_id.")
    return not _fixed_phase_replay_available(
        scenario_root=scenario_root,
        phase_id="phase1",
        candidate_drug_id=str(candidate_id),
    )


def _fixed_phase_replay_available(
    *,
    scenario_root: Path,
    phase_id: str,
    candidate_drug_id: str,
) -> bool:
    """Return whether the release contains evidence for one submitted path."""

    cases = _load_fixed_phase_replay_cases(Path(scenario_root).parent)
    return any(
        str(case.request.scenario_id) == str(Path(scenario_root).name.removeprefix("scenario_"))
        and str(case.request.phase_id) == str(phase_id)
        and tuple(str(value) for value in case.request.candidate_drug_ids) == (str(candidate_drug_id),)
        for case in cases
    )


def _load_fixed_phase_replay_cases(bundle_root: Path) -> tuple[TrialDevPhaseReplayCaseV1, ...]:
    """Load the immutable fixed-trajectory index from a participant release."""

    cases_path = Path(bundle_root) / "fixed_trajectories" / "cases.jsonl"
    if not cases_path.is_file():
        raise FileNotFoundError("TrialDev participant release is missing fixed_trajectories/cases.jsonl.")
    cases = tuple(
        TrialDevPhaseReplayCaseV1.model_validate_json(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not cases:
        raise ValueError("TrialDev participant release has an empty fixed-trajectory index.")
    return cases


def require_fixed_phase_replay_surface(bundle_root: Path) -> None:
    """Require every indexed randomized trajectory before provider execution."""

    root = Path(bundle_root)
    cases = _load_fixed_phase_replay_cases(root)
    materialized_root = root / "fixed_trajectories" / "materialized"
    for case in cases:
        request_root = materialized_root / f"world_{case.world_seed}" / f"request_{case.request.checksum()}"
        replicate_roots = tuple(path for path in request_root.glob("trial_seed_*") if path.is_dir())
        if len(replicate_roots) != 1:
            raise FileNotFoundError(
                "TrialDev participant release requires exactly one materialized replicate "
                f"for fixed request {case.request.checksum()}."
            )
        required = TRIALDEV_FIXED_TRAJECTORY_REPLICATE_MEMBERS
        observed = {path.name for path in replicate_roots[0].iterdir() if path.is_file()}
        missing = sorted(required - observed)
        if missing:
            raise FileNotFoundError(
                "TrialDev participant release has incomplete fixed randomized evidence "
                f"for request {case.request.checksum()}: {missing}."
            )


def _build_obs_review_submission_for_grader(
    *,
    program: Program,
    scenario_root: Path,
    agent_payload: dict[str, Any],
    out_path: Path,
) -> TrialDevelopmentSubmissionV1:
    """Wrap the agent's obs_review payload into a TrialDevelopmentSubmissionV1.

    The grader expects ``request + analysis_report + program_decision``.
    For obs_review there is no real trial design, so the request is a thin
    record with the catalog's full candidate list (asset hasn't been
    selected yet) and the agent's chosen objective.
    """
    candidate_ids = list(candidate_ids_by_role_v1(scenario_root=scenario_root)["investigational"])
    participant_submission = build_observational_review_v1(
        agent_payload,
        source_artifact_checksums=bridge.observational_source_artifact_checksums(
            scenario_root,
            objective_id=program.objective_id,
        ),
        identification_artifact_checksums=bridge.observational_identification_artifact_checksums(scenario_root),
    )
    agent_payload = participant_submission.model_dump(mode="json")

    submission: dict[str, object] = {
        "version": "v1",
        "scenario_id": program.scenario_id,
        "request": {
            "version": "v1",
            "scenario_id": program.scenario_id,
            "phase_id": "observational_review",
            "candidate_drug_ids": candidate_ids,
            "selection_objective": program.objective_id,
        },
        "analysis_report": {
            "response_branch": agent_payload.get("response_branch"),
            "primary_resolution_evidence_class": agent_payload.get("primary_resolution_evidence_class"),
            "ranked_drug_ids": agent_payload.get("ranked_drug_ids"),
            "selected_winner_drug_id": agent_payload.get("candidate_drug_id"),
            "candidate_utility_estimates": agent_payload.get("candidate_utility_estimates"),
            "identification_evidence": agent_payload.get("identification_evidence", []),
            "claimed_subgroup_variables": agent_payload.get("claimed_subgroup_variables", []),
            "diagnostic_artifacts": agent_payload.get("diagnostic_artifacts", []),
        },
        "program_decision": {
            "objective_id": program.objective_id,
            "decision_action": agent_payload.get("decision_action"),
            "recommended_drug_id": agent_payload.get("candidate_drug_id"),
            "supporting_evidence_ids": agent_payload.get("supporting_evidence_ids"),
        },
    }
    return trialdev_upstream.validate_and_write_submission(submission, path=out_path)


def _run_one_phase(
    *,
    phase_id: PhaseId,
    public_dir: Path,
    phase_dir: Path,
    state_path: Path,
    program: Program,
    loop: agent_mod.AgentLoop,
    usage: MaterializationUsage,
    options: RunOptions,
    src_root: Path,
    prior_phase_summaries: list[dict[str, Any]],
    violations: list[dict[str, Any]] | None = None,
    action_policy: TrialDevelopmentPhaseActionPolicyV1,
    action_spec: TrialDevelopmentPhaseActionSpecV1,
    checkpoint: Callable[
        [
            Literal[
                "phase_request",
                "materialize",
                "phase_analysis",
                "phase_decision",
                "advance_state",
            ],
            PhaseAttempt,
        ],
        None,
    ],
    resume_operation: Literal[
        "phase_request",
        "materialize",
        "phase_analysis",
        "phase_decision",
        "advance_state",
    ] = "phase_request",
    restored_attempt: PhaseAttempt | None = None,
) -> PhaseAttempt:
    """Run one materializing phase: request -> trial output -> analysis -> decision."""
    attempt = restored_attempt or PhaseAttempt(phase_id=phase_id)

    state_summary = bridge.summarize_program_state_for_agent(state_path)
    phase_module = prompts.get_phase_module(public_dir, phase_id)

    def drive_request() -> tuple[TrialDevelopmentRequestV1, Path]:
        return _drive_phase_request(
            phase_id=phase_id,
            phase_dir=phase_dir,
            loop=loop,
            usage=usage,
            state_summary=state_summary,
            phase_module=phase_module,
            prior_phase_summaries=prior_phase_summaries,
            violations=violations,
            program_id=program.program_id,
            scenario_id=program.scenario_id,
            program_objective=program.objective_id,
            start_step=attempt.request_path is None,
            step_checkpoint=lambda: checkpoint("phase_request", attempt),
        )

    # ── Request ─────────────────────────────────────────────────────────
    if resume_operation == "phase_request":
        request, request_path = drive_request()
        attempt.request_path = request_path
        checkpoint("materialize", attempt)
    else:
        request_path = _require_path(attempt.request_path, f"{phase_id} request")
        request = TrialDevelopmentRequestV1.model_validate_json(request_path.read_text(encoding="utf-8"))

    # ── Materialize (with retry on contract rejections) ─────────────────
    seed = bridge.derive_phase_seed(options.master_seed, program.program_id, phase_id)
    trial_output_root = phase_dir / "trial_output"
    max_retries = options.max_phase_retries
    if resume_operation in {"phase_request", "materialize"}:
        if trial_output_root.exists() and not (trial_output_root / "trial_output_manifest.json").is_file():
            _archive_incomplete_materialization(
                trial_output_root=trial_output_root,
                program_dir=phase_dir.parent.parent,
                phase_id=phase_id,
            )
        if not trial_output_root.exists():
            for retry_idx in range(max_retries + 1):
                try:
                    trialdev_upstream.materialize_phase(
                        scenario_root=src_root,
                        state_path=state_path,
                        request_path=request_path,
                        out_dir=trial_output_root,
                        seed=seed,
                        overwrite=False,
                    )
                    break
                except trialdev_upstream.TrialMaterializationRejectedError as exc:
                    archived_output = _archive_incomplete_materialization(
                        trial_output_root=trial_output_root,
                        program_dir=phase_dir.parent.parent,
                        phase_id=phase_id,
                    )
                    if violations is not None:
                        violation = {
                            "phase_id": phase_id,
                            "kind": "materialize_rejection",
                            "error": str(exc),
                        }
                        if archived_output is not None:
                            violation["artifact_relative_path"] = archived_output.relative_to(
                                phase_dir.parent.parent
                            ).as_posix()
                        violations.append(violation)
                    if retry_idx == max_retries:
                        raise
                    loop.append_user_message(
                        f"MATERIALIZATION REJECTED at {phase_id}: {exc.reason}\n"
                        "Consult the mounted phase contract, adjust the request, and submit again."
                    )
                    request, request_path = drive_request()
                    attempt.request_path = request_path
        usage.record(phase_id)
        attempt.trial_output_root = trial_output_root
    else:
        trial_output_root = _require_path(
            attempt.trial_output_root,
            f"{phase_id} trial output",
        )

    output_summary = bridge.summarize_trial_output_for_agent(trial_output_root, relative_to=phase_dir.parent)
    if not attempt.materializations:
        attempt.materializations.append(
            MaterializationRecord(
                phase_id=phase_id,
                seed=seed,
                request_path=request_path,
                request_checksum=output_summary["request_checksum"],
                trial_output_root=trial_output_root,
                trial_output_checksum=output_summary["trial_output_checksum"],
            )
        )
    else:
        materialization = attempt.materializations[-1]
        expected_materialization = (
            phase_id,
            seed,
            request_path.resolve(),
            output_summary["request_checksum"],
            trial_output_root.resolve(),
            output_summary["trial_output_checksum"],
        )
        restored_materialization = (
            materialization.phase_id,
            materialization.seed,
            materialization.request_path.resolve(),
            materialization.request_checksum,
            materialization.trial_output_root.resolve(),
            materialization.trial_output_checksum,
        )
        if restored_materialization != expected_materialization:
            raise ValueError("TrialDev restored materialization custody does not match its artifacts.")
    # ── Analysis ────────────────────────────────────────────────────────
    if resume_operation in {"phase_request", "materialize", "phase_analysis"}:
        analysis_path = _drive_phase_analysis(
            phase_id=phase_id,
            phase_dir=phase_dir,
            loop=loop,
            trial_output_summary=output_summary,
            program_id=str(program.program_id),
            scenario_id=str(program.scenario_id),
            program_objective=str(program.objective_id),
            start_step=resume_operation != "phase_analysis",
            step_checkpoint=lambda: checkpoint("phase_analysis", attempt),
        )
        attempt.analysis_path = analysis_path
    else:
        analysis_path = _require_path(attempt.analysis_path, f"{phase_id} analysis")

    # ── Decision ────────────────────────────────────────────────────────
    request_checksum = output_summary["request_checksum"]
    from trialagentbench_harness.adapters.trialdev_share import sha256_file_hex

    analysis_checksum = sha256_file_hex(analysis_path)
    if resume_operation != "advance_state":
        decision_path, decision_action, advance, candidate_drug_id = _drive_phase_decision(
            phase_id=phase_id,
            phase_dir=phase_dir,
            loop=loop,
            request_checksum=request_checksum,
            analysis_checksum=analysis_checksum,
            program_id=str(program.program_id),
            scenario_id=str(program.scenario_id),
            program_objective=str(program.objective_id),
            action_policy=action_policy,
            action_spec=action_spec,
            start_step=resume_operation != "phase_decision",
            step_checkpoint=lambda: checkpoint("phase_decision", attempt),
        )
        attempt.decision_path = decision_path
        attempt.decision_action = decision_action
        attempt.advance = advance
        attempt.candidate_drug_id = candidate_drug_id

    # Match the decision's chosen endpoint against an item if the program had
    # multiple endpoints at this phase — purely informational, recorded so the
    # aggregate can show which item this phase's submission scored against.
    attempt.matched_item_id = _match_item(program, phase_id, request.endpoint_id)
    checkpoint("advance_state", attempt)

    return attempt


def _archive_incomplete_materialization(
    *,
    trial_output_root: Path,
    program_dir: Path,
    phase_id: PhaseId,
) -> Path | None:
    """Move one incomplete materialization out of the canonical output path."""

    if not trial_output_root.exists():
        return None
    if not trial_output_root.is_dir() or trial_output_root.is_symlink():
        raise ValueError(f"Incomplete TrialDev materialization is not a regular directory: {trial_output_root}")
    attempt_root = program_dir / "materialization_attempts" / phase_id
    attempt_index = 1
    while (attempt_root / f"attempt-{attempt_index}").exists():
        attempt_index += 1
    archive = Path(attempt_root / f"attempt-{attempt_index}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    trial_output_root.rename(archive)
    return archive


def _drive_phase_request(
    *,
    phase_id: PhaseId,
    phase_dir: Path,
    loop: agent_mod.AgentLoop,
    usage: MaterializationUsage,
    state_summary: dict[str, Any],
    phase_module: dict[str, Any],
    prior_phase_summaries: list[dict[str, Any]],
    violations: list[dict[str, Any]] | None = None,
    program_id: str = "(unknown)",
    scenario_id: str = "(unknown)",
    program_objective: str = "(unknown)",
    start_step: bool,
    step_checkpoint: Callable[[], None] = lambda: None,
) -> tuple[TrialDevelopmentRequestV1, Path]:
    """Prompt the agent for a phase request, validate, persist, return (model, path).

    ``violations`` is a per-program list we append to whenever the agent
    submits a request that fails pydantic validation — useful as a
    diagnostic for "model confused about phase-legal fields".
    """
    if start_step:
        loop.begin_step(phase_id=phase_id, step_id="trial_design_request")
        loop.append_user_message(
            prompts.build_phase_request_block(
                phase_id=phase_id,
                state_summary=state_summary,
                phase_module=phase_module,
                prior_phase_summaries=prior_phase_summaries,
                program_id=program_id,
                program_objective=program_objective,
            )
        )
        step_checkpoint()
    tools = agent_mod.tools_for_phase_request(phase_module=phase_module)
    while True:
        capture = loop.run_until_submit(
            tools=tools,
            submit_tool_names={"submit_phase_request", "submit_phase_request_file"},
        )
        request, err = bridge.parse_request(
            capture.payload,
            scenario_id=scenario_id,
            phase_id=str(phase_id),
        )
        if err:
            if violations is not None:
                violations.append(
                    {
                        "phase_id": phase_id,
                        "kind": "schema_validation",
                        "error": err,
                    }
                )
            loop.append_tool_reply(
                capture.tool_call_id,
                err + "\nResubmit a corrected request.",
                tool_name=capture.name,
            )
            continue
        if request is None:
            raise RuntimeError("Request validation returned neither request nor error")
        request_path = phase_dir / "request.json"
        bridge.write_request(request, request_path)
        bridge.write_rationale_sidecar(phase_dir=phase_dir, request_payload=capture.payload)
        if any(
            value is None
            for value in (
                request.target_sample_size,
                request.follow_up_days,
                request.site_count_budget,
                request.enrollment_window_days,
            )
        ):
            raise ValueError(f"Randomized phase request lacks its complete resource vector: phase_id={phase_id!r}.")
        assert request.target_sample_size is not None
        assert request.follow_up_days is not None
        assert request.site_count_budget is not None
        assert request.enrollment_window_days is not None
        # Schema-bearing sidecar enabling strict offline analysis without
        # modifying upstream artifacts consumed by the state machine/grader.
        from trialagentbench_harness.contracts.core.runs import (
            TrialDevPhaseRequestSummaryV1,
            TrialDevPhaseStepSummaryV1,
        )
        from trialagentbench_harness.io.json import write_json_model

        write_json_model(
            phase_dir / "phase_step_summary.json",
            TrialDevPhaseStepSummaryV1(
                program_id=str(program_id),
                scenario_id=str(scenario_id),
                objective_id=str(program_objective),
                phase_id=str(phase_id),
                request=TrialDevPhaseRequestSummaryV1(
                    phase_id=str(phase_id),
                    endpoint_id=(str(request.endpoint_id) if getattr(request, "endpoint_id", None) else None),
                    selection_objective=(
                        str(request.selection_objective) if getattr(request, "selection_objective", None) else None
                    ),
                    target_sample_size=int(request.target_sample_size),
                    follow_up_days=int(request.follow_up_days),
                    allocation_ratio=str(request.allocation_ratio),
                    site_count_budget=int(request.site_count_budget),
                    enrollment_window_days=int(request.enrollment_window_days),
                ),
            ),
        )
        loop.append_tool_reply(
            capture.tool_call_id,
            json.dumps({"status": "request accepted; study records are being prepared"}),
            tool_name=capture.name,
        )
        return request, request_path


def _drive_phase_analysis(
    *,
    phase_id: PhaseId,
    phase_dir: Path,
    loop: agent_mod.AgentLoop,
    trial_output_summary: dict[str, Any],
    program_id: str,
    scenario_id: str,
    program_objective: str,
    start_step: bool,
    step_checkpoint: Callable[[], None],
) -> Path:
    if start_step:
        loop.begin_step(phase_id=phase_id, step_id="trial_analysis")
        loop.append_user_message(
            prompts.build_phase_analysis_block(
                phase_id=phase_id,
                trial_output_summary=trial_output_summary,
            )
        )
        step_checkpoint()
    while True:
        capture = loop.run_until_submit(
            tools=agent_mod.tools_for_phase_analysis(),
            submit_tool_names={"submit_phase_analysis", "submit_phase_analysis_file"},
        )
        analysis, err = bridge.parse_phase_analysis(
            capture.payload,
            scenario_id=scenario_id,
            phase_id=str(phase_id),
            request_checksum=str(trial_output_summary["request_checksum"]),
            trial_output_checksum=str(trial_output_summary["trial_output_checksum"]),
            effect_source_artifact_checksums=cast(
                dict[str, str], trial_output_summary["effect_source_artifact_checksums"]
            ),
            safety_source_artifact_checksums=cast(
                dict[str, str], trial_output_summary["safety_source_artifact_checksums"]
            ),
        )
        if err:
            loop.append_tool_reply(
                capture.tool_call_id,
                err + "\nResubmit a corrected analysis.",
                tool_name=capture.name,
            )
            continue
        if analysis is None:
            raise RuntimeError("Analysis validation returned neither analysis nor error")
        analysis_path = phase_dir / "analysis_submission.json"
        bridge.write_phase_submission(analysis, analysis_path)
        bridge.write_rationale_sidecar(phase_dir=phase_dir, analysis_payload=capture.payload)
        from trialagentbench_harness.contracts.core.runs import (
            TrialDevPhaseAnalysisSummaryV1,
            TrialDevPhaseStepSummaryV1,
        )
        from trialagentbench_harness.io.json import read_json_model, write_json_model

        summary_path = phase_dir / "phase_step_summary.json"
        try:
            summary = read_json_model(TrialDevPhaseStepSummaryV1, summary_path)
        except FileNotFoundError:
            summary = TrialDevPhaseStepSummaryV1(
                program_id=str(program_id),
                scenario_id=str(scenario_id),
                objective_id=str(program_objective),
                phase_id=str(phase_id),
            )
        summary.analysis = TrialDevPhaseAnalysisSummaryV1(
            phase_id=str(phase_id),
            ranked_drug_ids=[str(x) for x in (getattr(analysis, "ranked_drug_ids", None) or [])],
            selected_winner_drug_id=(
                str(analysis.selected_winner_drug_id) if getattr(analysis, "selected_winner_drug_id", None) else None
            ),
        )
        write_json_model(summary_path, summary)
        loop.append_tool_reply(
            capture.tool_call_id,
            "Analysis submission accepted.",
            tool_name=capture.name,
        )
        return analysis_path


def _drive_phase_decision(
    *,
    phase_id: PhaseId,
    phase_dir: Path,
    loop: agent_mod.AgentLoop,
    request_checksum: str,
    analysis_checksum: str,
    program_id: str,
    scenario_id: str,
    program_objective: str,
    action_policy: TrialDevelopmentPhaseActionPolicyV1,
    action_spec: TrialDevelopmentPhaseActionSpecV1,
    start_step: bool,
    step_checkpoint: Callable[[], None],
) -> tuple[Path, str, bool, str | None]:
    """Run the decision step.

    Returns ``(decision_path, decision_action, advance, candidate_drug_id)``.
    ``advance`` is True iff ``decision_action`` is a *non-terminal* action
    in this scenario's ``phase_action_policy`` (i.e. continue to the next
    phase). ``action_spec`` constrains the agent's tool enum to that phase's
    legal actions; ``action_policy`` lets the bridge re-check post-validation.
    """
    if start_step:
        loop.begin_step(phase_id=phase_id, step_id="phase_decision")
        loop.append_user_message(
            prompts.build_phase_decision_block(
                phase_id=phase_id,
                action_spec=action_spec,
            )
        )
        step_checkpoint()
    terminal_action_ids = set(action_spec.terminal_action_ids)
    while True:
        capture = loop.run_until_submit(
            tools=agent_mod.tools_for_phase_decision(action_spec),
            submit_tool_names={"submit_phase_decision", "submit_phase_decision_file"},
        )
        decision, err = bridge.parse_phase_decision(
            capture.payload,
            action_policy=action_policy,
            scenario_id=scenario_id,
            phase_id=str(phase_id),
            request_checksum=str(request_checksum),
            analysis_checksum=str(analysis_checksum),
        )
        if err:
            loop.append_tool_reply(
                capture.tool_call_id,
                err + "\nResubmit a corrected decision.",
                tool_name=capture.name,
            )
            continue
        if decision is None:
            raise RuntimeError("Decision validation returned neither decision nor error")
        decision_path = phase_dir / "decision_submission.json"
        bridge.write_phase_submission(decision, decision_path)
        from trialagentbench_harness.contracts.core.runs import (
            TrialDevPhaseDecisionSummaryV1,
            TrialDevPhaseStepSummaryV1,
        )
        from trialagentbench_harness.io.json import read_json_model, write_json_model

        summary_path = phase_dir / "phase_step_summary.json"
        try:
            summary = read_json_model(TrialDevPhaseStepSummaryV1, summary_path)
        except FileNotFoundError:
            summary = TrialDevPhaseStepSummaryV1(
                program_id=str(program_id),
                scenario_id=str(scenario_id),
                objective_id=str(program_objective),
                phase_id=str(phase_id),
            )
        summary.decision = TrialDevPhaseDecisionSummaryV1(
            phase_id=str(phase_id),
            decision_action=(str(decision.decision_action) if getattr(decision, "decision_action", None) else None),
            candidate_drug_id=(
                str(decision.candidate_drug_id) if getattr(decision, "candidate_drug_id", None) else None
            ),
        )
        write_json_model(summary_path, summary)
        action = str(decision.decision_action)
        advance = action not in terminal_action_ids if terminal_action_ids else False
        loop.append_tool_reply(
            capture.tool_call_id,
            json.dumps({"status": "decision accepted", "decision_action": action}),
            tool_name=capture.name,
        )
        return decision_path, action, advance, decision.candidate_drug_id


# ---------------------------------------------------------------------------
# Bookkeeping helpers
# ---------------------------------------------------------------------------


def _summarize_phase_attempt(attempt: PhaseAttempt) -> dict[str, Any]:
    """Compact summary surfaced to the agent in the next phase's prompt."""
    summary: dict[str, Any] = {
        "phase_id": attempt.phase_id,
        "decision_action": attempt.decision_action,
        "advance": attempt.advance,
        "candidate_drug_id": attempt.candidate_drug_id,
        "matched_item_id": attempt.matched_item_id,
    }
    if attempt.analysis_path and attempt.analysis_path.is_file():
        try:
            payload = json.loads(attempt.analysis_path.read_text(encoding="utf-8"))
            summary.update(
                {
                    "primary_effect": payload.get("primary_effect"),
                    "safety_estimate": payload.get("safety_estimate"),
                }
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid phase analysis payload JSON: {attempt.analysis_path}") from exc
    return summary


def _match_item(program: Program, phase_id: PhaseId, endpoint_id: str | None) -> str | None:
    items = program.items_by_phase.get(phase_id, ())
    if not items:
        return None
    for item in items:
        if item.endpoint_id == endpoint_id:
            return str(item.item_id)
    # No exact match — return the first item id for traceability
    return str(items[0].item_id)


def _stepwise_submission_payload(phase_workdir: Path) -> dict[str, Any] | None:
    """Build a TrialDevelopmentSubmissionV1 dict from saved phase JSON files.

    Mirrors ``trialagentbench_harness.trialdev.grading.sequential._full_submission_from_stepwise``.
    Returns None if any of the three sub-submissions is missing.

    NB: we deliberately do *not* write a ``submission.json`` next to the
    sub-submissions — ``grade_trajectory_v1`` uses ``rglob('submission.json')``
    and would pick it up, double-grading the same phase.
    """
    request_path = phase_workdir / "request.json"
    analysis_path = phase_workdir / "analysis_submission.json"
    decision_path = phase_workdir / "decision_submission.json"
    if not (request_path.is_file() and analysis_path.is_file() and decision_path.is_file()):
        return None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(request, dict) or not isinstance(analysis, dict) or not isinstance(decision, dict):
        return None

    return {
        "version": "v1",
        "scenario_id": str(request.get("scenario_id")),
        "request": request,
        "analysis_report": {
            "selected_winner_drug_id": analysis.get("selected_winner_drug_id"),
            "ranked_drug_ids": list(analysis.get("ranked_drug_ids", []) or []),
            "candidate_utility_estimates": list(analysis.get("candidate_utility_estimates", []) or []),
            "primary_effect": analysis.get("primary_effect"),
            "safety_estimate": analysis.get("safety_estimate"),
            "claimed_subgroup_variables": list(analysis.get("claimed_subgroup_variables", []) or []),
            "diagnostic_artifacts": list(analysis.get("diagnostic_artifacts", []) or []),
            "evidence_summary": analysis.get("evidence_summary", ""),
        },
        "program_decision": {
            "objective_id": request.get("selection_objective"),
            "decision_action": decision.get("decision_action"),
            "recommended_drug_id": decision.get("candidate_drug_id"),
            "supporting_evidence_ids": list(decision.get("supporting_evidence_ids", []) or []),
        },
    }


def _phase_path_stats_from_program_dir(program_dir: Path) -> dict[str, dict[str, int]]:
    """Count turns and tool calls per phase from the active conversation.

    Phases are delimited by user messages whose content starts with
    "PHASE: <phase_id>" (set by our prompt builders). We attribute each
    subsequent assistant message + its tool calls to the active phase.
    """
    out: dict[str, dict[str, int]] = {}
    checkpoint_path = program_dir / "checkpoint_conversation.json"
    conv_path = checkpoint_path if checkpoint_path.is_file() else program_dir / "conversation.json"
    if not conv_path.is_file():
        return out
    try:
        msgs = json.loads(conv_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return out
    current: str | None = None
    PHASE_TAGS = {
        "PHASE: observational_review": "observational_review",
        "PHASE: phase1": "phase1",
        "PHASE: phase2": "phase2",
        "PHASE: phase3": "phase3",
        "FINAL PROGRAM": "final_program",
    }
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(p.get("text") or p) if isinstance(p, dict) else str(p) for p in content)
        content_s = str(content)
        if role == "user":
            for tag, phase_id in PHASE_TAGS.items():
                if tag in content_s:
                    current = phase_id
                    out.setdefault(current, {"turns": 0, "execute_code": 0, "inspect_parquet": 0})
                    break
        elif role == "assistant" and current:
            out[current]["turns"] += 1
            for tc in m.get("tool_calls") or []:
                name = (tc.get("function") or {}).get("name", "")
                if name == "execute_code":
                    out[current]["execute_code"] += 1
                elif name == "inspect_parquet":
                    out[current]["inspect_parquet"] += 1
    return out


def _persist_chain_summary(program_dir: Path, run: ProgramRun, usage: MaterializationUsage) -> None:
    from trialagentbench_harness.contracts.core.runs import (
        TrialDevChainSummaryV1,
        TrialDevMaterializationUsageV1,
        TrialDevPhaseAttemptSummaryV1,
        TrialDevTrajectoryMetricsV1,
    )
    from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
        TrialDevGradeRecordV1,
        TrialDevTrajectoryGradeV1,
    )
    from trialagentbench_harness.io import read_json_model, write_json_model
    from trialagentbench_harness.trialdev.grade_wrappers import (
        phase_policy_modes_from_manifest,
        summarise_programme_analysis_quality,
        trajectory_metrics_from_grade,
    )

    trajectory_metrics = TrialDevTrajectoryMetricsV1()
    trajectory_grade = None
    if run.trajectory_grade_path and Path(run.trajectory_grade_path).is_file():
        trajectory_grade = read_json_model(TrialDevTrajectoryGradeV1, Path(run.trajectory_grade_path))
    observational_grade = (
        read_json_model(TrialDevGradeRecordV1, Path(run.obs_review_grade_path))
        if run.obs_review_grade_path and Path(run.obs_review_grade_path).is_file()
        else None
    )
    attempted_phase_ids = {str(phase.phase_id) for phase in run.phases}
    trajectory_metrics.analysis_quality = summarise_programme_analysis_quality(
        observational_report=observational_grade,
        phase_reports=() if trajectory_grade is None else trajectory_grade.phase_reports,
        attempted_phase_ids=attempted_phase_ids,
    )
    if trajectory_grade is not None:
        trajectory_metrics = trajectory_metrics_from_grade(
            trajectory_grade=trajectory_grade,
            observational_report=observational_grade,
            phase_policy_modes=phase_policy_modes_from_manifest(Path(run.workdir) / "program_loop_manifest.json"),
            analysis_quality=trajectory_metrics.analysis_quality,
        )
    # Per-phase turn / tool-call counts (computed from saved conversation if present).
    phase_path_stats = _phase_path_stats_from_program_dir(program_dir)

    phases_attempted = [
        TrialDevPhaseAttemptSummaryV1(
            phase_id=p.phase_id,
            matched_item_id=p.matched_item_id,
            decision_action=p.decision_action,
            advance=p.advance,
            candidate_drug_id=p.candidate_drug_id,
            n_materializations=len(p.materializations),
            turns=int(phase_path_stats.get(p.phase_id, {}).get("turns", 0)),
            execute_code_calls=int(phase_path_stats.get(p.phase_id, {}).get("execute_code", 0)),
            inspect_parquet_calls=int(phase_path_stats.get(p.phase_id, {}).get("inspect_parquet", 0)),
        )
        for p in run.phases
    ]
    materialization_usage = TrialDevMaterializationUsageV1(
        materialize_calls_by_phase=dict(usage.materialize_calls_by_phase),
    )
    summary = TrialDevChainSummaryV1(
        program_id=run.program_id,
        scenario_id=run.scenario_id,
        objective_id=run.objective_id,
        stopped_at_phase=run.stopped_at_phase,
        started_at_utc=run.started_at_utc,
        ended_at_utc=run.ended_at_utc,
        wall_seconds_total=run.wall_seconds_total,
        phases_attempted=phases_attempted,
        obs_review_path_stats=phase_path_stats.get("observational_review", {}) or {},
        materialization_usage=materialization_usage,
        obs_review_grade_path=(str(run.obs_review_grade_path) if run.obs_review_grade_path else None),
        trajectory_grade_path=(str(run.trajectory_grade_path) if run.trajectory_grade_path else None),
        trajectory_metrics=trajectory_metrics,
        execution_status=run.execution_status,
        error=run.error,
        violations_n=len(run.violations or []),
        violations=list(run.violations or []),
    )
    write_json_model(program_dir / "chain_summary.json", summary)


def _persist_conversation(program_dir: Path, loop: agent_mod.AgentLoop) -> None:
    from trialagentbench_harness.io import write_json

    out = program_dir / "conversation.json"
    write_json(out, loop.messages)
    transcript.write_transcript_md(loop.messages, program_dir / "transcript.md")


__all__ = ["RunOptions", "run_program"]
