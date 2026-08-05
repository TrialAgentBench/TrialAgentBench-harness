"""Run frozen matched TrialDev checkpoint continuations."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from trialagentbench_harness.adapters import ProviderRouting, get_provider
from trialagentbench_harness.contracts.experiments import (
    TrialDevCheckpointAssignmentV1,
    TrialDevCheckpointRunConfigV1,
    TrialDevCheckpointScheduleV1,
)
from trialagentbench_harness.contracts.trialdev.run_checkpoint import TrialDevRunCheckpointV1
from trialagentbench_harness.execution_policy import TRIALDEV_RELEASE_BUDGET_V1
from trialagentbench_harness.io import (
    read_json_model,
    sha256_dir_digest,
    sha256_file,
    write_json,
    write_json_model,
)
from trialagentbench_harness.trialdev.agent import DEFAULT_MAX_TURNS_PER_STEP
from trialagentbench_harness.trialdev.data import discover_programs
from trialagentbench_harness.trialdev.runner import RunOptions, resume_program
from trialagentbench_harness.util.provider_environment import load_provider_dotenv


def _same_run(left: TrialDevCheckpointRunConfigV1, right: TrialDevCheckpointRunConfigV1) -> bool:
    return left.model_dump(exclude={"timestamp_utc"}) == right.model_dump(exclude={"timestamp_utc"})


def _safe_source_path(root: Path, relative_path: str) -> Path:
    source = (root / relative_path).resolve()
    try:
        source.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Checkpoint source path escapes its root: {relative_path!r}.") from exc
    return source


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ValueError(f"Checkpoint source custody cannot be a symbolic link: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Checkpoint source custody contains a symbolic link: {path}")


def _validate_source_assignment(
    assignment: TrialDevCheckpointAssignmentV1,
    *,
    checkpoint_root: Path,
) -> tuple[Path, TrialDevRunCheckpointV1]:
    source_program = _safe_source_path(checkpoint_root, assignment.source_program_relative_path)
    if not source_program.is_dir():
        raise NotADirectoryError(source_program)
    _reject_symlinks(source_program)
    if (source_program / "chain_summary.json").exists():
        raise ValueError("Checkpoint replay requires partial programme custody, not a completed run.")
    source_checkpoint = _safe_source_path(checkpoint_root, assignment.source_checkpoint_relative_path)
    try:
        source_checkpoint.relative_to(source_program)
    except ValueError as exc:
        raise ValueError("Checkpoint file does not belong to its declared source programme.") from exc
    if sha256_file(source_checkpoint) != assignment.source_checkpoint_sha256:
        raise ValueError("Checkpoint source file checksum mismatch.")
    envelope = read_json_model(TrialDevRunCheckpointV1, source_checkpoint)
    payload = envelope.payload
    continuation = payload.continuation.payload
    if payload.run_identity_sha256 != assignment.source_run_identity_sha256:
        raise ValueError("Checkpoint source run identity mismatch.")
    if (
        continuation.program_id,
        continuation.scenario_id,
        continuation.objective_id,
    ) != (
        assignment.program_id,
        assignment.scenario_id,
        assignment.objective_id,
    ):
        raise ValueError("Checkpoint source programme identity mismatch.")
    if (
        continuation.pending_step.phase_id,
        continuation.pending_step.step_id,
    ) != (
        assignment.checkpoint_phase_id,
        assignment.checkpoint_step_id,
    ):
        raise ValueError("Checkpoint source semantic boundary mismatch.")
    if continuation.pending_step.turns_used != 0:
        raise ValueError("Checkpoint experiment sources must be captured before an assistant response.")
    checkpoint_files = sorted((source_program / "checkpoints").glob("*.json"))
    if not checkpoint_files or source_checkpoint != checkpoint_files[-1].resolve():
        raise ValueError("Checkpoint source must be the latest append-only programme checkpoint.")
    return source_program, envelope


def _stage_assignment(
    assignment: TrialDevCheckpointAssignmentV1,
    *,
    checkpoint_root: Path,
    assignment_root: Path,
) -> TrialDevRunCheckpointV1:
    source_program, checkpoint = _validate_source_assignment(
        assignment,
        checkpoint_root=checkpoint_root,
    )
    target_program = assignment_root / "programs" / assignment.program_id
    target_program.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_program, target_program)
    return checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    """Execute matched checkpoint assignments in frozen schedule order."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-dir", required=True)
    parser.add_argument("--checkpoint-source-dir", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True, choices=("openai", "openai_responses", "openrouter"))
    parser.add_argument("--openrouter-provider")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dotenv", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn,
    )
    parser.add_argument("--max-turns-per-step", type=int, default=DEFAULT_MAX_TURNS_PER_STEP)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=TRIALDEV_RELEASE_BUDGET_V1.provider_request_timeout_seconds,
    )
    parser.add_argument(
        "--program-watchdog-seconds",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.wall_time_limit_seconds,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.dotenv:
        load_provider_dotenv()

    participant_root = Path(args.participant_dir).resolve()
    checkpoint_root = Path(args.checkpoint_source_dir).resolve()
    schedule = read_json_model(TrialDevCheckpointScheduleV1, Path(args.schedule))
    participant_hash = sha256_dir_digest(participant_root)
    checkpoint_hash = sha256_dir_digest(checkpoint_root)
    if participant_hash != schedule.participant_release_sha256:
        raise ValueError("Participant release hash does not match the frozen checkpoint schedule.")
    if checkpoint_hash != schedule.checkpoint_source_sha256:
        raise ValueError("Checkpoint source hash does not match the frozen checkpoint schedule.")
    proposed = TrialDevCheckpointRunConfigV1(
        timestamp_utc=datetime.now(UTC),
        schedule_checksum=str(schedule.checksum),
        participant_release_sha256=participant_hash,
        checkpoint_source_sha256=checkpoint_hash,
        model=str(args.model),
        provider=args.provider,
        openrouter_provider=args.openrouter_provider,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        max_turns_per_step=int(args.max_turns_per_step),
        request_timeout_seconds=float(args.request_timeout_seconds),
        program_watchdog_seconds=int(args.program_watchdog_seconds),
    )
    output = Path(args.output_dir).resolve()
    if args.resume:
        config = read_json_model(TrialDevCheckpointRunConfigV1, output / "run_config.json")
        if not _same_run(config, proposed):
            raise ValueError("Resume arguments differ from the frozen checkpoint experiment run.")
        if read_json_model(TrialDevCheckpointScheduleV1, output / "schedule.json") != schedule:
            raise ValueError("Resume schedule differs from the persisted checkpoint schedule.")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json_model(output / "schedule.json", schedule)
        write_json_model(output / "run_config.json", proposed)
        config = proposed

    programs = {program.program_id: program for program in discover_programs(participant_root)}
    routing = ProviderRouting(
        provider=config.provider,
        openrouter_provider=config.openrouter_provider,
    )
    for index, assignment in enumerate(schedule.assignments, start=1):
        program = programs.get(assignment.program_id)
        if program is None:
            raise ValueError(f"Checkpoint schedule references unknown program {assignment.program_id!r}.")
        assignment_root = output / "assignments" / assignment.assignment_id
        program_dir = assignment_root / "programs" / assignment.program_id
        if (program_dir / "chain_summary.json").is_file():
            if not args.resume:
                raise FileExistsError(program_dir)
            continue
        if not program_dir.exists():
            checkpoint = _stage_assignment(
                assignment,
                checkpoint_root=checkpoint_root,
                assignment_root=assignment_root,
            )
        else:
            if not args.resume:
                raise FileExistsError(program_dir)
            _, checkpoint = _validate_source_assignment(
                assignment,
                checkpoint_root=checkpoint_root,
            )
        source_provider = checkpoint.payload.continuation.payload
        provider = get_provider(
            config.model,
            routing=routing,
            send_temperature=True,
            decoding_seed=assignment.decoding_seed,
            timeout_s=config.request_timeout_seconds,
        )
        if assignment.condition != "canonical_state" and (
            provider.model,
            provider.telemetry_route,
        ) != (
            source_provider.provider_model,
            source_provider.provider_route,
        ):
            raise ValueError("Checkpoint source provider identity does not match the scheduled live route.")
        options = RunOptions(
            bundle_root=participant_root,
            output_root=assignment_root,
            model=config.model,
            procedure_assistance="output_contract_only",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_timeout_seconds=config.request_timeout_seconds,
            max_turns_per_step=config.max_turns_per_step,
            program_watchdog_seconds=config.program_watchdog_seconds,
            run_identity_sha256=assignment.source_run_identity_sha256,
            checkpoint_context_mode=("exact" if assignment.condition == "endogenous" else "active_step_only"),
            continuation_budget_scope="checkpoint_local",
        )
        run = resume_program(program, options=options, provider=provider)
        if run.execution_status not in {
            "completed",
            "model_turn_limit",
            "model_invalid_submission",
        }:
            raise RuntimeError(f"Checkpoint assignment {assignment.assignment_id} did not terminate.")
        write_json(
            assignment_root / "assignment_receipt.json",
            {
                "assignment_id": assignment.assignment_id,
                "condition": assignment.condition,
                "source_checkpoint_sha256": assignment.source_checkpoint_sha256,
                "source_run_identity_sha256": assignment.source_run_identity_sha256,
                "canonical_reference_id": assignment.canonical_reference_id,
                "execution_status": run.execution_status,
            },
        )
        print(f"[{index}/{len(schedule.assignments)}] {assignment.assignment_id}: {run.execution_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
