"""Run a frozen TrialDev observational selection-versus-execution experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from trialagentbench_harness.adapters import ProviderRouting, get_provider
from trialagentbench_harness.contracts.experiments import (
    TrialDevObservationalSpecificationRunConfigV1,
    TrialDevObservationalSpecificationScheduleV1,
)
from trialagentbench_harness.execution_policy import TRIALDEV_RELEASE_BUDGET_V1
from trialagentbench_harness.io import (
    canonical_payload_sha256,
    read_json_model,
    sha256_dir_digest,
    write_json_model,
)
from trialagentbench_harness.trialdev.agent import DEFAULT_MAX_TURNS_PER_STEP
from trialagentbench_harness.trialdev.data import discover_programs
from trialagentbench_harness.trialdev.runner import RunOptions, resume_program, run_program
from trialagentbench_harness.util.provider_environment import load_provider_dotenv


def _same_run(
    left: TrialDevObservationalSpecificationRunConfigV1,
    right: TrialDevObservationalSpecificationRunConfigV1,
) -> bool:
    return left.model_dump(exclude={"timestamp_utc"}) == right.model_dump(exclude={"timestamp_utc"})


def main(argv: Sequence[str] | None = None) -> int:
    """Execute schedule assignments in their precommitted randomized order."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-dir", required=True)
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
    schedule = read_json_model(
        TrialDevObservationalSpecificationScheduleV1,
        Path(args.schedule),
    )
    release_hash = sha256_dir_digest(participant_root)
    if release_hash != schedule.participant_release_sha256:
        raise ValueError("Participant release hash does not match the frozen schedule.")
    proposed = TrialDevObservationalSpecificationRunConfigV1(
        timestamp_utc=datetime.now(UTC),
        schedule_checksum=str(schedule.checksum),
        participant_release_sha256=release_hash,
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
        persisted = read_json_model(
            TrialDevObservationalSpecificationRunConfigV1,
            output / "run_config.json",
        )
        if not _same_run(persisted, proposed):
            raise ValueError("Resume arguments differ from the frozen observational experiment run.")
        if read_json_model(TrialDevObservationalSpecificationScheduleV1, output / "schedule.json") != schedule:
            raise ValueError("Resume schedule differs from the persisted schedule.")
        config = persisted
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
        assignment_root = output / "assignments" / assignment.assignment_id
        program = programs.get(assignment.program_id)
        if program is None:
            raise ValueError(f"Schedule references unknown program {assignment.program_id!r}.")
        run_identity = canonical_payload_sha256(
            {
                "run": config.model_dump(mode="json", exclude={"timestamp_utc"}),
                "assignment": assignment.model_dump(mode="json"),
            }
        )
        options = RunOptions(
            bundle_root=participant_root,
            output_root=assignment_root,
            model=config.model,
            procedure_assistance="output_contract_only",
            execution_scope="observational_review_only",
            observational_analysis_specification=(
                assignment.method_specification if assignment.condition == "prespecified_execution" else None
            ),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_timeout_seconds=config.request_timeout_seconds,
            max_turns_per_step=config.max_turns_per_step,
            program_watchdog_seconds=config.program_watchdog_seconds,
            run_identity_sha256=run_identity,
        )
        program_dir = assignment_root / "programs" / assignment.program_id
        if (program_dir / "chain_summary.json").is_file():
            if not args.resume:
                raise FileExistsError(program_dir)
            continue
        provider = get_provider(
            config.model,
            routing=routing,
            send_temperature=True,
            decoding_seed=assignment.decoding_seed,
            timeout_s=config.request_timeout_seconds,
        )
        if program_dir.exists():
            if not args.resume:
                raise FileExistsError(program_dir)
            run = resume_program(program, options=options, provider=provider)
        else:
            run = run_program(program, options=options, provider=provider)
        if run.execution_status not in {
            "completed",
            "model_turn_limit",
            "model_invalid_submission",
        }:
            raise RuntimeError(f"Assignment {assignment.assignment_id} did not reach a terminal status.")
        print(f"[{index}/{len(schedule.assignments)}] {assignment.assignment_id}: {run.execution_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
