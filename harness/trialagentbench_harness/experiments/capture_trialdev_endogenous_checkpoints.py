"""Capture model-produced TrialDev checkpoints at planned semantic boundaries."""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

from trialagentbench_harness.adapters import ProviderRouting, get_provider, trialdev_upstream
from trialagentbench_harness.adapters.docker_code_execution import (
    resolve_executor_environment,
)
from trialagentbench_harness.contracts.experiments import (
    ProcedureAssistanceV1,
    TrialDevCheckpointBlockPlanV1,
    TrialDevCheckpointSchedulePlanV1,
    TrialDevEndogenousCheckpointSourcesV1,
    TrialDevEndogenousCheckpointSourceV1,
)
from trialagentbench_harness.contracts.trialdev.run_checkpoint import (
    TrialDevRunCheckpointV1,
)
from trialagentbench_harness.execution_policy import TRIALDEV_RELEASE_BUDGET_V1
from trialagentbench_harness.io import (
    canonical_payload_sha256,
    read_json_model,
    sha256_dir_digest,
    sha256_file,
    write_json_model,
)
from trialagentbench_harness.ports import CodeExecutionLimitsV1, LLMProvider
from trialagentbench_harness.trialdev import agent, data, prompts, runner
from trialagentbench_harness.trialdev.data import discover_programs
from trialagentbench_harness.trialdev.runner import (
    CheckpointCaptureComplete,
    RunOptions,
    run_program,
)
from trialagentbench_harness.util.provider_environment import load_provider_dotenv

ProviderFactory = Callable[[int], LLMProvider]


def _implementation_sha256() -> str:
    """Hash the exact modules that determine a captured TrialDev state."""

    modules: tuple[ModuleType, ...] = (
        runner,
        agent,
        prompts,
        data,
        trialdev_upstream,
    )
    paths: list[Path] = []
    for module in modules:
        source = inspect.getsourcefile(module)
        if source is None:
            raise RuntimeError(f"Cannot resolve TrialDev source identity for {module.__name__!r}.")
        paths.append(Path(source).resolve())
    return canonical_payload_sha256([{"path": path.name, "sha256": sha256_file(path)} for path in paths])


def _endogenous_output_root(
    *,
    checkpoint_root: Path,
    block: TrialDevCheckpointBlockPlanV1,
) -> Path:
    """Resolve the runner output root from one safe planned programme path."""

    program_path = Path(block.endogenous_program_relative_path)
    if program_path.name != block.program_id or program_path.parent.name != "programs":
        raise ValueError("endogenous_program_relative_path must end with programs/{program_id}.")
    if program_path.is_absolute() or ".." in program_path.parts:
        raise ValueError("Endogenous checkpoint source path must be safe and relative.")
    return (Path(checkpoint_root) / program_path.parent.parent).resolve()


def _capture_block(
    *,
    participant_root: Path,
    checkpoint_root: Path,
    block: TrialDevCheckpointBlockPlanV1,
    provider_factory: ProviderFactory,
    model: str,
    procedure_assistance: ProcedureAssistanceV1,
    master_seed: int,
    max_tokens: int,
    max_turns_per_step: int,
    request_timeout_seconds: float,
    program_watchdog_seconds: int,
    executor_image: str | None,
    executor_limits: CodeExecutionLimitsV1,
    implementation_sha256: str,
) -> TrialDevEndogenousCheckpointSourceV1:
    """Run one model trajectory until its planned pre-response checkpoint."""

    programs = {program.program_id: program for program in discover_programs(participant_root)}
    program = programs.get(block.program_id)
    if program is None:
        raise ValueError(f"Checkpoint plan references unknown programme {block.program_id!r}.")
    if (program.scenario_id, program.objective_id) != (
        block.scenario_id,
        block.objective_id,
    ):
        raise ValueError("Checkpoint plan programme, scenario, and objective identities disagree.")
    output_root = _endogenous_output_root(
        checkpoint_root=checkpoint_root,
        block=block,
    )
    provider = provider_factory(block.decoding_seed)
    if provider.model != model:
        raise ValueError("Checkpoint provider returned a model identity different from the requested model.")
    run_identity = canonical_payload_sha256(
        {
            "schema_id": "trialagentbench.endogenous_checkpoint_run/v1",
            "participant_release_sha256": sha256_dir_digest(participant_root),
            "implementation_sha256": implementation_sha256,
            "program_id": block.program_id,
            "scenario_id": block.scenario_id,
            "objective_id": block.objective_id,
            "replicate_id": block.replicate_id,
            "phase_id": block.checkpoint_phase_id,
            "step_id": block.checkpoint_step_id,
            "model": provider.model,
            "provider_route": provider.telemetry_route,
            "procedure_assistance": procedure_assistance,
            "master_seed": master_seed,
            "decoding_seed": block.decoding_seed,
            "max_tokens": max_tokens,
            "max_turns_per_step": max_turns_per_step,
            "request_timeout_seconds": request_timeout_seconds,
            "program_watchdog_seconds": program_watchdog_seconds,
            "executor_image": executor_image,
        }
    )
    captured: Path | None = None

    def observer(checkpoint: TrialDevRunCheckpointV1) -> None:
        nonlocal captured
        pending = checkpoint.payload.continuation.payload.pending_step
        if (pending.phase_id, pending.step_id) != (
            block.checkpoint_phase_id,
            block.checkpoint_step_id,
        ):
            return
        if pending.turns_used != 0:
            raise ValueError("Endogenous checkpoint must precede the first assistant response.")
        captured = (
            output_root / "programs" / block.program_id / "checkpoints" / f"{checkpoint.payload.sequence:08d}.json"
        )
        raise CheckpointCaptureComplete

    options = RunOptions(
        bundle_root=participant_root,
        output_root=output_root,
        model=provider.model,
        procedure_assistance=procedure_assistance,
        master_seed=master_seed,
        temperature=0.0,
        max_tokens=max_tokens,
        request_timeout_seconds=request_timeout_seconds,
        max_turns_per_step=max_turns_per_step,
        program_watchdog_seconds=program_watchdog_seconds,
        executor_image=executor_image,
        executor_limits=executor_limits,
        run_identity_sha256=run_identity,
        checkpoint_observer=observer,
    )
    try:
        run_program(program, options=options, provider=provider)
    except CheckpointCaptureComplete:
        pass
    if captured is None or not captured.is_file():
        raise RuntimeError("Model trajectory ended before the requested checkpoint was captured.")
    envelope = read_json_model(TrialDevRunCheckpointV1, captured)
    if envelope.payload.run_identity_sha256 != run_identity:
        raise ValueError("Captured checkpoint run identity does not match its planned execution.")
    program_relative = Path(block.endogenous_program_relative_path).as_posix()
    checkpoint_relative = captured.relative_to(Path(checkpoint_root).resolve()).as_posix()
    return TrialDevEndogenousCheckpointSourceV1(
        program_id=block.program_id,
        scenario_id=block.scenario_id,
        objective_id=block.objective_id,
        replicate_id=block.replicate_id,
        decoding_seed=block.decoding_seed,
        phase_id=block.checkpoint_phase_id,
        step_id=block.checkpoint_step_id,
        program_relative_path=program_relative,
        checkpoint_relative_path=checkpoint_relative,
        checkpoint_sha256=sha256_file(captured),
        run_identity_sha256=run_identity,
        provider_model=provider.model,
        provider_route=provider.telemetry_route,
        procedure_assistance=procedure_assistance,
    )


def capture_trialdev_endogenous_checkpoints_v1(
    *,
    participant_root: Path,
    checkpoint_root: Path,
    plan: TrialDevCheckpointSchedulePlanV1,
    provider_factory: ProviderFactory,
    model: str,
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only",
    master_seed: int = 42,
    max_tokens: int = 4096,
    max_turns_per_step: int = agent.DEFAULT_MAX_TURNS_PER_STEP,
    request_timeout_seconds: float = 300.0,
    program_watchdog_seconds: int = 1800,
) -> TrialDevEndogenousCheckpointSourcesV1:
    """Capture every unique endogenous source in a prospective checkpoint plan."""

    participant = Path(participant_root).resolve()
    output = Path(checkpoint_root).resolve()
    if not participant.is_dir():
        raise NotADirectoryError(participant)
    output.mkdir(parents=True, exist_ok=True)
    identities: dict[str, tuple[str, str, str, int]] = {}
    for block in plan.blocks:
        identity = (
            block.program_id,
            block.checkpoint_phase_id,
            block.checkpoint_step_id,
            block.decoding_seed,
        )
        prior = identities.setdefault(block.endogenous_program_relative_path, identity)
        if prior != identity:
            raise ValueError("One endogenous programme path cannot describe multiple checkpoints.")
    executor = resolve_executor_environment()
    implementation_sha256 = _implementation_sha256()
    records: list[TrialDevEndogenousCheckpointSourceV1] = []
    captured_paths: set[str] = set()
    for block in plan.blocks:
        if block.endogenous_program_relative_path in captured_paths:
            continue
        captured_paths.add(block.endogenous_program_relative_path)
        records.append(
            _capture_block(
                participant_root=participant,
                checkpoint_root=output,
                block=block,
                provider_factory=provider_factory,
                model=model,
                procedure_assistance=procedure_assistance,
                master_seed=master_seed,
                max_tokens=max_tokens,
                max_turns_per_step=max_turns_per_step,
                request_timeout_seconds=request_timeout_seconds,
                program_watchdog_seconds=program_watchdog_seconds,
                executor_image=executor.image_id,
                executor_limits=executor.limits,
                implementation_sha256=implementation_sha256,
            )
        )
    receipt = TrialDevEndogenousCheckpointSourcesV1(
        participant_release_sha256=sha256_dir_digest(participant),
        records=tuple(records),
    )
    write_json_model(output / "endogenous_checkpoint_sources.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    """Capture planned model-produced TrialDev checkpoint sources."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-source-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=("openai", "openai_responses", "openrouter"), required=True)
    parser.add_argument("--openrouter-provider")
    parser.add_argument(
        "--procedure-assistance",
        choices=("output_contract_only", "unordered_checklist", "ordered_sop"),
        default="output_contract_only",
    )
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn,
    )
    parser.add_argument("--max-turns-per-step", type=int, default=agent.DEFAULT_MAX_TURNS_PER_STEP)
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
    parser.add_argument("--dotenv", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.dotenv:
        load_provider_dotenv()
    if (args.provider == "openrouter") != bool(args.openrouter_provider):
        parser.error("OpenRouter capture requires one exact --openrouter-provider pin.")
    if args.provider == "openai_responses":
        parser.error("Checkpoint-source capture requires provider decoding seeds and cannot use openai_responses.")
    routing = ProviderRouting(
        provider=args.provider,
        openrouter_provider=args.openrouter_provider,
    )

    def provider_factory(decoding_seed: int) -> LLMProvider:
        return get_provider(
            args.model,
            routing=routing,
            send_temperature=True,
            decoding_seed=decoding_seed,
            timeout_s=float(args.request_timeout_seconds),
        )

    capture_trialdev_endogenous_checkpoints_v1(
        participant_root=args.participant_dir,
        checkpoint_root=args.checkpoint_source_dir,
        plan=read_json_model(TrialDevCheckpointSchedulePlanV1, args.plan),
        provider_factory=provider_factory,
        model=args.model,
        procedure_assistance=args.procedure_assistance,
        master_seed=args.master_seed,
        max_tokens=args.max_tokens,
        max_turns_per_step=args.max_turns_per_step,
        request_timeout_seconds=args.request_timeout_seconds,
        program_watchdog_seconds=args.program_watchdog_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
