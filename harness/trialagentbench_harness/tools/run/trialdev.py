r"""CLI orchestrator for the TrialDevBench sequential benchmark.

Discovers programs from the bundle manifests, runs the per-program
orchestrator in parallel via a ThreadPoolExecutor, and writes a unified
results tree under ``results/trialdevbench/{model_slug}/{timestamp}/``.

Example invocations
-------------------

Smoke (one program):

    trialagentbench run trialdev \\
        --bundle <trialdev-runtime-dir> \\
        --provider openai \\
        --model <model-id> \\
        --programs s01:benefit_risk

Full sweep:

    trialagentbench run trialdev \\
        --bundle <trialdev-runtime-dir> \\
        --provider openrouter \\
        --model <provider/model-id> \\
        --workers 4 \\
        --seed-variants 3
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime  # type: ignore[attr-defined]
from pathlib import Path
from types import ModuleType
from typing import Literal, TypedDict, TypeVar, cast

from pydantic import ValidationError
from pydantic.types import JsonValue

from trialagentbench_harness.adapters import ProviderRouting, get_provider  # noqa: E402
from trialagentbench_harness.adapters.docker_code_execution import resolve_executor_environment
from trialagentbench_harness.contracts.core.config import (
    DecodingConfigV1,
    ExperimentConditionV1,
    ReasoningEffortV1,
    TrialDevExecutionRequestV1,
)
from trialagentbench_harness.contracts.core.runs import (
    ExecutorEnvironmentV1,
    ProviderTelemetrySummaryV1,
    RunCoverageV1,
    TrialDevChainSummaryV1,
    TrialDevRunConfigV1,
    TrialDevRunStopV1,
)
from trialagentbench_harness.contracts.experiments import procedure_assistance as assistance_contract
from trialagentbench_harness.contracts.trialdev.portfolio_release import TrialDevPortfolioParticipantViewV1
from trialagentbench_harness.contracts.trialdev.portfolio_submission import (
    TrialDevPortfolioCheckpointGradeV1,
)
from trialagentbench_harness.execution_policy import (
    TRIALDEV_DEFAULT_WORKERS,
    TRIALDEV_RELEASE_BUDGET_V1,
)
from trialagentbench_harness.io import (
    canonical_payload_sha256,
    read_json_model,
    sha256_file,
    sha256_path,
    write_json,
)
from trialagentbench_harness.io.json import write_json_model
from trialagentbench_harness.trialdev import agent as trialdev_agent  # noqa: E402
from trialagentbench_harness.trialdev import data as trialdev_data  # noqa: E402
from trialagentbench_harness.trialdev import prompts as trialdev_prompts  # noqa: E402
from trialagentbench_harness.trialdev import runner as trialdev_runner  # noqa: E402
from trialagentbench_harness.trialdev import scoring as trialdev_scoring  # noqa: E402
from trialagentbench_harness.trialdev.data import discover_programs, write_coverage_report  # noqa: E402
from trialagentbench_harness.trialdev.runner import RunOptions, resume_program, run_program  # noqa: E402
from trialagentbench_harness.trialdev.schema import (  # noqa: E402
    Program,
    ProgramExecutionStatus,
    ProgramRun,
)
from trialagentbench_harness.util.experiment_condition import (
    resolve_experiment_condition_v1,
)
from trialagentbench_harness.util.provider_environment import load_provider_dotenv
from trialagentbench_harness.util.provider_telemetry import summarize_provider_telemetry_v1
from trialagentbench_harness.util.reported_cost import (
    ReportedCostBoundProvider,
    ReportedCostBudget,
    ReportedCostThresholdReached,
    ReportedCostUnavailable,
    RunStopRequested,
    RunStopSignal,
    StoppableProvider,
)

logger = logging.getLogger("trialagentbench_harness.trialdev.cli")

TJob = TypeVar("TJob")


def _build_experiment_condition(args: argparse.Namespace) -> ExperimentConditionV1:
    """Validate and bind one request condition before any run output is created."""

    return resolve_experiment_condition_v1(
        condition_id=args.condition_id,
        request_replicate_id=args.request_replicate_id,
        reasoning_effort=args.reasoning_effort,
        reasoning_capability_snapshot=args.reasoning_capability_snapshot,
        provider=args.provider,
        model=args.model,
        openrouter_provider=args.openrouter_provider or None,
        procedure_assistance=args.procedure_assistance,
        maximum_turns_per_step=args.max_turns_per_step,
        maximum_submission_attempts=args.max_submission_attempts,
        tool_choice=args.tool_choice,
    )


def _source_set_digest(*sources: Callable[..., object] | ModuleType) -> str:
    paths = []
    for source in sources:
        path = inspect.getsourcefile(source)
        if path is None:
            raise RuntimeError(f"Cannot resolve source identity for {source!r}.")
        paths.append(sha256_file(Path(path)))
    return str(canonical_payload_sha256(cast(JsonValue, paths)))


def _trialdev_runtime_source_digest() -> str:
    """Hash the complete TrialDev runtime and contract implementation."""

    package_root = Path(trialdev_runner.__file__).resolve().parent.parent
    source_paths = {
        Path(__file__).resolve(),
        *(package_root / "trialdev").rglob("*.py"),
        *(package_root / "contracts" / "trialdev").rglob("*.py"),
        *(package_root / "io").rglob("*.py"),
        *(package_root / "ports").rglob("*.py"),
        package_root / "adapters" / "docker_code_execution.py",
        package_root / "adapters" / "llm_providers.py",
        package_root / "adapters" / "trialdev_share.py",
        package_root / "adapters" / "trialdev_upstream.py",
        package_root / "contracts" / "core" / "config.py",
        package_root / "contracts" / "core" / "runs.py",
        package_root / "contracts" / "trace" / "observable.py",
        package_root / "execution_policy.py",
        package_root / "tools" / "workspace.py",
        package_root / "util" / "provider_environment.py",
        package_root / "util" / "provider_telemetry.py",
        package_root / "util" / "reported_cost.py",
        package_root / "util" / "runtime_context.py",
    }
    payload: list[JsonValue] = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(source_paths)
    ]
    return str(canonical_payload_sha256(payload))


def _filter_portfolio_views(
    views: list[TrialDevPortfolioParticipantViewV1],
    selectors: list[str] | None,
) -> list[TrialDevPortfolioParticipantViewV1]:
    if not selectors:
        return views
    selected: list[TrialDevPortfolioParticipantViewV1] = []
    for view in views:
        programme_id = str(view.programme_id)
        scenario_id = str(view.scenario_id)
        objective_id = str(view.objective_id)
        if any(
            selector == programme_id or selector == scenario_id or selector == f"{scenario_id}:{objective_id}"
            for selector in selectors
        ):
            selected.append(view)
    return selected


def _build_portfolio_run_config(
    *,
    args: argparse.Namespace,
    selected_programme_ids: list[str],
    master_seed: int,
    executor: ExecutorEnvironmentV1,
) -> TrialDevRunConfigV1:
    from trialagentbench_harness.trialdev import portfolio_grading, portfolio_release, portfolio_runtime

    base = _build_run_config(
        args,
        len(selected_programme_ids),
        master_seed,
        executor=executor,
        selected_program_ids=selected_programme_ids,
    )
    payload = base.model_dump(mode="python", exclude={"run_identity_sha256"})
    payload.update(
        scorer_source_sha256=_source_set_digest(portfolio_grading),
        runner_source_sha256=_trialdev_runtime_source_digest(),
        prompt_interface_sha256=_source_set_digest(portfolio_runtime, trialdev_agent),
        staging_source_sha256=_source_set_digest(portfolio_release),
    )
    return cast(TrialDevRunConfigV1, TrialDevRunConfigV1.create(**payload))


def _run_portfolio_master_seed(
    *,
    args: argparse.Namespace,
    views: list[TrialDevPortfolioParticipantViewV1],
    master_seed: int,
    output_root: Path,
    label: str | None,
) -> Path:
    from trialagentbench_harness.contracts.trialdev.portfolio_submission import (
        TrialDevPortfolioRunSummaryV1,
    )
    from trialagentbench_harness.trialdev.portfolio_runtime import (
        portfolio_resource_outcomes_v1,
        resume_portfolio_programme_v1,
        run_portfolio_programme_v1,
    )

    bundle = Path(args.bundle).resolve()
    append = args.append_run_dir is not None
    if append:
        run_root = Path(str(args.append_run_dir)).resolve()
        if not run_root.is_dir():
            raise FileNotFoundError(f"Append run directory does not exist: {run_root}")
    else:
        run_root = _make_run_root(output_root, args.model, master_seed, label)
    executor = resolve_executor_environment()
    programme_ids = [str(view.programme_id) for view in views]
    requested_config = _build_portfolio_run_config(
        args=args,
        selected_programme_ids=programme_ids,
        master_seed=master_seed,
        executor=executor,
    )
    run_config = _validate_append_identity(run_root, requested_config) if append else requested_config
    if not append:
        write_json_model(run_root / "run_config.json", run_config)
    if tuple(run_config.selected_program_ids) != tuple(programme_ids):
        raise ValueError("Selected portfolio programmes conflict with the immutable prospective denominator.")
    schedule_sha256 = canonical_payload_sha256(cast(JsonValue, programme_ids))
    coverage_path = run_root / "run_coverage.json"
    if append:
        coverage = read_json_model(RunCoverageV1, coverage_path)
        if (
            coverage.run_identity_sha256 != run_config.run_identity_sha256
            or coverage.schedule_sha256 != schedule_sha256
            or coverage.unit_ids != tuple(programme_ids)
        ):
            raise ValueError("Portfolio run coverage conflicts with the immutable prospective schedule.")
    else:
        coverage = RunCoverageV1(
            run_identity_sha256=run_config.run_identity_sha256,
            schedule_sha256=schedule_sha256,
            unit_ids=tuple(programme_ids),
        )
        _write_coverage_atomic(coverage_path, coverage)
    programs_root = run_root / "programs"
    persisted_ids = (
        {path.name for path in programs_root.iterdir() if path.is_dir()} if programs_root.is_dir() else set()
    )
    unexpected = sorted(persisted_ids - set(programme_ids))
    if unexpected:
        raise ValueError(
            f"Existing portfolio output contains programmes outside the immutable denominator: {unexpected}"
        )
    runnable_views: list[TrialDevPortfolioParticipantViewV1] = []
    completed_before: set[str] = set()
    for view in views:
        programme_id = str(view.programme_id)
        programme_root = programs_root / programme_id
        if not programme_root.exists():
            runnable_views.append(view)
            continue
        summary_path = programme_root / "portfolio_run_summary.json"
        if not summary_path.is_file():
            raise FileExistsError(f"Existing portfolio programme has no continuation summary: {programme_root}")
        summary = read_json_model(TrialDevPortfolioRunSummaryV1, summary_path)
        if (
            summary.programme_id != programme_id
            or summary.scenario_id != str(view.scenario_id)
            or summary.objective_id != str(view.objective_id)
        ):
            raise ValueError(f"Existing portfolio programme identity conflicts at {programme_root}")
        if summary.execution_status in {"completed", "model_noncompletion"}:
            completed_before.add(programme_id)
        else:
            runnable_views.append(view)
    claimed_completed = set(coverage.completed_unit_ids)
    if not claimed_completed.issubset(completed_before):
        raise ValueError("Portfolio coverage claims completion without an exact terminal programme outcome.")
    reported_cost_budget = _reported_cost_budget(
        threshold_usd=run_config.reported_cost_stop_usd,
        run_root=run_root,
        coverage=coverage,
        append=append,
    )
    stop_signal = RunStopSignal()
    routing = ProviderRouting(
        provider=args.provider,
        openrouter_provider=args.openrouter_provider or None,
    )
    decoding = _default_decoding(
        send_temperature=not args.omit_temperature,
        decoding_seed=args.decoding_seed,
        max_tokens=args.max_tokens,
    )

    def execute(view: TrialDevPortfolioParticipantViewV1) -> dict[str, object]:
        started = time.monotonic()
        summary_path = run_root / "programs" / str(view.programme_id) / "portfolio_run_summary.json"
        prior_wall_seconds = (
            read_json_model(TrialDevPortfolioRunSummaryV1, summary_path).wall_seconds_total
            if summary_path.is_file()
            else 0.0
        )
        provider = get_provider(
            args.model,
            routing=routing,
            send_temperature=decoding.send_temperature,
            decoding_seed=decoding.decoding_seed,
            reasoning_effort=run_config.experiment_condition.reasoning.effort,
            exclude_reasoning=run_config.experiment_condition.reasoning.exclude_from_response,
            timeout_s=args.request_timeout_seconds,
        )
        if reported_cost_budget is not None:
            provider = ReportedCostBoundProvider(provider=provider, budget=reported_cost_budget)
        provider = StoppableProvider(provider=provider, signal=stop_signal)

        def persist_incomplete(
            *,
            execution_status: Literal["infrastructure_failure", "run_stopped"],
            error: BaseException,
        ) -> None:
            programme_root = run_root / "programs" / str(view.programme_id)
            programme_root.mkdir(parents=True, exist_ok=True)
            from trialagentbench_harness.trialdev.portfolio_release import load_portfolio_catalogue_v1

            relative_paths = {
                name: tuple(
                    path.relative_to(programme_root).as_posix()
                    for path in sorted((programme_root / name).glob("*.json"))
                )
                for name in ("states", "submissions", "grades")
            }
            reached = tuple(
                read_json_model(
                    TrialDevPortfolioCheckpointGradeV1,
                    programme_root / relative_path,
                ).checkpoint_id
                for relative_path in relative_paths["grades"]
            )
            resources = portfolio_resource_outcomes_v1(programme_root)
            failure = TrialDevPortfolioRunSummaryV1(
                programme_id=str(view.programme_id),
                scenario_id=str(view.scenario_id),
                objective_id=str(view.objective_id),
                resource_budget_units=view.resource_budget_units,
                participant_view_checksum=str(view.checksum),
                release_source_identity=load_portfolio_catalogue_v1(bundle).source_identity,
                execution_status=execution_status,
                reached_checkpoint_ids=reached,
                state_relative_paths=relative_paths["states"],
                submission_relative_paths=relative_paths["submissions"],
                grade_relative_paths=relative_paths["grades"],
                wall_seconds_total=prior_wall_seconds + time.monotonic() - started,
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
                error=f"{type(error).__name__}: {error}",
            )
            write_json_model(programme_root / "portfolio_run_summary.json", failure)

        try:
            execute_programme = (
                resume_portfolio_programme_v1
                if (run_root / "programs" / str(view.programme_id)).exists()
                else run_portfolio_programme_v1
            )
            summary = execute_programme(
                release_root=bundle,
                programme_id=str(view.programme_id),
                output_root=run_root,
                provider=provider,
                max_turns_per_checkpoint=args.max_turns_per_step,
                max_tokens=args.max_tokens,
                max_context_characters=args.max_context_characters,
                watchdog_seconds=args.program_watchdog_seconds,
                max_submission_attempts=args.max_submission_attempts,
                procedure_assistance=args.procedure_assistance,
                tool_choice=args.tool_choice,
                executor_image=executor.image_id,
                executor_limits=executor.limits,
                verbose=args.verbose,
            )
        except (ReportedCostThresholdReached, ReportedCostUnavailable, RunStopRequested) as exc:
            persist_incomplete(execution_status="run_stopped", error=exc)
            raise
        except (OSError, RuntimeError, ValueError, ValidationError) as exc:
            persist_incomplete(execution_status="infrastructure_failure", error=exc)
            return {
                "program_id": str(view.programme_id),
                "execution_status": "infrastructure_failure",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": time.monotonic() - started,
            }
        return {
            "program_id": summary.programme_id,
            "execution_status": summary.execution_status,
            "terminal_disposition": summary.terminal_disposition,
            "n_checkpoints": len(summary.reached_checkpoint_ids),
            "elapsed_s": time.monotonic() - started,
        }

    batch = _run_bounded_batch(
        jobs=runnable_views,
        workers=args.workers,
        unit_id=lambda view: str(view.programme_id),
        execute=execute,
        reported_cost_budget=reported_cost_budget,
        stop_signal=stop_signal,
    )
    results = batch["results"]
    completed_now = {
        str(row["program_id"]) for row in results if row["execution_status"] in {"completed", "model_noncompletion"}
    }
    completed = tuple(
        programme_id for programme_id in programme_ids if programme_id in completed_before | completed_now
    )
    _write_coverage_atomic(
        coverage_path,
        coverage.model_copy(update={"completed_unit_ids": completed}),
    )
    write_json(
        run_root / "program_run_index.json",
        {
            "runs": sorted(results, key=lambda row: str(row["program_id"])),
            "completed_before_append": sorted(completed_before),
        },
    )
    telemetry = summarize_provider_telemetry_v1(
        run_root=run_root,
        coverage=read_json_model(RunCoverageV1, coverage_path),
    )
    if batch["stop_reason"] is not None:
        _write_run_stop(
            run_root=run_root,
            run_config=run_config,
            reason=batch["stop_reason"],
            unit_ids=programme_ids,
            completed_unit_ids=set(completed),
            not_started_unit_ids=batch["not_started_unit_ids"],
            telemetry=telemetry,
        )
        return run_root
    infrastructure_failures = tuple(
        str(row["program_id"]) for row in results if row["execution_status"] == "infrastructure_failure"
    )
    if infrastructure_failures:
        raise RuntimeError(
            "TrialDev portfolio infrastructure failed for programme(s): " + ", ".join(infrastructure_failures)
        )
    return run_root


def _run_portfolio_release(args: argparse.Namespace) -> int:
    from trialagentbench_harness.trialdev.portfolio_release import (
        load_portfolio_catalogue_v1,
        validate_portfolio_release_v1,
    )

    bundle = Path(args.bundle).resolve()
    validate_portfolio_release_v1(bundle)
    catalogue = load_portfolio_catalogue_v1(bundle)
    views = _filter_portfolio_views(list(catalogue.views), args.programs)
    if not views:
        raise ValueError("No portfolio programmes matched the requested filter.")
    output_root = Path(args.output_root) / _slugify_model(args.model)
    output_root.mkdir(parents=True, exist_ok=True)
    for variant in range(max(1, int(args.seed_variants))):
        master_seed = int(args.master_seed) + variant
        label = args.label
        if int(args.seed_variants) > 1:
            suffix = f"variant{variant}_seed{master_seed}"
            label = f"{label}_{suffix}" if label else suffix
        run_root = _run_portfolio_master_seed(
            args=args,
            views=views,
            master_seed=master_seed,
            output_root=output_root,
            label=label,
        )
        sys.stdout.write(f"Portfolio run artifacts written to: {run_root}\n")
    return 0


class _ProgressState(TypedDict):
    completed: int
    total: int


class _ProgramRunIndexEntry(TypedDict, total=False):
    program_id: str
    stopped_at_phase: str | None
    n_phases: int
    error: str | None
    execution_status: ProgramExecutionStatus
    elapsed_s: float
    skipped: bool
    reason: str


class _BoundedBatchResult(TypedDict):
    results: list[dict[str, object]]
    interrupted_unit_ids: list[str]
    not_started_unit_ids: list[str]
    stop_reason: str | None


def _run_bounded_batch(
    *,
    jobs: Sequence[TJob],
    workers: int,
    unit_id: Callable[[TJob], str],
    execute: Callable[[TJob], dict[str, object]],
    reported_cost_budget: ReportedCostBudget | None,
    stop_signal: RunStopSignal,
) -> _BoundedBatchResult:
    """Run a rolling worker window with exact cost and interruption custody."""

    if workers < 1:
        raise ValueError("workers must be at least 1.")
    pending_jobs = list(jobs)
    results: list[dict[str, object]] = []
    interrupted: list[str] = []
    submitted_ids: set[str] = set()
    stop_reason: str | None = None
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="trialdev")
    active: dict[Future[dict[str, object]], TJob] = {}

    def submit_available() -> None:
        nonlocal stop_reason
        if reported_cost_budget is not None:
            snapshot = reported_cost_budget.snapshot()
            if not snapshot.cost_complete:
                stop_reason = "reported_cost_unavailable"
            elif snapshot.threshold_reached:
                stop_reason = "reported_cost_threshold"
        while pending_jobs and len(active) < workers and stop_reason is None:
            job = pending_jobs.pop(0)
            submitted_ids.add(unit_id(job))
            active[executor.submit(execute, job)] = job

    try:
        submit_available()
        while active:
            try:
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                stop_reason = "keyboard_interrupt"
                stop_signal.request()
                for future, job in tuple(active.items()):
                    if future.cancel():
                        submitted_ids.remove(unit_id(job))
                        del active[future]
                continue
            for future in completed:
                job = active.pop(future)
                try:
                    results.append(future.result())
                except RunStopRequested:
                    interrupted.append(unit_id(job))
                except ReportedCostThresholdReached:
                    stop_reason = "reported_cost_threshold"
                    stop_signal.request()
                    interrupted.append(unit_id(job))
                except ReportedCostUnavailable:
                    stop_reason = "reported_cost_unavailable"
                    stop_signal.request()
                    interrupted.append(unit_id(job))
            submit_available()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    not_started = [unit_id(job) for job in jobs if unit_id(job) not in submitted_ids]
    return _BoundedBatchResult(
        results=results,
        interrupted_unit_ids=interrupted,
        not_started_unit_ids=not_started,
        stop_reason=stop_reason,
    )


def _default_decoding(
    *,
    send_temperature: bool,
    decoding_seed: int | None,
    max_tokens: int,
) -> DecodingConfigV1:
    """Single source of reference for pinned decoding parameters in TrialDev runs."""
    return DecodingConfigV1(
        temperature=0.0,
        max_tokens=max_tokens,
        send_temperature=send_temperature,
        decoding_seed=decoding_seed,
    )


def _maybe_load_dotenv(enabled: bool) -> None:
    if not enabled:
        return
    load_provider_dotenv()


def _slugify_model(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


def _filter_programs(
    programs: list[Program],
    selectors: list[str] | None,
) -> list[Program]:
    if not selectors:
        return programs
    wanted: set[tuple[str, str | None]] = set()
    programme_ids: set[str] = set()
    for raw in selectors:
        if ":" in raw:
            scen, obj = raw.split(":", 1)
            wanted.add((scen.strip(), obj.strip()))
        else:
            selector = raw.strip()
            wanted.add((selector, None))
            programme_ids.add(selector)
    out = []
    for program in programs:
        if (
            program.program_id in programme_ids
            or (program.scenario_id, program.objective_id) in wanted
            or (
                program.scenario_id,
                None,
            )
            in wanted
        ):
            out.append(program)
    return out


def _make_run_root(output_root: Path, model: str, master_seed: int, label: str | None) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    parts = [_slugify_model(model), timestamp]
    if label:
        parts.append(label)
    if master_seed != 42:
        parts.append(f"seed{master_seed}")
    run_root = output_root / "_".join(parts)
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def _build_run_config(
    args: argparse.Namespace,
    n_programs: int,
    master_seed: int,
    *,
    executor: ExecutorEnvironmentV1,
    selected_program_ids: list[str] | None = None,
) -> TrialDevRunConfigV1:
    from trialagentbench_harness.contracts.core.config import RoutingConfigV1

    selected_ids = list(selected_program_ids or [])
    payload: dict[str, JsonValue] = {
        "schema_id": "trialagentbench_trialdev_run_config_v1",
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "bundle": str(Path(args.bundle).resolve()),
        "bundle_sha256": sha256_path(Path(args.bundle).resolve()),
        "scorer_source_sha256": _source_digest(trialdev_scoring),
        "runner_source_sha256": _trialdev_runtime_source_digest(),
        "prompt_interface_sha256": _prompt_interface_digest(),
        "staging_source_sha256": _staging_source_digest(),
        "procedure_assistance": str(args.procedure_assistance),
        "model": str(args.model),
        "experiment_condition": _build_experiment_condition(args).model_dump(mode="json"),
        "master_seed": int(master_seed),
        "seed_variants": int(args.seed_variants),
        "workers": int(args.workers),
        "reported_cost_stop_usd": args.reported_cost_stop_usd,
        "decoding": _default_decoding(
            send_temperature=not args.omit_temperature,
            decoding_seed=args.decoding_seed,
            max_tokens=args.max_tokens,
        ).model_dump(mode="json"),
        "routing": RoutingConfigV1(
            provider=args.provider,
            openrouter_provider=(args.openrouter_provider or None),
            request_timeout_seconds=args.request_timeout_seconds,
        ).model_dump(mode="json"),
        "executor": executor.model_dump(mode="json"),
        "max_turns_per_step": int(args.max_turns_per_step),
        "max_context_characters": int(args.max_context_characters),
        "max_phase_retries": int(args.max_phase_retries),
        "max_submission_attempts": int(args.max_submission_attempts),
        "program_watchdog_seconds": int(args.program_watchdog_seconds),
        "programs_filter": list(args.programs or []),
        "selected_program_ids": cast(list[JsonValue], selected_ids),
        "n_programs_selected": int(n_programs),
        "label": getattr(args, "label", None),
    }
    return cast(TrialDevRunConfigV1, TrialDevRunConfigV1.create(**payload))


def _source_digest(function: Callable[..., object] | ModuleType) -> str:
    source = inspect.getsourcefile(function)
    if source is None:
        raise RuntimeError(f"Cannot resolve source identity for {function!r}.")
    return str(sha256_file(Path(source)))


def _prompt_interface_digest() -> str:
    sources = (inspect.getsourcefile(trialdev_prompts), inspect.getsourcefile(trialdev_agent))
    if any(source is None for source in sources):
        raise RuntimeError("Cannot resolve TrialDev prompt/interface source identity.")
    payload = [sha256_file(Path(source)) for source in sources if source is not None]
    return str(canonical_payload_sha256(cast(JsonValue, payload)))


def _staging_source_digest() -> str:
    """Hash sources that determine assistance-dependent workspace staging."""

    sources = (
        inspect.getsourcefile(trialdev_data),
        inspect.getsourcefile(assistance_contract),
    )
    if any(source is None for source in sources):
        raise RuntimeError("Cannot resolve TrialDev staging source identity.")
    payload = [sha256_file(Path(source)) for source in sources if source is not None]
    return str(canonical_payload_sha256(cast(JsonValue, payload)))


def _write_coverage_atomic(path: Path, coverage: RunCoverageV1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(coverage.model_dump(mode="json"), indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reported_cost_budget(
    *,
    threshold_usd: float | None,
    run_root: Path,
    coverage: RunCoverageV1,
    append: bool,
) -> ReportedCostBudget | None:
    """Create or restore one run-level provider-reported cost stop."""

    if threshold_usd is None:
        return None
    if not append:
        return ReportedCostBudget(threshold_usd=threshold_usd)
    summary = summarize_provider_telemetry_v1(run_root=run_root, coverage=coverage)
    return ReportedCostBudget.from_observation(
        threshold_usd=threshold_usd,
        observed_usd=summary.reported_cost_usd,
        response_count=summary.response_count,
        cost_complete=summary.responses_with_reported_cost == summary.response_count,
    )


def _write_run_stop(
    *,
    run_root: Path,
    run_config: TrialDevRunConfigV1,
    reason: str,
    unit_ids: Sequence[str],
    completed_unit_ids: set[str],
    not_started_unit_ids: Sequence[str],
    telemetry: ProviderTelemetrySummaryV1,
) -> TrialDevRunStopV1:
    """Persist exact scheduled-unit custody after an orderly run stop."""

    if reason not in {"reported_cost_threshold", "reported_cost_unavailable", "keyboard_interrupt"}:
        raise ValueError(f"Unsupported run stop reason: {reason!r}.")
    threshold = run_config.reported_cost_stop_usd
    if reason.startswith("reported_cost") and threshold is None:
        raise ValueError("A reported-cost stop requires a configured threshold.")
    not_started = set(not_started_unit_ids)
    interrupted = set(unit_ids) - completed_unit_ids - not_started
    observed_cost = telemetry.reported_cost_usd
    overshoot = (
        max(0.0, observed_cost - threshold) if reason == "reported_cost_threshold" and threshold is not None else 0.0
    )
    record = TrialDevRunStopV1(
        run_identity_sha256=run_config.run_identity_sha256,
        reason=cast(
            Literal["reported_cost_threshold", "reported_cost_unavailable", "keyboard_interrupt"],
            reason,
        ),
        unit_ids=tuple(unit_ids),
        completed_unit_ids=tuple(item for item in unit_ids if item in completed_unit_ids),
        interrupted_unit_ids=tuple(item for item in unit_ids if item in interrupted),
        not_started_unit_ids=tuple(item for item in unit_ids if item in not_started),
        reported_cost_threshold_usd=threshold,
        observed_reported_cost_usd=observed_cost,
        reported_cost_overshoot_usd=overshoot,
        provider_response_count=telemetry.response_count,
        cost_complete=telemetry.responses_with_reported_cost == telemetry.response_count,
    )
    write_json_model(run_root / "run_stop.json", record)
    return record


_APPEND_IDENTITY_EXCLUDED_FIELDS = {
    "timestamp_utc",
    "workers",
    "programs_filter",
    "n_programs_selected",
    "label",
}


def _validate_append_identity(run_root: Path, requested: TrialDevRunConfigV1) -> TrialDevRunConfigV1:
    persisted = read_json_model(TrialDevRunConfigV1, run_root / "run_config.json")
    persisted_identity = persisted.model_dump(mode="json", exclude=_APPEND_IDENTITY_EXCLUDED_FIELDS)
    requested_identity = requested.model_dump(mode="json", exclude=_APPEND_IDENTITY_EXCLUDED_FIELDS)
    if persisted_identity != requested_identity:
        mismatched = sorted(
            key for key in persisted_identity if persisted_identity.get(key) != requested_identity.get(key)
        )
        raise ValueError(f"Append run identity conflicts with persisted run_config.json fields: {mismatched}")
    return cast(TrialDevRunConfigV1, persisted)


def _programs_to_append(run_root: Path, programs: list[Program]) -> tuple[list[Program], list[str]]:
    runnable: list[Program] = []
    skipped: list[str] = []
    seen: set[str] = set()
    programs_root = run_root / "programs"
    persisted_ids = (
        {path.name for path in programs_root.iterdir() if path.is_dir()} if programs_root.is_dir() else set()
    )
    selected_ids = {program.program_id for program in programs}
    extra_ids = sorted(persisted_ids - selected_ids)
    if extra_ids:
        raise ValueError(f"Existing TrialDev output contains programs outside the immutable denominator: {extra_ids}")
    for program in programs:
        if program.program_id in seen:
            raise ValueError(f"Duplicate selected TrialDev program_id: {program.program_id}")
        seen.add(program.program_id)
        program_dir = run_root / "programs" / program.program_id
        if not program_dir.exists():
            runnable.append(program)
            continue
        if not program_dir.is_dir():
            raise FileExistsError(f"TrialDev program destination is not a directory: {program_dir}")
        if not (program_dir / "chain_summary.json").is_file():
            if not (program_dir / "checkpoints").is_dir():
                raise FileExistsError(
                    f"Existing TrialDev program has neither completion nor continuation custody: {program_dir}"
                )
            runnable.append(program)
            continue
        chain = read_json_model(TrialDevChainSummaryV1, program_dir / "chain_summary.json")
        expected_identity = (program.program_id, program.scenario_id, program.objective_id)
        persisted_identity = (chain.program_id, chain.scenario_id, chain.objective_id)
        if persisted_identity != expected_identity:
            raise ValueError(
                f"Existing TrialDev program identity conflict at {program_dir}: "
                f"expected {expected_identity}, found {persisted_identity}"
            )
        terminal = (chain.execution_status == "completed" and chain.error is None) or (
            chain.execution_status in {"model_turn_limit", "model_invalid_submission"} and chain.error is not None
        )
        if not terminal:
            if not (program_dir / "checkpoints").is_dir():
                raise FileExistsError(f"Existing TrialDev program has no exact continuation checkpoint: {program_dir}")
            runnable.append(program)
            continue
        skipped.append(program.program_id)
    return runnable, skipped


def _program_progress_status(run: ProgramRun) -> str:
    """Render the scientific outcome separately from infrastructure failures."""

    if run.execution_status in {"model_turn_limit", "model_invalid_submission"}:
        return str(run.execution_status)
    if run.error:
        return f"ERROR ({run.error.splitlines()[0][:120]})"
    if run.stopped_at_phase:
        return f"stopped_at={run.stopped_at_phase}"
    return "completed"


def _execute_one(
    program: Program,
    *,
    options: RunOptions,
    model: str,
    routing: ProviderRouting,
    send_temperature: bool,
    decoding_seed: int | None,
    reasoning_effort: ReasoningEffortV1 | None,
    exclude_reasoning: bool,
    reported_cost_budget: ReportedCostBudget | None,
    stop_signal: RunStopSignal,
    state_lock: threading.Lock,
    progress: _ProgressState,
) -> _ProgramRunIndexEntry:
    started = time.monotonic()
    provider = get_provider(
        model,
        routing=routing,
        send_temperature=send_temperature,
        decoding_seed=decoding_seed,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=exclude_reasoning,
        timeout_s=options.request_timeout_seconds,
    )
    if reported_cost_budget is not None:
        provider = ReportedCostBoundProvider(provider=provider, budget=reported_cost_budget)
    provider = StoppableProvider(provider=provider, signal=stop_signal)
    program_dir = options.output_root / "programs" / program.program_id
    if program_dir.exists():
        run = resume_program(program, options=options, provider=provider)
    else:
        run = run_program(program, options=options, provider=provider)
    elapsed = time.monotonic() - started
    with state_lock:
        progress["completed"] += 1
        n_done = progress["completed"]
        n_total = progress["total"]
        status = _program_progress_status(run)
        sys.stdout.write(
            f"  [{n_done}/{n_total}] {program.program_id} {status} phases={len(run.phases)} elapsed={elapsed:.1f}s\n"
        )
        sys.stdout.flush()
    return {
        "program_id": program.program_id,
        "stopped_at_phase": run.stopped_at_phase,
        "n_phases": len(run.phases),
        "error": run.error,
        "execution_status": run.execution_status,
        "elapsed_s": elapsed,
    }


def _attach_run_log(run_root: Path, *, append: bool) -> logging.Handler:
    """Attach a file handler for harness events and dependency warnings."""
    handler = logging.FileHandler(run_root / "run.log", mode="a" if append else "x", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _configure_logging(*, verbose: bool) -> None:
    """Enable harness debug logs without exposing full provider payloads."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    level = logging.DEBUG if verbose else logging.NOTSET
    for logger_name in ("trialagentbench_harness.trialdev", "trialagentbench_harness"):
        logging.getLogger(logger_name).setLevel(level)


def run_one_master_seed(
    *,
    args: argparse.Namespace,
    programs: list[Program],
    master_seed: int,
    output_root: Path,
    label: str | None,
) -> Path:
    append_dir = getattr(args, "append_run_dir", None)
    append = append_dir is not None
    if append and int(args.seed_variants) != 1:
        raise ValueError("--append-run-dir requires --seed-variants 1.")
    if append:
        run_root = Path(str(append_dir)).resolve()
        if not run_root.is_dir():
            raise FileNotFoundError(f"Append run directory does not exist: {run_root}")
        sys.stdout.write(f"Appending missing programs to existing run dir: {run_root}\n")
    else:
        run_root = _make_run_root(output_root, args.model, master_seed, label)
    executor_environment = resolve_executor_environment()
    requested_config = _build_run_config(
        args,
        len(programs),
        master_seed,
        executor=executor_environment,
        selected_program_ids=[program.program_id for program in programs],
    )
    if append:
        run_config = _validate_append_identity(run_root, requested_config)
        runnable_programs, skipped_program_ids = _programs_to_append(run_root, programs)
    else:
        run_config = requested_config
        runnable_programs = programs
        skipped_program_ids = []
        write_json_model(run_root / "run_config.json", run_config)
        write_coverage_report(programs, run_root / "coverage_report.json")
    selected_program_ids = tuple(run_config.selected_program_ids)
    if selected_program_ids != tuple(program.program_id for program in programs):
        raise ValueError("Selected TrialDev programs conflict with the immutable prospective denominator.")
    schedule_sha256 = canonical_payload_sha256(list(selected_program_ids))
    custody_path = run_root / "run_coverage.json"
    if append:
        custody = read_json_model(RunCoverageV1, custody_path)
        if (
            custody.run_identity_sha256 != run_config.run_identity_sha256
            or custody.schedule_sha256 != schedule_sha256
            or custody.unit_ids != selected_program_ids
        ):
            raise ValueError("TrialDev run coverage conflicts with the immutable prospective schedule.")
    else:
        custody = RunCoverageV1(
            run_identity_sha256=run_config.run_identity_sha256,
            schedule_sha256=schedule_sha256,
            unit_ids=selected_program_ids,
        )
        _write_coverage_atomic(custody_path, custody)
    completed_program_ids = set(skipped_program_ids)
    if any(program_id not in completed_program_ids for program_id in custody.completed_unit_ids):
        raise ValueError("TrialDev coverage claims completion without an exact completed program.")
    completed_in_order = tuple(
        program_id for program_id in selected_program_ids if program_id in completed_program_ids
    )
    if custody.completed_unit_ids != completed_in_order:
        custody = custody.model_copy(update={"completed_unit_ids": completed_in_order})
        _write_coverage_atomic(custody_path, custody)
    reported_cost_budget = _reported_cost_budget(
        threshold_usd=run_config.reported_cost_stop_usd,
        run_root=run_root,
        coverage=custody,
        append=append,
    )
    stop_signal = RunStopSignal()
    log_handler = _attach_run_log(run_root, append=append)

    routing = ProviderRouting(
        provider=args.provider,
        openrouter_provider=(args.openrouter_provider or None),
    )

    decoding = _default_decoding(
        send_temperature=not args.omit_temperature,
        decoding_seed=args.decoding_seed,
        max_tokens=args.max_tokens,
    )
    options = RunOptions(
        bundle_root=Path(args.bundle).resolve(),
        output_root=run_root,
        model=args.model,
        procedure_assistance=args.procedure_assistance,
        tool_choice=args.tool_choice,
        master_seed=master_seed,
        temperature=float(decoding.temperature),
        max_tokens=int(decoding.max_tokens),
        max_context_chars=int(run_config.max_context_characters),
        request_timeout_seconds=float(args.request_timeout_seconds),
        max_turns_per_step=args.max_turns_per_step,
        max_phase_retries=int(getattr(args, "max_phase_retries", 10)),
        program_watchdog_seconds=int(
            getattr(
                args,
                "program_watchdog_seconds",
                TRIALDEV_RELEASE_BUDGET_V1.wall_time_limit_seconds,
            )
        ),
        verbose=bool(args.verbose),
        executor_image=executor_environment.image_id,
        executor_limits=executor_environment.limits,
        run_identity_sha256=run_config.run_identity_sha256,
    )

    state_lock = threading.Lock()
    progress = _ProgressState(completed=0, total=len(runnable_programs))
    results: list[_ProgramRunIndexEntry] = [
        {"program_id": program_id, "skipped": True, "reason": "exact completed program exists"}
        for program_id in skipped_program_ids
    ]

    def record_result(result: _ProgramRunIndexEntry) -> None:
        results.append(result)
        if result.get("execution_status") in {
            "completed",
            "model_turn_limit",
            "model_invalid_submission",
        }:
            completed_program_ids.add(result["program_id"])
            completed_now = tuple(
                program_id for program_id in selected_program_ids if program_id in completed_program_ids
            )
            _write_coverage_atomic(
                custody_path,
                custody.model_copy(update={"completed_unit_ids": completed_now}),
            )

    def execute_program(program: Program) -> dict[str, object]:
        return cast(
            dict[str, object],
            _execute_one(
                program,
                options=options,
                model=args.model,
                routing=routing,
                send_temperature=decoding.send_temperature,
                decoding_seed=decoding.decoding_seed,
                reasoning_effort=run_config.experiment_condition.reasoning.effort,
                exclude_reasoning=run_config.experiment_condition.reasoning.exclude_from_response,
                reported_cost_budget=reported_cost_budget,
                stop_signal=stop_signal,
                state_lock=state_lock,
                progress=progress,
            ),
        )

    batch = _run_bounded_batch(
        jobs=runnable_programs,
        workers=args.workers,
        unit_id=lambda program: program.program_id,
        execute=execute_program,
        reported_cost_budget=reported_cost_budget,
        stop_signal=stop_signal,
    )
    for result in batch["results"]:
        record_result(cast(_ProgramRunIndexEntry, result))

    (run_root / "program_run_index.json").write_text(
        json.dumps({"runs": sorted(results, key=lambda r: r.get("program_id", ""))}, indent=2),
        encoding="utf-8",
    )
    telemetry = summarize_provider_telemetry_v1(
        run_root=run_root,
        coverage=read_json_model(RunCoverageV1, custody_path),
    )
    if batch["stop_reason"] is not None:
        _write_run_stop(
            run_root=run_root,
            run_config=run_config,
            reason=batch["stop_reason"],
            unit_ids=selected_program_ids,
            completed_unit_ids=completed_program_ids,
            not_started_unit_ids=batch["not_started_unit_ids"],
            telemetry=telemetry,
        )
        sys.stdout.write(f"\nRun stopped with exact custody at: {run_root}\n")
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()
        return run_root
    infrastructure_failures = tuple(
        str(result["program_id"])
        for result in results
        if result.get("execution_status") in {"infrastructure_timeout", "infrastructure_error"}
    )
    if infrastructure_failures:
        raise RuntimeError("TrialDev infrastructure failed for programme(s): " + ", ".join(infrastructure_failures))

    sys.stdout.write(f"\nRun artifacts written to: {run_root}\n")
    sys.stdout.write("Grade this immutable run with `trialagentbench grade trialdev` before analysis.\n")

    logging.getLogger().removeHandler(log_handler)
    log_handler.close()
    return run_root


def _request_path(value: str, *, base: Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def _execution_request_argv(request: TrialDevExecutionRequestV1, *, base: Path) -> tuple[str, ...]:
    """Translate one typed execution request into the canonical CLI surface."""

    arguments = [
        "--bundle",
        _request_path(request.bundle, base=base),
        "--model",
        request.model,
        "--provider",
        request.provider,
    ]
    if request.dotenv:
        arguments.append("--dotenv")
    scalar_options: tuple[tuple[str, object | None], ...] = (
        ("--openrouter-provider", request.openrouter_provider),
        ("--condition-id", request.condition_id),
        ("--request-replicate-id", request.request_replicate_id),
        ("--reasoning-effort", request.reasoning_effort),
        ("--master-seed", request.master_seed),
        ("--seed-variants", request.seed_variants),
        ("--max-phase-retries", request.max_phase_retries),
        ("--max-submission-attempts", request.max_submission_attempts),
        ("--program-watchdog-seconds", request.program_watchdog_seconds),
        ("--workers", request.workers),
        ("--reported-cost-stop-usd", request.reported_cost_stop_usd),
        ("--max-turns-per-step", request.max_turns_per_step),
        ("--max-tokens", request.max_tokens),
        ("--max-context-characters", request.max_context_characters),
        ("--procedure-assistance", request.procedure_assistance),
        ("--tool-choice", request.tool_choice),
        ("--label", request.label),
        ("--request-timeout-seconds", request.request_timeout_seconds),
        ("--decoding-seed", request.decoding_seed),
    )
    for option, value in scalar_options:
        if value is not None:
            arguments.extend((option, str(value)))
    path_options = (
        ("--reasoning-capability-snapshot", request.reasoning_capability_snapshot),
        ("--output-root", request.output_root),
        ("--append-run-dir", request.append_run_dir),
    )
    for option, value in path_options:
        if value is not None:
            arguments.extend((option, _request_path(value, base=base)))
    if request.programs is not None:
        arguments.append("--programs")
        arguments.extend(request.programs)
    if request.omit_temperature:
        arguments.append("--omit-temperature")
    return tuple(arguments)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_arguments = tuple(sys.argv[1:] if argv is None else argv)
    if "--experiment-config" in raw_arguments:
        if len(raw_arguments) != 2 or raw_arguments[0] != "--experiment-config":
            raise ValueError("--experiment-config must be used alone.")
        config_path = Path(raw_arguments[1]).resolve()
        request = read_json_model(TrialDevExecutionRequestV1, config_path)
        return parse_args(_execution_request_argv(request, base=config_path.parent))
    parser = argparse.ArgumentParser(description="TrialDevBench harness CLI")
    parser.add_argument(
        "--experiment-config",
        help="Run one strict trialagentbench.trialdev_execution_request/v1 JSON configuration.",
    )
    parser.add_argument(
        "--dotenv",
        action="store_true",
        help="Load environment variables from a local .env file (off by default).",
    )
    parser.add_argument("--bundle", required=True, help="Path to an extracted TrialDevBench release root")
    parser.add_argument("--model", required=True, help="Exact model identifier for the selected provider.")
    parser.add_argument(
        "--condition-id",
        default="primary",
        help="Stable identifier for this experimental condition.",
    )
    parser.add_argument(
        "--request-replicate-id",
        default="request-1",
        help="Stable identifier for this provider request replicate.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Exact provider-supported reasoning effort; omitted by default.",
    )
    parser.add_argument(
        "--reasoning-capability-snapshot",
        type=Path,
        default=None,
        help="Source-bound capability record required when --reasoning-effort is set.",
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=("openai", "openai_responses", "openrouter"),
        help=(
            "Exact API transport: openai uses Chat Completions, openai_responses "
            "uses Responses, and openrouter uses pinned OpenRouter Chat Completions."
        ),
    )
    parser.add_argument(
        "--programs",
        nargs="*",
        default=None,
        help=(
            "Run only the listed programme identifiers. Single-asset releases also accept "
            "'scenario:objective' selectors such as 's01:benefit_risk'. Default: all programmes."
        ),
    )
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument(
        "--seed-variants",
        type=int,
        default=1,
        help="Run the same set of programs N times with master_seed in {seed, seed+1, ...}. Default 1.",
    )
    parser.add_argument(
        "--max-phase-retries",
        type=int,
        default=10,
        help="How many times the materializer can reject before the program errors out. Default: 10.",
    )
    parser.add_argument(
        "--max-submission-attempts",
        type=int,
        default=3,
        help="Maximum corrected portfolio submissions at one checkpoint. Default: 3.",
    )
    parser.add_argument(
        "--program-watchdog-seconds",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.wall_time_limit_seconds,
        help="Per-program wall-clock budget checked before each provider/tool operation. Default 1800 (30 min).",
    )
    parser.add_argument("--workers", type=int, default=TRIALDEV_DEFAULT_WORKERS)
    parser.add_argument(
        "--reported-cost-stop-usd",
        type=float,
        default=None,
        help=(
            "Stop new provider requests after cumulative provider-reported cost reaches this positive "
            "USD threshold. Concurrent in-flight responses may produce a reported overshoot."
        ),
    )
    parser.add_argument(
        "--max-turns-per-step",
        type=int,
        default=trialdev_agent.DEFAULT_MAX_TURNS_PER_STEP,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn,
        help="Maximum completion tokens requested for each model turn.",
    )
    parser.add_argument(
        "--max-context-characters",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.maximum_context_characters,
        help="Maximum retained conversation size before deterministic context compaction.",
    )
    parser.add_argument(
        "--procedure-assistance",
        choices=("output_contract_only", "unordered_checklist", "ordered_sop"),
        default="output_contract_only",
        help=(
            "Participant-facing analysis support. output_contract_only is the primary "
            "benchmark surface; checklist and SOP conditions are assistance ablations."
        ),
    )
    parser.add_argument(
        "--tool-choice",
        choices=("auto", "required"),
        default="auto",
        help="Provider tool-choice policy. Auto is the benchmark default; required is an interface ablation.",
    )
    parser.add_argument(
        "--output-root",
        default="results/trialdevbench",
        help="Output directory root for runs.",
    )
    parser.add_argument("--label", default=None, help="Optional suffix added to the run dir name.")
    parser.add_argument(
        "--append-run-dir",
        default=None,
        help="Append missing programs to an identity-matched run. Exact completed "
        "programs are skipped; conflicting or incomplete destinations fail.",
    )
    parser.add_argument(
        "--openrouter-provider",
        type=str,
        default=None,
        help=("Required exact upstream-provider pin when --provider=openrouter (for example, GMICloud)."),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=TRIALDEV_RELEASE_BUDGET_V1.provider_request_timeout_seconds,
        help="Per-attempt provider request timeout (default: 300 seconds).",
    )
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help="Do not send temperature when the selected endpoint does not support it.",
    )
    parser.add_argument(
        "--decoding-seed",
        type=int,
        help="Optional non-negative provider decoding seed.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.provider == "openai_responses" and args.decoding_seed is not None:
        sys.stderr.write("--decoding-seed is not supported by --provider openai_responses.\n")
        return 2
    try:
        _build_experiment_condition(args)
    except (OSError, ValueError, ValidationError) as exc:
        sys.stderr.write(f"Invalid experiment condition: {exc}\n")
        return 2
    _maybe_load_dotenv(bool(args.dotenv))
    _configure_logging(verbose=bool(args.verbose))

    bundle = Path(args.bundle).resolve()
    if not bundle.is_dir():
        sys.stderr.write(f"--bundle not found: {bundle}\n")
        return 2

    if (bundle / "participant_catalogue.json").is_file():
        try:
            return _run_portfolio_release(args)
        except (OSError, RuntimeError, ValueError, ValidationError) as exc:
            sys.stderr.write(f"Portfolio run failed: {exc}\n")
            return 2

    programs = discover_programs(bundle)
    programs = _filter_programs(programs, args.programs)
    if not programs:
        sys.stderr.write("No programs matched the filter — nothing to run.\n")
        return 2
    try:
        trialdev_runner.require_fixed_phase_replay_surface(bundle)
    except (OSError, ValueError, ValidationError) as exc:
        sys.stderr.write(f"Single-asset release preflight failed: {exc}\n")
        return 2

    sys.stdout.write(f"Selected {len(programs)} program(s) for model={args.model}.\n")
    output_root = Path(args.output_root) / _slugify_model(args.model)
    output_root.mkdir(parents=True, exist_ok=True)

    seed_variants = max(1, int(args.seed_variants))
    if args.append_run_dir is not None and seed_variants != 1:
        sys.stderr.write("--append-run-dir requires --seed-variants 1.\n")
        return 2
    user_label = args.label or None
    for k in range(seed_variants):
        master_seed = int(args.master_seed) + k
        if seed_variants == 1:
            label = user_label
        else:
            variant_tag = f"variant{k}_seed{master_seed}"
            label = f"{user_label}_{variant_tag}" if user_label else variant_tag
        sys.stdout.write(f"\n=== run with master_seed={master_seed} ===\n")
        run_one_master_seed(
            args=args,
            programs=programs,
            master_seed=master_seed,
            output_root=output_root,
            label=label,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
