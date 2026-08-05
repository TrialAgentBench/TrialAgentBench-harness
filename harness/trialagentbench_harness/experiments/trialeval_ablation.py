"""Execute a precommitted TrialEval prompt or interface ablation schedule."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import threading
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic.types import JsonValue

from trialagentbench_harness.adapters import ProviderRouting, get_provider
from trialagentbench_harness.adapters.docker_code_execution import resolve_executor_environment
from trialagentbench_harness.contracts.core.config import DecodingConfigV1, RoutingConfigV1
from trialagentbench_harness.contracts.core.runs import (
    ProviderTelemetrySummaryV1,
    RunCoverageV1,
    TrialEvalAblationItemResultV1,
    TrialEvalAblationRunConfigV1,
    TrialEvalAgentOutputV1,
)
from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationAssignmentV1,
    TrialEvalAblationScheduleV1,
    TrialEvalSchemaAffordanceInventoryV1,
)
from trialagentbench_harness.contracts.trace.observable import BenchmarkRuntimeTraceEventV1
from trialagentbench_harness.execution_policy import (
    TRIALEVAL_DEFAULT_WORKERS,
    TRIALEVAL_RELEASE_BUDGET_V1,
)
from trialagentbench_harness.io import (
    read_json_model,
    sha256_dir_digest,
    write_json_model,
)
from trialagentbench_harness.trialeval.agent import DEFAULT_MAX_TURNS, run_agent
from trialagentbench_harness.trialeval.conditions import (
    condition_contrasts_markdown_v1,
    prompt_set_sha256_v1,
    schema_affordance_inventory_v1,
)
from trialagentbench_harness.trialeval.data import (
    discover_participant_items,
    participant_analysis_surface_sha256,
)
from trialagentbench_harness.trialeval.grade_submission import (
    grade_trialeval_submission_v1,
)
from trialagentbench_harness.util.provider_environment import load_provider_dotenv
from trialagentbench_harness.util.provider_telemetry import summarize_provider_telemetry_v1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-dir", required=True, help="Extracted TrialEval public participant root.")
    parser.add_argument("--schedule", required=True, help="Precommitted ablation schedule JSON.")
    parser.add_argument("--model", required=True, help="Exact agent model identifier.")
    parser.add_argument("--provider", required=True, choices=("openai", "openai_responses", "openrouter"))
    parser.add_argument("--output-dir", required=True, help="Experiment output directory.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete output directory after validating its frozen run configuration.",
    )
    parser.add_argument("--turns", type=int, help="Maximum agent turns.")
    parser.add_argument(
        "--item-watchdog-seconds",
        type=int,
        default=TRIALEVAL_RELEASE_BUDGET_V1.wall_time_limit_seconds,
        help="Per-assignment wall-clock budget across all turns and tools (default: 3600 seconds).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=TRIALEVAL_DEFAULT_WORKERS,
        help="Concurrent assignment executions.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=TRIALEVAL_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn,
    )
    parser.add_argument(
        "--max-context-characters",
        type=int,
        default=TRIALEVAL_RELEASE_BUDGET_V1.maximum_context_characters,
        help="Maximum retained conversation size before deterministic context compaction.",
    )
    parser.add_argument(
        "--openrouter-provider",
        help="Required exact upstream-provider pin when --provider=openrouter.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=TRIALEVAL_RELEASE_BUDGET_V1.provider_request_timeout_seconds,
        help="Per-attempt provider request timeout (default: 300 seconds).",
    )
    parser.add_argument("--omit-temperature", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dotenv", action="store_true")
    return parser


def _same_run_configuration(
    existing: TrialEvalAblationRunConfigV1,
    proposed: TrialEvalAblationRunConfigV1,
) -> bool:
    """Return whether two run configurations differ only in invocation time."""

    excluded = {"timestamp_utc", "workers"}
    return existing.model_dump(exclude=excluded) == proposed.model_dump(exclude=excluded)


def _source_digest(function: Callable[..., object]) -> str:
    source = inspect.getsourcefile(function)
    if source is None:
        raise RuntimeError(f"Cannot resolve source identity for {function!r}.")
    return sha256_dir_digest(Path(source).parent) if Path(source).is_dir() else _sha256_file(Path(source))


def _sha256_file(path: Path) -> str:
    from trialagentbench_harness.io import sha256_file

    return sha256_file(path)


def _write_model_exclusive(path: Path, model: TrialEvalAblationItemResultV1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


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


def _archive_partial_assignment(output: Path, assignment_id: str) -> None:
    """Preserve incomplete artifacts before retrying one assignment."""

    candidates = (
        output / "logs" / f"{assignment_id}_provider_responses.jsonl",
        output / "traces" / f"{assignment_id}.json",
        output / "events" / f"{assignment_id}_events.jsonl",
        output / "traces" / f"{assignment_id}_workspace",
    )
    present = tuple(path for path in candidates if path.exists())
    if not present:
        return
    attempt_root = output / "failed_attempts" / assignment_id
    attempt_index = 1
    while (attempt_root / f"attempt-{attempt_index}").exists():
        attempt_index += 1
    archive = attempt_root / f"attempt-{attempt_index}"
    archive.mkdir(parents=True)
    for path in present:
        if path.is_dir():
            path.rename(archive / f"{path.name}.archived")
        else:
            suffix = ".jsonl" if path.suffix == ".jsonl" else ".json"
            path.rename(archive / f"{path.stem}.archived{suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one immutable participant-only ablation schedule."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.workers != TRIALEVAL_DEFAULT_WORKERS:
        raise ValueError(
            "TrialEval ablations require --workers 1 so execution follows the "
            "precommitted randomized assignment order."
        )
    if args.turns is not None and args.turns < 1:
        raise ValueError("--turns must be at least 1")
    if args.item_watchdog_seconds < 1:
        raise ValueError("--item-watchdog-seconds must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if args.max_context_characters < 1:
        raise ValueError("--max-context-characters must be at least 1")
    max_turns = int(args.turns) if args.turns is not None else DEFAULT_MAX_TURNS
    if args.dotenv:
        load_provider_dotenv()

    participant_root = Path(args.participant_dir)
    schedule = read_json_model(TrialEvalAblationScheduleV1, Path(args.schedule))
    observed_release_hash = sha256_dir_digest(participant_root)
    if observed_release_hash != schedule.participant_release_sha256:
        raise ValueError("Participant release hash does not match the precommitted ablation schedule.")
    observed_prompt_hash = prompt_set_sha256_v1()
    if observed_prompt_hash != schedule.prompt_set_sha256:
        raise ValueError("Installed prompt set does not match the precommitted ablation schedule.")

    task_ids = tuple(sorted({assignment.task_id for assignment in schedule.assignments}))
    items = discover_participant_items(participant_root, task_ids=task_ids)
    timestamp = datetime.now(UTC)
    executor = resolve_executor_environment()
    config_payload: dict[str, JsonValue] = {
        "schema_id": "trialagentbench.trialeval_ablation_run/v1",
        "timestamp_utc": timestamp.isoformat(),
        "experiment_id": schedule.experiment_id,
        "schedule_checksum": cast(str, schedule.checksum),
        "participant_release_sha256": observed_release_hash,
        "prompt_set_sha256": observed_prompt_hash,
        "scorer_source_sha256": _source_digest(grade_trialeval_submission_v1),
        "agent_source_sha256": _source_digest(run_agent),
        "model": args.model,
        "max_turns": max_turns,
        "max_context_characters": args.max_context_characters,
        "item_watchdog_seconds": args.item_watchdog_seconds,
        "decoding": DecodingConfigV1(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            send_temperature=not args.omit_temperature,
        ).model_dump(mode="json"),
        "routing": RoutingConfigV1(
            provider=args.provider,
            openrouter_provider=args.openrouter_provider,
            request_timeout_seconds=args.request_timeout_seconds,
        ).model_dump(mode="json"),
        "executor": executor.model_dump(mode="json"),
        "workers": args.workers,
        "n_assignments": len(schedule.assignments),
    }
    run_config = TrialEvalAblationRunConfigV1.create(**config_payload)
    output = Path(args.output_dir).resolve()
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError(f"Cannot resume missing ablation output directory: {output}")
        if (output / "provider_telemetry_summary.json").exists():
            read_json_model(ProviderTelemetrySummaryV1, output / "provider_telemetry_summary.json")
            raise ValueError("Cannot resume a completed TrialEval ablation run.")
        persisted_schedule = read_json_model(TrialEvalAblationScheduleV1, output / "schedule.json")
        if persisted_schedule != schedule:
            raise ValueError("Resume schedule does not match the persisted ablation schedule.")
        persisted_config = read_json_model(TrialEvalAblationRunConfigV1, output / "run_config.json")
        if not _same_run_configuration(persisted_config, run_config):
            raise ValueError("Resume arguments do not match the persisted ablation run configuration.")
        if (
            read_json_model(
                TrialEvalSchemaAffordanceInventoryV1,
                output / "schema_affordance_inventory.json",
            )
            != schema_affordance_inventory_v1()
        ):
            raise ValueError("Persisted schema-affordance inventory does not match the current prompt authority.")
        if (output / "condition_contrasts.md").read_text(encoding="utf-8") != condition_contrasts_markdown_v1():
            raise ValueError("Persisted condition contrasts do not match the current prompt authority.")
        run_config = persisted_config
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json_model(output / "schedule.json", schedule)
        write_json_model(output / "run_config.json", run_config)
        write_json_model(output / "schema_affordance_inventory.json", schema_affordance_inventory_v1())
        (output / "condition_contrasts.md").write_text(
            condition_contrasts_markdown_v1(),
            encoding="utf-8",
        )

    assignment_ids = tuple(assignment.assignment_id for assignment in schedule.assignments)
    coverage_path = output / "coverage.json"
    if args.resume:
        coverage = read_json_model(RunCoverageV1, coverage_path)
        if (
            coverage.run_identity_sha256 != run_config.run_identity_sha256
            or coverage.schedule_sha256 != schedule.checksum
            or coverage.unit_ids != assignment_ids
        ):
            raise ValueError("Persisted ablation coverage conflicts with its prospective schedule.")
    else:
        coverage = RunCoverageV1(
            run_identity_sha256=run_config.run_identity_sha256,
            schedule_sha256=cast(str, schedule.checksum),
            unit_ids=assignment_ids,
        )
        _write_coverage_atomic(coverage_path, coverage)

    existing_results: dict[str, TrialEvalAblationItemResultV1] = {}
    for path in sorted((output / "assignments").glob("*.json")):
        result = read_json_model(TrialEvalAblationItemResultV1, path)
        assignment_id = result.assignment.assignment_id
        if path.stem != assignment_id:
            raise ValueError(f"Persisted ablation result identity mismatch: {path}")
        if result.run_config != run_config:
            raise ValueError(f"Persisted ablation result run configuration drift: {assignment_id!r}.")
        if assignment_id in existing_results:
            raise ValueError(f"Duplicate persisted ablation assignment: {assignment_id!r}.")
        existing_results[assignment_id] = result

    expected_assignments = {assignment.assignment_id: assignment for assignment in schedule.assignments}
    extra = sorted(set(existing_results).difference(expected_assignments))
    if extra:
        raise ValueError(f"Persisted ablation output contains unknown assignments: {extra}")
    for assignment_id, result in existing_results.items():
        if result.assignment != expected_assignments[assignment_id]:
            raise ValueError(f"Persisted ablation assignment drift: {assignment_id!r}.")
    checkpoint_ids = tuple(assignment_id for assignment_id in assignment_ids if assignment_id in existing_results)
    if any(assignment_id not in existing_results for assignment_id in coverage.completed_unit_ids):
        raise ValueError("Ablation coverage claims completion without an immutable checkpoint.")
    if coverage.completed_unit_ids != checkpoint_ids:
        coverage = coverage.model_copy(update={"completed_unit_ids": checkpoint_ids})
        _write_coverage_atomic(coverage_path, coverage)

    pending = [assignment for assignment in schedule.assignments if assignment.assignment_id not in existing_results]
    if args.resume:
        for assignment in pending:
            _archive_partial_assignment(output, assignment.assignment_id)

    lock = threading.Lock()
    completed = len(existing_results)
    completed_assignment_ids = set(existing_results)

    def process(assignment: TrialEvalAblationAssignmentV1) -> None:
        nonlocal completed
        item = items[assignment.task_id]
        if (
            item.context_tier != assignment.context_tier
            or item.data_preparation != assignment.data_preparation
            or item.analysis_specification != assignment.analysis_specification
        ):
            raise ValueError("Ablation assignment evidence factors disagree with the participant release.")
        observed_surface_sha256 = participant_analysis_surface_sha256(item)
        if observed_surface_sha256 != assignment.analysis_surface_sha256:
            raise ValueError("Projected participant analysis surface does not match its precommitted assignment.")
        provider = get_provider(
            args.model,
            routing=ProviderRouting(provider=args.provider, openrouter_provider=args.openrouter_provider),
            send_temperature=run_config.decoding.send_temperature,
            decoding_seed=assignment.decoding_seed,
            timeout_s=run_config.routing.request_timeout_seconds,
        )
        raw = cast(
            dict[str, JsonValue],
            run_agent(
                item,
                provider,
                max_turns=max_turns,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                verbose=not args.quiet and args.workers == 1,
                procedure_assistance=assignment.procedure_assistance,
                analysis_specification=assignment.analysis_specification,
                prompt_condition=assignment.prompt_condition,
                submission_interface=assignment.submission_interface,
                provider_log_path=output / "logs" / f"{assignment.assignment_id}_provider_responses.jsonl",
                conversation_log_path=output / "traces" / f"{assignment.assignment_id}.json",
                event_log_path=output / "events" / f"{assignment.assignment_id}_events.jsonl",
                executor_image=run_config.executor.image_id,
                executor_limits=run_config.executor.limits,
                max_elapsed_seconds=float(run_config.item_watchdog_seconds),
                max_context_chars=int(run_config.max_context_characters),
            ),
        )
        conversation = raw.pop("conversation", [])
        events_payload = raw.pop("events", [])
        if not isinstance(events_payload, list):
            raise ValueError("TrialEval runtime events must be a JSON array.")
        events = [BenchmarkRuntimeTraceEventV1.model_validate(event) for event in events_payload]
        agent_output = TrialEvalAgentOutputV1.model_validate(raw)

        result = TrialEvalAblationItemResultV1(
            timestamp_utc=datetime.now(UTC),
            assignment=assignment,
            run_config=run_config,
            agent_output=agent_output,
        )
        _write_model_exclusive(output / "assignments" / f"{assignment.assignment_id}.json", result)
        if conversation:
            trace_path = output / "traces" / f"{assignment.assignment_id}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps(conversation, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if events:
            event_path = output / "events" / f"{assignment.assignment_id}_events.jsonl"
            event_path.parent.mkdir(parents=True, exist_ok=True)
            event_path.write_text(
                "".join(json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n" for event in events),
                encoding="utf-8",
            )
        with lock:
            completed += 1
            completed_assignment_ids.add(assignment.assignment_id)
            completed_in_order = tuple(
                assignment_id for assignment_id in assignment_ids if assignment_id in completed_assignment_ids
            )
            _write_coverage_atomic(
                coverage_path,
                coverage.model_copy(update={"completed_unit_ids": completed_in_order}),
            )
            if not args.quiet:
                print(f"[DONE {completed}/{len(schedule.assignments)}] {assignment.assignment_id}")

    for assignment in pending:
        process(assignment)

    summarize_provider_telemetry_v1(
        run_root=output,
        coverage=read_json_model(RunCoverageV1, coverage_path),
    )

    print(f"Ablation artifacts saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
