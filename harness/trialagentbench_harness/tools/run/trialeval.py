"""Run TrialEvalBench against a benchmark release surface."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import threading
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime  # type: ignore[attr-defined]
from pathlib import Path
from typing import cast

from pydantic.types import JsonValue

from trialagentbench_harness.adapters import ProviderRouting, get_provider
from trialagentbench_harness.adapters.docker_code_execution import resolve_executor_environment
from trialagentbench_harness.contracts.core.config import (
    DecodingConfigV1,
    ReasoningEffortV1,
    RoutingConfigV1,
)
from trialagentbench_harness.contracts.core.runs import (
    ProviderTelemetrySummaryV1,
    RunCoverageV1,
    TrialEvalAgentOutputV1,
    TrialEvalItemResultV1,
    TrialEvalRunConfigV1,
)
from trialagentbench_harness.contracts.core.trialeval_factors import (
    TrialEvalAnalysisSpecificationV1,
)
from trialagentbench_harness.execution_policy import (
    TRIALEVAL_DEFAULT_WORKERS,
    TRIALEVAL_RELEASE_BUDGET_V1,
)
from trialagentbench_harness.io import canonical_payload_sha256, read_json_model, sha256_path
from trialagentbench_harness.io.json import write_json_model
from trialagentbench_harness.trialeval.agent import DEFAULT_MAX_TURNS, run_agent
from trialagentbench_harness.trialeval.conditions import prompt_set_sha256_v1
from trialagentbench_harness.trialeval.data import discover_participant_items, participant_task_factors
from trialagentbench_harness.trialeval.grade_submission import (
    grade_trialeval_submission_v1,
)
from trialagentbench_harness.trialeval.schema import BenchmarkItem
from trialagentbench_harness.util.experiment_condition import (
    resolve_experiment_condition_v1,
)
from trialagentbench_harness.util.provider_environment import load_provider_dotenv
from trialagentbench_harness.util.provider_telemetry import summarize_provider_telemetry_v1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-dir", required=True, help="Unpacked TrialEvalBench participant release.")
    parser.add_argument(
        "--task-id",
        nargs="+",
        help="Optional exact opaque task IDs. Omit to run the complete participant release.",
    )
    parser.add_argument("--model", required=True, help="Agent model identifier.")
    parser.add_argument(
        "--provider",
        required=True,
        choices=("openai", "openai_responses", "openrouter"),
        help=(
            "Exact API transport: openai uses Chat Completions, openai_responses "
            "uses Responses, and openrouter uses pinned OpenRouter Chat Completions."
        ),
    )
    parser.add_argument("--turns", type=int, help="Maximum agent turns; context defaults apply when omitted.")
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
        "--item-watchdog-seconds",
        type=int,
        default=TRIALEVAL_RELEASE_BUDGET_V1.wall_time_limit_seconds,
        help="Per-item wall-clock budget across all turns and tools (default: 3600 seconds).",
    )
    parser.add_argument("--output-dir", default="results", help="Parent directory for the timestamped run.")
    parser.add_argument(
        "--resume-run-dir",
        "--append-run-dir",
        dest="resume_run_dir",
        help="Resume pending items in an identity-matched run with a frozen prospective schedule.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=TRIALEVAL_DEFAULT_WORKERS,
        help="Concurrent item executions.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress turn-level output.")
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
    parser.add_argument("--dotenv", action="store_true", help="Load provider credentials from .env.")
    return parser


def _save_item_result(
    *,
    output_dir: Path,
    item_id: str,
    agent_output: dict[str, JsonValue],
    run_config: TrialEvalRunConfigV1,
) -> None:
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    agent = TrialEvalAgentOutputV1(
        status=str(agent_output.get("status") or ""),
        turns_used=_integer(agent_output.get("turns_used")),
        report=str(agent_output.get("report")) if agent_output.get("report") is not None else None,
        result=agent_output.get("result"),
        condition_provenance=agent_output.get("condition_provenance"),
    )
    result = TrialEvalItemResultV1(
        item_id=item_id,
        timestamp_utc=datetime.now(UTC),
        run_config=run_config,
        agent_output=agent,
    )
    _write_json_model_exclusive(items_dir / f"{item_id}.json", result)

    conversation = agent_output.get("conversation")
    if isinstance(conversation, list) and conversation:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / f"{item_id}_conversation.json").write_text(
            json.dumps(conversation, indent=2, default=str, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    events = agent_output.get("events")
    if isinstance(events, list) and events:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / f"{item_id}_events.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )


def _write_json_model_exclusive(path: Path, model: TrialEvalItemResultV1) -> None:
    """Durably create one immutable checkpoint without overwrite races."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_coverage_atomic(path: Path, coverage: RunCoverageV1) -> None:
    """Atomically replace the derived coverage view after validating its contract."""

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


def _source_digest(function: Callable[..., object]) -> str:
    source = inspect.getsourcefile(function)
    if source is None:
        raise RuntimeError(f"Cannot resolve source identity for {function!r}.")
    return str(sha256_path(Path(source)))


def _archive_partial_item(output_dir: Path, item_id: str) -> None:
    """Move raw incomplete-attempt logs aside without matching active telemetry discovery."""

    candidates = (
        output_dir / "logs" / f"{item_id}_provider_responses.jsonl",
        output_dir / "logs" / f"{item_id}_conversation.json",
        output_dir / "logs" / f"{item_id}_events.jsonl",
        output_dir / "logs" / f"{item_id}_workspace",
    )
    present = tuple(path for path in candidates if path.exists())
    if not present:
        return
    attempt_root = output_dir / "failed_attempts" / item_id
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


def _integer(value: JsonValue) -> int:
    """Return an explicitly persisted integer."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"Expected persisted integer, observed {type(value).__name__}.")


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a TrialEvalBench run and persist scoreable artifacts."""
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    max_turns = int(args.turns) if args.turns is not None else DEFAULT_MAX_TURNS
    if max_turns < 1:
        raise ValueError("--turns must be at least 1")
    if args.item_watchdog_seconds < 1:
        raise ValueError("--item-watchdog-seconds must be at least 1")
    if args.provider == "openai_responses" and args.decoding_seed is not None:
        raise ValueError("--decoding-seed is not supported by --provider openai_responses")
    experiment_condition = resolve_experiment_condition_v1(
        condition_id=args.condition_id,
        request_replicate_id=args.request_replicate_id,
        reasoning_effort=cast(ReasoningEffortV1 | None, args.reasoning_effort),
        reasoning_capability_snapshot=args.reasoning_capability_snapshot,
        provider=args.provider,
        model=args.model,
        openrouter_provider=args.openrouter_provider or None,
        procedure_assistance="output_contract_only",
        maximum_turns_per_step=max_turns,
        maximum_submission_attempts=None,
        tool_choice="auto",
    )
    if args.dotenv:
        load_provider_dotenv()

    participant_dir = Path(args.participant_dir).resolve()
    if not participant_dir.is_dir():
        raise FileNotFoundError(f"TrialEvalBench participant release does not exist: {participant_dir}")
    declared_task_ids, declared_factors = participant_task_factors(participant_dir)
    selected_task_ids = list(args.task_id) if args.task_id else declared_task_ids
    if len(set(selected_task_ids)) != len(selected_task_ids):
        raise ValueError("--task-id values must be unique.")
    unknown_task_ids = sorted(set(selected_task_ids) - set(declared_task_ids))
    if unknown_task_ids:
        raise ValueError(f"Requested task IDs are absent from the participant release: {unknown_task_ids}")
    item_by_id = discover_participant_items(participant_dir, task_ids=tuple(selected_task_ids))
    items = [item_by_id[task_id] for task_id in selected_task_ids]
    if not items:
        raise ValueError("The participant release contains no selected benchmark items.")

    timestamp = datetime.now(UTC)
    model_slug = args.model.replace("/", "_").replace(".", "_")
    if args.resume_run_dir:
        output_dir = Path(args.resume_run_dir).resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume missing TrialEval run directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir) / model_slug / timestamp.strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=False)
    executor = resolve_executor_environment()
    config_payload: dict[str, JsonValue] = {
        "schema_id": "trialagentbench_trialeval_run_config_v1",
        "schema_version": 1,
        "timestamp_utc": timestamp.isoformat(),
        "model": args.model,
        "output_mode": "structured",
        "item_watchdog_seconds": args.item_watchdog_seconds,
        "experiment_condition": experiment_condition.model_dump(mode="json"),
        "decoding": DecodingConfigV1(
            temperature=0.0,
            max_tokens=TRIALEVAL_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn,
            send_temperature=not args.omit_temperature,
            decoding_seed=args.decoding_seed,
        ).model_dump(mode="json"),
        "routing": RoutingConfigV1(
            provider=args.provider,
            openrouter_provider=args.openrouter_provider,
            request_timeout_seconds=args.request_timeout_seconds,
        ).model_dump(mode="json"),
        "executor": executor.model_dump(mode="json"),
        "participant_dir": str(participant_dir),
        "participant_release_sha256": sha256_path(participant_dir),
        "prompt_set_sha256": prompt_set_sha256_v1(),
        "scorer_source_sha256": _source_digest(grade_trialeval_submission_v1),
        "agent_source_sha256": _source_digest(run_agent),
        "task_evidence_factors": {
            task_id: declared_factors[task_id].model_dump(mode="json") for task_id in selected_task_ids
        },
        "prompt_condition": "neutral",
        "submission_interface": "structured",
        "task_ids": cast(list[JsonValue], selected_task_ids),
        "data_format": "trialagentbench_v1",
        "data_version": "trialagentbench_v1",
        "workers": args.workers,
        "n_items": len(items),
    }
    proposed_config = TrialEvalRunConfigV1.create(**config_payload)
    if args.resume_run_dir:
        run_config = read_json_model(TrialEvalRunConfigV1, output_dir / "run_config.json")
        if run_config.run_identity_sha256 != proposed_config.run_identity_sha256:
            raise ValueError("Resume arguments or source content conflict with the persisted TrialEval identity.")
        if run_config.task_ids != selected_task_ids:
            raise ValueError("Resume selection conflicts with the immutable TrialEval denominator.")
    else:
        run_config = proposed_config
        write_json_model(output_dir / "run_config.json", run_config)

    schedule_sha256 = canonical_payload_sha256(cast(JsonValue, run_config.task_ids))
    coverage_path = output_dir / "coverage.json"
    if args.resume_run_dir:
        coverage = read_json_model(RunCoverageV1, coverage_path)
        if (
            coverage.run_identity_sha256 != run_config.run_identity_sha256
            or coverage.schedule_sha256 != schedule_sha256
            or coverage.unit_ids != tuple(run_config.task_ids)
        ):
            raise ValueError("Persisted TrialEval coverage conflicts with the immutable run schedule.")
    else:
        coverage = RunCoverageV1(
            run_identity_sha256=run_config.run_identity_sha256,
            schedule_sha256=schedule_sha256,
            unit_ids=tuple(run_config.task_ids),
        )
        _write_coverage_atomic(coverage_path, coverage)

    existing_results: dict[str, TrialEvalItemResultV1] = {}
    for path in sorted((output_dir / "items").glob("*.json")):
        result = read_json_model(TrialEvalItemResultV1, path)
        if path.stem != result.item_id or result.item_id not in run_config.task_ids:
            raise ValueError(f"Persisted TrialEval checkpoint has an unknown identity: {path}")
        if result.run_config != run_config:
            raise ValueError(f"Persisted TrialEval checkpoint has run identity drift: {result.item_id}")
        if result.item_id in existing_results:
            raise ValueError(f"Duplicate TrialEval checkpoint identity: {result.item_id}")
        existing_results[result.item_id] = result
    checkpoint_ids = tuple(task_id for task_id in run_config.task_ids if task_id in existing_results)
    if any(task_id not in existing_results for task_id in coverage.completed_unit_ids):
        raise ValueError("TrialEval coverage claims completion without an immutable item checkpoint.")
    if coverage.completed_unit_ids != checkpoint_ids:
        coverage = coverage.model_copy(update={"completed_unit_ids": checkpoint_ids})
        _write_coverage_atomic(coverage_path, coverage)
    if (output_dir / "provider_telemetry_summary.json").exists():
        telemetry = read_json_model(ProviderTelemetrySummaryV1, output_dir / "provider_telemetry_summary.json")
        if len(existing_results) != len(run_config.task_ids):
            raise ValueError("Completed TrialEval marker exists with incomplete schedule coverage.")
        if (
            telemetry.run_identity_sha256 != coverage.run_identity_sha256
            or telemetry.schedule_sha256 != coverage.schedule_sha256
            or telemetry.unit_ids != coverage.unit_ids
            or telemetry.completed_unit_ids != coverage.completed_unit_ids
        ):
            raise ValueError("Completed TrialEval telemetry does not match its run denominator.")
        print(f"Run already complete: {output_dir}")
        return 0

    routing = ProviderRouting(provider=args.provider, openrouter_provider=args.openrouter_provider)
    provider = get_provider(
        args.model,
        routing=routing,
        send_temperature=run_config.decoding.send_temperature,
        decoding_seed=run_config.decoding.decoding_seed,
        reasoning_effort=run_config.experiment_condition.reasoning.effort,
        exclude_reasoning=run_config.experiment_condition.reasoning.exclude_from_response,
        timeout_s=run_config.routing.request_timeout_seconds,
    )
    work: list[tuple[int, BenchmarkItem]] = []
    for item in items:
        if item.item_id not in existing_results:
            if args.resume_run_dir:
                _archive_partial_item(output_dir, item.item_id)
            work.append((len(work), item))

    lock = threading.Lock()
    completed_indices: set[int] = set()
    completed_item_ids: set[str] = set(existing_results)

    def process(index: int, item: BenchmarkItem) -> None:
        agent_output = cast(
            dict[str, JsonValue],
            run_agent(
                item,
                provider,
                max_turns=run_config.experiment_condition.maximum_turns_per_step,
                temperature=float(run_config.decoding.temperature),
                max_tokens=int(run_config.decoding.max_tokens),
                verbose=not args.quiet and args.workers == 1,
                provider_log_path=output_dir / "logs" / f"{item.item_id}_provider_responses.jsonl",
                conversation_log_path=output_dir / "logs" / f"{item.item_id}_conversation.json",
                event_log_path=output_dir / "logs" / f"{item.item_id}_events.jsonl",
                executor_image=run_config.executor.image_id,
                executor_limits=run_config.executor.limits,
                max_elapsed_seconds=float(run_config.item_watchdog_seconds),
                procedure_assistance=run_config.experiment_condition.procedure_assistance,
                analysis_specification=cast(
                    TrialEvalAnalysisSpecificationV1,
                    item.analysis_specification,
                ),
                prompt_condition=run_config.prompt_condition,
                submission_interface=run_config.submission_interface,
            ),
        )

        _save_item_result(
            output_dir=output_dir,
            item_id=item.item_id,
            agent_output=agent_output,
            run_config=run_config,
        )
        with lock:
            completed_indices.add(index)
            completed_item_ids.add(item.item_id)
            completed_in_order = tuple(task_id for task_id in run_config.task_ids if task_id in completed_item_ids)
            _write_coverage_atomic(
                coverage_path,
                coverage.model_copy(update={"completed_unit_ids": completed_in_order}),
            )
            if not args.quiet:
                print(f"[DONE {len(completed_item_ids)}/{len(run_config.task_ids)}] {item.item_id}")

    if args.workers == 1:
        for row in work:
            process(*row)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process, *row) for row in work]
            for future in as_completed(futures):
                future.result()

    summarize_provider_telemetry_v1(
        run_root=output_dir,
        coverage=read_json_model(RunCoverageV1, coverage_path),
    )
    print(f"Run artifacts saved to {output_dir}")
    print(f"Completed items: {len(completed_item_ids)}/{len(run_config.task_ids)}")
    print("Grade with trialagentbench grade trialeval and the paired evaluator release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
