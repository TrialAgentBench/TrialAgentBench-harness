"""Grade frozen TrialEval ablation responses with the canonical scorer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from trialagentbench_harness.analysis.experiments.trialeval_endpoint_scoring import (
    score_trialeval_ablation_submission_v1,
    trialeval_scoring_implementation_sha256_v1,
)
from trialagentbench_harness.analysis.experiments.trialeval_transcription import (
    validate_narrative_transcription_v1,
)
from trialagentbench_harness.contracts.core.runs import (
    ProviderTelemetrySummaryV1,
    RunCoverageV1,
    TrialEvalAblationItemResultV1,
    TrialEvalAblationRunConfigV1,
)
from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationEndpointRowV1,
    TrialEvalAblationEndpointSetV1,
    TrialEvalAblationScheduleV1,
    TrialEvalNarrativeTranscriptionV1,
)
from trialagentbench_harness.contracts.scoring.assumption_evidence import (
    read_assumption_evidence_domains,
)
from trialagentbench_harness.grading import ScoringKeyStoreV1
from trialagentbench_harness.io import read_json_model, sha256_path, write_json_model
from trialagentbench_harness.trialeval.data import discover_items


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Immutable TrialEval ablation run directory.")
    parser.add_argument("--suite-dir", required=True, help="Exact evaluator release used for grading.")
    parser.add_argument(
        "--transcriptions-dir",
        help="Narrative transcriptions root containing manual/ and optional automated/ JSON files.",
    )
    parser.add_argument("--out", required=True, help="New scored endpoint-set JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Grade every scheduled assignment without model or importer calls."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    run_dir = Path(args.run_dir)
    suite_dir = Path(args.suite_dir)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite ablation endpoint set: {output}")
    schedule = read_json_model(TrialEvalAblationScheduleV1, run_dir / "schedule.json")
    run_config = read_json_model(TrialEvalAblationRunConfigV1, run_dir / "run_config.json")
    if run_config.schedule_checksum != schedule.checksum:
        raise ValueError("Ablation run configuration does not match its schedule.")
    if run_config.participant_release_sha256 != schedule.participant_release_sha256:
        raise ValueError("Ablation run participant release does not match its schedule.")
    public_release = suite_dir / "public"
    if not public_release.is_dir():
        raise FileNotFoundError(f"Evaluator release lacks its paired public surface: {public_release}")
    if sha256_path(public_release) != schedule.participant_release_sha256:
        raise ValueError("Evaluator public surface does not match the participant release used for the run.")

    assignment_ids = tuple(assignment.assignment_id for assignment in schedule.assignments)
    coverage = read_json_model(RunCoverageV1, run_dir / "coverage.json")
    if (
        coverage.run_identity_sha256 != run_config.run_identity_sha256
        or coverage.schedule_sha256 != schedule.checksum
        or coverage.unit_ids != assignment_ids
        or coverage.completed_unit_ids != assignment_ids
    ):
        raise ValueError("Ablation run coverage is incomplete or inconsistent.")
    telemetry = read_json_model(ProviderTelemetrySummaryV1, run_dir / "provider_telemetry_summary.json")
    if (
        telemetry.run_identity_sha256 != coverage.run_identity_sha256
        or telemetry.schedule_sha256 != coverage.schedule_sha256
        or telemetry.unit_ids != coverage.unit_ids
        or telemetry.completed_unit_ids != coverage.completed_unit_ids
    ):
        raise ValueError("Ablation provider telemetry is not bound to the completed run denominator.")

    discovered = discover_items(suite_dir)
    items = {item.task_id: item for item in discovered}
    if len(items) != len(discovered):
        raise ValueError("Evaluator release contains duplicate TrialEval task IDs.")
    expected_ids = {assignment.assignment_id for assignment in schedule.assignments}
    result_paths = {path.stem: path for path in sorted((run_dir / "assignments").glob("*.json"))}
    missing = sorted(expected_ids.difference(result_paths))
    extra = sorted(set(result_paths).difference(expected_ids))
    if missing or extra:
        raise ValueError(f"Ablation run denominator mismatch: missing={missing}, extra={extra}")

    evaluator_task_ids = tuple(sorted(items))
    scoring_keys = ScoringKeyStoreV1.from_release(
        suite_dir,
        expected_item_ids=evaluator_task_ids,
    )
    assumption_evidence = read_assumption_evidence_domains(release_root=suite_dir)
    if set(assumption_evidence) != set(evaluator_task_ids):
        raise ValueError("assumption-evidence denominator must match the scheduled scoring-key denominator")
    transcription_root = Path(args.transcriptions_dir) if args.transcriptions_dir else None
    endpoints: list[TrialEvalAblationEndpointRowV1] = []
    for assignment in schedule.assignments:
        result = read_json_model(TrialEvalAblationItemResultV1, result_paths[assignment.assignment_id])
        if result.assignment != assignment:
            raise ValueError(f"Ablation result assignment drift: {assignment.assignment_id!r}.")
        if result.run_config != run_config:
            raise ValueError(f"Ablation result run configuration drift: {assignment.assignment_id!r}.")
        item = items.get(assignment.task_id)
        if item is None:
            raise ValueError(f"Evaluator release lacks scheduled task {assignment.task_id!r}.")

        if assignment.submission_interface == "structured":
            endpoints.append(
                score_trialeval_ablation_submission_v1(
                    scoring_key=scoring_keys.for_item(assignment.task_id),
                    assumption_evidence=assumption_evidence[assignment.task_id],
                    item=item,
                    result=result,
                    submission=result.agent_output.result,
                    normalization_source="direct_structured",
                    normalization_status="not_applicable",
                )
            )
            continue

        if transcription_root is None:
            raise ValueError("Narrative assignments require --transcriptions-dir.")
        report = result.agent_output.report or ""
        sources: tuple[tuple[str, Literal["manual_masked", "automated_importer"]], ...] = (
            ("manual", "manual_masked"),
            ("automated", "automated_importer"),
        )
        scored_sources = 0
        for directory, normalization_source in sources:
            path = transcription_root / directory / f"{assignment.assignment_id}.json"
            if not path.is_file():
                continue
            transcription = read_json_model(TrialEvalNarrativeTranscriptionV1, path)
            if transcription.source != normalization_source:
                raise ValueError(f"Narrative transcription source mismatch: {path}")
            validate_narrative_transcription_v1(
                transcription=transcription,
                frozen_report=report,
                expected_assignment_id=assignment.assignment_id,
                expected_task_id=assignment.task_id,
            )
            endpoints.append(
                score_trialeval_ablation_submission_v1(
                    scoring_key=scoring_keys.for_item(assignment.task_id),
                    assumption_evidence=assumption_evidence[assignment.task_id],
                    item=item,
                    result=result,
                    submission=transcription.submission,
                    normalization_source=normalization_source,
                    normalization_status=transcription.status,
                    normalization_failure_reason=transcription.abstention_reason,
                )
            )
            scored_sources += 1
        if scored_sources == 0:
            raise ValueError(f"Narrative assignment lacks any frozen transcription: {assignment.assignment_id!r}.")

    endpoint_set = TrialEvalAblationEndpointSetV1(
        experiment_id=schedule.experiment_id,
        schedule_checksum=cast(str, schedule.checksum),
        evaluator_release_sha256=sha256_path(suite_dir),
        scoring_implementation_sha256=trialeval_scoring_implementation_sha256_v1(),
        endpoints=tuple(endpoints),
    )
    write_json_model(output, endpoint_set)
    print(f"Graded {len(endpoints)} TrialEval ablation endpoints: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
