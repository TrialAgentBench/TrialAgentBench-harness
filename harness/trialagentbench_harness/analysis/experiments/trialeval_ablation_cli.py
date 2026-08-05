"""Build a checksummed TrialEval prompt/interface ablation analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import TypeAdapter

from trialagentbench_harness.analysis.experiments.resource_attainment import (
    trialeval_resource_attainment_v1,
)
from trialagentbench_harness.analysis.experiments.trialeval_endpoint_rows import (
    join_ablation_evaluator_labels_v1,
)
from trialagentbench_harness.analysis.experiments.trialeval_inference import (
    analysis_specification_contrasts_v1,
    factorial_ablation_arm_summaries_v1,
    factorial_ablation_contrasts_v1,
    factorial_observable_contrasts_v1,
    targeted_control_contrasts_v1,
    trialeval_route_multiplicity_summaries_v1,
    trialeval_tolerance_sensitivity_summaries_v1,
)
from trialagentbench_harness.analysis.experiments.trialeval_transcription import (
    evaluate_representation_fixtures_v1,
)
from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationAnalysisConfigV1,
    TrialEvalAblationAnalysisReportV1,
    TrialEvalAblationEndpointRowV1,
    TrialEvalAblationEndpointSetV1,
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalAblationScheduleV1,
    TrialEvalRepresentationFidelityRowV1,
    TrialEvalRepresentationFixtureV1,
)
from trialagentbench_harness.io import read_json, read_json_model, sha256_path, write_json_model
from trialagentbench_harness.trialeval.action_trace import collect_trialeval_ablation_observables


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, help="Original checksummed ablation schedule JSON.")
    parser.add_argument("--endpoint-set", required=True, help="Checksummed scored ablation endpoint set JSON.")
    parser.add_argument("--evaluator-labels", required=True, help="Evaluator-only task identity and labels JSON.")
    parser.add_argument("--evaluator-release", required=True, help="Exact evaluator release used for scoring.")
    parser.add_argument("--analysis-config", required=True, help="Prospective ablation analysis configuration JSON.")
    parser.add_argument(
        "--run-dir",
        required=True,
        action="append",
        help="Frozen ablation run directory; repeat for each model.",
    )
    parser.add_argument(
        "--design",
        required=True,
        choices=("factorial_interface", "targeted_control"),
        help="Precommitted ablation design.",
    )
    parser.add_argument("--representation-fixtures", help="Optional JSON array of fixed-answer fixtures.")
    parser.add_argument("--out", required=True, help="New analysis report JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate frozen inputs and emit paired, cluster-aware ablation evidence."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite ablation analysis: {output}")

    endpoint_set = read_json_model(TrialEvalAblationEndpointSetV1, Path(args.endpoint_set))
    schedule = read_json_model(TrialEvalAblationScheduleV1, Path(args.schedule))
    evaluator_labels = read_json_model(TrialEvalAblationEvaluatorLabelsV1, Path(args.evaluator_labels))
    config = read_json_model(TrialEvalAblationAnalysisConfigV1, Path(args.analysis_config))
    if config.execution_scope == "canary":
        raise ValueError(
            "Canary schedules qualify runtime execution only and cannot produce inferential analysis reports."
        )
    evaluator_hash = sha256_path(Path(args.evaluator_release))
    if endpoint_set.evaluator_release_sha256 != evaluator_hash:
        raise ValueError("Endpoint set is not bound to the supplied evaluator release.")
    if evaluator_labels.evaluator_release_sha256 != evaluator_hash:
        raise ValueError("Applicability labels are not bound to the supplied evaluator release.")

    design = cast(Literal["factorial_interface", "targeted_control"], args.design)
    if schedule.design != design:
        raise ValueError("Requested analysis design does not match the precommitted schedule.")
    if config.design != design:
        raise ValueError("Prospective analysis contract design does not match the requested analysis.")
    if schedule.analysis_config_sha256 != config.checksum:
        raise ValueError("Prospective analysis contract does not match the checksum frozen in the schedule.")
    if endpoint_set.experiment_id != schedule.experiment_id or endpoint_set.schedule_checksum != schedule.checksum:
        raise ValueError("Endpoint set does not match the supplied ablation schedule.")
    _validate_endpoint_denominator(endpoint_set=endpoint_set, schedule=schedule)
    grade_rows = join_ablation_evaluator_labels_v1(
        endpoints=endpoint_set.endpoints,
        evaluator_labels=evaluator_labels,
        design=design,
    )
    if design == "factorial_interface":
        arm_summaries = factorial_ablation_arm_summaries_v1(rows=grade_rows, config=config)
        contrasts = factorial_ablation_contrasts_v1(
            rows=grade_rows,
            config=config,
        ) + analysis_specification_contrasts_v1(rows=grade_rows, config=config)
        observable_rows = collect_trialeval_ablation_observables([Path(path) for path in args.run_dir])
        observable_contrasts = factorial_observable_contrasts_v1(
            rows=observable_rows,
            evaluator_labels=evaluator_labels,
            config=config,
        )
    else:
        arm_summaries = ()
        contrasts = targeted_control_contrasts_v1(rows=grade_rows, config=config)
        observable_rows = collect_trialeval_ablation_observables([Path(path) for path in args.run_dir])
        observable_contrasts = ()

    representation_rows: tuple[TrialEvalRepresentationFidelityRowV1, ...] = ()
    if args.representation_fixtures:
        fixture_payload = read_json(Path(args.representation_fixtures))
        fixtures = TypeAdapter(tuple[TrialEvalRepresentationFixtureV1, ...]).validate_python(fixture_payload)
        representation_rows = evaluate_representation_fixtures_v1(fixtures)

    report = TrialEvalAblationAnalysisReportV1(
        experiment_id=endpoint_set.experiment_id,
        design=design,
        schedule_checksum=cast(str, schedule.checksum),
        endpoint_set_checksum=cast(str, endpoint_set.checksum),
        evaluator_labels_checksum=cast(str, evaluator_labels.checksum),
        analysis_config=config,
        grade_rows=grade_rows,
        arm_summaries=arm_summaries,
        contrasts=contrasts,
        observable_rows=observable_rows,
        observable_contrasts=observable_contrasts,
        resource_attainment=trialeval_resource_attainment_v1(
            grades=grade_rows,
            observables=observable_rows,
        ),
        route_multiplicity=trialeval_route_multiplicity_summaries_v1(grade_rows),
        tolerance_sensitivity=trialeval_tolerance_sensitivity_summaries_v1(grade_rows),
        representation_fidelity=representation_rows,
    )
    write_json_model(output, report)
    print(f"Wrote TrialEval ablation analysis: {output}")
    return 0


def _validate_endpoint_denominator(
    *,
    endpoint_set: TrialEvalAblationEndpointSetV1,
    schedule: TrialEvalAblationScheduleV1,
) -> None:
    """Require one official endpoint for every scheduled assignment."""

    assignments = {row.assignment_id: row for row in schedule.assignments}
    observed_ids = {row.assignment_id for row in endpoint_set.endpoints}
    missing = sorted(set(assignments).difference(observed_ids))
    extra = sorted(observed_ids.difference(assignments))
    if missing or extra:
        raise ValueError(f"Ablation endpoint denominator mismatch: missing={missing}, extra={extra}")

    by_assignment: dict[str, list[TrialEvalAblationEndpointRowV1]] = {}
    for endpoint in endpoint_set.endpoints:
        assignment = assignments[endpoint.assignment_id]
        if (
            endpoint.task_id != assignment.task_id
            or endpoint.context_tier != assignment.context_tier
            or endpoint.analysis_specification != assignment.analysis_specification
            or endpoint.replicate_id != assignment.replicate_id
            or endpoint.procedure_assistance != assignment.procedure_assistance
            or endpoint.prompt_condition != assignment.prompt_condition
            or endpoint.submission_interface != assignment.submission_interface
        ):
            raise ValueError(f"Ablation endpoint treatment provenance drift: {endpoint.assignment_id!r}.")
        by_assignment.setdefault(endpoint.assignment_id, []).append(endpoint)

    narrative_assignment_ids = {
        assignment_id
        for assignment_id, assignment in assignments.items()
        if assignment.submission_interface == "narrative"
    }
    automated_assignment_ids = {
        assignment_id
        for assignment_id, rows in by_assignment.items()
        if any(row.normalization_source == "automated_importer" for row in rows)
    }
    if automated_assignment_ids and automated_assignment_ids != narrative_assignment_ids:
        raise ValueError("Automated narrative normalization must cover every scheduled narrative assignment.")

    for assignment_id, assignment in assignments.items():
        sources = {row.normalization_source for row in by_assignment[assignment_id]}
        if len(sources) != len(by_assignment[assignment_id]):
            raise ValueError(f"Ablation assignment {assignment_id!r} repeats a normalization source.")
        required = {"direct_structured"} if assignment.submission_interface == "structured" else {"manual_masked"}
        allowed = required if assignment.submission_interface == "structured" else required | {"automated_importer"}
        if not required.issubset(sources) or not sources.issubset(allowed):
            raise ValueError(
                f"Ablation assignment {assignment_id!r} has invalid endpoint sources: {sorted(sources)!r}."
            )


if __name__ == "__main__":
    raise SystemExit(main())
