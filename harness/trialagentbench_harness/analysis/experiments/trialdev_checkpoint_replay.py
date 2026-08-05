"""Analyse matched TrialDev checkpoint continuations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Literal

from trialagentbench_harness.analysis.experiments.crossed_bootstrap import (
    crossed_cluster_bootstrap_mean,
)
from trialagentbench_harness.contracts.core.runs import TrialDevChainSummaryV1
from trialagentbench_harness.contracts.experiments import (
    TrialDevCheckpointAnalysisV1,
    TrialDevCheckpointAssignmentV1,
    TrialDevCheckpointContrastIdV1,
    TrialDevCheckpointContrastV1,
    TrialDevCheckpointDescriptiveV1,
    TrialDevCheckpointMetricV1,
    TrialDevCheckpointObservedContrastV1,
    TrialDevCheckpointQualityV1,
    TrialDevCheckpointScheduleV1,
    TrialDevCheckpointScoreRowV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevGradeRecordV1,
    TrialDevTrajectoryGradeV1,
)
from trialagentbench_harness.io import read_json_model, write_json_model
from trialagentbench_harness.numeric_policy import (
    PUBLICATION_BOOTSTRAP_REPLICATES_V1,
    PUBLICATION_CONFIDENCE_LEVEL_V1,
    TRIALDEV_CHECKPOINT_ANALYSIS_SEED_V1,
)

_METRICS: tuple[TrialDevCheckpointMetricV1, ...] = (
    "checkpoint_primary_score",
    "checkpoint_decision_correct",
    "downstream_primary_score",
    "downstream_decision_score",
    "checkpoint_design_validity",
    "checkpoint_phase_evaluation_validity",
    "checkpoint_primary_effect_point_agreement",
    "checkpoint_primary_effect_interval_agreement",
    "checkpoint_safety_evidence_agreement",
    "downstream_design_validity",
    "downstream_phase_evaluation_validity",
    "downstream_primary_effect_point_agreement",
    "downstream_primary_effect_interval_agreement",
    "downstream_safety_evidence_agreement",
)
_CONTRASTS: tuple[
    tuple[TrialDevCheckpointContrastIdV1, Literal["context_reset", "canonical_state"]],
    ...,
] = (
    ("context_reset_minus_endogenous", "context_reset"),
    ("canonical_state_minus_endogenous", "canonical_state"),
)
_PHASE_INDEX = {"phase1": 1, "phase2": 2, "phase3": 3}


def _metric_value(
    row: TrialDevCheckpointScoreRowV1,
    metric: TrialDevCheckpointMetricV1,
) -> float | None:
    direct_metrics = {
        "checkpoint_primary_score",
        "checkpoint_decision_correct",
        "downstream_primary_score",
        "downstream_decision_score",
    }
    if metric in direct_metrics:
        return float(getattr(row, metric))
    scope, quality_metric = metric.split("_", maxsplit=1)
    quality = row.checkpoint_quality if scope == "checkpoint" else row.downstream_quality
    return getattr(quality, quality_metric)


def _applicable_metrics(rows: Sequence[TrialDevCheckpointScoreRowV1]) -> tuple[TrialDevCheckpointMetricV1, ...]:
    return tuple(metric for metric in _METRICS if all(_metric_value(row, metric) is not None for row in rows))


def _quality_from_reports(
    reports: Sequence[TrialDevGradeRecordV1],
    *,
    require_primary_effect: bool,
) -> TrialDevCheckpointQualityV1:
    if not reports:
        raise ValueError("Checkpoint quality requires at least one phase report.")
    design_values: list[float] = []
    phase_validity_values: list[float] = []
    point_values: list[float] = []
    interval_values: list[float] = []
    safety_values: list[float] = []
    for report in reports:
        if report.design_efficiency is None:
            raise ValueError(f"Checkpoint phase {report.phase_id!r} lacks design-efficiency evidence.")
        quality = report.analysis_quality
        design_values.append(float(report.design_efficiency.design_valid))
        phase_validity_values.append(float(quality.phase_evaluation_valid))
        if quality.randomized_primary_effect_eligible:
            if (
                quality.randomized_primary_effect_point_agreement is None
                or quality.randomized_primary_effect_interval_agreement is None
            ):
                raise ValueError(f"Checkpoint phase {report.phase_id!r} lacks primary-effect agreement.")
            point_values.append(float(quality.randomized_primary_effect_point_agreement))
            interval_values.append(float(quality.randomized_primary_effect_interval_agreement))
        if not quality.safety_evidence_eligible or quality.safety_evidence_agreement is None:
            raise ValueError(f"Checkpoint phase {report.phase_id!r} lacks safety-evidence agreement.")
        safety_values.append(float(quality.safety_evidence_agreement))
    if require_primary_effect and not point_values:
        raise ValueError("Checkpoint quality requires a randomized primary-effect endpoint.")
    return TrialDevCheckpointQualityV1(
        design_validity=fmean(design_values),
        phase_evaluation_validity=fmean(phase_validity_values),
        primary_effect_point_agreement=(fmean(point_values) if point_values else None),
        primary_effect_interval_agreement=(fmean(interval_values) if interval_values else None),
        safety_evidence_agreement=fmean(safety_values),
    )


def _failed_quality(*, primary_effect_eligible: bool) -> TrialDevCheckpointQualityV1:
    return TrialDevCheckpointQualityV1(
        design_validity=0.0,
        phase_evaluation_validity=0.0,
        primary_effect_point_agreement=(0.0 if primary_effect_eligible else None),
        primary_effect_interval_agreement=(0.0 if primary_effect_eligible else None),
        safety_evidence_agreement=0.0,
    )


def _score_assignment(
    *,
    root: Path,
    assignment: TrialDevCheckpointAssignmentV1,
) -> TrialDevCheckpointScoreRowV1:
    assignment_id = assignment.assignment_id
    program_id = assignment.program_id
    program_dir = root / "assignments" / assignment_id / "programs" / program_id
    chain = read_json_model(TrialDevChainSummaryV1, program_dir / "chain_summary.json")
    grade = read_json_model(TrialDevTrajectoryGradeV1, program_dir / "trajectory_grade.json")
    phase_id = assignment.checkpoint_phase_id
    threshold = _PHASE_INDEX[phase_id]
    reports = [
        report
        for report in grade.phase_reports
        if report.phase_id in _PHASE_INDEX and _PHASE_INDEX[str(report.phase_id)] >= threshold
    ]
    report_ids = [str(report.phase_id) for report in reports]
    if len(report_ids) != len(set(report_ids)):
        raise ValueError(f"Checkpoint assignment {assignment_id} has duplicate downstream phase reports.")
    completed = chain.execution_status == "completed"
    if not completed:
        values = (0.0, 0.0, 0.0, 0.0)
        checkpoint_quality = _failed_quality(primary_effect_eligible=phase_id in {"phase2", "phase3"})
        downstream_quality = _failed_quality(primary_effect_eligible=True)
    else:
        checkpoint_reports = [report for report in reports if report.phase_id == phase_id]
        if len(checkpoint_reports) != 1:
            raise ValueError(f"Checkpoint assignment {assignment_id} lacks exactly one checkpoint phase report.")
        missing_regret = [report_id for report_id in report_ids if report_id not in grade.decision_regret_by_phase]
        if missing_regret:
            raise ValueError(f"Checkpoint assignment {assignment_id} lacks decision outcomes for {missing_regret!r}.")
        checkpoint_report = checkpoint_reports[0]
        decision_scores = [1.0 - float(grade.decision_regret_by_phase[report_id]) for report_id in report_ids]
        values = (
            float(checkpoint_report.primary_score),
            1.0 - float(grade.decision_regret_by_phase[phase_id]),
            fmean(float(report.primary_score) for report in reports),
            fmean(decision_scores),
        )
        checkpoint_quality = _quality_from_reports(
            checkpoint_reports,
            require_primary_effect=phase_id in {"phase2", "phase3"},
        )
        downstream_quality = _quality_from_reports(
            reports,
            require_primary_effect=any(report.phase_id in {"phase2", "phase3"} for report in reports),
        )
    return TrialDevCheckpointScoreRowV1(
        assignment_id=assignment_id,
        block_id=assignment.block_id,
        program_id=program_id,
        scenario_id=assignment.scenario_id,
        objective_id=assignment.objective_id,
        replicate_id=assignment.replicate_id,
        condition=assignment.condition,
        checkpoint_phase_id=phase_id,
        checkpoint_primary_score=values[0],
        checkpoint_decision_correct=values[1],
        downstream_primary_score=values[2],
        downstream_decision_score=values[3],
        downstream_phase_count=len(reports),
        checkpoint_quality=checkpoint_quality,
        downstream_quality=downstream_quality,
        completed=completed,
    )


def _matched_blocks(
    rows: Sequence[TrialDevCheckpointScoreRowV1],
) -> dict[str, dict[str, TrialDevCheckpointScoreRowV1]]:
    blocks: dict[str, dict[str, TrialDevCheckpointScoreRowV1]] = {}
    for row in rows:
        block = blocks.setdefault(row.block_id, {})
        if row.condition in block:
            raise ValueError(f"Graded checkpoint results duplicate {row.condition!r} in block {row.block_id!r}.")
        block[row.condition] = row
    if any(set(block) != {"endogenous", "context_reset", "canonical_state"} for block in blocks.values()):
        raise ValueError("Graded checkpoint results are not complete matched triads.")
    return blocks


def _observed_contrasts(
    blocks: dict[str, dict[str, TrialDevCheckpointScoreRowV1]],
) -> tuple[TrialDevCheckpointObservedContrastV1, ...]:
    contrasts: list[TrialDevCheckpointObservedContrastV1] = []
    rows = tuple(row for block in blocks.values() for row in block.values())
    for contrast_id, treatment in _CONTRASTS:
        for metric in _applicable_metrics(rows):
            differences = []
            for block in blocks.values():
                treated = _metric_value(block[treatment], metric)
                endogenous = _metric_value(block["endogenous"], metric)
                if treated is None or endogenous is None:
                    raise AssertionError("Applicable checkpoint metric unexpectedly resolved to null.")
                differences.append(treated - endogenous)
            contrasts.append(
                TrialDevCheckpointObservedContrastV1(
                    metric=metric,
                    contrast_id=contrast_id,
                    n_blocks=len(differences),
                    estimate=fmean(differences),
                )
            )
    return tuple(contrasts)


def summarise_trialdev_checkpoint_replay_v1(*, graded_root: Path) -> TrialDevCheckpointDescriptiveV1:
    """Return observed matched rows and contrasts without inferential claims."""

    root = Path(graded_root)
    schedule = read_json_model(TrialDevCheckpointScheduleV1, root / "schedule.json")
    rows = tuple(_score_assignment(root=root, assignment=assignment) for assignment in schedule.assignments)
    return TrialDevCheckpointDescriptiveV1(
        schedule_checksum=str(schedule.checksum),
        rows=rows,
        observed_contrasts=_observed_contrasts(_matched_blocks(rows)),
    )


def analyse_trialdev_checkpoint_replay_v1(
    *,
    graded_root: Path,
    confidence_level: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> TrialDevCheckpointAnalysisV1:
    """Return checkpoint-local contrasts with crossed-cluster uncertainty."""

    root = Path(graded_root)
    schedule = read_json_model(TrialDevCheckpointScheduleV1, root / "schedule.json")
    rows = tuple(_score_assignment(root=root, assignment=assignment) for assignment in schedule.assignments)
    blocks = _matched_blocks(rows)

    contrasts: list[TrialDevCheckpointContrastV1] = []
    applicable_metrics = _applicable_metrics(rows)
    for contrast_index, (contrast_id, treatment) in enumerate(_CONTRASTS):
        for metric_index, metric in enumerate(applicable_metrics):
            values = []
            for block in blocks.values():
                endogenous = block["endogenous"]
                treated = block[treatment]
                treated_value = _metric_value(treated, metric)
                endogenous_value = _metric_value(endogenous, metric)
                if treated_value is None or endogenous_value is None:
                    raise AssertionError("Applicable checkpoint metric unexpectedly resolved to null.")
                values.append(
                    (
                        endogenous.scenario_id,
                        endogenous.replicate_id,
                        treated_value - endogenous_value,
                    )
                )
            estimate = crossed_cluster_bootstrap_mean(
                values=values,
                min_row_clusters=2,
                min_column_clusters=2,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=bootstrap_seed + 100 * contrast_index + metric_index,
                contrast_id=f"{contrast_id}:{metric}",
                weighting="observation",
            )
            contrasts.append(
                TrialDevCheckpointContrastV1(
                    metric=metric,
                    contrast_id=contrast_id,
                    n_blocks=len(values),
                    n_scenarios=estimate.n_row_clusters,
                    n_replicates=estimate.n_column_clusters,
                    estimate=estimate.estimate,
                    interval_low=estimate.interval_low,
                    interval_high=estimate.interval_high,
                )
            )
    return TrialDevCheckpointAnalysisV1(
        schedule_checksum=str(schedule.checksum),
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        rows=rows,
        contrasts=tuple(contrasts),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write the canonical matched checkpoint analysis artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graded_root")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--descriptive-only",
        action="store_true",
        help="Write observed pilot rows and contrasts without confidence intervals.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report: TrialDevCheckpointDescriptiveV1 | TrialDevCheckpointAnalysisV1
    if args.descriptive_only:
        report = summarise_trialdev_checkpoint_replay_v1(graded_root=Path(args.graded_root))
    else:
        report = analyse_trialdev_checkpoint_replay_v1(
            graded_root=Path(args.graded_root),
            confidence_level=PUBLICATION_CONFIDENCE_LEVEL_V1,
            bootstrap_resamples=PUBLICATION_BOOTSTRAP_REPLICATES_V1,
            bootstrap_seed=TRIALDEV_CHECKPOINT_ANALYSIS_SEED_V1,
        )
    write_json_model(Path(args.out), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
