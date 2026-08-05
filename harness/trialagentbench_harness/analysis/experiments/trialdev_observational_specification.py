"""Analyse the paired TrialDev observational specification experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from trialagentbench_harness.analysis.experiments.crossed_bootstrap import (
    crossed_cluster_bootstrap_mean,
)
from trialagentbench_harness.contracts.experiments import (
    TrialDevObservationalSpecificationAnalysisV1,
    TrialDevObservationalSpecificationContrastV1,
    TrialDevObservationalSpecificationScheduleV1,
    TrialDevObservationalSpecificationScoreRowV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevGradeRecordV1,
)
from trialagentbench_harness.io import read_json_model, write_json_model
from trialagentbench_harness.numeric_policy import (
    PUBLICATION_BOOTSTRAP_REPLICATES_V1,
    PUBLICATION_CONFIDENCE_LEVEL_V1,
    TRIALDEV_OBSERVATIONAL_ANALYSIS_SEED_V1,
)

_METRICS = ("primary_score", "analysis_valid", "analysis_score", "ranking_score")


def analyse_trialdev_observational_specification_v1(
    *,
    graded_root: Path,
    confidence_level: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> TrialDevObservationalSpecificationAnalysisV1:
    """Return method-stratified paired effects with crossed-cluster uncertainty."""

    root = Path(graded_root)
    schedule = read_json_model(TrialDevObservationalSpecificationScheduleV1, root / "schedule.json")
    rows: list[TrialDevObservationalSpecificationScoreRowV1] = []
    for assignment in schedule.assignments:
        grade = read_json_model(
            TrialDevGradeRecordV1,
            root
            / "assignments"
            / assignment.assignment_id
            / "programs"
            / assignment.program_id
            / "obs_review"
            / "grade_report.json",
        )
        quality = grade.analysis_quality
        if (
            not quality.observational_analysis_eligible
            or quality.observational_analysis_valid is None
            or quality.observational_analysis_score is None
        ):
            raise ValueError(f"Assignment {assignment.assignment_id} lacks an observational analysis endpoint.")
        rows.append(
            TrialDevObservationalSpecificationScoreRowV1(
                assignment_id=assignment.assignment_id,
                pair_id=assignment.pair_id,
                scenario_id=assignment.scenario_id,
                objective_id=assignment.objective_id,
                replicate_id=assignment.replicate_id,
                condition=assignment.condition,
                method_route_id=assignment.method_specification.method_route_id,
                primary_score=grade.primary_score,
                analysis_valid=quality.observational_analysis_valid,
                analysis_score=quality.observational_analysis_score,
                ranking_score=grade.ranking_score,
            )
        )
    by_pair: dict[str, dict[str, TrialDevObservationalSpecificationScoreRowV1]] = {}
    for row in rows:
        by_pair.setdefault(row.pair_id, {})[row.condition] = row
    if any(set(pair) != {"open_selection", "prespecified_execution"} for pair in by_pair.values()):
        raise ValueError("Graded observational specification results are not complete paired blocks.")

    method_ids = sorted({row.method_route_id for row in rows})
    contrasts: list[TrialDevObservationalSpecificationContrastV1] = []
    for method_id in (None, *method_ids):
        pairs = [
            pair
            for pair in by_pair.values()
            if method_id is None or pair["open_selection"].method_route_id == method_id
        ]
        for metric_index, metric in enumerate(_METRICS):
            values = []
            for pair in pairs:
                open_row = pair["open_selection"]
                specified_row = pair["prespecified_execution"]
                delta = float(getattr(specified_row, metric)) - float(getattr(open_row, metric))
                values.append((open_row.scenario_id, open_row.replicate_id, delta))
            estimate = crossed_cluster_bootstrap_mean(
                values=values,
                min_row_clusters=2,
                min_column_clusters=2,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=bootstrap_seed
                + metric_index
                + (0 if method_id is None else 100 * (method_ids.index(method_id) + 1)),
                contrast_id=f"{method_id or 'all'}:{metric}",
                weighting="observation",
            )
            contrasts.append(
                TrialDevObservationalSpecificationContrastV1(
                    metric=cast(
                        Literal["primary_score", "analysis_valid", "analysis_score", "ranking_score"],
                        metric,
                    ),
                    method_route_id=method_id,
                    n_pairs=len(pairs),
                    n_scenarios=estimate.n_row_clusters,
                    n_replicates=estimate.n_column_clusters,
                    estimate=estimate.estimate,
                    interval_low=estimate.interval_low,
                    interval_high=estimate.interval_high,
                )
            )
    return TrialDevObservationalSpecificationAnalysisV1(
        schedule_checksum=str(schedule.checksum),
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        rows=tuple(rows),
        contrasts=tuple(contrasts),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write the canonical paired analysis artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graded_root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = analyse_trialdev_observational_specification_v1(
        graded_root=Path(args.graded_root),
        confidence_level=PUBLICATION_CONFIDENCE_LEVEL_V1,
        bootstrap_resamples=PUBLICATION_BOOTSTRAP_REPLICATES_V1,
        bootstrap_seed=TRIALDEV_OBSERVATIONAL_ANALYSIS_SEED_V1,
    )
    write_json_model(Path(args.out), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
