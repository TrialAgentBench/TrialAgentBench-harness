"""Build the outcome-blind sampling frame for narrative-normalizer qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalNormalizerFrameUnitV1,
    TrialEvalNormalizerFrameV1,
)
from trialagentbench_harness.experiments.trialeval_run_artifacts import (
    CompletedTrialEvalAblationRun,
    load_completed_trialeval_ablation_run,
)
from trialagentbench_harness.grading.key_store import ScoringKeyStoreV1
from trialagentbench_harness.io import read_json_model, sha256_dir_digest, write_json_model

_ResultShape = Literal["scalar", "identified_interval", "vector", "test", "non_identification", "mixed"]
_RESULT_SHAPE_BY_KIND: dict[str, _ResultShape] = {
    "numeric_point": "scalar",
    "numeric_interval": "identified_interval",
    "numeric_vector": "vector",
    "statistical_test": "test",
    "sensitivity_set": "vector",
    "identification_bound": "identified_interval",
    "limitation": "non_identification",
    "abstention": "non_identification",
    "decision": "non_identification",
}


def _planned_result_shapes(evaluator_root: Path) -> dict[str, _ResultShape]:
    manifest_path = evaluator_root / "grader" / "scoring_key_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item_ids = tuple(str(item_id) for item_id in manifest.get("item_ids", ()))
    store = ScoringKeyStoreV1.from_release(evaluator_root, expected_item_ids=item_ids)
    result: dict[str, _ResultShape] = {}
    for key in store.all():
        shapes = {_RESULT_SHAPE_BY_KIND[route.method.result_kind] for route in key.credit_eligible_routes}
        result[key.item_id] = next(iter(shapes)) if len(shapes) == 1 else "mixed"
    return result


def _validate_campaign_runs(runs: tuple[CompletedTrialEvalAblationRun, ...]) -> None:
    identities = tuple(run.run_config.run_identity_sha256 for run in runs)
    if len(identities) != len(set(identities)):
        raise ValueError("Normalizer frame contains duplicate run identities.")
    schedule_checksums = {run.run_config.schedule_checksum for run in runs}
    participant_checksums = {run.run_config.participant_release_sha256 for run in runs}
    experiment_ids = {run.run_config.experiment_id for run in runs}
    if len(schedule_checksums) != 1 or len(participant_checksums) != 1 or len(experiment_ids) != 1:
        raise ValueError("Normalizer frame runs must share one experiment, schedule, and participant release.")


def build_trialeval_normalizer_frame_v1(
    *,
    run_dirs: tuple[Path, ...],
    evaluator_root: Path,
    evaluator_labels: TrialEvalAblationEvaluatorLabelsV1,
) -> TrialEvalNormalizerFrameV1:
    """Build report metadata without consulting responses, scores, or normalizer outputs."""

    if not run_dirs:
        raise ValueError("Normalizer qualification requires at least one completed run.")
    evaluator_candidate = Path(evaluator_root)
    if evaluator_candidate.is_symlink():
        raise ValueError(f"Evaluator root must be a regular directory: {evaluator_candidate}")
    evaluator = evaluator_candidate.resolve(strict=True)
    if not evaluator.is_dir():
        raise ValueError(f"Evaluator root must be a regular directory: {evaluator}")
    if evaluator_labels.evaluator_release_sha256 != sha256_dir_digest(evaluator):
        raise ValueError("Normalizer frame evaluator labels do not match the evaluator release.")
    runs = tuple(load_completed_trialeval_ablation_run(path) for path in run_dirs)
    _validate_campaign_runs(runs)
    identities = {row.task_id: row for row in evaluator_labels.task_identities}
    shapes = _planned_result_shapes(evaluator)

    frame: list[TrialEvalNormalizerFrameUnitV1] = []
    for run in runs:
        for result in run.results:
            assignment = result.assignment
            if assignment.submission_interface != "narrative":
                continue
            identity = identities.get(assignment.task_id)
            if identity is None:
                raise ValueError(f"Narrative assignment refers to unknown evaluator task {assignment.task_id!r}.")
            result_shape = shapes.get(assignment.task_id)
            if result_shape is None:
                raise ValueError(f"Narrative assignment lacks required-primary result shape: {assignment.task_id!r}.")
            report = result.agent_output.report
            report_bytes = b"" if report is None else report.encode("utf-8")
            unit_digest = hashlib.sha256(
                f"{run.run_config.run_identity_sha256}:{assignment.assignment_id}".encode()
            ).hexdigest()
            frame.append(
                TrialEvalNormalizerFrameUnitV1(
                    unit_id=f"normalizer-unit-{unit_digest}",
                    run_identity_sha256=run.run_config.run_identity_sha256,
                    assignment_id=assignment.assignment_id,
                    task_id=assignment.task_id,
                    base_trial_id=identity.base_trial_id,
                    report_sha256=hashlib.sha256(report_bytes).hexdigest(),
                    regime_cell_id=identity.regime_cell_id,
                    design_tier=identity.design_tier,
                    assumption_tier=identity.assumption_tier,
                    context_configuration=identity.context_tier,
                    data_preparation=identity.data_preparation,
                    analysis_specification=identity.analysis_specification,
                    result_shape=result_shape,
                    model_id=run.run_config.model,
                )
            )
        run.assert_unchanged()
    ordered = tuple(sorted(frame, key=lambda row: row.unit_id))
    keys = tuple((row.run_identity_sha256, row.assignment_id) for row in ordered)
    if not ordered or len(keys) != len(set(keys)):
        raise ValueError("Normalizer qualification frame is empty or contains duplicate run assignments.")
    first = runs[0]
    return TrialEvalNormalizerFrameV1(
        evaluator_release_sha256=evaluator_labels.evaluator_release_sha256,
        participant_release_sha256=first.run_config.participant_release_sha256,
        schedule_sha256=first.run_config.schedule_checksum,
        run_identity_sha256s=tuple(sorted(run.run_config.run_identity_sha256 for run in runs)),
        units=ordered,
    ).with_checksum()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="Completed run; repeat for each model run.")
    parser.add_argument("--evaluator-dir", required=True)
    parser.add_argument("--evaluator-labels", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write one canonical evaluator-owned normalizer sampling frame."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite normalizer qualification frame: {output}")
    labels = read_json_model(TrialEvalAblationEvaluatorLabelsV1, Path(args.evaluator_labels))
    frame = build_trialeval_normalizer_frame_v1(
        run_dirs=tuple(Path(path) for path in cast(list[str], args.run_dir)),
        evaluator_root=Path(args.evaluator_dir),
        evaluator_labels=labels,
    )
    write_json_model(output, frame)
    print(f"Wrote {len(frame.units)} eligible narrative reports: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_trialeval_normalizer_frame_v1", "main"]
