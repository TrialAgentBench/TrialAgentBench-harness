"""Canonical reader for freshly graded TrialEval item artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from trialagentbench_harness.analysis.run_identity import trialeval_run_id
from trialagentbench_harness.contracts.core.runs import TrialEvalItemResultV1
from trialagentbench_harness.contracts.scoring.trialeval_scores import TrialEvalGradedItemRowV1
from trialagentbench_harness.io import read_json_model


class TrialEvalScoreSourceError(RuntimeError):
    """Raised when a canonical graded TrialEval run is incomplete or malformed."""


def _graded_row(path: Path, *, run_id: str, record: TrialEvalItemResultV1) -> TrialEvalGradedItemRowV1:
    scores = record.scores
    if scores is None:
        raise TrialEvalScoreSourceError(f"TrialEval item has no canonical grade: {path}")
    if scores.grade.usable_primary != (record.agent_output.result is not None):
        raise TrialEvalScoreSourceError(
            f"TrialEval grade usability disagrees with the persisted primary submission: {path}"
        )
    if scores.item_id != record.item_id or scores.task_id != record.item_id:
        raise TrialEvalScoreSourceError(f"TrialEval score wrapper identity mismatch: {path}")
    if scores.model != record.run_config.model:
        raise TrialEvalScoreSourceError(f"TrialEval score model disagrees with the immutable run config: {path}")
    factors = record.run_config.task_evidence_factors[record.item_id]
    condition = record.run_config.experiment_condition
    capability = condition.reasoning.capability
    return TrialEvalGradedItemRowV1(
        model_id=scores.model,
        run_id=run_id,
        task_id=record.item_id,
        condition_id=condition.condition_id,
        request_replicate_id=condition.request_replicate_id,
        reasoning_effort=condition.reasoning.effort,
        reasoning_capability_sha256=(None if capability is None else capability.checksum),
        procedure_assistance=condition.procedure_assistance,
        analysis_specification=factors.analysis_specification,
        trial_name=scores.trial_name or "",
        design_tier=scores.design_tier,
        design_subtype=scores.design_subtype,
        assumption_tier=scores.assumption_tier,
        context_tier=scores.context_tier,
        estimand_mode=scores.estimand_mode,
        output_mode=scores.output_mode,
        agent_status=scores.agent_status,
        turns_used=scores.turns_used,
        credit_eligible_route_count=scores.credit_eligible_route_count,
        grade=scores.grade,
        planning=scores.planning,
        source_json_path=path.as_posix(),
    )


def iter_trialeval_score_rows(run_dir: Path) -> Iterator[TrialEvalGradedItemRowV1]:
    """Yield one canonical grade row for every scheduled item in a stored run."""

    item_dir = Path(run_dir) / "items"
    if not item_dir.is_dir():
        raise TrialEvalScoreSourceError(f"TrialEval JSON score item directory not found: {run_dir}")
    paths = sorted(item_dir.glob("*.json"))
    if not paths:
        raise TrialEvalScoreSourceError(f"TrialEval score item directory is empty: {item_dir}")
    run_id = trialeval_run_id(run_dir)
    records = tuple(read_json_model(TrialEvalItemResultV1, path) for path in paths)
    task_surfaces = {tuple(record.run_config.task_ids) for record in records}
    if len(task_surfaces) != 1:
        raise TrialEvalScoreSourceError(f"TrialEval items disagree on the scheduled task surface: {run_dir}")
    expected = task_surfaces.pop()
    observed = tuple(record.item_id for record in records)
    if len(set(observed)) != len(observed):
        raise TrialEvalScoreSourceError(f"TrialEval run contains duplicate item identities: {run_dir}")
    if set(observed) != set(expected):
        raise TrialEvalScoreSourceError(
            "TrialEval item artifacts do not match the scheduled task surface: "
            f"missing={sorted(set(expected) - set(observed))!r}, "
            f"unexpected={sorted(set(observed) - set(expected))!r}."
        )
    by_id = {record.item_id: (path, record) for path, record in zip(paths, records, strict=True)}
    for task_id in expected:
        path, record = by_id[task_id]
        yield _graded_row(path, run_id=run_id, record=record)


def iter_trialeval_score_rows_under_root(root_dir: Path) -> Iterator[TrialEvalGradedItemRowV1]:
    """Yield canonical grade rows for every run below a graded root."""

    item_dirs = sorted(path for path in Path(root_dir).rglob("items") if path.is_dir())
    if not item_dirs:
        raise TrialEvalScoreSourceError(f"TrialEval graded root contains no item directories: {root_dir}")
    for item_dir in item_dirs:
        yield from iter_trialeval_score_rows(item_dir.parent)


__all__ = [
    "TrialEvalScoreSourceError",
    "iter_trialeval_score_rows",
    "iter_trialeval_score_rows_under_root",
]
