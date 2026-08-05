"""Tests for deterministic one-view-per-trial factorial sampling."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalAblationTaskIdentityV1,
    TrialEvalExperimentProtocolV1,
)
from trialagentbench_harness.experiments.select_trialeval_factorial_tasks import (
    select_trialeval_factorial_tasks_v1,
)

_HARNESS_ROOT = Path(__file__).resolve().parents[2]
_DESIGN = TrialEvalExperimentProtocolV1.model_validate_json(
    (_HARNESS_ROOT / "experiment_configs/trialeval_experiment_protocol_v1.json").read_text(encoding="utf-8")
)


def _evaluator_labels() -> TrialEvalAblationEvaluatorLabelsV1:
    contexts = {
        "C1": ("analysis_ready", "locked_sap"),
        "C2": ("analysis_ready", "protocol_only"),
        "C3": ("raw_domains", "locked_sap"),
        "C4": ("raw_domains", "protocol_only"),
        "C5": ("raw_domains_declared_defect", "protocol_only"),
    }
    cells = tuple(
        (f"TE-S{series:02d}-A{assumption}", f"TE-S{series:02d}", f"A{assumption}")
        for series, tiers in enumerate((3, 3, 2, 2, 3, 3, 3, 3, 3), start=1)
        for assumption in range(1, tiers + 1)
    )
    assert len(cells) == _DESIGN.precision.regime_cell_count
    identities: list[TrialEvalAblationTaskIdentityV1] = []
    task_number = 1
    for regime_cell_id, evaluation_series_id, assumption_tier in cells:
        for replicate in range(_DESIGN.precision.base_trial_replicates_per_regime_cell):
            base_trial_id = f"{regime_cell_id}:world-{replicate}"
            for context, (data_preparation, analysis_specification) in contexts.items():
                identities.append(
                    TrialEvalAblationTaskIdentityV1(
                        task_id=f"TASK{task_number:04d}",
                        base_trial_id=base_trial_id,
                        regime_cell_id=regime_cell_id,
                        evaluation_series_id=evaluation_series_id,
                        design_tier="D1",
                        design_subtype="individual_randomized",
                        assumption_tier=assumption_tier,
                        context_tier=context,
                        data_preparation=data_preparation,
                        analysis_specification=analysis_specification,
                    )
                )
                task_number += 1
    return TrialEvalAblationEvaluatorLabelsV1(
        evaluator_release_sha256="e" * 64,
        task_identities=tuple(identities),
    )


def test_factorial_selection_is_balanced_deterministic_and_one_per_base_trial() -> None:
    labels = _evaluator_labels()
    sample = select_trialeval_factorial_tasks_v1(
        experiment_design=_DESIGN,
        evaluator_labels=labels,
        participant_release_sha256="p" * 64,
    )
    repeated = select_trialeval_factorial_tasks_v1(
        experiment_design=_DESIGN,
        evaluator_labels=labels,
        participant_release_sha256="p" * 64,
    )
    identity_by_task = {row.task_id: row for row in labels.task_identities}

    assert sample == repeated
    assert sample.context_allocation == _DESIGN.compute_envelope.factorial_context_allocation
    assert len(sample.task_ids) == _DESIGN.precision.retained_independent_base_trials
    assert len({identity_by_task[task_id].base_trial_id for task_id in sample.task_ids}) == (
        _DESIGN.precision.retained_independent_base_trials
    )
    assert len({identity_by_task[task_id].regime_cell_id for task_id in sample.task_ids}) == (
        _DESIGN.precision.regime_cell_count
    )
    context_by_cell: dict[str, set[str]] = {}
    for task_id in sample.task_ids:
        identity = identity_by_task[task_id]
        context_by_cell.setdefault(str(identity.regime_cell_id), set()).add(str(identity.context_tier))
    assert {len(contexts) for contexts in context_by_cell.values()} == {
        _DESIGN.precision.base_trial_replicates_per_regime_cell
    }
    omitted_contexts = {
        context for contexts in context_by_cell.values() for context in set(("C1", "C2", "C3", "C4", "C5")) - contexts
    }
    assert omitted_contexts == {"C1", "C2", "C3", "C4", "C5"}


def test_factorial_selection_rejects_incomplete_base_trial_block() -> None:
    labels = _evaluator_labels()
    incomplete = labels.model_copy(update={"task_identities": labels.task_identities[:-1], "checksum": None})
    incomplete = TrialEvalAblationEvaluatorLabelsV1.model_validate(incomplete.model_dump(mode="python"))

    with pytest.raises(ValueError, match="invalid canonical context blocks"):
        select_trialeval_factorial_tasks_v1(
            experiment_design=_DESIGN,
            evaluator_labels=incomplete,
            participant_release_sha256="p" * 64,
        )


def test_factorial_selection_rejects_family_replication_drift() -> None:
    labels = _evaluator_labels()
    first_base = labels.task_identities[0].base_trial_id
    changed = tuple(
        row.model_copy(update={"regime_cell_id": "TE-S99-A1"}) if row.base_trial_id == first_base else row
        for row in labels.task_identities
    )
    drifted = TrialEvalAblationEvaluatorLabelsV1(
        evaluator_release_sha256=labels.evaluator_release_sha256,
        task_identities=changed,
    )

    with pytest.raises(ValueError, match="regime-cell replication"):
        select_trialeval_factorial_tasks_v1(
            experiment_design=_DESIGN,
            evaluator_labels=drifted,
            participant_release_sha256="p" * 64,
        )
