"""Select one balanced context view per TrialEval base trial."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalAblationTaskIdentityV1,
    TrialEvalExperimentProtocolV1,
    TrialEvalFactorialTaskSampleV1,
)
from trialagentbench_harness.io import read_json_model, sha256_dir_digest, write_json_model

_CONTEXTS = ("C1", "C2", "C3", "C4", "C5")


def select_trialeval_factorial_tasks_v1(
    *,
    experiment_design: TrialEvalExperimentProtocolV1,
    evaluator_labels: TrialEvalAblationEvaluatorLabelsV1,
    participant_release_sha256: str,
) -> TrialEvalFactorialTaskSampleV1:
    """Return the deterministic balanced factorial sample from complete blocks."""

    by_base: dict[str, dict[str, TrialEvalAblationTaskIdentityV1]] = {}
    for identity in evaluator_labels.task_identities:
        context_rows = by_base.setdefault(identity.base_trial_id, {})
        if identity.context_tier in context_rows:
            raise ValueError(
                f"Base trial {identity.base_trial_id!r} has duplicate {identity.context_tier} evaluator identities."
            )
        context_rows[identity.context_tier] = identity

    expected_base_trials = experiment_design.precision.retained_independent_base_trials
    if len(by_base) != expected_base_trials:
        raise ValueError(
            "Evaluator labels do not contain the frozen independent-base-trial population: "
            f"expected={expected_base_trials} observed={len(by_base)}."
        )
    incomplete = sorted(base_id for base_id, rows in by_base.items() if set(rows) != set(_CONTEXTS))
    if incomplete:
        raise ValueError(f"Evaluator labels contain invalid canonical context blocks: {incomplete!r}.")

    inconsistent_blocks = sorted(
        base_id
        for base_id, context_rows in by_base.items()
        if len({row.regime_cell_id for row in context_rows.values()}) != 1
        or len({row.evaluation_series_id for row in context_rows.values()}) != 1
        or len({row.design_tier for row in context_rows.values()}) != 1
        or len({row.design_subtype for row in context_rows.values()}) != 1
        or len({row.assumption_tier for row in context_rows.values()}) != 1
    )
    if inconsistent_blocks:
        raise ValueError(
            f"Evaluator labels contain internally inconsistent base-trial blocks: {inconsistent_blocks!r}."
        )

    cell_counts = Counter(str(next(iter(context_rows.values())).regime_cell_id) for context_rows in by_base.values())
    expected_replicates = experiment_design.precision.base_trial_replicates_per_regime_cell
    if len(cell_counts) != experiment_design.precision.regime_cell_count or any(
        count != expected_replicates for count in cell_counts.values()
    ):
        raise ValueError(
            "Evaluator base-trial blocks do not match the frozen regime-cell replication: "
            f"expected_cell_count={experiment_design.precision.regime_cell_count} "
            f"observed_counts={dict(sorted(cell_counts.items()))!r}."
        )

    base_ids_by_cell: dict[str, list[str]] = {}
    for base_id, context_rows in by_base.items():
        identity = next(iter(context_rows.values()))
        base_ids_by_cell.setdefault(str(identity.regime_cell_id), []).append(base_id)

    ranked_cells = sorted(
        base_ids_by_cell,
        key=lambda cell_id: hashlib.sha256(f"{experiment_design.checksum}\0cell\0{cell_id}".encode()).hexdigest(),
    )
    selected_task_ids: list[str] = []
    allocation = Counter[str]()
    for cell_index, cell_id in enumerate(ranked_cells):
        omitted_context = _CONTEXTS[cell_index % len(_CONTEXTS)]
        retained_contexts = [context for context in _CONTEXTS if context != omitted_context]
        ranked_contexts = sorted(
            retained_contexts,
            key=lambda context: hashlib.sha256(
                f"{experiment_design.checksum}\0context\0{cell_id}\0{context}".encode()
            ).hexdigest(),
        )
        ranked_base_ids = sorted(
            base_ids_by_cell[cell_id],
            key=lambda base_id: hashlib.sha256(
                f"{experiment_design.checksum}\0base\0{cell_id}\0{base_id}".encode()
            ).hexdigest(),
        )
        if len(ranked_base_ids) != len(ranked_contexts):
            raise ValueError(
                "Stratified factorial selection requires one fewer base-trial replicate per regime cell "
                f"than context tiers: cell={cell_id!r} base_trials={len(ranked_base_ids)} "
                f"retained_contexts={len(ranked_contexts)}."
            )
        for base_id, context in zip(ranked_base_ids, ranked_contexts, strict=True):
            identity = by_base[base_id][context]
            selected_task_ids.append(str(identity.task_id))
            allocation[context] += 1
    observed_allocation = tuple(allocation[context] for context in _CONTEXTS)
    if observed_allocation != experiment_design.compute_envelope.factorial_context_allocation:
        raise ValueError(
            "Deterministic factorial selection did not meet the frozen context allocation: "
            f"expected={experiment_design.compute_envelope.factorial_context_allocation!r} "
            f"observed={observed_allocation!r}."
        )
    return TrialEvalFactorialTaskSampleV1(
        experiment_design_sha256=experiment_design.checksum,
        participant_release_sha256=participant_release_sha256,
        evaluator_labels_sha256=str(evaluator_labels.checksum),
        selection_method=experiment_design.compute_envelope.factorial_selection_method,
        task_ids=tuple(selected_task_ids),
        context_allocation=observed_allocation,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-design", required=True)
    parser.add_argument("--evaluator-labels", required=True)
    parser.add_argument("--participant-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write one immutable participant-safe factorial task sample."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite TrialEval factorial task sample: {output}")
    sample = select_trialeval_factorial_tasks_v1(
        experiment_design=read_json_model(TrialEvalExperimentProtocolV1, Path(args.experiment_design)),
        evaluator_labels=read_json_model(TrialEvalAblationEvaluatorLabelsV1, Path(args.evaluator_labels)),
        participant_release_sha256=sha256_dir_digest(Path(args.participant_dir)),
    )
    write_json_model(output, sample)
    print(f"Wrote TrialEval factorial task sample: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
