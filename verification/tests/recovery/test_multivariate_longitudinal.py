from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.external.recovery.multivariate_longitudinal import (
    MultivariateCellReferenceV1,
    MultivariateLongitudinalDesignV1,
    MultivariateLongitudinalFittedModelV1,
    MultivariatePairReferenceV1,
    MultivariateTreatmentTruthV1,
    _fit_ancova,
    _summarize_cells,
)


def _design(
    *, latent_correlation: tuple[tuple[float, ...], ...]
) -> MultivariateLongitudinalDesignV1:
    outcomes = ("a", "b")
    times = (0.0, 1.0)
    cells = tuple(
        MultivariateCellReferenceV1(
            arm_id=arm,
            outcome_id=outcome,
            time=time,
            observations=50,
            mean=0.0,
            standard_deviation=1.0,
            quantile_10=-1.0,
            quantile_50=0.0,
            quantile_90=1.0,
        )
        for arm in ("control", "intervention")
        for outcome in outcomes
        for time in times
    )
    return MultivariateLongitudinalDesignV1(
        schema_id="trialagentbench.multivariate_longitudinal_design/v1",
        seed=12,
        trial_id="trial",
        source_sha256="0" * 64,
        participants=100,
        worlds=2,
        arm_ids=("control", "intervention"),
        control_arm_id="control",
        outcome_ids=outcomes,
        time_values=times,
        source_complete_trajectories=50,
        fitted_model=MultivariateLongitudinalFittedModelV1(
            control_mean_values={"a": (0.0, 0.0), "b": (0.0, 0.0)},
            latent_correlation=latent_correlation,
            marginal_probabilities=(0.1, 0.3, 0.5, 0.7, 0.9),
            marginal_residual_values={
                "a": ((-2.0, -1.0, 0.0, 1.0, 2.0),) * 2,
                "b": ((-2.0, -1.0, 0.0, 1.0, 2.0),) * 2,
            },
            arm_visit_shifts={"intervention": {"a": (0.0, 1.0), "b": (0.0, 1.0)}},
            dropout_logit_intercepts=(-3.0,),
            dropout_treatment_coefficient=0.5,
            measurement_probabilities={"a": (1.0, 0.95), "b": (0.9, 1.0)},
        ),
        cells=cells,
        cross_outcome_pairs=(
            MultivariatePairReferenceV1(
                outcome_a="a",
                time_a=0.0,
                outcome_b="b",
                time_b=0.0,
                complete_pairs=50,
                spearman_correlation=0.5,
            ),
        ),
        treatment_truth=(
            MultivariateTreatmentTruthV1(
                arm_id="intervention",
                outcome_id="a",
                final_time=1.0,
                contrast=1.0,
            ),
        ),
    )


def test_design_rejects_latent_correlation_with_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="square by outcome-visit cell"):
        _design(latent_correlation=((1.0, 0.0), (0.0, 1.0)))


def test_design_rejects_incomplete_observation_process() -> None:
    correlation = tuple(tuple(float(i == j) for j in range(4)) for i in range(4))
    design = _design(latent_correlation=correlation)
    payload = design.model_dump(mode="python")
    payload["fitted_model"]["dropout_logit_intercepts"] = ()

    with pytest.raises(ValueError, match="at least 1 item"):
        MultivariateLongitudinalDesignV1.model_validate(payload)


def test_ancova_recovers_randomized_final_contrast() -> None:
    rng = np.random.default_rng(451)
    rows: list[dict[str, str | int | float]] = []
    for participant_pair in range(200):
        baseline = rng.normal()
        noise = rng.normal(scale=0.5)
        for arm_index, arm in enumerate(("control", "intervention")):
            participant = 2 * participant_pair + arm_index
            final = 2.0 + 0.8 * baseline + 1.5 * arm_index + noise
            rows.extend(
                [
                    {
                        "participant_id": str(participant),
                        "arm": arm,
                        "time": 0.0,
                        "value": baseline,
                    },
                    {
                        "participant_id": str(participant),
                        "arm": arm,
                        "time": 1.0,
                        "value": final,
                    },
                ]
            )

    estimate, standard_error = _fit_ancova(
        pd.DataFrame(rows),
        arm_id="intervention",
        control_arm_id="control",
        baseline_time=0.0,
        final_time=1.0,
    )

    assert estimate == pytest.approx(1.5, abs=0.1)
    assert 0.0 < standard_error < 0.1


def test_cell_summary_keeps_observation_and_distribution_metrics_separate() -> None:
    correlation = tuple(tuple(float(i == j) for j in range(4)) for i in range(4))
    design = _design(latent_correlation=correlation)
    rows: list[dict[str, str | int | float]] = []
    samples = ((-1.0, -0.5, 0.0, 0.5, 1.0), (-1.0, 0.0, 1.0))
    for world_index, sample in enumerate(samples):
        for cell in design.cells:
            rows.extend(
                {
                    "world_index": world_index,
                    "mode": "source_anchored",
                    "arm": cell.arm_id,
                    "outcome_id": cell.outcome_id,
                    "time": cell.time,
                    "value": value,
                }
                for value in sample
            )

    summaries = _summarize_cells(pd.DataFrame(rows), design)

    assert len(summaries) == len(design.cells)
    assert {summary.observations_median for summary in summaries} == {4.0}
    assert {summary.source_observations_predictive_rank for summary in summaries} == {
        1.0
    }
    assert all(
        summary.mean_interval_95_low
        <= summary.mean_interval_50_low
        <= summary.mean_median
        <= summary.mean_interval_50_high
        <= summary.mean_interval_95_high
        for summary in summaries
    )
    assert (
        max(summary.standardized_quantile_error_median for summary in summaries) < 0.3
    )
