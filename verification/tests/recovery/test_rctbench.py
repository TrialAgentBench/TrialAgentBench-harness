"""Tests for independent source-sized RCT qualification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.external.recovery.rctbench import (
    RctQualificationTrialV1,
    RctSourceFitV1,
    RctWorldEstimateV1,
    _dependence_divergence,
    _dose_responses,
    _linkage_dose_responses,
    _mechanism_truth,
    _point_estimator_crosscheck,
    _verify_equal_arm_marginals,
)


def _source_fit(*, outcome_kind: str) -> RctSourceFitV1:
    probabilities = tuple((np.arange(40, dtype=float) + 0.5) / 40)
    return RctSourceFitV1(
        outcome_kind=outcome_kind,
        source_subjects=40,
        source_control_subjects=20,
        source_active_subjects=20,
        source_event_rate=0.5 if outcome_kind == "binary" else None,
        active_source_level="active",
        intercept=0.0,
        treatment_coefficient=0.8,
        age_coefficient=0.04,
        bmi_coefficient=-0.08,
        analysis_treatment_effect=0.8,
        analysis_age_coefficient=0.04,
        analysis_bmi_coefficient=-0.08,
        age_center=50.0,
        bmi_center=25.0,
        source_adjusted_standard_error=0.2,
        source_unadjusted_standard_error=0.21,
        source_adjusted_to_unadjusted_se_ratio=0.2 / 0.21,
        residual_probabilities=probabilities,
        residual_quantiles=tuple(np.linspace(-1.0, 1.0, 40)),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treatment": ["control", "active"] * 20,
            "outcome": np.zeros(40),
            "age": np.linspace(35.0, 65.0, 40),
            "bmi": 25.0 + 3.0 * np.sin(np.arange(40)),
        }
    )


def test_continuous_mechanism_truth_recovers_declared_doses() -> None:
    treatment, prognostic = _mechanism_truth(
        _frame(),
        source=_source_fit(outcome_kind="continuous"),
        prognostic_scale=1.5,
        treatment_scale=2.0,
    )

    assert treatment == pytest.approx(1.6)
    assert prognostic == pytest.approx(1.5)


def test_binary_mechanism_truth_is_finite_and_monotone() -> None:
    source = _source_fit(outcome_kind="binary")
    treatment_truth = [
        _mechanism_truth(
            _frame(),
            source=source,
            prognostic_scale=1.0,
            treatment_scale=scale,
        )[0]
        for scale in (0.0, 0.5, 1.0, 1.5, 2.0)
    ]

    assert np.isfinite(treatment_truth).all()
    assert treatment_truth == sorted(treatment_truth)
    assert treatment_truth[0] == pytest.approx(0.0, abs=1e-12)
    assert treatment_truth[-1] < 1.0


def test_scipy_point_estimator_crosscheck_matches_statsmodels() -> None:
    frame = _frame()
    frame["outcome"] = (
        0.7 * frame["treatment"].eq("active").astype(float)
        + 0.04 * frame["age"]
        - 0.03 * frame["bmi"]
    )

    assert (
        _point_estimator_crosscheck(
            frame,
            treatment_estimate=0.7,
        )
        < 1e-12
    )


def test_dose_response_compares_recovered_effect_with_estimand_truth() -> None:
    source = _source_fit(outcome_kind="continuous")
    trial = RctQualificationTrialV1(
        trial_id="RCTBENCH-001",
        source_data_sha256="a" * 64,
        source_dictionary_sha256="b" * 64,
        worlds=3,
        fitted_analysis=source,
    )
    estimates = []
    for world_index in range(3):
        for scale in (0.0, 0.5, 1.0, 1.5, 2.0):
            estimates.append(
                RctWorldEstimateV1(
                    trial_id=trial.trial_id,
                    world_id=f"world-{world_index}",
                    world_index=world_index,
                    mode="source_anchored",
                    response_axis="treatment",
                    prognostic_scale=1.0,
                    treatment_scale=scale,
                    linkage_retention=1.0,
                    marginal_error=0.1,
                    dependence_error=0.1,
                    source_outcome_mean=0.0,
                    generated_outcome_mean=0.0,
                    standardized_outcome_mean_difference=0.0,
                    generated_to_source_outcome_sd_ratio=1.0,
                    treatment_estimate=source.treatment_coefficient * scale,
                    treatment_standard_error=0.2,
                    treatment_truth=source.treatment_coefficient * scale,
                    treatment_covered=True,
                    adjusted_to_unadjusted_se_ratio=1.0,
                    prognostic_projection=1.0,
                    prognostic_truth_projection=1.0,
                    point_estimator_crosscheck_absolute_difference=0.0,
                )
            )
            estimates.append(
                RctWorldEstimateV1(
                    trial_id=trial.trial_id,
                    world_id=f"world-{world_index}",
                    world_index=world_index,
                    mode="source_anchored",
                    response_axis="prognostic",
                    prognostic_scale=scale,
                    treatment_scale=1.0,
                    linkage_retention=1.0,
                    marginal_error=0.1,
                    dependence_error=0.1,
                    source_outcome_mean=0.0,
                    generated_outcome_mean=0.0,
                    standardized_outcome_mean_difference=0.0,
                    generated_to_source_outcome_sd_ratio=1.0,
                    treatment_estimate=source.analysis_treatment_effect,
                    treatment_standard_error=0.2,
                    treatment_truth=source.analysis_treatment_effect,
                    treatment_covered=True,
                    adjusted_to_unadjusted_se_ratio=1.0,
                    prognostic_projection=scale,
                    prognostic_truth_projection=scale,
                    point_estimator_crosscheck_absolute_difference=0.0,
                )
            )

    responses = _dose_responses(tuple(estimates), trial_by_id={trial.trial_id: trial})

    assert len(responses) == 2
    assert all(response.mean_slope == pytest.approx(1.0) for response in responses)
    assert all(abs(response.slope_bias) < 1e-12 for response in responses)


def test_linkage_controls_preserve_arm_marginals_and_measure_paired_divergence() -> (
    None
):
    intact = _frame()
    intact["outcome"] = np.linspace(-2.0, 2.0, len(intact))
    disrupted = intact.copy()
    for treatment in ("control", "active"):
        rows = disrupted["treatment"].eq(treatment)
        disrupted.loc[rows, "outcome"] = disrupted.loc[rows, "outcome"].to_numpy()[::-1]
        disrupted.loc[rows, "age"] = disrupted.loc[rows, "age"].to_numpy()[::-1]

    _verify_equal_arm_marginals(intact, disrupted, mode="independent_marginal")

    assert _dependence_divergence(intact, intact) == pytest.approx(0.0)
    assert _dependence_divergence(intact, disrupted) > 0.0

    changed = disrupted.copy()
    changed.loc[0, "bmi"] += 1.0
    with pytest.raises(ValueError, match="changed the 'bmi' marginal"):
        _verify_equal_arm_marginals(intact, changed, mode="invalid")


def test_linkage_response_uses_disruption_from_paired_intact_world() -> None:
    estimates = []
    for world_index in range(3):
        for retention in (1.0, 0.75, 0.5, 0.25, 0.0):
            disruption = 1.0 - retention
            estimates.append(
                RctWorldEstimateV1(
                    trial_id="RCTBENCH-001",
                    world_id=f"world-{world_index}",
                    world_index=world_index,
                    mode={
                        1.0: "whole_subject",
                        0.75: "linkage_75",
                        0.5: "linkage_50",
                        0.25: "linkage_25",
                        0.0: "independent_marginal",
                    }[retention],
                    response_axis="source_reference",
                    prognostic_scale=1.0,
                    treatment_scale=1.0,
                    linkage_retention=retention,
                    marginal_error=0.1,
                    dependence_error=0.1,
                    linkage_dependence_divergence=0.2 * disruption,
                    source_outcome_mean=0.0,
                    generated_outcome_mean=0.0,
                    standardized_outcome_mean_difference=0.0,
                    generated_to_source_outcome_sd_ratio=1.0,
                    treatment_estimate=0.5 + 0.1 * disruption,
                    treatment_standard_error=0.2,
                    treatment_truth=0.5,
                    treatment_covered=True,
                    adjusted_to_unadjusted_se_ratio=1.0,
                    prognostic_projection=1.0,
                    prognostic_truth_projection=1.0,
                    point_estimator_crosscheck_absolute_difference=0.0,
                )
            )

    response = _linkage_dose_responses(tuple(estimates))[0]

    assert response.mean_dependence_divergence_slope == pytest.approx(0.2)
    assert response.fraction_with_increasing_divergence == pytest.approx(1.0)
    assert response.mean_analysis_perturbation_in_intact_se_slope == pytest.approx(0.5)
    assert response.fraction_with_increasing_perturbation == pytest.approx(1.0)
