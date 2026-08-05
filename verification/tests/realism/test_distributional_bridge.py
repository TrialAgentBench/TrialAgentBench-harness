from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.external.realism.distributional_bridge import (
    AnalysisReplicationV1,
    DependenceReplicationV1,
    MarginalReplicationV1,
    TrialAnalysisDesignV1,
    TrialReplicationWorldV1,
    evaluate_trial_replication_world,
    summarize_distributional_bridge,
)


def test_evaluate_trial_replication_world_reports_three_proof_levels() -> None:
    subjects = np.arange(80)
    source = pd.DataFrame(
        {
            "arm": np.where(subjects % 2, "control", "active"),
            "age": 35.0 + subjects / 2,
            "bmi": 20.0 + subjects / 10 + np.sin(subjects),
            "outcome": 2.0 * (subjects % 2 == 0) + subjects / 20 + np.cos(subjects),
        }
    )
    synthetic = source.sample(n=len(source), replace=True, random_state=17).reset_index(
        drop=True
    )
    design = TrialAnalysisDesignV1(
        trial_id="trial-1",
        treatment_column="arm",
        outcome_column="outcome",
        covariate_columns=("age", "bmi"),
    )

    result = evaluate_trial_replication_world(
        source,
        synthetic,
        design=design,
        method="whole_subject",
        world_index=0,
    )

    assert {row.variable for row in result.marginals} == {
        "outcome",
        "age",
        "bmi",
    }
    assert {(row.left_variable, row.right_variable) for row in result.dependence} == {
        ("outcome", "age"),
        ("outcome", "bmi"),
        ("age", "bmi"),
    }
    assert math.isfinite(result.analysis.synthetic_estimate)
    assert result.analysis.synthetic_standard_error > 0


def test_evaluate_trial_replication_world_fails_on_degenerate_covariate() -> None:
    frame = pd.DataFrame(
        {
            "arm": ["active", "control"] * 5,
            "age": range(10),
            "bmi": [25.0] * 10,
            "outcome": range(10),
        }
    )
    design = TrialAnalysisDesignV1(
        trial_id="trial-1",
        treatment_column="arm",
        outcome_column="outcome",
        covariate_columns=("age", "bmi"),
    )

    with pytest.raises(ValueError, match="must vary"):
        evaluate_trial_replication_world(
            frame,
            frame,
            design=design,
            method="whole_subject",
            world_index=0,
        )


def test_summarize_distributional_bridge_quantifies_trial_uncertainty() -> None:
    worlds = []
    for trial_index in range(3):
        for method in ("whole_subject", "independent_columns"):
            for world_index in range(100):
                method_bias = 0.01 if method == "whole_subject" else 0.18
                worlds.append(
                    TrialReplicationWorldV1(
                        trial_id=f"trial-{trial_index}",
                        method=method,
                        world_index=world_index,
                        source_subjects=100,
                        synthetic_subjects=100,
                        marginals=(
                            MarginalReplicationV1(
                                variable="age",
                                standardized_wasserstein=0.08,
                                standardized_mean_error=0.03,
                                absolute_log_sd_ratio=0.02,
                            ),
                            MarginalReplicationV1(
                                variable="bmi",
                                standardized_wasserstein=0.09,
                                standardized_mean_error=0.04,
                                absolute_log_sd_ratio=0.03,
                            ),
                        ),
                        dependence=(
                            DependenceReplicationV1(
                                left_variable="age",
                                right_variable="bmi",
                                source_spearman=0.4,
                                synthetic_spearman=(
                                    0.39 if method == "whole_subject" else 0.02
                                ),
                                absolute_error=(
                                    0.01 if method == "whole_subject" else 0.38
                                ),
                            ),
                        ),
                        analysis=AnalysisReplicationV1(
                            source_estimate=0.5,
                            source_standard_error=0.2,
                            synthetic_estimate=(
                                0.5
                                + method_bias
                                + 0.1 * math.sin(world_index + trial_index)
                            ),
                            synthetic_standard_error=(
                                0.21 if method == "whole_subject" else 0.29
                            ),
                        ),
                    )
                )

    summary = summarize_distributional_bridge(
        tuple(worlds),
        bootstrap_replicates=200,
        seed=7,
    )

    by_method = {row.method: row for row in summary.method_summaries}
    assert by_method["whole_subject"].median_dependence_error == pytest.approx(0.01)
    assert (
        by_method["whole_subject"].median_analysis_absolute_bias_in_source_se
        < by_method["independent_columns"].median_analysis_absolute_bias_in_source_se
    )
    assert by_method["whole_subject"].median_standard_error_ratio == pytest.approx(1.05)
    assert summary.paired_contrast.dependence_trials_favoring_whole_subject == 3
    assert summary.paired_contrast.analysis_trials_favoring_whole_subject == 3
    assert summary.paired_contrast.dependence_difference_ci_low > 0

    mismatched = tuple(
        (
            world.model_copy(
                update={
                    "analysis": world.analysis.model_copy(
                        update={"source_estimate": 0.6}
                    )
                }
            )
            if world.trial_id == "trial-0" and world.method == "independent_columns"
            else world
        )
        for world in worlds
    )
    with pytest.raises(ValueError, match="source analysis differs"):
        summarize_distributional_bridge(
            mismatched,
            bootstrap_replicates=200,
            seed=7,
        )
