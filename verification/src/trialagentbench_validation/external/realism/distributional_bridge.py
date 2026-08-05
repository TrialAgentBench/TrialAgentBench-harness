"""Independent distributional and analysis-impact verification."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import spearmanr, wasserstein_distance
from statsmodels.stats.proportion import proportion_confint


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialAnalysisDesignV1(_FrozenModel):
    """Columns defining one trial-level distributional comparison."""

    schema_id: Literal["trialagentbench.trial_analysis_design/v1"] = (
        "trialagentbench.trial_analysis_design/v1"
    )
    trial_id: str = Field(min_length=1)
    treatment_column: str = Field(min_length=1)
    outcome_column: str = Field(min_length=1)
    covariate_columns: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _columns_are_unique(self) -> TrialAnalysisDesignV1:
        columns = (self.treatment_column, self.outcome_column, *self.covariate_columns)
        if len(columns) != len(set(columns)):
            raise ValueError("trial analysis columns must be unique")
        return self


class MarginalReplicationV1(_FrozenModel):
    """Standardized discrepancy for one endpoint or continuous covariate."""

    variable: str = Field(min_length=1)
    standardized_wasserstein: float = Field(ge=0, allow_inf_nan=False)
    standardized_mean_error: float = Field(ge=0, allow_inf_nan=False)
    absolute_log_sd_ratio: float = Field(ge=0, allow_inf_nan=False)


class DependenceReplicationV1(_FrozenModel):
    """Real and synthetic rank dependence for one variable pair."""

    left_variable: str = Field(min_length=1)
    right_variable: str = Field(min_length=1)
    source_spearman: float = Field(ge=-1, le=1, allow_inf_nan=False)
    synthetic_spearman: float = Field(ge=-1, le=1, allow_inf_nan=False)
    absolute_error: float = Field(ge=0, le=2, allow_inf_nan=False)


class AnalysisReplicationV1(_FrozenModel):
    """HC3-adjusted treatment-effect estimates in real and synthetic data."""

    source_estimate: float = Field(allow_inf_nan=False)
    source_standard_error: float = Field(gt=0, allow_inf_nan=False)
    synthetic_estimate: float = Field(allow_inf_nan=False)
    synthetic_standard_error: float = Field(gt=0, allow_inf_nan=False)


class TrialReplicationWorldV1(_FrozenModel):
    """One independently analyzable real-to-synthetic trial comparison."""

    schema_id: Literal["trialagentbench.trial_replication_world/v1"] = (
        "trialagentbench.trial_replication_world/v1"
    )
    trial_id: str = Field(min_length=1)
    method: Literal["whole_subject", "independent_columns"]
    world_index: int = Field(ge=0)
    source_subjects: int = Field(ge=8)
    synthetic_subjects: int = Field(ge=8)
    marginals: tuple[MarginalReplicationV1, ...] = Field(min_length=2)
    dependence: tuple[DependenceReplicationV1, ...] = Field(min_length=1)
    analysis: AnalysisReplicationV1


class TrialMethodReplicationSummaryV1(_FrozenModel):
    """Trial-level operating characteristics for one resampling method."""

    trial_id: str = Field(min_length=1)
    method: Literal["whole_subject", "independent_columns"]
    worlds: int = Field(ge=100)
    source_subjects: int = Field(ge=8)
    source_estimate: float = Field(allow_inf_nan=False)
    source_standard_error: float = Field(gt=0, allow_inf_nan=False)
    median_standardized_wasserstein: float = Field(ge=0, allow_inf_nan=False)
    median_dependence_error: float = Field(ge=0, allow_inf_nan=False)
    analysis_bias: float = Field(allow_inf_nan=False)
    analysis_bias_mcse: float = Field(gt=0, allow_inf_nan=False)
    analysis_bias_ci_low: float = Field(allow_inf_nan=False)
    analysis_bias_ci_high: float = Field(allow_inf_nan=False)
    analysis_absolute_bias_in_source_se: float = Field(ge=0, allow_inf_nan=False)
    median_standard_error_ratio: float = Field(gt=0, allow_inf_nan=False)
    replication_interval_low: float = Field(allow_inf_nan=False)
    replication_interval_high: float = Field(allow_inf_nan=False)
    source_estimate_captured: bool


class MethodReplicationSummaryV1(_FrozenModel):
    """Across-trial summary with trial-clustered uncertainty."""

    method: Literal["whole_subject", "independent_columns"]
    trials: int = Field(ge=3)
    worlds: int = Field(ge=300)
    median_standardized_wasserstein: float = Field(ge=0, allow_inf_nan=False)
    median_standardized_wasserstein_ci_low: float = Field(ge=0, allow_inf_nan=False)
    median_standardized_wasserstein_ci_high: float = Field(ge=0, allow_inf_nan=False)
    median_dependence_error: float = Field(ge=0, allow_inf_nan=False)
    median_dependence_error_ci_low: float = Field(ge=0, allow_inf_nan=False)
    median_dependence_error_ci_high: float = Field(ge=0, allow_inf_nan=False)
    median_analysis_absolute_bias_in_source_se: float = Field(ge=0, allow_inf_nan=False)
    median_analysis_absolute_bias_in_source_se_ci_low: float = Field(
        ge=0, allow_inf_nan=False
    )
    median_analysis_absolute_bias_in_source_se_ci_high: float = Field(
        ge=0, allow_inf_nan=False
    )
    median_standard_error_ratio: float = Field(gt=0, allow_inf_nan=False)
    median_standard_error_ratio_ci_low: float = Field(gt=0, allow_inf_nan=False)
    median_standard_error_ratio_ci_high: float = Field(gt=0, allow_inf_nan=False)
    replication_interval_capture: float = Field(ge=0, le=1)
    replication_interval_capture_ci_low: float = Field(ge=0, le=1)
    replication_interval_capture_ci_high: float = Field(ge=0, le=1)


class MethodContrastSummaryV1(_FrozenModel):
    """Paired across-trial contrast of linkage-breaking versus whole subjects."""

    contrast: Literal["independent_columns_minus_whole_subject"] = (
        "independent_columns_minus_whole_subject"
    )
    trials: int = Field(ge=3)
    marginal_difference: float = Field(allow_inf_nan=False)
    marginal_difference_ci_low: float = Field(allow_inf_nan=False)
    marginal_difference_ci_high: float = Field(allow_inf_nan=False)
    marginal_trials_favoring_whole_subject: int = Field(ge=0)
    dependence_difference: float = Field(allow_inf_nan=False)
    dependence_difference_ci_low: float = Field(allow_inf_nan=False)
    dependence_difference_ci_high: float = Field(allow_inf_nan=False)
    dependence_trials_favoring_whole_subject: int = Field(ge=0)
    analysis_bias_difference_in_source_se: float = Field(allow_inf_nan=False)
    analysis_bias_difference_in_source_se_ci_low: float = Field(allow_inf_nan=False)
    analysis_bias_difference_in_source_se_ci_high: float = Field(allow_inf_nan=False)
    analysis_trials_favoring_whole_subject: int = Field(ge=0)


class DistributionalBridgeSummaryV1(_FrozenModel):
    """Public reconstruction of multilevel trial replication evidence."""

    schema_id: Literal["trialagentbench.distributional_bridge_summary/v1"] = (
        "trialagentbench.distributional_bridge_summary/v1"
    )
    trial_summaries: tuple[TrialMethodReplicationSummaryV1, ...] = Field(min_length=6)
    method_summaries: tuple[MethodReplicationSummaryV1, ...] = Field(
        min_length=2, max_length=2
    )
    paired_contrast: MethodContrastSummaryV1


def evaluate_trial_replication_world(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    design: TrialAnalysisDesignV1,
    method: Literal["whole_subject", "independent_columns"],
    world_index: int,
) -> TrialReplicationWorldV1:
    """Compare one synthetic trial with its real participant-level source."""

    source_values = _analysis_frame(source, design=design, label="source")
    synthetic_values = _analysis_frame(synthetic, design=design, label="synthetic")
    analysis_variables = (design.outcome_column, *design.covariate_columns)
    marginals = []
    for variable in analysis_variables:
        observed = source_values[variable].to_numpy(dtype=float)
        generated = synthetic_values[variable].to_numpy(dtype=float)
        observed_sd = float(np.std(observed, ddof=1))
        generated_sd = float(np.std(generated, ddof=1))
        if observed_sd <= 0 or generated_sd <= 0:
            raise ValueError(f"{variable} must vary in source and synthetic trials")
        marginals.append(
            MarginalReplicationV1(
                variable=variable,
                standardized_wasserstein=float(
                    wasserstein_distance(observed, generated) / observed_sd
                ),
                standardized_mean_error=float(
                    abs(observed.mean() - generated.mean()) / observed_sd
                ),
                absolute_log_sd_ratio=float(abs(math.log(generated_sd / observed_sd))),
            )
        )
    dependence = []
    for left_index, left in enumerate(analysis_variables):
        for right in analysis_variables[left_index + 1 :]:
            source_correlation = _spearman(
                source_values[left], source_values[right], label="source"
            )
            synthetic_correlation = _spearman(
                synthetic_values[left],
                synthetic_values[right],
                label="synthetic",
            )
            dependence.append(
                DependenceReplicationV1(
                    left_variable=left,
                    right_variable=right,
                    source_spearman=source_correlation,
                    synthetic_spearman=synthetic_correlation,
                    absolute_error=abs(source_correlation - synthetic_correlation),
                )
            )
    source_estimate, source_se = _adjusted_treatment_effect(
        source_values, design=design
    )
    synthetic_estimate, synthetic_se = _adjusted_treatment_effect(
        synthetic_values, design=design
    )
    return TrialReplicationWorldV1(
        trial_id=design.trial_id,
        method=method,
        world_index=world_index,
        source_subjects=len(source_values),
        synthetic_subjects=len(synthetic_values),
        marginals=tuple(marginals),
        dependence=tuple(dependence),
        analysis=AnalysisReplicationV1(
            source_estimate=source_estimate,
            source_standard_error=source_se,
            synthetic_estimate=synthetic_estimate,
            synthetic_standard_error=synthetic_se,
        ),
    )


def summarize_distributional_bridge(
    worlds: tuple[TrialReplicationWorldV1, ...],
    *,
    bootstrap_replicates: int = 2_000,
    seed: int = 451012,
) -> DistributionalBridgeSummaryV1:
    """Summarize marginal, dependence, and analysis replication by trial."""

    if bootstrap_replicates < 200:
        raise ValueError("at least 200 trial-bootstrap replicates are required")
    grouped: dict[tuple[str, str], list[TrialReplicationWorldV1]] = defaultdict(list)
    for world in worlds:
        grouped[(world.trial_id, world.method)].append(world)
    methods_by_trial: dict[str, set[str]] = defaultdict(set)
    trial_summaries: list[TrialMethodReplicationSummaryV1] = []
    for (trial_id, method), group_rows in sorted(grouped.items()):
        methods_by_trial[trial_id].add(method)
        if len(group_rows) < 100:
            raise ValueError("each trial and method requires at least 100 worlds")
        indexes = sorted(row.world_index for row in group_rows)
        if indexes != list(range(len(group_rows))):
            raise ValueError("world indexes must form a complete zero-based sequence")
        source_estimates = {row.analysis.source_estimate for row in group_rows}
        source_ses = {row.analysis.source_standard_error for row in group_rows}
        source_subject_counts = {row.source_subjects for row in group_rows}
        if (
            len(source_estimates) != 1
            or len(source_ses) != 1
            or len(source_subject_counts) != 1
        ):
            raise ValueError("source analysis must be invariant within trial")
        synthetic_estimates = np.asarray(
            [row.analysis.synthetic_estimate for row in group_rows]
        )
        source_estimate = next(iter(source_estimates))
        source_se = next(iter(source_ses))
        synthetic_ses = np.asarray(
            [row.analysis.synthetic_standard_error for row in group_rows]
        )
        differences = synthetic_estimates - source_estimate
        sampling_sd = float(np.std(synthetic_estimates, ddof=1))
        if sampling_sd <= 0:
            raise ValueError("synthetic treatment estimates must vary across worlds")
        interval_low, interval_high = np.quantile(synthetic_estimates, [0.025, 0.975])
        mean_analysis_bias = float(differences.mean())
        analysis_bias_mcse = float(sampling_sd / math.sqrt(len(group_rows)))
        trial_summaries.append(
            TrialMethodReplicationSummaryV1(
                trial_id=trial_id,
                method=cast(Literal["whole_subject", "independent_columns"], method),
                worlds=len(group_rows),
                source_subjects=next(iter(source_subject_counts)),
                source_estimate=source_estimate,
                source_standard_error=source_se,
                median_standardized_wasserstein=float(
                    np.median(
                        [
                            value.standardized_wasserstein
                            for row in group_rows
                            for value in row.marginals
                        ]
                    )
                ),
                median_dependence_error=float(
                    np.median(
                        [
                            value.absolute_error
                            for row in group_rows
                            for value in row.dependence
                        ]
                    )
                ),
                analysis_bias=mean_analysis_bias,
                analysis_bias_mcse=analysis_bias_mcse,
                analysis_bias_ci_low=mean_analysis_bias - 1.96 * analysis_bias_mcse,
                analysis_bias_ci_high=mean_analysis_bias + 1.96 * analysis_bias_mcse,
                analysis_absolute_bias_in_source_se=abs(mean_analysis_bias) / source_se,
                median_standard_error_ratio=float(np.median(synthetic_ses / source_se)),
                replication_interval_low=float(interval_low),
                replication_interval_high=float(interval_high),
                source_estimate_captured=bool(
                    interval_low <= source_estimate <= interval_high
                ),
            )
        )
    expected_methods = {"whole_subject", "independent_columns"}
    if any(methods != expected_methods for methods in methods_by_trial.values()):
        raise ValueError("every trial must contain both resampling methods")
    method_summaries: list[MethodReplicationSummaryV1] = []
    for method in sorted(expected_methods):
        method_rows = [row for row in trial_summaries if row.method == method]
        if len(method_rows) < 3:
            raise ValueError(
                "distributional bridge requires at least three independent trials"
            )
        wasserstein = np.asarray(
            [row.median_standardized_wasserstein for row in method_rows]
        )
        dependence = np.asarray([row.median_dependence_error for row in method_rows])
        analysis_biases = np.asarray(
            [row.analysis_absolute_bias_in_source_se for row in method_rows]
        )
        standard_error_ratio = np.asarray(
            [row.median_standard_error_ratio for row in method_rows]
        )
        capture_count = sum(row.source_estimate_captured for row in method_rows)
        capture_low, capture_high = proportion_confint(
            capture_count,
            len(method_rows),
            alpha=0.05,
            method="beta",
        )
        wasserstein_ci = _trial_bootstrap_median(
            wasserstein,
            replicates=bootstrap_replicates,
            seed=seed,
        )
        dependence_ci = _trial_bootstrap_median(
            dependence,
            replicates=bootstrap_replicates,
            seed=seed + 1,
        )
        analysis_bias_ci = _trial_bootstrap_median(
            analysis_biases,
            replicates=bootstrap_replicates,
            seed=seed + 2,
        )
        standard_error_ratio_ci = _trial_bootstrap_median(
            standard_error_ratio,
            replicates=bootstrap_replicates,
            seed=seed + 3,
        )
        method_summaries.append(
            MethodReplicationSummaryV1(
                method=cast(Literal["whole_subject", "independent_columns"], method),
                trials=len(method_rows),
                worlds=sum(row.worlds for row in method_rows),
                median_standardized_wasserstein=float(np.median(wasserstein)),
                median_standardized_wasserstein_ci_low=wasserstein_ci[0],
                median_standardized_wasserstein_ci_high=wasserstein_ci[1],
                median_dependence_error=float(np.median(dependence)),
                median_dependence_error_ci_low=dependence_ci[0],
                median_dependence_error_ci_high=dependence_ci[1],
                median_analysis_absolute_bias_in_source_se=float(
                    np.median(analysis_biases)
                ),
                median_analysis_absolute_bias_in_source_se_ci_low=analysis_bias_ci[0],
                median_analysis_absolute_bias_in_source_se_ci_high=analysis_bias_ci[1],
                median_standard_error_ratio=float(np.median(standard_error_ratio)),
                median_standard_error_ratio_ci_low=standard_error_ratio_ci[0],
                median_standard_error_ratio_ci_high=standard_error_ratio_ci[1],
                replication_interval_capture=capture_count / len(method_rows),
                replication_interval_capture_ci_low=float(capture_low),
                replication_interval_capture_ci_high=float(capture_high),
            )
        )
    contrast = _paired_method_contrast(
        tuple(trial_summaries),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 10,
    )
    return DistributionalBridgeSummaryV1(
        trial_summaries=tuple(trial_summaries),
        method_summaries=tuple(method_summaries),
        paired_contrast=contrast,
    )


def read_trial_replication_worlds(path: Path) -> tuple[TrialReplicationWorldV1, ...]:
    """Read a public JSONL archive of trial replication metrics."""

    return tuple(
        TrialReplicationWorldV1.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def write_distributional_bridge_summary(
    path: Path,
    summary: DistributionalBridgeSummaryV1,
) -> None:
    """Write a public distributional-bridge summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _analysis_frame(
    frame: pd.DataFrame,
    *,
    design: TrialAnalysisDesignV1,
    label: str,
) -> pd.DataFrame:
    columns = (
        design.treatment_column,
        design.outcome_column,
        *design.covariate_columns,
    )
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} trial is missing columns: {missing}")
    if len(frame) < 8:
        raise ValueError(f"{label} trial requires at least eight subjects")
    output = pd.DataFrame(frame.loc[:, list(columns)].copy())
    for column in (design.outcome_column, *design.covariate_columns):
        output[column] = pd.to_numeric(output[column], errors="raise")
        if (
            output[column].isna().any()
            or not np.isfinite(output[column].to_numpy(dtype=float)).all()
        ):
            raise ValueError(f"{label} {column} must be complete and finite")
    treatment = output[design.treatment_column].astype("string")
    if treatment.isna().any() or treatment.nunique() != 2:
        raise ValueError(f"{label} treatment must contain exactly two complete levels")
    output[design.treatment_column] = treatment
    return output


def _adjusted_treatment_effect(
    frame: pd.DataFrame,
    *,
    design: TrialAnalysisDesignV1,
) -> tuple[float, float]:
    treatment_levels = tuple(
        sorted(frame[design.treatment_column].astype(str).unique())
    )
    treatment = (
        frame[design.treatment_column].astype(str).eq(treatment_levels[1]).astype(float)
    )
    covariates = frame.loc[:, design.covariate_columns].astype(float)
    standardized = (covariates - covariates.mean()) / covariates.std(ddof=1)
    if standardized.isna().to_numpy().any():
        raise ValueError("analysis covariates must have positive variance")
    design_matrix = sm.add_constant(
        pd.concat([treatment.rename("treatment"), standardized], axis=1),
        has_constant="add",
    )
    if (
        np.linalg.matrix_rank(design_matrix.to_numpy(dtype=float))
        != design_matrix.shape[1]
    ):
        raise ValueError("adjusted treatment analysis is rank deficient")
    fitted = sm.OLS(
        frame[design.outcome_column].to_numpy(dtype=float),
        design_matrix.to_numpy(dtype=float),
    ).fit(cov_type="HC3")
    estimate = float(fitted.params[1])
    standard_error = float(fitted.bse[1])
    if (
        not math.isfinite(estimate)
        or not math.isfinite(standard_error)
        or standard_error <= 0
    ):
        raise ValueError("adjusted treatment analysis produced an invalid estimate")
    return estimate, standard_error


def _spearman(left: pd.Series, right: pd.Series, *, label: str) -> float:
    result = float(
        spearmanr(left.to_numpy(dtype=float), right.to_numpy(dtype=float)).statistic
    )
    if not math.isfinite(result):
        raise ValueError(f"{label} Spearman correlation is undefined")
    return result


def _trial_bootstrap_median(
    values: npt.NDArray[np.float64],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    medians = np.asarray(
        [
            np.median(rng.choice(values, size=len(values), replace=True))
            for _ in range(replicates)
        ]
    )
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def _paired_method_contrast(
    rows: tuple[TrialMethodReplicationSummaryV1, ...],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> MethodContrastSummaryV1:
    by_trial: dict[str, dict[str, TrialMethodReplicationSummaryV1]] = defaultdict(dict)
    for row in rows:
        by_trial[row.trial_id][row.method] = row
    expected_methods = {"whole_subject", "independent_columns"}
    if any(set(methods) != expected_methods for methods in by_trial.values()):
        raise ValueError("paired contrast requires both methods for every trial")
    for trial_id, methods in by_trial.items():
        whole_subject = methods["whole_subject"]
        independent = methods["independent_columns"]
        if (
            whole_subject.worlds != independent.worlds
            or whole_subject.source_subjects != independent.source_subjects
            or whole_subject.source_estimate != independent.source_estimate
            or whole_subject.source_standard_error != independent.source_standard_error
        ):
            raise ValueError(
                f"paired contrast source analysis differs between methods for {trial_id}"
            )

    def differences(field: str) -> npt.NDArray[np.float64]:
        values = []
        for methods in by_trial.values():
            independent = getattr(methods["independent_columns"], field)
            whole_subject = getattr(methods["whole_subject"], field)
            values.append(float(independent - whole_subject))
        return np.asarray(values)

    marginal = differences("median_standardized_wasserstein")
    dependence = differences("median_dependence_error")
    analysis = differences("analysis_absolute_bias_in_source_se")
    marginal_ci = _trial_bootstrap_median(
        marginal,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    dependence_ci = _trial_bootstrap_median(
        dependence,
        replicates=bootstrap_replicates,
        seed=seed + 1,
    )
    analysis_ci = _trial_bootstrap_median(
        analysis,
        replicates=bootstrap_replicates,
        seed=seed + 2,
    )
    return MethodContrastSummaryV1(
        trials=len(by_trial),
        marginal_difference=float(np.median(marginal)),
        marginal_difference_ci_low=marginal_ci[0],
        marginal_difference_ci_high=marginal_ci[1],
        marginal_trials_favoring_whole_subject=int((marginal > 0).sum()),
        dependence_difference=float(np.median(dependence)),
        dependence_difference_ci_low=dependence_ci[0],
        dependence_difference_ci_high=dependence_ci[1],
        dependence_trials_favoring_whole_subject=int((dependence > 0).sum()),
        analysis_bias_difference_in_source_se=float(np.median(analysis)),
        analysis_bias_difference_in_source_se_ci_low=analysis_ci[0],
        analysis_bias_difference_in_source_se_ci_high=analysis_ci[1],
        analysis_trials_favoring_whole_subject=int((analysis > 0).sum()),
    )


__all__ = [
    "DistributionalBridgeSummaryV1",
    "MethodContrastSummaryV1",
    "TrialAnalysisDesignV1",
    "TrialReplicationWorldV1",
    "evaluate_trial_replication_world",
    "read_trial_replication_worlds",
    "summarize_distributional_bridge",
    "write_distributional_bridge_summary",
]
