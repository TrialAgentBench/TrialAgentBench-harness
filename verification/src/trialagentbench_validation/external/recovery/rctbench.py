"""Independent verification of source-sized RCT production qualification."""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field
from scipy.linalg import lstsq
from scipy.special import expit
from scipy.stats import spearmanr, t, wasserstein_distance

from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.statistics import proportion_interval


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _FitResult(Protocol):
    params: pd.Series
    bse: pd.Series


class RctSourceFitV1(_FrozenModel):
    """Source regression parameters required for independent recovery."""

    outcome_kind: Literal["binary", "continuous"]
    source_subjects: int = Field(ge=40)
    source_control_subjects: int = Field(ge=1)
    source_active_subjects: int = Field(ge=1)
    source_event_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    active_source_level: str = Field(min_length=1)
    intercept: float = Field(allow_inf_nan=False)
    treatment_coefficient: float = Field(allow_inf_nan=False)
    age_coefficient: float = Field(allow_inf_nan=False)
    bmi_coefficient: float = Field(allow_inf_nan=False)
    analysis_treatment_effect: float = Field(allow_inf_nan=False)
    analysis_age_coefficient: float = Field(allow_inf_nan=False)
    analysis_bmi_coefficient: float = Field(allow_inf_nan=False)
    age_center: float = Field(allow_inf_nan=False)
    bmi_center: float = Field(allow_inf_nan=False)
    source_adjusted_standard_error: float = Field(gt=0.0, allow_inf_nan=False)
    source_unadjusted_standard_error: float = Field(gt=0.0, allow_inf_nan=False)
    source_adjusted_to_unadjusted_se_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    residual_probabilities: tuple[float, ...] = Field(min_length=5)
    residual_quantiles: tuple[float, ...] = Field(min_length=5)


class RctQualificationTrialV1(_FrozenModel):
    """One source-bound RCT qualification declaration."""

    trial_id: str = Field(pattern=r"^RCTBENCH-[0-9]{3}$")
    source_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dictionary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: int = Field(ge=2)
    fitted_analysis: RctSourceFitV1


class RctQualificationDesignV1(_FrozenModel):
    """Independent view of the RCT qualification design."""

    schema_id: Literal["trialagentbench.rctbench_production_qualification_design/v1"]
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    seed: int = Field(ge=0)
    dose_levels: tuple[float, ...] = Field(min_length=3)
    trials: tuple[RctQualificationTrialV1, ...] = Field(min_length=3)


class RctWorldReceiptV1(_FrozenModel):
    """Independent view of one RCT world receipt."""

    world_id: str = Field(pattern=r"^rctbench-[0-9]{3}-[0-9a-f]{16}$")
    trial_id: str = Field(pattern=r"^RCTBENCH-[0-9]{3}$")
    world_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    analysis_path: str = Field(min_length=1)
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_spec_sha256_by_cell: dict[str, str] = Field(min_length=1)
    generated_bundle_sha256_by_cell: dict[str, str] = Field(min_length=1)


class RctQualificationReceiptV1(_FrozenModel):
    """Independent view of the complete RCT qualification receipt."""

    schema_id: Literal["trialagentbench.rctbench_production_qualification_receipt/v1"]
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[RctWorldReceiptV1, ...] = Field(min_length=1)


class RctWorldEstimateV1(_FrozenModel):
    """One independently estimated RCT world cell."""

    trial_id: str
    world_id: str
    world_index: int = Field(ge=0)
    mode: Literal[
        "whole_subject",
        "linkage_75",
        "linkage_50",
        "linkage_25",
        "independent_marginal",
        "source_anchored",
    ]
    response_axis: Literal["source_reference", "treatment", "prognostic"]
    prognostic_scale: float = Field(ge=0.0, allow_inf_nan=False)
    treatment_scale: float = Field(ge=0.0, allow_inf_nan=False)
    linkage_retention: float = Field(ge=0.0, le=1.0)
    marginal_error: float = Field(ge=0.0, allow_inf_nan=False)
    dependence_error: float = Field(ge=0.0, allow_inf_nan=False)
    linkage_dependence_divergence: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )
    source_outcome_mean: float = Field(allow_inf_nan=False)
    generated_outcome_mean: float = Field(allow_inf_nan=False)
    standardized_outcome_mean_difference: float = Field(allow_inf_nan=False)
    generated_to_source_outcome_sd_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    treatment_estimate: float = Field(allow_inf_nan=False)
    treatment_standard_error: float = Field(gt=0.0, allow_inf_nan=False)
    treatment_truth: float = Field(allow_inf_nan=False)
    treatment_covered: bool
    adjusted_to_unadjusted_se_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    prognostic_projection: float = Field(allow_inf_nan=False)
    prognostic_truth_projection: float = Field(allow_inf_nan=False)
    point_estimator_crosscheck_absolute_difference: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )


class RctCellSummaryV1(_FrozenModel):
    """Repeated-world operating characteristics for one RCT cell."""

    trial_id: str
    mode: str
    response_axis: str
    prognostic_scale: float
    treatment_scale: float
    worlds: int = Field(ge=2)
    mean_marginal_error: float = Field(ge=0.0, allow_inf_nan=False)
    mean_dependence_error: float = Field(ge=0.0, allow_inf_nan=False)
    mean_standardized_outcome_mean_difference: float = Field(allow_inf_nan=False)
    standardized_outcome_mean_difference_ci_low: float = Field(allow_inf_nan=False)
    standardized_outcome_mean_difference_ci_high: float = Field(allow_inf_nan=False)
    mean_generated_to_source_outcome_sd_ratio: float = Field(
        gt=0.0, allow_inf_nan=False
    )
    generated_to_source_outcome_sd_ratio_ci_low: float = Field(
        gt=0.0, allow_inf_nan=False
    )
    generated_to_source_outcome_sd_ratio_ci_high: float = Field(
        gt=0.0, allow_inf_nan=False
    )
    treatment_bias: float = Field(allow_inf_nan=False)
    treatment_bias_ci_low: float = Field(allow_inf_nan=False)
    treatment_bias_ci_high: float = Field(allow_inf_nan=False)
    treatment_coverage: float = Field(ge=0.0, le=1.0)
    treatment_coverage_ci_low: float = Field(ge=0.0, le=1.0)
    treatment_coverage_ci_high: float = Field(ge=0.0, le=1.0)
    mean_adjusted_to_unadjusted_se_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    adjusted_to_unadjusted_se_ratio_interval_low: float = Field(
        gt=0.0,
        allow_inf_nan=False,
    )
    adjusted_to_unadjusted_se_ratio_interval_high: float = Field(
        gt=0.0,
        allow_inf_nan=False,
    )
    source_adjusted_to_unadjusted_se_ratio: float = Field(
        gt=0.0,
        allow_inf_nan=False,
    )
    source_se_ratio_contained: bool
    mean_prognostic_projection: float = Field(allow_inf_nan=False)
    mean_prognostic_truth_projection: float = Field(allow_inf_nan=False)
    prognostic_projection_bias: float = Field(allow_inf_nan=False)
    maximum_point_estimator_crosscheck_difference: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )


class RctDoseResponseV1(_FrozenModel):
    """Paired common-random-number response slope for one RCT and mechanism."""

    trial_id: str
    response_axis: Literal["treatment", "prognostic"]
    worlds: int = Field(ge=2)
    expected_slope: float = Field(allow_inf_nan=False)
    mean_slope: float = Field(allow_inf_nan=False)
    slope_bias: float = Field(allow_inf_nan=False)
    slope_bias_ci_low: float = Field(allow_inf_nan=False)
    slope_bias_ci_high: float = Field(allow_inf_nan=False)
    directionally_concordant_fraction: float = Field(ge=0.0, le=1.0)
    directionally_concordant_fraction_ci_low: float = Field(ge=0.0, le=1.0)
    directionally_concordant_fraction_ci_high: float = Field(ge=0.0, le=1.0)


class RctLinkageDoseResponseV1(_FrozenModel):
    """Paired response to progressive participant-linkage disruption."""

    trial_id: str
    worlds: int = Field(ge=2)
    mean_dependence_divergence_slope: float = Field(allow_inf_nan=False)
    dependence_divergence_slope_ci_low: float = Field(allow_inf_nan=False)
    dependence_divergence_slope_ci_high: float = Field(allow_inf_nan=False)
    fraction_with_increasing_divergence: float = Field(ge=0.0, le=1.0)
    mean_analysis_perturbation_in_intact_se_slope: float = Field(allow_inf_nan=False)
    fraction_with_increasing_perturbation: float = Field(ge=0.0, le=1.0)


class RctQualificationReportV1(_FrozenModel):
    """Independent RCT realism and recoverability report."""

    schema_id: Literal[
        "trialagentbench.rctbench_production_qualification_report/v1"
    ] = "trialagentbench.rctbench_production_qualification_report/v1"
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trials: int = Field(ge=3)
    worlds: int = Field(ge=6)
    estimates: tuple[RctWorldEstimateV1, ...] = Field(min_length=1)
    cell_summaries: tuple[RctCellSummaryV1, ...] = Field(min_length=1)
    dose_responses: tuple[RctDoseResponseV1, ...] = Field(min_length=1)
    linkage_dose_responses: tuple[RctLinkageDoseResponseV1, ...] = Field(min_length=1)


def evaluate_rctbench_qualification(
    *,
    release_dir: Path,
    source_root: Path,
    minimum_worlds_per_trial: int = 100,
) -> RctQualificationReportV1:
    """Verify released RCT worlds and independently recover all analyses."""
    design_path = release_dir / "design.json"
    receipt_path = release_dir / "receipt.json"
    design = RctQualificationDesignV1.model_validate_json(
        design_path.read_text(encoding="utf-8")
    )
    receipt = RctQualificationReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    design_sha256 = _canonical_json_sha(design_path)
    if receipt.design_sha256 != design_sha256:
        raise ValueError("RCT qualification receipt does not bind the design bytes.")
    if _git_revision(source_root) != design.source_revision:
        raise ValueError(
            "RCT Bench source revision differs from the qualification design."
        )
    trial_by_id = {trial.trial_id: trial for trial in design.trials}
    sources = {
        trial.trial_id: _load_source(source_root, trial) for trial in design.trials
    }
    for trial in design.trials:
        _verify_source_fit(sources[trial.trial_id], trial.fitted_analysis)
    receipts_by_trial: dict[str, list[RctWorldReceiptV1]] = defaultdict(list)
    estimates: list[RctWorldEstimateV1] = []
    for world in receipt.worlds:
        if world.trial_id not in trial_by_id:
            raise ValueError(
                f"RCT receipt references unknown trial {world.trial_id!r}."
            )
        path = (release_dir / world.analysis_path).resolve()
        if not path.is_relative_to(release_dir.resolve()):
            raise ValueError("RCT world path escapes the release directory.")
        if sha256_file(path) != world.analysis_sha256:
            raise ValueError(f"RCT world checksum mismatch for {world.world_id!r}.")
        receipts_by_trial[world.trial_id].append(world)
        frame = pd.read_parquet(path)
        estimates.extend(
            _evaluate_world(
                frame,
                source=sources[world.trial_id],
                trial=trial_by_id[world.trial_id],
                world=world,
            )
        )
    for trial in design.trials:
        rows = receipts_by_trial[trial.trial_id]
        if len(rows) < minimum_worlds_per_trial:
            raise ValueError(
                f"{trial.trial_id} has {len(rows)} worlds; {minimum_worlds_per_trial} are required."
            )
        indexes = sorted(row.world_index for row in rows)
        if indexes != list(range(len(rows))):
            raise ValueError(
                f"{trial.trial_id} world indexes are not complete and zero based."
            )
    estimate_tuple = tuple(estimates)
    return RctQualificationReportV1(
        design_sha256=design_sha256,
        receipt_sha256=_canonical_json_sha(receipt_path),
        trials=len(design.trials),
        worlds=len(receipt.worlds),
        estimates=estimate_tuple,
        cell_summaries=_summarize_cells(
            estimate_tuple,
            trial_by_id=trial_by_id,
        ),
        dose_responses=_dose_responses(estimate_tuple, trial_by_id=trial_by_id),
        linkage_dose_responses=_linkage_dose_responses(estimate_tuple),
    )


def _evaluate_world(
    frame: pd.DataFrame,
    *,
    source: pd.DataFrame,
    trial: RctQualificationTrialV1,
    world: RctWorldReceiptV1,
) -> tuple[RctWorldEstimateV1, ...]:
    required = {
        "world_id",
        "trial_id",
        "mode",
        "response_axis",
        "prognostic_scale",
        "treatment_scale",
        "linkage_retention",
        "treatment",
        "outcome",
        "age",
        "bmi",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(
            f"RCT world {world.world_id!r} is missing columns: {missing!r}."
        )
    if set(frame["world_id"].astype(str)) != {world.world_id}:
        raise ValueError("RCT world identity differs from its receipt.")
    group_columns = [
        "mode",
        "response_axis",
        "prognostic_scale",
        "treatment_scale",
        "linkage_retention",
    ]
    groups = tuple(frame.groupby(group_columns, sort=True, dropna=False))
    linkage_frames = {
        str(key[0]): generated.loc[
            :, ["treatment", "outcome", "age", "bmi"]
        ].reset_index(drop=True)
        for key, generated in groups
        if str(key[0])
        in {
            "whole_subject",
            "linkage_75",
            "linkage_50",
            "linkage_25",
            "independent_marginal",
        }
    }
    intact = linkage_frames.get("whole_subject")
    if intact is None:
        raise ValueError(
            f"RCT world {world.world_id!r} lacks its intact linkage control."
        )
    for linkage_mode, values in linkage_frames.items():
        _verify_equal_arm_marginals(intact, values, mode=linkage_mode)
    estimates: list[RctWorldEstimateV1] = []
    for (
        mode,
        axis,
        prognostic_scale,
        treatment_scale,
        linkage_retention,
    ), generated in groups:
        values = generated.loc[:, ["treatment", "outcome", "age", "bmi"]].reset_index(
            drop=True
        )
        if len(values) != trial.fitted_analysis.source_subjects:
            raise ValueError(
                f"RCT cell {world.world_id!r} does not preserve source sample size."
            )
        source_outcome = source["outcome"].to_numpy(dtype=float)
        generated_outcome = values["outcome"].to_numpy(dtype=float)
        source_outcome_sd = float(np.std(source_outcome, ddof=1))
        generated_outcome_sd = float(np.std(generated_outcome, ddof=1))
        if source_outcome_sd <= 0.0 or generated_outcome_sd <= 0.0:
            raise ValueError("RCT source and generated outcomes must vary.")
        adjusted = _fit(values, adjusted=True)
        unadjusted = _fit(values, adjusted=False)
        crosscheck_difference = _point_estimator_crosscheck(
            values,
            treatment_estimate=float(adjusted.params["treatment"]),
        )
        if str(axis) == "source_reference":
            truth = trial.fitted_analysis.analysis_treatment_effect
            prognostic_truth = 1.0
        else:
            truth, prognostic_truth = _mechanism_truth(
                values,
                source=trial.fitted_analysis,
                prognostic_scale=float(str(prognostic_scale)),
                treatment_scale=float(str(treatment_scale)),
            )
        estimates.append(
            RctWorldEstimateV1(
                trial_id=world.trial_id,
                world_id=world.world_id,
                world_index=world.world_index,
                mode=str(mode),
                response_axis=str(axis),
                prognostic_scale=float(str(prognostic_scale)),
                treatment_scale=float(str(treatment_scale)),
                linkage_retention=float(str(linkage_retention)),
                marginal_error=_marginal_error(source, values),
                dependence_error=_dependence_error(source, values),
                linkage_dependence_divergence=(
                    _dependence_divergence(intact, values)
                    if str(mode) in linkage_frames
                    else None
                ),
                source_outcome_mean=float(np.mean(source_outcome)),
                generated_outcome_mean=float(np.mean(generated_outcome)),
                standardized_outcome_mean_difference=float(
                    (np.mean(generated_outcome) - np.mean(source_outcome))
                    / source_outcome_sd
                ),
                generated_to_source_outcome_sd_ratio=generated_outcome_sd
                / source_outcome_sd,
                treatment_estimate=float(adjusted.params["treatment"]),
                treatment_standard_error=float(adjusted.bse["treatment"]),
                treatment_truth=truth,
                treatment_covered=bool(
                    adjusted.params["treatment"] - 1.96 * adjusted.bse["treatment"]
                    <= truth
                    <= adjusted.params["treatment"] + 1.96 * adjusted.bse["treatment"]
                ),
                adjusted_to_unadjusted_se_ratio=float(
                    adjusted.bse["treatment"] / unadjusted.bse["treatment"]
                ),
                prognostic_projection=_prognostic_projection(
                    adjusted, trial.fitted_analysis
                ),
                prognostic_truth_projection=prognostic_truth,
                point_estimator_crosscheck_absolute_difference=crosscheck_difference,
            )
        )
    return tuple(estimates)


def _fit(frame: pd.DataFrame, *, adjusted: bool) -> _FitResult:
    design = _analysis_design(frame, adjusted=adjusted)
    outcome = pd.to_numeric(frame["outcome"], errors="raise").astype("float64")
    return cast(_FitResult, sm.OLS(outcome, design).fit(cov_type="HC3"))


def _analysis_design(frame: pd.DataFrame, *, adjusted: bool) -> pd.DataFrame:
    treatment = frame["treatment"].astype("string").eq("active").astype("float64")
    design = pd.DataFrame({"intercept": 1.0, "treatment": treatment})
    if adjusted:
        design["age"] = pd.to_numeric(frame["age"], errors="raise")
        design["bmi"] = pd.to_numeric(frame["bmi"], errors="raise")
    return design


def _point_estimator_crosscheck(
    frame: pd.DataFrame,
    *,
    treatment_estimate: float,
) -> float:
    design = _analysis_design(frame, adjusted=True)
    outcome = pd.to_numeric(frame["outcome"], errors="raise").to_numpy(dtype=float)
    coefficients, _, rank, _ = lstsq(
        design.to_numpy(dtype=float),
        outcome,
        check_finite=True,
        lapack_driver="gelsd",
    )
    if rank != design.shape[1]:
        raise ValueError("RCT adjusted analysis design is rank deficient.")
    difference = abs(float(coefficients[1]) - treatment_estimate)
    if difference > 1e-9:
        raise ValueError(
            "Statsmodels and SciPy treatment point estimates differ by "
            f"{difference:.3g}."
        )
    return difference


def _marginal_error(source: pd.DataFrame, generated: pd.DataFrame) -> float:
    errors = []
    for column in ("outcome", "age", "bmi"):
        observed = source[column].to_numpy(dtype=float)
        simulated = generated[column].to_numpy(dtype=float)
        scale = float(np.std(observed, ddof=1))
        if scale <= 0.0:
            raise ValueError(f"RCT source variable {column!r} does not vary.")
        errors.append(float(wasserstein_distance(observed, simulated) / scale))
    return float(np.mean(errors))


def _dependence_error(source: pd.DataFrame, generated: pd.DataFrame) -> float:
    errors = []
    columns = ("outcome", "age", "bmi")
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            observed = float(spearmanr(source[left], source[right]).statistic)
            simulated = float(spearmanr(generated[left], generated[right]).statistic)
            if not math.isfinite(observed) or not math.isfinite(simulated):
                raise ValueError("RCT dependence correlation is not finite.")
            errors.append(abs(observed - simulated))
    return float(np.mean(errors))


def _dependence_divergence(reference: pd.DataFrame, generated: pd.DataFrame) -> float:
    errors = []
    columns = ("outcome", "age", "bmi")
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            intact = float(spearmanr(reference[left], reference[right]).statistic)
            disrupted = float(spearmanr(generated[left], generated[right]).statistic)
            if not math.isfinite(intact) or not math.isfinite(disrupted):
                raise ValueError("RCT paired dependence correlation is not finite.")
            errors.append(abs(intact - disrupted))
    return float(np.mean(errors))


def _verify_equal_arm_marginals(
    reference: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    mode: str,
) -> None:
    for treatment in ("control", "active"):
        intact = reference.loc[reference["treatment"].eq(treatment)]
        disrupted = generated.loc[generated["treatment"].eq(treatment)]
        for column in ("outcome", "age", "bmi"):
            expected = np.sort(intact[column].to_numpy(dtype=float))
            observed = np.sort(disrupted[column].to_numpy(dtype=float))
            if not np.array_equal(expected, observed):
                raise ValueError(
                    f"RCT linkage mode {mode!r} changed the {column!r} "
                    f"marginal for treatment arm {treatment!r}."
                )


def _prognostic_projection(fit: _FitResult, source: RctSourceFitV1) -> float:
    denominator = (
        source.analysis_age_coefficient**2 + source.analysis_bmi_coefficient**2
    )
    if denominator <= 1e-16:
        return 0.0
    return float(
        (
            float(fit.params["age"]) * source.analysis_age_coefficient
            + float(fit.params["bmi"]) * source.analysis_bmi_coefficient
        )
        / denominator
    )


def _mechanism_truth(
    frame: pd.DataFrame,
    *,
    source: RctSourceFitV1,
    prognostic_scale: float,
    treatment_scale: float,
) -> tuple[float, float]:
    treatment = (
        frame["treatment"].astype("string").eq("active").to_numpy(dtype=np.float64)
    )
    age = pd.to_numeric(frame["age"], errors="raise").to_numpy(dtype=np.float64)
    bmi = pd.to_numeric(frame["bmi"], errors="raise").to_numpy(dtype=np.float64)
    baseline_predictor = (
        source.intercept
        + source.age_coefficient * prognostic_scale * (age - source.age_center)
        + source.bmi_coefficient * prognostic_scale * (bmi - source.bmi_center)
    )
    treatment_shift = source.treatment_coefficient * treatment_scale
    if source.outcome_kind == "binary":
        untreated = expit(baseline_predictor)
        treated = expit(baseline_predictor + treatment_shift)
        treatment_truth = float(np.mean(treated - untreated))
        expected_outcome = untreated + treatment * (treated - untreated)
    else:
        treatment_truth = treatment_shift
        expected_outcome = baseline_predictor + treatment_shift * treatment
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "treatment": treatment,
            "age": age,
            "bmi": bmi,
        }
    )
    truth_fit = cast(_FitResult, sm.OLS(expected_outcome, design).fit())
    return treatment_truth, _prognostic_projection(truth_fit, source)


def _summarize_cells(
    estimates: tuple[RctWorldEstimateV1, ...],
    *,
    trial_by_id: dict[str, RctQualificationTrialV1],
) -> tuple[RctCellSummaryV1, ...]:
    grouped: dict[tuple[str, str, str, float, float], list[RctWorldEstimateV1]] = (
        defaultdict(list)
    )
    for row in estimates:
        grouped[
            (
                row.trial_id,
                row.mode,
                row.response_axis,
                row.prognostic_scale,
                row.treatment_scale,
            )
        ].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        biases = np.asarray(
            [row.treatment_estimate - row.treatment_truth for row in rows]
        )
        prognostic_biases = np.asarray(
            [
                row.prognostic_projection - row.prognostic_truth_projection
                for row in rows
            ]
        )
        outcome_mean_differences = np.asarray(
            [row.standardized_outcome_mean_difference for row in rows],
            dtype=float,
        )
        outcome_mean_interval = _mean_interval(outcome_mean_differences)
        outcome_sd_ratios = np.asarray(
            [row.generated_to_source_outcome_sd_ratio for row in rows],
            dtype=float,
        )
        outcome_sd_interval = _mean_interval(outcome_sd_ratios)
        bias = float(biases.mean())
        interval = _mean_interval(biases)
        covered = sum(row.treatment_covered for row in rows)
        coverage = proportion_interval(covered, len(rows))
        se_ratios = np.asarray(
            [row.adjusted_to_unadjusted_se_ratio for row in rows],
            dtype=float,
        )
        se_interval = np.quantile(se_ratios, [0.025, 0.975])
        source_se_ratio = trial_by_id[
            key[0]
        ].fitted_analysis.source_adjusted_to_unadjusted_se_ratio
        output.append(
            RctCellSummaryV1(
                trial_id=key[0],
                mode=key[1],
                response_axis=key[2],
                prognostic_scale=key[3],
                treatment_scale=key[4],
                worlds=len(rows),
                mean_marginal_error=float(
                    np.mean([row.marginal_error for row in rows])
                ),
                mean_dependence_error=float(
                    np.mean([row.dependence_error for row in rows])
                ),
                mean_standardized_outcome_mean_difference=float(
                    np.mean(outcome_mean_differences)
                ),
                standardized_outcome_mean_difference_ci_low=outcome_mean_interval[0],
                standardized_outcome_mean_difference_ci_high=outcome_mean_interval[1],
                mean_generated_to_source_outcome_sd_ratio=float(
                    np.mean(outcome_sd_ratios)
                ),
                generated_to_source_outcome_sd_ratio_ci_low=outcome_sd_interval[0],
                generated_to_source_outcome_sd_ratio_ci_high=outcome_sd_interval[1],
                treatment_bias=bias,
                treatment_bias_ci_low=interval[0],
                treatment_bias_ci_high=interval[1],
                treatment_coverage=covered / len(rows),
                treatment_coverage_ci_low=float(coverage[0]),
                treatment_coverage_ci_high=float(coverage[1]),
                mean_adjusted_to_unadjusted_se_ratio=float(np.mean(se_ratios)),
                adjusted_to_unadjusted_se_ratio_interval_low=float(se_interval[0]),
                adjusted_to_unadjusted_se_ratio_interval_high=float(se_interval[1]),
                source_adjusted_to_unadjusted_se_ratio=source_se_ratio,
                source_se_ratio_contained=bool(
                    se_interval[0] <= source_se_ratio <= se_interval[1]
                ),
                mean_prognostic_projection=float(
                    np.mean([row.prognostic_projection for row in rows])
                ),
                mean_prognostic_truth_projection=float(
                    np.mean([row.prognostic_truth_projection for row in rows])
                ),
                prognostic_projection_bias=float(np.mean(prognostic_biases)),
                maximum_point_estimator_crosscheck_difference=max(
                    row.point_estimator_crosscheck_absolute_difference for row in rows
                ),
            )
        )
    return tuple(output)


def _dose_responses(
    estimates: tuple[RctWorldEstimateV1, ...],
    *,
    trial_by_id: dict[str, RctQualificationTrialV1],
) -> tuple[RctDoseResponseV1, ...]:
    output = []
    for trial_id in sorted(trial_by_id):
        for axis in ("treatment", "prognostic"):
            rows = [
                row
                for row in estimates
                if row.trial_id == trial_id
                and row.mode == "source_anchored"
                and row.response_axis == axis
            ]
            by_world: dict[int, list[RctWorldEstimateV1]] = defaultdict(list)
            for row in rows:
                by_world[row.world_index].append(row)
            slopes = []
            for world_rows in by_world.values():
                x = np.asarray(
                    [
                        (
                            row.treatment_truth
                            if axis == "treatment"
                            else row.prognostic_truth_projection
                        )
                        for row in world_rows
                    ]
                )
                y = np.asarray(
                    [
                        (
                            row.treatment_estimate
                            if axis == "treatment"
                            else row.prognostic_projection
                        )
                        for row in world_rows
                    ]
                )
                if len(np.unique(x)) < 3:
                    raise ValueError(
                        f"RCT {axis} response requires at least three dose levels."
                    )
                slopes.append(float(np.polyfit(x, y, deg=1)[0]))
            expected = 1.0
            bias_values = np.asarray(slopes) - expected
            interval = _mean_interval(bias_values)
            directionally_concordant = np.asarray(slopes) > 0.0
            direction_interval = proportion_interval(
                int(np.sum(directionally_concordant)),
                len(directionally_concordant),
            )
            output.append(
                RctDoseResponseV1(
                    trial_id=trial_id,
                    response_axis=axis,
                    worlds=len(slopes),
                    expected_slope=expected,
                    mean_slope=float(np.mean(slopes)),
                    slope_bias=float(np.mean(bias_values)),
                    slope_bias_ci_low=interval[0],
                    slope_bias_ci_high=interval[1],
                    directionally_concordant_fraction=float(
                        np.mean(directionally_concordant)
                    ),
                    directionally_concordant_fraction_ci_low=direction_interval[0],
                    directionally_concordant_fraction_ci_high=direction_interval[1],
                )
            )
    return tuple(output)


def _linkage_dose_responses(
    estimates: tuple[RctWorldEstimateV1, ...],
) -> tuple[RctLinkageDoseResponseV1, ...]:
    output = []
    trial_ids = sorted({row.trial_id for row in estimates})
    linkage_modes = {
        "whole_subject",
        "linkage_75",
        "linkage_50",
        "linkage_25",
        "independent_marginal",
    }
    for trial_id in trial_ids:
        by_world: dict[int, list[RctWorldEstimateV1]] = defaultdict(list)
        for row in estimates:
            if row.trial_id == trial_id and row.mode in linkage_modes:
                by_world[row.world_index].append(row)
        dependence_slopes = []
        perturbation_slopes = []
        for rows in by_world.values():
            retention = np.asarray([row.linkage_retention for row in rows])
            if sorted(retention.tolist()) != [0.0, 0.25, 0.5, 0.75, 1.0]:
                raise ValueError(
                    f"{trial_id} linkage response does not contain all five retention levels."
                )
            intact = next(row for row in rows if row.linkage_retention == 1.0)
            if any(row.linkage_dependence_divergence is None for row in rows):
                raise ValueError(
                    f"{trial_id} linkage response lacks paired dependence divergence."
                )
            disruption = 1.0 - retention
            dependence_slopes.append(
                float(
                    np.polyfit(
                        disruption,
                        np.asarray([row.linkage_dependence_divergence for row in rows]),
                        deg=1,
                    )[0]
                )
            )
            perturbation_slopes.append(
                float(
                    np.polyfit(
                        disruption,
                        np.asarray(
                            [
                                abs(row.treatment_estimate - intact.treatment_estimate)
                                / intact.treatment_standard_error
                                for row in rows
                            ]
                        ),
                        deg=1,
                    )[0]
                )
            )
        dependence_interval = _mean_interval(np.asarray(dependence_slopes))
        output.append(
            RctLinkageDoseResponseV1(
                trial_id=trial_id,
                worlds=len(dependence_slopes),
                mean_dependence_divergence_slope=float(np.mean(dependence_slopes)),
                dependence_divergence_slope_ci_low=dependence_interval[0],
                dependence_divergence_slope_ci_high=dependence_interval[1],
                fraction_with_increasing_divergence=float(
                    np.mean(np.asarray(dependence_slopes) > 0.0)
                ),
                mean_analysis_perturbation_in_intact_se_slope=float(
                    np.mean(perturbation_slopes)
                ),
                fraction_with_increasing_perturbation=float(
                    np.mean(np.asarray(perturbation_slopes) > 0.0)
                ),
            )
        )
    return tuple(output)


def _load_source(source_root: Path, trial: RctQualificationTrialV1) -> pd.DataFrame:
    numeric_id = int(trial.trial_id.rsplit("-", 1)[1])
    data_path = source_root / "cleaned_data" / f"trial{numeric_id}.csv"
    dictionary_path = (
        source_root / "data_dictionary" / f"trial{numeric_id}_dictionary.csv"
    )
    if sha256_file(data_path) != trial.source_data_sha256:
        raise ValueError(f"{trial.trial_id} source data checksum mismatch.")
    if sha256_file(dictionary_path) != trial.source_dictionary_sha256:
        raise ValueError(f"{trial.trial_id} source dictionary checksum mismatch.")
    dictionary = pd.read_csv(dictionary_path)
    roles = {
        role: dictionary.loc[dictionary["variable_role"].eq(role), "variable_name"]
        .astype(str)
        .tolist()
        for role in ("Treatment assignment", "Primary outcome")
    }
    if any(len(values) != 1 for values in roles.values()):
        raise ValueError(
            f"{trial.trial_id} treatment and outcome roles must be unique."
        )
    covariates = dictionary.loc[
        dictionary["variable_role"].eq("Baseline covariate"), "variable_name"
    ].astype(str)
    age_pattern = re.compile(r"(?:^|_)age(?:_|$)", flags=re.IGNORECASE)
    bmi_pattern = re.compile(r"(?:^|_)(?:bmi|body_mass)(?:_|$)", flags=re.IGNORECASE)
    age = [value for value in covariates if age_pattern.search(value)]
    bmi = [value for value in covariates if bmi_pattern.search(value)]
    if len(age) != 1 or len(bmi) != 1:
        raise ValueError(f"{trial.trial_id} age and BMI mappings must be unique.")
    columns = [
        roles["Treatment assignment"][0],
        roles["Primary outcome"][0],
        age[0],
        bmi[0],
    ]
    frame = (
        pd.read_csv(data_path, usecols=columns)
        .dropna()
        .rename(
            columns={
                columns[0]: "treatment",
                columns[1]: "outcome",
                columns[2]: "age",
                columns[3]: "bmi",
            }
        )
    )
    levels = sorted(frame["treatment"].astype(str).unique())
    frame["treatment"] = (
        frame["treatment"]
        .astype(str)
        .eq(levels[1])
        .map({False: "control", True: "active"})
    )
    if trial.fitted_analysis.outcome_kind == "binary":
        outcome_levels = sorted(frame["outcome"].unique())
        frame["outcome"] = frame["outcome"].eq(outcome_levels[1]).astype("float64")
    return frame.reset_index(drop=True)


def _verify_source_fit(source: pd.DataFrame, declared: RctSourceFitV1) -> None:
    if len(source) != declared.source_subjects:
        raise ValueError(
            "RCT source complete-case count differs from the declared fit."
        )
    control = int(source["treatment"].eq("control").sum())
    active = int(source["treatment"].eq("active").sum())
    if (control, active) != (
        declared.source_control_subjects,
        declared.source_active_subjects,
    ):
        raise ValueError("RCT source arm allocation differs from the declared fit.")
    if not math.isclose(
        float(source["age"].mean()), declared.age_center, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("RCT source age center differs from the declared fit.")
    if not math.isclose(
        float(source["bmi"].mean()), declared.bmi_center, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("RCT source BMI center differs from the declared fit.")
    if declared.outcome_kind == "binary":
        if declared.source_event_rate is None or not math.isclose(
            float(source["outcome"].mean()),
            declared.source_event_rate,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("RCT source event rate differs from the declared fit.")
    elif declared.source_event_rate is not None:
        raise ValueError("Continuous RCT source fit must not declare an event rate.")
    centered = source.assign(
        age=source["age"] - declared.age_center,
        bmi=source["bmi"] - declared.bmi_center,
    )
    analysis_adjusted = _fit(centered, adjusted=True)
    analysis_unadjusted = _fit(centered, adjusted=False)
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "treatment": centered["treatment"].eq("active").astype("float64"),
            "age": centered["age"],
            "bmi": centered["bmi"],
        }
    )
    if declared.outcome_kind == "binary":
        generation_fit = cast(
            _FitResult,
            sm.GLM(source["outcome"], design, family=sm.families.Binomial()).fit(),
        )
    else:
        generation_fit = analysis_adjusted
    observed = np.asarray(
        [
            float(generation_fit.params["intercept"]),
            float(generation_fit.params["treatment"]),
            float(generation_fit.params["age"]),
            float(generation_fit.params["bmi"]),
            float(analysis_adjusted.params["treatment"]),
            float(analysis_adjusted.params["age"]),
            float(analysis_adjusted.params["bmi"]),
            float(analysis_adjusted.bse["treatment"]),
            float(analysis_unadjusted.bse["treatment"]),
        ]
    )
    expected = np.asarray(
        [
            declared.intercept,
            declared.treatment_coefficient,
            declared.age_coefficient,
            declared.bmi_coefficient,
            declared.analysis_treatment_effect,
            declared.analysis_age_coefficient,
            declared.analysis_bmi_coefficient,
            declared.source_adjusted_standard_error,
            declared.source_unadjusted_standard_error,
        ]
    )
    if not np.allclose(observed, expected, rtol=1e-10, atol=1e-10):
        raise ValueError(
            "RCT source regression does not reproduce the declared fitted analysis."
        )
    ratio = float(
        analysis_adjusted.bse["treatment"] / analysis_unadjusted.bse["treatment"]
    )
    if not math.isclose(
        ratio,
        declared.source_adjusted_to_unadjusted_se_ratio,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError(
            "RCT source standard-error ratio differs from the declared fitted analysis."
        )
    linear_predictor = design.to_numpy(dtype=float) @ observed[:4]
    fitted_values = (
        expit(linear_predictor)
        if declared.outcome_kind == "binary"
        else linear_predictor
    )
    residuals = source["outcome"].to_numpy(dtype=float) - fitted_values
    probabilities = (np.arange(len(residuals), dtype=float) + 0.5) / len(residuals)
    if not np.allclose(
        probabilities, declared.residual_probabilities, rtol=0.0, atol=1e-15
    ):
        raise ValueError(
            "RCT residual probabilities differ from the declared fitted analysis."
        )
    if not np.allclose(
        np.sort(residuals), declared.residual_quantiles, rtol=1e-10, atol=1e-10
    ):
        raise ValueError(
            "RCT residual quantiles differ from the declared fitted analysis."
        )


def _mean_interval(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("Repeated-world uncertainty requires at least two values.")
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    half_width = float(t.ppf(0.975, df=len(values) - 1) * sem)
    return mean - half_width, mean + half_width


def _canonical_json_sha(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    import hashlib

    return hashlib.sha256(canonical).hexdigest()


def _git_revision(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


__all__ = ["RctQualificationReportV1", "evaluate_rctbench_qualification"]
