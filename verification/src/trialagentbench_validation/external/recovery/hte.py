"""Independent verification of treatment-effect heterogeneity qualifications."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.linalg import lstsq
from scipy.special import expit
from scipy.stats import t

from trialagentbench_validation.external.recovery.rctbench import (
    RctQualificationTrialV1,
)
from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.statistics import proportion_interval


class _EstimationError(ValueError):
    """Expected non-estimability in one otherwise valid world."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HteCellV1(_FrozenModel):
    """One interaction dose and information level."""

    cell_id: str = Field(min_length=1)
    interaction_scale: float = Field(ge=0, allow_inf_nan=False)
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    worlds_per_trial: int = Field(ge=2)


class HteTrialV1(_FrozenModel):
    """One RCT source model and standardized interaction."""

    qualification: RctQualificationTrialV1
    age_standard_deviation: float = Field(gt=0, allow_inf_nan=False)
    base_standardized_interaction: float = Field(gt=0, allow_inf_nan=False)


class HteQualificationDesignV1(_FrozenModel):
    """Path-free HTE qualification design."""

    schema_id: Literal["trialagentbench.hte_qualification_design/v1"]
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    seed: int = Field(ge=0, le=2**32 - 1)
    trials: tuple[HteTrialV1, ...] = Field(min_length=3)
    cells: tuple[HteCellV1, ...] = Field(min_length=4)

    @model_validator(mode="after")
    def _complete(self) -> HteQualificationDesignV1:
        trial_ids = [trial.qualification.trial_id for trial in self.trials]
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(trial_ids) != len(set(trial_ids)) or len(cell_ids) != len(set(cell_ids)):
            raise ValueError("HTE trial and cell identities must be unique.")
        multipliers = sorted({cell.sample_size_multiplier for cell in self.cells})
        for multiplier in multipliers:
            levels = sorted(
                {
                    cell.interaction_scale
                    for cell in self.cells
                    if cell.sample_size_multiplier == multiplier
                }
            )
            if len(levels) < 4 or levels[0] != 0:
                raise ValueError(
                    "Every HTE information level requires four doses beginning at zero."
                )
        if len(multipliers) < 2:
            raise ValueError(
                "HTE qualification requires at least two information levels."
            )
        return self


class HteWorldReceiptV1(_FrozenModel):
    """Checksum-bound identity for one HTE world."""

    world_id: str = Field(pattern=r"^hte-[0-9a-f]{20}$")
    trial_id: str = Field(pattern=r"^RCTBENCH-[0-9]{3}$")
    cell_id: str = Field(min_length=1)
    world_index: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    subjects: int = Field(ge=40)
    analysis_path: str = Field(pattern=r"^worlds/hte-[0-9a-f]{20}\.parquet$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HteQualificationReceiptV1(_FrozenModel):
    """Complete HTE qualification inventory."""

    schema_id: Literal["trialagentbench.hte_qualification_receipt/v1"]
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[HteWorldReceiptV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> HteQualificationReceiptV1:
        keys = [(row.trial_id, row.cell_id, row.world_index) for row in self.worlds]
        if len(keys) != len(set(keys)):
            raise ValueError("HTE world identities must be unique.")
        return self


class HteWorldEstimateV1(_FrozenModel):
    """Independent interaction estimate for one HTE world."""

    world_id: str
    trial_id: str
    cell_id: str
    world_index: int
    outcome_kind: Literal["binary", "continuous"]
    interaction_scale: float = Field(ge=0, allow_inf_nan=False)
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    truth: float = Field(allow_inf_nan=False)
    estimate: float = Field(allow_inf_nan=False)
    standard_error: float = Field(gt=0, allow_inf_nan=False)
    classical_standard_error: float = Field(gt=0, allow_inf_nan=False)
    covered: bool
    rejected_null: bool
    point_estimator_crosscheck_absolute_difference: float = Field(
        ge=0,
        allow_inf_nan=False,
    )


class HteCellSummaryV1(_FrozenModel):
    """Repeated-world interaction operating characteristics."""

    trial_id: str
    cell_id: str
    outcome_kind: Literal["binary", "continuous"]
    interaction_scale: float = Field(ge=0, allow_inf_nan=False)
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    truth: float = Field(allow_inf_nan=False)
    worlds: int = Field(ge=2)
    successful_worlds: int = Field(ge=0)
    failures: int = Field(ge=0)
    bias: float | None = Field(default=None, allow_inf_nan=False)
    bias_ci_low: float | None = Field(default=None, allow_inf_nan=False)
    bias_ci_high: float | None = Field(default=None, allow_inf_nan=False)
    rmse: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    coverage: float | None = Field(default=None, ge=0, le=1)
    coverage_ci_low: float | None = Field(default=None, ge=0, le=1)
    coverage_ci_high: float | None = Field(default=None, ge=0, le=1)
    rejection_rate: float | None = Field(default=None, ge=0, le=1)
    rejection_rate_ci_low: float | None = Field(default=None, ge=0, le=1)
    rejection_rate_ci_high: float | None = Field(default=None, ge=0, le=1)
    model_to_empirical_se_ratio: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )
    maximum_point_estimator_crosscheck_difference: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )


class HteDoseResponseV1(_FrozenModel):
    """Within-world recovered-versus-configured interaction response."""

    trial_id: str
    outcome_kind: Literal["binary", "continuous"]
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    worlds: int = Field(ge=2)
    truth_levels: tuple[float, ...] = Field(min_length=4)
    mean_slope: float = Field(allow_inf_nan=False)
    slope_ci_low: float = Field(allow_inf_nan=False)
    slope_ci_high: float = Field(allow_inf_nan=False)


class HteInformationResponseV1(_FrozenModel):
    """Paired standard-error response to sample-size amplification."""

    trial_id: str
    outcome_kind: Literal["binary", "continuous"]
    interaction_scale: float = Field(ge=0, allow_inf_nan=False)
    lower_sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    higher_sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    worlds: int = Field(ge=2)
    asymptotic_standard_error_ratio: float = Field(gt=0, allow_inf_nan=False)
    geometric_mean_standard_error_ratio: float = Field(gt=0, allow_inf_nan=False)
    ratio_ci_low: float = Field(gt=0, allow_inf_nan=False)
    ratio_ci_high: float = Field(gt=0, allow_inf_nan=False)
    geometric_mean_classical_standard_error_ratio: float = Field(
        gt=0,
        allow_inf_nan=False,
    )
    classical_ratio_ci_low: float = Field(gt=0, allow_inf_nan=False)
    classical_ratio_ci_high: float = Field(gt=0, allow_inf_nan=False)


class HteQualificationReportV1(_FrozenModel):
    """Independent HTE qualification report."""

    schema_id: Literal["trialagentbench.hte_qualification_report/v1"] = (
        "trialagentbench.hte_qualification_report/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimates: tuple[HteWorldEstimateV1, ...]
    cell_summaries: tuple[HteCellSummaryV1, ...]
    dose_responses: tuple[HteDoseResponseV1, ...]
    information_responses: tuple[HteInformationResponseV1, ...]


def evaluate_hte_qualification(
    *,
    release_dir: Path,
    minimum_null_worlds: int = 100,
    minimum_nonnull_worlds: int = 50,
) -> HteQualificationReportV1:
    """Verify released HTE worlds and independently recover interactions."""

    if minimum_null_worlds < 2 or minimum_nonnull_worlds < 2:
        raise ValueError("HTE replication floors must be at least two.")
    design_path = release_dir / "design.json"
    receipt_path = release_dir / "receipt.json"
    payload = json.loads(design_path.read_text(encoding="utf-8"))
    design = HteQualificationDesignV1.model_validate(payload)
    design_sha256 = _json_sha(payload)
    receipt = HteQualificationReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    if receipt.design_sha256 != design_sha256:
        raise ValueError("HTE receipt does not bind its design.")
    trials = {row.qualification.trial_id: row for row in design.trials}
    cells = {row.cell_id: row for row in design.cells}
    expected = {
        (trial_id, cell.cell_id, world_index)
        for trial_id in trials
        for cell in design.cells
        for world_index in range(cell.worlds_per_trial)
    }
    observed = {(row.trial_id, row.cell_id, row.world_index) for row in receipt.worlds}
    if expected != observed:
        raise ValueError("HTE receipt does not contain the complete design.")
    for cell in design.cells:
        floor = (
            minimum_null_worlds
            if cell.interaction_scale == 0
            else minimum_nonnull_worlds
        )
        if cell.worlds_per_trial < floor:
            raise ValueError(f"HTE cell {cell.cell_id!r} misses its replication floor.")

    estimates = []
    failures: defaultdict[tuple[str, str], int] = defaultdict(int)
    for world in receipt.worlds:
        trial = trials[world.trial_id]
        cell = cells[world.cell_id]
        if world.seed != _world_seed(design.seed, world.trial_id, world.world_index):
            raise ValueError(f"HTE world seed mismatch: {world.world_id}.")
        if world.world_id != _world_id(
            design_sha256,
            world.trial_id,
            world.cell_id,
            world.world_index,
        ):
            raise ValueError(f"HTE world identity mismatch: {world.world_id}.")
        expected_subjects = int(
            round(
                trial.qualification.fitted_analysis.source_subjects
                * cell.sample_size_multiplier
            )
        )
        if world.subjects != expected_subjects:
            raise ValueError(f"HTE world subject count mismatch: {world.world_id}.")
        path = _release_path(release_dir, world.analysis_path)
        if sha256_file(path) != world.analysis_sha256:
            raise ValueError(f"HTE world checksum mismatch: {world.world_id}.")
        frame = _validate_world(
            pd.read_parquet(path),
            world=world,
            expected_subjects=expected_subjects,
            outcome_kind=trial.qualification.fitted_analysis.outcome_kind,
        )
        try:
            estimates.append(_fit_world(frame, world=world, trial=trial, cell=cell))
        except (np.linalg.LinAlgError, _EstimationError):
            failures[(world.trial_id, world.cell_id)] += 1
    return HteQualificationReportV1(
        design_sha256=design_sha256,
        receipt_sha256=sha256_file(receipt_path),
        estimates=tuple(estimates),
        cell_summaries=tuple(_summarize_cells(estimates, design, failures)),
        dose_responses=tuple(_dose_responses(estimates, design)),
        information_responses=tuple(_information_responses(estimates, design)),
    )


def _validate_world(
    frame: pd.DataFrame,
    *,
    world: HteWorldReceiptV1,
    expected_subjects: int,
    outcome_kind: Literal["binary", "continuous"],
) -> pd.DataFrame:
    required = {
        "world_id",
        "trial_id",
        "cell_id",
        "participant_id",
        "arm",
        "age",
        "bmi",
        "outcome",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"HTE world is missing columns: {missing!r}.")
    for column, expected in (
        ("world_id", world.world_id),
        ("trial_id", world.trial_id),
        ("cell_id", world.cell_id),
    ):
        if set(frame[column].astype(str)) != {expected}:
            raise ValueError(f"HTE world carries inconsistent {column}.")
    if (
        frame["participant_id"].nunique() != expected_subjects
        or len(frame) != expected_subjects
    ):
        raise ValueError("HTE world must contain one row per declared subject.")
    if set(frame["arm"].astype(str)) != {"control", "active"}:
        raise ValueError("HTE world must contain control and active arms.")
    for column in ("age", "bmi", "outcome"):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"HTE world {column!r} must be finite.")
    if outcome_kind == "binary" and set(frame["outcome"].astype(float)) - {0.0, 1.0}:
        raise ValueError("Binary HTE outcome must contain only zero and one.")
    return frame


def _fit_world(
    frame: pd.DataFrame,
    *,
    world: HteWorldReceiptV1,
    trial: HteTrialV1,
    cell: HteCellV1,
) -> HteWorldEstimateV1:
    fitted = trial.qualification.fitted_analysis
    treatment = frame["arm"].astype("string").eq("active").to_numpy(dtype=float)
    age = frame["age"].to_numpy(dtype=float)
    bmi = frame["bmi"].to_numpy(dtype=float)
    if fitted.outcome_kind == "binary":
        age_term = (age - age.mean()) / age.std(ddof=0)
    else:
        age_term = age - fitted.age_center
    design = np.column_stack(
        [
            np.ones(len(frame)),
            treatment,
            age_term,
            bmi - fitted.bmi_center,
            treatment * age_term,
        ]
    )
    outcome = frame["outcome"].to_numpy(dtype=float)
    classical_fit = sm.OLS(outcome, design).fit()
    fit = classical_fit.get_robustcov_results(cov_type="HC3")
    if fitted.outcome_kind == "binary":
        estimate = float(fit.params[4])
        standard_error = float(fit.bse[4])
        classical_standard_error = float(classical_fit.bse[4])
        mechanism_interaction = (
            trial.base_standardized_interaction * cell.interaction_scale
        )
        untreated_predictor = (
            fitted.intercept
            + fitted.age_coefficient * (age - fitted.age_center)
            + fitted.bmi_coefficient * (bmi - fitted.bmi_center)
        )
        untreated_risk = expit(untreated_predictor)
        treated_risk = expit(
            untreated_predictor
            + fitted.treatment_coefficient
            + mechanism_interaction * age_term
        )
        expected_outcome = untreated_risk + treatment * (treated_risk - untreated_risk)
        truth_coefficients, _, truth_rank, _ = lstsq(
            design,
            expected_outcome,
            check_finite=True,
            lapack_driver="gelsd",
        )
        if truth_rank != design.shape[1]:
            raise _EstimationError("Binary HTE truth projection is rank deficient.")
        truth = float(truth_coefficients[4])
        crosscheck = _linear_crosscheck(
            design,
            outcome,
            estimate=estimate,
            scale=1.0,
        )
    else:
        estimate = float(fit.params[4] * trial.age_standard_deviation)
        standard_error = float(fit.bse[4] * trial.age_standard_deviation)
        classical_standard_error = float(
            classical_fit.bse[4] * trial.age_standard_deviation
        )
        truth = trial.base_standardized_interaction * cell.interaction_scale
        crosscheck = _linear_crosscheck(
            design,
            outcome,
            estimate=estimate,
            scale=trial.age_standard_deviation,
        )
    if (
        not np.isfinite([estimate, standard_error, classical_standard_error]).all()
        or standard_error <= 0
        or classical_standard_error <= 0
    ):
        raise _EstimationError("HTE fit returned a non-finite interaction estimate.")
    return HteWorldEstimateV1(
        world_id=world.world_id,
        trial_id=world.trial_id,
        cell_id=world.cell_id,
        world_index=world.world_index,
        outcome_kind=fitted.outcome_kind,
        interaction_scale=cell.interaction_scale,
        sample_size_multiplier=cell.sample_size_multiplier,
        truth=truth,
        estimate=estimate,
        standard_error=standard_error,
        classical_standard_error=classical_standard_error,
        covered=estimate - 1.96 * standard_error
        <= truth
        <= estimate + 1.96 * standard_error,
        rejected_null=abs(estimate / standard_error) > 1.96,
        point_estimator_crosscheck_absolute_difference=crosscheck,
    )


def _linear_crosscheck(
    design: npt.NDArray[np.float64],
    outcome: npt.NDArray[np.float64],
    *,
    estimate: float,
    scale: float,
) -> float:
    coefficients, _, rank, _ = lstsq(
        design,
        outcome,
        check_finite=True,
        lapack_driver="gelsd",
    )
    if rank != design.shape[1]:
        raise _EstimationError("HTE linear analysis is rank deficient.")
    difference = abs(float(coefficients[4] * scale) - estimate)
    if difference > 1e-9:
        raise ValueError("Statsmodels and SciPy HTE estimates differ.")
    return difference


def _summarize_cells(
    estimates: list[HteWorldEstimateV1],
    design: HteQualificationDesignV1,
    failures: defaultdict[tuple[str, str], int],
) -> list[HteCellSummaryV1]:
    grouped: defaultdict[tuple[str, str], list[HteWorldEstimateV1]] = defaultdict(list)
    for row in estimates:
        grouped[(row.trial_id, row.cell_id)].append(row)
    output = []
    for trial in design.trials:
        trial_id = trial.qualification.trial_id
        outcome_kind = trial.qualification.fitted_analysis.outcome_kind
        for cell in design.cells:
            rows = grouped[(trial_id, cell.cell_id)]
            values = np.asarray([row.estimate for row in rows], dtype=float)
            ses = np.asarray([row.standard_error for row in rows], dtype=float)
            truth = (
                float(np.mean([row.truth for row in rows]))
                if rows
                else trial.base_standardized_interaction * cell.interaction_scale
            )
            bias: float | None
            low: float | None
            high: float | None
            rmse: float | None
            coverage: float | None
            rejection: float | None
            coverage_bounds: tuple[float | None, float | None]
            rejection_bounds: tuple[float | None, float | None]
            se_ratio: float | None
            maximum_difference: float | None
            if rows:
                biases = np.asarray(
                    [row.estimate - row.truth for row in rows],
                    dtype=float,
                )
                low, high = _mean_interval(biases)
                covered = np.asarray([row.covered for row in rows], dtype=bool)
                rejected = np.asarray([row.rejected_null for row in rows], dtype=bool)
                coverage_bounds = proportion_interval(int(covered.sum()), len(covered))
                rejection_bounds = proportion_interval(
                    int(rejected.sum()), len(rejected)
                )
                empirical_se = float(values.std(ddof=1))
                se_ratio = (
                    None
                    if empirical_se <= 0
                    else float(np.sqrt(np.mean(np.square(ses))) / empirical_se)
                )
                bias = float(biases.mean())
                rmse = float(np.sqrt(np.mean(np.square(biases))))
                coverage = float(covered.mean())
                rejection = float(rejected.mean())
                maximum_difference = max(
                    row.point_estimator_crosscheck_absolute_difference for row in rows
                )
            else:
                bias = low = high = rmse = coverage = rejection = None
                coverage_bounds = rejection_bounds = (None, None)
                se_ratio = maximum_difference = None
            output.append(
                HteCellSummaryV1(
                    trial_id=trial_id,
                    cell_id=cell.cell_id,
                    outcome_kind=outcome_kind,
                    interaction_scale=cell.interaction_scale,
                    sample_size_multiplier=cell.sample_size_multiplier,
                    truth=truth,
                    worlds=cell.worlds_per_trial,
                    successful_worlds=len(rows),
                    failures=failures[(trial_id, cell.cell_id)],
                    bias=bias,
                    bias_ci_low=low,
                    bias_ci_high=high,
                    rmse=rmse,
                    coverage=coverage,
                    coverage_ci_low=coverage_bounds[0],
                    coverage_ci_high=coverage_bounds[1],
                    rejection_rate=rejection,
                    rejection_rate_ci_low=rejection_bounds[0],
                    rejection_rate_ci_high=rejection_bounds[1],
                    model_to_empirical_se_ratio=se_ratio,
                    maximum_point_estimator_crosscheck_difference=maximum_difference,
                )
            )
    return output


def _dose_responses(
    estimates: list[HteWorldEstimateV1],
    design: HteQualificationDesignV1,
) -> list[HteDoseResponseV1]:
    output = []
    for trial in design.trials:
        trial_id = trial.qualification.trial_id
        outcome_kind = trial.qualification.fitted_analysis.outcome_kind
        for multiplier in sorted(
            {cell.sample_size_multiplier for cell in design.cells}
        ):
            rows = [
                row
                for row in estimates
                if row.trial_id == trial_id and row.sample_size_multiplier == multiplier
            ]
            by_world: defaultdict[int, list[HteWorldEstimateV1]] = defaultdict(list)
            for row in rows:
                by_world[row.world_index].append(row)
            slopes = []
            levels: tuple[float, ...] | None = None
            for world_rows in by_world.values():
                ordered = sorted(world_rows, key=lambda row: row.truth)
                world_levels = tuple(row.truth for row in ordered)
                if len(set(world_levels)) < 4:
                    continue
                slopes.append(
                    float(
                        np.polyfit(
                            world_levels,
                            [row.estimate for row in ordered],
                            1,
                        )[0]
                    )
                )
                levels = world_levels
            if len(slopes) < 2 or levels is None:
                continue
            values = np.asarray(slopes, dtype=float)
            low, high = _mean_interval(values)
            output.append(
                HteDoseResponseV1(
                    trial_id=trial_id,
                    outcome_kind=outcome_kind,
                    sample_size_multiplier=multiplier,
                    worlds=len(values),
                    truth_levels=levels,
                    mean_slope=float(values.mean()),
                    slope_ci_low=low,
                    slope_ci_high=high,
                )
            )
    return output


def _information_responses(
    estimates: list[HteWorldEstimateV1],
    design: HteQualificationDesignV1,
) -> list[HteInformationResponseV1]:
    multipliers = sorted({cell.sample_size_multiplier for cell in design.cells})
    lower, higher = multipliers[0], multipliers[-1]
    expected = float(np.sqrt(lower / higher))
    by_key = {
        (
            row.trial_id,
            row.interaction_scale,
            row.sample_size_multiplier,
            row.world_index,
        ): row
        for row in estimates
    }
    output = []
    for trial in design.trials:
        trial_id = trial.qualification.trial_id
        for scale in sorted({cell.interaction_scale for cell in design.cells}):
            ratios = []
            classical_ratios = []
            for world_index in range(
                min(
                    cell.worlds_per_trial
                    for cell in design.cells
                    if cell.interaction_scale == scale
                )
            ):
                low_row = by_key.get((trial_id, scale, lower, world_index))
                high_row = by_key.get((trial_id, scale, higher, world_index))
                if low_row is not None and high_row is not None:
                    ratios.append(high_row.standard_error / low_row.standard_error)
                    classical_ratios.append(
                        high_row.classical_standard_error
                        / low_row.classical_standard_error
                    )
            if len(ratios) < 2:
                continue
            values = np.asarray(ratios, dtype=float)
            log_values = np.log(values)
            log_low, log_high = _mean_interval(log_values)
            low_ci, high_ci = float(np.exp(log_low)), float(np.exp(log_high))
            classical_values = np.asarray(classical_ratios, dtype=float)
            classical_log_low, classical_log_high = _mean_interval(
                np.log(classical_values)
            )
            output.append(
                HteInformationResponseV1(
                    trial_id=trial_id,
                    outcome_kind=trial.qualification.fitted_analysis.outcome_kind,
                    interaction_scale=scale,
                    lower_sample_size_multiplier=lower,
                    higher_sample_size_multiplier=higher,
                    worlds=len(values),
                    asymptotic_standard_error_ratio=expected,
                    geometric_mean_standard_error_ratio=float(
                        np.exp(log_values.mean())
                    ),
                    ratio_ci_low=low_ci,
                    ratio_ci_high=high_ci,
                    geometric_mean_classical_standard_error_ratio=float(
                        np.exp(np.log(classical_values).mean())
                    ),
                    classical_ratio_ci_low=float(np.exp(classical_log_low)),
                    classical_ratio_ci_high=float(np.exp(classical_log_high)),
                )
            )
    return output


def _mean_interval(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("HTE interval requires at least two finite values.")
    half_width = float(
        t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    mean = float(values.mean())
    return mean - half_width, mean + half_width


def _release_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("HTE world path escapes the release directory.")
    return path


def _json_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _world_seed(seed: int, trial_id: str, world_index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{trial_id}:{world_index}".encode()).digest()[:4],
        "big",
    )


def _world_id(
    design_sha256: str,
    trial_id: str,
    cell_id: str,
    world_index: int,
) -> str:
    digest = hashlib.sha256(
        f"{design_sha256}:{trial_id}:{cell_id}:{world_index}".encode()
    ).hexdigest()
    return f"hte-{digest[:20]}"


__all__ = [
    "HteCellV1",
    "HteQualificationDesignV1",
    "HteQualificationReceiptV1",
    "HteTrialV1",
    "HteWorldReceiptV1",
    "evaluate_hte_qualification",
]
