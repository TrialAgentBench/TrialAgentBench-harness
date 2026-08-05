"""Independent verification of confounding and overlap qualification worlds."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import least_squares, minimize, root
from scipy.special import expit, logit
from scipy.stats import t
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationWarning,
)

from trialagentbench_validation.external.recovery.rctbench import (
    RctQualificationTrialV1,
)
from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.statistics import (
    proportion_interval,
    scale_aware_tolerance,
)

FailureReason = Literal[
    "no_variation",
    "statsmodels_fit",
    "scipy_fit",
    "estimator_disagreement",
    "invalid_weights",
]


class _EstimationError(ValueError):
    """Expected non-estimability in one otherwise valid world."""

    def __init__(self, reason: FailureReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConfoundingCellV1(_FrozenModel):
    """One assignment-strength and information-level cell."""

    cell_id: str = Field(min_length=1)
    assignment_strength: float = Field(ge=-3, le=3, allow_inf_nan=False)
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    worlds_per_trial: int = Field(ge=2)


class ConfoundingTrialV1(_FrozenModel):
    """One source trial and its baseline standardization constants."""

    qualification: RctQualificationTrialV1
    age_mean: float = Field(allow_inf_nan=False)
    age_standard_deviation: float = Field(gt=0, allow_inf_nan=False)
    bmi_mean: float = Field(allow_inf_nan=False)
    bmi_standard_deviation: float = Field(gt=0, allow_inf_nan=False)


class ConfoundingQualificationDesignV1(_FrozenModel):
    """Path-free production confounding qualification design."""

    schema_id: Literal["trialagentbench.confounding_qualification_design/v1"]
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    seed: int = Field(ge=0, le=2**32 - 1)
    outcome_intercept: float = Field(allow_inf_nan=False)
    exposure_log_odds_coefficient: float = Field(allow_inf_nan=False)
    age_log_odds_coefficient: float = Field(allow_inf_nan=False)
    bmi_log_odds_coefficient: float = Field(allow_inf_nan=False)
    propensity_minimum: float = Field(gt=0, lt=0.5)
    propensity_maximum: float = Field(gt=0.5, lt=1)
    estimator_absolute_tolerance: float = Field(default=1e-6, gt=0, allow_inf_nan=False)
    estimator_standard_error_fraction: float = Field(
        default=1e-3, gt=0, allow_inf_nan=False
    )
    trials: tuple[ConfoundingTrialV1, ...] = Field(min_length=3)
    cells: tuple[ConfoundingCellV1, ...] = Field(min_length=10)

    @model_validator(mode="after")
    def _complete(self) -> ConfoundingQualificationDesignV1:
        trial_ids = [row.qualification.trial_id for row in self.trials]
        cell_ids = [row.cell_id for row in self.cells]
        if len(trial_ids) != len(set(trial_ids)) or len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Confounding trial and cell identities must be unique.")
        multipliers = sorted({row.sample_size_multiplier for row in self.cells})
        if len(multipliers) < 2:
            raise ValueError(
                "Confounding qualification requires two information levels."
            )
        for multiplier in multipliers:
            strengths = sorted(
                {
                    row.assignment_strength
                    for row in self.cells
                    if row.sample_size_multiplier == multiplier
                }
            )
            if (
                len(strengths) < 5
                or 0.0 not in strengths
                or min(strengths) >= 0
                or max(strengths) <= 0
            ):
                raise ValueError(
                    "Every information level requires a null and bidirectional "
                    "five-level assignment response."
                )
        if self.propensity_minimum >= self.propensity_maximum:
            raise ValueError("Propensity bounds must be strictly ordered.")
        return self


class ConfoundingWorldReceiptV1(_FrozenModel):
    """Checksum-bound identity for one confounding world."""

    world_id: str = Field(pattern=r"^conf-[0-9a-f]{20}$")
    trial_id: str = Field(pattern=r"^RCTBENCH-[0-9]{3}$")
    cell_id: str = Field(min_length=1)
    world_index: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    subjects: int = Field(ge=40)
    analysis_path: str = Field(pattern=r"^worlds/conf-[0-9a-f]{20}\.parquet$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ConfoundingQualificationReceiptV1(_FrozenModel):
    """Complete inventory of confounding qualification worlds."""

    schema_id: Literal["trialagentbench.confounding_qualification_receipt/v1"]
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[ConfoundingWorldReceiptV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> ConfoundingQualificationReceiptV1:
        keys = [(row.trial_id, row.cell_id, row.world_index) for row in self.worlds]
        if len(keys) != len(set(keys)):
            raise ValueError("Confounding world identities must be unique.")
        return self


class ConfoundingWorldEstimateV1(_FrozenModel):
    """Independent assignment, outcome, and overlap estimates for one world."""

    world_id: str
    trial_id: str
    cell_id: str
    world_index: int
    assignment_strength: float = Field(allow_inf_nan=False)
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    propensity_truth: float = Field(allow_inf_nan=False)
    propensity_estimate: float = Field(allow_inf_nan=False)
    propensity_standard_error: float = Field(gt=0, allow_inf_nan=False)
    propensity_covered: bool
    naive_exposure_estimate: float = Field(allow_inf_nan=False)
    adjusted_exposure_estimate: float = Field(allow_inf_nan=False)
    adjusted_exposure_standard_error: float = Field(gt=0, allow_inf_nan=False)
    adjusted_exposure_covered: bool
    risk_difference_truth: float = Field(ge=-1, le=1, allow_inf_nan=False)
    oracle_ipw_risk_difference: float = Field(ge=-1, le=1, allow_inf_nan=False)
    estimated_ipw_risk_difference: float = Field(ge=-1, le=1, allow_inf_nan=False)
    exposed_fraction: float = Field(gt=0, lt=1, allow_inf_nan=False)
    extreme_propensity_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    oracle_effective_sample_fraction: float = Field(gt=0, le=1, allow_inf_nan=False)
    score_mean_difference: float = Field(allow_inf_nan=False)
    maximum_point_estimator_crosscheck_difference: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    maximum_point_estimator_crosscheck_tolerance: float = Field(
        gt=0,
        allow_inf_nan=False,
    )


class ConfoundingCellSummaryV1(_FrozenModel):
    """Repeated-world operating characteristics for one qualification cell."""

    trial_id: str
    cell_id: str
    assignment_strength: float = Field(allow_inf_nan=False)
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    worlds: int = Field(ge=2)
    successful_worlds: int = Field(ge=0)
    failures: int = Field(ge=0)
    propensity_bias: float | None = Field(default=None, allow_inf_nan=False)
    propensity_bias_ci_low: float | None = Field(default=None, allow_inf_nan=False)
    propensity_bias_ci_high: float | None = Field(default=None, allow_inf_nan=False)
    propensity_coverage_successful: float | None = Field(default=None, ge=0, le=1)
    propensity_coverage_successful_ci_low: float | None = Field(
        default=None, ge=0, le=1
    )
    propensity_coverage_successful_ci_high: float | None = Field(
        default=None, ge=0, le=1
    )
    propensity_coverage_scheduled: float = Field(ge=0, le=1)
    propensity_coverage_scheduled_ci_low: float = Field(ge=0, le=1)
    propensity_coverage_scheduled_ci_high: float = Field(ge=0, le=1)
    adjusted_bias: float | None = Field(default=None, allow_inf_nan=False)
    adjusted_bias_ci_low: float | None = Field(default=None, allow_inf_nan=False)
    adjusted_bias_ci_high: float | None = Field(default=None, allow_inf_nan=False)
    adjusted_rmse: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    adjusted_coverage_successful: float | None = Field(default=None, ge=0, le=1)
    adjusted_coverage_successful_ci_low: float | None = Field(default=None, ge=0, le=1)
    adjusted_coverage_successful_ci_high: float | None = Field(default=None, ge=0, le=1)
    adjusted_coverage_scheduled: float = Field(ge=0, le=1)
    adjusted_coverage_scheduled_ci_low: float = Field(ge=0, le=1)
    adjusted_coverage_scheduled_ci_high: float = Field(ge=0, le=1)
    maximum_coverage_denominator_gap: float | None = Field(default=None, ge=0, le=1)
    naive_bias: float | None = Field(default=None, allow_inf_nan=False)
    oracle_ipw_risk_difference_bias: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )
    estimated_ipw_risk_difference_bias: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )
    mean_extreme_propensity_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    mean_oracle_effective_sample_fraction: float | None = Field(
        default=None,
        gt=0,
        le=1,
        allow_inf_nan=False,
    )
    maximum_point_estimator_crosscheck_difference: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    maximum_point_estimator_crosscheck_tolerance: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def _consistent_denominators(self) -> ConfoundingCellSummaryV1:
        if self.successful_worlds + self.failures != self.worlds:
            raise ValueError("Confounding cell counts must partition scheduled worlds.")
        return self


class ConfoundingDoseResponseV1(_FrozenModel):
    """Within-world response to graded covariate-dependent assignment."""

    trial_id: str
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    response: Literal[
        "propensity_coefficient",
        "naive_exposure_bias",
        "adjusted_exposure_bias",
        "score_imbalance",
    ]
    worlds: int = Field(ge=2)
    assignment_strengths: tuple[float, ...] = Field(min_length=5)
    mean_slope: float = Field(allow_inf_nan=False)
    slope_ci_low: float = Field(allow_inf_nan=False)
    slope_ci_high: float = Field(allow_inf_nan=False)


class OverlapResponseV1(_FrozenModel):
    """Within-world response to absolute assignment strength."""

    trial_id: str
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    response: Literal[
        "extreme_propensity_fraction",
        "effective_sample_loss",
    ]
    worlds: int = Field(ge=2)
    absolute_strengths: tuple[float, ...] = Field(min_length=3)
    mean_slope: float = Field(allow_inf_nan=False)
    slope_ci_low: float = Field(allow_inf_nan=False)
    slope_ci_high: float = Field(allow_inf_nan=False)


class ConfoundingInformationResponseV1(_FrozenModel):
    """Paired precision response from source size to fourfold information."""

    trial_id: str
    assignment_strength: float = Field(allow_inf_nan=False)
    worlds: int = Field(ge=2)
    asymptotic_standard_error_ratio: float = Field(gt=0, allow_inf_nan=False)
    geometric_mean_standard_error_ratio: float = Field(gt=0, allow_inf_nan=False)
    ratio_ci_low: float = Field(gt=0, allow_inf_nan=False)
    ratio_ci_high: float = Field(gt=0, allow_inf_nan=False)


class ConfoundingFailureV1(_FrozenModel):
    """Reason-coded failed analysis for one otherwise valid world."""

    world_id: str
    trial_id: str
    cell_id: str
    world_index: int
    reason: FailureReason


class ConfoundingQualificationReportV1(_FrozenModel):
    """Independent report for production confounding qualification."""

    schema_id: Literal["trialagentbench.confounding_qualification_report/v1"] = (
        "trialagentbench.confounding_qualification_report/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimates: tuple[ConfoundingWorldEstimateV1, ...]
    cell_summaries: tuple[ConfoundingCellSummaryV1, ...]
    dose_responses: tuple[ConfoundingDoseResponseV1, ...]
    overlap_responses: tuple[OverlapResponseV1, ...]
    information_responses: tuple[ConfoundingInformationResponseV1, ...]
    failures: tuple[ConfoundingFailureV1, ...]


def evaluate_confounding_qualification(
    *,
    release_dir: Path,
    minimum_null_worlds: int = 100,
    minimum_nonnull_worlds: int = 50,
) -> ConfoundingQualificationReportV1:
    """Verify released worlds and recover confounding and overlap properties."""

    if minimum_null_worlds < 2 or minimum_nonnull_worlds < 2:
        raise ValueError("Confounding replication floors must be at least two.")
    design_path = release_dir / "design.json"
    receipt_path = release_dir / "receipt.json"
    payload = json.loads(design_path.read_text(encoding="utf-8"))
    design = ConfoundingQualificationDesignV1.model_validate(payload)
    design_sha256 = _json_sha(payload)
    receipt = ConfoundingQualificationReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    if receipt.design_sha256 != design_sha256:
        raise ValueError("Confounding receipt does not bind its design.")
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
        raise ValueError("Confounding receipt does not contain the complete design.")
    for cell in design.cells:
        floor = (
            minimum_null_worlds
            if cell.assignment_strength == 0
            else minimum_nonnull_worlds
        )
        if cell.worlds_per_trial < floor:
            raise ValueError(
                f"Confounding cell {cell.cell_id!r} misses its replication floor."
            )

    estimates: list[ConfoundingWorldEstimateV1] = []
    failures: defaultdict[tuple[str, str], int] = defaultdict(int)
    failure_records: list[ConfoundingFailureV1] = []
    for world in receipt.worlds:
        trial = trials[world.trial_id]
        cell = cells[world.cell_id]
        if world.seed != _world_seed(
            design.seed,
            world.trial_id,
            world.world_index,
        ):
            raise ValueError(f"Confounding world seed mismatch: {world.world_id}.")
        if world.world_id != _world_id(
            design_sha256,
            world.trial_id,
            world.cell_id,
            world.world_index,
        ):
            raise ValueError(f"Confounding world identity mismatch: {world.world_id}.")
        expected_subjects = int(
            round(
                trial.qualification.fitted_analysis.source_subjects
                * cell.sample_size_multiplier
            )
        )
        if world.subjects != expected_subjects:
            raise ValueError(f"Confounding subject count mismatch: {world.world_id}.")
        path = _release_path(release_dir, world.analysis_path)
        if sha256_file(path) != world.analysis_sha256:
            raise ValueError(f"Confounding world checksum mismatch: {world.world_id}.")
        frame = _validate_world(
            pd.read_parquet(path),
            world=world,
            expected_subjects=expected_subjects,
        )
        try:
            estimates.append(
                _fit_world(
                    frame,
                    world=world,
                    trial=trial,
                    cell=cell,
                    design=design,
                )
            )
        except np.linalg.LinAlgError:
            failures[(world.trial_id, world.cell_id)] += 1
            failure_records.append(
                ConfoundingFailureV1(
                    world_id=world.world_id,
                    trial_id=world.trial_id,
                    cell_id=world.cell_id,
                    world_index=world.world_index,
                    reason="statsmodels_fit",
                )
            )
        except _EstimationError as exc:
            failures[(world.trial_id, world.cell_id)] += 1
            failure_records.append(
                ConfoundingFailureV1(
                    world_id=world.world_id,
                    trial_id=world.trial_id,
                    cell_id=world.cell_id,
                    world_index=world.world_index,
                    reason=exc.reason,
                )
            )
    return ConfoundingQualificationReportV1(
        design_sha256=design_sha256,
        receipt_sha256=sha256_file(receipt_path),
        estimates=tuple(estimates),
        cell_summaries=tuple(_summarize_cells(estimates, design, failures)),
        dose_responses=tuple(_dose_responses(estimates, design)),
        overlap_responses=tuple(_overlap_responses(estimates, design)),
        information_responses=tuple(_information_responses(estimates, design)),
        failures=tuple(failure_records),
    )


def _validate_world(
    frame: pd.DataFrame,
    *,
    world: ConfoundingWorldReceiptV1,
    expected_subjects: int,
) -> pd.DataFrame:
    required = {
        "world_id",
        "trial_id",
        "cell_id",
        "participant_id",
        "age",
        "bmi",
        "exposed",
        "outcome",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Confounding world is missing columns: {missing!r}.")
    for column, expected in (
        ("world_id", world.world_id),
        ("trial_id", world.trial_id),
        ("cell_id", world.cell_id),
    ):
        if set(frame[column].astype(str)) != {expected}:
            raise ValueError(f"Confounding world carries inconsistent {column}.")
    if (
        len(frame) != expected_subjects
        or frame["participant_id"].nunique() != expected_subjects
    ):
        raise ValueError("Confounding world must contain one row per declared subject.")
    for column in ("age", "bmi", "exposed", "outcome"):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Confounding world {column!r} must be finite.")
    for column in ("exposed", "outcome"):
        if set(frame[column].astype(float)) - {0.0, 1.0}:
            raise ValueError(f"Confounding world {column!r} must be binary.")
    return frame


def _fit_world(
    frame: pd.DataFrame,
    *,
    world: ConfoundingWorldReceiptV1,
    trial: ConfoundingTrialV1,
    cell: ConfoundingCellV1,
    design: ConfoundingQualificationDesignV1,
) -> ConfoundingWorldEstimateV1:
    age_z = (
        frame["age"].to_numpy(dtype=float) - trial.age_mean
    ) / trial.age_standard_deviation
    bmi_z = (
        frame["bmi"].to_numpy(dtype=float) - trial.bmi_mean
    ) / trial.bmi_standard_deviation
    score = (age_z + bmi_z) / np.sqrt(2.0)
    exposed = frame["exposed"].to_numpy(dtype=float)
    outcome = frame["outcome"].to_numpy(dtype=float)
    if len(np.unique(exposed)) != 2 or len(np.unique(outcome)) != 2:
        raise _EstimationError(
            "no_variation",
            "Confounding estimation requires exposure and outcome variation.",
        )
    propensity = np.clip(
        expit(cell.assignment_strength * score),
        design.propensity_minimum,
        design.propensity_maximum,
    )
    propensity_design = np.column_stack([np.ones(len(frame)), score])
    propensity_truth = _scipy_logistic(
        propensity_design,
        propensity,
        context="propensity projection",
    )[1]
    propensity_fit = _statsmodels_logistic(
        propensity_design,
        exposed,
        context="propensity",
    )
    outcome_design = np.column_stack([np.ones(len(frame)), exposed, age_z, bmi_z])
    adjusted_fit = _statsmodels_logistic(
        outcome_design,
        outcome,
        context="adjusted outcome",
    )
    naive_fit = _statsmodels_logistic(
        np.column_stack([np.ones(len(frame)), exposed]),
        outcome,
        context="naive outcome",
    )
    propensity_crosscheck, propensity_tolerance = _crosscheck_logistic(
        propensity_design,
        exposed,
        reference=propensity_fit,
        coefficient_index=1,
        absolute_tolerance=design.estimator_absolute_tolerance,
        standard_error_fraction=design.estimator_standard_error_fraction,
        context="propensity",
    )
    adjusted_crosscheck, adjusted_tolerance = _crosscheck_logistic(
        outcome_design,
        outcome,
        reference=adjusted_fit,
        coefficient_index=1,
        absolute_tolerance=design.estimator_absolute_tolerance,
        standard_error_fraction=design.estimator_standard_error_fraction,
        context="adjusted outcome",
    )
    maximum_difference = max(propensity_crosscheck, adjusted_crosscheck)
    maximum_tolerance = max(propensity_tolerance, adjusted_tolerance)

    untreated_risk = expit(
        design.outcome_intercept
        + design.age_log_odds_coefficient * age_z
        + design.bmi_log_odds_coefficient * bmi_z
    )
    treated_risk = expit(logit(untreated_risk) + design.exposure_log_odds_coefficient)
    risk_difference_truth = float(np.mean(treated_risk - untreated_risk))
    estimated_propensity = np.clip(
        expit(propensity_design @ propensity_fit.params),
        design.propensity_minimum,
        design.propensity_maximum,
    )
    oracle_weights = exposed / propensity + (1.0 - exposed) / (1.0 - propensity)
    estimated_weights = exposed / estimated_propensity + (1.0 - exposed) / (
        1.0 - estimated_propensity
    )
    return ConfoundingWorldEstimateV1(
        world_id=world.world_id,
        trial_id=world.trial_id,
        cell_id=world.cell_id,
        world_index=world.world_index,
        assignment_strength=cell.assignment_strength,
        sample_size_multiplier=cell.sample_size_multiplier,
        propensity_truth=float(propensity_truth),
        propensity_estimate=float(propensity_fit.params[1]),
        propensity_standard_error=float(propensity_fit.standard_errors[1]),
        propensity_covered=bool(
            propensity_fit.params[1] - 1.96 * propensity_fit.standard_errors[1]
            <= propensity_truth
            <= propensity_fit.params[1] + 1.96 * propensity_fit.standard_errors[1]
        ),
        naive_exposure_estimate=float(naive_fit.params[1]),
        adjusted_exposure_estimate=float(adjusted_fit.params[1]),
        adjusted_exposure_standard_error=float(adjusted_fit.standard_errors[1]),
        adjusted_exposure_covered=bool(
            adjusted_fit.params[1] - 1.96 * adjusted_fit.standard_errors[1]
            <= design.exposure_log_odds_coefficient
            <= adjusted_fit.params[1] + 1.96 * adjusted_fit.standard_errors[1]
        ),
        risk_difference_truth=risk_difference_truth,
        oracle_ipw_risk_difference=_weighted_risk_difference(
            outcome,
            exposed,
            oracle_weights,
        ),
        estimated_ipw_risk_difference=_weighted_risk_difference(
            outcome,
            exposed,
            estimated_weights,
        ),
        exposed_fraction=float(exposed.mean()),
        extreme_propensity_fraction=float(
            np.mean((propensity < 0.1) | (propensity > 0.9))
        ),
        oracle_effective_sample_fraction=_effective_sample_fraction(oracle_weights),
        score_mean_difference=float(
            score[exposed == 1].mean() - score[exposed == 0].mean()
        ),
        maximum_point_estimator_crosscheck_difference=maximum_difference,
        maximum_point_estimator_crosscheck_tolerance=maximum_tolerance,
    )


class _LogisticFit:
    def __init__(
        self,
        *,
        params: npt.NDArray[np.float64],
        standard_errors: npt.NDArray[np.float64],
    ) -> None:
        self.params = params
        self.standard_errors = standard_errors


def _statsmodels_logistic(
    design: npt.NDArray[np.float64],
    outcome: npt.NDArray[np.float64],
    *,
    context: str,
) -> _LogisticFit:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", PerfectSeparationWarning)
        warnings.simplefilter("error", RuntimeWarning)
        try:
            fit = sm.GLM(
                outcome,
                design,
                family=sm.families.Binomial(),
            ).fit(cov_type="HC3", maxiter=100)
        except (
            ConvergenceWarning,
            PerfectSeparationWarning,
            RuntimeWarning,
            ValueError,
        ) as exc:
            raise _EstimationError(
                "statsmodels_fit",
                f"{context} logistic fit failed.",
            ) from exc
    if not bool(fit.converged):
        raise _EstimationError(
            "statsmodels_fit",
            f"{context} logistic fit did not converge.",
        )
    params = np.asarray(fit.params, dtype=float)
    standard_errors = np.asarray(fit.bse, dtype=float)
    if (
        not np.isfinite(params).all()
        or not np.isfinite(standard_errors).all()
        or np.any(standard_errors <= 0)
    ):
        raise _EstimationError(
            "statsmodels_fit",
            f"{context} logistic fit returned invalid values.",
        )
    return _LogisticFit(params=params, standard_errors=standard_errors)


def _scipy_logistic(
    design: npt.NDArray[np.float64],
    outcome: npt.NDArray[np.float64],
    *,
    context: str,
    initial: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    def objective(beta: npt.NDArray[np.float64]) -> float:
        linear = design @ beta
        return float(np.sum(np.logaddexp(0.0, linear) - outcome * linear))

    def gradient(beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(
            design.T @ (expit(design @ beta) - outcome),
            dtype=float,
        )

    def hessian(beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        probabilities = expit(design @ beta)
        variances = probabilities * (1.0 - probabilities)
        return np.asarray(
            design.T @ (design * variances[:, None]),
            dtype=float,
        )

    if initial is None:
        initial = np.zeros(design.shape[1], dtype=float)
    if initial.shape != (design.shape[1],) or not np.isfinite(initial).all():
        raise ValueError("SciPy logistic initial parameters must match the design.")
    first = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=gradient,
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1_000},
    )
    start = (
        np.asarray(first.x, dtype=float)
        if np.isfinite(first.x).all()
        else np.zeros(design.shape[1], dtype=float)
    )
    polished = minimize(
        objective,
        start,
        method="trust-ncg",
        jac=gradient,
        hess=hessian,
        options={"gtol": 1e-10, "maxiter": 250},
    )
    if polished.success and np.isfinite(polished.x).all():
        return np.asarray(polished.x, dtype=float)
    if first.success and np.isfinite(first.x).all():
        start = np.asarray(first.x, dtype=float)
    solved = root(
        gradient,
        start,
        method="hybr",
        jac=hessian,
        tol=1e-10,
    )
    if solved.success and np.isfinite(solved.x).all():
        return np.asarray(solved.x, dtype=float)
    least_squares_fit = least_squares(
        gradient,
        start,
        jac=hessian,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=1_000,
    )
    if (
        least_squares_fit.success
        and np.isfinite(least_squares_fit.x).all()
        and float(np.linalg.norm(gradient(least_squares_fit.x), ord=np.inf)) <= 1e-7
    ):
        return np.asarray(least_squares_fit.x, dtype=float)
    if float(np.linalg.norm(gradient(start), ord=np.inf)) <= 1e-7:
        return start
    raise _EstimationError(
        "scipy_fit",
        f"{context} SciPy logistic fit failed.",
    )


def _crosscheck_logistic(
    design: npt.NDArray[np.float64],
    outcome: npt.NDArray[np.float64],
    *,
    reference: _LogisticFit,
    coefficient_index: int,
    absolute_tolerance: float,
    standard_error_fraction: float,
    context: str,
) -> tuple[float, float]:
    tolerance = scale_aware_tolerance(
        float(reference.standard_errors[coefficient_index]),
        absolute_tolerance=absolute_tolerance,
        standard_error_fraction=standard_error_fraction,
    )
    independent = _scipy_logistic(
        design,
        outcome,
        context=f"{context} cross-check",
    )
    difference = abs(
        float(independent[coefficient_index] - reference.params[coefficient_index])
    )
    if difference <= tolerance:
        return difference, tolerance
    refined = _scipy_logistic(
        design,
        outcome,
        context=f"{context} cross-check refinement",
        initial=reference.params,
    )
    difference = abs(
        float(refined[coefficient_index] - reference.params[coefficient_index])
    )
    if difference > tolerance:
        raise _EstimationError(
            "estimator_disagreement",
            f"Independent {context} logistic point estimates disagree after refinement.",
        )
    return difference, tolerance


def _weighted_risk_difference(
    outcome: npt.NDArray[np.float64],
    exposed: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> float:
    treated = exposed == 1
    control = ~treated
    if not treated.any() or not control.any():
        raise _EstimationError(
            "no_variation",
            "IPW contrast requires both exposure groups.",
        )
    treated_mean = float(np.average(outcome[treated], weights=weights[treated]))
    control_mean = float(np.average(outcome[control], weights=weights[control]))
    return treated_mean - control_mean


def _effective_sample_fraction(weights: npt.NDArray[np.float64]) -> float:
    value = float(np.square(weights.sum()) / np.square(weights).sum() / len(weights))
    if not np.isfinite(value) or value <= 0 or value > 1 + 1e-12:
        raise _EstimationError(
            "invalid_weights",
            "IPW effective sample fraction is invalid.",
        )
    return min(value, 1.0)


def _summarize_cells(
    estimates: list[ConfoundingWorldEstimateV1],
    design: ConfoundingQualificationDesignV1,
    failures: defaultdict[tuple[str, str], int],
) -> list[ConfoundingCellSummaryV1]:
    grouped: defaultdict[
        tuple[str, str],
        list[ConfoundingWorldEstimateV1],
    ] = defaultdict(list)
    for row in estimates:
        grouped[(row.trial_id, row.cell_id)].append(row)
    output = []
    for trial in design.trials:
        trial_id = trial.qualification.trial_id
        for cell in design.cells:
            rows = grouped[(trial_id, cell.cell_id)]
            if rows:
                propensity_biases = np.asarray(
                    [row.propensity_estimate - row.propensity_truth for row in rows],
                    dtype=float,
                )
                adjusted_biases = np.asarray(
                    [
                        row.adjusted_exposure_estimate
                        - design.exposure_log_odds_coefficient
                        for row in rows
                    ],
                    dtype=float,
                )
                propensity_low, propensity_high = _mean_interval(propensity_biases)
                adjusted_low, adjusted_high = _mean_interval(adjusted_biases)
                propensity_covered = sum(row.propensity_covered for row in rows)
                adjusted_covered = sum(row.adjusted_exposure_covered for row in rows)
                propensity_successful = propensity_covered / len(rows)
                adjusted_successful = adjusted_covered / len(rows)
                propensity_successful_interval = proportion_interval(
                    propensity_covered, len(rows)
                )
                adjusted_successful_interval = proportion_interval(
                    adjusted_covered, len(rows)
                )
                propensity_scheduled = propensity_covered / cell.worlds_per_trial
                adjusted_scheduled = adjusted_covered / cell.worlds_per_trial
                propensity_scheduled_interval = proportion_interval(
                    propensity_covered, cell.worlds_per_trial
                )
                adjusted_scheduled_interval = proportion_interval(
                    adjusted_covered, cell.worlds_per_trial
                )
                values: dict[str, float | None] = {
                    "propensity_bias": float(propensity_biases.mean()),
                    "propensity_bias_ci_low": propensity_low,
                    "propensity_bias_ci_high": propensity_high,
                    "propensity_coverage_successful": propensity_successful,
                    "propensity_coverage_successful_ci_low": propensity_successful_interval[
                        0
                    ],
                    "propensity_coverage_successful_ci_high": propensity_successful_interval[
                        1
                    ],
                    "propensity_coverage_scheduled": propensity_scheduled,
                    "propensity_coverage_scheduled_ci_low": propensity_scheduled_interval[
                        0
                    ],
                    "propensity_coverage_scheduled_ci_high": propensity_scheduled_interval[
                        1
                    ],
                    "adjusted_bias": float(adjusted_biases.mean()),
                    "adjusted_bias_ci_low": adjusted_low,
                    "adjusted_bias_ci_high": adjusted_high,
                    "adjusted_rmse": float(
                        np.sqrt(np.mean(np.square(adjusted_biases)))
                    ),
                    "adjusted_coverage_successful": adjusted_successful,
                    "adjusted_coverage_successful_ci_low": adjusted_successful_interval[
                        0
                    ],
                    "adjusted_coverage_successful_ci_high": adjusted_successful_interval[
                        1
                    ],
                    "adjusted_coverage_scheduled": adjusted_scheduled,
                    "adjusted_coverage_scheduled_ci_low": adjusted_scheduled_interval[
                        0
                    ],
                    "adjusted_coverage_scheduled_ci_high": adjusted_scheduled_interval[
                        1
                    ],
                    "maximum_coverage_denominator_gap": max(
                        propensity_successful - propensity_scheduled,
                        adjusted_successful - adjusted_scheduled,
                    ),
                    "naive_bias": float(
                        np.mean(
                            [
                                row.naive_exposure_estimate
                                - design.exposure_log_odds_coefficient
                                for row in rows
                            ]
                        )
                    ),
                    "oracle_ipw_risk_difference_bias": float(
                        np.mean(
                            [
                                row.oracle_ipw_risk_difference
                                - row.risk_difference_truth
                                for row in rows
                            ]
                        )
                    ),
                    "estimated_ipw_risk_difference_bias": float(
                        np.mean(
                            [
                                row.estimated_ipw_risk_difference
                                - row.risk_difference_truth
                                for row in rows
                            ]
                        )
                    ),
                    "mean_extreme_propensity_fraction": float(
                        np.mean([row.extreme_propensity_fraction for row in rows])
                    ),
                    "mean_oracle_effective_sample_fraction": float(
                        np.mean([row.oracle_effective_sample_fraction for row in rows])
                    ),
                    "maximum_point_estimator_crosscheck_difference": max(
                        row.maximum_point_estimator_crosscheck_difference
                        for row in rows
                    ),
                    "maximum_point_estimator_crosscheck_tolerance": max(
                        row.maximum_point_estimator_crosscheck_tolerance for row in rows
                    ),
                }
            else:
                scheduled_interval = proportion_interval(0, cell.worlds_per_trial)
                values = {
                    key: None
                    for key in (
                        "propensity_bias",
                        "propensity_bias_ci_low",
                        "propensity_bias_ci_high",
                        "propensity_coverage_successful",
                        "propensity_coverage_successful_ci_low",
                        "propensity_coverage_successful_ci_high",
                        "adjusted_bias",
                        "adjusted_bias_ci_low",
                        "adjusted_bias_ci_high",
                        "adjusted_rmse",
                        "adjusted_coverage_successful",
                        "adjusted_coverage_successful_ci_low",
                        "adjusted_coverage_successful_ci_high",
                        "maximum_coverage_denominator_gap",
                        "naive_bias",
                        "oracle_ipw_risk_difference_bias",
                        "estimated_ipw_risk_difference_bias",
                        "mean_extreme_propensity_fraction",
                        "mean_oracle_effective_sample_fraction",
                        "maximum_point_estimator_crosscheck_difference",
                        "maximum_point_estimator_crosscheck_tolerance",
                    )
                }
                values.update(
                    {
                        "propensity_coverage_scheduled": 0.0,
                        "propensity_coverage_scheduled_ci_low": scheduled_interval[0],
                        "propensity_coverage_scheduled_ci_high": scheduled_interval[1],
                        "adjusted_coverage_scheduled": 0.0,
                        "adjusted_coverage_scheduled_ci_low": scheduled_interval[0],
                        "adjusted_coverage_scheduled_ci_high": scheduled_interval[1],
                    }
                )
            output.append(
                ConfoundingCellSummaryV1(
                    trial_id=trial_id,
                    cell_id=cell.cell_id,
                    assignment_strength=cell.assignment_strength,
                    sample_size_multiplier=cell.sample_size_multiplier,
                    worlds=cell.worlds_per_trial,
                    successful_worlds=len(rows),
                    failures=failures[(trial_id, cell.cell_id)],
                    **values,
                )
            )
    return output


def _dose_responses(
    estimates: list[ConfoundingWorldEstimateV1],
    design: ConfoundingQualificationDesignV1,
) -> list[ConfoundingDoseResponseV1]:
    output = []
    for trial in design.trials:
        trial_id = trial.qualification.trial_id
        for multiplier in sorted(
            {cell.sample_size_multiplier for cell in design.cells}
        ):
            rows = [
                row
                for row in estimates
                if row.trial_id == trial_id and row.sample_size_multiplier == multiplier
            ]
            for response in (
                "propensity_coefficient",
                "naive_exposure_bias",
                "adjusted_exposure_bias",
                "score_imbalance",
            ):
                slopes, levels = _paired_slopes(
                    rows,
                    x=lambda row: row.assignment_strength,
                    y=partial(
                        _dose_value,
                        response=response,
                        exposure_truth=design.exposure_log_odds_coefficient,
                    ),
                    minimum_levels=5,
                )
                if len(slopes) < 2:
                    continue
                values = np.asarray(slopes, dtype=float)
                low, high = _mean_interval(values)
                output.append(
                    ConfoundingDoseResponseV1(
                        trial_id=trial_id,
                        sample_size_multiplier=multiplier,
                        response=response,
                        worlds=len(values),
                        assignment_strengths=levels,
                        mean_slope=float(values.mean()),
                        slope_ci_low=low,
                        slope_ci_high=high,
                    )
                )
    return output


def _overlap_responses(
    estimates: list[ConfoundingWorldEstimateV1],
    design: ConfoundingQualificationDesignV1,
) -> list[OverlapResponseV1]:
    output = []
    for trial in design.trials:
        trial_id = trial.qualification.trial_id
        for multiplier in sorted(
            {cell.sample_size_multiplier for cell in design.cells}
        ):
            rows = [
                row
                for row in estimates
                if row.trial_id == trial_id and row.sample_size_multiplier == multiplier
            ]
            for response in (
                "extreme_propensity_fraction",
                "effective_sample_loss",
            ):
                slopes, levels = _paired_slopes(
                    rows,
                    x=lambda row: abs(row.assignment_strength),
                    y=partial(_overlap_value, response=response),
                    minimum_levels=3,
                    average_duplicate_x=True,
                )
                if len(slopes) < 2:
                    continue
                values = np.asarray(slopes, dtype=float)
                low, high = _mean_interval(values)
                output.append(
                    OverlapResponseV1(
                        trial_id=trial_id,
                        sample_size_multiplier=multiplier,
                        response=response,
                        worlds=len(values),
                        absolute_strengths=levels,
                        mean_slope=float(values.mean()),
                        slope_ci_low=low,
                        slope_ci_high=high,
                    )
                )
    return output


def _dose_value(
    row: ConfoundingWorldEstimateV1,
    *,
    response: str,
    exposure_truth: float,
) -> float:
    if response == "propensity_coefficient":
        return row.propensity_estimate
    if response == "naive_exposure_bias":
        return row.naive_exposure_estimate - exposure_truth
    if response == "adjusted_exposure_bias":
        return row.adjusted_exposure_estimate - exposure_truth
    if response == "score_imbalance":
        return row.score_mean_difference
    raise ValueError(f"Unsupported confounding response: {response!r}.")


def _overlap_value(
    row: ConfoundingWorldEstimateV1,
    *,
    response: str,
) -> float:
    if response == "extreme_propensity_fraction":
        return row.extreme_propensity_fraction
    if response == "effective_sample_loss":
        return 1.0 - row.oracle_effective_sample_fraction
    raise ValueError(f"Unsupported overlap response: {response!r}.")


def _paired_slopes(
    rows: list[ConfoundingWorldEstimateV1],
    *,
    x: Callable[[ConfoundingWorldEstimateV1], float],
    y: Callable[[ConfoundingWorldEstimateV1], float],
    minimum_levels: int,
    average_duplicate_x: bool = False,
) -> tuple[list[float], tuple[float, ...]]:
    by_world: defaultdict[int, list[ConfoundingWorldEstimateV1]] = defaultdict(list)
    for row in rows:
        by_world[row.world_index].append(row)
    slopes = []
    levels: tuple[float, ...] = ()
    for world_rows in by_world.values():
        pairs = [(float(x(row)), float(y(row))) for row in world_rows]
        if average_duplicate_x:
            grouped: defaultdict[float, list[float]] = defaultdict(list)
            for level, value in pairs:
                grouped[level].append(value)
            pairs = [
                (level, float(np.mean(grouped[level]))) for level in sorted(grouped)
            ]
        else:
            pairs = sorted(pairs)
        if len({level for level, _ in pairs}) < minimum_levels:
            continue
        levels = tuple(level for level, _ in pairs)
        slopes.append(
            float(
                np.polyfit(
                    np.asarray(levels, dtype=float),
                    np.asarray([value for _, value in pairs], dtype=float),
                    1,
                )[0]
            )
        )
    return slopes, levels


def _information_responses(
    estimates: list[ConfoundingWorldEstimateV1],
    design: ConfoundingQualificationDesignV1,
) -> list[ConfoundingInformationResponseV1]:
    multipliers = sorted({cell.sample_size_multiplier for cell in design.cells})
    lower, higher = multipliers[0], multipliers[-1]
    asymptotic = float(np.sqrt(lower / higher))
    by_key = {
        (
            row.trial_id,
            row.assignment_strength,
            row.sample_size_multiplier,
            row.world_index,
        ): row
        for row in estimates
    }
    output = []
    for trial in design.trials:
        trial_id = trial.qualification.trial_id
        for strength in sorted({cell.assignment_strength for cell in design.cells}):
            ratios = []
            for world_index in range(
                min(
                    cell.worlds_per_trial
                    for cell in design.cells
                    if cell.assignment_strength == strength
                )
            ):
                low_row = by_key.get((trial_id, strength, lower, world_index))
                high_row = by_key.get((trial_id, strength, higher, world_index))
                if low_row is not None and high_row is not None:
                    ratios.append(
                        high_row.adjusted_exposure_standard_error
                        / low_row.adjusted_exposure_standard_error
                    )
            if len(ratios) < 2:
                continue
            values = np.asarray(ratios, dtype=float)
            log_low, log_high = _mean_interval(np.log(values))
            output.append(
                ConfoundingInformationResponseV1(
                    trial_id=trial_id,
                    assignment_strength=strength,
                    worlds=len(values),
                    asymptotic_standard_error_ratio=asymptotic,
                    geometric_mean_standard_error_ratio=float(
                        np.exp(np.log(values).mean())
                    ),
                    ratio_ci_low=float(np.exp(log_low)),
                    ratio_ci_high=float(np.exp(log_high)),
                )
            )
    return output


def _mean_interval(
    values: npt.NDArray[np.float64],
) -> tuple[float, float]:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Confounding interval requires at least two finite values.")
    half_width = float(
        t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    mean = float(values.mean())
    return mean - half_width, mean + half_width


def _release_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Confounding world path escapes the release directory.")
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
    return f"conf-{digest[:20]}"


__all__ = [
    "ConfoundingCellV1",
    "ConfoundingQualificationDesignV1",
    "ConfoundingQualificationReceiptV1",
    "ConfoundingTrialV1",
    "ConfoundingWorldReceiptV1",
    "evaluate_confounding_qualification",
]
