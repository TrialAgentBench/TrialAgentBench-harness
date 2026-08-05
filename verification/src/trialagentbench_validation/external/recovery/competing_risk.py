"""Independent verification of native competing-risk qualifications."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import minimize
from scipy.stats import t
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationWarning,
)

from trialagentbench_validation.external.recovery.rctbench import (
    RctQualificationTrialV1,
)
from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.statistics import proportion_interval


class _EstimationError(ValueError):
    """Expected non-estimability in one otherwise valid world."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CompetingRiskCellV1(_FrozenModel):
    """One pair of cause-specific treatment coefficients."""

    cell_id: str = Field(min_length=1)
    mechanism_id: str = Field(min_length=1)
    primary_treatment_coefficient: float = Field(allow_inf_nan=False)
    competing_treatment_coefficient: float = Field(allow_inf_nan=False)
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    worlds_per_trial: int = Field(ge=2)
    primary_response: bool = False
    competing_response: bool = False


class CompetingRiskTrialV1(_FrozenModel):
    """One RCT source model used for baseline generation."""

    qualification: RctQualificationTrialV1


class CompetingRiskDesignV1(_FrozenModel):
    """Path-free competing-risk qualification design."""

    schema_id: Literal["trialagentbench.competing_risk_design/v1"]
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    seed: int = Field(ge=0, le=2**32 - 1)
    followup_intervals: int = Field(ge=2)
    primary_intercept: float = Field(allow_inf_nan=False)
    competing_intercept: float = Field(allow_inf_nan=False)
    age_coefficient: float = Field(allow_inf_nan=False)
    bmi_coefficient: float = Field(allow_inf_nan=False)
    trials: tuple[CompetingRiskTrialV1, ...] = Field(min_length=3)
    cells: tuple[CompetingRiskCellV1, ...] = Field(min_length=7)

    @model_validator(mode="after")
    def _complete(self) -> CompetingRiskDesignV1:
        trial_ids = [row.qualification.trial_id for row in self.trials]
        cell_ids = [row.cell_id for row in self.cells]
        if len(trial_ids) != len(set(trial_ids)) or len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Competing-risk trial and cell identities must be unique.")
        mechanisms: defaultdict[str, set[float]] = defaultdict(set)
        for row in self.cells:
            mechanisms[row.mechanism_id].add(row.sample_size_multiplier)
        if any(levels != {1.0, 4.0} for levels in mechanisms.values()):
            raise ValueError(
                "Each competing-risk mechanism requires source-sized and fourfold cells."
            )
        primary_levels = {
            row.primary_treatment_coefficient
            for row in self.cells
            if row.primary_response
        }
        competing_levels = {
            row.competing_treatment_coefficient
            for row in self.cells
            if row.competing_response
        }
        if len(primary_levels) < 4 or len(competing_levels) < 5:
            raise ValueError(
                "Competing-risk design requires graded primary and bidirectional competing doses."
            )
        if (
            len(
                {
                    row.competing_treatment_coefficient
                    for row in self.cells
                    if row.primary_response
                }
            )
            != 1
        ):
            raise ValueError(
                "Primary response cells must hold the competing coefficient fixed."
            )
        if (
            len(
                {
                    row.primary_treatment_coefficient
                    for row in self.cells
                    if row.competing_response
                }
            )
            != 1
        ):
            raise ValueError(
                "Competing response cells must hold the primary coefficient fixed."
            )
        if not any(
            row.primary_treatment_coefficient == 0
            and row.competing_treatment_coefficient == 0
            for row in self.cells
        ):
            raise ValueError("Competing-risk design requires a joint null cell.")
        return self


class CompetingRiskWorldReceiptV1(_FrozenModel):
    """Checksum-bound identity for one competing-risk world."""

    world_id: str = Field(pattern=r"^cr-[0-9a-f]{20}$")
    trial_id: str = Field(pattern=r"^RCTBENCH-[0-9]{3}$")
    cell_id: str = Field(min_length=1)
    world_index: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    subjects: int = Field(ge=40)
    analysis_path: str = Field(pattern=r"^worlds/cr-[0-9a-f]{20}\.parquet$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CompetingRiskReceiptV1(_FrozenModel):
    """Complete competing-risk release inventory."""

    schema_id: Literal["trialagentbench.competing_risk_receipt/v1"]
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[CompetingRiskWorldReceiptV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> CompetingRiskReceiptV1:
        keys = [(row.trial_id, row.cell_id, row.world_index) for row in self.worlds]
        if len(keys) != len(set(keys)):
            raise ValueError("Competing-risk world identities must be unique.")
        return self


class CompetingRiskWorldEstimateV1(_FrozenModel):
    """Independent cause-specific recovery and cumulative incidence."""

    world_id: str
    trial_id: str
    cell_id: str
    world_index: int
    primary_truth: float = Field(allow_inf_nan=False)
    competing_truth: float = Field(allow_inf_nan=False)
    primary_estimate: float = Field(allow_inf_nan=False)
    primary_standard_error: float = Field(gt=0, allow_inf_nan=False)
    primary_covered: bool
    competing_estimate: float = Field(allow_inf_nan=False)
    competing_standard_error: float = Field(gt=0, allow_inf_nan=False)
    competing_covered: bool
    primary_cumulative_incidence_difference: float = Field(
        ge=-1,
        le=1,
        allow_inf_nan=False,
    )
    any_event_cumulative_incidence_difference: float = Field(
        ge=-1,
        le=1,
        allow_inf_nan=False,
    )
    primary_event_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    competing_event_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    maximum_point_estimator_crosscheck_difference: float = Field(
        ge=0,
        allow_inf_nan=False,
    )


class CauseRecoverySummaryV1(_FrozenModel):
    """Repeated-world recovery for one cause-specific coefficient."""

    trial_id: str
    cell_id: str
    cause: Literal["primary", "competing"]
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
        default=None, gt=0, allow_inf_nan=False
    )
    maximum_point_estimator_crosscheck_difference: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )


class CompetingRiskDoseResponseV1(_FrozenModel):
    """Within-world response to a cause-specific coefficient."""

    trial_id: str
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)
    response: Literal[
        "primary_coefficient",
        "competing_coefficient",
        "primary_cumulative_incidence",
        "any_event_cumulative_incidence",
    ]
    worlds: int = Field(ge=2)
    coefficient_levels: tuple[float, ...] = Field(min_length=4)
    mean_slope: float = Field(allow_inf_nan=False)
    slope_ci_low: float = Field(allow_inf_nan=False)
    slope_ci_high: float = Field(allow_inf_nan=False)


class CompetingRiskInformationResponseV1(_FrozenModel):
    """Paired precision response from source size to fourfold information."""

    trial_id: str
    mechanism_id: str
    cause: Literal["primary", "competing"]
    worlds: int = Field(ge=2)
    geometric_mean_standard_error_ratio: float = Field(gt=0, allow_inf_nan=False)
    ratio_ci_low: float = Field(gt=0, allow_inf_nan=False)
    ratio_ci_high: float = Field(gt=0, allow_inf_nan=False)
    asymptotic_standard_error_ratio: float = 0.5


class CompetingRiskReportV1(_FrozenModel):
    """Independent competing-risk qualification report."""

    schema_id: Literal["trialagentbench.competing_risk_report/v1"] = (
        "trialagentbench.competing_risk_report/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimates: tuple[CompetingRiskWorldEstimateV1, ...]
    cause_summaries: tuple[CauseRecoverySummaryV1, ...]
    dose_responses: tuple[CompetingRiskDoseResponseV1, ...]
    information_responses: tuple[CompetingRiskInformationResponseV1, ...]


def evaluate_competing_risk_qualification(
    *,
    release_dir: Path,
    minimum_null_worlds: int = 100,
    minimum_nonnull_worlds: int = 50,
) -> CompetingRiskReportV1:
    """Verify and independently analyze competing-risk worlds."""

    design_path = release_dir / "design.json"
    receipt_path = release_dir / "receipt.json"
    payload = json.loads(design_path.read_text(encoding="utf-8"))
    design = CompetingRiskDesignV1.model_validate(payload)
    design_sha256 = _json_sha(payload)
    receipt = CompetingRiskReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    if receipt.design_sha256 != design_sha256:
        raise ValueError("Competing-risk receipt does not bind its design.")
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
        raise ValueError("Competing-risk receipt does not contain the complete design.")
    for cell in design.cells:
        is_null = (
            cell.primary_treatment_coefficient == 0
            and cell.competing_treatment_coefficient == 0
        )
        floor = minimum_null_worlds if is_null else minimum_nonnull_worlds
        if cell.worlds_per_trial < floor:
            raise ValueError(
                f"Competing-risk cell {cell.cell_id!r} misses its replication floor."
            )

    estimates = []
    failures: defaultdict[tuple[str, str], int] = defaultdict(int)
    for world in receipt.worlds:
        trial = trials[world.trial_id]
        cell = cells[world.cell_id]
        if world.seed != _world_seed(design.seed, world.trial_id, world.world_index):
            raise ValueError(f"Competing-risk world seed mismatch: {world.world_id}.")
        if world.world_id != _world_id(
            design_sha256,
            world.trial_id,
            world.cell_id,
            world.world_index,
        ):
            raise ValueError(
                f"Competing-risk world identity mismatch: {world.world_id}."
            )
        expected_subjects = int(
            round(
                trial.qualification.fitted_analysis.source_subjects
                * cell.sample_size_multiplier
            )
        )
        if world.subjects != expected_subjects:
            raise ValueError(
                f"Competing-risk subject count mismatch: {world.world_id}."
            )
        path = _release_path(release_dir, world.analysis_path)
        if sha256_file(path) != world.analysis_sha256:
            raise ValueError(f"Competing-risk checksum mismatch: {world.world_id}.")
        frame = _validate_world(
            pd.read_parquet(path),
            world=world,
            expected_subjects=expected_subjects,
            intervals=design.followup_intervals,
        )
        try:
            estimates.append(
                _fit_world(
                    frame,
                    world=world,
                    cell=cell,
                    intervals=design.followup_intervals,
                )
            )
        except (np.linalg.LinAlgError, _EstimationError):
            failures[(world.trial_id, world.cell_id)] += 1
    return CompetingRiskReportV1(
        design_sha256=design_sha256,
        receipt_sha256=sha256_file(receipt_path),
        estimates=tuple(estimates),
        cause_summaries=tuple(_summarize_causes(estimates, design, failures)),
        dose_responses=tuple(_dose_responses(estimates, design)),
        information_responses=tuple(_information_responses(estimates, design)),
    )


def _validate_world(
    frame: pd.DataFrame,
    *,
    world: CompetingRiskWorldReceiptV1,
    expected_subjects: int,
    intervals: int,
) -> pd.DataFrame:
    required = {
        "world_id",
        "trial_id",
        "cell_id",
        "participant_id",
        "arm",
        "age",
        "bmi",
        "event_time",
        "event_cause",
        "event",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Competing-risk world is missing columns: {missing!r}.")
    for column, expected in (
        ("world_id", world.world_id),
        ("trial_id", world.trial_id),
        ("cell_id", world.cell_id),
    ):
        if set(frame[column].astype(str)) != {expected}:
            raise ValueError(f"Competing-risk world carries inconsistent {column}.")
    if (
        len(frame) != expected_subjects
        or frame["participant_id"].nunique() != expected_subjects
    ):
        raise ValueError("Competing-risk world must contain one row per subject.")
    if set(frame["arm"].astype(str)) != {"control", "active"}:
        raise ValueError("Competing-risk world must contain both randomized arms.")
    if set(frame["event_cause"].astype(str)) - {"primary", "competing", "censored"}:
        raise ValueError("Competing-risk world contains an unknown event cause.")
    event = frame["event"].astype(bool)
    if not frame.loc[~event, "event_cause"].astype(str).eq("censored").all():
        raise ValueError("Censored subjects must carry the censored cause.")
    times = pd.to_numeric(frame["event_time"], errors="raise").to_numpy(dtype=float)
    rounded_times = np.rint(times)
    if (
        not np.isfinite(times).all()
        or np.any(np.abs(times - rounded_times) > 1e-8)
        or np.any(rounded_times < 1)
        or np.any(rounded_times > intervals)
    ):
        raise ValueError(
            "Competing-risk event times must lie on the declared interval grid."
        )
    frame = frame.copy()
    frame["event_time"] = rounded_times.astype(np.int64)
    return frame


def _fit_world(
    frame: pd.DataFrame,
    *,
    world: CompetingRiskWorldReceiptV1,
    cell: CompetingRiskCellV1,
    intervals: int,
) -> CompetingRiskWorldEstimateV1:
    risk = _risk_rows(frame, intervals=intervals)
    design = np.column_stack(
        [
            np.ones(len(risk)),
            risk["active"].to_numpy(dtype=float),
            risk["age_z"].to_numpy(dtype=float),
            risk["bmi_z"].to_numpy(dtype=float),
        ]
    )
    outcome = risk["outcome"].to_numpy(dtype=int)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", HessianInversionWarning)
            warnings.simplefilter("error", PerfectSeparationWarning)
            warnings.simplefilter("error", RuntimeWarning)
            fit = sm.MNLogit(outcome, design).fit(
                method="newton",
                disp=False,
                maxiter=100,
            )
    except (
        ConvergenceWarning,
        HessianInversionWarning,
        PerfectSeparationWarning,
        RuntimeWarning,
    ) as error:
        raise _EstimationError(
            "Competing-risk multinomial fit is not estimable."
        ) from error
    if not bool(fit.mle_retvals["converged"]):
        raise _EstimationError("Competing-risk multinomial fit did not converge.")
    params = np.asarray(fit.params, dtype=float)
    standard_errors = np.asarray(fit.bse, dtype=float)
    if params.shape != (4, 2) or standard_errors.shape != (4, 2):
        raise _EstimationError(
            "Competing-risk multinomial fit has an unexpected shape."
        )
    primary_estimate = float(params[1, 0])
    competing_estimate = float(params[1, 1])
    primary_se = float(standard_errors[1, 0])
    competing_se = float(standard_errors[1, 1])
    if not np.isfinite(
        [primary_estimate, competing_estimate, primary_se, competing_se]
    ).all():
        raise _EstimationError("Competing-risk multinomial fit is non-finite.")
    crosscheck = _multinomial_crosscheck(
        design,
        outcome,
        start=params,
        expected=(primary_estimate, competing_estimate),
    )
    active = frame["arm"].astype("string").eq("active")
    primary = frame["event_cause"].astype("string").eq("primary")
    any_event = frame["event"].astype(bool)
    return CompetingRiskWorldEstimateV1(
        world_id=world.world_id,
        trial_id=world.trial_id,
        cell_id=world.cell_id,
        world_index=world.world_index,
        primary_truth=cell.primary_treatment_coefficient,
        competing_truth=cell.competing_treatment_coefficient,
        primary_estimate=primary_estimate,
        primary_standard_error=primary_se,
        primary_covered=(
            primary_estimate - 1.96 * primary_se
            <= cell.primary_treatment_coefficient
            <= primary_estimate + 1.96 * primary_se
        ),
        competing_estimate=competing_estimate,
        competing_standard_error=competing_se,
        competing_covered=(
            competing_estimate - 1.96 * competing_se
            <= cell.competing_treatment_coefficient
            <= competing_estimate + 1.96 * competing_se
        ),
        primary_cumulative_incidence_difference=float(
            primary[active].mean() - primary[~active].mean()
        ),
        any_event_cumulative_incidence_difference=float(
            any_event[active].mean() - any_event[~active].mean()
        ),
        primary_event_fraction=float(primary.mean()),
        competing_event_fraction=float(
            frame["event_cause"].astype("string").eq("competing").mean()
        ),
        maximum_point_estimator_crosscheck_difference=crosscheck,
    )


def _risk_rows(frame: pd.DataFrame, *, intervals: int) -> pd.DataFrame:
    age = frame["age"].to_numpy(dtype=float)
    bmi = frame["bmi"].to_numpy(dtype=float)
    age_z = (age - age.mean()) / age.std(ddof=0)
    bmi_z = (bmi - bmi.mean()) / bmi.std(ddof=0)
    if not np.isfinite(age_z).all() or not np.isfinite(bmi_z).all():
        raise ValueError(
            "Competing-risk baseline predictors require positive variation."
        )
    event_times = frame["event_time"].to_numpy(dtype=np.int64)
    records = []
    for position, row in enumerate(frame.itertuples(index=False)):
        event_time = int(event_times[position])
        cause = str(row.event_cause)
        for interval in range(1, intervals + 1):
            if interval > event_time:
                break
            outcome = (
                1
                if interval == event_time and cause == "primary"
                else 2 if interval == event_time and cause == "competing" else 0
            )
            records.append(
                {
                    "active": str(row.arm) == "active",
                    "age_z": age_z[position],
                    "bmi_z": bmi_z[position],
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(records)


def _multinomial_crosscheck(
    design: npt.NDArray[np.float64],
    outcome: npt.NDArray[np.int_],
    *,
    start: npt.NDArray[np.float64],
    expected: tuple[float, float],
) -> float:
    rows, columns = start.shape

    def objective(flat: npt.NDArray[np.float64]) -> float:
        beta = flat.reshape(rows, columns)
        logits = design @ beta
        maximum = np.maximum(0.0, logits.max(axis=1))
        denominator = maximum + np.log(
            np.exp(-maximum) + np.exp(logits - maximum[:, None]).sum(axis=1)
        )
        selected = np.zeros(len(outcome), dtype=float)
        for cause in (1, 2):
            mask = outcome == cause
            selected[mask] = logits[mask, cause - 1]
        return float(np.sum(denominator - selected))

    def gradient(flat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        probabilities = _multinomial_probabilities(
            design,
            flat.reshape(rows, columns),
        )
        indicators = np.column_stack((outcome == 1, outcome == 2)).astype(float)
        return np.asarray(
            design.T @ (probabilities - indicators),
            dtype=np.float64,
        ).ravel()

    def hessian(flat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        probabilities = _multinomial_probabilities(
            design,
            flat.reshape(rows, columns),
        )
        weights = -np.einsum(
            "ni,nj->nij",
            probabilities,
            probabilities,
        )
        weights[:, 0, 0] += probabilities[:, 0]
        weights[:, 1, 1] += probabilities[:, 1]
        matrix = np.einsum(
            "np,nq,ncd->pcqd",
            design,
            design,
            weights,
            optimize=True,
        )
        return np.asarray(
            matrix.reshape(rows * columns, rows * columns),
            dtype=np.float64,
        )

    result = minimize(
        objective,
        np.zeros_like(start).ravel(),
        jac=gradient,
        method="L-BFGS-B",
        options={"gtol": 1e-10, "ftol": 1e-14, "maxiter": 500},
    )
    coefficients = result.x.reshape(rows, columns)
    if not np.isfinite(coefficients).all():
        raise _EstimationError("Independent multinomial fit is non-finite.")
    difference = _coefficient_difference(coefficients, expected)
    if difference > 1e-6:
        result = minimize(
            objective,
            result.x,
            jac=gradient,
            hess=hessian,
            method="trust-ncg",
            options={"gtol": 1e-10, "maxiter": 100},
        )
        coefficients = result.x.reshape(rows, columns)
        if not np.isfinite(coefficients).all():
            raise _EstimationError("Independent multinomial refinement is non-finite.")
        difference = _coefficient_difference(coefficients, expected)
    if difference > 1e-6:
        raise _EstimationError("Statsmodels and SciPy competing-risk estimates differ.")
    return difference


def _coefficient_difference(
    coefficients: npt.NDArray[np.float64],
    expected: tuple[float, float],
) -> float:
    return max(
        abs(float(coefficients[1, 0]) - expected[0]),
        abs(float(coefficients[1, 1]) - expected[1]),
    )


def _multinomial_probabilities(
    design: npt.NDArray[np.float64],
    coefficients: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    logits = design @ coefficients
    maximum = np.maximum(0.0, logits.max(axis=1))
    denominator = np.exp(-maximum) + np.exp(logits - maximum[:, None]).sum(axis=1)
    return np.asarray(
        np.exp(logits - maximum[:, None]) / denominator[:, None],
        dtype=np.float64,
    )


def _summarize_causes(
    estimates: list[CompetingRiskWorldEstimateV1],
    design: CompetingRiskDesignV1,
    failures: defaultdict[tuple[str, str], int],
) -> list[CauseRecoverySummaryV1]:
    groups: defaultdict[tuple[str, str], list[CompetingRiskWorldEstimateV1]] = (
        defaultdict(list)
    )
    for row in estimates:
        groups[(row.trial_id, row.cell_id)].append(row)
    output = []
    for trial in design.trials:
        trial_id = trial.qualification.trial_id
        for cell in design.cells:
            rows = groups[(trial_id, cell.cell_id)]
            for cause in ("primary", "competing"):
                truth = (
                    cell.primary_treatment_coefficient
                    if cause == "primary"
                    else cell.competing_treatment_coefficient
                )
                values = np.asarray(
                    [
                        (
                            row.primary_estimate
                            if cause == "primary"
                            else row.competing_estimate
                        )
                        for row in rows
                    ],
                    dtype=float,
                )
                ses = np.asarray(
                    [
                        (
                            row.primary_standard_error
                            if cause == "primary"
                            else row.competing_standard_error
                        )
                        for row in rows
                    ],
                    dtype=float,
                )
                covered = np.asarray(
                    [
                        (
                            row.primary_covered
                            if cause == "primary"
                            else row.competing_covered
                        )
                        for row in rows
                    ],
                    dtype=bool,
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
                crosscheck: float | None
                if rows:
                    biases = values - truth
                    low, high = _mean_interval(biases)
                    rejected = np.abs(values / ses) > 1.96
                    coverage_bounds = proportion_interval(
                        int(covered.sum()), len(covered)
                    )
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
                    crosscheck = max(
                        row.maximum_point_estimator_crosscheck_difference
                        for row in rows
                    )
                else:
                    bias = None
                    low = None
                    high = None
                    rmse = None
                    coverage = None
                    rejection = None
                    coverage_bounds = rejection_bounds = (None, None)
                    se_ratio = None
                    crosscheck = None
                output.append(
                    CauseRecoverySummaryV1(
                        trial_id=trial_id,
                        cell_id=cell.cell_id,
                        cause=cause,
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
                        maximum_point_estimator_crosscheck_difference=crosscheck,
                    )
                )
    return output


def _dose_responses(
    estimates: list[CompetingRiskWorldEstimateV1],
    design: CompetingRiskDesignV1,
) -> list[CompetingRiskDoseResponseV1]:
    output = []
    for trial in design.trials:
        trial_id = trial.qualification.trial_id
        trial_rows = [row for row in estimates if row.trial_id == trial_id]
        for multiplier in (1.0, 4.0):
            primary_cells = {
                cell.cell_id: cell.primary_treatment_coefficient
                for cell in design.cells
                if cell.primary_response and cell.sample_size_multiplier == multiplier
            }
            competing_cells = {
                cell.cell_id: cell.competing_treatment_coefficient
                for cell in design.cells
                if cell.competing_response and cell.sample_size_multiplier == multiplier
            }
            for response, cells, attribute in (
                ("primary_coefficient", primary_cells, "primary_estimate"),
                ("competing_coefficient", competing_cells, "competing_estimate"),
                (
                    "primary_cumulative_incidence",
                    competing_cells,
                    "primary_cumulative_incidence_difference",
                ),
                (
                    "any_event_cumulative_incidence",
                    competing_cells,
                    "any_event_cumulative_incidence_difference",
                ),
            ):
                by_world: defaultdict[
                    int,
                    list[CompetingRiskWorldEstimateV1],
                ] = defaultdict(list)
                for row in trial_rows:
                    if row.cell_id in cells:
                        by_world[row.world_index].append(row)
                slopes = []
                levels: tuple[float, ...] | None = None
                for world_rows in by_world.values():
                    ordered = sorted(
                        world_rows,
                        key=lambda row: cells[row.cell_id],
                    )
                    world_levels = tuple(cells[row.cell_id] for row in ordered)
                    if len(set(world_levels)) < 4:
                        continue
                    slopes.append(
                        float(
                            np.polyfit(
                                world_levels,
                                [float(getattr(row, attribute)) for row in ordered],
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
                    CompetingRiskDoseResponseV1(
                        trial_id=trial_id,
                        sample_size_multiplier=multiplier,
                        response=response,
                        worlds=len(values),
                        coefficient_levels=levels,
                        mean_slope=float(values.mean()),
                        slope_ci_low=low,
                        slope_ci_high=high,
                    )
                )
    return output


def _information_responses(
    estimates: list[CompetingRiskWorldEstimateV1],
    design: CompetingRiskDesignV1,
) -> list[CompetingRiskInformationResponseV1]:
    cells = {row.cell_id: row for row in design.cells}
    groups: defaultdict[
        tuple[str, str, int],
        dict[float, CompetingRiskWorldEstimateV1],
    ] = defaultdict(dict)
    for row in estimates:
        cell = cells[row.cell_id]
        groups[(row.trial_id, cell.mechanism_id, row.world_index)][
            cell.sample_size_multiplier
        ] = row
    output = []
    for trial in design.trials:
        trial_id = trial.qualification.trial_id
        for mechanism_id in sorted({row.mechanism_id for row in design.cells}):
            for cause in ("primary", "competing"):
                ratios = []
                for (group_trial, group_mechanism, _), paired in groups.items():
                    if group_trial != trial_id or group_mechanism != mechanism_id:
                        continue
                    if set(paired) != {1.0, 4.0}:
                        continue
                    source = paired[1.0]
                    amplified = paired[4.0]
                    source_se = (
                        source.primary_standard_error
                        if cause == "primary"
                        else source.competing_standard_error
                    )
                    amplified_se = (
                        amplified.primary_standard_error
                        if cause == "primary"
                        else amplified.competing_standard_error
                    )
                    ratios.append(amplified_se / source_se)
                values = np.asarray(ratios, dtype=np.float64)
                mean_ratio, low, high = _geometric_interval(values)
                output.append(
                    CompetingRiskInformationResponseV1(
                        trial_id=trial_id,
                        mechanism_id=mechanism_id,
                        cause=cause,
                        worlds=len(values),
                        geometric_mean_standard_error_ratio=mean_ratio,
                        ratio_ci_low=low,
                        ratio_ci_high=high,
                    )
                )
    return output


def _geometric_interval(
    values: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    if len(values) < 2 or np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError("Ratio interval requires at least two finite positive values.")
    logs = np.log(values)
    low, high = _mean_interval(logs)
    return float(np.exp(logs.mean())), float(np.exp(low)), float(np.exp(high))


def _mean_interval(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Competing-risk interval requires at least two finite values.")
    half_width = float(
        t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    mean = float(values.mean())
    return mean - half_width, mean + half_width


def _release_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Competing-risk path escapes the release directory.")
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
    return f"cr-{digest[:20]}"


__all__ = [
    "CompetingRiskCellV1",
    "CompetingRiskDesignV1",
    "CompetingRiskReceiptV1",
    "CompetingRiskTrialV1",
    "CompetingRiskWorldReceiptV1",
    "evaluate_competing_risk_qualification",
]
