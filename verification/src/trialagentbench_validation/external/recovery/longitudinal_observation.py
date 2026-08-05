"""Independent recovery for native longitudinal dropout qualifications."""

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
from scipy.stats import t as student_t
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from trialagentbench_validation.external.recovery.longitudinal import (
    LongitudinalQualificationTrialV1,
)
from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.statistics import proportion_interval

ObservationRoute = Literal["available_case", "estimated_ipcw", "oracle_ipcw"]
CorrectionRoute = Literal["estimated_ipcw", "oracle_ipcw"]
DropoutMechanism = Literal["none", "lagged_outcome"]


class _EstimationError(ValueError):
    """Expected non-estimability in one otherwise valid simulation world."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ObservationCellV1(_FrozenModel):
    """One controlled longitudinal observation-process setting."""

    cell_id: str = Field(min_length=1)
    mechanism: DropoutMechanism
    logit_intercept: float | None = Field(default=None, allow_inf_nan=False)
    lagged_outcome_coefficient: float = Field(ge=0, allow_inf_nan=False)
    worlds_per_trial: int = Field(ge=2)
    sample_size_multiplier: float = Field(ge=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _complete(self) -> ObservationCellV1:
        if self.mechanism == "none":
            if self.logit_intercept is not None or self.lagged_outcome_coefficient != 0:
                raise ValueError(
                    "Observation cell without dropout cannot configure a dropout model."
                )
        elif self.logit_intercept is None:
            raise ValueError("Lagged-outcome dropout requires a logit intercept.")
        return self


class ObservationTrialV1(_FrozenModel):
    """Longitudinal source model plus its dropout-predictor standardization."""

    qualification: LongitudinalQualificationTrialV1
    predictor_center: float = Field(allow_inf_nan=False)
    predictor_scale: float = Field(gt=0, allow_inf_nan=False)


class LongitudinalObservationDesignV1(_FrozenModel):
    """Path-free design for longitudinal observation-process qualification."""

    schema_id: Literal["trialagentbench.longitudinal_observation_design/v1"]
    seed: int = Field(ge=0, le=2**32 - 1)
    trials: tuple[ObservationTrialV1, ...] = Field(min_length=2)
    cells: tuple[ObservationCellV1, ...] = Field(min_length=5)

    @model_validator(mode="after")
    def _complete(self) -> LongitudinalObservationDesignV1:
        trial_ids = [trial.qualification.trial_id for trial in self.trials]
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(trial_ids) != len(set(trial_ids)) or len(cell_ids) != len(set(cell_ids)):
            raise ValueError(
                "Observation qualification trial and cell identities must be unique."
            )
        none_cells = [cell for cell in self.cells if cell.mechanism == "none"]
        informative = [
            cell for cell in self.cells if cell.mechanism == "lagged_outcome"
        ]
        coefficients = sorted({cell.lagged_outcome_coefficient for cell in informative})
        if len(none_cells) != 1 or len(coefficients) < 4 or coefficients[0] != 0:
            raise ValueError(
                "Observation qualification requires one complete-data cell and at least four "
                "lagged-outcome levels beginning at zero."
            )
        return self


class ObservationWorldReceiptV1(_FrozenModel):
    """Checksum-bound identity of one observation-process world."""

    world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    trial_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    world_index: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    subjects: int = Field(ge=20)
    dropout_events: int = Field(ge=0)
    analysis_path: str = Field(pattern=r"^worlds/world_[0-9a-f]{20}\.parquet$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ground_truth_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LongitudinalObservationReceiptV1(_FrozenModel):
    """Complete observation-process release inventory."""

    schema_id: Literal["trialagentbench.longitudinal_observation_receipt/v1"]
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[ObservationWorldReceiptV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> LongitudinalObservationReceiptV1:
        identities = [
            (row.trial_id, row.cell_id, row.world_index) for row in self.worlds
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Observation-process world identities must be unique.")
        return self


class DropoutWorldEstimateV1(_FrozenModel):
    """Recovered lagged-outcome dropout mechanism in one world."""

    world_id: str
    trial_id: str
    cell_id: str
    world_index: int
    truth: float = Field(ge=0, allow_inf_nan=False)
    estimate: float = Field(allow_inf_nan=False)
    standard_error: float = Field(gt=0, allow_inf_nan=False)
    covered: bool
    rejected_null: bool
    mean_probability_residual: float = Field(allow_inf_nan=False)
    mean_predictor_weighted_probability_residual: float = Field(allow_inf_nan=False)
    risk_visits: int = Field(ge=1)
    dropout_events: int = Field(ge=1)


class ObservationWorldEstimateV1(_FrozenModel):
    """Recovered treatment-trajectory contrast in one world."""

    world_id: str
    trial_id: str
    cell_id: str
    world_index: int
    arm_id: str
    route: ObservationRoute
    truth: float = Field(allow_inf_nan=False)
    estimate: float = Field(allow_inf_nan=False)
    standard_error: float = Field(gt=0, allow_inf_nan=False)
    covered: bool
    effective_sample_fraction: float = Field(gt=0, le=1, allow_inf_nan=False)
    maximum_weight: float = Field(ge=1, allow_inf_nan=False)


class DropoutCellSummaryV1(_FrozenModel):
    """Across-world recovery of one dropout coefficient."""

    trial_id: str
    cell_id: str
    truth: float = Field(ge=0, allow_inf_nan=False)
    worlds: int = Field(ge=2)
    successful_worlds: int = Field(ge=0)
    failures: int = Field(ge=0)
    mean_estimate: float | None = Field(default=None, allow_inf_nan=False)
    bias: float | None = Field(default=None, allow_inf_nan=False)
    bias_ci_low: float | None = Field(default=None, allow_inf_nan=False)
    bias_ci_high: float | None = Field(default=None, allow_inf_nan=False)
    coverage: float | None = Field(default=None, ge=0, le=1)
    coverage_ci_low: float | None = Field(default=None, ge=0, le=1)
    coverage_ci_high: float | None = Field(default=None, ge=0, le=1)
    rejection_rate: float | None = Field(default=None, ge=0, le=1)
    rejection_rate_ci_low: float | None = Field(default=None, ge=0, le=1)
    rejection_rate_ci_high: float | None = Field(default=None, ge=0, le=1)
    mean_probability_residual: float | None = Field(default=None, allow_inf_nan=False)
    probability_residual_ci_low: float | None = Field(default=None, allow_inf_nan=False)
    probability_residual_ci_high: float | None = Field(
        default=None, allow_inf_nan=False
    )
    mean_predictor_weighted_probability_residual: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )
    predictor_weighted_probability_residual_ci_low: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )
    predictor_weighted_probability_residual_ci_high: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )
    mean_dropout_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)


class ObservationCellSummaryV1(_FrozenModel):
    """Across-world operating characteristics for one analysis route."""

    trial_id: str
    cell_id: str
    arm_id: str
    route: ObservationRoute
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
    model_to_empirical_se_ratio: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    mean_effective_sample_fraction: float | None = Field(
        default=None, gt=0, le=1, allow_inf_nan=False
    )
    median_maximum_weight: float | None = Field(default=None, ge=1, allow_inf_nan=False)


class ObservationDoseResponseV1(_FrozenModel):
    """Within-world treatment-estimate response to dropout dependence."""

    trial_id: str
    arm_id: str
    route: ObservationRoute
    worlds: int = Field(ge=2)
    coefficient_levels: tuple[float, ...] = Field(min_length=4)
    mean_slope: float = Field(allow_inf_nan=False)
    slope_ci_low: float = Field(allow_inf_nan=False)
    slope_ci_high: float = Field(allow_inf_nan=False)


class DropoutDoseResponseV1(_FrozenModel):
    """Within-world recovery response to configured dropout dependence."""

    trial_id: str
    worlds: int = Field(ge=2)
    coefficient_levels: tuple[float, ...] = Field(min_length=4)
    mean_slope: float = Field(allow_inf_nan=False)
    slope_ci_low: float = Field(allow_inf_nan=False)
    slope_ci_high: float = Field(allow_inf_nan=False)


class ObservationRouteContrastV1(_FrozenModel):
    """Paired improvement over available-case analysis."""

    trial_id: str
    cell_id: str
    arm_id: str
    correction_route: CorrectionRoute
    lagged_outcome_coefficient: float = Field(ge=0, allow_inf_nan=False)
    worlds: int = Field(ge=2)
    mean_absolute_error_reduction: float = Field(allow_inf_nan=False)
    reduction_ci_low: float = Field(allow_inf_nan=False)
    reduction_ci_high: float = Field(allow_inf_nan=False)
    improvement_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    improvement_fraction_ci_low: float = Field(ge=0, le=1, allow_inf_nan=False)
    improvement_fraction_ci_high: float = Field(ge=0, le=1, allow_inf_nan=False)
    rmse_ratio: float = Field(ge=0, allow_inf_nan=False)


class LongitudinalObservationReportV1(_FrozenModel):
    """Independent observation-process qualification report."""

    schema_id: Literal["trialagentbench.longitudinal_observation_report/v1"] = (
        "trialagentbench.longitudinal_observation_report/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dropout_estimates: tuple[DropoutWorldEstimateV1, ...]
    treatment_estimates: tuple[ObservationWorldEstimateV1, ...]
    dropout_cells: tuple[DropoutCellSummaryV1, ...]
    treatment_cells: tuple[ObservationCellSummaryV1, ...]
    dropout_response: tuple[DropoutDoseResponseV1, ...]
    treatment_response: tuple[ObservationDoseResponseV1, ...]
    route_contrasts: tuple[ObservationRouteContrastV1, ...]


def evaluate_longitudinal_observation(
    *,
    release_dir: Path,
    minimum_worlds_per_trial_cell: int = 100,
) -> LongitudinalObservationReportV1:
    """Verify and independently analyze longitudinal observation-process worlds."""

    if minimum_worlds_per_trial_cell < 2:
        raise ValueError(
            "Observation qualification replication floor must be at least two."
        )
    design_path = release_dir / "design.json"
    receipt_path = release_dir / "receipt.json"
    design_payload = json.loads(design_path.read_text(encoding="utf-8"))
    design = LongitudinalObservationDesignV1.model_validate(design_payload)
    design_sha256 = _json_sha(design_payload)
    receipt = LongitudinalObservationReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    if receipt.design_sha256 != design_sha256:
        raise ValueError("Observation qualification receipt does not match its design.")
    trials = {trial.qualification.trial_id: trial for trial in design.trials}
    cells = {cell.cell_id: cell for cell in design.cells}
    expected = {
        (trial.qualification.trial_id, cell.cell_id, world_index)
        for trial in design.trials
        for cell in design.cells
        for world_index in range(cell.worlds_per_trial)
    }
    observed = {(row.trial_id, row.cell_id, row.world_index) for row in receipt.worlds}
    if expected != observed:
        raise ValueError(
            "Observation qualification receipt does not contain the complete design."
        )
    short = {
        cell.cell_id: cell.worlds_per_trial
        for cell in design.cells
        if cell.worlds_per_trial < minimum_worlds_per_trial_cell
    }
    if short:
        raise ValueError(
            f"Observation qualification cells miss the replication floor: {short!r}."
        )

    dropout_estimates: list[DropoutWorldEstimateV1] = []
    treatment_estimates: list[ObservationWorldEstimateV1] = []
    dropout_failures: defaultdict[tuple[str, str], int] = defaultdict(int)
    treatment_failures: defaultdict[tuple[str, str, str, ObservationRoute], int] = (
        defaultdict(int)
    )
    dropout_fraction: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for world in receipt.worlds:
        trial = trials[world.trial_id]
        cell = cells[world.cell_id]
        if world.seed != _world_seed(design.seed, world.trial_id, world.world_index):
            raise ValueError(f"Observation world seed mismatch: {world.world_id}.")
        if world.world_id != _world_id(
            design_sha256, world.trial_id, world.cell_id, world.world_index
        ):
            raise ValueError(f"Observation world identity mismatch: {world.world_id}.")
        expected_subjects = int(
            round(trial.qualification.participants * cell.sample_size_multiplier)
        )
        if world.subjects != expected_subjects:
            raise ValueError(
                f"Observation world subject count mismatch: {world.world_id}."
            )
        path = _release_path(release_dir, world.analysis_path)
        if sha256_file(path) != world.analysis_sha256:
            raise ValueError(f"Observation world checksum mismatch: {world.world_id}.")
        frame = _validate_world(
            pd.read_parquet(path),
            world=world,
            trial=trial.qualification,
        )
        source_trial = trial.qualification
        fraction = float(
            frame.groupby("participant_id", sort=False)["observed"]
            .min()
            .eq(False)
            .mean()
        )
        dropout_fraction[(source_trial.trial_id, cell.cell_id)].append(fraction)
        if cell.mechanism == "lagged_outcome":
            try:
                dropout_estimates.append(
                    _fit_dropout(frame, world=world, cell=cell, trial=trial)
                )
            except (np.linalg.LinAlgError, PerfectSeparationError, _EstimationError):
                dropout_failures[(source_trial.trial_id, cell.cell_id)] += 1
        for arm_id in source_trial.arm_ids:
            if arm_id == source_trial.control_arm_id:
                continue
            for route in ("available_case", "estimated_ipcw", "oracle_ipcw"):
                typed_route: ObservationRoute = route
                key = (source_trial.trial_id, cell.cell_id, arm_id, typed_route)
                try:
                    treatment_estimates.append(
                        _fit_treatment(
                            frame,
                            world=world,
                            trial=trial,
                            cell=cell,
                            arm_id=arm_id,
                            route=typed_route,
                        )
                    )
                except (
                    np.linalg.LinAlgError,
                    PerfectSeparationError,
                    _EstimationError,
                ):
                    treatment_failures[key] += 1
    return LongitudinalObservationReportV1(
        design_sha256=design_sha256,
        receipt_sha256=sha256_file(receipt_path),
        dropout_estimates=tuple(dropout_estimates),
        treatment_estimates=tuple(treatment_estimates),
        dropout_cells=tuple(
            _summarize_dropout(
                dropout_estimates,
                design=design,
                failures=dropout_failures,
                dropout_fraction=dropout_fraction,
            )
        ),
        treatment_cells=tuple(
            _summarize_treatment(
                treatment_estimates,
                design=design,
                failures=treatment_failures,
            )
        ),
        dropout_response=tuple(_dropout_response(dropout_estimates, design)),
        treatment_response=tuple(_treatment_response(treatment_estimates, design)),
        route_contrasts=tuple(_route_contrasts(treatment_estimates, design)),
    )


def _validate_world(
    frame: pd.DataFrame,
    *,
    world: ObservationWorldReceiptV1,
    trial: LongitudinalQualificationTrialV1,
) -> pd.DataFrame:
    required = {
        "world_id",
        "trial_id",
        "cell_id",
        "participant_id",
        "arm",
        "time",
        "value",
        "observed",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Observation world omits columns: {sorted(missing)!r}.")
    if (
        set(frame["world_id"].astype(str)) != {world.world_id}
        or set(frame["trial_id"].astype(str)) != {world.trial_id}
        or set(frame["cell_id"].astype(str)) != {world.cell_id}
    ):
        raise ValueError(
            f"Observation world identity columns disagree: {world.world_id}."
        )
    output = frame.copy()
    output["participant_id"] = output["participant_id"].astype("string")
    output["arm"] = output["arm"].astype("string")
    output["time"] = pd.to_numeric(output["time"], errors="raise").astype("float64")
    output["value"] = pd.to_numeric(output["value"], errors="coerce").astype("float64")
    output["observed"] = output["observed"].astype("boolean").fillna(False)
    if output.duplicated(["participant_id", "time"]).any():
        raise ValueError("Observation world contains duplicate participant-time rows.")
    if set(output["arm"].astype(str)) != set(trial.arm_ids):
        raise ValueError(
            "Observation world arm identities differ from its trial design."
        )
    if set(output["time"]) != set(trial.time_values):
        raise ValueError("Observation world schedule differs from its trial design.")
    counts = output.groupby("participant_id", sort=False)["time"].nunique()
    if len(counts) != world.subjects or not counts.eq(len(trial.time_values)).all():
        raise ValueError(
            "Observation world is not a complete planned participant-time grid."
        )
    if not np.array_equal(
        output["observed"].to_numpy(dtype=bool),
        output["value"].notna().to_numpy(dtype=bool),
    ):
        raise ValueError("Observation indicator and value availability disagree.")
    for _, rows in output.sort_values("time").groupby("participant_id", sort=False):
        observed = rows["observed"].to_numpy(dtype=bool)
        if not observed[0] or np.any(np.diff(observed.astype(np.int8)) > 0):
            raise ValueError(
                "Observation world violates baseline presence or monotone dropout."
            )
    dropout_events = int(
        output.groupby("participant_id", sort=False)["observed"].min().eq(False).sum()
    )
    if dropout_events != world.dropout_events:
        raise ValueError("Observation world dropout count differs from its receipt.")
    return output


def _fit_dropout(
    frame: pd.DataFrame,
    *,
    world: ObservationWorldReceiptV1,
    cell: ObservationCellV1,
    trial: ObservationTrialV1,
) -> DropoutWorldEstimateV1:
    if cell.logit_intercept is None:
        raise _EstimationError(
            "Dropout coefficient fit requires a configured intercept."
        )
    risk = _risk_rows(
        frame,
        trial=trial.qualification,
        center=trial.predictor_center,
        scale=trial.predictor_scale,
    )
    events = int(risk["dropout"].sum())
    if events == 0 or events == len(risk):
        raise _EstimationError(
            "Dropout coefficient is not estimable without both events and non-events."
        )
    fit = sm.GLM(
        risk["dropout"].to_numpy(dtype=float),
        sm.add_constant(risk["mechanism_value_z"].to_numpy(dtype=float)),
        family=sm.families.Binomial(),
    ).fit(cov_type="HC3")
    estimate = float(fit.params[1])
    standard_error = float(fit.bse[1])
    if not np.isfinite([estimate, standard_error]).all() or standard_error <= 0:
        raise _EstimationError("Dropout coefficient fit is non-finite.")
    truth = cell.lagged_outcome_coefficient
    predictor = risk["mechanism_value_z"].to_numpy(dtype=float)
    probability = _expit(cell.logit_intercept + truth * predictor)
    probability_residual = risk["dropout"].to_numpy(dtype=float) - probability
    return DropoutWorldEstimateV1(
        world_id=world.world_id,
        trial_id=world.trial_id,
        cell_id=world.cell_id,
        world_index=world.world_index,
        truth=truth,
        estimate=estimate,
        standard_error=standard_error,
        covered=estimate - 1.96 * standard_error
        <= truth
        <= estimate + 1.96 * standard_error,
        rejected_null=abs(estimate / standard_error) > 1.96,
        mean_probability_residual=float(np.mean(probability_residual)),
        mean_predictor_weighted_probability_residual=float(
            np.mean(probability_residual * predictor)
        ),
        risk_visits=len(risk),
        dropout_events=events,
    )


def _fit_treatment(
    frame: pd.DataFrame,
    *,
    world: ObservationWorldReceiptV1,
    trial: ObservationTrialV1,
    cell: ObservationCellV1,
    arm_id: str,
    route: ObservationRoute,
) -> ObservationWorldEstimateV1:
    source_trial = trial.qualification
    subset = frame.loc[
        frame["arm"].astype("string").isin([source_trial.control_arm_id, arm_id])
    ].copy()
    observed = subset.loc[subset["observed"]].copy()
    if route == "available_case" or cell.mechanism == "none":
        weights = pd.Series(1.0, index=observed.index)
    else:
        weights = _ipcw_weights(
            subset,
            cell=cell,
            trial=trial,
            arm_id=arm_id,
            oracle=route == "oracle_ipcw",
        ).reindex(observed.index)
    if weights.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise _EstimationError(
            "Observation-process weights must be finite and positive."
        )
    treatment = observed["arm"].astype("string").eq(arm_id).to_numpy(dtype=float)
    time = observed["time"].to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(observed)), time, treatment, treatment * time]
    )
    fit = sm.WLS(observed["value"].to_numpy(dtype=float), design, weights=weights).fit()
    robust = fit.get_robustcov_results(
        cov_type="cluster",
        groups=observed["participant_id"].astype(str).to_numpy(),
        use_correction=True,
    )
    estimate = float(robust.params[3])
    standard_error = float(robust.bse[3])
    if not np.isfinite([estimate, standard_error]).all() or standard_error <= 0:
        raise _EstimationError("Treatment trajectory fit is non-finite.")
    truth = _truth_slope(source_trial, arm_id)
    weight_values = weights.to_numpy(dtype=float)
    effective_fraction = float(
        (weight_values.sum() ** 2 / np.square(weight_values).sum()) / len(weight_values)
    )
    return ObservationWorldEstimateV1(
        world_id=world.world_id,
        trial_id=world.trial_id,
        cell_id=world.cell_id,
        world_index=world.world_index,
        arm_id=arm_id,
        route=route,
        truth=truth,
        estimate=estimate,
        standard_error=standard_error,
        covered=estimate - 1.96 * standard_error
        <= truth
        <= estimate + 1.96 * standard_error,
        effective_sample_fraction=effective_fraction,
        maximum_weight=float(weight_values.max()),
    )


def _risk_rows(
    frame: pd.DataFrame,
    *,
    trial: LongitudinalQualificationTrialV1,
    center: float,
    scale: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for participant_id, rows in frame.sort_values("time").groupby(
        "participant_id", sort=False
    ):
        ordered = rows.reset_index()
        for position in range(1, len(ordered)):
            previous = ordered.iloc[position - 1]
            current = ordered.iloc[position]
            if not bool(previous["observed"]):
                break
            arm = str(current["arm"])
            previous_time = float(previous["time"])
            arm_shift = _arm_shift(trial, arm_id=arm, time=previous_time)
            records.append(
                {
                    "row_index": int(current["index"]),
                    "participant_id": str(participant_id),
                    "arm": arm,
                    "time": float(current["time"]),
                    "lagged_value_z": (float(previous["value"]) - center) / scale,
                    "mechanism_value_z": (float(previous["value"]) - arm_shift - center)
                    / scale,
                    "observed": bool(current["observed"]),
                    "dropout": not bool(current["observed"]),
                }
            )
            if not bool(current["observed"]):
                break
    if not records:
        raise _EstimationError(
            "Observation world contains no post-baseline risk visits."
        )
    return pd.DataFrame(records)


def _ipcw_weights(
    frame: pd.DataFrame,
    *,
    cell: ObservationCellV1,
    trial: ObservationTrialV1,
    arm_id: str,
    oracle: bool,
) -> pd.Series:
    risk = _risk_rows(
        frame,
        trial=trial.qualification,
        center=trial.predictor_center,
        scale=trial.predictor_scale,
    )
    if oracle:
        if cell.logit_intercept is None:
            raise _EstimationError(
                "Oracle IPCW requires a configured dropout mechanism."
            )
        denominator_probability = 1.0 - _expit(
            cell.logit_intercept
            + cell.lagged_outcome_coefficient
            * risk["mechanism_value_z"].to_numpy(dtype=float)
        )
    else:
        treatment = risk["arm"].astype("string").eq(arm_id).to_numpy(dtype=float)
        time = risk["time"].to_numpy(dtype=float)
        time_z = (time - time.mean()) / time.std(ddof=0)
        denominator_design = np.column_stack(
            [
                np.ones(len(risk)),
                treatment,
                time_z,
                treatment * time_z,
                risk["lagged_value_z"].to_numpy(dtype=float),
            ]
        )
        denominator = sm.GLM(
            risk["observed"].to_numpy(dtype=float),
            denominator_design,
            family=sm.families.Binomial(),
        ).fit()
        denominator_probability = np.asarray(
            denominator.predict(denominator_design), dtype=float
        )
    if not np.isfinite(denominator_probability).all() or np.any(
        denominator_probability <= 0
    ):
        raise _EstimationError(
            "IPCW observation probabilities must be finite and positive."
        )
    risk = risk.assign(
        ratio=1.0 / denominator_probability,
    )
    weight_by_index: dict[int, float] = {}
    for _, rows in risk.groupby("participant_id", sort=False):
        cumulative = 1.0
        for row_index, ratio, row_observed in zip(
            rows["row_index"].to_numpy(dtype=np.int64),
            rows["ratio"].to_numpy(dtype=float),
            rows["observed"].to_numpy(dtype=bool),
            strict=True,
        ):
            cumulative *= float(ratio)
            if row_observed:
                weight_by_index[int(row_index)] = cumulative
    baseline = frame.groupby("participant_id", sort=False)["time"].idxmin()
    weight_by_index.update({int(index): 1.0 for index in baseline})
    return pd.Series(weight_by_index, dtype="float64")


def _truth_slope(trial: LongitudinalQualificationTrialV1, arm_id: str) -> float:
    effect = next(row for row in trial.fitted_model.arm_effects if row.arm_id == arm_id)
    design = np.column_stack(
        [np.ones(len(trial.time_values)), np.asarray(trial.time_values, dtype=float)]
    )
    return float(
        np.linalg.lstsq(
            design, np.asarray(effect.visit_shifts, dtype=float), rcond=None
        )[0][1]
    )


def _arm_shift(
    trial: LongitudinalQualificationTrialV1,
    *,
    arm_id: str,
    time: float,
) -> float:
    if arm_id == trial.control_arm_id:
        return 0.0
    effect = next(row for row in trial.fitted_model.arm_effects if row.arm_id == arm_id)
    shifts = dict(zip(trial.time_values, effect.visit_shifts, strict=True))
    try:
        return float(shifts[time])
    except KeyError as error:
        raise ValueError(
            f"Observation time {time} is absent from declared arm shifts."
        ) from error


def _summarize_dropout(
    estimates: list[DropoutWorldEstimateV1],
    *,
    design: LongitudinalObservationDesignV1,
    failures: defaultdict[tuple[str, str], int],
    dropout_fraction: defaultdict[tuple[str, str], list[float]],
) -> list[DropoutCellSummaryV1]:
    grouped = {
        (trial, cell): rows for (trial, cell), rows in _group_dropout(estimates).items()
    }
    output = []
    for trial in design.trials:
        for cell in design.cells:
            if cell.mechanism == "none":
                continue
            trial_id = trial.qualification.trial_id
            rows = grouped.get((trial_id, cell.cell_id), [])
            configured = cell.worlds_per_trial
            mean_estimate: float | None
            bias: float | None
            low: float | None
            high: float | None
            coverage: float | None
            coverage_low: float | None
            coverage_high: float | None
            rejection: float | None
            rejection_low: float | None
            rejection_high: float | None
            probability_residual: float | None
            probability_residual_low: float | None
            probability_residual_high: float | None
            predictor_score: float | None
            predictor_score_low: float | None
            predictor_score_high: float | None
            if rows:
                values = np.asarray([row.estimate for row in rows], dtype=float)
                biases = values - cell.lagged_outcome_coefficient
                low, high = _mean_interval(biases)
                coverage = float(np.mean([row.covered for row in rows]))
                rejection = float(np.mean([row.rejected_null for row in rows]))
                coverage_bounds = proportion_interval(
                    int(sum(row.covered for row in rows)),
                    len(rows),
                )
                rejection_bounds = proportion_interval(
                    int(sum(row.rejected_null for row in rows)),
                    len(rows),
                )
                coverage_low, coverage_high = coverage_bounds
                rejection_low, rejection_high = rejection_bounds
                mean_estimate = float(values.mean())
                bias = float(biases.mean())
                probability_residuals = np.asarray(
                    [row.mean_probability_residual for row in rows],
                    dtype=float,
                )
                probability_residual = float(np.mean(probability_residuals))
                probability_residual_low, probability_residual_high = _mean_interval(
                    probability_residuals
                )
                predictor_scores = np.asarray(
                    [row.mean_predictor_weighted_probability_residual for row in rows],
                    dtype=float,
                )
                predictor_score = float(np.mean(predictor_scores))
                predictor_score_low, predictor_score_high = _mean_interval(
                    predictor_scores
                )
            else:
                mean_estimate = bias = low = high = coverage = None
                coverage_low = coverage_high = rejection = None
                rejection_low = rejection_high = None
                probability_residual = probability_residual_low = (
                    probability_residual_high
                ) = None
                predictor_score = predictor_score_low = predictor_score_high = None
            output.append(
                DropoutCellSummaryV1(
                    trial_id=trial_id,
                    cell_id=cell.cell_id,
                    truth=cell.lagged_outcome_coefficient,
                    worlds=configured,
                    successful_worlds=len(rows),
                    failures=failures[(trial_id, cell.cell_id)],
                    mean_estimate=mean_estimate,
                    bias=bias,
                    bias_ci_low=low,
                    bias_ci_high=high,
                    coverage=coverage,
                    coverage_ci_low=coverage_low,
                    coverage_ci_high=coverage_high,
                    rejection_rate=rejection,
                    rejection_rate_ci_low=rejection_low,
                    rejection_rate_ci_high=rejection_high,
                    mean_probability_residual=probability_residual,
                    probability_residual_ci_low=probability_residual_low,
                    probability_residual_ci_high=probability_residual_high,
                    mean_predictor_weighted_probability_residual=predictor_score,
                    predictor_weighted_probability_residual_ci_low=predictor_score_low,
                    predictor_weighted_probability_residual_ci_high=predictor_score_high,
                    mean_dropout_fraction=float(
                        np.mean(dropout_fraction[(trial_id, cell.cell_id)])
                    ),
                )
            )
    return output


def _summarize_treatment(
    estimates: list[ObservationWorldEstimateV1],
    *,
    design: LongitudinalObservationDesignV1,
    failures: defaultdict[tuple[str, str, str, ObservationRoute], int],
) -> list[ObservationCellSummaryV1]:
    groups: defaultdict[
        tuple[str, str, str, ObservationRoute], list[ObservationWorldEstimateV1]
    ] = defaultdict(list)
    for row in estimates:
        groups[(row.trial_id, row.cell_id, row.arm_id, row.route)].append(row)
    output = []
    for trial in design.trials:
        source_trial = trial.qualification
        for cell in design.cells:
            for arm_id in source_trial.arm_ids:
                if arm_id == source_trial.control_arm_id:
                    continue
                for route in ("available_case", "estimated_ipcw", "oracle_ipcw"):
                    typed_route: ObservationRoute = route
                    key = (source_trial.trial_id, cell.cell_id, arm_id, typed_route)
                    rows = groups[key]
                    bias: float | None
                    low: float | None
                    high: float | None
                    rmse: float | None
                    coverage: float | None
                    coverage_low: float | None
                    coverage_high: float | None
                    se_ratio: float | None
                    ess: float | None
                    maximum_weight: float | None
                    if rows:
                        values = np.asarray([row.estimate for row in rows], dtype=float)
                        ses = np.asarray(
                            [row.standard_error for row in rows], dtype=float
                        )
                        biases = values - rows[0].truth
                        low, high = _mean_interval(biases)
                        coverage = float(np.mean([row.covered for row in rows]))
                        coverage_low, coverage_high = proportion_interval(
                            int(sum(row.covered for row in rows)),
                            len(rows),
                        )
                        empirical_se = float(values.std(ddof=1))
                        se_ratio = (
                            None
                            if empirical_se <= 0
                            else float(np.sqrt(np.mean(ses**2)) / empirical_se)
                        )
                        bias = float(biases.mean())
                        rmse = float(np.sqrt(np.mean(biases**2)))
                        ess = float(
                            np.mean([row.effective_sample_fraction for row in rows])
                        )
                        maximum_weight = float(
                            np.median([row.maximum_weight for row in rows])
                        )
                    else:
                        bias = low = high = rmse = coverage = coverage_low = (
                            coverage_high
                        ) = None
                        se_ratio = ess = maximum_weight = None
                    output.append(
                        ObservationCellSummaryV1(
                            trial_id=source_trial.trial_id,
                            cell_id=cell.cell_id,
                            arm_id=arm_id,
                            route=typed_route,
                            truth=_truth_slope(source_trial, arm_id),
                            worlds=cell.worlds_per_trial,
                            successful_worlds=len(rows),
                            failures=failures[key],
                            bias=bias,
                            bias_ci_low=low,
                            bias_ci_high=high,
                            rmse=rmse,
                            coverage=coverage,
                            coverage_ci_low=(
                                None if coverage_low is None else float(coverage_low)
                            ),
                            coverage_ci_high=(
                                None if coverage_high is None else float(coverage_high)
                            ),
                            model_to_empirical_se_ratio=se_ratio,
                            mean_effective_sample_fraction=ess,
                            median_maximum_weight=maximum_weight,
                        )
                    )
    return output


def _dropout_response(
    estimates: list[DropoutWorldEstimateV1],
    design: LongitudinalObservationDesignV1,
) -> list[DropoutDoseResponseV1]:
    output = []
    for trial in design.trials:
        trial_id = trial.qualification.trial_id
        rows = [row for row in estimates if row.trial_id == trial_id]
        by_world: defaultdict[int, list[DropoutWorldEstimateV1]] = defaultdict(list)
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
                float(np.polyfit(world_levels, [row.estimate for row in ordered], 1)[0])
            )
            levels = world_levels
        if len(slopes) < 2 or levels is None:
            continue
        low, high = _mean_interval(np.asarray(slopes, dtype=float))
        output.append(
            DropoutDoseResponseV1(
                trial_id=trial_id,
                worlds=len(slopes),
                coefficient_levels=levels,
                mean_slope=float(np.mean(slopes)),
                slope_ci_low=low,
                slope_ci_high=high,
            )
        )
    return output


def _treatment_response(
    estimates: list[ObservationWorldEstimateV1],
    design: LongitudinalObservationDesignV1,
) -> list[ObservationDoseResponseV1]:
    cell_level = {
        cell.cell_id: cell.lagged_outcome_coefficient
        for cell in design.cells
        if cell.mechanism == "lagged_outcome"
    }
    output = []
    for trial in design.trials:
        source_trial = trial.qualification
        for arm_id in source_trial.arm_ids:
            if arm_id == source_trial.control_arm_id:
                continue
            for route in ("available_case", "estimated_ipcw", "oracle_ipcw"):
                typed_route: ObservationRoute = route
                rows = [
                    row
                    for row in estimates
                    if row.trial_id == source_trial.trial_id
                    and row.arm_id == arm_id
                    and row.route == typed_route
                    and row.cell_id in cell_level
                ]
                by_world: defaultdict[int, list[ObservationWorldEstimateV1]] = (
                    defaultdict(list)
                )
                for row in rows:
                    by_world[row.world_index].append(row)
                slopes = []
                levels: tuple[float, ...] | None = None
                for world_rows in by_world.values():
                    ordered = sorted(
                        world_rows, key=lambda row: cell_level[row.cell_id]
                    )
                    world_levels = tuple(cell_level[row.cell_id] for row in ordered)
                    if len(set(world_levels)) < 4:
                        continue
                    slopes.append(
                        float(
                            np.polyfit(
                                world_levels, [row.estimate for row in ordered], 1
                            )[0]
                        )
                    )
                    levels = world_levels
                if len(slopes) < 2 or levels is None:
                    continue
                low, high = _mean_interval(np.asarray(slopes, dtype=float))
                output.append(
                    ObservationDoseResponseV1(
                        trial_id=source_trial.trial_id,
                        arm_id=arm_id,
                        route=typed_route,
                        worlds=len(slopes),
                        coefficient_levels=levels,
                        mean_slope=float(np.mean(slopes)),
                        slope_ci_low=low,
                        slope_ci_high=high,
                    )
                )
    return output


def _route_contrasts(
    estimates: list[ObservationWorldEstimateV1],
    design: LongitudinalObservationDesignV1,
) -> list[ObservationRouteContrastV1]:
    by_key = {
        (row.trial_id, row.cell_id, row.arm_id, row.route, row.world_index): row
        for row in estimates
    }
    output = []
    for trial in design.trials:
        source_trial = trial.qualification
        for cell in design.cells:
            if cell.mechanism != "lagged_outcome":
                continue
            for arm_id in source_trial.arm_ids:
                if arm_id == source_trial.control_arm_id:
                    continue
                for correction_route in ("estimated_ipcw", "oracle_ipcw"):
                    typed_route: CorrectionRoute = correction_route
                    reductions = []
                    available_errors = []
                    corrected_errors = []
                    for world_index in range(cell.worlds_per_trial):
                        available = by_key.get(
                            (
                                source_trial.trial_id,
                                cell.cell_id,
                                arm_id,
                                "available_case",
                                world_index,
                            )
                        )
                        corrected = by_key.get(
                            (
                                source_trial.trial_id,
                                cell.cell_id,
                                arm_id,
                                typed_route,
                                world_index,
                            )
                        )
                        if available is None or corrected is None:
                            continue
                        available_error = available.estimate - available.truth
                        corrected_error = corrected.estimate - corrected.truth
                        available_errors.append(available_error)
                        corrected_errors.append(corrected_error)
                        reductions.append(abs(available_error) - abs(corrected_error))
                    if len(reductions) < 2:
                        continue
                    values = np.asarray(reductions, dtype=float)
                    low, high = _mean_interval(values)
                    improved = values > 0
                    fraction_bounds = proportion_interval(
                        int(improved.sum()),
                        len(improved),
                    )
                    available_rmse = float(
                        np.sqrt(
                            np.mean(
                                np.square(np.asarray(available_errors, dtype=float))
                            )
                        )
                    )
                    corrected_rmse = float(
                        np.sqrt(
                            np.mean(
                                np.square(np.asarray(corrected_errors, dtype=float))
                            )
                        )
                    )
                    if available_rmse <= 0:
                        raise ValueError(
                            "Available-case RMSE must be positive for a route contrast."
                        )
                    output.append(
                        ObservationRouteContrastV1(
                            trial_id=source_trial.trial_id,
                            cell_id=cell.cell_id,
                            arm_id=arm_id,
                            correction_route=typed_route,
                            lagged_outcome_coefficient=cell.lagged_outcome_coefficient,
                            worlds=len(values),
                            mean_absolute_error_reduction=float(values.mean()),
                            reduction_ci_low=low,
                            reduction_ci_high=high,
                            improvement_fraction=float(improved.mean()),
                            improvement_fraction_ci_low=float(fraction_bounds[0]),
                            improvement_fraction_ci_high=float(fraction_bounds[1]),
                            rmse_ratio=corrected_rmse / available_rmse,
                        )
                    )
    return output


def _group_dropout(
    estimates: list[DropoutWorldEstimateV1],
) -> defaultdict[tuple[str, str], list[DropoutWorldEstimateV1]]:
    groups: defaultdict[tuple[str, str], list[DropoutWorldEstimateV1]] = defaultdict(
        list
    )
    for row in estimates:
        groups[(row.trial_id, row.cell_id)].append(row)
    return groups


def _mean_interval(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Observation summary requires at least two finite values.")
    half = float(
        student_t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    mean = float(values.mean())
    return mean - half, mean + half


def _expit(values: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    return np.asarray(1.0 / (1.0 + np.exp(-array)), dtype=float)


def _json_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _world_seed(seed: int, trial_id: str, world_index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{trial_id}:{world_index}".encode()).digest()[:4],
        byteorder="big",
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
    return f"world_{digest[:20]}"


def _release_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Observation release path escapes its root: {relative!r}.")
    return candidate


__all__ = [
    "LongitudinalObservationDesignV1",
    "LongitudinalObservationReceiptV1",
    "LongitudinalObservationReportV1",
    "ObservationCellV1",
    "ObservationTrialV1",
    "ObservationWorldReceiptV1",
    "evaluate_longitudinal_observation",
]
