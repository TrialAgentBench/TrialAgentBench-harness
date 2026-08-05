"""Independent recovery of native clinical-mechanism stress worlds."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, spearmanr
from scipy.stats import t as student_t
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError

from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.process_pool import (
    single_threaded_numerical_process_pool,
)

Family = Literal["time_varying_hazard", "recurrent_adverse_event"]
Parameter = Literal[
    "early_log_hazard_ratio",
    "late_log_hazard_ratio",
    "hazard_contrast",
    "log_rate_ratio",
    "gamma_frailty_variance",
]
Route = Literal[
    "segmented_cox",
    "binary_endpoint",
    "poisson_rate",
    "binary_any_event",
    "nb2_profile_likelihood",
]
ResponseAxis = Literal[
    "hazard_contrast",
    "treatment_log_rate_ratio",
    "subject_rate_frailty_variance",
]

_PROFILE_CONFIDENCE_LEVEL = 0.95
_PROFILE_LR_CUTOFF = float(chi2.ppf(_PROFILE_CONFIDENCE_LEVEL, df=1))
_PROFILE_ALPHA_FLOOR = 1e-6
_PROFILE_ALPHA_CEILING = 50.0
_PROFILE_LOG_ALPHA_TOLERANCE = 1e-5
_PROFILE_ROOT_TOLERANCE = 1e-6


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _AnchorV1(_FrozenModel):
    anchor_id: str = Field(pattern=r"^anchor_[0-9a-f]{16}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _CellV1(_FrozenModel):
    cell_id: str
    family: Family
    response_axis: ResponseAxis
    level: float = Field(ge=0, allow_inf_nan=False)
    worlds_per_anchor: int = Field(ge=2)
    sample_size_multiplier: float = Field(gt=0, allow_inf_nan=False)
    minimum_sample_size: int = Field(ge=20)
    early_treatment_log_hazard_ratio: float | None = Field(
        default=None, allow_inf_nan=False
    )
    late_treatment_log_hazard_ratio: float | None = Field(
        default=None, allow_inf_nan=False
    )
    treatment_log_rate_ratio: float | None = Field(default=None, allow_inf_nan=False)
    subject_rate_frailty_variance: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _family_parameters(self) -> _CellV1:
        hazard = (
            self.early_treatment_log_hazard_ratio,
            self.late_treatment_log_hazard_ratio,
        )
        if self.family == "time_varying_hazard":
            if (
                self.response_axis != "hazard_contrast"
                or any(value is None for value in hazard)
                or self.treatment_log_rate_ratio is not None
                or self.subject_rate_frailty_variance != 0.0
            ):
                raise ValueError("Time-varying-hazard cell truth is incomplete.")
            early = cast(float, hazard[0])
            late = cast(float, hazard[1])
            expected = abs(late - early)
        else:
            if (
                any(value is not None for value in hazard)
                or self.treatment_log_rate_ratio is None
            ):
                raise ValueError("Recurrent-adverse-event cell truth is incomplete.")
            if self.response_axis == "treatment_log_rate_ratio":
                expected = abs(self.treatment_log_rate_ratio)
            elif self.response_axis == "subject_rate_frailty_variance":
                if self.treatment_log_rate_ratio != 0.0:
                    raise ValueError(
                        "Frailty-response cells require a null treatment effect."
                    )
                expected = self.subject_rate_frailty_variance
            else:
                raise ValueError(
                    "Recurrent-adverse-event cell response axis is invalid."
                )
        if abs(self.level - expected) > 1e-12:
            raise ValueError(
                "Native stress cell level does not match its configured contrast."
            )
        return self


class NativeStressPublicDesignV1(_FrozenModel):
    """Path-free native stress design."""

    schema_id: Literal["trialagentbench.native_stress_design/v1"]
    anchors: tuple[_AnchorV1, ...] = Field(min_length=2)
    cells: tuple[_CellV1, ...] = Field(min_length=6)
    seed: int = Field(ge=0, le=2**32 - 1)
    followup_horizon_dy: float = Field(gt=1, allow_inf_nan=False)
    interval_width_dy: float = Field(gt=0, allow_inf_nan=False)
    change_point_dy: float = Field(gt=0, allow_inf_nan=False)
    minimum_late_support_participants: int = Field(ge=2)
    adverse_event_baseline_rate_per_day: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _complete(self) -> NativeStressPublicDesignV1:
        if len({row.anchor_id for row in self.anchors}) != len(self.anchors):
            raise ValueError("Native stress anchors must be unique.")
        if len({row.cell_id for row in self.cells}) != len(self.cells):
            raise ValueError("Native stress cells must be unique.")
        if {row.family for row in self.cells} != {
            "time_varying_hazard",
            "recurrent_adverse_event",
        }:
            raise ValueError("Native stress design requires both mechanism families.")
        for family, response_axis in {
            (row.family, row.response_axis) for row in self.cells
        }:
            levels = [
                row.level
                for row in self.cells
                if row.family == family and row.response_axis == response_axis
            ]
            if len(levels) < 3 or 0.0 not in levels or len(levels) != len(set(levels)):
                raise ValueError(
                    "Each native stress response axis requires unique null and positive dose levels."
                )
        if not self.change_point_dy < self.followup_horizon_dy:
            raise ValueError("Native stress change point must precede follow-up.")
        intervals = self.followup_horizon_dy / self.interval_width_dy
        change_intervals = self.change_point_dy / self.interval_width_dy
        if (
            abs(intervals - round(intervals)) > 1e-12
            or abs(change_intervals - round(change_intervals)) > 1e-12
        ):
            raise ValueError(
                "Native stress follow-up and change point must align with complete intervals."
            )
        return self


class _WorldV1(_FrozenModel):
    world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    anchor_id: str = Field(pattern=r"^anchor_[0-9a-f]{16}$")
    cell_id: str
    family: Family
    world_index: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    subjects: int = Field(ge=20)
    events: int = Field(ge=0)
    analysis_path: str = Field(pattern=r"^worlds/world_[0-9a-f]{20}\.parquet$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ground_truth_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resampling_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class NativeStressPublicReceiptV1(_FrozenModel):
    """Public inventory independently bound to a design."""

    schema_id: Literal["trialagentbench.native_stress_receipt/v1"]
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[_WorldV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _checksum(self) -> NativeStressPublicReceiptV1:
        payload = self.model_dump(mode="json")
        supplied = str(payload.pop("checksum"))
        if _sha256_json(payload) != supplied:
            raise ValueError(
                "Native stress receipt checksum does not match its payload."
            )
        identities = [
            (row.anchor_id, row.cell_id, row.world_index) for row in self.worlds
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Native stress receipt world identities must be unique.")
        return self


class NativeStressWorldEstimateV1(_FrozenModel):
    """One independently estimated parameter in one world."""

    world_id: str
    anchor_id: str
    cell_id: str
    world_index: int
    family: Family
    parameter: Parameter
    route: Route
    truth: float = Field(allow_inf_nan=False)
    subjects: int = Field(ge=20)
    events: int = Field(ge=0)
    estimate: float | None = Field(default=None, allow_inf_nan=False)
    standard_error: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    interval_low: float | None = Field(default=None, allow_inf_nan=False)
    interval_high: float | None = Field(default=None, allow_inf_nan=False)
    covered: bool | None = None
    rejected_null: bool | None = None
    failure: Literal["convergence", "singular_information", "invalid_world"] | None = (
        None
    )

    @model_validator(mode="after")
    def _result_or_failure(self) -> NativeStressWorldEstimateV1:
        complete = all(
            value is not None
            for value in (
                self.estimate,
                self.interval_low,
                self.interval_high,
                self.covered,
                self.rejected_null,
            )
        )
        if complete == (self.failure is not None):
            raise ValueError(
                "Native stress estimate requires either a complete result or one failure."
            )
        if complete and not (
            cast(float, self.interval_low)
            <= cast(float, self.estimate)
            <= cast(float, self.interval_high)
        ):
            raise ValueError("Native stress interval must contain its estimate.")
        return self


class NativeStressCellSummaryV1(_FrozenModel):
    """Equal-anchor operating characteristics for one cell and route."""

    cell_id: str
    family: Family
    level: float = Field(ge=0, allow_inf_nan=False)
    parameter: Parameter
    route: Route
    anchors: int = Field(ge=2)
    worlds: int = Field(ge=2)
    successful_worlds: int = Field(ge=0)
    failures: int = Field(ge=0)
    mean_estimate: float | None = Field(default=None, allow_inf_nan=False)
    bias: float | None = Field(default=None, allow_inf_nan=False)
    bias_ci_low: float | None = Field(default=None, allow_inf_nan=False)
    bias_ci_high: float | None = Field(default=None, allow_inf_nan=False)
    rmse: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    coverage: float | None = Field(default=None, ge=0, le=1)
    coverage_ci_low: float | None = Field(default=None, ge=0, le=1)
    coverage_ci_high: float | None = Field(default=None, ge=0, le=1)
    coverage_scheduled: float = Field(ge=0, le=1)
    coverage_scheduled_ci_low: float = Field(ge=0, le=1)
    coverage_scheduled_ci_high: float = Field(ge=0, le=1)
    rejection_rate: float | None = Field(default=None, ge=0, le=1)
    rejection_rate_ci_low: float | None = Field(default=None, ge=0, le=1)
    rejection_rate_ci_high: float | None = Field(default=None, ge=0, le=1)
    rejection_rate_scheduled: float = Field(ge=0, le=1)
    rejection_rate_scheduled_ci_low: float = Field(ge=0, le=1)
    rejection_rate_scheduled_ci_high: float = Field(ge=0, le=1)
    model_to_empirical_se_ratio: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    model_to_empirical_se_ratio_ci_low: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    model_to_empirical_se_ratio_ci_high: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def _ordered_intervals(self) -> NativeStressCellSummaryV1:
        if self.successful_worlds + self.failures != self.worlds:
            raise ValueError(
                "Native stress successful and failed fits must partition scheduled worlds."
            )
        for estimate, low, high in (
            (self.bias, self.bias_ci_low, self.bias_ci_high),
            (self.coverage, self.coverage_ci_low, self.coverage_ci_high),
            (
                self.coverage_scheduled,
                self.coverage_scheduled_ci_low,
                self.coverage_scheduled_ci_high,
            ),
            (
                self.rejection_rate,
                self.rejection_rate_ci_low,
                self.rejection_rate_ci_high,
            ),
            (
                self.rejection_rate_scheduled,
                self.rejection_rate_scheduled_ci_low,
                self.rejection_rate_scheduled_ci_high,
            ),
            (
                self.model_to_empirical_se_ratio,
                self.model_to_empirical_se_ratio_ci_low,
                self.model_to_empirical_se_ratio_ci_high,
            ),
        ):
            if (low is None) != (high is None):
                raise ValueError(
                    "Native stress interval bounds must be present together."
                )
            if (
                low is not None
                and high is not None
                and (estimate is None or not low - 1e-12 <= estimate <= high + 1e-12)
            ):
                raise ValueError("Native stress interval must contain its estimate.")
        return self


class NativeStressResponseCurveV1(_FrozenModel):
    """Within-anchor estimated response to controlled mechanism strength."""

    family: Family
    route: Route
    parameter: Parameter
    anchors: int = Field(ge=2)
    points_per_anchor: int = Field(ge=3)
    slope_mean: float = Field(allow_inf_nan=False)
    slope_ci_low: float = Field(allow_inf_nan=False)
    slope_ci_high: float = Field(allow_inf_nan=False)
    expected_slope: float | None = Field(default=None, allow_inf_nan=False)


class NativeStressWorldMechanismV1(_FrozenModel):
    """Directly observed mechanism information in one released world."""

    world_id: str
    anchor_id: str
    cell_id: str
    family: Family
    subjects: int = Field(ge=20)
    events: int = Field(ge=0)
    early_events: int | None = Field(default=None, ge=0)
    late_events: int | None = Field(default=None, ge=0)
    late_risk_set_fraction: float | None = Field(default=None, ge=0, le=1)
    mean_followup: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    any_event_fraction: float | None = Field(default=None, ge=0, le=1)
    multiple_event_fraction: float | None = Field(default=None, ge=0, le=1)
    followup_varies: bool | None = None
    visit_followup_spearman: float | None = Field(default=None, ge=-1, le=1)
    configured_frailty_variance: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    direct_frailty_moment: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def _family_fields(self) -> NativeStressWorldMechanismV1:
        hazard = (self.early_events, self.late_events, self.late_risk_set_fraction)
        safety = (
            self.mean_followup,
            self.any_event_fraction,
            self.multiple_event_fraction,
            self.followup_varies,
            self.configured_frailty_variance,
            self.direct_frailty_moment,
        )
        if self.family == "time_varying_hazard":
            if (
                any(value is None for value in hazard)
                or any(value is not None for value in safety)
                or self.visit_followup_spearman is not None
            ):
                raise ValueError("Time-varying-hazard mechanism fields are incomplete.")
        else:
            if any(value is not None for value in hazard) or any(
                value is None for value in safety
            ):
                raise ValueError(
                    "Recurrent-adverse-event mechanism fields are incomplete."
                )
            if self.followup_varies == (self.visit_followup_spearman is None):
                raise ValueError(
                    "Follow-up association availability must match realized follow-up variation."
                )
        return self


class NativeStressRecoveryReportV1(_FrozenModel):
    """Complete independent recovery report."""

    schema_id: Literal["trialagentbench.native_stress_recovery/v1"] = (
        "trialagentbench.native_stress_recovery/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimates: tuple[NativeStressWorldEstimateV1, ...] = Field(min_length=1)
    mechanisms: tuple[NativeStressWorldMechanismV1, ...] = Field(min_length=1)
    cells: tuple[NativeStressCellSummaryV1, ...] = Field(min_length=1)
    curves: tuple[NativeStressResponseCurveV1, ...] = Field(min_length=2)


def evaluate_native_stress_release(
    *,
    release_dir: Path,
    minimum_null_worlds_per_anchor: int = 100,
    minimum_nonnull_worlds_per_anchor: int = 50,
    workers: int = 1,
) -> NativeStressRecoveryReportV1:
    """Verify archive custody and recover all declared mechanism cells."""

    if minimum_null_worlds_per_anchor < 2 or minimum_nonnull_worlds_per_anchor < 2:
        raise ValueError("Native stress replication floors must each be at least two.")
    if workers < 1:
        raise ValueError("Native stress recovery workers must be at least one.")
    design_path = release_dir / "design.json"
    receipt_path = release_dir / "receipt.json"
    design_payload = json.loads(design_path.read_text(encoding="utf-8"))
    design = NativeStressPublicDesignV1.model_validate(design_payload)
    design_sha = _sha256_json(design_payload)
    receipt = NativeStressPublicReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    if receipt.design_sha256 != design_sha:
        raise ValueError("Native stress receipt does not match the public design.")
    cells = {row.cell_id: row for row in design.cells}
    anchors = {row.anchor_id for row in design.anchors}
    expected = {
        (anchor, cell.cell_id, index)
        for anchor in anchors
        for cell in cells.values()
        for index in range(cell.worlds_per_anchor)
    }
    observed = {(row.anchor_id, row.cell_id, row.world_index) for row in receipt.worlds}
    if observed != expected:
        raise ValueError("Native stress receipt does not contain the complete design.")
    arguments = []
    for world in receipt.worlds:
        cell = cells.get(world.cell_id)
        if cell is None or world.family != cell.family:
            raise ValueError(
                f"Native stress world has inconsistent cell identity: {world.world_id}."
            )
        if world.anchor_id not in anchors:
            raise ValueError(
                f"Native stress world has unknown anchor: {world.world_id}."
            )
        if world.seed != _world_seed(
            design.seed, world.anchor_id, world.cell_id, world.world_index
        ):
            raise ValueError(f"Native stress world seed mismatch: {world.world_id}.")
        if world.world_id != _world_id(
            design_sha, world.anchor_id, world.cell_id, world.world_index
        ):
            raise ValueError(
                f"Native stress world identity mismatch: {world.world_id}."
            )
        arguments.append(
            (
                release_dir,
                world,
                cell,
                design.followup_horizon_dy,
                design.change_point_dy,
                design.adverse_event_baseline_rate_per_day,
            )
        )
    if workers == 1:
        world_results = [_evaluate_native_world(*argument) for argument in arguments]
    else:
        with single_threaded_numerical_process_pool(
            workers=min(workers, len(arguments))
        ) as executor:
            futures = [
                executor.submit(_evaluate_native_world, *argument)
                for argument in arguments
            ]
            world_results = [future.result() for future in futures]
    mechanisms = [mechanism for mechanism, _ in world_results]
    estimates = [
        estimate for _, world_estimates in world_results for estimate in world_estimates
    ]
    short = {
        cell.cell_id: cell.worlds_per_anchor
        for cell in cells.values()
        if cell.worlds_per_anchor
        < (
            minimum_null_worlds_per_anchor
            if cell.level == 0.0
            else minimum_nonnull_worlds_per_anchor
        )
    }
    if short:
        raise ValueError(
            "Native stress cells do not meet per-anchor replication floors: "
            f"{short!r}."
        )
    summaries = _summaries(estimates, cells)
    curves = _curves(estimates, cells)
    return NativeStressRecoveryReportV1(
        design_sha256=design_sha,
        receipt_sha256=sha256_file(receipt_path),
        estimates=tuple(estimates),
        mechanisms=tuple(mechanisms),
        cells=tuple(summaries),
        curves=tuple(curves),
    )


def _evaluate_native_world(
    release_dir: Path,
    world: _WorldV1,
    cell: _CellV1,
    followup_horizon_dy: float,
    change_point_dy: float,
    adverse_event_baseline_rate_per_day: float,
) -> tuple[NativeStressWorldMechanismV1, tuple[NativeStressWorldEstimateV1, ...]]:
    """Verify and evaluate one independently released stress world."""

    path = _release_path(release_dir, world.analysis_path)
    if sha256_file(path) != world.analysis_sha256:
        raise ValueError(f"Native stress analysis checksum mismatch: {world.world_id}.")
    frame = _validate_frame(
        pd.read_parquet(path),
        world,
        followup_horizon_dy=followup_horizon_dy,
    )
    mechanism = _mechanism_summary(
        frame,
        world,
        cell=cell,
        change_point=change_point_dy,
        adverse_event_baseline_rate_per_day=adverse_event_baseline_rate_per_day,
    )
    estimates = (
        _fit_hazard_world(frame, world, cell, change_point_dy)
        if cell.family == "time_varying_hazard"
        else (
            _fit_safety_world(frame, world, cell)
            if cell.response_axis == "treatment_log_rate_ratio"
            else _fit_frailty_world(frame, world, cell)
        )
    )
    return mechanism, tuple(estimates)


def _mechanism_summary(
    frame: pd.DataFrame,
    world: _WorldV1,
    *,
    cell: _CellV1,
    change_point: float,
    adverse_event_baseline_rate_per_day: float,
) -> NativeStressWorldMechanismV1:
    """Summarize realized information without fitting an outcome model."""

    if world.family == "time_varying_hazard":
        return NativeStressWorldMechanismV1(
            world_id=world.world_id,
            anchor_id=world.anchor_id,
            cell_id=world.cell_id,
            family=world.family,
            subjects=world.subjects,
            events=world.events,
            early_events=int(
                ((frame["event"] == 1) & (frame["time"] <= change_point)).sum()
            ),
            late_events=int(
                ((frame["event"] == 1) & (frame["time"] > change_point)).sum()
            ),
            late_risk_set_fraction=float((frame["time"] > change_point).mean()),
        )
    counts = frame["recurrent_event_count"].to_numpy(dtype=float)
    if cell.treatment_log_rate_ratio is None:
        raise ValueError("Recurrent-event cell omits its configured treatment effect.")
    expected_counts = (
        adverse_event_baseline_rate_per_day
        * np.exp(
            float(cell.treatment_log_rate_ratio)
            * frame["treatment"].to_numpy(dtype=float)
        )
        * frame["followup"].to_numpy(dtype=float)
    )
    squared_mean_sum = float(np.square(expected_counts).sum())
    if squared_mean_sum <= 0.0:
        raise ValueError("Recurrent-event expected counts have no information.")
    direct_frailty_moment = float(
        (np.square(counts - expected_counts) - counts).sum() / squared_mean_sum
    )
    followup_varies = bool(frame["followup"].nunique() > 1)
    association = (
        spearmanr(
            frame["empirical_visit_count_z"].to_numpy(dtype=float),
            frame["followup"].to_numpy(dtype=float),
        ).statistic
        if followup_varies
        else None
    )
    if association is not None and not np.isfinite(association):
        raise ValueError(
            f"Native stress follow-up association is undefined: {world.world_id}."
        )
    return NativeStressWorldMechanismV1(
        world_id=world.world_id,
        anchor_id=world.anchor_id,
        cell_id=world.cell_id,
        family=world.family,
        subjects=world.subjects,
        events=world.events,
        mean_followup=float(frame["followup"].mean()),
        any_event_fraction=float(np.mean(counts > 0)),
        multiple_event_fraction=float(np.mean(counts > 1)),
        followup_varies=followup_varies,
        visit_followup_spearman=None if association is None else float(association),
        configured_frailty_variance=cell.subject_rate_frailty_variance,
        direct_frailty_moment=direct_frailty_moment,
    )


def _fit_hazard_world(
    frame: pd.DataFrame,
    world: _WorldV1,
    cell: _CellV1,
    change_point: float,
) -> list[NativeStressWorldEstimateV1]:
    if (
        cell.early_treatment_log_hazard_ratio is None
        or cell.late_treatment_log_hazard_ratio is None
    ):
        raise ValueError("Time-varying-hazard cell omits its configured coefficients.")
    early = float(cell.early_treatment_log_hazard_ratio)
    late = float(cell.late_treatment_log_hazard_ratio)
    participant_ids = frame["participant_id"].astype("string").tolist()
    times = frame["time"].to_numpy(dtype=float)
    events = frame["event"].to_numpy(dtype=int)
    treatment = frame["treatment"].to_numpy(dtype=float)
    visit = frame["empirical_visit_count_z"].to_numpy(dtype=float)
    rows: list[tuple[str, float, float, int, float, float, float]] = []
    for participant_id, time, event, treated, visit_z in zip(
        participant_ids,
        times,
        events,
        treatment,
        visit,
        strict=True,
    ):
        stop = min(float(time), change_point)
        rows.append(
            (
                str(participant_id),
                0.0,
                stop,
                int(event == 1 and time <= change_point),
                float(treated),
                0.0,
                float(visit_z),
            )
        )
        if time > change_point:
            rows.append(
                (
                    str(participant_id),
                    change_point,
                    float(time),
                    int(event),
                    float(treated),
                    float(treated),
                    float(visit_z),
                )
            )
    split = pd.DataFrame(
        rows,
        columns=[
            "id",
            "start",
            "stop",
            "event",
            "treatment",
            "treatment_late",
            "visit_z",
        ],
    )
    truths: dict[Parameter, float] = {
        "early_log_hazard_ratio": early,
        "late_log_hazard_ratio": late,
        "hazard_contrast": late - early,
    }
    results: list[NativeStressWorldEstimateV1]
    failure: Literal["convergence", "singular_information"] | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", RuntimeWarning)
            fit = PHReg(
                endog=split["stop"],
                exog=split.loc[:, ["treatment", "treatment_late", "visit_z"]],
                status=split["event"],
                entry=split["start"],
                ties="breslow",
            ).fit()
        covariance = np.asarray(fit.cov_params(), dtype=float)
        _require_full_rank_covariance(covariance)
        contrasts: dict[Parameter, npt.NDArray[np.float64]] = {
            "early_log_hazard_ratio": np.asarray([1.0, 0.0, 0.0]),
            "late_log_hazard_ratio": np.asarray([1.0, 1.0, 0.0]),
            "hazard_contrast": np.asarray([0.0, 1.0, 0.0]),
        }
        results = [
            _complete(
                world,
                parameter=parameter,
                route="segmented_cox",
                truth=truths[parameter],
                estimate=float(vector @ fit.params),
                standard_error=float(np.sqrt(vector @ covariance @ vector)),
            )
            for parameter, vector in contrasts.items()
        ]
    except (ValueError, ConvergenceWarning):
        failure = "convergence"
    except (np.linalg.LinAlgError, RuntimeWarning):
        failure = "singular_information"
    if failure is not None:
        results = [
            _failed(
                world,
                parameter=parameter,
                route="segmented_cox",
                truth=truth,
                failure=failure,
            )
            for parameter, truth in truths.items()
        ]
    try:
        predictors = sm.add_constant(
            frame.loc[:, ["treatment", "empirical_visit_count_z"]], has_constant="add"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            shortcut = sm.GLM(
                frame["event"],
                predictors,
                family=sm.families.Binomial(),
            ).fit(cov_type="HC3")
        if not bool(shortcut.converged):
            raise ConvergenceWarning("Binary endpoint shortcut did not converge.")
        results.append(
            _complete(
                world,
                parameter="hazard_contrast",
                route="binary_endpoint",
                truth=late - early,
                estimate=float(shortcut.params["treatment"]),
                standard_error=float(shortcut.bse["treatment"]),
            )
        )
    except (ValueError, ConvergenceWarning, PerfectSeparationError):
        results.append(
            _failed(
                world,
                parameter="hazard_contrast",
                route="binary_endpoint",
                truth=late - early,
                failure="convergence",
            )
        )
    except np.linalg.LinAlgError:
        results.append(
            _failed(
                world,
                parameter="hazard_contrast",
                route="binary_endpoint",
                truth=late - early,
                failure="singular_information",
            )
        )
    return results


def _require_full_rank_covariance(covariance: npt.NDArray[np.float64]) -> None:
    """Reject covariance matrices whose information is numerically singular."""

    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not np.isfinite(covariance).all()
        or np.linalg.matrix_rank(covariance) != covariance.shape[0]
    ):
        raise np.linalg.LinAlgError("Estimator covariance is not finite and full rank.")


def _fit_safety_world(
    frame: pd.DataFrame,
    world: _WorldV1,
    cell: _CellV1,
) -> list[NativeStressWorldEstimateV1]:
    if cell.treatment_log_rate_ratio is None:
        raise ValueError(
            "Recurrent-adverse-event cell omits its configured log rate ratio."
        )
    truth = float(cell.treatment_log_rate_ratio)
    predictors = sm.add_constant(
        frame.loc[:, ["treatment", "empirical_visit_count_z"]], has_constant="add"
    )
    results = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fit = sm.GLM(
                frame["recurrent_event_count"],
                predictors,
                family=sm.families.Poisson(),
                offset=np.log(frame["followup"]),
            ).fit(cov_type="HC3")
        if not bool(fit.converged):
            raise ConvergenceWarning("Poisson rate model did not converge.")
        results.append(
            _complete(
                world,
                parameter="log_rate_ratio",
                route="poisson_rate",
                truth=truth,
                estimate=float(fit.params["treatment"]),
                standard_error=float(fit.bse["treatment"]),
            )
        )
    except (ValueError, ConvergenceWarning, PerfectSeparationError):
        results.append(
            _failed(
                world,
                parameter="log_rate_ratio",
                route="poisson_rate",
                truth=truth,
                failure="convergence",
            )
        )
    except np.linalg.LinAlgError:
        results.append(
            _failed(
                world,
                parameter="log_rate_ratio",
                route="poisson_rate",
                truth=truth,
                failure="singular_information",
            )
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            shortcut = sm.GLM(
                (frame["recurrent_event_count"] > 0).astype(int),
                predictors,
                family=sm.families.Binomial(),
            ).fit(cov_type="HC3")
        if not bool(shortcut.converged):
            raise ConvergenceWarning("Binary shortcut did not converge.")
        results.append(
            _complete(
                world,
                parameter="log_rate_ratio",
                route="binary_any_event",
                truth=truth,
                estimate=float(shortcut.params["treatment"]),
                standard_error=float(shortcut.bse["treatment"]),
            )
        )
    except (ValueError, ConvergenceWarning, PerfectSeparationError):
        results.append(
            _failed(
                world,
                parameter="log_rate_ratio",
                route="binary_any_event",
                truth=truth,
                failure="convergence",
            )
        )
    except np.linalg.LinAlgError:
        results.append(
            _failed(
                world,
                parameter="log_rate_ratio",
                route="binary_any_event",
                truth=truth,
                failure="singular_information",
            )
        )
    return results


def _fit_frailty_world(
    frame: pd.DataFrame,
    world: _WorldV1,
    cell: _CellV1,
) -> list[NativeStressWorldEstimateV1]:
    """Estimate NB2 frailty variance by profile likelihood."""

    truth = float(cell.subject_rate_frailty_variance)
    try:
        estimate, low, high = _nb2_profile_interval(
            counts=frame["recurrent_event_count"].to_numpy(dtype=float),
            predictors=sm.add_constant(
                frame.loc[:, ["treatment", "empirical_visit_count_z"]],
                has_constant="add",
            ).to_numpy(dtype=float),
            offset=np.log(frame["followup"].to_numpy(dtype=float)),
        )
        return [
            _complete(
                world,
                parameter="gamma_frailty_variance",
                route="nb2_profile_likelihood",
                truth=truth,
                estimate=estimate,
                interval_low=low,
                interval_high=high,
            )
        ]
    except (ValueError, ConvergenceWarning):
        return [
            _failed(
                world,
                parameter="gamma_frailty_variance",
                route="nb2_profile_likelihood",
                truth=truth,
                failure="convergence",
            )
        ]
    except np.linalg.LinAlgError:
        return [
            _failed(
                world,
                parameter="gamma_frailty_variance",
                route="nb2_profile_likelihood",
                truth=truth,
                failure="singular_information",
            )
        ]


def _nb2_profile_interval(
    *,
    counts: npt.NDArray[np.float64],
    predictors: npt.NDArray[np.float64],
    offset: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Fit an NB2 dispersion coefficient and invert its profile likelihood."""

    def profile_log_likelihood(alpha: float) -> float:
        family: sm.families.Family = (
            sm.families.Poisson()
            if alpha == 0.0
            else sm.families.NegativeBinomial(alpha=alpha)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fit = sm.GLM(
                counts,
                predictors,
                family=family,
                offset=offset,
            ).fit()
        if not bool(fit.converged) or not np.isfinite(fit.llf):
            raise ConvergenceWarning("NB2 profile mean model did not converge.")
        return float(fit.llf)

    poisson_log_likelihood = profile_log_likelihood(0.0)
    optimum = minimize_scalar(
        lambda log_alpha: -profile_log_likelihood(float(np.exp(log_alpha))),
        bounds=(np.log(_PROFILE_ALPHA_FLOOR), np.log(_PROFILE_ALPHA_CEILING)),
        method="bounded",
        options={"xatol": _PROFILE_LOG_ALPHA_TOLERANCE},
    )
    if not bool(optimum.success) or not np.isfinite(optimum.fun):
        raise ConvergenceWarning(
            "NB2 dispersion profile optimization did not converge."
        )
    positive_estimate = float(np.exp(optimum.x))
    positive_log_likelihood = profile_log_likelihood(positive_estimate)
    if poisson_log_likelihood >= positive_log_likelihood:
        estimate = 0.0
        maximum_log_likelihood = poisson_log_likelihood
    else:
        estimate = positive_estimate
        maximum_log_likelihood = positive_log_likelihood

    def profile_deviance(alpha: float) -> float:
        return (
            2.0 * (maximum_log_likelihood - profile_log_likelihood(alpha))
            - _PROFILE_LR_CUTOFF
        )

    null_deviance = 2.0 * (maximum_log_likelihood - poisson_log_likelihood)
    if null_deviance <= _PROFILE_LR_CUTOFF:
        low = 0.0
    else:
        low = float(
            brentq(
                profile_deviance,
                _PROFILE_ALPHA_FLOOR,
                estimate,
                xtol=_PROFILE_ROOT_TOLERANCE,
            )
        )
    upper_bracket = max(0.1, 2.0 * max(estimate, _PROFILE_ALPHA_FLOOR))
    while (
        upper_bracket < _PROFILE_ALPHA_CEILING
        and profile_deviance(upper_bracket) <= 0.0
    ):
        upper_bracket = min(_PROFILE_ALPHA_CEILING, 2.0 * upper_bracket)
    if profile_deviance(upper_bracket) <= 0.0:
        raise ConvergenceWarning(
            "NB2 dispersion profile has no finite upper interval bound."
        )
    high = float(
        brentq(
            profile_deviance,
            max(estimate, _PROFILE_ALPHA_FLOOR),
            upper_bracket,
            xtol=_PROFILE_ROOT_TOLERANCE,
        )
    )
    return estimate, low, high


def _complete(
    world: _WorldV1,
    *,
    parameter: Parameter,
    route: Route,
    truth: float,
    estimate: float,
    standard_error: float | None = None,
    interval_low: float | None = None,
    interval_high: float | None = None,
) -> NativeStressWorldEstimateV1:
    if not np.isfinite(estimate):
        return _failed(
            world,
            parameter=parameter,
            route=route,
            truth=truth,
            failure="singular_information",
        )
    if (interval_low is None) != (interval_high is None):
        raise ValueError("Native stress interval bounds must be supplied together.")
    if interval_low is None:
        if (
            standard_error is None
            or not np.isfinite(standard_error)
            or standard_error <= 0
        ):
            return _failed(
                world,
                parameter=parameter,
                route=route,
                truth=truth,
                failure="singular_information",
            )
        low = estimate - 1.96 * standard_error
        high = estimate + 1.96 * standard_error
    else:
        low = float(interval_low)
        high = cast(float, interval_high)
        if not np.isfinite([low, high]).all() or not low <= estimate <= high:
            return _failed(
                world,
                parameter=parameter,
                route=route,
                truth=truth,
                failure="singular_information",
            )
    return NativeStressWorldEstimateV1(
        world_id=world.world_id,
        anchor_id=world.anchor_id,
        cell_id=world.cell_id,
        world_index=world.world_index,
        family=world.family,
        parameter=parameter,
        route=route,
        truth=truth,
        subjects=world.subjects,
        events=world.events,
        estimate=estimate,
        standard_error=standard_error,
        interval_low=low,
        interval_high=high,
        covered=low - 1e-12 <= truth <= high + 1e-12,
        rejected_null=low > 0 or high < 0,
    )


def _failed(
    world: _WorldV1,
    *,
    parameter: Parameter,
    route: Route,
    truth: float,
    failure: Literal["convergence", "singular_information", "invalid_world"],
) -> NativeStressWorldEstimateV1:
    return NativeStressWorldEstimateV1(
        world_id=world.world_id,
        anchor_id=world.anchor_id,
        cell_id=world.cell_id,
        world_index=world.world_index,
        family=world.family,
        parameter=parameter,
        route=route,
        truth=truth,
        subjects=world.subjects,
        events=world.events,
        failure=failure,
    )


def _summaries(
    estimates: list[NativeStressWorldEstimateV1],
    cells: dict[str, _CellV1],
) -> list[NativeStressCellSummaryV1]:
    grouped: dict[tuple[str, Parameter, Route], list[NativeStressWorldEstimateV1]] = (
        defaultdict(list)
    )
    for row in estimates:
        grouped[(row.cell_id, row.parameter, row.route)].append(row)
    output = []
    for (cell_id, parameter, route), rows in sorted(grouped.items()):
        successful = [row for row in rows if row.failure is None]
        anchor_ids = sorted({row.anchor_id for row in rows})
        scheduled_metrics = np.asarray(
            [
                (
                    np.mean(
                        [row.covered is True for row in rows if row.anchor_id == anchor]
                    ),
                    np.mean(
                        [
                            row.rejected_null is True
                            for row in rows
                            if row.anchor_id == anchor
                        ]
                    ),
                )
                for anchor in anchor_ids
            ],
            dtype=float,
        )
        scheduled_coverage = _bounded_mean_ci(scheduled_metrics[:, 0])
        scheduled_rejection = _bounded_mean_ci(scheduled_metrics[:, 1])
        if not successful:
            output.append(
                NativeStressCellSummaryV1(
                    cell_id=cell_id,
                    family=cells[cell_id].family,
                    level=cells[cell_id].level,
                    parameter=parameter,
                    route=route,
                    anchors=len(anchor_ids),
                    worlds=len(rows),
                    successful_worlds=0,
                    failures=len(rows),
                    coverage_scheduled=scheduled_coverage[0],
                    coverage_scheduled_ci_low=scheduled_coverage[1],
                    coverage_scheduled_ci_high=scheduled_coverage[2],
                    rejection_rate_scheduled=scheduled_rejection[0],
                    rejection_rate_scheduled_ci_low=scheduled_rejection[1],
                    rejection_rate_scheduled_ci_high=scheduled_rejection[2],
                )
            )
            continue
        anchor_metrics = []
        for anchor in anchor_ids:
            values = [row for row in successful if row.anchor_id == anchor]
            if not values:
                continue
            estimates_a = np.asarray([cast(float, row.estimate) for row in values])
            truths_a = np.asarray([row.truth for row in values])
            ses = [row.standard_error for row in values]
            empirical_sd = (
                float(estimates_a.std(ddof=1)) if len(estimates_a) > 1 else np.nan
            )
            anchor_metrics.append(
                (
                    float(estimates_a.mean()),
                    float((estimates_a - truths_a).mean()),
                    float(np.sqrt(np.mean((estimates_a - truths_a) ** 2))),
                    float(np.mean([bool(row.covered) for row in values])),
                    float(np.mean([bool(row.rejected_null) for row in values])),
                    (
                        float(np.mean(cast(list[float], ses)) / empirical_sd)
                        if empirical_sd > 0 and all(value is not None for value in ses)
                        else np.nan
                    ),
                )
            )
        if len(anchor_metrics) < 2:
            raise ValueError(
                f"Native stress cell {cell_id}/{parameter}/{route} has fewer than two successful anchors."
            )
        metric = np.asarray(anchor_metrics, dtype=float)
        bias_mean, bias_low, bias_high = _mean_ci(metric[:, 1])
        coverage_mean, coverage_low, coverage_high = _bounded_mean_ci(metric[:, 3])
        rejection_mean, rejection_low, rejection_high = _bounded_mean_ci(metric[:, 4])
        se_ratios = metric[:, 5][np.isfinite(metric[:, 5])]
        if len(se_ratios) >= 2:
            se_ratio_mean, se_ratio_low, se_ratio_high = _mean_ci(se_ratios)
            se_ratio_interval: tuple[float | None, float | None] = (
                max(0.0, se_ratio_low),
                se_ratio_high,
            )
        else:
            se_ratio_mean = None
            se_ratio_interval = (None, None)
        output.append(
            NativeStressCellSummaryV1(
                cell_id=cell_id,
                family=cells[cell_id].family,
                level=cells[cell_id].level,
                parameter=parameter,
                route=route,
                anchors=len(anchor_ids),
                worlds=len(rows),
                successful_worlds=len(successful),
                failures=len(rows) - len(successful),
                mean_estimate=float(metric[:, 0].mean()),
                bias=bias_mean,
                bias_ci_low=bias_low,
                bias_ci_high=bias_high,
                rmse=float(metric[:, 2].mean()),
                coverage=coverage_mean,
                coverage_ci_low=coverage_low,
                coverage_ci_high=coverage_high,
                coverage_scheduled=scheduled_coverage[0],
                coverage_scheduled_ci_low=scheduled_coverage[1],
                coverage_scheduled_ci_high=scheduled_coverage[2],
                rejection_rate=rejection_mean,
                rejection_rate_ci_low=rejection_low,
                rejection_rate_ci_high=rejection_high,
                rejection_rate_scheduled=scheduled_rejection[0],
                rejection_rate_scheduled_ci_low=scheduled_rejection[1],
                rejection_rate_scheduled_ci_high=scheduled_rejection[2],
                model_to_empirical_se_ratio=se_ratio_mean,
                model_to_empirical_se_ratio_ci_low=se_ratio_interval[0],
                model_to_empirical_se_ratio_ci_high=se_ratio_interval[1],
            )
        )
    return output


def _curves(
    estimates: list[NativeStressWorldEstimateV1],
    cells: dict[str, _CellV1],
) -> list[NativeStressResponseCurveV1]:
    definitions: tuple[tuple[Family, Parameter, Route, float | None], ...] = (
        ("time_varying_hazard", "hazard_contrast", "segmented_cox", 1.0),
        ("time_varying_hazard", "hazard_contrast", "binary_endpoint", None),
        ("recurrent_adverse_event", "log_rate_ratio", "poisson_rate", 1.0),
        ("recurrent_adverse_event", "log_rate_ratio", "binary_any_event", None),
        (
            "recurrent_adverse_event",
            "gamma_frailty_variance",
            "nb2_profile_likelihood",
            1.0,
        ),
    )
    output = []
    for family, parameter, route, expected in definitions:
        selected = [
            row
            for row in estimates
            if row.family == family
            and row.parameter == parameter
            and row.route == route
            and row.failure is None
        ]
        slopes = []
        points = []
        for anchor in sorted({row.anchor_id for row in selected}):
            by_cell: dict[str, list[float]] = defaultdict(list)
            for row in selected:
                if row.anchor_id == anchor:
                    by_cell[row.cell_id].append(cast(float, row.estimate))
            if len(by_cell) < 3:
                continue
            x = np.asarray([_truth(cells[cell], parameter) for cell in sorted(by_cell)])
            y = np.asarray([np.mean(by_cell[cell]) for cell in sorted(by_cell)])
            slopes.append(float(np.polyfit(x, y, 1)[0]))
            points.append(len(x))
        if len(slopes) < 2 or len(set(points)) != 1:
            raise ValueError(
                f"Response curve {family}/{route} lacks complete independent anchors."
            )
        mean, low, high = _mean_ci(np.asarray(slopes))
        output.append(
            NativeStressResponseCurveV1(
                family=family,
                route=route,
                parameter=parameter,
                anchors=len(slopes),
                points_per_anchor=points[0],
                slope_mean=mean,
                slope_ci_low=low,
                slope_ci_high=high,
                expected_slope=expected,
            )
        )
    return output


def _truth(cell: _CellV1, parameter: Parameter) -> float:
    if parameter == "hazard_contrast":
        if (
            cell.late_treatment_log_hazard_ratio is None
            or cell.early_treatment_log_hazard_ratio is None
        ):
            raise ValueError("Hazard contrast cell omits early or late truth.")
        return float(cell.late_treatment_log_hazard_ratio) - float(
            cell.early_treatment_log_hazard_ratio
        )
    if parameter == "log_rate_ratio":
        if cell.treatment_log_rate_ratio is None:
            raise ValueError("Safety cell omits log rate-ratio truth.")
        return float(cell.treatment_log_rate_ratio)
    if parameter == "gamma_frailty_variance":
        return float(cell.subject_rate_frailty_variance)
    raise ValueError(f"Unsupported response parameter: {parameter}.")


def _validate_frame(
    frame: pd.DataFrame,
    world: _WorldV1,
    *,
    followup_horizon_dy: float,
) -> pd.DataFrame:
    common = {"participant_id", "treatment", "empirical_visit_count_z"}
    family = (
        {"time", "event"}
        if world.family == "time_varying_hazard"
        else {"followup", "recurrent_event_count"}
    )
    required = common | family
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(
            f"Native stress world is missing analysis columns: {missing!r}."
        )
    values = frame.loc[:, sorted(required)].copy()
    if len(values) != world.subjects or values["participant_id"].duplicated().any():
        raise ValueError(
            f"Native stress world has invalid participant identity: {world.world_id}."
        )
    for column in required - {"participant_id"}:
        values[column] = pd.to_numeric(values[column], errors="raise")
    if set(values["treatment"].unique()) != {0, 1}:
        raise ValueError(
            f"Native stress world requires both randomized arms: {world.world_id}."
        )
    if not np.isfinite(
        values.loc[:, list(required - {"participant_id"})].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"Native stress world contains non-finite analysis values: {world.world_id}."
        )
    if world.family == "time_varying_hazard":
        if (
            not set(values["event"].unique()).issubset({0, 1})
            or (values["time"] <= 0).any()
            or (values["time"] > followup_horizon_dy + 1e-12).any()
        ):
            raise ValueError(
                f"Native stress event-time values are invalid: {world.world_id}."
            )
        event_count = int(values["event"].sum())
    else:
        counts = values["recurrent_event_count"].to_numpy(dtype=float)
        if (
            (values["followup"] <= 0).any()
            or (values["followup"] > followup_horizon_dy + 1e-12).any()
            or (counts < 0).any()
            or not np.equal(counts, np.floor(counts)).all()
        ):
            raise ValueError(
                f"Native stress recurrent-event values are invalid: {world.world_id}."
            )
        event_count = int(counts.sum())
    if event_count != world.events:
        raise ValueError(
            f"Native stress event count does not match receipt: {world.world_id}."
        )
    return values


def _mean_ci(values: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    if len(values) < 2:
        raise ValueError("Mean uncertainty requires at least two independent anchors.")
    mean = float(values.mean())
    half = float(
        student_t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return mean, mean - half, mean + half


def _bounded_mean_ci(values: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    mean, low, high = _mean_ci(values)
    return mean, max(0.0, low), min(1.0, high)


def _release_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("Native stress analysis paths must be relative.")
    resolved = (root.resolve() / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(
            "Native stress analysis path is missing or escapes the release."
        )
    return resolved


def _sha256_json(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _world_seed(seed: int, anchor_id: str, cell_id: str, world_index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{anchor_id}:{cell_id}:{world_index}".encode()).digest()[
            :4
        ],
        byteorder="big",
    )


def _world_id(checksum: str, anchor_id: str, cell_id: str, world_index: int) -> str:
    value = hashlib.sha256(
        f"{checksum}:{anchor_id}:{cell_id}:{world_index}".encode()
    ).hexdigest()
    return f"world_{value[:20]}"
