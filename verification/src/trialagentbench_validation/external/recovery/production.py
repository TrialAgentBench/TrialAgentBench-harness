"""Independent recovery of production-core empirical qualification worlds."""

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
from scipy.stats import norm
from scipy.stats import t as student_t
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.process_pool import (
    single_threaded_numerical_process_pool,
)
from trialagentbench_validation.statistics import proportion_interval

_INTERVAL_ROUNDOFF_TOLERANCE = 1e-12


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _PublicAnchorV1(_FrozenModel):
    anchor_id: str = Field(pattern=r"^anchor_[0-9a-f]{16}$")
    source_subjects: int = Field(ge=20)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _PublicCellV1(_FrozenModel):
    cell_id: str = Field(min_length=1)
    response_axis: Literal[
        "source_size", "reference", "information", "effect", "mechanism"
    ]
    level: float = Field(allow_inf_nan=False)
    sample_size_multiplier: float = Field(gt=0, allow_inf_nan=False)
    minimum_sample_size: int = Field(ge=20)
    treatment_log_hazard_ratio: float = Field(allow_inf_nan=False)
    empirical_visit_count_log_hazard_ratio: float = Field(allow_inf_nan=False)
    baseline_hazard: float = Field(gt=0, allow_inf_nan=False)


class _ExcludedPublicAnchorV1(_FrozenModel):
    anchor_id: str = Field(pattern=r"^anchor_[0-9a-f]{16}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: Literal[
        "constant_empirical_visit_count",
        "insufficient_empirical_visit_count_support",
    ]


class ProductionCorePublicDesignV1(_FrozenModel):
    """Path-free qualification design released for independent replay."""

    schema_id: Literal["trialagentbench.production_core_qualification_design/v1"]
    anchors: tuple[_PublicAnchorV1, ...] = Field(min_length=1)
    excluded_anchors: tuple[_ExcludedPublicAnchorV1, ...] = ()
    cells: tuple[_PublicCellV1, ...] = Field(min_length=1)
    worlds_per_anchor_cell: int = Field(ge=2)
    seed: int = Field(ge=0, le=2**32 - 1)
    treatment_execution: Literal["randomized_fully_adherent"] = (
        "randomized_fully_adherent"
    )
    followup_horizon_dy: float = Field(gt=1, allow_inf_nan=False)
    interval_width_dy: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _complete_design(self) -> ProductionCorePublicDesignV1:
        anchor_ids = [anchor.anchor_id for anchor in self.anchors]
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("Production-core anchor identifiers must be unique.")
        excluded_ids = [anchor.anchor_id for anchor in self.excluded_anchors]
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError(
                "Excluded production-core anchor identifiers must be unique."
            )
        if set(anchor_ids) & set(excluded_ids):
            raise ValueError(
                "Production-core anchors cannot be both included and excluded."
            )
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Production-core cell identifiers must be unique.")
        source_size_cells = [
            cell for cell in self.cells if cell.response_axis == "source_size"
        ]
        if len(source_size_cells) != 1:
            raise ValueError(
                "Production-core design requires exactly one source-size cell."
            )
        source_size_cell = source_size_cells[0]
        if (
            source_size_cell.level != 1.0
            or source_size_cell.sample_size_multiplier != 1.0
        ):
            raise ValueError(
                "Source-size cell must use unit level and sample-size multiplier."
            )
        if source_size_cell.minimum_sample_size > min(
            anchor.source_subjects for anchor in self.anchors
        ):
            raise ValueError("Source-size cell cannot impose a sample-size floor.")
        intervals = self.followup_horizon_dy / self.interval_width_dy
        if abs(intervals - round(intervals)) > 1e-12:
            raise ValueError(
                "Production-core follow-up must contain complete intervals."
            )
        return self


class _WorldReceiptV1(_FrozenModel):
    world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    anchor_id: str = Field(pattern=r"^anchor_[0-9a-f]{16}$")
    cell_id: str = Field(min_length=1)
    world_index: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    subjects: int = Field(ge=20)
    events: int = Field(ge=0)
    truth_log_hazard_ratio: float = Field(allow_inf_nan=False)
    truth_empirical_visit_count_log_hazard_ratio: float = Field(allow_inf_nan=False)
    analysis_path: str = Field(pattern=r"^worlds/world_[0-9a-f]{20}\.parquet$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ground_truth_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resampling_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProductionCorePublicReceiptV1(_FrozenModel):
    """Public world inventory consumed without production-generator imports."""

    schema_id: Literal["trialagentbench.production_core_qualification_receipt/v1"]
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[_WorldReceiptV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _checksum_matches(self) -> ProductionCorePublicReceiptV1:
        payload = self.model_dump(mode="json")
        supplied = str(payload.pop("checksum"))
        if _sha256_json(payload) != supplied:
            raise ValueError(
                "Production-core receipt checksum does not match its payload."
            )
        identities = [
            (world.anchor_id, world.cell_id, world.world_index) for world in self.worlds
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Production-core world identities must be unique.")
        seeds = [world.seed for world in self.worlds]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Production-core world seeds must be unique.")
        return self


class ProductionCorePublicRequestV1(_FrozenModel):
    """Candidate-neutral qualification request released for verification."""

    schema_id: Literal["trialagentbench.production_core_qualification_request/v1"]
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimands: tuple[
        Literal[
            "treatment_log_hazard_ratio",
            "empirical_visit_count_log_hazard_ratio",
        ],
        ...,
    ] = Field(min_length=1)
    analysis_version: Literal["production_core_cox_v1"]
    estimator_version: Literal["statsmodels_phreg_breslow_v1"]
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _checksum_matches(self) -> ProductionCorePublicRequestV1:
        payload = self.model_dump(mode="json")
        supplied = str(payload.pop("checksum"))
        if _sha256_json(payload) != supplied:
            raise ValueError(
                "Production-core request checksum does not match its payload."
            )
        if len(self.estimands) != len(set(self.estimands)):
            raise ValueError("Production-core request estimands must be unique.")
        return self


class ProductionCorePublicRunV1(_FrozenModel):
    """Public binding between a candidate, request, design, and receipt."""

    schema_id: Literal["trialagentbench.production_core_qualification_run/v1"]
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _checksum_matches(self) -> ProductionCorePublicRunV1:
        payload = self.model_dump(mode="json")
        supplied = str(payload.pop("checksum"))
        if _sha256_json(payload) != supplied:
            raise ValueError("Production-core run checksum does not match its payload.")
        return self


class ProductionCoreWorldEstimateV1(_FrozenModel):
    """One independently fitted route in one synthetic world."""

    world_id: str
    anchor_id: str
    cell_id: str
    world_index: int
    estimand: Literal[
        "treatment_log_hazard_ratio", "empirical_visit_count_log_hazard_ratio"
    ]
    route: Literal["adjusted_cox", "binary_endpoint_shortcut"]
    truth_log_hazard_ratio: float = Field(allow_inf_nan=False)
    subjects: int = Field(ge=20)
    events: int = Field(ge=0)
    estimate: float | None = Field(default=None, allow_inf_nan=False)
    standard_error: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    covered: bool | None = None
    rejected_null: bool | None = None
    failure: Literal["convergence", "singular_information", "invalid_world"] | None = (
        None
    )

    @model_validator(mode="after")
    def _result_or_failure(self) -> ProductionCoreWorldEstimateV1:
        complete = all(
            value is not None
            for value in (
                self.estimate,
                self.standard_error,
                self.covered,
                self.rejected_null,
            )
        )
        if complete == (self.failure is not None):
            raise ValueError(
                "World estimate must contain either a complete result or one failure."
            )
        return self


class ProductionCoreCellSummaryV1(_FrozenModel):
    """Operating characteristics and Monte Carlo uncertainty for one cell."""

    anchor_id: str | None = Field(default=None, pattern=r"^anchor_[0-9a-f]{16}$")
    anchors: int = Field(ge=1)
    uncertainty_unit: Literal["world", "anchor"]
    cell_id: str
    response_axis: str
    level: float = Field(allow_inf_nan=False)
    estimand: Literal[
        "treatment_log_hazard_ratio", "empirical_visit_count_log_hazard_ratio"
    ]
    route: Literal["adjusted_cox", "binary_endpoint_shortcut"]
    truth_log_hazard_ratio: float = Field(allow_inf_nan=False)
    worlds: int = Field(ge=2)
    successful_worlds: int = Field(ge=0)
    failures: int = Field(ge=0)
    mean_subjects: float = Field(ge=20, allow_inf_nan=False)
    mean_events: float = Field(ge=0, allow_inf_nan=False)
    mean_event_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    mean_estimate: float | None = Field(default=None, allow_inf_nan=False)
    bias: float | None = Field(default=None, allow_inf_nan=False)
    bias_mcse: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    bias_ci_low: float | None = Field(default=None, allow_inf_nan=False)
    bias_ci_high: float | None = Field(default=None, allow_inf_nan=False)
    rmse: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    rmse_ci_low: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    rmse_ci_high: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    empirical_standard_deviation: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    mean_model_standard_error: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    model_to_empirical_se_ratio: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    model_to_empirical_se_ratio_ci_low: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    model_to_empirical_se_ratio_ci_high: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
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

    @model_validator(mode="after")
    def _ordered_intervals(self) -> ProductionCoreCellSummaryV1:
        for estimate, low, high in (
            (self.bias, self.bias_ci_low, self.bias_ci_high),
            (self.rmse, self.rmse_ci_low, self.rmse_ci_high),
            (
                self.model_to_empirical_se_ratio,
                self.model_to_empirical_se_ratio_ci_low,
                self.model_to_empirical_se_ratio_ci_high,
            ),
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
        ):
            if (low is None) != (high is None):
                raise ValueError(
                    "Cell-summary interval bounds must be present together."
                )
            if low is not None and high is not None:
                if estimate is None or not (
                    low - _INTERVAL_ROUNDOFF_TOLERANCE
                    <= estimate
                    <= high + _INTERVAL_ROUNDOFF_TOLERANCE
                ):
                    raise ValueError("Cell-summary interval must contain its estimate.")
        return self


class ProductionCoreResponseCurveV1(_FrozenModel):
    """Anchor-level response of an analysis result to a controlled input."""

    curve: Literal[
        "treatment_effect_recovery",
        "empirical_process_recovery",
        "information_precision",
        "time_discarding_route",
    ]
    estimand: Literal[
        "treatment_log_hazard_ratio", "empirical_visit_count_log_hazard_ratio"
    ]
    route: Literal["adjusted_cox", "binary_endpoint_shortcut"]
    response: Literal["mean_estimate", "empirical_standard_deviation"]
    input_scale: Literal["truth_log_hazard_ratio", "log_mean_subjects"]
    anchors: int = Field(ge=2)
    points_per_anchor: int = Field(ge=3)
    slope_mean: float = Field(allow_inf_nan=False)
    slope_ci_low: float = Field(allow_inf_nan=False)
    slope_ci_high: float = Field(allow_inf_nan=False)
    intercept_mean: float = Field(allow_inf_nan=False)
    intercept_ci_low: float = Field(allow_inf_nan=False)
    intercept_ci_high: float = Field(allow_inf_nan=False)
    expected_slope: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered_intervals(self) -> ProductionCoreResponseCurveV1:
        if not self.slope_ci_low <= self.slope_mean <= self.slope_ci_high:
            raise ValueError("Response-curve slope interval must contain its mean.")
        if not self.intercept_ci_low <= self.intercept_mean <= self.intercept_ci_high:
            raise ValueError("Response-curve intercept interval must contain its mean.")
        return self


class ProductionCoreRecoveryReportV1(_FrozenModel):
    """Independent production-core recovery report."""

    schema_id: Literal["trialagentbench.production_core_recovery_report/v1"] = (
        "trialagentbench.production_core_recovery_report/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimates: tuple[ProductionCoreWorldEstimateV1, ...] = Field(min_length=1)
    cells: tuple[ProductionCoreCellSummaryV1, ...] = Field(min_length=1)
    anchor_cells: tuple[ProductionCoreCellSummaryV1, ...] = Field(min_length=1)
    curves: tuple[ProductionCoreResponseCurveV1, ...] = ()


class ProductionCoreCandidateCellComparisonV1(_FrozenModel):
    """Matched operating-characteristic change between two candidates."""

    anchor_id: str | None = Field(default=None, pattern=r"^anchor_[0-9a-f]{16}$")
    cell_id: str
    estimand: Literal[
        "treatment_log_hazard_ratio",
        "empirical_visit_count_log_hazard_ratio",
    ]
    route: Literal["adjusted_cox", "binary_endpoint_shortcut"]
    worlds: int = Field(ge=2)
    successful_worlds_change: int
    failures_change: int
    mean_subjects_change: float = Field(allow_inf_nan=False)
    mean_events_change: float = Field(allow_inf_nan=False)
    mean_event_fraction_change: float = Field(allow_inf_nan=False)
    bias_change: float | None = Field(default=None, allow_inf_nan=False)
    rmse_change: float | None = Field(default=None, allow_inf_nan=False)
    model_to_empirical_se_ratio_change: float | None = Field(
        default=None, allow_inf_nan=False
    )
    coverage_change: float | None = Field(default=None, ge=-1, le=1)
    coverage_scheduled_change: float = Field(ge=-1, le=1)
    rejection_rate_change: float | None = Field(default=None, ge=-1, le=1)
    rejection_rate_scheduled_change: float = Field(ge=-1, le=1)


class ProductionCoreCandidateComparisonV1(_FrozenModel):
    """Strictly matched comparison of two production candidates."""

    schema_id: Literal["trialagentbench.production_core_candidate_comparison/v1"] = (
        "trialagentbench.production_core_candidate_comparison/v1"
    )
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison_candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cells: tuple[ProductionCoreCandidateCellComparisonV1, ...] = Field(min_length=1)


def compare_production_core_candidates(
    reference: ProductionCoreRecoveryReportV1,
    comparison: ProductionCoreRecoveryReportV1,
) -> ProductionCoreCandidateComparisonV1:
    """Compare only reports generated from the same frozen request."""

    if reference.candidate_sha256 == comparison.candidate_sha256:
        raise ValueError(
            "Candidate comparison requires two distinct candidate identities."
        )
    if (
        reference.request_sha256 != comparison.request_sha256
        or reference.design_sha256 != comparison.design_sha256
    ):
        raise ValueError(
            "Candidate comparison requires the same qualification request and design."
        )
    reference_cells = {_cell_key(row): row for row in reference.cells}
    comparison_cells = {_cell_key(row): row for row in comparison.cells}
    if len(reference_cells) != len(reference.cells) or len(comparison_cells) != len(
        comparison.cells
    ):
        raise ValueError(
            "Candidate comparison reports contain duplicate operating-characteristic cells."
        )
    if reference_cells.keys() != comparison_cells.keys():
        raise ValueError(
            "Candidate comparison requires exactly matched operating-characteristic cells."
        )
    cells = []
    for key in sorted(reference_cells):
        left = reference_cells[key]
        right = comparison_cells[key]
        if left.worlds != right.worlds:
            raise ValueError(
                "Candidate comparison requires equal scheduled-world denominators."
            )
        cells.append(
            ProductionCoreCandidateCellComparisonV1(
                anchor_id=left.anchor_id,
                cell_id=left.cell_id,
                estimand=left.estimand,
                route=left.route,
                worlds=left.worlds,
                successful_worlds_change=right.successful_worlds
                - left.successful_worlds,
                failures_change=right.failures - left.failures,
                mean_subjects_change=right.mean_subjects - left.mean_subjects,
                mean_events_change=right.mean_events - left.mean_events,
                mean_event_fraction_change=right.mean_event_fraction
                - left.mean_event_fraction,
                bias_change=_difference(right.bias, left.bias),
                rmse_change=_difference(right.rmse, left.rmse),
                model_to_empirical_se_ratio_change=_difference(
                    right.model_to_empirical_se_ratio,
                    left.model_to_empirical_se_ratio,
                ),
                coverage_change=_difference(right.coverage, left.coverage),
                coverage_scheduled_change=right.coverage_scheduled
                - left.coverage_scheduled,
                rejection_rate_change=_difference(
                    right.rejection_rate, left.rejection_rate
                ),
                rejection_rate_scheduled_change=(
                    right.rejection_rate_scheduled - left.rejection_rate_scheduled
                ),
            )
        )
    return ProductionCoreCandidateComparisonV1(
        request_sha256=reference.request_sha256,
        design_sha256=reference.design_sha256,
        reference_candidate_sha256=reference.candidate_sha256,
        comparison_candidate_sha256=comparison.candidate_sha256,
        cells=tuple(cells),
    )


def evaluate_production_core_release(
    *,
    release_dir: Path,
    minimum_worlds_per_cell: int = 100,
    workers: int = 1,
) -> ProductionCoreRecoveryReportV1:
    """Verify released bytes and independently recover every response cell."""

    if minimum_worlds_per_cell < 2:
        raise ValueError("minimum_worlds_per_cell must be at least two.")
    if workers < 1:
        raise ValueError("Production-core recovery workers must be at least one.")
    design_path = release_dir / "qualification_design.json"
    receipt_path = release_dir / "qualification_receipt.json"
    design_payload = json.loads(design_path.read_text(encoding="utf-8"))
    design = ProductionCorePublicDesignV1.model_validate(design_payload)
    if design.worlds_per_anchor_cell < minimum_worlds_per_cell:
        raise ValueError(
            "Qualification design does not meet the minimum worlds per anchor-cell: "
            f"observed={design.worlds_per_anchor_cell}, required={minimum_worlds_per_cell}."
        )
    design_sha256 = _sha256_json(design.model_dump(mode="json"))
    receipt = ProductionCorePublicReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    if receipt.design_sha256 != design_sha256:
        raise ValueError(
            "Qualification receipt does not match the released public design."
        )
    candidate_sha256, request_sha256 = _verify_run_binding(
        release_dir=release_dir,
        design_sha256=design_sha256,
        receipt_path=receipt_path,
    )
    cell_by_id = {cell.cell_id: cell for cell in design.cells}
    source_size_cell = next(
        cell.cell_id for cell in design.cells if cell.response_axis == "source_size"
    )
    anchor_ids = {anchor.anchor_id for anchor in design.anchors}
    source_subjects = {
        anchor.anchor_id: anchor.source_subjects for anchor in design.anchors
    }
    expected_worlds = {
        (anchor_id, cell_id, world_index)
        for anchor_id in anchor_ids
        for cell_id in cell_by_id
        for world_index in range(design.worlds_per_anchor_cell)
    }
    observed_worlds = {
        (world.anchor_id, world.cell_id, world.world_index) for world in receipt.worlds
    }
    if observed_worlds != expected_worlds:
        missing = sorted(expected_worlds - observed_worlds)
        unexpected = sorted(observed_worlds - expected_worlds)
        raise ValueError(
            "Qualification receipt does not contain the complete public design: "
            f"missing={missing[:5]!r}, unexpected={unexpected[:5]!r}."
        )
    for world in receipt.worlds:
        if world.cell_id not in cell_by_id:
            raise ValueError(
                f"World references unknown response cell {world.cell_id!r}."
            )
        cell = cell_by_id[world.cell_id]
        if (
            world.truth_log_hazard_ratio != cell.treatment_log_hazard_ratio
            or world.truth_empirical_visit_count_log_hazard_ratio
            != cell.empirical_visit_count_log_hazard_ratio
        ):
            raise ValueError(
                f"World truth does not match response cell for {world.world_id}."
            )
        expected_seed = _world_seed(
            design_seed=design.seed,
            anchor_id=world.anchor_id,
            cell_id=world.cell_id,
            world_index=world.world_index,
        )
        expected_world_id = _world_id(
            design_sha256=design_sha256,
            anchor_id=world.anchor_id,
            cell_id=world.cell_id,
            world_index=world.world_index,
        )
        if world.seed != expected_seed or world.world_id != expected_world_id:
            raise ValueError(
                f"World identity derivation mismatch for {world.world_id}."
            )
        if (
            world.cell_id == source_size_cell
            and world.subjects != source_subjects[world.anchor_id]
        ):
            raise ValueError(
                f"Source-size world {world.world_id} has {world.subjects} subjects; "
                f"expected {source_subjects[world.anchor_id]}."
            )
    arguments = [(release_dir, world) for world in receipt.worlds]
    if workers == 1:
        world_estimates = [_fit_release_world(*argument) for argument in arguments]
    else:
        with single_threaded_numerical_process_pool(
            workers=min(workers, len(arguments))
        ) as executor:
            futures = [
                executor.submit(_fit_release_world, *argument) for argument in arguments
            ]
            world_estimates = [future.result() for future in futures]
    estimates = [estimate for world_rows in world_estimates for estimate in world_rows]
    summaries = _summarize_cells(estimates, cell_by_id=cell_by_id)
    anchor_summaries = _summarize_cells(
        estimates,
        cell_by_id=cell_by_id,
        group_by_anchor=True,
    )
    curves = _summarize_response_curves(anchor_summaries)
    return ProductionCoreRecoveryReportV1(
        design_sha256=design_sha256,
        receipt_sha256=sha256_file(receipt_path),
        candidate_sha256=candidate_sha256,
        request_sha256=request_sha256,
        estimates=tuple(estimates),
        cells=tuple(summaries),
        anchor_cells=tuple(anchor_summaries),
        curves=tuple(curves),
    )


def _fit_release_world(
    release_dir: Path,
    world: _WorldReceiptV1,
) -> tuple[ProductionCoreWorldEstimateV1, ...]:
    """Verify and fit one independent released world."""

    path = _resolve_release_path(release_dir, world.analysis_path)
    if sha256_file(path) != world.analysis_sha256:
        raise ValueError(f"Analysis checksum mismatch for {world.world_id}.")
    frame = _validate_analysis_frame(pd.read_parquet(path), world=world)
    return tuple(_fit_world(frame, world=world))


def _verify_run_binding(
    *,
    release_dir: Path,
    design_sha256: str,
    receipt_path: Path,
) -> tuple[str, str]:
    request_path = release_dir / "qualification_request.json"
    run_path = release_dir / "qualification_run.json"
    if not request_path.is_file() or not run_path.is_file():
        raise ValueError("Qualification request and run binding are required.")
    request = ProductionCorePublicRequestV1.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    run = ProductionCorePublicRunV1.model_validate_json(
        run_path.read_text(encoding="utf-8")
    )
    if request.design_sha256 != design_sha256 or run.design_sha256 != design_sha256:
        raise ValueError(
            "Qualification run binding does not match the released design."
        )
    if run.request_sha256 != request.checksum:
        raise ValueError(
            "Qualification run binding does not match the released request."
        )
    if run.receipt_sha256 != sha256_file(receipt_path):
        raise ValueError(
            "Qualification run binding does not match the released receipt."
        )
    return run.candidate_sha256, run.request_sha256


def _cell_key(row: ProductionCoreCellSummaryV1) -> tuple[str, str, str, str]:
    return (row.anchor_id or "", row.cell_id, row.estimand, row.route)


def _difference(right: float | None, left: float | None) -> float | None:
    if right is None or left is None:
        if right is left:
            return None
        raise ValueError("Candidate comparison requires matched estimability.")
    return right - left


def _fit_world(
    frame: pd.DataFrame, *, world: _WorldReceiptV1
) -> list[ProductionCoreWorldEstimateV1]:
    if world.events == 0:
        return [
            *_failed_adjusted_estimates(world, failure="singular_information"),
            _failed_estimate(
                world,
                route="binary_endpoint_shortcut",
                failure="singular_information",
            ),
        ]
    estimates = []
    standardized = frame.copy()
    visit_sd = float(standardized["empirical_visit_count"].std(ddof=0))
    if not np.isfinite(visit_sd) or visit_sd <= 0:
        return [
            *_failed_adjusted_estimates(world, failure="invalid_world"),
            _failed_estimate(
                world,
                route="binary_endpoint_shortcut",
                failure="invalid_world",
            ),
        ]
    standardized["visit_count_z"] = (
        standardized["empirical_visit_count"]
        - standardized["empirical_visit_count"].mean()
    ) / visit_sd
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", RuntimeWarning)
            fit = PHReg(
                endog=standardized["time"].to_numpy(dtype=float),
                exog=standardized.loc[:, ["treatment", "visit_count_z"]].to_numpy(
                    dtype=float
                ),
                status=standardized["event"].to_numpy(dtype=int),
                ties="breslow",
            ).fit()
            parameters = fit.params
            standard_errors = fit.bse
        estimates.append(
            _completed_estimate(
                world,
                estimand="treatment_log_hazard_ratio",
                route="adjusted_cox",
                estimate=float(parameters[0]),
                standard_error=float(standard_errors[0]),
            )
        )
        estimates.append(
            _completed_estimate(
                world,
                estimand="empirical_visit_count_log_hazard_ratio",
                route="adjusted_cox",
                estimate=float(parameters[1]),
                standard_error=float(standard_errors[1]),
            )
        )
    except ConvergenceWarning:
        estimates.extend(_failed_adjusted_estimates(world, failure="convergence"))
    except RuntimeWarning:
        estimates.extend(
            _failed_adjusted_estimates(world, failure="singular_information")
        )
    except np.linalg.LinAlgError:
        estimates.extend(
            _failed_adjusted_estimates(world, failure="singular_information")
        )

    try:
        predictors = sm.add_constant(
            standardized.loc[:, ["treatment"]], has_constant="add"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", PerfectSeparationWarning)
            fit = sm.GLM(
                standardized["event"], predictors, family=sm.families.Binomial()
            ).fit(cov_type="HC3")
        if not bool(fit.converged):
            raise ConvergenceWarning("Endpoint-only fit did not converge.")
        estimates.append(
            _completed_estimate(
                world,
                estimand="treatment_log_hazard_ratio",
                route="binary_endpoint_shortcut",
                estimate=float(fit.params["treatment"]),
                standard_error=float(fit.bse["treatment"]),
            )
        )
    except PerfectSeparationError:
        estimates.append(
            _failed_estimate(
                world, route="binary_endpoint_shortcut", failure="convergence"
            )
        )
    except PerfectSeparationWarning:
        estimates.append(
            _failed_estimate(
                world, route="binary_endpoint_shortcut", failure="convergence"
            )
        )
    except ConvergenceWarning:
        estimates.append(
            _failed_estimate(
                world, route="binary_endpoint_shortcut", failure="convergence"
            )
        )
    except np.linalg.LinAlgError:
        estimates.append(
            _failed_estimate(
                world, route="binary_endpoint_shortcut", failure="singular_information"
            )
        )
    return estimates


def _completed_estimate(
    world: _WorldReceiptV1,
    *,
    estimand: Literal[
        "treatment_log_hazard_ratio", "empirical_visit_count_log_hazard_ratio"
    ],
    route: Literal["adjusted_cox", "binary_endpoint_shortcut"],
    estimate: float,
    standard_error: float,
) -> ProductionCoreWorldEstimateV1:
    if (
        not np.isfinite(estimate)
        or not np.isfinite(standard_error)
        or standard_error <= 0
    ):
        return _failed_estimate(
            world,
            estimand=estimand,
            route=route,
            failure="singular_information",
        )
    truth = (
        world.truth_log_hazard_ratio
        if estimand == "treatment_log_hazard_ratio"
        else world.truth_empirical_visit_count_log_hazard_ratio
    )
    lower = estimate - float(norm.ppf(0.975)) * standard_error
    upper = estimate + float(norm.ppf(0.975)) * standard_error
    return ProductionCoreWorldEstimateV1(
        world_id=world.world_id,
        anchor_id=world.anchor_id,
        cell_id=world.cell_id,
        world_index=world.world_index,
        estimand=estimand,
        route=route,
        truth_log_hazard_ratio=truth,
        subjects=world.subjects,
        events=world.events,
        estimate=estimate,
        standard_error=standard_error,
        covered=lower <= truth <= upper,
        rejected_null=not (lower <= 0.0 <= upper),
    )


def _failed_estimate(
    world: _WorldReceiptV1,
    *,
    estimand: Literal[
        "treatment_log_hazard_ratio", "empirical_visit_count_log_hazard_ratio"
    ] = "treatment_log_hazard_ratio",
    route: Literal["adjusted_cox", "binary_endpoint_shortcut"],
    failure: Literal["convergence", "singular_information", "invalid_world"],
) -> ProductionCoreWorldEstimateV1:
    return ProductionCoreWorldEstimateV1(
        world_id=world.world_id,
        anchor_id=world.anchor_id,
        cell_id=world.cell_id,
        world_index=world.world_index,
        estimand=estimand,
        route=route,
        truth_log_hazard_ratio=(
            world.truth_log_hazard_ratio
            if estimand == "treatment_log_hazard_ratio"
            else world.truth_empirical_visit_count_log_hazard_ratio
        ),
        subjects=world.subjects,
        events=world.events,
        failure=failure,
    )


def _failed_adjusted_estimates(
    world: _WorldReceiptV1,
    *,
    failure: Literal["convergence", "singular_information", "invalid_world"],
) -> list[ProductionCoreWorldEstimateV1]:
    return [
        _failed_estimate(
            world,
            estimand=estimand,
            route="adjusted_cox",
            failure=failure,
        )
        for estimand in (
            cast(
                Literal[
                    "treatment_log_hazard_ratio",
                    "empirical_visit_count_log_hazard_ratio",
                ],
                "treatment_log_hazard_ratio",
            ),
            cast(
                Literal[
                    "treatment_log_hazard_ratio",
                    "empirical_visit_count_log_hazard_ratio",
                ],
                "empirical_visit_count_log_hazard_ratio",
            ),
        )
    ]


def _summarize_cells(
    estimates: list[ProductionCoreWorldEstimateV1],
    *,
    cell_by_id: dict[str, _PublicCellV1],
    group_by_anchor: bool = False,
) -> list[ProductionCoreCellSummaryV1]:
    grouped: dict[
        tuple[str | None, str, str, str], list[ProductionCoreWorldEstimateV1]
    ] = defaultdict(list)
    for estimate in estimates:
        anchor_id = estimate.anchor_id if group_by_anchor else None
        grouped[
            (anchor_id, estimate.cell_id, estimate.estimand, estimate.route)
        ].append(estimate)
    summaries = []
    for (anchor_id, cell_id, estimand, route), rows in sorted(
        grouped.items(),
        key=lambda item: tuple("" if value is None else value for value in item[0]),
    ):
        successful = [row for row in rows if row.failure is None]
        anchor_count = len({row.anchor_id for row in rows})
        base = {
            "anchor_id": anchor_id,
            "anchors": anchor_count,
            "uncertainty_unit": (
                "anchor" if not group_by_anchor and anchor_count >= 2 else "world"
            ),
            "cell_id": cell_id,
            "response_axis": cell_by_id[cell_id].response_axis,
            "level": cell_by_id[cell_id].level,
            "estimand": estimand,
            "route": route,
            "truth_log_hazard_ratio": rows[0].truth_log_hazard_ratio,
            "worlds": len(rows),
            "successful_worlds": len(successful),
            "failures": len(rows) - len(successful),
            "mean_subjects": float(np.mean([row.subjects for row in rows])),
            "mean_events": float(np.mean([row.events for row in rows])),
            "mean_event_fraction": float(
                np.mean([row.events / row.subjects for row in rows])
            ),
        }
        scheduled_coverage = _scheduled_proportion_summary(
            rows,
            attribute="covered",
            group_by_anchor=group_by_anchor,
        )
        scheduled_rejection = _scheduled_proportion_summary(
            rows,
            attribute="rejected_null",
            group_by_anchor=group_by_anchor,
        )
        base.update(
            {
                "coverage_scheduled": scheduled_coverage[0],
                "coverage_scheduled_ci_low": scheduled_coverage[1],
                "coverage_scheduled_ci_high": scheduled_coverage[2],
                "rejection_rate_scheduled": scheduled_rejection[0],
                "rejection_rate_scheduled_ci_low": scheduled_rejection[1],
                "rejection_rate_scheduled_ci_high": scheduled_rejection[2],
            }
        )
        if len(successful) < 2:
            summaries.append(ProductionCoreCellSummaryV1(**base))
            continue
        values = np.asarray(
            [cast(float, row.estimate) for row in successful], dtype=np.float64
        )
        standard_errors = np.asarray(
            [cast(float, row.standard_error) for row in successful],
            dtype=np.float64,
        )
        truth = np.asarray(
            [row.truth_log_hazard_ratio for row in successful], dtype=float
        )
        errors = values - truth
        coverage = np.asarray([bool(row.covered) for row in successful], dtype=float)
        rejected = np.asarray(
            [bool(row.rejected_null) for row in successful], dtype=float
        )
        empirical_sd_value: float | None
        se_ratio_interval: tuple[float | None, float | None]
        if group_by_anchor or anchor_count < 2:
            empirical_sd_value = float(values.std(ddof=1))
            bias_mcse = float(errors.std(ddof=1) / np.sqrt(len(errors)))
            bias_half_width = float(
                student_t.ppf(0.975, df=len(errors) - 1) * bias_mcse
            )
            coverage_interval = proportion_interval(int(coverage.sum()), len(coverage))
            rejection_interval = proportion_interval(int(rejected.sum()), len(rejected))
            mean_estimate = float(values.mean())
            bias = float(errors.mean())
            bias_interval = (bias - bias_half_width, bias + bias_half_width)
            mse = np.square(errors)
            mse_interval = _positive_mean_interval(mse)
            rmse = float(np.sqrt(mse.mean()))
            rmse_interval = tuple(float(np.sqrt(bound)) for bound in mse_interval)
            mean_model_standard_error = float(standard_errors.mean())
            model_to_empirical_se_ratio = (
                float(mean_model_standard_error / empirical_sd_value)
                if empirical_sd_value > 0
                else None
            )
            se_ratio_interval = (None, None)
            coverage_mean = float(coverage.mean())
            rejection_mean = float(rejected.mean())
        else:
            successful_by_anchor: dict[str, list[int]] = defaultdict(list)
            for index, row in enumerate(successful):
                successful_by_anchor[row.anchor_id].append(index)
            if len(successful_by_anchor) < 2:
                summaries.append(ProductionCoreCellSummaryV1(**base))
                continue
            anchor_errors = np.asarray(
                [errors[indexes].mean() for indexes in successful_by_anchor.values()],
                dtype=float,
            )
            anchor_estimates = np.asarray(
                [values[indexes].mean() for indexes in successful_by_anchor.values()],
                dtype=float,
            )
            anchor_mse = np.asarray(
                [
                    np.square(errors[indexes]).mean()
                    for indexes in successful_by_anchor.values()
                ],
                dtype=float,
            )
            anchor_empirical_sd = np.asarray(
                [
                    values[indexes].std(ddof=1)
                    for indexes in successful_by_anchor.values()
                    if len(indexes) >= 2
                ],
                dtype=float,
            )
            anchor_model_se = np.asarray(
                [
                    standard_errors[indexes].mean()
                    for indexes in successful_by_anchor.values()
                ],
                dtype=float,
            )
            anchor_se_ratio = np.asarray(
                [
                    standard_errors[indexes].mean() / values[indexes].std(ddof=1)
                    for indexes in successful_by_anchor.values()
                    if len(indexes) >= 2 and values[indexes].std(ddof=1) > 0
                ],
                dtype=float,
            )
            anchor_coverage = np.asarray(
                [coverage[indexes].mean() for indexes in successful_by_anchor.values()],
                dtype=float,
            )
            anchor_rejection = np.asarray(
                [rejected[indexes].mean() for indexes in successful_by_anchor.values()],
                dtype=float,
            )
            bias_mcse = float(anchor_errors.std(ddof=1) / np.sqrt(len(anchor_errors)))
            _, bias_low, bias_high = _unbounded_mean_interval(anchor_errors)
            bias_interval = (bias_low, bias_high)
            coverage_interval = _mean_interval(anchor_coverage)
            rejection_interval = _mean_interval(anchor_rejection)
            mean_estimate = float(anchor_estimates.mean())
            bias = float(anchor_errors.mean())
            mse_interval = _positive_mean_interval(anchor_mse)
            rmse = float(np.sqrt(anchor_mse.mean()))
            rmse_interval = tuple(float(np.sqrt(bound)) for bound in mse_interval)
            empirical_sd_value = (
                float(anchor_empirical_sd.mean()) if len(anchor_empirical_sd) else None
            )
            mean_model_standard_error = float(anchor_model_se.mean())
            model_to_empirical_se_ratio = (
                float(anchor_se_ratio.mean()) if len(anchor_se_ratio) else None
            )
            if len(anchor_se_ratio) >= 2:
                _, ratio_low, ratio_high = _unbounded_mean_interval(anchor_se_ratio)
                se_ratio_interval = (max(0.0, ratio_low), ratio_high)
            else:
                se_ratio_interval = (None, None)
            coverage_mean = float(anchor_coverage.mean())
            rejection_mean = float(anchor_rejection.mean())
        summaries.append(
            ProductionCoreCellSummaryV1(
                **base,
                mean_estimate=mean_estimate,
                bias=bias,
                bias_mcse=bias_mcse,
                bias_ci_low=bias_interval[0],
                bias_ci_high=bias_interval[1],
                rmse=rmse,
                rmse_ci_low=rmse_interval[0],
                rmse_ci_high=rmse_interval[1],
                empirical_standard_deviation=empirical_sd_value,
                mean_model_standard_error=mean_model_standard_error,
                model_to_empirical_se_ratio=model_to_empirical_se_ratio,
                model_to_empirical_se_ratio_ci_low=se_ratio_interval[0],
                model_to_empirical_se_ratio_ci_high=se_ratio_interval[1],
                coverage=coverage_mean,
                coverage_ci_low=coverage_interval[0],
                coverage_ci_high=coverage_interval[1],
                rejection_rate=rejection_mean,
                rejection_rate_ci_low=rejection_interval[0],
                rejection_rate_ci_high=rejection_interval[1],
            )
        )
    return summaries


def _scheduled_proportion_summary(
    rows: list[ProductionCoreWorldEstimateV1],
    *,
    attribute: Literal["covered", "rejected_null"],
    group_by_anchor: bool,
) -> tuple[float, float, float]:
    """Summarize a conservative operating characteristic over scheduled worlds."""

    by_anchor: dict[str, list[ProductionCoreWorldEstimateV1]] = defaultdict(list)
    for row in rows:
        by_anchor[row.anchor_id].append(row)
    if group_by_anchor or len(by_anchor) < 2:
        successes = sum(getattr(row, attribute) is True for row in rows)
        low, high = proportion_interval(successes, len(rows))
        return successes / len(rows), low, high
    values = np.asarray(
        [
            sum(getattr(row, attribute) is True for row in anchor_rows)
            / len(anchor_rows)
            for anchor_rows in by_anchor.values()
        ],
        dtype=float,
    )
    low, high = _mean_interval(values)
    return float(values.mean()), low, high


def _summarize_response_curves(
    anchor_cells: list[ProductionCoreCellSummaryV1],
) -> list[ProductionCoreResponseCurveV1]:
    anchors = {row.anchor_id for row in anchor_cells}
    axes = {row.response_axis for row in anchor_cells}
    if len(anchors) < 2 or not {
        "reference",
        "information",
        "effect",
        "mechanism",
    }.issubset(axes):
        return []
    return [
        _summarize_response_curve(
            anchor_cells,
            curve="treatment_effect_recovery",
            response_axis="effect",
            estimand="treatment_log_hazard_ratio",
            route="adjusted_cox",
            response="mean_estimate",
            input_scale="truth_log_hazard_ratio",
            expected_slope=1.0,
        ),
        _summarize_response_curve(
            anchor_cells,
            curve="empirical_process_recovery",
            response_axis="mechanism",
            estimand="empirical_visit_count_log_hazard_ratio",
            route="adjusted_cox",
            response="mean_estimate",
            input_scale="truth_log_hazard_ratio",
            expected_slope=1.0,
        ),
        _summarize_response_curve(
            anchor_cells,
            curve="information_precision",
            response_axis="information",
            estimand="treatment_log_hazard_ratio",
            route="adjusted_cox",
            response="empirical_standard_deviation",
            input_scale="log_mean_subjects",
            expected_slope=-0.5,
        ),
        _summarize_response_curve(
            anchor_cells,
            curve="time_discarding_route",
            response_axis="effect",
            estimand="treatment_log_hazard_ratio",
            route="binary_endpoint_shortcut",
            response="mean_estimate",
            input_scale="truth_log_hazard_ratio",
            expected_slope=None,
        ),
    ]


def _summarize_response_curve(
    anchor_cells: list[ProductionCoreCellSummaryV1],
    *,
    curve: Literal[
        "treatment_effect_recovery",
        "empirical_process_recovery",
        "information_precision",
        "time_discarding_route",
    ],
    response_axis: Literal["information", "effect", "mechanism"],
    estimand: Literal[
        "treatment_log_hazard_ratio", "empirical_visit_count_log_hazard_ratio"
    ],
    route: Literal["adjusted_cox", "binary_endpoint_shortcut"],
    response: Literal["mean_estimate", "empirical_standard_deviation"],
    input_scale: Literal["truth_log_hazard_ratio", "log_mean_subjects"],
    expected_slope: float | None,
) -> ProductionCoreResponseCurveV1:
    selected = [
        row
        for row in anchor_cells
        if row.estimand == estimand
        and row.route == route
        and row.response_axis in ("reference", response_axis)
    ]
    anchor_ids = sorted({cast(str, row.anchor_id) for row in selected})
    if len(anchor_ids) < 2:
        raise ValueError(f"Response curve {curve!r} requires at least two anchors.")
    point_counts = {
        anchor_id: sum(row.anchor_id == anchor_id for row in selected)
        for anchor_id in anchor_ids
    }
    if len(set(point_counts.values())) != 1:
        raise ValueError(f"Response curve {curve!r} has incomplete anchor cell sets.")
    points_per_anchor = next(iter(point_counts.values()))
    if points_per_anchor < 3:
        raise ValueError(
            f"Response curve {curve!r} requires at least three points per anchor."
        )

    slopes = []
    intercepts = []
    for anchor_id in anchor_ids:
        rows = [row for row in selected if row.anchor_id == anchor_id]
        x = np.asarray(
            [
                (
                    row.truth_log_hazard_ratio
                    if input_scale == "truth_log_hazard_ratio"
                    else np.log(row.mean_subjects)
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        raw_y = [getattr(row, response) for row in rows]
        if any(value is None for value in raw_y):
            raise ValueError(
                f"Response curve {curve!r} contains an unavailable result for {anchor_id}."
            )
        y = np.asarray([cast(float, value) for value in raw_y], dtype=np.float64)
        if response == "empirical_standard_deviation":
            if np.any(y <= 0):
                raise ValueError(
                    f"Response curve {curve!r} requires positive precision estimates."
                )
            y = np.log(y)
        if len(np.unique(x)) != len(x):
            raise ValueError(
                f"Response curve {curve!r} contains repeated input levels for {anchor_id}."
            )
        slope, intercept = np.polyfit(x, y, deg=1)
        slopes.append(float(slope))
        intercepts.append(float(intercept))

    slope_mean, slope_low, slope_high = _unbounded_mean_interval(
        np.asarray(slopes, dtype=np.float64)
    )
    intercept_mean, intercept_low, intercept_high = _unbounded_mean_interval(
        np.asarray(intercepts, dtype=np.float64)
    )
    return ProductionCoreResponseCurveV1(
        curve=curve,
        estimand=estimand,
        route=route,
        response=response,
        input_scale=input_scale,
        anchors=len(anchor_ids),
        points_per_anchor=points_per_anchor,
        slope_mean=slope_mean,
        slope_ci_low=slope_low,
        slope_ci_high=slope_high,
        intercept_mean=intercept_mean,
        intercept_ci_low=intercept_low,
        intercept_ci_high=intercept_high,
        expected_slope=expected_slope,
    )


def _validate_analysis_frame(
    frame: pd.DataFrame, *, world: _WorldReceiptV1
) -> pd.DataFrame:
    required = {"participant_id", "treatment", "time", "event", "empirical_visit_count"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(
            f"Analysis world {world.world_id} is missing columns: {missing!r}."
        )
    if len(frame) != world.subjects or frame["participant_id"].duplicated().any():
        raise ValueError(
            f"Analysis world {world.world_id} has invalid participant identity or count."
        )
    values = frame.loc[:, sorted(required)].copy()
    for column in ("treatment", "time", "event", "empirical_visit_count"):
        values[column] = pd.to_numeric(values[column], errors="raise")
    if not set(values["treatment"].unique()).issubset({0, 1}) or set(
        values["treatment"].unique()
    ) != {0, 1}:
        raise ValueError(
            f"Analysis world {world.world_id} must contain both binary treatment arms."
        )
    if not set(values["event"].unique()).issubset({0, 1}):
        raise ValueError(
            f"Analysis world {world.world_id} event indicator must be binary."
        )
    if int(values["event"].sum()) != world.events or (values["time"] <= 0).any():
        raise ValueError(
            f"Analysis world {world.world_id} has inconsistent event or time data."
        )
    return values


def _resolve_release_path(root: Path, relative_path: str) -> Path:
    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError("Release analysis paths must be relative.")
    resolved = (root.resolve() / requested).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Release analysis path escapes the release directory.")
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Release analysis file does not exist: {relative_path}."
        )
    return resolved


def _mean_interval(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    if len(values) < 2:
        return float(values[0]), float(values[0])
    half_width = float(
        student_t.ppf(0.975, df=len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    mean = float(values.mean())
    return max(0.0, mean - half_width), min(1.0, mean + half_width)


def _positive_mean_interval(
    values: npt.NDArray[np.float64],
) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0])
        return value, value
    half_width = float(
        student_t.ppf(0.975, df=len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    mean = float(values.mean())
    return max(0.0, mean - half_width), mean + half_width


def _unbounded_mean_interval(
    values: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    if len(values) < 2:
        raise ValueError("A mean interval requires at least two independent values.")
    mean = float(values.mean())
    half_width = float(
        student_t.ppf(0.975, df=len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return mean, mean - half_width, mean + half_width


def _sha256_json(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _world_seed(
    *,
    design_seed: int,
    anchor_id: str,
    cell_id: str,
    world_index: int,
) -> int:
    payload = f"{design_seed}:{anchor_id}:{cell_id}:{world_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big")


def _world_id(
    *,
    design_sha256: str,
    anchor_id: str,
    cell_id: str,
    world_index: int,
) -> str:
    payload = f"{design_sha256}:{anchor_id}:{cell_id}:{world_index}".encode()
    return f"world_{hashlib.sha256(payload).hexdigest()[:20]}"
