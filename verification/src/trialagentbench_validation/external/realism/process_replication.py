"""Independent marginal replication checks for empirical clinical processes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import t as student_t
from scipy.stats import wasserstein_distance

from trialagentbench_validation.external.recovery.production import (
    ProductionCorePublicDesignV1,
    ProductionCorePublicReceiptV1,
    _verify_run_binding,
)
from trialagentbench_validation.io import sha256_file


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VisitCountBinV1(_FrozenModel):
    """Aggregate participant count at one observed visit count."""

    visits: int = Field(ge=0)
    participants: int = Field(ge=1)


class SourceVisitCountFingerprintV1(_FrozenModel):
    """Non-disclosing visit-count distribution for one empirical anchor."""

    schema_id: Literal["trialagentbench.source_visit_count_fingerprint/v1"] = (
        "trialagentbench.source_visit_count_fingerprint/v1"
    )
    anchor_id: str = Field(pattern=r"^anchor_[0-9a-f]{16}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participants: int = Field(ge=20)
    bins: tuple[VisitCountBinV1, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _complete_distribution(self) -> SourceVisitCountFingerprintV1:
        visits = [row.visits for row in self.bins]
        if visits != sorted(visits) or len(visits) != len(set(visits)):
            raise ValueError("Visit-count fingerprint bins must be unique and ordered.")
        if sum(row.participants for row in self.bins) != self.participants:
            raise ValueError("Visit-count fingerprint bins do not sum to participants.")
        return self


class VisitCountAnchorCellV1(_FrozenModel):
    """Mean replication discrepancy over worlds for one source anchor and cell."""

    anchor_id: str = Field(pattern=r"^anchor_[0-9a-f]{16}$")
    cell_id: str = Field(min_length=1)
    worlds: int = Field(ge=2)
    source_participants: int = Field(ge=20)
    synthetic_participants_mean: float = Field(ge=20, allow_inf_nan=False)
    standardized_wasserstein_mean: float = Field(ge=0, allow_inf_nan=False)
    standardized_mean_error_mean: float = Field(ge=0, allow_inf_nan=False)
    absolute_log_sd_ratio_mean: float = Field(ge=0, allow_inf_nan=False)


class VisitCountCellSummaryV1(_FrozenModel):
    """Equal-anchor replication summary for one production response cell."""

    cell_id: str = Field(min_length=1)
    anchors: int = Field(ge=1)
    worlds: int = Field(ge=2)
    standardized_wasserstein_mean: float = Field(ge=0, allow_inf_nan=False)
    standardized_wasserstein_ci_low: float = Field(ge=0, allow_inf_nan=False)
    standardized_wasserstein_ci_high: float = Field(ge=0, allow_inf_nan=False)
    standardized_mean_error_mean: float = Field(ge=0, allow_inf_nan=False)
    standardized_mean_error_ci_low: float = Field(ge=0, allow_inf_nan=False)
    standardized_mean_error_ci_high: float = Field(ge=0, allow_inf_nan=False)
    absolute_log_sd_ratio_mean: float = Field(ge=0, allow_inf_nan=False)
    absolute_log_sd_ratio_ci_low: float = Field(ge=0, allow_inf_nan=False)
    absolute_log_sd_ratio_ci_high: float = Field(ge=0, allow_inf_nan=False)


class VisitCountNegativeControlV1(_FrozenModel):
    """Wrong-source sensitivity check for the visit-count discrepancy."""

    mismatched_anchor_pairs: int = Field(ge=2)
    mismatched_standardized_wasserstein_minimum: float = Field(
        ge=0, allow_inf_nan=False
    )
    mismatched_standardized_wasserstein_median: float = Field(ge=0, allow_inf_nan=False)
    mismatched_standardized_wasserstein_maximum: float = Field(
        ge=0, allow_inf_nan=False
    )
    within_anchor_reference_mean: float = Field(ge=0, allow_inf_nan=False)
    within_anchor_reference_ci_high: float = Field(ge=0, allow_inf_nan=False)
    median_separation_ratio: float = Field(gt=0, allow_inf_nan=False)
    mismatched_pairs_at_or_below_within_reference_ci_high: int = Field(ge=0)


class VisitCountReplicationReportV1(_FrozenModel):
    """Exact-candidate empirical visit-count replication evidence."""

    schema_id: Literal["trialagentbench.visit_count_replication_report/v1"] = (
        "trialagentbench.visit_count_replication_report/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_fingerprints_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    anchor_cells: tuple[VisitCountAnchorCellV1, ...] = Field(min_length=1)
    cells: tuple[VisitCountCellSummaryV1, ...] = Field(min_length=1)
    negative_control: VisitCountNegativeControlV1 | None = None


def evaluate_visit_count_replication(
    *,
    release_dir: Path,
    source_fingerprints: Path,
) -> VisitCountReplicationReportV1:
    """Compare every released world's visit-count margin with its source anchor."""

    design_path = release_dir / "qualification_design.json"
    receipt_path = release_dir / "qualification_receipt.json"
    design = ProductionCorePublicDesignV1.model_validate_json(
        design_path.read_text(encoding="utf-8")
    )
    receipt = ProductionCorePublicReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    design_sha256 = _sha256_json(design.model_dump(mode="json"))
    if receipt.design_sha256 != design_sha256:
        raise ValueError("Production receipt and design identities differ.")
    candidate_sha256, request_sha256 = _verify_run_binding(
        release_dir=release_dir,
        design_sha256=design_sha256,
        receipt_path=receipt_path,
    )
    fingerprints = _read_fingerprints(source_fingerprints)
    fingerprint_by_anchor = {row.anchor_id: row for row in fingerprints}
    expected_anchors = {row.anchor_id: row.manifest_sha256 for row in design.anchors}
    observed_anchors = {row.anchor_id: row.manifest_sha256 for row in fingerprints}
    if observed_anchors != expected_anchors:
        raise ValueError(
            "Source visit-count fingerprints do not match the production anchors."
        )
    declared_subjects = {row.anchor_id: row.source_subjects for row in design.anchors}
    fingerprint_subjects = {row.anchor_id: row.participants for row in fingerprints}
    if fingerprint_subjects != declared_subjects:
        raise ValueError(
            "Source visit-count fingerprints do not match declared anchor sizes."
        )

    reference_cells = [
        cell.cell_id for cell in design.cells if cell.response_axis == "reference"
    ]
    if len(reference_cells) != 1:
        raise ValueError("Visit-count replication requires exactly one reference cell.")
    reference_cell_id = reference_cells[0]
    grouped: dict[tuple[str, str], list[tuple[int, float, float, float]]] = defaultdict(
        list
    )
    reference_world_values: dict[str, list[npt.NDArray[np.float64]]] = defaultdict(list)
    for world in receipt.worlds:
        fingerprint = fingerprint_by_anchor[world.anchor_id]
        path = _resolve_release_path(release_dir, world.analysis_path)
        if sha256_file(path) != world.analysis_sha256:
            raise ValueError(f"Analysis checksum mismatch for {world.world_id}.")
        frame = pd.read_parquet(path, columns=["empirical_visit_count"])
        synthetic_values = pd.to_numeric(
            frame["empirical_visit_count"], errors="raise"
        ).to_numpy(dtype=float)
        if (
            len(synthetic_values) != world.subjects
            or not np.isfinite(synthetic_values).all()
            or (synthetic_values < 0).any()
        ):
            raise ValueError(f"Invalid visit counts in {world.world_id}.")
        source_values = np.asarray(
            [row.visits for row in fingerprint.bins], dtype=float
        )
        source_weights = np.asarray(
            [row.participants for row in fingerprint.bins], dtype=float
        )
        source_mean = float(np.average(source_values, weights=source_weights))
        source_variance = float(
            np.average(np.square(source_values - source_mean), weights=source_weights)
        )
        source_sd = float(np.sqrt(source_variance))
        synthetic_sd = float(synthetic_values.std(ddof=0))
        if source_sd <= 0 or synthetic_sd <= 0:
            raise ValueError(
                f"Visit counts must vary in source and synthetic world {world.world_id}."
            )
        grouped[(world.anchor_id, world.cell_id)].append(
            (
                len(synthetic_values),
                float(
                    wasserstein_distance(
                        source_values,
                        synthetic_values,
                        u_weights=source_weights,
                    )
                    / source_sd
                ),
                float(abs(synthetic_values.mean() - source_mean) / source_sd),
                float(abs(np.log(synthetic_sd / source_sd))),
            )
        )
        if world.cell_id == reference_cell_id:
            reference_world_values[world.anchor_id].append(synthetic_values)
    expected_groups = {
        (anchor.anchor_id, cell.cell_id)
        for anchor in design.anchors
        for cell in design.cells
    }
    if set(grouped) != expected_groups:
        raise ValueError("Visit-count replication does not cover the complete design.")

    anchor_cells: list[VisitCountAnchorCellV1] = []
    for (anchor_id, cell_id), world_rows in sorted(grouped.items()):
        if len(world_rows) != design.worlds_per_anchor_cell:
            raise ValueError(
                f"Incomplete visit-count worlds for {anchor_id}:{cell_id}."
            )
        world_values = np.asarray(world_rows, dtype=float)
        anchor_cells.append(
            VisitCountAnchorCellV1(
                anchor_id=anchor_id,
                cell_id=cell_id,
                worlds=len(world_rows),
                source_participants=fingerprint_by_anchor[anchor_id].participants,
                synthetic_participants_mean=float(world_values[:, 0].mean()),
                standardized_wasserstein_mean=float(world_values[:, 1].mean()),
                standardized_mean_error_mean=float(world_values[:, 2].mean()),
                absolute_log_sd_ratio_mean=float(world_values[:, 3].mean()),
            )
        )

    cells = []
    for cell in design.cells:
        cell_rows = [row for row in anchor_cells if row.cell_id == cell.cell_id]
        if len(cell_rows) != len(design.anchors):
            raise ValueError(
                f"Incomplete anchor set for visit-count cell {cell.cell_id}."
            )
        wasserstein = np.asarray(
            [row.standardized_wasserstein_mean for row in cell_rows]
        )
        mean_error = np.asarray([row.standardized_mean_error_mean for row in cell_rows])
        sd_error = np.asarray([row.absolute_log_sd_ratio_mean for row in cell_rows])
        cells.append(
            VisitCountCellSummaryV1(
                cell_id=cell.cell_id,
                anchors=len(cell_rows),
                worlds=sum(row.worlds for row in cell_rows),
                **_metric_interval("standardized_wasserstein", wasserstein),
                **_metric_interval("standardized_mean_error", mean_error),
                **_metric_interval("absolute_log_sd_ratio", sd_error),
            )
        )
    negative_control = _wrong_source_negative_control(
        fingerprints=fingerprints,
        reference_world_values=reference_world_values,
        reference_summary=next(
            row for row in cells if row.cell_id == reference_cell_id
        ),
    )
    return VisitCountReplicationReportV1(
        design_sha256=design_sha256,
        receipt_sha256=sha256_file(receipt_path),
        candidate_sha256=candidate_sha256,
        request_sha256=request_sha256,
        source_fingerprints_sha256=sha256_file(source_fingerprints),
        anchor_cells=tuple(anchor_cells),
        cells=tuple(cells),
        negative_control=negative_control,
    )


def _wrong_source_negative_control(
    *,
    fingerprints: tuple[SourceVisitCountFingerprintV1, ...],
    reference_world_values: dict[str, list[npt.NDArray[np.float64]]],
    reference_summary: VisitCountCellSummaryV1,
) -> VisitCountNegativeControlV1 | None:
    """Compare reference worlds with every nonmatching source fingerprint."""

    if len(fingerprints) < 2:
        return None
    pair_discrepancies = []
    for fingerprint in fingerprints:
        source_values = np.asarray(
            [row.visits for row in fingerprint.bins], dtype=float
        )
        source_weights = np.asarray(
            [row.participants for row in fingerprint.bins], dtype=float
        )
        source_mean = float(np.average(source_values, weights=source_weights))
        source_sd = float(
            np.sqrt(
                np.average(
                    np.square(source_values - source_mean),
                    weights=source_weights,
                )
            )
        )
        for synthetic_anchor, worlds in reference_world_values.items():
            if synthetic_anchor == fingerprint.anchor_id:
                continue
            pair_discrepancies.append(
                float(
                    np.mean(
                        [
                            wasserstein_distance(
                                source_values,
                                world,
                                u_weights=source_weights,
                            )
                            / source_sd
                            for world in worlds
                        ]
                    )
                )
            )
    mismatched = np.asarray(pair_discrepancies, dtype=float)
    median = float(np.median(mismatched))
    return VisitCountNegativeControlV1(
        mismatched_anchor_pairs=len(mismatched),
        mismatched_standardized_wasserstein_minimum=float(mismatched.min()),
        mismatched_standardized_wasserstein_median=median,
        mismatched_standardized_wasserstein_maximum=float(mismatched.max()),
        within_anchor_reference_mean=reference_summary.standardized_wasserstein_mean,
        within_anchor_reference_ci_high=reference_summary.standardized_wasserstein_ci_high,
        median_separation_ratio=median
        / reference_summary.standardized_wasserstein_mean,
        mismatched_pairs_at_or_below_within_reference_ci_high=int(
            np.sum(mismatched <= reference_summary.standardized_wasserstein_ci_high)
        ),
    )


def _read_fingerprints(path: Path) -> tuple[SourceVisitCountFingerprintV1, ...]:
    rows = [
        SourceVisitCountFingerprintV1.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({row.anchor_id for row in rows}) != len(rows):
        raise ValueError("Source visit-count fingerprints must be nonempty and unique.")
    return tuple(rows)


def _metric_interval(
    prefix: str,
    values: npt.NDArray[np.float64],
) -> dict[str, float]:
    mean = float(values.mean())
    half_width = (
        0.0
        if len(values) == 1
        else float(
            student_t.ppf(0.975, df=len(values) - 1)
            * values.std(ddof=1)
            / np.sqrt(len(values))
        )
    )
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_ci_low": max(0.0, mean - half_width),
        f"{prefix}_ci_high": mean + half_width,
    }


def _resolve_release_path(root: Path, relative_path: str) -> Path:
    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError("Release paths must be relative.")
    resolved = (root.resolve() / requested).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("Release analysis path is absent or escapes the release.")
    return resolved


def _sha256_json(payload: dict[str, object]) -> str:
    import hashlib

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SourceVisitCountFingerprintV1",
    "VisitCountBinV1",
    "VisitCountNegativeControlV1",
    "VisitCountReplicationReportV1",
    "evaluate_visit_count_replication",
]
