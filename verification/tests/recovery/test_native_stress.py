"""Independent native clinical-mechanism recovery tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.external.recovery.native_stress import (
    NativeStressWorldEstimateV1,
    _CellV1,
    _require_full_rank_covariance,
    _summaries,
    evaluate_native_stress_release,
)
from trialagentbench_validation.io import sha256_file


def test_native_stress_release_recovers_complete_checksum_bound_design(
    tmp_path: Path,
) -> None:
    cells = [
        _hazard_cell("hazard_constant", 0.0),
        _hazard_cell("hazard_lower", 0.3),
        _hazard_cell("hazard_reference", 0.6),
        _hazard_cell("hazard_higher", 0.9),
        _safety_cell("safety_null", 0.0),
        _safety_cell("safety_lower", float(np.log(1.5))),
        _safety_cell("safety_reference", float(np.log(2.0))),
        _safety_cell("safety_higher", float(np.log(3.0))),
        _frailty_cell("frailty_null", 0.0),
        _frailty_cell("frailty_lower", 0.4),
        _frailty_cell("frailty_reference", 0.8),
        _frailty_cell("frailty_higher", 1.4),
    ]
    design: dict[str, object] = {
        "schema_id": "trialagentbench.native_stress_design/v1",
        "anchors": [
            {"anchor_id": f"anchor_{index:016x}", "manifest_sha256": str(index) * 64}
            for index in (1, 2)
        ],
        "cells": cells,
        "seed": 451,
        "followup_horizon_dy": 360.0,
        "interval_width_dy": 30.0,
        "change_point_dy": 180.0,
        "minimum_late_support_participants": 20,
        "adverse_event_baseline_rate_per_day": 0.003,
    }
    design_sha = _json_sha(design)
    worlds_dir = tmp_path / "worlds"
    worlds_dir.mkdir()
    worlds = []
    for anchor_index in (1, 2):
        anchor_id = f"anchor_{anchor_index:016x}"
        for cell in cells:
            for world_index in range(2):
                seed = _world_seed(451, anchor_id, str(cell["cell_id"]), world_index)
                world_id = _world_id(
                    design_sha, anchor_id, str(cell["cell_id"]), world_index
                )
                if cell["family"] == "time_varying_hazard":
                    frame = _hazard_frame(
                        seed=seed,
                        early=float(cell["early_treatment_log_hazard_ratio"]),
                        late=float(cell["late_treatment_log_hazard_ratio"]),
                    )
                    events = int(frame["event"].sum())
                else:
                    frame = _safety_frame(
                        seed=seed,
                        log_rate_ratio=float(cell["treatment_log_rate_ratio"]),
                        frailty_variance=float(cell["subject_rate_frailty_variance"]),
                    )
                    events = int(frame["recurrent_event_count"].sum())
                path = worlds_dir / f"{world_id}.parquet"
                frame.to_parquet(path, index=False)
                worlds.append(
                    {
                        "world_id": world_id,
                        "anchor_id": anchor_id,
                        "cell_id": cell["cell_id"],
                        "family": cell["family"],
                        "world_index": world_index,
                        "seed": seed,
                        "subjects": len(frame),
                        "events": events,
                        "analysis_path": f"worlds/{world_id}.parquet",
                        "analysis_sha256": sha256_file(path),
                        "generator_spec_sha256": "a" * 64,
                        "ground_truth_manifest_sha256": "b" * 64,
                        "resampling_report_sha256": "c" * 64,
                        "generated_bundle_sha256": "d" * 64,
                    }
                )
    receipt: dict[str, object] = {
        "schema_id": "trialagentbench.native_stress_receipt/v1",
        "design_sha256": design_sha,
        "worlds": worlds,
    }
    receipt["checksum"] = _json_sha(receipt)
    (tmp_path / "design.json").write_text(json.dumps(design), encoding="utf-8")
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    report = evaluate_native_stress_release(
        release_dir=tmp_path,
        minimum_null_worlds_per_anchor=2,
        minimum_nonnull_worlds_per_anchor=2,
    )
    parallel_report = evaluate_native_stress_release(
        release_dir=tmp_path,
        minimum_null_worlds_per_anchor=2,
        minimum_nonnull_worlds_per_anchor=2,
        workers=2,
    )

    assert len(report.estimates) == 112
    assert parallel_report == report
    assert len(report.mechanisms) == 48
    assert not [row for row in report.estimates if row.failure is not None]
    assert all(
        row.early_events + row.late_events == row.events
        for row in report.mechanisms
        if row.family == "time_varying_hazard"
        and row.early_events is not None
        and row.late_events is not None
    )
    assert len(report.curves) == 5
    assert {row.route for row in report.curves} == {
        "segmented_cox",
        "binary_endpoint",
        "poisson_rate",
        "binary_any_event",
        "nb2_profile_likelihood",
    }
    assert all(
        row.model_to_empirical_se_ratio_ci_low is not None
        and row.model_to_empirical_se_ratio_ci_high is not None
        for row in report.cells
        if row.route != "nb2_profile_likelihood"
    )
    assert all(
        row.model_to_empirical_se_ratio is None
        for row in report.cells
        if row.route == "nb2_profile_likelihood"
    )
    frailty_estimates = [
        row for row in report.estimates if row.route == "nb2_profile_likelihood"
    ]
    assert all(
        row.interval_low is not None
        and row.interval_high is not None
        and row.interval_low >= 0.0
        for row in frailty_estimates
    )
    assert all(row.coverage_scheduled == row.coverage for row in report.cells)
    assert all(
        row.rejection_rate_scheduled == row.rejection_rate for row in report.cells
    )
    recurrent_mechanisms = [
        row for row in report.mechanisms if row.family == "recurrent_adverse_event"
    ]
    assert all(
        row.configured_frailty_variance is not None
        and row.direct_frailty_moment is not None
        for row in recurrent_mechanisms
    )

    safety_world = next(
        row for row in worlds if row["family"] == "recurrent_adverse_event"
    )
    safety_path = tmp_path / str(safety_world["analysis_path"])
    safety_frame = pd.read_parquet(safety_path)
    safety_frame.loc[0, "followup"] = 361.0
    safety_frame.to_parquet(safety_path, index=False)
    safety_world["analysis_sha256"] = sha256_file(safety_path)
    receipt.pop("checksum")
    receipt["checksum"] = _json_sha(receipt)
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="recurrent-event values are invalid"):
        evaluate_native_stress_release(
            release_dir=tmp_path,
            minimum_null_worlds_per_anchor=2,
            minimum_nonnull_worlds_per_anchor=2,
        )

    world_path = tmp_path / str(worlds[0]["analysis_path"])
    frame = pd.read_parquet(world_path)
    frame.loc[0, "treatment"] = 1 - int(frame.loc[0, "treatment"])
    frame.to_parquet(world_path, index=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluate_native_stress_release(
            release_dir=tmp_path,
            minimum_null_worlds_per_anchor=2,
            minimum_nonnull_worlds_per_anchor=2,
        )


def test_native_stress_scheduled_coverage_retains_failed_fits() -> None:
    """Failure-inclusive summaries do not condition away non-estimability."""

    cell = _CellV1.model_validate(_hazard_cell("hazard_constant", 0.0))
    estimates = [
        _estimate(anchor=1, world=0, covered=True),
        _estimate(anchor=1, world=1, covered=None),
        _estimate(anchor=2, world=0, covered=True),
        _estimate(anchor=2, world=1, covered=True),
    ]

    summary = _summaries(estimates, {cell.cell_id: cell})[0]

    assert summary.coverage == 1.0
    assert summary.coverage_scheduled == 0.75
    assert summary.successful_worlds == 3
    assert summary.failures == 1


def test_native_stress_rejects_numerically_singular_covariance() -> None:
    """Converged estimators still require identifiable information."""

    covariance = np.diag([1.0, 1e24, 1.0])

    with pytest.raises(np.linalg.LinAlgError, match="full rank"):
        _require_full_rank_covariance(covariance)


def _hazard_cell(cell_id: str, contrast: float) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "family": "time_varying_hazard",
        "response_axis": "hazard_contrast",
        "level": contrast,
        "worlds_per_anchor": 2,
        "sample_size_multiplier": 1.0,
        "minimum_sample_size": 192,
        "early_treatment_log_hazard_ratio": -0.4 - contrast / 2,
        "late_treatment_log_hazard_ratio": -0.4 + contrast / 2,
    }


def _estimate(
    *,
    anchor: int,
    world: int,
    covered: bool | None,
) -> NativeStressWorldEstimateV1:
    common = {
        "world_id": f"world_{anchor:010x}{world:010x}",
        "anchor_id": f"anchor_{anchor:016x}",
        "cell_id": "hazard_constant",
        "world_index": world,
        "family": "time_varying_hazard",
        "parameter": "hazard_contrast",
        "route": "segmented_cox",
        "truth": 0.0,
        "subjects": 100,
        "events": 20,
    }
    if covered is None:
        return NativeStressWorldEstimateV1(**common, failure="convergence")
    return NativeStressWorldEstimateV1(
        **common,
        estimate=0.01,
        standard_error=0.1,
        interval_low=-0.186,
        interval_high=0.206,
        covered=covered,
        rejected_null=False,
    )


def _safety_cell(cell_id: str, log_rate_ratio: float) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "family": "recurrent_adverse_event",
        "response_axis": "treatment_log_rate_ratio",
        "level": abs(log_rate_ratio),
        "worlds_per_anchor": 2,
        "sample_size_multiplier": 1.0,
        "minimum_sample_size": 192,
        "treatment_log_rate_ratio": log_rate_ratio,
        "subject_rate_frailty_variance": 0.8,
    }


def _frailty_cell(cell_id: str, frailty_variance: float) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "family": "recurrent_adverse_event",
        "response_axis": "subject_rate_frailty_variance",
        "level": frailty_variance,
        "worlds_per_anchor": 2,
        "sample_size_multiplier": 1.0,
        "minimum_sample_size": 192,
        "treatment_log_rate_ratio": 0.0,
        "subject_rate_frailty_variance": frailty_variance,
    }


def _hazard_frame(*, seed: int, early: float, late: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = 400
    treatment = np.tile([0, 1], subjects // 2)
    rng.shuffle(treatment)
    visit = rng.normal(size=subjects)
    early_rate = 0.008 * np.exp(early * treatment + 0.2 * visit)
    late_rate = 0.008 * np.exp(late * treatment + 0.2 * visit)
    target = -np.log(rng.uniform(size=subjects))
    early_hazard = early_rate * 180.0
    time = np.where(
        target <= early_hazard,
        target / early_rate,
        180.0 + (target - early_hazard) / late_rate,
    )
    event = (time <= 360.0).astype(int)
    return pd.DataFrame(
        {
            "participant_id": [f"P{index:04d}" for index in range(subjects)],
            "treatment": treatment,
            "empirical_visit_count_z": visit,
            "time": np.minimum(time, 360.0),
            "event": event,
        }
    )


def _safety_frame(
    *, seed: int, log_rate_ratio: float, frailty_variance: float
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = 400
    treatment = np.tile([0, 1], subjects // 2)
    rng.shuffle(treatment)
    visit = rng.normal(size=subjects)
    followup = rng.uniform(180.0, 360.0, size=subjects)
    rate = 0.003 * np.exp(log_rate_ratio * treatment)
    frailty = (
        np.ones(subjects)
        if frailty_variance == 0.0
        else rng.gamma(1.0 / frailty_variance, frailty_variance, subjects)
    )
    return pd.DataFrame(
        {
            "participant_id": [f"P{index:04d}" for index in range(subjects)],
            "treatment": treatment,
            "empirical_visit_count_z": visit,
            "followup": followup,
            "recurrent_event_count": rng.poisson(rate * followup * frailty),
        }
    )


def _json_sha(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


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
