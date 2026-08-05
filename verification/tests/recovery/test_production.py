"""Independent production-core recovery tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.cli import main as cli_main
from trialagentbench_validation.external.realism.process_replication import (
    SourceVisitCountFingerprintV1,
    VisitCountBinV1,
    VisitCountCellSummaryV1,
    _wrong_source_negative_control,
    evaluate_visit_count_replication,
)
from trialagentbench_validation.external.recovery.production import (
    ProductionCoreCellSummaryV1,
    ProductionCoreWorldEstimateV1,
    _fit_world,
    _PublicCellV1,
    _summarize_cells,
    _summarize_response_curves,
    _WorldReceiptV1,
    compare_production_core_candidates,
    evaluate_production_core_release,
)
from trialagentbench_validation.io import sha256_file, write_model


def test_recovery_validates_complete_release_without_construction_imports(
    tmp_path: Path,
) -> None:
    design: dict[str, object] = {
        "schema_id": "trialagentbench.production_core_qualification_design/v1",
        "anchors": [
            {
                "anchor_id": "anchor_0000000000000001",
                "source_subjects": 120,
                "manifest_sha256": "1" * 64,
            }
        ],
        "excluded_anchors": [],
        "cells": [
            {
                "cell_id": "source_size",
                "response_axis": "source_size",
                "level": 1.0,
                "sample_size_multiplier": 1.0,
                "minimum_sample_size": 20,
                "treatment_log_hazard_ratio": -0.4,
                "empirical_visit_count_log_hazard_ratio": 0.3,
                "baseline_hazard": 0.01,
            },
            {
                "cell_id": "reference",
                "response_axis": "reference",
                "level": 1.0,
                "sample_size_multiplier": 1.0,
                "minimum_sample_size": 80,
                "treatment_log_hazard_ratio": -0.4,
                "empirical_visit_count_log_hazard_ratio": 0.3,
                "baseline_hazard": 0.01,
            },
        ],
        "worlds_per_anchor_cell": 2,
        "seed": 451,
        "treatment_execution": "randomized_fully_adherent",
        "followup_horizon_dy": 360.0,
        "interval_width_dy": 30.0,
    }
    design_sha = _json_sha256(design)
    worlds_dir = tmp_path / "worlds"
    worlds_dir.mkdir()
    receipts: list[dict[str, object]] = []
    for cell_id in ("source_size", "reference"):
        for world_index in range(2):
            world_id = _world_id(design_sha, world_index, cell_id=cell_id)
            frame = _world_frame(seed=world_index)
            path = worlds_dir / f"{world_id}.parquet"
            frame.to_parquet(path, index=False)
            receipts.append(
                {
                    "world_id": world_id,
                    "anchor_id": "anchor_0000000000000001",
                    "cell_id": cell_id,
                    "world_index": world_index,
                    "seed": _world_seed(world_index, cell_id=cell_id),
                    "subjects": len(frame),
                    "events": int(frame["event"].sum()),
                    "truth_log_hazard_ratio": -0.4,
                    "truth_empirical_visit_count_log_hazard_ratio": 0.3,
                    "analysis_path": f"worlds/{world_id}.parquet",
                    "analysis_sha256": sha256_file(path),
                    "generator_spec_sha256": "2" * 64,
                    "ground_truth_manifest_sha256": "3" * 64,
                    "resampling_report_sha256": "4" * 64,
                    "generated_bundle_sha256": "5" * 64,
                }
            )
    receipt: dict[str, object] = {
        "schema_id": "trialagentbench.production_core_qualification_receipt/v1",
        "design_sha256": design_sha,
        "worlds": receipts,
    }
    receipt["checksum"] = _json_sha256(receipt)
    (tmp_path / "qualification_design.json").write_text(
        json.dumps(design),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "qualification_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    request: dict[str, object] = {
        "schema_id": "trialagentbench.production_core_qualification_request/v1",
        "design_sha256": design_sha,
        "estimands": [
            "treatment_log_hazard_ratio",
            "empirical_visit_count_log_hazard_ratio",
        ],
        "analysis_version": "production_core_cox_v1",
        "estimator_version": "statsmodels_phreg_breslow_v1",
    }
    request["checksum"] = _json_sha256(request)
    (tmp_path / "qualification_request.json").write_text(
        json.dumps(request),
        encoding="utf-8",
    )
    _write_run_binding(
        root=tmp_path,
        design_sha=design_sha,
        request_sha=str(request["checksum"]),
        receipt_path=receipt_path,
    )

    report = evaluate_production_core_release(
        release_dir=tmp_path,
        minimum_worlds_per_cell=2,
        workers=2,
    )
    with pytest.raises(ValueError, match="workers must be at least one"):
        evaluate_production_core_release(
            release_dir=tmp_path,
            minimum_worlds_per_cell=2,
            workers=0,
        )
    with pytest.raises(ValueError, match="minimum worlds per anchor-cell"):
        evaluate_production_core_release(
            release_dir=tmp_path,
            minimum_worlds_per_cell=3,
        )

    assert len(report.estimates) == 12
    assert len(report.cells) == 6
    assert all(row.successful_worlds == 2 for row in report.cells)
    reference = report.model_copy(
        update={
            "candidate_sha256": "6" * 64,
            "request_sha256": "7" * 64,
        }
    )
    comparison = report.model_copy(
        update={
            "candidate_sha256": "8" * 64,
            "request_sha256": "7" * 64,
        }
    )
    candidate_comparison = compare_production_core_candidates(reference, comparison)
    assert all(row.bias_change == 0 for row in candidate_comparison.cells)
    reference_path = tmp_path / "reference_recovery.json"
    comparison_path = tmp_path / "comparison_recovery.json"
    comparison_output = tmp_path / "candidate_comparison.json"
    write_model(reference_path, reference)
    write_model(comparison_path, comparison)
    assert (
        cli_main(
            [
                "generator-core-compare",
                "--reference",
                str(reference_path),
                "--comparison",
                str(comparison_path),
                "--output",
                str(comparison_output),
            ]
        )
        == 0
    )
    assert comparison_output.is_file()
    with pytest.raises(ValueError, match="same qualification request"):
        compare_production_core_candidates(
            reference,
            comparison.model_copy(update={"request_sha256": "9" * 64}),
        )

    source_fingerprints = tmp_path / "source_visit_count_fingerprints.jsonl"
    fingerprint = SourceVisitCountFingerprintV1(
        anchor_id="anchor_0000000000000001",
        manifest_sha256="1" * 64,
        participants=120,
        bins=tuple(
            VisitCountBinV1(visits=visits, participants=10) for visits in range(2, 14)
        ),
    )
    source_fingerprints.write_text(
        fingerprint.model_dump_json() + "\n", encoding="utf-8"
    )
    process_report = evaluate_visit_count_replication(
        release_dir=tmp_path,
        source_fingerprints=source_fingerprints,
    )
    assert len(process_report.anchor_cells) == 2
    assert all(cell.worlds == 2 for cell in process_report.cells)
    assert process_report.negative_control is None

    original_seed = receipts[1]["seed"]
    receipts[1]["seed"] = receipts[0]["seed"]
    receipt["checksum"] = _json_sha256(
        {key: value for key, value in receipt.items() if key != "checksum"}
    )
    receipt_path.write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="world seeds must be unique"):
        evaluate_production_core_release(
            release_dir=tmp_path,
            minimum_worlds_per_cell=2,
        )
    receipts[1]["seed"] = original_seed

    receipts.pop()
    receipt["worlds"] = receipts
    receipt["checksum"] = _json_sha256(
        {key: value for key, value in receipt.items() if key != "checksum"}
    )
    receipt_path.write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    _write_run_binding(
        root=tmp_path,
        design_sha=design_sha,
        request_sha=str(request["checksum"]),
        receipt_path=receipt_path,
    )
    with pytest.raises(ValueError, match="complete public design"):
        evaluate_production_core_release(
            release_dir=tmp_path,
            minimum_worlds_per_cell=2,
        )


def test_visit_count_negative_control_distinguishes_wrong_source() -> None:
    fingerprints = tuple(
        SourceVisitCountFingerprintV1(
            anchor_id=f"anchor_{index:016x}",
            manifest_sha256=str(index) * 64,
            participants=20,
            bins=(
                VisitCountBinV1(visits=offset, participants=10),
                VisitCountBinV1(visits=offset + 1, participants=10),
            ),
        )
        for index, offset in ((1, 1), (2, 10))
    )
    result = _wrong_source_negative_control(
        fingerprints=fingerprints,
        reference_world_values={
            "anchor_0000000000000001": [np.tile([1.0, 2.0], 10)],
            "anchor_0000000000000002": [np.tile([10.0, 11.0], 10)],
        },
        reference_summary=VisitCountCellSummaryV1(
            cell_id="reference",
            anchors=2,
            worlds=4,
            standardized_wasserstein_mean=0.1,
            standardized_wasserstein_ci_low=0.0,
            standardized_wasserstein_ci_high=0.2,
            standardized_mean_error_mean=0.1,
            standardized_mean_error_ci_low=0.0,
            standardized_mean_error_ci_high=0.2,
            absolute_log_sd_ratio_mean=0.1,
            absolute_log_sd_ratio_ci_low=0.0,
            absolute_log_sd_ratio_ci_high=0.2,
        ),
    )

    assert result is not None
    assert result.mismatched_anchor_pairs == 2
    assert result.mismatched_pairs_at_or_below_within_reference_ci_high == 0
    assert result.median_separation_ratio > 100


def test_response_curves_use_within_anchor_slopes() -> None:
    rows: list[ProductionCoreCellSummaryV1] = []
    for anchor_index in (1, 2):
        anchor_id = f"anchor_{anchor_index:016x}"
        for cell_id, axis, truth, subjects, standard_deviation in (
            ("effect_null", "effect", 0.0, 200.0, 0.12),
            ("effect_lower", "effect", -0.2, 200.0, 0.12),
            ("reference", "reference", -0.4, 200.0, 1.0 / np.sqrt(200.0)),
            ("effect_higher", "effect", -0.6, 200.0, 0.12),
        ):
            rows.append(
                _cell_summary(
                    anchor_id=anchor_id,
                    cell_id=cell_id,
                    response_axis=axis,
                    estimand="treatment_log_hazard_ratio",
                    route="adjusted_cox",
                    truth=truth,
                    estimate=truth,
                    subjects=subjects,
                    standard_deviation=standard_deviation,
                )
            )
            rows.append(
                _cell_summary(
                    anchor_id=anchor_id,
                    cell_id=cell_id,
                    response_axis=axis,
                    estimand="treatment_log_hazard_ratio",
                    route="binary_endpoint_shortcut",
                    truth=truth,
                    estimate=2.0 * truth,
                    subjects=subjects,
                    standard_deviation=standard_deviation,
                )
            )
        for cell_id, axis, truth in (
            ("mechanism_null", "mechanism", 0.0),
            ("mechanism_lower", "mechanism", 0.15),
            ("reference", "reference", 0.3),
            ("mechanism_higher", "mechanism", 0.45),
        ):
            rows.append(
                _cell_summary(
                    anchor_id=anchor_id,
                    cell_id=cell_id,
                    response_axis=axis,
                    estimand="empirical_visit_count_log_hazard_ratio",
                    route="adjusted_cox",
                    truth=truth,
                    estimate=truth,
                    subjects=200.0,
                    standard_deviation=0.1,
                )
            )
        for cell_id, axis, subjects in (
            ("information_lower", "information", 100.0),
            ("information_higher", "information", 400.0),
        ):
            rows.append(
                _cell_summary(
                    anchor_id=anchor_id,
                    cell_id=cell_id,
                    response_axis=axis,
                    estimand="treatment_log_hazard_ratio",
                    route="adjusted_cox",
                    truth=-0.4,
                    estimate=-0.4,
                    subjects=subjects,
                    standard_deviation=1.0 / np.sqrt(subjects),
                )
            )

    curves = {row.curve: row for row in _summarize_response_curves(rows)}

    assert curves["treatment_effect_recovery"].slope_mean == pytest.approx(1.0)
    assert curves["empirical_process_recovery"].slope_mean == pytest.approx(1.0)
    assert curves["information_precision"].slope_mean == pytest.approx(-0.5)
    assert curves["time_discarding_route"].slope_mean == pytest.approx(2.0)
    assert all(row.anchors == 2 for row in curves.values())


def test_portfolio_se_calibration_excludes_between_anchor_location() -> None:
    estimates = []
    standard_error = float(np.sqrt(0.02))
    for anchor_index, values in ((1, (0.9, 1.1)), (2, (-1.1, -0.9))):
        for world_index, estimate in enumerate(values):
            estimates.append(
                ProductionCoreWorldEstimateV1(
                    world_id=f"world_{anchor_index:02d}_{world_index}",
                    anchor_id=f"anchor_{anchor_index:016x}",
                    cell_id="reference",
                    world_index=world_index,
                    estimand="treatment_log_hazard_ratio",
                    route="adjusted_cox",
                    truth_log_hazard_ratio=0.0,
                    subjects=100,
                    events=50,
                    estimate=estimate,
                    standard_error=standard_error,
                    covered=False,
                    rejected_null=True,
                )
            )
    cell = _PublicCellV1(
        cell_id="reference",
        response_axis="reference",
        level=1.0,
        sample_size_multiplier=1.0,
        minimum_sample_size=100,
        treatment_log_hazard_ratio=0.0,
        empirical_visit_count_log_hazard_ratio=0.0,
        baseline_hazard=0.01,
    )

    summary = _summarize_cells(estimates, cell_by_id={"reference": cell})[0]

    assert summary.mean_estimate == pytest.approx(0.0)
    assert summary.empirical_standard_deviation == pytest.approx(standard_error)
    assert summary.model_to_empirical_se_ratio == pytest.approx(1.0)
    assert summary.bias_ci_low < summary.bias < summary.bias_ci_high
    assert summary.rmse_ci_low <= summary.rmse <= summary.rmse_ci_high
    assert (
        summary.model_to_empirical_se_ratio_ci_low
        <= summary.model_to_empirical_se_ratio
        <= summary.model_to_empirical_se_ratio_ci_high
    )


def test_scheduled_coverage_counts_failed_world_as_not_covered() -> None:
    cell = _PublicCellV1(
        cell_id="reference",
        response_axis="reference",
        level=1.0,
        sample_size_multiplier=1.0,
        minimum_sample_size=100,
        treatment_log_hazard_ratio=-0.4,
        empirical_visit_count_log_hazard_ratio=0.3,
        baseline_hazard=0.01,
    )
    estimates = [
        ProductionCoreWorldEstimateV1(
            world_id="world_success",
            anchor_id="anchor_0000000000000001",
            cell_id="reference",
            world_index=0,
            estimand="treatment_log_hazard_ratio",
            route="adjusted_cox",
            truth_log_hazard_ratio=-0.4,
            subjects=100,
            events=40,
            estimate=-0.4,
            standard_error=0.1,
            covered=True,
            rejected_null=True,
        ),
        ProductionCoreWorldEstimateV1(
            world_id="world_failure",
            anchor_id="anchor_0000000000000001",
            cell_id="reference",
            world_index=1,
            estimand="treatment_log_hazard_ratio",
            route="adjusted_cox",
            truth_log_hazard_ratio=-0.4,
            subjects=100,
            events=2,
            failure="singular_information",
        ),
    ]

    summary = _summarize_cells(estimates, cell_by_id={"reference": cell})[0]

    assert summary.coverage is None
    assert summary.coverage_scheduled == 0.5
    assert summary.rejection_rate is None
    assert summary.rejection_rate_scheduled == 0.5
    assert summary.failures == 1


def test_zero_event_world_is_reported_as_nonestimable() -> None:
    frame = _world_frame(seed=123)
    frame["event"] = 0
    world = _WorldReceiptV1(
        world_id="world_00000000000000000000",
        anchor_id="anchor_0000000000000001",
        cell_id="source_size",
        world_index=0,
        seed=123,
        subjects=len(frame),
        events=0,
        truth_log_hazard_ratio=-0.4,
        truth_empirical_visit_count_log_hazard_ratio=0.3,
        analysis_path="worlds/world_00000000000000000000.parquet",
        analysis_sha256="1" * 64,
        generator_spec_sha256="2" * 64,
        ground_truth_manifest_sha256="3" * 64,
        resampling_report_sha256="4" * 64,
        generated_bundle_sha256="5" * 64,
    )

    estimates = _fit_world(frame, world=world)

    assert len(estimates) == 3
    assert all(row.failure == "singular_information" for row in estimates)


def test_sparse_cox_information_warning_is_reported_as_nonestimable() -> None:
    frame = _world_frame(seed=456)
    frame["event"] = 0
    frame.loc[0, "event"] = 1
    world = _WorldReceiptV1(
        world_id="world_00000000000000000001",
        anchor_id="anchor_0000000000000001",
        cell_id="source_size",
        world_index=1,
        seed=456,
        subjects=len(frame),
        events=1,
        truth_log_hazard_ratio=-0.4,
        truth_empirical_visit_count_log_hazard_ratio=0.3,
        analysis_path="worlds/world_00000000000000000001.parquet",
        analysis_sha256="1" * 64,
        generator_spec_sha256="2" * 64,
        ground_truth_manifest_sha256="3" * 64,
        resampling_report_sha256="4" * 64,
        generated_bundle_sha256="5" * 64,
    )

    estimates = _fit_world(frame, world=world)

    adjusted = [row for row in estimates if row.route == "adjusted_cox"]
    assert len(adjusted) == 2
    assert all(row.failure == "singular_information" for row in adjusted)


def test_cell_summary_accepts_machine_roundoff_at_interval_boundary() -> None:
    summary = _cell_summary(
        anchor_id="anchor_0000000000000001",
        cell_id="reference",
        response_axis="reference",
        estimand="treatment_log_hazard_ratio",
        route="adjusted_cox",
        truth=0.0,
        estimate=0.0,
        subjects=100.0,
        standard_deviation=0.1,
    )
    payload = summary.model_dump(mode="python")
    payload.update(
        coverage=1.0,
        coverage_ci_low=0.7,
        coverage_ci_high=np.nextafter(1.0, 0.0),
    )

    recovered = ProductionCoreCellSummaryV1.model_validate(payload)

    assert recovered.coverage == 1.0
    payload["coverage_ci_high"] = 0.999
    with pytest.raises(ValueError, match="must contain"):
        ProductionCoreCellSummaryV1.model_validate(payload)


def _cell_summary(
    *,
    anchor_id: str,
    cell_id: str,
    response_axis: str,
    estimand: str,
    route: str,
    truth: float,
    estimate: float,
    subjects: float,
    standard_deviation: float,
) -> ProductionCoreCellSummaryV1:
    return ProductionCoreCellSummaryV1.model_validate(
        {
            "anchor_id": anchor_id,
            "anchors": 1,
            "uncertainty_unit": "world",
            "cell_id": cell_id,
            "response_axis": response_axis,
            "level": truth,
            "estimand": estimand,
            "route": route,
            "truth_log_hazard_ratio": truth,
            "worlds": 10,
            "successful_worlds": 10,
            "failures": 0,
            "mean_subjects": subjects,
            "mean_events": 50.0,
            "mean_event_fraction": 0.25,
            "mean_estimate": estimate,
            "bias": estimate - truth,
            "bias_mcse": 0.01,
            "rmse": abs(estimate - truth),
            "empirical_standard_deviation": standard_deviation,
            "mean_model_standard_error": standard_deviation,
            "model_to_empirical_se_ratio": 1.0,
            "coverage": 0.95,
            "coverage_ci_low": 0.8,
            "coverage_ci_high": 1.0,
            "coverage_scheduled": 0.95,
            "coverage_scheduled_ci_low": 0.8,
            "coverage_scheduled_ci_high": 1.0,
            "rejection_rate": 0.8,
            "rejection_rate_ci_low": 0.5,
            "rejection_rate_ci_high": 1.0,
            "rejection_rate_scheduled": 0.8,
            "rejection_rate_scheduled_ci_low": 0.5,
            "rejection_rate_scheduled_ci_high": 1.0,
        }
    )


def _world_frame(*, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = 120
    treatment = np.tile([0, 1], subjects // 2)
    visit_count = rng.integers(2, 14, size=subjects).astype(float)
    visit_z = (visit_count - visit_count.mean()) / visit_count.std(ddof=0)
    hazard = 0.01 * np.exp(-0.4 * treatment + 0.3 * visit_z)
    event_time = rng.exponential(1.0 / hazard)
    censor_time = rng.uniform(20.0, 180.0, size=subjects)
    event = (event_time <= censor_time).astype(int)
    return pd.DataFrame(
        {
            "participant_id": [f"S{index:04d}" for index in range(subjects)],
            "treatment": treatment,
            "time": np.minimum(event_time, censor_time),
            "event": event,
            "empirical_visit_count": visit_count,
        }
    )


def _world_seed(world_index: int, *, cell_id: str = "reference") -> int:
    payload = f"451:anchor_0000000000000001:{cell_id}:{world_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big")


def _world_id(design_sha: str, world_index: int, *, cell_id: str = "reference") -> str:
    payload = f"{design_sha}:anchor_0000000000000001:{cell_id}:{world_index}".encode()
    return f"world_{hashlib.sha256(payload).hexdigest()[:20]}"


def _json_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_run_binding(
    *,
    root: Path,
    design_sha: str,
    request_sha: str,
    receipt_path: Path,
) -> None:
    run: dict[str, object] = {
        "schema_id": "trialagentbench.production_core_qualification_run/v1",
        "candidate_sha256": "6" * 64,
        "request_sha256": request_sha,
        "design_sha256": design_sha,
        "receipt_sha256": sha256_file(receipt_path),
    }
    run["checksum"] = _json_sha256(run)
    (root / "qualification_run.json").write_text(
        json.dumps(run),
        encoding="utf-8",
    )
