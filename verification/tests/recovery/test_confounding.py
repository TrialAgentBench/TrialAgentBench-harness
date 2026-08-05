"""Public confounding and overlap verifier tests."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from trialagentbench_validation.cli import main
from trialagentbench_validation.external.recovery.confounding import (
    ConfoundingQualificationDesignV1,
    ConfoundingWorldEstimateV1,
    ConfoundingWorldReceiptV1,
    _summarize_cells,
)
from trialagentbench_validation.io import sha256_file


def test_confounding_receipt_uses_generator_spec_hash_name() -> None:
    receipt = ConfoundingWorldReceiptV1.model_validate(
        {
            "world_id": "conf-" + "a" * 20,
            "trial_id": "RCTBENCH-001",
            "cell_id": "assignment_+0.0__n_1",
            "world_index": 0,
            "seed": 451,
            "subjects": 100,
            "analysis_path": "worlds/conf-" + "a" * 20 + ".parquet",
            "analysis_sha256": "b" * 64,
            "generator_spec_sha256": "c" * 64,
            "generated_bundle_sha256": "d" * 64,
        }
    )

    assert receipt.generator_spec_sha256 == "c" * 64
    assert set(receipt.model_dump()).issuperset({"generator_spec_sha256"})


def test_confounding_receipt_rejects_unknown_provenance_field() -> None:
    with pytest.raises(ValueError, match="unknown_provenance_field"):
        ConfoundingWorldReceiptV1.model_validate(
            {
                "world_id": "conf-" + "a" * 20,
                "trial_id": "RCTBENCH-001",
                "cell_id": "assignment_+0.0__n_1",
                "world_index": 0,
                "seed": 451,
                "subjects": 100,
                "analysis_path": "worlds/conf-" + "a" * 20 + ".parquet",
                "analysis_sha256": "b" * 64,
                "unknown_provenance_field": "c" * 64,
                "generated_bundle_sha256": "d" * 64,
            }
        )


def test_confounding_qualification_recovers_curves_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    worlds_dir = tmp_path / "worlds"
    worlds_dir.mkdir()
    cells = [
        {
            "cell_id": f"assignment_{strength:+.1f}__n_{multiplier:g}",
            "assignment_strength": strength,
            "sample_size_multiplier": multiplier,
            "worlds_per_trial": 2,
        }
        for multiplier in (1.0, 4.0)
        for strength in (-2.0, -1.0, 0.0, 1.0, 2.0)
    ]
    trials = [_trial(f"RCTBENCH-{index:03d}") for index in (1, 2, 3)]
    design = {
        "schema_id": "trialagentbench.confounding_qualification_design/v1",
        "source_revision": "a" * 40,
        "seed": 451,
        "outcome_intercept": -1.5,
        "exposure_log_odds_coefficient": 0.8,
        "age_log_odds_coefficient": 0.5,
        "bmi_log_odds_coefficient": 0.4,
        "propensity_minimum": 0.02,
        "propensity_maximum": 0.98,
        "trials": trials,
        "cells": cells,
    }
    design_sha256 = _json_sha(design)
    receipts = []
    for trial in trials:
        trial_id = trial["qualification"]["trial_id"]
        for cell in cells:
            for world_index in range(2):
                seed = _world_seed(451, trial_id, world_index)
                world_id = _world_id(
                    design_sha256,
                    trial_id,
                    cell["cell_id"],
                    world_index,
                )
                subjects = int(300 * cell["sample_size_multiplier"])
                frame = _world(
                    seed=seed,
                    subjects=subjects,
                    assignment_strength=cell["assignment_strength"],
                )
                frame.insert(0, "cell_id", cell["cell_id"])
                frame.insert(0, "trial_id", trial_id)
                frame.insert(0, "world_id", world_id)
                path = worlds_dir / f"{world_id}.parquet"
                frame.to_parquet(path, index=False)
                receipts.append(
                    {
                        "world_id": world_id,
                        "trial_id": trial_id,
                        "cell_id": cell["cell_id"],
                        "world_index": world_index,
                        "seed": seed,
                        "subjects": subjects,
                        "analysis_path": f"worlds/{world_id}.parquet",
                        "analysis_sha256": sha256_file(path),
                        "generator_spec_sha256": "b" * 64,
                        "generated_bundle_sha256": "c" * 64,
                    }
                )
    receipt = {
        "schema_id": "trialagentbench.confounding_qualification_receipt/v1",
        "design_sha256": design_sha256,
        "worlds": receipts,
    }
    (tmp_path / "design.json").write_text(json.dumps(design), encoding="utf-8")
    (tmp_path / "receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "confounding-validation",
            "--release-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--minimum-null-worlds",
            "2",
            "--minimum-nonnull-worlds",
            "2",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(report["estimates"]) == 60
    assert len(report["dose_responses"]) == 24
    assert len(report["overlap_responses"]) == 12
    assert len(report["information_responses"]) == 15
    assert all(row["failures"] == 0 for row in report["cell_summaries"])
    propensity = [
        row["mean_slope"]
        for row in report["dose_responses"]
        if row["response"] == "propensity_coefficient"
    ]
    naive = [
        row["mean_slope"]
        for row in report["dose_responses"]
        if row["response"] == "naive_exposure_bias"
    ]
    adjusted = [
        row["mean_slope"]
        for row in report["dose_responses"]
        if row["response"] == "adjusted_exposure_bias"
    ]
    assert min(propensity) > 0.8
    assert min(naive) > 0.2
    assert max(abs(value) for value in adjusted) < 0.2
    assert min(row["mean_slope"] for row in report["overlap_responses"]) > 0
    assert all(
        row["maximum_point_estimator_crosscheck_difference"]
        <= row["maximum_point_estimator_crosscheck_tolerance"]
        for row in report["cell_summaries"]
    )
    assert all(
        row["propensity_coverage_successful"] == row["propensity_coverage_scheduled"]
        and row["adjusted_coverage_successful"] == row["adjusted_coverage_scheduled"]
        and row["maximum_coverage_denominator_gap"] == 0.0
        for row in report["cell_summaries"]
    )

    tampered = tmp_path / receipts[0]["analysis_path"]
    frame = pd.read_parquet(tampered)
    frame.at[0, "outcome"] = 1 - frame.at[0, "outcome"]
    frame.to_parquet(tampered, index=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        main(
            [
                "confounding-validation",
                "--release-dir",
                str(tmp_path),
                "--output",
                str(tmp_path / "tampered.json"),
                "--minimum-null-worlds",
                "2",
                "--minimum-nonnull-worlds",
                "2",
            ]
        )


def test_confounding_summary_retains_failed_worlds_in_scheduled_coverage() -> None:
    cells = tuple(
        {
            "cell_id": f"assignment_{strength:+.1f}__n_{multiplier:g}",
            "assignment_strength": strength,
            "sample_size_multiplier": multiplier,
            "worlds_per_trial": 3,
        }
        for multiplier in (1.0, 4.0)
        for strength in (-2.0, -1.0, 0.0, 1.0, 2.0)
    )
    design = ConfoundingQualificationDesignV1.model_validate(
        {
            "schema_id": "trialagentbench.confounding_qualification_design/v1",
            "source_revision": "a" * 40,
            "seed": 451,
            "outcome_intercept": -1.5,
            "exposure_log_odds_coefficient": 0.8,
            "age_log_odds_coefficient": 0.5,
            "bmi_log_odds_coefficient": 0.4,
            "propensity_minimum": 0.02,
            "propensity_maximum": 0.98,
            "trials": [_trial(f"RCTBENCH-{index:03d}") for index in (1, 2, 3)],
            "cells": cells,
        }
    )
    trial_id = "RCTBENCH-001"
    cell_id = "assignment_+2.0__n_1"
    estimates = [
        ConfoundingWorldEstimateV1(
            world_id=f"world-{index}",
            trial_id=trial_id,
            cell_id=cell_id,
            world_index=index,
            assignment_strength=2.0,
            sample_size_multiplier=1.0,
            propensity_truth=2.0,
            propensity_estimate=2.0,
            propensity_standard_error=0.2,
            propensity_covered=index == 0,
            naive_exposure_estimate=1.0,
            adjusted_exposure_estimate=0.8,
            adjusted_exposure_standard_error=0.2,
            adjusted_exposure_covered=index == 0,
            risk_difference_truth=0.1,
            oracle_ipw_risk_difference=0.1,
            estimated_ipw_risk_difference=0.1,
            exposed_fraction=0.5,
            extreme_propensity_fraction=0.2,
            oracle_effective_sample_fraction=0.7,
            score_mean_difference=0.4,
            maximum_point_estimator_crosscheck_difference=2e-6,
            maximum_point_estimator_crosscheck_tolerance=2e-4,
        )
        for index in range(2)
    ]
    failures: defaultdict[tuple[str, str], int] = defaultdict(int)
    for trial in design.trials:
        for cell in design.cells:
            failures[(trial.qualification.trial_id, cell.cell_id)] = (
                cell.worlds_per_trial
            )
    failures[(trial_id, cell_id)] = 1

    summary = next(
        row
        for row in _summarize_cells(estimates, design, failures)
        if row.trial_id == trial_id and row.cell_id == cell_id
    )

    assert summary.successful_worlds == 2
    assert summary.failures == 1
    assert summary.propensity_coverage_successful == 0.5
    assert summary.propensity_coverage_scheduled == pytest.approx(1 / 3)
    assert summary.adjusted_coverage_successful == 0.5
    assert summary.adjusted_coverage_scheduled == pytest.approx(1 / 3)
    assert summary.maximum_coverage_denominator_gap == pytest.approx(1 / 6)
    assert summary.propensity_coverage_scheduled_ci_low < 1 / 3
    assert summary.propensity_coverage_scheduled_ci_high > 1 / 3


def _trial(trial_id: str) -> dict[str, object]:
    return {
        "qualification": {
            "trial_id": trial_id,
            "source_data_sha256": "d" * 64,
            "source_dictionary_sha256": "e" * 64,
            "worlds": 2,
            "fitted_analysis": {
                "outcome_kind": "binary",
                "source_subjects": 300,
                "source_control_subjects": 150,
                "source_active_subjects": 150,
                "source_event_rate": 0.3,
                "active_source_level": "active",
                "intercept": -1.0,
                "treatment_coefficient": 0.5,
                "age_coefficient": 0.1,
                "bmi_coefficient": 0.2,
                "analysis_treatment_effect": 0.1,
                "analysis_age_coefficient": 0.1,
                "analysis_bmi_coefficient": 0.2,
                "age_center": 50.0,
                "bmi_center": 25.0,
                "source_adjusted_standard_error": 0.1,
                "source_unadjusted_standard_error": 0.1,
                "source_adjusted_to_unadjusted_se_ratio": 1.0,
                "residual_probabilities": [0.1, 0.25, 0.5, 0.75, 0.9],
                "residual_quantiles": [-2.0, -1.0, 0.0, 1.0, 2.0],
            },
        },
        "age_mean": 50.0,
        "age_standard_deviation": 10.0,
        "bmi_mean": 25.0,
        "bmi_standard_deviation": 4.0,
    }


def _world(
    *,
    seed: int,
    subjects: int,
    assignment_strength: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age_z = rng.normal(size=subjects)
    bmi_z = 0.35 * age_z + np.sqrt(1.0 - 0.35**2) * rng.normal(size=subjects)
    score = (age_z + bmi_z) / np.sqrt(2.0)
    propensity = np.clip(expit(assignment_strength * score), 0.02, 0.98)
    exposed = (rng.random(subjects) < propensity).astype(np.int8)
    outcome_probability = expit(-1.5 + 0.8 * exposed + 0.5 * age_z + 0.4 * bmi_z)
    outcome = (rng.random(subjects) < outcome_probability).astype(np.int8)
    return pd.DataFrame(
        {
            "participant_id": [f"P{index:05d}" for index in range(subjects)],
            "age": 50.0 + 10.0 * age_z,
            "bmi": 25.0 + 4.0 * bmi_z,
            "exposed": exposed,
            "outcome": outcome,
        }
    )


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
