"""Public longitudinal observation-process verifier tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.cli import main
from trialagentbench_validation.external.realism.longitudinal import (
    fingerprint_longitudinal_trial,
)
from trialagentbench_validation.io import sha256_file


def test_longitudinal_observation_recovers_mechanism_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    worlds_dir = tmp_path / "worlds"
    worlds_dir.mkdir()
    trial = _trial("TRIAL-A", "a")
    trial_b = _trial("TRIAL-B", "b")
    cells = [
        _cell("complete_data", "none", None, 0.0),
        _cell("dropout_independent", "lagged_outcome", -2.5, 0.0),
        _cell("dropout_lower", "lagged_outcome", -2.5, 0.5),
        _cell("dropout_reference", "lagged_outcome", -2.5, 1.0),
        _cell("dropout_higher", "lagged_outcome", -2.5, 1.5),
    ]
    design = {
        "schema_id": "trialagentbench.longitudinal_observation_design/v1",
        "seed": 451,
        "trials": [
            {
                "qualification": trial,
                "predictor_center": 10.0,
                "predictor_scale": 1.0,
            },
            {
                "qualification": trial_b,
                "predictor_center": 10.0,
                "predictor_scale": 1.0,
            },
        ],
        "cells": cells,
    }
    design_sha256 = _json_sha(design)
    receipts = []
    for trial_id in ("TRIAL-A", "TRIAL-B"):
        for cell in cells:
            for world_index in range(2):
                cell_id = cast(str, cell["cell_id"])
                coefficient = cast(float, cell["lagged_outcome_coefficient"])
                mechanism = cast(str, cell["mechanism"])
                seed = _world_seed(451, trial_id, world_index)
                world_id = _world_id(design_sha256, trial_id, cell_id, world_index)
                frame = _world(
                    seed=seed,
                    coefficient=coefficient,
                    dropout=mechanism != "none",
                )
                frame.insert(0, "cell_id", cell_id)
                frame.insert(0, "trial_id", trial_id)
                frame.insert(0, "world_id", world_id)
                path = worlds_dir / f"{world_id}.parquet"
                frame.to_parquet(path, index=False)
                receipts.append(
                    {
                        "world_id": world_id,
                        "trial_id": trial_id,
                        "cell_id": cell_id,
                        "world_index": world_index,
                        "seed": seed,
                        "subjects": 80,
                        "dropout_events": int(
                            frame.groupby("participant_id", sort=False)["observed"]
                            .min()
                            .eq(False)
                            .sum()
                        ),
                        "analysis_path": f"worlds/{world_id}.parquet",
                        "analysis_sha256": sha256_file(path),
                        "generator_spec_sha256": "c" * 64,
                        "ground_truth_manifest_sha256": "d" * 64,
                        "generated_bundle_sha256": "e" * 64,
                    }
                )
    receipt = {
        "schema_id": "trialagentbench.longitudinal_observation_receipt/v1",
        "design_sha256": design_sha256,
        "worlds": receipts,
    }
    (tmp_path / "design.json").write_text(json.dumps(design), encoding="utf-8")
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "longitudinal-observation",
            "--release-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--minimum-worlds-per-trial-cell",
            "2",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(report["dropout_response"]) == 2
    assert all(row["mean_slope"] > 0 for row in report["dropout_response"])
    assert all(
        np.isfinite(
            [
                row["mean_probability_residual"],
                row["mean_predictor_weighted_probability_residual"],
            ]
        ).all()
        for row in report["dropout_cells"]
    )
    assert {row["route"] for row in report["treatment_estimates"]} == {
        "available_case",
        "estimated_ipcw",
        "oracle_ipcw",
    }
    assert all(row["failures"] == 0 for row in report["treatment_cells"])
    assert len(report["route_contrasts"]) == 16
    assert {row["correction_route"] for row in report["route_contrasts"]} == {
        "estimated_ipcw",
        "oracle_ipcw",
    }

    tampered = tmp_path / receipts[0]["analysis_path"]
    frame = pd.read_parquet(tampered)
    frame.at[0, "value"] = cast(float, frame.at[0, "value"]) + 1.0
    frame.to_parquet(tampered, index=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        main(
            [
                "longitudinal-observation",
                "--release-dir",
                str(tmp_path),
                "--output",
                str(tmp_path / "tampered.json"),
                "--minimum-worlds-per-trial-cell",
                "2",
            ]
        )


def _trial(trial_id: str, checksum_prefix: str) -> dict[str, object]:
    source = _complete_panel(seed=451)
    fingerprint = fingerprint_longitudinal_trial(
        source,
        trial_id=trial_id,
        source="fixture",
        measurement="score",
        measurement_unit="points",
        time_unit="day",
    )
    covariance = [
        [1.0, 0.7, 0.5, 0.3],
        [0.7, 1.0, 0.7, 0.5],
        [0.5, 0.7, 1.0, 0.7],
        [0.3, 0.5, 0.7, 1.0],
    ]
    return {
        "trial_id": trial_id,
        "source_sha256": checksum_prefix * 64,
        "participants": 80,
        "worlds": 2,
        "time_values": [0.0, 1.0, 2.0, 3.0],
        "arm_ids": ["control", "active"],
        "control_arm_id": "control",
        "fitted_model": {
            "control_mean_values": [10.0, 10.1, 10.2, 10.3],
            "source_covariance": covariance,
            "latent_correlation": covariance,
            "measurement_probabilities": [1.0, 1.0, 1.0, 1.0],
            "marginal_probabilities": [0.1, 0.25, 0.5, 0.75, 0.9],
            "marginal_residual_values": [[-2.0, -1.0, 0.0, 1.0, 2.0]] * 4,
            "arm_effects": [
                {
                    "arm_id": "active",
                    "visit_shifts": [0.0, -0.2, -0.4, -0.6],
                }
            ],
        },
        "source_fingerprint": fingerprint.model_dump(mode="json"),
    }


def _cell(
    cell_id: str,
    mechanism: str,
    intercept: float | None,
    coefficient: float,
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "mechanism": mechanism,
        "logit_intercept": intercept,
        "lagged_outcome_coefficient": coefficient,
        "worlds_per_trial": 2,
        "sample_size_multiplier": 1.0,
    }


def _complete_panel(seed: int) -> pd.DataFrame:
    return _world(seed=seed, coefficient=0.0, dropout=False).drop(columns=["observed"])


def _world(*, seed: int, coefficient: float, dropout: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for participant in range(80):
        arm = "control" if participant < 40 else "active"
        natural = 10.0 + rng.normal()
        previous_natural = natural
        observed = True
        for time in (0.0, 1.0, 2.0, 3.0):
            natural_value = natural + 0.1 * time + rng.normal(scale=0.35)
            if time > 0 and observed and dropout:
                probability = 1.0 / (
                    1.0 + np.exp(-(-2.5 + coefficient * (previous_natural - 10.0)))
                )
                observed = bool(rng.random() >= probability)
            value = (
                natural_value - (0.2 * time if arm == "active" else 0.0)
                if observed
                else np.nan
            )
            rows.append(
                {
                    "participant_id": f"P{participant:03d}",
                    "arm": arm,
                    "time": time,
                    "value": value,
                    "observed": observed,
                }
            )
            previous_natural = natural_value
    return pd.DataFrame(rows)


def _json_sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    return f"world_{digest[:20]}"
