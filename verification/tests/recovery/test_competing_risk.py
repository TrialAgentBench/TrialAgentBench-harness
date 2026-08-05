"""Public competing-risk verifier tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.cli import main
from trialagentbench_validation.io import sha256_file


def test_competing_risk_qualification_replays_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    worlds_dir = tmp_path / "worlds"
    worlds_dir.mkdir()
    cells = _cells()
    trials = [_trial(f"RCTBENCH-{index:03d}") for index in (1, 2, 3)]
    design = {
        "schema_id": "trialagentbench.competing_risk_design/v1",
        "source_revision": "a" * 40,
        "seed": 451,
        "followup_intervals": 4,
        "primary_intercept": -2.7,
        "competing_intercept": -3.0,
        "age_coefficient": 0.3,
        "bmi_coefficient": 0.2,
        "trials": trials,
        "cells": cells,
    }
    design_sha256 = _json_sha(design)
    receipts = []
    for trial in trials:
        qualification = cast(dict[str, object], trial["qualification"])
        trial_id = cast(str, qualification["trial_id"])
        for cell in cells:
            cell_id = cast(str, cell["cell_id"])
            primary = cast(float, cell["primary_treatment_coefficient"])
            competing = cast(float, cell["competing_treatment_coefficient"])
            multiplier = cast(float, cell["sample_size_multiplier"])
            for world_index in range(2):
                seed = _world_seed(451, trial_id, world_index)
                world_id = _world_id(
                    design_sha256,
                    trial_id,
                    cell_id,
                    world_index,
                )
                frame = _world(
                    seed=seed,
                    subjects=int(100 * multiplier),
                    primary=primary,
                    competing=competing,
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
                        "subjects": int(100 * multiplier),
                        "analysis_path": f"worlds/{world_id}.parquet",
                        "analysis_sha256": sha256_file(path),
                        "generator_spec_sha256": "b" * 64,
                        "generated_bundle_sha256": "c" * 64,
                    }
                )
    receipt = {
        "schema_id": "trialagentbench.competing_risk_receipt/v1",
        "design_sha256": design_sha256,
        "worlds": receipts,
    }
    (tmp_path / "design.json").write_text(json.dumps(design), encoding="utf-8")
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "competing-risk-validation",
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
    assert len(report["estimates"]) == 96
    assert len(report["cause_summaries"]) == 96
    assert len(report["dose_responses"]) == 24
    assert len(report["information_responses"]) == 48
    assert all(row["failures"] == 0 for row in report["cause_summaries"])
    assert (
        max(
            row["maximum_point_estimator_crosscheck_difference"]
            for row in report["cause_summaries"]
        )
        < 1e-6
    )

    tampered = tmp_path / receipts[0]["analysis_path"]
    frame = pd.read_parquet(tampered)
    frame.at[0, "age"] = cast(float, frame.at[0, "age"]) + 1.0
    frame.to_parquet(tampered, index=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        main(
            [
                "competing-risk-validation",
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


def _cells() -> list[dict[str, object]]:
    mechanisms = [
        {
            "mechanism_id": "joint_null",
            "primary_treatment_coefficient": 0.0,
            "competing_treatment_coefficient": 0.0,
            "worlds_per_trial": 2,
            "primary_response": True,
        },
        *[
            {
                "mechanism_id": f"primary_{value}",
                "primary_treatment_coefficient": value,
                "competing_treatment_coefficient": 0.0,
                "worlds_per_trial": 2,
                "primary_response": True,
                "competing_response": value == 0.8,
            }
            for value in (0.4, 0.8, 1.2)
        ],
        *[
            {
                "mechanism_id": f"competing_{value}",
                "primary_treatment_coefficient": 0.8,
                "competing_treatment_coefficient": value,
                "worlds_per_trial": 2,
                "competing_response": True,
            }
            for value in (-0.8, -0.4, 0.4, 0.8)
        ],
    ]
    return [
        {
            **mechanism,
            "cell_id": (f"{cast(str, mechanism['mechanism_id'])}__n_{multiplier:g}"),
            "sample_size_multiplier": multiplier,
        }
        for multiplier in (1.0, 4.0)
        for mechanism in mechanisms
    ]


def _trial(trial_id: str) -> dict[str, object]:
    return {
        "qualification": {
            "trial_id": trial_id,
            "source_data_sha256": "d" * 64,
            "source_dictionary_sha256": "e" * 64,
            "worlds": 2,
            "fitted_analysis": {
                "outcome_kind": "binary",
                "source_subjects": 100,
                "source_control_subjects": 50,
                "source_active_subjects": 50,
                "source_event_rate": 0.3,
                "active_source_level": "active",
                "intercept": -1.0,
                "treatment_coefficient": 0.5,
                "age_coefficient": 0.1,
                "bmi_coefficient": 0.2,
                "analysis_treatment_effect": 0.5,
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
        }
    }


def _world(
    *,
    seed: int,
    subjects: int,
    primary: float,
    competing: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(50.0, 10.0, subjects)
    bmi = rng.normal(25.0, 4.0, subjects)
    active = np.resize(np.asarray([0.0, 1.0]), subjects)
    event_time = np.full(subjects, 4, dtype=int)
    event_cause = np.full(subjects, "censored", dtype=object)
    at_risk = np.ones(subjects, dtype=bool)
    for interval in range(1, 5):
        linear_primary = -2.7 + primary * active + 0.3 * (age - age.mean()) / age.std()
        linear_competing = (
            -3.0 + competing * active + 0.2 * (bmi - bmi.mean()) / bmi.std()
        )
        weights = np.column_stack(
            (
                np.ones(subjects),
                np.exp(linear_primary),
                np.exp(linear_competing),
            )
        )
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        draws = np.asarray(
            [rng.choice(3, p=probabilities[index]) for index in range(subjects)]
        )
        for cause, label in ((1, "primary"), (2, "competing")):
            selected = at_risk & (draws == cause)
            event_time[selected] = interval
            event_cause[selected] = label
            at_risk[selected] = False
    return pd.DataFrame(
        {
            "participant_id": [f"P{index:04d}" for index in range(subjects)],
            "arm": np.where(active == 1.0, "active", "control"),
            "age": age,
            "bmi": bmi,
            "event_time": event_time,
            "event_cause": event_cause,
            "event": event_cause != "censored",
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
    return f"cr-{digest[:20]}"
