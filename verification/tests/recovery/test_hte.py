"""Public treatment-effect heterogeneity verifier tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.cli import main
from trialagentbench_validation.io import sha256_file


def test_hte_qualification_recovers_dose_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    worlds_dir = tmp_path / "worlds"
    worlds_dir.mkdir()
    cells = [
        {
            "cell_id": f"interaction_{scale:g}__n_{multiplier:g}",
            "interaction_scale": scale,
            "sample_size_multiplier": multiplier,
            "worlds_per_trial": 2,
        }
        for multiplier in (1.0, 4.0)
        for scale in (0.0, 0.5, 1.0, 1.5)
    ]
    trials = [_trial(f"RCTBENCH-{index:03d}") for index in (1, 2, 3)]
    design = {
        "schema_id": "trialagentbench.hte_qualification_design/v1",
        "source_revision": "a" * 40,
        "seed": 451,
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
                subjects = int(80 * cell["sample_size_multiplier"])
                frame = _world(
                    seed=seed,
                    subjects=subjects,
                    interaction_scale=cell["interaction_scale"],
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
        "schema_id": "trialagentbench.hte_qualification_receipt/v1",
        "design_sha256": design_sha256,
        "worlds": receipts,
    }
    (tmp_path / "design.json").write_text(json.dumps(design), encoding="utf-8")
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "hte-validation",
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
    assert len(report["estimates"]) == 48
    assert len(report["dose_responses"]) == 6
    assert len(report["information_responses"]) == 12
    assert all(row["failures"] == 0 for row in report["cell_summaries"])
    assert all(abs(row["mean_slope"] - 1.0) < 1e-10 for row in report["dose_responses"])
    assert (
        max(
            row["maximum_point_estimator_crosscheck_difference"]
            for row in report["cell_summaries"]
        )
        < 1e-10
    )

    tampered = tmp_path / receipts[0]["analysis_path"]
    frame = pd.read_parquet(tampered)
    frame.at[0, "outcome"] += 1.0
    frame.to_parquet(tampered, index=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        main(
            [
                "hte-validation",
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


def _trial(trial_id: str) -> dict[str, object]:
    fitted = {
        "outcome_kind": "continuous",
        "source_subjects": 80,
        "source_control_subjects": 40,
        "source_active_subjects": 40,
        "source_event_rate": None,
        "active_source_level": "active",
        "intercept": 1.0,
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
    }
    return {
        "qualification": {
            "trial_id": trial_id,
            "source_data_sha256": "d" * 64,
            "source_dictionary_sha256": "e" * 64,
            "worlds": 2,
            "fitted_analysis": fitted,
        },
        "age_standard_deviation": 10.0,
        "base_standardized_interaction": 1.0,
    }


def _world(
    *,
    seed: int,
    subjects: int,
    interaction_scale: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(50.0, 10.0, size=subjects)
    bmi = rng.normal(25.0, 4.0, size=subjects)
    treatment = np.resize(np.asarray([0.0, 1.0]), subjects)
    noise = rng.normal(scale=0.5, size=subjects)
    outcome = (
        1.0
        + 0.1 * (age - 50.0)
        + 0.2 * (bmi - 25.0)
        + 0.5 * treatment
        + interaction_scale / 10.0 * treatment * (age - 50.0)
        + noise
    )
    return pd.DataFrame(
        {
            "participant_id": [f"P{index:04d}" for index in range(subjects)],
            "arm": np.where(treatment == 1.0, "active", "control"),
            "age": age,
            "bmi": bmi,
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
    return f"hte-{digest[:20]}"
