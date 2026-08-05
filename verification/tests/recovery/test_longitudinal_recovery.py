"""Independent longitudinal qualification release tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trialagentbench_validation.cli import main
from trialagentbench_validation.external.realism.longitudinal import (
    fingerprint_longitudinal_trial,
)
from trialagentbench_validation.io import sha256_file

_MODES = {
    "independent_marginal": 0.0,
    "linkage_25": 0.25,
    "linkage_50": 0.5,
    "linkage_75": 0.75,
    "source_anchored": 1.0,
}


def test_longitudinal_qualification_recovers_linkage_dose_response(
    tmp_path: Path,
) -> None:
    worlds_dir = tmp_path / "worlds"
    worlds_dir.mkdir()
    source = _panel(seed=451, retention=1.0)
    fingerprint = fingerprint_longitudinal_trial(
        source,
        trial_id="TRIAL-A",
        source="fixture",
        measurement="score",
        measurement_unit="points",
        time_unit="day",
    )
    trial = {
        "trial_id": "TRIAL-A",
        "source_sha256": "a" * 64,
        "participants": 24,
        "worlds": 2,
        "time_values": [0.0, 1.0, 2.0, 3.0],
        "arm_ids": ["control", "active"],
        "control_arm_id": "control",
        "fitted_model": {
            "control_mean_values": [10.0, 10.2, 10.4, 10.6],
            "source_covariance": [
                [1.0, 0.8, 0.6, 0.4],
                [0.8, 1.0, 0.8, 0.6],
                [0.6, 0.8, 1.0, 0.8],
                [0.4, 0.6, 0.8, 1.0],
            ],
            "latent_correlation": [
                [1.0, 0.8, 0.6, 0.4],
                [0.8, 1.0, 0.8, 0.6],
                [0.6, 0.8, 1.0, 0.8],
                [0.4, 0.6, 0.8, 1.0],
            ],
            "marginal_probabilities": [0.1, 0.25, 0.5, 0.75, 0.9],
            "marginal_residual_values": [[-2.0, -1.0, 0.0, 1.0, 2.0]] * 4,
            "arm_effects": [
                {"arm_id": "active", "visit_shifts": [0.0, -0.2, -0.4, -0.6]}
            ],
            "measurement_probabilities": [1.0, 1.0, 1.0, 1.0],
        },
        "source_fingerprint": fingerprint.model_copy(
            update={"trial_id": "TRIAL-A"}
        ).model_dump(mode="json"),
    }
    trial_b = json.loads(json.dumps(trial))
    trial_b["trial_id"] = "TRIAL-B"
    trial_b["source_sha256"] = "b" * 64
    trial_b["source_fingerprint"]["trial_id"] = "TRIAL-B"
    design = {
        "schema_id": "trialagentbench.longitudinal_qualification_design/v1",
        "seed": 451,
        "trials": [trial, trial_b],
    }
    design_sha256 = _json_sha(design)
    worlds = []
    for trial_id in ("TRIAL-A", "TRIAL-B"):
        for world_index in range(2):
            seed = _world_seed(451, trial_id, world_index)
            world_id = _world_id(design_sha256, trial_id, world_index)
            frames = [
                _panel(seed=seed, retention=retention).assign(mode=mode)
                for mode, retention in _MODES.items()
            ]
            frames.append(_panel(seed=seed, retention=1.0).assign(mode="whole_subject"))
            frame = pd.concat(frames, ignore_index=True)
            frame.insert(0, "trial_id", trial_id)
            frame.insert(0, "world_id", world_id)
            path = worlds_dir / f"{world_id}.parquet"
            frame.to_parquet(path, index=False)
            worlds.append(
                {
                    "world_id": world_id,
                    "trial_id": trial_id,
                    "world_index": world_index,
                    "seed": seed,
                    "analysis_path": f"worlds/{world_id}.parquet",
                    "analysis_sha256": sha256_file(path),
                    "generator_spec_sha256": "c" * 64,
                    "ground_truth_manifest_sha256": "d" * 64,
                    "generated_bundle_sha256": "e" * 64,
                }
            )
    receipt = {
        "schema_id": "trialagentbench.longitudinal_qualification_receipt/v1",
        "design_sha256": design_sha256,
        "worlds": worlds,
    }
    (tmp_path / "design.json").write_text(json.dumps(design), encoding="utf-8")
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "longitudinal-validation",
            "--release-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--minimum-worlds-per-trial",
            "2",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(report["linkage_dose_response"]) == 2
    assert all(row["mean_slope"] > 0.0 for row in report["linkage_dose_response"])
    assert all(
        row["endpoint_contrast_mean"] > 0.0 for row in report["linkage_dose_response"]
    )


def _panel(*, seed: int, retention: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for participant in range(24):
        arm = "control" if participant < 12 else "active"
        subject_effect = np.sqrt(retention) * rng.normal()
        for time in (0.0, 1.0, 2.0, 3.0):
            rows.append(
                {
                    "participant_id": f"P{participant:03d}",
                    "arm": arm,
                    "time": time,
                    "value": 10.0
                    + 0.2 * time
                    - (0.2 * time if arm == "active" else 0.0)
                    + subject_effect
                    + np.sqrt(1.0 - 0.7 * retention) * rng.normal(),
                }
            )
    return pd.DataFrame(rows)


def _json_sha(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _world_seed(seed: int, trial_id: str, world_index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{trial_id}:{world_index}".encode()).digest()[:4], "big"
    )


def _world_id(design_sha256: str, trial_id: str, world_index: int) -> str:
    digest = hashlib.sha256(
        f"{design_sha256}:{trial_id}:{world_index}".encode()
    ).hexdigest()
    return f"world_{digest[:20]}"
