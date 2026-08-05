"""Longitudinal trial fingerprint tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trialagentbench_validation.cli import main
from trialagentbench_validation.external.realism.longitudinal import (
    LongitudinalTrialFingerprintV1,
    fingerprint_longitudinal_trial,
)


def _panel() -> pd.DataFrame:
    rows = []
    for participant in range(24):
        arm = "active" if participant % 2 else "control"
        baseline = float(participant)
        for time in (0.0, 1.0, 3.0, 7.0):
            rows.append(
                {
                    "participant_id": f"P{participant:03d}",
                    "arm": arm,
                    "time": time,
                    "value": baseline + 0.5 * time,
                }
            )
    return pd.DataFrame(rows)


def test_longitudinal_fingerprint_preserves_panel_denominators() -> None:
    result = fingerprint_longitudinal_trial(
        _panel(),
        trial_id="TRIAL-1",
        source="public_trial",
        measurement="test outcome",
        measurement_unit="points",
        time_unit="day",
    )

    assert result.participants == 24
    assert result.arms == 2
    assert result.timepoints == 4
    assert result.observation_fraction == 1.0
    assert result.observed_timepoints_mean == 4.0
    assert result.followup_mean == 7.0
    assert result.adjacent_measurement_correlation > 0.99
    assert result.baseline_final_correlation == pytest.approx(1.0)
    assert result.baseline_final_change_mean == pytest.approx(3.5)


def test_longitudinal_fingerprint_rejects_duplicate_and_degenerate_panels() -> None:
    duplicated = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="at most one row"):
        fingerprint_longitudinal_trial(
            duplicated,
            trial_id="TRIAL-1",
            source="public_trial",
            measurement="test outcome",
            measurement_unit="points",
            time_unit="day",
        )


def test_longitudinal_fingerprint_cli_reads_standard_csv(tmp_path) -> None:
    input_path = tmp_path / "panel.csv"
    output_path = tmp_path / "fingerprint.json"
    _panel().to_csv(input_path, index=False)

    exit_code = main(
        [
            "longitudinal-fingerprint",
            "--input",
            str(input_path),
            "--trial-id",
            "TRIAL-1",
            "--source",
            "public_trial",
            "--measurement",
            "test outcome",
            "--measurement-unit",
            "points",
            "--time-unit",
            "day",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    result = LongitudinalTrialFingerprintV1.model_validate_json(output_path.read_text())
    assert result.participants == 24
    assert result.observation_fraction == 1.0

    constant = _panel()
    constant["value"] = np.where(constant["time"].eq(0), 1.0, 2.0)
    with pytest.raises(ValueError, match="measurements must vary"):
        fingerprint_longitudinal_trial(
            constant,
            trial_id="TRIAL-1",
            source="public_trial",
            measurement="test outcome",
            measurement_unit="points",
            time_unit="day",
        )
