"""Independent repeated-trial analysis reliability tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from trialagentbench_validation.cli import main
from trialagentbench_validation.trialeval.analysis_reliability import (
    _bias_interval,
    verify_analysis_reliability,
)
from trialagentbench_validation.validation_figures.report import (
    _analysis_reliability_figure,
)


def _spec() -> dict[str, object]:
    return {
        "qualification_id": "TE-S02-A3:censoring_ignorability:interval_coverage",
        "regime_cell_id": "TE-S02-A3",
        "assumption_id": "censoring_ignorability",
        "comparison_mode": "interval_coverage",
        "default_estimator_id": "observed:km_rmst_tau",
        "corrected_estimator_id": "observed:km_ipcw_rmst_tau",
    }


def _world(
    *,
    world_id: str,
    default_estimate: float,
    corrected_estimate: float,
    default_interval: tuple[float, float],
    corrected_interval: tuple[float, float],
) -> dict[str, object]:
    reference = 1.0
    consequence = (
        not (default_interval[0] <= reference <= default_interval[1])
        and corrected_interval[0] <= reference <= corrected_interval[1]
    )
    return {
        "world_id": world_id,
        "practical_consequence_records": [
            {
                "qualification_spec": _spec(),
                "status": "success",
                "reference_value": reference,
                "default_estimate": default_estimate,
                "corrected_estimate": corrected_estimate,
                "default_interval_low": default_interval[0],
                "default_interval_high": default_interval[1],
                "corrected_interval_low": corrected_interval[0],
                "corrected_interval_high": corrected_interval[1],
                "consequence_observed": consequence,
            }
        ],
    }


def _write_inputs(
    root: Path, *, reported_default_coverage: float = 0.5
) -> tuple[Path, Path]:
    worlds = (
        _world(
            world_id="w1",
            default_estimate=0.8,
            corrected_estimate=0.95,
            default_interval=(0.70, 0.90),
            corrected_interval=(0.85, 1.05),
        ),
        _world(
            world_id="w2",
            default_estimate=1.1,
            corrected_estimate=1.0,
            default_interval=(0.95, 1.25),
            corrected_interval=(0.90, 1.10),
        ),
    )
    worlds_path = root / "world_records.jsonl"
    worlds_path.write_text(
        "".join(json.dumps(world, sort_keys=True) + "\n" for world in worlds),
        encoding="utf-8",
    )
    default_rmse = (0.025) ** 0.5
    corrected_rmse = (0.00125) ** 0.5
    operating = {
        "practical_consequences": [
            {
                "qualification_spec": _spec(),
                "n_worlds": 2,
                "n_success": 2,
                "n_consequences": 1,
                "failure_rate": 0.0,
                "paired_recovery_rate": 0.5,
                "default_coverage": reported_default_coverage,
                "corrected_coverage": 1.0,
                "default_rmse": default_rmse,
                "corrected_rmse": corrected_rmse,
                "relative_rmse_reduction": (default_rmse - corrected_rmse)
                / default_rmse,
                "passed": True,
            }
        ]
    }
    operating_path = root / "operating_characteristics.json"
    operating_path.write_text(json.dumps(operating), encoding="utf-8")
    return worlds_path, operating_path


def test_analysis_reliability_recomputes_coverage_and_rmse(tmp_path: Path) -> None:
    worlds, operating = _write_inputs(tmp_path)

    result = verify_analysis_reliability(
        world_records_path=worlds,
        operating_characteristics_path=operating,
    )[0]

    assert result.series_id == "TE-S02"
    assert result.independent_trials == 2
    assert result.fit_failures == 0
    assert result.default_coverage == 0.5
    assert result.qualified_coverage == 1.0
    assert result.default_rmse == pytest.approx(0.025**0.5)
    assert result.qualified_rmse == pytest.approx(0.00125**0.5)
    assert result.default_bias == pytest.approx(-0.05)
    assert result.default_bias_low <= result.default_bias <= result.default_bias_high
    assert result.qualified_bias == pytest.approx(-0.025)
    assert (
        result.qualified_bias_low <= result.qualified_bias <= result.qualified_bias_high
    )
    assert result.qualified_to_default_rmse_ratio == pytest.approx(
        (0.00125 / 0.025) ** 0.5
    )
    assert (
        result.rmse_ratio_low
        <= result.qualified_to_default_rmse_ratio
        <= result.rmse_ratio_high
    )
    assert (
        result.paired_recovery_rate_low
        <= result.paired_recovery_rate
        <= result.paired_recovery_rate_high
    )
    assert result.paired_loss_rate == 0.0
    assert (
        result.paired_loss_rate_low
        <= result.paired_loss_rate
        <= result.paired_loss_rate_high
    )
    assert result.fit_failure_rate == 0.0
    assert (
        result.fit_failure_rate_low
        <= result.fit_failure_rate
        <= result.fit_failure_rate_high
    )


def test_analysis_reliability_rejects_summary_drift(tmp_path: Path) -> None:
    worlds, operating = _write_inputs(tmp_path, reported_default_coverage=1.0)

    with pytest.raises(ValueError, match="summary disagrees"):
        verify_analysis_reliability(
            world_records_path=worlds,
            operating_characteristics_path=operating,
        )


def test_analysis_reliability_rejects_inconsistent_recovery_flag(
    tmp_path: Path,
) -> None:
    worlds, operating = _write_inputs(tmp_path)
    payloads = [
        json.loads(line) for line in worlds.read_text(encoding="utf-8").splitlines()
    ]
    payloads[0]["practical_consequence_records"][0]["consequence_observed"] = False
    worlds.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recovery status disagrees"):
        verify_analysis_reliability(
            world_records_path=worlds,
            operating_characteristics_path=operating,
        )


def test_bias_interval_handles_identical_trial_errors() -> None:
    point, low, high = _bias_interval(np.full(12, 0.04, dtype=np.float64))

    assert point == pytest.approx(0.04)
    assert low == pytest.approx(point)
    assert high == pytest.approx(point)


def test_analysis_reliability_cli_writes_tidy_results(tmp_path: Path) -> None:
    worlds, operating = _write_inputs(tmp_path)
    output = tmp_path / "analysis_reliability.csv"

    exit_code = main(
        [
            "trialeval-analysis-reliability",
            "--world-records",
            str(worlds),
            "--operating-characteristics",
            str(operating),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert (
        output.read_text(encoding="utf-8")
        .splitlines()[0]
        .startswith("series_id,regime_cell_id,assumption")
    )


def test_analysis_reliability_figure_uses_native_uncertainty(tmp_path: Path) -> None:
    data = tmp_path / "data"
    figures = tmp_path / "figures"
    data.mkdir()
    figures.mkdir()
    rows = []
    for index, series_id in enumerate(
        ("TE-S01", "TE-S02", "TE-S04", "TE-S05", "TE-S06", "TE-S07")
    ):
        rows.append(
            {
                "series_id": series_id,
                "independent_trials": 278,
                "default_coverage": 0.70 + index * 0.02,
                "default_coverage_low": 0.65 + index * 0.02,
                "default_coverage_high": 0.75 + index * 0.02,
                "qualified_coverage": 0.95,
                "qualified_coverage_low": 0.92,
                "qualified_coverage_high": 0.97,
                "qualified_to_default_rmse_ratio": 0.5 + index * 0.05,
                "rmse_ratio_low": 0.4 + index * 0.05,
                "rmse_ratio_high": 0.6 + index * 0.05,
                "paired_recovery_rate": 0.25 - index * 0.02,
                "paired_recovery_rate_low": 0.20 - index * 0.02,
                "paired_recovery_rate_high": 0.30 - index * 0.02,
                "paired_loss_rate": 0.02,
                "paired_loss_rate_low": 0.005,
                "paired_loss_rate_high": 0.05,
                "fit_failure_rate": 0.01,
                "fit_failure_rate_low": 0.0,
                "fit_failure_rate_high": 0.03,
            }
        )
    pd.DataFrame(rows).to_csv(data / "analysis_reliability.csv", index=False)

    figure = _analysis_reliability_figure(figures)

    assert len(figure.axes) == 3
    assert figure.axes[1].get_yscale() == "log"
    assert figure.axes[0].get_ylabel() == "Coverage probability"
    plt.close(figure)
