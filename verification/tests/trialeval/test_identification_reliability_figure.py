"""A4 reliability figure tests."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from trialagentbench_validation.validation_figures.report import (
    _identification_reliability_figure,
)


def test_identification_reliability_figure_separates_coverage_width_and_execution(
    tmp_path: Path,
) -> None:
    """A4 coverage cannot conceal vacuous ranges or inactive monitoring."""

    data = tmp_path / "data"
    figures = tmp_path / "figures"
    data.mkdir()
    figures.mkdir()
    rows = []
    for series_id, conclusion, role, width, parameter, early_stop in (
        (
            "TE-S04",
            "identified_range",
            "prespecified_bounded_departure",
            0.18,
            0.2,
            None,
        ),
        ("TE-S04", "identified_range", "unrestricted_worst_case", 1.20, None, None),
        (
            "TE-S06",
            "identified_range",
            "prespecified_bounded_departure",
            0.12,
            0.2,
            None,
        ),
        ("TE-S06", "identified_range", "unrestricted_worst_case", 1.10, None, None),
        ("TE-S09", "repeated_interval", "repeated_monitoring", 0.08, None, 0.35),
    ):
        rows.append(
            {
                "series_id": series_id,
                "conclusion_type": conclusion,
                "analysis_role": role,
                "effect_scale": "risk_difference_tau",
                "sensitivity_parameter": parameter,
                "sensitivity_parameter_unit": (
                    "risk_probability_difference" if parameter is not None else None
                ),
                "independent_trials": 278,
                "coverage": 0.95,
                "coverage_low": 0.92,
                "coverage_high": 0.97,
                "mean_width": width,
                "mean_width_low": width - 0.01,
                "mean_width_high": width + 0.01,
                "fit_failure_rate": 0.01,
                "fit_failure_rate_low": 0.003,
                "fit_failure_rate_high": 0.03,
                "early_stop_rate": early_stop,
                "early_stop_rate_low": None if early_stop is None else 0.30,
                "early_stop_rate_high": None if early_stop is None else 0.41,
                "bias": None if series_id != "TE-S09" else 0.002,
                "bias_low": None if series_id != "TE-S09" else -0.003,
                "bias_high": None if series_id != "TE-S09" else 0.007,
                "rmse": None if series_id != "TE-S09" else 0.025,
                "rmse_low": None if series_id != "TE-S09" else 0.020,
                "rmse_high": None if series_id != "TE-S09" else 0.031,
            }
        )
    pd.DataFrame(rows).to_csv(data / "identification_reliability.csv", index=False)

    figure = _identification_reliability_figure(figures)

    assert len(figure.axes) == 4
    assert figure.axes[0].get_ylabel() == "Trial proportion"
    assert "(0-1 scale)" in figure.axes[1].get_ylabel()
    assert figure.axes[2].get_title() == "Execution"
    assert len(figure.axes[2].get_legend().get_texts()) == 2
    assert figure.axes[3].get_title() == "Sequential estimation error"
    assert "(0-1 scale)" in figure.axes[3].get_ylabel()
    plt.close(figure)
