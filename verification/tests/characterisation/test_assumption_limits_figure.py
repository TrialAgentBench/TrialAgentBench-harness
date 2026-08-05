"""Checks for the A4 identified-set response figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from trialagentbench_validation.validation_figures.report import (
    _assumption_limits_figure,
)


def test_assumption_limits_figure_shows_the_full_departure_response(
    tmp_path: Path,
) -> None:
    """Every partially identified trial contributes a monotone sensitivity curve."""

    data = tmp_path / "data"
    figures = tmp_path / "figures"
    data.mkdir()
    figures.mkdir()
    identified = []
    for series_id, multiplier in (("TE-S04", 0.8), ("TE-S06", 1.0)):
        for replicate_index in range(1, 5):
            centre = -0.01 + 0.002 * replicate_index
            for delta in (0.05, 0.10, 0.20):
                half_width = multiplier * delta
                identified.append(
                    {
                        "task_id": f"TASK{replicate_index:032X}",
                        "series_id": series_id,
                        "replicate_index": replicate_index,
                        "assumption": "bounded_deviation",
                        "sensitivity_parameter": delta,
                        "lower": centre - half_width,
                        "upper": centre + half_width,
                        "width": 2.0 * half_width,
                        "result_unit": "risk difference",
                    }
                )
            identified.append(
                {
                    "task_id": f"TASK{replicate_index:032X}",
                    "series_id": series_id,
                    "replicate_index": replicate_index,
                    "assumption": "worst_case",
                    "sensitivity_parameter": None,
                    "lower": centre - 0.5,
                    "upper": centre + 0.5,
                    "width": 1.0,
                    "result_unit": "risk difference",
                }
            )
    pd.DataFrame(identified).to_csv(
        data / "assumption_identification_results.csv", index=False
    )
    sequential = [
        {
            "series_id": "TE-S09",
            "replicate_index": replicate_index,
            "assumption_tier": "A4",
            "default_status": "incompatible",
            "qualified_shape": "point",
            "qualified_value": -0.02,
            "qualified_interval_low": -0.04,
            "qualified_interval_high": 0.0,
            "result_unit": "risk difference",
        }
        for replicate_index in range(1, 5)
    ]
    pd.DataFrame(sequential).to_csv(data / "assumption_bridges.csv", index=False)

    figure = _assumption_limits_figure(figures)

    assert len(figure.axes) == 4
    assert figure.axes[0].get_xticklabels()[-1].get_text() == "No bound"
    assert (
        figure.axes[0].get_xlabel()
        == "Maximum event-risk departure (percentage points)"
    )
    assert "(0-1 scale)" in figure.axes[0].get_ylabel()
    assert (
        figure.axes[2].get_xlabel()
        == "Treated - control event-risk difference (0-1 scale)"
    )
    assert figure.axes[3].get_title() == "Sequential monitoring"
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "Individual trial",
        "Mean loss-to-follow-up response",
        "Mean endpoint-validation response",
        "Repeated 95% interval",
    ]
    plt.close(figure)
