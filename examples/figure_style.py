"""Shared figure language for the worked demonstrations: one palette, one type scale, one export path.

Colours carry a fixed meaning across every demonstration and always come with a second cue (marker
shape, line style or a direct label), so the figures survive greyscale printing and colour-vision
differences.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

INK = "#0A0E14"
MUTED = "#5B636B"
GRID = "#E3E5EA"
AUDIT = "#0072B2"
"""Audit-corrected (validated-event discrepancy) analysis; circle marker."""
ROUTINE = "#B54823"
"""Routine-record analysis; square marker."""
JOINT = "#008269"
"""Joint likelihood over routine and adjudicated records; triangle marker."""
CAUTION = "#AD7800"
"""Thresholds and data-blind policies."""

MARKERS = {"routine": "s", "audit": "o", "joint": "^"}
COLOURS = {"routine": ROUTINE, "audit": AUDIT, "joint": JOINT}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "savefig.dpi": 200,
        "text.color": INK,
    }
)


def save_figure(figure: Figure, data: pd.DataFrame, output: Path, name: str) -> tuple[Path, ...]:
    """Write the figure as SVG and PNG with its exact plotted values as CSV.

    Parameters
    ----------
    figure
        The finished figure.
    data
        The values the figure draws, one row per plotted mark.
    output
        Destination directory, created if absent.
    name
        File stem shared by the three outputs.

    Returns
    -------
    tuple of Path
        The SVG, PNG and CSV paths.
    """
    output.mkdir(parents=True, exist_ok=True)
    paths = (output / f"{name}.svg", output / f"{name}.png", output / f"{name}.csv")
    figure.savefig(paths[0], metadata={"Date": None})
    figure.savefig(paths[1])
    data.to_csv(paths[2], index=False, float_format="%.6g")
    plt.close(figure)
    return paths
