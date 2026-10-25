"""Shared figure language for the worked demonstrations: one palette, one type scale, one export path.

Colours carry a fixed meaning across every demonstration and always come with a second cue (marker
shape, line style or a direct label), so the figures survive greyscale printing and colour-vision
differences. Every figure is 7.2 inches wide; the type scale keeps the smallest text at 7.5 pt, about 10 px
where the documentation and site show a figure at its designed size.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from trialagentbench.figure_style import (  # noqa: E402
    ALTERNATIVE as ALTERNATIVE,
)
from trialagentbench.figure_style import (
    CAUTION as CAUTION,
)
from trialagentbench.figure_style import (
    CREDITED as CREDITED,
)
from trialagentbench.figure_style import (
    DEMONSTRATION_STYLE,
)
from trialagentbench.figure_style import (
    DISTORTED as DISTORTED,
)
from trialagentbench.figure_style import (
    GRID as GRID,
)
from trialagentbench.figure_style import (
    INK as INK,
)
from trialagentbench.figure_style import (
    MUTED as MUTED,
)
from trialagentbench.figure_style import (
    NOTE_SIZE as NOTE_SIZE,
)
from trialagentbench.figure_style import (
    PASSING as PASSING,
)
from trialagentbench.figure_style import (
    WIDTH as WIDTH,
)

plt.rcParams.update(DEMONSTRATION_STYLE)


def add_note(figure: Figure, text: str, *, x: float = 0.01, ha: str = "left") -> None:
    """Write a figure's reading note (what labels, bars or intervals mean) along its bottom edge.

    Parameters
    ----------
    figure
        The figure.
    text
        The note; line breaks are kept.
    x
        Horizontal position in figure coordinates.
    ha
        Horizontal alignment of the note at ``x``.
    """
    figure.text(x, 0.015, text, ha=ha, va="bottom", fontsize=NOTE_SIZE, color=MUTED, linespacing=1.3)


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
    # Without trailing spaces the SVG is byte-identical to the committed copy, which the repository hooks strip.
    svg = paths[0].read_text(encoding="utf-8").splitlines()
    paths[0].write_text("\n".join(line.rstrip() for line in svg) + "\n", encoding="utf-8")
    figure.savefig(paths[1])
    data.to_csv(paths[2], index=False, float_format="%.6g")
    plt.close(figure)
    return paths
