"""Worked demonstration of whether a data-blind strategy could score well on TrialEval.

The release-evidence record runs fixed answer policies over 1,200 prior-drawn worlds of every cell of both
TrialEval splits and counts the cells each policy fails, at the declared economics and at every point of the
disclosed sensitivity band (value of success and trial costs halved and doubled). This demonstration reads that
record only (no task data) and draws public failed-cell counts beside withheld rule outcomes only: whether each
policy fails at least one withheld cell.

Selection rule: every policy and every band point in the record's ``economics.band_failing_cells`` (public split)
and ``policy_withheld.fails_a_cell`` (withheld split). Nothing is omitted; no withheld failed-cell count is read
or published.

Run ``python release_panel.py --output results/release-panel`` (add ``--release-root`` for a local copy
of the dataset).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_style import DISTORTED, INK, PASSING, WIDTH, add_note, save_figure
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from release_access import PublicRelease, ReleaseEvidence, add_release_arguments

REFERENCE = "scripted_reference"
POLICY_LABELS = {
    REFERENCE: "Scripted reference (credited)",
    "always_robust": "Always robust (trap's correction)",
    "follow_plan": "Follow the plan",
    "always_naive": "Always naive (excluded analysis)",
    "plan_hedged_3x": "Plan estimate, interval ×3",
    "prior_only": "Prior only (ignores the data)",
    "null_point_plan_interval": "Null estimate, plan interval",
    "plan_overconfident_0.3x": "Plan estimate, interval ×0.3",
}
"""The policy names of the methods' policy table, with what each answers where the name alone does not say."""
FAILING = LinearSegmentedColormap.from_list("failing", ["#F0CDBF", DISTORTED])
"""One failed cell is the lightest tint, every cell the full failing colour; no failure is ``PASSING``."""
BAND_LABELS = {
    "declared": "Declared",
    "value x0.5": "Value\n×½",
    "value x2": "Value\n×2",
    "cost x0.5": "Costs\n×½",
    "cost x2": "Costs\n×2",
}
SPLITS = ("public", "withheld")


def policy_table(evidence: ReleaseEvidence) -> pd.DataFrame:
    """Return public failed-cell counts and withheld rule outcomes, one row per policy and band point.

    Parameters
    ----------
    evidence
        The release-evidence record.

    Returns
    -------
    pandas.DataFrame
        ``split``, ``policy`` and ``band``, with ``failing_cells`` and ``cells`` only for public rows and
        boolean ``fails_a_cell`` only for withheld rows.

    Raises
    ------
    ValueError
        If the public band's declared point differs from its policy panel, or the splits differ in band points
        or policies.
    """
    trialeval = evidence.trialeval
    public = trialeval.economics.band_failing_cells
    withheld = trialeval.policy_withheld.fails_a_cell
    panel = trialeval.policy_public
    if public.get("declared") != panel.failing_cells:
        raise ValueError("The public band's declared point differs from its policy panel")
    if set(public) != set(withheld):
        raise ValueError("The public and withheld bands differ in band points")
    for split, band in (("public", public), ("withheld", withheld)):
        if any(set(values) != set(panel.failing_cells) for values in band.values()):
            raise ValueError(f"A {split} band point omits a policy")
    rows = [
        {"split": "public", "policy": policy, "band": point, "failing_cells": count, "cells": panel.cells}
        for point, counts in public.items()
        for policy, count in counts.items()
    ] + [
        {"split": "withheld", "policy": policy, "band": point, "fails_a_cell": fails}
        for point, outcomes in withheld.items()
        for policy, fails in outcomes.items()
    ]
    return pd.DataFrame(rows).astype({"failing_cells": "Int64", "cells": "Int64", "fails_a_cell": "boolean"})


def _luminance(colour: tuple[float, float, float]) -> float:
    """Return the WCAG relative luminance of an RGB colour."""
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in colour]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _heatmap(axis: Axes, rows: pd.DataFrame, order: list[str], *, withheld: bool) -> None:
    grid = rows.pivot(index="policy", columns="band", values="fails_a_cell" if withheld else "failing_cells")
    grid = grid.loc[order, [b for b in BAND_LABELS if b in grid] + [b for b in grid.columns if b not in BAND_LABELS]]
    cells = 1 if withheld else int(rows["cells"].iloc[0])
    values = grid.to_numpy(dtype=int)
    # Any failure is tinted, so a policy failing one cell never reads as the reference's pass.
    fills = np.array(
        [
            [
                to_rgb(PASSING) if value == 0 else FAILING((value - 1) / (cells - 1) if cells > 1 else 1.0)[:3]
                for value in row
            ]
            for row in values
        ]
    )
    axis.imshow(fills, aspect="auto")
    ink = _luminance(to_rgb(INK))
    for (row, column), value in np.ndenumerate(values):
        background = _luminance(tuple(fills[row, column]))
        axis.text(
            column,
            row,
            ("fails" if value else "passes") if withheld else str(value),
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if 1.05 / (background + 0.05) > (background + 0.05) / (ink + 0.05) else INK,
            fontweight="bold" if grid.index[row] == REFERENCE else "normal",
        )
    axis.set_xticks(range(grid.shape[1]), [BAND_LABELS.get(b, b) for b in grid.columns])
    axis.set_yticks(
        range(grid.shape[0]), [POLICY_LABELS.get(p, p.replace("_", " ")) for p in grid.index] if not withheld else []
    )
    axis.tick_params(length=0)
    axis.xaxis.set_ticks_position("top")
    for spine in axis.spines.values():
        spine.set_visible(False)


def panel_figure(policies: pd.DataFrame) -> Figure:
    """Draw public failed-cell counts beside withheld fails/passes outcomes by policy and band point.

    Parameters
    ----------
    policies
        Output of ``policy_table``.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    declared = policies.loc[policies["split"].eq("public") & policies["band"].eq("declared")]
    order = list(declared.sort_values(["failing_cells", "policy"], kind="stable")["policy"])
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), gridspec_kw={"wspace": 0.06})
    for letter, axis, split in zip("ab", axes, SPLITS, strict=True):
        rows = policies.loc[policies["split"].eq(split)]
        _heatmap(axis, rows, order, withheld=split == "withheld")
        title = (
            "Withheld: fails at least one cell"
            if split == "withheld"
            else f"Public cells failed, of {int(rows['cells'].iloc[0])}"
        )
        axis.set_title(f"{letter}  {title}", pad=30)
    add_note(
        figure,
        "Columns: the declared economics, then the value of success or the trial costs halved and doubled.\n"
        "Each policy answers prior-drawn worlds of every cell. The withheld split is stated by outcome only.",
    )
    figure.subplots_adjust(left=0.29, right=0.99, top=0.78, bottom=0.15)
    return figure


def main(argv: list[str] | None = None) -> int:
    """Draw the release panel from the release-evidence record.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` reads the process arguments.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_release_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the figure and its data.")
    arguments = parser.parse_args(argv)
    policies = policy_table(PublicRelease.from_arguments(arguments).evidence())
    save_figure(panel_figure(policies), policies, arguments.output, "release_panel")
    for split in SPLITS:
        rows = policies.loc[policies["split"].eq(split)]
        declared = rows.loc[rows["band"].eq("declared")]
        if split == "withheld":
            for _, row in declared.iterrows():
                outcome = "fails at least one cell" if row["fails_a_cell"] else "passes every cell"
                print(f"{split:<9} {row['policy']:<26} {outcome} at declared economics")
            continue
        declared = declared.sort_values("failing_cells")
        for _, row in declared.iterrows():
            fewest = int(rows.loc[rows["policy"].eq(row["policy"]), "failing_cells"].min())
            print(
                f"{split:<9} {row['policy']:<26} fails {row['failing_cells']:>2} of {row['cells']} cells "
                f"(at least {fewest} at every band point)"
            )
    print(f"Wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
