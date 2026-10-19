"""Worked demonstration of whether a data-blind strategy could score well on TrialEval.

The release-evidence record runs fixed answer policies over 1,200 prior-drawn worlds of every public
cell and counts the cells each policy fails, at the declared economics and at every point of the
disclosed sensitivity band (value of success and trial costs halved and doubled). It also records how far
each trap's excluded analysis moves the estimate, by tertile of the true effect. This demonstration
reads that record only (no task data) and draws both.

Selection rule: every policy and every band point in the record's ``economics.band_failing_cells``;
every trap cell in its ``consequence`` list. Nothing is omitted.

Run ``python release_panel.py --output results/release-panel`` (add ``--release-root`` for a local copy
of the dataset).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_style import CAUTION, INK, MUTED, ROUTINE, save_figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from release_access import PublicRelease, ReleaseEvidence, add_release_arguments

REFERENCE = "scripted_reference"
POLICY_LABELS = {
    REFERENCE: "Reference: the credited analysis",
    "always_robust": "Always the trap's correction",
    "follow_plan": "Follow the analysis plan",
    "always_naive": "Always the excluded shortcut",
    "plan_hedged_3x": "Plan estimate, interval ×3",
    "prior_only": "Ignore the data (prior only)",
    "null_point_plan_interval": "Null estimate, plan interval",
    "plan_overconfident_0.3x": "Plan estimate, interval ×0.3",
}
BAND_LABELS = {
    "declared": "Declared",
    "value x0.5": "Value\n×½",
    "value x2": "Value\n×2",
    "cost x0.5": "Costs\n×½",
    "cost x2": "Costs\n×2",
}
TIERS = ("lower", "middle", "upper")


def policy_table(evidence: ReleaseEvidence) -> pd.DataFrame:
    """Return failing cells by policy and band point, one row per pair.

    Parameters
    ----------
    evidence
        The release-evidence record.

    Returns
    -------
    pandas.DataFrame
        ``policy``, ``band``, ``failing_cells`` and ``cells``.

    Raises
    ------
    ValueError
        If the band omits the declared economics or a band point omits a policy.
    """
    band = evidence.trialeval.economics.band_failing_cells
    if "declared" not in band or band["declared"] != evidence.trialeval.policy_public.failing_cells:
        raise ValueError("The band's declared point differs from the policy panel")
    policies = set(band["declared"])
    if any(set(counts) != policies for counts in band.values()):
        raise ValueError("A band point omits a policy")
    cells = evidence.trialeval.policy_public.cells
    return pd.DataFrame(
        [
            {"policy": policy, "band": point, "failing_cells": count, "cells": cells}
            for point, counts in band.items()
            for policy, count in counts.items()
        ]
    )


def consequence_table(evidence: ReleaseEvidence) -> pd.DataFrame:
    """Return each trap's tertile shift of the excluded analysis, in standard errors.

    Parameters
    ----------
    evidence
        The release-evidence record.

    Returns
    -------
    pandas.DataFrame
        ``cell``, ``tertile``, ``shift_se`` (signed), ``shift_consequential`` and ``interval_consequential``.
    """
    return pd.DataFrame(
        [
            {
                "cell": trap.regime_cell_id,
                "tertile": tertile,
                "shift_se": shift,
                "shift_consequential": trap.shift_consequential,
                "interval_consequential": trap.interval_consequential,
                "width_ratio": trap.width_ratio,
            }
            for trap in evidence.trialeval.consequence
            for tertile, shift in zip(TIERS, trap.tertile_bias_sd, strict=True)
        ]
    )


def panel_figure(policies: pd.DataFrame, traps: pd.DataFrame) -> Figure:
    """Draw failing cells by policy and band, beside the trap shifts against the one-SE bar.

    Parameters
    ----------
    policies
        Output of ``policy_table``.
    traps
        Output of ``consequence_table``.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    grid = policies.pivot(index="policy", columns="band", values="failing_cells")
    grid = grid.loc[
        grid["declared"].sort_values(kind="stable").index,
        [b for b in BAND_LABELS if b in grid] + [b for b in grid.columns if b not in BAND_LABELS],
    ]
    cells = int(policies["cells"].iloc[0])
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": (1.25, 1.0), "wspace": 0.42}
    )
    colours = LinearSegmentedColormap.from_list("failing", ["#F5F6F8", "#E9B8A6", ROUTINE])
    left.imshow(grid.to_numpy(), cmap=colours, vmin=0, vmax=cells, aspect="auto")
    for (row, column), value in np.ndenumerate(grid.to_numpy()):
        left.text(
            column,
            row,
            str(value),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if value > 0.6 * cells else INK,
            fontweight="bold" if grid.index[row] == REFERENCE else "normal",
        )
    left.set_xticks(range(grid.shape[1]), [BAND_LABELS.get(b, b) for b in grid.columns])
    left.set_yticks(range(grid.shape[0]), [POLICY_LABELS.get(p, p.replace("_", " ")) for p in grid.index])
    left.tick_params(length=0)
    left.xaxis.set_ticks_position("top")
    for spine in left.spines.values():
        spine.set_visible(False)
    left.set_title(f"a  Cells failed, of {cells}, by fixed policy", pad=30)
    left.text(
        0.5,
        -0.06,
        "Declared economics, and value of success\nor trial costs halved and doubled",
        transform=left.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )

    order = list(dict.fromkeys(traps["cell"]))
    for position, cell in enumerate(order):
        rows = traps.loc[traps["cell"].eq(cell)]
        shifts = rows["shift_se"].abs().to_numpy()
        y = len(order) - 1 - position
        admitted_by_interval = bool(rows["interval_consequential"].iloc[0]) and not bool(
            rows["shift_consequential"].iloc[0]
        )
        right.plot([shifts.min(), shifts.max()], [y, y], color=MUTED, linewidth=1.0)
        right.plot(
            shifts,
            [y] * 3,
            linestyle="none",
            marker="o",
            markersize=5.5,
            color=ROUTINE,
            markerfacecolor="white" if admitted_by_interval else ROUTINE,
            markeredgewidth=1.2,
        )
        if admitted_by_interval:
            right.annotate(
                f"open: admitted by its\ninterval (width ×{rows['width_ratio'].iloc[0]:.2f})",
                (shifts.max(), y),
                xytext=(8, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
                color=MUTED,
            )
    right.axvline(1.0, color=CAUTION, linestyle=(0, (4, 2)), linewidth=1.2)
    right.annotate(
        "1 SE",
        (1.0, len(order) - 0.45),
        xytext=(3, 0),
        textcoords="offset points",
        color=CAUTION,
        fontsize=7.2,
        va="center",
    )
    right.set_yticks(range(len(order)), order[::-1])
    right.set_ylim(-0.7, len(order) - 0.3)
    right.set_xlim(0, max(2.5, float(traps["shift_se"].abs().max()) * 1.1))
    right.set_xlabel("Shift of the excluded analysis (standard\nerrors; one dot per tertile of the true effect)")
    right.set_title("b  Trap shift against 1 SE", pad=30)
    right.grid(axis="x", color="#E3E5EA", linewidth=0.6)
    figure.subplots_adjust(left=0.255, right=0.975, top=0.78, bottom=0.2)
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
    evidence = PublicRelease.from_arguments(arguments).evidence()
    policies, traps = policy_table(evidence), consequence_table(evidence)
    data = pd.concat([policies.assign(panel="a"), traps.assign(panel="b")], ignore_index=True)
    save_figure(panel_figure(policies, traps), data, arguments.output, "release_panel")
    declared = policies.loc[policies["band"].eq("declared")].sort_values("failing_cells")
    for _, row in declared.iterrows():
        worst = int(policies.loc[policies["policy"].eq(row["policy"]), "failing_cells"].max())
        print(
            f"{row['policy']:<26} fails {row['failing_cells']:>2} of {row['cells']} cells (up to {worst} across the band)"
        )
    print(f"Wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
