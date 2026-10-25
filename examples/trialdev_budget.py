"""Worked demonstration of whether a better evidence strategy converts budget into development value.

Every public TrialDev programme world is run at its three budget rungs (constrained, lean and generous)
by an oracle that knows every true effect and by the scripted reference policy, which sees only the
evidence it commissions. Each world is run under shared recruitment seeds (the same seeds for every
rung and policy), because one realised programme carries an approval draw worth hundreds of millions
of dollars. The release-evidence record publishes the resulting resource-gradient panel at the
declared profit scale; this demonstration reads that record only (no task data) and draws, for each
clinical programme, realised development value against budget beside stopping at once (value 0) and
ranking candidates by their early readouts.

Selection rule: every clinical programme in the record's ``trialdev.gradient``, each summarising every
public world of that programme. Nothing is omitted.

Run ``python trialdev_budget.py --output results/trialdev-budget`` (add ``--release-root`` for a local
copy of the dataset).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from figure_style import CAUTION, CREDITED, GRID, INK, MUTED, NOTE_SIZE, WIDTH, add_note, save_figure
from matplotlib.figure import Figure
from release_access import ProgrammeGradient, PublicRelease, ReleaseEvidence, add_release_arguments

RUNGS = ("constrained", "lean", "generous")
POLICIES = {"oracle": ("Attainable oracle", INK, "D"), "reference": ("Reference", CREDITED, "o")}
READOUT_RANK = "readout_rank"


def gradient(evidence: ReleaseEvidence) -> dict[str, ProgrammeGradient]:
    """Return the record's resource-gradient panel, by clinical programme.

    Parameters
    ----------
    evidence
        The release-evidence record.

    Returns
    -------
    dict
        Gradient summary of each clinical programme.

    Raises
    ------
    ValueError
        If the record publishes no resource-gradient panel.
    """
    panel = evidence.trialdev.gradient
    if not panel:
        raise ValueError("This release-evidence record publishes no TrialDev resource-gradient panel")
    return panel


def _signed(value: float) -> str:
    rounded = round(value)
    return f"{rounded:+d}".replace("-", "−") if rounded else "0"


def budget_table(panel: dict[str, ProgrammeGradient]) -> pd.DataFrame:
    """Return realised value by programme, policy and budget rung.

    Parameters
    ----------
    panel
        Output of ``gradient``.

    Returns
    -------
    pandas.DataFrame
        One row per programme, policy and rung, with the step from the previous rung and its 95% interval.
    """
    rows = []
    for programme, summary in panel.items():
        for policy in POLICIES:
            values = getattr(summary, policy)
            steps = (None, values.constrained_to_lean, values.lean_to_generous)
            for rung, budget, value, step in zip(
                RUNGS, summary.ladder_usd, values.realised_mean_by_rung_musd, steps, strict=True
            ):
                rows.append(
                    {
                        "programme": programme,
                        "policy": policy,
                        "rung": rung,
                        "budget_musd": budget / 1e6,
                        "realised_value_musd": value,
                        "step_musd": None if step is None else step.mean_musd,
                        "step_lower_musd": None if step is None else step.lower_musd,
                        "step_upper_musd": None if step is None else step.upper_musd,
                        "worlds": values.constrained_to_lean.worlds,
                    }
                )
        rows.append(
            {
                "programme": programme,
                "policy": READOUT_RANK,
                "rung": "main selection",
                "realised_value_musd": summary.mean_musd_by_policy[READOUT_RANK],
            }
        )
    return pd.DataFrame(rows)


def budget_figure(table: pd.DataFrame) -> Figure:
    """Draw realised value against budget for every clinical programme.

    Parameters
    ----------
    table
        Output of ``budget_table``.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per clinical programme on a shared value axis.
    """
    programmes = list(dict.fromkeys(table["programme"]))
    # Two programmes per row leave each panel wide enough for its budget labels and step labels.
    columns = min(2, len(programmes))
    lines = math.ceil(len(programmes) / columns)
    figure, grid = plt.subplots(
        lines,
        columns,
        figsize=(WIDTH, 1.5 + 2.15 * lines),
        sharey=True,
        squeeze=False,
        gridspec_kw={"wspace": 0.08, "hspace": 0.62},
    )
    axes = list(grid.flat)
    for spare in axes[len(programmes) :]:
        spare.set_visible(False)
    top = 1.3 * float(table["realised_value_musd"].max())
    for axis, programme in zip(axes, programmes, strict=False):
        part = table.loc[table["programme"].eq(programme)]
        for policy, (label, colour, marker) in POLICIES.items():
            rows = part.loc[part["policy"].eq(policy)]
            values = rows["realised_value_musd"].tolist()
            axis.plot(
                range(len(RUNGS)), values, color=colour, marker=marker, markersize=4.5, linewidth=1.4, label=label
            )
            # Each step is labelled at its segment's midpoint, above the oracle's line and below the reference's.
            for position, (_, row) in enumerate(rows.iterrows()):
                if position == 0:
                    continue
                low, mean, high = (_signed(row[key]) for key in ("step_lower_musd", "step_musd", "step_upper_musd"))
                axis.annotate(
                    f"{mean}\n({low}, {high})",
                    (position - 0.5, (values[position - 1] + values[position]) / 2),
                    xytext=(0, 6 if policy == "oracle" else -6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if policy == "oracle" else "top",
                    fontsize=NOTE_SIZE,
                    color=colour,
                    linespacing=1.0,
                    bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none"},
                )
        rank = float(part.loc[part["policy"].eq(READOUT_RANK), "realised_value_musd"].iloc[0])
        axis.axhline(rank, color=CAUTION, linestyle=(0, (4, 2)), linewidth=1.1, label="Readout ranking")
        axis.axhline(0, color=MUTED, linewidth=0.9, label="Stop")
        budgets = part.loc[part["policy"].eq("oracle"), "budget_musd"]
        axis.set_xticks(
            range(len(RUNGS)), [f"{rung}\n${budget:.0f}M" for rung, budget in zip(RUNGS, budgets, strict=True)]
        )
        axis.set_xlim(-0.35, len(RUNGS) - 0.65)
        axis.set_ylim(-0.06 * top, top)
        axis.set_title(programme.replace("_", " ").capitalize().replace("Copd", "COPD"))
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.yaxis.set_tick_params(labelleft=True)
    figure.supylabel("Mean realised development value (US$ millions)", x=0.015, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, columnspacing=1.6)
    worlds = int(table["worlds"].dropna().iloc[0])
    add_note(
        figure,
        f"x: budget rung and budget (US$ millions). Each point averages {worlds} programme worlds under shared "
        "recruitment seeds. Labels: change from the\nprevious budget, US$ millions (95% interval over worlds). "
        "Dashed: Readout ranking on the main selection (one budget per world).",
    )
    height = figure.get_figheight()
    figure.subplots_adjust(left=0.1, right=0.99, top=1 - 0.6 / height, bottom=0.85 / height)
    return figure


def main(argv: list[str] | None = None) -> int:
    """Draw the budget demonstration from the release-evidence record.

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
    if evidence.trialdev.gradient is None and evidence.trialdev.gradient_status == "pending":
        message = "pending: published as an evidence update with unchanged data"
        figure, axis = plt.subplots(figsize=(WIDTH, 2.4))
        axis.set_axis_off()
        axis.text(0.5, 0.65, "TrialDev resource-gradient evaluation", ha="center", transform=axis.transAxes)
        axis.text(
            0.5,
            0.35,
            "pending: published as an evidence update\nwith unchanged data",
            ha="center",
            transform=axis.transAxes,
        )
        save_figure(figure, pd.DataFrame({"status": [message]}), arguments.output, "trialdev_budget")
        print(message)
        return 0
    table = budget_table(gradient(evidence))
    save_figure(budget_figure(table), table, arguments.output, "trialdev_budget")
    for (programme, policy), rows in table.loc[table["policy"].isin(POLICIES)].groupby(
        ["programme", "policy"], sort=False
    ):
        values = " -> ".join(f"{value:.0f}" for value in rows["realised_value_musd"])
        print(f"{programme:<22} {policy:<10} ${values}M across the budget rungs")
    print(f"Wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
