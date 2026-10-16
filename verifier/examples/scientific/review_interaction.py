"""Explain executed research actions and the materiality of trial interpretations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import t

INK = "#23323d"
TEAL = "#226879"
RUST = "#a45b35"


def _save(figure: Figure, output: Path, stem: str) -> None:
    """Export editable vector artwork and a matching preview."""
    for extension in ("svg", "pdf", "png"):
        figure.savefig(output / f"{stem}.{extension}", dpi=180, transparent=extension == "svg")
    plt.close(figure)


def research_episode(root: Path, output: Path) -> pd.DataFrame:
    """Join native actions, accrued costs and observed readouts in execution order.

    Parameters
    ----------
    root
        Bundled TrialDev directory with action requests, ledger and patient tables.
    output
        Existing output directory containing independently recalculated readouts.

    Returns
    -------
    pandas.DataFrame
        One row per study start, readout and terminal decision. Costs represent
        the ledger state at that action; later study setup is not charged early.

    Raises
    ------
    ValueError
        If action requests, dates or readouts disagree with the native ledger.
    """
    ledger = json.loads((root / "ledger.json").read_text())
    prices = json.loads((root / "preferences.json").read_text())
    fits = pd.read_csv(output / "trialdev_readout_estimates.csv").set_index("phase")
    studies = {row["study_id"]: row for row in ledger["recruitment_commitments"]}
    patients = pd.DataFrame(
        [
            dict(study_id=entry["study_id"], **patient)
            for entry in ledger["enrollments"]
            for patient in entry["patients"]
        ]
    )
    latest_cost = {}
    latest_liability = {}
    started = set()
    observed = 0
    rows: list[dict[str, str | float | int]] = []
    for receipt in ledger["receipts"]:
        study_id, day = receipt["study_id"], float(receipt["programme_day"])
        study = studies[study_id]
        latest_cost[study_id] = float(receipt["cost"]["accrued_usd"])
        latest_liability[study_id] = float(receipt["cost"]["remaining_liability_usd"])
        first = study_id not in started
        if not first and receipt["kind"] != "readout":
            continue
        action = "start_study" if first else "readout"
        request_name = study_id if first else f"{study_id}-look"
        request = json.loads((root / "actions" / f"{request_name}.request.json").read_text())["request"]
        if request["action"] != action or (first and day != study["start_day"]):
            raise ValueError("Action does not match the recorded research sequence")
        started.add(study_id)
        estimate = lower = upper = np.nan
        if first:
            count = sum(arm["participant_count"] for arm in request["arms"])
            response = (
                f"Open {study['sites']} sites; recruit {count} patients; assess at Day {study['follow_up_days']}."
            )
        else:
            readout = json.loads((root / "readouts" / f"{study_id}-look" / "study.json").read_text())
            if day != readout["readout_day"]:
                raise ValueError("Readout clock differs from the native ledger")
            fit = fits.loc[study["phase"]]
            estimate, lower, upper = float(fit.estimate), float(fit.lower), float(fit.upper)
            available = int(fit.observed_active + fit.observed_control)
            observed += available
            response = f"{available} observed outcomes; treatment minus control {estimate:+.2f} DAS28 points."
        study_cost = sum(latest_cost.values())
        overhead = day * prices["programme_cost_per_day_usd"]
        rows.append(
            dict(
                sequence=len(rows),
                action=action,
                study_id=study_id,
                programme_day=day,
                follow_up_days=study["follow_up_days"],
                planned_patients=sum(
                    arm["participant_count"]
                    for arm in json.loads((root / "actions" / f"{study_id}.request.json").read_text())["request"][
                        "arms"
                    ]
                ),
                enrolled_to_date=int((patients.study_id.isin(started) & patients.enrollment_day.le(day)).sum()),
                available_outcomes=observed,
                study_cost_usd=study_cost,
                overhead_usd=overhead,
                total_cost_usd=study_cost + overhead,
                committed_followup_usd=sum(latest_liability.values()),
                budget_remaining_usd=ledger["budget_usd"] - study_cost - sum(latest_liability.values()),
                estimate=estimate,
                lower=lower,
                upper=upper,
                response=response,
            )
        )
    terminal = ledger["terminal"]
    request = json.loads((root / "actions/terminal.request.json").read_text())["request"]
    if request["action"] != terminal["decision"]:
        raise ValueError("Terminal request differs from the executed decision")
    final = rows[-1].copy()
    final.update(
        sequence=len(rows),
        action=terminal["decision"],
        study_id="programme",
        estimate=np.nan,
        lower=np.nan,
        upper=np.nan,
        response=(
            "Submit the candidate; confirmation evidence meets the simulator's approval rule."
            if terminal["regulatory_review"]["approved"]
            else "Submit the candidate; confirmation evidence does not meet the approval rule."
        ),
    )
    if final["programme_day"] != terminal["programme_day"]:
        raise ValueError("Episode requires terminal submission at the final readout")
    rows.append(final)
    result = pd.DataFrame(rows)
    result.to_csv(output / "interaction-timeline.csv", index=False)
    for compact in (False, True):
        _plot_episode(result, output, compact=compact)
    return result


@mpl.rc_context(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
def _plot_episode(rows: pd.DataFrame, output: Path, *, compact: bool) -> None:
    """Align evidence and resources on the programme clock."""
    looks = rows.loc[rows.action.eq("readout")]
    width = 4.2 if compact else 9
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(width, 9 if compact else 7.7),
        layout="constrained",
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.5, 1.1]},
    )
    stages = ("Early safety", "Exploration", "Confirmation")
    for i, (row, label) in enumerate(zip(looks.itertuples(), stages, strict=True)):
        start = rows.loc[rows.action.eq("start_study") & rows.study_id.eq(row.study_id)].iloc[0]
        axes[0].plot([start.programme_day, row.programme_day], [i, i], color=TEAL, linewidth=3)
        axes[0].scatter([start.programme_day, row.programme_day], [i, i], color=TEAL, s=28)
        axes[0].text(
            (start.programme_day + row.programme_day) / 2,
            i - 0.18,
            f"{label}\n{row.planned_patients} patients",
            ha="center",
            va="bottom",
            fontsize=11 if compact else 9,
        )
        axes[1].errorbar(
            row.programme_day,
            row.estimate,
            yerr=[
                [cast(float, row.estimate) - cast(float, row.lower)],
                [cast(float, row.upper) - cast(float, row.estimate)],
            ],
            color=TEAL,
            fmt="o",
            capsize=4,
            markersize=6,
        )
        axes[1].annotate(
            (
                f"Day {row.follow_up_days:g}\n{row.estimate:+.2f}"
                if compact
                else f"Day-{row.follow_up_days:g} outcome\n{row.estimate:+.2f} [{row.lower:+.2f}, {row.upper:+.2f}]"
            ),
            (row.programme_day, row.upper),
            xytext=(row.programme_day, 2.2) if compact else (0, 10),
            textcoords="data" if compact else "offset points",
            ha="center",
            fontsize=11 if compact else 9,
        )
    axes[0].set(ylim=(-0.85, 2.5), yticks=[], title="A  Open a study, then wait for evidence")
    axes[0].invert_yaxis()
    axes[1].axhline(0, color=INK, linewidth=0.8)
    axes[1].set(
        ylabel="DAS28 difference\n(lower favours treatment)",
        ylim=(-1.7, 2.7),
        title="B  Interpret each study’s readout",
    )
    axes[2].plot(rows.programme_day, rows.study_cost_usd / 1e6, color=TEAL, marker="o", label="Studies")
    axes[2].plot(rows.programme_day, rows.total_cost_usd / 1e6, color=RUST, marker="s", label="Studies + overhead")
    axes[2].set(
        ylabel="Accrued cost (million USD)",
        xlabel="Programme day",
        title="C  Follow the cost of research",
        ylim=(0, 48),
    )
    axes[2].legend(frameon=False, fontsize=10 if compact else 9, loc="upper left")
    axes[2].set_xticks([0, 150, 400, 810])
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(-100, 935)
        title = axis.get_title()
        axis.set_title("")
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis.tick_params(labelsize=11 if compact else 9)
    _save(figure, output, "research-episode-compact" if compact else "research-episode")


@mpl.rc_context(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
def interpretation_materiality(output: Path) -> None:
    """Apply one explicit conclusion rule to paired same-target analyses.

    Parameters
    ----------
    output
        Directory containing freshly reproduced repeated-study estimates.

    Notes
    -----
    The rule is an analysis scenario, not a clinical recommendation or an agent
    decision: a 95% interval above a specified response gain supports that target;
    an interval below it rules it out; otherwise it remains unresolved. Thresholds
    range from zero to ten percentage points. No costs or treatment value are
    inferred from these classifications.
    """
    source = pd.read_csv(output / "repeated_study_estimates.csv")
    paired = source.loc[source.case.eq("TE-S15-A3-favourable")].copy()
    if len(paired) != 64 or paired.groupby("seed").method.nunique().ne(2).any():
        raise ValueError("Materiality comparison requires all 32 paired colorectal studies")
    rows = []
    for threshold in np.arange(0, 10.01, 0.25):
        for row in paired.itertuples():
            lower, upper, reference = (
                100 * cast(float, row.lower),
                100 * cast(float, row.upper),
                100 * cast(float, row.reference),
            )
            conclusion = (
                "supports target" if lower > threshold else ("rules out target" if upper < threshold else "unresolved")
            )
            reference_margin = 100 * t.ppf(0.975, 63) * cast(float, row.reference_se)
            reference_conclusion = (
                "above target"
                if reference - reference_margin > threshold
                else ("below target" if reference + reference_margin < threshold else "reference unresolved")
            )
            rows.append(
                dict(
                    seed=row.seed,
                    method=row.method,
                    threshold_pp=threshold,
                    estimate_pp=100 * cast(float, row.estimate),
                    lower_pp=lower,
                    upper_pp=upper,
                    reference_pp=reference,
                    reference_lower_pp=reference - reference_margin,
                    reference_upper_pp=reference + reference_margin,
                    conclusion=conclusion,
                    reference_conclusion=reference_conclusion,
                )
            )
    data = pd.DataFrame(rows)
    data.to_csv(output / "interpretation-materiality.csv", index=False)
    summary = data.groupby(["threshold_pp", "method", "conclusion"]).size().rename("studies").reset_index()
    summary.to_csv(output / "interpretation-materiality-summary.csv", index=False)
    pairs = data.pivot(index=["threshold_pp", "seed"], columns="method", values="conclusion")
    pairs["different_conclusion"] = pairs["Routine endpoint"] != pairs["Stratified adjudication"]
    pairs.reset_index().to_csv(output / "interpretation-materiality-pairs.csv", index=False)
    for compact in (False, True):
        figure, axes = plt.subplots(
            2, 1, figsize=(4.2 if compact else 8, 7.4 if compact else 6), layout="constrained", sharex=True
        )
        colours = {"supports target": TEAL, "unresolved": "#bdc8cc", "rules out target": RUST}
        for axis, method in zip(axes, ("Routine endpoint", "Stratified adjudication"), strict=True):
            selected = (
                summary.loc[summary.method.eq(method)]
                .pivot(index="threshold_pp", columns="conclusion", values="studies")
                .fillna(0)
            )
            bottom = np.zeros(len(selected))
            for label, colour in colours.items():
                values = selected[label].to_numpy() if label in selected else np.zeros(len(selected))
                axis.fill_between(
                    selected.index, bottom, bottom + values, step="mid", color=colour, label=label.capitalize()
                )
                bottom += values
            reference_row = data.iloc[0]
            axis.axvspan(
                reference_row.reference_lower_pp,
                reference_row.reference_upper_pp,
                facecolor="none",
                edgecolor=INK,
                hatch="////",
                linewidth=0,
            )
            axis.axvline(reference_row.reference_pp, color=INK, linestyle="--", linewidth=1)
            axis.set(ylim=(0, 32), yticks=[0, 8, 16, 24, 32], ylabel="Trials (of 32)", title=method)
            axis.set_title("")
            axis.set_title(method, loc="left", fontsize=11, fontweight="bold")
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.55 if compact else 1.3),
            ncol=1 if compact else 3,
            fontsize=11 if compact else 9,
        )
        axes[1].set(
            xlabel=(
                "Required tumour-response gain\n(percentage points)"
                if compact
                else "Required gain in Day-168 tumour response (percentage points)"
            ),
            xlim=(0, 10),
        )
        axes[1].text(reference_row.reference_pp + 0.2, 1, "World reference", rotation=90, fontsize=9, va="bottom")
        _save(figure, output, "interpretation-materiality-compact" if compact else "interpretation-materiality")
