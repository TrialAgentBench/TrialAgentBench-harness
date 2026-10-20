"""Worked demonstration of how TrialEval prices an answer: the decision layer and the cost of a distorted report.

Every TrialEval answer, an estimate with its 95% interval, drives one development decision through the
verifier's public decision layer (``trialagentbench_verifier.decision_consequence``): stop, or run a
confirmatory trial of a chosen multiple of the analysed trial's size, whichever has the larger expected value
under the disclosed effect prior and the trial's declared economics. This demonstration uses that layer alone,
with representative declared economics and no task data, to answer two questions.

``a``: how does a report map to a decision? It sweeps the reported estimate, in standard errors of the analysed
trial, for an interval of the honest width (1.96 standard errors each side), half that width and twice it, and
draws the confirmatory trial each report drives (zero is stop). Because a success is worth far more than a
confirmatory trial costs, the layer confirms on modest evidence; it confirms largest where the evidence is
promising but uncertain, and a wider interval keeps a confirmatory trial worth running on weaker evidence.

``b``: what does a distorted report cost? In worlds drawn from the disclosed prior, the honest report is the
trial's estimate with its 95% interval. Each distortion of it (an estimate moved by one standard error either way,
an interval halved or doubled, or the agent's own shrinkage towards the prior, which the layer then applies a
second time) is priced against the honest report of the same data: its posterior-expected regret, in US$
millions, and the share of worlds in which the stop-or-confirm decision changes at a regret of at least $5M.
Because the layer already combines the report with the prior, no distortion gains in expectation.

Selection rule: ``--worlds`` worlds (default 2,000) drawn from the disclosed prior with a fixed seed, every
one priced; nothing is omitted. The economics are representative values inside the declared ranges, not any one
trial's.

Run ``python decision_layer.py --output results/decision-layer``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_style import ALTERNATIVE, CREDITED, DISTORTED, GRID, INK, MUTED, NOTE_SIZE, WIDTH, add_note, save_figure
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from trialagentbench_verifier.decision_consequence import (
    COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS,
    COCHRANE_SNR_MIXTURE_WEIGHTS,
    DecisionEconomicsV1,
    DecisionReportV1,
    decision_consequence_v1,
    optimal_action_v1,
)

ECONOMICS = DecisionEconomicsV1(value_of_success_musd=1_000.0, replicate_cost_musd=30.0, fixed_trial_cost_musd=5.9)
"""Representative declared economics (US$ millions): a confirmatory success, a confirmatory trial the size of the
analysed trial beyond its fixed cost, and the fixed cost of an industry confirmatory trial."""
SCALE = 1.0
"""The analysed trial's standard error; estimates are in its units."""
PARTICIPANTS = 500
"""Participants in the analysed trial; it only converts size multiples to participants."""
Z95 = 1.959963984540054
WIDTHS = {0.5: "Interval ×½ (overconfident)", 1.0: "Honest interval", 2.0: "Interval ×2 (over-wide)"}
DISTORTIONS = {
    "estimate +1 SE": (1.0, 1.0),
    "estimate −1 SE": (-1.0, 1.0),
    "interval ×½": (0.0, 0.5),
    "interval ×2": (0.0, 2.0),
    "shrunk to the prior": None,
}
"""Each distortion's shift of the estimate (in SE) and scale of the interval; ``None`` is the agent's shrinkage."""


def _report(estimate: float, standard_error: float) -> DecisionReportV1:
    return DecisionReportV1(
        estimate=estimate, lower=estimate - Z95 * standard_error, upper=estimate + Z95 * standard_error
    )


def shrunk_report(estimate: float, standard_error: float = SCALE) -> DecisionReportV1:
    """Return the posterior mean and its normal 95% interval under the disclosed prior: an agent-side shrinkage.

    Parameters
    ----------
    estimate
        The trial's estimate.
    standard_error
        Its standard error.

    Returns
    -------
    DecisionReportV1
        The shrunk report, which the decision layer would combine with the prior a second time.
    """
    weights = np.asarray(COCHRANE_SNR_MIXTURE_WEIGHTS)
    prior = (np.asarray(COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS) * SCALE) ** 2
    total = prior + standard_error**2
    log_weight = np.log(weights) - 0.5 * np.log(total) - 0.5 * estimate**2 / total
    posterior = np.exp(log_weight - log_weight.max())
    posterior /= posterior.sum()
    means, variances = prior / total * estimate, prior / total * standard_error**2
    mean = float(posterior @ means)
    return _report(mean, math.sqrt(float(posterior @ (variances + means**2)) - mean**2))


def action_table(estimates: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None) -> pd.DataFrame:
    """Return the action each report drives, by reported estimate and interval width.

    Parameters
    ----------
    estimates
        Reported estimates in standard errors; default a grid from −3 to 5.

    Returns
    -------
    pandas.DataFrame
        ``estimate_se``, ``width``, ``confirm`` and ``size_multiple`` (0 when the layer stops).
    """
    grid = np.linspace(-3.0, 5.0, 321) if estimates is None else estimates
    rows = []
    for width in WIDTHS:
        for estimate in grid:
            action = optimal_action_v1(
                _report(float(estimate), SCALE * width),
                scale=SCALE,
                favorable_direction="higher",
                economics=ECONOMICS,
            )
            rows.append(
                {
                    "estimate_se": float(estimate),
                    "width": width,
                    "confirm": action.confirm,
                    "size_multiple": action.size_multiple,
                }
            )
    return pd.DataFrame(rows)


def distortion_table(worlds: int = 2_000, seed: int = 20261020) -> pd.DataFrame:
    """Return what each distortion of the honest report costs, over worlds drawn from the disclosed prior.

    Parameters
    ----------
    worlds
        Number of worlds.
    seed
        Seed of the effect and sampling draws.

    Returns
    -------
    pandas.DataFrame
        One row per distortion: mean posterior-expected regret with its Monte Carlo SE, and the share of worlds
        whose decision changes at a regret of at least $5M.
    """
    rng = np.random.default_rng(seed)
    component = rng.choice(len(COCHRANE_SNR_MIXTURE_WEIGHTS), size=worlds, p=COCHRANE_SNR_MIXTURE_WEIGHTS)
    truth = rng.normal(0.0, np.asarray(COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS)[component] * SCALE)
    observed = truth + SCALE * rng.normal(size=worlds)
    rows = []
    for name, distortion in DISTORTIONS.items():
        regrets, consequential = [], []
        for effect, estimate in zip(truth, observed, strict=True):
            honest = _report(float(estimate), SCALE)
            if distortion is None:
                answer = shrunk_report(float(estimate))
            else:
                shift, width = distortion
                answer = _report(float(estimate) + shift * SCALE, SCALE * width)
            consequence = decision_consequence_v1(
                answer,
                (honest,),
                truth=float(effect),
                scale=SCALE,
                participants=PARTICIPANTS,
                favorable_direction="higher",
                economics=ECONOMICS,
            )
            regrets.append(consequence.regret_musd)
            consequential.append(consequence.consequential)
        values = np.asarray(regrets)
        rows.append(
            {
                "report": name,
                "worlds": worlds,
                "mean_regret_musd": float(values.mean()),
                "regret_se_musd": float(values.std(ddof=1) / math.sqrt(worlds)),
                "consequential_share": float(np.mean(consequential)),
            }
        )
    return pd.DataFrame(rows)


def decision_figure(actions: pd.DataFrame, distortions: pd.DataFrame) -> Figure:
    """Draw the action each report drives beside what each distortion costs.

    Parameters
    ----------
    actions
        Output of ``action_table``.
    distortions
        Output of ``distortion_table``.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(WIDTH, 3.6), gridspec_kw={"width_ratios": (1.1, 1.0), "wspace": 0.72}
    )
    styles = {0.5: (DISTORTED, (0, (4, 2))), 1.0: (CREDITED, "-"), 2.0: (ALTERNATIVE, (0, (1, 1.5)))}
    for width, label in WIDTHS.items():
        rows = actions.loc[actions["width"].eq(width)]
        colour, line = styles[width]
        left.step(rows["estimate_se"], rows["size_multiple"], where="mid", color=colour, linestyle=line, label=label)
    left.axhline(0, color=MUTED, linewidth=0.8)
    # The ticks reach past the largest trial; the axis continues above them only to hold the legend.
    locator = MaxNLocator(nbins=5, integer=True)
    ticks = [float(tick) for tick in locator.tick_values(0, float(actions["size_multiple"].max())) if tick >= 0]
    left.set_ylim(-0.3, 1.5 * ticks[-1])
    left.set_yticks(ticks, ["Stop" if tick == 0 else f"{tick:g}" for tick in ticks])
    left.spines["left"].set_bounds(0, ticks[-1])
    left.set_xlabel("Reported estimate (standard errors\nof the analysed trial)")
    left.set_ylabel("Confirmatory trial (multiple of\nthe analysed trial's size)")
    bottom, top = left.get_ylim()
    left.yaxis.set_label_coords(-0.14, (ticks[-1] / 2 - bottom) / (top - bottom))
    left.set_title("a  The decision each report drives")
    left.grid(color=GRID, linewidth=0.6)
    left.legend(loc="upper left", borderaxespad=0.2)

    order = distortions.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(order))
    right.barh(positions, order["mean_regret_musd"], color=DISTORTED, height=0.55)
    right.errorbar(
        order["mean_regret_musd"],
        positions,
        xerr=1.96 * order["regret_se_musd"],
        fmt="none",
        ecolor=INK,
        elinewidth=0.8,
        capsize=2,
    )
    for position, row in order.iterrows():
        right.annotate(
            f"{100 * row['consequential_share']:.0f}% consequential",
            (row["mean_regret_musd"] + 1.96 * row["regret_se_musd"], position),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=NOTE_SIZE,
            color=INK,
        )
    right.set_yticks(positions, order["report"])
    right.set_xlim(0, float((order["mean_regret_musd"] + 1.96 * order["regret_se_musd"]).max()) * 1.75)
    right.set_xlabel("Posterior-expected regret against the\nhonest report (US$ millions per answer)")
    right.set_title("b  What a distorted report costs")
    right.grid(axis="x", color=GRID, linewidth=0.6)
    right.set_axisbelow(True)
    add_note(
        figure,
        f"b: mean over {int(distortions['worlds'].iloc[0]):,} worlds drawn from the disclosed prior, with its 95% "
        "Monte Carlo interval. Consequential: the share of worlds in which the\ndistortion flips stop or confirm "
        "at a regret of at least US$5M. Representative declared economics; no task data.",
    )
    figure.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.3)
    return figure


def main(argv: list[str] | None = None) -> int:
    """Draw the decision-layer demonstration.

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
    parser.add_argument("--output", type=Path, required=True, help="Directory for the figure and its data.")
    parser.add_argument("--worlds", type=int, default=2_000, help="Worlds drawn from the disclosed prior.")
    arguments = parser.parse_args(argv)
    actions, distortions = action_table(), distortion_table(arguments.worlds)
    data = pd.concat([actions.assign(panel="a"), distortions.assign(panel="b")], ignore_index=True)
    save_figure(decision_figure(actions, distortions), data, arguments.output, "decision_layer")
    honest = actions.loc[actions["width"].eq(1.0) & actions["confirm"], "estimate_se"]
    print(f"An honest report confirms from an estimate of {honest.min():.2f} standard errors upwards.")
    for _, row in distortions.iterrows():
        print(
            f"{row['report']:<20} costs ${row['mean_regret_musd']:.2f}M ± {row['regret_se_musd']:.2f} per answer; "
            f"{100 * row['consequential_share']:.1f}% of worlds are consequential errors"
        )
    print(f"Wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
