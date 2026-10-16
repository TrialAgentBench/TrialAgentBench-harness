"""Replay one colorectal study from observed labels through numerical recovery.

The first archived study is selected by index before scoring. All 32 paired
studies remain in the exports; 64 disjoint complete studies supply the reference.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from review_mechanisms import summarize_repetitions
from review_recovery import estimates
from review_scoring import numerical_loss as reader_loss
from scipy.stats import norm, t

from trialagentbench_verifier.numerical_scoring import bounded_recovery, numerical_loss

CASE = "TE-S15-A3-favourable"
SEED = 86_828_000
METHODS = ("Routine endpoint", "Stratified adjudication")
COLOURS = ("#536370", "#226879")
INK = "#23323d"
REFERENCE = "#79552a"
MISS = "#92332c"


class ScoredEstimate(BaseModel):
    """One reproduced answer and additive contributions to its numerical loss."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    method: str
    estimate: float
    lower: float
    upper: float
    point_loss: float
    width_loss: float
    miss_loss: float
    total_loss: float
    recovery_score: float


class WorkedStudy(BaseModel):
    """Public figure inputs bound to one selected study and reference bank."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_id: Literal["trialagentbench.worked_adjudication/v1"] = "trialagentbench.worked_adjudication/v1"
    case: str = CASE
    seed: int = SEED
    study_index: int = 0
    selection: str = "First archived analysis study (index 0), selected without inspecting scores."
    estimand: str = (
        "Combination minus standard therapy probability of centrally adjudicated objective tumour response "
        "at Day 168 in the randomized metastatic colorectal-cancer population."
    )
    unit: str = "risk difference (proportion); multiply by 100 for percentage points"
    participants: int
    adjudicated_participants: int
    reference: float
    reference_se: float
    reference_lower: float
    reference_upper: float
    reference_worlds: int = 64
    analysis_worlds: int = 32
    scale: float
    scale_basis: str = "Sample SD of the 64 complete-study contrasts; reference_se × sqrt(64)."
    confidence_level: float = 0.95
    loss_formula: str = (
        "L = (abs(estimate-reference) + upper-lower + 40*max(lower-reference, reference-upper, 0))/(2*scale)"
    )
    score_formula: str = "recovery_score = 1/(1+L)"
    scope: str = (
        "Numerical recovery in this colorectal study, using the benchmark scoring function. "
        "These are reproducible observed-data analyses, not agent-campaign or held-out release grades. "
        "The reference is a Monte Carlo estimate, not an error-free clinical treatment effect."
    )
    estimates: list[ScoredEstimate]
    source_paths: list[str]


def score_rows(rows: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Calculate native recovery and check parity with the independent reader.

    Parameters
    ----------
    rows
        Observed-data estimates with interval limits and the separate reference.
    scale
        Standard deviation of complete-study contrasts in the reference bank.

    Returns
    -------
    pandas.DataFrame
        Original estimates and additive, half-weighted loss components.

    Raises
    ------
    AssertionError
        If the public native scorer, reader equation or decomposition disagree.
    """
    scored = rows.copy()
    scored["point_loss"] = abs(scored.estimate - scored.reference) / (2 * scale)
    scored["width_loss"] = (scored.upper - scored.lower) / (2 * scale)
    scored["miss_loss"] = (
        40
        * np.maximum.reduce(
            np.array(
                [scored.lower - scored.reference, scored.reference - scored.upper, np.zeros(len(scored))], dtype=float
            )
        )
        / (2 * scale)
    )
    losses = []
    for values in scored[
        ["estimate", "lower", "upper", "reference", "point_loss", "width_loss", "miss_loss"]
    ].to_numpy(dtype=float):
        estimate, lower, upper, reference, point, width, miss = map(float, values)
        loss = numerical_loss(
            estimate=estimate,
            lower=lower,
            upper=upper,
            target=reference,
            scale=scale,
            confidence_level=0.95,
        )
        if loss is None:
            raise ValueError("Finite teaching results must produce a finite loss")
        np.testing.assert_allclose(
            [loss, reader_loss(estimate, lower, upper, reference, scale)],
            point + width + miss,
            rtol=1e-12,
        )
        losses.append(loss)
    scored["total_loss"] = losses
    scored["recovery_score"] = [bounded_recovery(loss) for loss in losses]
    return scored


def plot_worked(worked: WorkedStudy, output: Path, *, compact: bool) -> None:
    """Draw effects and grading separately, with common scientific colour roles.

    Each figure has a compact composition rather than a scaled desktop panel.
    SVG and PDF retain editable text; all exports have transparent backgrounds.
    """
    with plt.rc_context(
        {
            "font.size": 15,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "sans-serif"],
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#9ca8ad",
            "savefig.transparent": True,
        }
    ):
        for grading in (False, True):
            size = (4.375, 4.8 if grading else 4.7) if compact else (10, 4.25)
            fig, ax = plt.subplots(figsize=size)
            ax.set_facecolor("none")
            fig.subplots_adjust(left=0.07, right=0.96, bottom=0.22 if grading else 0.18, top=0.97)
            ax.set(yticks=[], ylim=(-0.55, 2.2))
            ax.tick_params(axis="x", length=0, pad=9)
            ax.grid(axis="x", color="#c8d1d7", linewidth=0.6, alpha=0.6)
            ax.set_axisbelow(True)
            if grading:
                ax.set(xlim=(0, 77), xticks=[0, 20, 40, 60], xlabel="Numerical loss · lower is better")
                for index, row in enumerate(worked.estimates):
                    y = 1.25 - index * 1.15
                    ax.text(0, y + 0.31, row.method, color=COLOURS[index], weight="bold")
                    left = 0.0
                    for field, colour, hatch in (
                        ("point_loss", INK, None),
                        ("width_loss", "#aebdc4", "///"),
                        ("miss_loss", MISS, None),
                    ):
                        value = getattr(row, field)
                        ax.barh(y, value, left=left, height=0.22, color=colour, hatch=hatch, linewidth=0)
                        left += value
                    ax.text(left + 1.8, y, f"{row.total_loss:.2f}", va="center", weight="bold")
                routine = worked.estimates[0]
                ax.text(
                    routine.point_loss + routine.width_loss,
                    0.87,
                    f"Miss penalty: {routine.miss_loss:.2f}",
                    color=MISS,
                    fontsize=14,
                )
                for x, colour, label, hatch in (
                    (0.04, INK, "Point error", None),
                    (0.37, "#aebdc4", "Width", "///"),
                    (0.62, MISS, "Miss penalty", None),
                ):
                    fig.add_artist(
                        matplotlib.patches.Rectangle(
                            (x, 0.045),
                            0.025,
                            0.025,
                            transform=fig.transFigure,
                            facecolor=colour,
                            hatch=hatch,
                            linewidth=0,
                        )
                    )
                    fig.text(x + 0.037, 0.045, label, fontsize=13, va="bottom")
            else:
                reference = 100 * worked.reference
                ax.set(xlim=(-5, 13), xticks=[-5, 0, 5, 10], xlabel="Response difference (percentage points)")
                if compact:
                    ax.set_xlabel("Response difference\n(percentage points)")
                ax.axvspan(100 * worked.reference_lower, 100 * worked.reference_upper, color=REFERENCE, alpha=0.1)
                ax.axvline(reference, color=REFERENCE, lw=1.6, ls=(0, (4, 3)))
                ax.axvline(0, color="#87959c", lw=0.9)
                ax.text(
                    0.02,
                    0.96,
                    f"World reference: +{reference:.2f}",
                    transform=ax.transAxes,
                    color=REFERENCE,
                    weight="bold",
                    va="top",
                )
                for index, row in enumerate(worked.estimates):
                    y = 1.1 - index * 1.05
                    ax.text(
                        0.02,
                        y + 0.26,
                        row.method,
                        transform=ax.get_yaxis_transform(),
                        color=COLOURS[index],
                        weight="bold",
                    )
                    ax.errorbar(
                        100 * row.estimate,
                        y,
                        xerr=[[100 * (row.estimate - row.lower)], [100 * (row.upper - row.estimate)]],
                        fmt="o" if index == 0 else "s",
                        color=COLOURS[index],
                        capsize=5,
                        markersize=8,
                        linewidth=2,
                    )
                    ax.text(
                        0.02,
                        y - 0.32,
                        f"{100 * row.estimate:+.2f}  [{100 * row.lower:+.2f}, {100 * row.upper:+.2f}]",
                        transform=ax.get_yaxis_transform(),
                        color=COLOURS[index],
                        fontsize=14,
                    )
            name = "adjudication-grading" if grading else "adjudication-worked"
            if compact:
                name += "-compact"
            for suffix in ("svg", "pdf", "png"):
                destination = output / f"{name}.{suffix}"
                fig.savefig(destination, dpi=160, transparent=True)
                if suffix == "svg":
                    destination.write_text(
                        "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n"
                    )
            fig.savefig(output / f"{name}-placed.png", dpi=80, transparent=False, facecolor="#fafbf9")
            plt.close(fig)


def plot_baseline_anchors(data: pd.DataFrame, output: Path) -> None:
    """Compare two published baseline proportions with the generated study.

    Parameters
    ----------
    data
        Participant rows of the selected colorectal teaching study.
    output
        Destination for the comparison CSV and editable SVGs.

    Notes
    -----
    Published counts are Table 1 of the TRIBE/TRIBE2 analysis, DOI
    10.1016/j.esmoop.2022.100606: 1,146 patients without baseline bone disease.
    These selected marginal anchors do not validate the authored joint model.
    """
    rows = []
    for label, field, numerator in (
        ("Primary tumour removed", "primary_tumour_resected", 664),
        ("Activity limitation (ECOG 1–2)", "performance_status", 141),
    ):
        generated = int(data[field].gt(0).sum())
        rows.append(
            dict(
                characteristic=label,
                field=field,
                published_numerator=numerator,
                published_denominator=1146,
                published_percent=100 * numerator / 1146,
                generated_numerator=generated,
                generated_denominator=len(data),
                generated_percent=100 * generated / len(data),
                source_doi="10.1016/j.esmoop.2022.100606",
                source_table="1; no bone metastases at baseline",
            )
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "colorectal-baselines.csv", index=False)
    for compact in (False, True):
        width, height = (350, 388) if compact else (800, 370)
        left, plot_width = (16, 278) if compact else (24, 656)
        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            '<title id="title">Clinical source and generated patient characteristics</title>',
            '<desc id="desc">Published TRIBE/TRIBE2 proportions and the generated colorectal teaching study agree '
            "closely for prior primary tumour resection and ECOG performance status one to two. "
            "Both bars show percentages of their own study population.</desc>",
            '<style>text{font-family:Arial,"Liberation Sans",sans-serif;font-size:17px;fill:#23323d}'
            ".title{font-weight:700}.small{font-size:15px;fill:#536370}</style>",
        ]
        for index, (_, row) in enumerate(table.iterrows()):
            y = 27 + index * 150
            elements.append(f'<text class="title" x="{left}" y="{y}">{row.characteristic}</text>')
            for j, (label, field, colour) in enumerate(
                (
                    ("TRIBE/TRIBE2", "published_percent", "#687b86"),
                    ("Generated study", "generated_percent", "#226879"),
                )
            ):
                yy = y + 39 + j * 46
                value = float(row[field])
                length = plot_width * value / 70
                elements.extend(
                    [
                        f'<text class="small" x="{left}" y="{yy - 8}">{label}</text>',
                        f'<rect x="{left}" y="{yy}" width="{length:.4f}" height="12" fill="{colour}"/>',
                        f'<text x="{left + length + 9:.4f}" y="{yy + 12}">{value:.1f}%</text>',
                    ]
                )
        axis_y = 324
        elements.append(f'<path d="M{left} {axis_y}H{left + plot_width}" stroke="#9ca8ad"/>')
        for value in (0, 20, 40, 60):
            x = left + plot_width * value / 70
            elements.append(f'<text class="small" x="{x:.4f}" y="{axis_y + 24}" text-anchor="middle">{value}</text>')
        elements.append(
            f'<text class="small" x="{left}" y="{axis_y + 50}">Percentage of participants</text>'
            if compact
            else f'<text class="small" x="{width - 20}" y="{axis_y + 24}" text-anchor="end">% of participants</text>'
        )
        elements.append("</svg>")
        name = "colorectal-baselines-compact.svg" if compact else "colorectal-baselines.svg"
        (output / name).write_text("\n".join(elements) + "\n", encoding="utf-8")


def export_worked(source: Path, output: Path) -> WorkedStudy:
    """Refit all paired trials before introducing the disjoint reference bank."""
    rows, reference_draws = [], []
    with zipfile.ZipFile(source / "recovery-records.zip") as archive:
        study_bytes = archive.read(f"{CASE}/study.json")
        study = json.loads(study_bytes)
        question = study["estimands"][0]
        first = pd.read_parquet(archive.open(f"{CASE}/observed-0.parquet"))
        for index in range(32):
            data = pd.read_parquet(archive.open(f"{CASE}/observed-{index}.parquet"))
            for method, estimate, se in estimates(
                data,
                "TE-S15",
                question["endpoint_id"],
                question["assessment_time"],
                0,
            ):
                margin = norm.ppf(0.975) * se
                rows.append(
                    dict(
                        case=CASE,
                        series="TE-S15",
                        state="favourable",
                        seed=SEED + index,
                        method=method,
                        estimate=estimate,
                        lower=estimate - margin,
                        upper=estimate + margin,
                        complete=True,
                        participants=len(data),
                    )
                )
        # References enter only after the observed-data fits above are finished.
        for index in range(32, 96):
            totals = pd.read_csv(archive.open(f"{CASE}/complete-{index}.csv")).set_index("assigned_arm")
            risks = totals["sum"] / totals["count"]
            reference_draws.append(dict(seed=SEED + index, contrast=float(risks.active - risks.control)))
    draws = pd.DataFrame(reference_draws)
    reference, scale = float(draws.contrast.mean()), float(draws.contrast.std(ddof=1))
    reference_se = scale / np.sqrt(len(draws))
    results = pd.DataFrame(rows).assign(reference=reference, reference_se=reference_se)
    results["error"] = results.estimate - reference
    original = pd.read_csv(source / "mechanisms/study_estimates.csv")
    original = original.loc[original.case.eq(CASE)].sort_values(["seed", "method"])
    np.testing.assert_allclose(
        results.sort_values(["seed", "method"])[["estimate", "lower", "upper", "reference"]],
        original[["estimate", "lower", "upper", "reference"]],
        rtol=0,
        atol=1e-12,
    )
    references = pd.DataFrame(
        [
            dict(
                case=CASE,
                reference_group=CASE,
                reference=reference,
                reference_se=reference_se,
                empirical_reference=reference,
                empirical_reference_se=reference_se,
                reference_worlds=64,
                analysis_worlds=32,
            )
        ]
    )
    _, summaries = summarize_repetitions(results, references)
    scored = score_rows(results, scale)
    summaries = summaries.merge(
        scored.groupby("method").recovery_score.mean().rename("mean_recovery_score"),
        on="method",
    )
    selected = scored.loc[scored.seed.eq(SEED)]
    margin = t.ppf(0.975, 63) * reference_se
    worked = WorkedStudy(
        participants=len(first),
        adjudicated_participants=int(first.validation_selected.sum()),
        reference=reference,
        reference_se=reference_se,
        reference_lower=reference - margin,
        reference_upper=reference + margin,
        scale=scale,
        estimates=[
            ScoredEstimate.model_validate({key: row[key] for key in ScoredEstimate.model_fields})
            for row in selected.to_dict(orient="records")
        ],
        source_paths=[
            "worked-study-estimates.csv",
            "repeated-study-scored.csv",
            "recovery-summary.csv",
            "reference-draws.csv",
            "worked-study-observed.csv",
            "worked-study-strata.csv",
            "observation-transitions.csv",
            "study.json",
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "study.json").write_bytes(study_bytes)
    (output / "worked-study.json").write_text(worked.model_dump_json(indent=2) + "\n", encoding="utf-8")
    selected.to_csv(output / "worked-study-estimates.csv", index=False)
    scored.to_csv(output / "repeated-study-scored.csv", index=False)
    summaries.to_csv(output / "recovery-summary.csv", index=False)
    draws.to_csv(output / "reference-draws.csv", index=False)
    first.to_csv(output / "worked-study-observed.csv", index=False)
    transitions = (
        first.loc[first.validation_selected]
        .groupby(["assigned_arm", "adjudicated_endpoint", question["endpoint_id"]])
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [["active", "control"], [0, 1], [0, 1]],
                names=["assigned_arm", "adjudicated_endpoint", question["endpoint_id"]],
            ),
            fill_value=0,
        )
        .rename("participants")
        .reset_index()
    )
    transitions.to_csv(output / "observation-transitions.csv", index=False)
    strata = first.groupby(["assigned_arm", "validation_stratum"]).agg(
        participants=("participant_id", "size"),
        adjudicated=("validation_selected", "sum"),
        routine_response=(question["endpoint_id"], "mean"),
        adjudicated_response=("adjudicated_endpoint", "mean"),
    )
    strata["arm_weight"] = strata.participants / strata.participants.groupby(level=0).transform("sum")
    strata["observed_validation_fraction"] = strata.adjudicated / strata.participants
    strata.to_csv(output / "worked-study-strata.csv")
    plot_baseline_anchors(first, output)
    for compact in (False, True):
        plot_worked(worked, output, compact=compact)
    return worked


def main() -> None:
    """Export a fixed paired example, checking every fit and scoring equation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worked = export_worked(args.source.resolve(strict=True), args.output)
    print(worked.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
