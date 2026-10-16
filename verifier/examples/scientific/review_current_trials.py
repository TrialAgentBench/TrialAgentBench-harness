"""Describe trial mechanisms in one record set per current clinical study.

The archive pairs C2 analysis-ready tables with C4 clean source records from the
same study. It contains participant-facing data only. Context views are not
counted as independent studies. No evaluator reference enters these summaries.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PETROL, OCHRE = "#226879", "#B87525"
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def _read(archive: zipfile.ZipFile, study: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(archive.read(f"{study}/{name}.parquet")))


def _icc(values: pd.Series, groups: pd.Series) -> float:
    """Return the one-way unequal-size ANOVA intraclass correlation."""
    frame = pd.DataFrame({"y": values, "g": groups}).dropna()
    counts = frame.groupby("g").y.count()
    if len(counts) < 2 or len(frame) <= len(counts):
        return float("nan")
    means = frame.groupby("g").y.mean()
    between = float((counts * (means - frame.y.mean()) ** 2).sum() / (len(counts) - 1))
    within = float(((frame.y - frame.g.map(means)) ** 2).sum() / (len(frame) - len(counts)))
    size = (len(frame) - float((counts**2).sum()) / len(frame)) / (len(counts) - 1)
    denominator = between + (size - 1) * within
    return (between - within) / denominator if denominator > 0 else float("nan")


def summarize_records(root: Path, output: Path) -> None:
    """Calculate observable patterns from the complete bundled study archive.

    Parameters
    ----------
    root
        Directory containing current-trial-records.zip.
    output
        Directory receiving CSV summaries and source identification.

    Raises
    ------
    ValueError
        If a terminal endpoint cannot be identified uniquely per participant.
    """
    output.mkdir(parents=True, exist_ok=True)
    census, recruitment, clinical, observation, exposure, allocation, monitoring = [], [], [], [], [], [], []
    with zipfile.ZipFile(root / "current-trial-records.zip") as archive:
        studies = pd.read_csv(archive.open("studies.csv"))
        for _, item in studies.iterrows():
            p = _read(archive, item.study, "participant_records")
            data = _read(archive, item.study, "analysis")
            terminal = (
                data.loc[np.isclose(data.occurrence_time, item.assessment_horizon_days)]
                if "occurrence_time" in data
                else data
            )
            if not terminal.participant_id.is_unique:
                raise ValueError(f"Terminal endpoint has duplicate participants: {item.study}")
            terminal = p.merge(
                terminal.drop(columns=[c for c in p if c != "participant_id" and c in terminal]),
                on="participant_id",
                how="left",
                validate="one_to_one",
            )
            common = {
                "study": item.study,
                "cell": item.cell,
                "family_id": item.family_id,
                "paired_world": item.paired_world,
            }
            endpoint = (
                "routine_endpoint"
                if "routine_endpoint" in terminal
                else ("primary_endpoint" if "primary_endpoint" in terminal else str(item.endpoint_id))
            )
            known = (
                (terminal.event_observed | terminal.observed_time.ge(item.assessment_horizon_days))
                if "event_observed" in terminal
                else terminal[endpoint].notna()
            )
            sizes = p.groupby("site_id").size()
            row = {
                **item.to_dict(),
                "participants": len(p),
                "occupied_sites": len(sizes),
                "active_fraction": float(p.assigned_arm.eq("active").mean()),
                "enrolment_p10_day": p.enrollment_day.quantile(0.1),
                "enrolment_p90_day": p.enrollment_day.quantile(0.9),
                "site_size_cv": sizes.std() / sizes.mean(),
                "final_known_fraction": known.mean(),
                "routine_recorded_fraction": known.mean(),
                "routine_endpoint_field": endpoint,
                "adjudicated_fraction": (
                    terminal.adjudicated_endpoint.notna().mean() if "adjudicated_endpoint" in terminal else np.nan
                ),
                "age_median": p.age_years.median(),
                "records_per_participant": len(data) / len(p),
            }
            if "adjudicated_endpoint" in terminal:
                paired_validation = terminal[[endpoint, "adjudicated_endpoint"]].dropna()
                difference = paired_validation[endpoint].astype(float) - paired_validation.adjudicated_endpoint.astype(
                    float
                )
                row["validation_agreement_fraction"] = difference.eq(0).mean()
                row["validation_mean_absolute_difference"] = difference.abs().mean()
            if "cluster_id" in p:
                row["randomized_clusters"] = p.cluster_id.nunique()
                residual = terminal[endpoint].astype(float) - terminal.groupby("assigned_arm")[endpoint].transform(
                    "mean"
                )
                row["arm_adjusted_outcome_icc"] = _icc(residual, terminal.cluster_id)
            if "treatment_discontinuation_time" in data:
                active = data.loc[data.assigned_arm.eq("active"), "treatment_discontinuation_time"]
                row["active_discontinuation_fraction"] = active.notna().mean()
            census.append(row)
            for probability in np.linspace(0, 1, 21):
                recruitment.append(
                    {
                        **common,
                        "kind": "participant_enrolment",
                        "fraction": probability,
                        "day": p.enrollment_day.quantile(probability),
                    }
                )
                recruitment.append(
                    {
                        **common,
                        "kind": "site_first_enrolment",
                        "fraction": probability,
                        "day": p.groupby("site_id").enrollment_day.min().quantile(probability),
                    }
                )
            for field in [
                c
                for c in p.columns.astype(str)
                if c == "age_years"
                or c.startswith("baseline_")
                or c.startswith("years_since_")
                or c == "severe_exacerbations_previous_year"
            ]:
                values = p[field].astype(float)
                for q in QUANTILES:
                    clinical.append(
                        {
                            **common,
                            "field": field,
                            "statistic": "quantile",
                            "quantile": q,
                            "value": values.quantile(q),
                            "n": values.notna().sum(),
                        }
                    )
                paired = terminal.loc[known, [field, endpoint]].dropna()
                if paired[field].nunique() > 1 and paired[endpoint].nunique() > 1:
                    clinical.append(
                        {
                            **common,
                            "field": field,
                            "statistic": "outcome_pearson_r",
                            "quantile": np.nan,
                            "value": paired[field].astype(float).corr(paired[endpoint].astype(float)),
                            "n": len(paired),
                        }
                    )
            visits = (
                data.groupby(["occurrence_time", "assigned_arm"])
                if "occurrence_time" in data
                else [((item.assessment_horizon_days, arm), frame) for arm, frame in terminal.groupby("assigned_arm")]
            )
            for (day, arm), frame in visits:
                observed = (
                    (frame.event_observed | frame.observed_time.ge(item.assessment_horizon_days))
                    if "event_observed" in frame
                    else frame[endpoint].notna()
                )
                observation.append(
                    {
                        **common,
                        "day": day,
                        "arm": arm,
                        "participants": len(frame),
                        "known": int(observed.sum()),
                        "observed_fraction": observed.mean(),
                        "routine_recorded_fraction": observed.mean(),
                        "routine_endpoint_field": endpoint,
                        "adjudicated": (
                            int(frame.adjudicated_endpoint.notna().sum())
                            if "adjudicated_endpoint" in frame
                            else np.nan
                        ),
                        "adjudicated_fraction": (
                            frame.adjudicated_endpoint.notna().mean() if "adjudicated_endpoint" in frame else np.nan
                        ),
                        "outcome_mean": frame.loc[observed, endpoint].astype(float).mean(),
                        "outcome_sd": frame.loc[observed, endpoint].astype(float).std(),
                        "retrieved": int(frame.endpoint_recovered.sum()) if "endpoint_recovered" in frame else 0,
                    }
                )
            received = _read(archive, item.study, "exposure_records").merge(
                p[["participant_id", "assigned_arm"]], on="participant_id", validate="many_to_one"
            )
            for (day, arm), frame in received.groupby(["occurrence_time", "assigned_arm"]):
                exposure.append(
                    {
                        **common,
                        "day": day,
                        "assigned_arm": arm,
                        "records": len(frame),
                        "received_active_fraction": frame.received_arm.eq("active").mean(),
                    }
                )
            if "cluster_id" in p:
                group_columns = ["cluster_id", "calendar_period"] if "calendar_period" in p else ["cluster_id"]
                for group, frame in terminal.groupby(group_columns):
                    group = group if isinstance(group, tuple) else (group,)
                    allocation.append(
                        {
                            **common,
                            "cluster": group[0],
                            "period": group[1] if len(group) > 1 else 0,
                            "participants": len(frame),
                            "active_fraction": frame.assigned_arm.eq("active").mean(),
                            "outcome_fraction": frame[endpoint].astype(float).mean(),
                            "enrolment_mean_day": frame.enrollment_day.mean(),
                            "switch_day": frame.switch_time.iloc[0] if "switch_time" in frame else np.nan,
                        }
                    )
            member = f"{item.study}/monitoring_results.parquet"
            if member in archive.namelist():
                for record in _read(archive, item.study, "monitoring_results").to_dict("records"):
                    monitoring.append({**common, **{str(key): value for key, value in record.items()}})
        (output / "SOURCE.md").write_bytes(archive.read("SOURCE.md"))
    for name, rows in (
        ("trial_census", census),
        ("recruitment", recruitment),
        ("clinical_distributions", clinical),
        ("observation", observation),
        ("treatment_exposure", exposure),
        ("allocation", allocation),
        ("monitoring", monitoring),
    ):
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)


def plot_patterns(output: Path) -> None:
    """Render readable mechanism figures directly from the exported tables."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    census = pd.read_csv(output / "trial_census.csv")
    recruitment = pd.read_csv(output / "recruitment.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for family, colour in (("TE-S01", PETROL), ("TE-S09", OCHRE)):
        study = census.loc[census.family_id.eq(family)].sort_values("study").iloc[0]
        points = recruitment.loc[recruitment.study.eq(study.study)]
        for ax, kind in zip(axes, ("participant_enrolment", "site_first_enrolment"), strict=True):
            line = points.loc[points.kind.eq(kind)]
            ax.plot(
                line.day, line.fraction * 100, color=colour, linewidth=2.5, label=study.study_name.split(" trial")[0]
            )
            ax.set(xlabel="Days from programme start", ylim=(0, 102), ylabel="Cumulative percentage")
            ax.grid(axis="y", alpha=0.15)
    axes[0].set_title("Patients enter over time", loc="left")
    axes[1].set_title("Sites begin contributing at different times", loc="left")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(output / "recruitment.svg", transparent=True)
    fig.savefig(output / "recruitment.png", dpi=180, transparent=True)
    fig.set_size_inches(4.2, 8.0)
    grid = fig.add_gridspec(2, 1)
    for number, ax in enumerate(axes):
        ax.set_subplotspec(grid[number, 0])
        ax.set_title(fill(ax.get_title(loc="left") or ax.get_title(), width=31), loc="left")
        ax.set_title("")
        if ax.get_legend() is not None:
            for label in ax.get_legend().get_texts():
                label.set_fontsize(10)
    fig.savefig(output / "recruitment-compact.svg", transparent=True)
    plt.close(fig)
    allocation = pd.read_csv(output / "allocation.csv")
    study = census.loc[census.cell.eq("TE-S08-A2")].sort_values("study").iloc[0]
    data = allocation.loc[allocation.study.eq(study.study)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for switch, frame in data.groupby("switch_day"):
        sums = (
            frame.assign(active=frame.active_fraction * frame.participants)
            .groupby("period")[["active", "participants"]]
            .sum()
        )
        axes[0].step(
            sums.index,
            sums.active / sums.participants * 100,
            where="mid",
            marker="o",
            label=f"Switch on day {switch:g}",
        )
    axes[0].set(
        xlabel="Calendar period",
        ylabel="Assigned pathway (%)",
        ylim=(-3, 103),
        title="Maternity units change care in sequence",
    )
    axes[0].legend(frameon=False, fontsize=9)
    for arm, colour in ((0, PETROL), (1, OCHRE)):
        frame = data.loc[data.active_fraction.eq(arm)].copy()
        frame["events"] = frame.outcome_fraction * frame.participants
        sums = frame.groupby("period")[["events", "participants"]].sum()
        axes[1].plot(
            sums.index,
            100 * sums.events / sums.participants,
            marker="o",
            color=colour,
            label="Pathway" if arm else "Usual care",
        )
    axes[1].set(
        xlabel="Calendar period",
        ylabel="Observed severe morbidity (%)",
        title="Treatment and time both distinguish observations",
    )
    axes[0].set_xticks([0, 1, 2])
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=9)
    fig.savefig(output / "rollout.svg", transparent=True)
    fig.savefig(output / "rollout.png", dpi=180, transparent=True)
    fig.set_size_inches(4.2, 8.0)
    grid = fig.add_gridspec(2, 1)
    for number, ax in enumerate(axes):
        ax.set_subplotspec(grid[number, 0])
        ax.set_title(fill(ax.get_title(loc="left") or ax.get_title(), width=31), loc="left")
        ax.set_title("")
        if ax.get_legend() is not None:
            for label in ax.get_legend().get_texts():
                label.set_fontsize(10)
    fig.savefig(output / "rollout-compact.svg", transparent=True)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for ax, family, title, colour in zip(
        axes, ("TE-S15", "TE-S16"), ("Tumour response", "Severe asthma attacks"), (PETROL, OCHRE), strict=True
    ):
        studies = census.loc[census.family_id.eq(family) & census.assumption.eq("A3")].sort_values("paired_world")
        positions = np.arange(len(studies))
        ax.barh(
            positions,
            100 * studies.routine_recorded_fraction,
            color="#d8e2e5",
            height=0.6,
            label="Routine measurement",
        )
        ax.barh(positions, 100 * studies.adjudicated_fraction, color=colour, height=0.6, label="Also adjudicated")
        for position, fraction in zip(positions, studies.adjudicated_fraction, strict=True):
            ax.text(100 * fraction + 2, position, f"{100 * fraction:.1f}%", va="center", fontsize=10)
        ax.set(
            yticks=positions,
            yticklabels=[str(i + 1) for i in positions],
            xlim=(0, 103),
            xlabel="Patients with each measurement (%)",
            ylabel="Study",
            title=title,
        )
        ax.invert_yaxis()
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.4), fontsize=10)
    fig.savefig(output / "followup.svg", transparent=True)
    fig.savefig(output / "followup.png", dpi=180, transparent=True)
    fig.set_size_inches(4.2, 8.0)
    grid = fig.add_gridspec(2, 1)
    for number, ax in enumerate(axes):
        ax.set_subplotspec(grid[number, 0])
        ax.set_title(fill(ax.get_title(loc="left") or ax.get_title(), width=31), loc="left")
        ax.set_title("")
        if ax.get_legend() is not None:
            for label in ax.get_legend().get_texts():
                label.set_fontsize(10)
    fig.savefig(output / "followup-compact.svg", transparent=True)
    plt.close(fig)


def run(root: Path, output: Path) -> None:
    """Reproduce the current-trial characterization from its public archive."""
    summarize_records(root, output)
    plot_patterns(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
