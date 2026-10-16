"""Expose clinical distributions, evidence structures and same-target analyses.

The family archive contains September teaching records, not the pending scored
release. Calculations use disclosed records and ordinary statistical libraries;
references enter only after observed-data analyses have finished.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from review_families import design_controls
from review_mechanisms import summarize_repetitions
from review_scoring import numerical_loss
from review_trialeval import fixed_trial

TEACHING = "september-teaching"
PROBE = "october-current-probe"
PETROL, OCHRE, GREY = "#226879", "#B87525", "#687279"
DESIGNS = {
    "TE-S03": ("pragmatic", "D2"),
    "TE-S04": ("pragmatic", "D2"),
    "TE-S05": ("standardization", "D3"),
    "TE-S06": ("endpoint validation", "D3"),
    "TE-S07": ("cluster parallel", "D4"),
    "TE-S08": ("stepped wedge", "D4"),
    "TE-S09": ("group sequential", "D4"),
}
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def _table(archive: zipfile.ZipFile, cell: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(archive.read(f"{cell}/{name}.parquet")))


def project_records(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Project all teaching cells at participant, record and clinical units.

    Parameters
    ----------
    root
        Bundled scientific examples, including the disclosed family archive.

    Returns
    -------
    tuple of pandas.DataFrame
        Complete cell census, tidy clinical summaries and selected design marks.

    Raises
    ------
    ValueError
        If a participant is duplicated or a final endpoint has multiple records.
    """
    labels = pd.read_csv(root / "mechanisms/clinical_settings.csv").set_index("family_id").clinical_setting
    census_rows, clinical_rows, design_rows = [], [], []
    with zipfile.ZipFile(root / "family-records.zip") as archive:
        cells = pd.read_csv(archive.open("cells.csv"), keep_default_na=False)
        for cell in cells.itertuples(index=False):
            identity = str(cell.cell)
            family_id, assumption = identity.rsplit("-", 1)
            horizon = float(cast(float, cell.horizon))
            participants = _table(archive, identity, "participant_records")
            data = _table(archive, identity, "analysis")
            if not participants.participant_id.is_unique:
                raise ValueError(f"Participant identities are duplicated: {identity}")
            terminal = data.loc[np.isclose(data.occurrence_time, horizon)] if "occurrence_time" in data else data
            if not terminal.participant_id.is_unique:
                raise ValueError(f"Final endpoint has multiple participant records: {identity}")
            if "event_observed" in terminal:
                known = terminal.event_observed | terminal.observed_time.ge(horizon)
                event_fraction = float(terminal.event_observed.mean())
                median_followup = float(terminal.observed_time.median())
            else:
                endpoint = "adjudicated_endpoint" if "adjudicated_endpoint" in terminal else str(cell.endpoint)
                known = terminal[endpoint].notna()
                event_fraction = np.nan
                median_followup = horizon
            clusters = participants.groupby("cluster_id").size() if "cluster_id" in participants else None
            design, tier = DESIGNS.get(family_id, ("individual parallel", "D1"))
            census_rows.append(
                {
                    "source_id": TEACHING,
                    "cell": identity,
                    "family_id": family_id,
                    "clinical_label": str(labels.loc[family_id]),
                    "design": design,
                    "design_tier": tier,
                    "assumption": assumption,
                    "question": cell.question,
                    "condition_description": cell.condition,
                    "endpoint": cell.endpoint,
                    "endpoint_family": cell.family,
                    "effect_measure": cell.scale,
                    "seed": cell.seed,
                    "participants": len(participants),
                    "planned_participants": cell.planned_n,
                    "active_participants": int(participants.assigned_arm.eq("active").sum()),
                    "control_participants": int(participants.assigned_arm.eq("control").sum()),
                    "occupied_sites": participants.site_id.nunique(),
                    "randomized_clusters": len(clusters) if clusters is not None else 0,
                    "horizon_days": horizon,
                    "records_per_participant": len(data) / len(participants),
                    "final_observed_fraction": float(known.sum() / len(participants)),
                    "observed_event_fraction": event_fraction,
                    "median_followup_days": median_followup,
                    "cluster_size_cv": float(clusters.std() / clusters.mean()) if clusters is not None else np.nan,
                    "paired_family_seed": f"{family_id}:{cell.seed}",
                }
            )
            common = {"source_id": TEACHING, "cell": identity, "family_id": family_id}
            laws = pd.read_csv(archive.open(f"{identity}/baseline_law.csv"), keep_default_na=False)
            for law in laws.itertuples(index=False):
                values = participants[str(law.field)].dropna().astype(float)
                metadata = {
                    **common,
                    "domain": "baseline",
                    "field": law.field,
                    "label": law.label,
                    "unit": law.unit,
                    "arm": "all",
                    "group": "all participants",
                    "n": len(values),
                    "source_role": "generated; marginal anchor only where identified",
                    "source_ids": law.source_ids,
                }
                for probability in QUANTILES:
                    clinical_rows.append(
                        {**metadata, "statistic": f"quantile_{probability:g}", "value": values.quantile(probability)}
                    )
                for statistic, value in (
                    ("mean", values.mean()),
                    ("sd", values.std()),
                    ("recording_resolution", law.recording_resolution),
                ):
                    clinical_rows.append({**metadata, "statistic": statistic, "value": value})
                if identity == "TE-S16-A1" and law.field in ("baseline_body_mass_index", "baseline_eosinophils"):
                    frequencies = values.value_counts().sort_index()
                    for measurement, probability in (frequencies.cumsum() / len(values)).items():
                        clinical_rows.append(
                            {**metadata, "statistic": "ecdf", "group": str(measurement), "value": probability}
                        )
            categories = pd.read_csv(archive.open(f"{identity}/baseline_categories.csv"), keep_default_na=False)
            for category in categories.itertuples(index=False):
                clinical_rows.append(
                    {
                        **common,
                        "domain": "baseline",
                        "field": category.field,
                        "label": category.label,
                        "unit": "proportion",
                        "arm": "all",
                        "group": category.category,
                        "statistic": "proportion",
                        "value": float(participants[str(category.field)].eq(category.category).mean()),
                        "n": len(participants),
                        "source_role": "generated; marginal anchor only where identified",
                        "source_ids": category.source_ids,
                    }
                )
            for arm, frame in terminal.groupby("assigned_arm"):
                endpoint = "adjudicated_endpoint" if "adjudicated_endpoint" in frame else str(cell.endpoint)
                values = (
                    frame.event_observed.astype(float)
                    if "event_observed" in frame
                    else frame[endpoint].dropna().astype(float)
                )
                for statistic, value in (("mean", values.mean()), ("sd", values.std())):
                    clinical_rows.append(
                        {
                            **common,
                            "domain": "outcome",
                            "field": cell.endpoint,
                            "label": str(cell.endpoint).replace("_", " "),
                            "unit": (
                                "proportion"
                                if cell.family in ("binary", "time_to_event")
                                else "events" if cell.family == "count" else "endpoint units"
                            ),
                            "arm": arm,
                            "group": "final observed records",
                            "statistic": statistic,
                            "value": value,
                            "n": len(values),
                            "source_role": "generated observed outcome",
                            "source_ids": "",
                        }
                    )
            if identity == "TE-S16-A1":
                for arm, frame in data.groupby("assigned_arm"):
                    frequencies = frame[str(cell.endpoint)].value_counts().sort_index()
                    for count, frequency in frequencies.items():
                        clinical_rows.append(
                            {
                                **common,
                                "domain": "outcome",
                                "field": cell.endpoint,
                                "label": "Severe exacerbations during 365 days",
                                "unit": "proportion",
                                "arm": arm,
                                "group": str(count),
                                "statistic": "frequency",
                                "value": frequency / len(frame),
                                "n": int(frequency),
                                "source_role": "generated observed outcome",
                                "source_ids": "",
                            }
                        )
                    for severity, severity_group in frame.groupby(
                        frame.severe_exacerbations_previous_year.clip(upper=5)
                    ):
                        clinical_rows.append(
                            {
                                **common,
                                "domain": "dependence",
                                "field": cell.endpoint,
                                "label": "Prior severity and subsequent exacerbations",
                                "unit": "events per participant",
                                "arm": arm,
                                "group": "5+" if severity == 5 else str(severity),
                                "statistic": "conditional_mean",
                                "value": severity_group[str(cell.endpoint)].mean(),
                                "n": len(severity_group),
                                "source_role": "authored joint mechanism; not fitted to QUEST",
                                "source_ids": "",
                            }
                        )
            if identity == "TE-S08-A2":
                for (rollout_cluster, rollout_period, rollout_arm), frame in data.groupby(
                    ["cluster_id", "calendar_period", "assigned_arm"]
                ):
                    design_rows.append(
                        {
                            "cell": identity,
                            "panel": "rollout",
                            "group": str(rollout_cluster),
                            "period": rollout_period,
                            "arm": rollout_arm,
                            "metric": "assigned_active",
                            "value": float(rollout_arm == "active"),
                            "n": len(frame),
                            "unit": "proportion",
                            "method": "observed allocation",
                            "lower": np.nan,
                            "upper": np.nan,
                            "target_role": "randomized unit-period assignment",
                        }
                    )
                for (rollout_period, rollout_arm), frame in data.groupby(["calendar_period", "assigned_arm"]):
                    for metric, value, unit in (
                        ("births", len(frame), "participants"),
                        ("observed_events", frame.event_observed.sum(), "events"),
                        ("observed_event_risk", frame.event_observed.mean(), "proportion"),
                    ):
                        design_rows.append(
                            {
                                "cell": identity,
                                "panel": "calendar",
                                "group": "all units",
                                "period": rollout_period,
                                "arm": rollout_arm,
                                "metric": metric,
                                "value": value,
                                "n": len(frame),
                                "unit": unit,
                                "method": "observed records",
                                "lower": np.nan,
                                "upper": np.nan,
                                "target_role": "descriptive observed risk; not treatment effect",
                            }
                        )
                ordered = data.sort_values("entry_time")
                for position in (0, len(ordered) // 2, len(ordered) - 1):
                    mother = ordered.iloc[position]
                    delivery = float(mother.entry_time)
                    for metric, value in (
                        ("delivery_day", delivery),
                        ("followup_end_calendar_day", delivery + float(mother.observed_time)),
                        ("planned_followup_end_calendar_day", delivery + horizon),
                    ):
                        design_rows.append(
                            {
                                "cell": identity,
                                "panel": "clocks",
                                "group": str(mother.participant_id),
                                "period": mother.calendar_period,
                                "arm": mother.assigned_arm,
                                "metric": metric,
                                "value": value,
                                "n": 1,
                                "unit": "calendar day",
                                "method": "actual participant record",
                                "lower": np.nan,
                                "upper": np.nan,
                                "target_role": "42-day individual follow-up after delivery",
                            }
                        )
            if identity == "TE-S12-A4":
                for visit, frame in data.groupby("occurrence_time"):
                    for visit_metric, value in (
                        ("scheduled", len(frame)),
                        ("observed", frame.observed.sum()),
                        ("recovered", frame.endpoint_recovered.sum()),
                    ):
                        design_rows.append(
                            {
                                "cell": identity,
                                "panel": "visits",
                                "group": "all participants",
                                "period": visit,
                                "arm": "all",
                                "metric": visit_metric,
                                "value": value,
                                "n": len(participants),
                                "unit": "participants",
                                "method": "source visit records",
                                "lower": np.nan,
                                "upper": np.nan,
                                "target_role": "scheduled and observed visits differ",
                            }
                        )
            if identity == "TE-S09-A1":
                monitoring = _table(archive, identity, "monitoring")
                for look in monitoring.itertuples(index=False):
                    for metric, value in (
                        ("z_statistic", look.z_statistic),
                        ("efficacy_z_boundary", look.efficacy_z_boundary),
                        ("information_fraction", look.information_fraction),
                        ("control_events", look.control_events),
                        ("active_events", look.treatment_events),
                    ):
                        design_rows.append(
                            {
                                "cell": identity,
                                "panel": "monitoring",
                                "group": look.decision,
                                "period": look.information_time,
                                "arm": "all",
                                "metric": metric,
                                "value": value,
                                "n": cast(int, look.control_participants) + cast(int, look.treatment_participants),
                                "unit": (
                                    "z units"
                                    if "z_" in metric
                                    else "proportion" if metric == "information_fraction" else "events"
                                ),
                                "method": look.look_id,
                                "lower": np.nan,
                                "upper": np.nan,
                                "target_role": "two executed looks; three planned",
                            }
                        )
    # These reported summaries anchor selected marginals only. No source density
    # or outcome distribution is inferred from their mean, SD or proportion.
    for field, label, statistic, value, unit, group in (
        ("baseline_body_mass_index", "Body mass index", "mean", 29.20, "kg/m2", "overall population"),
        ("baseline_body_mass_index", "Body mass index", "sd", 6.77, "kg/m2", "overall population"),
        ("baseline_eosinophils", "Blood eosinophils", "mean", 360.0, "cells/uL", "overall population"),
        ("baseline_eosinophils", "Blood eosinophils", "sd", 366.0, "cells/uL", "overall population"),
        ("sex", "Recorded female sex", "proportion", 1197 / 1902, "proportion", "female"),
        ("high_dose_ics", "High-dose inhaled corticosteroids", "proportion", 979 / 1902, "proportion", "yes"),
    ):
        clinical_rows.append(
            {
                "source_id": "TAB-SRC-081",
                "cell": "QUEST",
                "family_id": "TE-S16",
                "domain": "published baseline summary",
                "field": field,
                "label": label,
                "unit": unit,
                "arm": "all",
                "group": group,
                "statistic": statistic,
                "value": value,
                "n": 1902,
                "source_role": "construction anchor; distinct eligibility",
                "source_ids": "TAB-SRC-081",
            }
        )
    return pd.DataFrame(census_rows), pd.DataFrame(clinical_rows), pd.DataFrame(design_rows).assign(source_id=TEACHING)


def design_analyses(root: Path) -> pd.DataFrame:
    """Reuse the existing observed-record estimator and repeated-study summaries."""
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(root / "family-records.zip") as archive:
            cells = pd.read_csv(archive.open("cells.csv"))
            selected = cells.loc[cells.cell.eq("TE-S08-A2")]
            for identifier in selected.cell:
                path = directory / str(identifier)
                path.mkdir()
                (path / "analysis.parquet").write_bytes(archive.read(f"{identifier}/analysis.parquet"))
            intervals = design_controls(directory, selected)
    rows = []
    for result in intervals.itertuples(index=False):
        rows.append(
            {
                "source_id": TEACHING,
                "cell": result.cell,
                "panel": "rollout analysis",
                "group": "one realized trial",
                "period": 42,
                "arm": "active minus control",
                "metric": "risk_difference",
                "value": result.estimate,
                "n": 7200,
                "unit": "proportion",
                "method": result.analysis,
                "lower": result.lower,
                "upper": result.upper,
                "target_role": "period-adjusted target; no disclosed population reference",
            }
        )
    estimates = pd.read_csv(root / "mechanisms/study_estimates.csv")
    references = pd.read_csv(root / "mechanisms/population_references.csv")
    case = "TE-S07-A1-favourable"
    _, summaries = summarize_repetitions(
        estimates.loc[estimates.case.eq(case)], references.loc[references.case.eq(case)]
    )
    for result in summaries.itertuples(index=False):
        for metric, value, lower, upper, unit in (
            ("coverage", result.coverage, result.coverage_lower, result.coverage_upper, "proportion"),
            ("mean_interval_width", result.mean_interval_width, np.nan, np.nan, "proportion"),
            ("bias", result.bias, result.bias_lower, result.bias_upper, "proportion"),
        ):
            rows.append(
                {
                    "source_id": "september-repeated-studies",
                    "cell": case,
                    "panel": "cluster recovery",
                    "group": "32 paired analysis studies",
                    "period": 180,
                    "arm": "active minus control",
                    "metric": metric,
                    "value": value,
                    "n": result.studies,
                    "unit": unit,
                    "method": result.method,
                    "lower": lower,
                    "upper": upper,
                    "target_role": "population reference from 64 separate studies",
                }
            )
    return pd.DataFrame(rows)


def method_results(root: Path, output: Path) -> pd.DataFrame:
    """Expose the two executed current-record analyses of one fixed clinical target."""
    if not (output / "trialeval_analyses.csv").is_file():
        fixed_trial(root / "trialeval", output)
    analyses = pd.read_csv(output / "trialeval_analyses.csv")
    selected = analyses.loc[analyses.analysis.str.startswith("Current records:")].copy()
    reference = pd.read_csv(output / "reference_precision.csv").iloc[0]
    observed = pd.read_csv(root / "trialeval/analysis_ready.csv")
    if len(selected) != 2 or not np.isfinite(selected[["estimate", "lower", "upper"]]).all().all():
        raise ValueError("Method plurality requires both finite current-record analyses")
    selected = selected.rename(
        columns={"analysis": "method", "scale_points": "reference_scale", "mean_loss": "numerical_loss"}
    )
    selected["source_id"] = "bundled-parkinson-trial"
    selected["case"] = "Parkinson Day-180 teaching trial"
    selected["target_id"] = "population-motor-disability-day180"
    selected["target_definition"] = (
        "Rehabilitation minus usual-care population mean motor-disability score at Day 180 in all randomized participants"
    )
    selected["reference_mcse"] = reference.monte_carlo_se
    selected["reference_lower"] = reference.reference_lower
    selected["reference_upper"] = reference.reference_upper
    selected["reference_worlds"] = int(reference.worlds)
    selected["point_loss"] = selected.absolute_error / selected.reference_scale
    selected["noncoverage_penalty"] = 40 * np.maximum(
        np.maximum(selected.lower - selected.reference, selected.reference - selected.upper), 0
    )
    selected["interval_loss"] = (selected.interval_width + selected.noncoverage_penalty) / selected.reference_scale
    selected["participants"] = len(observed)
    selected["observed_n"] = int(observed.motor_disability_score.notna().sum())
    selected["unit"] = "motor-disability score points"
    calculated = np.array(
        [
            numerical_loss(
                cast(float, row.estimate),
                cast(float, row.lower),
                cast(float, row.upper),
                cast(float, row.reference),
                cast(float, row.reference_scale),
            )
            for row in selected.itertuples(index=False)
        ]
    )
    np.testing.assert_allclose(calculated, selected.numerical_loss, rtol=0, atol=1e-12)
    return selected.drop(columns=["complete"])


def _save(figure: Figure, output: Path, name: str) -> None:
    for extension in ("svg", "pdf", "png"):
        figure.savefig(output / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(figure)


def _title(axis: Axes, text: str) -> None:
    axis.set_title(text, loc="left", fontsize=11, pad=12, weight="semibold")
    axis.spines[["top", "right"]].set_visible(False)


def plot_census(census: pd.DataFrame, output: Path) -> None:
    """Show every disclosed cell once, retaining family and condition identities."""
    census = census.sort_values(["design_tier", "design", "family_id", "assumption"], kind="stable")
    figure, axes = plt.subplots(1, 4, figsize=(12, 10), sharey=True)
    figure.subplots_adjust(left=0.16, right=0.97, bottom=0.12, top=0.88, wspace=0.35)
    y = np.arange(len(census))
    for axis, field, label in zip(
        axes,
        ("participants", "horizon_days", "records_per_participant", "final_observed_fraction"),
        (
            "Randomized participants",
            "Individual horizon (days)",
            "Outcome rows / participant",
            "Known final endpoint (%)",
        ),
        strict=True,
    ):
        values = census[field].to_numpy(dtype=float) * (100 if field == "final_observed_fraction" else 1)
        axis.scatter(values, y, color=PETROL, s=26, marker="o")
        axis.set_xlabel(label, labelpad=10)
        _title(
            axis,
            {
                "participants": "Participant number",
                "horizon_days": "Individual follow-up",
                "records_per_participant": "Observation burden",
                "final_observed_fraction": "Endpoint availability",
            }[field],
        )
        axis.grid(axis="y", color="#e7eaeb", linewidth=0.6)
        if field == "participants":
            axis.set_xscale("log")
        if field == "final_observed_fraction":
            axis.set_xlim(-3, 103)
        axis.set_ylim(len(census) - 0.5, -0.5)
    axes[0].set_yticks(y, census.cell + " · " + census.design_tier, fontsize=9)
    figure.suptitle(
        "Thirty disclosed clinical cells, seventeen families",
        x=0.14,
        y=0.97,
        ha="left",
        fontsize=17,
        weight="semibold",
    )
    figure.text(
        0.14,
        0.925,
        "September teaching archive · one realization per cell · same-family conditions share seeds",
        fontsize=10,
        color=GREY,
    )
    figure.text(
        0.14,
        0.025,
        "Rows are design/assumption cells, not independent release trials or information views.\nEvent-free follow-up to the horizon or an observed event establishes a known final survival endpoint.",
        fontsize=9,
        color=GREY,
    )
    _save(figure, output, "trial-census")


def _rollout(axis: Axes, evidence: pd.DataFrame) -> None:
    rollout = evidence.loc[evidence.panel.eq("rollout")]
    wide = rollout.pivot(index="group", columns="period", values="value")
    wide.index = wide.index.astype(int)
    wide = wide.sort_index(kind="stable")
    wide = wide.loc[wide[1].sort_values(ascending=False, kind="stable").index]
    axis.imshow(
        wide.to_numpy(),
        cmap=ListedColormap(("#eedbc1", PETROL)),
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
        extent=(0, 42, len(wide) + 0.5, 0.5),
    )
    axis.set(
        xticks=(7, 21, 35),
        xticklabels=("0–14", "14–28", "28–42"),
        yticks=(1, 10, 20, 30),
        yticklabels=("1", "10", "20", "30"),
        xlabel="Delivery calendar period (days)",
        ylabel="Randomized maternity units",
    )
    axis.set_yticks(np.arange(1.5, len(wide), 1), minor=True)
    axis.grid(which="minor", axis="y", color="white", linewidth=0.4)
    axis.tick_params(which="minor", left=False)
    _title(axis, "A  Units switch; births enter across time")
    axis.text(0.01, -0.43, "Usual care", transform=axis.transAxes, color=OCHRE, fontsize=9)
    axis.text(0.48, -0.43, "Postpartum pathway", transform=axis.transAxes, color=PETROL, fontsize=9)


def _calendar(axis: Axes, evidence: pd.DataFrame) -> None:
    calendar = evidence.loc[evidence.panel.eq("calendar") & evidence.metric.eq("observed_event_risk")]
    for arm, colour, marker, label in (("control", OCHRE, "s", "Usual care"), ("active", PETROL, "o", "Pathway")):
        frame = calendar.loc[calendar.arm.eq(arm)]
        x = frame.period.to_numpy(dtype=float)
        axis.plot(x, 100 * frame.value, color=colour, marker=marker, label=label)
        for row in frame.itertuples(index=False):
            events = evidence.loc[
                evidence.panel.eq("calendar")
                & evidence.metric.eq("observed_events")
                & evidence.period.eq(row.period)
                & evidence.arm.eq(arm),
                "value",
            ].iloc[0]
            axis.annotate(
                f"{int(events)} / {int(cast(int, row.n)):,}",
                (cast(float, row.period), 100 * cast(float, row.value)),
                xytext=(0, 8 if arm == "control" else -15),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set(
        xticks=(0, 1, 2),
        xticklabels=("0–14", "14–28", "28–42"),
        xlabel="Delivery calendar period (days)",
        ylabel="Observed severe morbidity (%)",
        ylim=(0.25, 1.6),
        xlim=(-0.3, 2.3),
    )
    _title(axis, "B  Observed risk during rollout")
    axis.legend(frameon=False, fontsize=8, loc="upper right")


def _clocks(axis: Axes, evidence: pd.DataFrame) -> None:
    clocks = (
        evidence.loc[evidence.panel.eq("clocks")]
        .pivot(index="group", columns="metric", values="value")
        .sort_values("delivery_day")
    )
    for position, (delivery, observed_end, planned_end) in enumerate(
        clocks[["delivery_day", "followup_end_calendar_day", "planned_followup_end_calendar_day"]].to_numpy(
            dtype=float
        )
    ):
        axis.plot([delivery, planned_end], [position, position], color=GREY, linewidth=1.5, linestyle="--")
        axis.plot([delivery, observed_end], [position, position], color=PETROL, linewidth=2)
        axis.scatter([delivery, observed_end], [position, position], color=PETROL, marker="|", s=100)
    axis.axvspan(0, 42, color=OCHRE, alpha=0.12)
    axis.set(
        xlabel="Calendar day",
        yticks=range(3),
        yticklabels=("First delivery", "Middle delivery", "Last delivery"),
        xlim=(0, 86),
        ylim=(-0.5, 3.0),
    )
    axis.invert_yaxis()
    _title(axis, "C  Follow-up has its own clock")
    axis.text(
        0.03,
        0.03,
        "Solid: observed; dashed: planned Day 42\nShading: rollout calendar",
        transform=axis.transAxes,
        fontsize=8,
        color=GREY,
    )


def _rollout_analysis(axis: Axes, evidence: pd.DataFrame) -> None:
    rows = evidence.loc[evidence.panel.eq("rollout analysis")]
    for position, (estimate, lower, upper) in enumerate(rows[["value", "lower", "upper"]].to_numpy(dtype=float)):
        axis.errorbar(
            100 * estimate,
            position,
            xerr=100 * np.array([[estimate - lower], [upper - estimate]]),
            color=OCHRE if position == 0 else PETROL,
            fmt="s" if position == 0 else "o",
            capsize=3,
        )
    axis.axvline(0, color=GREY, linestyle=":", linewidth=1)
    axis.set(
        yticks=(0, 1),
        yticklabels=("Unadjusted", "Period-aware;\ncluster SE"),
        xlabel="Risk difference (percentage points)",
        ylim=(-0.7, 1.7),
    )
    axis.invert_yaxis()
    _title(axis, "D  The analysis accounts for rollout")
    axis.text(
        0.03,
        0.02,
        "One study; 95% intervals. No population reference here.",
        transform=axis.transAxes,
        fontsize=8,
        color=GREY,
    )


def _visits(axis: Axes, evidence: pd.DataFrame) -> None:
    visits = evidence.loc[evidence.panel.eq("visits")]
    for metric, label, colour, style in (
        ("scheduled", "Scheduled", GREY, "--"),
        ("observed", "Observed", PETROL, "-"),
        ("recovered", "Randomly recovered", OCHRE, ":"),
    ):
        rows = visits.loc[visits.metric.eq(metric)]
        axis.plot(rows.period, rows.value, marker="o", color=colour, linestyle=style, label=label)
    axis.set(xlabel="Individual follow-up day", ylabel="Participants with a visit record", ylim=(-10, 320))
    _title(axis, "E  Scheduled and observed asthma visits")
    axis.legend(frameon=False, fontsize=8, loc="center right")


def _monitoring(axis: Axes, evidence: pd.DataFrame) -> None:
    looks = evidence.loc[evidence.panel.eq("monitoring")]
    for metric, label, colour, style in (
        ("z_statistic", "Observed comparison", PETROL, "-"),
        ("efficacy_z_boundary", "Efficacy boundary", OCHRE, "--"),
    ):
        rows = looks.loc[looks.metric.eq(metric)]
        axis.plot(rows.period, rows.value, color=colour, marker="o", linestyle=style, label=label)
    for row in looks.loc[looks.metric.eq("information_fraction")].itertuples(index=False):
        axis.text(
            cast(float, row.period), 0.85, f"{100 * cast(float, row.value):.0f}% information", fontsize=8, ha="center"
        )
    axis.set(
        xticks=(365, 540),
        xlabel="Trial calendar day at released look",
        ylabel="Comparative z statistic",
        ylim=(0.5, 3.65),
        xlim=(325, 580),
    )
    _title(axis, "F  The second look crossed its boundary")
    axis.legend(frameon=False, fontsize=8, loc="upper right")


def plot_assignment_atlas(census: pd.DataFrame, evidence: pd.DataFrame, output: Path) -> None:
    """Explain assignment level with schematics bound to three disclosed trials.

    Parameters
    ----------
    census
        Reconstructed trial-level participant and randomized-cluster counts.
    evidence
        Actual unit-period assignments and scheduled observation times.
    output
        Destination for editable diagrams and the selected census rows.

    Raises
    ------
    ValueError
        If any named trial or the two observed switch patterns is unavailable.
    """
    cells = ("TE-S12-A4", "TE-S07-A1", "TE-S08-A2")
    selected = census.loc[census.cell.isin(cells)].set_index("cell")
    if set(selected.index) != set(cells) or len(selected) != len(cells):
        raise ValueError("The atlas requires one census row for each named trial")
    selected.loc[list(cells)].reset_index().to_csv(output / "trial-assignment-atlas.csv", index=False)
    rollout = evidence.loc[evidence.panel.eq("rollout")].pivot(index="group", columns="period", values="value")
    patterns = rollout.drop_duplicates().to_numpy(dtype=float)
    if patterns.shape != (2, 3):
        raise ValueError("The atlas requires the two disclosed three-period rollout patterns")
    visits = sorted(evidence.loc[evidence.panel.eq("visits") & evidence.metric.eq("scheduled"), "period"].unique())
    ink, grey, petrol = "#23323d", "#aebdc4", "#226879"
    for compact in (False, True):
        width, height = (350, 1175) if compact else (960, 530)
        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            '<title id="title">Who receives treatment together changes the analysis</title>',
            '<desc id="desc">Schematics distinguish individual assignment from whole-home assignment. '
            "The rollout grid shows the two observed switch patterns in a maternity study. "
            "Actual trial counts are given below each example. Repeated visits belong to the same person.</desc>",
            '<style>text{font-family:Arial,"Liberation Sans",sans-serif;font-size:17px;fill:'
            + ink
            + "}.title{font-size:21px;font-weight:700}.small{font-size:15px;fill:#536370}.strong{font-weight:700}</style>",
            '<circle cx="19" cy="20" r="7" fill="' + petrol + '"/><text x="34" y="26">Treatment</text>',
            '<circle cx="152" cy="20" r="7" fill="' + grey + '"/><text x="167" y="26">Control</text>',
        ]
        for index, cell in enumerate(cells):
            x, y = (12, 67 + index * 305) if compact else (18 + index * 315, 75)
            row = selected.loc[[cell]].iloc[0]
            heading = ("Individual assignment", "Whole-home assignment", "Stepped-wedge rollout")[index]
            elements.append(f'<g id="design-{index}"><text class="title" x="{x}" y="{y}">{heading}</text>')
            if index < 2:
                elements.append(
                    f'<text class="small" x="{x}" y="{y + 26}">Schematic: each circle represents a person</text>'
                )
                for group in range(2):
                    if index == 1:
                        elements.append(
                            f'<rect x="{x + group * 158}" y="{y + 43}" width="136" height="69" '
                            'rx="3" fill="none" stroke="#536370"/>'
                        )
                    for person in range(4):
                        treatment = person % 2 == 0 if index == 0 else group == 0
                        cx, cy = x + 19 + group * 158 + person * 30, y + 77
                        elements.append(f'<circle cx="{cx}" cy="{cy}" r="10" fill="{petrol if treatment else grey}"/>')
                if index == 1:
                    elements.append(
                        f'<text class="small" x="{x + 12}" y="{y + 135}">Home A</text>'
                        f'<text class="small" x="{x + 170}" y="{y + 135}">Home B</text>'
                    )
                else:
                    elements.append(
                        f'<text class="small" x="{x}" y="{y + 135}">Assignment varies from person to person</text>'
                    )
            else:
                elements.append(f'<text class="small" x="{x}" y="{y + 26}">Two recorded switch patterns</text>')
                for period, label in enumerate(("0–14", "14–28", "28–42")):
                    elements.append(
                        f'<text class="small" x="{x + 72 + period * 78}" y="{y + 57}" '
                        f'text-anchor="middle">{label}</text>'
                    )
                for unit, pattern in enumerate(patterns):
                    elements.append(f'<text class="small" x="{x}" y="{y + 88 + unit * 35}">{("A", "B")[unit]}</text>')
                    for period, treated in enumerate(pattern):
                        elements.append(
                            f'<rect x="{x + 38 + period * 78}" y="{y + 67 + unit * 35}" '
                            f'width="70" height="26" fill="{petrol if treated else grey}"/>'
                        )
                elements.append(f'<text class="small" x="{x + 38}" y="{y + 147}">Calendar days of delivery →</text>')
            counts = (
                f"Asthma: {int(row.participants):,} people",
                f"Infection: {int(row.participants):,} residents",
                f"Maternity: {int(row.participants):,} births",
            )[index]
            extra = (
                f"{int(row.active_participants)} treatment; {int(row.control_participants)} control",
                f"{int(row.randomized_clusters):,} randomized care homes",
                f"{int(row.randomized_clusters):,} units switch at different times",
            )[index]
            message = (
                ("Compare the assigned groups;", "preserve each person's identity."),
                ("People in one home share conditions.", "Account for their dependence."),
                ("Treatment changes as time advances.", "Separate time from treatment."),
            )[index]
            elements.extend(
                [
                    f'<text class="strong" x="{x}" y="{y + 177}">{counts}</text>',
                    f'<text class="small" x="{x}" y="{y + 202}">{extra}</text>',
                    f'<text x="{x}" y="{y + 233}">{message[0]}</text>',
                    f'<text x="{x}" y="{y + 258}">{message[1]}</text></g>',
                ]
            )
        y = 1005 if compact else 398
        left, right = 22, width - 24
        elements.extend(
            [
                f'<path d="M12 {y - 36}H{width - 12}" stroke="#c8d1d7"/>',
                f'<text class="title" x="12" y="{y}">Six visits are still one person</text>',
                f'<path d="M{left} {y + 46}H{right}" stroke="#536370" stroke-width="1.5"/>',
            ]
        )
        for day in visits:
            x = left + (right - left) * float(day) / max(visits)
            elements.extend(
                [
                    f'<circle cx="{x:.3f}" cy="{y + 46}" r="5" fill="#536370"/>',
                    f'<text class="small" x="{x:.3f}" y="{y + 73}" text-anchor="middle">{day:.0f}</text>',
                ]
            )
        elements.extend(
            [
                f'<text class="small" x="12" y="{y + 104}">Days since randomization · asthma schedule</text>',
                f'<text x="12" y="{y + 137}">Repeated records are not extra participants.</text>' if compact else "",
                "</svg>",
            ]
        )
        name = "trial-assignment-atlas-compact.svg" if compact else "trial-assignment-atlas.svg"
        (output / name).write_text("\n".join(elements) + "\n", encoding="utf-8")


def plot_designs(evidence: pd.DataFrame, output: Path, *, compact: bool = False) -> None:
    """Align actual rollout, observation clocks and executed analytical consequences."""
    figure, axes = plt.subplots(
        3 if not compact else 4, 2 if not compact else 1, figsize=(12, 10) if not compact else (5.2, 13)
    )
    figure.subplots_adjust(
        left=0.09 if not compact else 0.24,
        right=0.98,
        bottom=0.13,
        top=0.86 if compact else 0.90,
        wspace=0.60,
        hspace=0.85,
    )
    drawers = (
        (_rollout, _calendar, _clocks, _rollout_analysis, _visits, _monitoring)
        if not compact
        else (_rollout, _calendar, _clocks, _rollout_analysis)
    )
    for axis, draw in zip(np.ravel(axes), drawers, strict=True):
        draw(axis, evidence)
    figure.suptitle(
        "Trial design determines\nthe evidence available" if compact else "Design determines the evidence available",
        x=0.09 if not compact else 0.06,
        y=0.98,
        ha="left",
        fontsize=17 if not compact else 15,
        weight="semibold",
    )
    figure.text(
        0.09 if not compact else 0.06,
        0.91 if compact else 0.935,
        "Disclosed September teaching records"
        + (
            " · maternity, asthma and anticoagulation"
            if not compact
            else "\nMaternity rollout: 36 units, 7,200 births"
        ),
        fontsize=10,
        color=GREY,
    )
    figure.text(
        0.09 if not compact else 0.06,
        0.022,
        (
            "The maternity comparison uses calendar periods, baseline characteristics and clustered uncertainty.\nIt illustrates an analytical consequence; one realization does not establish bias or coverage."
            if not compact
            else "One realized study. Adjusted and unadjusted analyses\nhave different estimates and uncertainty; no repeated-study\nrecovery claim is made for this maternity specimen."
        ),
        fontsize=9,
        color=GREY,
    )
    _save(figure, output, "trial-designs-compact" if compact else "trial-designs")


def registry_distributions(root: Path) -> pd.DataFrame:
    """Preserve current-probe enrollment and the actual calibration-cohort quantiles."""
    from review_registry import compare_trials

    targets = json.loads((root / "registry/calibration_targets.json").read_text())
    native = pd.read_csv(root / "registry/native_trials.csv")
    registry = pd.read_csv(root / "registry/registry_trials.csv")
    compared = compare_trials(native, registry, targets)
    rows = []
    for row in compared.loc[compared.metric.eq("Randomized participants")].itertuples(index=False):
        for statistic, value in (
            ("probe", row.native_value),
            ("median", row.registry_median),
            ("quantile_0.25", row.registry_lower_quartile),
            ("quantile_0.75", row.registry_upper_quartile),
        ):
            rows.append(
                {
                    "source_id": PROBE if statistic == "probe" else "aact_20260801",
                    "cell": row.cell if statistic == "probe" else "registry cohort",
                    "family_id": row.family_id,
                    "domain": "trial enrollment",
                    "field": "participants",
                    "label": row.clinical_setting,
                    "unit": "participants per trial",
                    "arm": "all",
                    "group": "one current-generator trial" if statistic == "probe" else "source registry trials",
                    "statistic": statistic,
                    "value": value,
                    "n": 1 if statistic == "probe" else row.registry_trials,
                    "source_role": "source-cohort construction comparison; not held out",
                    "source_ids": "aact_20260801",
                }
            )
    return pd.DataFrame(rows)


def _ecdf(axis: Axes, clinical: pd.DataFrame, field: str, label: str, panel: str) -> None:
    rows = clinical.loc[
        clinical.cell.eq("TE-S16-A1") & clinical.field.eq(field) & clinical.statistic.eq("ecdf")
    ].copy()
    rows["x"] = rows.group.astype(float)
    rows = rows.sort_values("x")
    axis.step(rows.x, rows.value, where="post", color=PETROL, linewidth=1.8)
    summary = clinical.loc[
        clinical.cell.eq("QUEST") & clinical.field.eq(field) & clinical.statistic.eq("mean"), "value"
    ].iloc[0]
    axis.axvline(summary, color=OCHRE, linestyle="--", linewidth=1.5)
    axis.set(xlabel=label, ylabel="Fraction at or below value", ylim=(0, 1.02))
    _title(axis, f"{panel}  " + ("Body mass index" if field == "baseline_body_mass_index" else "Blood eosinophils"))
    axis.text(
        0.96,
        0.12,
        f"QUEST mean {summary:g}\n(summary only)",
        transform=axis.transAxes,
        fontsize=8,
        ha="right",
        color=OCHRE,
    )
    if field == "baseline_eosinophils":
        axis.set_xscale("symlog", linthresh=10)
        axis.set_xlim(0, float(rows.x.max()) * 1.05)
        axis.set_xticks((0, 10, 100, 1000), ("0", "10", "100", "1,000"))


def _categories(axis: Axes, clinical: pd.DataFrame) -> None:
    for position, (field, group) in enumerate((("sex", "female"), ("high_dose_ics", "yes"))):
        rows = clinical.loc[
            clinical.field.eq(field)
            & clinical.group.eq(group)
            & clinical.statistic.eq("proportion")
            & clinical.cell.isin(["TE-S16-A1", "QUEST"])
        ]
        for cell, colour, marker, offset in (("TE-S16-A1", PETROL, "o", -0.07), ("QUEST", OCHRE, "D", 0.07)):
            value = float(rows.loc[rows.cell.eq(cell), "value"].iloc[0])
            axis.scatter(value * 100, position + offset, color=colour, marker=marker, s=44)
            axis.annotate(
                f"{value * 100:.1f}%",
                (value * 100, position + offset),
                xytext=(7, -3),
                textcoords="offset points",
                fontsize=9,
            )
    axis.set(
        yticks=(0, 1),
        yticklabels=("Female sex", "High-dose ICS"),
        xlabel="Recorded participants (%)",
        xlim=(30, 85),
        ylim=(-0.5, 1.5),
    )
    axis.invert_yaxis()
    _title(axis, "D  Anchored proportions")
    axis.text(0.02, 0.05, "● Synthetic records   ◆ QUEST summary", transform=axis.transAxes, fontsize=8)


def _outcomes(axis: Axes, clinical: pd.DataFrame) -> None:
    rows = clinical.loc[clinical.cell.eq("TE-S16-A1") & clinical.statistic.eq("frequency")]
    for arm, colour, offset, label in (
        ("control", OCHRE, -0.2, "Placebo"),
        ("active", PETROL, 0.2, "Add-on biologic"),
    ):
        frame = rows.loc[rows.arm.eq(arm)]
        axis.bar(frame.group.astype(float) + offset, frame.value * 100, width=0.38, color=colour, label=label)
    axis.set(
        xlabel="Severe exacerbations in 365 days",
        ylabel="Participants (%)",
        xticks=range(0, int(rows.group.astype(float).max()) + 1, 5),
    )
    _title(axis, "E  Exacerbation counts")
    axis.legend(frameon=False, fontsize=8)


def _dependence(axis: Axes, clinical: pd.DataFrame) -> None:
    rows = clinical.loc[clinical.cell.eq("TE-S16-A1") & clinical.statistic.eq("conditional_mean")]
    for arm, colour, marker, label in (("control", OCHRE, "s", "Placebo"), ("active", PETROL, "o", "Add-on biologic")):
        frame = rows.loc[rows.arm.eq(arm)].copy()
        frame["x"] = frame.group.str.removesuffix("+").astype(float)
        frame = frame.sort_values("x")
        axis.plot(frame.x, frame.value, marker=marker, color=colour, label=label)
    axis.set(
        xlabel="Severe exacerbations in previous year",
        ylabel="Mean subsequent count",
        ylim=(0, None),
        xticks=range(6),
        xticklabels=("0", "1", "2", "3", "4", "5+"),
    )
    _title(axis, "F  Prior severity and subsequent outcomes")
    axis.legend(frameon=False, fontsize=8)
    axis.text(
        0.01,
        0.96,
        "Authored joint mechanism; not a fitted QUEST relationship",
        transform=axis.transAxes,
        fontsize=8,
        va="top",
        color=GREY,
    )


def plot_clinical(clinical: pd.DataFrame, output: Path, *, compact: bool = False) -> None:
    """Compare empirical anchors, generated distributions and clinical dependence."""
    if compact:
        figure, axes = plt.subplots(4, 1, figsize=(5.2, 12))
        figure.subplots_adjust(left=0.24, right=0.97, top=0.85, bottom=0.13, hspace=0.85)
        _ecdf(axes[0], clinical, "baseline_body_mass_index", "Body mass index (kg/m²)", "A")
        _categories(axes[1], clinical)
        _outcomes(axes[2], clinical)
        _dependence(axes[3], clinical)
    else:
        figure = plt.figure(figsize=(13, 10))
        grid = figure.add_gridspec(
            3, 3, width_ratios=(1.3, 1, 1), left=0.19, right=0.98, bottom=0.16, top=0.89, wspace=0.85, hspace=0.85
        )
        registry_axis = figure.add_subplot(grid[:, 0])
        registry = clinical.loc[clinical.domain.eq("trial enrollment")]
        for position, (_family, frame) in enumerate(registry.groupby("family_id", sort=True)):
            values = frame.set_index("statistic").value
            registry_axis.plot(
                [values["quantile_0.25"], values["quantile_0.75"]],
                [position, position],
                color="#bac2c5",
                linewidth=5,
                solid_capstyle="butt",
            )
            registry_axis.scatter(values["median"], position, color=GREY, marker="|", s=100)
            registry_axis.scatter(values["probe"], position, color=PETROL, s=28)
        families = sorted(registry.family_id.unique())
        registry_axis.set(
            yticks=range(len(families)),
            yticklabels=families,
            xlabel="Participants per trial",
            xscale="log",
            ylim=(len(families) - 0.5, -0.5),
        )
        _title(registry_axis, "A  Trial size and source cohort")
        registry_axis.text(
            0.0,
            -0.09,
            "Grey: registry median and IQR\n● One October probe per family",
            transform=registry_axis.transAxes,
            fontsize=9,
            color=GREY,
        )
        _ecdf(figure.add_subplot(grid[0, 1]), clinical, "baseline_body_mass_index", "Body mass index (kg/m²)", "B")
        _ecdf(figure.add_subplot(grid[0, 2]), clinical, "baseline_eosinophils", "Blood eosinophils (cells/µL)", "C")
        _categories(figure.add_subplot(grid[1, 1]), clinical)
        _outcomes(figure.add_subplot(grid[1, 2]), clinical)
        _dependence(figure.add_subplot(grid[2, 1:]), clinical)
    figure.suptitle(
        "Clinical resemblance\nhas several units" if compact else "Clinical resemblance has several units",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=17 if not compact else 15,
        weight="semibold",
    )
    figure.text(
        0.06,
        0.91 if compact else 0.935,
        (
            "Current-generator trial-size probe; September asthma patient records"
            if not compact
            else "September asthma teaching records, n = 604\nQUEST supplies selected baseline summaries"
        ),
        fontsize=10,
        color=GREY,
    )
    figure.text(
        0.06,
        0.022,
        (
            "QUEST anchors selected marginals. Distribution shapes, eligibility, joint dependence and treatment effects differ.\nRegistry IQR describes variation between trials; it is not patient uncertainty or an acceptance band."
            if not compact
            else "Published summaries inform selected marginals.\nGenerated shapes, dependence and treatment effects\nare not independently validated by those summaries."
        ),
        fontsize=9,
        color=GREY,
    )
    _save(figure, output, "clinical-distributions-compact" if compact else "clinical-distributions")


def plot_methods(results: pd.DataFrame, output: Path, *, compact: bool = False) -> None:
    """Compare two observed-data analyses and their common numerical-loss rule."""
    from matplotlib.patches import Patch

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans"],
            "font.size": 11,
            "text.color": "#202a2e",
            "svg.fonttype": "none",
            "svg.hashsalt": "method-plurality",
            "pdf.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(
            2 if compact else 1, 1 if compact else 2, figsize=(3.5, 8.1) if compact else (10, 4.8)
        )
        figure.subplots_adjust(
            left=0.08,
            right=0.96,
            bottom=0.24 if compact else 0.29,
            top=0.79 if compact else 0.68,
            wspace=0.20,
            hspace=0.75,
        )
        estimate_axis, loss_axis = axes
        first = results.iloc[0]
        estimate_axis.axvspan(first.reference_lower, first.reference_upper, color="#efe5d8")
        estimate_axis.axvline(first.reference, color=OCHRE, linestyle="--", linewidth=1.5)
        estimate_axis.axvline(0, color=GREY, linewidth=0.8)
        labels = ("Difference of means", "Baseline-adjusted regression")
        for position, row in enumerate(results.itertuples(index=False)):
            y = 1 - position
            estimate, lower, upper = cast(float, row.estimate), cast(float, row.lower), cast(float, row.upper)
            estimate_axis.errorbar(
                estimate,
                y,
                xerr=[[estimate - lower], [upper - estimate]],
                color=PETROL,
                fmt="o" if position == 0 else "s",
                capsize=4,
            )
            estimate_axis.text(-8.0, y + 0.25, labels[position], fontsize=11)
            estimate_axis.text(-8.0, y - 0.30, f"{estimate:.2f} [{lower:.2f}, {upper:.2f}]", fontsize=10.5, color=GREY)
            components = (
                cast(float, row.point_loss) / 2,
                cast(float, row.interval_width) / (2 * cast(float, row.reference_scale)),
                cast(float, row.noncoverage_penalty) / (2 * cast(float, row.reference_scale)),
            )
            start = 0.0
            for value, colour, hatch in zip(components, (PETROL, "#a9bdc1", OCHRE), (None, "///", None), strict=True):
                loss_axis.barh(y, value, left=start, color=colour, hatch=hatch, height=0.18)
                start += value
            loss_axis.text(0, y + 0.25, labels[position], fontsize=11)
            loss_axis.text(start + 0.04, y, f"{cast(float, row.numerical_loss):.3f}", va="center", fontsize=10.5)
        estimate_axis.set(
            xlim=(-8.2, 0.3),
            ylim=(-0.5, 1.55),
            yticks=[],
            xticks=[-8, -6, -4, -2, 0],
            xlabel="Motor-disability points\nrehabilitation minus usual care",
        )
        loss_axis.set(
            xlim=(0, 2.7), ylim=(-0.5, 1.55), yticks=[], xticks=[0, 1, 2], xlabel="Numerical loss · lower is better"
        )
        for axis in axes:
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.grid(axis="x", color="#d5dcdc", linewidth=0.5)
            axis.set_axisbelow(True)
        estimate_axis.set_title("Estimates and 95% intervals", loc="left", fontsize=12, pad=14)
        loss_axis.set_title("Loss components", loc="left", fontsize=12, pad=14)
        figure.text(
            0.08,
            0.975,
            "Parkinson rehabilitation\nDay 180" if compact else "Parkinson rehabilitation · Day 180",
            fontsize=15 if compact else 18,
            weight="semibold",
            va="top",
        )
        figure.text(
            0.08,
            0.875 if compact else 0.82,
            "Gold: estimated reference (32 studies)\nBand: 95% Monte Carlo interval",
            fontsize=10.5,
            color=OCHRE,
            va="top",
            linespacing=1.4,
        )
        figure.legend(
            handles=[
                Patch(facecolor=PETROL, label="Point error"),
                Patch(facecolor="#a9bdc1", hatch="///", label="Interval width"),
                Patch(facecolor=OCHRE, label="Missed reference"),
            ],
            frameon=False,
            fontsize=10.5,
            ncol=1 if compact else 3,
            loc="lower left",
            bbox_to_anchor=(0.05, 0.075 if compact else 0.04),
        )
        name = "method-plurality-compact" if compact else "method-plurality"
        for extension in ("svg", "pdf", "png"):
            figure.savefig(
                output / f"{name}.{extension}",
                dpi=180,
                transparent=True,
                metadata={"Date": None} if extension == "svg" else None,
            )
        plt.close(figure)


def review_environment(root: Path, output: Path) -> None:
    """Write the complete scientific environment evidence and editable figures.

    Parameters
    ----------
    root
        Public disclosed scientific examples directory.
    output
        Existing review output directory. Environment artifacts are written in
        its ``environment`` subdirectory.

    Returns
    -------
    None
        Save four tidy CSVs, source/caption notes and SVG, PDF and PNG exports.

    Raises
    ------
    ValueError
        If disclosed identities or the executed method comparison are invalid.

    Examples
    --------
    >>> review_environment(Path("scientific"), Path("scientific-review"))
    """
    destination = output / "environment"
    destination.mkdir(parents=True, exist_ok=False)
    census, clinical, design = project_records(root)
    clinical = pd.concat([clinical, registry_distributions(root)], ignore_index=True)
    design = pd.concat([design, design_analyses(root)], ignore_index=True)
    methods = method_results(root, output)
    plot_assignment_atlas(census, design, destination)
    census.to_csv(destination / "trial_census.csv", index=False)
    clinical.to_csv(destination / "clinical_distributions.csv", index=False)
    design.to_csv(destination / "design_evidence.csv", index=False)
    methods.to_csv(destination / "method_plurality.csv", index=False)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        plot_census(census, destination)
        for compact in (False, True):
            plot_designs(design, destination, compact=compact)
            plot_clinical(clinical, destination, compact=compact)
            plot_methods(methods, destination, compact=compact)
    teaching = json.loads((root / "mechanisms/source.json").read_text())
    registry = json.loads((root / "registry/source.json").read_text())
    figures = [
        {
            "filename": "trial-census.svg",
            "data_path": "trial_census.csv",
            "caption": "Every one of the 30 September teaching design/assumption cells appears once. The 17 families span seven design profiles and five endpoint families. Conditions within a family share seeds; these are not 30 independent release trials. Participant horizon, outcome-row burden and endpoint availability have different units.",
            "alt_text": "Thirty named cell rows align randomized participant counts, individual follow-up horizons, outcome rows per participant and final endpoint availability.",
        },
        {
            "filename": "trial-designs.svg",
            "data_path": "design_evidence.csv",
            "caption": "Maternity units switch treatment across three calendar periods; patient follow-up runs for 42 days after delivery. The period-adjusted comparison accounts for the rollout with cluster uncertainty. Asthma visits show observation loss and randomized retrieval; anticoagulation readouts show a stopping boundary crossed at the second look.",
            "alt_text": "Thirty-six maternity units change from usual care to a postpartum pathway across three calendar periods. Counts and observed risk vary by period. Three actual delivery timelines distinguish observed follow-up from the planned 42-day endpoint. Adjusted and unadjusted risk intervals differ. Asthma visit counts fall below schedules, and the second anticoagulation look crosses its efficacy boundary.",
        },
        {
            "filename": "clinical-distributions.svg",
            "data_path": "clinical_distributions.csv",
            "caption": "Current trial sizes appear against AACT registry medians and interquartile ranges. September asthma teaching records show patient distributions and proportions beside QUEST baseline summaries, with exacerbation counts and their relationship to prior severity. The published summaries anchor selected construction parameters; the generated shapes and joint relationships remain visible for review.",
            "alt_text": "Current-probe trial sizes appear against registry medians and interquartile ranges. September asthma empirical cumulative distributions show body mass index and eosinophils beside reported QUEST means. Female-sex and high-dose inhaled corticosteroid proportions are compared with published summaries. Exacerbation-count distributions and severity-group means show the synthetic outcome and joint relationship.",
        },
        {
            "filename": "method-plurality.svg",
            "data_path": "method_plurality.csv",
            "caption": "Two analyses of the same 509 observed Parkinson outcomes recover the Day-180 population contrast with nearly equal numerical loss. The reference band comes from 32 separate complete studies.",
            "alt_text": "Two Parkinson estimate intervals overlap the separate population reference. Stacked losses separate point error, interval width and noncoverage penalty; no method label enters scoring.",
        },
    ]
    compact_descriptions = {
        "trial-designs": (
            "Maternity units switch treatment across three calendar periods. The patients' 42-day follow-up has a separate clock. The adjusted comparison accounts for period and patient characteristics with cluster uncertainty; the full figure also contains asthma visits and sequential anticoagulation readouts.",
            "Thirty-six maternity units switch across three calendar periods. Three patient timelines separate follow-up from rollout; two estimate intervals show the analytical consequence of adjusting for period and clustering.",
        ),
        "clinical-distributions": (
            "September asthma teaching records show the body-mass-index distribution and categorical baseline proportions beside QUEST summaries. Exacerbation counts and their relationship to prior severity expose outcome variation and dependence. The full figure also shows registry trial-size comparisons and eosinophils.",
            "Asthma body-mass-index cumulative distribution and baseline proportions appear beside reported QUEST summaries; exacerbation counts and severity-group means show outcome variation and dependence.",
        ),
        "method-plurality": (
            "Welch and adjusted HC3 analyses use the same 509 observed Parkinson outcomes and Day-180 target. Both intervals include the separate-world reference. Greater point error offsets a narrower adjusted interval, leaving nearly equal loss.",
            "Two Parkinson estimate intervals overlap a separate population reference; stacked losses show point error and interval width.",
        ),
    }
    for name, (caption, alt_text) in compact_descriptions.items():
        full = next(figure for figure in figures if figure["filename"] == f"{name}.svg")
        figures.append(
            {
                **full,
                "filename": f"{name}-compact.svg",
                "caption": caption,
                "alt_text": alt_text,
            }
        )
    sources = {
        "teaching_source": teaching["family_archive"],
        "current_probe_source": registry["native_source"],
        "population_sources": [
            {
                "source_id": "TAB-SRC-081",
                "title": "Dupilumab improves lung function in patients with uncontrolled, moderate-to-severe asthma",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6983496/",
                "locator": "Table 1, overall QUEST population, n=1902",
                "role": "construction anchor",
                "properties": [
                    "BMI mean and SD",
                    "blood eosinophil mean and SD",
                    "female-sex proportion",
                    "high-dose ICS proportion",
                ],
                "scope": "Compatible marginal summaries only; benchmark eligibility, distribution shapes, dependence and treatment effects differ.",
            }
        ],
        "targets": [
            {
                "target_id": "population-motor-disability-day180",
                "source": "trialeval/reference_worlds.csv",
                "reference_worlds": 32,
                "role": "complete-study population reference; distinct from realized-sample or conditional-law contrast",
                "definition": str(methods.target_definition.iloc[0]),
            },
            {
                "target_id": "maternity-period-adjusted-risk-day42",
                "source": "family-records.zip/TE-S08-A2",
                "role": "observed-data comparison; no disclosed population reference or repeated-study validation",
            },
            {
                "target_id": "cluster-infection-risk-day180",
                "source": "mechanisms/study_estimates.csv and population_references.csv",
                "analysis_studies": 32,
                "reference_worlds": 64,
                "role": "paired repeated-study recovery; distinct target from Parkinson example",
            },
        ],
        "counting_units": "Clinical families contain design/assumption cells. Same-family teaching cells share seeds and can share patient populations. Information views of one study are not independent trials. The pending 31-cell × 4-law × 5-view release has 124 law/study instances and 620 views; it is not enumerated by this archived census.",
        "census_missing_values": "randomized_clusters is zero for participant allocation; cluster_size_cv and observed_event_fraction are empty where the quantity does not apply. Final survival endpoint availability means an observed event or observed event-free follow-up through the requested horizon.",
        "figures": figures,
    }
    (destination / "environment_sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(figures).to_csv(destination / "figures.csv", index=False)
    source_rows = [
        (
            TEACHING,
            "September disclosed clinical teaching archive",
            "30 cells across 17 families; archive last changed 2026-09-25",
            "teaching illustration",
            str(teaching["family_archive"]["coverage"])
            + "; conditions share seeds; predates October registry changes",
            "family-records.zip; mechanisms/source.json",
            "",
        ),
        (
            PROBE,
            "October current-generator trial-size probe",
            "One A1 realization per family; generated 2026-10-11",
            "construction-source comparison",
            str(registry["native_source"]["interpretation"]),
            "registry/native_trials.csv; registry/source.json",
            "",
        ),
        (
            "aact_20260801",
            "AACT trial registry calibration cohort",
            "Family-specific selected registry trials, August 2026 snapshot",
            "construction anchor",
            "Enrollment is a trial-level quantity; IQR is between-trial variation, not patient uncertainty or an acceptance band. This cohort informed construction and is not held out.",
            "registry/registry_trials.csv; registry/calibration_targets.json",
            "https://aact.ctti-clinicaltrials.org/",
        ),
        (
            "TAB-SRC-081",
            "QUEST published baseline summaries",
            "Castro et al., ERJ Open Research 2020; Table 1, overall population, n=1902",
            "construction anchor",
            "BMI/eosinophil means and SDs, female-sex and high-dose ICS proportions only. Benchmark eligibility, shapes, dependence and treatment effects differ.",
            "Published Table 1; family-records.zip/TE-S16-A1/baseline_law.csv and baseline_categories.csv",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6983496/",
        ),
        (
            "bundled-parkinson-trial",
            "Parkinson same-target method comparison",
            "516 randomized; 509 observed Day-180 outcomes; 32 separate complete reference studies",
            "population-reference comparison",
            str(methods.target_definition.iloc[0])
            + "; estimate intervals describe patient sampling and reference intervals describe Monte Carlo precision. The reference does not enter either estimator.",
            "trialeval/analysis_ready.csv; trialeval/reference_worlds.csv; review_trialeval.fixed_trial",
            "",
        ),
        (
            "september-repeated-studies",
            "Repeated cluster-trial recovery",
            "32 paired study seeds; 64 disjoint population-reference studies",
            "repeated-study comparison",
            "Day-180 respiratory-infection risk difference. Both methods use the same point estimate with different uncertainty. This is a different target from the Parkinson demonstration.",
            "mechanisms/study_estimates.csv; mechanisms/population_references.csv",
            "",
        ),
        (
            "maternity-period-adjusted-risk-day42",
            "Maternity rollout analysis",
            "TE-S08-A2: 36 units; 7200 births; three delivery periods; 42-day individual horizon",
            "single-study analytical consequence",
            "Adjusted and unadjusted analyses use observed records. No disclosed population reference or repeated-study recovery result is available for this specimen.",
            "family-records.zip/TE-S08-A2/analysis.parquet; review_families.design_controls",
            "",
        ),
    ]
    pd.DataFrame(source_rows, columns=["source_id", "title", "locator", "role", "scope", "source_path", "url"]).to_csv(
        destination / "sources.csv", index=False
    )


def main() -> None:
    """Run the public ordinary-library environment calculation directly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    review_environment(arguments.source.resolve(strict=True), arguments.output)
    print(f"Scientific environment figures and tables: {arguments.output / 'environment'}")


if __name__ == "__main__":
    main()
