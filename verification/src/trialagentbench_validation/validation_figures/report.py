"""Render the public simulation-validity report figures from packaged data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from trialagentbench_validation.external.release.bundle import (
    installed_validation_root,
)

_BLUE = "#0072B2"
_VERMILLION = "#D55E00"
_CHARCOAL = "#3F3F3F"
_LIGHT_GREY = "#D9D9D9"


def render_validation_report_figures(
    *,
    validation_root: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Render every report figure from the packaged CSV source tables."""

    root = validation_root.resolve()
    figures = root / "figures"
    if not figures.is_dir():
        raise FileNotFoundError(
            f"Validation figure data directory is missing: {figures}"
        )
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    renderers: tuple[tuple[str, Callable[[Path], Figure]], ...] = (
        ("outcome_survival", _survival_figure),
        ("outcome_ordinal", _ordinal_figure),
        ("outcome_longitudinal", _longitudinal_figure),
        ("joint_structure", _joint_structure_figure),
        ("parameter_recovery", _parameter_recovery_figure),
        ("mechanism_response", _mechanism_response_figure),
        ("negative_control", _negative_control_figure),
        ("worked_trial", _worked_trial_figure),
        ("trial_programme", _trial_programme_figure),
        ("trial_designs", _trial_designs_figure),
        ("design_consequences", _design_consequences_figure),
        ("assumption_response", _assumption_response_figure),
        ("assumption_limits", _assumption_limits_figure),
        ("context_workflow", _context_workflow_figure),
        ("generator_realism", _generator_realism_figure),
    )
    paths: list[Path] = []
    with plt.rc_context(cast(Any, _style())):
        for stem, renderer in renderers:
            figure = renderer(figures)
            paths.extend(_save(figure, output / stem))
            plt.close(figure)
    return tuple(paths)


def _worked_trial_figure(root: Path) -> Figure:
    from lifelines import KaplanMeierFitter

    data_root = root.parent / "data"
    participants = pd.read_csv(data_root / "worked_trial_participants.csv")
    _require_columns(
        participants,
        {
            "participant_id",
            "arm",
            "age_years",
            "bmi_kg_m2",
            "time_days",
            "attendance_rate",
            "exposure_adherence",
            "any_intercurrent_event",
            "discontinued",
            "rescue_therapy",
            "treatment_switch",
            "per_protocol",
            "event",
        },
    )
    lineage = pd.read_json(data_root / "worked_trial_lineage.json", typ="series")
    required_lineage = {"estimate", "interval_low", "interval_high"}
    if missing := sorted(required_lineage - set(lineage.index)):
        raise ValueError(f"Worked-trial lineage is missing fields: {missing!r}")
    arms = tuple(
        str(arm) for arm in participants["arm"].drop_duplicates().sort_values()
    )
    if len(arms) != 2:
        raise ValueError("Worked-trial figure requires exactly two randomized arms")
    colours = dict(zip(arms, (_BLUE, _VERMILLION), strict=True))
    linestyles = {
        arms[0]: "-",
        arms[1]: "--",
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    adherence, intercurrent, survival, estimate = axes.ravel()
    for arm in arms:
        rows = participants.loc[participants["arm"].eq(arm)]
        ordered_adherence = np.sort(rows["exposure_adherence"].to_numpy(dtype=float))
        adherence.step(
            ordered_adherence,
            np.arange(1, len(rows) + 1) / len(rows),
            where="post",
            color=colours[arm],
            linestyle=cast(Any, linestyles[arm]),
            linewidth=1.5,
        )
        km = KaplanMeierFitter().fit(
            rows["time_days"], event_observed=rows["event"], label=arm
        )
        survival.step(
            km.survival_function_.index,
            km.survival_function_[arm],
            where="post",
            color=colours[arm],
            linestyle=cast(Any, linestyles[arm]),
            linewidth=1.5,
        )
    handles = [
        Line2D(
            [0],
            [0],
            color=colours[arm],
            linestyle=cast(Any, linestyles[arm]),
            linewidth=1.6,
            label=arm.capitalize(),
        )
        for arm in arms
    ]
    adherence.legend(
        handles=handles, frameon=False, title="Randomized arm", loc="upper left"
    )
    adherence.set(
        xlabel="Received / prescribed exposure",
        ylabel="Participants at or below value",
        xlim=(0, 1.01),
    )
    mechanisms = (
        ("any_intercurrent_event", "Any post-\nrandomization event"),
        ("discontinued", "Discontinued"),
        ("rescue_therapy", "Rescue"),
        ("treatment_switch", "Switched"),
        ("per_protocol", "Per-protocol\neligible"),
    )
    positions = np.arange(len(mechanisms))
    width = 0.34
    for arm_index, arm in enumerate(arms):
        rows = participants.loc[participants["arm"].eq(arm)]
        intercurrent.bar(
            positions + (arm_index - 0.5) * width,
            [float(rows[column].mean()) for column, _ in mechanisms],
            width=width,
            color=colours[arm],
            edgecolor=_CHARCOAL,
            linewidth=0.4,
            label=arm.capitalize(),
        )
    intercurrent.set(
        xticks=positions,
        xticklabels=[label for _, label in mechanisms],
        ylabel="Participant proportion",
        ylim=(0, 1),
    )
    intercurrent.tick_params(axis="x", labelsize=6)
    intercurrent.legend(frameon=False)
    survival.set(
        xlabel="Days since randomization",
        ylabel="Event-free probability",
        ylim=(0.9, 1.0),
    )
    point = float(lineage["estimate"])
    low = float(lineage["interval_low"])
    high = float(lineage["interval_high"])
    estimate.errorbar(
        point,
        0,
        xerr=np.array([[point - low], [high - point]]),
        color=_BLUE,
        marker="D",
        markerfacecolor="white",
        capsize=4,
        linewidth=1.5,
    )
    estimate.axvline(0, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=1)
    estimate.set(
        xlabel="Treated - control event risk (probability)",
        yticks=(),
        ylim=(-0.5, 0.5),
        xlim=(min(low - 0.01, -0.035), max(high + 0.01, 0.015)),
    )
    estimate.text(
        point, 0.13, f"{point:.3f} ({low:.3f} to {high:.3f})", ha="center", fontsize=7
    )
    for panel, axis in enumerate(axes.ravel()):
        axis.text(
            -0.12,
            1.04,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
        axis.grid(axis="both")
    return _finish(figure, bottom=0.1, left=0.11)


def _trial_programme_figure(root: Path) -> Figure:
    data_root = root.parent / "data"
    profiles = pd.read_csv(data_root / "programme_profiles.csv")
    estimates = pd.read_csv(data_root / "programme_estimates.csv")
    _require_columns(
        profiles,
        {
            "independence_unit_id",
            "design_profile_id",
            "participant_count",
            "follow_up_horizon_days",
        },
    )
    _require_columns(
        estimates, {"trial_id", "property_id", "group", "time", "estimate"}
    )
    profile_map = profiles.set_index("independence_unit_id")["design_profile_id"]
    estimates["design_profile_id"] = estimates["trial_id"].map(profile_map)
    if estimates["design_profile_id"].isna().any():
        raise ValueError(
            "Programme estimates contain trials absent from the profile census"
        )
    displays = (
        (
            "Participants per trial",
            profiles.rename(columns={"participant_count": "value"}),
            "Participants",
        ),
        (
            "Age-BMI rank correlation",
            _select_programme_estimate(estimates, "dependence.age_bmi.spearman"),
            "Spearman correlation coefficient",
        ),
        (
            "Participant attendance",
            _select_programme_estimate(estimates, "observation.attendance_rate.mean"),
            "Mean attendance proportion",
        ),
        (
            "Primary follow-up",
            profiles.rename(columns={"follow_up_horizon_days": "value"}),
            "Days",
        ),
    )
    profile_ids = tuple(
        sorted(str(value) for value in profiles["design_profile_id"].unique())
    )
    if profile_ids != tuple(f"TE-DP0{index}" for index in range(1, 8)):
        raise ValueError("Programme figure requires all seven design profiles")
    profile_labels = (
        "Individual",
        "Pragmatic",
        "Standardized",
        "Ascertainment",
        "Cluster",
        "Stepped-wedge",
        "Sequential",
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    for panel, (axis, (label, rows, y_label)) in enumerate(
        zip(axes.ravel(), displays, strict=True)
    ):
        for position, profile_id in enumerate(profile_ids):
            values = (
                rows.loc[rows["design_profile_id"].eq(profile_id), "value"]
                .astype(float)
                .sort_values()
                .to_numpy()
            )
            if not len(values):
                raise ValueError(f"Programme figure lacks {label} for {profile_id}")
            offsets = np.linspace(-0.15, 0.15, len(values))
            axis.scatter(
                np.full(len(values), position) + offsets,
                values,
                color=_BLUE,
                marker="o",
                facecolors="none",
                s=14,
                linewidths=0.7,
                alpha=0.7,
            )
            axis.scatter(
                position,
                np.median(values),
                color=_VERMILLION,
                marker="D",
                s=26,
                zorder=3,
            )
        axis.set_xticks(
            range(len(profile_ids)), profile_labels, rotation=30, ha="right"
        )
        axis.tick_params(axis="x", labelsize=6)
        axis.set(title=label, ylabel=y_label)
        if panel < 2:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel("Design profile")
        axis.text(
            -0.12,
            1.04,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
        axis.grid(axis="y")
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=_BLUE,
                marker="o",
                markerfacecolor="white",
                linestyle="none",
                label="Trial",
            ),
            Line2D(
                [0],
                [0],
                color=_VERMILLION,
                marker="D",
                linestyle="none",
                label="Profile median",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    return _finish(figure, bottom=0.15, left=0.11)


def _context_workflow_figure(root: Path) -> Figure:
    workflow = pd.read_csv(root / "context_workflow.csv")
    context = pd.read_csv(root.parent / "data/context_invariance.csv")
    standards = json.loads(
        (root.parent / "data/cdisc_reference_evidence.json").read_text(encoding="utf-8")
    )
    _require_columns(
        workflow,
        {
            "context_id",
            "recovery_path",
            "route_count",
            "successful_route_count",
            "maximum_absolute_difference",
        },
    )
    _require_columns(
        context,
        {
            "matched_set_id",
            "context_count",
            "generation_seed_count",
            "estimand_count",
            "analysis_ready_hash_count_c1_c2",
            "raw_domain_hash_count_c3_c4",
            "status",
        },
    )
    expected_contexts = ("C1", "C2", "C3", "C4", "C5")
    workflow = (
        workflow.set_index("context_id").loc[list(expected_contexts)].reset_index()
    )
    if not workflow["route_count"].eq(workflow["successful_route_count"]).all():
        raise ValueError("Context reconstruction contains an unsuccessful route")
    if len(context) != 100 or not context["status"].eq("pass").all():
        raise ValueError("Context reconstruction requires 100 passing matched panels")

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    route_axis, replay_axis, identity_axis, standards_axis = axes.ravel()
    path_style = {
        "direct_analysis_ready": (_BLUE, ""),
        "reconstruct_raw_domains": (_VERMILLION, "///"),
        "repair_then_reconstruct_raw_domains": (_CHARCOAL, "\\\\\\"),
    }
    for position, row in workflow.iterrows():
        colour, hatch = path_style[str(row["recovery_path"])]
        route_axis.bar(
            position,
            float(row["route_count"]),
            color=colour,
            edgecolor=_CHARCOAL,
            linewidth=0.7,
            hatch=hatch,
        )
        route_axis.text(
            position,
            float(row["route_count"]) + 5,
            str(int(row["route_count"])),
            ha="center",
        )
    route_axis.set(
        title="Analysis routes",
        xticks=range(5),
        xticklabels=expected_contexts,
        xlabel="Context",
        ylabel="Independently replayed routes",
        ylim=(0, 190),
    )
    route_axis.grid(axis="y")

    replay_axis.plot(
        range(5),
        workflow["maximum_absolute_difference"] * 1.0e14,
        color=_BLUE,
        marker="o",
        markerfacecolor="white",
        linewidth=1.3,
    )
    replay_axis.set(
        title="Estimate replay",
        xticks=range(5),
        xticklabels=expected_contexts,
        xlabel="Context",
        ylabel="Maximum absolute difference (10^-14)",
    )
    replay_axis.grid(axis="y")

    identity_checks = (
        ("Complete\nC1-C5", context["context_count"].eq(5)),
        ("One generation\nseed", context["generation_seed_count"].eq(1)),
        ("One\nestimand", context["estimand_count"].eq(1)),
        (
            "C1/C2 analysis\ndata identical",
            context["analysis_ready_hash_count_c1_c2"].eq(1),
        ),
        ("C3/C4 raw\ndata identical", context["raw_domain_hash_count_c3_c4"].eq(1)),
    )
    identity_counts = [int(check.sum()) for _, check in identity_checks]
    identity_axis.barh(
        range(len(identity_checks)),
        identity_counts,
        color=_BLUE,
        edgecolor=_CHARCOAL,
        linewidth=0.7,
    )
    for position, ((label, _), count) in enumerate(
        zip(identity_checks, identity_counts, strict=True)
    ):
        identity_axis.text(
            2,
            position,
            label.replace("\n", " "),
            ha="left",
            va="center",
            color="white",
            fontsize=6,
        )
        identity_axis.text(
            count - 2, position, str(count), ha="right", va="center", color="white"
        )
    identity_axis.set(
        title="Matched contexts",
        yticks=(),
        xlabel="Matched trial panels (n)",
        xlim=(0, 105),
    )
    identity_axis.grid(axis="x")

    standards_rows = (
        (
            "Transport cells",
            int(standards["cells_compared"]) - int(standards["mismatched_cells"]),
            int(standards["cells_compared"]),
        ),
        (
            "Define datasets",
            int(standards["define_xml_datasets_present"]),
            int(standards["define_xml_datasets_required"]),
        ),
        (
            "Analysis result",
            int(float(standards["analysis_absolute_difference"]) == 0.0),
            1,
        ),
        (
            "Corruption controls",
            int(standards["negative_controls_detected"]),
            int(standards["negative_controls_total"]),
        ),
    )
    labels = [row[0] for row in standards_rows]
    fractions = [row[1] / row[2] for row in standards_rows]
    bars = standards_axis.barh(
        labels, fractions, color=_BLUE, edgecolor=_CHARCOAL, linewidth=0.7
    )
    for bar, (_, numerator, denominator) in zip(bars, standards_rows, strict=True):
        standards_axis.text(
            0.98,
            bar.get_y() + bar.get_height() / 2,
            f"{numerator:,}/{denominator:,}",
            ha="right",
            va="center",
            color="white",
            fontsize=7,
        )
    standards_axis.set(
        title="Standards workflow",
        xlabel="Checks passed / checks performed",
        xlim=(0, 1.02),
    )
    standards_axis.grid(axis="x")

    for panel, axis in enumerate(axes.ravel()):
        axis.text(
            -0.12,
            1.04,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
    figure.legend(
        handles=[
            Patch(facecolor=_BLUE, edgecolor=_CHARCOAL, label="Analysis-ready"),
            Patch(
                facecolor=_VERMILLION,
                edgecolor=_CHARCOAL,
                hatch="///",
                label="Raw reconstruction",
            ),
            Patch(
                facecolor=_CHARCOAL,
                edgecolor=_CHARCOAL,
                hatch="\\\\\\",
                label="Repair and reconstruction",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    return _finish(figure, bottom=0.18, left=0.12)


def _generator_realism_figure(root: Path) -> Figure:
    data = pd.read_csv(root / "generator_realism.csv")
    _require_columns(
        data,
        {
            "construct",
            "validation_domain",
            "relative_distance",
            "interval_low",
            "interval_high",
            "external_trials",
            "synthetic_trials",
        },
    )
    expected_domains = ("Marginal distributions", "Joint structure", "Analysis impact")
    if tuple(data["validation_domain"].drop_duplicates()) != expected_domains:
        raise ValueError(
            "Generator-realism constructs must retain the declared domain order"
        )
    colours = {
        "Marginal distributions": _BLUE,
        "Joint structure": _VERMILLION,
        "Analysis impact": _CHARCOAL,
    }
    markers = {
        "Marginal distributions": "o",
        "Joint structure": "D",
        "Analysis impact": "s",
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    positions = np.arange(len(data))[::-1]
    for position, (_, row) in zip(positions, data.iterrows(), strict=True):
        value = float(row["relative_distance"])
        low = float(row["interval_low"])
        high = float(row["interval_high"])
        domain = str(row["validation_domain"])
        axis.errorbar(
            value,
            position,
            xerr=np.array([[value - low], [high - value]]),
            color=colours[domain],
            marker=markers[domain],
            markerfacecolor="white",
            linestyle="none",
            capsize=3,
            linewidth=1.2,
        )
    axis.axvline(
        1,
        color=_CHARCOAL,
        linestyle=(0, (5, 2)),
        linewidth=1.1,
        label="95th percentile between external trial splits",
    )
    axis.set_xscale("log")
    axis.set(
        xlabel="Generated-external discrepancy (external 95th percentile = 1)",
        yticks=positions,
        yticklabels=data["construct"],
        xlim=(0.15, 2.1),
    )
    axis.set_xticks((0.2, 0.5, 1.0, 2.0), ("0.2", "0.5", "1", "2"))
    axis.grid(axis="x")
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=colours[domain],
                marker=markers[domain],
                markerfacecolor="white",
                linestyle="none",
                label=domain,
            )
            for domain in expected_domains
        ]
        + [
            Line2D(
                [0],
                [0],
                color=_CHARCOAL,
                linestyle=(0, (5, 2)),
                label="External split reference",
            )
        ],
        frameon=False,
        loc="lower center",
        ncol=4,
    )
    return _finish(figure, bottom=0.23, left=0.3)


def _assumption_response_figure(root: Path) -> Figure:
    data = pd.read_csv(root / "assumption_response.csv")
    _require_columns(
        data,
        {
            "series_id",
            "assumption_tier",
            "mechanism_value_mean",
            "mechanism_value_interval_low",
            "mechanism_value_interval_high",
            "mechanism_label",
            "mechanism_unit",
            "consequence_value_mean",
            "consequence_interval_low",
            "consequence_interval_high",
            "consequence_unit",
            "consequence_label",
        },
    )
    labels = {
        "TE-S01": "Time-varying effect",
        "TE-S02": "Censoring in survival-time analysis",
        "TE-S03": "Nonadherence",
        "TE-S04": "Censoring in a pragmatic trial",
        "TE-S05": "Nonlinear prognosis and response",
        "TE-S06": "Endpoint classification",
        "TE-S07": "Censoring in a cluster trial",
        "TE-S08": "Calendar trend",
    }
    tiers = {"A1": 0, "A2": 1, "A3": 2}
    if set(data["series_id"]) != set(labels):
        raise ValueError(
            "Assumption-response figure requires all eight point-identified series"
        )
    if not set(data["assumption_tier"]).issubset(tiers):
        raise ValueError("Assumption-response figure accepts only A1-A3 cells")

    figure, axes = plt.subplots(8, 2, figsize=(7.2, 9.8))
    for panel, (series_id, label) in enumerate(labels.items()):
        mechanism_axis, consequence_axis = axes[panel]
        rows = data.loc[data["series_id"].eq(series_id)].copy()
        rows["x"] = rows["assumption_tier"].map(tiers)
        rows = rows.sort_values("x")
        x = rows["x"].to_numpy(dtype=float)
        mechanism = rows["mechanism_value_mean"].to_numpy(dtype=float)
        mechanism_low = rows["mechanism_value_interval_low"].to_numpy(dtype=float)
        mechanism_high = rows["mechanism_value_interval_high"].to_numpy(dtype=float)
        mechanism_axis.errorbar(
            x,
            mechanism,
            yerr=np.vstack((mechanism - mechanism_low, mechanism_high - mechanism)),
            color=_BLUE,
            marker="o",
            markerfacecolor="white",
            linestyle="-",
            linewidth=1.3,
            markersize=4,
            capsize=2,
            label="Observed mechanism",
        )
        consequence = rows["consequence_value_mean"].to_numpy(dtype=float)
        consequence_low = rows["consequence_interval_low"].to_numpy(dtype=float)
        consequence_high = rows["consequence_interval_high"].to_numpy(dtype=float)
        consequence_axis.errorbar(
            x,
            consequence,
            yerr=np.vstack(
                (consequence - consequence_low, consequence_high - consequence)
            ),
            color=_VERMILLION,
            marker="D",
            markerfacecolor="white",
            linestyle=(0, (5, 2)),
            linewidth=1.3,
            markersize=4,
            capsize=2,
            label="Change in the trial result",
        )
        mechanism_units = tuple(rows["mechanism_unit"].dropna().astype(str).unique())
        if len(mechanism_units) != 1:
            raise ValueError(f"{series_id} requires one mechanism unit")
        mechanism_labels = tuple(rows["mechanism_label"].dropna().astype(str).unique())
        if len(mechanism_labels) != 1:
            raise ValueError(f"{series_id} requires one mechanism definition")
        result_units = tuple(rows["consequence_unit"].dropna().astype(str).unique())
        if len(result_units) != 1:
            raise ValueError(f"{series_id} requires one consequence unit")
        consequence_labels = tuple(
            rows["consequence_label"].dropna().astype(str).unique()
        )
        if len(consequence_labels) != 1:
            raise ValueError(f"{series_id} requires one consequence definition")
        mechanism_unit = {
            "log hazard ratio": "log hazard ratio",
            "proportion": "proportion",
        }.get(mechanism_units[0])
        if mechanism_unit is None:
            raise ValueError(
                f"{series_id} has an unsupported mechanism unit: {mechanism_units[0]!r}"
            )
        result_unit = {
            "risk difference": "estimate difference\n(0-1 risk scale)",
            "standardized risk difference": "estimate difference\n(0-1 risk scale)",
            "days": "estimate difference\n(days)",
        }.get(result_units[0])
        if result_unit is None:
            raise ValueError(
                f"{series_id} has an unsupported consequence unit: {result_units[0]!r}"
            )
        if series_id == "TE-S03":
            result_unit = "attenuation from A1\n(0-1 risk scale)"
        mechanism_axis.set_title(
            f"{series_id.removeprefix('TE-')}  {label}",
            fontsize=7,
            loc="left",
            pad=2,
        )
        consequence_axis.set_title(
            consequence_labels[0], fontsize=5.5, loc="left", pad=2
        )
        tier_labels = (
            ("A1\nreference", "A2\nstressed")
            if tuple(rows["assumption_tier"]) == ("A1", "A2")
            else ("A1\nreference", "A2\nintermediate", "A3\nstrong")
        )
        for response_axis in (mechanism_axis, consequence_axis):
            response_axis.set_xticks(x, tier_labels)
            response_axis.set_xlim(float(x.min()) - 0.15, float(x.max()) + 0.15)
            response_axis.tick_params(axis="both", labelsize=6)
            response_axis.grid(axis="y")
        mechanism_axis.set_ylabel(
            f"{mechanism_labels[0]}\n({mechanism_unit})",
            color=_BLUE,
            fontsize=6,
        )
        consequence_axis.set_ylabel(result_unit, color=_VERMILLION, fontsize=6)
        mechanism_axis.tick_params(axis="y", colors=_BLUE)
        consequence_axis.tick_params(axis="y", colors=_VERMILLION)
        mechanism_display_high = max(
            float(rows["mechanism_value_interval_high"].max()), 0.01
        )
        consequence_floor = 0.1 if result_units[0] == "days" else 0.005
        consequence_display_low = min(
            float(rows["consequence_interval_low"].min()), 0.0
        )
        consequence_display_high = max(
            float(rows["consequence_interval_high"].max()), consequence_floor
        )
        mechanism_axis.set_ylim(
            -0.04 * mechanism_display_high, 1.08 * mechanism_display_high
        )
        consequence_span = max(
            consequence_display_high - consequence_display_low, consequence_floor
        )
        consequence_axis.set_ylim(
            consequence_display_low - 0.04 * consequence_span,
            consequence_display_high + 0.08 * consequence_span,
        )
        consequence_axis.axhline(
            0.0, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.6
        )
        mechanism_axis.text(
            -0.1,
            1.03,
            chr(ord("a") + panel),
            transform=mechanism_axis.transAxes,
            fontweight="bold",
            ha="right",
        )
    axes[0, 0].text(
        0.5,
        1.45,
        "Observed mechanism",
        color=_BLUE,
        ha="center",
        transform=axes[0, 0].transAxes,
        fontsize=8,
        fontweight="bold",
    )
    axes[0, 1].text(
        0.5,
        1.45,
        "Analysis consequence",
        color=_VERMILLION,
        ha="center",
        transform=axes[0, 1].transAxes,
        fontsize=8,
        fontweight="bold",
    )
    figure.supxlabel("Prespecified assumption condition", y=0.03)
    figure.subplots_adjust(
        left=0.11, right=0.94, top=0.94, bottom=0.07, wspace=0.4, hspace=0.78
    )
    return figure


def _assumption_limits_figure(root: Path) -> Figure:
    """Show the supported results when an A4 condition precludes the routine analysis."""

    data_root = root.parent / "data"
    identified = pd.read_csv(data_root / "assumption_identification_results.csv")
    _require_columns(
        identified,
        {
            "task_id",
            "series_id",
            "replicate_index",
            "assumption",
            "sensitivity_parameter",
            "lower",
            "upper",
            "width",
            "result_unit",
        },
    )
    if set(identified["series_id"]) != {"TE-S04", "TE-S06"}:
        raise ValueError(
            "A4 identified-set results require censoring and endpoint-validation trials"
        )
    if not identified["result_unit"].eq("risk difference").all():
        raise ValueError("A4 identified-set results must use the risk-difference scale")
    if not np.isfinite(
        identified[["lower", "upper", "width"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("A4 identified-set results must be finite")

    bridges = pd.read_csv(data_root / "assumption_bridges.csv")
    _require_columns(
        bridges,
        {
            "series_id",
            "replicate_index",
            "assumption_tier",
            "default_status",
            "qualified_shape",
            "qualified_value",
            "qualified_interval_low",
            "qualified_interval_high",
            "result_unit",
        },
    )
    sequential = bridges.loc[
        bridges["series_id"].eq("TE-S09") & bridges["assumption_tier"].eq("A4")
    ].sort_values("replicate_index")
    if len(sequential) != 4 or not sequential["qualified_shape"].eq("point").all():
        raise ValueError("A4 figure requires four sequential-monitoring point analyses")

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    series = {
        "TE-S04": ("Outcome-related loss to follow-up", _BLUE),
        "TE-S06": ("Incomplete endpoint validation", _VERMILLION),
    }
    positions = np.arange(4)
    labels = ("5", "10", "20", "No bound")
    for panel, (series_id, (title, colour)) in enumerate(series.items()):
        axis = axes[0, panel]
        rows = identified.loc[identified["series_id"].eq(series_id)].copy()
        if len(rows) != 16 or rows["task_id"].nunique() != 4:
            raise ValueError(
                f"{series_id} A4 response requires four analyses for each of four trials"
            )
        for _, trial in rows.groupby("replicate_index", sort=True):
            trial = trial.sort_values(
                ["sensitivity_parameter"],
                na_position="last",
            )
            widths = trial["width"].to_numpy(dtype=float)
            if len(widths) != 4 or np.any(np.diff(widths) < -1e-12):
                raise ValueError(
                    f"{series_id} identified ranges must widen with the allowed departure"
                )
            axis.plot(
                positions,
                widths,
                color=_LIGHT_GREY,
                marker="o",
                markersize=3,
                linewidth=0.8,
                zorder=1,
            )
        mean_width = (
            rows.assign(
                response_order=np.where(
                    rows["assumption"].eq("worst_case"),
                    3,
                    rows["sensitivity_parameter"].map({0.05: 0, 0.10: 1, 0.20: 2}),
                )
            )
            .groupby("response_order", sort=True)["width"]
            .mean()
            .reindex(positions)
            .to_numpy(dtype=float)
        )
        axis.plot(
            positions,
            mean_width,
            color=colour,
            marker="s",
            linewidth=1.8,
            markersize=5,
            label="Mean across four trials",
            zorder=2,
        )
        axis.set(
            title=title,
            xlabel="Maximum event-risk departure (percentage points)",
            ylabel="Risk-difference range width\n(0-1 scale)",
            xticks=positions,
            xticklabels=labels,
        )
        axis.grid(axis="y")

    selected = identified.loc[identified["sensitivity_parameter"].eq(0.20)].copy()
    selected = selected.sort_values(["series_id", "replicate_index"])
    if len(selected) != 8:
        raise ValueError(
            "A4 display requires the 0.20 range from eight independent trials"
        )
    bound_axis = axes[1, 0]
    for series_id, (_, colour) in series.items():
        rows = selected.loc[selected["series_id"].eq(series_id)]
        y = np.arange(len(selected))[selected["series_id"].eq(series_id).to_numpy()]
        midpoint = (
            rows["lower"].to_numpy(dtype=float) + rows["upper"].to_numpy(dtype=float)
        ) / 2.0
        bound_axis.errorbar(
            midpoint,
            y,
            xerr=np.vstack(
                (
                    midpoint - rows["lower"].to_numpy(dtype=float),
                    rows["upper"].to_numpy(dtype=float) - midpoint,
                )
            ),
            color=colour,
            marker="s",
            linestyle="none",
            capsize=2.5,
            linewidth=1.1,
        )
    bound_axis.axvline(0.0, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8)
    bound_axis.set(
        title="Ranges allowing a 20 percentage-point departure",
        xlabel="Treated - control event-risk difference (0-1 scale)",
        ylabel="Independent trial",
        yticks=np.arange(len(selected)),
        yticklabels=[
            f"{series_id.removeprefix('TE-')} trial {replicate_index}"
            for series_id, replicate_index in zip(
                [str(value) for value in selected["series_id"].tolist()],
                [int(value) for value in selected["replicate_index"].tolist()],
                strict=True,
            )
        ],
    )
    bound_axis.grid(axis="x")

    sequential_axis = axes[1, 1]
    estimate = sequential["qualified_value"].to_numpy(dtype=float)
    low = sequential["qualified_interval_low"].to_numpy(dtype=float)
    high = sequential["qualified_interval_high"].to_numpy(dtype=float)
    sequential_axis.errorbar(
        estimate,
        np.arange(1, 5),
        xerr=np.vstack((estimate - low, high - estimate)),
        color=_BLUE,
        marker="o",
        markerfacecolor="white",
        linestyle="none",
        capsize=3,
        linewidth=1.2,
    )
    sequential_axis.axvline(0.0, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8)
    sequential_axis.set(
        title="Sequential monitoring",
        xlabel="Treated - control event-risk difference (0-1 scale)",
        ylabel="Independent trial",
        yticks=np.arange(1, 5),
        yticklabels=[f"Trial {value}" for value in range(1, 5)],
    )
    sequential_axis.grid(axis="x")
    for panel, axis in enumerate(axes.ravel()):
        axis.text(
            -0.12,
            1.08,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
            ha="right",
        )
    figure.legend(
        handles=[
            Line2D([0], [0], color=_LIGHT_GREY, marker="o", label="Individual trial"),
            Line2D(
                [0],
                [0],
                color=_BLUE,
                marker="s",
                label="Mean loss-to-follow-up response",
            ),
            Line2D(
                [0],
                [0],
                color=_VERMILLION,
                marker="s",
                label="Mean endpoint-validation response",
            ),
            Line2D(
                [0],
                [0],
                color=_BLUE,
                marker="o",
                markerfacecolor="white",
                linestyle="none",
                label="Repeated 95% interval",
            ),
        ],
        frameon=False,
        loc="lower center",
        ncol=2,
    )
    return _finish(figure, top=0.91, bottom=0.18, left=0.14, wspace=0.5, hspace=0.52)


def _analysis_reliability_figure(root: Path) -> Figure:
    data = pd.read_csv(root.parent / "data/analysis_reliability.csv")
    _require_columns(
        data,
        {
            "series_id",
            "independent_trials",
            "default_coverage",
            "default_coverage_low",
            "default_coverage_high",
            "qualified_coverage",
            "qualified_coverage_low",
            "qualified_coverage_high",
            "qualified_to_default_rmse_ratio",
            "rmse_ratio_low",
            "rmse_ratio_high",
            "paired_recovery_rate",
            "paired_recovery_rate_low",
            "paired_recovery_rate_high",
            "paired_loss_rate",
            "paired_loss_rate_low",
            "paired_loss_rate_high",
            "fit_failure_rate",
            "fit_failure_rate_low",
            "fit_failure_rate_high",
        },
    )
    labels = {
        "TE-S01": "Time-varying\neffect",
        "TE-S02": "Outcome-related\ncensoring",
        "TE-S04": "Baseline-dependent\ncensoring",
        "TE-S05": "Nonlinear\nprognosis",
        "TE-S06": "Endpoint\nclassification",
        "TE-S07": "Site-related\ncensoring",
    }
    if set(data["series_id"]) != set(labels):
        raise ValueError(
            "Analysis-reliability figure requires all six recoverable A3 series"
        )
    if data["series_id"].duplicated().any():
        raise ValueError("Analysis-reliability figure requires one result per series")
    if (
        (
            data[
                ["qualified_to_default_rmse_ratio", "rmse_ratio_low", "rmse_ratio_high"]
            ]
            <= 0
        )
        .any()
        .any()
    ):
        raise ValueError("Analysis-reliability RMSE ratios must be positive")
    ordered = data.set_index("series_id").loc[list(labels)].reset_index()
    x = np.arange(len(ordered))
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.3))

    width = 0.17
    for offset, prefix, colour, marker, label in (
        (-width, "default", _VERMILLION, "s", "Routine analysis"),
        (width, "qualified", _BLUE, "o", "Prespecified alternative"),
    ):
        values = ordered[f"{prefix}_coverage"].to_numpy(dtype=float)
        axes[0].errorbar(
            x + offset,
            values,
            yerr=np.vstack(
                (
                    values - ordered[f"{prefix}_coverage_low"].to_numpy(dtype=float),
                    ordered[f"{prefix}_coverage_high"].to_numpy(dtype=float) - values,
                )
            ),
            color=colour,
            marker=marker,
            markerfacecolor="white",
            linestyle="none",
            capsize=2,
            label=label,
        )
    axes[0].axhline(0.95, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8)
    axes[0].text(
        0.98,
        0.95,
        "95% target",
        transform=axes[0].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6,
    )
    axes[0].set(
        ylabel="Coverage probability",
        ylim=(0, 1.03),
        title="Interval coverage",
    )
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")

    ratio = ordered["qualified_to_default_rmse_ratio"].to_numpy(dtype=float)
    axes[1].errorbar(
        x,
        ratio,
        yerr=np.vstack(
            (
                ratio - ordered["rmse_ratio_low"].to_numpy(dtype=float),
                ordered["rmse_ratio_high"].to_numpy(dtype=float) - ratio,
            )
        ),
        color=_BLUE,
        marker="o",
        markerfacecolor="white",
        linestyle="none",
        capsize=2,
    )
    rmse_display_floor = min(
        max(float(ordered["rmse_ratio_low"].min()) * 0.9, 0.05),
        0.95,
    )
    axes[1].axhspan(
        rmse_display_floor,
        1.0,
        color=_BLUE,
        alpha=0.07,
        linewidth=0,
    )
    axes[1].axhline(1, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8)
    axes[1].text(
        0.98,
        1,
        "Equal error",
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6,
    )
    axes[1].set_yscale("log")
    axes[1].set(
        ylabel="RMSE ratio\n(alternative / routine)",
        title="Relative estimation error",
    )
    axes[1].text(
        0.02,
        0.03,
        "Below 1: lower error with the alternative analysis",
        transform=axes[1].transAxes,
        color=_BLUE,
        fontsize=5.7,
        va="bottom",
    )

    for offset, prefix, colour, marker, label in (
        (
            -width,
            "paired_recovery_rate",
            _BLUE,
            "o",
            "Routine misses; alternative covers",
        ),
        (
            0.0,
            "paired_loss_rate",
            _VERMILLION,
            "s",
            "Routine covers; alternative misses",
        ),
        (width, "fit_failure_rate", _CHARCOAL, "^", "Analysis fit failure"),
    ):
        values = ordered[prefix].to_numpy(dtype=float)
        axes[2].errorbar(
            x + offset,
            values,
            yerr=np.vstack(
                (
                    values - ordered[f"{prefix}_low"].to_numpy(dtype=float),
                    ordered[f"{prefix}_high"].to_numpy(dtype=float) - values,
                )
            ),
            color=colour,
            marker=marker,
            markerfacecolor="white",
            linestyle="none",
            capsize=2,
            label=label,
        )
    axes[2].set(
        ylabel="Trial proportion",
        ylim=(-0.02, 1.03),
        title="Trial-level comparison",
    )
    axes[2].legend(frameon=False, fontsize=5.7, loc="upper right")

    tick_labels = tuple(labels[series_id] for series_id in ordered["series_id"])
    for panel, axis in enumerate(axes):
        axis.set_xticks(x, tick_labels, rotation=35, ha="right")
        axis.tick_params(axis="x", labelsize=6)
        axis.set_title(axis.get_title(), fontsize=8.5, pad=7)
        axis.set_ylabel(axis.get_ylabel(), fontsize=7)
        axis.grid(axis="y")
        axis.text(
            -0.2,
            1.16,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8,
        )
    trial_counts = tuple(int(value) for value in ordered["independent_trials"].unique())
    if len(trial_counts) != 1:
        raise ValueError(
            "Analysis-reliability series require a common independent-trial count"
        )
    figure.text(
        0.5,
        0.01,
        f"{trial_counts[0]} independent trials per analysis pair",
        ha="center",
        fontsize=7,
    )
    figure.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.3, wspace=0.42)
    return figure


def _identification_reliability_figure(root: Path) -> Figure:
    """Show coverage, information, and execution for every A4 analysis route."""

    data = pd.read_csv(root.parent / "data/identification_reliability.csv")
    _require_columns(
        data,
        {
            "series_id",
            "conclusion_type",
            "analysis_role",
            "effect_scale",
            "sensitivity_parameter",
            "sensitivity_parameter_unit",
            "independent_trials",
            "coverage",
            "coverage_low",
            "coverage_high",
            "mean_width",
            "mean_width_low",
            "mean_width_high",
            "fit_failure_rate",
            "fit_failure_rate_low",
            "fit_failure_rate_high",
            "early_stop_rate",
            "early_stop_rate_low",
            "early_stop_rate_high",
            "bias",
            "bias_low",
            "bias_high",
            "rmse",
            "rmse_low",
            "rmse_high",
        },
    )
    order = (
        ("TE-S04", "prespecified_bounded_departure"),
        ("TE-S04", "unrestricted_worst_case"),
        ("TE-S06", "prespecified_bounded_departure"),
        ("TE-S06", "unrestricted_worst_case"),
        ("TE-S09", "repeated_monitoring"),
    )
    labels = {
        order[0]: "Loss to follow-up\n20 percentage-point\ndeparture",
        order[1]: "Loss to follow-up\nworst case",
        order[2]: "Endpoint validation\n20 percentage-point\ndeparture",
        order[3]: "Endpoint validation\nworst case",
        order[4]: "Sequential analysis\nrepeated interval",
    }
    identities = list(zip(data["series_id"], data["analysis_role"], strict=True))
    if len(identities) != len(set(identities)) or set(identities) != set(order):
        raise ValueError(
            "Identification reliability requires every declared A4 analysis route exactly once"
        )
    ordered = (
        data.set_index(["series_id", "analysis_role"]).loc[list(order)].reset_index()
    )
    expected_conclusions = ("identified_range",) * 4 + ("repeated_interval",)
    if tuple(ordered["conclusion_type"]) != expected_conclusions:
        raise ValueError("A4 conclusion types do not match their analysis routes")
    if set(ordered["effect_scale"]) != {"risk_difference_tau"}:
        raise ValueError(
            "A4 reliability widths must share the fixed-horizon risk-difference scale"
        )
    bounded = ordered["analysis_role"].eq("prespecified_bounded_departure")
    if (
        not ordered.loc[bounded, "sensitivity_parameter"].eq(0.20).all()
        or not ordered.loc[bounded, "sensitivity_parameter_unit"]
        .eq("risk_probability_difference")
        .all()
        or ordered.loc[~bounded, "sensitivity_parameter"].notna().any()
    ):
        raise ValueError(
            "A4 bounded-departure rows must carry the prespecified 0.20 risk-probability limit"
        )
    x = np.arange(len(ordered))
    figure, axes_grid = plt.subplots(2, 2, figsize=(7.2, 6.1))
    axes = axes_grid.ravel()
    role_styles = {
        "prespecified_bounded_departure": (_BLUE, "o"),
        "unrestricted_worst_case": (_CHARCOAL, "D"),
        "repeated_monitoring": (_VERMILLION, "s"),
    }

    for position, (_, row) in enumerate(ordered.iterrows()):
        color, marker = role_styles[str(row["analysis_role"])]
        axes[0].errorbar(
            position,
            row["coverage"],
            yerr=np.array(
                [
                    [row["coverage"] - row["coverage_low"]],
                    [row["coverage_high"] - row["coverage"]],
                ]
            ),
            color=color,
            marker=marker,
            markerfacecolor="white",
            linestyle="none",
            capsize=2,
        )
    axes[0].axhline(0.95, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8)
    axes[0].set(
        title="Contains the configured effect",
        ylabel="Trial proportion",
        ylim=(0, 1.03),
    )

    for position, (_, row) in enumerate(ordered.iterrows()):
        color, marker = role_styles[str(row["analysis_role"])]
        axes[1].errorbar(
            position,
            row["mean_width"],
            yerr=np.array(
                [
                    [row["mean_width"] - row["mean_width_low"]],
                    [row["mean_width_high"] - row["mean_width"]],
                ]
            ),
            color=color,
            marker=marker,
            markerfacecolor="white",
            linestyle="none",
            capsize=2,
        )
    axes[1].set(
        title="Range or interval width",
        ylabel="Mean risk-difference width\n(0-1 scale)",
        ylim=(0, max(float(ordered["mean_width_high"].max()) * 1.12, 0.01)),
    )

    for position, (_, row) in enumerate(ordered.iterrows()):
        axes[2].errorbar(
            position - 0.08,
            row["fit_failure_rate"],
            yerr=np.array(
                [
                    [row["fit_failure_rate"] - row["fit_failure_rate_low"]],
                    [row["fit_failure_rate_high"] - row["fit_failure_rate"]],
                ]
            ),
            color=_CHARCOAL,
            marker="x",
            linestyle="none",
            capsize=2,
        )
    sequential = ordered.loc[ordered["series_id"].eq("TE-S09")].iloc[0]
    early_stop = float(sequential["early_stop_rate"])
    axes[2].errorbar(
        4.08,
        early_stop,
        yerr=np.array(
            [
                [early_stop - float(sequential["early_stop_rate_low"])],
                [float(sequential["early_stop_rate_high"]) - early_stop],
            ]
        ),
        color=_BLUE,
        marker="o",
        markerfacecolor="white",
        linestyle="none",
        capsize=2,
    )
    axes[2].set(title="Execution", ylabel="Trial proportion", ylim=(0, 1.03))
    axes[2].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=_CHARCOAL,
                marker="x",
                linestyle="none",
                label="Analysis failure",
            ),
            Line2D(
                [0],
                [0],
                color=_BLUE,
                marker="o",
                markerfacecolor="white",
                linestyle="none",
                label="Stopped at an interim look",
            ),
        ],
        frameon=False,
        fontsize=5.7,
        loc="upper left",
    )

    bias = float(sequential["bias"])
    rmse = float(sequential["rmse"])
    axes[3].errorbar(
        0,
        bias,
        yerr=np.array(
            [
                [bias - float(sequential["bias_low"])],
                [float(sequential["bias_high"]) - bias],
            ]
        ),
        color=_VERMILLION,
        marker="s",
        markerfacecolor="white",
        linestyle="none",
        capsize=2,
        label="Mean signed error",
    )
    axes[3].errorbar(
        1,
        rmse,
        yerr=np.array(
            [
                [rmse - float(sequential["rmse_low"])],
                [float(sequential["rmse_high"]) - rmse],
            ]
        ),
        color=_BLUE,
        marker="o",
        markerfacecolor="white",
        linestyle="none",
        capsize=2,
        label="Root mean squared error",
    )
    axes[3].axhline(0, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8)
    axes[3].set(
        title="Sequential estimation error",
        ylabel="Risk difference (0-1 scale)",
        xticks=(0, 1),
        xticklabels=("Signed bias", "RMSE"),
    )
    axes[3].tick_params(axis="x", labelsize=6)

    tick_labels = [labels[identity] for identity in order]
    for axis in axes[:3]:
        axis.set_xticks(x, tick_labels, rotation=25, ha="right")
        axis.tick_params(axis="x", labelsize=5.6)
    for panel, axis in enumerate(axes):
        axis.grid(axis="y")
        axis.text(
            -0.16,
            1.06,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
    trial_counts = tuple(int(value) for value in ordered["independent_trials"].unique())
    if len(trial_counts) != 1:
        raise ValueError(
            "Identification-reliability series require a common trial count"
        )
    figure.text(
        0.5,
        0.01,
        f"{trial_counts[0]} independent trials per analysis; points show estimates and 95% intervals",
        ha="center",
        fontsize=7,
    )
    figure.subplots_adjust(
        left=0.09, right=0.99, top=0.91, bottom=0.18, hspace=0.62, wspace=0.34
    )
    return figure


def _select_programme_estimate(
    estimates: pd.DataFrame, property_id: str
) -> pd.DataFrame:
    rows = estimates.loc[
        estimates["property_id"].eq(property_id) & estimates["group"].eq("overall"),
        ["trial_id", "design_profile_id", "estimate"],
    ].rename(columns={"estimate": "value"})
    if rows["trial_id"].duplicated().any():
        raise ValueError(f"Programme estimate is not unique by trial: {property_id}")
    return rows


def _trial_designs_figure(root: Path) -> Figure:
    data = pd.read_csv(root.parent / "data/design_properties.csv")
    _require_columns(
        data,
        {
            "independence_unit_id",
            "design_profile_id",
            "property_id",
            "estimate",
            "unit",
        },
    )
    figure, axes = plt.subplots(4, 2, figsize=(7.2, 9.2))
    panels = axes.ravel()

    balance = _design_property(data, "allocation.maximum_baseline_smd").sort_values(
        "estimate"
    )
    randomization = _design_property(
        data, "allocation.randomization_p95_maximum_baseline_smd"
    ).set_index("independence_unit_id")["estimate"]
    positions = np.arange(1, len(balance) + 1)
    panels[0].scatter(
        positions,
        balance["estimate"],
        color=_BLUE,
        marker="o",
        facecolors="none",
        s=15,
        label="Observed",
    )
    panels[0].scatter(
        positions,
        [randomization.loc[trial_id] for trial_id in balance["independence_unit_id"]],
        color=_VERMILLION,
        marker="_",
        s=28,
        label="Randomization 95th percentile",
    )
    panels[0].set(
        xlabel="Independent trial",
        ylabel="Largest standardized\narm difference",
        title="Baseline balance",
        ylim=(0.018, 0.14),
    )
    panels[0].legend(frameon=False, loc="upper left", fontsize=7)

    _pragmatic_arm_scatter(
        panels[1],
        data,
    )
    panels[1].set(title="Pragmatic trial processes", ylim=(-0.02, 1.12))

    _category_scatter(
        panels[2],
        data,
        profile_id="TE-DP03",
        property_ids=(
            "covariate.age_bmi_spearman",
            "covariate.age_event_spearman",
            "covariate.bmi_event_spearman",
        ),
        labels=("Age-BMI", "Age-event", "BMI-event"),
        ylabel="Spearman correlation",
    )
    panels[2].axhline(0, color=_CHARCOAL, linewidth=0.6)
    panels[2].set_title("Baseline and outcome dependence")

    _tier_category_scatter(
        panels[3],
        data,
        profile_id="TE-DP04",
        property_ids=(
            "ascertainment.validation_fraction",
            "ascertainment.sensitivity",
            "ascertainment.specificity",
            "ascertainment.observed_event_fraction",
            "ascertainment.adjudicated_event_fraction",
        ),
        labels=(
            "Validation\nsample",
            "Sensitivity",
            "Specificity",
            "Observed\npositive",
            "Adjudicated\npositive",
        ),
        ylabel="Participant proportion",
    )
    panels[3].set_title("Endpoint validation study")
    panels[3].tick_params(axis="x", labelsize=6.5, rotation=0)
    for label in panels[3].get_xticklabels():
        label.set_horizontalalignment("center")

    cluster_categories = (
        ("TE-DP05", 12, "Cluster\nparallel"),
        ("TE-DP06", 8, "Stepped\nwedge"),
    )
    cluster_effect = data.loc[data["property_id"].eq("cluster.design_effect")]
    for position, (profile_id, expected_trials, _) in enumerate(cluster_categories):
        values = cluster_effect.loc[
            cluster_effect["design_profile_id"].eq(profile_id),
            "estimate",
        ].sort_values()
        if len(values) != expected_trials:
            raise ValueError(
                f"Cluster design display requires {expected_trials} {profile_id} trials"
            )
        offsets = np.linspace(-0.10, 0.10, len(values))
        panels[4].scatter(
            position + offsets,
            values,
            color=_BLUE,
            marker="o",
            facecolors="none",
            s=15,
        )
        panels[4].scatter(
            position,
            float(values.median()),
            color=_VERMILLION,
            marker="D",
            s=18,
            zorder=3,
        )
    panels[4].axhline(1, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8)
    panels[4].set(
        xticks=np.arange(len(cluster_categories)),
        xticklabels=[label for _, _, label in cluster_categories],
        ylabel="Event-variance inflation",
        title="Observed cluster dependence",
    )
    panels[4].tick_params(axis="x", labelsize=7)

    rollout = data.loc[
        data["property_id"].str.startswith("rollout.switch_day.sequence_")
    ].copy()
    rollout["sequence"] = (
        rollout["property_id"].str.rsplit("_", n=1).str[-1].astype(int)
    )
    switch_days = rollout.groupby("sequence")["estimate"].agg(["min", "max"])
    if not np.allclose(switch_days["min"], switch_days["max"]):
        raise ValueError(
            "Staggered-wedge trials disagree on the public rollout schedule"
        )
    starts = switch_days["min"].to_numpy(dtype=float)
    intervention = starts[None, :] >= starts[:, None]
    for sequence, period in np.ndindex(intervention.shape):
        panels[5].add_patch(
            Rectangle(
                (period - 0.5, sequence - 0.5),
                1,
                1,
                facecolor=_BLUE if intervention[sequence, period] else _LIGHT_GREY,
                edgecolor="white",
                linewidth=0.6,
            )
        )
    panels[5].set(
        xlim=(-0.5, len(starts) - 0.5),
        ylim=(len(starts) - 0.5, -0.5),
        xticks=np.arange(len(starts)),
        xticklabels=[f"Day {int(day)}" for day in starts],
        yticks=np.arange(len(starts)),
        yticklabels=[f"Sequence {index}" for index in range(1, len(starts) + 1)],
        xlabel="Calendar period start",
        ylabel="Randomized sequence",
        title="Treatment rollout",
    )
    panels[5].tick_params(axis="x", labelsize=7)
    panels[5].tick_params(axis="y", labelsize=7)
    panels[5].text(
        1,
        len(starts) - 1,
        "Control",
        color=_CHARCOAL,
        ha="center",
        va="center",
        fontsize=7,
    )
    panels[5].text(
        len(starts) - 1,
        0,
        "Intervention",
        color="white",
        ha="center",
        va="center",
        fontsize=7,
    )

    period_rows = data.loc[
        data["design_profile_id"].eq("TE-DP06")
        & data["property_id"].str.startswith("rollout.event_rate.period_")
        & data["assumption_tier"].isin(("A1", "A2"))
    ].copy()
    period_rows["period"] = (
        period_rows["property_id"].str.rsplit("_", n=1).str[-1].astype(int)
    )
    for tier, colour, linestyle, label in (
        ("A1", _BLUE, "-", "No secular trend"),
        ("A2", _VERMILLION, (0, (5, 2)), "Secular trend"),
    ):
        tier_rows = period_rows.loc[period_rows["assumption_tier"].eq(tier)]
        matrix = tier_rows.pivot(
            index="independence_unit_id", columns="period", values="estimate"
        )
        relative = matrix.div(matrix[1], axis=0)
        summary = pd.DataFrame(
            {
                "period": relative.columns,
                "mean": relative.mean(axis=0),
                "sem": relative.sem(axis=0),
            }
        )
        independent_trials = int(tier_rows["independence_unit_id"].nunique())
        critical = float(stats.t.ppf(0.975, df=independent_trials - 1))
        panels[6].errorbar(
            summary["period"],
            summary["mean"],
            yerr=summary["sem"] * critical,
            color=colour,
            linestyle=linestyle,
            marker="o" if tier == "A1" else "s",
            markerfacecolor="white",
            linewidth=1.0,
            capsize=2,
            label=label,
        )
    panels[6].set(
        xticks=(1, 2, 3, 4),
        xlabel="Calendar period",
        ylabel="Baseline rate ratio to period 1",
        title="Treatment-adjusted calendar trend",
    )
    panels[6].legend(frameon=False, loc="best", fontsize=7)

    sequential = data.loc[data["design_profile_id"].eq("TE-DP07")]
    planned_fractions = []
    planned_boundaries = []
    for look in range(1, 4):
        fractions = _design_property(
            sequential, f"interim.information_fraction.look_{look}"
        )["estimate"]
        boundaries = _design_property(sequential, f"interim.critical_z.look_{look}")[
            "estimate"
        ]
        if not np.allclose(fractions, fractions.iloc[0]) or not np.allclose(
            boundaries, boundaries.iloc[0]
        ):
            raise ValueError(
                "Group-sequential trials disagree on the released monitoring plan"
            )
        planned_fractions.append(float(fractions.iloc[0]))
        planned_boundaries.append(float(boundaries.iloc[0]))
    panels[7].plot(
        planned_fractions,
        planned_boundaries,
        color=_CHARCOAL,
        linestyle=(0, (5, 2)),
        marker="_",
        linewidth=1.2,
        label="Efficacy boundary",
    )
    observed = (
        _design_property(sequential, "interim.analysis_information_fraction")
        .loc[:, ["independence_unit_id", "estimate"]]
        .rename(columns={"estimate": "information_fraction"})
        .merge(
            _design_property(sequential, "interim.observed_absolute_z")
            .loc[:, ["independence_unit_id", "estimate"]]
            .rename(columns={"estimate": "absolute_z"}),
            on="independence_unit_id",
            validate="one_to_one",
        )
        .merge(
            _design_property(sequential, "interim.boundary_margin")
            .loc[:, ["independence_unit_id", "estimate"]]
            .rename(columns={"estimate": "boundary_margin"}),
            on="independence_unit_id",
            validate="one_to_one",
        )
    )
    crossed = observed["boundary_margin"].ge(0)
    panels[7].scatter(
        observed.loc[~crossed, "information_fraction"],
        observed.loc[~crossed, "absolute_z"],
        color=_BLUE,
        marker="o",
        facecolors="white",
        s=24,
        label="Did not cross",
        zorder=3,
    )
    panels[7].scatter(
        observed.loc[crossed, "information_fraction"],
        observed.loc[crossed, "absolute_z"],
        color=_VERMILLION,
        marker="D",
        s=24,
        label="Crossed",
        zorder=3,
    )
    panels[7].set(
        xticks=planned_fractions,
        xlabel="Information fraction",
        ylabel="Absolute Z statistic",
        title="Observed monitoring decisions",
        ylim=(0.0, 3.45),
    )
    panels[7].legend(frameon=False, loc="best", fontsize=7)
    for panel, axis in enumerate(panels):
        axis.text(
            -0.25,
            1.06,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
        axis.grid(axis="y")
    figure.subplots_adjust(
        left=0.1, right=0.98, top=0.95, bottom=0.07, hspace=0.62, wspace=0.42
    )
    return figure


def _design_consequences_figure(root: Path) -> Figure:
    data = pd.read_csv(root.parent / "data/design_comparisons.csv")
    _require_columns(
        data,
        {
            "design_profile_id",
            "qualified_estimate",
            "qualified_interval_low",
            "qualified_interval_high",
            "naive_estimate",
            "naive_interval_low",
            "naive_interval_high",
        },
    )
    panel_specs = (
        ("TE-DP03", "Covariate adjustment", None),
        ("TE-DP04", "Endpoint correction", None),
        ("TE-DP05", "Cluster uncertainty", None),
        ("TE-DP06", "Calendar-trend bias", "bias"),
        ("TE-DP06", "Calendar-trend coverage", "coverage"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 5.5))
    for panel, (axis, (profile_id, title, tier)) in enumerate(
        zip(axes.ravel()[:5], panel_specs, strict=True)
    ):
        if profile_id == "TE-DP05":
            cluster = pd.read_csv(
                root.parent
                / "data/operating_characteristics/clustered_design/cluster_response_summary.csv"
            )
            _require_columns(
                cluster,
                {
                    "hazard_ratio_90_to_10",
                    "measure",
                    "mean",
                    "ci_low",
                    "ci_high",
                },
            )
            for measure, colour, marker, linestyle, label in (
                ("cluster_robust_covered", _BLUE, "o", "-", "Cluster-aware"),
                (
                    "participant_independent_covered",
                    _VERMILLION,
                    "s",
                    (0, (5, 2)),
                    "Participant-independent",
                ),
            ):
                coverage = cluster.loc[cluster["measure"].eq(measure)].sort_values(
                    "hazard_ratio_90_to_10"
                )
                x = coverage["hazard_ratio_90_to_10"].to_numpy(dtype=float)
                value = coverage["mean"].to_numpy(dtype=float)
                axis.errorbar(
                    x,
                    value,
                    yerr=np.vstack(
                        (
                            value - coverage["ci_low"].to_numpy(dtype=float),
                            coverage["ci_high"].to_numpy(dtype=float) - value,
                        )
                    ),
                    color=colour,
                    marker=marker,
                    markerfacecolor="white",
                    linestyle=linestyle,
                    linewidth=1.2,
                    capsize=2,
                    label=label,
                )
            axis.axhline(0.95, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8)
            release_ratio = float(
                cluster.loc[
                    cluster["setting"].eq("source_anchored"), "hazard_ratio_90_to_10"
                ].iloc[0]
            )
            maximum_ratio = float(cluster["hazard_ratio_90_to_10"].max())
            axis.set(
                xlabel="90th:10th percentile hazard ratio",
                ylabel="Empirical coverage",
                ylim=(0.68, 0.99),
                xticks=(1.0, release_ratio, maximum_ratio),
                xticklabels=(
                    "1.0",
                    f"{release_ratio:.2f}\nexternal reference",
                    f"{maximum_ratio:.1f}",
                ),
                title="Cluster interval coverage",
            )
            axis.legend(frameon=False, loc="lower left", fontsize=7)
            axis.text(
                -0.14,
                1.11,
                chr(ord("a") + panel),
                transform=axis.transAxes,
                fontweight="bold",
            )
            axis.grid(axis="y")
            continue
        if profile_id == "TE-DP06":
            response = pd.read_csv(
                root.parent
                / "data/operating_characteristics/stepped_wedge/stepped_wedge_response_summary.csv"
            )
            _require_columns(
                response,
                {
                    "secular_hazard_ratio_period_4_to_1",
                    "measure",
                    "mean",
                    "ci_low",
                    "ci_high",
                },
            )
            suffix = "bias" if tier == "bias" else "covered"
            for prefix, colour, marker, linestyle, label in (
                ("period_adjusted", _BLUE, "o", "-", "Period-adjusted"),
                ("period_omitting", _VERMILLION, "s", (0, (5, 2)), "Period omitted"),
            ):
                values = response.loc[
                    response["measure"].eq(f"{prefix}_{suffix}")
                ].sort_values("secular_hazard_ratio_period_4_to_1")
                x = values["secular_hazard_ratio_period_4_to_1"].to_numpy(dtype=float)
                mean = values["mean"].to_numpy(dtype=float)
                axis.errorbar(
                    x,
                    mean,
                    yerr=np.vstack(
                        (
                            mean - values["ci_low"].to_numpy(dtype=float),
                            values["ci_high"].to_numpy(dtype=float) - mean,
                        )
                    ),
                    color=colour,
                    marker=marker,
                    markerfacecolor="white",
                    linestyle=linestyle,
                    linewidth=1.2,
                    capsize=2,
                    label=label,
                )
                if tier == "coverage":
                    axis.annotate(
                        label,
                        (x[-1], mean[-1]),
                        xytext=(-5, 0),
                        textcoords="offset points",
                        ha="right",
                        va="center",
                        fontsize=7,
                        color=colour,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
                    )
            axis.set(
                xlabel="Period 4:1 secular hazard ratio",
                xticks=(1.0, 1.65, 2.0, 3.0),
                xticklabels=("1.0", "1.65\nprespecified", "2.0", "3.0"),
            )
            if suffix == "bias":
                axis.axhline(0, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8)
                axis.set(
                    ylabel="Treatment risk-difference bias",
                    title="Calendar-trend bias",
                )
                axis.legend(frameon=False, loc="upper left", fontsize=7)
            else:
                axis.axhline(
                    0.95, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8
                )
                axis.set(
                    ylabel="Empirical coverage",
                    ylim=(-0.03, 1.02),
                    xlim=(0.92, 3.15),
                    title="Calendar-trend coverage",
                )
            axis.text(
                -0.14,
                1.11,
                chr(ord("a") + panel),
                transform=axis.transAxes,
                fontweight="bold",
            )
            axis.grid(axis="y")
            continue
        rows = data.loc[data["design_profile_id"].eq(profile_id)]
        if tier is not None:
            rows = rows.loc[rows["assumption_tier"].eq(tier)]
        rows = rows.sort_values("independence_unit_id").reset_index(drop=True)
        if len(rows) != 4:
            suffix = f" {tier}" if tier is not None else ""
            raise ValueError(
                f"Design analysis requires four comparisons for {profile_id}{suffix}"
            )
        y = np.arange(1, len(rows) + 1)
        for position, row in zip(y, rows.itertuples(index=False), strict=True):
            axis.plot(
                (row.qualified_estimate, row.naive_estimate),
                (position - 0.08, position + 0.08),
                color=_LIGHT_GREY,
                linewidth=0.8,
                zorder=0,
            )
        for offset, prefix, colour, marker, label in (
            (-0.08, "qualified", _BLUE, "o", "Prespecified analysis"),
            (0.08, "naive", _VERMILLION, "s", "Key design feature omitted"),
        ):
            estimate = rows[f"{prefix}_estimate"].to_numpy()
            low = rows[f"{prefix}_interval_low"].to_numpy()
            high = rows[f"{prefix}_interval_high"].to_numpy()
            axis.errorbar(
                estimate,
                y + offset,
                xerr=np.vstack((estimate - low, high - estimate)),
                color=colour,
                marker=marker,
                markerfacecolor="white" if prefix == "qualified" else colour,
                linestyle="none",
                capsize=2,
                linewidth=1.2,
                markersize=4,
                label=label,
            )
        axis.axvline(0, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8)
        axis.set(
            yticks=y,
            yticklabels=[str(index) for index in y],
            ylim=(0.7, 4.65),
            xlabel="Treated - control event risk",
            title=title,
        )
        qualified_width = (
            rows["qualified_interval_high"] - rows["qualified_interval_low"]
        )
        naive_width = rows["naive_interval_high"] - rows["naive_interval_low"]
        if profile_id in {"TE-DP03", "TE-DP05"}:
            summary = f"Median interval-width ratio {np.median(qualified_width / naive_width):.2f}"
        else:
            summary = (
                "Median absolute effect shift "
                f"{np.median(np.abs(rows['qualified_estimate'] - rows['naive_estimate'])):.3f}"
            )
        axis.text(
            0.98,
            0.98,
            summary,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color=_CHARCOAL,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )
        if panel in (0, 3):
            axis.set_ylabel("Independent trial")
        axis.text(
            -0.14,
            1.11,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
        axis.grid(axis="x")
    sequential_axis = axes.ravel()[5]
    sequential = pd.read_csv(
        root.parent
        / "data/operating_characteristics/group_sequential/group_sequential_operating_characteristics.csv"
    )
    _require_columns(
        sequential,
        {
            "signal_to_final_boundary",
            "repeated_interval_coverage",
            "repeated_interval_coverage_ci_low",
            "repeated_interval_coverage_ci_high",
            "ordinary_interval_coverage",
            "ordinary_interval_coverage_ci_low",
            "ordinary_interval_coverage_ci_high",
        },
    )
    sequential = sequential.sort_values("signal_to_final_boundary")
    x = sequential["signal_to_final_boundary"].to_numpy(dtype=float)
    for prefix, colour, marker, linestyle, label in (
        ("repeated_interval", _BLUE, "o", "-", "Sequential interval"),
        ("ordinary_interval", _VERMILLION, "s", (0, (5, 2)), "Ordinary 1.96 interval"),
    ):
        value = sequential[f"{prefix}_coverage"].to_numpy(dtype=float)
        sequential_axis.errorbar(
            x,
            value,
            yerr=np.vstack(
                (
                    value
                    - sequential[f"{prefix}_coverage_ci_low"].to_numpy(dtype=float),
                    sequential[f"{prefix}_coverage_ci_high"].to_numpy(dtype=float)
                    - value,
                )
            ),
            color=colour,
            marker=marker,
            markerfacecolor="white",
            linestyle=linestyle,
            linewidth=1.2,
            capsize=2,
            label=label,
        )
        sequential_axis.annotate(
            label,
            (x[-1], value[-1]),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7,
            color=colour,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        )
    sequential_axis.axhline(0.95, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=0.8)
    sequential_axis.set(
        xlabel="Final mean Z / final boundary",
        ylabel="Empirical coverage",
        ylim=(0.925, 0.985),
        xlim=(-0.05, 2.15),
        title="Sequential interval coverage",
    )
    sequential_axis.text(
        -0.14,
        1.11,
        "f",
        transform=sequential_axis.transAxes,
        fontweight="bold",
    )
    sequential_axis.grid(axis="y")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(
        handles=handles,
        labels=labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        fontsize=7,
    )
    return _finish(figure, bottom=0.11, left=0.09, hspace=0.58)


def _design_property(data: pd.DataFrame, property_id: str) -> pd.DataFrame:
    rows = data.loc[data["property_id"].eq(property_id)].copy()
    if rows.empty or rows["independence_unit_id"].duplicated().any():
        raise ValueError(
            f"Design property must have one value per contributing trial: {property_id}"
        )
    return rows


def _category_scatter(
    axis: Axes,
    data: pd.DataFrame,
    *,
    profile_id: str,
    property_ids: tuple[str, ...],
    labels: tuple[str, ...],
    ylabel: str,
) -> None:
    selected = data.loc[
        data["design_profile_id"].eq(profile_id)
        & data["property_id"].isin(property_ids),
        ["independence_unit_id", "property_id", "estimate"],
    ]
    matrix = selected.pivot(
        index="independence_unit_id", columns="property_id", values="estimate"
    ).reindex(columns=property_ids)
    if matrix.empty or matrix.isna().any().any():
        raise ValueError(f"Design display is incomplete for {profile_id}")
    x = np.arange(len(property_ids))
    for position, property_id in enumerate(property_ids):
        values = matrix[property_id].sort_values().to_numpy(dtype=float)
        offset = (
            np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
        )
        axis.scatter(
            np.full(len(values), position) + offset,
            values,
            color=_BLUE,
            marker="o",
            facecolors="none",
            s=15,
        )
        axis.scatter(
            position, np.median(values), color=_VERMILLION, marker="D", s=18, zorder=3
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel(ylabel)


def _tier_category_scatter(
    axis: Axes,
    data: pd.DataFrame,
    *,
    profile_id: str,
    property_ids: tuple[str, ...],
    labels: tuple[str, ...],
    ylabel: str,
) -> None:
    tiers = ("A1", "A2", "A3", "A4")
    markers = ("o", "s", "^", "D")
    tier_offsets = np.linspace(-0.15, 0.15, len(tiers))
    selected = data.loc[
        data["design_profile_id"].eq(profile_id)
        & data["property_id"].isin(property_ids),
        ["independence_unit_id", "assumption_tier", "property_id", "estimate"],
    ]
    for position, property_id in enumerate(property_ids):
        for tier, marker, tier_offset in zip(tiers, markers, tier_offsets, strict=True):
            values = selected.loc[
                selected["property_id"].eq(property_id)
                & selected["assumption_tier"].eq(tier),
                "estimate",
            ].sort_values()
            if len(values) != 4:
                raise ValueError(
                    f"Tiered design display requires four {profile_id}/{tier}/{property_id} trials"
                )
            jitter = np.linspace(-0.018, 0.018, len(values))
            axis.scatter(
                position + tier_offset + jitter,
                values,
                color=_BLUE,
                marker=marker,
                facecolors="none",
                linewidths=0.7,
                s=11,
            )
            axis.scatter(
                position + tier_offset,
                float(values.median()),
                color=_VERMILLION,
                marker=marker,
                s=18,
                zorder=3,
            )
    axis.set_xticks(np.arange(len(property_ids)), labels)
    axis.set_ylabel(ylabel)
    axis.legend(
        handles=tuple(
            (
                Line2D(
                    [0],
                    [0],
                    color=_BLUE,
                    marker="o",
                    markerfacecolor="white",
                    linestyle="none",
                    label="Trial",
                ),
                Line2D(
                    [0],
                    [0],
                    color=_VERMILLION,
                    marker="o",
                    linestyle="none",
                    label="Tier median",
                ),
            )
        )
        + tuple(
            Line2D(
                [0],
                [0],
                color=_CHARCOAL,
                marker=marker,
                markerfacecolor="white",
                linestyle="none",
                label=tier,
            )
            for tier, marker in zip(tiers, markers, strict=True)
        ),
        frameon=False,
        loc="lower center",
        ncols=3,
        fontsize=7,
    )


def _pragmatic_arm_scatter(axis: Axes, data: pd.DataFrame) -> None:
    property_names = (
        "mean_exposure_adherence",
        "discontinuation_fraction",
        "rescue_fraction",
        "switch_fraction",
        "intercurrent_event_fraction",
        "per_protocol_fraction",
    )
    labels = (
        "Dose\nreceived",
        "Discontinued",
        "Rescue\ntreatment",
        "Treatment\nswitch",
        "Post-randomization\nevent",
        "Per-protocol\neligible",
    )
    x = np.arange(len(property_names), dtype=float)
    arm_styles = (
        ("control", -0.13, _BLUE, "o"),
        ("treated", 0.13, _VERMILLION, "s"),
    )
    for arm, arm_offset, colour, marker in arm_styles:
        property_ids = tuple(f"pragmatic.{name}.{arm}" for name in property_names)
        selected = data.loc[
            data["design_profile_id"].eq("TE-DP02")
            & data["property_id"].isin(property_ids),
            ["independence_unit_id", "property_id", "estimate"],
        ]
        matrix = selected.pivot(
            index="independence_unit_id",
            columns="property_id",
            values="estimate",
        ).reindex(columns=property_ids)
        if len(matrix) != 24 or matrix.isna().any().any():
            raise ValueError(f"Pragmatic arm display is incomplete for {arm}")
        for position, property_id in enumerate(property_ids):
            values = matrix[property_id].sort_values().to_numpy(dtype=float)
            jitter = np.linspace(-0.05, 0.05, len(values))
            axis.scatter(
                position + arm_offset + jitter,
                values,
                color=colour,
                marker=marker,
                facecolors="none",
                linewidths=0.65,
                s=10,
            )
            axis.scatter(
                position + arm_offset,
                float(np.median(values)),
                color=colour,
                marker=marker,
                edgecolors=colour,
                linewidths=0.8,
                s=22,
                zorder=3,
            )
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.tick_params(axis="x", labelsize=7)
    axis.set_ylabel("Participant proportion")
    axis.legend(
        handles=(
            Line2D(
                [0],
                [0],
                color=_BLUE,
                marker="o",
                markerfacecolor="white",
                linestyle="none",
                label="Control",
            ),
            Line2D(
                [0],
                [0],
                color=_VERMILLION,
                marker="s",
                markerfacecolor="white",
                linestyle="none",
                label="Treated",
            ),
            Line2D(
                [0],
                [0],
                color=_CHARCOAL,
                marker="o",
                markerfacecolor="white",
                linestyle="none",
                label="Trial",
            ),
            Line2D(
                [0],
                [0],
                color=_CHARCOAL,
                marker="o",
                markerfacecolor=_CHARCOAL,
                linestyle="none",
                label="Arm median",
            ),
        ),
        frameon=False,
        loc="upper center",
        ncols=2,
        fontsize=7,
    )


def _survival_figure(root: Path) -> Figure:
    data = pd.read_csv(root / "outcome_survival.csv")
    _require_columns(
        data,
        {
            "arm",
            "time",
            "source_survival",
            "mean_survival",
            "predictive_95_low",
            "predictive_95_high",
            "source_at_risk",
            "mean_at_risk",
            "at_risk_predictive_95_low",
            "at_risk_predictive_95_high",
        },
    )
    figure, (survival_axis, risk_axis) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.3),
        sharex=True,
        gridspec_kw={"height_ratios": (1.8, 1.0)},
    )
    colours = {"Conventional": _BLUE, "No-touch": _VERMILLION}
    for arm, rows in data.groupby("arm", sort=False):
        ordered = rows.sort_values("time")
        colour = colours[str(arm)]
        survival_axis.fill_between(
            ordered["time"],
            ordered["predictive_95_low"],
            ordered["predictive_95_high"],
            color=colour,
            alpha=0.14,
            linewidth=0,
        )
        survival_axis.plot(
            ordered["time"],
            ordered["source_survival"],
            color=colour,
            linestyle="-",
            marker="D",
            markersize=4,
            linewidth=1.6,
        )
        survival_axis.plot(
            ordered["time"],
            ordered["mean_survival"],
            color=colour,
            linestyle=(0, (5, 2)),
            marker="o",
            markerfacecolor="white",
            markersize=4,
            linewidth=1.6,
        )
        risk_axis.fill_between(
            ordered["time"],
            ordered["at_risk_predictive_95_low"],
            ordered["at_risk_predictive_95_high"],
            color=colour,
            alpha=0.14,
            linewidth=0,
        )
        risk_axis.plot(
            ordered["time"],
            ordered["source_at_risk"],
            color=colour,
            linestyle="-",
            marker="D",
            markersize=4,
            linewidth=1.4,
        )
        risk_axis.plot(
            ordered["time"],
            ordered["mean_at_risk"],
            color=colour,
            linestyle=(0, (5, 2)),
            marker="o",
            markerfacecolor="white",
            markersize=4,
            linewidth=1.4,
        )
    arm_handles = [
        Line2D([0], [0], color=colour, linewidth=2, label=arm)
        for arm, colour in colours.items()
    ]
    series_handles = [
        Line2D(
            [0], [0], color=_CHARCOAL, marker="D", linewidth=1.6, label="Source trial"
        ),
        Line2D(
            [0],
            [0],
            color=_CHARCOAL,
            marker="o",
            markerfacecolor="white",
            linestyle=(0, (5, 2)),
            linewidth=1.6,
            label="Repeated-trial mean",
        ),
    ]
    first = survival_axis.legend(
        handles=arm_handles, loc="lower left", frameon=False, title="Randomized arm"
    )
    survival_axis.add_artist(first)
    survival_axis.legend(
        handles=series_handles, loc="lower right", frameon=False, title="Series"
    )
    survival_axis.set(ylabel="MACCE-free survival probability", ylim=(0.9, 1.0))
    survival_axis.grid(axis="y")
    risk_axis.set(xlabel="Days since randomization", ylabel="Participants at risk")
    risk_axis.grid(axis="y")
    survival_axis.text(
        -0.08, 1.04, "a", transform=survival_axis.transAxes, fontweight="bold"
    )
    risk_axis.text(-0.08, 1.04, "b", transform=risk_axis.transAxes, fontweight="bold")
    return _finish(figure, bottom=0.11, left=0.12, hspace=0.18)


def _ordinal_figure(root: Path) -> Figure:
    data = pd.read_csv(root / "outcome_ordinal.csv")
    _require_columns(
        data,
        {
            "arm",
            "category",
            "category_label",
            "category_probability_observed",
            "category_probability_simulated_mean",
            "category_probability_predictive_95_low",
            "category_probability_predictive_95_high",
            "cumulative_probability_observed",
            "cumulative_probability_simulated_mean",
            "cumulative_probability_predictive_95_low",
            "cumulative_probability_predictive_95_high",
        },
    )
    arms = tuple(data["arm"].drop_duplicates())
    if len(arms) != 2:
        raise ValueError("Ordinal figure requires exactly two randomized arms")
    figure = plt.figure(figsize=(7.2, 6.0))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    category_axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    cumulative_axis = figure.add_subplot(grid[1, :])
    for panel, (axis, arm) in enumerate(zip(category_axes, arms, strict=True)):
        rows = data.loc[data["arm"].eq(arm)].sort_values("category")
        x = np.arange(len(rows))
        axis.bar(
            x - 0.18,
            rows["category_probability_observed"],
            width=0.36,
            color=_LIGHT_GREY,
            edgecolor=_CHARCOAL,
            hatch="///",
            linewidth=0.7,
        )
        axis.bar(
            x + 0.18,
            rows["category_probability_simulated_mean"],
            width=0.36,
            color=_BLUE,
            edgecolor=_CHARCOAL,
            linewidth=0.7,
        )
        mean = rows["category_probability_simulated_mean"].to_numpy(dtype=float)
        axis.errorbar(
            x + 0.18,
            mean,
            yerr=np.vstack(
                (
                    mean
                    - rows["category_probability_predictive_95_low"].to_numpy(
                        dtype=float
                    ),
                    rows["category_probability_predictive_95_high"].to_numpy(
                        dtype=float
                    )
                    - mean,
                )
            ),
            color=_CHARCOAL,
            linestyle="none",
            linewidth=0.8,
            capsize=2,
        )
        axis.set_xticks(x, rows["category_label"], rotation=35, ha="right")
        axis.set_title(str(arm), fontsize=9)
        axis.text(
            -0.08,
            1.04,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
        axis.grid(axis="y")
    category_axes[0].set_ylabel("Participant probability")
    category_axes[1].sharey(category_axes[0])
    category_axes[1].tick_params(labelleft=False)
    category_axes[0].legend(
        handles=[
            Patch(
                facecolor=_LIGHT_GREY,
                edgecolor=_CHARCOAL,
                hatch="///",
                label="Source trial",
            ),
            Patch(
                facecolor=_BLUE,
                edgecolor=_CHARCOAL,
                label="Repeated-trial mean and 95% range",
            ),
        ],
        loc="upper right",
        frameon=False,
        fontsize=6,
    )

    colours = {
        arm: colour for arm, colour in zip(arms, (_BLUE, _VERMILLION), strict=True)
    }
    arm_handles: list[Line2D] = []
    for arm in arms:
        rows = data.loc[data["arm"].eq(arm)].sort_values("category")
        x = rows["category"].to_numpy(dtype=float)
        colour = colours[arm]
        cumulative_axis.plot(
            x,
            rows["cumulative_probability_observed"],
            color=colour,
            marker="D",
            linestyle="-",
            linewidth=1.4,
            markersize=4,
        )
        mean = rows["cumulative_probability_simulated_mean"].to_numpy(dtype=float)
        cumulative_axis.errorbar(
            x,
            mean,
            yerr=np.vstack(
                (
                    mean
                    - rows["cumulative_probability_predictive_95_low"].to_numpy(
                        dtype=float
                    ),
                    rows["cumulative_probability_predictive_95_high"].to_numpy(
                        dtype=float
                    )
                    - mean,
                )
            ),
            color=colour,
            marker="o",
            markerfacecolor="white",
            linestyle=(0, (5, 2)),
            linewidth=1.3,
            markersize=4,
            capsize=2,
        )
        arm_handles.extend(
            (
                Line2D(
                    [0],
                    [0],
                    color=colour,
                    marker="D",
                    linestyle="-",
                    label=f"{arm}: observed",
                ),
                Line2D(
                    [0],
                    [0],
                    color=colour,
                    marker="o",
                    markerfacecolor="white",
                    linestyle=(0, (5, 2)),
                    label=f"{arm}: simulated",
                ),
            )
        )
    cumulative_axis.set(
        xlabel="Modified Rankin Scale cutpoint",
        ylabel="Probability at or below cutpoint",
        xticks=data["category"].drop_duplicates(),
        ylim=(0, 1.03),
    )
    cumulative_axis.grid(axis="y")
    cumulative_axis.legend(
        handles=arm_handles, frameon=False, fontsize=6, ncol=2, loc="lower right"
    )
    cumulative_axis.text(
        -0.04, 1.04, "c", transform=cumulative_axis.transAxes, fontweight="bold"
    )
    figure.subplots_adjust(
        left=0.1, right=0.98, top=0.94, bottom=0.1, wspace=0.25, hspace=0.48
    )
    return figure


def _longitudinal_figure(root: Path) -> Figure:
    data = pd.read_csv(root / "outcome_longitudinal.csv")
    _require_columns(
        data,
        {
            "arm",
            "visit",
            "source_mean",
            "mean_median",
            "mean_interval_50_low",
            "mean_interval_50_high",
            "mean_interval_95_low",
            "mean_interval_95_high",
            "source_observations",
            "observations_median",
            "observations_interval_95_low",
            "observations_interval_95_high",
        },
    )
    operating_root = root.parent / "data/operating_characteristics/longitudinal"
    treatment = pd.read_csv(operating_root / "tereco_treatment_recovery.csv")
    linkage = pd.read_csv(operating_root / "tereco_linkage_response.csv")
    _require_columns(
        treatment,
        {
            "outcome",
            "worlds",
            "standardized_bias",
            "standardized_bias_simultaneous_ci_low",
            "standardized_bias_simultaneous_ci_high",
            "coverage",
        },
    )
    _require_columns(
        linkage,
        {
            "linkage_retention",
            "worlds",
            "correlation_mae_mean",
            "correlation_mae_ci_low",
            "correlation_mae_ci_high",
            "source_bootstrap_mean",
            "source_bootstrap_low",
            "source_bootstrap_high",
        },
    )
    if not treatment["worlds"].eq(200).all() or not linkage["worlds"].eq(200).all():
        raise ValueError("TERECO figure requires 200 independent worlds per analysis")
    arms = tuple(data["arm"].drop_duplicates())
    if len(arms) != 2:
        raise ValueError("Longitudinal figure requires exactly two randomized arms")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.1))
    colours = {
        _arm: colour for _arm, colour in zip(arms, (_BLUE, _VERMILLION), strict=True)
    }
    markers = {_arm: marker for _arm, marker in zip(arms, ("o", "s"), strict=True)}

    trajectory_axis = axes[0, 0]
    for arm in arms:
        rows = data.loc[data["arm"].eq(arm)].copy()
        x = np.arange(len(rows))
        median = rows["mean_median"].to_numpy()
        interval_95 = np.vstack(
            (
                median - rows["mean_interval_95_low"].to_numpy(),
                rows["mean_interval_95_high"].to_numpy() - median,
            )
        )
        interval_50 = np.vstack(
            (
                median - rows["mean_interval_50_low"].to_numpy(),
                rows["mean_interval_50_high"].to_numpy() - median,
            )
        )
        trajectory_axis.errorbar(
            x,
            median,
            yerr=interval_95,
            color=colours[arm],
            linestyle=(0, (5, 2)),
            marker=markers[arm],
            markerfacecolor="white",
            capsize=3,
            linewidth=1.4,
        )
        trajectory_axis.errorbar(
            x,
            median,
            yerr=interval_50,
            color=colours[arm],
            linewidth=3.2,
            linestyle="none",
        )
        trajectory_axis.plot(
            x,
            rows["source_mean"],
            color=colours[arm],
            linestyle="-",
            marker="D",
            markersize=4,
            linewidth=1.5,
        )
    trajectory_axis.set(
        title="Six-minute walk trajectory",
        ylabel="Distance (m)",
        xticks=np.arange(len(data.loc[data["arm"].eq(arms[0])])),
        xticklabels=data.loc[data["arm"].eq(arms[0]), "visit"],
    )
    arm_handles = [
        Line2D([0], [0], color=colours[arm], marker=markers[arm], label=str(arm))
        for arm in arms
    ]
    first_legend = trajectory_axis.legend(
        handles=arm_handles,
        frameon=False,
        fontsize=6,
        title="Randomized arm",
        loc="upper left",
    )
    trajectory_axis.add_artist(first_legend)
    trajectory_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=_CHARCOAL,
                marker="D",
                linestyle="-",
                label="Source trial",
            ),
            Line2D(
                [0],
                [0],
                color=_CHARCOAL,
                marker="o",
                markerfacecolor="white",
                linestyle=(0, (5, 2)),
                label="Repeated-trial median",
            ),
        ],
        frameon=False,
        fontsize=6,
        title="Series",
        loc="lower right",
    )
    trajectory_axis.grid(axis="y")
    trajectory_axis.text(
        -0.08, 1.04, "a", transform=trajectory_axis.transAxes, fontweight="bold"
    )
    trajectory_low = float(
        min(
            data["source_mean"].min(),
            data["mean_interval_95_low"].min(),
        )
    )
    trajectory_high = float(
        max(
            data["source_mean"].max(),
            data["mean_interval_95_high"].max(),
        )
    )
    trajectory_padding = 0.04 * (trajectory_high - trajectory_low)
    trajectory_axis.set_ylim(
        trajectory_low - trajectory_padding,
        trajectory_high + trajectory_padding,
    )

    treatment_axis = axes[0, 1]
    treatment = treatment.sort_values("standardized_bias").reset_index(drop=True)
    outcome_labels = {
        "Forced expiratory volume": "FEV1",
        "Forced vital capacity": "FVC",
        "SF-12 mental": "SF-12 mental",
        "SF-12 physical": "SF-12 physical",
        "Six-minute walk distance": "Six-minute walk",
        "Squat repetitions": "Squat repetitions",
    }
    if set(treatment["outcome"]) != set(outcome_labels):
        raise ValueError("TERECO treatment recovery requires all six declared outcomes")
    positions = np.arange(len(treatment))
    point = treatment["standardized_bias"].to_numpy(dtype=float)
    treatment_axis.errorbar(
        point,
        positions,
        xerr=np.vstack(
            (
                point
                - treatment["standardized_bias_simultaneous_ci_low"].to_numpy(
                    dtype=float
                ),
                treatment["standardized_bias_simultaneous_ci_high"].to_numpy(
                    dtype=float
                )
                - point,
            )
        ),
        color=_BLUE,
        marker="o",
        markerfacecolor="white",
        linestyle="none",
        capsize=2,
        linewidth=1.1,
    )
    treatment_axis.axvline(0, color=_CHARCOAL, linewidth=0.8, linestyle=(0, (3, 2)))
    treatment_axis.set(
        title="Treatment-effect recovery",
        xlabel="Bias (source SD)",
        yticks=positions,
        yticklabels=[outcome_labels[str(outcome)] for outcome in treatment["outcome"]],
    )
    treatment_axis.grid(axis="x")
    treatment_axis.text(
        -0.08, 1.04, "b", transform=treatment_axis.transAxes, fontweight="bold"
    )

    retention_axis = axes[1, 0]
    offsets = {_arm: offset for _arm, offset in zip(arms, (-0.08, 0.08), strict=True)}
    for arm in arms:
        rows = data.loc[data["arm"].eq(arm)].copy()
        x_positions = np.arange(len(rows), dtype=float) + offsets[arm]
        median = rows["observations_median"].to_numpy(dtype=float)
        interval = np.vstack(
            (
                median - rows["observations_interval_95_low"].to_numpy(dtype=float),
                rows["observations_interval_95_high"].to_numpy(dtype=float) - median,
            )
        )
        retention_axis.errorbar(
            x_positions,
            median,
            yerr=interval,
            color=colours[arm],
            marker=markers[arm],
            markerfacecolor="white",
            linestyle=(0, (5, 2)),
            linewidth=1.2,
            capsize=2,
        )
        retention_axis.scatter(
            np.arange(len(rows), dtype=float),
            rows["source_observations"],
            color=colours[arm],
            marker=markers[arm],
            s=18,
            zorder=3,
        )
    retention_axis.set(
        title="Follow-up retained",
        ylabel="Participants observed",
        xticks=np.arange(len(data.loc[data["arm"].eq(arms[0])])),
        xticklabels=data.loc[data["arm"].eq(arms[0]), "visit"],
    )
    retention_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=colours[arm],
                marker=markers[arm],
                linestyle="-",
                label=f"{arm}: source",
            )
            for arm in arms
        ]
        + [
            Line2D(
                [0],
                [0],
                color=colours[arm],
                marker=markers[arm],
                markerfacecolor="white",
                linestyle=(0, (5, 2)),
                label=f"{arm}: repeated trials",
            )
            for arm in arms
        ],
        frameon=False,
        fontsize=5.8,
        loc="lower left",
        ncols=2,
    )
    retention_axis.grid(axis="y")
    retention_axis.text(
        -0.08, 1.04, "c", transform=retention_axis.transAxes, fontweight="bold"
    )

    linkage_axis = axes[1, 1]
    linkage = linkage.sort_values("linkage_retention")
    linkage_value = linkage["correlation_mae_mean"].to_numpy(dtype=float)
    linkage_retention_percent = 100.0 * linkage["linkage_retention"].to_numpy(
        dtype=float
    )
    linkage_axis.axhspan(
        float(linkage["source_bootstrap_low"].iloc[0]),
        float(linkage["source_bootstrap_high"].iloc[0]),
        color=_LIGHT_GREY,
        alpha=0.7,
        label="Source resampling range",
    )
    linkage_axis.errorbar(
        linkage_retention_percent,
        linkage_value,
        yerr=np.vstack(
            (
                linkage_value - linkage["correlation_mae_ci_low"].to_numpy(dtype=float),
                linkage["correlation_mae_ci_high"].to_numpy(dtype=float)
                - linkage_value,
            )
        ),
        color=_BLUE,
        marker="o",
        markerfacecolor="white",
        linestyle="-",
        linewidth=1.2,
        capsize=2,
        label="Simulated trials",
    )
    linkage_axis.set(
        title="Joint correlation recovery",
        xlabel="Participant linkage retained (%)",
        ylabel="Mean absolute correlation error\n(correlation units)",
        xticks=(0, 25, 50, 75, 100),
    )
    linkage_axis.legend(frameon=False, fontsize=6, loc="upper right")
    linkage_axis.grid(axis="y")
    linkage_axis.text(
        -0.08, 1.04, "d", transform=linkage_axis.transAxes, fontweight="bold"
    )

    return _finish(figure, bottom=0.1, left=0.12, hspace=0.44, wspace=0.48)


def _joint_structure_figure(root: Path) -> Figure:
    data = pd.read_csv(root / "joint_structure_methods.csv")
    trials = pd.read_csv(root / "joint_structure_trials.csv")
    _require_columns(
        data, {"method", "metric", "estimate", "interval_low", "interval_high"}
    )
    _require_columns(trials, {"trial_id", "method", "metric", "estimate"})
    metrics = tuple(data["metric"].drop_duplicates())
    figure, axes = plt.subplots(1, len(metrics), figsize=(7.2, 3.7))
    colours = {"Whole-subject": _BLUE, "Column-wise": _VERMILLION}
    markers = {"Whole-subject": "o", "Column-wise": "s"}
    labels = {
        "Whole-subject": "Linked subjects",
        "Column-wise": "Independent columns",
    }
    for panel, (axis, metric) in enumerate(
        zip(np.atleast_1d(axes), metrics, strict=True)
    ):
        trial_rows = trials.loc[trials["metric"].eq(metric)]
        paired = trial_rows.pivot(index="trial_id", columns="method", values="estimate")
        expected_methods = ("Whole-subject", "Column-wise")
        if tuple(sorted(paired.columns)) != tuple(sorted(expected_methods)):
            raise ValueError(f"{metric} lacks both linkage methods")
        for _, row in paired.iterrows():
            axis.plot(
                (0, 1),
                (float(row["Whole-subject"]), float(row["Column-wise"])),
                color="#B5B5B5",
                linewidth=0.7,
                alpha=0.75,
                zorder=1,
            )
        summary = data.loc[data["metric"].eq(metric)].set_index("method")
        medians = tuple(
            float(cast(float, summary.loc[method, "estimate"]))
            for method in expected_methods
        )
        axis.plot((0, 1), medians, color=_CHARCOAL, linewidth=1.5, zorder=2)
        for x, method in enumerate(expected_methods):
            axis.scatter(
                x,
                medians[x],
                color=colours[method],
                marker=markers[method],
                facecolor="white" if method == "Whole-subject" else colours[method],
                s=32,
                linewidth=1.2,
                zorder=3,
            )
        linked_lower = int((paired["Whole-subject"] < paired["Column-wise"]).sum())
        takeaway = (
            "Marginals unchanged"
            if metric == "Marginal distance (source SD)"
            else f"Linked lower in {linked_lower}/{len(paired)} trials"
        )
        axis.text(
            0.5,
            0.98,
            takeaway,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )
        axis.set_xticks(
            (0, 1),
            tuple(labels[method].replace(" ", "\n") for method in expected_methods),
        )
        axis.set_ylabel(str(metric))
        axis.text(
            -0.08,
            1.04,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
        axis.grid(axis="y")
    figure.legend(
        handles=[
            Line2D([0], [0], color="#B5B5B5", linewidth=0.8, label="Independent trial"),
            Line2D(
                [0],
                [0],
                color=_BLUE,
                marker="o",
                markerfacecolor="white",
                linestyle="none",
                label="Linked-subject median",
            ),
            Line2D(
                [0],
                [0],
                color=_VERMILLION,
                marker="s",
                markerfacecolor=_VERMILLION,
                linestyle="none",
                label="Independent-column median",
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    return _finish(figure, left=0.11, bottom=0.23, wspace=0.4)


def _parameter_recovery_figure(root: Path) -> Figure:
    data = pd.read_csv(root / "parameter_recovery.csv")
    _require_columns(
        data,
        {
            "outcome",
            "dose_multiplier",
            "truth",
            "estimate",
            "interval_low",
            "interval_high",
            "bias",
            "bias_interval_low",
            "bias_interval_high",
        },
    )
    outcomes = tuple(data["outcome"].drop_duplicates())
    figure, axes = plt.subplots(2, len(outcomes), figsize=(7.2, 5.4), sharex="col")
    for panel, outcome in enumerate(outcomes):
        estimate_axis = axes[0, panel]
        bias_axis = axes[1, panel]
        rows = data.loc[data["outcome"].eq(outcome)].sort_values("dose_multiplier")
        x = rows["dose_multiplier"].to_numpy()
        estimate = rows["estimate"].to_numpy()
        interval = np.vstack(
            (
                estimate - rows["interval_low"].to_numpy(),
                rows["interval_high"].to_numpy() - estimate,
            )
        )
        estimate_axis.plot(
            x, rows["truth"], color=_CHARCOAL, linestyle=(0, (5, 2)), linewidth=1.6
        )
        estimate_axis.errorbar(
            x,
            estimate,
            yerr=interval,
            color=_BLUE,
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            capsize=3,
            linewidth=1.4,
        )
        estimate_axis.axhline(0, color="#999999", linewidth=0.7)
        estimate_axis.set_title(str(outcome), fontsize=9)
        estimate_axis.text(
            -0.08,
            1.04,
            chr(ord("a") + panel),
            transform=estimate_axis.transAxes,
            fontweight="bold",
        )
        estimate_axis.grid(axis="y")

        bias = rows["bias"].to_numpy()
        bias_axis.errorbar(
            x,
            bias,
            yerr=np.vstack(
                (
                    bias - rows["bias_interval_low"].to_numpy(),
                    rows["bias_interval_high"].to_numpy() - bias,
                )
            ),
            color=_BLUE,
            marker="o",
            markerfacecolor="white",
            linestyle="-",
            capsize=3,
            linewidth=1.4,
        )
        bias_axis.axhline(0, color=_CHARCOAL, linestyle=(0, (2, 2)), linewidth=1)
        bias_axis.set_xticks((0, 1, 2, 4))
        bias_axis.set_xlabel("Known effect multiplier")
        bias_axis.text(
            -0.08,
            1.04,
            chr(ord("c") + panel),
            transform=bias_axis.transAxes,
            fontweight="bold",
        )
        bias_axis.grid(axis="y")
    axes[0, 0].set_ylabel("Recovered log effect")
    axes[1, 0].set_ylabel("Bias in recovered effect")
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=_CHARCOAL,
                linestyle=(0, (5, 2)),
                label="Known generating effect",
            ),
            Line2D(
                [0],
                [0],
                color=_BLUE,
                marker="o",
                markerfacecolor="white",
                linestyle="none",
                label="Recovered mean",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    return _finish(figure, bottom=0.12, hspace=0.34)


def _mechanism_response_figure(root: Path) -> Figure:
    data = pd.read_csv(root / "mechanism_response.csv")
    _require_columns(
        data,
        {
            "panel",
            "series",
            "setting",
            "dose",
            "estimate",
            "interval_low",
            "interval_high",
            "reference",
            "unit",
            "independent_units",
        },
    )
    expected_panels = {
        "treatment_heterogeneity",
        "competing_events",
        "confounding",
        "dropout",
        "recurrent_events",
        "cross_domain_linkage",
    }
    if set(data["panel"]) != expected_panels:
        raise ValueError(
            "Mechanism-response source data do not contain the six required panels"
        )

    figure, axes = plt.subplots(3, 2, figsize=(7.2, 8.6))
    panels = axes.ravel()
    _forest_panel(
        panels[0],
        _replace_setting_labels(
            data.loc[data["panel"].eq("treatment_heterogeneity")],
            {
                "Binary, N": "Binary\nsource size",
                "Binary, 4N": "Binary\n4x size",
                "Continuous, N": "Continuous\nsource size",
                "Continuous, 4N": "Continuous\n4x size",
            },
        ),
        title="Treatment-effect heterogeneity",
        x_label="Recovered / generating change",
        reference=1.0,
    )
    _forest_panel(
        panels[1],
        _replace_setting_labels(
            data.loc[data["panel"].eq("competing_events")],
            {
                "Primary event, N": "Primary event\nsource size",
                "Any event, N": "Any event\nsource size",
                "Primary event, 4N": "Primary event\n4x size",
                "Any event, 4N": "Any event\n4x size",
            },
        ),
        title="Competing-event consequences",
        x_label="Probability change per coefficient unit",
        reference=0.0,
    )
    _forest_panel(
        panels[2],
        _replace_setting_labels(
            data.loc[data["panel"].eq("confounding")],
            {
                "Unadjusted, N": "Unadjusted\nsource size",
                "Adjusted, N": "Adjusted\nsource size",
                "Unadjusted, 4N": "Unadjusted\n4x size",
                "Adjusted, 4N": "Adjusted\n4x size",
            },
        ),
        title="Confounding and adjustment",
        x_label="Bias change per assignment-strength unit",
        reference=0.0,
    )

    dropout = data.loc[data["panel"].eq("dropout")]
    dropout_styles = {
        "Skin barrier": (_CHARCOAL, "^", ":"),
        "PENG 0.375": (_BLUE, "o", "-"),
        "PENG 0.50": (_VERMILLION, "s", "--"),
    }
    dropout_labels = {
        "Skin barrier": "Skin-barrier trial",
        "PENG 0.375": "PENG 0.375%",
        "PENG 0.50": "PENG 0.50%",
    }
    for series, (colour, marker, linestyle) in dropout_styles.items():
        rows = dropout.loc[dropout["series"].eq(series)].sort_values("dose")
        panels[3].errorbar(
            rows["dose"],
            rows["estimate"],
            yerr=np.vstack(
                (
                    rows["estimate"] - rows["interval_low"],
                    rows["interval_high"] - rows["estimate"],
                )
            ),
            color=colour,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.0,
            capsize=2,
            label=dropout_labels[series],
        )
    panels[3].axhline(0.0, color=_CHARCOAL, linewidth=0.8)
    panels[3].set(
        title="Dropout weighting trade-off",
        xlabel="Prior-outcome coefficient for dropout",
        ylabel="Absolute-error reduction",
    )
    panels[3].legend(frameon=False, fontsize=7)
    panels[3].grid(axis="y")

    recurrent = data.loc[data["panel"].eq("recurrent_events")].sort_values("dose")
    panels[4].errorbar(
        recurrent["dose"],
        recurrent["estimate"],
        yerr=np.vstack(
            (
                recurrent["estimate"] - recurrent["interval_low"],
                recurrent["interval_high"] - recurrent["estimate"],
            )
        ),
        color=_BLUE,
        marker="o",
        linestyle="none",
        capsize=2,
    )
    limit = float(max(recurrent["dose"].max(), recurrent["interval_high"].max()) * 1.04)
    panels[4].plot(
        [0.0, limit], [0.0, limit], color=_CHARCOAL, linestyle="--", linewidth=0.9
    )
    panels[4].set(
        title="Recurrent-event heterogeneity",
        xlabel="Known generating frailty variance",
        ylabel="Recovered frailty variance",
        xlim=(-0.04, limit),
        ylim=(-0.04, limit),
    )
    panels[4].grid()

    _forest_panel(
        panels[5],
        data.loc[data["panel"].eq("cross_domain_linkage")],
        title="Cross-domain linkage",
        x_label="Change after full linkage disruption",
        reference=0.0,
    )

    for panel, axis in enumerate(panels):
        axis.text(
            -0.16,
            1.05,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
    return _finish(figure, bottom=0.08, left=0.16, hspace=0.5, wspace=0.42)


def _replace_setting_labels(data: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Return figure rows with a complete, explicit setting-label mapping."""

    observed = set(data["setting"].astype(str))
    if observed != set(labels):
        raise ValueError(
            f"Mechanism-response settings differ from their display labels: {sorted(observed)!r}"
        )
    relabelled = data.copy()
    relabelled["setting"] = relabelled["setting"].map(labels)
    return relabelled


def _forest_panel(
    axis: Axes,
    rows: pd.DataFrame,
    *,
    title: str,
    x_label: str,
    reference: float,
) -> None:
    if rows.empty:
        raise ValueError(f"Mechanism-response panel has no rows: {title}")
    positions = np.arange(len(rows))
    axis.errorbar(
        rows["estimate"],
        positions,
        xerr=np.vstack(
            (
                rows["estimate"] - rows["interval_low"],
                rows["interval_high"] - rows["estimate"],
            )
        ),
        color=_BLUE,
        marker="o",
        linestyle="none",
        capsize=2,
    )
    axis.axvline(reference, color=_CHARCOAL, linestyle="--", linewidth=0.8)
    axis.set_yticks(positions, rows["setting"])
    axis.set(title=title, xlabel=x_label)
    axis.invert_yaxis()
    axis.grid(axis="x")


def _negative_control_figure(root: Path) -> Figure:
    data = pd.read_csv(root / "negative_control.csv")
    _require_columns(
        data,
        {
            "outcome",
            "intact_error",
            "intact_ci_low",
            "intact_ci_high",
            "broken_error",
            "broken_ci_low",
            "broken_ci_high",
            "difference",
            "difference_ci_low",
            "difference_ci_high",
        },
    )
    figure, axis = plt.subplots(figsize=(7.2, 3.7))
    y = np.arange(len(data), dtype=float)
    offset = 0.13
    for row_index, (_, row) in enumerate(data.iterrows()):
        axis.plot(
            [float(row["intact_error"]), float(row["broken_error"])],
            [y[row_index] - offset, y[row_index] + offset],
            color=_LIGHT_GREY,
            linewidth=1.2,
            zorder=1,
        )
    for label, prefix, colour, marker, displacement in (
        ("Intact linkage", "intact", _BLUE, "o", -offset),
        ("Broken linkage", "broken", _VERMILLION, "s", offset),
    ):
        estimates = data[f"{prefix}_error"].to_numpy(dtype=float)
        lows = data[f"{prefix}_ci_low"].to_numpy(dtype=float)
        highs = data[f"{prefix}_ci_high"].to_numpy(dtype=float)
        axis.errorbar(
            estimates,
            y + displacement,
            xerr=np.vstack((estimates - lows, highs - estimates)),
            color=colour,
            marker=marker,
            markerfacecolor="white" if prefix == "intact" else colour,
            markeredgecolor=colour,
            linestyle="none",
            linewidth=1.3,
            capsize=3,
            label=label,
            zorder=2,
        )
    maximum = float(data["broken_ci_high"].max())
    for row_index, (_, row) in enumerate(data.iterrows()):
        axis.text(
            maximum * 1.02,
            y[row_index],
            (
                f"Paired increase {float(row['difference']):.5f}\n"
                f"95% CI {float(row['difference_ci_low']):.5f} to "
                f"{float(row['difference_ci_high']):.5f}"
            ),
            va="center",
            fontsize=7,
        )
    axis.set_yticks(y, data["outcome"])
    axis.invert_yaxis()
    axis.set(
        xlabel="Mean absolute probability error",
        xlim=(0, maximum * 2.1),
    )
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=2,
    )
    axis.grid(axis="x")
    return _finish(figure, bottom=0.3, left=0.29)


def _require_columns(data: pd.DataFrame, required: set[str]) -> None:
    if missing := sorted(required - set(data.columns)):
        raise ValueError(f"Figure data are missing columns: {missing!r}")


def _style() -> dict[str, object]:
    return {
        "axes.edgecolor": _CHARCOAL,
        "axes.labelcolor": _CHARCOAL,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.6,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "trialagentbench-validation",
        "xtick.color": _CHARCOAL,
        "ytick.color": _CHARCOAL,
    }


def _finish(
    figure: Figure,
    *,
    bottom: float = 0.16,
    left: float = 0.1,
    top: float = 0.9,
    hspace: float = 0.35,
    wspace: float = 0.35,
) -> Figure:
    figure.subplots_adjust(
        left=left, right=0.98, top=top, bottom=bottom, wspace=wspace, hspace=hspace
    )
    return figure


def _save(figure: Figure, stem: Path) -> tuple[Path, Path]:
    svg = stem.with_suffix(".svg")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(
        svg,
        format="svg",
        metadata={"Creator": "trialagentbench-validation", "Date": None},
    )
    svg.write_text(
        "\n".join(
            line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    figure.savefig(
        pdf,
        format="pdf",
        metadata={
            "Creator": "trialagentbench-validation",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    return svg, pdf


def main() -> None:
    """Render all report figures from an installed or supplied result bundle."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render_validation_report_figures(
        validation_root=args.validation_root or installed_validation_root(),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


__all__ = ["render_validation_report_figures"]
