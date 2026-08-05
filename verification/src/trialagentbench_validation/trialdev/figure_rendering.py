"""Deterministic publication rendering for TrialDev verification results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from statsmodels.stats.proportion import proportion_confint

_WIDTH_IN = 183.0 / 25.4
_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_GREY = "#6B7280"
_LIGHT_GREY = "#D1D5DB"
_EXPORT_METADATA = {
    "Creator": "trialagentbench-validation",
    "Date": "2026-08-02",
}
_AXIS_LABELS = {
    "information": "Information",
    "confounding": "Residual confounding",
    "overlap": "Covariate overlap",
    "resources": "Resource budget",
    "stopping": "Stopping evidence",
    "reallocation": "Reserve reallocation",
    "efficacy": "Efficacy",
    "safety": "Safety",
    "operations": "Operational disruption",
    "asset_correlation": "Asset correlation",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "trialdev-v1",
        }
    )


def _save(fig: Figure, output_stem: Path) -> tuple[Path, Path]:
    output = Path(output_stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    svg = output.with_suffix(".svg")
    fig.savefig(
        pdf,
        metadata={
            "Creator": _EXPORT_METADATA["Creator"],
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(svg, metadata=_EXPORT_METADATA)
    plt.close(fig)
    return pdf, svg


def _clean_axis(axis: Axes) -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6, zorder=0)


def write_operating_effect_source_data_v1(
    *,
    summary_csv: Path,
    axes: tuple[str, ...],
    output_csv: Path,
) -> Path:
    """Export exact paired operating effects for one figure."""

    frame = pd.read_csv(summary_csv).set_index("axis").loc[list(axes)].reset_index()
    fields = (
        "experiment_id",
        "axis",
        "primary_metric",
        "world_count",
        "reference_mean",
        "reference_lower",
        "reference_upper",
        "intervention_mean",
        "intervention_lower",
        "intervention_upper",
        "paired_difference",
        "paired_bootstrap_lower",
        "paired_bootstrap_upper",
        "expected_direction",
    )
    missing = set(fields) - set(str(column) for column in frame.columns)
    if missing:
        raise ValueError(
            f"Operating summary lacks exact figure fields: {sorted(missing)!r}."
        )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, fields].to_csv(output, index=False, lineterminator="\n")
    return output


def write_observational_replay_source_data_v1(
    *, replay_root: Path, output_csv: Path
) -> Path:
    """Export exact independent-replay values used by the recoverability figure."""

    records: list[dict[str, object]] = []
    paths = tuple(sorted(Path(replay_root).glob("observational_replay_world_*.json")))
    if not paths:
        raise ValueError("Observational-replay figure requires replay reports.")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("status") != "pass":
            raise ValueError(f"Observational replay must pass before plotting: {path}.")
        scenario_id = str(payload["scenario_id"])
        tolerance = float(payload["absolute_tolerance"])
        methods = payload.get("methods")
        if not isinstance(methods, list) or not methods:
            raise ValueError(f"Observational replay lacks method results: {path}.")
        for method in methods:
            if not isinstance(method, dict):
                raise ValueError(f"Observational method replay is malformed: {path}.")
            route_id = str(method["method_route_id"])
            result_form = str(method["result_form"])
            candidate_results = method.get("candidate_results")
            if not isinstance(candidate_results, list):
                raise ValueError(
                    f"Observational replay candidate results are malformed: {path}."
                )
            for candidate in candidate_results:
                if not isinstance(candidate, dict):
                    raise ValueError(
                        f"Observational replay candidate is malformed: {path}."
                    )
                records.append(
                    {
                        "record_kind": "candidate",
                        "scenario_id": scenario_id,
                        "method_route_id": route_id,
                        "result_form": result_form,
                        "objective_id": str(candidate["objective_id"]),
                        "candidate_id": str(candidate["candidate_drug_id"]),
                        "metric_id": "utility",
                        "expected": float(candidate["expected_utility"]),
                        "replayed": float(candidate["replayed_utility"]),
                        "estimate": float(candidate["utility_absolute_error"]),
                        "plot_value": float(candidate["utility_absolute_error"]),
                        "tolerance": tolerance,
                        "match": bool(candidate["within_tolerance"]),
                    }
                )
            for metric_id, field in (
                ("utility point", "maximum_utility_absolute_error"),
                ("standard error", "maximum_standard_error_absolute_error"),
                ("interval endpoint", "maximum_interval_endpoint_absolute_error"),
            ):
                estimate = float(method[field])
                records.append(
                    {
                        "record_kind": "method_error",
                        "scenario_id": scenario_id,
                        "method_route_id": route_id,
                        "result_form": result_form,
                        "objective_id": "",
                        "candidate_id": "",
                        "metric_id": metric_id,
                        "expected": "",
                        "replayed": "",
                        "estimate": estimate,
                        "plot_value": max(estimate, tolerance * 1e-4),
                        "tolerance": tolerance,
                        "match": bool(method["status"] == "pass"),
                    }
                )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output, index=False, lineterminator="\n")
    return output


def render_identification_recoverability_v1(
    *, source_csv: Path, output_stem: Path
) -> tuple[Path, Path]:
    """Render independent numeric replay and qualified non-estimability."""

    _style()
    frame = pd.read_csv(source_csv)
    required = {
        "record_kind",
        "scenario_id",
        "method_route_id",
        "result_form",
        "metric_id",
        "expected",
        "replayed",
        "estimate",
        "plot_value",
        "tolerance",
        "match",
    }
    if not required <= set(frame):
        raise ValueError("Recoverability source table lacks required exact values.")
    candidates = frame.loc[frame["record_kind"].eq("candidate")].copy()
    errors = frame.loc[frame["record_kind"].eq("method_error")].copy()
    if candidates.empty or errors.empty:
        raise ValueError(
            "Recoverability figure requires numeric and method-level replay records."
        )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(_WIDTH_IN, 2.45),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.08, 1.2, 0.78)},
    )

    method_ids = tuple(sorted(frame["method_route_id"].unique()))
    method_labels = {
        method_id: f"Method {index + 1}" for index, method_id in enumerate(method_ids)
    }
    method_styles = dict(zip(method_ids, ((_BLUE, "o"), (_ORANGE, "s")), strict=True))
    for method_id in method_ids:
        rows = candidates.loc[candidates["method_route_id"].eq(method_id)]
        colour, marker = method_styles[method_id]
        axes[0].scatter(
            rows["expected"].to_numpy(float),
            rows["replayed"].to_numpy(float),
            s=7,
            marker=marker,
            facecolors="none",
            edgecolors=colour,
            linewidths=0.45,
            alpha=0.75,
            label=method_labels[method_id],
        )
    limits = (
        float(min(candidates["expected"].min(), candidates["replayed"].min())),
        float(max(candidates["expected"].max(), candidates["replayed"].max())),
    )
    padding = max(0.01, (limits[1] - limits[0]) * 0.04)
    axes[0].plot(
        (limits[0] - padding, limits[1] + padding),
        (limits[0] - padding, limits[1] + padding),
        color="#111827",
        linestyle="--",
        linewidth=0.7,
    )
    axes[0].set_xlim(limits[0] - padding, limits[1] + padding)
    axes[0].set_ylim(limits[0] - padding, limits[1] + padding)
    axes[0].set_xlabel("Released utility estimate")
    axes[0].set_ylabel("Independent replay estimate")
    axes[0].legend(frameon=False, loc="upper left")

    metric_order = ("utility point", "standard error", "interval endpoint")
    metric_styles = dict(
        zip(metric_order, ((_BLUE, "o"), (_GREEN, "s"), (_ORANGE, "^")), strict=True)
    )
    y_by_scenario = {
        scenario: index
        for index, scenario in enumerate(sorted(frame["scenario_id"].unique()))
    }
    for metric_id in metric_order:
        rows = errors.loc[errors["metric_id"].eq(metric_id)]
        colour, marker = metric_styles[metric_id]
        offsets = np.where(rows["method_route_id"].eq(method_ids[0]), -0.10, 0.10)
        axes[1].scatter(
            rows["plot_value"].to_numpy(float),
            np.asarray(
                [y_by_scenario[value] for value in rows["scenario_id"]], dtype=float
            )
            + offsets,
            s=8,
            marker=marker,
            facecolors="none",
            edgecolors=colour,
            linewidths=0.5,
            label=metric_id.capitalize(),
        )
    tolerance_values = errors["tolerance"].unique()
    if len(tolerance_values) != 1:
        raise ValueError(
            "Recoverability figure requires one declared numerical tolerance."
        )
    axes[1].axvline(
        float(tolerance_values[0]), color="#111827", linestyle="--", linewidth=0.7
    )
    axes[1].set_xscale("log")
    axes[1].set_xticks(tuple(10.0**exponent for exponent in range(-8, -3)))
    axes[1].xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"1e{int(np.log10(value))}")
    )
    axes[1].tick_params(axis="x", which="minor", bottom=False)
    axes[1].set_yticks(())
    axes[1].set_xlabel("Maximum absolute replay error")
    axes[1].legend(frameon=False, loc="lower right")

    method_census = (
        errors.loc[errors["metric_id"].eq("utility point")]
        .groupby(["method_route_id", "result_form"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(method_ids, fill_value=0)
    )
    estimable = method_census.get(
        "point_estimates", pd.Series(0, index=method_ids)
    ).to_numpy(float)
    nonestimable = method_census.get(
        "qualified_non_nomination", pd.Series(0, index=method_ids)
    ).to_numpy(float)
    positions = np.arange(len(method_ids))
    axes[2].bar(positions, estimable, color=_BLUE, width=0.58, label="Estimable")
    axes[2].bar(
        positions,
        nonestimable,
        bottom=estimable,
        color=_ORANGE,
        width=0.58,
        label="Non-estimable",
    )
    axes[2].set_xticks(
        positions,
        [method_labels[value] for value in method_ids],
        rotation=25,
        ha="right",
    )
    axes[2].set_ylabel("Released worlds")
    axes[2].legend(frameon=False, loc="upper right")

    for label, axis in zip(("a", "b", "c"), axes, strict=True):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)
        axis.text(
            -0.18, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=8
        )
    return _save(fig, output_stem)


def render_paired_effect_forest_v1(
    *,
    summary_csv: Path,
    axes: tuple[str, ...],
    output_stem: Path,
    panel_label: str,
) -> tuple[Path, Path]:
    """Render paired intervention effects and bootstrap intervals."""

    _style()
    frame = pd.read_csv(summary_csv).set_index("axis").loc[list(axes)].reset_index()
    values = frame["paired_difference"].to_numpy(float)
    lower = frame["paired_bootstrap_lower"].to_numpy(float)
    upper = frame["paired_bootstrap_upper"].to_numpy(float)
    if not np.isfinite(np.concatenate((values, lower, upper))).all():
        raise ValueError(
            "Paired-effect figure requires finite estimates and intervals."
        )
    positions = np.arange(len(frame))[::-1]
    fig, plot_axes = plt.subplots(
        1,
        2,
        figsize=(_WIDTH_IN, 1.45 + 0.34 * len(frame)),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.05, 1.0)},
    )
    reference = frame["reference_mean"].to_numpy(float)
    intervention = frame["intervention_mean"].to_numpy(float)
    for position, start, end in zip(positions, reference, intervention, strict=True):
        plot_axes[0].plot(
            (start, end),
            (position, position),
            color=_LIGHT_GREY,
            linewidth=1.1,
            zorder=1,
        )
    plot_axes[0].errorbar(
        reference,
        positions + 0.08,
        xerr=np.vstack(
            (
                reference - frame["reference_lower"].to_numpy(float),
                frame["reference_upper"].to_numpy(float) - reference,
            )
        ),
        fmt="o",
        color=_GREY,
        markerfacecolor="white",
        markersize=3.6,
        capsize=1.5,
        label="Reference",
        zorder=3,
    )
    plot_axes[0].errorbar(
        intervention,
        positions - 0.08,
        xerr=np.vstack(
            (
                intervention - frame["intervention_lower"].to_numpy(float),
                frame["intervention_upper"].to_numpy(float) - intervention,
            )
        ),
        fmt="s",
        color=_ORANGE,
        markerfacecolor=_ORANGE,
        markersize=3.4,
        capsize=1.5,
        label="Intervention",
        zorder=3,
    )
    plot_axes[0].set_yticks(positions, [_AXIS_LABELS[value] for value in frame["axis"]])
    plot_axes[0].set_xlabel("Operating-characteristic estimate")
    plot_axes[0].legend(
        frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12)
    )
    _clean_axis(plot_axes[0])

    plot_axes[1].axvline(0.0, color="#111827", linewidth=0.8, linestyle="--", zorder=1)
    plot_axes[1].errorbar(
        values,
        positions,
        xerr=np.vstack((values - lower, upper - values)),
        fmt="o",
        color=_BLUE,
        ecolor=_BLUE,
        markerfacecolor="white",
        markeredgewidth=1.0,
        markersize=4.2,
        capsize=2.0,
        zorder=3,
    )
    plot_axes[1].set_yticks(positions, ())
    plot_axes[1].set_xlabel("Paired difference (intervention − reference)")
    padding = max(0.05, 0.08 * (float(upper.max()) - float(lower.min())))
    plot_axes[1].set_xlim(
        float(min(0.0, lower.min())) - padding, float(max(0.0, upper.max())) + padding
    )
    _clean_axis(plot_axes[1])
    for label, axis in zip((panel_label, "b"), plot_axes, strict=True):
        axis.text(
            -0.12, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=8
        )
    return _save(fig, output_stem)


def render_operating_characteristics_v1(
    *,
    summary_csv: Path,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render paired arm rates and uncertainty for every prespecified experiment."""

    _style()
    frame = pd.read_csv(summary_csv).sort_values("axis")
    required = {
        "axis",
        "reference_mean",
        "reference_lower",
        "reference_upper",
        "intervention_mean",
        "intervention_lower",
        "intervention_upper",
    }
    if not required <= set(frame):
        raise ValueError("Operating-characteristic table lacks required arm summaries.")
    positions = np.arange(len(frame))[::-1]
    fig, axis = plt.subplots(
        figsize=(_WIDTH_IN, 2.4 + 0.22 * len(frame)), constrained_layout=True
    )
    reference = frame["reference_mean"].to_numpy(float)
    intervention = frame["intervention_mean"].to_numpy(float)
    for y, start, end in zip(positions, reference, intervention, strict=True):
        axis.plot((start, end), (y, y), color=_LIGHT_GREY, linewidth=1.2, zorder=1)
    axis.errorbar(
        reference,
        positions + 0.08,
        xerr=np.vstack(
            (
                reference - frame["reference_lower"].to_numpy(float),
                frame["reference_upper"].to_numpy(float) - reference,
            )
        ),
        fmt="o",
        color=_GREY,
        markerfacecolor="white",
        markersize=3.6,
        capsize=1.5,
        label="Reference",
        zorder=3,
    )
    axis.errorbar(
        intervention,
        positions - 0.08,
        xerr=np.vstack(
            (
                intervention - frame["intervention_lower"].to_numpy(float),
                frame["intervention_upper"].to_numpy(float) - intervention,
            )
        ),
        fmt="s",
        color=_ORANGE,
        markerfacecolor=_ORANGE,
        markersize=3.4,
        capsize=1.5,
        label="Intervention",
        zorder=3,
    )
    axis.set_yticks(positions, [_AXIS_LABELS[value] for value in frame["axis"]])
    axis.set_xlabel("Prespecified operating-characteristic estimate")
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    _clean_axis(axis)
    axis.text(
        -0.055, 1.04, "a", transform=axis.transAxes, fontweight="bold", fontsize=8
    )
    return _save(fig, output_stem)


def write_grader_control_source_data_v1(
    *, controls_csv: Path, output_csv: Path
) -> Path:
    """Export the exact positive and single-fault grader-control census."""

    frame = pd.read_csv(controls_csv)
    required = {"control_id", "control_kind", "detected"}
    if not required <= set(frame):
        raise ValueError("Control table lacks the declared control outcomes.")
    if "responsibility" not in frame:
        responsibility_by_control = {
            "accepted_reference": "all graded responsibilities",
            "numeric_evidence_error": "numeric evidence",
            "efficacy_evidence_error": "efficacy evidence",
            "pairwise_uncertainty_error": "pairwise uncertainty",
            "evidence_reference_error": "analysis evidence binding",
            "provenance_error": "supporting evidence binding",
            "unsupported_action": "action support",
            "scheduled_design_error": "study design",
            "stale_state": "state custody",
        }
        unknown = set(frame["control_id"].astype(str)) - set(responsibility_by_control)
        if unknown:
            raise ValueError(
                f"Control table contains unknown controls: {sorted(unknown)!r}."
            )
        frame = frame.assign(
            responsibility=frame["control_id"].map(responsibility_by_control)
        )
    frame = frame.assign(detected=frame["detected"].astype(bool))
    grouped = cast(
        pd.DataFrame,
        frame.groupby(
            ["control_id", "responsibility", "control_kind"], as_index=False, sort=True
        )["detected"]
        .agg(["sum", "count"])
        .reset_index(),
    )
    grouped = grouped.rename(
        columns={"sum": "expected_behavior_count", "count": "evaluated_count"}
    )
    grouped["expected_behavior_rate"] = (
        grouped["expected_behavior_count"] / grouped["evaluated_count"]
    )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output, index=False, lineterminator="\n")
    return output


def render_failure_decomposition_v1(
    *, source_csv: Path, output_stem: Path
) -> tuple[Path, Path]:
    """Render exact coherent-reference acceptance and isolated-fault detection."""

    _style()
    frame = pd.read_csv(source_csv)
    required = {
        "control_id",
        "responsibility",
        "control_kind",
        "expected_behavior_count",
        "evaluated_count",
        "expected_behavior_rate",
    }
    if not required <= set(frame):
        raise ValueError(
            "Grader-control source table lacks required exact census fields."
        )
    controls = tuple(frame["control_id"])
    positions = np.arange(len(frame))[::-1]
    rates = frame["expected_behavior_rate"].to_numpy(float)
    colours = np.where(frame["control_kind"].eq("positive"), _BLUE, _ORANGE)
    fig, axis = plt.subplots(
        figsize=(_WIDTH_IN, 1.45 + 0.30 * len(frame)), constrained_layout=True
    )
    for position, rate, colour in zip(positions, rates, colours, strict=True):
        axis.plot(
            (0.0, rate),
            (position, position),
            color=_LIGHT_GREY,
            linewidth=1.0,
            zorder=1,
        )
        axis.scatter(
            rate,
            position,
            s=22,
            marker="o",
            facecolor="white",
            edgecolor=colour,
            linewidth=1.0,
            zorder=3,
        )
    expected_counts = frame["expected_behavior_count"].to_numpy(int)
    evaluated_counts = frame["evaluated_count"].to_numpy(int)
    for position, expected, evaluated in zip(
        positions, expected_counts, evaluated_counts, strict=True
    ):
        axis.text(
            1.015,
            position,
            f"{expected}/{evaluated}",
            fontsize=6.3,
            va="center",
            ha="left",
        )
    labels = [
        f"{'complete reference' if control == 'accepted_reference' else control.replace('_', ' ')}\n"
        f"{responsibility.replace('_', ' ')}"
        for control, responsibility in zip(
            controls, frame["responsibility"], strict=True
        )
    ]
    axis.set_yticks(positions, labels)
    axis.set_xlim(0.0, 1.12)
    axis.set_xticks((0.0, 0.5, 1.0))
    axis.set_xlabel("Expected outcome agreement across applicable views")
    axis.legend(
        handles=(
            Patch(facecolor=_BLUE, label="Coherent reference"),
            Patch(facecolor=_ORANGE, label="Isolated fault"),
        ),
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )
    _clean_axis(axis)
    axis.text(
        -0.055, 1.04, "a", transform=axis.transAxes, fontweight="bold", fontsize=8
    )
    return _save(fig, output_stem)


def write_clinical_realism_source_data_v1(
    *, release_audit_json: Path, output_csv: Path
) -> Path:
    """Export the released randomized and observational quantities used in the realism figure."""

    payload = json.loads(Path(release_audit_json).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "pass":
        raise ValueError(
            "Clinical-realism source data require a passing release audit."
        )
    episodes = pd.DataFrame(payload.get("episode_realism"))
    observational = payload.get("observational_realism")
    if episodes.empty or not isinstance(observational, list) or not observational:
        raise ValueError(
            "Clinical-realism source data require randomized and observational evidence."
        )
    required_episode_fields = {
        "episode_id",
        "phase_id",
        "row_count",
        "follow_up_days",
        "efficacy_event_rate_treated",
        "efficacy_event_rate_control",
        "serious_ae_rate_treated",
        "discontinuation_rate_treated",
        "loss_to_follow_up_rate",
    }
    if not required_episode_fields <= set(episodes):
        raise ValueError("Clinical-realism audit lacks required episode quantities.")
    records: list[dict[str, object]] = []
    for row in episodes.to_dict(orient="records"):
        episode_id = str(row["episode_id"])
        phase_id = str(row["phase_id"])
        records.extend(
            (
                {
                    "panel": "episode_scale",
                    "unit_id": episode_id,
                    "phase_id": phase_id,
                    "series_id": "participants",
                    "estimate": int(row["row_count"]),
                    "reference": "",
                },
                {
                    "panel": "follow_up",
                    "unit_id": episode_id,
                    "phase_id": phase_id,
                    "series_id": "follow-up days",
                    "estimate": int(row["follow_up_days"]),
                    "reference": "",
                },
            )
        )
        treated = row["efficacy_event_rate_treated"]
        control = row["efficacy_event_rate_control"]
        if (
            treated is not None
            and control is not None
            and not pd.isna(treated)
            and not pd.isna(control)
        ):
            records.append(
                {
                    "panel": "efficacy_contrast",
                    "unit_id": episode_id,
                    "phase_id": phase_id,
                    "series_id": "treated minus control",
                    "estimate": float(treated) - float(control),
                    "reference": 0.0,
                }
            )
        for series_id, field in (
            ("Serious adverse event", "serious_ae_rate_treated"),
            ("Discontinuation", "discontinuation_rate_treated"),
            ("Loss to follow-up", "loss_to_follow_up_rate"),
        ):
            records.append(
                {
                    "panel": "event_rate",
                    "unit_id": episode_id,
                    "phase_id": phase_id,
                    "series_id": series_id,
                    "estimate": float(row[field]),
                    "reference": "",
                }
            )
    for world in observational:
        if not isinstance(world, dict) or not isinstance(
            world.get("treatment_counts"), dict
        ):
            raise ValueError("Observational realism record lacks treatment counts.")
        world_id = str(world.get("world_id", ""))
        counts = {
            str(treatment): int(count)
            for treatment, count in world["treatment_counts"].items()
        }
        if "control" not in counts or len(counts) < 2:
            raise ValueError(
                "Every observational world requires control and investigational arms."
            )
        records.extend(
            {
                "panel": "observational_support",
                "unit_id": f"{world_id}:{treatment}",
                "phase_id": "observational",
                "series_id": (
                    "Control" if treatment == "control" else "Investigational candidate"
                ),
                "estimate": count,
                "reference": "",
            }
            for treatment, count in counts.items()
        )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output, index=False, lineterminator="\n")
    return output


def render_clinical_realism_v1(
    *, source_csv: Path, output_stem: Path
) -> tuple[Path, Path]:
    """Render scale, follow-up, efficacy, retention, and observational support."""

    _style()
    frame = pd.read_csv(source_csv)
    required = {"panel", "unit_id", "phase_id", "series_id", "estimate", "reference"}
    if not required <= set(frame):
        raise ValueError("Clinical-realism source table lacks required exact values.")
    phases = ("phase1", "phase2", "phase3")
    phase_names = ("Phase 1", "Phase 2", "Phase 3")
    fig, axes = plt.subplots(2, 2, figsize=(_WIDTH_IN, 4.6), constrained_layout=True)
    flat_axes = tuple(axes.flat)

    scale = frame.loc[frame["panel"].eq("episode_scale")]
    follow_up = frame.loc[frame["panel"].eq("follow_up")]
    follow_up_days = tuple(
        follow_up.loc[follow_up["phase_id"].eq(phase), "estimate"]
        .drop_duplicates()
        .to_numpy(int)
        for phase in phases
    )
    if any(len(values) != 1 for values in follow_up_days):
        raise ValueError(
            "Clinical-realism figure requires one follow-up duration per phase."
        )
    labels = tuple(
        f"{phase_name}\n{int(days[0])} days"
        for phase_name, days in zip(phase_names, follow_up_days, strict=True)
    )
    episode_n = [
        scale.loc[scale["phase_id"].eq(phase), "estimate"].to_numpy(float)
        for phase in phases
    ]
    flat_axes[0].boxplot(
        episode_n,
        tick_labels=labels,
        widths=0.5,
        showfliers=False,
        medianprops={"color": _BLUE, "linewidth": 1.2},
        boxprops={"color": _GREY},
        whiskerprops={"color": _GREY},
        capprops={"color": _GREY},
    )
    for position, values in enumerate(episode_n, start=1):
        jitter = np.linspace(-0.13, 0.13, len(values))
        flat_axes[0].scatter(
            position + jitter,
            values,
            s=6,
            facecolors="none",
            edgecolors=_BLUE,
            linewidths=0.5,
        )
    flat_axes[0].set_ylabel("Participants per randomized episode")
    flat_axes[0].tick_params(axis="x", rotation=25)

    efficacy = frame.loc[frame["panel"].eq("efficacy_contrast")]
    efficacy_phases = ("phase2", "phase3")
    efficacy_values = [
        efficacy.loc[efficacy["phase_id"].eq(phase), "estimate"].to_numpy(float)
        for phase in efficacy_phases
    ]
    flat_axes[1].axhline(0.0, color="#111827", linewidth=0.7, linestyle="--")
    flat_axes[1].boxplot(
        efficacy_values,
        tick_labels=("Phase 2", "Phase 3"),
        widths=0.45,
        showfliers=False,
        medianprops={"color": _GREEN, "linewidth": 1.2},
        boxprops={"color": _GREY},
        whiskerprops={"color": _GREY},
        capprops={"color": _GREY},
    )
    for position, values in enumerate(efficacy_values, start=1):
        jitter = np.linspace(-0.12, 0.12, len(values))
        flat_axes[1].scatter(
            position + jitter,
            values,
            s=6,
            facecolors="none",
            edgecolors=_GREEN,
            linewidths=0.5,
        )
    flat_axes[1].set_ylabel("Treated − control efficacy-event rate")

    events = frame.loc[frame["panel"].eq("event_rate") & frame["phase_id"].eq("phase3")]
    event_order = ("Serious adverse event", "Discontinuation", "Loss to follow-up")
    event_values = [
        events.loc[events["series_id"].eq(metric), "estimate"].to_numpy(float)
        for metric in event_order
    ]
    flat_axes[2].boxplot(
        event_values,
        tick_labels=("Serious adverse\nevent", "Discontinuation", "Loss to\nfollow-up"),
        widths=0.45,
        showfliers=False,
        medianprops={"color": _ORANGE, "linewidth": 1.2},
        boxprops={"color": _GREY},
        whiskerprops={"color": _GREY},
        capprops={"color": _GREY},
    )
    flat_axes[2].set_ylabel("Phase 3 participant proportion")

    support = frame.loc[frame["panel"].eq("observational_support")]
    support_order = ("Control", "Investigational candidate")
    count_values = [
        support.loc[support["series_id"].eq(series_id), "estimate"].to_numpy(float)
        for series_id in support_order
    ]
    flat_axes[3].boxplot(
        count_values,
        tick_labels=["Control", "Investigational\ncandidate"],
        widths=0.5,
        showfliers=False,
        medianprops={"color": _GREEN, "linewidth": 1.2},
        boxprops={"color": _GREY},
        whiskerprops={"color": _GREY},
        capprops={"color": _GREY},
    )
    flat_axes[3].set_ylabel("Participants per observational arm")

    for label, axis in zip(("a", "b", "c", "d"), flat_axes, strict=True):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)
        axis.text(
            -0.18, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=8
        )
    return _save(fig, output_stem)


def write_decision_difficulty_source_data_v1(
    *,
    decision_boundary_json: Path,
    portfolio_difficulty_json: Path,
    output_csv: Path,
) -> Path:
    """Export the exact tidy values used by the TrialDev difficulty figure."""

    boundary = json.loads(Path(decision_boundary_json).read_text(encoding="utf-8"))
    difficulty = json.loads(Path(portfolio_difficulty_json).read_text(encoding="utf-8"))
    if not isinstance(boundary, dict) or boundary.get("status") != "pass":
        raise ValueError(
            "Decision-difficulty source data require a passing boundary report."
        )
    if not isinstance(difficulty, dict) or difficulty.get("status") != "pass":
        raise ValueError(
            "Decision-difficulty source data require a passing shortcut report."
        )
    cells = boundary.get("cells")
    strategies = difficulty.get("strategies")
    if (
        not isinstance(cells, list)
        or not cells
        or not isinstance(strategies, list)
        or not strategies
    ):
        raise ValueError(
            "Decision-difficulty reports require non-empty cells and strategies."
        )
    records: list[dict[str, object]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Decision-boundary cells must be objects.")
        denominator = int(cell["world_count"])
        classification_rates = (
            float(cell["clear_pass_rate"]),
            float(cell["clear_fail_rate"]),
            float(cell["indeterminate_rate"]),
        )
        if denominator < 1 or abs(sum(classification_rates) - 1.0) > 1e-12:
            raise ValueError(
                "Decision-boundary classifications must partition every repeated world."
            )
        decisive_rate = classification_rates[0] + classification_rates[1]
        successes = int(round(decisive_rate * denominator))
        lower, upper = proportion_confint(successes, denominator, method="wilson")
        records.append(
            {
                "panel": str(cell["axis"]),
                "series_id": f"N = {int(cell['information_size'])}",
                "strategy_class": "",
                "label": "",
                "x": float(cell["mechanism_value"]),
                "estimate": decisive_rate,
                "lower": float(lower),
                "upper": float(upper),
                "denominator": denominator,
                "threshold": float(boundary[f"{cell['axis']}_threshold"]),
                "support_ceiling": "",
                "uncertainty_role": "Wilson Monte Carlo interval",
            }
        )
    ceilings = {
        "action_only": float(difficulty["maximum_action_only_shortcut_support_rate"]),
        "point_estimate_only": float(
            difficulty["maximum_point_estimate_shortcut_support_rate"]
        ),
    }
    for strategy in strategies:
        if not isinstance(strategy, dict):
            raise ValueError("Shortcut strategies must be objects.")
        strategy_class = str(strategy["strategy_class"])
        records.append(
            {
                "panel": "strategy",
                "series_id": str(strategy["strategy_id"]),
                "strategy_class": strategy_class,
                "label": str(strategy["strategy_id"]),
                "x": "",
                "estimate": float(strategy["supported_view_rate"]),
                "lower": "",
                "upper": "",
                "denominator": int(strategy["evaluated_view_count"]),
                "threshold": "",
                "support_ceiling": ceilings.get(strategy_class, ""),
                "uncertainty_role": "exact released-view census",
            }
        )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output, index=False, lineterminator="\n")
    return output


def render_decision_difficulty_v1(
    *, source_csv: Path, output_stem: Path
) -> tuple[Path, Path]:
    """Render information-dependent decision resolution and shortcut performance."""

    _style()
    frame = pd.read_csv(source_csv)
    required = {
        "panel",
        "series_id",
        "strategy_class",
        "label",
        "x",
        "estimate",
        "lower",
        "upper",
        "denominator",
        "threshold",
        "support_ceiling",
        "uncertainty_role",
    }
    if not required <= set(frame):
        raise ValueError("Decision-difficulty source table lacks required fields.")
    boundary = frame.loc[frame["panel"].isin(("efficacy", "safety"))].copy()
    strategies = frame.loc[frame["panel"].eq("strategy")].copy()
    if boundary.empty or strategies.empty:
        raise ValueError(
            "Decision-difficulty figure requires boundary and strategy rows."
        )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(_WIDTH_IN, 2.45),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.15)},
    )
    colour_by_series = {"N = 80": _BLUE, "N = 240": _ORANGE, "N = 600": _GREEN}
    marker_by_series = {"N = 80": "o", "N = 240": "s", "N = 600": "^"}
    x_labels = {
        "efficacy": "Favourable-response probability",
        "safety": "Serious-event probability",
    }
    for axis, panel in zip(axes[:2], ("efficacy", "safety"), strict=True):
        panel_frame = boundary.loc[boundary["panel"].eq(panel)]
        for series_id in colour_by_series:
            rows = panel_frame.loc[panel_frame["series_id"].eq(series_id)].sort_values(
                "x"
            )
            if rows.empty:
                raise ValueError(
                    f"Decision-boundary panel {panel!r} lacks {series_id!r}."
                )
            x = rows["x"].to_numpy(float)
            estimate = rows["estimate"].to_numpy(float)
            lower = rows["lower"].to_numpy(float)
            upper = rows["upper"].to_numpy(float)
            axis.plot(
                x,
                estimate,
                color=colour_by_series[series_id],
                marker=marker_by_series[series_id],
                markersize=3.2,
                label=series_id,
                zorder=3,
            )
            axis.fill_between(
                x,
                lower,
                upper,
                color=colour_by_series[series_id],
                alpha=0.10,
                linewidth=0,
            )
        threshold_values = panel_frame["threshold"].dropna().unique()
        if len(threshold_values) != 1:
            raise ValueError(
                f"Decision-boundary panel {panel!r} requires one threshold."
            )
        axis.axvline(
            float(threshold_values[0]),
            color="#111827",
            linestyle="--",
            linewidth=0.7,
            zorder=1,
        )
        axis.set_xlabel(x_labels[panel])
        axis.set_ylim(-0.02, 1.03)
        axis.set_yticks((0.0, 0.5, 1.0))
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Decisive classification rate")
    axes[0].legend(frameon=False, loc="lower left")

    strategy_order = (
        "evidence_and_policy",
        "adjusted_point_pair",
        "always_withhold",
        "raw_observed_pair",
        "alphabetical_pair",
    )
    labels = {
        "evidence_and_policy": "Complete analysis",
        "adjusted_point_pair": "Adjusted point ranks",
        "always_withhold": "Always withhold",
        "raw_observed_pair": "Unadjusted point ranks",
        "alphabetical_pair": "Alphabetical pair",
    }
    class_styles = {
        "complete_analysis": (_GREEN, "D"),
        "point_estimate_only": (_ORANGE, "s"),
        "action_only": (_GREY, "o"),
    }
    indexed = strategies.set_index("series_id")
    if set(indexed.index) != set(strategy_order):
        raise ValueError(
            "Decision-difficulty figure requires the complete strategy inventory."
        )
    positions = np.arange(len(strategy_order))[::-1]
    for position, strategy_id in zip(positions, strategy_order, strict=True):
        row = cast(pd.Series, indexed.loc[strategy_id])
        colour, marker = class_styles[str(row["strategy_class"])]
        axes[2].plot(
            (0.0, float(row["estimate"])),
            (position, position),
            color=_LIGHT_GREY,
            linewidth=0.8,
            zorder=1,
        )
        axes[2].scatter(
            float(row["estimate"]),
            position,
            color=colour,
            marker=marker,
            s=18,
            zorder=3,
        )
    action_ceiling = float(
        cast(
            float,
            strategies.loc[
                strategies["strategy_class"].eq("action_only"), "support_ceiling"
            ]
            .dropna()
            .iloc[0],
        )
    )
    point_ceiling = float(
        cast(
            float,
            strategies.loc[
                strategies["strategy_class"].eq("point_estimate_only"),
                "support_ceiling",
            ]
            .dropna()
            .iloc[0],
        )
    )
    axes[2].axvline(action_ceiling, color=_GREY, linestyle=":", linewidth=0.8)
    axes[2].axvline(point_ceiling, color=_GREY, linestyle="--", linewidth=0.8)
    axes[2].text(
        action_ceiling,
        4.28,
        "Action-only ceiling",
        color=_GREY,
        fontsize=6.5,
        ha="center",
        va="bottom",
    )
    axes[2].text(
        point_ceiling,
        4.58,
        "Point-estimate ceiling",
        color=_GREY,
        fontsize=6.5,
        ha="center",
        va="bottom",
    )
    axes[2].set_yticks(positions, [labels[value] for value in strategy_order])
    axes[2].set_ylim(-0.5, 4.9)
    axes[2].set_xlim(0.0, 1.03)
    axes[2].set_xticks((0.0, 0.5, 1.0))
    axes[2].set_xlabel("Views with supported action")
    _clean_axis(axes[2])
    for label, axis in zip(("a", "b", "c"), axes, strict=True):
        axis.text(
            -0.18, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=8
        )
    return _save(fig, output_stem)


def render_policy_value_v1(*, source_csv: Path, output_stem: Path) -> tuple[Path, Path]:
    """Render reference-action coverage, supported-set regret, and budget response."""

    _style()
    frame = pd.read_csv(source_csv)
    required = {
        "scenario_id",
        "information_size",
        "resource_budget_units",
        "oracle_action_supported_rate",
        "oracle_action_supported_lower",
        "oracle_action_supported_upper",
        "best_supported_regret",
        "best_supported_regret_lower",
        "best_supported_regret_upper",
        "worst_supported_regret",
        "worst_supported_regret_lower",
        "worst_supported_regret_upper",
        "adjusted_point_regret",
        "adjusted_point_regret_lower",
        "adjusted_point_regret_upper",
        "alphabetical_regret",
        "alphabetical_regret_lower",
        "alphabetical_regret_upper",
        "oracle_terminal_success_probability",
    }
    if not required <= set(frame):
        raise ValueError("Policy-value source table lacks required fields.")
    if not np.isfinite(frame[list(required - {"scenario_id"})].to_numpy(float)).all():
        raise ValueError("Policy-value figure requires finite numeric source values.")
    scenario_order = ("clear_separation", "near_tie", "threshold_uncertainty")
    scenario_labels = {
        "clear_separation": "Clear separation",
        "near_tie": "Near tie",
        "threshold_uncertainty": "Near efficacy threshold",
    }
    colours = dict(zip(scenario_order, (_BLUE, _GREEN, _ORANGE), strict=True))
    markers = dict(zip(scenario_order, ("o", "s", "^"), strict=True))
    budgets = tuple(sorted(frame["resource_budget_units"].astype(int).unique()))
    if budgets != (8, 10):
        raise ValueError(
            "Policy-value figure requires the declared 8- and 10-unit budgets."
        )
    information = tuple(sorted(frame["information_size"].astype(int).unique()))
    fig, axes = plt.subplots(1, 3, figsize=(_WIDTH_IN, 2.45), constrained_layout=True)

    coverage = frame.loc[frame["resource_budget_units"].eq(10)]
    for scenario_id in scenario_order:
        rows = coverage.loc[coverage["scenario_id"].eq(scenario_id)].sort_values(
            "information_size"
        )
        if tuple(rows["information_size"].astype(int)) != information:
            raise ValueError(
                f"Policy-value coverage lacks a complete {scenario_id!r} series."
            )
        x = rows["information_size"].to_numpy(float)
        estimate = rows["oracle_action_supported_rate"].to_numpy(float)
        lower = rows["oracle_action_supported_lower"].to_numpy(float)
        upper = rows["oracle_action_supported_upper"].to_numpy(float)
        axes[0].plot(
            x,
            estimate,
            color=colours[scenario_id],
            marker=markers[scenario_id],
            markersize=3.2,
            label=scenario_labels[scenario_id],
            zorder=3,
        )
        axes[0].fill_between(
            x, lower, upper, color=colours[scenario_id], alpha=0.10, linewidth=0
        )
    axes[0].set_xlabel("Information per candidate")
    axes[0].set_ylabel("Best modeled action retained (proportion)")
    axes[0].set_ylim(0.72, 1.01)
    axes[0].legend(frameon=False, loc="lower right")

    low_information = min(information)
    regret = frame.loc[
        frame["resource_budget_units"].eq(8)
        & frame["information_size"].eq(low_information)
    ].set_index("scenario_id")
    positions = np.arange(len(scenario_order))[::-1]
    for position, scenario_id in zip(positions, scenario_order, strict=True):
        row = cast(pd.Series, regret.loc[scenario_id])
        best = float(row["best_supported_regret"])
        worst = float(row["worst_supported_regret"])
        point = float(row["adjusted_point_regret"])
        alphabetical = float(row["alphabetical_regret"])
        axes[1].plot(
            (best, worst),
            (position, position),
            color=colours[scenario_id],
            linewidth=2.2,
            zorder=2,
        )
        for regret_estimate, prefix in (
            (best, "best_supported"),
            (worst, "worst_supported"),
        ):
            axes[1].errorbar(
                regret_estimate,
                position,
                xerr=np.asarray(
                    [
                        [regret_estimate - float(row[f"{prefix}_regret_lower"])],
                        [float(row[f"{prefix}_regret_upper"]) - regret_estimate],
                    ]
                ),
                color=colours[scenario_id],
                marker="|",
                markersize=6,
                linewidth=0.8,
                capsize=1.5,
                zorder=3,
            )
        axes[1].errorbar(
            point,
            position,
            xerr=np.asarray(
                [
                    [point - float(row["adjusted_point_regret_lower"])],
                    [float(row["adjusted_point_regret_upper"]) - point],
                ]
            ),
            color="#111827",
            marker="o",
            markerfacecolor="white",
            markersize=3.6,
            linewidth=0.8,
            capsize=1.5,
            zorder=4,
        )
        axes[1].errorbar(
            alphabetical,
            position,
            xerr=np.asarray(
                [
                    [alphabetical - float(row["alphabetical_regret_lower"])],
                    [float(row["alphabetical_regret_upper"]) - alphabetical],
                ]
            ),
            color=_ORANGE,
            marker="x",
            markersize=4.0,
            linewidth=0.8,
            capsize=1.5,
            zorder=4,
        )
    axes[1].set_yticks(positions, [scenario_labels[value] for value in scenario_order])
    axes[1].set_xlabel(f"Terminal-success regret at N = {low_information}")
    axes[1].plot([], [], color=_GREY, linewidth=2.2, label="Supported-action range")
    axes[1].scatter(
        [],
        [],
        facecolor="white",
        edgecolor="#111827",
        marker="o",
        s=15,
        label="Point ranks",
    )
    axes[1].scatter(
        [], [], color=_ORANGE, marker="x", s=15, label="Alphabetical labels"
    )
    axes[1].legend(
        frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.27)
    )

    high_information = max(information)
    for scenario_id in scenario_order:
        rows = frame.loc[
            frame["scenario_id"].eq(scenario_id)
            & frame["information_size"].eq(high_information)
        ].sort_values("resource_budget_units")
        axes[2].plot(
            rows["resource_budget_units"].to_numpy(float),
            rows["oracle_terminal_success_probability"].to_numpy(float),
            color=colours[scenario_id],
            marker=markers[scenario_id],
            markersize=3.2,
            label=scenario_labels[scenario_id],
            zorder=3,
        )
    axes[2].set_xticks(budgets)
    axes[2].set_xlabel("Resource budget (units)")
    axes[2].set_ylabel("Best modeled terminal-success probability")

    for label, axis in zip(("a", "b", "c"), axes, strict=True):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)
        axis.text(
            -0.18, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=8
        )
    return _save(fig, output_stem)


def write_portfolio_route_source_data_v1(
    *, route_report_json: Path, output_csv: Path
) -> Path:
    """Export family-level route coverage and complexity from the exact release."""

    report = json.loads(Path(route_report_json).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "pass":
        raise ValueError(
            "Portfolio route source data require a passing exact-release audit."
        )
    families = report.get("families")
    if not isinstance(families, list) or len(families) != 12:
        raise ValueError(
            "Portfolio route audit must contain the twelve construction families."
        )
    required_actions = (
        "select_lead_and_reserve",
        "withhold_selection",
        "advance_lead_to_proof_of_concept",
        "promote_reserve_to_proof_of_concept",
        "advance_active_to_confirmation",
        "terminate_portfolio",
        "declare_success",
        "declare_failure",
        "declare_inconclusive",
    )
    records: list[dict[str, object]] = []
    for family in families:
        if not isinstance(family, dict):
            raise ValueError("Portfolio family route summaries must be objects.")
        family_id = str(family["family_id"])
        label = family_id.split("_", maxsplit=1)[0]
        supported = set(str(value) for value in family["supported_action_ids"])
        for action_id in required_actions:
            records.append(
                {
                    "panel": "action_coverage",
                    "family_id": family_id,
                    "family_label": label,
                    "action_id": action_id,
                    "reachable": int(action_id in supported),
                    "terminal_route_count_min": "",
                    "terminal_route_count_max": "",
                }
            )
        records.append(
            {
                "panel": "route_complexity",
                "family_id": family_id,
                "family_label": label,
                "action_id": "",
                "reachable": "",
                "terminal_route_count_min": int(family["terminal_route_count_min"]),
                "terminal_route_count_max": int(family["terminal_route_count_max"]),
            }
        )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output, index=False, lineterminator="\n")
    return output


def render_portfolio_routes_v1(
    *, source_csv: Path, output_stem: Path
) -> tuple[Path, Path]:
    """Render supported-action coverage and terminal-route complexity by family."""

    _style()
    frame = pd.read_csv(source_csv)
    action_rows = frame.loc[frame["panel"].eq("action_coverage")]
    complexity = frame.loc[frame["panel"].eq("route_complexity")]
    family_order = tuple(dict.fromkeys(action_rows["family_label"].astype(str)))
    action_order = tuple(dict.fromkeys(action_rows["action_id"].astype(str)))
    if len(family_order) != 12 or len(action_order) != 9 or len(complexity) != 12:
        raise ValueError(
            "Portfolio route figure requires twelve families and nine action classes."
        )
    matrix = (
        action_rows.pivot(index="family_label", columns="action_id", values="reachable")
        .reindex(index=family_order, columns=action_order)
        .to_numpy(float)
    )
    action_labels = {
        "select_lead_and_reserve": "Select\nlead + reserve",
        "withhold_selection": "Withhold\nselection",
        "advance_lead_to_proof_of_concept": "Advance\nlead",
        "promote_reserve_to_proof_of_concept": "Promote\nreserve",
        "advance_active_to_confirmation": "Advance to\nconfirmation",
        "terminate_portfolio": "Terminate\nportfolio",
        "declare_success": "Declare\nsuccess",
        "declare_failure": "Declare\nfailure",
        "declare_inconclusive": "Declare\ninconclusive",
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.1),
        gridspec_kw={"width_ratios": (2.25, 1.0)},
        constrained_layout=True,
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            reachable = matrix[row_index, column_index] == 1
            axes[0].add_patch(
                Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1.0,
                    1.0,
                    facecolor=_BLUE if reachable else "#F3F4F6",
                    edgecolor="white",
                    linewidth=0.5,
                )
            )
            if reachable:
                axes[0].scatter(
                    column_index,
                    row_index,
                    s=5,
                    marker="o",
                    facecolor="white",
                    edgecolor="white",
                    linewidth=0.4,
                )
    axes[0].set_xlim(-0.5, len(action_order) - 0.5)
    axes[0].set_ylim(len(family_order) - 0.5, -0.5)
    axes[0].set_xticks(
        range(len(action_order)),
        [action_labels[value] for value in action_order],
        rotation=45,
        ha="right",
    )
    axes[0].set_yticks(range(len(family_order)), family_order)
    axes[0].set_ylabel("Trial family")
    axes[0].tick_params(length=0)
    complexity = complexity.set_index("family_label").reindex(family_order)
    positions = np.arange(len(family_order))
    lower = complexity["terminal_route_count_min"].to_numpy(float)
    upper = complexity["terminal_route_count_max"].to_numpy(float)
    axes[1].hlines(positions, lower, upper, color=_GREY, linewidth=1.8)
    axes[1].scatter(
        lower,
        positions,
        color=_ORANGE,
        marker="|",
        s=32,
        label="Minimum across views",
        zorder=3,
    )
    axes[1].scatter(
        upper,
        positions,
        color=_GREEN,
        marker="o",
        s=13,
        label="Maximum across views",
        zorder=3,
    )
    axes[1].set_yticks(positions, family_order)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Supported terminal routes per view")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(axis="x", color="#E5E7EB", linewidth=0.6, zorder=0)
    for label, axis in zip(("a", "b"), axes, strict=True):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.15, 1.03, label, transform=axis.transAxes, fontweight="bold", fontsize=8
        )
    return _save(fig, output_stem)


def build_trialdev_scientific_figures_v1(
    *,
    operating_summary_csv: Path,
    observational_replay_root: Path,
    release_audit_json: Path,
    grader_controls_csv: Path,
    decision_boundary_json: Path,
    portfolio_difficulty_json: Path,
    portfolio_routes_json: Path,
    policy_value_csv: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Build TrialDev result figures and their exact tidy source tables."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to mix scientific-figure builds: {output}.")
    source_root = output / "source_data"
    source_root.mkdir(parents=True)
    identification = write_observational_replay_source_data_v1(
        replay_root=observational_replay_root,
        output_csv=source_root / "02_identification_uncertainty.csv",
    )
    policy_response = write_operating_effect_source_data_v1(
        summary_csv=operating_summary_csv,
        axes=("resources", "stopping", "reallocation"),
        output_csv=source_root / "03_policy_response.csv",
    )
    mechanism_response = write_operating_effect_source_data_v1(
        summary_csv=operating_summary_csv,
        axes=("efficacy", "safety", "operations", "asset_correlation"),
        output_csv=source_root / "04_mechanism_response.csv",
    )
    realism = write_clinical_realism_source_data_v1(
        release_audit_json=release_audit_json,
        output_csv=source_root / "05_clinical_realism.csv",
    )
    operating_frame = pd.read_csv(operating_summary_csv)
    if set(operating_frame["axis"]) != set(_AXIS_LABELS):
        raise ValueError(
            "Operating-characteristic figure requires the complete ten-axis inventory."
        )
    operating = source_root / "06_operating_characteristics.csv"
    operating_frame.to_csv(operating, index=False, lineterminator="\n")
    controls = write_grader_control_source_data_v1(
        controls_csv=grader_controls_csv,
        output_csv=source_root / "07_grader_controls.csv",
    )
    difficulty = write_decision_difficulty_source_data_v1(
        decision_boundary_json=decision_boundary_json,
        portfolio_difficulty_json=portfolio_difficulty_json,
        output_csv=source_root / "08_decision_difficulty.csv",
    )
    policy_frame = pd.read_csv(policy_value_csv)
    policy = source_root / "09_policy_value.csv"
    policy_frame.to_csv(policy, index=False, lineterminator="\n")
    routes = write_portfolio_route_source_data_v1(
        route_report_json=portfolio_routes_json,
        output_csv=source_root / "10_portfolio_routes.csv",
    )

    paths: list[Path] = []
    paths.extend(
        render_identification_recoverability_v1(
            source_csv=identification,
            output_stem=output / "02_identification_uncertainty",
        )
    )
    paths.extend(
        render_paired_effect_forest_v1(
            summary_csv=policy_response,
            axes=("resources", "stopping", "reallocation"),
            output_stem=output / "03_policy_response",
            panel_label="a",
        )
    )
    paths.extend(
        render_paired_effect_forest_v1(
            summary_csv=mechanism_response,
            axes=("efficacy", "safety", "operations", "asset_correlation"),
            output_stem=output / "04_mechanism_response",
            panel_label="a",
        )
    )
    paths.extend(
        render_clinical_realism_v1(
            source_csv=realism, output_stem=output / "05_clinical_realism"
        )
    )
    paths.extend(
        render_operating_characteristics_v1(
            summary_csv=operating, output_stem=output / "06_operating_characteristics"
        )
    )
    paths.extend(
        render_failure_decomposition_v1(
            source_csv=controls, output_stem=output / "07_grader_controls"
        )
    )
    paths.extend(
        render_decision_difficulty_v1(
            source_csv=difficulty, output_stem=output / "08_decision_difficulty"
        )
    )
    paths.extend(
        render_policy_value_v1(
            source_csv=policy, output_stem=output / "09_policy_value"
        )
    )
    paths.extend(
        render_portfolio_routes_v1(
            source_csv=routes, output_stem=output / "10_portfolio_routes"
        )
    )
    return tuple(paths)


__all__ = [
    "build_trialdev_scientific_figures_v1",
    "render_clinical_realism_v1",
    "render_decision_difficulty_v1",
    "render_failure_decomposition_v1",
    "render_identification_recoverability_v1",
    "render_operating_characteristics_v1",
    "render_paired_effect_forest_v1",
    "render_policy_value_v1",
    "render_portfolio_routes_v1",
    "write_clinical_realism_source_data_v1",
    "write_decision_difficulty_source_data_v1",
    "write_grader_control_source_data_v1",
    "write_observational_replay_source_data_v1",
    "write_portfolio_route_source_data_v1",
    "write_operating_effect_source_data_v1",
]
