"""Two worked demonstrations on one trial design: the right analysis depends on the data, not the protocol.

Both demonstrations analyse the heart-failure trial series in which routine site records of
hospitalization are checked against blinded adjudication in a validation sample. They read the public
participant tables of every public world, compute three analyses, and price each answer with the
verifier's public decision layer: the expected-value-optimal development action (stop, or a
confirmatory trial of k times this trial's size) and its regret against the credited analysis.

``trap`` (figure ``trap_decision``) answers: what happens when the default analysis is wrong for these
data? In the series' trap cell (tier A3) routine records miss more events in the active arm, so the
routine-record risk difference is biased; the audit-corrected analysis is credited.

``twin`` (figure ``twin_overcorrection``) answers: does the benchmark punish reflexive correction? In
the trap's twin cell, which shares the trap's base worlds, routine records miss events equally in both
arms. The joint likelihood over routine and adjudicated records is credited there, and the trap's audit
correction, though unbiased, is less precise and changes decisions.

Selection rule, fixed before any result was seen: series TE-S06; its A3 cell and the one other TE-S06
cell whose base worlds are the A3 cell's (the twin); every public world of both cells at evidence
context C5 (raw source tables with declared record defects). No world is excluded. Each credited
analysis recomputed here must reproduce the credited reference in the public evaluator record, or the
demonstration stops.

Run ``python trap_decision.py --output results/trap-decision`` (add ``--release-root`` for a local copy
of the dataset).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from figure_style import COLOURS, GRID, INK, MARKERS, MUTED, save_figure
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pydantic import BaseModel, ConfigDict
from release_access import PublicRelease, ReleaseEvidence, add_release_arguments
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from scipy.stats import t as student_t
from trialagentbench_verifier.decision_consequence import (
    DecisionEconomicsV1,
    DecisionReportV1,
    decision_consequence_v1,
)

from trialagentbench.grading.outcome import TrialEvalEvaluatorRecordV1

SERIES = "TE-S06"
CONTEXT = "C5"
DELETE_GROUPS = 20
TOLERANCE = 1e-4
"""Largest accepted difference from the evaluator record's credited estimate or interval (risk scale)."""
TASK_INDEX = "evaluation/grader/task_index.json"
EVALUATOR_RECORDS = "evaluation/grader/evaluator_records.jsonl"
RAW_TABLES = ("participant_records.parquet", "event_history.parquet", "identifier_crosswalk.parquet")

Analysis = Literal["routine", "audit", "joint"]
Role = Literal["trap", "twin"]
LABELS: dict[str, str] = {
    "routine": "Routine records only",
    "audit": "Audit-corrected",
    "joint": "Joint likelihood",
}
CREDITED: dict[str, Analysis] = {
    "observed:validated_event_discrepancy_risk_difference": "audit",
    "observed:validated_endpoint_joint_likelihood": "joint",
}
_Floats = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Report:
    """One analysis' estimate and two-sided 95% interval on the risk-difference scale."""

    estimate: float
    lower: float
    upper: float


# ---------------------------------------------------------------------------------------------- analyses


def _interval(estimate: float, standard_error: float, critical: float) -> Report:
    if not math.isfinite(standard_error) or standard_error <= 0:
        raise ValueError("Analysis has no positive finite standard error")
    return Report(estimate, estimate - critical * standard_error, estimate + critical * standard_error)


def routine_risk_difference(frame: pd.DataFrame) -> Report:
    """Return the active-minus-control risk difference in routine records, with a Wald interval.

    Parameters
    ----------
    frame
        Analysis frame from ``analysis_frame``.

    Returns
    -------
    Report
        Estimate and 95% interval.
    """
    active = frame["assigned_arm"].eq("active")
    p1, n1 = float(frame.loc[active, "routine"].mean()), int(active.sum())
    p0, n0 = float(frame.loc[~active, "routine"].mean()), int((~active).sum())
    return _interval(float(p1 - p0), math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0), float(norm.ppf(0.975)))


def _audit_contrast(frame: pd.DataFrame) -> float:
    corrected = []
    for arm in ("control", "active"):
        rows = frame.loc[frame["assigned_arm"].eq(arm)]
        total = 0.0
        for _, cell in rows.groupby("validation_stratum"):
            audited = cell.loc[cell["selected"]]
            if audited.empty:
                raise ValueError("An arm and validation stratum has no audited participant")
            discrepancy = float((audited["adjudicated"] - audited["routine"]).mean())
            total += len(cell) * (float(cell["routine"].mean()) + discrepancy)
        corrected.append(total / len(rows))
    return corrected[1] - corrected[0]


def _delete_groups(frame: pd.DataFrame) -> npt.NDArray[np.int64]:
    """Assign participants to 20 hash-ordered delete groups, balanced within each arm."""
    groups = np.full(len(frame), -1, dtype=np.int64)
    identifiers = [str(value) for value in frame["participant_id"]]
    for arm in sorted(set(frame["assigned_arm"])):
        indices = np.flatnonzero(frame["assigned_arm"].to_numpy() == arm)
        ordered = sorted(indices, key=lambda index: hashlib.sha256(identifiers[index].encode()).digest())
        for offset, index in enumerate(ordered):
            groups[index] = offset % DELETE_GROUPS
    return groups


def audit_corrected_risk_difference(frame: pd.DataFrame) -> Report:
    """Return the audit-corrected risk difference with a delete-a-group jackknife interval.

    Within each arm and validation stratum, the mean adjudicated-minus-routine discrepancy in the audit
    sample is added to the routine risk; the stratum-weighted corrected arm risks are contrasted. The
    complete calculation is refitted without each of 20 balanced participant groups (two-sided 95% t
    interval, 19 degrees of freedom).

    Parameters
    ----------
    frame
        Analysis frame from ``analysis_frame``.

    Returns
    -------
    Report
        Estimate and 95% interval.
    """
    groups = _delete_groups(frame)
    replicates = np.array([_audit_contrast(frame.loc[groups != group]) for group in range(DELETE_GROUPS)])
    variance = (DELETE_GROUPS - 1) / DELETE_GROUPS * float(np.sum((replicates - replicates.mean()) ** 2))
    critical = float(student_t.ppf(0.975, DELETE_GROUPS - 1))
    return _interval(_audit_contrast(frame), math.sqrt(variance), critical)


def _negative_log_likelihood(
    parameters: _Floats,
    *,
    arm: npt.NDArray[np.int64],
    routine: _Floats,
    selected: npt.NDArray[np.bool_],
    truth: _Floats,
) -> float:
    risk_control, risk_active, sensitivity, specificity = expit(parameters)
    risk = np.where(arm == 1, risk_active, risk_control)
    routine_probability = risk * sensitivity + (1 - risk) * (1 - specificity)
    unaudited = np.where(routine == 1, routine_probability, 1 - routine_probability)
    audited = np.where(truth == 1, risk, 1 - risk) * np.where(
        truth == 1,
        np.where(routine == 1, sensitivity, 1 - sensitivity),
        np.where(routine == 0, specificity, 1 - specificity),
    )
    return -float(np.log(np.where(selected, audited, unaudited)).sum())


def _hessian(function: Callable[[_Floats], float], point: _Floats) -> _Floats:
    step = 1e-4 * np.maximum(np.abs(point), 1.0)
    size = len(point)
    hessian = np.empty((size, size))
    for i in range(size):
        for j in range(i, size):
            shifts = []
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                moved = point.copy()
                moved[i] += si * step[i]
                moved[j] += sj * step[j]
                shifts.append(function(moved))
            hessian[i, j] = hessian[j, i] = (shifts[0] - shifts[1] - shifts[2] + shifts[3]) / (4 * step[i] * step[j])
    return hessian


def joint_likelihood_risk_difference(frame: pd.DataFrame) -> Report:
    """Return the risk difference from the joint likelihood of routine and adjudicated records.

    Within each validation stratum, arm risks and a common routine sensitivity and specificity are
    fitted by maximum likelihood (all routine records, plus adjudication where audited). Stratum
    differences are weighted by stratum size, with a normal interval that adds the variance of the
    estimated stratum composition.

    Parameters
    ----------
    frame
        Analysis frame from ``analysis_frame``.

    Returns
    -------
    Report
        Estimate and 95% interval.
    """
    rows = []
    for _, stratum in frame.groupby("validation_stratum", sort=True):
        arm = np.asarray(stratum["assigned_arm"].eq("active"), dtype=np.int64)
        objective = partial(
            _negative_log_likelihood,
            arm=arm,
            routine=stratum["routine"].to_numpy(dtype=float),
            selected=stratum["selected"].to_numpy(dtype=bool),
            truth=stratum["adjudicated"].fillna(0).to_numpy(dtype=float),
        )
        fitted = minimize(
            objective,
            np.zeros(4),
            method="L-BFGS-B",
            jac="3-point",
            bounds=[(-30.0, 30.0)] * 4,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000, "maxls": 50},
        )
        if not fitted.success:
            raise ValueError("Joint likelihood did not converge")
        risks = expit(fitted.x[:2])
        gradient = np.array([-risks[0] * (1 - risks[0]), risks[1] * (1 - risks[1]), 0.0, 0.0])
        variance = float(gradient @ np.linalg.solve(_hessian(objective, fitted.x), gradient))
        rows.append((len(stratum) / len(frame), risks[0], risks[1], variance))
    values = np.array(rows)
    weights = values[:, 0]
    control, active = weights @ values[:, 1:3]
    differences = values[:, 2] - values[:, 1]
    composition = weights @ (differences - (active - control)) ** 2 / len(frame)
    return _interval(float(active - control), math.sqrt(float(weights**2 @ values[:, 3] + composition)), 1.959964)


ANALYSES: dict[Analysis, Callable[[pd.DataFrame], Report]] = {
    "routine": routine_risk_difference,
    "audit": audit_corrected_risk_difference,
    "joint": joint_likelihood_risk_difference,
}


def analysis_frame(tables: dict[str, bytes]) -> pd.DataFrame:
    """Build one row per randomized participant from the released raw tables.

    The source-record rules are those the task's data-integrity policy documents: former source
    identifiers are mapped through the identifier crosswalk, and each participant must then hold exactly
    one routine assessment and at most one adjudicated assessment.

    Parameters
    ----------
    tables
        Raw files keyed by their path below ``data/raw/``.

    Returns
    -------
    pandas.DataFrame
        ``participant_id``, ``assigned_arm``, ``validation_stratum``, ``routine``, ``selected`` and
        ``adjudicated`` (missing outside the audit).

    Raises
    ------
    ValueError
        If the records break the documented identity rules.
    """
    participants = pd.read_parquet(io.BytesIO(tables["participant_records.parquet"]))
    events = pd.read_parquet(io.BytesIO(tables["event_history.parquet"]))
    identifiers = events["participant_id"].astype(str)
    if "identifier_crosswalk.parquet" in tables:
        crosswalk = pd.read_parquet(io.BytesIO(tables["identifier_crosswalk.parquet"]))
        mapping = dict(zip(crosswalk["source_identifier"], crosswalk["current_identifier"], strict=True))
        identifiers = identifiers.map(mapping).fillna(identifiers)
    events["participant_id"] = identifiers.astype("int64")
    assessed = events.loc[events["event_kind"].eq("endpoint_assessment")]
    if assessed.duplicated(["participant_id", "event_id"]).any():
        raise ValueError("A participant holds two assessments of one kind")
    status = assessed.pivot(index="participant_id", columns="event_id", values="event_by_contact")
    routine_column = next(column for column in map(str, status.columns) if column.startswith("routine_"))
    adjudicated_column = next(column for column in map(str, status.columns) if column.startswith("adjudicated_"))
    frame = participants[["participant_id", "assigned_arm", "validation_stratum"]].merge(
        status, left_on="participant_id", right_index=True, how="left", validate="one_to_one"
    )
    if frame[routine_column].isna().any() or len(status) != len(frame):
        raise ValueError("Routine assessments do not cover the randomized cohort exactly")
    return pd.DataFrame(
        {
            "participant_id": frame["participant_id"],
            "assigned_arm": frame["assigned_arm"],
            "validation_stratum": frame["validation_stratum"],
            "routine": frame[routine_column].astype(float),
            "selected": frame[adjudicated_column].notna(),
            "adjudicated": frame[adjudicated_column].astype(float),
        }
    )


# ---------------------------------------------------------------------------------------------- release


class IndexEntry(BaseModel):
    """The fields of one evaluator task-index entry that the selection rule reads."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    task_id: str
    regime_cell_id: str
    evaluation_series_id: str
    assumption_tier: str
    context_tier: str
    base_world_id: str


@dataclass(frozen=True)
class World:
    """One public task: its cell, evaluator record and the three analyses of its data."""

    entry: IndexEntry
    record: TrialEvalEvaluatorRecordV1
    reports: dict[Analysis, Report]


def select_tasks(index: list[IndexEntry]) -> dict[Role, list[IndexEntry]]:
    """Apply the stated selection rule to the evaluator task index.

    Parameters
    ----------
    index
        Entries of ``evaluation/grader/task_index.json``.

    Returns
    -------
    dict
        ``trap`` and ``twin`` task entries at context C5: every public world of each cell.

    Raises
    ------
    ValueError
        If the series lacks exactly one A3 cell and one twin over the same base worlds.
    """
    cells: dict[str, list[IndexEntry]] = {}
    for entry in index:
        if entry.evaluation_series_id == SERIES and entry.context_tier == CONTEXT:
            cells.setdefault(entry.regime_cell_id, []).append(entry)
    traps = [cell for cell, entries in cells.items() if entries[0].assumption_tier == "A3"]
    if len(traps) != 1:
        raise ValueError(f"{SERIES} has {len(traps)} public A3 cells; expected one")
    worlds = {entry.base_world_id for entry in cells[traps[0]]}
    twins = [
        cell
        for cell, entries in cells.items()
        if cell != traps[0] and {entry.base_world_id for entry in entries} == worlds
    ]
    if len(twins) != 1:
        raise ValueError(f"{SERIES} has {len(twins)} cells sharing the trap's base worlds; expected one")
    return {"trap": cells[traps[0]], "twin": cells[twins[0]]}


def load_worlds(release: PublicRelease) -> dict[Role, list[World]]:
    """Read the selected public tasks and analyse every one.

    Parameters
    ----------
    release
        The public dataset.

    Returns
    -------
    dict
        ``trap`` and ``twin`` worlds, each ordered by true effect.

    Raises
    ------
    ValueError
        If a recomputed credited analysis differs from the evaluator record's credited reference.
    """
    shared = release.members("trialeval", "evaluator", (TASK_INDEX, EVALUATOR_RECORDS))
    index = [IndexEntry.model_validate(entry) for entry in json.loads(shared[TASK_INDEX])["entries"]]
    records = {
        record.task_id: record
        for record in map(TrialEvalEvaluatorRecordV1.model_validate_json, shared[EVALUATOR_RECORDS].splitlines())
    }
    selected = select_tasks(index)
    released = {
        entry.task_id: [
            name for name in RAW_TABLES if f"data/raw/{name}" in records[entry.task_id].participant_artifact_paths
        ]
        for entries in selected.values()
        for entry in entries
    }
    payloads = release.members(
        "trialeval",
        "evaluator",
        [f"participant/tasks/{task}/data/raw/{name}" for task, names in released.items() for name in names],
    )
    worlds: dict[Role, list[World]] = {}
    for role, entries in selected.items():
        worlds[role] = []
        for entry in entries:
            task = entry.task_id
            frame = analysis_frame(
                {name: payloads[f"participant/tasks/{task}/data/raw/{name}"] for name in released[task]}
            )
            record = records[task]
            reports = {name: analyse(frame) for name, analyse in ANALYSES.items()}
            (credited_id,) = record.reference_analysis_ids
            consequence = record.target.consequence
            if consequence is None or len(consequence.credited) != 1:
                raise ValueError(f"{task} lacks one credited reference")
            mine = reports[CREDITED[credited_id]]
            if (
                max(
                    abs(a - b)
                    for a, b in zip((mine.estimate, mine.lower, mine.upper), consequence.credited[0], strict=True)
                )
                > TOLERANCE
            ):
                raise ValueError(f"{task}: recomputed {credited_id} differs from the evaluator record")
            worlds[role].append(World(entry, record, reports))
    # Number base worlds once, by the trap's true effect, so world k is the same base world in both cells.
    order = [
        world.entry.base_world_id for world in sorted(worlds["trap"], key=lambda world: world.record.target.value)
    ]
    return {
        role: sorted(items, key=lambda world: order.index(world.entry.base_world_id)) for role, items in worlds.items()
    }


# ---------------------------------------------------------------------------------------------- decisions


def price(world: World, analysis: Analysis) -> dict[str, object]:
    """Price one analysis of one world with the verifier's public decision layer.

    Parameters
    ----------
    world
        The analysed world.
    analysis
        Which of its three analyses to price.

    Returns
    -------
    dict
        One tidy row: the report, the action it drives, and its regret against the credited analysis.
    """
    target = world.record.target
    consequence = target.consequence
    if consequence is None:
        raise ValueError(f"{world.entry.task_id} has no decision consequence")
    report = world.reports[analysis]
    priced = decision_consequence_v1(
        DecisionReportV1(estimate=report.estimate, lower=report.lower, upper=report.upper),
        tuple(DecisionReportV1(estimate=e, lower=lo, upper=hi) for e, lo, hi in consequence.credited),
        truth=target.value,
        scale=target.reference_sampling_standard_deviation,
        participants=consequence.participants,
        favorable_direction=target.binding.favorable_direction,
        economics=DecisionEconomicsV1.model_validate(consequence.economics.model_dump(exclude={"cost_type"})),
    )
    (credited_id,) = world.record.reference_analysis_ids
    return {
        "cell": world.entry.regime_cell_id,
        "tier": world.entry.assumption_tier,
        "base_world_id": world.entry.base_world_id,
        "task_id": world.entry.task_id,
        "true_risk_difference": target.value,
        "analysis": analysis,
        "credited": CREDITED[credited_id] == analysis,
        "estimate": report.estimate,
        "lower": report.lower,
        "upper": report.upper,
        "confirm": priced.agent.confirm,
        "size_multiple": priced.agent.size_multiple,
        "confirmatory_participants": priced.agent_confirmatory_participants,
        "regret_musd": priced.regret_musd,
        "regret_at_truth_musd": priced.realised_regret_musd,
        "decision_differs": priced.go_decision_differs,
        "consequential": priced.consequential,
    }


def decision_table(worlds: dict[Role, list[World]]) -> pd.DataFrame:
    """Price every analysis of every selected world.

    Parameters
    ----------
    worlds
        Output of ``load_worlds``.

    Returns
    -------
    pandas.DataFrame
        One row per role, world and analysis; world k is the same base world in both roles.
    """
    rows = [
        {"role": role, "world": number} | price(world, analysis)
        for role, items in worlds.items()
        for number, world in enumerate(items, start=1)
        for analysis in ANALYSES
    ]
    return pd.DataFrame(rows)


def release_scale(evidence: ReleaseEvidence, trap_cell: str, twin_cell: str) -> dict[str, float]:
    """Read the cell-level consequence of each excluded analysis from the release-evidence record.

    Parameters
    ----------
    evidence
        The parsed release-evidence record.
    trap_cell, twin_cell
        The selected cells.

    Returns
    -------
    dict
        Mean regret ($M), its standard error and consequential share of the routine analysis in the trap
        cell, and mean regret and consequential share of the tempting correction in the twin cell.

    Raises
    ------
    ValueError
        If the record does not price both cells.
    """
    trialeval = evidence.trialeval
    naive = [cost for cost in trialeval.policy_public.naive_costs if cost.regime_cell_id == trap_cell]
    over = [cost for cost in trialeval.over_corrections if cost.regime_cell_id == twin_cell]
    if len(naive) != 1 or len(over) != 1:
        raise ValueError("The release-evidence record does not price both selected cells")
    return {
        "trap_naive_regret_musd": naive[0].regret_musd,
        "trap_naive_regret_se_musd": naive[0].regret_se_musd,
        "trap_naive_consequential_share": naive[0].consequential_share,
        "twin_correction_regret_musd": over[0].mean_regret_musd,
        "twin_correction_consequential_share": over[0].consequential_share,
        "worlds_per_cell": float(trialeval.policy_public.worlds_per_cell),
    }


# ---------------------------------------------------------------------------------------------- figures


def _pp(value: Any) -> Any:
    """Express a proportion (or a column of them) in percentage points."""
    return 100.0 * value


def _decision_label(row: pd.Series) -> str:
    return "stop" if not row["confirm"] else f"confirm {row['size_multiple']:.1f}×"


def trap_figure(table: pd.DataFrame, scale: dict[str, float]) -> Figure:
    """Draw the trap: estimates against truth, and the development decision each estimate drives.

    Parameters
    ----------
    table
        Output of ``decision_table``.
    scale
        Output of ``release_scale``.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    rows = table.loc[table["role"].eq("trap") & table["analysis"].isin(("routine", "audit"))]
    worlds = sorted(rows["world"].unique())
    participants = int(rows["confirmatory_participants"].max() / rows["size_multiple"].max())
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": (1.4, 1.0), "wspace": 0.06}, sharey=True
    )
    offsets = {"routine": 0.17, "audit": -0.17}
    for analysis, offset in offsets.items():
        part = rows.loc[rows["analysis"].eq(analysis)]
        y = len(worlds) - part["world"].to_numpy() + offset
        style = {"color": COLOURS[analysis], "marker": MARKERS[analysis], "markersize": 5.5, "linestyle": "none"}
        left.errorbar(
            _pp(part["estimate"]),
            y,
            xerr=(_pp(part["estimate"] - part["lower"]), _pp(part["upper"] - part["estimate"])),
            elinewidth=1.2,
            capsize=0,
            label=LABELS[analysis] + (" (credited)" if part["credited"].all() else ""),
            **style,
        )
        right.plot(part["size_multiple"], y, **style)
        for _, row in part.iterrows():
            yy = len(worlds) - row["world"] + offset
            text = _decision_label(row)
            if row["regret_musd"] >= 0.5:
                text += f"  ·  regret ${row['regret_musd']:.0f}M"
            weight = "bold" if row["consequential"] else "normal"
            right.annotate(
                text,
                (row["size_multiple"], yy),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=COLOURS[analysis],
                fontweight=weight,
            )
    truth = rows.drop_duplicates("world")
    left.scatter(
        _pp(truth["true_risk_difference"]),
        len(worlds) - truth["world"],
        marker="D",
        s=26,
        color=INK,
        zorder=5,
        label="True effect",
    )
    left.axvline(0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    left.set_yticks([len(worlds) - world for world in worlds], [f"World {world}" for world in worlds])
    left.set_ylim(-0.6, len(worlds) - 0.4)
    left.set_xlabel("Risk difference in heart-failure hospitalization\n(percentage points; below 0 favours treatment)")
    left.set_title("a  Estimate and 95% interval against the true effect")
    left.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.07), ncol=3, handletextpad=0.3, columnspacing=1.2)
    left.grid(axis="x", color=GRID, linewidth=0.6)
    right.set_xlim(-0.15, 5.3)
    right.set_xticks([0, 1, 2, 3], ["stop", "1×", "2×", "3×"])
    right.set_xlabel(
        f"Development decision: confirmatory trial size\n(multiple of this trial's {participants:,} participants)"
    )
    right.set_title("b  Decision each estimate drives")
    right.grid(axis="x", color=GRID, linewidth=0.6)
    right.tick_params(axis="y", length=0)
    figure.text(
        0.01,
        0.01,
        f"Across {scale['worlds_per_cell']:,.0f} prior-drawn worlds of this cell, the routine-records analysis costs "
        f"${scale['trap_naive_regret_musd']:.1f}M per trial in expected regret (SE {scale['trap_naive_regret_se_musd']:.1f}) "
        f"and changes the stop-or-confirm decision consequentially in {scale['trap_naive_consequential_share']:.0%} of them. "
        "Bold: consequential (decision flipped, regret ≥ $5M).",
        fontsize=7.2,
        color=MUTED,
        wrap=True,
    )
    figure.subplots_adjust(left=0.1, right=0.985, top=0.83, bottom=0.27)
    return figure


def twin_figure(table: pd.DataFrame, scale: dict[str, float]) -> Figure:
    """Draw the twin beside the trap: the regret of each analysis on the same base worlds.

    Parameters
    ----------
    table
        Output of ``decision_table``.
    scale
        Output of ``release_scale``.

    Returns
    -------
    matplotlib.figure.Figure
        The paired figure.
    """
    worlds = sorted(table["world"].unique())
    limit = 1.12 * float(table["regret_musd"].max())
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True, sharex=True, gridspec_kw={"wspace": 0.06})
    offsets = {"routine": 0.24, "joint": 0.0, "audit": -0.24}
    notes = {
        "trap": (
            "Trap: routine records miss more events on treatment",
            f"Over {scale['worlds_per_cell']:,.0f} prior-drawn worlds of this cell, routine records only\n"
            f"cost ${scale['trap_naive_regret_musd']:.1f}M per trial; decision flipped in {scale['trap_naive_consequential_share']:.0%}.",
        ),
        "twin": (
            "Twin: routine records miss events equally",
            f"Over {scale['worlds_per_cell']:,.0f} prior-drawn worlds of this cell, the audit correction\n"
            f"costs ${scale['twin_correction_regret_musd']:.1f}M per trial; decision flipped in {scale['twin_correction_consequential_share']:.0%}.",
        ),
    }
    for axis, role, letter in zip(axes, ("trap", "twin"), "ab", strict=True):
        part = table.loc[table["role"].eq(role)]
        for analysis, offset in offsets.items():
            rows = part.loc[part["analysis"].eq(analysis)]
            y = len(worlds) - rows["world"].to_numpy() + offset
            credited = bool(rows["credited"].all())
            axis.hlines(y, 0, rows["regret_musd"], color=COLOURS[analysis], linewidth=1.4)
            axis.plot(
                rows["regret_musd"],
                y,
                linestyle="none",
                marker=MARKERS[analysis],
                markersize=5.5,
                color=COLOURS[analysis],
                markerfacecolor=COLOURS[analysis] if credited else "white",
                markeredgewidth=1.2,
                label=LABELS[analysis],
            )
            flipped = rows.loc[rows["consequential"]]
            axis.scatter(
                flipped["regret_musd"],
                len(worlds) - flipped["world"] + offset,
                s=150,
                facecolor="none",
                edgecolor=INK,
                linewidth=0.8,
                zorder=4,
            )
            for _, row in rows.loc[rows["regret_musd"] >= 0.5].iterrows():
                axis.annotate(
                    f"${row['regret_musd']:.0f}M",
                    (row["regret_musd"], len(worlds) - row["world"] + offset),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=6.8,
                    color=COLOURS[analysis],
                )
        title, note = notes[role]
        axis.set_title(f"{letter}  {title}", fontsize=8.6)
        axis.text(0.0, -0.27, note, transform=axis.transAxes, fontsize=7.2, color=MUTED, va="top")
        axis.set_xlim(-0.02 * limit, limit)
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_xlabel("Expected regret of the decision it drives ($M)")
        axis.axvline(0, color=MUTED, linewidth=0.7)
    axes[0].set_yticks([len(worlds) - world for world in worlds], [f"World {world}" for world in worlds])
    axes[0].set_ylim(-0.6, len(worlds) - 0.3)
    axes[1].tick_params(axis="y", length=0)
    handles = [
        Line2D([], [], linestyle="none", marker=MARKERS[name], markersize=5.5, color=COLOURS[name]) for name in offsets
    ]
    handles += [
        Line2D([], [], linestyle="none", marker="o", markersize=5.5, color=MUTED, markerfacecolor="white"),
        Line2D([], [], linestyle="none", marker="o", markersize=10, markerfacecolor="none", markeredgecolor=INK),
    ]
    figure.legend(
        handles,
        [*(LABELS[name] for name in offsets), "Open: not credited in that cell", "Stop-or-confirm decision flipped"],
        loc="upper left",
        bbox_to_anchor=(0.085, 0.995),
        ncol=3,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    figure.subplots_adjust(left=0.1, right=0.98, top=0.78, bottom=0.27)
    return figure


def main(argv: list[str] | None = None) -> int:
    """Run both demonstrations and write their figures, tables and source data.

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
    parser.add_argument("--output", type=Path, required=True, help="Directory for figures and tables.")
    arguments = parser.parse_args(argv)
    release = PublicRelease.from_arguments(arguments)
    table = decision_table(load_worlds(release))
    cells = table.groupby("role")["cell"].first()
    scale = release_scale(release.evidence(), str(cells["trap"]), str(cells["twin"]))
    trap = table.loc[table["role"].eq("trap") & table["analysis"].isin(("routine", "audit"))]
    save_figure(trap_figure(table, scale), trap, arguments.output, "trap_decision")
    save_figure(twin_figure(table, scale), table, arguments.output, "twin_overcorrection")
    pd.DataFrame([scale]).to_csv(arguments.output / "release_scale.csv", index=False, float_format="%.6g")
    for role in ("trap", "twin"):
        part = table.loc[table["role"].eq(role)]
        print(f"{role} cell {part['cell'].iloc[0]} ({part['tier'].iloc[0]}):")
        for _, row in part.iterrows():
            print(
                f"  world {row['world']} truth {_pp(row['true_risk_difference']):+.2f} pp | {LABELS[row['analysis']]:<21} "
                f"{_pp(row['estimate']):+.2f} [{_pp(row['lower']):+.2f}, {_pp(row['upper']):+.2f}] -> "
                f"{_decision_label(row):<13} regret ${row['regret_musd']:.1f}M{'  consequential' if row['consequential'] else ''}"
            )
    print(f"Wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
