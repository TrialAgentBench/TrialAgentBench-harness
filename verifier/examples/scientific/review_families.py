"""Check disclosed family records and compare illustrative point estimates.

This command uses ordinary data and survival libraries only. Complete outcomes
are deliberately disclosed diagnostics, not usable participant evidence. One
realisation cannot establish bias, coverage or a release-level ranking.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from numpy.typing import NDArray
from review_standards import read_tables, reconcile_tables
from scipy.stats import norm, t


def contrast(values: pd.Series, arms: pd.Series, *, ratio: bool) -> float:
    """Calculate an arm-mean contrast on the stated scale."""
    means = values.groupby(arms).mean()
    if set(means.index) != {"active", "control"} or not np.isfinite(means).all():
        raise ValueError("Both arms require finite outcome means")
    if ratio and means["control"] <= 0:
        raise ValueError("A rate ratio requires a positive comparator mean")
    return float(means["active"] / means["control"] if ratio else means["active"] - means["control"])


def reconstruct_survival(analysis: pd.DataFrame, events: pd.DataFrame, endpoint: str) -> None:
    """Recover the observed follow-up and event flag from source event records."""
    indexed = analysis.set_index("participant_id")
    event_time = (
        events.loc[(events.event_kind == "endpoint") & (events.event_id == endpoint)]
        .groupby("participant_id")
        .occurrence_time.min()
        .reindex(indexed.index)
        .fillna(np.inf)
    )
    censor_time = (
        events.loc[events.event_kind == "follow_up"]
        .groupby("participant_id")
        .occurrence_time.min()
        .reindex(indexed.index)
        .fillna(np.inf)
    )
    times = np.minimum(event_time, censor_time)
    if not np.isfinite(times).all():
        raise ValueError("Every participant needs an endpoint or end-of-follow-up record")
    np.testing.assert_allclose(times, indexed.observed_time, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(event_time <= censor_time, indexed.event_observed)


def reconstruct_validation(analysis: pd.DataFrame, events: pd.DataFrame, endpoint: str) -> None:
    """Recover routine and adjudicated binary endpoints from assessment records."""
    indexed = analysis.set_index("participant_id")
    for prefix, column in (("routine", "routine_endpoint"), ("adjudicated", "adjudicated_endpoint")):
        source = events.loc[events.event_id == f"{prefix}_{endpoint}_assessment"].set_index("participant_id")
        if source.index.duplicated().any():
            raise ValueError("An assessment record must be unique per participant")
        observed = source.event_by_contact.reindex(indexed.index).astype(float)
        np.testing.assert_allclose(observed, indexed[column].astype(float), equal_nan=True)
        if prefix == "adjudicated":
            np.testing.assert_array_equal(observed.notna(), indexed.validation_selected)


def check_cell(directory: Path, *, endpoint: str, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check reference arithmetic, participant membership and raw/ready reconstruction."""
    analysis = pd.read_parquet(directory / "analysis.parquet")
    participants = pd.read_parquet(directory / "participant_records.parquet")
    events = pd.read_parquet(directory / "event_history.parquet")
    complete = pd.read_parquet(directory / "disclosed_complete_outcomes.parquet")
    oracle = pd.read_csv(directory / "oracle_totals.csv").set_index("arm").sort_index()
    sums = complete.groupby("assigned_arm").complete_endpoint.agg(total="sum", n="size").sort_index()
    np.testing.assert_allclose(sums.total, oracle.total, rtol=1e-12, atol=1e-10)
    np.testing.assert_array_equal(sums.n, oracle.n)
    if participants.participant_id.duplicated().any():
        raise ValueError("Participant source identifiers must be unique")
    for source in (analysis, events):
        if not set(source.participant_id) <= set(participants.participant_id):
            raise ValueError("Source records contain unexposed participants")
    baseline_fields = [str(column) for column in participants if column in analysis and column != "participant_id"]
    aligned_baseline = participants.set_index("participant_id").loc[analysis.participant_id, baseline_fields]
    pd.testing.assert_frame_equal(
        aligned_baseline.reset_index(drop=True), analysis[baseline_fields].reset_index(drop=True), check_dtype=False
    )
    if family == "time_to_event":
        if "routine_endpoint" in analysis:
            reconstruct_validation(analysis, events, endpoint)
        else:
            reconstruct_survival(analysis, events, endpoint)
    else:
        observations = pd.read_parquet(directory / "endpoint_observations.parquet")
        key = ["participant_id", "occurrence_time"] if family == "longitudinal" else ["participant_id"]
        if observations.duplicated(key).any():
            raise ValueError("Endpoint source keys must be unique")
        rebuilt = observations.merge(participants, on="participant_id", validate="many_to_one")
        # Public observation-status projection removes engine bookkeeping fields.
        shared = sorted(str(column) for column in set(rebuilt) & set(analysis))
        pd.testing.assert_frame_equal(
            rebuilt.sort_values(key)[shared].reset_index(drop=True),
            analysis.sort_values(key)[shared].reset_index(drop=True),
            check_dtype=False,
        )
        if endpoint not in shared or "assigned_arm" not in shared:
            raise ValueError("Reconstruction must include endpoint and treatment")
    return analysis, complete


def paired_survival_targets(directory: Path, *, horizon: float) -> pd.DataFrame:
    """Reconstruct both finite-trial contrasts from deliberately disclosed histories."""
    complete = pd.read_parquet(directory / "disclosed_complete_outcomes.parquet")
    totals = pd.read_csv(directory / "paired_oracle_totals.csv")
    mapping = {
        "risk_difference": ("complete_event_by_horizon", "proportion"),
        "restricted_mean_survival_time_difference": ("complete_event_free_time", "day"),
    }
    if (
        not np.isfinite(horizon)
        or horizon <= 0
        or complete.participant_id.isna().any()
        or complete.participant_id.duplicated().any()
        or set(complete.assigned_arm) != {"active", "control"}
        or not complete.complete_event_by_horizon.isin([True, False]).all()
        or not complete.complete_event_free_time.between(0, horizon).all()
        or not complete.loc[~complete.complete_event_by_horizon.astype(bool), "complete_event_free_time"]
        .eq(horizon)
        .all()
        or set(totals.effect_measure) != set(mapping)
        or totals.target_id.nunique() != 2
        or totals.target_id.isna().any()
        or len(totals) != 4
    ):
        raise ValueError("Paired survival evidence requires two targets and complete participant histories")
    rows = []
    for measure, (column, unit) in mapping.items():
        expected = totals.loc[totals.effect_measure.eq(measure)].set_index("arm").sort_index()
        if (
            set(expected.index) != {"active", "control"}
            or expected.target_id.nunique() != 1
            or not expected.unit.eq(unit).all()
        ):
            raise ValueError("Paired target identity, unit or arm assignment disagrees")
        calculated = complete.groupby("assigned_arm")[column].agg(total="sum", n="size").sort_index()
        np.testing.assert_array_equal(calculated.n, expected.n)
        np.testing.assert_allclose(calculated.total, expected.total, rtol=1e-12, atol=1e-10)
        rows.append(
            {
                "target_id": expected.target_id.iloc[0],
                "effect_measure": measure,
                "unit": unit,
                "finite_complete_contrast": contrast(complete[column], complete.assigned_arm, ratio=False),
                "participants": len(complete),
            }
        )
    return pd.DataFrame(rows)


def paired_survival_estimates(data: pd.DataFrame, *, horizon: float) -> pd.DataFrame:
    """Estimate both marginal survival contrasts with Greenwood uncertainty.

    Parameters
    ----------
    data
        Observed times, event indicators and randomized arms.
    horizon
        Common analysis horizon, supported in both observed arms.

    Returns
    -------
    pandas.DataFrame
        KM and crude contrasts with marginal normal intervals. These are
        descriptive analyses, not simultaneous confidence intervals.

    Raises
    ------
    ValueError
        If arm membership, time support or Greenwood risk sets are invalid.
    """
    if set(data.assigned_arm) != {"active", "control"} or not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Paired survival analysis requires two arms and a positive horizon")
    arms = []
    for arm, frame in data.groupby("assigned_arm"):
        times_observed = frame.observed_time.to_numpy(dtype=float)
        if (
            not np.isfinite(times_observed).all()
            or (times_observed < 0).any()
            or not frame.event_observed.isin([0, 1]).all()
            or max(times_observed) < horizon
        ):
            raise ValueError("Paired survival analysis lacks valid observed horizon support")
        fit = KaplanMeierFitter().fit(frame.observed_time, frame.event_observed)
        curve = fit.survival_function_.loc[fit.survival_function_.index < horizon]
        areas = curve.iloc[:, 0].to_numpy() * np.diff(np.r_[curve.index.to_numpy(), horizon])
        tail_integrals = np.cumsum(areas[::-1])[::-1]
        events = fit.event_table.loc[(fit.event_table.index <= horizon) & fit.event_table.observed.gt(0)]
        if (events.at_risk <= events.observed).any():
            raise ValueError("Greenwood approximation requires positive risk sets after events")
        factors = events.observed / (events.at_risk * (events.at_risk - events.observed))
        before = events.index < horizon
        positions = np.searchsorted(curve.index.to_numpy(), events.index[before].to_numpy())
        survival = float(fit.predict(horizon))
        for measure, column, point, variance in (
            ("risk_difference", "event_observed", 1 - survival, survival**2 * float(factors.sum())),
            (
                "restricted_mean_survival_time_difference",
                "observed_time",
                float(areas.sum()),
                float(np.sum(factors.loc[before].to_numpy() * tail_integrals[positions] ** 2)),
            ),
        ):
            for method, estimate, sampling_variance in (
                ("Kaplan–Meier", point, variance),
                ("Crude observed records", float(frame[column].mean()), float(frame[column].var() / len(frame))),
            ):
                arms.append(
                    {
                        "arm": arm,
                        "effect_measure": measure,
                        "method": method,
                        "estimate": estimate,
                        "variance": sampling_variance,
                    }
                )
    rows = []
    for (group_measure, group_method), group in pd.DataFrame(arms).groupby(["effect_measure", "method"]):
        active = float(group.loc[group.arm.eq("active"), "estimate"].iloc[0])
        control = float(group.loc[group.arm.eq("control"), "estimate"].iloc[0])
        estimate = active - control
        se = float(np.sqrt(group.variance.sum()))
        rows.append(
            {
                "effect_measure": str(group_measure),
                "method": str(group_method),
                "estimate": estimate,
                "se": se,
                "lower": estimate - norm.ppf(0.975) * se,
                "upper": estimate + norm.ppf(0.975) * se,
            }
        )
    return pd.DataFrame(rows)


def paired_survival_law(directory: Path, *, horizon: float) -> pd.DataFrame:
    """Integrate the disclosed event law conditional on this realised population.

    This checks event generation, not population-reference precision or interval
    coverage. Calibration fixes the control risk across all disclosed covariates
    before applying the actual randomized arms. Censoring does not enter either
    complete-data functional.
    """
    from scipy.integrate import quad_vec
    from scipy.optimize import brentq

    law = pd.read_csv(directory / "survival_law.csv")
    covariates = pd.read_csv(directory / "survival_covariates.csv").set_index("participant_id")
    complete = pd.read_parquet(directory / "disclosed_complete_outcomes.parquet").set_index("participant_id")
    finite = paired_survival_targets(directory, horizon=horizon).set_index("effect_measure")
    if (
        not covariates.index.is_unique
        or covariates.index.hasnans
        or set(covariates.index) != set(complete.index)
        or not np.isfinite(covariates.log_hazard_multiplier).all()
        or law.empty
        or not np.isfinite(law.to_numpy(dtype=float)).all()
        or law[["shape", "scale", "reference_risk"]].nunique().ne(1).any()
        or law.start.iloc[0] != 0
        or law.end.iloc[-1] != horizon
        or not law.start.iloc[1:].reset_index(drop=True).equals(law.end.iloc[:-1].reset_index(drop=True))
        or not law.end.gt(law.start).all()
    ):
        raise ValueError("Conditional survival law has invalid population or period support")
    covariates = covariates.loc[complete.index]
    if not covariates.assigned_arm.equals(complete.assigned_arm):
        raise ValueError("Conditional survival law changes randomized arms")
    shape, scale, target = law.iloc[0][["shape", "scale", "reference_risk"]]
    if shape <= 0 or scale <= 0 or not 0 < target < 1:
        raise ValueError("Conditional survival parameters are outside their support")
    terms = pd.read_csv(directory / "survival_risk_terms.csv")
    if terms.empty or terms.variable.isna().any() or not np.isfinite(terms.coefficient).all():
        raise ValueError("Conditional survival law requires explicit finite risk terms")
    predictor = np.zeros(len(covariates), dtype=float)
    for term in terms.itertuples(index=False):
        predictor += covariates[term.variable].to_numpy(dtype=float) * term.coefficient
    np.testing.assert_allclose(predictor, covariates.log_hazard_multiplier, rtol=1e-12, atol=1e-12)
    baseline = np.exp(predictor)

    def cumulative(time: float, arm: str) -> NDArray[np.float64]:
        increments = np.maximum((np.minimum(time, law.end) / scale) ** shape - (law.start / scale) ** shape, 0)
        return np.asarray(baseline * float(np.sum(np.exp(law[f"{arm}_log_hazard"]) * increments)), dtype=np.float64)

    control_hazard = cumulative(horizon, "control")
    target_hazard = np.log(-np.log1p(-target))
    lower = target_hazard - np.log(control_hazard).max() - 1
    upper = target_hazard - np.log(control_hazard).min() + 1
    calibration = np.exp(
        brentq(lambda value: np.mean(-np.expm1(-np.exp(value) * control_hazard)) - target, lower, upper)
    )
    active = complete.assigned_arm.eq("active").to_numpy()

    def survival(time: float) -> NDArray[np.float64]:
        hazard = np.where(active, cumulative(time, "active"), cumulative(time, "control"))
        return np.asarray(np.exp(-calibration * hazard), dtype=np.float64)

    def integrand(time: float) -> NDArray[np.float64]:
        probability = survival(time)
        return np.asarray(2 * time * probability, dtype=np.float64)

    mean, error = quad_vec(survival, 0, horizon, points=law.end.iloc[:-1].tolist(), epsabs=1e-9, epsrel=1e-10)
    second, second_error = quad_vec(
        integrand,
        0,
        horizon,
        points=law.end.iloc[:-1].tolist(),
        epsabs=1e-9,
        epsrel=1e-10,
    )
    risk = 1 - survival(horizon)
    quantities = (
        ("risk_difference", risk, risk * (1 - risk)),
        ("restricted_mean_survival_time_difference", mean, second - mean**2),
    )
    rows = []
    for measure, expected, variance in quantities:
        if not np.isfinite(expected).all() or not np.isfinite(variance).all() or np.any(variance < 0):
            raise ValueError("Conditional survival moments are invalid")
        value = float(expected[active].mean() - expected[~active].mean())
        standard_error = float(
            np.sqrt(variance[active].sum() / active.sum() ** 2 + variance[~active].sum() / (~active).sum() ** 2)
        )
        realised = float(finite.loc[[measure], "finite_complete_contrast"].to_numpy(dtype=float)[0])
        rows.append(
            {
                "effect_measure": measure,
                "conditional_expected_contrast": value,
                "conditional_sampling_se": standard_error,
                "conditional_normal_lower": value - 1.96 * standard_error,
                "conditional_normal_upper": value + 1.96 * standard_error,
                "finite_complete_contrast": realised,
                "standardized_deviation": (realised - value) / standard_error,
                "mean_integration_error_estimate": float(error) if measure != "risk_difference" else 0.0,
                "second_moment_integration_error_estimate": (
                    float(second_error) if measure != "risk_difference" else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_paired_survival_law(source: Path, output: Path) -> None:
    """Plot conditional expectations and complete-trial contrasts from exact CSV values."""
    from matplotlib.lines import Line2D

    data = pd.read_csv(source)
    labels = {
        "TE-S01-A1": "Mortality\nProportional hazards",
        "TE-S01-A2": "Mortality\nVarying treatment effect",
        "TE-S02-A1": "Kidney failure or death\nIndependent censoring",
        "TE-S02-A2": "Kidney failure or death\nPrognosis-related censoring",
    }
    cells = sorted(data.cell.unique())
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
    panels = (
        ("risk_difference", "Risk difference\n(percentage points)", 100),
        ("restricted_mean_survival_time_difference", "Restricted mean time difference\n(days)", 1),
    )
    for axis, (measure, label, factor) in zip(axes, panels, strict=True):
        selected = data.loc[data.effect_measure.eq(measure)].set_index("cell").loc[cells]
        for position, (lower, upper, expected, realised) in enumerate(
            selected[
                [
                    "conditional_normal_lower",
                    "conditional_normal_upper",
                    "conditional_expected_contrast",
                    "finite_complete_contrast",
                ]
            ].to_numpy(dtype=float)
        ):
            axis.plot(
                factor * np.array([lower, upper]),
                [position, position],
                color="#777777",
                linewidth=1.5,
            )
            axis.scatter(factor * expected, position, marker="|", s=100, color="#333333", zorder=3)
            axis.scatter(factor * realised, position, marker="o", s=35, color="#24788a", zorder=4)
        axis.axvline(0, color="#dddddd", linewidth=0.8)
        axis.set_xlabel(label, fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(cells)), [labels[cell] for cell in cells], fontsize=9)
    axes[0].invert_yaxis()
    figure.suptitle("Expected effects and one realised trial", fontsize=12)
    figure.legend(
        [Line2D([], [], color="#777777", marker="|"), Line2D([], [], color="#24788a", marker="o", linestyle="none")],
        ["Conditional expectation ± 1.96 sampling SD", "Realised complete-data contrast"],
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.17, 1, 0.93))
    for extension in ("pdf", "svg", "png"):
        with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            figure.savefig(output / f"paired_conditional_law.{extension}", dpi=200)
    plt.close(figure)
    (output / "paired_conditional_law.md").write_text(
        "# Expected effects and one realised trial\n\n"
        "Risk differences and restricted mean time differences describe the same complete event histories. "
        "Grey ticks are expectations calculated from the disclosed event law with SciPy, conditional on the "
        "realised covariates and treatment assignments. Bars span ±1.96 conditional sampling standard deviations; "
        "circles show the complete-data contrasts in one simulated trial. Mortality uses a 730-day horizon and "
        "kidney failure or death uses 1,095 days. All contrasts are active treatment minus control.\n\n"
        "The model calibrates the baseline hazard to its specified control risk across the realised covariate "
        "population. Risk is one minus survival at the horizon; restricted mean time is the integral of survival. "
        "The first two moments give the conditional sampling variance. Within each clinical family, conditions "
        "share seeds; the two targets and the paired conditions are dependent. Censoring changes observed evidence "
        "while leaving the complete event law unchanged in the kidney example.\n\n"
        "These are conditional generation checks. The normal ranges are illustrative; repeated-trial coverage "
        "and population-reference Monte Carlo precision require separate evidence.\n\n"
        "Alternative text: Two horizontal panels compare expected and realised risk and restricted mean time "
        "differences for four survival cases, with grey conditional sampling ranges and teal realised values.\n",
        encoding="utf-8",
    )


def point_estimates(
    data: pd.DataFrame,
    *,
    endpoint: str,
    family: str,
    scale: str,
    horizon: float,
    cell: str,
    covariates: tuple[str, ...],
) -> tuple[str, float, str, float]:
    """Compare transparent illustrative analyses without recommending a graded method."""
    ratio = scale == "rate_ratio"
    if "validation_selected" in data:
        routine = "routine_endpoint" if "routine_endpoint" in data else endpoint
        naive = contrast(data[routine], data.assigned_arm, ratio=ratio)
        cell_means = data.groupby(["assigned_arm", "validation_stratum"]).adjudicated_endpoint.mean()
        weights = data.groupby(["assigned_arm", "validation_stratum"]).size()
        if cell_means.isna().any():
            raise ValueError("Every validation cell needs adjudicated observations")
        means = (cell_means * weights).groupby(level=0).sum() / weights.groupby(level=0).sum()
        estimate = float(means["active"] / means["control"] if ratio else means["active"] - means["control"])
        return "stratum-weighted adjudication", estimate, "routine endpoint", naive
    if family == "time_to_event":
        sample = data.loc[data.bmi_kg_m2 >= 35] if cell.startswith("TE-S05") else data
        survival_means: dict[str, float] = {}
        for arm, frame in sample.groupby("assigned_arm"):
            fit = KaplanMeierFitter().fit(frame.observed_time, frame.event_observed)
            if frame.observed_time.max() < horizon:
                raise ValueError("The requested survival horizon lacks observed support")
            survival_means[str(arm)] = (
                float(restricted_mean_survival_time(fit, t=horizon))
                if scale == "restricted_mean_survival_time_difference"
                else 1 - float(fit.predict(horizon))
            )
        estimate = survival_means["active"] - survival_means["control"]
        naive = contrast(
            data.observed_time if scale == "restricted_mean_survival_time_difference" else data.event_observed,
            data.assigned_arm,
            ratio=False,
        )
        label = "KM RMST" if scale == "restricted_mean_survival_time_difference" else "KM horizon risk"
        if cell.startswith("TE-S05"):
            label += "; target subgroup"
        return label, estimate, "mean observed time / crude events; all participants", naive
    if family == "longitudinal":
        terminal = data.loc[np.isclose(data.occurrence_time, horizon)]
        if terminal.participant_id.duplicated().any() or terminal.empty:
            raise ValueError("The requested final endpoint needs unique participant records")
        return (
            "observed final-window arm means",
            contrast(terminal[endpoint], terminal.assigned_arm, ratio=ratio),
            "pool all observed visits",
            contrast(data[endpoint], data.assigned_arm, ratio=ratio),
        )
    observed = data.dropna(subset=[endpoint])
    naive = contrast(observed[endpoint], observed.assigned_arm, ratio=ratio)
    # OLS is an illustrative precision adjustment, not a prescribed answer.
    matrix = np.column_stack(
        [
            np.ones(len(observed)),
            (observed.assigned_arm == "active").astype(float),
            observed.loc[:, list(covariates)].to_numpy(dtype=float),
        ]
    )
    coefficients = np.linalg.lstsq(matrix, observed[endpoint].to_numpy(dtype=float), rcond=None)[0]
    active = matrix.copy()
    control = matrix.copy()
    active[:, 1], control[:, 1] = 1, 0
    active_mean, control_mean = float(np.mean(active @ coefficients)), float(np.mean(control @ coefficients))
    if ratio and control_mean <= 0:
        raise ValueError("Illustrative adjusted comparator mean cannot support a rate ratio")
    adjusted = active_mean / control_mean if ratio else active_mean - control_mean
    return "OLS baseline-standardized arm means", adjusted, "observed arm means", naive


def recovery_estimate(data: pd.DataFrame, endpoint: str, probability: float) -> tuple[float, float]:
    """Estimate an arm contrast using a randomized retrieval sampling probability."""
    if not 0 < probability <= 1:
        raise ValueError("Endpoint recovery requires its declared sampling probability")
    recovered = data.endpoint_recovered.astype(bool)
    routine = data.routine_observed.astype(bool) if "routine_observed" in data else data[endpoint].notna() & ~recovered
    weights = routine.astype(float) + recovered.astype(float) / probability
    values = data[endpoint].fillna(0).astype(float) * weights
    means = values.groupby(data.assigned_arm).mean()
    variances = values.groupby(data.assigned_arm).var() / data.groupby("assigned_arm").size()
    return float(means["active"] - means["control"]), float(np.sqrt(variances.sum()))


def horizon_risk(data: pd.DataFrame, horizon: float) -> tuple[float, float]:
    """Return the KM risk difference and its independent-arm Greenwood SE."""
    estimates = paired_survival_estimates(data, horizon=horizon)
    risk = estimates.loc[estimates.effect_measure.eq("risk_difference") & estimates.method.eq("Kaplan–Meier")].iloc[0]
    return float(risk.estimate), float(risk.se)


def cluster_risk_deletions(data: pd.DataFrame, horizon: float) -> NDArray[np.float64]:
    """Calculate exact delete-cluster KM contrasts from common event-time counts.

    Sorting each arm once avoids fitting hundreds of identical model objects.
    The calculation retains censoring ties and the full cluster deletion.
    """
    clusters = data.cluster_id.unique()
    differences = np.zeros(len(clusters))
    for arm, sign in (("active", 1), ("control", -1)):
        frame = data.loc[data.assigned_arm.eq(arm)]
        times, positions = np.unique(frame.observed_time, return_inverse=True)
        removals = np.bincount(positions, minlength=len(times))
        deaths = np.bincount(positions, weights=frame.event_observed, minlength=len(times))
        at_risk = len(frame) - np.r_[0, np.cumsum(removals)[:-1]]
        for index, cluster in enumerate(clusters):
            selected = frame.cluster_id.eq(cluster).to_numpy()
            removed = np.bincount(positions[selected], minlength=len(times))
            removed_deaths = np.bincount(
                positions[selected], weights=frame.event_observed.to_numpy()[selected], minlength=len(times)
            )
            risk = at_risk - (selected.sum() - np.r_[0, np.cumsum(removed)[:-1]])
            events = deaths - removed_deaths
            included = (times <= horizon) & (events > 0)
            if np.any(risk[included] <= events[included]):
                raise ValueError("A cluster deletion exhausts the horizon risk set")
            survival = np.exp(np.log1p(-events[included] / risk[included]).sum())
            differences[index] += sign * (1 - survival)
    return differences


def design_controls(source: Path, cells: pd.DataFrame) -> pd.DataFrame:
    """Expose design-sensitive intervals without claiming repeated-world coverage."""
    rows = []
    for cell in cells.to_dict(orient="records"):
        identifier, horizon = str(cell["cell"]), float(cell["horizon"])
        data = pd.read_parquet(source / identifier / "analysis.parquet")
        comparisons: list[tuple[str, float, float, float]] = []
        if float(cell["recovery_probability"]) > 0:
            endpoint = "horizon_endpoint" if "horizon_endpoint" in data else str(cell["endpoint"])
            terminal = data.loc[np.isclose(data.occurrence_time, horizon)] if "occurrence_time" in data else data
            estimate, se = recovery_estimate(terminal, endpoint, float(cell["recovery_probability"]))
            comparisons.append(("randomized retrieval HT; normal approximation", estimate, se, 1.959963984540054))
        if identifier.startswith(("TE-S07", "TE-S08", "TE-S09")):
            estimate, se = horizon_risk(data, horizon)
            comparisons.append(("unadjusted KM; independent-person interval", estimate, se, 1.959963984540054))
        if identifier.startswith("TE-S07"):
            clusters = data.cluster_id.unique()
            leave_out = cluster_risk_deletions(data, horizon)
            jackknife = float(
                np.sqrt((len(clusters) - 1) / len(clusters) * np.square(leave_out - leave_out.mean()).sum())
            )
            comparisons.append(
                ("KM; delete-cluster jackknife", estimate, jackknife, float(t.ppf(0.975, len(clusters) - 2)))
            )
        if identifier.startswith("TE-S08"):
            # IPCW event pseudo-outcomes target horizon risk. Calendar adjustment
            # addresses the rollout's changing treatment proportions. Conditional
            # censoring models remain an assumption, not guaranteed by this fit.
            data = data.copy()
            data["pseudo_outcome"] = 0.0
            for _, frame in data.groupby(["assigned_arm", "calendar_period"]):
                censoring = KaplanMeierFitter().fit(frame.observed_time, ~frame.event_observed)
                retention = np.asarray(censoring.predict(np.nextafter(frame.observed_time.to_numpy(), -np.inf)))
                if np.any(retention <= 0):
                    raise ValueError("Horizon-risk weighting lacks positive censoring support")
                data.loc[frame.index, "pseudo_outcome"] = frame.event_observed.to_numpy(dtype=float) / retention
            data["active"] = data.assigned_arm.eq("active").astype(float)
            formula = "pseudo_outcome ~ active + C(calendar_period) + " + " + ".join(
                str(cell["analysis_covariates"]).split(";")
            )
            fit = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data.cluster_id}, use_t=True)
            comparisons.append(
                (
                    "IPCW risk; period/baseline adjustment, cluster SE",
                    float(fit.params["active"]),
                    float(fit.bse["active"]),
                    float(t.ppf(0.975, data.cluster_id.nunique() - 1)),
                )
            )
        if identifier.startswith("TE-S09"):
            looks = int(cell["planned_looks"])
            if looks < len(pd.read_parquet(source / identifier / "monitoring.parquet")):
                raise ValueError("Observed monitoring exceeds the declared number of looks")
            comparisons.append(
                (
                    "KM; Bonferroni across all planned looks, asymptotic",
                    estimate,
                    se,
                    float(norm.ppf(1 - 0.05 / (2 * looks))),
                )
            )
        for method, estimate, se, critical in comparisons:
            rows.append(
                {
                    "cell": identifier,
                    "analysis": method,
                    "estimate": estimate,
                    "standard_error": se,
                    "lower": estimate - critical * se,
                    "upper": estimate + critical * se,
                }
            )
    return pd.DataFrame(rows)


def baseline_checks(source: Path, cells: pd.DataFrame, output: Path) -> None:
    """Describe natural-unit marginals and joint structure without sample-fit gates."""
    marginals, joint, clocks, categories = [], [], [], []
    for identifier in cells.cell:
        directory = source / identifier
        baseline = pd.read_parquet(directory / "clinical_baseline.parquet")
        if "years_since_multiple_sclerosis_diagnosis" in baseline:
            if baseline.years_since_multiple_sclerosis_diagnosis.gt(baseline.age_years).any():
                raise ValueError("Diagnosis duration cannot exceed recorded age")
        if "parous" in baseline:
            nulliparous = baseline.parous.eq("nulliparous")
            if (
                baseline.loc[nulliparous, "previous_caesarean"].ne("no").any()
                or baseline.loc[nulliparous, "previous_postpartum_haemorrhage"].ne("no").any()
            ):
                raise ValueError("Nulliparous participants cannot have a previous delivery history")
        law = pd.read_csv(directory / "baseline_law.csv")
        for specification in law.to_dict(orient="records"):
            values = baseline[str(specification["field"])].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Declared baseline measurements must be finite")
            lower, upper = float(specification["lower"]), float(specification["upper"])
            if (np.isfinite(lower) and np.min(values) < lower) or (np.isfinite(upper) and np.max(values) > upper):
                raise ValueError("Clinical baseline observations violate declared support")
            marginals.append(
                dict(
                    {str(key): value for key, value in specification.items()},
                    cell=identifier,
                    participants=len(values),
                    observed_mean=float(np.mean(values)),
                    observed_sd=float(np.std(values, ddof=1)),
                    observed_minimum=float(np.min(values)),
                    observed_maximum=float(np.max(values)),
                )
            )
        codes = {}
        category_law = pd.read_csv(directory / "baseline_categories.csv")
        for field, rows in category_law.groupby("field", sort=False):
            category_values = baseline[field]
            if category_values.isna().any() or not category_values.isin(rows.category).all():
                raise ValueError("Baseline contains an undeclared or missing category")
            codes[field] = dict(zip(rows.category, rows.source_code, strict=True))
            for row in rows.itertuples(index=False):
                categories.append(
                    {
                        "cell": identifier,
                        "field": field,
                        "label": row.label,
                        "category": row.category,
                        "participants": len(category_values),
                        "requested_probability_before_recording": row.requested_probability_before_recording,
                        "observed_probability": float(category_values.eq(row.category).mean()),
                    }
                )
        for specification in pd.read_csv(directory / "baseline_dependence.csv").to_dict(orient="records"):
            a, b = str(specification["field_a"]), str(specification["field_b"])
            left = baseline[a].map(codes[a]) if a in codes else baseline[a]
            right = baseline[b].map(codes[b]) if b in codes else baseline[b]
            constant = left.nunique() < 2 or right.nunique() < 2
            continuous = {"normal", "truncated_normal", "uniform", "lognormal", "gamma", "beta", "student_t"}
            laws = law.set_index("field").law
            target = specification["requested_spearman_before_recording"]
            converted = (
                specification["requested_metric"] == "latent_gaussian"
                and a in laws
                and b in laws
                and laws[a] in continuous
                and laws[b] in continuous
            )
            if converted:
                # Gaussian copula identity for continuous margins, before recording ties.
                target = 6 / np.pi * np.arcsin(float(specification["requested_correlation"]) / 2)
            joint.append(
                dict(
                    {str(key): value for key, value in specification.items()},
                    cell=identifier,
                    target_spearman_before_recording=target,
                    comparison_status=(
                        "constant_margin"
                        if constant
                        else (
                            "before_recording_reference"
                            if specification["requested_metric"] == "spearman" or converted
                            else "different_scales"
                        )
                    ),
                    observed_spearman=None if constant else float(left.corr(right, method="spearman")),
                )
            )
        events = pd.read_parquet(directory / "event_history.parquet")
        recorded = events[["occurrence_time", "observation_time", "documentation_time", "availability_time"]].dropna()
        if (np.diff(recorded.to_numpy(dtype=float), axis=1) < -1e-10).any():
            raise ValueError("Clean event records violate occurrence-to-availability chronology")
        clocks.append({"cell": identifier, "event_records": len(events), "complete_chronology_records": len(recorded)})
    pd.DataFrame(marginals).to_csv(output / "clinical_baseline_marginals.csv", index=False)
    pd.DataFrame(joint).to_csv(output / "clinical_baseline_dependence.csv", index=False)
    pd.DataFrame(clocks).to_csv(output / "record_chronology.csv", index=False)
    pd.DataFrame(categories).to_csv(output / "clinical_baseline_categories.csv", index=False)


def asthma_trial(source: Path, output: Path) -> None:
    """Report the disclosed asthma trial from participant-visible source records."""
    directory = source / "TE-S04-A4"
    baseline = pd.read_parquet(directory / "clinical_baseline.parquet")
    events = pd.read_parquet(directory / "event_history.parquet")
    endpoint = "severe_asthma_exacerbation"
    indexed = baseline.set_index("participant_id")
    primary = events.loc[events.event_id.eq(endpoint)].set_index("participant_id")
    followup = events.loc[events.event_kind.eq("follow_up")].set_index("participant_id")
    recovery = events.loc[events.event_id.eq(f"recovered_{endpoint}_assessment")].set_index("participant_id")
    if primary.index.has_duplicates or followup.index.has_duplicates or recovery.index.has_duplicates:
        raise ValueError("The worked trial requires one terminal record of each kind per patient")
    event_time = primary.occurrence_time.reindex(indexed.index).fillna(np.inf)
    censor_time = followup.occurrence_time.reindex(indexed.index).fillna(np.inf)
    indexed["observed_time"] = np.minimum(event_time, censor_time)
    indexed["event_observed"] = event_time.le(censor_time)
    indexed["routine_observed"] = indexed.event_observed | indexed.observed_time.ge(365)
    indexed["endpoint_recovered"] = indexed.index.isin(recovery.index)
    indexed["horizon_endpoint"] = indexed.event_observed.astype(float).where(indexed.routine_observed)
    indexed.loc[recovery.index, "horizon_endpoint"] = recovery.event_by_contact.astype(float)
    if not np.isfinite(indexed.observed_time).all():
        raise ValueError("Every randomized patient needs an event or end-of-follow-up record")
    ready = pd.read_parquet(directory / "analysis.parquet").set_index("participant_id")
    for field in ("observed_time", "event_observed", "routine_observed", "endpoint_recovered", "horizon_endpoint"):
        np.testing.assert_allclose(indexed[field], ready.loc[indexed.index, field], equal_nan=True)
    indexed.reset_index().to_csv(output / "asthma_reconstructed.csv", index=False)
    planning = json.loads((directory / "information_plan.json").read_text())
    requirements = planning["requirements"]
    if len(requirements) != 1 or requirements[0]["requirement_type"] != "derived_minimum_subjects":
        raise ValueError("The worked planning example requires one event-information requirement")
    request = requirements[0]["design_request"]
    if request["design_type"] != "survival":
        raise ValueError("The worked planning example requires survival information")
    allocation = request["treatment_allocation"]
    required_events = (norm.ppf(1 - request["two_sided_alpha"] / 2) + norm.ppf(request["power"])) ** 2 / (
        allocation * (1 - allocation) * np.log(request["hazard_ratio"]) ** 2
    )
    block_size = planning["allocation_multiple"]
    selected_n = int(np.ceil(np.ceil(required_events) / request["event_probability"] / block_size) * block_size)
    if selected_n != planning["minimum_subjects"] or planning["selected_subjects"] != len(baseline):
        raise ValueError("The independent planning calculation differs from the supplied trial size")
    if len(baseline) < selected_n:
        raise ValueError("The supplied trial is smaller than its planned minimum")
    pd.DataFrame(
        [{"required_events": required_events, "event_yield": request["event_probability"], "planned_n": selected_n}]
    ).to_csv(output / "asthma_sample_size.csv", index=False)
    flow, table1, estimates = [], [], []
    for arm, frame in indexed.groupby("assigned_arm"):
        flow.append(
            {
                "arm": arm,
                "randomized": len(frame),
                "routine_status": int(frame.routine_observed.sum()),
                "routine_events": int(frame.event_observed.sum()),
                "status_missing_routinely": int((~frame.routine_observed).sum()),
                "status_retrieved": int(frame.endpoint_recovered.sum()),
                "still_missing": int(frame.horizon_endpoint.isna().sum()),
                "safety_events": events.loc[
                    events.event_kind.eq("safety_event") & events.participant_id.isin(frame.index), "participant_id"
                ].nunique(),
            }
        )
        for field in ("age_years", "severe_exacerbations_previous_year", "baseline_fev1_percent_predicted"):
            table1.append(
                {
                    "arm": arm,
                    "characteristic": field,
                    "n": len(frame),
                    "mean": frame[field].mean(),
                    "sd": frame[field].std(),
                }
            )
    flow_frame = pd.DataFrame(flow).set_index("arm")
    flow_frame.to_csv(output / "asthma_flow.csv")
    pd.DataFrame(table1).to_csv(output / "asthma_table1.csv", index=False)
    # The first comparator incorrectly treats unobserved endpoints as absent.
    means = indexed.groupby("assigned_arm").event_observed.mean()
    crude = float(means["active"] - means["control"])
    crude_se = float(
        np.sqrt(sum(means[a] * (1 - means[a]) / len(indexed.loc[indexed.assigned_arm.eq(a)]) for a in means.index))
    )
    question = pd.read_csv(source / "cells.csv").set_index("cell").loc[["TE-S04-A4"]]
    question[["question", "endpoint", "horizon", "recovery_probability"]].to_csv(
        output / "asthma_analysis_question.csv", index=False
    )
    km, km_se = horizon_risk(indexed, float(question.horizon.iloc[0]))
    probability = float(question.recovery_probability.iloc[0])
    recovered, recovered_se = recovery_estimate(indexed, "horizon_endpoint", probability)
    for label, point, se in (
        ("Count recorded events", crude, crude_se),
        ("Kaplan–Meier", km, km_se),
        ("Randomized status retrieval", recovered, recovered_se),
    ):
        estimates.append(
            {
                "analysis": label,
                "estimate": point,
                "lower": point - norm.ppf(0.975) * se,
                "upper": point + norm.ppf(0.975) * se,
            }
        )
    results = pd.DataFrame(estimates)
    results.to_csv(output / "asthma_results.csv", index=False)
    # Complete outcomes enter only after every observed-data estimate is fixed.
    complete = pd.read_parquet(directory / "disclosed_complete_outcomes.parquet")
    reference = contrast(complete.complete_endpoint, complete.assigned_arm, ratio=False)
    pd.DataFrame([{"quantity": "complete allocated-arm contrast", "estimate": reference}]).to_csv(
        output / "asthma_disclosed_contrast.csv", index=False
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained", gridspec_kw={"width_ratios": [1, 1.35]})
    colors = ("#466785", "#168A8B", "#D7DDE2")
    left = np.zeros(2)
    for field, label, color in zip(
        ("routine_status", "status_retrieved", "still_missing"),
        ("Routine follow-up", "Retrieved status", "Still unavailable"),
        colors,
        strict=True,
    ):
        values = flow_frame.loc[["control", "active"], field].to_numpy()
        axes[0].barh([0, 1], values, left=left, color=color, label=label, edgecolor="white")
        for y, width, start in zip([0, 1], values, left, strict=True):
            axes[0].text(
                start + width / 2,
                y + ({"routine_status": 0, "status_retrieved": 0.17, "still_missing": -0.17}[field]),
                str(width),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if field == "routine_status" else "#142A35",
            )
        left += values
    axes[0].set(
        yticks=[0, 1],
        yticklabels=["Placebo", "Add-on biologic"],
        xlabel="Randomized participants",
        title="A  Follow-up available",
    )
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.22), frameon=False, fontsize=8)
    for i, row in results.iterrows():
        axes[1].errorbar(
            100 * row.estimate,
            i,
            xerr=[[100 * (row.estimate - row.lower)], [100 * (row.upper - row.estimate)]],
            color="#168A8B" if i == 2 else "#466785",
            marker="o",
            capsize=3,
        )
    axes[1].axvline(0, color="#777777", linestyle=":", linewidth=0.8)
    axes[1].axvline(
        100 * reference, color="#B85B36", linestyle="--", linewidth=1, label="Complete allocated-arm contrast"
    )
    axes[1].set(
        yticks=range(3),
        yticklabels=["Count recorded\nevents", "Kaplan–Meier", "Randomized status\nretrieval"],
        xlabel="Day-365 risk difference (percentage points)",
        title="B  Estimated treatment effect",
    )
    axes[1].invert_yaxis()
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -0.22), frameon=False, fontsize=8)
    for ax in axes:
        ax.title.set_fontsize(10)
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_fontsize(9)
        ax.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "asthma_trial.png", dpi=200)
    figure.savefig(output / "asthma_trial.pdf")
    plt.close(figure)
    lines = [
        "# A randomized asthma trial",
        "",
        "## Question and design",
        "",
        f"Does adding a biologic to optimized inhaled treatment reduce the risk of a severe asthma exacerbation by Day 365? This simulated trial randomized {len(indexed):,} adults 1:1 to add-on biologic or placebo, in blocks of {block_size}. The question concerns assignment, including outcomes after treatment discontinuation.",
        "",
        f"The cohort size comes from a prespecified nominal proportional-hazards planning alternative: hazard ratio {request['hazard_ratio']:.4f}, two-sided alpha {request['two_sided_alpha']:g}, power {request['power']:g}, equal allocation, and event yield {request['event_probability']:.4f}. Schoenfeld's event formula gives {required_events:.1f} events, rounded up to {int(np.ceil(required_events))}; division by event yield and rounding to blocks of {block_size} gives {selected_n:,} patients. This plans the reference scenario, not achieved power under this example's losses and treatment discontinuation.",
        "",
        "## Table 1. Participants at randomization",
        "",
        "| Characteristic, mean (SD) | Placebo | Add-on biologic |",
        "| --- | ---: | ---: |",
    ]
    for field, label in (
        ("age_years", "Age, years"),
        ("severe_exacerbations_previous_year", "Severe exacerbations in the previous year"),
        ("baseline_fev1_percent_predicted", "FEV1, % predicted"),
    ):
        groups = [indexed.loc[indexed.assigned_arm.eq(arm), field] for arm in ("control", "active")]
        lines.append(
            f"| {label} | {groups[0].mean():.1f} ({groups[0].std():.1f}) | {groups[1].mean():.1f} ({groups[1].std():.1f}) |"
        )
    lines += [
        "",
        "## Follow-up and reconstruction",
        "",
        "| Participants | Placebo | Add-on biologic |",
        "| --- | ---: | ---: |",
    ]
    for field, label in (
        ("randomized", "Randomized"),
        ("routine_events", "Exacerbation recorded before loss/end of follow-up"),
        ("routine_status", "Day-365 status known from routine follow-up"),
        ("status_missing_routinely", "Status initially unavailable"),
        ("status_retrieved", "Status retrieved in the randomized follow-up sample"),
        ("still_missing", "Status still unavailable"),
        ("safety_events", "Recorded systemic hypersensitivity"),
    ):
        lines.append(f"| {label} | {flow_frame.loc['control', field]} | {flow_frame.loc['active', field]} |")
    lines += [
        "",
        "Patient identifiers link assignment and baseline records to events, follow-up endings and retrieved assessments. Retrieval establishes whether an exacerbation occurred by Day 365; it does not reconstruct its exact date. The script rebuilds these fields from source events and checks them against the supplied analysis table.",
        "",
        "## Analysis and clinical conclusion",
        "",
        "| Analysis | Risk difference, percentage points (nominal 95% interval) |",
        "| --- | ---: |",
    ]
    for _, result in results.iterrows():
        lines.append(
            f"| {result.analysis} | {100*result.estimate:.2f} ({100*result.lower:.2f}, {100*result.upper:.2f}) |"
        )
    lines += [
        "",
        "Counting only recorded events treats missing follow-up as absence of disease. Kaplan–Meier uses time at risk but depends on a censoring assumption. The protocol instead supplies an independently sampled retrieval of missing statuses, with probability 0.5: routine known outcomes receive weight one, retrieved outcomes weight two, and every randomized participant remains in the denominator. This estimates the full-cohort risk without assuming that routine losses have the same outcomes as retained patients.",
        "",
        f"The retrieval estimate is {100 * recovered:.2f} percentage points "
        f"(nominal 95% interval {100 * (recovered - norm.ppf(0.975) * recovered_se):.2f} to "
        f"{100 * (recovered + norm.ppf(0.975) * recovered_se):.2f}). "
        "Recorded hypersensitivity counts describe observed safety and do not correct for incomplete safety ascertainment.",
        "",
        "![Follow-up availability and analysis results.](asthma_trial.png)",
        "",
        "## Comparison with complete outcomes",
        "",
        f"Complete outcomes in the teaching dataset give an allocated-arm risk difference of {100*reference:.2f} percentage points. They are used only after the observed-data analyses, to inspect the consequence of incomplete observation. This comparison concerns the patients allocated in this trial. Population-effect recovery across repeated trials is shown in the accompanying recovery figure.",
        "",
        "Intervals here use Greenwood variance for Kaplan–Meier and independent-arm normal approximations for the crude and retrieval analyses. Retrieval uses the sample variance of weighted contributions.",
    ]
    (output / "asthma_trial.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def review(source: Path, output: Path) -> None:
    """Write finite-world diagnostics, preserving target-specific limitations."""
    output.mkdir(parents=True, exist_ok=False)
    cells = pd.read_csv(source / "cells.csv")
    baseline_checks(source, cells, output)
    if "TE-S04-A4" in set(cells.cell):
        asthma_trial(source, output)
    rows = []
    standards_rows = []
    paired_rows = []
    law_rows = []
    paired_estimates = []
    population_totals = []
    for cell in cells.to_dict(orient="records"):
        identifier = str(cell["cell"])
        data, complete = check_cell(source / identifier, endpoint=str(cell["endpoint"]), family=str(cell["family"]))
        if (source / identifier / "paired_oracle_totals.csv").exists():
            paired_rows.append(
                paired_survival_targets(source / identifier, horizon=float(cell["horizon"])).assign(cell=identifier)
            )
        if (source / identifier / "population_reference_totals.csv").exists():
            paired_estimates.append(
                paired_survival_estimates(data, horizon=float(cell["horizon"])).assign(cell=identifier)
            )
            population_totals.append(
                pd.read_csv(source / identifier / "population_reference_totals.csv").assign(cell=identifier)
            )
            horizon = float(cell["horizon"])
            times = np.unique(np.r_[0.0, data.loc[data.observed_time.le(horizon), "observed_time"], horizon])
            curves = []
            for arm, records in data.groupby("assigned_arm"):
                fit = KaplanMeierFitter().fit(records.observed_time, records.event_observed)
                curves.append(pd.DataFrame({"arm": arm, "time_day": times, "survival": fit.predict(times).to_numpy()}))
            pd.concat(curves, ignore_index=True).to_csv(output / "paired_survival_curves.csv", index=False)
        if (source / identifier / "survival_law.csv").exists():
            law_rows.append(
                paired_survival_law(source / identifier, horizon=float(cell["horizon"])).assign(cell=identifier)
            )
        participant_records = pd.read_parquet(source / identifier / "participant_records.parquet")
        events = pd.read_parquet(source / identifier / "event_history.parquet")
        safety = events.loc[
            events.event_type.eq("safety") & events.event_at_contact.eq(True) & events.reported.eq(True)
        ]
        exposure_path = source / identifier / "exposure_records.parquet"
        exposure = pd.read_parquet(exposure_path) if exposure_path.exists() else None
        checked = reconcile_tables(
            participant_records,
            data,
            source / identifier / "standards.zip",
            safety=safety,
            intercurrent=events.loc[events.event_type.eq("intercurrent") & events.event_at_contact.eq(True)],
            exposure=exposure,
            endpoint_event=None if pd.isna(cell.get("endpoint_event")) else cell.get("endpoint_event"),
        )
        standards_rows.append(checked.assign(cell=identifier))
        if identifier == "TE-S04-A4":
            exported = read_tables(source / identifier / "standards.zip")
            tte = exported["ADTTE.parquet"].sort_values("SRCROW")
            pd.DataFrame(
                {"assigned_arm": tte.TRTP, "observed_time": tte.AVAL, "event_observed": tte.CNSR.eq(0)}
            ).to_csv(output / "asthma_standard_analysis.csv", index=False)
        reference = contrast(complete.complete_endpoint, complete.assigned_arm, ratio=cell["scale"] == "rate_ratio")
        method, estimate, shortcut, naive = point_estimates(
            data,
            endpoint=str(cell["endpoint"]),
            family=str(cell["family"]),
            scale=str(cell["scale"]),
            horizon=float(cell["horizon"]),
            cell=identifier,
            covariates=tuple(str(cell["analysis_covariates"]).split(";")),
        )
        rows.append(
            {
                "cell": identifier,
                "family": cell["family"],
                "visible_n": cell["visible_n"],
                "planned_n": cell["planned_n"],
                "scale": cell["scale"],
                "finite_complete_contrast": reference,
                "analysis": method,
                "estimate": estimate,
                "shortcut": shortcut,
                "shortcut_estimate": naive,
                "absolute_discrepancy": abs(estimate - reference),
                "shortcut_absolute_discrepancy": abs(naive - reference),
                "reference_arithmetic_and_raw_reconstruction": "passed",
            }
        )
    if paired_estimates:
        pd.concat(paired_estimates, ignore_index=True).to_csv(output / "paired_observed_estimates.csv", index=False)
        pd.concat(population_totals, ignore_index=True).to_csv(output / "paired_population_totals.csv", index=False)
    if law_rows:
        pd.concat(law_rows, ignore_index=True).to_csv(output / "paired_conditional_law.csv", index=False)
        plot_paired_survival_law(output / "paired_conditional_law.csv", output)
    if paired_rows:
        pd.concat(paired_rows, ignore_index=True).to_csv(output / "paired_finite_trial_targets.csv", index=False)
    pd.concat(standards_rows, ignore_index=True).to_csv(output / "standards_reconciliation.csv", index=False)
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(output / "family_diagnostics.csv", index=False)
    intervals = design_controls(source, cells)
    if intervals.empty:
        return
    intervals.to_csv(output / "design_sensitive_intervals.csv", index=False)
    panels = (
        ("TE-S04-A4", "Randomized endpoint retrieval", ("Retrieval-weighted",)),
        ("TE-S07-A2", "Cluster-randomized vaccination", ("Person-level uncertainty", "Cluster uncertainty")),
        ("TE-S08-A2", "Randomized obstetric rollout", ("Unadjusted", "Calendar adjustment\n+ cluster uncertainty")),
        ("TE-S09-A1", "Sequential anticoagulation trial", ("Single-look interval", "All planned looks")),
    )
    selected_panels = tuple(panel for panel in panels if panel[0] in set(intervals.cell))
    figure, axes = plt.subplots(
        (len(selected_panels) + 1) // 2,
        min(2, len(selected_panels)),
        figsize=(10, 3.25 * ((len(selected_panels) + 1) // 2)),
        squeeze=False,
    )
    for axis in axes.flat[len(selected_panels) :]:
        axis.set_visible(False)
    for axis, (identifier, title, labels) in zip(axes.flat[: len(selected_panels)], selected_panels, strict=True):
        selected = intervals.loc[intervals.cell.eq(identifier)]
        reference = float(
            diagnostics.loc[diagnostics.cell.eq(identifier), "finite_complete_contrast"].to_numpy(dtype=float)[0]
        )
        for position, (lower, upper, point) in enumerate(
            selected[["lower", "upper", "estimate"]].to_numpy(dtype=float)
        ):
            axis.plot([100 * lower, 100 * upper], [position, position], color="#24788a", linewidth=2)
            axis.scatter([100 * point], [position], color="#24788a", zorder=3)
        reference_line = axis.axvline(100 * reference, linestyle="--", color="#555555", linewidth=1)
        axis.axvline(0, color="#cccccc", linewidth=0.8)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_ylim(-0.6, max(0, len(labels) - 1) + 0.6)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Risk difference (percentage points)", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    figure.legend([reference_line], ["Disclosed complete-outcome contrast"], loc="upper center", frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("pdf", "png"):
        figure.savefig(output / f"design_sensitive_intervals.{extension}", dpi=200)
    plt.close(figure)


def main() -> None:
    """Run checks against a deliberately disclosed native export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Family data directory or ZIP; defaults to the bundled disclosed corpus."
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selected_source = args.source if args.source is not None else Path(__file__).with_name("family-records.zip")
    if selected_source.is_dir():
        review(selected_source, args.output)
    else:
        with tempfile.TemporaryDirectory(prefix="trialagentbench-examples-") as temporary:
            source = Path(temporary)
            with zipfile.ZipFile(selected_source) as archive:
                if any(not (source / name).resolve().is_relative_to(source) for name in archive.namelist()):
                    raise ValueError("Family archive contains an unsafe path")
                archive.extractall(source)
            review(source, args.output)


if __name__ == "__main__":
    main()
