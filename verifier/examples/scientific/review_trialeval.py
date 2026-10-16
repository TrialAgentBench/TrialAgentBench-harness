"""Reconstruct completed trials and compare treatment-effect estimates."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from review_scoring import numerical_loss
from scipy import stats


def paired_survival_report(output: Path) -> None:
    """Score the two disclosed survival targets against separate reference studies.

    Parameters
    ----------
    output
        Existing family-review output containing observed analyses and complete
        reference-study totals.

    Raises
    ------
    ValueError
        If reference identities, arm counts or precision cannot support the
        paired teaching calculation.
    """
    references = pd.read_csv(output / "families/paired_population_totals.csv")
    estimates = pd.read_csv(output / "families/paired_observed_estimates.csv")
    finite = pd.read_csv(output / "families/paired_finite_trial_targets.csv")
    bindings = finite[["cell", "effect_measure", "target_id", "unit"]]
    checked = references.merge(bindings, on=["cell", "effect_measure", "target_id", "unit"], validate="many_to_one")
    if len(checked) != len(references):
        raise ValueError("Paired teaching reference identity or unit differs from its source target")
    summaries = []
    for (cell, measure), group in references.groupby(["cell", "effect_measure"]):
        if (
            len(group) != 64
            or group.seed.nunique() != 64
            or group.target_id.nunique() != 1
            or group.unit.nunique() != 1
            or (group[["active_n", "control_n"]] <= 0).any().any()
        ):
            raise ValueError("Paired teaching references require 64 separate complete studies per target")
        values = group.active_total / group.active_n - group.control_total / group.control_n
        sd = float(values.std(ddof=1))
        if not np.isfinite(values).all() or not np.isfinite(sd) or sd <= 0:
            raise ValueError("Paired teaching references require a finite positive scale")
        mean = float(values.mean())
        half = stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))
        summaries.append(
            {
                "cell": cell,
                "effect_measure": measure,
                "target_id": str(group.target_id.iloc[0]),
                "unit": str(group.unit.iloc[0]),
                "reference": mean,
                "reference_sd": sd,
                "reference_mcse": sd / np.sqrt(len(values)),
                "reference_lower": mean - half,
                "reference_upper": mean + half,
            }
        )
    reference_table = pd.DataFrame(summaries)
    if len(reference_table) != 2 or reference_table.cell.nunique() != 1:
        raise ValueError("Worked paired proof requires both targets from one study")
    result = estimates.merge(reference_table, on=["cell", "effect_measure"], validate="many_to_one")
    if len(result) != 4:
        raise ValueError("Worked paired proof requires both analyses for both targets")
    result["loss"] = np.vectorize(numerical_loss)(
        *(
            result[column].to_numpy(dtype=float)
            for column in ("estimate", "lower", "upper", "reference", "reference_sd")
        )
    )
    result.to_csv(output / "paired_survival_analyses.csv", index=False)
    reference_table.to_csv(output / "paired_survival_references.csv", index=False)
    plot_paired_survival(output)


def plot_paired_survival(output: Path) -> None:
    """Connect observed survival curves to both estimates and separate reference uncertainty."""
    curves = pd.read_csv(output / "families/paired_survival_curves.csv")
    results = pd.read_csv(output / "paired_survival_analyses.csv")
    references = pd.read_csv(output / "paired_survival_references.csv")
    wide = curves.pivot(index="time_day", columns="arm", values="survival").sort_index()
    figure = plt.figure(figsize=(9, 6.2), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.3, 1))
    curve_axis = figure.add_subplot(grid[0, :])
    for arm, label, colour, line in (
        ("active", "Maintenance therapy", "#0072B2", "-"),
        ("control", "Observation", "#D55E00", "--"),
    ):
        curve_axis.step(wide.index, wide[arm], where="post", label=label, color=colour, linestyle=line)
    curve_axis.fill_between(wide.index, wide.active, wide.control, step="post", color="#0072B2", alpha=0.15)
    curve_axis.set(xlabel="Days since randomization", ylabel="Estimated survival probability", ylim=(0, 1.02))
    curve_axis.set_xlim(0, float(wide.index.max()))
    curve_axis.set_title("A  Same observed histories, two survival questions", loc="left", fontsize=11)
    curve_axis.legend(loc="lower left", frameon=False)
    curve_axis.text(
        0.98,
        0.08,
        "Area between curves → restricted mean time difference\nHorizon gap → death-risk difference (opposite sign)",
        transform=curve_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    for column, (measure, label, multiplier, panel) in enumerate(
        (
            ("risk_difference", "Death-risk difference (percentage points)", 100, "B"),
            ("restricted_mean_survival_time_difference", "Restricted mean time difference (days)", 1, "C"),
        )
    ):
        axis = figure.add_subplot(grid[1, column])
        selected = results.loc[results.effect_measure.eq(measure)].set_index("method")
        for method, y, marker in (("Kaplan–Meier", 2, "o"), ("Crude observed records", 1, "^")):
            row = selected.loc[method]
            axis.errorbar(
                multiplier * row.estimate,
                y,
                xerr=multiplier * np.array([[row.estimate - row.lower], [row.upper - row.estimate]]),
                fmt=marker,
                color="#333333",
                capsize=3,
            )
        reference = references.loc[references.effect_measure.eq(measure)].iloc[0]
        axis.errorbar(
            multiplier * reference.reference,
            0,
            xerr=multiplier
            * np.array(
                [[reference.reference - reference.reference_lower], [reference.reference_upper - reference.reference]]
            ),
            fmt="s",
            color="#009E73",
            capsize=3,
        )
        axis.axvline(0, color="#999999", linestyle=":", linewidth=1)
        axis.set(
            xlabel=label,
            yticks=[2, 1, 0],
            yticklabels=["Kaplan–Meier", "Crude observed", "64-study reference"],
            ylim=(-0.5, 2.5),
        )
        axis.set_title(
            f"{panel}  "
            + ("Death risk at Day 730" if measure == "risk_difference" else "Mean survival time through Day 730"),
            loc="left",
            fontsize=10,
        )
    for axis in figure.axes:
        axis.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"paired_survival.{extension}", dpi=180)
    plt.close(figure)
    (output / "paired_survival.md").write_text(
        "# Two targets from one survival study\n\n"
        "Observed Kaplan–Meier curves for the same 1,896 randomized teaching participants. "
        "The signed area between the curves is the restricted mean survival-time contrast; "
        "the opposite-signed horizon gap is the death-risk contrast at Day 730. "
        "Panels B and C show marginal 95% normal patient-sampling intervals for two analyses, "
        "and a separate 95% t interval for Monte Carlo uncertainty in the 64-study reference. "
        "The latter is not a patient-sampling interval. Contrasts are maintenance therapy minus observation. "
        "No simultaneous coverage, method superiority or clinical materiality is established.\n\n"
        "Alternative text: Maintenance-therapy and observation step curves share a 730-day horizon. "
        "Their gap and integrated gap define two correlated targets. Below, two observed-data analyses "
        "are compared with a separate population-reference estimate on risk and time scales.\n",
        encoding="utf-8",
    )


def recovery_loss_report(review: Path) -> None:
    """Summarize fixed teaching losses with matched-study and reference uncertainty.

    Sampling intervals condition on the teaching reference and scale. Reference
    sensitivity is separate, and scale-estimation uncertainty is not propagated.
    These exhaustive method comparisons are exploratory, not clinical approval.

    Parameters
    ----------
    review : Path
        Existing review directory containing study estimates and reference tables.

    Returns
    -------
    None
        Write study losses, summaries, matched contrasts and their scope in place.

    Raises
    ------
    ValueError
        If methods are incomplete, scales are invalid or matched inputs disagree.

    Examples
    --------
    >>> recovery_loss_report(Path("scientific-review"))
    """
    rows = pd.read_csv(review / "repeated_study_estimates.csv")
    refs = pd.read_csv(review / "population_references.csv").set_index("case")
    rows["scale"] = rows.case.map(refs.empirical_reference_se * np.sqrt(refs.reference_worlds))
    if not (rows.scale > 0).all() or not rows.complete.all():
        raise ValueError("Recovery loss requires complete methods and positive reference scales")

    def loss(frame: pd.DataFrame, truth: float | pd.Series) -> npt.NDArray[np.float64]:
        return np.asarray(
            np.vectorize(numerical_loss, otypes=[float])(frame.estimate, frame.lower, frame.upper, truth, frame.scale),
            dtype=np.float64,
        )

    rows["normalized_point_loss"] = abs(rows.estimate - rows.reference) / rows.scale
    rows["mean_loss"] = loss(rows, rows.reference)
    rows["normalized_interval_loss"] = 2 * rows.mean_loss - rows.normalized_point_loss
    rows.to_csv(review / "repeated_study_losses.csv", index=False)
    summaries = []
    for (case, method), group in rows.groupby(["case", "method"], sort=True):
        if len(group) != 32 or group.seed.nunique() != 32:
            raise ValueError("Recovery loss requires all 32 declared study seeds")
        se = group.mean_loss.std(ddof=1) / np.sqrt(len(group))
        delta = stats.t.ppf(0.975, len(group) - 1) * se
        ref = refs.loc[[str(case)]].iloc[0]
        truth = ref.reference
        shift = stats.t.ppf(0.975, 63) * ref.reference_se
        knots = np.clip(
            np.r_[truth - shift, truth + shift, group.estimate, group.lower, group.upper], truth - shift, truth + shift
        )
        sensitivity = [loss(group, x).mean() for x in knots]
        summaries.append(
            {
                "case": case,
                "method": method,
                "n": len(group),
                "scale": group.scale.iloc[0],
                "mean_loss": group.mean_loss.mean(),
                "sampling_mcse_conditional_on_reference_and_scale": se,
                "sampling_lower": group.mean_loss.mean() - delta,
                "sampling_upper": group.mean_loss.mean() + delta,
                "reference_sensitivity_min": min(sensitivity),
                "reference_sensitivity_max": max(sensitivity),
            }
        )
    pd.DataFrame(summaries).to_csv(review / "mean_loss_summary.csv", index=False)
    comparisons = []

    def contrast(label: str, left: pd.DataFrame, right: pd.DataFrame) -> None:
        a = left.set_index("seed").sort_index()
        b = right.set_index("seed").sort_index()
        if not a.index.equals(b.index) or len(a) != 32:
            raise ValueError("Recovery comparisons require identical study seeds")
        if a.reference.iloc[0] != b.reference.iloc[0] or a.scale.iloc[0] != b.scale.iloc[0]:
            raise ValueError("Paired loss sensitivity requires a shared reference and scale")
        d = a.mean_loss - b.mean_loss
        se = d.std(ddof=1) / np.sqrt(len(d))
        half = stats.t.ppf(0.975, len(d) - 1) * se
        truth = a.reference.iloc[0]
        shift = stats.t.ppf(0.975, 63) * a.reference_se.iloc[0]
        knots = np.clip(
            np.r_[truth - shift, truth + shift, a.estimate, a.lower, a.upper, b.estimate, b.lower, b.upper],
            truth - shift,
            truth + shift,
        )
        sensitivity = [float((loss(a, x) - loss(b, x)).mean()) for x in knots]
        comparisons.append(
            {
                "comparison": label,
                "n": len(d),
                "mean_paired_loss_difference": d.mean(),
                "sampling_mcse_conditional_on_reference_and_scale": se,
                "sampling_lower": d.mean() - half,
                "sampling_upper": d.mean() + half,
                "reference_sensitivity_min": min(sensitivity),
                "reference_sensitivity_max": max(sensitivity),
            }
        )

    for case, group in rows.groupby("case"):
        methods = sorted(group.method.unique())
        for a, b in itertools.combinations(methods, 2):
            contrast(f"{str(case)}: {a} minus {b}", group[group.method == a], group[group.method == b])
    for tier in ["A4r25", "A4r100"]:
        for method in ["Crude event risk", "Randomized retrieval"]:
            contrast(
                f"TE-S04 {tier} minus A4: {method}",
                rows[(rows.case == f"TE-S04-{tier}-favourable") & (rows.method == method)],
                rows[(rows.case == "TE-S04-A4-favourable") & (rows.method == method)],
            )
    pd.DataFrame(comparisons).to_csv(review / "paired_loss_contrasts.csv", index=False)
    (review / "loss_scope.json").write_text(
        json.dumps(
            {
                "studies_per_condition": 32,
                "method_condition_groups": len(summaries),
                "method_study_results": len(rows),
                "paired_contrasts": len(comparisons),
                "scale": "Empirical complete-study contrast SD from each disjoint 64-study teaching reference bank, held fixed for these comparisons. Not a released scored-world normalization estimate.",
                "uncertainty": "Sampling t intervals condition on the reference and scale. Shared-reference sensitivity is the exact piecewise-linear range over the reference mean t interval. Scale-estimation uncertainty and clinical materiality remain unresolved; do not interpret these as final qualification or universal superiority.",
            },
            indent=2,
        )
        + "\n"
    )


def fixed_trial(root: Path, output: Path) -> None:
    """Reconstruct source revisions and compare analyses of the same trial."""
    participants = pd.read_csv(root / "participants.csv")
    records = pd.read_csv(root / "endpoint_observations.csv")
    ready = pd.read_csv(root / "analysis_ready.csv")
    endpoint = "motor_disability_score"
    current = records.sort_values("record_version").drop_duplicates("participant_id", keep="last")
    analysis = participants.merge(current[["participant_id", endpoint]], validate="one_to_one")
    pd.testing.assert_frame_equal(
        analysis[ready.columns].sort_values("participant_id").reset_index(drop=True),
        ready.sort_values("participant_id").reset_index(drop=True),
    )
    references = pd.read_csv(root / "reference_worlds.csv")
    differences = references.active_total / references.active_n - references.control_total / references.control_n
    target = float(differences.mean())
    scale = float(differences.std(ddof=1))
    reference_se = scale / np.sqrt(len(differences))
    reference_halfwidth = float(stats.t.ppf(0.975, len(differences) - 1) * reference_se)
    observation_errors = references.observed_difference - differences
    pd.DataFrame(
        [
            {
                "reference": target,
                "monte_carlo_se": reference_se,
                "worlds": len(differences),
                "normalization_scale": scale,
                "reference_lower": target - reference_halfwidth,
                "reference_upper": target + reference_halfwidth,
                "paired_observation_error": float(observation_errors.mean()),
                "paired_observation_error_se": float(observation_errors.std(ddof=1) / np.sqrt(len(differences))),
            }
        ]
    ).to_csv(output / "reference_precision.csv", index=False)
    law = pd.read_csv(root / "law_constants.csv").set_index("arm")
    expected = {}
    for arm, law_row in law.iterrows():
        probability = law_row.initial_progressive_probability
        for _ in range(int(law_row.transition_count)):
            probability = (1 - probability) * law_row.progression_probability + probability * (
                1 - law_row.recovery_probability
            )
        expected[arm] = law_row.direct_effect + probability * law_row.progressive_shift
    before_rounding = expected["active"] - expected["control"]
    epsilon = np.finfo(float).eps
    rounding_bound = float((law.rounding_increment / 2 + law.residual_sd * np.sqrt(epsilon / (1 - epsilon))).sum())
    pd.DataFrame(
        [
            {
                "pre_rounding_law_contrast": before_rounding,
                "recording_error_bound": rounding_bound,
                "reference": target,
                "reference_monte_carlo_se": reference_se,
            }
        ]
    ).to_csv(output / "known_law.csv", index=False)
    rows = []
    stale = records.sort_values("record_version").drop_duplicates("participant_id", keep="first")
    shortcut = participants.merge(stale[["participant_id", endpoint]], validate="one_to_one")
    for method, frame, adjusted in (
        ("Current records: Welch", analysis, False),
        ("Current records: adjusted HC3", analysis, True),
        ("Superseded records: Welch", shortcut, False),
    ):
        frame = frame.dropna(subset=[endpoint]).copy()
        if adjusted:
            design = frame[
                ["age_years", "years_since_parkinson_disease_diagnosis", "baseline_motor_disability_score"]
            ].copy()
            design["active"] = frame.assigned_arm.eq("active").astype(int)
            result = sm.OLS(frame[endpoint], sm.add_constant(design)).fit(cov_type="HC3", use_t=True)
            estimate = float(result.params["active"])
            lower, upper = result.conf_int().loc["active"]
        else:
            active = frame.loc[frame.assigned_arm.eq("active"), endpoint]
            control = frame.loc[frame.assigned_arm.eq("control"), endpoint]
            estimate = float(active.mean() - control.mean())
            va, vc = active.var(ddof=1) / len(active), control.var(ddof=1) / len(control)
            df = (va + vc) ** 2 / (va**2 / (len(active) - 1) + vc**2 / (len(control) - 1))
            halfwidth = stats.t.ppf(0.975, df) * np.sqrt(va + vc)
            lower, upper = estimate - halfwidth, estimate + halfwidth
        if not np.isfinite([estimate, lower, upper]).all():
            raise ValueError(f"Analysis did not produce finite estimates: {method}")
        # Apply the stated mathematical definition directly, separately from the
        # production implementation. The reference-world SD fixes the scale before
        # examining any submitted analysis; it is distinct from reference MCSE.
        interval_loss = upper - lower + 40 * max(lower - target, target - upper, 0)
        mean_loss = numerical_loss(estimate, lower, upper, target, scale)
        rows.append(
            {
                "analysis": method,
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
                "reference": target,
                "absolute_error": abs(estimate - target),
                "interval_width": upper - lower,
                "interval_loss": interval_loss,
                "scale_points": scale,
                "mean_loss": mean_loss,
                "complete": True,
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(output / "trialeval_analyses.csv", index=False)
    sensitivity = []
    for reference_range, left, right in (
        ("Reference effect ± numerical uncertainty", target - reference_halfwidth, target + reference_halfwidth),
        ("Known-law recording bound", before_rounding - rounding_bound, before_rounding + rounding_bound),
    ):
        # Loss differences are piecewise linear in the shared truth. Endpoints
        # and all point/interval knots therefore suffice for their exact extrema.
        knots = sorted(
            {
                left,
                right,
                *(min(right, max(left, row[field])) for row in rows for field in ("estimate", "lower", "upper")),
            }
        )
        baseline_row = rows[0]
        for alternative in rows[1:]:
            gaps = [
                numerical_loss(alternative["estimate"], alternative["lower"], alternative["upper"], value, scale)
                - numerical_loss(baseline_row["estimate"], baseline_row["lower"], baseline_row["upper"], value, scale)
                for value in knots
            ]
            sensitivity.append(
                {
                    "analysis": alternative["analysis"],
                    "compared_with": baseline_row["analysis"],
                    "reference_range": reference_range,
                    "minimum_loss_difference": min(gaps),
                    "maximum_loss_difference": max(gaps),
                }
            )
    pd.DataFrame(sensitivity).to_csv(output / "trialeval_reference_sensitivity.csv", index=False)
    analysis.to_csv(output / "reconstructed_analysis.csv", index=False)
    analysis.groupby("assigned_arm")[
        [endpoint, "age_years", "years_since_parkinson_disease_diagnosis", "baseline_motor_disability_score"]
    ].agg(["count", "mean", "std", "min", "max"]).to_csv(output / "population_summary.csv")
    figure, axis = plt.subplots(figsize=(8, 3.3), layout="constrained")
    axis.axvspan(
        target - reference_halfwidth,
        target + reference_halfwidth,
        color="#dddddd",
        label="Reference effect ± numerical uncertainty",
    )
    axis.axvline(target, color="#444444", linestyle="--")
    for index, row in enumerate(rows):
        axis.errorbar(
            row["estimate"],
            index,
            xerr=[[row["estimate"] - row["lower"]], [row["upper"] - row["estimate"]]],
            fmt="o",
            color="#0072B2" if index < 2 else "#D55E00",
            capsize=4,
        )
    axis.set_yticks(range(len(rows)), ["Group averages", "Baseline-adjusted", "Outdated assessments"])
    axis.set_xlabel("Rehabilitation − usual care (disability points)\nNegative values favour rehabilitation")
    axis.invert_yaxis()
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=8, frameon=False)
    figure.savefig(output / "trialeval_analyses.png", dpi=180)
    figure.savefig(output / "trialeval_analyses.pdf")
    plt.close(figure)


def population_display(output: Path) -> None:
    """Show clinical features and observed outcomes in the worked trial."""
    data = pd.read_csv(output / "reconstructed_analysis.csv")
    baseline = "baseline_motor_disability_score"
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    axes[0].hist(data.age_years, bins=15, color="#168a8b", edgecolor="white")
    axes[0].set(xlabel="Age (years)", ylabel="Patients", title="Who enters the trial")
    endpoint = "motor_disability_score"
    for arm, colour, marker, label in (
        ("control", "#687888", "o", "Usual care"),
        ("active", "#0072B2", "^", "Rehabilitation"),
    ):
        group = data.loc[data.assigned_arm.eq(arm)]
        axes[1].scatter(group[baseline], group[endpoint], s=13, alpha=0.4, color=colour, marker=marker, label=label)
    axes[1].set(
        xlabel="Baseline disability (points)",
        ylabel="Day-180 disability (points)",
        title="Baseline health and later outcome",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "population_and_response.png", dpi=180)
    figure.savefig(output / "population_and_response.pdf")
    plt.close(figure)
