"""Analyse observational evidence, acquired studies and development decisions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from review_scoring import numerical_loss
from scipy import stats
from scipy.integrate import quad
from scipy.special import expit, logit, softmax
from statsmodels.stats.proportion import confint_proportions_2indep


def decode_initial_factors(data: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Decode the example's explicit categorical codebook for numerical analysis."""
    decoded = data[factors.index].copy()
    for field, row in factors.iterrows():
        codes = json.loads(row.source_codes)
        if codes:
            if not decoded[field].dropna().isin(codes).all():
                raise ValueError(f"Initial evidence has an unknown category for {field}")
            decoded[field] = decoded[field].map(codes)
    return decoded


def initial_context_offsets(root: Path, data: pd.DataFrame, decoded: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct disclosed group intercepts and raw-value slopes using table joins."""
    terms = pd.read_csv(root / "initial_context.csv", keep_default_na=False, dtype={"group_level": str})
    offsets = pd.DataFrame(0.0, index=data.index, columns=["outcome", "safety"])
    recorded = data.assign(**{str(column): decoded[column] for column in decoded})
    for (endpoint, _, group, covariate), rows in terms.groupby(
        ["endpoint", "effect_id", "group_field", "input_field"], sort=False
    ):
        if rows.group_level.duplicated().any():
            raise ValueError("Disclosed context repeats a group coefficient")
        coefficients = pd.to_numeric(rows.set_index("group_level").coefficient, errors="raise")
        values = recorded[group].astype(str).map(coefficients)
        if not np.isfinite(values).all():
            raise ValueError("Disclosed context lacks a finite coefficient for an observed group")
        if covariate:
            values = values * pd.to_numeric(recorded[covariate], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError("Disclosed context slope requires complete finite recorded values")
        offsets[endpoint] += values
    return offsets


def initial_law_reference(root: Path, data: pd.DataFrame, output: Path) -> None:
    """Derive the disclosed measured-only law using SciPy truncated-normal probabilities."""
    factors = pd.read_csv(root / "initial_factors.csv").set_index("field")
    measurement = pd.read_csv(root / "initial_measurement.csv").iloc[0]
    candidates = pd.read_csv(root / "initial_candidates.csv").set_index("asset_id")
    decoded = decode_initial_factors(data, factors)
    standardized = (decoded - factors["mean"]) / factors.sd
    context = initial_context_offsets(root, data, decoded)
    support = measurement.upper - measurement.lower
    thresholds = np.arange(measurement.lower, measurement.upper, measurement.increment) + measurement.increment / 2
    limits = np.asarray(stats.norm.ppf([np.finfo(float).eps, 1 - np.finfo(float).eps]), dtype=float)
    means = []
    for asset, candidate in candidates.iterrows():
        coefficients = factors.outcome_coefficient.to_numpy() + candidate[
            [field + "_modifier" for field in factors.index]
        ].to_numpy(dtype=float)
        location = candidate.outcome_intercept + standardized.to_numpy() @ coefficients + context.outcome.to_numpy()
        a, b = (measurement.lower - location) / measurement.sd, (measurement.upper - location) / measurement.sd
        # E[rounded X] is its lowest value plus the spacing times the sum of
        # probabilities above each rounding midpoint. No simulation is needed.
        expectation = np.full(len(data), measurement.lower)
        for chunk in np.array_split(thresholds, 20):
            expectation += measurement.increment * stats.truncnorm.sf(
                (chunk[:, None] - location) / measurement.sd, a, b
            ).sum(axis=0)
        safety = np.asarray(
            expit(
                logit(candidate.reference_safety_probability)
                + standardized.to_numpy() @ factors.safety_coefficient.to_numpy()
                + context.safety.to_numpy()
            ),
            dtype=float,
        )
        # The native generator additionally clips absolute Gaussian quantiles.
        # Any recording change is bounded by support width times clipped mass.
        clipped_mass = np.asarray(
            stats.truncnorm.cdf(limits[0], a, b) + stats.truncnorm.sf(limits[1], a, b), dtype=float
        )
        means.append(
            {
                "asset_id": asset,
                "outcome_mean_difference": float(expectation.mean()),
                "safety_risk_difference": float(safety.mean()),
                "tail_clipping_bound": float(support * clipped_mass.mean()),
            }
        )
    derived = pd.DataFrame(means).set_index("asset_id")
    columns = ["outcome_mean_difference", "safety_risk_difference"]
    derived[columns] = derived[columns] - derived.loc["control", columns]
    derived["tail_clipping_bound"] += float(derived["tail_clipping_bound"].loc["control"])
    derived = derived.drop(index="control")
    native = pd.read_csv(root / "initial_references.csv").set_index("asset_id").loc[derived.index]
    for column in columns:
        derived[column + "_native_error"] = derived[column] - native[column]
        if not (derived[column + "_native_error"].abs() <= derived.tail_clipping_bound + 1e-11).all():
            raise ValueError("Native association reference disagrees with the disclosed conditional law")
    derived.to_csv(output / "trialdev_initial_reference_derivation.csv")


def initial_associations(root: Path, output: Path) -> None:
    """Estimate observed associations over the fixed recorded baseline population."""
    data = pd.read_csv(root / "initial" / "participants.csv")
    for name in ("treatments", "clinical_assessments", "safety_assessments"):
        data = data.merge(pd.read_csv(root / "initial" / f"{name}.csv"), validate="one_to_one")
    ready = pd.read_csv(root / "initial_ready.csv")
    pd.testing.assert_frame_equal(data[ready.columns], ready)
    initial_law_reference(root, data, output)
    references = pd.read_csv(root / "initial_references.csv").set_index("asset_id")
    measurement = pd.read_csv(root / "initial_measurement.csv").iloc[0]
    factors = pd.read_csv(root / "initial_factors.csv").set_index("field")
    baseline = decode_initial_factors(data, factors)
    centered = (baseline - baseline.mean()) / baseline.std(ddof=0)
    rows = []
    for method in ("unadjusted", "linear standardization", "quadratic standardization"):
        basis = pd.DataFrame({"intercept": np.ones(len(data))})
        if method != "unadjusted":
            basis = pd.concat([basis, centered], axis=1)
        if method == "quadratic standardization":
            for field in ("baseline_das28_score", "age_years"):
                basis[field + "_squared"] = centered[field] ** 2
            basis["severity_by_age"] = centered.baseline_das28_score * centered.age_years
        mean_basis = basis.mean().to_numpy()
        for endpoint, column, reference_column in (
            ("clinical_outcome", "das28_score_end_of_follow_up", "outcome_mean_difference"),
            ("safety_event", "infection_aesi_end_of_follow_up", "safety_risk_difference"),
        ):
            fits = {
                asset: sm.OLS(data.loc[data.asset_id.eq(asset), column], basis.loc[data.asset_id.eq(asset)]).fit(
                    cov_type="HC3"
                )
                for asset in sorted(data.asset_id.unique())
            }
            control = fits["control"]
            precision_column = (
                "outcome_numerical_standard_error"
                if endpoint == "clinical_outcome"
                else "safety_numerical_standard_error"
            )
            for asset, fit in fits.items():
                if asset == "control":
                    continue
                estimate = float(mean_basis @ (fit.params - control.params))
                variance = float(mean_basis @ (fit.cov_params() + control.cov_params()) @ mean_basis)
                margin = float(stats.t.ppf(0.975, min(fit.df_resid, control.df_resid)) * np.sqrt(variance))
                reference = float(references[reference_column].loc[asset])
                scale = (
                    measurement.clinical_loss_scale
                    if endpoint == "clinical_outcome"
                    else measurement.safety_loss_scale
                )
                rows.append(
                    {
                        "asset_id": asset,
                        "endpoint": endpoint,
                        "analysis": method,
                        "estimate": estimate,
                        "lower": estimate - margin,
                        "upper": estimate + margin,
                        "association_reference": references.loc[asset, reference_column],
                        "normalization_scale": scale,
                        "mean_loss": numerical_loss(estimate, estimate - margin, estimate + margin, reference, scale),
                        "reference_scramble_se": references.loc[asset, precision_column],
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(output / "trialdev_initial_associations.csv", index=False)
    results.groupby("analysis").agg(mean_loss=("mean_loss", "mean"), completed_targets=("mean_loss", "count")).to_csv(
        output / "trialdev_initial_analysis_summary.csv"
    )
    baseline.groupby(data.asset_id).mean().to_csv(output / "trialdev_initial_case_mix.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    assets = sorted(references.index)
    for axis, endpoint in zip(axes, ("clinical_outcome", "safety_event"), strict=True):
        unit = 1 if endpoint == "clinical_outcome" else 100
        for method_index, (method, (color, marker)) in enumerate(
            {
                "unadjusted": ("#777777", "o"),
                "linear standardization": ("#0072B2", "s"),
                "quadratic standardization": ("#009E73", "^"),
            }.items()
        ):
            selected = (
                results.loc[results.endpoint.eq(endpoint) & results.analysis.eq(method)]
                .set_index("asset_id")
                .loc[assets]
            )
            axis.errorbar(
                unit * selected.estimate,
                np.arange(len(assets)) + (method_index - 1) * 0.22,
                xerr=unit * np.array([selected.estimate - selected.lower, selected.upper - selected.estimate]),
                fmt=marker,
                color=color,
                capsize=3,
                label=method,
            )
        axis.plot(
            unit * selected.association_reference,
            np.arange(len(assets)),
            "D",
            color="black",
            label="Observed-law reference",
        )
        axis.axvline(0, color="#777777", linestyle=":", linewidth=0.8)
        axis.set_yticks(range(len(assets)))
        axis.invert_yaxis()
    axes[0].set_yticklabels(assets)
    axes[1].set_yticklabels([])
    axes[0].set_xlabel("Standardized DAS28 association (points)")
    axes[1].set_xlabel("Standardized safety association (percentage points)")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False, fontsize=8)
    figure.savefig(output / "trialdev_initial_associations.png", dpi=180)
    figure.savefig(output / "trialdev_initial_associations.pdf")
    plt.close(figure)


def development(root: Path, output: Path) -> None:
    """Recalculate finite truth, phased estimates, programme costs and terminal value."""
    from review_recovery import development_samples, programme_value, study_cost, welch_interval

    ledger = json.loads((root / "ledger.json").read_text())
    preferences = json.loads((root / "preferences.json").read_text())
    native = json.loads((root / "native_value.json").read_text())
    potentials = pd.read_csv(root / "terminal_potentials.csv", keep_default_na=False)
    means = potentials.groupby(["asset_id", "dose"])[["target_outcome", "safety_event"]].mean()
    control = means.loc[("control", "")]
    truth = (means - control).drop(index=("control", "")).reset_index()
    truth = truth.rename(columns={"target_outcome": "outcome_difference", "safety_event": "safety_difference"})
    truth.to_csv(output / "trialdev_finite_truth.csv", index=False)
    estimates, costs = [], []
    flow, phase_baselines, time_controls = [], [], []
    previous_samples = None
    previous_study_id = None
    previous_horizon = None
    for study in ledger["recruitment_commitments"]:
        study_id = study["study_id"]
        horizon = study["follow_up_days"]
        enrollment = next(row for row in ledger["enrollments"] if row["study_id"] == study_id)
        n = len(enrollment["patients"])
        # These three disclosed studies all finish every enrolled follow-up and scheduled assessment.
        total = study_cost(study, enrollment)
        recorded = [row for row in ledger["receipts"] if row["study_id"] == study_id][-1]["cost"]
        np.testing.assert_allclose(
            total, recorded["accrued_usd"] + recorded["remaining_liability_usd"], rtol=0, atol=1e-6
        )
        costs.append(
            {
                "phase": study["phase"],
                "screened": enrollment["screened"],
                "enrolled": n,
                "follow_up_days": horizon,
                "cost_usd": total,
            }
        )
        samples = {}
        baseline_samples = {}
        wrong_clock_samples = {}
        risks = {}
        safety_counts = {}
        for arm in ("control", "active"):
            directory = root / "readouts" / f"{study_id}-look" / "arms" / arm
            baseline = pd.read_csv(directory / "baseline.csv", float_precision="round_trip")
            measurements = pd.read_csv(directory / "measurements.csv", float_precision="round_trip")
            events = pd.read_csv(directory / "safety_events.csv")
            selected_samples = development_samples(baseline, measurements, horizon)
            samples[arm] = selected_samples["scheduled_visit"]
            baseline_samples[arm] = selected_samples["baseline_as_followup"]
            wrong_clock_samples[arm] = selected_samples["programme_clock_as_duration"]
            risks[arm] = events.participant_id.nunique() / len(baseline)
            safety_counts[arm] = (events.participant_id.nunique(), len(baseline))
            flow.append(
                {
                    "phase": study["phase"],
                    "arm": arm,
                    "randomized": len(baseline),
                    "outcome_observed": len(samples[arm]),
                    "safety_events": events.participant_id.nunique(),
                    "follow_up_days": study["follow_up_days"],
                }
            )
            for field in ("baseline_das28_score", "age_years", "prior_biologic_dmard_use"):
                phase_baselines.append(
                    {
                        "phase": study["phase"],
                        "arm": arm,
                        "field": field,
                        "n": len(baseline),
                        "mean": baseline[field].mean(),
                        "sd": baseline[field].std(),
                    }
                )
        active, comparator = samples["active"], samples["control"]
        estimate, lower, upper = welch_interval(active, comparator)
        study_potentials = pd.read_csv(root / f"potentials_{horizon}.csv", keep_default_na=False)
        study_means = study_potentials.groupby(["asset_id", "dose"])[["target_outcome", "safety_event"]].mean()
        active_partition = next(arm["partition"] for arm in study["arms"] if arm["arm_id"] == "active")
        difference = (
            study_means.loc[(active_partition["asset_id"], active_partition["dose"])]
            - study_means.loc[("control", "")]
        )
        methods = [
            ("scheduled_visit", samples, study_id, horizon),
            ("baseline_as_followup", baseline_samples, study_id, 0),
            ("programme_clock_as_duration", wrong_clock_samples, study_id, None),
        ]
        if previous_samples is not None:
            methods.append(("previous_study", previous_samples, previous_study_id, previous_horizon))
        for method, selected_samples, source_study, source_horizon in methods:
            observed_active, observed_control = len(selected_samples["active"]), len(selected_samples["control"])
            available = min(observed_active, observed_control) >= 2
            interval = welch_interval(selected_samples["active"], selected_samples["control"]) if available else None
            point, lo, hi = interval if interval is not None else (None,) * 3
            time_controls.append(
                {
                    "study_id": study_id,
                    "phase": study["phase"],
                    "target_follow_up_days": horizon,
                    "source_study_id": source_study,
                    "source_follow_up_days": source_horizon,
                    "method": method,
                    "status": "estimated" if available else "insufficient_observations",
                    "observed_active": observed_active,
                    "observed_control": observed_control,
                    "estimate": point,
                    "lower": lo,
                    "upper": hi,
                    "finite_outcome_reference": difference.target_outcome,
                    "error": interval[0] - difference.target_outcome if interval is not None else None,
                    "clinical_loss": (
                        numerical_loss(*interval, difference.target_outcome, preferences["benefit_unit"])
                        if interval is not None
                        else None
                    ),
                }
            )
        previous_samples, previous_study_id, previous_horizon = samples, study_id, horizon
        safety_lower, safety_upper = confint_proportions_2indep(
            *safety_counts["active"],
            *safety_counts["control"],
            method="newcomb",
            alpha=0.05,
        )
        estimates.append(
            {
                "phase": study["phase"],
                "follow_up_days": horizon,
                "estimate": estimate,
                "lower": lower,
                "upper": upper,
                "safety_difference": risks["active"] - risks["control"],
                "safety_lower": safety_lower,
                "safety_upper": safety_upper,
                "finite_outcome_reference": difference.target_outcome,
                "finite_safety_reference": difference.safety_event,
                "clinical_loss": numerical_loss(
                    estimate,
                    lower,
                    upper,
                    difference.target_outcome,
                    preferences["benefit_unit"],
                ),
                "safety_loss": numerical_loss(
                    risks["active"] - risks["control"], safety_lower, safety_upper, difference.safety_event, 0.01
                ),
                "observed_active": len(active),
                "observed_control": len(comparator),
            }
        )
    pd.DataFrame(time_controls).to_csv(output / "trialdev_time_controls.csv", index=False)
    pd.DataFrame(flow).to_csv(output / "trialdev_participant_flow.csv", index=False)
    pd.DataFrame(phase_baselines).to_csv(output / "trialdev_phase_baselines.csv", index=False)
    pd.DataFrame(costs).to_csv(output / "trialdev_resources.csv", index=False)
    pd.DataFrame(estimates).to_csv(output / "trialdev_readout_estimates.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for index, row in enumerate(estimates):
        axes[0].errorbar(
            row["estimate"],
            index,
            xerr=[[row["estimate"] - row["lower"]], [row["upper"] - row["estimate"]]],
            fmt="o",
            color="#0072B2",
            capsize=4,
            label="Observed difference and nominal 95% interval" if index == 0 else None,
        )
        axes[0].plot(
            row["finite_outcome_reference"],
            index,
            "D",
            color="#D55E00",
            label="Complete pool at this follow-up" if index == 0 else None,
        )
        axes[1].errorbar(
            100 * row["safety_difference"],
            index,
            xerr=[
                [100 * (row["safety_difference"] - row["safety_lower"])],
                [100 * (row["safety_upper"] - row["safety_difference"])],
            ],
            fmt="o",
            color="#0072B2",
            capsize=4,
        )
        axes[1].plot(100 * row["finite_safety_reference"], index, "D", color="#D55E00")
    labels = [
        f"{row['phase'].replace('_', ' ').title()}\n{cost['enrolled']} patients · Day {row['follow_up_days']}"
        for row, cost in zip(estimates, costs, strict=True)
    ]
    for axis in axes:
        axis.axvline(0, color="#777777", linewidth=0.8, linestyle=":")
        axis.set_yticks(range(len(estimates)))
        axis.invert_yaxis()
    axes[0].set_yticklabels(labels)
    axes[1].set_yticklabels([])
    axes[0].set_xlabel("DAS28 difference (active − control)")
    axes[1].set_xlabel("Safety risk difference (percentage points)")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", frameon=False, fontsize=8)
    figure.savefig(output / "trialdev_phases.png", dpi=180)
    figure.savefig(output / "trialdev_phases.pdf")
    plt.close(figure)
    expenditure = sum(row["cost_usd"] for row in costs)
    value = programme_value(root, ledger, preferences, expenditure)
    for key, recalculated in value.items():
        np.testing.assert_allclose(recalculated, native[key], rtol=0, atol=1e-6, err_msg=key)
    fixed, delay = value["net_value_usd"], value["time_cost_usd"]
    observed_final = estimates[-1]
    # Annual profit from the final observed contrasts; discounting cannot change its sign.
    predicted_profit = (
        -observed_final["estimate"] / preferences["benefit_unit"] * preferences["annual_profit_per_benefit_unit_usd"]
        - observed_final["safety_difference"] * 100 * preferences["annual_cost_per_safety_percentage_point_usd"]
    )
    components = pd.DataFrame(
        {
            "component": ["Clinical benefit", "Safety burden", "Studies", "Programme overhead", "Net value"],
            "value_usd": [
                value["clinical_value_usd"],
                -value["safety_cost_usd"],
                -expenditure,
                -delay,
                fixed,
            ],
        }
    )
    components.to_csv(output / "trialdev_value_components.csv", index=False)
    figure, axis = plt.subplots(figsize=(6, 3.6), layout="constrained")
    axis.barh(
        components.component,
        components.value_usd / 1_000_000,
        color=["#0072B2", "#D55E00", "#D55E00", "#D55E00", "#666666"],
    )
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.invert_yaxis()
    axis.set_xlabel("Declared sponsor value (USD millions)")
    figure.savefig(output / "trialdev_value.png", dpi=180)
    figure.savefig(output / "trialdev_value.pdf")
    plt.close(figure)
    decisions = pd.DataFrame(
        [
            {"policy": "Stop before spending", "decision": "stop", "net_value_usd": 0.0},
            {"policy": "Fixed three-phase advance", "decision": "advance", "net_value_usd": fixed},
            {
                "policy": "Same studies; plug-in terminal choice",
                "decision": "advance" if predicted_profit > 0 else "stop",
                "net_value_usd": fixed if predicted_profit > 0 else -expenditure - delay,
            },
        ]
    )
    adaptive = json.loads((root / "adaptive" / "ledger.json").read_text())
    adaptive_value = json.loads((root / "adaptive" / "native_value.json").read_text())
    adaptive_decisions = pd.read_csv(root / "adaptive" / "decisions.csv")
    fixed_enrollments = {row["study_id"]: row for row in ledger["enrollments"]}
    if any(row != fixed_enrollments[row["study_id"]] for row in adaptive["enrollments"]):
        raise ValueError("The stopping comparison did not retain matched recruitment")
    for index, (_, observed) in enumerate(adaptive_decisions.iterrows()):
        phase_estimate = estimates[index]
        np.testing.assert_allclose(
            [observed.clinical_difference, observed.safety_difference],
            [phase_estimate["estimate"], phase_estimate["safety_difference"]],
            rtol=0,
            atol=1e-12,
        )
        stop = observed.clinical_difference >= 0 or observed.safety_difference > 0.05
        expected_decision = "stop" if stop else "advance" if index == 2 else "continue"
        if observed.decision != expected_decision:
            raise ValueError("The stopping policy changed its declared observed-data rule")
    adaptive_cost = sum(row["cost_usd"] for row in costs[: len(adaptive_decisions)])
    adaptive_net = programme_value(root, adaptive, preferences, adaptive_cost)["net_value_usd"]
    np.testing.assert_allclose(adaptive_net, adaptive_value["net_value_usd"], rtol=0, atol=1e-6)
    decisions.loc[len(decisions)] = {
        "policy": "Observed-evidence stopping rule",
        "decision": adaptive["terminal"]["decision"],
        "net_value_usd": adaptive_net,
    }
    decisions.to_csv(output / "trialdev_policy_comparison.csv", index=False)


def development_timeline(root: Path, output: Path) -> None:
    """Plot every early-safety participant's recorded and scheduled chronology."""
    readout = root / "readouts" / "early_safety-look"
    study = json.loads((readout / "study.json").read_text())
    frames = []
    for arm in ("control", "active"):
        directory = readout / "arms" / arm
        baseline = pd.read_csv(directory / "baseline.csv", float_precision="round_trip")
        measurements = pd.read_csv(directory / "measurements.csv", float_precision="round_trip")
        frames.append(
            measurements.merge(
                baseline[["participant_id", "site_id", "enrollment_day"]],
                on="participant_id",
                validate="many_to_one",
            ).assign(arm=arm, study_id=study["study_id"], follow_up_days=study["follow_up_days"])
        )
    records = pd.concat(frames, ignore_index=True).sort_values(
        ["arm", "site_id", "enrollment_day", "participant_id", "occurrence_time"]
    )
    records.to_csv(output / "trialdev_participant_timeline.csv", index=False)
    # Render the exported table so the figure has one inspectable data source.
    records = pd.read_csv(output / "trialdev_participant_timeline.csv", float_precision="round_trip")
    figure, axis = plt.subplots(figsize=(7.1, 5.8), layout="constrained")
    labels = []
    for position, (participant, visits) in enumerate(records.groupby("participant_id", sort=False)):
        first = visits.iloc[0]
        if not isinstance(participant, str):
            raise ValueError("Participant timeline requires string identifiers")
        labels.append(f"{first.arm} · {participant.split(':')[-1]} · {first.site_id}")
        axis.plot(
            [first.enrollment_day, first.enrollment_day + first.follow_up_days],
            [position, position],
            color="#999999",
            linewidth=1,
            zorder=1,
        )
        for observed, marker, color in ((True, "o", "#226B93"), (False, "x", "#B45333")):
            selected = visits.loc[visits.observed.eq(observed)]
            axis.scatter(selected.occurrence_time, [position] * len(selected), marker=marker, color=color, s=20)
    axis.axvline(study["readout_day"], color="#555555", linestyle="--", linewidth=1, label="Readout cutoff")
    axis.scatter([], [], marker="o", color="#226B93", s=20, label="Observed assessment")
    axis.scatter([], [], marker="x", color="#B45333", s=20, label="Unavailable assessment")
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlim(0, study["readout_day"] + 8)
    axis.set_xlabel(f"Programme day (day 0: {study['calendar_origin']})")
    axis.spines[["top", "right"]].set_visible(False)
    figure.legend(*axis.get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False, fontsize=8)
    for suffix in ("svg", "pdf", "png"):
        figure.savefig(output / f"trialdev_participant_timeline.{suffix}", dpi=180)
    plt.close(figure)


def development_report(root: Path, output: Path) -> None:
    """Write clinical baseline, phase denominators and an observed-evidence account."""
    initial = pd.read_csv(root / "initial_ready.csv")
    arms = ["control", *sorted(set(initial.asset_id) - {"control"})]
    ledger = json.loads((root / "ledger.json").read_text())
    preferences = json.loads((root / "preferences.json").read_text())
    rule = json.loads((root / "design_space.json").read_text())["approval"]
    terminal = ledger["terminal"]
    review = terminal["regulatory_review"]
    selected_asset = terminal["asset_id"]
    resources = pd.read_csv(output / "trialdev_resources.csv").set_index("phase")
    early = json.loads((root / "readouts" / "early_safety-look" / "study.json").read_text())
    early_n, early_day = int(resources.enrolled["early_safety"]), early["readout_day"]
    summary = []
    for arm in arms:
        group = initial.loc[initial.asset_id.eq(arm)]
        summary.append(
            {
                "treatment": arm,
                "n": len(group),
                "baseline_das28_mean": group.baseline_das28_score.mean(),
                "baseline_das28_sd": group.baseline_das28_score.std(),
                "age_mean": group.age_years.mean(),
                "age_sd": group.age_years.std(),
                "prior_biologic_n": int(group.prior_biologic_dmard_use.sum()),
                "endpoint_n": int(group.das28_score_end_of_follow_up.notna().sum()),
                "safety_n": int(group.infection_aesi_end_of_follow_up.notna().sum()),
            }
        )
    pd.DataFrame(summary).to_csv(output / "trialdev_table1.csv", index=False)
    lines = [
        "# Developing a rheumatoid-arthritis treatment",
        "",
        "## Question and starting population",
        "",
        f"Which candidate warrants further development for adults with active rheumatoid arthritis despite treatment? The programme seeks lower DAS28 disease activity without an unacceptable increase in infection risk. It starts with {len(initial):,} patients receiving standard care or one of {len(arms) - 1} candidates.",
        "",
        "## Table 1. Initial evidence",
        "",
        "| Characteristic | Standard care | " + " | ".join(arms[1:]) + " |",
        "| --- | " + " | ".join("---:" for _ in arms) + " |",
    ]
    lines.append("| Patients | " + " | ".join(str(row["n"]) for row in summary) + " |")
    for field, label in (("baseline_das28", "Baseline DAS28, mean (SD)"), ("age", "Age, years, mean (SD)")):
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{row[field+'_mean']:.2f} ({row[field+'_sd']:.2f})" for row in summary)
            + " |"
        )
    lines.append(
        "| Prior biologic treatment, n (%) | "
        + " | ".join(f"{row['prior_biologic_n']} ({100*row['prior_biologic_n']/row['n']:.1f})" for row in summary)
        + " |"
    )
    lines += [
        "",
        "A raw outcome difference mixes treatment-associated differences with differences in patient characteristics. This example fits separate outcome models by treatment, predicts each treatment for every baseline profile, and averages those predictions over the same population. This estimates the requested standardized association; it does not establish that all confounding has been removed.",
        "",
        "## What the initial analysis changes",
        "",
        "| Candidate | Raw DAS28 difference (95% interval) | Standardized difference (95% interval) |",
        "| --- | ---: | ---: |",
    ]
    associations = pd.read_csv(output / "trialdev_initial_associations.csv")
    for arm in arms[1:]:
        rows = associations.loc[
            associations.asset_id.eq(arm) & associations.endpoint.eq("clinical_outcome")
        ].set_index("analysis")
        values = [rows.loc[name] for name in ("unadjusted", "linear standardization")]
        lines.append(
            "| "
            + arm
            + " | "
            + " | ".join(f"{row.estimate:.2f} ({row.lower:.2f}, {row.upper:.2f})" for row in values)
            + " |"
        )
    lines += [
        "",
        f"Compare the raw and standardized differences to see how measured case mix affects the apparent benefit. Lower DAS28 favours the candidate; infection risk and the sponsor's stated preferences also affect whether development is worthwhile. The illustrative programme investigates {selected_asset} through three randomized studies.",
        "",
        "## Authored studies and their readouts",
        "",
        f"The designs use a shared clinical question, 1:1 randomization and progressively longer follow-up. {early_n}, {resources.enrolled['exploration']} and {resources.enrolled['confirmation']} patients represent different precision and expenditure choices. Each confirmation arm holds {resources.enrolled['confirmation'] // 2} patients, the phase minimum, so one completed study can meet the regulator's exposure rule of {rule['minimum_completed_treated']} treated patients. For perspective, ten treated patients have a 65% chance of revealing at least one event with 10% risk, but only a 9.6% chance at 1% risk. An uneventful early study cannot exclude rare harm. Each activated site adds a fixed charge, so the smaller studies use fewer sites; the table records each study's site count.",
        "",
        "| Phase | Sites | Randomized, active/control | Outcome observed, active/control | Infection events, active/control | Follow-up |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    frames = pd.read_csv(output / "trialdev_participant_flow.csv")
    sites = {study["phase"]: study["sites"] for study in ledger["recruitment_commitments"]}
    for phase in ("early_safety", "exploration", "confirmation"):
        rows = frames.loc[frames.phase.eq(phase)].set_index("arm")
        lines.append(
            f"| {phase.replace('_', ' ').title()} | {sites[phase]} | "
            + " | ".join(
                f"{rows.loc['active',field]}/{rows.loc['control',field]}"
                for field in ("randomized", "outcome_observed", "safety_events")
            )
            + f" | {rows.iloc[0].follow_up_days:g} days |"
        )
    lines += [
        "",
        "DAS28 differences below use observed outcomes at each study's own horizon, with Welch intervals. Infection differences use all enrolled patients at the completed readout and Newcombe intervals. Phase-specific baseline summaries are in `trialdev_phase_baselines.csv`. The denominators show how many enrolled patients contributed an outcome.",
        "",
        "| Phase | DAS28 difference (95% interval) | Infection risk difference, percentage points (95% interval) |",
        "| --- | ---: | ---: |",
    ]
    readouts = pd.read_csv(output / "trialdev_readout_estimates.csv")
    for _, readout in readouts.iterrows():
        lines.append(
            f"| {readout.phase.replace('_', ' ').title()} | {readout.estimate:.2f} ({readout.lower:.2f}, {readout.upper:.2f}) | {100*readout.safety_difference:.1f} ({100*readout.safety_lower:.1f}, {100*readout.safety_upper:.1f}) |"
        )
    lines += [
        "",
        "![Study readouts with their uncertainty.](trialdev_phases.png)",
        "",
        "The intervals show how much each readout leaves unresolved. Compare both their width and their position relative to zero. Longer follow-up changes the clinical horizon as well as the amount of information: these are not three interchangeable estimates of one fixed effect.",
        "",
        "![All early-safety participants, grouped by treatment and site, with their scheduled follow-up and observed or unavailable assessments on the programme calendar.](trialdev_participant_timeline.png)",
        "",
        f"The timeline includes all {early_n} early-safety participants; none is selected by outcome. Each grey segment runs from enrollment to the scheduled {early['follow_up_days']}-day endpoint, not continuous observation. Circles are observed assessments and crosses unavailable assessments. The dashed line is the actual day-{early_day:g} readout cutoff. Enrollment and follow-up use each participant's clock: day {early_day:g} is neither {early_day:g} days of individual follow-up nor an assessment at that time. Site labels identify recruitment membership. These teaching records have no reporting lag between occurrence and availability; they do not illustrate delayed transfer. Exact occurrence, availability and enrollment times are in [the timeline table](trialdev_participant_timeline.csv).",
        "",
        "## Reproduce a submission from the visible records",
        "",
        "In a fresh working directory, copy one disclosed `trialdev/readouts/<phase>-look` directory to `readout/`, create `scratch/`, and save the following as `scratch/analysis.py`. Run `python scratch/analysis.py` from that working directory. Running the same command in a second fresh Python session reproduces `scratch/estimates.csv`; no interactive variables, population potentials or reference tables are needed. Repeat with the next completed readout in its own working directory. The target identifiers and horizon come from that readout's `study.json`, rather than from a previous study.",
        "",
        "This reproduces the two analyses in the table above for these completed, two-arm rheumatoid-arthritis teaching studies. It is an observed-outcome analysis, not a correction for informative missingness or incomplete safety ascertainment. The example uses the disclosed CSV projection; a live task supplies its own readout paths and file formats. Save and execute the corresponding analysis before requesting another readout or stopping, when the current results are captured. `check_analysis` can check fresh-session reproduction but cannot validate the clinical question or reveal the reference.",
        "",
        "```python",
        """import json
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import confint_proportions_2indep

readout = Path("readout")
study = json.loads((readout / "study.json").read_text())
targets = {row["endpoint"]: row for row in study["targets"]}
assert len(study["targets"]) == 2
assert set(targets) == {"clinical_outcome", "safety_risk"}
assert all(row["arm_id"] == "active" and row["confidence_level"] == 0.95
           for row in targets.values())
values, events, counts = {}, {}, {}
for arm in ("active", "control"):
    directory = readout / "arms" / arm
    baseline = pd.read_csv(directory / "baseline.csv", float_precision="round_trip")
    measurements = pd.read_csv(directory / "measurements.csv", float_precision="round_trip")
    safety = pd.read_csv(directory / "safety_events.csv")
    assert baseline.participant_id.is_unique
    enrollment = baseline.set_index("participant_id").enrollment_day
    assert set(measurements.participant_id) <= set(enrollment.index)
    assert set(safety.participant_id) <= set(enrollment.index)
    scheduled_time = measurements.participant_id.map(enrollment) + study["follow_up_days"]
    selected = measurements.loc[
        measurements.occurrence_time.eq(scheduled_time)
        & measurements.observed & measurements.outcome_id.eq("clinical_outcome")
    ]
    assert selected.participant_id.is_unique
    values[arm] = selected.observed_outcome.dropna()
    assert len(values[arm]) >= 2
    events[arm] = safety.participant_id.nunique()
    counts[arm] = len(baseline)

clinical = values["active"].mean() - values["control"].mean()
interval = ttest_ind(values["active"], values["control"], equal_var=False).confidence_interval()
lower, upper = confint_proportions_2indep(
    events["active"], counts["active"], events["control"], counts["control"],
    method="newcomb", alpha=0.05,
)
rows = [
    (targets["clinical_outcome"]["target_id"], clinical, interval.low, interval.high),
    (targets["safety_risk"]["target_id"],
     events["active"] / counts["active"] - events["control"] / counts["control"], lower, upper),
]
pd.DataFrame(rows, columns=["target_id", "estimate", "lower", "upper"]).to_csv(
    "scratch/estimates.csv", index=False
)""",
        "```",
        "",
        "A CSV with only one row is a partial answer: the available target can be retained, but it does not yield a complete two-target loss. A later deliberate stop retains study and delay costs. Neither a reproducible CSV nor a permitted stop establishes that the analysis recovered the intended effect or that acquisition was worthwhile.",
        "",
        "## Which records answer this question?",
        "",
        "[Time and readout controls](trialdev_time_controls.csv) repeat the analysis using the scheduled visit, baseline substituted for follow-up, programme time mistaken for duration since enrollment, and the previous completed study where available. Each row retains its source study, target horizon and contributing counts. An empty extraction has no estimate or loss; it is not a zero effect.",
        "",
        "Baseline and stale-readout estimates can reproduce exactly while answering a different question. Their Welch intervals quantify sampling variation in the selected records, not validity for the requested target. The correct visit can still be biased by missing outcomes. The three fixed teaching studies do not establish repeated-study coverage, clinical materiality or power. Previous-study comparisons also change participants and sometimes follow-up, so their differences do not isolate a clock effect. No method is required to have the smallest realised loss.",
        "",
        "| Target study | Analysis records | Active/control observed | Difference (95% interval), DAS28 | Finite-target loss |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in pd.read_csv(output / "trialdev_time_controls.csv").itertuples():
        result = (
            f"{row.estimate:.3f} ({row.lower:.3f}, {row.upper:.3f})" if row.status == "estimated" else "No estimate"
        )
        loss = f"{row.clinical_loss:.3f}" if row.status == "estimated" else "Not defined"
        lines.append(
            f"| {row.study_id} | {row.method} ({row.source_study_id}) | "
            f"{row.observed_active}/{row.observed_control} | {result} | {loss} |"
        )
    lines += [
        "",
        "## Stop, continue or advance",
        "",
        "A prespecified illustrative rule stops when observed benefit is nonpositive or the infection excess exceeds five percentage points. Another policy completes all three studies before choosing whether to advance from the observed estimates. Recruitment is matched through their shared studies. The table compares their decisions and costs with fixed advancement and immediate stopping. Neither observed-evidence policy consults complete outcomes.",
        "",
        "| Policy | Decision | Net development value, USD |",
        "| --- | --- | ---: |",
    ]
    for row in pd.read_csv(output / "trialdev_policy_comparison.csv").itertuples():
        lines.append(f"| {row.policy} | {row.decision} | {row.net_value_usd:,.0f} |")
    lines += [
        "",
        f"An advanced treatment earns market value only if the regulator approves it. At submission on day {terminal['programme_day']:g}, the regulator analysed the complete recorded outcomes of the {review['completed_treated']} treated and {review['completed_control']} control confirmation patients whose follow-up had ended. Its pooled Welch statistic was z = {review['benefit_z_statistic']:.2f}; approval requires a two-sided {100 * rule['two_sided_alpha']:g}% test of benefit and at least {rule['minimum_completed_treated']} completed treated patients, so the treatment was {'approved' if review['approved'] else 'rejected'}.",
        "",
        f"Approved value is annual profit of ${preferences['annual_profit_per_benefit_unit_usd'] / 1e6:g}M per {preferences['benefit_unit']:g} DAS28-point benefit, less ${preferences['annual_cost_per_safety_percentage_point_usd'] / 1e6:.2f}M per percentage point of excess infection risk. Both use Day-336 contrasts in the full programme population. Profit runs from launch, {preferences['regulatory_review_days']} days after submission, to exclusivity end on programme day {preferences['exclusivity_end_day']:,}, discounted at {100 * preferences['discount_rate_per_year']:g}% a year to day zero, so each year of delay loses roughly 12% of the value. Study costs are nominal, and programme overhead accrues at ${preferences['programme_cost_per_day_usd']:,.0f} a day. The treatment contrasts average complete recorded outcomes over the same participants, retaining measurement variation, rounding and score limits before missed visits hide observations. Stopping avoids the treatment payoff while retaining incurred costs. Compare values within this programme: one example cannot establish which policy performs best across programmes with different benefits, harms and resource constraints.",
        "",
        "The independent reader derives starting-cohort references by numerical integration, verifies matched recruitment, reproduces the observed-data decisions, repeats the regulator's approval test and recalculates costs and terminal value from the supplied records. The report, code and tables make the reasoning inspectable; numerical grading does not award credit for choosing these particular models or policies.",
    ]
    (output / "trialdev_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def latent_selection_control(root: Path, output: Path) -> None:
    """Integrate a disclosed latent-selection law and distinguish association from intervention."""
    candidates = pd.read_csv(root / "candidates.csv").set_index("asset_id")
    factors = pd.read_csv(root / "factors.csv").set_index("field")
    cases = pd.read_csv(root / "cases.csv").set_index("participant_id")
    native = pd.read_csv(root / "native_expectations.csv")
    if not np.allclose(factors["mean"], 0) or not np.allclose(factors.sd, 1):
        raise ValueError("This disclosed control uses two independent standard-normal factors")
    measured_effect = float(factors["outcome_coefficient"]["baseline_risk"])
    hidden_effect = float(factors["outcome_coefficient"]["hidden_risk"])
    records = []
    for row in native.to_dict(orient="records"):
        x = float(cases["baseline_risk"][int(row["participant_id"])])
        index = list(candidates.index).index(row["asset_id"])
        intercept = float(candidates.iloc[index].intercept)
        intercepts = candidates.assignment_intercept.to_numpy() + candidates.measured_assignment.to_numpy() * x
        slopes = candidates.hidden_assignment.to_numpy()
        mean0 = intercept + measured_effect * x
        count = row["family"] != "continuous"

        def weight(
            u: float,
            intercepts: npt.NDArray[np.float64] = intercepts,
            slopes: npt.NDArray[np.float64] = slopes,
            index: int = index,
        ) -> float:
            return float(softmax(intercepts + slopes * u)[index] * stats.norm.pdf(u))

        denominator, denominator_error = quad(weight, -10, 10, epsabs=1e-12, epsrel=1e-12)
        numerator, numerator_error = quad(
            lambda u, base=mean0, is_count=count, density=weight: density(u)
            * (np.exp(base + hidden_effect * u) if is_count else base + hidden_effect * u),
            -10,
            10,
            epsabs=1e-12,
            epsrel=1e-12,
        )
        derived = numerator / denominator
        intervention = float(np.exp(mean0 + hidden_effect**2 / 2)) if count else mean0
        tail_mass = 2 * stats.norm.sf(10)
        numerator_tail = (
            intervention * (stats.norm.sf(10 - hidden_effect) + stats.norm.cdf(-10 - hidden_effect))
            if count
            else abs(mean0) * tail_mass + abs(hidden_effect) * 2 * stats.norm.pdf(10)
        )
        tail_bound = numerator_tail / denominator + abs(numerator) * tail_mass / denominator**2
        np.testing.assert_allclose(derived, row["outcome_mean"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(denominator, row["assignment_probability"], rtol=0, atol=1e-8)
        records.append(
            {
                "family": row["family"],
                "participant_id": row["participant_id"],
                "asset_id": row["asset_id"],
                "observed_conditional_mean": derived,
                "intervention_mean": intervention,
                "native_error": derived - float(row["outcome_mean"]),
                "reported_quadrature_error": numerator_error / denominator
                + abs(numerator) * denominator_error / denominator**2,
                "normal_tail_error_bound": tail_bound,
            }
        )
    pd.DataFrame(records).to_csv(output / "latent_selection_control.csv", index=False)
