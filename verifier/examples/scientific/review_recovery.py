"""Independent repeated-study recovery and design displays from exported records."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from review_families import cluster_risk_deletions, horizon_risk, recovery_estimate
from scipy.stats import norm, t
from statsmodels.stats.proportion import proportion_confint

COLOURS = ("#226B93", "#B45333", "#287768")


def save(figure: Figure, output: Path, name: str) -> None:
    """Save the same final construction as editable vector and raster exports."""
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def mean_contrast(values: pd.Series, arms: pd.Series) -> tuple[float, float]:
    """Return an independent-arm difference and its estimated standard error."""
    groups = values.groupby(arms)
    means, counts = groups.mean(), groups.count()
    if set(means.index) != {"active", "control"} or counts.min() < 2:
        raise ValueError("Both arms need at least two observed outcomes")
    return float(means.active - means.control), float(np.sqrt((groups.var() / counts).sum()))


def estimates(
    data: pd.DataFrame, series: str, endpoint: str, horizon: float, probability: float
) -> list[tuple[str, float, float]]:
    """Fit observed-record methods without accepting latent outcomes or references."""
    if series == "TE-S10":
        return [("Observed mean", *mean_contrast(data[endpoint], data.assigned_arm))]
    if series == "TE-S04":
        return [
            ("Crude event risk", *mean_contrast(data.event_observed.astype(float), data.assigned_arm)),
            ("Randomized retrieval", *recovery_estimate(data, "horizon_endpoint", probability)),
        ]
    if series == "TE-S15":
        grouped = data.groupby(["assigned_arm", "validation_stratum"])
        # Post-stratify the validation sample. Its influence function retains both
        # random case-mix composition and within-stratum adjudication uncertainty.
        cell_mean = grouped.adjudicated_endpoint.transform("mean")
        selection = grouped.validation_selected.transform("mean")
        if cell_mean.isna().any() or (selection <= 0).any():
            raise ValueError("Every validation stratum requires adjudicated observations")
        pseudo = (
            cell_mean
            + data.validation_selected * (data.adjudicated_endpoint.fillna(cell_mean) - cell_mean) / selection
        )
        return [
            ("Routine endpoint", *mean_contrast(data[endpoint], data.assigned_arm)),
            ("Stratified adjudication", *mean_contrast(pseudo, data.assigned_arm)),
        ]
    if series == "TE-S07":
        estimate, se = horizon_risk(data, horizon)
        clusters = data.cluster_id.unique()
        deleted = cluster_risk_deletions(data, horizon)
        jackknife = np.sqrt((len(clusters) - 1) / len(clusters) * np.square(deleted - deleted.mean()).sum())
        # Express the t interval as a normal-equivalent SE for common storage.
        adjusted = jackknife * t.ppf(0.975, len(clusters) - 2) / norm.ppf(0.975)
        return [("Person-level uncertainty", estimate, se), ("Cluster jackknife", estimate, float(adjusted))]
    raise ValueError(f"Unsupported experiment: {series}")


def welch_interval(active: pd.Series, control: pd.Series) -> tuple[float, float, float]:
    """Calculate an observed mean contrast and unequal-variance 95% interval."""
    if min(len(active), len(control)) < 2:
        raise ValueError("Welch uncertainty requires at least two observations per arm")
    va, vc = active.var(ddof=1) / len(active), control.var(ddof=1) / len(control)
    if not np.isfinite([va, vc]).all() or va + vc <= 0:
        raise ValueError("Welch uncertainty requires finite positive sampling variance")
    df = (va + vc) ** 2 / (va**2 / (len(active) - 1) + vc**2 / (len(control) - 1))
    estimate = float(active.mean() - control.mean())
    margin = float(t.ppf(0.975, df) * np.sqrt(va + vc))
    return estimate, estimate - margin, estimate + margin


def development_samples(baseline: pd.DataFrame, records: pd.DataFrame, horizon: int) -> dict[str, pd.Series]:
    """Select participant values without accepting reference outcomes.

    Parameters
    ----------
    baseline
        Unique enrolled identities and their programme enrollment clocks.
    records
        Observed measurements, or recurrent events timed from enrolment with an
        observation-end record.
    horizon
        Requested follow-up in days since each participant's enrollment.

    Returns
    -------
    dict of str to pandas.Series
        Named observed analyses and deliberately incorrect clock/visit controls.
        Empty extractions remain empty; no unavailable outcome becomes zero.

    Raises
    ------
    ValueError
        If identities, observation endings or scheduled measurements are ambiguous.
    """
    if not baseline.participant_id.is_unique or not set(records.participant_id) <= set(baseline.participant_id):
        raise ValueError("Development records require unique enrolled identities")
    enrollment = baseline.set_index("participant_id").enrollment_day
    if "event" in records:
        calendar = records.loc[~records.event.astype(bool)].set_index("participant_id").occurrence_time
        if not calendar.index.is_unique or set(calendar.index) != set(enrollment.index):
            raise ValueError("Recurrent records require one observation end per participant")
        counts = (
            records.loc[records.event.astype(bool)]
            .groupby("participant_id")
            .size()
            .reindex(enrollment.index, fill_value=0)
        )
        # Recurrent times are released as programme days; the programme-clock control
        # keeps the enrolment date deliberately.
        duration = calendar - enrollment.reindex(calendar.index)
        if (duration < 0).any():
            raise ValueError("Observation cannot end before enrollment")
        return {
            "observed_duration": 365 * counts.loc[duration.gt(0)] / duration.loc[duration.gt(0)],
            "full_horizon_denominator": 365 * counts / horizon,
            "programme_clock_denominator": 365 * counts.loc[calendar.gt(0)] / calendar.loc[calendar.gt(0)],
        }
    clock = records.participant_id.map(enrollment)
    selectors = {
        "scheduled_visit": records.occurrence_time.eq(clock + horizon),
        "baseline_as_followup": records.occurrence_time.eq(clock),
        "programme_clock_as_duration": records.occurrence_time.eq(horizon),
    }
    samples = {}
    for method, selected in selectors.items():
        values = records.loc[selected & records.observed & records.outcome_id.eq("clinical_outcome")]
        if not values.participant_id.is_unique:
            raise ValueError("A scheduled clinical measurement occurs more than once")
        samples[method] = values.observed_outcome.dropna()
    return samples


def potentials_file(horizon: int) -> str:
    """Name the disclosed complete-outcome table at one follow-up of the Day-336 RA examples."""
    return "terminal_potentials.csv" if horizon == 336 else f"potentials_{horizon}.csv"


def market_years(preferences: dict[str, Any], decision_day: float) -> float:
    """Discount one dollar a year from launch, one review after the decision, to exclusivity end."""
    launch = (decision_day + preferences["regulatory_review_days"]) / 365.25
    end = preferences["exclusivity_end_day"] / 365.25
    rate = np.log1p(preferences["discount_rate_per_year"])
    return max(0.0, float((np.exp(-rate * launch) - np.exp(-rate * end)) / rate))


def regulatory_approval(root: Path, ledger: dict[str, Any], rule: dict[str, Any]) -> bool:
    """Repeat the regulator's pooled Welch test on completed confirmation patients.

    Each confirmation study of the advanced treatment contributes the Welch
    difference in complete recorded outcomes of patients whose committed
    follow-up ended by the submission day. The recorded review must agree.
    """
    terminal = ledger["terminal"]
    enrolled = {row["study_id"]: row["patients"] for row in ledger["enrollments"]}
    weights, differences, counts = [], [], {"treated": 0, "control": 0}
    for study in ledger["recruitment_commitments"]:
        arms = {(arm["partition"]["asset_id"], arm["partition"]["dose"]): arm["arm_id"] for arm in study["arms"]}
        if study["phase"] != "confirmation" or (terminal["asset_id"], terminal["dose"]) not in arms:
            continue
        horizon = study["follow_up_days"]
        potentials = pd.read_csv(root / potentials_file(horizon), keep_default_na=False)
        completed = [
            p for p in enrolled[study["study_id"]] if p["enrollment_day"] + horizon <= terminal["programme_day"]
        ]
        groups = []
        for role, (asset, dose) in (
            ("treated", (terminal["asset_id"], terminal["dose"])),
            ("control", ("control", None)),
        ):
            ids = [p["participant_id"] for p in completed if p["arm_id"] == arms[(asset, dose)]]
            outcomes = potentials.loc[potentials.asset_id.eq(asset) & potentials.dose.eq(dose or "")]
            groups.append(outcomes.set_index("participant_id").target_outcome.loc[ids].to_numpy(dtype=float))
            counts[role] += len(ids)
        if min(map(len, groups)) >= 2:
            variance = sum(np.var(group, ddof=1) / len(group) for group in groups)
            weights.append(1 / variance)
            differences.append(groups[0].mean() - groups[1].mean())
    statistic = float(-np.dot(weights, differences) / np.sqrt(np.sum(weights))) if weights else None
    recorded = terminal["regulatory_review"]
    if (recorded["completed_treated"], recorded["completed_control"]) != (counts["treated"], counts["control"]):
        raise ValueError("Recorded regulatory review counts differ from the completed confirmation patients")
    if (statistic is None) != (recorded["benefit_z_statistic"] is None) or (
        statistic is not None and abs(statistic - recorded["benefit_z_statistic"]) > 1e-9
    ):
        raise ValueError("Recorded regulatory statistic differs from the completed confirmation outcomes")
    approved = (
        statistic is not None
        and statistic >= norm.ppf(1 - rule["two_sided_alpha"] / 2)
        and counts["treated"] >= rule["minimum_completed_treated"]
    )
    if approved != recorded["approved"]:
        raise ValueError("Recorded regulatory decision differs from the declared approval rule")
    return approved


def study_cost(study: dict[str, Any], enrollment: dict[str, Any]) -> float:
    """Price one study whose enrolled patients all finished follow-up and scheduled visits.

    This equation is not a general runtime emulator: it omits partial follow-up,
    dropped arms and early closure.
    """
    tariff = study["tariff"]
    visits = len(set(study["assessment_days"]) | set(study["surveillance_days"]))
    return float(
        tariff["setup"]
        + study["sites"] * tariff["per_site"]
        + enrollment["screened"] * tariff["screening"]
        + len(enrollment["patients"])
        * (tariff["enrollment"] + study["follow_up_days"] * tariff["followup_day"] + visits * tariff["assessment"])
    )


def programme_value(
    root: Path, ledger: dict[str, Any], preferences: dict[str, Any], studies_usd: float
) -> dict[str, float]:
    """Value a terminal choice from complete records, the approval rule and sponsor economics.

    An approved treatment earns its finite-population benefit less its safety cost
    over discounted exclusive market years; every decision bears study and overhead costs.
    """
    terminal = ledger["terminal"]
    day = terminal["programme_day"]
    approved, clinical, safety = False, 0.0, 0.0
    if terminal["decision"] == "advance":
        rule = json.loads((root / "design_space.json").read_text())["approval"]
        approved = regulatory_approval(root, ledger, rule)
    if approved:
        potentials = pd.read_csv(root / "terminal_potentials.csv", keep_default_na=False)
        means = potentials.groupby(["asset_id", "dose"])[["target_outcome", "safety_event"]].mean()
        selected = means.loc[(terminal["asset_id"], terminal["dose"])] - means.loc[("control", "")]
        years = market_years(preferences, day)
        clinical = (
            -selected.target_outcome
            / preferences["benefit_unit"]
            * preferences["annual_profit_per_benefit_unit_usd"]
            * years
        )
        safety = selected.safety_event * 100 * preferences["annual_cost_per_safety_percentage_point_usd"] * years
    delay = day * preferences["programme_cost_per_day_usd"]
    return {
        "approved": approved,
        "clinical_value_usd": clinical,
        "safety_cost_usd": safety,
        "study_cost_usd": studies_usd,
        "time_cost_usd": delay,
        "net_value_usd": clinical - safety - studies_usd - delay,
    }


def development_recovery(
    root: Path, output: Path, *, score: Callable[[float, float, float, float, float], float]
) -> None:
    """Compare repeated allocations and clock shortcuts in four disclosed finite pools."""
    rows = []
    with TemporaryDirectory() as temporary:
        with zipfile.ZipFile(root / "development-records.zip") as archive:
            archive.extractall(temporary)
        for profile in sorted(Path(temporary).iterdir()):
            metadata = json.loads((profile / "profile.json").read_text())
            population: dict[str, tuple[pd.Series, pd.DataFrame, bool]] = {}
            for arm in ("active", "control"):
                complete_root = profile / "complete" / arm
                ids = pd.read_parquet(complete_root / "baseline.parquet").participant_id
                recurrent_path = complete_root / "recurrent_truth.parquet"
                records = pd.read_parquet(
                    recurrent_path if recurrent_path.is_file() else complete_root / "complete_measurements.parquet"
                )
                population[arm] = (ids, records, recurrent_path.is_file())
            for run in sorted(path for path in profile.iterdir() if path.name.isdigit()):
                previous: tuple[str, list[pd.Series]] | None = None
                # Use actual published readout clocks, not alphabetic phase order.
                readouts = [
                    (path, json.loads((path / "study.json").read_text())) for path in (run / "readouts").iterdir()
                ]
                for readout, study in sorted(readouts, key=lambda item: item[1]["readout_day"]):
                    horizon = study["follow_up_days"]
                    target = next(
                        target
                        for target in study["targets"]
                        if target["endpoint"] == "clinical_outcome" and target["arm_id"] == "active"
                    )
                    references: list[float] = []
                    methods: dict[str, list[pd.Series]] = {}
                    enrollments: list[int] = []
                    for arm in ("active", "control"):
                        ids, complete, is_recurrent = population[arm]
                        if is_recurrent:
                            counts = (
                                complete.loc[complete.event.eq(1) & complete.event_time.le(horizon)]
                                .groupby("participant_id")
                                .size()
                            )
                            values = counts.reindex(ids, fill_value=0) * 365 / horizon
                        else:
                            scheduled = complete.loc[complete.occurrence_time.eq(horizon)]
                            if not scheduled.participant_id.is_unique or set(scheduled.participant_id) != set(ids):
                                raise ValueError("Finite teaching reference lacks unique complete scheduled outcomes")
                            values = scheduled.complete_recorded_outcome
                        references.append(float(values.mean()))
                        directory = readout / "arms" / arm
                        baseline = pd.read_parquet(directory / "baseline.parquet")
                        path = directory / ("recurrent_events.parquet" if is_recurrent else "measurements.parquet")
                        for method, sample in development_samples(baseline, pd.read_parquet(path), horizon).items():
                            methods.setdefault(method, []).append(sample)
                        enrollments.append(len(baseline))
                    reference = references[0] - references[1]
                    primary = "observed_duration" if population["active"][2] else "scheduled_visit"
                    if previous is not None:
                        methods["previous_readout"] = previous[1]
                    for method, samples in methods.items():
                        active, control = samples
                        point = float(active.mean() - control.mean()) if min(len(active), len(control)) else np.nan
                        lower = upper = loss = np.nan
                        status = "insufficient_observations"
                        if min(len(active), len(control)) >= 2:
                            status = "insufficient_variation"
                            if active.var(ddof=1) + control.var(ddof=1) > 0:
                                point, lower, upper = welch_interval(active, control)
                                # One native unit is an illustrative, not clinical, scale.
                                loss = score(point, lower, upper, reference, 1.0)
                                status = "estimated"
                        rows.append(
                            dict(
                                profile=profile.name,
                                world_id=metadata["world_id"],
                                repeat=int(run.name),
                                checkpoint=readout.name,
                                horizon=horizon,
                                target_id=target["target_id"],
                                unit=target["unit"],
                                method=method,
                                primary_method=primary,
                                source_checkpoint=(
                                    previous[0]
                                    if method == "previous_readout" and previous is not None
                                    else readout.name
                                ),
                                enrolled_active=enrollments[0],
                                enrolled_control=enrollments[1],
                                observed_active=len(active),
                                observed_control=len(control),
                                status=status,
                                estimate=point,
                                lower=lower,
                                upper=upper,
                                reference=reference,
                                error=point - reference,
                                loss=loss,
                            )
                        )
                    previous = (readout.name, methods[primary])
    results = pd.DataFrame(rows)
    results.to_csv(output / "trialdev_recovery_estimates.csv", index=False)
    summaries = []
    for (profile_id, checkpoint, summary_method), group in results.groupby(
        ["profile", "checkpoint", "method"], sort=False
    ):
        valid = group.loc[group.status.eq("estimated")]
        n, attempted = len(valid), len(group)
        covered = valid.lower.le(valid.reference) & valid.upper.ge(valid.reference)
        low, high = proportion_confint(int(covered.sum()), n, method="wilson") if n else (np.nan, np.nan)
        bias = float(valid.error.mean())
        half = float(t.ppf(0.975, n - 1) * valid.error.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        summaries.append(
            dict(
                profile=profile_id,
                checkpoint=checkpoint,
                method=summary_method,
                attempted=attempted,
                fitted=n,
                bias=bias,
                bias_lower=bias - half,
                bias_upper=bias + half,
                mean_width=(valid.upper - valid.lower).mean(),
                coverage=covered.mean(),
                coverage_lower=low,
                coverage_upper=high,
                all_attempt_coverage=covered.sum() / attempted,
                mean_loss=valid.loss.mean(),
            )
        )
    pd.DataFrame(summaries).to_csv(output / "trialdev_recovery_summary.csv", index=False)
    contrasts = []
    for (profile_id, checkpoint), group in results.groupby(["profile", "checkpoint"], sort=False):
        primary = group.primary_method.iloc[0]
        primary_losses = group.loc[group.method.eq(primary), ["repeat", "loss"]].rename(
            columns={"loss": "primary_loss"}
        )
        for contrast_method, frame in group.loc[group.method.ne(primary)].groupby("method", sort=False):
            paired = frame.merge(primary_losses, on="repeat", validate="one_to_one")
            difference = (paired.loss - paired.primary_loss).dropna()
            n = len(difference)
            mean = difference.mean()
            half = t.ppf(0.975, n - 1) * difference.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            contrasts.append(
                dict(
                    profile=profile_id,
                    checkpoint=checkpoint,
                    method=contrast_method,
                    primary_method=primary,
                    attempted=len(paired),
                    paired=n,
                    mean_loss_difference=mean,
                    lower=mean - half,
                    upper=mean + half,
                    shortcut_lower_loss=int(difference.lt(0).sum()),
                )
            )
    contrast_table = pd.DataFrame(contrasts)
    contrast_table.to_csv(output / "trialdev_recovery_contrasts.csv", index=False)
    final = contrast_table.loc[contrast_table.checkpoint.eq("confirmation-1") & contrast_table.paired.gt(0)]
    labels = {
        "baseline_as_followup": "Baseline for follow-up",
        "previous_readout": "Previous readout",
        "full_horizon_denominator": "Full planned duration",
        "programme_clock_denominator": "Wrong calendar denominator",
    }
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        figure, axes = plt.subplots(2, 2, figsize=(12, 5), layout="constrained")
        for ax, (profile_id, group) in zip(axes.flat, final.groupby("profile", sort=False), strict=True):
            plotted_mean, plotted_lower, plotted_upper = (
                group[["mean_loss_difference", "lower", "upper"]].to_numpy(dtype=float).T
            )
            ax.errorbar(
                plotted_mean,
                np.arange(len(group)),
                xerr=[plotted_mean - plotted_lower, plotted_upper - plotted_mean],
                fmt="o",
                color=COLOURS[0],
                capsize=3,
            )
            ax.axvline(0, color=".5", ls="--", lw=1)
            ax.set(
                title=str(profile_id).replace("_", " ").title().replace("Copd", "COPD"),
                yticks=range(len(group)),
                yticklabels=[labels[str(method)] for method in group.method],
                xlabel="Mean paired loss difference (shortcut − primary)",
            )
            ax.invert_yaxis()
        save(figure, output, "trialdev_recovery")


def recovery(root: Path, output: Path) -> None:
    """Calculate errors, interval coverage and reference sensitivity by study."""
    rows, references = [], []
    with TemporaryDirectory() as temporary:
        with zipfile.ZipFile(root / "recovery-records.zip") as archive:
            archive.extractall(temporary)
        for directory in sorted(Path(temporary).iterdir()):
            prefix, number, tier, state = directory.name.split("-", 3)
            series = prefix + "-" + number
            study = json.loads((directory / "study.json").read_text())
            question = study["estimands"][0]
            reference_draws: list[float] = []
            for index in range(32, 96):
                totals = (
                    pd.read_csv(directory / f"complete-{index}.csv").set_index("assigned_arm").to_dict(orient="index")
                )
                reference_draws.append(
                    float(totals["active"]["sum"]) / float(totals["active"]["count"])
                    - float(totals["control"]["sum"]) / float(totals["control"]["count"])
                )
            centre = float(np.mean(reference_draws))
            ref_se = float(np.std(reference_draws, ddof=1) / np.sqrt(len(reference_draws)))
            empirical, empirical_se = centre, ref_se
            if state == "null":
                centre, ref_se = 0.0, 0.0
            references.append(
                dict(
                    case=directory.name,
                    reference_group="TE-S04-A4-favourable" if "A4r" in directory.name else directory.name,
                    reference=centre,
                    reference_se=ref_se,
                    empirical_reference=empirical,
                    empirical_reference_se=empirical_se,
                    reference_worlds=len(reference_draws),
                    analysis_worlds=32,
                )
            )
            probability = (study.get("survival") or {}).get("horizon_recovery_probability", 0)
            for index in range(32):
                data = pd.read_parquet(directory / f"observed-{index}.parquet")
                for method, estimate, se in estimates(
                    data, series, question["endpoint_id"], question["assessment_time"], probability
                ):
                    if not np.isfinite([estimate, se]).all() or se <= 0:
                        raise ValueError("A recovery fit did not produce finite uncertainty")
                    width = norm.ppf(0.975) * se
                    rows.append(
                        dict(
                            case=directory.name,
                            series=series,
                            state=state,
                            seed=86_828_000 + index,
                            method=method,
                            estimate=estimate,
                            lower=estimate - width,
                            upper=estimate + width,
                            reference=centre,
                            reference_se=ref_se,
                            error=estimate - centre,
                            complete=True,
                            participants=len(data),
                        )
                    )
    results = pd.DataFrame(rows)
    results.to_csv(output / "repeated_study_estimates.csv", index=False)
    pd.DataFrame(references).to_csv(output / "population_references.csv", index=False)
    summary_rows = []
    for (case, group_method), group in results.groupby(["case", "method"], sort=False):
        n = len(group)
        reference, ref_se = group.reference.iloc[0], group.reference_se.iloc[0]
        covered = group.lower.le(reference) & group.upper.ge(reference)
        low, high = proportion_confint(int(covered.sum()), n, alpha=0.05, method="wilson")
        shift = t.ppf(0.975, 63) * ref_se
        # Extremes over the whole reference interval, not just its endpoints.
        breaks = np.r_[group.lower.to_numpy(), group.upper.to_numpy()]
        grid = np.r_[
            reference - shift, reference + shift, breaks, np.nextafter(breaks, -np.inf), np.nextafter(breaks, np.inf)
        ]
        grid = grid[(grid >= reference - shift) & (grid <= reference + shift)]
        coverage = [float((group.lower.le(value) & group.upper.ge(value)).mean()) for value in grid]
        error_se = np.sqrt(group.error.var(ddof=1) / n + ref_se**2)
        summary_rows.append(
            dict(
                case=case,
                method=group_method,
                completed=n,
                attempted=32,
                bias=group.error.mean(),
                bias_lower=group.error.mean() - t.ppf(0.975, n - 1) * error_se,
                bias_upper=group.error.mean() + t.ppf(0.975, n - 1) * error_se,
                rmse=np.sqrt(np.mean(group.error**2)),
                mean_interval_width=(group.upper - group.lower).mean(),
                coverage=covered.mean(),
                coverage_lower=low,
                coverage_upper=high,
                reference_coverage_min=min(coverage),
                reference_coverage_max=max(coverage),
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "recovery_operating_characteristics.csv", index=False)
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained", gridspec_kw={"height_ratios": [1, 1.7]})
        for ax, series, title, unit in zip(
            axes[0],
            ("TE-S10", "TE-S04"),
            ("A  Disability outcome", "B  Exacerbation risk"),
            ("Active − control (score points)", "Active − control (risk difference)"),
            strict=True,
        ):
            states = (
                ("favourable", "near_null", "null", "unfavourable")
                if series == "TE-S10"
                else ("favourable", "near_null", "unfavourable")
            )
            tier = "A1" if series == "TE-S10" else "A4"
            for x, state in enumerate(states):
                group = results.loc[results.case.eq(f"{series}-{tier}-{state}")]
                ref = group.reference.iloc[0]
                ref_se = group.reference_se.iloc[0]
                ax.errorbar(
                    x,
                    ref,
                    yerr=t.ppf(0.975, 63) * ref_se,
                    fmt="D",
                    color="black",
                    capsize=4,
                    label="Population reference" if x == 0 else None,
                )
                for offset, (plot_method, values) in enumerate(group.groupby("method", sort=False)):
                    centre = float(values.estimate.to_numpy(dtype=float).mean())
                    interval = t.ppf(0.975, len(values) - 1) * float(
                        values.estimate.to_numpy(dtype=float).std(ddof=1) / np.sqrt(len(values))
                    )
                    ax.errorbar(
                        x + 0.14 * (offset + 1),
                        centre,
                        yerr=interval,
                        fmt="o" if offset == 0 else "s",
                        color=COLOURS[offset],
                        capsize=3,
                        label=str(plot_method) if x == 0 else None,
                    )
            ax.axhline(0, color=".7", lw=0.8)
            names = {
                "favourable": "Favourable",
                "near_null": "Zero direct\ncomponent",
                "null": "Equal-arm\nnull",
                "unfavourable": "Unfavourable",
            }
            ax.set(xticks=range(len(states)), xticklabels=[names[state] for state in states], ylabel=unit, title=title)
            ax.legend(fontsize=7, loc="best")
        # Mechanism comparisons at one fixed effect setting keep the lower panels readable.
        selected = summary.loc[
            summary.case.str.endswith("favourable")
            & ~summary.case.str.endswith("unfavourable")
            & ~(summary.case.str.contains("A4r") & summary.method.eq("Crude event risk"))
        ].reset_index(drop=True)
        labels = []
        short = {"TE-S10": "Observation", "TE-S04": "Retrieval", "TE-S15": "Validation", "TE-S07": "Clustering"}
        for index, (_, row) in enumerate(selected.iterrows()):
            label = f"{short[row['case'][:6]]}: {row['method']}"
            if row["method"] == "Randomized retrieval":
                probability = "25%" if "r25" in row["case"] else "100%" if "r100" in row["case"] else "50%"
                label = "Retrieval: " + probability + " of missing outcomes"
            labels.append(label)
            scale = results.loc[results.case.eq(row["case"])].reference.abs().iloc[0]
            # Scale only the plot, retaining original clinical units in the CSV.
            axes[1, 0].errorbar(
                row.bias / scale,
                index,
                xerr=[[(row.bias - row.bias_lower) / scale], [(row.bias_upper - row.bias) / scale]],
                fmt="o",
                color=COLOURS[0],
                capsize=3,
            )
            axes[1, 1].errorbar(
                row.coverage,
                index,
                xerr=[[row.coverage - row.coverage_lower], [row.coverage_upper - row.coverage]],
                fmt="o",
                color=COLOURS[0],
                capsize=3,
            )
            axes[1, 1].plot(
                [row.reference_coverage_min, row.reference_coverage_max],
                [index, index],
                color=COLOURS[1],
                lw=4,
                alpha=0.7,
            )
        axes[1, 0].axvline(0, color=".6", lw=0.8)
        axes[1, 0].set(
            title="C  Bias relative to effect magnitude",
            xlabel="Mean error / |population effect|",
            yticks=range(len(labels)),
            yticklabels=labels,
        )
        axes[1, 1].axvline(0.95, color=".4", ls="--", lw=1)
        axes[1, 1].set(
            title="D  Coverage of nominal 95% intervals",
            xlabel="Fraction covering the population reference",
            xlim=(0, 1.02),
            yticks=range(len(labels)),
            yticklabels=[],
        )
        for ax in axes[1]:
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.2)
        save(fig, output, "repeated_study_recovery")


def design_display(root: Path, output: Path) -> None:
    """Display baseline and dependence recovery for the disclosed clinical-cell census."""
    marginals = pd.read_csv(output / "families/clinical_baseline_marginals.csv")
    dependence = pd.read_csv(output / "families/clinical_baseline_dependence.csv")
    marginals["mean_shift_sd"] = (
        marginals.observed_mean - marginals.requested_mean_before_recording
    ) / marginals.requested_sd_before_recording
    marginals["sd_ratio"] = marginals.observed_sd / marginals.requested_sd_before_recording
    marginals.to_csv(output / "baseline_recovery.csv", index=False)
    rows, rollout, clusters = [], None, None
    with zipfile.ZipFile(root / "family-records.zip") as archive:
        cells = pd.read_csv(archive.open("cells.csv"))
        for case in cells.itertuples(index=False):
            data = pd.read_parquet(archive.open(f"{case.cell}/analysis.parquet"))
            participants = data.drop_duplicates("participant_id")
            endpoint = "horizon_endpoint" if "horizon_endpoint" in data else case.endpoint
            observed = data[endpoint].notna().mean() if endpoint in data else 1.0
            rows.append(
                dict(
                    case=case.cell,
                    participants=len(participants),
                    active_fraction=participants.assigned_arm.eq("active").mean(),
                    observed_fraction=observed,
                    event_fraction=data.event_observed.mean() if "event_observed" in data else np.nan,
                    median_followup=data.observed_time.median() if "observed_time" in data else np.nan,
                    horizon=case.horizon,
                )
            )
            if case.cell == "TE-S08-A2":
                rollout = (
                    data.assign(active=data.assigned_arm.eq("active"))
                    .groupby("calendar_period")
                    .agg(active_fraction=("active", "mean"), participants=("participant_id", "size"))
                )
            if case.cell == "TE-S07-A1":
                clusters = data.groupby("cluster_id").size()
    census = pd.DataFrame(rows)
    census.to_csv(output / "study_design_census.csv", index=False)
    if rollout is None or clusters is None:
        raise ValueError("The design census lacks its cluster and rollout examples")
    rollout.to_csv(output / "calendar_rollout.csv")
    clusters.rename("participants").to_csv(output / "cluster_sizes.csv")
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(12, 5), layout="constrained")
        axes[0, 0].scatter(marginals.mean_shift_sd, marginals.sd_ratio, s=15, alpha=0.55, color=COLOURS[0])
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=4))
        axes[0, 0].axvline(0, color=".6", lw=0.8)
        axes[0, 0].axhline(1, color=".6", lw=0.8)
        axes[0, 0].set(
            title="A  Baseline measurements", xlabel="Mean departure / target SD", ylabel="Observed SD / target SD"
        )
        axes[0, 1].scatter(
            dependence.target_spearman_before_recording,
            dependence.observed_spearman,
            s=15,
            alpha=0.55,
            color=COLOURS[0],
        )
        axes[0, 1].plot([-0.6, 0.6], [-0.6, 0.6], color=".6", lw=0.8)
        axes[0, 1].set(
            title="B  Baseline dependence",
            xlabel="Spearman reference before recording",
            ylabel="Recorded sample Spearman correlation",
        )
        axes[0, 2].scatter(census.participants, census.active_fraction, s=22, color=COLOURS[0])
        axes[0, 2].axhline(0.5, color=".6", lw=0.8)
        axes[0, 2].set(
            title="C  Treatment allocation",
            xlabel="Participants per study",
            ylabel="Proportion assigned active",
            ylim=(0, 1),
        )
        axes[1, 0].hist(clusters, bins=12, color=COLOURS[0], edgecolor="white")
        axes[1, 0].set(title="D  Cluster randomization", xlabel="Participants per facility", ylabel="Facilities")
        axes[1, 1].plot(rollout.index, rollout.active_fraction, marker="o", color=COLOURS[0])
        axes[1, 1].set(
            title="E  Calendar rollout",
            xlabel="Enrollment period",
            ylabel="Proportion entering on active",
            ylim=(-0.03, 1.03),
        )
        survival = census.dropna(subset=["event_fraction"])
        axes[1, 2].scatter(
            survival.median_followup / survival.horizon, survival.event_fraction, s=22, color=COLOURS[0]
        )
        axes[1, 2].set(
            title="F  Follow-up and observed events",
            xlabel="Median follow-up / endpoint horizon",
            ylabel="Observed event proportion",
            ylim=(0, 1),
        )
        save(fig, output, "study_design_recovery")
