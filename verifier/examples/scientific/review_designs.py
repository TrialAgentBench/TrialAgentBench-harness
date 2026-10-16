"""Recalculate paired design and policy results from participant records and ledgers."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from review_recovery import COLOURS, mean_contrast, potentials_file, programme_value, save, study_cost
from scipy.stats import t


def compare_designs(root: Path, output: Path) -> None:
    """Reconstruct estimates, resource charges and terminal value without the grader."""
    study_rows, programme_rows, preference_rows = [], [], []
    with TemporaryDirectory() as temporary:
        with zipfile.ZipFile(root / "design-records.zip") as archive:
            archive.extractall(temporary)
        for world in sorted(Path(temporary).iterdir()):
            identity = pd.read_csv(world / "world.csv", keep_default_na=False).iloc[0]
            preferences = json.loads((world / "preferences.json").read_text())
            preference_rows.append({"world": world.name, **preferences})
            target_sites = pd.read_parquet(world / "target_population.parquet").site_id.value_counts(normalize=True)
            terminal = pd.read_csv(world / "terminal_potentials.csv", keep_default_na=False)
            targets = terminal.groupby(["asset_id", "dose"])[["target_outcome", "safety_event"]].mean()
            target = targets.loc[(identity.asset, identity.dose)] - targets.loc[("control", "")]
            potentials = {
                horizon: pd.read_csv(world / potentials_file(horizon), keep_default_na=False)
                .groupby(["asset_id", "dose"])
                .target_outcome.mean()
                for horizon in (28, 84, 168, 336)
            }
            for run in sorted(world.iterdir()):
                if not run.is_dir():
                    continue
                design, repeat, policy = run.name.split("-")
                ledger = json.loads((run / "ledger.json").read_text())
                native = json.loads((run / "native_value.json").read_text())
                decisions = pd.read_csv(run / "decisions.csv").set_index("phase")
                cost, enrolled = 0.0, 0
                for study in ledger["recruitment_commitments"]:
                    study_id, horizon = study["study_id"], study["follow_up_days"]
                    enrollment = next(row for row in ledger["enrollments"] if row["study_id"] == study_id)
                    n = len(enrollment["patients"])
                    enrolled += n
                    if any(
                        patient["enrollment_day"] + horizon > ledger["terminal"]["programme_day"]
                        for patient in enrollment["patients"]
                    ):
                        raise ValueError("This completed-follow-up cost derivation cannot value unfinished follow-up")
                    cost += study_cost(study, enrollment)
                    samples, arms, safety, recruited_sites = [], [], [], []
                    arm_counts, infection_counts, arm_variances = [], [], []
                    for arm in ("active", "control"):
                        directory = run / "readouts" / f"{study_id}-look" / "arms" / arm
                        baseline = pd.read_parquet(directory / "baseline.parquet")
                        recruited_sites.extend(baseline.site_id.to_list())
                        data = pd.read_parquet(directory / "measurements.parquet").merge(
                            baseline[["participant_id", "enrollment_day"]], validate="many_to_one"
                        )
                        values = data.loc[
                            data.occurrence_time.eq(data.enrollment_day + horizon) & data.observed,
                            "observed_outcome",
                        ].dropna()
                        events = pd.read_parquet(directory / "safety_events.parquet")
                        safety.append(events.participant_id.nunique() / len(baseline))
                        arm_counts.append(len(values))
                        infection_counts.append(events.participant_id.nunique())
                        arm_variances.append(values.var(ddof=1) / len(values))
                        samples.extend(values.to_list())
                        arms.extend([arm] * len(values))
                    estimate, se = mean_contrast(pd.Series(samples), pd.Series(arms))
                    degrees = sum(arm_variances) ** 2 / sum(
                        variance**2 / (count - 1) for variance, count in zip(arm_variances, arm_counts, strict=True)
                    )
                    half_width = float(t.ppf(0.975, degrees)) * se
                    site_shares = pd.Series(recruited_sites).value_counts(normalize=True)
                    site_distance = site_shares.sub(target_sites, fill_value=0).abs().sum() / 2
                    decision = decisions.loc[study["phase"]].to_dict()
                    np.testing.assert_allclose(estimate, decision["clinical_difference"], atol=1e-12)
                    np.testing.assert_allclose(safety[0] - safety[1], decision["safety_difference"], atol=1e-12)
                    expected_stop = policy == "responsive" and (estimate >= 0 or safety[0] - safety[1] > 0.05)
                    if bool(decision["stop"]) != expected_stop:
                        raise ValueError("The policy decision disagrees with its observed-evidence rule")
                    potential = potentials[horizon]
                    reference = potential.loc[(identity.asset, identity.dose)] - potential.loc[("control", "")]
                    study_rows.append(
                        dict(
                            world=world.name,
                            candidates=identity.candidates,
                            benefit_class=identity.benefit_class,
                            design=design,
                            repeat=int(repeat),
                            policy=policy,
                            phase=study["phase"],
                            horizon=horizon,
                            patients=n,
                            activated_sites=study["sites"],
                            represented_sites=len(site_shares),
                            target_sites=len(target_sites),
                            site_total_variation=site_distance,
                            observed=len(samples),
                            estimate=estimate,
                            standard_error=se,
                            lower=estimate - half_width,
                            upper=estimate + half_width,
                            treated_observed=arm_counts[0],
                            control_observed=arm_counts[1],
                            treated_infections=infection_counts[0],
                            control_infections=infection_counts[1],
                            reference=reference,
                            error=estimate - reference,
                            cumulative_cost=cost,
                            day=study["looks"][0]["deadline_day"],
                            budget_usd=ledger["budget_usd"],
                            programme_horizon_days=ledger["horizon_days"],
                        )
                    )
                value = programme_value(world, ledger, preferences, cost)
                for key, recalculated in value.items():
                    np.testing.assert_allclose(recalculated, native[key], atol=1e-6, rtol=0, err_msg=key)
                programme_rows.append(
                    dict(
                        world=world.name,
                        candidates=identity.candidates,
                        benefit_class=identity.benefit_class,
                        design=design,
                        repeat=int(repeat),
                        policy=policy,
                        decision=ledger["terminal"]["decision"],
                        approved=value["approved"],
                        outcome_difference=target.target_outcome,
                        safety_difference=target.safety_event,
                        patients=enrolled,
                        cost_usd=cost,
                        elapsed_days=ledger["terminal"]["programme_day"],
                        benefit_usd=value["clinical_value_usd"],
                        safety_cost_usd=value["safety_cost_usd"],
                        delay_cost_usd=value["time_cost_usd"],
                        value_usd=value["net_value_usd"],
                        complete=True,
                    )
                )
    studies, programmes = pd.DataFrame(study_rows), pd.DataFrame(programme_rows)
    studies.to_csv(output / "authored_study_estimates.csv", index=False)
    programmes.to_csv(output / "authored_programme_values.csv", index=False)
    pd.DataFrame(preference_rows).to_csv(output / "authored_utility_preferences.csv", index=False)
    summaries = []
    confirmation = studies.loc[studies.policy.eq("fixed") & studies.phase.eq("confirmation")]
    for (world_key, design_key), group in confirmation.groupby(["world", "design"]):
        summaries.append(
            dict(
                world=str(world_key),
                design=str(design_key),
                candidates=group.candidates.iloc[0],
                benefit_class=group.benefit_class.iloc[0],
                repeats=len(group),
                patients=group.patients.mean(),
                horizon=group.horizon.iloc[0],
                mean_se=group.standard_error.mean(),
                rmse=np.sqrt(np.mean(group.error**2)),
                cost_million_usd=group.cumulative_cost.mean() / 1e6,
            )
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "authored_design_summary.csv", index=False)
    means = programmes.groupby(["world", "design", "policy"], as_index=False).agg(
        value_usd=("value_usd", "mean"), elapsed_days=("elapsed_days", "mean"), cost_usd=("cost_usd", "mean")
    )
    means.to_csv(output / "authored_policy_summary.csv", index=False)
    shapes = {"balanced": "o", "larger": "s", "allocation": "^", "longer": "D", "staged": "P"}
    labels = {
        "balanced": "300:300, 168 days",
        "larger": "600:600, 168 days",
        "allocation": "300:600, 168 days",
        "longer": "300:300, 336 days",
        "staged": "450:450, 168 days; 6 early sites",
    }
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
        for _, matched in summary.loc[summary.horizon.eq(168)].groupby("world"):
            ordered = matched.sort_values("patients")
            axes[0].plot(ordered.patients, ordered.mean_se, color=".7", lw=0.7, zorder=0)
            axes[1].plot(ordered.cost_million_usd, ordered.rmse, color=".7", lw=0.7, zorder=0)
        for design_key, group in summary.groupby("design"):
            design = str(design_key)
            # Each symbol represents one of the six fixed worlds, averaged over
            # two matched allocation seeds. No population inference is implied.
            axes[0].scatter(
                group.patients,
                group.mean_se,
                marker=shapes[design],
                label=labels[design],
                s=42,
                facecolors="none" if design == "longer" else COLOURS[0],
                edgecolors=COLOURS[0],
            )
            axes[1].scatter(
                group.cost_million_usd,
                group.rmse,
                marker=shapes[design],
                s=42,
                facecolors="none" if design == "longer" else COLOURS[0],
                edgecolors=COLOURS[0],
            )
        axes[0].set(
            title="A  Acquired precision",
            xlabel="Confirmation participants",
            ylabel="Mean estimated SE (DAS28 points)",
        )
        axes[0].legend(fontsize=7, loc="best")
        axes[1].set(
            title="B  Resources and estimation error",
            xlabel="Total study expenditure (million USD)",
            ylabel="RMSE at the studied horizon (DAS28 points)",
        )
        index = 0
        ticklabels = []
        for _world, group in programmes.loc[programmes.design.eq("balanced")].groupby("world", sort=False):
            ident = group.iloc[0]
            ticklabels.append(f"{int(ident.candidates)} candidates; ΔDAS28 {ident.outcome_difference:+.2f}")
            for shift, policy, colour, marker in (
                (-0.12, "fixed", COLOURS[0], "o"),
                (0.12, "responsive", COLOURS[1], "s"),
            ):
                values = group.loc[group.policy.eq(policy)].value_usd / 1e6
                axes[2].plot([values.min(), values.max()], [index + shift, index + shift], color=colour, lw=1)
                axes[2].scatter(
                    values.mean(),
                    index + shift,
                    color=colour,
                    marker=marker,
                    label=policy.title() if index == 0 else None,
                    s=30,
                )
            index += 1
        axes[2].axvline(0, color=".4", ls="--", lw=1, label="Immediate stop")
        axes[2].set(
            title="C  Decisions after acquiring evidence",
            xlabel="Net programme value (million USD)",
            yticks=range(index),
            yticklabels=ticklabels,
        )
        axes[2].legend(fontsize=7, loc="best")
        axes[2].invert_yaxis()
        save(fig, output, "authored_design_utility")
    clinical_world_figure(root, output)
    programme_decisions(output / "authored_programme_values.csv", output)
    utility_sensitivity(output / "authored_programme_values.csv", output / "authored_utility_preferences.csv", output)


def clinical_world_figure(root: Path, output: Path) -> None:
    """Derive the arthritis world's day-28 target beside its observed pilot.

    Parameters
    ----------
    root
        Public scientific example directory containing design-records.zip.
    output
        Directory for the exact four-row comparison and editable diagrams.

    Raises
    ------
    ValueError
        If the complete treatment conditions do not contain the same people or
        the observed pilot has duplicate outcomes at its stated follow-up.
    """
    world = "tdw-70ff3604b8a7"
    rows = []
    with zipfile.ZipFile(root / "design-records.zip") as archive:
        identity = pd.read_csv(archive.open(f"{world}/world.csv"), keep_default_na=False).iloc[0]
        potential = pd.read_csv(archive.open(f"{world}/potentials_28.csv"), keep_default_na=False)
        treated = potential.loc[potential.asset_id.eq(identity.asset) & potential.dose.eq(identity.dose)]
        control = potential.loc[potential.asset_id.eq("control")]
        if (
            not treated.participant_id.is_unique
            or not control.participant_id.is_unique
            or set(treated.participant_id) != set(control.participant_id)
        ):
            raise ValueError("The world comparison must use the same unique people under both treatments")
        for arm, complete in (("treatment", treated), ("control", control)):
            rows.append(
                dict(
                    view="complete population",
                    arm=arm,
                    participants=len(complete),
                    follow_up_days=28,
                    mean=float(complete.target_outcome.mean()),
                )
            )
            record_arm = "active" if arm == "treatment" else "control"
            directory = f"{world}/staged-1-fixed/readouts/early_safety-look/arms/{record_arm}"
            baseline = pd.read_parquet(archive.open(f"{directory}/baseline.parquet"))
            measured = pd.read_parquet(archive.open(f"{directory}/measurements.parquet")).merge(
                baseline[["participant_id", "enrollment_day"]], validate="many_to_one"
            )
            observed = measured.loc[
                np.isclose(measured.occurrence_time - measured.enrollment_day, 28) & measured.observed
            ]
            if not observed.participant_id.is_unique:
                raise ValueError("The pilot requires one observed endpoint per person at day 28")
            rows.append(
                dict(
                    view="observed pilot",
                    arm=arm,
                    participants=len(observed),
                    follow_up_days=28,
                    mean=float(observed.observed_outcome.mean()),
                )
            )
    values = pd.DataFrame(rows)
    means = values.pivot(index="view", columns="arm", values="mean")
    t_mean, c_mean = map(float, means.loc[["complete population"], ["treatment", "control"]].to_numpy(dtype=float)[0])
    pt_mean, pc_mean = map(float, means.loc[["observed pilot"], ["treatment", "control"]].to_numpy(dtype=float)[0])
    target, estimate = t_mean - c_mean, pt_mean - pc_mean
    values["treatment_minus_control"] = values.view.map({"complete population": target, "observed pilot": estimate})
    output.mkdir(parents=True, exist_ok=True)
    values.to_csv(output / "arthritis-world.csv", index=False)
    population = len(treated)
    pilot_n = int(values.loc[values.view.eq("observed pilot"), "participants"].sum())
    style = """<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10" fill="#536370"/></marker></defs>
<style>text{font-family:Arial,"Liberation Sans",sans-serif;font-size:20px;fill:#23323d}.title{font-weight:700;font-size:23px}.small{font-size:18px;fill:#536370}.reference{fill:#79552a}.observed{fill:#226879}.path{stroke:#536370;stroke-width:1.8;fill:none;marker-end:url(#arrow)}.rule{stroke:#536370;stroke-width:1.4;fill:none}.panel{fill:none;stroke:#c8d1d7;stroke-width:1.2}</style>"""
    description = (
        f"The same {population:,} simulated people have day-28 outcomes under candidate treatment and control. "
        f"Their mean difference is {target:+.3f} arthritis-activity points. A recruited pilot reveals each "
        f"participant's assigned-treatment outcome; its observed difference is {estimate:+.3f}. "
        "Patient characteristics, treatment, follow-up time and random variation determine outcomes. "
        "Lower disease activity is better. The comparison does not identify why the pilot differs."
    )
    desktop = f"""<g id="model"><text class="title" x="20" y="35">One clinical world</text>
<rect class="panel" x="20" y="70" width="280" height="222" rx="3"/>
<text class="title" x="38" y="105">{population:,} simulated people</text>
<text x="38" y="142">Patient characteristics</text><text x="38" y="175">+ treatment + follow-up time</text>
<path class="path" d="M150 190V216"/><text class="small" x="38" y="244">Specified relationships</text><text class="small" x="38" y="272">and random variation</text>
<text class="small" x="20" y="325">A structured causal model</text></g>
<path class="rule" d="M300 151H311V92H607"/><path class="path" d="M607 92H443V102"/><path class="path" d="M607 92H771V102"/><path class="path" d="M285 292V431H318"/>
<g id="evaluator"><rect class="panel" x="322" y="14" width="618" height="290" rx="3"/>
<text class="title reference" x="342" y="48">Evaluator: both treatment conditions</text>
<text class="small" x="342" y="79">The same {population:,} people, assessed after 28 days</text>
<text x="360" y="122">Candidate treatment</text><text x="703" y="122">Control</text>
<text class="title" x="360" y="161">Mean {t_mean:.3f}</text><text class="title" x="703" y="161">Mean {c_mean:.3f}</text>
<path class="rule" d="M443 178V197H771V178M607 197V216"/>
<text class="title reference" x="607" y="247" text-anchor="middle">{t_mean:.3f} − {c_mean:.3f} = {target:+.3f}</text>
<text class="small" x="607" y="280" text-anchor="middle">Population treatment effect</text></g>
<g id="observed"><rect class="panel" x="322" y="347" width="618" height="229" rx="3"/>
<text class="title observed" x="342" y="382">Agent: recruited participants and collected records</text>
<text x="342" y="419">Recruit {pilot_n} people; randomly assign 10 to each treatment</text>
<text x="342" y="451">Observe their assigned-treatment outcome after 28 days</text>
<path class="path" d="M607 467V490"/>
<text class="title observed" x="607" y="522" text-anchor="middle">{pt_mean:.3f} − {pc_mean:.3f} = {estimate:+.3f}</text>
<text class="small" x="607" y="553" text-anchor="middle">Observed pilot estimate</text></g>
<text class="small" x="20" y="620">Both comparisons use day-28 arthritis activity (DAS28). Lower is better.</text>"""
    compact = f"""<style>text{{font-size:17px}}.title{{font-size:19px}}.small{{font-size:15px}}</style>
<g id="model"><text class="title" x="12" y="29">One clinical world</text>
<rect class="panel" x="12" y="48" width="326" height="161" rx="3"/>
<text class="title" x="27" y="79">{population:,} simulated people</text><text x="27" y="111">Patient characteristics, treatment</text><text x="27" y="136">and follow-up time</text>
<text class="small" x="27" y="170">Specified relationships + random variation</text><text class="small" x="27" y="194">A structured causal model</text></g>
<path class="path" d="M175 209V252"/>
<g id="evaluator"><rect class="panel" x="12" y="256" width="326" height="260" rx="3"/>
<text class="title reference" x="27" y="288">Evaluator: both treatments</text>
<text class="small" x="27" y="316">Same {population:,} people; 28-day follow-up</text>
<path class="path" d="M175 327H74V334"/><path class="path" d="M175 327H264V334"/>
<text x="27" y="354">Candidate</text><text x="220" y="354">Control</text>
<text class="title" x="27" y="384">{t_mean:.3f}</text><text class="title" x="220" y="384">{c_mean:.3f}</text>
<text class="small" x="27" y="405">Mean activity</text><text class="small" x="220" y="405">Mean activity</text>
<path class="rule" d="M74 417V434H264V417M175 434V450"/>
<text class="title reference" x="175" y="477" text-anchor="middle">Difference {target:+.3f}</text>
<text class="small" x="175" y="503" text-anchor="middle">Population treatment effect</text></g>
<path class="path" d="M338 128H346V653H337"/>
<g id="observed"><rect class="panel" x="12" y="557" width="318" height="255" rx="3"/>
<text class="title observed" x="27" y="590">Agent: collected records</text>
<text x="27" y="626">Recruit {pilot_n}; randomly assign</text><text x="27" y="651">10 to treatment and 10 to control</text>
<text x="27" y="687">Observe each assigned treatment</text><text x="27" y="712">after 28 days</text>
<path class="path" d="M175 728V748"/>
<text class="title observed" x="175" y="778" text-anchor="middle">{pt_mean:.3f} − {pc_mean:.3f} = {estimate:+.3f}</text>
<text class="small" x="175" y="801" text-anchor="middle">Observed pilot estimate</text></g>
<text class="small" x="12" y="850">Arthritis activity (DAS28): lower is better.</text>"""
    for name, viewbox, content in (
        ("arthritis-world.svg", "0 0 960 644", desktop),
        ("arthritis-world-compact.svg", "0 0 350 875", compact),
    ):
        (output / name).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" role="img" aria-labelledby="title desc">\n'
            f'<title id="title">How the world defines a reference independently of the agent</title><desc id="desc">{description}</desc>\n'
            + style
            + "\n"
            + content
            + "\n</svg>\n",
            encoding="utf-8",
        )


def programme_decisions(programmes_path: Path, output: Path) -> None:
    """Compare three studies and their uncertainty in two saved clinical worlds.

    Parameters
    ----------
    programmes_path
        Programme results reconstructed from the disclosed patient records.
    output
        Destination for editable figures and their exact programme/study rows.

    Raises
    ------
    ValueError
        If the selected matched programmes or three-study trajectories are absent.
    """
    worlds = ("tdw-70ff3604b8a7", "tdw-84fc2dcd0306")
    programmes = pd.read_csv(programmes_path)
    selected = programmes.loc[
        programmes.world.isin(worlds) & programmes.design.eq("staged") & programmes.repeat.eq(1)
    ].copy()
    studies = pd.read_csv(programmes_path.with_name("authored_study_estimates.csv"))
    studies = studies.loc[
        studies.world.isin(worlds) & studies.design.eq("staged") & studies.repeat.eq(1) & studies.policy.eq("fixed")
    ].copy()
    if len(selected) != 4 or selected.duplicated(["world", "policy"]).any() or len(studies) != 6:
        raise ValueError("The decision figure requires two matched programme pairs and their three studies")
    selected.to_csv(output / "programme-decisions.csv", index=False)
    studies.to_csv(output / "programme-studies.csv", index=False)
    stages = (
        ("early_safety", "Pilot", "Both uncertain"),
        ("exploration", "Exploration", "A favours treatment; B uncertain"),
        ("confirmation", "Confirmation", "Opposite directions"),
    )
    for compact in (False, True):
        width, height = (350, 886) if compact else (960, 660)
        plot_left, plot_right = (26, 326) if compact else (350, 924)
        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">',
            '<title id="title">What later studies reveal in two arthritis worlds</title>',
            '<desc id="description">Programme A and Programme B use the same study plan in different simulated '
            "worlds. Both pilots are uncertain. Exploration favours treatment in A while B remains uncertain. "
            "Confirmation favours treatment in A and control in B. Each point is a treatment-minus-control "
            "difference with a 95% Welch interval. The three studies have different follow-up horizons.</desc>",
            '<style>text{font-family:Arial,"Liberation Sans",sans-serif;fill:#23323d;'
            f"font-size:{15 if compact else 18}px}}"
            f".heading{{font-size:{21 if compact else 24}px;font-weight:700}}"
            f".stage{{font-size:{18 if compact else 21}px;font-weight:700}}"
            ".strong{font-weight:700}.rule{stroke:#c8d1d7;stroke-width:1}"
            ".zero{stroke:#78858d;stroke-width:1.2;stroke-dasharray:4 3}"
            ".interval{stroke-width:3}.estimate{stroke:white;stroke-width:1.5}</style>",
        ]

        def label(x: float, y: float, content: str, css: str = "", elements: list[str] = elements) -> None:
            elements.append(f'<text x="{x}" y="{y}" class="{css}">{content}</text>')

        def axis(value: float, left: float = plot_left, right: float = plot_right) -> float:
            return left + (value + 1.6) / 3.7 * (right - left)

        def signed(value: float) -> str:
            return f"{value:+.3f}".replace("-", "−")

        label(14, 28, "What later studies reveal", "heading")
        if compact:
            label(14, 55, "Treatment minus control")
            label(14, 76, "Arthritis activity (DAS28 points)")
            label(14, 101, "Negative: favours treatment")
            label(14, 122, "Positive: favours control")
            label(14, 143, "Lines: 95% Welch intervals")
        else:
            label(14, 59, "Treatment minus control · arthritis activity (DAS28 points)")
            label(14, 87, "Negative favours treatment; positive favours control. Lines: 95% Welch intervals.")
        for index, (phase, title, interpretation) in enumerate(stages):
            y = (182 + index * 240) if compact else (132 + index * 180)
            clinical = studies.loc[studies.phase.eq(phase)].set_index("world")
            first = clinical.loc[[worlds[0]]].iloc[0]
            if not clinical.patients.eq(first.patients).all() or not clinical.horizon.eq(first.horizon).all():
                raise ValueError("The paired study display requires a shared sample size and follow-up per stage")
            if compact:
                label(14, y, f"{title} · {int(first.patients):,} participants", "stage")
                label(14, y + 23, f"{int(first.horizon)}-day patient follow-up")
                label(14, y + 46, interpretation)
                first_row, row_gap = y + 97, 51
            else:
                label(14, y, title, "stage")
                label(14, y + 29, f"{int(first.patients):,} participants")
                label(14, y + 54, f"{int(first.horizon)}-day patient follow-up")
                if phase == "exploration":
                    label(14, y + 84, "A favours treatment;")
                    label(14, y + 108, "B uncertain")
                else:
                    label(14, y + 84, interpretation)
                first_row, row_gap = y + 21, 51
            elements.append(
                f'<line class="zero" x1="{axis(0)}" x2="{axis(0)}" '
                f'y1="{first_row - 10}" y2="{first_row + row_gap + 12}"/>'
            )
            for world_index, world in enumerate(worlds):
                row = clinical.loc[[world]].iloc[0]
                cy = first_row + world_index * row_gap
                cx = axis(float(row.estimate))
                colour = "#226879" if world_index == 0 else "#a45d45"
                name = "A" if world_index == 0 else "B"
                elements.append(f'<g data-world="{world}" data-phase="{phase}">')
                label(plot_left, cy - 21, f"{name}  {signed(row.estimate)} [{signed(row.lower)}, {signed(row.upper)}]")
                elements.append(
                    f'<line class="interval" x1="{axis(float(row.lower))}" x2="{axis(float(row.upper))}" '
                    f'y1="{cy}" y2="{cy}" stroke="{colour}"/>'
                )
                for endpoint in (row.lower, row.upper):
                    elements.append(
                        f'<path d="M{axis(float(endpoint))} {cy - 5}v10" ' f'stroke="{colour}" stroke-width="1.5"/>'
                    )
                if world_index == 0:
                    elements.append(f'<circle class="estimate" cx="{cx}" cy="{cy}" r="5.5" fill="{colour}"/>')
                else:
                    elements.append(
                        f'<rect class="estimate" x="{cx - 4.5}" y="{cy - 4.5}" '
                        f'width="9" height="9" transform="rotate(45 {cx} {cy})" fill="{colour}"/>'
                    )
                elements.append("</g>")
            baseline = first_row + row_gap + 23
            elements.append(f'<path class="rule" d="M{plot_left} {baseline}H{plot_right}"/>')
            for tick in (-1.5, -1, 0, 1, 2):
                x = axis(tick)
                elements.append(f'<path class="rule" d="M{x} {baseline}v5"/>')
                value = f"{tick:g}".replace("-", "−")
                elements.append(f'<text x="{x}" y="{baseline + 23}" text-anchor="middle">{value}</text>')
        elements.append("</svg>")
        name = "programme-decisions-compact.svg" if compact else "programme-decisions.svg"
        (output / name).write_text("\n".join(elements) + "\n", encoding="utf-8")


def plot_policy_panel(paired: pd.DataFrame, output: Path, *, compact: bool = False) -> None:
    """Show every matched policy difference within its authored world and design."""
    from matplotlib.patches import Rectangle

    baseline = paired.loc[paired.scenario.eq("baseline")]
    identities = baseline.drop_duplicates("world").sort_values(["benefit_class", "candidates"])
    designs = ("balanced", "allocation", "larger", "longer", "staged")
    labels = ("Balanced", "Allocation", "Larger", "Longer", "Staged")
    colours = {"positive": "#dcece8", "negative": "#ead4c9", "tie": "#eff1ee"}
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans"],
            "font.size": 11,
            "text.color": "#202a2e",
            "svg.fonttype": "none",
            "svg.hashsalt": "policy-panel",
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=(3.5, 14.7) if compact else (10, 8.5))
        grid = figure.add_gridspec(
            6 if compact else 3,
            1 if compact else 2,
            left=0.04 if compact else 0.035,
            right=0.96,
            top=0.871 if compact else 0.84,
            bottom=0.015,
            hspace=0.18 if compact else 0.25,
            wspace=0.16,
        )
        for index, identity in enumerate(identities.itertuples(index=False)):
            subset = baseline.loc[baseline.world.eq(identity.world)]
            mean = float(subset.responsive_minus_fixed_usd.mean()) / 1e6
            axis = figure.add_subplot(grid[index] if compact else grid[index // 2, index % 2])
            axis.set(xlim=(0, 3), ylim=(-0.5, 7.0))
            axis.axis("off")
            axis.text(
                0,
                6.55,
                f"{str(identity.benefit_class).title()} · {identity.candidates} candidates",
                fontsize=12 if compact else 14,
                weight="semibold",
            )
            axis.text(
                0, 5.65, f"World mean: {mean:+.2f}M" if mean else "World mean: 0.00M", fontsize=11 if compact else 13
            )
            axis.text(0, 4.75, "Design", fontsize=10.5 if compact else 13)
            for repeat in (0, 1):
                axis.text(
                    1.7 + repeat * 0.85, 4.75, f"Repeat {repeat + 1}", ha="center", fontsize=10.5 if compact else 13
                )
            for row_index, (design, label) in enumerate(zip(designs, labels, strict=True)):
                y = 3.85 - row_index * 0.92
                axis.text(0, y, label, va="center", fontsize=11 if compact else 13)
                for repeat in (0, 1):
                    row = subset.loc[subset.design.eq(design) & subset.repeat.eq(repeat)]
                    if len(row) != 1:
                        raise ValueError("The policy panel requires one matched pair per world/design/repeat")
                    difference = float(row.iloc[0].responsive_minus_fixed_usd) / 1e6
                    category = "tie" if abs(difference) < 1e-12 else "positive" if difference > 0 else "negative"
                    x = 1.7 + repeat * 0.85
                    axis.add_patch(
                        Rectangle((x - 0.405, y - 0.36), 0.81, 0.72, facecolor=colours[category], linewidth=0)
                    )
                    axis.text(
                        x,
                        y,
                        "0" if category == "tie" else f"{difference:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=11 if compact else 13,
                    )
        figure.text(
            0.04,
            0.988,
            "What early stopping\ngains — and loses" if compact else "What early stopping gains — and loses",
            fontsize=15 if compact else 18,
            weight="semibold",
            va="top",
        )
        figure.text(
            0.04,
            0.945 if compact else 0.932,
            "Early-stop minus complete-all value ($M)\n" "+ favours early stop; − complete all; 0 tie",
            fontsize=10.5 if compact else 13,
            va="top",
            linespacing=1.5,
        )
        figure.text(
            0.04,
            0.907 if compact else 0.878,
            "6 worlds · 5 designs · 2 allocation repeats",
            fontsize=10.5 if compact else 13,
            va="top",
        )
        name = "authored-policy-panel-compact" if compact else "authored-policy-panel"
        for suffix in ("svg", "pdf", "png"):
            figure.savefig(
                output / f"{name}.{suffix}",
                dpi=180,
                transparent=True,
                metadata={"Date": None} if suffix == "svg" else None,
            )
        plt.close(figure)


def utility_sensitivity(programmes_path: Path, preferences_path: Path, output: Path) -> None:
    """Revalue recorded policies with fixed actions, approvals and study clocks.

    Parameters
    ----------
    programmes_path
        Existing native programme values, with one row per world/design/repeat/policy.
    preferences_path
        Per-world disclosed economic preferences copied from the native design records.
    output
        Directory for exact programme, paired and summary values plus the figure.

    Raises
    ------
    ValueError
        If pairs, complete values or economic identities do not support revaluation.

    Notes
    -----
    This is descriptive objective sensitivity, not a rerun of adaptive decisions.
    World, design and allocation weights are fixed and explicitly equal. The
    selected worlds do not identify a real-world programme distribution.
    """
    programmes = pd.read_csv(programmes_path, keep_default_na=False)
    preferences = pd.read_csv(preferences_path, keep_default_na=False)
    keys = ["world", "design", "repeat"]
    if programmes.duplicated([*keys, "policy"]).any() or not programmes.complete.eq(True).all():
        raise ValueError("Utility sensitivity requires unique complete recorded policies")
    pairs = programmes.groupby(keys).policy.agg(set)
    if not pairs.map(lambda values: values == {"fixed", "responsive"}).all():
        raise ValueError("Every utility comparison requires matched fixed and responsive policies")
    data = programmes.merge(preferences, on="world", validate="many_to_one", how="left")
    numeric = [
        "benefit_usd",
        "safety_cost_usd",
        "cost_usd",
        "delay_cost_usd",
        "value_usd",
        "elapsed_days",
        "discount_rate_per_year",
        "regulatory_review_days",
        "exclusivity_end_day",
        "programme_cost_per_day_usd",
    ]
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Utility sensitivity requires finite values and preferences for every world")
    if (data.discount_rate_per_year <= 0).any():
        raise ValueError("These disclosed examples require a positive annual discount rate")
    native = data.benefit_usd - data.safety_cost_usd - data.cost_usd - data.delay_cost_usd
    if not np.allclose(native, data.value_usd, rtol=0, atol=1e-6):
        raise ValueError("Recorded programme components do not sum to the recorded value")
    if not np.allclose(data.elapsed_days * data.programme_cost_per_day_usd, data.delay_cost_usd, rtol=0, atol=1e-6):
        raise ValueError("Recorded programme overhead differs from the disclosed daily tariff")

    scenarios = [("baseline", "market_value", 1.0, 1.0, 1.0, 0.0)]
    scenarios += [
        (f"market_value_x{value:g}", "market_value", value, 1.0, 1.0, 0.0) for value in (0.0, 0.25, 0.5, 1.5, 2.0)
    ]
    scenarios += [(f"study_cost_x{value:g}", "study_cost", 1.0, value, 1.0, 0.0) for value in (0.5, 2.0)]
    scenarios += [(f"overhead_x{value:g}", "overhead", 1.0, 1.0, value, 0.0) for value in (0.5, 2.0)]
    scenarios += [
        (f"launch_delay_{days:g}_days", "launch_delay", 1.0, 1.0, 1.0, days) for days in (180.0, 365.0, 730.0)
    ]
    rows = []
    rate = np.log1p(data.discount_rate_per_year)
    launch = (data.elapsed_days + data.regulatory_review_days) / 365.25
    end = data.exclusivity_end_day / 365.25
    years = np.maximum(0, (np.exp(-rate * launch) - np.exp(-rate * end)) / rate)
    if (years <= 0).any():
        raise ValueError("The recorded examples require positive baseline market opportunity")
    for name, axis, market, study, overhead, delay in scenarios:
        delayed_years = np.maximum(0, (np.exp(-rate * (launch + delay / 365.25)) - np.exp(-rate * end)) / rate)
        frame = programmes.copy()
        frame["scenario"] = name
        frame["varied_quantity"] = axis
        frame["market_value_multiplier"] = market
        frame["study_cost_multiplier"] = study
        frame["overhead_multiplier"] = overhead
        frame["additional_launch_delay_days"] = delay
        frame["revalued_usd"] = (
            market * (data.benefit_usd - data.safety_cost_usd) * delayed_years / years
            - study * data.cost_usd
            - overhead * data.delay_cost_usd
        )
        rows.append(frame)
    values = pd.concat(rows, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    values.to_csv(output / "authored_utility_sensitivity.csv", index=False)
    paired = values.pivot(index=["scenario", *keys], columns="policy", values="revalued_usd").reset_index()
    paired["responsive_minus_fixed_usd"] = paired.responsive - paired.fixed
    identity = programmes.drop_duplicates("world")[["world", "candidates", "benefit_class"]]
    paired = paired.merge(identity, on="world", validate="many_to_one")
    baseline = paired.loc[paired.scenario.eq("baseline"), [*keys, "responsive_minus_fixed_usd"]].rename(
        columns={"responsive_minus_fixed_usd": "baseline_difference_usd"}
    )
    paired = paired.merge(baseline, on=keys, validate="many_to_one")
    paired["ranking_reversed"] = paired.responsive_minus_fixed_usd * paired.baseline_difference_usd < -(1e-6**2)
    paired.to_csv(output / "authored_utility_comparisons.csv", index=False)
    for compact in (False, True):
        plot_policy_panel(paired, output, compact=compact)
    world_means = paired.groupby(["scenario", "world"]).responsive_minus_fixed_usd.mean()
    summary_rows = []
    for scenario, group in paired.groupby("scenario"):
        for world, subset in [("all_selected_worlds", group), *group.groupby("world")]:
            means = (
                world_means.loc[scenario]
                if world == "all_selected_worlds"
                else pd.Series([subset.responsive_minus_fixed_usd.mean()])
            )
            difference = subset.responsive_minus_fixed_usd
            summary_rows.append(
                {
                    "scenario": scenario,
                    "world": world,
                    "worlds": len(means),
                    "paired_programmes": len(subset),
                    "mean_responsive_minus_fixed_usd": means.mean(),
                    "responsive_better": int((difference > 1e-6).sum()),
                    "fixed_better": int((difference < -1e-6).sum()),
                    "ties": int((difference.abs() <= 1e-6).sum()),
                    "ranking_reversals_from_baseline": int(subset.ranking_reversed.sum()),
                }
            )
    summaries = pd.DataFrame(summary_rows)
    summaries.to_csv(output / "authored_utility_summary.csv", index=False)
    # Plot the exported values, not a separate calculation.
    summaries = pd.read_csv(output / "authored_utility_summary.csv", keep_default_na=False)
    scenario_values = values.drop_duplicates("scenario").set_index("scenario")
    market_rows = summaries.loc[summaries.scenario.map(scenario_values.varied_quantity).eq("market_value")].copy()
    market_rows["multiplier"] = market_rows.scenario.map(scenario_values.market_value_multiplier)
    overall = market_rows.loc[market_rows.world.eq("all_selected_worlds")].sort_values("multiplier")
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.3]})
        axes[0].plot(overall.multiplier, overall.mean_responsive_minus_fixed_usd / 1e6, "o-", color=COLOURS[0])
        axes[0].axhline(0, color=".4", ls="--", lw=1)
        axes[0].axvline(1, color=".65", ls=":", lw=1)
        axes[0].set(
            title=f"A  Equal-weight mean across {len(identity)} authored worlds",
            xlabel="Multiplier of discounted net treatment value",
            ylabel="Responsive minus fixed value (USD millions)",
        )
        labels = []
        for index, row in enumerate(identity.sort_values(["benefit_class", "candidates"]).itertuples()):
            labels.append(f"{str(row.benefit_class).title()}, {str(row.candidates)} candidates")
            subset = summaries.loc[summaries.world.eq(row.world)].set_index("scenario")
            points = [
                cast(float, subset.loc[scenario, "mean_responsive_minus_fixed_usd"]) / 1e6
                for scenario in ("baseline", "market_value_x2")
            ]
            axes[1].plot(points, [index, index], color=".55", lw=1)
            for point, colour, marker, label in zip(
                points, COLOURS[:2], ("o", "s"), ("Disclosed value", "Twice the value"), strict=True
            ):
                axes[1].scatter(point, index, color=colour, marker=marker, s=35, label=label if index == 0 else None)
        axes[1].axvline(0, color=".4", ls="--", lw=1)
        axes[1].set(
            title=f"B  Preserve the {len(identity)} different worlds",
            yticks=range(len(labels)),
            yticklabels=labels,
            xlabel="Mean responsive minus fixed value (USD millions)",
        )
        axes[1].invert_yaxis()
        axes[1].legend(loc="lower left", fontsize=9)
        fig.savefig(output / "authored_utility_sensitivity.svg", bbox_inches="tight")
        save(fig, output, "authored_utility_sensitivity")
    (output / "authored_utility_sensitivity.md").write_text(
        "# Revalue the same recorded programmes\n\n"
        "Actions, enrollment, analyses, approval outcomes and submission times are held fixed. "
        "The treatment-value multiplier scales both discounted benefit and safety value together. "
        "Study-cost and daily-overhead scenarios alter their stated prices. Additional launch delay "
        "changes discounted market years after submission, without adding pre-submission overhead. "
        "The original exclusivity end is fixed.\n\n"
        f"The panel contains {len(identity)} authored worlds, {programmes.design.nunique()} designs and "
        f"{programmes.repeat.nunique()} matched allocation repeats per design; "
        f"each of the {len(pairs)} pairs contains a fixed and a responsive policy. Average paired differences equally "
        f"within each world, then equally across worlds. The {len(programmes)} programme rows are not "
        f"{len(programmes)} independent worlds. "
        "The plot shows exact finite-panel means, with no population confidence interval. Positive values "
        "favour the recorded responsive policy. The world labels refer to the authored benefit classes.\n\n"
        "These economic multipliers are illustrative stress settings, not empirically calibrated ranges. "
        "A changed ranking describes sensitivity of this objective on these recorded actions. It does not "
        "show how policies would adapt to changed incentives, optimal research decisions, general policy "
        "superiority or external clinical decision transfer. Study designs also remain fixed when a revised "
        "price would violate the original budget.\n\n"
        "Sources: authored_programme_values.csv and authored_utility_preferences.csv. Exact revalued "
        "programmes, matched differences, ties and direction reversals are in the accompanying CSV files.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Revalue fixed recorded programmes under disclosed economic changes.")
    parser.add_argument("--programmes", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    programme_decisions(arguments.programmes, arguments.output)
    utility_sensitivity(arguments.programmes, arguments.preferences, arguments.output)
