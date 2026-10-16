"""Present scientific reference comparisons and the combined evidence report."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reference_display(root: Path, output: Path) -> None:
    """Plot independently derived expectations beside the disclosed native references."""
    law = pd.read_csv(output / "known_law.csv").iloc[0]
    precision = pd.read_csv(output / "reference_precision.csv").iloc[0]
    derived = pd.read_csv(output / "trialdev_initial_reference_derivation.csv").set_index("asset_id")
    native = pd.read_csv(root / "trialdev" / "initial_references.csv").set_index("asset_id")
    rows = [
        {
            "question": "Rehabilitation",
            "unit": "disability points",
            "method": "Pre-recording law",
            "estimate": law.pre_rounding_law_contrast,
            "lower": np.nan,
            "upper": np.nan,
        },
        {
            "question": "Rehabilitation",
            "unit": "disability points",
            "method": "32-world reference",
            "estimate": law.reference,
            "lower": precision.reference_lower,
            "upper": precision.reference_upper,
        },
    ]
    for asset in derived.index:
        for endpoint, unit in (
            ("outcome_mean_difference", "DAS28 points"),
            ("safety_risk_difference", "risk difference"),
        ):
            for method, frame in (("Independent integration", derived), ("Native reference", native)):
                rows.append(
                    {
                        "question": str(asset),
                        "unit": unit,
                        "method": method,
                        "estimate": float(frame.loc[asset, endpoint]),
                        "lower": np.nan,
                        "upper": np.nan,
                    }
                )
    pd.DataFrame(rows).to_csv(output / "reference_agreement.csv", index=False)
    values = pd.read_csv(output / "reference_agreement.csv")
    figure = plt.figure(figsize=(8, 6.4), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(0.75, 1))
    axes = (figure.add_subplot(grid[0, :]), figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1]))
    for axis, unit, title, multiplier in zip(
        axes,
        ("disability points", "DAS28 points", "risk difference"),
        ("A  A known disease process", "B  Clinical association", "C  Safety association"),
        (1, 1, 100),
        strict=True,
    ):
        panel = values.loc[values.unit.eq(unit)]
        questions = list(panel.question.unique())
        methods = list(panel.method.unique())
        for index, question in enumerate(questions):
            selected = panel.loc[panel.question.eq(question)]
            for method_index, method in enumerate(methods):
                row = selected.loc[selected.method.eq(method)].iloc[0]
                y = index + (-0.12 if method_index == 0 else 0.12)
                colour = "#0072B2" if method_index == 0 else "#B35C1E"
                if pd.notna(row.lower):
                    axis.plot(multiplier * np.array([row.lower, row.upper]), [y, y], color=colour, linewidth=2)
                axis.scatter(
                    multiplier * row.estimate,
                    y,
                    marker="o" if method_index == 0 else "D",
                    color=colour,
                    s=35,
                    label=method if index == 0 else None,
                    zorder=3,
                )
        axis.set_yticks(range(len(questions)), questions)
        axis.set_ylim(len(questions) - 0.5, -0.5)
        axis.set_xlabel(
            "Safety risk difference (percentage points)"
            if multiplier == 100
            else ("DAS28 points" if unit == "DAS28 points" else unit.capitalize())
        )
        axis.set_title(title, loc="left", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)
    figure.savefig(output / "reference_agreement.png", dpi=180)
    figure.savefig(output / "reference_agreement.pdf")
    plt.close(figure)


def proof_report(root: Path, output: Path, *, r_executed: bool = False, compact: bool = False) -> None:
    """Explain the world, executed interactions and interpretation consequences.

    Parameters
    ----------
    root
        Public scientific input directory.
    output
        Completed ordinary-library analysis outputs.
    r_executed
        Whether this run also completed the optional R comparisons.
    compact
        Present only the bundled analysis and native development episode.
    """
    from html import escape

    def table(frame: pd.DataFrame) -> str:
        return (
            '<div class="table-scroll">'
            + frame.to_html(index=False, border=0, float_format=lambda x: f"{x:,.3g}")
            + "</div>"
        )

    def figure(name: str, alt: str, caption: str, *, responsive: bool = False) -> str:
        mobile = f'<source media="(max-width:650px)" srcset="{name}-compact.svg">' if responsive else ""
        extension = "svg" if (output / f"{name}.svg").exists() else "png"
        return (
            f'<figure><a href="{name}.{"pdf" if (output / f"{name}.pdf").exists() else extension}"><picture>{mobile}<img src="{name}.{extension}" '
            f'alt="{escape(alt)}" loading="lazy"></picture></a><figcaption>{caption}</figcaption></figure>'
        )

    episode = pd.read_csv(output / "interaction-timeline.csv")
    final = episode.iloc[-1]
    sections = [
        '<header><p class="eyebrow">TrialAgentBench · Scientific demonstrations</p>'
        "<h1>Investigate a clinical world. Measure what the interpretation changes.</h1>"
        '<p class="lead">An AI agent must learn from imperfect trial evidence and decide what research '
        "is worth doing next. The simulator controls the underlying clinical mechanisms, so we can "
        "compare the agent’s conclusions with the world—and observe the consequences of its choices.</p>"
        "<p>These executable examples use conventional statistical analyses and recorded, scripted research "
        "programmes to explain the task. Model evaluation is a separate experiment.</p></header>",
        '<nav aria-label="Demonstrations"><a href="#world">Clinical world</a><a href="#interaction">Research episode</a>'
        + (
            ""
            if compact
            else '<a href="#trials">Trial patterns</a><a href="#materiality">Interpretation</a><a href="#choices">Research choices</a>'
        )
        + '<a href="#reproduce">Calculations</a></nav>',
        '<section id="world"><h2>The world defines the question’s reference</h2>'
        "<p>Patient characteristics, treatment and disease progression enter explicit outcome mechanisms. "
        "Trial conduct then determines who is recruited, assigned treatment and followed; observation "
        "determines which outcomes reach the analyst. The agent receives documents and records, while "
        "the evaluator can calculate the corresponding reference from the constructed world.</p>"
        '<div class="world"><div><h3>Clinical mechanisms</h3><p>Patient characteristics → disease and treatment response.</p></div>'
        "<div><h3>Two views of the same world</h3><p><strong>Evaluator:</strong> outcomes under the relevant interventions define the target.</p>"
        "<p><strong>Agent:</strong> recruitment, conduct and observation produce the available evidence.</p></div>"
        "<div><h3>Assessment</h3><p>Compare the submitted estimate and uncertainty with the target. Different analyses can recover the same answer.</p></div></div>"
        "<p>For a randomized mean contrast the question may be the average outcome under treatment minus "
        "the average under control. An observational starting cohort instead asks for its specified association. "
        "The target’s population, outcome and follow-up must stay fixed when methods are compared.</p>"
        "<details><summary>Inspect the constructed response</summary>"
        "<p>The Parkinson example propagates disease-state probabilities and a direct treatment contribution. "
        "The arthritis starting cohort uses independent integration of its clinical and safety associations. "
        "These are separate clinical questions, each with its own reference.</p>"
        '<p><a href="known_law.csv">Parkinson response calculation</a> · '
        '<a href="trialdev_initial_reference_derivation.csv">Arthritis association calculation</a> · '
        '<a href="trialeval_analyses.csv">Alternative Parkinson analyses and scores</a></p></details></section>',
        '<section id="interaction"><h2>Research changes what can be learned</h2>'
        "<p>This rheumatoid-arthritis programme starts from observational evidence, then runs three randomized "
        "studies with 20, 120 and 600 patients. Opening sites commits resources. Recruitment and follow-up "
        "take time. A readout returns the outcomes available on that date.</p>"
        + figure(
            "research-episode",
            "Three executed studies, their DAS28 readouts and accrued spending on a shared programme clock.",
            "The first readout is inconclusive. Later studies support lower disease activity. "
            "Each estimates its own Day-28, Day-84 or Day-168 contrast; the intervals are 95% Welch intervals. "
            "Costs connect recorded states; intermediate accrual is not measured by the connecting lines. "
            '<a href="interaction-timeline.csv">Actions and responses</a> · '
            '<a href="trialdev_readout_estimates.csv">Readout estimates</a>',
            responsive=True,
        )
        + f"<p>At programme day {final.programme_day:,.0f}, the studies have enrolled {final.enrolled_to_date:,.0f} "
        f"patients and cost ${final.study_cost_usd/1e6:.2f} million. Programme overhead adds "
        f"${final.overhead_usd/1e6:.2f} million. The candidate is submitted and meets the simulator’s confirmation rule. "
        "The research budget covers study expenditure and commitments; overhead separately reduces terminal value.</p>"
        + table(
            episode[["programme_day", "action", "response"]]
            .replace(
                {"action": {"start_study": "Open study", "readout": "Obtain readout", "advance": "Submit candidate"}}
            )
            .rename(columns={"programme_day": "Day", "action": "Action", "response": "World response"})
        )
        + "<p>The script follows a predetermined programme. This demonstrates the environment’s executed "
        "responses; it does not measure an AI agent’s adaptive judgement. "
        '<a href="trialdev_study.md">Follow the patient records and analyses</a>.</p></section>',
    ]
    if not compact:
        cells = pd.read_csv(output / "families/family_diagnostics.csv")
        coverage = cells[
            ["cell", "family", "scale", "visible_n", "reference_arithmetic_and_raw_reconstruction"]
        ].rename(columns={"reference_arithmetic_and_raw_reconstruction": "source_reconstruction"})
        coverage["realizations"] = 1
        standards = pd.read_csv(output / "families/standards_reconciliation.csv")
        if set(standards.cell) != set(cells.cell) or not standards.status.eq("passed").all():
            raise ValueError("Source comparisons do not cover all example cells")
        coverage["benchmark_standards_export"] = "source reconciliation passed"
        coverage.to_csv(output / "proof_coverage.csv", index=False)
        sections.extend(_full_demonstrations(output, figure))
    sections.append(
        '<section id="reproduce"><h2>Recalculate the evidence</h2>'
        "<p>The reader reconstructs analysis tables, fits the statistical models and regenerates the figures "
        "using NumPy, pandas, SciPy, statsmodels, lifelines and Matplotlib. "
        '<a href="trialdev_resources.csv">Study charges</a>, '
        '<a href="trialdev_value_components.csv">programme value</a> and '
        '<a href="trialdev_time_controls.csv">alternative readout interpretations</a> retain the numerical calculations.</p>'
        "<p>The public code separates the clinical target, observed analysis and assessment. Missing records "
        "remain missing; evaluator outcomes enter the comparison after observed-data fitting.</p>"
    )
    if r_executed:
        sections.append(
            "<p>R also recalculated sample-size and survival results in this execution.</p>"
            + table(pd.read_csv(output / "clinical-reference-results.csv"))
            + '<p><a href="planning-results.csv">R planning calculations</a></p>'
        )
    elif not compact:
        sections.append(
            "<p>Optional R comparisons were not executed in this run. Add <code>--rscript Rscript</code> to repeat planning and survival calculations in R.</p>"
        )
    sections.append("</section><footer>TrialAgentBench · Executable scientific examples</footer>")
    css = """
    :root{color-scheme:light}body{margin:0;background:#f4f3ed;color:#23323d;font:17px/1.65 system-ui,sans-serif}
    main{max-width:1000px;margin:auto;padding:48px 28px}h1{font-size:clamp(2rem,5vw,3.3rem);line-height:1.1;max-width:900px}
    h2{font-size:1.65rem;line-height:1.25;margin-top:0}h3{font-size:1.05rem}.lead{font-size:1.2rem;max-width:850px}
    .eyebrow{font-size:.82rem;text-transform:uppercase;letter-spacing:.1em;color:#226879}a{color:#075a82;text-underline-offset:3px}
    nav{display:flex;flex-wrap:wrap;gap:12px 24px;padding:20px 0;border-block:1px solid #cbd5da}
    section{padding:38px 0;border-bottom:1px solid #cbd5da}figure{margin:30px 0}img{max-width:100%;height:auto;display:block}
    figcaption{font-size:.9rem;color:#41555f;margin-top:14px}table{border-collapse:collapse;font-size:.86rem;width:100%}
    td,th{padding:10px 12px;text-align:left;border-bottom:1px solid #ccd5d7;vertical-align:top}th{background:#e6eded}
    .table-scroll{overflow-x:auto;margin:24px 0}.world{display:grid;grid-template-columns:1fr 1.4fr 1fr;gap:24px;margin:28px 0}
    .world>div{border-top:3px solid #226879;padding-top:8px}summary{cursor:pointer;font-weight:600;padding:14px 0}
    footer{font-size:.85rem;color:#41555f;margin-top:30px}details{margin:20px 0}
    @media(max-width:650px){main{padding:24px 16px}body{font-size:16px}.world{grid-template-columns:1fr;gap:8px}}
    @media print{body,main{background:white}main{padding:0}nav{display:none}figure,table{break-inside:avoid}}
    """
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>TrialAgentBench — investigate a clinical world</title><style>"
        + css
        + "</style></head><body><main>"
        + "\n".join(sections)
        + "</main></body></html>\n",
        encoding="utf-8",
    )


def _full_demonstrations(output: Path, figure: Callable[..., str]) -> list[str]:
    """Connect current-record characterization with controlled scientific experiments."""
    parts = ['<section id="trials"><h2>Trial mechanisms leave visible patterns</h2>']
    if (output / "current-trials/trial_census.csv").exists():
        census = pd.read_csv(output / "current-trials/trial_census.csv")
        parts.append(
            f"<p>The current-record census contains {len(census)} studies, each counted once. "
            "Recruitment, assignment, follow-up and outcomes are measured from their records. "
            "Context views of the same study do not create additional independent trials.</p>"
            '<p><a href="current-trials/trial_census.csv">Current study census</a> · '
            '<a href="current-trials/observation.csv">Observation</a> · <a href="current-trials/allocation.csv">Allocation</a></p>'
        )
        for name, alt, caption in (
            (
                "recruitment",
                "Enrolment and site recruitment over time across current trials.",
                "Recruitment timing is measured from participant records, not inferred from a trial label.",
            ),
            (
                "rollout",
                "Treatment exposure by site and calendar period in current stepped-wedge trials.",
                "Staggered rollout makes treatment exposure depend on site and calendar time.",
            ),
            (
                "followup",
                "Routine measurement and central-review coverage in current colorectal and asthma trials.",
                "Routine measurements cover every participant in these adjudication studies; central review supplies a validation sample of approximately 30% to identify measurement errors. The channels serve different analytical purposes.",
            ),
        ):
            parts.append(figure(f"current-trials/{name}", alt, caption, responsive=True))
    else:
        parts.append(
            "<p>This input bundle does not include the current trial archive. The historical "
            "worked designs below explain mechanisms; the current-record atlas is produced "
            "when <code>current-trial-records.zip</code> is included.</p>"
        )
    parts.append(
        figure(
            "environment/trial-designs",
            "Staggered treatment assignment, sequential monitoring and repeated observations in worked trials.",
            "Different trial designs produce different data structures. These worked specimens explain "
            "rollout, stopping and observation; the current census supplies benchmark-wide distributions. "
            '<a href="environment/design_evidence.csv">Worked design values</a>',
            responsive=True,
        )
        + '</section><section id="materiality"><h2>When an interpretation changes the conclusion</h2>'
        "<p>The colorectal experiment compares routine response labels with sampled central adjudication "
        "in the same 32 trials. Both analyses address the Day-168 tumour-response gain from combination "
        "versus standard therapy. Routine false positives in the control arm and missed responses in "
        "the treatment arm obscure the gain.</p>"
    )
    data = pd.read_csv(output / "interpretation-materiality.csv")
    summary = pd.read_csv(output / "interpretation-materiality-summary.csv")
    pairs = pd.read_csv(output / "interpretation-materiality-pairs.csv")
    at_five = summary.loc[summary.threshold_pp.eq(5)]
    routine_out = int(
        at_five.loc[at_five.method.eq("Routine endpoint") & at_five.conclusion.eq("rules out target"), "studies"].sum()
    )
    corrected_out = int(
        at_five.loc[
            at_five.method.eq("Stratified adjudication") & at_five.conclusion.eq("rules out target"), "studies"
        ].sum()
    )
    changed = int(pairs.loc[pairs.threshold_pp.eq(5), "different_conclusion"].sum())
    reference = data.iloc[0]
    parts.append(
        "<p>Apply one fixed interpretation rule: a 95% interval entirely above a required response "
        "gain supports that target; an interval entirely below it rules the target out; otherwise "
        "the conclusion remains unresolved. Varying the required gain reveals when the method changes "
        "the interpretation. These thresholds are analysis scenarios, not clinical recommendations.</p>"
        + figure(
            "interpretation-materiality",
            "Conclusions from two analyses across required response gains from zero to ten percentage points.",
            "Each vertical slice classifies the same 32 trials. The dashed line is the separate "
            "world-reference estimate; hatching is its 95% Monte Carlo interval. "
            '<a href="interpretation-materiality.csv">Every classification</a> · '
            '<a href="interpretation-materiality-pairs.csv">Paired changes</a>',
            responsive=True,
        )
        + f"<p>At a five-percentage-point target, routine analysis rules out that gain in {routine_out}/32 "
        f"trials; adjudication does so in {corrected_out}/32. The two methods produce different "
        f"conclusions in {changed}/32 trials. The separate world reference is "
        f"{reference.reference_pp:.2f} percentage points (95% Monte Carlo interval "
        f"{reference.reference_lower_pp:.2f}–{reference.reference_upper_pp:.2f}).</p>"
        "<p>This experiment isolates interpretation using identical evidence and a common rule. "
        "It does not assign a funding decision or monetary value to the resulting conclusions.</p>"
        "<details><summary>Effect recovery and grading</summary>"
        "<p>Repeated-study bias, coverage and numerical loss explain the change in conclusions. "
        "The score combines estimation error, interval width and a penalty when the interval misses "
        "the reference; a method receives no credit merely for its name.</p>"
        + figure(
            "repeated_study_recovery",
            "Repeated-study bias and coverage under clinical and observation mechanisms.",
            "Within-condition methods share trial records. Each condition has 32 trials; references "
            "come from separate complete-outcome studies. "
            '<a href="recovery_operating_characteristics.csv">Bias and coverage</a> · '
            '<a href="paired_loss_contrasts.csv">Paired loss comparisons</a> · <a href="reference_agreement.csv">Reference calculations</a>',
        )
        + '</details></section><section id="choices"><h2>Which research is worth its cost?</h2>'
        "<p>Two scripted strategies pursue the same treatment in paired worlds. One stops after an "
        "unfavourable readout; another completes its research programme. The same decision rule can "
        "abandon a useful treatment in one world and avoid wasted research in another.</p>"
        + figure(
            "programme-decisions",
            "Paired research paths showing readouts, continuation and stopping consequences.",
            "Two selected worlds explain the decision. Each later study has its own follow-up target. "
            '<a href="programme-decisions.csv">Programme outcomes</a> · '
            '<a href="programme-studies.csv">Evidence at each stage</a>',
            responsive=True,
        )
        + "<p>The full experiment includes six worlds, five designs and two allocation repeats: "
        "60 matched strategy pairs. Study costs and elapsed time are measured from execution; "
        "terminal value uses the environment’s treatment-value and harm-cost assumptions.</p>"
        + figure(
            "authored-policy-panel",
            "Every paired strategy difference across six worlds and five designs.",
            "Within-world allocation repeats do not create new clinical worlds. "
            '<a href="authored_programme_values.csv">All programme outcomes</a> · '
            '<a href="authored_utility_summary.csv">Sensitivity to treatment value and costs</a>',
            responsive=True,
        )
        + "</section>"
    )
    return parts
