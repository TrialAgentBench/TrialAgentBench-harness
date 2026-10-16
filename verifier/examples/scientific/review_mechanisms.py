"""Show every disclosed clinical condition and every repeated-study method fit.

The small bundled CSV inputs are projections of public synthetic study records.
No generation, private benchmark imports or provider calls are needed.
"""

from __future__ import annotations

import argparse
import io
import itertools
import zipfile
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import t
from statsmodels.stats.proportion import proportion_confint


def project_cells(archive_path: Path) -> pd.DataFrame:
    """Recalculate the finite illustrative comparisons from every native clinical cell."""
    from review_families import contrast, point_estimates

    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        cells = pd.read_csv(io.BytesIO(archive.read("cells.csv")))
        for cell in cells.itertuples():
            data = pd.read_parquet(io.BytesIO(archive.read(f"{cell.cell}/analysis.parquet")))
            complete = pd.read_parquet(io.BytesIO(archive.read(f"{cell.cell}/disclosed_complete_outcomes.parquet")))
            method, estimate, shortcut, crude = point_estimates(
                data,
                endpoint=str(cell.endpoint),
                family=str(cell.family),
                scale=str(cell.scale),
                horizon=cast(float, cell.horizon),
                cell=str(cell.cell),
                covariates=tuple(str(cell.analysis_covariates).split(";")),
            )
            rows.append(
                {
                    "cell": cell.cell,
                    "family_id": str(cell.cell).rsplit("-", 1)[0],
                    "condition": cell.condition,
                    "question": cell.question,
                    "seed": cell.seed,
                    "participants": cell.visible_n,
                    "scale": cell.scale,
                    "finite_complete_contrast": contrast(
                        complete.complete_endpoint, complete.assigned_arm, ratio=cell.scale == "rate_ratio"
                    ),
                    "analysis": method,
                    "estimate": estimate,
                    "shortcut": shortcut,
                    "shortcut_estimate": crude,
                }
            )
    return pd.DataFrame(rows)


def losses(rows: pd.DataFrame, reference: float | pd.Series | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Use the reader's single existing definition of normalized numerical loss."""
    from review_scoring import numerical_loss

    return np.asarray(
        np.vectorize(numerical_loss, otypes=[float])(rows.estimate, rows.lower, rows.upper, reference, rows.scale),
        dtype=np.float64,
    )


def summarize_repetitions(estimates: pd.DataFrame, references: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate every method's bias, coverage, interval width and numerical loss."""
    if references.case.duplicated().any() or estimates.duplicated(["case", "seed", "method"]).any():
        raise ValueError("Each reference and condition/seed/method result must be unique")
    refs = references.set_index("case")
    rows = estimates.copy()
    if set(rows.case) != set(refs.index):
        raise ValueError("Every disclosed reference population requires its full analysis panel")
    rows["scale"] = rows.case.map(refs.empirical_reference_se * np.sqrt(refs.reference_worlds))
    if (
        not rows.complete.all()
        or not np.isfinite(rows[["estimate", "lower", "upper", "scale"]]).all().all()
        or not rows.scale.gt(0).all()
    ):
        raise ValueError("The disclosed panel requires every attempted fit and a positive reference scale")
    if not (rows.lower.le(rows.estimate) & rows.estimate.le(rows.upper)).all():
        raise ValueError("A method interval does not contain its estimate")
    for field in ("reference", "reference_se"):
        if not np.allclose(rows[field], rows.case.map(refs[field]), rtol=0, atol=1e-12):
            raise ValueError("Repeated estimates disagree with their independent reference bank")
    rows["mean_loss"] = losses(rows, rows.reference)
    rows["point_loss"] = abs(rows.estimate - rows.reference) / rows.scale
    rows["interval_loss"] = 2 * rows.mean_loss - rows.point_loss
    rows["interval_width"] = rows.upper - rows.lower
    rows["covered"] = rows.lower.le(rows.reference) & rows.upper.ge(rows.reference)
    summaries = []
    for (case, method), group in rows.groupby(["case", "method"], sort=True):
        ref = refs.loc[[str(case)]].iloc[0]
        n = len(group)
        if n != ref.analysis_worlds or group.seed.nunique() != n or ref.reference_worlds != 64:
            raise ValueError("The repeated panel must retain all declared independent study seeds and references")
        error = group.estimate - ref.reference
        bias_mcse = np.sqrt(error.var(ddof=1) / n + ref.reference_se**2)
        half = t.ppf(0.975, n - 1) * bias_mcse
        coverage_lower, coverage_upper = proportion_confint(int(group.covered.sum()), n, method="wilson")
        loss_mcse = group.mean_loss.std(ddof=1) / np.sqrt(n)
        loss_half = t.ppf(0.975, n - 1) * loss_mcse
        summaries.append(
            {
                "case": case,
                "series": group.series.iloc[0],
                "state": group.state.iloc[0],
                "method": method,
                "studies": n,
                "reference_group": ref.reference_group,
                "reference": ref.reference,
                "reference_mcse": ref.reference_se,
                "scale": group.scale.iloc[0],
                "bias": error.mean(),
                "bias_lower": error.mean() - half,
                "bias_upper": error.mean() + half,
                "coverage": group.covered.mean(),
                "coverage_lower": coverage_lower,
                "coverage_upper": coverage_upper,
                "mean_interval_width": group.interval_width.mean(),
                "mean_loss": group.mean_loss.mean(),
                "loss_lower": group.mean_loss.mean() - loss_half,
                "loss_upper": group.mean_loss.mean() + loss_half,
            }
        )
    return rows, pd.DataFrame(summaries)


def paired_comparisons(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep all within-condition method pairs and the declared retrieval controls."""
    comparisons = []

    def compare(label: str, first: pd.DataFrame, second: pd.DataFrame, kind: str) -> None:
        left, right = (frame.set_index("seed").sort_index() for frame in (first, second))
        if not left.index.equals(right.index) or len(left) < 2:
            raise ValueError("Each paired comparison requires the same independent study seeds")
        for field in ("reference", "reference_se", "scale", "participants"):
            if not np.allclose(left[field], right[field], rtol=0, atol=1e-12):
                raise ValueError("Paired comparisons require matching target, reference, scale and cohort size")
        reference = float(left.reference.iloc[0])
        shift = t.ppf(0.975, 63) * float(left.reference_se.iloc[0])
        knots = np.clip(
            np.r_[
                reference - shift,
                reference + shift,
                left.estimate,
                left.lower,
                left.upper,
                right.estimate,
                right.lower,
                right.upper,
            ],
            reference - shift,
            reference + shift,
        )
        sensitivity = [float(np.mean(losses(left, value) - losses(right, value))) for value in knots]
        difference = left.mean_loss - right.mean_loss
        mcse = difference.std(ddof=1) / np.sqrt(len(difference))
        half = t.ppf(0.975, len(difference) - 1) * mcse
        comparisons.append(
            {
                "comparison": label,
                "kind": kind,
                "first_case": left.case.iloc[0],
                "second_case": right.case.iloc[0],
                "first_method": left.method.iloc[0],
                "second_method": right.method.iloc[0],
                "studies": len(difference),
                "mean_loss_difference": difference.mean(),
                "conditional_mcse": mcse,
                "conditional_lower": difference.mean() - half,
                "conditional_upper": difference.mean() + half,
                "reference_sensitivity_min": min(sensitivity),
                "reference_sensitivity_max": max(sensitivity),
                "mean_point_loss_difference": (left.point_loss - right.point_loss).mean(),
                "mean_interval_loss_difference": (left.interval_loss - right.interval_loss).mean(),
                "mean_interval_width_difference": (left.interval_width - right.interval_width).mean(),
                "coverage_difference": (left.covered.astype(float) - right.covered.astype(float)).mean(),
                "first_has_higher_loss_fraction": difference.gt(0).mean(),
                "equal_loss_fraction": difference.eq(0).mean(),
            }
        )

    for case, group in rows.groupby("case"):
        for first, second in itertools.combinations(sorted(group.method.unique()), 2):
            compare(
                f"{case}: {first} minus {second}",
                group.loc[group.method.eq(first)],
                group.loc[group.method.eq(second)],
                "same_trial_method",
            )
    for tier in ("A4r25", "A4r100"):
        for method in ("Crude event risk", "Randomized retrieval"):
            compare(
                f"Asthma retrieval {tier[3:]}% minus 50%: {method}",
                rows.loc[rows.case.eq(f"TE-S04-{tier}-favourable") & rows.method.eq(method)],
                rows.loc[rows.case.eq("TE-S04-A4-favourable") & rows.method.eq(method)],
                "same_trial_retrieval_probability",
            )
    return pd.DataFrame(comparisons)


def review(source: Path, output: Path, *, family_archive: Path | None = None) -> None:
    """Reproduce exhaustive mechanism tables and a figure without selecting successes."""
    output.mkdir(parents=True, exist_ok=True)
    inputs = source / "mechanisms"
    cells = project_cells(family_archive) if family_archive else pd.read_csv(inputs / "clinical_cells.csv")
    labels = pd.read_csv(inputs / "conditions.csv")
    if cells.cell.duplicated().any() or labels.case.duplicated().any():
        raise ValueError("Each disclosed clinical cell and repeated condition requires one identity")
    settings = pd.read_csv(inputs / "clinical_settings.csv").set_index("family_id").clinical_setting
    cells["clinical_setting"] = cells.family_id.map(settings)
    cells["analysis_minus_shortcut"] = cells.estimate - cells.shortcut_estimate
    cells["analysis_minus_finite_contrast"] = cells.estimate - cells.finite_complete_contrast
    cells["shortcut_minus_finite_contrast"] = cells.shortcut_estimate - cells.finite_complete_contrast
    cells.to_csv(output / "mechanism_clinical_cells.csv", index=False)
    baseline = cells.loc[cells.cell.str.endswith("-A1")].set_index("family_id")
    changes = cells.loc[~cells.cell.str.endswith("-A1")].copy()
    if not (changes.seed == changes.family_id.map(baseline.seed)).all():
        raise ValueError("Within-family mechanism contrasts require shared construction seeds")
    changes["baseline_cell"] = changes.family_id.map(baseline.cell)
    changes["baseline_condition"] = changes.family_id.map(baseline.condition)
    for field in ("finite_complete_contrast", "estimate", "shortcut_estimate"):
        changes[f"baseline_{field}"] = changes.family_id.map(baseline[field])
        changes[f"change_in_{field}"] = changes[field] - changes[f"baseline_{field}"]
    changes.to_csv(output / "mechanism_condition_changes.csv", index=False)
    rows, summary = summarize_repetitions(
        pd.read_csv(inputs / "study_estimates.csv"), pd.read_csv(inputs / "population_references.csv")
    )
    if set(summary.case) != set(labels.case):
        raise ValueError("The condition descriptions must cover the complete repeated panel")
    summary = summary.merge(labels, on="case", validate="many_to_one")
    summary["clinical_setting"] = summary.series.map(settings)
    paired = paired_comparisons(rows)
    rows.to_csv(output / "mechanism_study_losses.csv", index=False)
    summary.to_csv(output / "mechanism_recovery_summary.csv", index=False)
    paired.to_csv(output / "mechanism_paired_comparisons.csv", index=False)
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        groups = list(summary.groupby("series", sort=True))
        figure, axes = plt.subplots(
            4,
            1,
            figsize=(7.2, 11.5),
            layout="constrained",
            gridspec_kw={"height_ratios": [len(group) for _, group in groups]},
        )
        clinical_titles = {
            "TE-S04": "First severe asthma exacerbation",
            "TE-S07": "Respiratory infection: cluster randomization",
            "TE-S10": "Parkinson disease: motor disability",
            "TE-S15": "Colorectal cancer: tumour response",
        }
        for axis, (series, group) in zip(axes, groups, strict=True):
            group = group.sort_values(["display_order", "method"])
            center = group.bias / group.scale
            lower, upper = group.bias_lower / group.scale, group.bias_upper / group.scale
            y = np.arange(len(group))
            axis.errorbar(center, y, xerr=[center - lower, upper - center], fmt="o", color="#226B93", capsize=3)
            axis.axvline(0, color="0.5", lw=0.8, ls="--")
            axis.set_yticks(y, [f"{row.condition_label}\n{row.method}" for row in group.itertuples()])
            axis.set_title(clinical_titles[str(series)] + f" ({series})", loc="left", fontsize=10)
            axis.set_xlabel("Mean signed error / complete-study contrast SD")
            axis.invert_yaxis()
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(output / f"mechanism_recovery.{suffix}", dpi=180)
        plt.close(figure)
    lines = [
        "# Which mechanisms change the analysis?",
        "",
        f"The finite census retains all {cells.cell.nunique()} clinical conditions across {cells.family_id.nunique()} clinical families. The separate repeated experiment retains all {summary.case.nunique()} disclosed conditions in four families, with 32 trial seeds per condition and {len(rows)} method fits. Its {rows[['case','seed']].drop_duplicates().shape[0]} condition–seed realizations are not mutually independent: methods share trials, retrieval-probability controls share the same underlying event histories, and benefit states reuse construction seeds. Inference is within a condition or paired on study seed, never pooled as independent trials across conditions. This authored teaching panel illustrates selected mechanisms; it is not a representative sample of clinical trials or a prospectively registered comparison of methods.",
        "",
        "The family and recovery teaching archives were last revised on 25 September 2026. They predate the registry-driven realism changes introduced on 7–8 October 2026. The results establish what happens in these disclosed records; they do not qualify every mechanism in the current generator. Exact archive identities and revision records are retained in source.json.",
        "",
        "![Every repeated condition and method, including null and adverse controls.](mechanism_recovery.png)",
        "",
        "The figure shows mean signed estimation error divided by the empirical complete-study contrast standard deviation from 64 separate reference studies. Bars are approximate 95% Monte Carlo intervals for bias, combining between-analysis-study variance with uncertainty in the independent reference mean. The equal-arm Parkinson null uses the known zero reference. The normalization scale is estimated and held fixed. These bars are not patient-sampling intervals or clinical equivalence bands. The reference banks are shared where specified; there is no multiplicity-adjusted claim across conditions.",
        "",
        "## Every repeated method comparison",
        "",
        "Positive differences mean that the first method has greater numerical loss. The loss averages absolute point error and the 95% interval score, divided by the same complete-study contrast SD. Wider intervals have a cost; coverage alone does not determine which method has lower loss. Intervals below condition on the reference mean and scale. The separate sensitivity range moves the shared reference over its 95% t interval; it is not a joint confidence interval. Scale-estimation uncertainty is not propagated.",
        "",
        "| Comparison | Mean loss difference (conditional 95% MC interval) | Reference sensitivity range |",
        "| --- | ---: | ---: |",
    ]
    names = labels.set_index("case").condition_label
    for row in paired.itertuples():
        clinical = str(settings[str(row.first_case)[:6]])
        label = (
            f"{clinical}: {str(names[row.first_case])}; {row.first_method} minus {row.second_method}"
            if row.kind == "same_trial_method"
            else str(row.comparison)
        )
        lines.append(
            f"| {label} | {row.mean_loss_difference:.3f} ({row.conditional_lower:.3f}, {row.conditional_upper:.3f}) | {row.reference_sensitivity_min:.3f} to {row.reference_sensitivity_max:.3f} |"
        )
    lines += [
        "",
        "Routine tumour-response classification has substantially greater loss than stratified adjudication in this repeated panel. That result does not generalize to every mechanism: crude asthma event risk has lower mean numerical loss than retrieval weighting in the unfavourable-effect condition, despite using incomplete event ascertainment. The retrieval method pays a variance and interval-width cost. Lowering retrieval from 50% to 25% increases retrieval-weighted loss; full retrieval does not resolve a loss advantage over 50% in this bounded experiment. The crude event estimator is exactly unchanged by retrieval probability, an adverse control that checks the intended isolation of the retrieval mechanism.",
        "",
        "Cluster-aware and person-level intervals have the same point estimates and different uncertainty. Their conditional loss contrast does not resolve an advantage in 32 studies. The Parkinson series includes beneficial, near-null, harmful and exact-null mechanisms with one observed-mean analysis; its complete operating-characteristic rows are retained even though there is no within-condition method pair. These are examples of what the mechanisms do to estimation, not proofs of clinical materiality or of universal method superiority.",
        "",
        "## Every clinical cell",
        "",
        "The following single-realization table uses the same source-record analyses as the family walkthrough. Differences expose which analysis operations matter in the disclosed trial. Its complete-data contrast is usually an allocated-arm finite contrast. Baseline standardization averages over a different covariate distribution, and the stepped-wedge reference compares fixed strategies. Consequently, closeness to this finite value is not a ranking of population-estimand accuracy. These comparisons do not add repeated evidence for the other 13 families.",
        "All 13 non-A1 conditions are also paired with their same-family, same-seed A1 condition in mechanism_condition_changes.csv. It retains changes in the complete contrast, illustrative estimate and shortcut. These are one-realization mechanism contrasts without uncertainty intervals; altered conditions may change both the target and the available evidence. They do not isolate a population causal effect of a single mechanism.",
        "",
        "| Clinical condition | Complete contrast | Illustrative analysis | Shortcut | Native scale |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in cells.itertuples():
        lines.append(
            f"| {row.clinical_setting}: {row.condition} ({row.cell}) | {row.finite_complete_contrast:.5g} | {row.estimate:.5g} | {row.shortcut_estimate:.5g} | {row.scale} |"
        )
    lines += [
        "",
        "The full CSV gives each clinical question, analysis and shortcut definition, seed, units and participant count. Survival shortcuts count recorded events as risks or average observed time; the longitudinal shortcut pools visits rather than selecting the final time. Adjudication uses the sampled endpoint-validation strata. Baseline-adjusted and crude arm means can differ solely because the realized groups have different prognostic covariates. No effect is asserted for a mechanism merely because it was declared in a specification.",
        "",
        "Reproduction: the bundled study estimates and independent-reference summaries regenerate all losses, uncertainty and paired comparisons. With --family-archive, the reader also refits every finite-cell analysis from the public participant-level native records. The original recovery reader refits the repeated estimates from recovery-records.zip; source.json pins the archives and projection files. No paid calls, generation or private benchmark imports are involved.",
        "",
        "Alternative text: Four panels show every repeated analysis in asthma, respiratory-infection clusters, Parkinson disability and tumour response. Points show signed normalized bias with Monte Carlo uncertainty. The routine tumour-response method has a large negative error while stratified adjudication is much closer to zero; asthma and cluster differences are more modest, with condition-specific reversals retained in the paired-loss table.",
    ]
    (output / "mechanism_comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run the disclosed mechanism analyses from bundled or separately supplied inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family-archive", type=Path)
    args = parser.parse_args()
    review(args.source, args.output, family_archive=args.family_archive)


if __name__ == "__main__":
    main()
