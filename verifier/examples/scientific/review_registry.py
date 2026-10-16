"""Compare disclosed trial designs with the registry cohort used for calibration.

Run with the bundled aggregate inputs; optional original archives reproduce their
projection. Registry trials are the comparison units, never synthetic patients.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class FamilyTarget(TypedDict):
    """Registry cohort identity and clinical labels used by this comparison."""

    family_id: str
    label: str
    tier: str


class RegistryTargets(TypedDict):
    """Family targets projected from the disclosed calibration document."""

    families: list[FamilyTarget]


def project_native_trials(archive_path: Path) -> pd.DataFrame:
    """Read enrollment, occupied sites and endpoint availability from native records."""
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        cells = pd.read_csv(io.BytesIO(archive.read("cells.csv")))
        for cell in cells.itertuples():
            participants = pd.read_parquet(io.BytesIO(archive.read(f"{cell.cell}/participant_records.parquet")))
            data = pd.read_parquet(io.BytesIO(archive.read(f"{cell.cell}/analysis.parquet")))
            if not participants.participant_id.is_unique:
                raise ValueError("Native enrollment requires unique participant identities")
            row = {
                "cell": cell.cell,
                "family_id": str(cell.cell).rsplit("-", 1)[0],
                "condition": cell.condition,
                "seed": cell.seed,
                "planned_n": cell.planned_n,
                "randomized_n": len(participants),
                "occupied_sites": participants.site_id.nunique(),
                "horizon_days": cell.horizon,
                "endpoint_family": cell.family,
            }
            for arm in ("active", "control"):
                row[f"{arm}_randomized_n"] = int(participants.assigned_arm.eq(arm).sum())
            # An absent measurement is not a recorded discontinuation. These
            # quantities are disclosed but never substituted for registry flow.
            if cell.family != "time_to_event":
                terminal = (
                    data.loc[np.isclose(data.occurrence_time, cast(float, cell.horizon))]
                    if "occurrence_time" in data
                    else data
                )
                if terminal.participant_id.duplicated().any():
                    raise ValueError("Endpoint availability needs one final record per participant")
                for arm in ("active", "control"):
                    randomized = participants.loc[participants.assigned_arm.eq(arm), "participant_id"]
                    observed = terminal.loc[terminal.assigned_arm.eq(arm)].set_index("participant_id")[cell.endpoint]
                    row[f"{arm}_endpoint_unavailable"] = float(observed.reindex(randomized).isna().mean())
            rows.append(row)
    return pd.DataFrame(rows)


def project_registry_trials(metrics: Path, targets: RegistryTargets) -> pd.DataFrame:
    """Keep the exact adopted cohort and the fields needed for direct comparisons."""
    trials = pd.read_csv(metrics, low_memory=False)
    tiers = {row["family_id"]: row["tier"] for row in targets["families"] if row["family_id"].startswith("TE-")}
    selected = trials.family_id.isin(tiers) & (trials.endpoint_matched | trials.family_id.map(tiers).eq("indication"))
    return trials.loc[
        selected,
        ["family_id", "nct_id", "endpoint_matched", "enrollment", "sites", "has_results"],
    ].copy()


def compare_trials(native: pd.DataFrame, registry: pd.DataFrame, targets: RegistryTargets) -> pd.DataFrame:
    """Compare each native realization with a finite, family-specific trial distribution."""
    if registry.duplicated(["family_id", "nct_id"]).any() or native.cell.duplicated().any():
        raise ValueError("Registry family/trial and native clinical-cell identities must be unique")
    families = {row["family_id"]: row for row in targets["families"]}
    rows = []
    for trial in native.itertuples():
        family = families[str(trial.family_id)]
        reference = registry.loc[registry.family_id.eq(trial.family_id)]
        for metric, field, registry_field, unit in (
            ("Randomized participants", "randomized_n", "enrollment", "participants"),
            ("Occupied recruitment sites", "occupied_sites", "sites", "sites"),
        ):
            values = pd.to_numeric(reference[registry_field], errors="raise")
            finite = values[np.isfinite(values)]
            if finite.empty:
                raise ValueError("The disclosed design comparison has no finite registry observations")
            q1, median, q3 = finite.quantile([0.25, 0.5, 0.75])
            value = float(getattr(trial, field))
            # The IQR describes heterogeneous registry studies, not a predictive
            # interval, a pass criterion or uncertainty in the registry median.
            sparse = len(finite) < 5
            relation = "below" if value < q1 else "above" if value > q3 else "inside"
            sites_caveat = metric == "Occupied recruitment sites"
            rows.append(
                {
                    "cell": trial.cell,
                    "family_id": trial.family_id,
                    "clinical_setting": family["label"],
                    "condition": trial.condition,
                    "metric": metric,
                    "unit": unit,
                    "native_value": value,
                    "registry_trials": len(finite),
                    "registry_missing_or_nonfinite": len(reference) - len(finite),
                    "registry_lower_quartile": q1,
                    "registry_median": median,
                    "registry_upper_quartile": q3,
                    "native_to_registry_median": value / median if median > 0 else np.nan,
                    "position_relative_to_iqr": "sparse_reference" if sparse else relation,
                    "cohort": family["tier"],
                    "comparison_role": "calibration_source_comparison",
                    "interpretation": (
                        "Registry registered facilities and occupied synthetic sites differ; facilities do not identify clusters."
                        if sites_caveat
                        else "Actual registry enrollment compared with synthetic randomized participants visible at the analysis cutoff."
                    ),
                }
            )
    return pd.DataFrame(rows)


def review(
    source: Path, output: Path, *, family_archive: Path | None = None, registry_metrics: Path | None = None
) -> None:
    """Write all-cell comparisons, semantic exclusions and a family-level display."""
    output.mkdir(parents=True, exist_ok=True)
    inputs = source / "registry"
    targets = cast(RegistryTargets, json.loads((inputs / "calibration_targets.json").read_text()))
    provenance = json.loads((inputs / "source.json").read_text())
    native = pd.read_csv(inputs / "native_trials.csv")
    if family_archive is not None:
        projected = project_native_trials(family_archive)
        try:
            pd.testing.assert_frame_equal(projected, native, check_exact=False, rtol=1e-12, atol=1e-12)
        except AssertionError as error:
            raise ValueError("Native archive projection differs from the declared source inputs") from error
        native = projected
    registry = (
        project_registry_trials(registry_metrics, targets)
        if registry_metrics
        else pd.read_csv(inputs / "registry_trials.csv")
    )
    registry.to_csv(output / "registry_reference_trials.csv", index=False)
    (output / "registry_sources.txt").write_text(json.dumps(provenance, indent=2) + "\n")
    comparisons = compare_trials(native, registry, targets)
    comparisons["generation_role"] = "current_generator_probe"
    comparisons["generation_date"] = provenance["native_source"]["generation_date"]
    comparisons.to_csv(output / "registry_comparisons.csv", index=False)
    archived = pd.read_csv(inputs / "archived_native_trials.csv")
    archived_comparisons = compare_trials(archived, registry, targets)
    archived_comparisons["generation_role"] = "archived_teaching_snapshot"
    archived_comparisons["archive_revision_date"] = "2026-09-25"
    archived_comparisons.to_csv(output / "registry_archived_comparisons.csv", index=False)
    native.to_csv(output / "registry_native_trials.csv", index=False)
    declarations = pd.read_csv(inputs / "current_loss_declarations.csv")
    loss_records = declarations.merge(native, on="cell", validate="one_to_one")
    observed_loss = []
    labels = {row["family_id"]: row["label"] for row in targets["families"]}
    for trial in loss_records.itertuples():
        process_note = (
            "Declared withdrawal acts alongside 98% visit-observation probability; final absence includes both."
            if trial.family_id in ("TE-S10", "TE-S11")
            else (
                "History-dependent contact loss uses normal-driver calibration; realized histories need not reproduce that marginal probability."
                if trial.driver == "observed_history"
                else (
                    "Baseline-risk-dependent final observation uses normal-driver calibration; realized risk mix affects final availability."
                    if trial.driver == "baseline_risk"
                    else "Declared random observation loss and final absence are shown for one realization; no repeated calibration test is performed."
                )
            )
        )
        for arm in ("active", "control"):
            declared = float(getattr(trial, f"{arm}_by_horizon"))
            observed = float(getattr(trial, f"{arm}_endpoint_unavailable"))
            denominator = int(getattr(trial, f"{arm}_randomized_n"))
            observed_loss.append(
                {
                    "cell": trial.cell,
                    "family_id": trial.family_id,
                    "clinical_setting": labels[str(trial.family_id)],
                    "arm": arm,
                    "randomized_participants": denominator,
                    "declared_native_loss_parameter": declared,
                    "observed_endpoint_unavailable_fraction": observed,
                    "observed_endpoint_unavailable_count": round(observed * denominator),
                    "driver": trial.driver,
                    "calibration_origin": trial.source,
                    "observation_process_note": process_note,
                    "comparison_role": "single_realization_native_observation_process",
                }
            )
    pd.DataFrame(observed_loss).to_csv(output / "registry_observation_loss.csv", index=False)
    exclusions = pd.read_csv(inputs / "comparison_limits.csv")
    exclusions.to_csv(output / "registry_comparison_limits.csv", index=False)
    families = comparisons.loc[comparisons.cell.str.endswith("-A1")].copy()
    if families.family_id.nunique() != native.family_id.nunique():
        raise ValueError("The family overview requires every family's baseline condition")
    families.to_csv(output / "registry_family_overview.csv", index=False)
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 7.3), layout="constrained", sharey=True)
        for axis, metric in zip(axes, ("Randomized participants", "Occupied recruitment sites"), strict=True):
            data = families.loc[families.metric.eq(metric)].sort_values("family_id")
            y = np.arange(len(data))
            axis.hlines(
                y, data.registry_lower_quartile, data.registry_upper_quartile, color="0.65", lw=4, label="Registry IQR"
            )
            axis.scatter(data.registry_median, y, marker="|", color="0.25", s=90, label="Registry median")
            axis.scatter(data.native_value, y, marker="o", color="#226B93", s=24, zorder=3, label="Native trial")
            axis.set_xscale("log")
            axis.set_xlabel(metric + " (log scale)")
            axis.set_yticks(y, [f"{row.clinical_setting}\n{row.family_id}" for row in data.itertuples()])
            axis.grid(axis="x", color="0.92")
        axes[0].invert_yaxis()
        counts = families.pivot(index="family_id", columns="metric", values="registry_trials")
        setting_labels = families.drop_duplicates("family_id").set_index("family_id").clinical_setting
        axes[0].set_yticklabels(
            [
                f"{setting_labels[family]}\n{family} · n={int(counts.loc[family, 'Randomized participants'])}/{int(counts.loc[family, 'Occupied recruitment sites'])}"
                for family in sorted(setting_labels.index)
            ]
        )
        figure.suptitle("Registry trial counts in labels: enrollment / sites", fontsize=10)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, legend_labels, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(output / f"registry_comparison.{suffix}", dpi=180)
        plt.close(figure)
    enrollment = families.loc[families.metric.eq("Randomized participants")]
    supported = enrollment.loc[enrollment.registry_trials.ge(5)]
    outside = supported.loc[~supported.position_relative_to_iqr.eq("inside")]
    lines = [
        "# How the disclosed trials compare with their registry calibration source",
        "",
        f"The comparison retains all {native.cell.nunique()} clinical conditions across {native.family_id.nunique()} families. The figure uses each family's A1 condition once; the full table retains every condition. The registry projection has {len(registry):,} family–trial matches and {registry.nct_id.nunique():,} unique registered trials. A trial can contribute to multiple families. Each grey band is an empirical interquartile range across trials; n is the number of finite contributing values.",
        "",
        f"The primary comparison uses a fresh fixed-seed A1 study in every family, generated on {provenance['native_source']['generation_date']} from construction revision {provenance['native_source']['source_commit']}. The 17 cells, seeds and quantities were declared before this bounded probe. Its generator source trees match the recorded qualification source revision, but the probe runs no release qualification or independent reference-bank generation. One realization per family cannot establish a generated population distribution or repeated-study performance. The exact study declarations and minimal participant/endpoint records are disclosed in current-records.zip.",
        "",
        "The separate registry_archived_comparisons.csv retains all 30 conditions from the September teaching archive. Those records predate the October registry-driven realism changes and are not substituted for the current probe. The mechanism walkthrough still concerns that explicitly dated archived teaching panel.",
        "",
        f"Of the {len(supported)} family cohorts with at least five enrollment observations, {len(outside)} native A1 trials fall outside the registry enrollment IQR. This describes design departures, not failures against a statistical tolerance. The benchmark's information-based sample sizes can differ substantially from typical registered enrollment.",
        "",
        "| Clinical setting | Native participants | Registry median (IQR) | Registry trials |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in outside.itertuples():
        lines.append(
            f"| {row.clinical_setting} ({row.family_id}) | {row.native_value:g} | {row.registry_median:g} ({row.registry_lower_quartile:g}–{row.registry_upper_quartile:g}) | {row.registry_trials} |"
        )
    lines += ["", "The sparse cohorts remain visible without an IQR position classification:", ""]
    for row in enrollment.loc[enrollment.registry_trials.lt(5)].itertuples():
        lines.append(
            f"- {row.clinical_setting}: {row.native_value:,.0f} native participants; registry median {row.registry_median:,.1f} from {row.registry_trials} trials."
        )
    lines += [
        "",
        "![Enrollment and recruitment sites compared with family-specific registry distributions.](registry_comparison.png)",
        "",
        "The AACT snapshot is the source used to choose calibration targets. Agreement therefore supports reproduction of those design envelopes, not independent external validation, clinical transportability, or performance on real patient data. Clinical indication, phase, design and primary-outcome text define the cohorts; endpoint matching is absent for the first-exacerbation asthma family, which uses its broader indication cohort. The respiratory-infection cluster cohort contains only four trials and the obstetric stepped-wedge cohort only one. Their plotted ranges are descriptive and do not establish representative distributions.",
        "",
        "Registered facilities are an imperfect comparator for occupied recruitment sites and cannot be interpreted as cluster counts. Synthetic sites with no recruited participants are absent from the occupied-site count. Primary-completion duration includes accrual and follow-up; it is not the individual endpoint horizon. Endpoint unavailability in the native records is disclosed separately, but it is not registry noncompletion. Oncology noncompletion can include progression and death. No endpoint-availability proportion is scored against a registry dropout band.",
        "",
        "The separate registry_observation_loss.csv reports the current native loss parameter and actual final endpoint unavailability in each randomized arm. It includes every nonsurvival A1 family, with exact arm denominators. These are model parameters and observed proportions in one finite realization, not a direct registry comparison or a repeated-study calibration test. The Parkinson and HbA1c families retain a 98% visit-observation probability alongside withdrawal. The asthma-control and albuminuria families apply contact loss through observed histories, calibrated under a normal driver distribution. Baseline-risk observation likewise depends on the realized risk mix. Equality of final unavailability to the declared loss parameter is therefore not assumed. The current archive supplies the exact study specifications and realized records for inspecting these processes.",
        "",
        "Registry safety windows, endpoint units, estimands and populations are not sufficiently aligned here to compare treatment effects, safety incidence or baseline patient distributions. Those exclusions remain in registry_comparison_limits.csv. Same-source endpoint calibration labels do not prove that these quantities match in generated trials. The registry cannot identify the cohort's causal truth or validate every authored analysis assumption.",
        "",
        "The analysis uses exact trial-level registry values for quantiles (linear interpolation), excludes missing and nonfinite values with explicit counts, and performs no hypothesis tests. Current native trial values come from unique participant records in current-records.zip; the historical comparison uses family-records.zip. The source record pins both archive identities. Optional --family-archive and --registry-metrics arguments repeat the current native and registry projections from the original records.",
        "",
        "Alternative text: Seventeen clinical settings appear in rows. Registry median ticks and interquartile bars are compared with native enrollment and occupied-site circles on logarithmic axes. Many enrollment circles lie above their registry bands; the two sparse design cohorts are retained with their sample counts.",
    ]
    (output / "registry_comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run the registry comparison using only public inputs and numerical libraries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family-archive", type=Path)
    parser.add_argument("--registry-metrics", type=Path)
    args = parser.parse_args()
    review(args.source, args.output, family_archive=args.family_archive, registry_metrics=args.registry_metrics)


if __name__ == "__main__":
    main()
