"""Recalculate the public real-trial analysis experiment without benchmark imports.

The separately acquired RCT Bench checkout stays outside this aggregate package.
This reproduces numerical analyses and paired resampling, not the Docker replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wasserstein_distance


class Selection(TypedDict):
    """Only the source-selection fields consumed by this reader."""

    trial_id: int
    source_sha256: str
    source_rows: int
    included: bool
    reason: str
    complete_rows: int
    source_endpoint_type: str
    arm_labels: list[str]
    historical_overlap: bool


class Metric(TypedDict):
    """One equal-trial-weight paired comparison."""

    metric: str
    whole_subject: float
    permuted_linkage: float
    adverse_minus_whole: float
    lower: float
    upper: float
    trials: int


class Reference(TypedDict):
    """Aggregate source identity and experiment settings used for reproduction."""

    source_revision: str
    source_metadata_sha256: str
    source_trial_count: int
    eligible_trial_count: int
    complete_participants: int
    fidelity_trial_count: int
    fidelity_complete_participants: int
    seed: int
    resamples_per_trial: int
    bootstrap_replicates: int
    selection: list[Selection]
    fidelity_selection: list[Selection]
    paired_metrics: list[Metric]


def _screen(data: pd.DataFrame, endpoint_type: str, fidelity: bool) -> tuple[str, list[str]]:
    """Repeat the structural screen before calculating any treatment contrast."""
    names = [str(column) for column in data.columns]
    if endpoint_type not in {"Continuous", "Binary", "Count"}:
        return "unsupported_source_endpoint_type", []
    if set(data) & {"Participant_ID", "Crossover_Sequence", "Crossover_Period", "Assessment_Window"}:
        return "explicit_repeated_or_crossover_design", []
    if "Treatment" not in data or data.Treatment.isna().any() or data.Treatment.nunique() != 2:
        return "requires_two_complete_arms", []
    outcomes = [column for column in names if column.startswith("YP_")]
    ages = [column for column in names if column.lower().startswith("x_age")]
    bmis = [column for column in names if column.lower().startswith("x_bmi")]
    if len(outcomes) != 1:
        return "requires_unique_primary_endpoint", []
    if fidelity and (len(ages) != 1 or len(bmis) != 1):
        return "requires_unique_age_bmi", []
    candidates = [
        column
        for column in names
        if column.startswith("X_") and pd.api.types.is_numeric_dtype(data[column]) and data[column].nunique() >= 10
    ]
    age = [column for column in candidates if column.lower().startswith("x_age")]
    bmi = [column for column in candidates if column.lower().startswith("x_bmi")]
    chosen = age if len(age) == 1 else bmi if len(bmi) == 1 else candidates
    if not fidelity and len(chosen) != 1:
        return "requires_unambiguous_numeric_covariate", []
    columns = [outcomes[0], ages[0], bmis[0]] if fidelity else [outcomes[0], chosen[0]]
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in columns):
        return "requires_numeric_measurements", columns
    complete = data[["Treatment", *columns]].dropna()
    if not np.isfinite(complete[columns].to_numpy(dtype=float)).all():
        return "nonfinite_measurement", columns
    if len(complete) < 40 or complete.groupby("Treatment").size().min() < 10 or complete.Treatment.nunique() != 2:
        return "insufficient_complete_case_arm_support", columns
    if any(complete[column].nunique() < 10 for column in columns[1:]):
        return "insufficient_continuous_covariate_support", columns
    return "eligible", columns


def _fit(data: pd.DataFrame, *, reverse: bool = False) -> tuple[float, float, float, float]:
    """Fit the declared adjusted OLS contrast, HC1 covariance and t interval."""
    active = data.arm.eq(sorted(data.arm.unique())[1]).to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(data)), 1 - active if reverse else active, data.iloc[:, 2:]))
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Declared OLS design is rank deficient; no silent fit exclusions")
    fit = sm.OLS(data.outcome, design).fit(cov_type="HC1", use_t=True)
    lower, upper = np.asarray(fit.conf_int(alpha=0.05))[1]
    return float(np.asarray(fit.params)[1]), float(np.asarray(fit.bse)[1]), float(lower), float(upper)


def _correlations(data: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Three pooled Spearman correlations, distinct from arm-specific metrics."""
    return data.iloc[:, 1:].corr(method="spearman").to_numpy()[np.triu_indices(3, 1)]


def _resamples(data: pd.DataFrame, trial_id: int, reference: Reference) -> list[dict[str, float | int | str]]:
    """Preserve arm sizes and marginals while perturbing participant linkage."""
    records: list[dict[str, float | int | str]] = []
    source_estimate, source_se, _, _ = _fit(data)
    correlations = _correlations(data)
    scale = data.iloc[:, 1:].std(ddof=1)
    for replicate in range(reference["resamples_per_trial"]):
        seed = int(np.random.SeedSequence([reference["seed"], trial_id, replicate]).generate_state(1)[0])
        rng = np.random.RandomState(seed)
        whole = pd.concat(
            [arm.iloc[rng.randint(len(arm), size=len(arm))] for _, arm in data.groupby("arm", sort=True)],
            ignore_index=True,
        )
        permuted = whole.copy()
        for indices in whole.groupby("arm", sort=True).indices.values():
            for column in data.columns[1:]:
                permuted.loc[indices, column] = rng.permutation(whole.loc[indices, column].to_numpy())
        for method, draw in (("whole_subject", whole), ("permuted_linkage", permuted)):
            estimate, se, _, _ = _fit(draw)
            records.append(
                {
                    "trial_id": trial_id,
                    "replicate": replicate,
                    "method": method,
                    "standardized_wasserstein": float(
                        np.mean(
                            [
                                wasserstein_distance(data[column], draw[column]) / scale[column]
                                for column in data.columns[1:]
                            ]
                        )
                    ),
                    "dependence_error": float(np.mean(abs(_correlations(draw) - correlations))),
                    "estimate_shift_in_source_se": (estimate - source_estimate) / source_se,
                    "standard_error_ratio": se / source_se,
                }
            )
    return records


def review(source: Path, reference_directory: Path, output: Path) -> None:
    """Reconstruct the pinned experiment from public participant source records.

    Parameters
    ----------
    source
        Pinned RCT Bench Git checkout with cleaned_data and website/assets data.
    reference_directory
        Aggregate reference JSON and CSVs distributed with this reader.
    output
        New directory for aggregate recalculations; no patient records are saved.

    Raises
    ------
    ValueError
        If source identity, selection, numerical support or results disagree.
    FileExistsError
        If the output directory already exists.
    """
    reference = cast(Reference, json.loads((reference_directory / "external-reconstruction.json").read_text()))
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    metadata_path = source / "website/assets/site-data.json"
    if (
        revision != reference["source_revision"]
        or hashlib.sha256(metadata_path.read_bytes()).hexdigest() != reference["source_metadata_sha256"]
    ):
        raise ValueError("Source revision or metadata bytes differ from the disclosed experiment")
    types = {row["id"]: row["primaryOutcomeType"] for row in json.loads(metadata_path.read_text())["trials"]}
    paths = {int(path.stem[5:]): path for path in (source / "cleaned_data").glob("trial*.csv")}
    if (
        set(paths) != {row["trial_id"] for row in reference["selection"]}
        or len(paths) != reference["source_trial_count"]
    ):
        raise ValueError("Source trial roster differs from the disclosed screen")
    output.mkdir(parents=True, exist_ok=False)
    reconstructed, resamples = [], []
    participants = [0, 0]
    counts = [0, 0]
    for fidelity, selections in enumerate((reference["selection"], reference["fidelity_selection"])):
        for trial in selections:
            trial_id = trial["trial_id"]
            path = paths[trial_id]
            if hashlib.sha256(path.read_bytes()).hexdigest() != trial["source_sha256"]:
                raise ValueError(f"Source bytes changed for trial {trial_id}")
            data = pd.read_csv(path)
            reason, columns = _screen(data, types[trial_id], bool(fidelity))
            if (
                reason != trial["reason"]
                or (reason == "eligible") != trial["included"]
                or len(data) != trial["source_rows"]
            ):
                raise ValueError(f"Structural selection disagrees for trial {trial_id}")
            if reason != "eligible":
                continue
            data = data[["Treatment", *columns]].dropna().reset_index(drop=True)
            data.columns = ["arm", "outcome", *[f"covariate{index}" for index in range(len(columns) - 1)]]
            data["arm"] = data.arm.astype(str)
            if len(data) != trial["complete_rows"] or sorted(data.arm.unique()) != trial["arm_labels"]:
                raise ValueError(f"Complete-case support or arm orientation changed for trial {trial_id}")
            counts[fidelity] += 1
            participants[fidelity] += len(data)
            if fidelity:
                resamples.extend(_resamples(data, trial_id, reference))
                print(f"Recalculated paired linkage experiment for trial {trial_id}", flush=True)
            else:
                estimate, se, lower, upper = _fit(data)
                reconstructed.append(
                    {
                        "trial_id": trial_id,
                        "estimate": estimate,
                        "standard_error": se,
                        "lower": lower,
                        "upper": upper,
                        "reversed_estimate": _fit(data, reverse=True)[0],
                    }
                )
    if counts != [reference["eligible_trial_count"], reference["fidelity_trial_count"]] or participants != [
        reference["complete_participants"],
        reference["fidelity_complete_participants"],
    ]:
        raise ValueError("Selected cohort counts differ from the reference")
    reconstruction = pd.DataFrame(reconstructed).set_index("trial_id").sort_index()
    expected = pd.read_csv(reference_directory / "source_reconstruction.csv").set_index("trial_id").sort_index()
    np.testing.assert_allclose(reconstruction, expected[reconstruction.columns], rtol=1e-8, atol=1e-10)
    reconstruction.to_csv(output / "source_reconstruction.csv")
    worlds = pd.DataFrame(resamples)
    if not np.isfinite(worlds.select_dtypes(include="number")).all().all():
        raise ValueError("Undefined resampling metric; no retries or silent exclusions")
    worlds.to_csv(output / "resample_metrics.csv", index=False)
    means = worlds.groupby(["trial_id", "method"], sort=True).mean(numeric_only=True).drop(columns="replicate")
    means["absolute_mean_shift_in_source_se"] = means.pop("estimate_shift_in_source_se").abs()
    expected_means = pd.read_csv(reference_directory / "trial_summary.csv").set_index(["trial_id", "method"])
    np.testing.assert_allclose(means, expected_means.loc[means.index, means.columns], rtol=1e-8, atol=1e-10)
    means.to_csv(output / "trial_summary.csv")
    metrics = []
    for metric in means:
        values = means[metric].unstack("method")
        difference = (values.permuted_linkage - values.whole_subject).to_numpy()
        indices = np.random.default_rng(reference["seed"]).integers(
            len(difference), size=(reference["bootstrap_replicates"], len(difference))
        )
        lower, upper = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
        metrics.append(
            {
                "metric": metric,
                "whole_subject": values.whole_subject.mean(),
                "permuted_linkage": values.permuted_linkage.mean(),
                "adverse_minus_whole": difference.mean(),
                "lower": lower,
                "upper": upper,
                "trials": len(difference),
            }
        )
    paired = pd.DataFrame(metrics).set_index("metric")
    expected_paired = pd.DataFrame(reference["paired_metrics"]).set_index("metric")
    np.testing.assert_allclose(paired, expected_paired.loc[paired.index, paired.columns], rtol=1e-8, atol=1e-10)
    paired.to_csv(output / "paired_metrics.csv")
    (output / "review.json").write_text(
        json.dumps(
            {
                "source_revision": revision,
                "source_trials_checked": len(paths),
                "reconstructed_trials": counts[0],
                "complete_participants": participants[0],
                "fidelity_trials": counts[1],
                "fidelity_complete_participants": participants[1],
                "resamples_per_trial": reference["resamples_per_trial"],
                "bootstrap_replicates": reference["bootstrap_replicates"],
                "maximum_reconstruction_discrepancy": float(
                    abs(reconstruction - expected[reconstruction.columns]).to_numpy().max()
                ),
                "maximum_trial_summary_discrepancy": float(
                    abs(means - expected_means[means.columns]).to_numpy().max()
                ),
                "maximum_paired_metric_discrepancy": float(
                    abs(paired - expected_paired[paired.columns]).to_numpy().max()
                ),
                "scope": "Direct OLS-HC1 and paired resampling reproduction; does not rerun or verify the archived Docker execution.",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Run a public source reconstruction with explicit source and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parent / "external")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review(args.source, args.reference, args.output)


if __name__ == "__main__":
    main()
