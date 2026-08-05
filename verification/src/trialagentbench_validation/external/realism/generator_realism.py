"""External realism fingerprints for generated clinical trials."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from io import BytesIO
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Literal, cast
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr, wasserstein_distance
from statsmodels.duration.hazard_regression import PHReg


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _TrialBaselineFieldsV1(_FrozenModel):
    """Shared trial-level marginal and joint-structure fields."""

    trial_id: str = Field(min_length=1)
    source: Literal["external_rct", "generated_trial"]
    participants: int = Field(ge=20)
    age_mean: float = Field(allow_inf_nan=False)
    age_sd: float = Field(gt=0, allow_inf_nan=False)
    bmi_mean: float = Field(allow_inf_nan=False)
    bmi_sd: float = Field(gt=0, allow_inf_nan=False)
    age_bmi_spearman: float = Field(ge=-1, le=1, allow_inf_nan=False)
    maximum_scaled_baseline_smd: float = Field(ge=0, allow_inf_nan=False)


class TrialBaselineFingerprintV1(_TrialBaselineFieldsV1):
    """Marginal and joint-structure summary of one independent trial."""

    schema_id: Literal["trialagentbench.trial_baseline_fingerprint/v1"] = (
        "trialagentbench.trial_baseline_fingerprint/v1"
    )


class TrialRealismFingerprintV1(_TrialBaselineFieldsV1):
    """Endpoint-aware, scale-free summary of one independent trial."""

    schema_id: Literal["trialagentbench.trial_realism_fingerprint/v1"] = (
        "trialagentbench.trial_realism_fingerprint/v1"
    )
    outcome_kind: Literal["binary", "continuous", "time_to_event"]
    adjustment_shift_in_unadjusted_se: float = Field(ge=0, allow_inf_nan=False)
    adjusted_to_unadjusted_se_ratio: float = Field(gt=0, allow_inf_nan=False)


class GeneratorRealismConstructV1(_FrozenModel):
    """External comparison for one trial-level realism construct."""

    construct_id: Literal[
        "age_mean",
        "age_sd",
        "bmi_mean",
        "bmi_sd",
        "age_bmi_spearman",
        "maximum_scaled_baseline_smd",
        "adjustment_shift_in_unadjusted_se",
        "adjusted_to_unadjusted_se_ratio",
    ]
    validation_domain: Literal["marginal", "joint_structure", "analysis_impact"]
    external_trials: int = Field(ge=3)
    synthetic_trials: int = Field(ge=3)
    external_median: float = Field(allow_inf_nan=False)
    external_median_ci_low: float = Field(allow_inf_nan=False)
    external_median_ci_high: float = Field(allow_inf_nan=False)
    synthetic_median: float = Field(allow_inf_nan=False)
    synthetic_median_ci_low: float = Field(allow_inf_nan=False)
    synthetic_median_ci_high: float = Field(allow_inf_nan=False)
    standardized_wasserstein: float = Field(ge=0, allow_inf_nan=False)
    standardized_wasserstein_ci_low: float = Field(ge=0, allow_inf_nan=False)
    standardized_wasserstein_ci_high: float = Field(ge=0, allow_inf_nan=False)
    external_split_reference_p95: float = Field(ge=0, allow_inf_nan=False)
    within_external_split_reference: bool


class GeneratorRealismSummaryV1(_FrozenModel):
    """Public production-generator realism comparison."""

    schema_id: Literal["trialagentbench.generator_realism_summary/v1"] = (
        "trialagentbench.generator_realism_summary/v1"
    )
    external_baseline_trials: int = Field(ge=3)
    external_analysis_trials: int = Field(ge=3)
    synthetic_trials: int = Field(ge=3)
    external_baseline_participants: int = Field(ge=20)
    external_analysis_participants: int = Field(ge=20)
    synthetic_participants: int = Field(ge=20)
    constructs: tuple[GeneratorRealismConstructV1, ...] = Field(
        min_length=8, max_length=8
    )


_CONSTRUCT_DOMAINS: dict[
    str, Literal["marginal", "joint_structure", "analysis_impact"]
] = {
    "age_mean": "marginal",
    "age_sd": "marginal",
    "bmi_mean": "marginal",
    "bmi_sd": "marginal",
    "age_bmi_spearman": "joint_structure",
    "maximum_scaled_baseline_smd": "joint_structure",
    "adjustment_shift_in_unadjusted_se": "analysis_impact",
    "adjusted_to_unadjusted_se_ratio": "analysis_impact",
}


def fingerprint_trial_baseline(
    frame: pd.DataFrame,
    *,
    trial_id: str,
    source: Literal["external_rct", "generated_trial"],
) -> TrialBaselineFingerprintV1:
    """Fingerprint baseline marginals, dependence, and arm balance."""

    values = _standard_frame(frame, outcome_columns=(), require_two_arms=False)
    return _build_baseline_fingerprint(
        values,
        trial_id=trial_id,
        source=source,
    )


def fingerprint_standard_trial(
    frame: pd.DataFrame,
    *,
    trial_id: str,
    source: Literal["external_rct", "generated_trial"],
    outcome_kind: Literal["binary", "continuous"],
) -> TrialRealismFingerprintV1:
    """Fingerprint a two-arm trial with a binary or continuous endpoint."""

    values = _standard_frame(frame, outcome_columns=("outcome",), require_two_arms=True)
    unadjusted = _ols_treatment_analysis(values, covariates=())
    adjusted = _ols_treatment_analysis(values, covariates=("age", "bmi"))
    return _build_fingerprint(
        values,
        trial_id=trial_id,
        source=source,
        outcome_kind=outcome_kind,
        unadjusted=unadjusted,
        adjusted=adjusted,
    )


def fingerprint_survival_trial(
    frame: pd.DataFrame,
    *,
    trial_id: str,
) -> TrialRealismFingerprintV1:
    """Fingerprint a generated two-arm time-to-event trial."""

    values = _standard_frame(
        frame, outcome_columns=("time", "event"), require_two_arms=True
    )
    if bool((values["time"] <= 0).any()):
        raise ValueError("survival durations must be positive")
    if not set(values["event"].unique()).issubset({0.0, 1.0}):
        raise ValueError("survival event indicator must be binary")
    if int(values["event"].sum()) < 10:
        raise ValueError("survival analysis requires at least ten events")
    unadjusted = _cox_treatment_analysis(values, covariates=())
    adjusted = _cox_treatment_analysis(values, covariates=("age", "bmi"))
    return _build_fingerprint(
        values,
        trial_id=trial_id,
        source="generated_trial",
        outcome_kind="time_to_event",
        unadjusted=unadjusted,
        adjusted=adjusted,
    )


def extract_generated_trial_fingerprints(
    participant_release: Path,
    *,
    target_participant_counts: tuple[int, ...] | None = None,
    seed: int = 451_015,
    skip_insufficient_endpoint_information: bool = False,
) -> tuple[TrialRealismFingerprintV1, ...]:
    """Extract one fingerprint per independent trial from public release bytes.

    When target participant counts are supplied, each trial is deterministically
    sampled without replacement before analysis. This provides a fidelity view
    at empirical trial sizes alongside the full-size qualification view.
    """

    if target_participant_counts is not None and (
        not target_participant_counts or min(target_participant_counts) < 20
    ):
        raise ValueError("target_participant_counts must contain values of at least 20")
    with ZipFile(participant_release) as archive:
        names = _validated_archive_members(archive)
        candidates: dict[
            tuple[str, str],
            list[tuple[Literal["prepared", "reconstructed", "raw"], str]],
        ] = defaultdict(list)
        for task_member in sorted(
            name
            for name in names
            if name.startswith("items/") and name.endswith("/task.json")
        ):
            prefix = task_member.rsplit("/", 1)[0]
            task = _json_object(archive, task_member)
            protocol_name = _required_text(task, "protocol_summary_file", task_member)
            protocol_member = f"{prefix}/{protocol_name}"
            protocol = _json_object(archive, protocol_member)
            trial_id = _required_text(protocol, "trial_id", protocol_member)
            study_id = _required_text(protocol, "study_id", protocol_member)
            view = _participant_view(names, prefix=prefix, task_member=task_member)
            candidates[(trial_id, study_id)].append((view, prefix))
        if not candidates:
            raise ValueError("participant release contains no trial tasks")

        fingerprints = []
        for trial_index, ((trial_id, study_id), views) in enumerate(
            sorted(candidates.items())
        ):
            analysis_views = sorted(
                (view, prefix)
                for view, prefix in views
                if view in {"prepared", "reconstructed"}
            )
            if not analysis_views:
                raise ValueError(
                    f"{trial_id}:{study_id} has raw Context views but no matched analysis-ready view"
                )
            preferred_view: Literal["prepared", "reconstructed"] = (
                "prepared"
                if any(view == "prepared" for view, _ in analysis_views)
                else "reconstructed"
            )
            preferred_prefixes = tuple(
                prefix for view, prefix in analysis_views if view == preferred_view
            )
            _validate_analysis_view_equivalence(
                archive,
                prefixes=preferred_prefixes,
                view=preferred_view,
                trial_id=trial_id,
                study_id=study_id,
            )
            prefix = preferred_prefixes[0]
            data_prefix = (
                f"{prefix}/data"
                if preferred_view == "prepared"
                else f"{prefix}/data/public_reconstruction"
            )
            task = _json_object(archive, f"{prefix}/task.json")
            endpoint_term = _required_text(
                task, "primary_endpoint_term", f"{prefix}/task.json"
            )
            protocol = _json_object(
                archive,
                f"{prefix}/{_required_text(task, 'protocol_summary_file', f'{prefix}/task.json')}",
            )
            design_family = _required_text(
                protocol, "design_family", f"{prefix}/protocol_summary.json"
            )
            endpoint = _json_object(archive, f"{prefix}/endpoint_definition.json")
            endpoint_id = _primary_endpoint_id(
                endpoint,
                endpoint_term=endpoint_term,
                member=f"{prefix}/endpoint_definition.json",
            )
            baseline = _read_parquet(archive, f"{data_prefix}/ADSL.parquet")
            time_to_event = _read_parquet(archive, f"{data_prefix}/ADTTE.parquet")
            treatment_arm_count = baseline["TRTA"].dropna().astype("string").nunique()
            if treatment_arm_count != 2:
                if (
                    design_family == "stepped_wedge_cluster_rollout"
                    and treatment_arm_count > 2
                ):
                    continue
                raise ValueError(
                    f"{trial_id}:{study_id} does not expose a two-arm treatment contrast"
                )
            analysis = _merge_generated_analysis_frame(
                baseline,
                time_to_event,
                endpoint_id=endpoint_id,
            )
            fingerprint_id = f"{trial_id}:{study_id}"
            if target_participant_counts is not None:
                target = target_participant_counts[
                    trial_index % len(target_participant_counts)
                ]
                if target > len(analysis):
                    raise ValueError(
                        f"{fingerprint_id} has {len(analysis)} participants, below target {target}"
                    )
                digest = hashlib.sha256(f"{seed}:{fingerprint_id}".encode()).digest()
                sample_seed = int.from_bytes(digest[:4], byteorder="big")
                analysis = analysis.sample(
                    n=target,
                    replace=False,
                    random_state=sample_seed,
                ).reset_index(drop=True)
            try:
                fingerprints.append(
                    fingerprint_survival_trial(
                        analysis,
                        trial_id=fingerprint_id,
                    )
                )
            except ValueError as error:
                if not (
                    skip_insufficient_endpoint_information
                    and str(error) == "survival analysis requires at least ten events"
                ):
                    raise
    if len(fingerprints) < 3:
        raise ValueError(
            "fewer than three generated trials meet the endpoint information requirement"
        )
    return tuple(fingerprints)


def _participant_view(
    names: set[str],
    *,
    prefix: str,
    task_member: str,
) -> Literal["prepared", "reconstructed", "raw"]:
    prepared = (
        f"{prefix}/data/ADSL.parquet",
        f"{prefix}/data/ADTTE.parquet",
    )
    reconstructed = (
        f"{prefix}/data/public_reconstruction/ADSL.parquet",
        f"{prefix}/data/public_reconstruction/ADTTE.parquet",
    )
    raw_required = (
        f"{prefix}/reconstruction_task.json",
        f"{prefix}/data/raw/subjects.parquet",
    )
    for label, members in (
        ("analysis-ready", prepared),
        ("public reconstruction", reconstructed),
        ("raw reconstruction", raw_required),
    ):
        present = tuple(member in names for member in members)
        if any(present) and not all(present):
            raise ValueError(f"{task_member} has an incomplete {label} view")
    available = tuple(
        view
        for view, members in (
            ("prepared", prepared),
            ("reconstructed", reconstructed),
            ("raw", raw_required),
        )
        if all(member in names for member in members)
    )
    if len(available) != 1:
        raise ValueError(f"{task_member} must expose exactly one declared data view")
    return cast(Literal["prepared", "reconstructed", "raw"], available[0])


def _validate_analysis_view_equivalence(
    archive: ZipFile,
    *,
    prefixes: tuple[str, ...],
    view: Literal["prepared", "reconstructed"],
    trial_id: str,
    study_id: str,
) -> None:
    if not prefixes:
        raise ValueError(f"{trial_id}:{study_id} has no {view} analysis view")
    suffix = "data" if view == "prepared" else "data/public_reconstruction"
    signatures = {
        tuple(
            (
                archive.getinfo(f"{prefix}/{suffix}/{table}.parquet").file_size,
                archive.getinfo(f"{prefix}/{suffix}/{table}.parquet").CRC,
            )
            for table in ("ADSL", "ADTTE")
        )
        for prefix in prefixes
    }
    if len(signatures) != 1:
        raise ValueError(
            f"{trial_id}:{study_id} has non-equivalent {view} Context views"
        )


def compare_generator_realism(
    external_baseline: tuple[TrialBaselineFingerprintV1, ...],
    external_analysis: tuple[TrialRealismFingerprintV1, ...],
    synthetic: tuple[TrialRealismFingerprintV1, ...],
    *,
    bootstrap_replicates: int = 2_000,
    seed: int = 451014,
) -> GeneratorRealismSummaryV1:
    """Compare production-generator fingerprints with external RCTs."""

    if bootstrap_replicates < 500:
        raise ValueError("at least 500 bootstrap replicates are required")
    _validate_fingerprint_set(external_baseline, expected_source="external_rct")
    _validate_fingerprint_set(external_analysis, expected_source="external_rct")
    _validate_fingerprint_set(synthetic, expected_source="generated_trial")
    constructs = []
    for index, (construct, validation_domain) in enumerate(_CONSTRUCT_DOMAINS.items()):
        external_rows: tuple[_TrialBaselineFieldsV1, ...] = (
            external_analysis
            if validation_domain == "analysis_impact"
            else external_baseline
        )
        external_values = np.asarray(
            [float(getattr(row, construct)) for row in external_rows]
        )
        synthetic_values = np.asarray(
            [float(getattr(row, construct)) for row in synthetic]
        )
        scale = float(np.std(external_values, ddof=1))
        if scale <= 0:
            raise ValueError(f"external construct {construct} must vary across trials")
        distance = float(
            wasserstein_distance(external_values, synthetic_values) / scale
        )
        rng = np.random.default_rng(seed + index)
        distance_samples = np.empty(bootstrap_replicates)
        external_medians = np.empty(bootstrap_replicates)
        synthetic_medians = np.empty(bootstrap_replicates)
        for replicate in range(bootstrap_replicates):
            sampled_external = rng.choice(
                external_values, size=len(external_values), replace=True
            )
            sampled_synthetic = rng.choice(
                synthetic_values, size=len(synthetic_values), replace=True
            )
            sampled_scale = float(np.std(sampled_external, ddof=1))
            distance_samples[replicate] = (
                distance
                if sampled_scale <= 0
                else float(
                    wasserstein_distance(sampled_external, sampled_synthetic)
                    / sampled_scale
                )
            )
            external_medians[replicate] = float(np.median(sampled_external))
            synthetic_medians[replicate] = float(np.median(sampled_synthetic))
        distance_ci = np.quantile(distance_samples, [0.025, 0.975])
        external_ci = np.quantile(external_medians, [0.025, 0.975])
        synthetic_ci = np.quantile(synthetic_medians, [0.025, 0.975])
        reference = _balanced_external_split_distances(
            external_values, seed=seed + 100 + index
        )
        reference_p95 = float(np.quantile(reference, 0.95))
        constructs.append(
            GeneratorRealismConstructV1(
                construct_id=cast(
                    Literal[
                        "age_mean",
                        "age_sd",
                        "bmi_mean",
                        "bmi_sd",
                        "age_bmi_spearman",
                        "maximum_scaled_baseline_smd",
                        "adjustment_shift_in_unadjusted_se",
                        "adjusted_to_unadjusted_se_ratio",
                    ],
                    construct,
                ),
                validation_domain=validation_domain,
                external_trials=len(external_rows),
                synthetic_trials=len(synthetic),
                external_median=float(np.median(external_values)),
                external_median_ci_low=float(external_ci[0]),
                external_median_ci_high=float(external_ci[1]),
                synthetic_median=float(np.median(synthetic_values)),
                synthetic_median_ci_low=float(synthetic_ci[0]),
                synthetic_median_ci_high=float(synthetic_ci[1]),
                standardized_wasserstein=distance,
                standardized_wasserstein_ci_low=float(distance_ci[0]),
                standardized_wasserstein_ci_high=float(distance_ci[1]),
                external_split_reference_p95=reference_p95,
                within_external_split_reference=distance <= reference_p95,
            )
        )
    return GeneratorRealismSummaryV1(
        external_baseline_trials=len(external_baseline),
        external_analysis_trials=len(external_analysis),
        synthetic_trials=len(synthetic),
        external_baseline_participants=sum(
            row.participants for row in external_baseline
        ),
        external_analysis_participants=sum(
            row.participants for row in external_analysis
        ),
        synthetic_participants=sum(row.participants for row in synthetic),
        constructs=tuple(constructs),
    )


def read_trial_realism_fingerprints(
    path: Path,
) -> tuple[TrialRealismFingerprintV1, ...]:
    """Read public trial fingerprints from JSON Lines."""

    return tuple(
        TrialRealismFingerprintV1.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def read_trial_baseline_fingerprints(
    path: Path,
) -> tuple[TrialBaselineFingerprintV1, ...]:
    """Read public baseline fingerprints from JSON Lines."""

    return tuple(
        TrialBaselineFingerprintV1.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _standard_frame(
    frame: pd.DataFrame,
    *,
    outcome_columns: tuple[str, ...],
    require_two_arms: bool,
) -> pd.DataFrame:
    columns = ("treatment", "age", "bmi", *outcome_columns)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"trial analysis frame is missing columns: {missing}")
    values = frame.loc[:, list(columns)].dropna().reset_index(drop=True).copy()
    if len(values) < 20:
        raise ValueError(
            "trial analysis requires at least twenty complete participants"
        )
    treatment = values["treatment"].astype("string")
    if treatment.isna().any() or treatment.nunique() < 2:
        raise ValueError("trial analysis requires at least two treatment arms")
    if require_two_arms and treatment.nunique() != 2:
        raise ValueError(
            "treatment-effect analysis requires exactly two treatment arms"
        )
    values["treatment"] = treatment
    for column in ("age", "bmi", *outcome_columns):
        values[column] = pd.to_numeric(values[column], errors="raise").astype(float)
        if not np.isfinite(values[column].to_numpy()).all():
            raise ValueError(f"{column} must be finite")
    if values["age"].std(ddof=1) <= 0 or values["bmi"].std(ddof=1) <= 0:
        raise ValueError("age and BMI must vary")
    return values


def _ols_treatment_analysis(
    frame: pd.DataFrame,
    *,
    covariates: tuple[str, ...],
) -> tuple[float, float]:
    design = _analysis_design(frame, covariates=covariates)
    fitted = sm.OLS(frame["outcome"].to_numpy(), design.to_numpy()).fit(cov_type="HC3")
    return _finite_estimate(float(fitted.params[1]), float(fitted.bse[1]))


def _cox_treatment_analysis(
    frame: pd.DataFrame,
    *,
    covariates: tuple[str, ...],
) -> tuple[float, float]:
    design = _analysis_design(frame, covariates=covariates).drop(columns="const")
    fitted = PHReg(
        endog=frame["time"].to_numpy(dtype=float),
        exog=design.to_numpy(dtype=float),
        status=frame["event"].to_numpy(dtype=int),
        ties="breslow",
    ).fit()
    return _finite_estimate(
        float(fitted.params[0]),
        float(fitted.bse[0]),
    )


def _analysis_design(
    frame: pd.DataFrame, *, covariates: tuple[str, ...]
) -> pd.DataFrame:
    levels = tuple(sorted(frame["treatment"].astype(str).unique()))
    treatment = (
        frame["treatment"].astype(str).eq(levels[1]).astype(float).rename("treatment")
    )
    parts = [treatment]
    for covariate in covariates:
        values = frame[covariate].astype(float)
        parts.append(((values - values.mean()) / values.std(ddof=1)).rename(covariate))
    design = sm.add_constant(pd.concat(parts, axis=1), has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("treatment analysis design is rank deficient")
    return cast(pd.DataFrame, design)


def _finite_estimate(estimate: float, standard_error: float) -> tuple[float, float]:
    if (
        not math.isfinite(estimate)
        or not math.isfinite(standard_error)
        or standard_error <= 0
    ):
        raise ValueError("treatment analysis produced an invalid estimate")
    return estimate, standard_error


def _build_fingerprint(
    frame: pd.DataFrame,
    *,
    trial_id: str,
    source: Literal["external_rct", "generated_trial"],
    outcome_kind: Literal["binary", "continuous", "time_to_event"],
    unadjusted: tuple[float, float],
    adjusted: tuple[float, float],
) -> TrialRealismFingerprintV1:
    baseline = _build_baseline_fingerprint(
        frame,
        trial_id=trial_id,
        source=source,
    )
    return TrialRealismFingerprintV1(
        **baseline.model_dump(exclude={"schema_id"}),
        outcome_kind=outcome_kind,
        adjustment_shift_in_unadjusted_se=abs(adjusted[0] - unadjusted[0])
        / unadjusted[1],
        adjusted_to_unadjusted_se_ratio=adjusted[1] / unadjusted[1],
    )


def _build_baseline_fingerprint(
    frame: pd.DataFrame,
    *,
    trial_id: str,
    source: Literal["external_rct", "generated_trial"],
) -> TrialBaselineFingerprintV1:
    correlation = float(spearmanr(frame["age"], frame["bmi"]).statistic)
    if not math.isfinite(correlation):
        raise ValueError("age-BMI Spearman correlation is undefined")
    return TrialBaselineFingerprintV1(
        trial_id=trial_id,
        source=source,
        participants=len(frame),
        age_mean=float(frame["age"].mean()),
        age_sd=float(frame["age"].std(ddof=1)),
        bmi_mean=float(frame["bmi"].mean()),
        bmi_sd=float(frame["bmi"].std(ddof=1)),
        age_bmi_spearman=correlation,
        maximum_scaled_baseline_smd=max(
            _maximum_scaled_standardized_mean_difference(frame, covariate="age"),
            _maximum_scaled_standardized_mean_difference(frame, covariate="bmi"),
        ),
    )


def _maximum_scaled_standardized_mean_difference(
    frame: pd.DataFrame, *, covariate: str
) -> float:
    groups = [
        group[covariate].to_numpy()
        for _, group in frame.groupby("treatment", sort=True)
    ]
    scaled_differences = []
    for left, right in combinations(groups, 2):
        pooled_variance = (
            (len(left) - 1) * np.var(left, ddof=1)
            + (len(right) - 1) * np.var(right, ddof=1)
        ) / (len(left) + len(right) - 2)
        if pooled_variance <= 0:
            raise ValueError(f"{covariate} pooled variance must be positive")
        effective_n = len(left) * len(right) / (len(left) + len(right))
        standardized_difference = abs(np.mean(right) - np.mean(left)) / math.sqrt(
            pooled_variance
        )
        scaled_differences.append(
            float(standardized_difference * math.sqrt(effective_n))
        )
    if not scaled_differences:
        raise ValueError(
            "standardized mean difference requires at least two treatment arms"
        )
    return max(scaled_differences)


def _merge_generated_analysis_frame(
    baseline: pd.DataFrame,
    time_to_event: pd.DataFrame,
    *,
    endpoint_id: str,
) -> pd.DataFrame:
    baseline_columns = {"USUBJID", "TRTA", "AGE", "BMI"}
    time_columns = {"USUBJID", "PARAMCD", "AVAL", "CNSR"}
    if missing := sorted(baseline_columns - set(baseline.columns)):
        raise ValueError(f"generated baseline evidence lacks columns: {missing}")
    if missing := sorted(time_columns - set(time_to_event.columns)):
        raise ValueError(f"generated time-to-event evidence lacks columns: {missing}")
    endpoint = time_to_event.loc[
        time_to_event["PARAMCD"].astype(str) == endpoint_id
    ].copy()
    if endpoint.empty or endpoint["USUBJID"].astype("string").duplicated().any():
        raise ValueError(
            "generated primary endpoint must contain one row per participant"
        )
    merged = baseline.loc[:, ["USUBJID", "TRTA", "AGE", "BMI"]].merge(
        endpoint.loc[:, ["USUBJID", "AVAL", "CNSR"]],
        on="USUBJID",
        how="inner",
        validate="one_to_one",
    )
    return merged.rename(
        columns={
            "TRTA": "treatment",
            "AGE": "age",
            "BMI": "bmi",
            "AVAL": "time",
            "CNSR": "censor",
        }
    ).assign(event=lambda values: 1.0 - pd.to_numeric(values["censor"], errors="raise"))


def _validated_archive_members(archive: ZipFile) -> set[str]:
    names = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if "\\" in member.filename or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe participant archive member: {member.filename!r}")
        if member.filename in names:
            raise ValueError(
                f"duplicate participant archive member: {member.filename!r}"
            )
        names.add(member.filename)
    return names


def _json_object(archive: ZipFile, member: str) -> dict[str, object]:
    if member not in archive.namelist():
        raise ValueError(f"participant release lacks {member}")
    payload = json.loads(archive.read(member))
    if not isinstance(payload, dict):
        raise ValueError(f"{member} is not a JSON object")
    return payload


def _required_text(payload: dict[str, object], field: str, member: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{member} lacks {field}")
    return value


def _primary_endpoint_id(
    payload: dict[str, object],
    *,
    endpoint_term: str,
    member: str,
) -> str:
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list) or not all(
        isinstance(endpoint, dict) for endpoint in endpoints
    ):
        raise ValueError(f"{member} must define endpoint objects")
    matches = [
        endpoint for endpoint in endpoints if endpoint.get("term") == endpoint_term
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{member} must identify one endpoint matching {endpoint_term!r}"
        )
    endpoint_id = matches[0].get("endpoint_id")
    if not isinstance(endpoint_id, str) or not endpoint_id:
        raise ValueError(f"{member} lacks endpoint_id")
    return endpoint_id


def _read_parquet(archive: ZipFile, member: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(BytesIO(archive.read(member)))
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"{member} is not a readable Parquet table") from exc


def _validate_fingerprint_set(
    rows: tuple[_TrialBaselineFieldsV1, ...],
    *,
    expected_source: Literal["external_rct", "generated_trial"],
) -> None:
    if len(rows) < 3:
        raise ValueError(
            "generator realism comparison requires at least three independent trials"
        )
    if any(row.source != expected_source for row in rows):
        raise ValueError(f"fingerprint set must contain only {expected_source} trials")
    identities = [row.trial_id for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("fingerprint trial identities must be unique")


def _balanced_external_split_distances(
    values: npt.NDArray[np.float64],
    *,
    seed: int,
) -> npt.NDArray[np.float64]:
    split_size = len(values) // 2
    if split_size < 2:
        raise ValueError("external split reference requires at least four trials")
    indexes = tuple(range(len(values)))
    possible_splits = math.comb(len(values), split_size)
    splits: Iterable[tuple[int, ...]]
    if possible_splits <= 5_000:
        splits = combinations(indexes, split_size)
    else:
        rng = np.random.default_rng(seed)
        splits = (
            tuple(sorted(rng.choice(indexes, size=split_size, replace=False).tolist()))
            for _ in range(5_000)
        )
    distances = []
    for selected in splits:
        left_indexes = np.asarray(selected)
        right_indexes = np.asarray(
            [index for index in indexes if index not in selected]
        )
        left = values[left_indexes]
        right = values[right_indexes]
        scale = float(np.std(values, ddof=1))
        distances.append(float(wasserstein_distance(left, right) / scale))
    return np.asarray(distances)


__all__ = [
    "GeneratorRealismSummaryV1",
    "TrialBaselineFingerprintV1",
    "TrialRealismFingerprintV1",
    "compare_generator_realism",
    "extract_generated_trial_fingerprints",
    "fingerprint_standard_trial",
    "fingerprint_survival_trial",
    "fingerprint_trial_baseline",
    "read_trial_baseline_fingerprints",
    "read_trial_realism_fingerprints",
]
