"""Study-level fitting and held-out construct concordance."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from hashlib import sha256
from typing import cast

import numpy as np
from scipy.stats import wasserstein_distance

from trialagentbench_validation.external.contracts import (
    CalibrationParameterObservationV1,
    ConstructConcordanceV1,
    ConstructDefinitionV1,
    ConstructKind,
    ConstructMapV1,
    DistributionProfileV1,
    ExternalValidationReportV1,
    FittedConstructionParameterV1,
    SelectedObservableProfileV1,
    StudyPartitionV1,
    StudySummaryV1,
    SyntheticConstructConcordanceV1,
    SyntheticConstructRole,
)
from trialagentbench_validation.io import sha256_model

_FORBIDDEN_USES = (
    "latent_causal_truth",
    "tier_prevalence",
    "credit_eligible_analysis_routes",
    "scoring_tolerances",
    "decision_policy",
    "model_facing_difficulty",
)
_STUDY_FIELD_BY_KIND: dict[ConstructKind, str] = {
    ConstructKind.ENROLLMENT: "enrollment",
    ConstructKind.ARM_COUNT: "arm_count",
    ConstructKind.BASELINE_COVARIATE_COUNT: "baseline_covariate_count",
    ConstructKind.BASELINE_MISSING_FRACTION: "baseline_missing_fraction",
    ConstructKind.PRIMARY_OUTCOME_MISSING_FRACTION: "primary_outcome_missing_fraction",
    ConstructKind.EVENT_FRACTION: "event_fraction",
    ConstructKind.FOLLOW_UP_TIME: "follow_up_time_median",
    ConstructKind.AGE_MEAN: "age_mean",
    ConstructKind.AGE_SD: "age_sd",
    ConstructKind.BMI_MEAN: "bmi_mean",
    ConstructKind.BMI_SD: "bmi_sd",
}


def split_studies(
    studies: Iterable[StudySummaryV1],
    *,
    seed: int,
    held_out_fraction: float = 0.25,
) -> StudyPartitionV1:
    """Create a deterministic split whose independent unit is the study."""

    if not 0 < held_out_fraction < 1:
        raise ValueError("held_out_fraction must lie strictly between zero and one")
    by_source: dict[str, list[str]] = {}
    for study in studies:
        by_source.setdefault(study.source_id, []).append(study.study_id)
    ids = [study_id for source_ids in by_source.values() for study_id in source_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("study identities must be globally unique")
    if len(ids) < 4:
        raise ValueError("at least four independent studies are required")
    calibration_ids: list[str] = []
    held_out_ids: list[str] = []
    for source_index, source_id in enumerate(sorted(by_source)):
        shuffled = sorted(by_source[source_id])
        if len(shuffled) < 4:
            raise ValueError(
                f"source {source_id!r} has fewer than four independent studies"
            )
        random.Random(seed + source_index).shuffle(shuffled)
        held_out_count = max(2, int(round(len(shuffled) * held_out_fraction)))
        if held_out_count >= len(shuffled):
            raise ValueError(
                f"held-out split leaves no calibration studies for source {source_id!r}"
            )
        held_out_ids.extend(shuffled[:held_out_count])
        calibration_ids.extend(shuffled[held_out_count:])
    held_out = tuple(sorted(held_out_ids))
    calibration = tuple(sorted(calibration_ids))
    return StudyPartitionV1(
        seed=seed,
        calibration_study_ids=calibration,
        held_out_study_ids=held_out,
    )


def fit_observable_profile(
    studies: Iterable[StudySummaryV1],
    *,
    partition: StudyPartitionV1,
    construct_map: ConstructMapV1,
    profile_id: str,
    source_manifest_sha256: str,
) -> SelectedObservableProfileV1:
    """Fit empirical nuisance distributions using calibration studies only."""

    by_id = _index_studies(studies)
    calibration = tuple(by_id[study_id] for study_id in partition.calibration_study_ids)
    distributions: list[DistributionProfileV1] = []
    for construct in construct_map.constructs:
        values = _construct_values(calibration, construct)
        if len(values) < construct.minimum_studies:
            continue
        transformed = _transform(values, construct.transformation)
        quantiles = np.quantile(transformed, [0.0, 0.25, 0.5, 0.75, 1.0])
        distributions.append(
            DistributionProfileV1(
                construct_id=construct.construct_id,
                unit=construct.unit,
                transformation=construct.transformation,
                n_studies=len(values),
                minimum=float(quantiles[0]),
                q25=float(quantiles[1]),
                median=float(quantiles[2]),
                q75=float(quantiles[3]),
                maximum=float(quantiles[4]),
            )
        )
    if not distributions:
        raise ValueError("no construct has sufficient calibration support")
    distribution_by_id = {row.construct_id: row for row in distributions}
    construction_parameters: list[FittedConstructionParameterV1] = []
    for construct in construct_map.constructs:
        if construct.construction_parameter is None:
            continue
        fitted = distribution_by_id.get(construct.construct_id)
        if fitted is None:
            raise ValueError(
                f"construction parameter lacks sufficient calibration support: {construct.construct_id}"
            )
        value = (
            math.expm1(fitted.median)
            if fitted.transformation == "log1p"
            else fitted.median
        )
        construction_parameters.append(
            FittedConstructionParameterV1(
                parameter_path=construct.construction_parameter,
                source_construct_id=construct.construct_id,
                value=float(value),
                unit=construct.unit,
                calibration_study_values=tuple(
                    CalibrationParameterObservationV1(
                        study_id=study.study_id,
                        value=float(observed),
                    )
                    for study in calibration
                    if study.source_id == construct.source_id
                    and (observed := _study_value(study, construct.kind)) is not None
                ),
            )
        )
    return SelectedObservableProfileV1(
        profile_id=profile_id,
        source_manifest_sha256=source_manifest_sha256,
        construct_map_sha256=sha256_model(construct_map),
        partition_sha256=sha256_model(partition),
        distributions=tuple(distributions),
        construction_parameters=tuple(construction_parameters),
        forbidden_uses=_FORBIDDEN_USES,
    )


def evaluate_held_out(
    studies: Iterable[StudySummaryV1],
    *,
    partition: StudyPartitionV1,
    construct_map: ConstructMapV1,
    profile: SelectedObservableProfileV1,
    bootstrap_replicates: int = 2_000,
    seed: int = 73,
) -> ExternalValidationReportV1:
    """Evaluate calibration-study empirical distributions on held-out studies."""

    if bootstrap_replicates < 200:
        raise ValueError("at least 200 bootstrap replicates are required")
    by_id = _index_studies(studies)
    calibration = tuple(by_id[study_id] for study_id in partition.calibration_study_ids)
    held_out = tuple(by_id[study_id] for study_id in partition.held_out_study_ids)
    fitted = {row.construct_id: row for row in profile.distributions}
    results: list[ConstructConcordanceV1] = []
    for construct in construct_map.constructs:
        calibration_values = _construct_values(calibration, construct)
        held_out_values = _construct_values(held_out, construct)
        if (
            construct.construct_id not in fitted
            or len(calibration_values) < construct.minimum_studies
            or len(held_out_values) < construct.minimum_studies
        ):
            results.append(
                ConstructConcordanceV1(
                    construct_id=construct.construct_id,
                    n_calibration_studies=len(calibration_values),
                    n_held_out_studies=len(held_out_values),
                    status="unsupported",
                    interpretation="Insufficient independent-study support for held-out inference.",
                )
            )
            continue
        x = _transform(calibration_values, construct.transformation)
        y = _transform(held_out_values, construct.transformation)
        distance = float(wasserstein_distance(x, y))
        low, high = _cluster_bootstrap_distance(
            x,
            y,
            replicates=bootstrap_replicates,
            seed=_construct_seed(seed, "held-out-bootstrap", construct.construct_id),
        )
        results.append(
            ConstructConcordanceV1(
                construct_id=construct.construct_id,
                n_calibration_studies=len(x),
                n_held_out_studies=len(y),
                status="supported",
                wasserstein_distance=distance,
                bootstrap_ci_low=low,
                bootstrap_ci_high=high,
                equivalence_margin=construct.equivalence_margin,
                equivalent=(
                    None
                    if construct.equivalence_margin is None
                    else high <= construct.equivalence_margin
                ),
                interpretation=(
                    "Held-out study-level distribution comparison; equivalence is not claimed without "
                    "a prospective native-unit margin."
                ),
            )
        )
    return ExternalValidationReportV1(
        profile_id=profile.profile_id,
        results=tuple(results),
        limitations=(
            "Observable construct alignment does not establish latent causal-truth validity.",
            "RCT Bench participant-level evidence covers individually randomized trials and does not validate D4 dependence.",
            "AACT aggregate records do not establish participant-level dependence or population representativeness.",
            "No global realism score is defined.",
        ),
    )


def evaluate_synthetic_concordance(
    synthetic_trials: Iterable[StudySummaryV1],
    external_studies: Iterable[StudySummaryV1],
    *,
    partition: StudyPartitionV1,
    construct_map: ConstructMapV1,
    bootstrap_replicates: int = 2_000,
    seed: int = 79,
) -> tuple[SyntheticConstructConcordanceV1, ...]:
    """Compare independent public synthetic trials with held-out external studies."""

    if bootstrap_replicates < 200:
        raise ValueError("at least 200 bootstrap replicates are required")
    synthetic = tuple(synthetic_trials)
    if len({study.study_id for study in synthetic}) != len(synthetic):
        raise ValueError("synthetic trial identities must be unique")
    external_by_id = _index_studies(external_studies)
    calibration = tuple(
        external_by_id[study_id] for study_id in partition.calibration_study_ids
    )
    held_out = tuple(
        external_by_id[study_id] for study_id in partition.held_out_study_ids
    )
    results: list[SyntheticConstructConcordanceV1] = []
    for construct in construct_map.constructs:
        synthetic_values = _construct_values_any_source(synthetic, construct)
        held_out_values = _construct_values(held_out, construct)
        role = _synthetic_construct_role(construct)
        if (
            len(synthetic_values) < construct.minimum_synthetic_trials
            or len(held_out_values) < construct.minimum_studies
        ):
            results.append(
                SyntheticConstructConcordanceV1(
                    construct_id=construct.construct_id,
                    role=role,
                    n_synthetic_trials=len(synthetic_values),
                    n_held_out_studies=len(held_out_values),
                    status="unsupported",
                    interpretation=(
                        "Insufficient independent trial support for a synthetic-to-held-out comparison."
                    ),
                )
            )
            continue
        x = _transform(synthetic_values, construct.transformation)
        y = _transform(held_out_values, construct.transformation)
        distance = float(wasserstein_distance(x, y))
        low, high = _cluster_bootstrap_distance(
            x,
            y,
            replicates=bootstrap_replicates,
            seed=_construct_seed(seed, "synthetic-bootstrap", construct.construct_id),
        )
        calibration_reference = None
        calibration_tail_probability = None
        within_reference = None
        if role == SyntheticConstructRole.EXTERNALLY_FITTED:
            calibration_values = _transform(
                _construct_values(calibration, construct),
                construct.transformation,
            )
            calibration_reference, calibration_tail_probability = (
                _calibration_reference_distance_distribution(
                    calibration_values,
                    observed_distance=distance,
                    synthetic_size=len(x),
                    held_out_size=len(y),
                    replicates=bootstrap_replicates,
                    seed=_construct_seed(
                        seed, "calibration-reference", construct.construct_id
                    ),
                )
            )
            within_reference = distance <= calibration_reference
        results.append(
            SyntheticConstructConcordanceV1(
                construct_id=construct.construct_id,
                role=role,
                n_synthetic_trials=len(x),
                n_held_out_studies=len(y),
                status="supported",
                wasserstein_distance=distance,
                bootstrap_ci_low=low,
                bootstrap_ci_high=high,
                equivalence_margin=construct.equivalence_margin,
                equivalent=(
                    None
                    if construct.equivalence_margin is None
                    else high <= construct.equivalence_margin
                ),
                calibration_reference_p95=calibration_reference,
                calibration_reference_tail_probability=calibration_tail_probability,
                within_calibration_reference=within_reference,
                interpretation=(
                    "Public synthetic trial-level distribution compared with held-out external studies. "
                    "For externally fitted constructs, the calibration reference is fixed from "
                    "calibration-study resampling before synthetic results are examined; it is not "
                    "a claim of clinical equivalence."
                ),
            )
        )
    return tuple(results)


def _index_studies(studies: Iterable[StudySummaryV1]) -> dict[str, StudySummaryV1]:
    indexed: dict[str, StudySummaryV1] = {}
    for study in studies:
        if study.study_id in indexed:
            raise ValueError(f"duplicate study identity: {study.study_id}")
        indexed[study.study_id] = study
    return indexed


def _construct_values(
    studies: Iterable[StudySummaryV1],
    construct: ConstructDefinitionV1,
) -> np.ndarray:
    values: list[float] = []
    for study in studies:
        if study.source_id != construct.source_id:
            continue
        value = _study_value(study, construct.kind)
        if value is not None:
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _construct_values_any_source(
    studies: Iterable[StudySummaryV1],
    construct: ConstructDefinitionV1,
) -> np.ndarray:
    values = [
        float(value)
        for study in studies
        if not any(
            exclusion.startswith(f"{construct.kind.value}:")
            for exclusion in study.observable_exclusions
        )
        and (value := _study_value(study, construct.kind)) is not None
    ]
    return np.asarray(values, dtype=float)


def _study_value(study: StudySummaryV1, kind: ConstructKind) -> int | float | None:
    return cast(int | float | None, getattr(study, _STUDY_FIELD_BY_KIND[kind]))


def _synthetic_construct_role(
    construct: ConstructDefinitionV1,
) -> SyntheticConstructRole:
    if construct.construction_parameter is not None:
        return SyntheticConstructRole.EXTERNALLY_FITTED
    if construct.kind == ConstructKind.ENROLLMENT:
        return SyntheticConstructRole.QUALIFICATION_CONTROL
    if construct.kind == ConstructKind.ARM_COUNT:
        return SyntheticConstructRole.PROTOCOL_CONTROL
    return SyntheticConstructRole.DESCRIPTIVE_ONLY


def _transform(values: np.ndarray, transformation: str) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("construct values must be finite")
    if transformation == "identity":
        return values
    if bool((values < 0).any()):
        raise ValueError("log1p transformation requires nonnegative values")
    return np.asarray(np.log1p(values), dtype=float)


def _cluster_bootstrap_distance(
    calibration: np.ndarray,
    held_out: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    distances = np.empty(replicates, dtype=float)
    for index in range(replicates):
        x = rng.choice(calibration, size=len(calibration), replace=True)
        y = rng.choice(held_out, size=len(held_out), replace=True)
        distances[index] = wasserstein_distance(x, y)
    low, high = np.quantile(distances, [0.025, 0.975])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise ValueError("bootstrap produced non-finite uncertainty")
    return float(low), float(high)


def _construct_seed(seed: int, analysis_id: str, construct_id: str) -> int:
    """Derive a stable random seed from analysis and construct identities."""

    payload = f"{seed}:{analysis_id}:{construct_id}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _calibration_reference_distance_distribution(
    calibration: np.ndarray,
    *,
    observed_distance: float,
    synthetic_size: int,
    held_out_size: int,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if len(calibration) < 2:
        raise ValueError(
            "calibration reference requires at least two independent studies"
        )
    rng = np.random.default_rng(seed)
    distances = np.empty(replicates, dtype=float)
    for index in range(replicates):
        synthetic_reference = rng.choice(
            calibration,
            size=synthetic_size,
            replace=True,
        )
        held_out_reference = rng.choice(
            calibration,
            size=held_out_size,
            replace=True,
        )
        distances[index] = wasserstein_distance(
            synthetic_reference,
            held_out_reference,
        )
    reference = float(np.quantile(distances, 0.95))
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("calibration reference distance must be finite and positive")
    tail_probability = float(
        (1 + np.count_nonzero(distances >= observed_distance)) / (replicates + 1)
    )
    return reference, tail_probability


__all__ = [
    "evaluate_held_out",
    "evaluate_synthetic_concordance",
    "fit_observable_profile",
    "split_studies",
]
