"""Independent verification of participant-clustered ordinal trial worlds."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import kendalltau
from statsmodels.genmod.cov_struct import Exchangeable

from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.statistics import proportion_interval


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClusteredOrdinalArmReferenceV1(_FrozenModel):
    """Source properties for one randomized arm."""

    arm_id: str = Field(min_length=1)
    participants: int = Field(ge=20)
    cluster_size_probabilities: tuple[float, ...] = Field(min_length=2)
    observation_probability: float = Field(gt=0, lt=1)
    category_probabilities: tuple[float, ...] = Field(min_length=3)
    source_kendall_tau: float = Field(ge=-1, le=1)
    latent_correlation: float = Field(ge=0, lt=1)


class ClusteredOrdinalDoseDistributionV1(_FrozenModel):
    """Ordinal probabilities for one arm and effect dose."""

    dose_multiplier: float = Field(ge=0)
    arm_id: str = Field(min_length=1)
    category_probabilities: tuple[float, ...] = Field(min_length=3)


class ClusteredOrdinalQualificationDesignV1(_FrozenModel):
    """Path-free design for a clustered ordinal qualification campaign."""

    schema_id: Literal["trialagentbench.clustered_ordinal_design/v1"] = (
        "trialagentbench.clustered_ordinal_design/v1"
    )
    trial_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: int = Field(ge=2)
    seed: int = Field(ge=0, le=2**32 - 1)
    categories: tuple[int, ...] = Field(min_length=3)
    control_arm_id: str = Field(min_length=1)
    treatment_arm_id: str = Field(min_length=1)
    source_log_occlusion_odds_ratio: float = Field(allow_inf_nan=False)
    dose_multipliers: tuple[float, ...] = Field(min_length=3)
    arms: tuple[ClusteredOrdinalArmReferenceV1, ClusteredOrdinalArmReferenceV1]
    fitted_distributions: tuple[ClusteredOrdinalDoseDistributionV1, ...] = Field(
        min_length=6
    )

    @model_validator(mode="after")
    def _complete(self) -> ClusteredOrdinalQualificationDesignV1:
        if tuple(range(1, len(self.categories) + 1)) != self.categories:
            raise ValueError(
                "Clustered ordinal categories must be consecutive integers from one"
            )
        if self.control_arm_id == self.treatment_arm_id:
            raise ValueError("Clustered ordinal arms must be distinct")
        arm_ids = {self.control_arm_id, self.treatment_arm_id}
        if {arm.arm_id for arm in self.arms} != arm_ids:
            raise ValueError("Clustered ordinal references must cover both arms")
        if tuple(sorted(self.dose_multipliers)) != self.dose_multipliers:
            raise ValueError("Clustered ordinal doses must be sorted")
        if 0.0 not in self.dose_multipliers or 1.0 not in self.dose_multipliers:
            raise ValueError("Clustered ordinal design requires null and source doses")
        expected = {(dose, arm) for dose in self.dose_multipliers for arm in arm_ids}
        if {
            (row.dose_multiplier, row.arm_id) for row in self.fitted_distributions
        } != expected:
            raise ValueError(
                "Clustered ordinal distributions do not cover the dose-by-arm grid"
            )
        for arm in self.arms:
            if not np.isclose(sum(arm.cluster_size_probabilities), 1.0):
                raise ValueError("Cluster-size probabilities must sum to one")
        probability_rows = [arm.category_probabilities for arm in self.arms] + [
            row.category_probabilities for row in self.fitted_distributions
        ]
        for probabilities in probability_rows:
            if len(probabilities) != len(self.categories):
                raise ValueError("Ordinal probabilities must match category support")
            values = np.asarray(probabilities, dtype=float)
            if np.any(values <= 0) or not np.isclose(values.sum(), 1.0):
                raise ValueError(
                    "Ordinal probabilities must be positive and sum to one"
                )
        return self


class ClusteredOrdinalWorldReceiptV1(_FrozenModel):
    """Checksum-bound identity for one generated world."""

    world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    world_index: int = Field(ge=0)
    analysis_path: str = Field(pattern=r"^worlds/world_[0-9a-f]{20}\.parquet$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ClusteredOrdinalQualificationReceiptV1(_FrozenModel):
    """Complete generated-world inventory."""

    schema_id: Literal["trialagentbench.clustered_ordinal_receipt/v1"] = (
        "trialagentbench.clustered_ordinal_receipt/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[ClusteredOrdinalWorldReceiptV1, ...] = Field(min_length=2)


class ClusteredOrdinalRecoveryV1(_FrozenModel):
    """Cluster-robust treatment recovery at one effect dose."""

    dose_multiplier: float = Field(ge=0)
    truth_log_odds_ratio: float = Field(allow_inf_nan=False)
    worlds: int = Field(ge=2)
    mean_log_odds_ratio: float = Field(allow_inf_nan=False)
    bias: float = Field(allow_inf_nan=False)
    bias_ci_low: float = Field(allow_inf_nan=False)
    bias_ci_high: float = Field(allow_inf_nan=False)
    predictive_95_low: float = Field(allow_inf_nan=False)
    predictive_95_high: float = Field(allow_inf_nan=False)
    coverage: float = Field(ge=0, le=1)
    coverage_ci_low: float = Field(ge=0, le=1)
    coverage_ci_high: float = Field(ge=0, le=1)


class ClusteredOrdinalArmResultV1(_FrozenModel):
    """Repeated-world fidelity summaries for one arm at source dose."""

    arm_id: str = Field(min_length=1)
    source_cluster_size_probabilities: tuple[float, ...] = Field(min_length=2)
    mean_cluster_size_probabilities: tuple[float, ...] = Field(min_length=2)
    cluster_size_mae: float = Field(ge=0, le=1)
    source_observation_probability: float = Field(ge=0, le=1)
    mean_observation_probability: float = Field(ge=0, le=1)
    observation_probability_bias: float = Field(allow_inf_nan=False)
    source_category_probabilities: tuple[float, ...] = Field(min_length=3)
    mean_category_probabilities: tuple[float, ...] = Field(min_length=3)
    category_probability_mae: float = Field(ge=0, le=1)
    source_kendall_tau: float = Field(ge=-1, le=1)
    mean_kendall_tau: float = Field(ge=-1, le=1)
    kendall_tau_ci_low: float = Field(ge=-1, le=1)
    kendall_tau_ci_high: float = Field(ge=-1, le=1)


class ClusteredOrdinalQualificationReportV1(_FrozenModel):
    """Independent report for clustered ordinal generation and analysis."""

    schema_id: Literal["trialagentbench.clustered_ordinal_report/v1"] = (
        "trialagentbench.clustered_ordinal_report/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: int = Field(ge=2)
    arm_results: tuple[ClusteredOrdinalArmResultV1, ClusteredOrdinalArmResultV1]
    recovery: tuple[ClusteredOrdinalRecoveryV1, ...] = Field(min_length=3)
    dose_response_slope: float = Field(allow_inf_nan=False)
    dose_response_slope_ci_low: float = Field(allow_inf_nan=False)
    dose_response_slope_ci_high: float = Field(allow_inf_nan=False)
    monotone_world_fraction: float = Field(ge=0, le=1)
    monotone_world_fraction_ci_low: float = Field(ge=0, le=1)
    monotone_world_fraction_ci_high: float = Field(ge=0, le=1)
    intact_source_dose_kendall_tau: float = Field(ge=-1, le=1)
    broken_cluster_linkage_kendall_tau: float = Field(ge=-1, le=1)


def evaluate_clustered_ordinal_qualification(
    *,
    release_dir: Path,
    minimum_worlds: int = 100,
) -> ClusteredOrdinalQualificationReportV1:
    """Verify released bytes and recompute clustered ordinal evidence."""

    design_path = release_dir / "design.json"
    receipt_path = release_dir / "receipt.json"
    design = ClusteredOrdinalQualificationDesignV1.model_validate_json(
        design_path.read_text(encoding="utf-8")
    )
    receipt = ClusteredOrdinalQualificationReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    design_sha = _json_sha(json.loads(design_path.read_text(encoding="utf-8")))
    if receipt.design_sha256 != design_sha:
        raise ValueError("Clustered ordinal receipt does not match its design")
    if len(receipt.worlds) != design.worlds or design.worlds < minimum_worlds:
        raise ValueError(
            "Clustered ordinal release has insufficient or inconsistent worlds"
        )
    expected = {item.analysis_path for item in receipt.worlds}
    actual = {
        path.relative_to(release_dir).as_posix()
        for path in (release_dir / "worlds").glob("*.parquet")
    }
    if actual != expected:
        raise ValueError("Clustered ordinal world inventory does not match the receipt")
    frames = []
    for item in receipt.worlds:
        path = release_dir / item.analysis_path
        if sha256_file(path) != item.analysis_sha256:
            raise ValueError(f"Clustered ordinal checksum mismatch: {item.world_id}")
        frame = pd.read_parquet(path)
        _validate_world(frame, design=design, world_id=item.world_id)
        frames.append(frame)
    values = pd.concat(frames, ignore_index=True)
    recovery, estimates = _recovery(values, design=design)
    arm_results = _arm_results(values, design=design)
    slopes = np.asarray(
        [
            np.polyfit(group["dose_multiplier"], group["estimate"], deg=1)[0]
            for _, group in estimates.groupby("world_id", sort=True)
        ],
        dtype=float,
    )
    slope, slope_low, slope_high = _mean_ci(slopes)
    monotone = np.sign(slopes) == np.sign(design.source_log_occlusion_odds_ratio)
    monotone_interval = proportion_interval(int(np.sum(monotone)), len(monotone))
    source_values = values.loc[
        values["dose_multiplier"].eq(1.0) & values["observed"].astype(bool)
    ]
    intact_tau = _pooled_pair_tau(source_values)
    broken_tau = _broken_cluster_tau(source_values)
    return ClusteredOrdinalQualificationReportV1(
        design_sha256=design_sha,
        receipt_sha256=sha256_file(receipt_path),
        worlds=design.worlds,
        arm_results=tuple(arm_results),
        recovery=tuple(recovery),
        dose_response_slope=slope,
        dose_response_slope_ci_low=slope_low,
        dose_response_slope_ci_high=slope_high,
        monotone_world_fraction=float(np.mean(monotone)),
        monotone_world_fraction_ci_low=monotone_interval[0],
        monotone_world_fraction_ci_high=monotone_interval[1],
        intact_source_dose_kendall_tau=intact_tau,
        broken_cluster_linkage_kendall_tau=broken_tau,
    )


def _validate_world(
    frame: pd.DataFrame,
    *,
    design: ClusteredOrdinalQualificationDesignV1,
    world_id: str,
) -> None:
    required = {
        "world_id",
        "dose_multiplier",
        "participant_id",
        "arm",
        "graft_index",
        "observed",
        "three_year_ct_result",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Clustered ordinal world is missing columns: {missing}")
    if set(frame["world_id"].astype(str)) != {world_id}:
        raise ValueError("Clustered ordinal world identity mismatch")
    if set(frame["dose_multiplier"].astype(float)) != set(design.dose_multipliers):
        raise ValueError("Clustered ordinal world has an incomplete dose surface")
    references = {arm.arm_id: arm for arm in design.arms}
    for (dose, arm), group in frame.groupby(["dose_multiplier", "arm"], observed=True):
        if str(arm) not in references:
            raise ValueError("Clustered ordinal world has an unknown arm")
        counts = group.groupby("participant_id", sort=False).size()
        if len(counts) != references[str(arm)].participants:
            raise ValueError(
                f"Clustered ordinal cell has wrong participant count: {dose}, {arm}"
            )
        if not counts.between(
            1, len(references[str(arm)].cluster_size_probabilities)
        ).all():
            raise ValueError("Clustered ordinal world has an invalid cluster size")
        if group.duplicated(["participant_id", "graft_index"]).any():
            raise ValueError("Clustered ordinal participant-graft keys must be unique")
        expected_indices = group.groupby("participant_id", sort=False)[
            "graft_index"
        ].apply(lambda values: tuple(sorted(values.astype(int))))
        if not all(
            indices == tuple(range(1, len(indices) + 1)) for indices in expected_indices
        ):
            raise ValueError("Clustered ordinal graft indices must be consecutive")
    observed = frame["observed"].astype(bool)
    if (
        frame.groupby(["world_id", "dose_multiplier", "participant_id"], observed=True)[
            "observed"
        ]
        .nunique()
        .max()
        != 1
    ):
        raise ValueError(
            "Clustered ordinal observation status must be participant-level"
        )
    outcome = frame.loc[observed, "three_year_ct_result"].astype(int)
    if (
        not outcome.isin(design.categories).all()
        or frame.loc[~observed, "three_year_ct_result"].notna().any()
    ):
        raise ValueError(
            "Clustered ordinal values violate observation or category support"
        )


def _recovery(
    values: pd.DataFrame,
    *,
    design: ClusteredOrdinalQualificationDesignV1,
) -> tuple[list[ClusteredOrdinalRecoveryV1], pd.DataFrame]:
    rows = []
    for (world_id, dose), group in values.loc[values["observed"].astype(bool)].groupby(
        ["world_id", "dose_multiplier"], observed=True, sort=True
    ):
        analysis = group.assign(
            occluded=group["three_year_ct_result"]
            .astype(int)
            .eq(max(design.categories))
            .astype(int),
            treatment=group["arm"].astype(str).eq(design.treatment_arm_id).astype(int),
        )
        fitted = sm.GEE.from_formula(
            "occluded ~ treatment",
            groups="participant_id",
            cov_struct=Exchangeable(),
            family=sm.families.Binomial(),
            data=analysis,
        ).fit()
        estimate = float(fitted.params["treatment"])
        standard_error = float(fitted.bse["treatment"])
        if (
            not np.isfinite(estimate)
            or not np.isfinite(standard_error)
            or standard_error <= 0
        ):
            raise ValueError(f"Clustered GEE failed for {world_id}, dose {dose}")
        rows.append(
            {
                "world_id": str(world_id),
                "dose_multiplier": float(cast(float, dose)),
                "estimate": estimate,
                "standard_error": standard_error,
            }
        )
    estimates = pd.DataFrame(rows)
    output = []
    for dose, group in estimates.groupby("dose_multiplier", sort=True):
        truth = float(cast(float, dose)) * design.source_log_occlusion_odds_ratio
        sample = group["estimate"].to_numpy(dtype=float)
        bias_sample = sample - truth
        bias, bias_low, bias_high = _mean_ci(bias_sample)
        predictive = np.quantile(sample, [0.025, 0.975])
        covered = (
            sample - 1.96 * group["standard_error"].to_numpy(dtype=float) <= truth
        ) & (sample + 1.96 * group["standard_error"].to_numpy(dtype=float) >= truth)
        covered_count = int(covered.sum())
        coverage_interval = proportion_interval(covered_count, len(covered))
        output.append(
            ClusteredOrdinalRecoveryV1(
                dose_multiplier=float(cast(float, dose)),
                truth_log_odds_ratio=truth,
                worlds=len(sample),
                mean_log_odds_ratio=float(np.mean(sample)),
                bias=bias,
                bias_ci_low=bias_low,
                bias_ci_high=bias_high,
                predictive_95_low=float(predictive[0]),
                predictive_95_high=float(predictive[1]),
                coverage=covered_count / len(covered),
                coverage_ci_low=coverage_interval[0],
                coverage_ci_high=coverage_interval[1],
            )
        )
    return output, estimates


def _arm_results(
    values: pd.DataFrame,
    *,
    design: ClusteredOrdinalQualificationDesignV1,
) -> list[ClusteredOrdinalArmResultV1]:
    source_dose = values.loc[values["dose_multiplier"].eq(1.0)]
    output = []
    for reference in sorted(design.arms, key=lambda item: item.arm_id):
        arm = source_dose.loc[source_dose["arm"].astype(str).eq(reference.arm_id)]
        size_rows = []
        observation_rows = []
        category_rows = []
        tau_rows = []
        for _, world in arm.groupby("world_id", sort=True):
            counts = world.groupby("participant_id", sort=False).size()
            size_rows.append(
                counts.value_counts(normalize=True)
                .reindex(
                    range(1, len(reference.cluster_size_probabilities) + 1),
                    fill_value=0.0,
                )
                .to_numpy(dtype=float)
            )
            participant_observed = (
                world.groupby("participant_id", sort=False)["observed"]
                .first()
                .astype(bool)
            )
            observation_rows.append(float(participant_observed.mean()))
            observed = world.loc[
                world["observed"].astype(bool), "three_year_ct_result"
            ].astype(int)
            category_rows.append(
                observed.value_counts(normalize=True)
                .reindex(design.categories, fill_value=0.0)
                .to_numpy(dtype=float)
            )
            tau_rows.append(_pooled_pair_tau(world.loc[world["observed"].astype(bool)]))
        size_mean = np.mean(np.vstack(size_rows), axis=0)
        category_mean = np.mean(np.vstack(category_rows), axis=0)
        tau, tau_low, tau_high = _mean_ci(np.asarray(tau_rows, dtype=float))
        output.append(
            ClusteredOrdinalArmResultV1(
                arm_id=reference.arm_id,
                source_cluster_size_probabilities=reference.cluster_size_probabilities,
                mean_cluster_size_probabilities=tuple(
                    float(value) for value in size_mean
                ),
                cluster_size_mae=float(
                    np.mean(
                        np.abs(
                            size_mean - np.asarray(reference.cluster_size_probabilities)
                        )
                    )
                ),
                source_observation_probability=reference.observation_probability,
                mean_observation_probability=float(np.mean(observation_rows)),
                observation_probability_bias=float(
                    np.mean(observation_rows) - reference.observation_probability
                ),
                source_category_probabilities=reference.category_probabilities,
                mean_category_probabilities=tuple(
                    float(value) for value in category_mean
                ),
                category_probability_mae=float(
                    np.mean(
                        np.abs(
                            category_mean - np.asarray(reference.category_probabilities)
                        )
                    )
                ),
                source_kendall_tau=reference.source_kendall_tau,
                mean_kendall_tau=tau,
                kendall_tau_ci_low=max(-1.0, tau_low),
                kendall_tau_ci_high=min(1.0, tau_high),
            )
        )
    return output


def _pooled_pair_tau(frame: pd.DataFrame) -> float:
    pairs: list[tuple[int, int]] = []
    for _, participant in frame.groupby("participant_id", sort=False):
        values = (
            participant.sort_values("graft_index")["three_year_ct_result"]
            .astype(int)
            .to_numpy()
        )
        pairs.extend(
            (int(values[left]), int(values[right]))
            for left in range(len(values))
            for right in range(left + 1, len(values))
        )
    if len(pairs) < 20:
        raise ValueError(
            "Clustered ordinal dependence requires at least 20 within-participant pairs"
        )
    array = np.asarray(pairs, dtype=float)
    statistic = float(kendalltau(array[:, 0], array[:, 1]).statistic)
    if not np.isfinite(statistic):
        raise ValueError("Clustered ordinal Kendall dependence is undefined")
    return statistic


def _broken_cluster_tau(frame: pd.DataFrame) -> float:
    broken = frame.copy()
    rng = np.random.default_rng(451_991)
    broken["three_year_ct_result"] = rng.permutation(
        broken["three_year_ct_result"].to_numpy()
    )
    return _pooled_pair_tau(broken)


def _mean_ci(sample: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    if len(sample) < 2 or not np.isfinite(sample).all():
        raise ValueError("Monte Carlo interval requires at least two finite estimates")
    mean = float(np.mean(sample))
    half_width = 1.96 * float(np.std(sample, ddof=1)) / math.sqrt(len(sample))
    return mean, mean - half_width, mean + half_width


def _json_sha(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ClusteredOrdinalArmReferenceV1",
    "ClusteredOrdinalDoseDistributionV1",
    "ClusteredOrdinalQualificationDesignV1",
    "ClusteredOrdinalQualificationReceiptV1",
    "ClusteredOrdinalQualificationReportV1",
    "ClusteredOrdinalWorldReceiptV1",
    "evaluate_clustered_ordinal_qualification",
]
