"""Independent qualification of multivariate longitudinal trial worlds."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import norm

from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.statistics import proportion_interval

MultivariateLongitudinalMode = Literal[
    "independent_outcomes",
    "linkage_25",
    "linkage_50",
    "linkage_75",
    "source_anchored",
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MultivariateCellReferenceV1(_FrozenModel):
    """Source reference for one randomized-arm outcome cell."""

    arm_id: str = Field(min_length=1)
    outcome_id: str = Field(min_length=1)
    time: float = Field(allow_inf_nan=False)
    observations: int = Field(ge=3)
    mean: float = Field(allow_inf_nan=False)
    standard_deviation: float = Field(gt=0, allow_inf_nan=False)
    quantile_10: float = Field(allow_inf_nan=False)
    quantile_50: float = Field(allow_inf_nan=False)
    quantile_90: float = Field(allow_inf_nan=False)


class MultivariatePairReferenceV1(_FrozenModel):
    """Source rank correlation for two outcome-by-visit cells."""

    outcome_a: str = Field(min_length=1)
    time_a: float = Field(allow_inf_nan=False)
    outcome_b: str = Field(min_length=1)
    time_b: float = Field(allow_inf_nan=False)
    complete_pairs: int = Field(ge=20)
    spearman_correlation: float = Field(ge=-1, le=1, allow_inf_nan=False)


class MultivariateTreatmentTruthV1(_FrozenModel):
    """Known randomized treatment contrast for one final outcome."""

    arm_id: str = Field(min_length=1)
    outcome_id: str = Field(min_length=1)
    final_time: float = Field(allow_inf_nan=False)
    contrast: float = Field(allow_inf_nan=False)


class MultivariateLongitudinalFittedModelV1(_FrozenModel):
    """Complete source-fitted joint longitudinal process."""

    control_mean_values: dict[str, tuple[float, ...]]
    latent_correlation: tuple[tuple[float, ...], ...]
    marginal_probabilities: tuple[float, ...] = Field(min_length=5)
    marginal_residual_values: dict[str, tuple[tuple[float, ...], ...]]
    arm_visit_shifts: dict[str, dict[str, tuple[float, ...]]]
    dropout_logit_intercepts: tuple[float, ...] = Field(min_length=1)
    dropout_treatment_coefficient: float = Field(allow_inf_nan=False)
    measurement_probabilities: dict[str, tuple[float, ...]]


class MultivariateLongitudinalDesignV1(_FrozenModel):
    """Public design and source reference for joint qualification."""

    schema_id: Literal["trialagentbench.multivariate_longitudinal_design/v1"]
    seed: int = Field(ge=0, le=2**32 - 1)
    trial_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participants: int = Field(ge=20)
    worlds: int = Field(ge=2)
    arm_ids: tuple[str, ...] = Field(min_length=2)
    control_arm_id: str = Field(min_length=1)
    outcome_ids: tuple[str, ...] = Field(min_length=2)
    time_values: tuple[float, ...] = Field(min_length=2)
    source_complete_trajectories: int = Field(ge=20)
    fitted_model: MultivariateLongitudinalFittedModelV1
    cells: tuple[MultivariateCellReferenceV1, ...] = Field(min_length=1)
    cross_outcome_pairs: tuple[MultivariatePairReferenceV1, ...] = Field(min_length=1)
    treatment_truth: tuple[MultivariateTreatmentTruthV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_support(self) -> MultivariateLongitudinalDesignV1:
        if self.control_arm_id not in self.arm_ids:
            raise ValueError("control_arm_id must be present in arm_ids")
        if len(set(self.arm_ids)) != len(self.arm_ids):
            raise ValueError("arm_ids must be unique")
        if len(set(self.outcome_ids)) != len(self.outcome_ids):
            raise ValueError("outcome_ids must be unique")
        if len(set(self.time_values)) != len(self.time_values):
            raise ValueError("time_values must be unique")
        outcomes = set(self.outcome_ids)
        fitted = self.fitted_model
        if (
            set(fitted.control_mean_values) != outcomes
            or set(fitted.marginal_residual_values) != outcomes
        ):
            raise ValueError("Fitted marginal keys must match outcome_ids")
        if set(fitted.measurement_probabilities) != outcomes:
            raise ValueError(
                "Fitted measurement-probability keys must match outcome_ids"
            )
        visits = len(self.time_values)
        probabilities = len(fitted.marginal_probabilities)
        if any(
            len(fitted.control_mean_values[outcome]) != visits
            or len(fitted.marginal_residual_values[outcome]) != visits
            or any(
                len(quantiles) != probabilities
                for quantiles in fitted.marginal_residual_values[outcome]
            )
            or len(fitted.measurement_probabilities[outcome]) != visits
            for outcome in self.outcome_ids
        ):
            raise ValueError(
                "Fitted marginal dimensions must match outcomes and visits"
            )
        if len(fitted.dropout_logit_intercepts) != visits - 1:
            raise ValueError(
                "Fitted dropout intercepts must cover every post-baseline visit"
            )
        if any(
            not 0.0 < probability <= 1.0
            for outcome in self.outcome_ids
            for probability in fitted.measurement_probabilities[outcome]
        ):
            raise ValueError("Fitted measurement probabilities must be in (0, 1]")
        cells = len(self.outcome_ids) * visits
        if len(fitted.latent_correlation) != cells or any(
            len(row) != cells for row in fitted.latent_correlation
        ):
            raise ValueError(
                "Fitted latent correlation must be square by outcome-visit cell"
            )
        return self


class MultivariateLongitudinalWorldReceiptV1(_FrozenModel):
    """Checksum-bound identity of one joint longitudinal world."""

    world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    world_index: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    analysis_path: str = Field(pattern=r"^worlds/world_[0-9a-f]{20}\.parquet$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ground_truth_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MultivariateLongitudinalReceiptV1(_FrozenModel):
    """Complete inventory for a joint longitudinal release."""

    schema_id: Literal["trialagentbench.multivariate_longitudinal_receipt/v1"]
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worlds: tuple[MultivariateLongitudinalWorldReceiptV1, ...] = Field(min_length=1)


class MultivariateCellPredictiveV1(_FrozenModel):
    """Across-world native-scale prediction for one source cell."""

    arm_id: str
    outcome_id: str
    time: float = Field(allow_inf_nan=False)
    worlds: int = Field(ge=2)
    source_observations: int = Field(ge=3)
    observations_median: float = Field(ge=0, allow_inf_nan=False)
    observations_interval_95_low: float = Field(ge=0, allow_inf_nan=False)
    observations_interval_95_high: float = Field(ge=0, allow_inf_nan=False)
    source_observations_predictive_rank: float = Field(ge=0, le=1)
    source_mean: float = Field(allow_inf_nan=False)
    mean_median: float = Field(allow_inf_nan=False)
    mean_interval_50_low: float = Field(allow_inf_nan=False)
    mean_interval_50_high: float = Field(allow_inf_nan=False)
    mean_interval_95_low: float = Field(allow_inf_nan=False)
    mean_interval_95_high: float = Field(allow_inf_nan=False)
    source_mean_predictive_rank: float = Field(ge=0, le=1)
    source_standard_deviation: float = Field(gt=0, allow_inf_nan=False)
    standard_deviation_median: float = Field(gt=0, allow_inf_nan=False)
    standardized_quantile_error_median: float = Field(ge=0, allow_inf_nan=False)


class MultivariateJointModeSummaryV1(_FrozenModel):
    """Joint-correlation fidelity for one linkage-retention mode."""

    mode: MultivariateLongitudinalMode
    linkage_retention: float = Field(ge=0, le=1)
    worlds: int = Field(ge=2)
    correlation_mae_mean: float = Field(ge=0, le=2)
    correlation_mae_ci_low: float = Field(ge=0, le=2)
    correlation_mae_ci_high: float = Field(ge=0, le=2)
    correlation_vector_alignment_mean: float = Field(ge=-1, le=1)
    sign_agreement_mean: float = Field(ge=0, le=1)


class MultivariatePairPredictiveV1(_FrozenModel):
    """Across-world recovery of one source cross-outcome correlation."""

    outcome_a: str
    time_a: float = Field(allow_inf_nan=False)
    outcome_b: str
    time_b: float = Field(allow_inf_nan=False)
    worlds: int = Field(ge=2)
    source_correlation: float = Field(ge=-1, le=1)
    source_anchored_correlation_mean: float = Field(ge=-1, le=1)
    source_anchored_correlation_ci_low: float = Field(ge=-1, le=1)
    source_anchored_correlation_ci_high: float = Field(ge=-1, le=1)
    absolute_error: float = Field(ge=0, le=2)


class MultivariateLinkageResponseV1(_FrozenModel):
    """Dose response of joint fidelity to retained outcome linkage."""

    retention_levels: tuple[float, ...] = Field(min_length=5)
    mean_correlation_mae: tuple[float, ...] = Field(min_length=5)
    mae_slope: float = Field(allow_inf_nan=False)
    mae_slope_ci_low: float = Field(allow_inf_nan=False)
    mae_slope_ci_high: float = Field(allow_inf_nan=False)
    negative_slope_fraction: float = Field(ge=0, le=1)
    negative_slope_fraction_ci_low: float = Field(ge=0, le=1)
    negative_slope_fraction_ci_high: float = Field(ge=0, le=1)
    endpoint_improvement_mean: float = Field(allow_inf_nan=False)
    endpoint_improvement_ci_low: float = Field(allow_inf_nan=False)
    endpoint_improvement_ci_high: float = Field(allow_inf_nan=False)


class MultivariateTreatmentRecoveryV1(_FrozenModel):
    """Across-world randomized ANCOVA recovery."""

    arm_id: str
    outcome_id: str
    worlds: int = Field(ge=2)
    truth_contrast: float = Field(allow_inf_nan=False)
    mean_estimate: float = Field(allow_inf_nan=False)
    bias: float = Field(allow_inf_nan=False)
    bias_ci_low: float = Field(allow_inf_nan=False)
    bias_ci_high: float = Field(allow_inf_nan=False)
    bias_simultaneous_ci_low: float = Field(allow_inf_nan=False)
    bias_simultaneous_ci_high: float = Field(allow_inf_nan=False)
    coverage: float = Field(ge=0, le=1)
    coverage_ci_low: float = Field(ge=0, le=1)
    coverage_ci_high: float = Field(ge=0, le=1)


class MultivariateLongitudinalQualificationReportV1(_FrozenModel):
    """Independent joint longitudinal qualification report."""

    schema_id: Literal["trialagentbench.multivariate_longitudinal_report/v1"] = (
        "trialagentbench.multivariate_longitudinal_report/v1"
    )
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cells: tuple[MultivariateCellPredictiveV1, ...]
    pairs: tuple[MultivariatePairPredictiveV1, ...]
    joint_modes: tuple[MultivariateJointModeSummaryV1, ...]
    linkage_response: MultivariateLinkageResponseV1
    treatment_recovery: tuple[MultivariateTreatmentRecoveryV1, ...]


def evaluate_multivariate_longitudinal_qualification(
    *,
    release_dir: Path,
    minimum_worlds: int = 100,
) -> MultivariateLongitudinalQualificationReportV1:
    """Verify and analyze a released multivariate longitudinal campaign."""

    design_path = release_dir / "design.json"
    receipt_path = release_dir / "receipt.json"
    design_payload = json.loads(design_path.read_text(encoding="utf-8"))
    design = MultivariateLongitudinalDesignV1.model_validate(design_payload)
    design_sha256 = _json_sha(design_payload)
    receipt = MultivariateLongitudinalReceiptV1.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    if receipt.design_sha256 != design_sha256:
        raise ValueError("Multivariate longitudinal receipt does not match its design")
    if design.worlds < minimum_worlds:
        raise ValueError(
            "Multivariate longitudinal campaign is below the replication floor"
        )
    if {world.world_index for world in receipt.worlds} != set(range(design.worlds)):
        raise ValueError("Multivariate longitudinal receipt is incomplete")

    frames = []
    for world in receipt.worlds:
        if world.seed != _world_seed(design.seed, world.world_index):
            raise ValueError(
                f"Multivariate longitudinal seed mismatch: {world.world_id}"
            )
        if world.world_id != _world_id(design_sha256, world.world_index):
            raise ValueError(
                f"Multivariate longitudinal identity mismatch: {world.world_id}"
            )
        path = _release_path(release_dir, world.analysis_path)
        if sha256_file(path) != world.analysis_sha256:
            raise ValueError(
                f"Multivariate longitudinal checksum mismatch: {world.world_id}"
            )
        frame = _validate_world(pd.read_parquet(path), design=design, world=world)
        frame["world_index"] = world.world_index
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    pair_metrics, pair_estimates = _world_pair_metrics(combined, design)
    return MultivariateLongitudinalQualificationReportV1(
        design_sha256=design_sha256,
        receipt_sha256=sha256_file(receipt_path),
        cells=tuple(_summarize_cells(combined, design)),
        pairs=tuple(_summarize_pairs(pair_estimates, design)),
        joint_modes=tuple(_summarize_joint_modes(pair_metrics)),
        linkage_response=_summarize_linkage_response(pair_metrics),
        treatment_recovery=tuple(_summarize_treatment_recovery(combined, design)),
    )


def _validate_world(
    frame: pd.DataFrame,
    *,
    design: MultivariateLongitudinalDesignV1,
    world: MultivariateLongitudinalWorldReceiptV1,
) -> pd.DataFrame:
    required = {
        "world_id",
        "mode",
        "participant_id",
        "arm",
        "outcome_id",
        "time",
        "value",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(
            f"Multivariate longitudinal world is missing columns: {missing}"
        )
    output = frame.loc[:, sorted(required)].copy()
    expected_modes = {
        "independent_outcomes",
        "linkage_25",
        "linkage_50",
        "linkage_75",
        "source_anchored",
    }
    if set(output["world_id"].astype("string")) != {world.world_id}:
        raise ValueError("Multivariate longitudinal world identity is inconsistent")
    if set(output["mode"].astype("string")) != expected_modes:
        raise ValueError("Multivariate longitudinal world lacks required linkage modes")
    if set(output["arm"].astype("string")) != set(design.arm_ids):
        raise ValueError("Multivariate longitudinal arm support differs from design")
    if set(output["outcome_id"].astype("string")) != set(design.outcome_ids):
        raise ValueError(
            "Multivariate longitudinal outcome support differs from design"
        )
    output["time"] = pd.to_numeric(output["time"], errors="raise").astype(float)
    output["value"] = pd.to_numeric(output["value"], errors="raise").astype(float)
    if set(output["time"]) != set(design.time_values):
        raise ValueError("Multivariate longitudinal visit support differs from design")
    if np.isinf(output["value"]).any():
        raise ValueError("Multivariate longitudinal measurements cannot be infinite")
    keys = ["mode", "participant_id", "outcome_id", "time"]
    if output.duplicated(keys).any():
        raise ValueError("Multivariate longitudinal world has duplicate planned cells")
    expected_rows = (
        len(expected_modes)
        * design.participants
        * len(design.outcome_ids)
        * len(design.time_values)
    )
    if len(output) != expected_rows:
        raise ValueError(
            "Multivariate longitudinal world lacks the complete planned grid"
        )
    return output


def _summarize_cells(
    frame: pd.DataFrame,
    design: MultivariateLongitudinalDesignV1,
) -> list[MultivariateCellPredictiveV1]:
    source_anchored = frame.loc[frame["mode"].eq("source_anchored")]
    grouped: dict[tuple[str, str, float], list[tuple[float, ...]]] = defaultdict(list)
    for (_, arm, outcome, time), values in source_anchored.groupby(
        ["world_index", "arm", "outcome_id", "time"],
        observed=True,
        sort=True,
    ):
        sample = values["value"].dropna().to_numpy(dtype=float)
        if len(sample) < 3:
            raise ValueError(
                "Multivariate cell requires at least three observed measurements"
            )
        grouped[(str(arm), str(outcome), float(cast(float, time)))].append(
            (
                float(np.mean(sample)),
                float(np.std(sample, ddof=1)),
                float(np.quantile(sample, 0.1)),
                float(np.quantile(sample, 0.5)),
                float(np.quantile(sample, 0.9)),
                float(len(sample)),
            )
        )
    references = {
        (cell.arm_id, cell.outcome_id, cell.time): cell for cell in design.cells
    }
    output = []
    for key, rows in sorted(grouped.items()):
        reference = references[key]
        summary_values = np.asarray(rows, dtype=float)
        source_quantiles = np.asarray(
            [reference.quantile_10, reference.quantile_50, reference.quantile_90]
        )
        quantile_errors = (
            np.mean(
                np.abs(summary_values[:, 2:5] - source_quantiles[None, :]),
                axis=1,
            )
            / reference.standard_deviation
        )
        means = np.asarray(summary_values[:, 0], dtype=np.float64)
        observations = np.asarray(summary_values[:, 5], dtype=np.float64)
        output.append(
            MultivariateCellPredictiveV1(
                arm_id=key[0],
                outcome_id=key[1],
                time=key[2],
                worlds=len(rows),
                source_observations=reference.observations,
                observations_median=float(np.median(observations)),
                observations_interval_95_low=float(np.quantile(observations, 0.025)),
                observations_interval_95_high=float(np.quantile(observations, 0.975)),
                source_observations_predictive_rank=_predictive_rank(
                    observations,
                    float(reference.observations),
                ),
                source_mean=reference.mean,
                mean_median=float(np.median(means)),
                mean_interval_50_low=float(np.quantile(means, 0.25)),
                mean_interval_50_high=float(np.quantile(means, 0.75)),
                mean_interval_95_low=float(np.quantile(means, 0.025)),
                mean_interval_95_high=float(np.quantile(means, 0.975)),
                source_mean_predictive_rank=_predictive_rank(means, reference.mean),
                source_standard_deviation=reference.standard_deviation,
                standard_deviation_median=float(np.median(summary_values[:, 1])),
                standardized_quantile_error_median=float(np.median(quantile_errors)),
            )
        )
    if set(grouped) != set(references):
        raise ValueError("Multivariate cell summaries do not match the design")
    return output


def _world_pair_metrics(
    frame: pd.DataFrame,
    design: MultivariateLongitudinalDesignV1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = np.asarray(
        [pair.spearman_correlation for pair in design.cross_outcome_pairs],
        dtype=float,
    )
    strong = np.abs(reference) >= 0.2
    rows = []
    pair_rows: list[dict[str, object]] = []
    for (world_index, mode), world in frame.groupby(
        ["world_index", "mode"],
        observed=True,
        sort=True,
    ):
        correlations = cross_outcome_correlations(
            world,
            design.cross_outcome_pairs,
        )
        rows.append(
            {
                "world_index": int(cast(int, world_index)),
                "mode": str(mode),
                "mae": float(np.mean(np.abs(correlations - reference))),
                "alignment": float(np.corrcoef(correlations, reference)[0, 1]),
                "sign_agreement": (
                    1.0
                    if not strong.any()
                    else float(
                        np.mean(
                            np.sign(correlations[strong]) == np.sign(reference[strong])
                        )
                    )
                ),
            }
        )
        pair_rows.extend(
            {
                "world_index": int(cast(int, world_index)),
                "mode": str(mode),
                "pair_index": pair_index,
                "correlation": float(correlation),
            }
            for pair_index, correlation in enumerate(correlations)
        )
    output = pd.DataFrame(rows)
    if not np.isfinite(output[["mae", "alignment", "sign_agreement"]]).all(axis=None):
        raise ValueError("Multivariate joint metrics must be finite")
    return output, pd.DataFrame(pair_rows)


def _summarize_pairs(
    estimates: pd.DataFrame,
    design: MultivariateLongitudinalDesignV1,
) -> list[MultivariatePairPredictiveV1]:
    source_anchored = estimates.loc[estimates["mode"].eq("source_anchored")]
    output = []
    for pair_index, values in source_anchored.groupby("pair_index", sort=True):
        reference = design.cross_outcome_pairs[int(cast(int, pair_index))]
        correlations = values["correlation"].to_numpy(dtype=float)
        ci_low, ci_high = _mean_interval(correlations)
        mean = float(np.mean(correlations))
        output.append(
            MultivariatePairPredictiveV1(
                outcome_a=reference.outcome_a,
                time_a=reference.time_a,
                outcome_b=reference.outcome_b,
                time_b=reference.time_b,
                worlds=len(correlations),
                source_correlation=reference.spearman_correlation,
                source_anchored_correlation_mean=mean,
                source_anchored_correlation_ci_low=ci_low,
                source_anchored_correlation_ci_high=ci_high,
                absolute_error=abs(mean - reference.spearman_correlation),
            )
        )
    if len(output) != len(design.cross_outcome_pairs):
        raise ValueError("Production pair summaries do not match the design")
    return output


def _summarize_joint_modes(
    metrics: pd.DataFrame,
) -> list[MultivariateJointModeSummaryV1]:
    retention = {
        "independent_outcomes": 0.0,
        "linkage_25": 0.25,
        "linkage_50": 0.5,
        "linkage_75": 0.75,
        "source_anchored": 1.0,
    }
    output = []
    for mode, group in metrics.groupby("mode", sort=True):
        ci_low, ci_high = _mean_interval(group["mae"].to_numpy(dtype=float))
        output.append(
            MultivariateJointModeSummaryV1(
                mode=str(mode),
                linkage_retention=retention[str(mode)],
                worlds=len(group),
                correlation_mae_mean=float(group["mae"].mean()),
                correlation_mae_ci_low=ci_low,
                correlation_mae_ci_high=ci_high,
                correlation_vector_alignment_mean=float(group["alignment"].mean()),
                sign_agreement_mean=float(group["sign_agreement"].mean()),
            )
        )
    return output


def _summarize_linkage_response(
    metrics: pd.DataFrame,
) -> MultivariateLinkageResponseV1:
    retention = {
        "independent_outcomes": 0.0,
        "linkage_25": 0.25,
        "linkage_50": 0.5,
        "linkage_75": 0.75,
        "source_anchored": 1.0,
    }
    metrics["retention"] = metrics["mode"].map(retention).astype(float)
    slopes = []
    improvements = []
    for _, world in metrics.groupby("world_index", sort=True):
        ordered = world.sort_values("retention")
        slopes.append(float(np.polyfit(ordered["retention"], ordered["mae"], 1)[0]))
        improvements.append(float(ordered["mae"].iloc[0] - ordered["mae"].iloc[-1]))
    slopes_array = np.asarray(slopes, dtype=float)
    improvements_array = np.asarray(improvements, dtype=float)
    slope_ci = _mean_interval(slopes_array)
    improvement_ci = _mean_interval(improvements_array)
    negative_interval = proportion_interval(
        int(np.sum(slopes_array < 0.0)),
        len(slopes_array),
    )
    means = metrics.groupby("retention", sort=True)["mae"].mean()
    return MultivariateLinkageResponseV1(
        retention_levels=tuple(float(value) for value in means.index),
        mean_correlation_mae=tuple(float(value) for value in means),
        mae_slope=float(np.mean(slopes_array)),
        mae_slope_ci_low=slope_ci[0],
        mae_slope_ci_high=slope_ci[1],
        negative_slope_fraction=float(np.mean(slopes_array < 0.0)),
        negative_slope_fraction_ci_low=negative_interval[0],
        negative_slope_fraction_ci_high=negative_interval[1],
        endpoint_improvement_mean=float(np.mean(improvements_array)),
        endpoint_improvement_ci_low=improvement_ci[0],
        endpoint_improvement_ci_high=improvement_ci[1],
    )


def _summarize_treatment_recovery(
    frame: pd.DataFrame,
    design: MultivariateLongitudinalDesignV1,
) -> list[MultivariateTreatmentRecoveryV1]:
    source_anchored = frame.loc[frame["mode"].eq("source_anchored")]
    estimates: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for truth in design.treatment_truth:
        outcome = source_anchored.loc[
            source_anchored["outcome_id"].eq(truth.outcome_id)
        ]
        for _, world in outcome.groupby("world_index", sort=True):
            estimate, standard_error = _fit_ancova(
                world,
                arm_id=truth.arm_id,
                control_arm_id=design.control_arm_id,
                baseline_time=min(design.time_values),
                final_time=truth.final_time,
            )
            estimates[(truth.arm_id, truth.outcome_id)].append(
                (estimate, standard_error)
            )
    truth_by_key = {
        (truth.arm_id, truth.outcome_id): truth for truth in design.treatment_truth
    }
    simultaneous_z = float(norm.ppf(1.0 - 0.05 / (2.0 * len(truth_by_key))))
    output = []
    for key, rows in sorted(estimates.items()):
        truth = truth_by_key[key]
        values = np.asarray(rows, dtype=float)
        bias = values[:, 0] - truth.contrast
        bias_ci = _mean_interval(bias)
        bias_simultaneous_ci = _mean_interval(bias, z=simultaneous_z)
        covered = np.abs(bias) <= norm.ppf(0.975) * values[:, 1]
        coverage_ci = proportion_interval(int(covered.sum()), len(covered))
        output.append(
            MultivariateTreatmentRecoveryV1(
                arm_id=key[0],
                outcome_id=key[1],
                worlds=len(rows),
                truth_contrast=truth.contrast,
                mean_estimate=float(np.mean(values[:, 0])),
                bias=float(np.mean(bias)),
                bias_ci_low=bias_ci[0],
                bias_ci_high=bias_ci[1],
                bias_simultaneous_ci_low=bias_simultaneous_ci[0],
                bias_simultaneous_ci_high=bias_simultaneous_ci[1],
                coverage=float(np.mean(covered)),
                coverage_ci_low=coverage_ci[0],
                coverage_ci_high=coverage_ci[1],
            )
        )
    return output


def _fit_ancova(
    frame: pd.DataFrame,
    *,
    arm_id: str,
    control_arm_id: str,
    baseline_time: float,
    final_time: float,
) -> tuple[float, float]:
    selected = frame.loc[
        frame["arm"].isin([control_arm_id, arm_id])
        & frame["time"].isin([baseline_time, final_time])
    ]
    wide = selected.pivot(
        index=["participant_id", "arm"], columns="time", values="value"
    ).dropna()
    if len(wide) < 20 or wide.index.get_level_values("arm").nunique() != 2:
        raise ValueError(
            "ANCOVA recovery requires both arms and at least 20 complete participants"
        )
    treatment = np.asarray(
        wide.index.get_level_values("arm").astype("string") == arm_id,
        dtype=float,
    )
    baseline = wide[baseline_time].to_numpy(dtype=float)
    centered_baseline = baseline - float(np.mean(baseline))
    design = sm.add_constant(
        np.column_stack(
            [
                treatment,
                centered_baseline,
                treatment * centered_baseline,
            ]
        ),
        has_constant="add",
    )
    fit = sm.OLS(wide[final_time].to_numpy(dtype=float), design).fit(cov_type="HC2")
    estimate = float(fit.params[1])
    standard_error = float(fit.bse[1])
    if not np.isfinite([estimate, standard_error]).all() or standard_error <= 0.0:
        raise ValueError("ANCOVA estimate and standard error must be finite")
    return estimate, standard_error


def cross_outcome_correlations(
    frame: pd.DataFrame,
    references: tuple[MultivariatePairReferenceV1, ...],
) -> NDArray[np.float64]:
    centered = frame.copy()
    centered["residual"] = centered["value"] - centered.groupby(
        ["arm", "outcome_id", "time"],
        observed=True,
    )["value"].transform("mean")
    wide = centered.pivot(
        index="participant_id",
        columns=["outcome_id", "time"],
        values="residual",
    )
    correlation_matrix = wide.corr(method="spearman")
    output = []
    for pair in references:
        column_a = (pair.outcome_a, pair.time_a)
        column_b = (pair.outcome_b, pair.time_b)
        values = wide.loc[:, [column_a, column_b]].dropna()
        if len(values) < 20:
            raise ValueError("Joint correlation requires at least 20 complete pairs")
        output.append(float(cast(float, correlation_matrix.loc[column_a, column_b])))
    return np.asarray(output, dtype=float)


def _predictive_rank(values: NDArray[np.float64], reference: float) -> float:
    return float(
        (
            np.count_nonzero(values < reference)
            + 0.5 * np.count_nonzero(values == reference)
        )
        / len(values)
    )


def _mean_interval(
    values: NDArray[np.float64],
    *,
    z: float = float(norm.ppf(0.975)),
) -> tuple[float, float]:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Monte Carlo interval requires at least two finite values")
    half_width = z * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    mean = float(np.mean(values))
    return mean - half_width, mean + half_width


def _world_seed(seed: int, world_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:joint:{world_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _world_id(design_sha256: str, world_index: int) -> str:
    return (
        "world_"
        + hashlib.sha256(f"{design_sha256}:{world_index}".encode()).hexdigest()[:20]
    )


def _json_sha(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _release_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Release path escapes its root")
    return path


__all__ = [
    "MultivariateCellReferenceV1",
    "MultivariateLongitudinalDesignV1",
    "MultivariateLongitudinalFittedModelV1",
    "MultivariateLongitudinalQualificationReportV1",
    "MultivariateLongitudinalReceiptV1",
    "MultivariateLongitudinalWorldReceiptV1",
    "MultivariatePairPredictiveV1",
    "MultivariatePairReferenceV1",
    "MultivariateTreatmentTruthV1",
    "cross_outcome_correlations",
    "evaluate_multivariate_longitudinal_qualification",
]
