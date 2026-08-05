"""Export compact PATENCY and HeadSOAR qualification evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_validation.external.recovery.ordinal import (
    OrdinalQualificationDesignV1,
    OrdinalQualificationReportV1,
)
from trialagentbench_validation.external.recovery.survival import (
    SurvivalQualificationDesignV1,
    SurvivalQualificationReportV1,
)
from trialagentbench_validation.external.release.artifacts import (
    write_external_artifact_manifest,
)

_OUTPUT_FILES = {
    "README.md",
    "artifact_manifest.json",
    "headsoar_category_predictive.csv",
    "headsoar_clean_wheel_replay.json",
    "headsoar_independent_report.json",
    "headsoar_proportional_odds_dose_recovery.csv",
    "headsoar_safety_predictive.csv",
    "open_outcome_analysis_concordance.csv",
    "open_outcome_summary.json",
    "patency_cox_dose_recovery.csv",
    "patency_clean_wheel_replay.json",
    "patency_independent_report.json",
    "patency_km_predictive.csv",
    "patency_risk_set_predictive.csv",
    "patency_rmst_predictive.csv",
}


class OpenOutcomeEvidenceSummaryV1(BaseModel):
    """Headline native-scale outcome realism evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.open_outcome_evidence/v1"] = (
        "trialagentbench.open_outcome_evidence/v1"
    )
    patency_participants: int = Field(ge=100)
    patency_worlds: int = Field(ge=100)
    patency_source_dose_curve_mae: float = Field(ge=0, le=1)
    patency_source_dose_risk_set_mae: float = Field(ge=0)
    patency_source_dose_rmst_mae_days: float = Field(ge=0)
    patency_intact_event_time_curve_mae: float = Field(ge=0, le=1)
    patency_intact_event_time_curve_mae_ci_low: float = Field(ge=0, le=1)
    patency_intact_event_time_curve_mae_ci_high: float = Field(ge=0, le=1)
    patency_broken_event_time_curve_mae: float = Field(ge=0, le=1)
    patency_broken_event_time_curve_mae_ci_low: float = Field(ge=0, le=1)
    patency_broken_event_time_curve_mae_ci_high: float = Field(ge=0, le=1)
    patency_broken_minus_intact_curve_mae: float = Field(ge=-1, le=1)
    patency_broken_minus_intact_curve_mae_ci_low: float = Field(allow_inf_nan=False)
    patency_broken_minus_intact_curve_mae_ci_high: float = Field(allow_inf_nan=False)
    patency_source_dose_log_hazard_bias: float = Field(allow_inf_nan=False)
    patency_source_dose_coverage: float = Field(ge=0, le=1)
    patency_source_dose_coverage_ci_low: float = Field(ge=0, le=1)
    patency_source_dose_coverage_ci_high: float = Field(ge=0, le=1)
    patency_dose_slope: float = Field(allow_inf_nan=False)
    patency_dose_slope_ci_low: float = Field(allow_inf_nan=False)
    patency_dose_slope_ci_high: float = Field(allow_inf_nan=False)
    patency_monotone_world_fraction: float = Field(ge=0, le=1)
    patency_monotone_world_fraction_ci_low: float = Field(ge=0, le=1)
    patency_monotone_world_fraction_ci_high: float = Field(ge=0, le=1)
    headsoar_participants: int = Field(ge=100)
    headsoar_worlds: int = Field(ge=100)
    headsoar_source_dose_category_mae: float = Field(ge=0, le=1)
    headsoar_source_dose_cumulative_mae: float = Field(ge=0, le=1)
    headsoar_safety_probability_mae: float = Field(ge=0, le=1)
    headsoar_intact_high_dose_log_odds: float = Field(allow_inf_nan=False)
    headsoar_intact_high_dose_log_odds_ci_low: float = Field(allow_inf_nan=False)
    headsoar_intact_high_dose_log_odds_ci_high: float = Field(allow_inf_nan=False)
    headsoar_broken_arm_linkage_log_odds: float = Field(allow_inf_nan=False)
    headsoar_broken_arm_linkage_log_odds_ci_low: float = Field(allow_inf_nan=False)
    headsoar_broken_arm_linkage_log_odds_ci_high: float = Field(allow_inf_nan=False)
    headsoar_intact_high_dose_category_mae: float = Field(ge=0, le=1)
    headsoar_intact_high_dose_category_mae_ci_low: float = Field(ge=0, le=1)
    headsoar_intact_high_dose_category_mae_ci_high: float = Field(ge=0, le=1)
    headsoar_broken_arm_category_mae: float = Field(ge=0, le=1)
    headsoar_broken_arm_category_mae_ci_low: float = Field(ge=0, le=1)
    headsoar_broken_arm_category_mae_ci_high: float = Field(ge=0, le=1)
    headsoar_broken_minus_intact_category_mae: float = Field(ge=-1, le=1)
    headsoar_broken_minus_intact_category_mae_ci_low: float = Field(allow_inf_nan=False)
    headsoar_broken_minus_intact_category_mae_ci_high: float = Field(
        allow_inf_nan=False
    )
    headsoar_source_dose_log_odds_bias: float = Field(allow_inf_nan=False)
    headsoar_source_dose_coverage: float = Field(ge=0, le=1)
    headsoar_source_dose_coverage_ci_low: float = Field(ge=0, le=1)
    headsoar_source_dose_coverage_ci_high: float = Field(ge=0, le=1)
    headsoar_dose_slope: float = Field(allow_inf_nan=False)
    headsoar_dose_slope_ci_low: float = Field(allow_inf_nan=False)
    headsoar_dose_slope_ci_high: float = Field(allow_inf_nan=False)
    headsoar_monotone_world_fraction: float = Field(ge=0, le=1)
    headsoar_monotone_world_fraction_ci_low: float = Field(ge=0, le=1)
    headsoar_monotone_world_fraction_ci_high: float = Field(ge=0, le=1)


def write_open_outcome_evidence(
    *,
    patency_design_path: Path,
    patency_report_path: Path,
    headsoar_design_path: Path,
    headsoar_report_path: Path,
    output_dir: Path,
) -> OpenOutcomeEvidenceSummaryV1:
    """Write native-scale survival and ordinal evidence tables."""

    existing = (
        {path.name for path in output_dir.iterdir()} if output_dir.exists() else set()
    )
    if unexpected := sorted(existing - _OUTPUT_FILES):
        raise ValueError(
            f"Open outcome evidence directory contains unexpected files: {unexpected}"
        )
    patency_design = SurvivalQualificationDesignV1.model_validate_json(
        patency_design_path.read_text(encoding="utf-8")
    )
    patency_report = SurvivalQualificationReportV1.model_validate_json(
        patency_report_path.read_text(encoding="utf-8")
    )
    headsoar_design = OrdinalQualificationDesignV1.model_validate_json(
        headsoar_design_path.read_text(encoding="utf-8")
    )
    headsoar_report = OrdinalQualificationReportV1.model_validate_json(
        headsoar_report_path.read_text(encoding="utf-8")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    patency_curves = pd.DataFrame(
        [row.model_dump(mode="json") for row in patency_report.curves]
    )
    patency_risk_sets = pd.DataFrame(
        [row.model_dump(mode="json") for row in patency_report.risk_sets]
    )
    patency_rmst = pd.DataFrame(
        [row.model_dump(mode="json") for row in patency_report.rmst]
    )
    patency_recovery = pd.DataFrame(
        [row.model_dump(mode="json") for row in patency_report.recovery]
    )
    headsoar_categories = pd.DataFrame(
        [row.model_dump(mode="json") for row in headsoar_report.categories]
    )
    headsoar_safety = pd.DataFrame(
        [row.model_dump(mode="json") for row in headsoar_report.safety]
    )
    headsoar_recovery = pd.DataFrame(
        [row.model_dump(mode="json") for row in headsoar_report.recovery]
    )
    patency_source = next(
        row for row in patency_report.recovery if row.dose_multiplier == 1.0
    )
    headsoar_source = next(
        row for row in headsoar_report.recovery if row.dose_multiplier == 1.0
    )
    summary = OpenOutcomeEvidenceSummaryV1(
        patency_participants=patency_design.participants,
        patency_worlds=patency_report.worlds,
        patency_source_dose_curve_mae=patency_report.source_dose_curve_mae,
        patency_source_dose_risk_set_mae=(patency_report.source_dose_risk_set_mae),
        patency_source_dose_rmst_mae_days=patency_report.source_dose_rmst_mae,
        patency_intact_event_time_curve_mae=(
            patency_report.intact_event_time_curve_mae
        ),
        patency_intact_event_time_curve_mae_ci_low=(
            patency_report.intact_event_time_curve_mae_ci_low
        ),
        patency_intact_event_time_curve_mae_ci_high=(
            patency_report.intact_event_time_curve_mae_ci_high
        ),
        patency_broken_event_time_curve_mae=(
            patency_report.broken_event_time_curve_mae
        ),
        patency_broken_event_time_curve_mae_ci_low=(
            patency_report.broken_event_time_curve_mae_ci_low
        ),
        patency_broken_event_time_curve_mae_ci_high=(
            patency_report.broken_event_time_curve_mae_ci_high
        ),
        patency_broken_minus_intact_curve_mae=(
            patency_report.broken_minus_intact_curve_mae
        ),
        patency_broken_minus_intact_curve_mae_ci_low=(
            patency_report.broken_minus_intact_curve_mae_ci_low
        ),
        patency_broken_minus_intact_curve_mae_ci_high=(
            patency_report.broken_minus_intact_curve_mae_ci_high
        ),
        patency_source_dose_log_hazard_bias=patency_source.bias,
        patency_source_dose_coverage=patency_source.coverage,
        patency_source_dose_coverage_ci_low=patency_source.coverage_ci_low,
        patency_source_dose_coverage_ci_high=patency_source.coverage_ci_high,
        patency_dose_slope=patency_report.dose_response_slope,
        patency_dose_slope_ci_low=patency_report.dose_response_slope_ci_low,
        patency_dose_slope_ci_high=patency_report.dose_response_slope_ci_high,
        patency_monotone_world_fraction=patency_report.monotone_world_fraction,
        patency_monotone_world_fraction_ci_low=patency_report.monotone_world_fraction_ci_low,
        patency_monotone_world_fraction_ci_high=patency_report.monotone_world_fraction_ci_high,
        headsoar_participants=headsoar_design.participants,
        headsoar_worlds=headsoar_report.worlds,
        headsoar_source_dose_category_mae=(headsoar_report.source_dose_category_mae),
        headsoar_source_dose_cumulative_mae=(
            headsoar_report.source_dose_cumulative_mae
        ),
        headsoar_safety_probability_mae=(headsoar_report.safety_probability_mae),
        headsoar_intact_high_dose_log_odds=(headsoar_report.intact_high_dose_log_odds),
        headsoar_intact_high_dose_log_odds_ci_low=(
            headsoar_report.intact_high_dose_log_odds_ci_low
        ),
        headsoar_intact_high_dose_log_odds_ci_high=(
            headsoar_report.intact_high_dose_log_odds_ci_high
        ),
        headsoar_broken_arm_linkage_log_odds=(
            headsoar_report.broken_arm_linkage_log_odds
        ),
        headsoar_broken_arm_linkage_log_odds_ci_low=(
            headsoar_report.broken_arm_linkage_log_odds_ci_low
        ),
        headsoar_broken_arm_linkage_log_odds_ci_high=(
            headsoar_report.broken_arm_linkage_log_odds_ci_high
        ),
        headsoar_intact_high_dose_category_mae=(
            headsoar_report.intact_high_dose_category_mae
        ),
        headsoar_intact_high_dose_category_mae_ci_low=(
            headsoar_report.intact_high_dose_category_mae_ci_low
        ),
        headsoar_intact_high_dose_category_mae_ci_high=(
            headsoar_report.intact_high_dose_category_mae_ci_high
        ),
        headsoar_broken_arm_category_mae=(headsoar_report.broken_arm_category_mae),
        headsoar_broken_arm_category_mae_ci_low=(
            headsoar_report.broken_arm_category_mae_ci_low
        ),
        headsoar_broken_arm_category_mae_ci_high=(
            headsoar_report.broken_arm_category_mae_ci_high
        ),
        headsoar_broken_minus_intact_category_mae=(
            headsoar_report.broken_minus_intact_category_mae
        ),
        headsoar_broken_minus_intact_category_mae_ci_low=(
            headsoar_report.broken_minus_intact_category_mae_ci_low
        ),
        headsoar_broken_minus_intact_category_mae_ci_high=(
            headsoar_report.broken_minus_intact_category_mae_ci_high
        ),
        headsoar_source_dose_log_odds_bias=headsoar_source.bias,
        headsoar_source_dose_coverage=headsoar_source.coverage,
        headsoar_source_dose_coverage_ci_low=headsoar_source.coverage_ci_low,
        headsoar_source_dose_coverage_ci_high=headsoar_source.coverage_ci_high,
        headsoar_dose_slope=headsoar_report.dose_response_slope,
        headsoar_dose_slope_ci_low=headsoar_report.dose_response_slope_ci_low,
        headsoar_dose_slope_ci_high=headsoar_report.dose_response_slope_ci_high,
        headsoar_monotone_world_fraction=headsoar_report.monotone_world_fraction,
        headsoar_monotone_world_fraction_ci_low=headsoar_report.monotone_world_fraction_ci_low,
        headsoar_monotone_world_fraction_ci_high=headsoar_report.monotone_world_fraction_ci_high,
    )
    concordance = pd.DataFrame(
        [
            {
                "trial": "PATENCY",
                "estimand": "Cox log hazard ratio",
                "dose_multiplier": row.dose_multiplier,
                "configured_value": row.truth_log_hazard_ratio,
                "production_mean": row.mean_log_hazard_ratio,
                "predictive_50_low": row.predictive_50_low,
                "predictive_50_high": row.predictive_50_high,
                "predictive_95_low": row.predictive_95_low,
                "predictive_95_high": row.predictive_95_high,
                "mean_bias": row.bias,
                "mean_bias_ci_low": row.bias_ci_low,
                "mean_bias_ci_high": row.bias_ci_high,
                "interval_coverage": row.coverage,
                "interval_coverage_ci_low": row.coverage_ci_low,
                "interval_coverage_ci_high": row.coverage_ci_high,
                "worlds": row.worlds,
            }
            for row in patency_report.recovery
        ]
        + [
            {
                "trial": "HeadSOAR",
                "estimand": "proportional-odds log common odds ratio",
                "dose_multiplier": row.dose_multiplier,
                "configured_value": row.truth_log_common_odds_ratio,
                "production_mean": row.mean_log_common_odds_ratio,
                "predictive_50_low": row.predictive_50_low,
                "predictive_50_high": row.predictive_50_high,
                "predictive_95_low": row.predictive_95_low,
                "predictive_95_high": row.predictive_95_high,
                "mean_bias": row.bias,
                "mean_bias_ci_low": row.bias_ci_low,
                "mean_bias_ci_high": row.bias_ci_high,
                "interval_coverage": row.coverage,
                "interval_coverage_ci_low": row.coverage_ci_low,
                "interval_coverage_ci_high": row.coverage_ci_high,
                "worlds": row.worlds,
            }
            for row in headsoar_report.recovery
        ]
    )
    for path, frame in (
        ("patency_km_predictive.csv", patency_curves),
        ("patency_risk_set_predictive.csv", patency_risk_sets),
        ("patency_rmst_predictive.csv", patency_rmst),
        ("patency_cox_dose_recovery.csv", patency_recovery),
        ("headsoar_category_predictive.csv", headsoar_categories),
        ("headsoar_safety_predictive.csv", headsoar_safety),
        ("headsoar_proportional_odds_dose_recovery.csv", headsoar_recovery),
        ("open_outcome_analysis_concordance.csv", concordance),
    ):
        frame.to_csv(output_dir / path, index=False, lineterminator="\n")
    (output_dir / "open_outcome_summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "patency_independent_report.json").write_text(
        patency_report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "headsoar_independent_report.json").write_text(
        headsoar_report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Native-scale outcome qualification\n\n"
        "PATENCY contributes 2,638 randomized participants with complete three-year "
        "MACCE follow-up. The production campaign fits the control survival curve, "
        "applies the source Cox contrast, reproduces random and administrative "
        "censoring, and tests four treatment-effect doses. The independent package "
        "recomputes Kaplan-Meier survival and cumulative hazard, numbers at risk, "
        "restricted mean survival time, and robust Cox estimates from "
        f"{patency_report.worlds} released source-sized worlds. Monte Carlo "
        "confidence intervals quantify the mean; "
        "Wilson intervals quantify recovery coverage and the fraction of worlds "
        "with the expected dose ordering; 50% and 95% predictive "
        "intervals quantify between-world variability. A "
        "negative control permutes event indicators away from follow-up times while "
        "retaining their marginals; a paired 95% confidence interval compares "
        "curve error before and after permutation within each world.\n\n"
        "HeadSOAR contributes 1,368 randomized participants, the complete 0-6 "
        "modified Rankin Scale, arm-specific missingness, and a published "
        "proportional-odds route. The production campaign samples fitted category "
        "distributions through the production multiclass generator. The independent "
        "package recomputes category and cumulative probabilities, mortality, "
        "missingness, five binary safety outcomes, and proportional-odds recovery "
        "across four effect doses and "
        f"{headsoar_report.worlds} source-sized worlds. An arm-permutation "
        "negative control retains the high-dose mRS marginal distribution and arm "
        "sizes while breaking the treatment-outcome linkage. A paired 95% "
        "confidence interval compares category-probability error before and after "
        "permutation within each world. Wilson intervals quantify interval coverage "
        "and the fraction of worlds with the expected dose ordering.\n",
        encoding="utf-8",
    )
    write_external_artifact_manifest(output_dir)
    return summary


def main() -> None:
    """Run the open-outcome evidence exporter."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--patency-design", type=Path, required=True)
    parser.add_argument("--patency-report", type=Path, required=True)
    parser.add_argument("--headsoar-design", type=Path, required=True)
    parser.add_argument("--headsoar-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_open_outcome_evidence(
        patency_design_path=args.patency_design,
        patency_report_path=args.patency_report,
        headsoar_design_path=args.headsoar_design,
        headsoar_report_path=args.headsoar_report,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "OpenOutcomeEvidenceSummaryV1",
    "write_open_outcome_evidence",
]
