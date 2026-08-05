"""Export compact native-scale evidence from the TERECO qualification."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from trialagentbench_validation.external.recovery.longitudinal import (
    LongitudinalQualificationReportV1,
)
from trialagentbench_validation.external.release.artifacts import (
    write_external_artifact_manifest,
)
from trialagentbench_validation.io import sha256_file

_TRIAL_ID = "TERECO:six_minute_walk_distance"


def write_tereco_validation_results(
    *,
    report_path: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write aggregate evidence tables from a verified report."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("TERECO qualification evidence directory must be empty")
    report = LongitudinalQualificationReportV1.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    trial_ids = {row.trial_id for row in report.modes}
    if trial_ids != {_TRIAL_ID}:
        raise ValueError(
            f"TERECO report has unexpected trial identities: {sorted(trial_ids)}"
        )
    worlds = {row.worlds for row in report.modes}
    if len(worlds) != 1 or min(worlds) < 200:
        raise ValueError("TERECO evidence requires at least 200 worlds in every mode")
    if not report.marginal_predictive:
        raise ValueError("TERECO report lacks native-scale predictive summaries")

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "tereco_marginal_predictive.csv": report.marginal_predictive,
        "tereco_mode_summary.csv": report.modes,
        "tereco_treatment_recovery.csv": report.treatment_recovery,
    }
    paths = []
    for name, rows in tables.items():
        path = output_dir / name
        pd.DataFrame([row.model_dump(mode="json") for row in rows]).to_csv(
            path,
            index=False,
            lineterminator="\n",
        )
        paths.append(path)

    dose = report.linkage_dose_response
    if len(dose) != 1:
        raise ValueError("TERECO report requires exactly one linkage response")
    modes_by_retention = {
        row.linkage_retention: row
        for row in report.modes
        if row.linkage_retention is not None
    }
    dose_rows = []
    for retention, correlation in zip(
        dose[0].retention_levels,
        dose[0].mean_correlations,
        strict=True,
    ):
        mode = modes_by_retention[retention]
        dose_rows.append(
            {
                "trial_id": _TRIAL_ID,
                "worlds": dose[0].worlds,
                "linkage_retention": retention,
                "mean_within_stratum_adjacent_correlation": correlation,
                "correlation_ci_low": mode.within_stratum_adjacent_correlation_ci_low,
                "correlation_ci_high": mode.within_stratum_adjacent_correlation_ci_high,
            }
        )
    dose_path = output_dir / "tereco_linkage_response.csv"
    pd.DataFrame(dose_rows).to_csv(dose_path, index=False, lineterminator="\n")
    paths.append(dose_path)

    source_anchored = next(row for row in report.modes if row.mode == "source_anchored")
    broken = next(row for row in report.modes if row.mode == "independent_marginal")
    treatment = report.treatment_recovery[0]
    summary = {
        "schema_id": "trialagentbench.tereco_qualification_summary/v1",
        "trial_id": _TRIAL_ID,
        "worlds": next(iter(worlds)),
        "source_sized": True,
        "analysis_population": 119,
        "design_sha256": report.design_sha256,
        "receipt_sha256": report.receipt_sha256,
        "report_sha256": sha256_file(report_path),
        "source_anchored": {
            "within_stratum_adjacent_correlation_mean": (
                source_anchored.within_stratum_adjacent_correlation_mean
            ),
            "within_stratum_adjacent_correlation_ci": [
                source_anchored.within_stratum_adjacent_correlation_ci_low,
                source_anchored.within_stratum_adjacent_correlation_ci_high,
            ],
            "marginal_standardized_error_mean": (
                source_anchored.marginal_standardized_error_mean
            ),
        },
        "independent_marginal_control": {
            "within_stratum_adjacent_correlation_mean": (
                broken.within_stratum_adjacent_correlation_mean
            ),
            "within_stratum_adjacent_correlation_ci": [
                broken.within_stratum_adjacent_correlation_ci_low,
                broken.within_stratum_adjacent_correlation_ci_high,
            ],
        },
        "linkage_response": dose[0].model_dump(mode="json"),
        "treatment_recovery": treatment.model_dump(mode="json"),
    }
    summary_path = output_dir / "tereco_qualification_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.append(summary_path)
    write_external_artifact_manifest(output_dir)
    paths.append(output_dir / "artifact_manifest.json")
    return tuple(paths)


def main(argv: Sequence[str] | None = None) -> int:
    """Export TERECO qualification evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    write_tereco_validation_results(
        report_path=args.report,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["write_tereco_validation_results"]
