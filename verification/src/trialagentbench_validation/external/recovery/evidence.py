"""Export compact evidence from independently verified qualifications."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from trialagentbench_validation.external.recovery.longitudinal import (
    LongitudinalQualificationReportV1,
)
from trialagentbench_validation.external.recovery.rctbench import (
    RctCellSummaryV1,
    RctDoseResponseV1,
    RctLinkageDoseResponseV1,
    RctQualificationDesignV1,
    RctQualificationReportV1,
)
from trialagentbench_validation.external.release.artifacts import (
    sha256_file,
    write_external_artifact_manifest,
)


def export_validation_results(
    *,
    longitudinal_report_path: Path,
    rct_report_path: Path,
    rct_design_path: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Validate full reports and export their scientific evidence tables."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Qualification evidence output directory must be empty.")
    longitudinal = LongitudinalQualificationReportV1.model_validate_json(
        longitudinal_report_path.read_text(encoding="utf-8")
    )
    rct = RctQualificationReportV1.model_validate_json(
        rct_report_path.read_text(encoding="utf-8")
    )
    rct_design = RctQualificationDesignV1.model_validate_json(
        rct_design_path.read_text(encoding="utf-8")
    )
    if rct.design_sha256 != _canonical_json_sha(rct_design_path):
        raise ValueError("RCT qualification report does not bind the supplied design.")
    outcome_kinds = [trial.fitted_analysis.outcome_kind for trial in rct_design.trials]
    if rct.trials != len(rct_design.trials):
        raise ValueError("RCT report trial count differs from its design.")
    outcome_kind_by_trial = {
        trial.trial_id: trial.fitted_analysis.outcome_kind
        for trial in rct_design.trials
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "longitudinal_source_anchored_recovery.csv": longitudinal.source_anchored_recovery,
        "longitudinal_treatment_recovery.csv": longitudinal.treatment_recovery,
    }
    output_paths = []
    for name, rows in tables.items():
        path = output_dir / name
        pd.DataFrame([row.model_dump(mode="json") for row in rows]).to_csv(
            path,
            index=False,
            lineterminator="\n",
        )
        output_paths.append(path)
    frames = {
        "longitudinal_linkage_curve.csv": _longitudinal_linkage_curve(longitudinal),
        "rct_linkage_curve.csv": _rct_linkage_curve(rct),
        "rct_cell_summaries.csv": _with_outcome_kind(
            rct.cell_summaries,
            outcome_kind_by_trial=outcome_kind_by_trial,
        ),
        "rct_mechanism_response.csv": _with_outcome_kind(
            rct.dose_responses,
            outcome_kind_by_trial=outcome_kind_by_trial,
        ),
        "rct_linkage_response.csv": _with_outcome_kind(
            rct.linkage_dose_responses,
            outcome_kind_by_trial=outcome_kind_by_trial,
        ),
    }
    for name, frame in frames.items():
        path = output_dir / name
        frame.to_csv(path, index=False, lineterminator="\n")
        output_paths.append(path)

    summary = {
        "schema_id": "trialagentbench.validation_results/v1",
        "rctbench": {
            "trials": rct.trials,
            "binary_trials": outcome_kinds.count("binary"),
            "continuous_trials": outcome_kinds.count("continuous"),
            "worlds": rct.worlds,
            "dose_levels": list(rct_design.dose_levels),
            "source_sized": True,
            "design_sha256": rct.design_sha256,
            "receipt_sha256": rct.receipt_sha256,
            "report_sha256": sha256_file(rct_report_path),
        },
        "longitudinal": {
            "trials": len(
                {row.trial_id for row in longitudinal.source_anchored_recovery}
            ),
            "worlds_per_trial": min(
                row.worlds for row in longitudinal.source_anchored_recovery
            ),
            "linkage_levels": list(
                longitudinal.linkage_dose_response[0].retention_levels
            ),
            "design_sha256": longitudinal.design_sha256,
            "receipt_sha256": longitudinal.receipt_sha256,
            "report_sha256": sha256_file(longitudinal_report_path),
        },
    }
    summary_path = output_dir / "qualification_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        """# External production qualification

This bundle is a compact projection of two independently verified
fit-generate-analyse qualifications. The RCT Bench evidence uses all ten
endpoint-capable trials at their observed analysis sizes and allocations,
including binary and continuous endpoints. The longitudinal evidence uses two
participant-level trials with repeated measurements.

The tables separate three questions:

1. **Marginal fidelity:** source-sized generated variables are compared with
   the observed trial variables in their native analysis populations.
   Outcome diagnostics report binary event-fraction differences and continuous
   mean differences in source standard deviations, plus outcome scale ratios.
2. **Dependence and analysis relevance:** progressive participant-linkage
   disruption holds arm-wise marginals fixed while measuring paired changes in
   correlation structure and fitted treatment analyses.
3. **Known-truth recoverability:** treatment, prognostic, and longitudinal
   trajectory mechanisms are varied over prespecified dose levels and
   independently re-estimated over repeated worlds with confidence intervals
   and coverage.

For continuous linear endpoints, common random numbers make the within-world
mechanism-response slope an algebraic identity. These rows are labelled
`is_algebraic_identity`; their residual is a coefficient-wiring check rather
than stochastic evidence. Binary response slopes and all repeated-world
operating characteristics remain stochastic.

`qualification_summary.json` binds the full report, design, and receipt
identities. The CSV files contain the compact trial-level and mechanism-level
results used for tables and figures; participant-level source data and
construction internals are not included.
""",
        encoding="utf-8",
    )
    output_paths.extend((summary_path, readme_path))
    write_external_artifact_manifest(output_dir)
    output_paths.append(output_dir / "artifact_manifest.json")
    return tuple(output_paths)


def _canonical_json_sha(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _with_outcome_kind(
    rows: Sequence[RctCellSummaryV1 | RctDoseResponseV1 | RctLinkageDoseResponseV1],
    *,
    outcome_kind_by_trial: Mapping[str, str],
) -> pd.DataFrame:
    records = []
    for row in rows:
        payload = row.model_dump(mode="json")
        outcome_kind = outcome_kind_by_trial[row.trial_id]
        payload["outcome_kind"] = outcome_kind
        if isinstance(row, RctDoseResponseV1):
            is_identity = outcome_kind == "continuous"
            payload["is_algebraic_identity"] = is_identity
            payload["identity_residual"] = row.mean_slope - 1.0 if is_identity else None
        records.append(payload)
    return pd.DataFrame(records)


def _rct_linkage_curve(report: RctQualificationReportV1) -> pd.DataFrame:
    rows = [
        row
        for row in report.estimates
        if row.mode
        in {
            "whole_subject",
            "linkage_75",
            "linkage_50",
            "linkage_25",
            "independent_marginal",
        }
    ]
    intact = {
        (row.trial_id, row.world_index): (
            row.treatment_estimate,
            row.treatment_standard_error,
        )
        for row in rows
        if row.linkage_retention == 1.0
    }
    records = []
    for row in rows:
        divergence = row.linkage_dependence_divergence
        if divergence is None:
            raise ValueError("RCT linkage estimate omits paired dependence divergence.")
        records.append(
            {
                "trial_id": row.trial_id,
                "world_index": row.world_index,
                "linkage_retention": row.linkage_retention,
                "linkage_disruption": 1.0 - row.linkage_retention,
                "dependence_divergence": divergence,
                "analysis_perturbation_in_intact_se": abs(
                    row.treatment_estimate - intact[(row.trial_id, row.world_index)][0]
                )
                / intact[(row.trial_id, row.world_index)][1],
            }
        )
    frame = pd.DataFrame(records)
    return _summarize_curve(
        frame,
        groups=("trial_id", "linkage_retention", "linkage_disruption"),
        values=("dependence_divergence", "analysis_perturbation_in_intact_se"),
    )


def _longitudinal_linkage_curve(
    report: LongitudinalQualificationReportV1,
) -> pd.DataFrame:
    retention = {
        "whole_subject": 1.0,
        "linkage_75": 0.75,
        "linkage_50": 0.5,
        "linkage_25": 0.25,
        "independent_marginal": 0.0,
    }
    seen = set()
    records = []
    for row in report.estimates:
        if row.mode not in retention:
            continue
        identity = (row.trial_id, row.world_index, row.mode)
        if identity in seen:
            continue
        seen.add(identity)
        records.append(
            {
                "trial_id": row.trial_id,
                "world_index": row.world_index,
                "linkage_retention": retention[row.mode],
                "adjacent_correlation": row.fingerprint.within_stratum_adjacent_correlation,
            }
        )
    return _summarize_curve(
        pd.DataFrame(records),
        groups=("trial_id", "linkage_retention"),
        values=("adjacent_correlation",),
    )


def _summarize_curve(
    frame: pd.DataFrame,
    *,
    groups: tuple[str, ...],
    values: tuple[str, ...],
) -> pd.DataFrame:
    records = []
    for keys, rows in frame.groupby(list(groups), sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(groups, key_values, strict=True))
        record["worlds"] = len(rows)
        for value in values:
            estimate = rows[value].to_numpy(dtype=float)
            if len(estimate) < 2 or not np.isfinite(estimate).all():
                raise ValueError(
                    f"Curve value {value!r} requires at least two finite worlds."
                )
            mean = float(np.mean(estimate))
            half_width = float(
                t.ppf(0.975, len(estimate) - 1)
                * np.std(estimate, ddof=1)
                / np.sqrt(len(estimate))
            )
            record[f"mean_{value}"] = mean
            record[f"{value}_ci_low"] = mean - half_width
            record[f"{value}_ci_high"] = mean + half_width
        records.append(record)
    return pd.DataFrame(records)


def main(argv: Sequence[str] | None = None) -> int:
    """Export compact qualification evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--longitudinal-report", type=Path, required=True)
    parser.add_argument("--rct-report", type=Path, required=True)
    parser.add_argument("--rct-design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    export_validation_results(
        longitudinal_report_path=args.longitudinal_report,
        rct_report_path=args.rct_report,
        rct_design_path=args.rct_design,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export_validation_results"]
