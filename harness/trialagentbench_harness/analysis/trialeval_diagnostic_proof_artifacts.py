"""Artifact writers for TrialEvalBench diagnostic proof surfaces."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import cast

from trialagentbench_harness.analysis.trialeval_diagnostic_proof_surface import (
    PH_METHOD_CHANGE_TIME_VARIATION_THRESHOLD,
    validate_trialeval_diagnostic_proof_surface_v1,
)
from trialagentbench_harness.contracts.analysis.diagnostic_proof import (
    TrialEvalDiagnosticProofReportV1,
    TrialEvalDiagnosticProofRowV1,
)
from trialagentbench_harness.execution_policy import TRIALEVAL_DIAGNOSTIC_PROOF_DEFAULT_WORKERS
from trialagentbench_harness.io.csv import write_csv_rows
from trialagentbench_harness.io.json import write_json_model


def _row_to_csv(row: TrialEvalDiagnosticProofRowV1) -> dict[str, object]:
    payload = cast(dict[str, object], row.model_dump(mode="json"))
    for key in (
        "inferred_route_families",
        "credit_eligible_route_families",
        "intended_assumption_statuses",
        "required_diagnostic_keys",
        "satisfied_diagnostic_keys",
        "missing_diagnostic_keys",
        "resolved_warning_keys",
        "assumption_replays",
        "public_input_paths",
        "public_input_hashes",
        "proof_surface_source_rows",
    ):
        payload[key] = _join_sequence_field(payload, key)
    payload["findings"] = _join_sequence_field(payload, "findings")
    return payload


def _join_sequence_field(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if isinstance(value, list | tuple):
        return ";".join(str(element) for element in value)
    return "" if value is None else str(value)


def _coverage_rows(rows: tuple[TrialEvalDiagnosticProofRowV1, ...]) -> list[dict[str, object]]:
    coverage: Counter[tuple[str, str, str, str, str]] = Counter()
    passed: Counter[tuple[str, str, str, str, str]] = Counter()
    for row in rows:
        key = (row.design_tier, row.assumption_tier, row.context_tier, row.evidence_class, row.design_family)
        coverage[key] += 1
        if row.status == "pass":
            passed[key] += 1
    return [
        {
            "design_tier": key[0],
            "assumption_tier": key[1],
            "context_tier": key[2],
            "evidence_class": key[3],
            "design_family": key[4],
            "total": total,
            "passed": passed[key],
            "failed": total - passed[key],
        }
        for key, total in sorted(coverage.items())
    ]


def _assumption_id_for_diagnostic_key(key: str) -> str:
    mapping = {
        "randomization_integrity_public": "randomization_integrity",
        "censoring_followup_public": "censoring_ignorability",
        "proportional_hazards_public": "proportional_hazards",
        "model_form_public": "model_form",
        "endpoint_ascertainment_public": "endpoint_ascertainment",
        "cluster_structure_public": "cluster_structure",
        "secular_trend_public": "secular_trend",
        "sequential_design_adjustment_public": "sequential_design_adjustment",
    }
    return mapping.get(key, key.removesuffix("_public"))


def _assumption_coverage_rows(rows: tuple[TrialEvalDiagnosticProofRowV1, ...]) -> list[dict[str, object]]:
    coverage: Counter[tuple[str, str, str]] = Counter()
    satisfied: Counter[tuple[str, str, str]] = Counter()
    missing: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        status_by_assumption = {
            pair.split("=", 1)[0]: pair.split("=", 1)[1] for pair in row.intended_assumption_statuses if "=" in pair
        }
        satisfied_keys = set(row.satisfied_diagnostic_keys)
        for diagnostic_key in row.required_diagnostic_keys:
            assumption_id = _assumption_id_for_diagnostic_key(diagnostic_key)
            counter_key = (assumption_id, status_by_assumption.get(assumption_id, "unknown"), diagnostic_key)
            coverage[counter_key] += 1
            if diagnostic_key in satisfied_keys:
                satisfied[counter_key] += 1
            else:
                missing[counter_key] += 1
    return [
        {
            "assumption_id": key[0],
            "assumption_status": key[1],
            "diagnostic_key": key[2],
            "total": total,
            "satisfied": satisfied[key],
            "missing": missing[key],
        }
        for key, total in sorted(coverage.items())
    ]


def _assumption_replay_rows(rows: tuple[TrialEvalDiagnosticProofRowV1, ...]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        for replay in row.assumption_replays:
            replay_payload = replay.model_dump(mode="json")
            decision_metric_names = replay_payload.pop("decision_metric_names")
            participant_values = replay_payload.pop("participant_decision_metric_values")
            evaluator_values = replay_payload.pop("evaluator_decision_metric_values")
            output.append(
                {
                    "task_id": row.task_id,
                    "item_id": row.item_id,
                    "design_tier": row.design_tier,
                    "assumption_tier": row.assumption_tier,
                    "context_tier": row.context_tier,
                    "design_family": row.design_family,
                    **replay_payload,
                    **{
                        f"{threshold_id}_decision_metric_name": decision_metric_names[threshold_id]
                        for threshold_id in ("stressed", "fragile", "broken")
                    },
                    **{
                        f"participant_{threshold_id}_decision_metric": participant_values[threshold_id]
                        for threshold_id in ("stressed", "fragile", "broken")
                    },
                    **{
                        f"evaluator_{threshold_id}_decision_metric": evaluator_values[threshold_id]
                        for threshold_id in ("stressed", "fragile", "broken")
                    },
                    "item_status": row.status,
                }
            )
    return output


def _ph_panel_rows(rows: tuple[TrialEvalDiagnosticProofRowV1, ...]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if not row.ph_diagnostic_required:
            continue
        output.append(
            {
                "task_id": row.task_id,
                "item_id": row.item_id,
                "design_tier": row.design_tier,
                "assumption_tier": row.assumption_tier,
                "context_tier": row.context_tier,
                "evidence_class": row.evidence_class,
                "n_subjects": row.n_subjects,
                "n_events": row.n_events,
                "schoenfeld_p_value": row.schoenfeld_p_value,
                "neg_log10_schoenfeld_p": row.neg_log10_schoenfeld_p,
                "scaled_schoenfeld_rank_slope": row.scaled_schoenfeld_rank_slope,
                "scaled_schoenfeld_rank_slope_standard_error": row.scaled_schoenfeld_rank_slope_standard_error,
                "simultaneous_lower_abs_time_varying_log_hazard_range": (
                    row.simultaneous_lower_abs_time_varying_log_hazard_range
                ),
                "ph_method_change_threshold_crossed": row.ph_method_change_threshold_crossed,
                "method_applicability_decision": row.method_applicability_decision,
                "resolved_warning_keys": ";".join(row.resolved_warning_keys),
                "status": row.status,
                "findings": ";".join(row.findings),
            }
        )
    return output


def _write_svg_ph_panel(path: Path, rows: tuple[TrialEvalDiagnosticProofRowV1, ...]) -> None:
    ph_rows = [
        row
        for row in rows
        if row.ph_diagnostic_required and row.simultaneous_lower_abs_time_varying_log_hazard_range is not None
    ]
    width, height = 900, 420
    margin = 60
    lower_ranges = [
        float(row.simultaneous_lower_abs_time_varying_log_hazard_range)
        for row in ph_rows
        if row.simultaneous_lower_abs_time_varying_log_hazard_range is not None
    ]
    max_y = max(lower_ranges, default=1.0)
    max_y = max(max_y, PH_METHOD_CHANGE_TIME_VARIATION_THRESHOLD * 1.25)
    colors = {"A1": "#537188", "A2": "#9a8f4f", "A3": "#c56b36", "A4": "#9f2f2f"}
    circles: list[str] = []
    for index, row in enumerate(ph_rows):
        lower_range = row.simultaneous_lower_abs_time_varying_log_hazard_range
        if lower_range is None:
            raise ValueError("PH panel rows require an estimable time-varying effect range.")
        x = margin + (width - 2 * margin) * (index / max(len(ph_rows) - 1, 1))
        y = height - margin - (height - 2 * margin) * (float(lower_range) / max_y)
        radius = 5.5
        color = colors.get(row.assumption_tier, "#555555")
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" opacity="0.72">'
            f"<title>{row.task_id} {row.assumption_tier} slope={row.scaled_schoenfeld_rank_slope} "
            f"se={row.scaled_schoenfeld_rank_slope_standard_error}</title></circle>"
        )
    threshold_y = height - margin - (height - 2 * margin) * (PH_METHOD_CHANGE_TIME_VARIATION_THRESHOLD / max_y)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{threshold_y:.1f}" x2="{width - margin}" y2="{threshold_y:.1f}" '
        'stroke="#b23a48" stroke-dasharray="6 5"/>',
        f'<text x="{margin}" y="{margin - 20}" font-family="sans-serif" font-size="16" fill="#111">'
        "Public PH diagnostic proof surface: lower 95% absolute time-varying log-HR range</text>",
        f'<text x="{margin + 8}" y="{threshold_y - 8:.1f}" font-family="sans-serif" font-size="12" fill="#b23a48">'
        "HR-ratio 1.25 stressed threshold</text>",
        *circles,
        "</svg>",
    ]
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def render_trialeval_diagnostic_proof_report_v1(report: TrialEvalDiagnosticProofReportV1) -> str:
    """Render a Markdown diagnostic proof-surface report."""

    lines = [
        "# TrialEvalBench Diagnostic Proof-Surface Report",
        "",
        f"- Public zip: `{report.public_zip}`",
        f"- Evaluator zip: `{report.evaluator_zip}`",
        f"- Status: `{report.status}`",
        f"- Total items: `{report.total_items}`",
        f"- Failed items: `{report.failed_items}`",
        f"- Items with resolved warnings: `{report.warning_items}`",
        f"- Items with unresolved warnings: `{report.unresolved_warning_items}`",
        f"- PH diagnostic rows: `{report.ph_diagnostic_rows}`",
        f"- PH method-change threshold rows: `{report.ph_method_change_rows}`",
        f"- Non-PH diagnostic rows: `{report.non_ph_diagnostic_rows}`",
        f"- Design-adjustment rows: `{report.design_adjustment_rows}`",
        "",
        "## Summary",
        "",
        "| Group | Value | Total | Passed | Failed | Retain official | Exclude/downgrade | Block release |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary_row in report.summaries:
        lines.append(
            f"| `{summary_row.group_kind}` | `{summary_row.group_value}` | {summary_row.total} | "
            f"{summary_row.passed} | {summary_row.failed} | {summary_row.retain_official} | "
            f"{summary_row.exclude_or_downgrade} | {summary_row.block_release} |"
        )
    failed = [row for row in report.rows if row.status == "fail"]
    if failed:
        lines.extend(["", "## Failed Items", ""])
        for failed_row in failed[:100]:
            lines.append(
                f"- `{failed_row.task_id}` `{failed_row.evidence_class}` `{failed_row.assumption_tier}`: "
                f"{', '.join(failed_row.findings)}"
            )
    return "\n".join(lines) + "\n"


def write_trialeval_diagnostic_proof_surface_artifacts_v1(
    *,
    public_zip: Path,
    evaluator_zip: Path,
    out_dir: Path,
    workers: int = TRIALEVAL_DIAGNOSTIC_PROOF_DEFAULT_WORKERS,
) -> TrialEvalDiagnosticProofReportV1:
    """Validate and write diagnostic proof-surface artifacts."""

    report = validate_trialeval_diagnostic_proof_surface_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
        workers=workers,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_model(out_dir / "trialeval_diagnostic_proof_report.json", report)
    (out_dir / "trialeval_diagnostic_proof_report.md").write_text(
        render_trialeval_diagnostic_proof_report_v1(report),
        encoding="utf-8",
    )
    row_csv = [_row_to_csv(row) for row in report.rows]
    write_csv_rows(out_dir / "trialeval_diagnostic_proof_rows.csv", row_csv, list(row_csv[0]) if row_csv else [])
    coverage = _coverage_rows(report.rows)
    write_csv_rows(
        out_dir / "trialeval_diagnostic_coverage_by_tier.csv",
        coverage,
        [
            "design_tier",
            "assumption_tier",
            "context_tier",
            "evidence_class",
            "design_family",
            "total",
            "passed",
            "failed",
        ],
    )
    assumption_coverage = _assumption_coverage_rows(report.rows)
    write_csv_rows(
        out_dir / "trialeval_assumption_diagnostic_coverage.csv",
        assumption_coverage,
        [
            "assumption_id",
            "assumption_status",
            "diagnostic_key",
            "total",
            "satisfied",
            "missing",
        ],
    )
    assumption_replays = _assumption_replay_rows(report.rows)
    write_csv_rows(
        out_dir / "trialeval_assumption_replay_rows.csv",
        assumption_replays,
        [
            "task_id",
            "item_id",
            "design_tier",
            "assumption_tier",
            "context_tier",
            "design_family",
            "assumption_id",
            "diagnosability",
            "severity_metric_name",
            "participant_severity_metric",
            "evaluator_severity_metric",
            "threshold_stressed",
            "threshold_fragile",
            "threshold_broken",
            "stressed_decision_metric_name",
            "fragile_decision_metric_name",
            "broken_decision_metric_name",
            "participant_stressed_decision_metric",
            "participant_fragile_decision_metric",
            "participant_broken_decision_metric",
            "evaluator_stressed_decision_metric",
            "evaluator_fragile_decision_metric",
            "evaluator_broken_decision_metric",
            "participant_band",
            "evaluator_band",
            "classification_applicable",
            "numeric_agreement",
            "classification_agreement",
            "nearest_threshold_margin",
            "item_status",
        ],
    )
    ph_panel = _ph_panel_rows(report.rows)
    write_csv_rows(
        out_dir / "trialeval_ph_diagnostic_panel_source.csv",
        ph_panel,
        [
            "task_id",
            "item_id",
            "design_tier",
            "assumption_tier",
            "context_tier",
            "evidence_class",
            "n_subjects",
            "n_events",
            "schoenfeld_p_value",
            "neg_log10_schoenfeld_p",
            "scaled_schoenfeld_rank_slope",
            "scaled_schoenfeld_rank_slope_standard_error",
            "simultaneous_lower_abs_time_varying_log_hazard_range",
            "ph_method_change_threshold_crossed",
            "method_applicability_decision",
            "resolved_warning_keys",
            "status",
            "findings",
        ],
    )
    _write_svg_ph_panel(out_dir / "trialeval_ph_diagnostic_panel.svg", report.rows)
    return report


def validate_existing_trialeval_diagnostic_proof_surface_v1(out_dir: Path) -> dict[str, object]:
    """Validate existing diagnostic proof-surface artifacts."""

    required = (
        "trialeval_diagnostic_proof_report.json",
        "trialeval_diagnostic_proof_report.md",
        "trialeval_diagnostic_proof_rows.csv",
        "trialeval_diagnostic_coverage_by_tier.csv",
        "trialeval_assumption_diagnostic_coverage.csv",
        "trialeval_assumption_replay_rows.csv",
        "trialeval_ph_diagnostic_panel_source.csv",
        "trialeval_ph_diagnostic_panel.svg",
    )
    findings: list[dict[str, str]] = []
    for name in required:
        if not (out_dir / name).is_file():
            findings.append({"code": "missing_diagnostic_proof_output", "path": (out_dir / name).as_posix()})
    if findings:
        return {
            "schema_id": "trialagentbench.trialeval_diagnostic_proof_existing_validation/v1",
            "status": "fail",
            "findings": findings,
        }
    payload = json.loads((out_dir / "trialeval_diagnostic_proof_report.json").read_text(encoding="utf-8"))
    report = TrialEvalDiagnosticProofReportV1.model_validate(payload)
    if report.status != "pass":
        findings.append({"code": "diagnostic_proof_report_not_pass", "path": out_dir.as_posix()})
    if report.total_items != 400:
        findings.append({"code": "diagnostic_proof_item_count_drift", "path": out_dir.as_posix()})
    if report.failed_items != 0:
        findings.append({"code": "diagnostic_proof_failures_present", "path": out_dir.as_posix()})
    if report.unresolved_warning_items != 0:
        findings.append({"code": "diagnostic_proof_unresolved_warnings_present", "path": out_dir.as_posix()})
    if report.source_table_count < 5:
        findings.append({"code": "diagnostic_proof_source_table_count_stale", "path": out_dir.as_posix()})
    return {
        "schema_id": "trialagentbench.trialeval_diagnostic_proof_existing_validation/v1",
        "status": "pass" if not findings else "fail",
        "total_items": report.total_items,
        "failed_items": report.failed_items,
        "warning_items": report.warning_items,
        "unresolved_warning_items": report.unresolved_warning_items,
        "ph_diagnostic_rows": report.ph_diagnostic_rows,
        "figure_source_count": report.figure_source_count,
        "findings": findings,
    }


__all__ = [
    "render_trialeval_diagnostic_proof_report_v1",
    "validate_existing_trialeval_diagnostic_proof_surface_v1",
    "write_trialeval_diagnostic_proof_surface_artifacts_v1",
]
