"""Validate TrialEvalBench public tier surfaces against grader item metadata."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict

from trialagentbench_harness.analysis.trialeval_release import load_evaluator_item_index
from trialagentbench_harness.contracts.core.trialeval_factors import (
    TrialEvalAnalysisSpecificationV1,
    TrialEvalDataPreparationV1,
    TrialEvalEvidenceFactorsV1,
)
from trialagentbench_harness.io.json import write_json_model

ContextTierSurfaceStatusV1 = Literal["pass", "fail"]
ContextTierSurfaceFindingSeverityV1 = Literal["error", "warning"]


class TrialEvalTierSurfaceRowV1(BaseModel):
    """Per-item TrialEvalBench public-surface compatibility row."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval_tier_surface_row/v1"] = (
        "trialagentbench.trialeval_tier_surface_row/v1"
    )
    task_id: str
    item_id: str
    variant_id: str
    design_tier: str
    assumption_tier: str
    context_tier: str
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    has_analysis_frame: bool
    has_public_reconstruction: bool
    has_raw_reconstruction_inputs: bool
    has_hidden_or_grader_public_member: bool
    reconstruction_row_count: int
    status: ContextTierSurfaceStatusV1
    findings: tuple[str, ...] = ()


class TrialEvalTierSurfaceSummaryRowV1(BaseModel):
    """Grouped TrialEvalBench public-surface compatibility summary."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval_tier_surface_summary_row/v1"] = (
        "trialagentbench.trialeval_tier_surface_summary_row/v1"
    )
    context_tier: str
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    total: int
    passed: int
    failed: int
    analysis_frame_items: int
    public_reconstruction_items: int
    raw_reconstruction_items: int
    hidden_or_grader_leak_items: int


class TrialEvalTierSurfaceCompatibilityReportV1(BaseModel):
    """TrialEvalBench C-tier public-surface compatibility report."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.trialeval_tier_surface_compatibility_report/v1"] = (
        "trialagentbench.trialeval_tier_surface_compatibility_report/v1"
    )
    public_zip: str
    evaluator_zip: str
    status: ContextTierSurfaceStatusV1
    total_items: int
    failed_items: int
    rows: tuple[TrialEvalTierSurfaceRowV1, ...]
    summaries: tuple[TrialEvalTierSurfaceSummaryRowV1, ...]


def _coerce_int(value: object, *, default: int = 0) -> int:
    """Coerce JSON scalar counts to int with an explicit default."""

    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid integer counts.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Expected integer-valued float, got {value!r}.")
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Expected integer-compatible value, got {type(value).__name__}.")


def _item_members(public_members: set[str], task_id: str) -> tuple[str, ...]:
    prefix = f"items/{task_id}/"
    return tuple(member for member in public_members if member.startswith(prefix))


def _row_findings(
    *,
    factors: TrialEvalEvidenceFactorsV1,
    has_analysis_frame: bool,
    has_public_reconstruction: bool,
    has_raw_reconstruction_inputs: bool,
    has_hidden_or_grader_public_member: bool,
    reconstruction_row_count: int,
) -> tuple[str, ...]:
    findings: list[str] = []
    if factors.data_preparation == "analysis_ready":
        if not has_analysis_frame:
            findings.append("analysis_ready_context_missing_analysis_frame")
        if has_public_reconstruction:
            findings.append("analysis_ready_context_contains_public_reconstruction_surface")
        if has_raw_reconstruction_inputs:
            findings.append("analysis_ready_context_contains_raw_reconstruction_inputs")
        if reconstruction_row_count != 0:
            findings.append("analysis_ready_context_has_reconstruction_reference_rows")
    else:
        if has_analysis_frame:
            findings.append("reconstruction_context_contains_analysis_frame_substitute")
        if has_public_reconstruction:
            findings.append("reconstruction_context_contains_completed_reference_output")
        if not has_raw_reconstruction_inputs:
            findings.append("reconstruction_context_missing_raw_reconstruction_inputs")
        if reconstruction_row_count <= 0:
            findings.append("evaluator_missing_reconstruction_reference_rows")
    if has_hidden_or_grader_public_member:
        findings.append("public_zip_contains_hidden_or_grader_member")
    return tuple(findings)


def validate_trialeval_tier_surface_compatibility_v1(
    *,
    public_zip: Path,
    evaluator_zip: Path,
) -> TrialEvalTierSurfaceCompatibilityReportV1:
    """Validate C-tier public evidence surfaces for a TrialEvalBench release."""

    entries = load_evaluator_item_index(evaluator_zip)
    with ZipFile(public_zip) as zf:
        public_members = set(zf.namelist())
    rows: list[TrialEvalTierSurfaceRowV1] = []
    for entry in entries:
        task_id = str(entry.get("task_id") or "")
        members = _item_members(public_members, task_id)
        factor_payload = entry.get("factors")
        if not isinstance(factor_payload, dict):
            raise ValueError(f"Evaluator task {task_id!r} lacks explicit evidence factors.")
        factors = TrialEvalEvidenceFactorsV1.model_validate(
            {
                "context_configuration": factor_payload.get("context_configuration"),
                "data_preparation": factor_payload.get("data_preparation"),
                "analysis_specification": factor_payload.get("analysis_specification"),
            }
        )
        context_tier = factors.context_configuration
        has_analysis_frame = {
            f"items/{task_id}/data/ADSL.parquet",
            f"items/{task_id}/data/ADTTE.parquet",
        }.issubset(members)
        has_public_reconstruction = any(
            member.startswith(f"items/{task_id}/data/public_reconstruction/") for member in members
        )
        has_raw_reconstruction_inputs = any(member.startswith(f"items/{task_id}/data/raw/") for member in members)
        has_hidden_or_grader_public_member = any(
            "/hidden/" in member
            or member.startswith(f"items/{task_id}/hidden/")
            or member.startswith(f"items/{task_id}/grader/")
            for member in members
        )
        reconstruction_row_count = _coerce_int(entry.get("reconstruction_row_count"), default=0)
        findings = _row_findings(
            factors=factors,
            has_analysis_frame=has_analysis_frame,
            has_public_reconstruction=has_public_reconstruction,
            has_raw_reconstruction_inputs=has_raw_reconstruction_inputs,
            has_hidden_or_grader_public_member=has_hidden_or_grader_public_member,
            reconstruction_row_count=reconstruction_row_count,
        )
        rows.append(
            TrialEvalTierSurfaceRowV1(
                task_id=task_id,
                item_id=str(entry.get("item_id") or ""),
                variant_id=str(entry.get("variant_id") or ""),
                design_tier=str(factor_payload.get("design_archetype") or ""),
                assumption_tier=str(factor_payload.get("assumption_regime") or ""),
                context_tier=context_tier,
                data_preparation=factors.data_preparation,
                analysis_specification=factors.analysis_specification,
                has_analysis_frame=has_analysis_frame,
                has_public_reconstruction=has_public_reconstruction,
                has_raw_reconstruction_inputs=has_raw_reconstruction_inputs,
                has_hidden_or_grader_public_member=has_hidden_or_grader_public_member,
                reconstruction_row_count=reconstruction_row_count,
                status="fail" if findings else "pass",
                findings=findings,
            )
        )
    summaries = _summary_rows(rows)
    failed_items = sum(1 for row in rows if row.status == "fail")
    return TrialEvalTierSurfaceCompatibilityReportV1(
        public_zip=public_zip.as_posix(),
        evaluator_zip=evaluator_zip.as_posix(),
        status="fail" if failed_items else "pass",
        total_items=len(rows),
        failed_items=failed_items,
        rows=tuple(rows),
        summaries=summaries,
    )


def _summary_rows(rows: list[TrialEvalTierSurfaceRowV1]) -> tuple[TrialEvalTierSurfaceSummaryRowV1, ...]:
    by_context: dict[str, list[TrialEvalTierSurfaceRowV1]] = {}
    for row in rows:
        by_context.setdefault(row.context_tier, []).append(row)
    summaries: list[TrialEvalTierSurfaceSummaryRowV1] = []
    for context_tier, context_rows in sorted(by_context.items()):
        status_counts = Counter(row.status for row in context_rows)
        evidence_factor_pairs = {(row.data_preparation, row.analysis_specification) for row in context_rows}
        if len(evidence_factor_pairs) != 1:
            raise ValueError(f"Context configuration {context_tier!r} contains inconsistent evidence factors.")
        data_preparation, analysis_specification = next(iter(evidence_factor_pairs))
        summaries.append(
            TrialEvalTierSurfaceSummaryRowV1(
                context_tier=context_tier,
                data_preparation=data_preparation,
                analysis_specification=analysis_specification,
                total=len(context_rows),
                passed=status_counts["pass"],
                failed=status_counts["fail"],
                analysis_frame_items=sum(1 for row in context_rows if row.has_analysis_frame),
                public_reconstruction_items=sum(1 for row in context_rows if row.has_public_reconstruction),
                raw_reconstruction_items=sum(1 for row in context_rows if row.has_raw_reconstruction_inputs),
                hidden_or_grader_leak_items=sum(1 for row in context_rows if row.has_hidden_or_grader_public_member),
            )
        )
    return tuple(summaries)


def render_trialeval_tier_surface_compatibility_report_v1(
    report: TrialEvalTierSurfaceCompatibilityReportV1,
) -> str:
    """Render a Markdown TrialEvalBench tier-surface compatibility report."""

    lines = [
        "# TrialEvalBench C-Tier Surface Compatibility Report",
        "",
        f"- Public zip: `{report.public_zip}`",
        f"- Evaluator zip: `{report.evaluator_zip}`",
        f"- Status: `{report.status}`",
        f"- Total items: `{report.total_items}`",
        f"- Failed items: `{report.failed_items}`",
        "",
        "## Summary By Context",
        "",
        "| Context | Data preparation | Analysis specification | Total | Passed | Failed | Analysis frame | Public reconstruction | Raw reconstruction | Hidden/grader leaks |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.summaries:
        lines.append(
            f"| `{row.context_tier}` | `{row.data_preparation}` | `{row.analysis_specification}` | "
            f"{row.total} | {row.passed} | {row.failed} | "
            f"{row.analysis_frame_items} | {row.public_reconstruction_items} | "
            f"{row.raw_reconstruction_items} | {row.hidden_or_grader_leak_items} |"
        )
    failed = [row for row in report.rows if row.status == "fail"]
    if failed:
        lines.extend(["", "## Failed Items", ""])
        for failed_row in failed:
            lines.append(f"- `{failed_row.task_id}` `{failed_row.context_tier}`: {', '.join(failed_row.findings)}")
    return "\n".join(lines) + "\n"


def write_trialeval_tier_surface_compatibility_artifacts_v1(
    *,
    public_zip: Path,
    evaluator_zip: Path,
    out_dir: Path,
) -> TrialEvalTierSurfaceCompatibilityReportV1:
    """Validate and write TrialEvalBench C-tier surface compatibility artifacts."""

    report = validate_trialeval_tier_surface_compatibility_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_model(out_dir / "trialeval_tier_surface_compatibility_report.json", report)
    (out_dir / "trialeval_tier_surface_compatibility_report.md").write_text(
        render_trialeval_tier_surface_compatibility_report_v1(report),
        encoding="utf-8",
    )
    return report


__all__ = [
    "TrialEvalTierSurfaceCompatibilityReportV1",
    "TrialEvalTierSurfaceRowV1",
    "TrialEvalTierSurfaceSummaryRowV1",
    "render_trialeval_tier_surface_compatibility_report_v1",
    "validate_trialeval_tier_surface_compatibility_v1",
    "write_trialeval_tier_surface_compatibility_artifacts_v1",
]
