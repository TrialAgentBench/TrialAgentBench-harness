"""Audit semantic differences among matched TrialEval context surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict

from trialagentbench_harness.analysis.trialeval_release import load_evaluator_json_model
from trialagentbench_harness.contracts.release.trialeval_context_panels import (
    TrialEvalContextPanelManifestV1,
    TrialEvalContextPanelV1,
)
from trialagentbench_harness.io.json import write_json_model

ContextDeltaStatusV1 = Literal["pass", "fail"]
ContextTierV1 = Literal["C1", "C2", "C3", "C4", "C5"]

_CONTRASTS: tuple[tuple[str, ContextTierV1, ContextTierV1], ...] = (
    ("C1-C2", "C1", "C2"),
    ("C3-C4", "C3", "C4"),
    ("C3-C1", "C3", "C1"),
    ("C4-C2", "C4", "C2"),
    ("C5-C4", "C5", "C4"),
)
_SCIENTIFIC_METADATA = frozenset(
    {
        "protocol_summary.json",
        "endpoint_definition.json",
        "intercurrent_event_strategy.json",
        "ascertainment_model.json",
    }
)
_C5_MUTABLE_RAW_PATHS = frozenset(
    {
        "data/raw/endpoint_reports.parquet",
        "data/raw/randomization.parquet",
    }
)
_PREPARATION_CONTRASTS = frozenset({"C3-C1", "C4-C2"})
_ENDPOINT_RECONSTRUCTION_KEYS = frozenset({"adjudication_fields", "adjudication_rule", "misclassification_parameters"})


class TrialEvalContextArtifactDeltaV1(BaseModel):
    """One matched context contrast over exact participant archive members."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval.context_artifact_delta/v1"] = (
        "trialagentbench.trialeval.context_artifact_delta/v1"
    )
    panel_id: str
    contrast_id: str
    left_task_id: str
    right_task_id: str
    left_only_paths: tuple[str, ...]
    right_only_paths: tuple[str, ...]
    byte_equal_shared_paths: tuple[str, ...]
    semantically_equal_shared_paths: tuple[str, ...]
    changed_shared_paths: tuple[str, ...]
    status: ContextDeltaStatusV1
    findings: tuple[str, ...] = ()


class TrialEvalContextArtifactDeltaReportV1(BaseModel):
    """Complete context-semantic audit over all evaluator-declared panels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval.context_artifact_delta_report/v1"] = (
        "trialagentbench.trialeval.context_artifact_delta_report/v1"
    )
    public_zip: str
    evaluator_zip: str
    status: ContextDeltaStatusV1
    panel_count: int
    contrast_count: int
    failed_contrast_count: int
    rows: tuple[TrialEvalContextArtifactDeltaV1, ...]


def _item_bytes(public: ZipFile, *, task_id: str) -> dict[str, bytes]:
    prefix = f"items/{task_id}/"
    files = {
        member.removeprefix(prefix): public.read(member)
        for member in public.namelist()
        if member.startswith(prefix) and not member.endswith("/")
    }
    if not files:
        raise FileNotFoundError(f"Context panel task is absent from the participant archive: {task_id}")
    return files


def _normalize(value: object, *, task_id: str) -> object:
    if isinstance(value, dict):
        return {str(key): _normalize(item, task_id=task_id) for key, item in value.items() if str(key) != "checksum"}
    if isinstance(value, list):
        return [_normalize(item, task_id=task_id) for item in value]
    if isinstance(value, str):
        return value.replace(task_id, "<TASK_ID>")
    return value


def _preparation_invariant_json(path: str, value: object) -> object:
    if not isinstance(value, dict):
        return value
    projected = dict(value)
    if path == "endpoint_definition.json":
        endpoints = projected.get("endpoints")
        if not isinstance(endpoints, list):
            return value
        projected_endpoints: list[object] = []
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                projected_endpoints.append(endpoint)
                continue
            endpoint_projection = dict(endpoint)
            definition = endpoint_projection.get("definition")
            if isinstance(definition, dict):
                definition_projection = dict(definition)
                detection = definition_projection.get("detection")
                if isinstance(detection, dict):
                    retained = {
                        key: item for key, item in detection.items() if key not in _ENDPOINT_RECONSTRUCTION_KEYS
                    }
                    if retained:
                        definition_projection["detection"] = retained
                    else:
                        definition_projection.pop("detection", None)
                endpoint_projection["definition"] = definition_projection
            projected_endpoints.append(endpoint_projection)
        projected["endpoints"] = projected_endpoints
    elif path == "protocol_summary.json":
        stepped = projected.get("stepped_wedge_plan")
        if isinstance(stepped, dict):
            stepped_projection = dict(stepped)
            source = stepped_projection.get("source_rel_path")
            if source in {"data/ADSL.parquet", "data/raw/randomization.parquet"}:
                stepped_projection["source_rel_path"] = "<PREPARATION_SPECIFIC_DESIGN_SOURCE>"
            notes = stepped_projection.get("notes")
            if isinstance(notes, str):
                stepped_projection["notes"] = notes.replace(
                    "data/ADSL.parquet", "<PREPARATION_SPECIFIC_DESIGN_SOURCE>"
                ).replace("data/raw/randomization.parquet", "<PREPARATION_SPECIFIC_DESIGN_SOURCE>")
            projected["stepped_wedge_plan"] = stepped_projection
    return projected


def _semantically_equal(
    path: str,
    left: bytes,
    right: bytes,
    *,
    left_task: str,
    right_task: str,
    preparation_contrast: bool,
) -> bool:
    if path.endswith(".json"):
        try:
            left_value = json.loads(left)
            right_value = json.loads(right)
        except json.JSONDecodeError:
            return False
        left_normalized = _normalize(left_value, task_id=left_task)
        right_normalized = _normalize(right_value, task_id=right_task)
        if preparation_contrast:
            left_normalized = _preparation_invariant_json(path, left_normalized)
            right_normalized = _preparation_invariant_json(path, right_normalized)
        return left_normalized == right_normalized
    if path.endswith(".md"):
        return left.decode("utf-8").replace(left_task, "<TASK_ID>") == right.decode("utf-8").replace(
            right_task, "<TASK_ID>"
        )
    return False


def _contrast_row(
    *,
    public: ZipFile,
    panel: TrialEvalContextPanelV1,
    contrast_id: str,
    left_tier: ContextTierV1,
    right_tier: ContextTierV1,
) -> TrialEvalContextArtifactDeltaV1:
    tasks = {task.context_tier: task.task_id for task in panel.tasks}
    left_task = tasks[left_tier]
    right_task = tasks[right_tier]
    left = _item_bytes(public, task_id=left_task)
    right = _item_bytes(public, task_id=right_task)
    left_paths = set(left)
    right_paths = set(right)
    shared = left_paths & right_paths
    byte_equal = tuple(sorted(path for path in shared if left[path] == right[path]))
    semantic_equal = tuple(
        sorted(
            path
            for path in shared
            if left[path] != right[path]
            and _semantically_equal(
                path,
                left[path],
                right[path],
                left_task=left_task,
                right_task=right_task,
                preparation_contrast=contrast_id in _PREPARATION_CONTRASTS,
            )
        )
    )
    changed = tuple(sorted(shared - set(byte_equal) - set(semantic_equal)))
    findings: list[str] = []
    changed_data = {path for path in changed if path.startswith("data/")}
    left_data = {path for path in left_paths if path.startswith("data/")}
    right_data = {path for path in right_paths if path.startswith("data/")}
    if contrast_id in {"C1-C2", "C3-C4"} and (left_data != right_data or changed_data):
        findings.append("matched_preparation_data_drift")
    if contrast_id == "C5-C4":
        if not changed_data:
            findings.append("declared_defect_surface_has_no_raw_data_delta")
        if not changed_data <= _C5_MUTABLE_RAW_PATHS:
            findings.append("declared_defect_changes_undeclared_raw_domain")
        if "data_integrity_policy.json" not in left_paths - right_paths:
            findings.append("declared_defect_policy_not_isolated_to_c5")
    for path in sorted(_SCIENTIFIC_METADATA & shared):
        if path in changed:
            findings.append(f"shared_scientific_metadata_drift:{path}")
    if contrast_id == "C3-C1":
        if "analysis_plan.json" not in shared or "analysis_plan.json" in changed:
            findings.append("locked_sap_drift_between_analysis_ready_and_raw")
    status: ContextDeltaStatusV1 = "fail" if findings else "pass"
    return TrialEvalContextArtifactDeltaV1(
        panel_id=panel.panel_id,
        contrast_id=contrast_id,
        left_task_id=left_task,
        right_task_id=right_task,
        left_only_paths=tuple(sorted(left_paths - right_paths)),
        right_only_paths=tuple(sorted(right_paths - left_paths)),
        byte_equal_shared_paths=byte_equal,
        semantically_equal_shared_paths=semantic_equal,
        changed_shared_paths=changed,
        status=status,
        findings=tuple(findings),
    )


def validate_trialeval_context_artifact_deltas_v1(
    *, public_zip: Path, evaluator_zip: Path
) -> TrialEvalContextArtifactDeltaReportV1:
    """Validate exact artifact semantics for every complete C1-C5 panel."""

    panels = load_evaluator_json_model(
        evaluator_zip=evaluator_zip,
        member="grader/domains/context_panels.json",
        model=TrialEvalContextPanelManifestV1,
    )
    rows: list[TrialEvalContextArtifactDeltaV1] = []
    with ZipFile(public_zip) as public:
        for panel in panels.panels:
            tiers = {task.context_tier for task in panel.tasks}
            if tiers != {"C1", "C2", "C3", "C4", "C5"}:
                raise ValueError(f"Context panel must contain exactly C1-C5: {panel.panel_id}")
            for contrast_id, left_tier, right_tier in _CONTRASTS:
                rows.append(
                    _contrast_row(
                        public=public,
                        panel=panel,
                        contrast_id=contrast_id,
                        left_tier=left_tier,
                        right_tier=right_tier,
                    )
                )
    failed = sum(row.status == "fail" for row in rows)
    return TrialEvalContextArtifactDeltaReportV1(
        public_zip=public_zip.as_posix(),
        evaluator_zip=evaluator_zip.as_posix(),
        status="fail" if failed else "pass",
        panel_count=len(panels.panels),
        contrast_count=len(rows),
        failed_contrast_count=failed,
        rows=tuple(rows),
    )


def write_trialeval_context_artifact_deltas_v1(
    *, public_zip: Path, evaluator_zip: Path, output_path: Path
) -> TrialEvalContextArtifactDeltaReportV1:
    """Validate and write machine- and human-readable context reports."""

    report = validate_trialeval_context_artifact_deltas_v1(public_zip=public_zip, evaluator_zip=evaluator_zip)
    write_json_model(output_path, report)
    output_path.with_suffix(".md").write_text(
        render_trialeval_context_artifact_deltas_markdown_v1(report),
        encoding="utf-8",
    )
    return report


def render_trialeval_context_artifact_deltas_markdown_v1(
    report: TrialEvalContextArtifactDeltaReportV1,
) -> str:
    """Render exact matched-context archive differences."""

    lines = [
        "# TrialEvalBench matched-context artifact contrasts",
        "",
        f"Status: **{report.status}**. Panels: {report.panel_count}. "
        f"Contrasts: {report.contrast_count}. Failed contrasts: {report.failed_contrast_count}.",
        "",
        "Byte-equal files are unchanged. Semantically equal files differ only in task identity or declared "
        "data-preparation representation. Changed files contain the intervention-specific evidence delta.",
    ]
    for row in report.rows:
        lines.extend(
            (
                "",
                f"## {row.panel_id}: {row.contrast_id}",
                "",
                f"Tasks: `{row.left_task_id}` versus `{row.right_task_id}`. Status: **{row.status}**.",
                "",
                f"- Left only: {', '.join(f'`{path}`' for path in row.left_only_paths) or 'none'}",
                f"- Right only: {', '.join(f'`{path}`' for path in row.right_only_paths) or 'none'}",
                f"- Changed shared: {', '.join(f'`{path}`' for path in row.changed_shared_paths) or 'none'}",
                f"- Semantically equal shared: "
                f"{', '.join(f'`{path}`' for path in row.semantically_equal_shared_paths) or 'none'}",
                f"- Findings: {', '.join(f'`{finding}`' for finding in row.findings) or 'none'}",
            )
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "TrialEvalContextArtifactDeltaReportV1",
    "TrialEvalContextArtifactDeltaV1",
    "render_trialeval_context_artifact_deltas_markdown_v1",
    "validate_trialeval_context_artifact_deltas_v1",
    "write_trialeval_context_artifact_deltas_v1",
]
