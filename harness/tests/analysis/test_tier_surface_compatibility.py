from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from trialagentbench_harness.analysis.tier_surface_compatibility import (
    validate_trialeval_tier_surface_compatibility_v1,
    write_trialeval_tier_surface_compatibility_artifacts_v1,
)


def _write_release_zips(
    tmp_path: Path,
    *,
    entries: list[dict[str, object]],
    public_members: dict[str, str],
) -> tuple[Path, Path]:
    public_zip = tmp_path / "public.zip"
    evaluator_zip = tmp_path / "evaluator.zip"
    with ZipFile(public_zip, "w") as zf:
        for name, payload in public_members.items():
            zf.writestr(name, payload)
    with ZipFile(evaluator_zip, "w") as zf:
        zf.writestr("grader/item_index.json", json.dumps({"entries": entries}))
    return public_zip, evaluator_zip


def _entry(task_id: str, context_tier: str, reconstruction_row_count: int) -> dict[str, object]:
    factors = {
        "C1": ("analysis_ready", "locked_sap"),
        "C2": ("analysis_ready", "protocol_only"),
        "C3": ("raw_domains", "locked_sap"),
        "C4": ("raw_domains", "protocol_only"),
        "C5": ("raw_domains_declared_defect", "protocol_only"),
    }
    data_preparation, analysis_specification = factors[context_tier]
    return {
        "task_id": task_id,
        "item_id": f"{task_id}_item",
        "variant_id": f"{task_id}_variant",
        "design_tier": "D1",
        "assumption_tier": "A1",
        "context_tier": context_tier,
        "factors": {
            "context_configuration": context_tier,
            "data_preparation": data_preparation,
            "analysis_specification": analysis_specification,
        },
        "reconstruction_row_count": reconstruction_row_count,
    }


def test_trialeval_tier_surface_accepts_expected_context_surfaces(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_zips(
        tmp_path,
        entries=[
            _entry("task_c1", "C1", 0),
            _entry("task_c2", "C2", 0),
            _entry("task_c3", "C3", 12),
            _entry("task_c4", "C4", 13),
            _entry("task_c5", "C5", 14),
        ],
        public_members={
            "items/task_c1/task.json": "{}",
            "items/task_c1/data/ADSL.parquet": "analysis-ready",
            "items/task_c1/data/ADTTE.parquet": "analysis-ready",
            "items/task_c2/task.json": "{}",
            "items/task_c2/data/ADSL.parquet": "analysis-ready",
            "items/task_c2/data/ADTTE.parquet": "analysis-ready",
            "items/task_c3/task.json": "{}",
            "items/task_c3/data/raw/source.csv": "raw",
            "items/task_c4/task.json": "{}",
            "items/task_c4/data/raw/source.csv": "raw",
            "items/task_c5/task.json": "{}",
            "items/task_c5/data/raw/source.csv": "raw",
        },
    )

    report = validate_trialeval_tier_surface_compatibility_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    assert report.status == "pass"
    assert report.total_items == 5
    assert report.failed_items == 0
    assert {row.context_tier: row.total for row in report.summaries} == {
        "C1": 1,
        "C2": 1,
        "C3": 1,
        "C4": 1,
        "C5": 1,
    }


def test_trialeval_tier_surface_rejects_analysis_ready_context_without_analysis_frame(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_zips(
        tmp_path,
        entries=[_entry("task_c1", "C1", 0)],
        public_members={"items/task_c1/task.json": "{}"},
    )

    report = validate_trialeval_tier_surface_compatibility_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    assert report.status == "fail"
    assert report.rows[0].findings == ("analysis_ready_context_missing_analysis_frame",)


def test_trialeval_tier_surface_rejects_c1_reconstruction_substitution(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_zips(
        tmp_path,
        entries=[_entry("task_c1", "C1", 7)],
        public_members={
            "items/task_c1/task.json": "{}",
            "items/task_c1/data/ADSL.parquet": "analysis-ready",
            "items/task_c1/data/ADTTE.parquet": "analysis-ready",
            "items/task_c1/data/public_reconstruction/reconstruction.csv": "rows",
            "items/task_c1/data/raw/source.csv": "raw",
        },
    )

    report = validate_trialeval_tier_surface_compatibility_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    assert report.status == "fail"
    assert set(report.rows[0].findings) == {
        "analysis_ready_context_contains_public_reconstruction_surface",
        "analysis_ready_context_contains_raw_reconstruction_inputs",
        "analysis_ready_context_has_reconstruction_reference_rows",
    }


def test_trialeval_tier_surface_rejects_reconstruction_context_analysis_frame_substitute(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_zips(
        tmp_path,
        entries=[_entry("task_c3", "C3", 0)],
        public_members={
            "items/task_c3/task.json": "{}",
            "items/task_c3/data/ADSL.parquet": "analysis-ready",
            "items/task_c3/data/ADTTE.parquet": "analysis-ready",
        },
    )

    report = validate_trialeval_tier_surface_compatibility_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    assert report.status == "fail"
    assert set(report.rows[0].findings) == {
        "reconstruction_context_contains_analysis_frame_substitute",
        "reconstruction_context_missing_raw_reconstruction_inputs",
        "evaluator_missing_reconstruction_reference_rows",
    }


def test_trialeval_tier_surface_rejects_completed_reference_output_in_reconstruction_context(
    tmp_path: Path,
) -> None:
    """C3-C5 participant surfaces must not contain completed reconstruction output."""

    public_zip, evaluator_zip = _write_release_zips(
        tmp_path,
        entries=[_entry("task_c3", "C3", 10)],
        public_members={
            "items/task_c3/task.json": "{}",
            "items/task_c3/data/raw/source.csv": "raw",
            "items/task_c3/data/public_reconstruction/ADSL.parquet": "reference",
        },
    )

    report = validate_trialeval_tier_surface_compatibility_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    assert report.status == "fail"
    assert report.rows[0].findings == ("reconstruction_context_contains_completed_reference_output",)


def test_trialeval_tier_surface_rejects_hidden_public_member(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_zips(
        tmp_path,
        entries=[_entry("task_c2", "C2", 0)],
        public_members={
            "items/task_c2/task.json": "{}",
            "items/task_c2/data/ADSL.parquet": "analysis-ready",
            "items/task_c2/data/ADTTE.parquet": "analysis-ready",
            "items/task_c2/grader/solution.json": "{}",
        },
    )

    report = validate_trialeval_tier_surface_compatibility_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    assert report.status == "fail"
    assert report.rows[0].findings == ("public_zip_contains_hidden_or_grader_member",)


def test_trialeval_tier_surface_writes_json_and_markdown_artifacts(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_release_zips(
        tmp_path,
        entries=[_entry("task_c1", "C1", 0)],
        public_members={
            "items/task_c1/task.json": "{}",
            "items/task_c1/data/ADSL.parquet": "analysis-ready",
            "items/task_c1/data/ADTTE.parquet": "analysis-ready",
        },
    )
    out_dir = tmp_path / "report"

    report = write_trialeval_tier_surface_compatibility_artifacts_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
        out_dir=out_dir,
    )

    assert report.status == "pass"
    assert (out_dir / "trialeval_tier_surface_compatibility_report.json").exists()
    assert (
        (out_dir / "trialeval_tier_surface_compatibility_report.md")
        .read_text(encoding="utf-8")
        .startswith("# TrialEvalBench C-Tier Surface Compatibility Report")
    )
