from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from trialagentbench_harness.tools.validate import validate_trialeval_tier_surface


def _write_minimal_zips(tmp_path: Path, *, valid: bool) -> tuple[Path, Path]:
    public_zip = tmp_path / "public.zip"
    evaluator_zip = tmp_path / "evaluator.zip"
    with ZipFile(public_zip, "w") as zf:
        zf.writestr("items/task_c1/task.json", "{}")
        if valid:
            zf.writestr("items/task_c1/data/ADSL.parquet", "analysis-ready")
            zf.writestr("items/task_c1/data/ADTTE.parquet", "analysis-ready")
    with ZipFile(evaluator_zip, "w") as zf:
        zf.writestr(
            "grader/item_index.json",
            json.dumps(
                {
                    "entries": [
                        {
                            "task_id": "task_c1",
                            "item_id": "task_c1_item",
                            "variant_id": "task_c1_variant",
                            "design_tier": "D1",
                            "assumption_tier": "A1",
                            "context_tier": "C1",
                            "factors": {
                                "context_configuration": "C1",
                                "data_preparation": "analysis_ready",
                                "analysis_specification": "locked_sap",
                            },
                            "reconstruction_row_count": 0,
                        }
                    ]
                }
            ),
        )
    return public_zip, evaluator_zip


def test_validate_trialeval_tier_surface_cli_writes_passing_report(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_minimal_zips(tmp_path, valid=True)
    out_dir = tmp_path / "out"

    exit_code = validate_trialeval_tier_surface.main(
        [
            "--public-zip",
            public_zip.as_posix(),
            "--evaluator-zip",
            evaluator_zip.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
        ]
    )

    assert exit_code == 0
    payload = json.loads((out_dir / "trialeval_tier_surface_compatibility_report.json").read_text())
    assert payload["status"] == "pass"


def test_validate_trialeval_tier_surface_cli_returns_failure_for_surface_drift(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _write_minimal_zips(tmp_path, valid=False)
    out_dir = tmp_path / "out"

    exit_code = validate_trialeval_tier_surface.main(
        [
            "--public-zip",
            public_zip.as_posix(),
            "--evaluator-zip",
            evaluator_zip.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
        ]
    )

    assert exit_code == 1
    payload = json.loads((out_dir / "trialeval_tier_surface_compatibility_report.json").read_text())
    assert payload["failed_items"] == 1
