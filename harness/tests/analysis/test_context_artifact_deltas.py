from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from trialagentbench_harness.analysis.context_artifact_deltas import (
    validate_trialeval_context_artifact_deltas_v1,
)
from trialagentbench_harness.io.json import read_json
from trialagentbench_harness.tools.validate.validate_trialeval_context_artifact_deltas import main


def _checksum(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _release_pair(
    tmp_path: Path,
    *,
    c4_raw: bytes = b"raw",
    c5_subjects: bytes = b"raw",
    preparation_specific_metadata: bool = False,
    endpoint_term_drift: bool = False,
    c3_reconstruction_sap_lane: bool = False,
) -> tuple[Path, Path]:
    public_zip = tmp_path / "public.zip"
    evaluator_zip = tmp_path / "evaluator.zip"
    tasks = [
        {"task_id": f"TASK-{tier}", "context_tier": tier, "variant_id": f"variant-{tier}"}
        for tier in ("C1", "C2", "C3", "C4", "C5")
    ]
    panel: dict[str, object] = {
        "panel_id": "panel-001",
        "item_id": "item-001",
        "design_tier": "D1",
        "design_subtype": "individual_randomized",
        "assumption_tier": "A1",
        "reference_signature_digest": "a" * 16,
        "tasks": tasks,
    }
    panel["checksum"] = _checksum(panel)
    manifest: dict[str, object] = {
        "schema_id": "trialagentbench.trialeval.context_panel_manifest/v1",
        "version": "v1",
        "panels": [panel],
    }
    manifest["checksum"] = _checksum(manifest)
    with ZipFile(evaluator_zip, "w") as evaluator:
        evaluator.writestr("grader/domains/context_panels.json", json.dumps(manifest))
    with ZipFile(public_zip, "w") as public:
        for tier in ("C1", "C2", "C3", "C4", "C5"):
            task_id = f"TASK-{tier}"
            prefix = f"items/{task_id}/"
            raw = tier in {"C3", "C4", "C5"}
            endpoint_definition: dict[str, object] = {
                "task_id": task_id,
                "endpoints": [
                    {
                        "endpoint_id": "death",
                        "term": "Mortality drift" if endpoint_term_drift and raw else "All-cause death",
                        "definition": {},
                    }
                ],
            }
            protocol_summary: dict[str, object] = {
                "task_id": task_id,
                "design_family": "stepped_wedge_cluster_rollout",
            }
            if preparation_specific_metadata:
                if raw:
                    endpoint_definition["endpoints"][0]["definition"] = {  # type: ignore[index]
                        "detection": {
                            "adjudication_rule": {"event_indicator": "declared source rule"},
                            "adjudication_fields": {"EVENT_WINDOW_START_DTC": "ISO-8601"},
                        }
                    }
                source = "data/raw/randomization.parquet" if raw else "data/ADSL.parquet"
                protocol_summary["stepped_wedge_plan"] = {
                    "source_rel_path": source,
                    "notes": f"Use `{source}` for the declared rollout sequence.",
                    "period_length_dy": 28.0,
                }
            for path, payload in {
                "protocol_summary.json": protocol_summary,
                "endpoint_definition.json": endpoint_definition,
                "intercurrent_event_strategy.json": {"task_id": task_id, "strategy": "treatment_policy"},
            }.items():
                public.writestr(prefix + path, json.dumps(payload))
            if tier in {"C1", "C3"}:
                lane_rules = [{"role": "primary", "effect_scale": "log_hr"}]
                multiplicity_strategy = "none"
                if tier == "C3" and c3_reconstruction_sap_lane:
                    lane_rules.insert(0, {"role": "reconstruction"})
                    multiplicity_strategy = "fixed_sequence"
                public.writestr(
                    prefix + "analysis_plan.json",
                    json.dumps(
                        {
                            "task_id": task_id,
                            "item_id": task_id,
                            "primary_estimand_id": "itt",
                            "lane_rules": lane_rules,
                            "multiplicity_strategy": multiplicity_strategy,
                            "checksum": tier.lower() * 32,
                        }
                    ),
                )
            if tier in {"C1", "C2"}:
                public.writestr(prefix + "data/ADSL.parquet", b"analysis")
            else:
                public.writestr(
                    prefix + "data/raw/subjects.parquet",
                    c5_subjects if tier == "C5" else (c4_raw if tier == "C4" else b"raw"),
                )
                public.writestr(
                    prefix + "data/raw/randomization.parquet",
                    b"defective-randomization" if tier == "C5" else b"randomization",
                )
            if tier == "C5":
                public.writestr(prefix + "data_integrity_policy.json", json.dumps({"entries": ["declared"]}))
    return public_zip, evaluator_zip


def test_context_artifact_deltas_accept_declared_context_semantics(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _release_pair(tmp_path)

    report = validate_trialeval_context_artifact_deltas_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    assert report.status == "pass"
    assert report.contrast_count == 5
    c3_c1 = next(row for row in report.rows if row.contrast_id == "C3-C1")
    assert "analysis_plan.json" in c3_c1.semantically_equal_shared_paths


def test_context_artifact_deltas_reject_c3_c4_raw_drift(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _release_pair(tmp_path, c4_raw=b"drift")

    report = validate_trialeval_context_artifact_deltas_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    row = next(row for row in report.rows if row.contrast_id == "C3-C4")
    assert row.status == "fail"
    assert "matched_preparation_data_drift" in row.findings


def test_context_artifact_deltas_accept_preparation_specific_instructions(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _release_pair(tmp_path, preparation_specific_metadata=True)

    report = validate_trialeval_context_artifact_deltas_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    assert report.status == "pass"
    c3_c1 = next(row for row in report.rows if row.contrast_id == "C3-C1")
    assert "endpoint_definition.json" in c3_c1.semantically_equal_shared_paths
    assert "protocol_summary.json" in c3_c1.semantically_equal_shared_paths


def test_context_artifact_deltas_reject_scientific_endpoint_drift(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _release_pair(
        tmp_path,
        preparation_specific_metadata=True,
        endpoint_term_drift=True,
    )

    report = validate_trialeval_context_artifact_deltas_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    row = next(candidate for candidate in report.rows if candidate.contrast_id == "C3-C1")
    assert row.status == "fail"
    assert "shared_scientific_metadata_drift:endpoint_definition.json" in row.findings


def test_context_artifact_deltas_reject_reconstruction_lane_in_locked_sap(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _release_pair(tmp_path, c3_reconstruction_sap_lane=True)

    report = validate_trialeval_context_artifact_deltas_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    row = next(candidate for candidate in report.rows if candidate.contrast_id == "C3-C1")
    assert row.status == "fail"
    assert "locked_sap_drift_between_analysis_ready_and_raw" in row.findings


def test_context_artifact_deltas_reject_undeclared_c5_domain_change(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _release_pair(tmp_path, c5_subjects=b"defective-subjects")

    report = validate_trialeval_context_artifact_deltas_v1(
        public_zip=public_zip,
        evaluator_zip=evaluator_zip,
    )

    row = next(row for row in report.rows if row.contrast_id == "C5-C4")
    assert row.status == "fail"
    assert "declared_defect_changes_undeclared_raw_domain" in row.findings


def test_context_artifact_delta_cli_writes_report(tmp_path: Path) -> None:
    public_zip, evaluator_zip = _release_pair(tmp_path)
    output = tmp_path / "context_deltas.json"

    rc = main(
        [
            "--public-zip",
            public_zip.as_posix(),
            "--evaluator-zip",
            evaluator_zip.as_posix(),
            "--output",
            output.as_posix(),
        ]
    )

    assert rc == 0
    assert read_json(output)["status"] == "pass"
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "# TrialEvalBench matched-context artifact contrasts" in markdown
    assert "## panel-001: C1-C2" in markdown
    assert "## panel-001: C5-C4" in markdown
    assert "`data_integrity_policy.json`" in markdown
