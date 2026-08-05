"""Tests for evaluator-owned targeted-control applicability export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trialagentbench_harness.experiments.export_trialeval_ablation_labels import (
    build_trialeval_ablation_evaluator_labels_v1,
)


def _assumption_manifest(*, task_id: str, status: str) -> dict[str, object]:
    band = "holds" if status == "holds" else "fragile"
    payload: dict[str, object] = {
        "version": "v1",
        "schema_id": "trial_benchmark_assumption_evidence_manifest_v1",
        "item_id": f"item-{task_id}",
        "base_case_id": f"case-{task_id}",
        "canonical_item_id": f"item-{task_id}",
        "variant_id": f"case-{task_id}__C1",
        "context_tier": "C1",
        "replicate_index": 0,
        "records": [
            {
                "assumption_id": "proportional_hazards",
                "expected_status": status,
                "computed_status": status,
                "expected_band": band,
                "computed_band": band,
                "diagnosability": "partially_diagnosable",
                "severity_metric": 0.0 if status == "holds" else 0.5,
                "severity_metric_name": "simultaneous_lower_abs_time_varying_log_hazard_range",
                "threshold_stressed": 0.1,
                "threshold_fragile": 0.4,
                "threshold_broken": 0.8,
                "decision_metric_names": {
                    "stressed": "simultaneous_lower_abs_time_varying_log_hazard_range",
                    "fragile": "simultaneous_lower_abs_time_varying_log_hazard_range",
                    "broken": "simultaneous_lower_abs_time_varying_log_hazard_range",
                },
                "supporting_metrics": {},
                "metric_units": {
                    "simultaneous_lower_abs_time_varying_log_hazard_range": "log_hazard_ratio",
                },
                "metric_public_evidence_basis": {
                    "simultaneous_lower_abs_time_varying_log_hazard_range": ["data/ADTTE.parquet"],
                },
                "notes": [],
            }
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    payload["checksum"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _evaluator_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    domains = root / "grader" / "domains"
    domains.mkdir(parents=True)
    cases = (
        ("TASK1001", "D3", 0, "holds"),
        ("TASK1002", "D1", 0, "stressed"),
        ("TASK1003", "D4", 0, "holds"),
        ("TASK1004", "D1", 2, "holds"),
        ("TASK1005", "D1", 0, "holds"),
    )
    entries = [
        {
            "task_id": task_id,
            "item_id": f"item-{task_id}",
            "base_case_id": f"case-{task_id}",
            "factors": {
                "evaluation_series_id": {
                    "D1": "individual_randomized",
                    "D3": "covariate_structure",
                    "D4": "cluster_randomized",
                }[design_tier],
                "design_archetype": design_tier,
                "design_subtype": {
                    "D1": "individual_randomized",
                    "D3": "covariate_structure",
                    "D4": "cluster_parallel",
                }[design_tier],
                "assumption_regime": "A1",
                "context_configuration": "C1",
                "data_preparation": "analysis_ready",
                "analysis_specification": "locked_sap",
                "procedure_assistance": "output_contract_only",
                "response_interface": "structured",
            },
            "data_integrity_reference_row_count": defect_count,
            "data_integrity_reference_row_offset": 0,
            "reconstruction_row_count": 0,
            "reconstruction_row_offset": 0,
            "scoring_row_count": 1,
            "scoring_row_offset": index,
            "variant_id": f"case-{task_id}__C1",
        }
        for index, (task_id, design_tier, defect_count, _) in enumerate(cases)
    ]
    (root / "grader" / "item_index.json").write_text(
        json.dumps(
            {
                "schema_id": "trialagentbench.trial_benchmark.grader_item_index/v1",
                "version": "v1",
                "checksum": "0" * 64,
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for task_id, _, _, status in cases:
        rows.append(
            {
                "schema_id": "trialagentbench.trial_benchmark.grader_domain_row/v1",
                "domain": "assumption_evidence",
                "task_id": task_id,
                "payload": {
                    "schema_id": "trialagentbench.trial_benchmark.assumption_evidence_manifest_row/v1",
                    "manifest": _assumption_manifest(task_id=task_id, status=status),
                },
            }
        )
    (domains / "assumption_evidence.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root


def test_export_derives_targeted_labels_from_frozen_evaluator_facts(tmp_path: Path) -> None:
    artifact = build_trialeval_ablation_evaluator_labels_v1(evaluator_root=_evaluator_release(tmp_path))
    labels = {(row.task_id, row.prompt_condition): row for row in artifact.labels}

    assert labels[("TASK1001", "targeted_covariate_structure")].applicability == "applicable"
    assert labels[("TASK1002", "targeted_survival_assumptions")].applicability == "applicable"
    assert labels[("TASK1003", "targeted_design_structure")].applicability == "applicable"
    assert labels[("TASK1004", "targeted_data_integrity")].applicability == "applicable"
    assert {
        labels[("TASK1005", condition)].applicability
        for condition in (
            "targeted_covariate_structure",
            "targeted_survival_assumptions",
            "targeted_design_structure",
            "targeted_data_integrity",
        )
    } == {"inapplicable"}
    assert labels[("TASK1001", "targeted_design_structure")].applicability == "mismatched"
    assert labels[("TASK1002", "targeted_survival_assumptions")].evidence_basis == (
        "assumption_evidence.proportional_hazards.computed_status!=holds",
    )
    identities = {row.task_id: row for row in artifact.task_identities}
    assert identities["TASK1001"].regime_cell_id == "case-TASK1001"
    assert identities["TASK1001"].design_tier == "D3"
    assert identities["TASK1001"].design_subtype == "covariate_structure"
    assert identities["TASK1001"].assumption_tier == "A1"


def test_export_rejects_incomplete_assumption_evidence_coverage(tmp_path: Path) -> None:
    root = _evaluator_release(tmp_path)
    path = root / "grader" / "domains" / "assumption_evidence.jsonl"
    path.write_text("\n".join(path.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="coverage differs"):
        build_trialeval_ablation_evaluator_labels_v1(evaluator_root=root)


def test_export_rejects_missing_scientific_stratum(tmp_path: Path) -> None:
    root = _evaluator_release(tmp_path)
    path = root / "grader" / "item_index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["entries"][0]["factors"]["design_subtype"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="design_subtype"):
        build_trialeval_ablation_evaluator_labels_v1(evaluator_root=root)
