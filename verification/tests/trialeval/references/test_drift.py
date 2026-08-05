"""Tests for public-evidence reference-drift validation."""

from __future__ import annotations

import json
from pathlib import Path

from trialagentbench_validation.trialeval.references.drift import (
    validate_public_evidence_reference_drift_validation_v1,
    write_public_evidence_reference_drift_validation_artifacts_v1,
)


def test_public_evidence_reference_drift_validation_rejects_dispositioned_official_blocked_rows(
    tmp_path: Path,
) -> None:
    drift_path = tmp_path / "drift.jsonl"
    _write_rows(
        drift_path,
        [_row("no_drift"), _row("blocked_derivation_gap", has_disposition=True)],
    )

    report = validate_public_evidence_reference_drift_validation_v1(drift_path)

    assert report.status == "fail"
    assert report.no_drift_count == 1
    assert report.blocked_derivation_gap_count == 1
    assert report.official_scoreable_blocked_derivation_gap_count == 1
    assert report.required_primary_blocked_derivation_gap_count == 1
    assert report.findings[0].code == "official_scoreable_blocked_derivation_gap"


def test_public_evidence_reference_drift_validation_accepts_dispositioned_non_official_blocked_rows(
    tmp_path: Path,
) -> None:
    drift_path = tmp_path / "drift.jsonl"
    _write_rows(
        drift_path,
        [
            _row("no_drift"),
            _row(
                "blocked_derivation_gap",
                has_disposition=True,
                variant_role="sensitivity_only",
            ),
        ],
    )

    report = validate_public_evidence_reference_drift_validation_v1(drift_path)

    assert report.status == "pass"
    assert report.blocked_derivation_gap_count == 1
    assert report.official_scoreable_blocked_derivation_gap_count == 0
    assert report.non_official_blocked_derivation_gap_count == 1
    assert report.blocked_variant_role_counts == {"sensitivity_only": 1}
    assert report.findings == ()


def test_public_evidence_reference_drift_validation_rejects_blocked_rows_without_disposition(
    tmp_path: Path,
) -> None:
    drift_path = tmp_path / "drift.jsonl"
    _write_rows(drift_path, [_row("blocked_derivation_gap", has_disposition=False)])

    report = validate_public_evidence_reference_drift_validation_v1(drift_path)

    assert report.status == "fail"
    assert {finding.code for finding in report.findings} == {
        "official_scoreable_blocked_derivation_gap",
        "missing_disposition_for_blocked_derivation_gap",
    }


def test_public_evidence_reference_drift_validation_rejects_score_affecting_drift(
    tmp_path: Path,
) -> None:
    drift_path = tmp_path / "drift.jsonl"
    _write_rows(drift_path, [_row("supported_reference_mismatch")])

    report = validate_public_evidence_reference_drift_validation_v1(drift_path)

    assert report.status == "fail"
    assert report.findings[0].code == "score_affecting_or_unresolved_drift"


def test_public_evidence_reference_drift_validation_writes_artifacts(
    tmp_path: Path,
) -> None:
    drift_path = tmp_path / "drift.jsonl"
    _write_rows(drift_path, [_row("no_drift")])
    out_dir = tmp_path / "out"

    report = write_public_evidence_reference_drift_validation_artifacts_v1(
        drift_jsonl=drift_path, out_dir=out_dir
    )

    assert report.status == "pass"
    assert (
        json.loads(
            (
                out_dir / "public_evidence_reference_drift_validation_report.json"
            ).read_text()
        )["status"]
        == "pass"
    )
    assert (
        "Status: `pass`"
        in (
            out_dir / "public_evidence_reference_drift_validation_report.md"
        ).read_text()
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    classification: str,
    *,
    has_disposition: bool = False,
    variant_role: str = "required_primary",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "trialagentbench.public_evidence_reference_drift_disposition/v1",
        "task_id": "TASK0001",
        "item_id": "d1a1_rct_clean_01",
        "lane_id": "primary_numeric.v1",
        "route_reference_id": f"TASK0001:{classification}",
        "variant_role": variant_role,
        "route_family": "km",
        "estimator_method_id": "observed:km",
        "effect_scale": "risk_difference_tau",
        "answer_shape": "point",
        "identification_class": "point_identified",
        "classification": classification,
        "public_surface_shape": "public_table:ADSL.parquet|public_table:ADTTE.parquet",
        "message": "fixture",
    }
    if has_disposition:
        payload["disposition"] = "input_surface_insufficient_for_registered_calculator"
        payload["required_release_action"] = (
            "Expose the required public analysis table."
        )
    return payload
