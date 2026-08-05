"""Tests for public-policy-owned TrialDev safety identities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import trialagentbench_harness.trialdev.share.inspect as inspect_module
from trialagentbench_harness.trialdev.share.hashing import compute_sha256_hex
from trialagentbench_harness.trialdev.share.inspect import inspect_scenario_bundle_v1
from trialagentbench_harness.trialdev.share.io import write_json
from trialagentbench_harness.trialdev.share.safety_policy import serious_event_definitions_v1


def _write_policy(public_dir: Path, *, duplicate_columns: bool = False) -> Path:
    definitions = [
        {
            "endpoint_id": "critical_event",
            "event_column": "SAFETY_OCCURRED",
            "time_column": "SAFETY_DAY",
            "seriousness_column": "SAFETY_SERIOUS",
            "severity_column": "SAFETY_GRADE",
        },
        {
            "endpoint_id": "organ_event",
            "event_column": "SAFETY_OCCURRED" if duplicate_columns else "ORGAN_OCCURRED",
            "time_column": "ORGAN_DAY",
            "seriousness_column": "ORGAN_SERIOUS",
            "severity_column": "ORGAN_GRADE",
        },
    ]
    payload = {
        "schema_id": "trialdev_safety_decision_policy_v1",
        "scenario_id": "scenario_test",
        "serious_event_definitions": definitions,
    }
    payload["checksum"] = compute_sha256_hex(payload)
    public_dir.mkdir(parents=True)
    path = public_dir / "safety_decision_policy.json"
    write_json(path, payload)
    return path


def test_safety_identity_is_independent_of_column_naming_convention(tmp_path: Path) -> None:
    """Policy identities need not use AE prefixes or event-name suffixes."""

    _write_policy(tmp_path / "public")
    definitions = serious_event_definitions_v1(scenario_root=tmp_path)

    assert tuple(definition.event_column for definition in definitions) == (
        "SAFETY_OCCURRED",
        "ORGAN_OCCURRED",
    )


def test_safety_identity_rejects_checksum_drift(tmp_path: Path) -> None:
    """Materialization must not consume an altered safety policy."""

    path = _write_policy(tmp_path / "public")
    path.write_text(path.read_text().replace("SAFETY_DAY", "ALTERED_DAY"), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        serious_event_definitions_v1(scenario_root=tmp_path)


def test_safety_identity_rejects_column_aliasing(tmp_path: Path) -> None:
    """Two endpoint definitions cannot claim the same physical column."""

    _write_policy(tmp_path / "public", duplicate_columns=True)

    with pytest.raises(ValueError, match="unique endpoint and column identities"):
        serious_event_definitions_v1(scenario_root=tmp_path)


def test_scenario_inspection_uses_declared_safety_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inspection must not rediscover safety endpoints from column names."""

    monkeypatch.setattr(inspect_module, "validate_public_scenario_bundle_v1", lambda *, scenario_root: None)
    public = tmp_path / "public"
    _write_policy(public)
    write_json(public / "variable_catalog.json", {"variables": []})
    write_json(public / "candidate_drug_catalog.json", {"candidate_drugs": []})
    write_json(
        public / "endpoint_catalog.json",
        {
            "endpoints": [
                {
                    "endpoint_id": "terminal",
                    "kind": "efficacy",
                    "mechanism_role": "terminal",
                }
            ]
        },
    )
    write_json(
        public / "eval_contract.json",
        {
            "scenario_id": "scenario_test",
            "phase_modules": [
                {
                    "phase_id": phase_id,
                    "allowed_endpoint_ids": ["terminal"],
                    "allowed_follow_up_days": [horizon],
                    "allowed_enrollment_window_days": [horizon],
                    "allowed_site_count_budgets": [1],
                    "allowed_allocation_ratios": ["1:1"],
                    "max_sample_size": 2,
                    "allowed_treatment_discontinuation_strategies": ["treatment_policy"],
                    "allowed_interim_policies": ["fixed_final"],
                    "allowed_site_strategies": ["high_enrolling"],
                    "allowed_selection_objectives": ["benefit_risk"],
                }
                for phase_id, horizon in (("phase2", 90), ("phase3", 365))
            ],
        },
    )
    pd.DataFrame(
        {
            "SAFETY_OCCURRED": [1, 0],
            "SAFETY_DAY": [20.0, 90.0],
            "SAFETY_SERIOUS": [1, 0],
            "SAFETY_GRADE": [3, 0],
            "ORGAN_OCCURRED": [0, 1],
            "ORGAN_DAY": [90.0, 30.0],
            "ORGAN_SERIOUS": [0, 1],
            "ORGAN_GRADE": [0, 3],
            "TREATMENT": ["control", "drug_a"],
            "EFF_terminal_T": [90.0, 30.0],
            "EFF_terminal_E": [0, 1],
        }
    ).to_parquet(public / "observational_extract.parquet", index=False)

    summary = inspect_scenario_bundle_v1(scenario_root=tmp_path)

    assert summary["ae_event_rates_phase2_horizon"] == {
        "critical_event": 0.5,
        "organ_event": 0.5,
    }
