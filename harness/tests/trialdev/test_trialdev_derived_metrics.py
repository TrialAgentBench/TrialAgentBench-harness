from __future__ import annotations

import json
from pathlib import Path

import pytest

from trialagentbench_harness.trialdev.derived_metrics import _recoverability_for, _reference_ranking_for


def _write_scenario(tmp_path: Path, *, value: object = 0.4) -> Path:
    scenario_root = tmp_path / "scenario_s01"
    public = scenario_root / "public"
    grader = scenario_root / "grader"
    public.mkdir(parents=True)
    grader.mkdir()
    (public / "candidate_drug_catalog.json").write_text(
        json.dumps(
            {
                "candidate_drugs": [
                    {"candidate_drug_id": "usual_care", "role": "control"},
                    {"candidate_drug_id": "control", "role": "investigational"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (grader / "drug_ranking_reference_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "phase_id": "observational_review",
                        "objective_id": "benefit_risk",
                        "metric": "objective_score",
                        "candidate_drug_ids": ["usual_care"],
                        "value": 0.8,
                    },
                    {
                        "phase_id": "observational_review",
                        "objective_id": "benefit_risk",
                        "metric": "objective_score",
                        "candidate_drug_ids": ["control"],
                        "value": value,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_candidate_only_ranking_uses_catalog_roles_not_identifier_spelling(tmp_path: Path) -> None:
    bundle_root = _write_scenario(tmp_path)

    assert _reference_ranking_for(
        bundle_root,
        scenario_id="s01",
        phase_id="observational_review",
        objective_id="benefit_risk",
    ) == [("control", 0.4)]


@pytest.mark.parametrize("value", [None, True, "0.4", float("nan")])
def test_ranking_reference_rejects_nonfinite_or_coerced_values(tmp_path: Path, value: object) -> None:
    bundle_root = _write_scenario(tmp_path, value=value)

    with pytest.raises(ValueError, match="finite numeric value"):
        _reference_ranking_for(
            bundle_root,
            scenario_id="s01",
            phase_id="observational_review",
            objective_id="benefit_risk",
        )


def test_ranking_reference_requires_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="policy reference ranking is missing"):
        _reference_ranking_for(
            tmp_path,
            scenario_id="s01",
            phase_id="observational_review",
            objective_id="benefit_risk",
        )


def test_ranking_reference_selects_the_submitted_method_route(tmp_path: Path) -> None:
    bundle_root = _write_scenario(tmp_path)
    path = bundle_root / "scenario_s01" / "grader" / "drug_ranking_reference_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    first = payload["records"][1]
    first["method_route_id"] = "method_a"
    payload["records"].append(
        {
            **first,
            "method_route_id": "method_b",
            "value": 0.6,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="requires a method_route_id"):
        _reference_ranking_for(
            bundle_root,
            scenario_id="s01",
            phase_id="observational_review",
            objective_id="benefit_risk",
        )
    assert _reference_ranking_for(
        bundle_root,
        scenario_id="s01",
        phase_id="observational_review",
        objective_id="benefit_risk",
        method_route_id="method_b",
    ) == [("control", 0.6)]


def test_recoverability_selects_the_submitted_method_route(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario_s01" / "grader"
    scenario.mkdir(parents=True)
    (scenario / "recoverability_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "phase_id": "observational_review",
                        "objective_id": "benefit_risk",
                        "method_route_id": method_route_id,
                        "candidate_records": [],
                        "acceptable_candidate_set": [candidate_id],
                        "acceptable_action_set": [],
                        "policy": "near_tie_set",
                    }
                    for method_route_id, candidate_id in (("method_a", "drug_a"), ("method_b", "drug_b"))
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires a method_route_id"):
        _recoverability_for(
            tmp_path,
            scenario_id="s01",
            phase_id="observational_review",
            objective_id="benefit_risk",
        )
    resolved = _recoverability_for(
        tmp_path,
        scenario_id="s01",
        phase_id="observational_review",
        objective_id="benefit_risk",
        method_route_id="method_b",
    )

    assert resolved["acceptable_candidate_set"] == ("drug_b",)
