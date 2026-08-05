from __future__ import annotations

import json
from pathlib import Path

import pytest
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)

from trialagentbench_harness.trialeval.data import (
    discover_items,
    discover_participant_items,
    load_visible_context,
    participant_task_factors,
    stage_participant_evidence,
)


def _evidence_factors(context: str) -> dict[str, str]:
    mapping = {
        "C1": ("analysis_ready", "locked_sap"),
        "C2": ("analysis_ready", "protocol_only"),
        "C3": ("raw_domains", "locked_sap"),
        "C4": ("raw_domains", "protocol_only"),
        "C5": ("raw_domains_declared_defect", "protocol_only"),
    }
    data_preparation, analysis_specification = mapping[context]
    return {
        "context_configuration": context,
        "data_preparation": data_preparation,
        "analysis_specification": analysis_specification,
    }


def _participant_release(root: Path) -> Path:
    participant = root / "participant"
    participant.mkdir(parents=True)
    write_minimal_trialeval_release_dictionaries(participant)
    item = participant / "items" / "TASK1"
    (item / "data").mkdir(parents=True)
    (participant / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_public_bundle_manifest/v1",
                "applied_baseline_profile_id": None,
                "applied_baseline_profile_sha256": None,
                "task_ids": ["TASK1"],
                "task_evidence_factors": {"TASK1": _evidence_factors("C1")},
            }
        ),
        encoding="utf-8",
    )
    (item / "task.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_task_v1",
                "task_id": "TASK1",
                "design_subtype": "individual_randomized",
                "primary_endpoint_id": "endpoint",
                "primary_paramcd": "endpoint",
                "primary_estimand_id": "estimand",
                "primary_effect_scale": "risk_difference_tau",
                "estimand_mode": "fixed_declared_estimand",
                "primary_effect_scale_options": ["risk_difference_tau"],
                "primary_result_unit": "probability_difference",
                "primary_tau_dy": 365.0,
                "primary_population_id": "itt",
                "primary_intercurrent_event_strategy_ids": ["treatment_policy"],
                "primary_control_arm_id": "control",
                "primary_treated_arm_id": "treated",
            }
        ),
        encoding="utf-8",
    )
    (item / "study_brief.md").write_text("Participant study brief.", encoding="utf-8")
    (item / "submission_contract.json").write_text(
        json.dumps(minimal_participant_output_contract("TASK1")),
        encoding="utf-8",
    )
    (item / "data" / "ADSL.parquet").write_bytes(b"participant-data")
    return participant


def test_evaluator_discovery_rejects_missing_item_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="item index is missing"):
        discover_items(tmp_path)


@pytest.mark.parametrize("payload", [[], {}, {"entries": []}, {"entries": ["TASK1"]}])
def test_evaluator_discovery_rejects_malformed_or_empty_item_index(
    tmp_path: Path,
    payload: object,
) -> None:
    index = tmp_path / "grader" / "item_index.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="item index|entries must be objects"):
        discover_items(tmp_path)


def test_participant_discovery_uses_only_public_manifest(tmp_path: Path) -> None:
    manifest = {
        "schema_id": "trial_analysis_public_bundle_manifest/v1",
        "applied_baseline_profile_id": None,
        "applied_baseline_profile_sha256": None,
        "task_ids": ["TASK2", "TASK1"],
        "task_evidence_factors": {
            "TASK1": _evidence_factors("C3"),
            "TASK2": _evidence_factors("C1"),
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "grader").mkdir()
    (tmp_path / "grader" / "item_index.json").write_text("not-json", encoding="utf-8")

    task_ids, factors = participant_task_factors(tmp_path)

    assert task_ids == ["TASK2", "TASK1"]
    assert {task_id: row.context_configuration for task_id, row in factors.items()} == {
        "TASK1": "C3",
        "TASK2": "C1",
    }


@pytest.mark.parametrize(
    "task_ids,contexts",
    [
        (["TASK1", "TASK1"], {"TASK1": "C1"}),
        (["TASK1"], {}),
        ([], {}),
    ],
)
def test_participant_discovery_rejects_ambiguous_manifest(
    tmp_path: Path,
    task_ids: list[str],
    contexts: dict[str, str],
) -> None:
    manifest = {
        "schema_id": "trial_analysis_public_bundle_manifest/v1",
        "applied_baseline_profile_id": None,
        "applied_baseline_profile_sha256": None,
        "task_ids": task_ids,
        "task_evidence_factors": {task_id: _evidence_factors(context) for task_id, context in contexts.items()},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="task_ids|task_evidence_factors"):
        participant_task_factors(tmp_path)


def test_participant_manifest_rejects_symlink_to_evaluator(tmp_path: Path) -> None:
    participant = tmp_path / "participant"
    participant.mkdir()
    write_minimal_trialeval_release_dictionaries(participant)
    evaluator_manifest = tmp_path / "evaluator" / "manifest.json"
    evaluator_manifest.parent.mkdir()
    evaluator_manifest.write_text("{}", encoding="utf-8")
    (participant / "manifest.json").symlink_to(evaluator_manifest)

    with pytest.raises(ValueError, match="must not be a symlink"):
        participant_task_factors(participant)


def test_participant_discovery_rejects_symlinked_item_directory(tmp_path: Path) -> None:
    participant = tmp_path / "participant"
    evaluator_item = tmp_path / "evaluator" / "items" / "TASK1"
    evaluator_item.mkdir(parents=True)
    participant.mkdir()
    write_minimal_trialeval_release_dictionaries(participant)
    (participant / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_public_bundle_manifest/v1",
                "applied_baseline_profile_id": None,
                "applied_baseline_profile_sha256": None,
                "task_ids": ["TASK1"],
                "task_evidence_factors": {"TASK1": _evidence_factors("C1")},
            }
        ),
        encoding="utf-8",
    )
    (participant / "items").mkdir()
    (participant / "items" / "TASK1").symlink_to(evaluator_item, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        discover_participant_items(participant, task_ids=("TASK1",))


def test_visible_context_rejects_allowed_text_symlink_to_evaluator(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    item = discover_participant_items(participant, task_ids=("TASK1",))["TASK1"]
    brief = item.visible_dir / "study_brief.md"
    brief.unlink()
    evaluator_brief = tmp_path / "evaluator" / "grader_notes.md"
    evaluator_brief.parent.mkdir()
    evaluator_brief.write_text("hidden evaluator notes", encoding="utf-8")
    brief.symlink_to(evaluator_brief)

    with pytest.raises(ValueError, match="must not be a symlink"):
        load_visible_context(item)


def test_staging_rejects_data_symlink_outside_participant_root(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    item = discover_participant_items(participant, task_ids=("TASK1",))["TASK1"]
    data_file = item.data_dir / "ADSL.parquet"
    data_file.unlink()
    outside_data = tmp_path / "evaluator" / "truth.parquet"
    outside_data.parent.mkdir()
    outside_data.write_bytes(b"evaluator-data")
    data_file.symlink_to(outside_data)
    destination = tmp_path / "staged"

    with pytest.raises(ValueError, match="must not be a symlink"):
        stage_participant_evidence(item, destination)
    assert not destination.exists()


def test_normal_participant_discovery_context_and_staging(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    dictionary = participant / "items" / "TASK1" / "data_dictionary.json"
    dictionary.write_text('{"large_reference":"mounted_not_inlined"}', encoding="utf-8")

    item = discover_participant_items(participant, task_ids=("TASK1",))["TASK1"]
    context = load_visible_context(item)
    staged = stage_participant_evidence(item, tmp_path / "staged")

    assert item.visible_dir == participant / "items" / "TASK1"
    assert "Participant study brief." in context
    assert "=== submission_contract.json ===" in context
    assert "trialagentbench.trialeval_semantic_submission_contract/v1" in context
    assert "mounted_not_inlined" not in context
    assert (staged / "data" / "ADSL.parquet").read_bytes() == b"participant-data"
    assert json.loads((staged / "submission_contract.json").read_text(encoding="utf-8")) == {
        **minimal_participant_output_contract("TASK1"),
    }
    assert json.loads((staged / "data_dictionary.json").read_text(encoding="utf-8")) == {
        "large_reference": "mounted_not_inlined"
    }
    method_dictionary = json.loads((staged / "method_dictionary.json").read_text(encoding="utf-8"))
    assert method_dictionary["methods"][0]["method_id"] == "km_rmst_greenwood"
    assert method_dictionary["methods"][0]["uncertainty_method_id"] == "greenwood"
    assert method_dictionary["methods"][0]["result_kind"] == "numeric_point"
    diagnostic_dictionary = json.loads((staged / "diagnostic_dictionary.json").read_text(encoding="utf-8"))
    censoring = diagnostic_dictionary["diagnostics"]["censoring_followup_public"]
    assert censoring["severity_thresholds"]["metric_name"] == "lower_abs_prognostic_censoring_log_hr"
    assert censoring["severity_thresholds"]["metric_unit"] == "log_hazard_ratio"
