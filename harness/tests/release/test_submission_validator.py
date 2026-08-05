"""Tests for the score-blind public submission validator."""

from __future__ import annotations

import json
from pathlib import Path

from trialagentbench_harness.contracts.submission import trialeval_submission_shape_catalogue
from trialagentbench_harness.tools.validate.validate_submission import main


def test_trialeval_submission_validator_uses_public_schema_only(tmp_path: Path) -> None:
    payload = (
        trialeval_submission_shape_catalogue().primary_submissions["numeric_point:scalar"].model_dump(mode="json")
    )
    payload["primary_analysis"]["estimator"].pop("implementation")
    payload.pop("limitations")
    submission = tmp_path / "submission.json"
    submission.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["--suite", "trialeval", "--submission", str(submission)]) == 0


def test_submission_validator_rejects_duplicate_fields_with_stable_json_report(
    tmp_path: Path,
    capsys,
) -> None:
    submission = tmp_path / "submission.json"
    submission.write_text('{"task_id":"TASK1","task_id":"TASK2"}', encoding="utf-8")

    assert (
        main(
            [
                "--suite",
                "trialeval",
                "--submission",
                str(submission),
                "--output-format",
                "json",
            ]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert report["scope"] == "schema_only"
    assert report["issues"] == [
        {
            "code": "duplicate_json_field",
            "json_pointer": "",
            "message": "Duplicate JSON field: 'task_id'.",
        }
    ]


def test_trialdev_submission_validator_uses_public_schema_only(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[2]
    examples = package_root / "examples" / "submissions"
    request = json.loads((examples / "trialdev_phase_request.json").read_text(encoding="utf-8"))
    analysis = json.loads((examples / "trialdev_phase_analysis.json").read_text(encoding="utf-8"))
    decision = json.loads((examples / "trialdev_phase_decision.json").read_text(encoding="utf-8"))
    request.update({"scenario_id": "example_scenario", "phase_id": "phase2"})
    analysis["primary_effect"]["source_artifact_checksums"] = {"data/example.parquet": "0" * 64}
    submission = tmp_path / "submission.json"
    submission.write_text(
        json.dumps(
            {
                "version": "v1",
                "scenario_id": "example_scenario",
                "request": {key: value for key, value in request.items() if key != "request_rationale"},
                "analysis_report": {
                    "selected_winner_drug_id": analysis["selected_winner_drug_id"],
                    "ranked_drug_ids": analysis["ranked_drug_ids"],
                    "primary_effect": analysis["primary_effect"],
                    "claimed_subgroup_variables": analysis["claimed_subgroup_variables"],
                    "diagnostic_artifacts": analysis["diagnostic_artifacts"],
                    "evidence_summary": analysis["evidence_summary"],
                },
                "program_decision": {
                    "objective_id": request["selection_objective"],
                    "decision_action": decision["decision_action"],
                    "recommended_drug_id": decision["candidate_drug_id"],
                    "supporting_evidence_ids": decision["supporting_evidence_ids"],
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(["--suite", "trialdev", "--submission", str(submission)]) == 0
