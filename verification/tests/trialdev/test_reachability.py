"""Tests for independent TrialDev programme reachability verification."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from trialagentbench_validation import cli
from trialagentbench_validation.trialdev.reachability import audit_trialdev_reachability


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True).encode("utf-8")


def _write_release_pair(
    tmp_path: Path,
    *,
    policies: dict[tuple[str, str], tuple[str, ...]],
    cases: tuple[dict[str, object], ...],
) -> tuple[Path, Path]:
    participant = tmp_path / "participant.zip"
    evaluator = tmp_path / "evaluator.zip"
    contexts = tuple(sorted(policies))
    with ZipFile(participant, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "benchmark_suite_manifest.json",
            _json(
                {
                    "items": [
                        {
                            "scenario_id": scenario_id,
                            "objective_id": objective_id,
                            "phase_id": "observational_review",
                        }
                        for scenario_id, objective_id in contexts
                    ]
                }
            ),
        )
        archive.writestr(
            "fixed_trajectories/cases.jsonl",
            b"".join(_json(case) + b"\n" for case in cases),
        )
    with ZipFile(evaluator, "w", compression=ZIP_DEFLATED) as archive:
        for scenario_id in sorted(
            {scenario_id for scenario_id, _objective_id in contexts}
        ):
            rows = []
            for policy_scenario, objective_id in contexts:
                if policy_scenario != scenario_id:
                    continue
                eligible = policies[(scenario_id, objective_id)]
                rows.append(
                    {
                        "objective_id": objective_id,
                        "reference_target_ids": [eligible[0]],
                        "credit_eligible_target_ids": list(eligible),
                    }
                )
            archive.writestr(
                f"scenario_{scenario_id}/grader/public_recoverability_report.json",
                _json(
                    {
                        "scenario_id": scenario_id,
                        "method_union_action_sensitivity": rows,
                    }
                ),
            )
    return participant, evaluator


def _cases(
    *,
    scenario_id: str,
    objective_ids: tuple[str, ...],
    candidate_id: str,
    phases: tuple[str, ...] = ("phase1", "phase2", "phase3"),
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "scenario_root": f"scenario_{scenario_id}",
            "program_objective_ids": list(objective_ids),
            "request": {
                "scenario_id": scenario_id,
                "phase_id": phase_id,
                "candidate_drug_ids": [candidate_id],
            },
        }
        for phase_id in phases
    )


def test_reachability_census_distinguishes_all_three_action_classes(
    tmp_path: Path,
) -> None:
    participant, evaluator = _write_release_pair(
        tmp_path,
        policies={
            ("s01", "benefit_risk"): ("withhold_nomination",),
            ("s02", "benefit_risk"): ("withhold_nomination", "drug_a"),
            ("s03", "benefit_risk"): ("drug_b",),
        },
        cases=(
            *_cases(
                scenario_id="s02",
                objective_ids=("benefit_risk",),
                candidate_id="drug_a",
            ),
            *_cases(
                scenario_id="s03",
                objective_ids=("benefit_risk",),
                candidate_id="drug_b",
            ),
        ),
    )

    report = audit_trialdev_reachability(
        participant_release=participant,
        evaluator_release=evaluator,
    )

    assert report.status == "pass"
    assert report.required_programme_count == 3
    assert report.fixed_replay_case_count == 6
    assert report.expanded_programme_case_count == 6
    assert report.reachability_counts == {
        "structural_nonreach": 1,
        "optional_nomination": 1,
        "nomination_required": 1,
    }


def test_reachability_rejects_an_accepted_candidate_missing_a_phase(
    tmp_path: Path,
) -> None:
    participant, evaluator = _write_release_pair(
        tmp_path,
        policies={("s01", "benefit_risk"): ("drug_a",)},
        cases=_cases(
            scenario_id="s01",
            objective_ids=("benefit_risk",),
            candidate_id="drug_a",
            phases=("phase1", "phase2"),
        ),
    )

    report = audit_trialdev_reachability(
        participant_release=participant,
        evaluator_release=evaluator,
    )

    assert report.status == "fail"
    assert report.failed_programme_count == 1
    assert report.programmes[0].findings == ("candidate_missing_phases:drug_a:phase3",)


def test_reachability_rejects_replay_for_a_stop_only_programme(tmp_path: Path) -> None:
    participant, evaluator = _write_release_pair(
        tmp_path,
        policies={("s01", "benefit_risk"): ("withhold_nomination",)},
        cases=_cases(
            scenario_id="s01",
            objective_ids=("benefit_risk",),
            candidate_id="drug_a",
        ),
    )

    report = audit_trialdev_reachability(
        participant_release=participant,
        evaluator_release=evaluator,
    )

    assert report.status == "fail"
    assert report.programmes[0].findings == (
        "replay_candidates_not_credit_eligible:drug_a",
    )


def test_reachability_rejects_duplicate_phase_evidence(tmp_path: Path) -> None:
    phase_cases = _cases(
        scenario_id="s01",
        objective_ids=("benefit_risk",),
        candidate_id="drug_a",
    )
    participant, evaluator = _write_release_pair(
        tmp_path,
        policies={("s01", "benefit_risk"): ("drug_a",)},
        cases=(*phase_cases, phase_cases[0]),
    )

    report = audit_trialdev_reachability(
        participant_release=participant,
        evaluator_release=evaluator,
    )

    assert report.status == "fail"
    assert report.global_findings == (
        "line_4:duplicate_case:s01:benefit_risk:drug_a:phase1",
    )


def test_reachability_cli_writes_a_passing_report(tmp_path: Path) -> None:
    participant, evaluator = _write_release_pair(
        tmp_path,
        policies={("s01", "benefit_risk"): ("drug_a",)},
        cases=_cases(
            scenario_id="s01",
            objective_ids=("benefit_risk",),
            candidate_id="drug_a",
        ),
    )
    output = tmp_path / "reachability.json"

    exit_code = cli.main(
        (
            "trialdev-reachability",
            "--participant-release",
            str(participant),
            "--evaluator-release",
            str(evaluator),
            "--output",
            str(output),
        )
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
