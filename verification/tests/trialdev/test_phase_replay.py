"""Independent randomized-phase replay tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import beta

from trialagentbench_validation.cli import main
from trialagentbench_validation.trialdev import phase_replay
from trialagentbench_validation.trialdev.phase_replay import (
    TrialDevPhaseRequestV1,
    validate_trialdev_phase_replay,
)


def test_independent_phase3_interval_policy_has_one_conclusion() -> None:
    """The verifier distinguishes success, failure, and inconclusive confirmation."""

    common = {
        "phase_id": "phase3",
        "margin": 0.015,
        "stop": ("declare_failure", "declare_inconclusive"),
        "advance": ("declare_success",),
        "direct_completion_margin": None,
    }
    assert phase_replay._actions_for_interval(
        interval=(0.06, 0.02, 0.10), **common
    ) == ("declare_success",)
    assert phase_replay._actions_for_interval(
        interval=(0.02, -0.01, 0.05), **common
    ) == ("declare_inconclusive",)
    assert phase_replay._actions_for_interval(
        interval=(-0.03, -0.06, 0.01), **common
    ) == ("declare_failure",)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _surface(root: Path) -> tuple[Path, Path, Path, Path]:
    bundle = root / "bundle"
    scenario = bundle / "seed_17" / "scenario_s1"
    public = scenario / "public"
    output_root = root / "materialized"
    request = TrialDevPhaseRequestV1(
        scenario_id="s1",
        phase_id="phase1",
        candidate_drug_ids=("drug_a",),
        target_sample_size=20,
        follow_up_days=28,
        enrollment_window_days=42,
        site_count_budget=8,
        allocation_ratio="1:1",
        design_cell_id="phase1.fixed",
        interim_policy="fixed_final",
        site_strategy="high_enrolling",
        selection_objective="benefit_risk",
    )
    relative_output = f"world_17/request_{request.checksum()}/trial_seed_31"
    trial_output = output_root / relative_output
    rows = []
    for arm in ("CONTROL", "TREATMENT"):
        for index in range(10):
            rows.append(
                {
                    "USUBJID": f"{arm}-{index}",
                    "ARM": arm,
                    "EVENT": 0,
                    "COMPETING_EVENT": 0,
                    "TIME": 28.0,
                    "AE_EVENT_E": 0,
                    "AE_EVENT_T": 28.0,
                    "AE_SERIOUS": None,
                    "DISCONTINUATION_E": 0,
                    "DISCONTINUATION_T": 28.0,
                    "LTFU_E": 0,
                    "LTFU_T": 28.0,
                    "TERMINAL_EVENT": 0,
                    "TERMINAL_TIME": 28.0,
                }
            )
    frame = pd.DataFrame(rows)
    trial_output.mkdir(parents=True)
    frame[["USUBJID", "ARM", "EVENT", "COMPETING_EVENT", "TIME"]].to_parquet(
        trial_output / "endpoints.parquet",
        index=False,
    )
    frame[
        [
            "USUBJID",
            "ARM",
            "AE_EVENT_E",
            "AE_EVENT_T",
            "AE_SERIOUS",
            "DISCONTINUATION_E",
            "DISCONTINUATION_T",
            "LTFU_E",
            "LTFU_T",
            "TERMINAL_EVENT",
            "TERMINAL_TIME",
        ]
    ].to_parquet(trial_output / "safety.parquet", index=False)
    _write_json(
        trial_output / "request.json",
        request.model_dump(mode="json", exclude_none=True),
    )
    _write_json(
        trial_output / "execution_summary.json",
        {
            "payload": {
                "loss_to_follow_up_assignment": "arm_conditional_random_permutation_v1",
            }
        },
    )
    _write_json(
        trial_output / "arm_mapping.json",
        {
            "control_arm_id": "CONTROL",
            "candidate_arm_ids": ["TREATMENT"],
            "drug_id_by_arm": {"CONTROL": "control", "TREATMENT": "drug_a"},
            "arm_weight_by_id": {"CONTROL": 0.5, "TREATMENT": 0.5},
        },
    )
    _write_json(
        public / "phase_decision_evidence_policy.json",
        {
            "confidence_level": 0.95,
            "phase_rules": [{"phase_id": "phase1", "evaluation_horizon_days": 28}],
        },
    )
    _write_json(
        public / "phase_action_policy.json",
        {
            "action_specs": [
                {
                    "phase_id": "phase1",
                    "stop_action_ids": ["stop"],
                    "advance_action_ids": ["advance"],
                }
            ]
        },
    )
    _write_json(
        public / "safety_decision_policy.json",
        {
            "serious_event_definitions": [
                {
                    "endpoint_id": "ae",
                    "event_column": "AE_EVENT_E",
                    "time_column": "AE_EVENT_T",
                    "seriousness_column": "AE_SERIOUS",
                }
            ],
            "thresholds": [
                {
                    "phase_id": "phase1",
                    "component_id": "serious_ae",
                    "role": "hard_gate",
                    "max_absolute_rate": 0.5,
                    "max_excess_vs_control": 0.5,
                    "sensitivity_max_absolute_rates": {
                        "strict": 0.2,
                        "primary": 0.5,
                        "permissive": 0.7,
                    },
                    "sensitivity_max_excess_vs_control": {
                        "strict": 0.2,
                        "primary": 0.5,
                        "permissive": 0.7,
                    },
                },
                {
                    "phase_id": "phase1",
                    "component_id": "discontinuation",
                    "role": "diagnostic_only",
                    "max_absolute_rate": 0.5,
                    "max_excess_vs_control": 0.5,
                    "sensitivity_max_absolute_rates": {
                        "strict": 0.2,
                        "primary": 0.5,
                        "permissive": 0.7,
                    },
                    "sensitivity_max_excess_vs_control": {
                        "strict": 0.2,
                        "primary": 0.5,
                        "permissive": 0.7,
                    },
                },
            ],
        },
    )
    absolute_power = 0.6777995264000001
    excess_power = 0.2435110687287947
    _write_json(
        public / "phase_design_policy.json",
        {
            "confidence_level": 0.95,
            "phase_rules": [
                {
                    "phase_id": "phase1",
                    "design_cell_id": "phase1.fixed",
                    "supported_interim_policy": "fixed_final",
                    "evaluation_horizon_days": 28,
                    "primary_endpoint_id": None,
                    "planning_information_fraction_by_drug_id": {
                        "control": 1.0,
                        "drug_a": 1.0,
                    },
                    "serious_ae_unacceptable_absolute_risk": 0.5,
                    "serious_ae_unacceptable_excess_risk": 0.5,
                    "planning_safety_control_risk": 0.1,
                    "planning_safety_absolute_treatment_risk": 0.8,
                    "planning_safety_excess_treatment_risk": 0.8,
                    "target_safety_decision_power": 0.2,
                    "target_power": None,
                }
            ],
        },
    )
    _write_json(
        public / "phase_design_frontiers.json",
        {
            "operational_support": [
                {
                    "phase_id": "phase1",
                    "enrollment_window_days": 42,
                    "site_count_budget": 8,
                    "site_strategy": "high_enrolling",
                    "eligible_subject_count": 20,
                }
            ],
            "strata": [
                {
                    "phase_id": "phase1",
                    "candidate_drug_ids": ["drug_a"],
                    "endpoint_id": None,
                    "treatment_discontinuation_strategy": None,
                    "design_cell_id": "phase1.fixed",
                    "interim_policy": "fixed_final",
                    "frontier": [
                        {
                            "target_sample_size": 20,
                            "follow_up_days": 28,
                            "allocation_ratio": "1:1",
                            "achieved_power": None,
                            "achieved_safety_absolute_risk_power": absolute_power,
                            "achieved_safety_excess_risk_power": excess_power,
                        }
                    ],
                }
            ],
        },
    )
    cases = root / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "scenario_root": "seed_17/scenario_s1",
                "world_seed": 17,
                "program_objective_ids": ["benefit_risk"],
                "request": request.model_dump(mode="json"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    upper = float(beta.ppf(0.975, 1, 10))
    interval = {"estimate": 0.0, "lower": 0.0, "upper": upper}
    excess = {"estimate": 0.0, "lower": -upper, "upper": upper}
    source_checksums = {
        f"public/{name}": _sha256(public / name)
        for name in (
            "phase_action_policy.json",
            "phase_decision_evidence_policy.json",
            "phase_design_frontiers.json",
            "phase_design_policy.json",
            "safety_decision_policy.json",
        )
    }
    source_checksums.update(
        {
            f"trial_output/{name}": _sha256(trial_output / name)
            for name in (
                "arm_mapping.json",
                "endpoints.parquet",
                "execution_summary.json",
                "request.json",
                "safety.parquet",
            )
        }
    )

    def component(component_id: str, role: str) -> dict[str, object]:
        return {
            "component_id": component_id,
            "role": role,
            "treated": interval,
            "control": interval,
            "excess": excess,
            "absolute_limit": 0.5,
            "excess_limit": 0.5,
        }

    records = root / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "schema_id": "trialagentbench.trialdev_public_phase_replay/v1",
                "scenario_id": "s1",
                "world_seed": 17,
                "trial_seed": 31,
                "request_checksum": request.checksum(),
                "trial_output_path": relative_output,
                "phase_id": "phase1",
                "endpoint_id": None,
                "treatment_discontinuation_strategy": None,
                "follow_up_days": 28,
                "target_sample_size": 20,
                "allocation_ratio": "1:1",
                "objective_ids": ["benefit_risk"],
                "candidate_drug_ids": ["drug_a"],
                "acceptable_action_ids": ["advance"],
                "stop_action_ids": ["stop"],
                "advance_action_ids": ["advance"],
                "sensitivity_action_sets": {
                    "primary": ["advance"],
                    "safety_profile::strict": ["advance", "stop"],
                    "safety_profile::primary": ["advance", "stop"],
                    "safety_profile::permissive": ["advance", "stop"],
                },
                "public_decision_witness_checksum": "b" * 64,
                "public_source_checksums": source_checksums,
                "candidate_decision_evidence": [
                    {
                        "candidate_arm_id": "drug_a",
                        "acceptable_action_ids": ["advance"],
                        "safety_state": "acceptable",
                        "efficacy": None,
                        "minimum_efficacy_benefit": None,
                        "safety_components": [
                            component("serious_ae", "hard_gate"),
                            component("discontinuation", "diagnostic_only"),
                        ],
                    }
                ],
                "public_safety_state": "acceptable",
                "design_adequate": True,
                "design_failures": [],
                "design_frontier": [
                    {
                        "target_sample_size": 20,
                        "follow_up_days": 28,
                        "allocation_ratio": "1:1",
                        "achieved_power": None,
                        "achieved_safety_absolute_risk_power": absolute_power,
                        "achieved_safety_excess_risk_power": excess_power,
                    }
                ],
                "design_on_frontier": True,
                "design_dominated_by_frontier": False,
                "minimum_frontier_participants": 20,
                "minimum_frontier_follow_up_days": 28,
                "participant_excess_vs_minimum": 0,
                "participant_shortage_vs_minimum": 0,
                "follow_up_excess_days_vs_minimum": 0,
                "follow_up_shortage_days_vs_minimum": 0,
                "achieved_power": None,
                "target_power": None,
                "achieved_safety_absolute_risk_power": absolute_power,
                "achieved_safety_excess_risk_power": excess_power,
                "target_safety_decision_power": 0.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle, output_root, cases, records


def test_independent_phase_replay_accepts_public_evidence(tmp_path: Path) -> None:
    bundle, materialized, cases, records = _surface(tmp_path)

    report = validate_trialdev_phase_replay(
        bundle_root=bundle,
        materialized_root=materialized,
        cases_path=cases,
        records_path=records,
    )

    assert report.status == "pass"
    assert report.records[0].maximum_absolute_error == 0.0


def test_independent_phase_replay_rejects_numeric_and_checksum_drift(
    tmp_path: Path,
) -> None:
    bundle, materialized, cases, records = _surface(tmp_path)
    payload = json.loads(records.read_text())
    payload["candidate_decision_evidence"][0]["safety_components"][0]["treated"][
        "upper"
    ] += 0.1
    payload["public_source_checksums"]["trial_output/safety.parquet"] = "0" * 64
    records.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    report = validate_trialdev_phase_replay(
        bundle_root=bundle,
        materialized_root=materialized,
        cases_path=cases,
        records_path=records,
    )

    assert report.status == "fail"
    assert not report.records[0].source_checksums_match
    assert not report.records[0].numeric_evidence_match


def test_independent_phase_replay_rejects_uncontrolled_ltfu_construction(
    tmp_path: Path,
) -> None:
    bundle, materialized, cases, records = _surface(tmp_path)
    execution_summary = next(materialized.rglob("execution_summary.json"))
    payload = json.loads(execution_summary.read_text(encoding="utf-8"))
    payload["payload"]["loss_to_follow_up_assignment"] = "outcome_dependent_assignment"
    _write_json(execution_summary, payload)

    report = validate_trialdev_phase_replay(
        bundle_root=bundle,
        materialized_root=materialized,
        cases_path=cases,
        records_path=records,
    )

    assert report.status == "fail"
    assert not report.records[0].source_checksums_match
    assert not report.records[0].ltfu_construction_match


def test_independent_phase_replay_rejects_sensitivity_overcredit(
    tmp_path: Path,
) -> None:
    bundle, materialized, cases, records = _surface(tmp_path)
    payload = json.loads(records.read_text())
    payload["sensitivity_action_sets"]["safety_profile::strict"] = ["advance"]
    records.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    report = validate_trialdev_phase_replay(
        bundle_root=bundle,
        materialized_root=materialized,
        cases_path=cases,
        records_path=records,
    )

    assert report.status == "fail"
    assert not report.records[0].action_match


def test_independent_phase_replay_rejects_design_power_drift(tmp_path: Path) -> None:
    bundle, materialized, cases, records = _surface(tmp_path)
    payload = json.loads(records.read_text())
    payload["achieved_safety_excess_risk_power"] += 0.01
    records.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    report = validate_trialdev_phase_replay(
        bundle_root=bundle,
        materialized_root=materialized,
        cases_path=cases,
        records_path=records,
    )

    assert report.status == "fail"
    assert not report.records[0].design_projection_match


def test_independent_phase_replay_requires_operational_frontier_support(
    tmp_path: Path,
) -> None:
    bundle, materialized, cases, records = _surface(tmp_path)
    frontier_path = (
        bundle / "seed_17" / "scenario_s1" / "public" / "phase_design_frontiers.json"
    )
    frontier = json.loads(frontier_path.read_text())
    frontier["operational_support"][0]["eligible_subject_count"] = 19
    _write_json(frontier_path, frontier)
    payload = json.loads(records.read_text())
    payload["public_source_checksums"]["public/phase_design_frontiers.json"] = _sha256(
        frontier_path
    )
    records.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    report = validate_trialdev_phase_replay(
        bundle_root=bundle,
        materialized_root=materialized,
        cases_path=cases,
        records_path=records,
    )

    assert report.status == "fail"
    assert report.records[0].source_checksums_match
    assert not report.records[0].design_projection_match


def test_independent_phase_replay_cli_writes_report(tmp_path: Path) -> None:
    bundle, materialized, cases, records = _surface(tmp_path)
    output = tmp_path / "report.json"

    status = main(
        [
            "trialdev-phase-replay",
            "--bundle-root",
            str(bundle),
            "--materialized-root",
            str(materialized),
            "--cases",
            str(cases),
            "--records",
            str(records),
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert json.loads(output.read_text())["status"] == "pass"
