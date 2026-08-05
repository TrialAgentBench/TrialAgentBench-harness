"""Public-evidence TrialDev phase replay tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.trialdev.trialdev_public_phase_replay import (
    TrialDevPhaseReplayCaseV1,
)
from trialagentbench_harness.trialdev import public_phase_replay
from trialagentbench_harness.trialdev.grading.decision_evidence import (
    TrialDevPhaseDecisionWitnessV1,
    TrialDevPhaseDesignWitnessV1,
)
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevDesignEfficiencyV1,
    TrialDevDesignFrontierPointV1,
)
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1


def _request() -> TrialDevelopmentRequestV1:
    return TrialDevelopmentRequestV1(
        scenario_id="s1",
        phase_id="phase2",
        candidate_drug_ids=("drug_a",),
        target_sample_size=120,
        endpoint_id="PRIMARY",
        follow_up_days=180,
        enrollment_window_days=42,
        site_count_budget=8,
        allocation_ratio="1:1",
        design_cell_id="trialdev.phase2.fixed_final_operating_characteristics.v1",
        treatment_discontinuation_strategy="treatment_policy",
        interim_policy="fixed_final",
        site_strategy="high_enrolling",
        selection_objective="cost_effective_best",
    )


def _decision_witness() -> TrialDevPhaseDecisionWitnessV1:
    candidate = {
        "acceptable_action_ids": ["advance"],
        "hard_safety_stop_action_ids": ["stop"],
        "efficacy_action_ids": ["advance"],
        "safety_state": "acceptable",
        "safety": {
            "treated_serious_rate": 0.03,
            "treated_serious_rate_interval": (0.01, 0.05),
            "control_serious_rate": 0.02,
            "control_serious_rate_interval": (0.0, 0.04),
            "serious_rate_excess": 0.01,
            "serious_rate_excess_interval": (-0.02, 0.04),
            "absolute_limit": 0.1,
            "excess_limit": 0.05,
            "sensitivity_states": {
                "strict": "unacceptable",
                "primary": "acceptable",
                "permissive": "acceptable",
            },
        },
        "efficacy": {
            "evaluated": True,
            "risk_difference_control_minus_treatment": 0.04,
            "confidence_interval": (0.01, 0.07),
            "minimum_benefit": 0.02,
            "margin_sensitivity_action_sets": {
                "0.010000": ["advance"],
                "0.020000": ["advance", "stop"],
            },
        },
    }
    return TrialDevPhaseDecisionWitnessV1(
        phase_id="phase2",
        acceptable_action_ids=("advance",),
        recoverability_class="decision_recoverable",
        safety_state="acceptable",
        safety_action_ids=("advance",),
        efficacy_action_ids=("advance",),
        stop_action_ids=("stop",),
        advance_action_ids=("advance",),
        candidates=(),
        evidence={
            "source_checksums": {
                "public/phase_action_policy.json": "a" * 64,
                "public/phase_decision_evidence_policy.json": "b" * 64,
                "public/safety_decision_policy.json": "c" * 64,
                "trial_output/arm_mapping.json": "d" * 64,
                "trial_output/endpoints.parquet": "e" * 64,
                "trial_output/execution_summary.json": "1" * 64,
                "trial_output/request.json": "f" * 64,
                "trial_output/safety.parquet": "0" * 64,
            },
            "candidates": {"drug_a": candidate},
        },
    )


def _design_witness() -> TrialDevPhaseDesignWitnessV1:
    return TrialDevPhaseDesignWitnessV1(
        phase_id="phase2",
        adequate=True,
        achieved_power=0.82,
        achieved_safety_absolute_risk_power=0.91,
        achieved_safety_excess_risk_power=0.90,
        target_power=0.80,
        target_safety_decision_power=0.80,
        failures=(),
        evidence={},
    )


def _design_efficiency() -> TrialDevDesignEfficiencyV1:
    frontier = (
        TrialDevDesignFrontierPointV1(
            target_sample_size=120,
            follow_up_days=180,
            allocation_ratio="1:1",
            achieved_power=0.82,
            achieved_safety_absolute_risk_power=0.91,
            achieved_safety_excess_risk_power=0.90,
        ),
    )
    return TrialDevDesignEfficiencyV1(
        statistically_adequate=True,
        operationally_feasible=True,
        design_valid=True,
        on_frontier=True,
        dominated_by_frontier=False,
        operational_support=150,
        operational_headroom=30,
        operational_shortage=0,
        minimum_frontier_participants=120,
        minimum_frontier_follow_up_days=180,
        participant_excess_vs_minimum=0,
        participant_shortage_vs_minimum=0,
        follow_up_excess_days_vs_minimum=0,
        follow_up_shortage_days_vs_minimum=0,
        achieved_power=0.82,
        target_power=0.80,
        achieved_safety_absolute_risk_power=0.91,
        achieved_safety_excess_risk_power=0.90,
        target_safety_decision_power=0.80,
        frontier=frontier,
    )


def test_public_phase_replay_emits_only_public_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = tmp_path / "scenario_s1"
    public = scenario / "public"
    public.mkdir(parents=True)
    for name in ("phase_design_frontiers.json", "phase_design_policy.json"):
        (public / name).write_text("{}\n", encoding="utf-8")

    def materialize(**kwargs: object) -> SimpleNamespace:
        output = Path(str(kwargs["out_dir"]))
        output.mkdir(parents=True)
        return SimpleNamespace(
            audit=SimpleNamespace(feasibility_status="accepted", rejection_reason=None),
            trial_tables_dir=str(output),
        )

    monkeypatch.setattr(public_phase_replay, "materialize_trial_view_v1", materialize)
    monkeypatch.setattr(
        public_phase_replay,
        "derive_phase_decision_witness_v1",
        lambda **_: _decision_witness(),
    )
    monkeypatch.setattr(
        public_phase_replay,
        "derive_phase_design_witness_v1",
        lambda **_: _design_witness(),
    )
    monkeypatch.setattr(
        public_phase_replay,
        "derive_phase_design_efficiency_v1",
        lambda **_: _design_efficiency(),
    )
    case = TrialDevPhaseReplayCaseV1(
        scenario_root="scenario_s1",
        world_seed=17,
        program_objective_ids=("cost_effective_best",),
        request=_request(),
    )

    records = public_phase_replay.replay_trialdev_public_phases_v1(
        bundle_root=tmp_path,
        materialized_root=tmp_path / "trials",
        cases=(case,),
        trial_seeds=(31, 32),
    )

    assert {record.trial_seed for record in records} == {31, 32}
    assert all(record.design_on_frontier for record in records)
    assert all(record.sensitivity_action_sets["safety_profile::strict"] == ("stop",) for record in records)
    assert all("public/phase_design_policy.json" in record.public_source_checksums for record in records)
    assert all("public/phase_design_frontiers.json" in record.public_source_checksums for record in records)
    payload = records[0].model_dump(mode="json")
    assert "diagnostic_reference_action_ids" not in payload
    assert "oracle_safety_state" not in payload


def test_public_phase_replay_preserves_materialization_rejection_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = tmp_path / "scenario_s1"
    scenario.mkdir()
    monkeypatch.setattr(
        public_phase_replay,
        "materialize_trial_view_v1",
        lambda **_: SimpleNamespace(
            audit=SimpleNamespace(
                feasibility_status="rejected",
                rejection_reason="insufficient_site_budget_support",
            ),
            trial_tables_dir=None,
        ),
    )
    case = TrialDevPhaseReplayCaseV1(
        scenario_root="scenario_s1",
        world_seed=17,
        program_objective_ids=("cost_effective_best",),
        request=_request(),
    )

    with pytest.raises(ValueError, match="insufficient_site_budget_support"):
        public_phase_replay.replay_trialdev_public_phases_v1(
            bundle_root=tmp_path,
            materialized_root=tmp_path / "trials",
            cases=(case,),
            trial_seeds=(31,),
        )


def test_public_phase_replay_rejects_unsafe_case_or_duplicate_seed(tmp_path: Path) -> None:
    payload = {
        "scenario_root": "../scenario_s1",
        "world_seed": 17,
        "program_objective_ids": ["cost_effective_best"],
        "request": _request().model_dump(mode="json"),
    }
    with pytest.raises(ValidationError, match="safe path"):
        TrialDevPhaseReplayCaseV1.model_validate(payload)

    case = TrialDevPhaseReplayCaseV1(
        scenario_root="scenario_s1",
        world_seed=17,
        program_objective_ids=("cost_effective_best",),
        request=_request(),
    )
    with pytest.raises(ValueError, match="trial seeds must be unique"):
        public_phase_replay.replay_trialdev_public_phases_v1(
            bundle_root=tmp_path,
            materialized_root=tmp_path / "trials",
            cases=(case,),
            trial_seeds=(31, 31),
        )
