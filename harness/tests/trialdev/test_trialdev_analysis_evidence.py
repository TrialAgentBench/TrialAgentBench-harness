"""Black-box qualification of TrialDev public-evidence effect replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trialagentbench_harness.trialdev.grading.analysis_evidence import (
    aalen_johansen_cif_variance_v1,
    derive_effect_references_v1,
    interval_equivalence_score_v1,
    point_equivalence_score_v1,
    point_interval_equivalence_score_v1,
    reporting_tolerance_v1,
)
from trialagentbench_harness.trialdev.share.hashing import compute_sha256_hex


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _checksummed(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["checksum"] = compute_sha256_hex(result)
    return result


def _method(phase_id: str) -> dict[str, object]:
    phase1 = phase_id == "phase1"
    return {
        "method_route_id": (
            "trialdev.phase1.aalen_johansen_safety_bundle.v1"
            if phase1
            else f"trialdev.{phase_id}.aalen_johansen_efficacy_safety.v1"
        ),
        "phase_id": phase_id,
        "calculator_id": ("aalen_johansen_safety_bundle_v1" if phase1 else "aalen_johansen_efficacy_safety_bundle_v1"),
        "estimator_id": "observed:aalen_johansen_cif_tau",
        "efficacy_estimand_id_template": (
            None if phase1 else "{treatment_discontinuation_strategy}:cumulative_incidence_at_horizon"
        ),
        "efficacy_effect_scale_id": (None if phase1 else "risk_difference_control_minus_treatment"),
        "efficacy_orientation_id": (None if phase1 else "positive_values_favour_treatment"),
        "safety_estimand_id_template": ("{safety_component_id}:cumulative_incidence_at_horizon"),
        "safety_absolute_risk_scale_id": "absolute_risk",
        "safety_excess_risk_scale_id": "risk_difference_treatment_minus_control",
        "safety_reported_measure_ids": [
            "treatment_absolute_risk",
            "control_absolute_risk",
            "risk_difference_treatment_minus_control",
        ],
        "safety_uncertainty_scope_id": ("two_sided_confidence_interval_per_safety_component_and_measure"),
        "safety_orientation_id": ("absolute_risk_higher_is_worse_and_risk_difference_positive_is_treatment_harm"),
        "result_shape": "safety_component_bundle" if phase1 else "efficacy_safety_bundle",
        "uncertainty_kind": "two_sided_confidence_interval",
        "confidence_level": 0.95,
        "horizon_source": "request.follow_up_days",
        "analysis_population": "all_randomized_participants",
        "censoring_assumption_id": "independent_censoring_conditional_on_randomized_arm",
        "loss_to_follow_up_construction_id": "arm_conditional_random_permutation_v1",
        "safety_component_ids": ["serious_ae", "discontinuation"],
    }


def _surface(
    tmp_path: Path,
    *,
    events: tuple[float, ...] = (1.0, 0.0),
    competing_events: tuple[float, ...] = (0.0, 0.0),
) -> tuple[Path, Path]:
    scenario = tmp_path / "scenario"
    output = tmp_path / "output"
    _write_json(
        scenario / "public" / "phase_decision_evidence_policy.json",
        {
            "schema_id": "trialdev_phase_decision_evidence_policy_v1",
            "confidence_level": 0.95,
        },
    )
    _write_json(
        scenario / "public" / "objective_charter.json",
        {"numeric_reporting_decimal_places": 3},
    )
    _write_json(
        scenario / "public" / "phase_analysis_method_catalog.json",
        _checksummed(
            {
                "schema_id": "trialdev_phase_analysis_method_catalog_v1",
                "version": "v1",
                "scenario_id": "s01",
                "confidence_level": 0.95,
                "methods": [_method(phase) for phase in ("phase1", "phase2", "phase3")],
            }
        ),
    )
    _write_json(
        output / "request.json",
        {
            "version": "v1",
            "scenario_id": "s01",
            "phase_id": "phase2",
            "candidate_drug_ids": ["drug_a"],
            "target_sample_size": 4,
            "endpoint_id": "E1",
            "follow_up_days": 10,
            "enrollment_window_days": 10,
            "site_count_budget": 1,
            "allocation_ratio": "1:1",
            "design_cell_id": "trialdev.phase2.fixed_final_operating_characteristics.v1",
            "treatment_discontinuation_strategy": "treatment_policy",
            "interim_policy": "fixed_final",
            "site_strategy": "high_enrolling",
            "selection_objective": "benefit_risk",
        },
    )
    _write_json(
        output / "arm_mapping.json",
        {
            "control_arm_id": "C",
            "candidate_arm_ids": ["T"],
            "drug_id_by_arm": {"C": "control", "T": "drug_a"},
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ARM": ["C", "C", "T", "T"],
            "TIME": [5.0, 10.0, 10.0, 10.0],
            "EVENT": [*events, 0.0, 0.0],
            "COMPETING_EVENT": [*competing_events, 0.0, 0.0],
            "TREATMENT_DISCONTINUATION_STRATEGY": ["treatment_policy"] * 4,
        }
    ).to_parquet(output / "endpoints.parquet", index=False)
    return scenario, output


def test_public_replay_recovers_known_cumulative_incidence_difference(tmp_path: Path) -> None:
    scenario, output = _surface(tmp_path)

    references = {
        record.estimator_id: record
        for record in derive_effect_references_v1(
            scenario_root=scenario,
            trial_output_root=output,
        )
    }

    reference = references["observed:aalen_johansen_cif_tau"]
    assert reference.estimate == pytest.approx(0.5)
    assert reference.estimand_id == "treatment_policy:cumulative_incidence_at_horizon"
    assert reference.effect_scale_id == "risk_difference_control_minus_treatment"
    assert set(dict(reference.source_checksums)) == {
        "trial_output/arm_mapping.json",
        "trial_output/endpoints.parquet",
        "trial_output/request.json",
    }


def test_public_replay_truth_and_provenance_change_with_source_data(tmp_path: Path) -> None:
    scenario, output = _surface(tmp_path)
    before = derive_effect_references_v1(scenario_root=scenario, trial_output_root=output)[0]
    endpoints = pd.read_parquet(output / "endpoints.parquet")
    endpoints.loc[2, ["TIME", "EVENT"]] = [4.0, 1.0]
    endpoints.to_parquet(output / "endpoints.parquet", index=False)

    after = derive_effect_references_v1(scenario_root=scenario, trial_output_root=output)[0]

    assert after.estimate != before.estimate
    assert after.checksum != before.checksum
    assert (
        dict(after.source_checksums)["trial_output/endpoints.parquet"]
        != dict(before.source_checksums)["trial_output/endpoints.parquet"]
    )


def test_public_replay_rejects_nonbinary_events(tmp_path: Path) -> None:
    scenario, output = _surface(tmp_path, events=(0.5, 0.0))

    with pytest.raises(ValueError, match="binary EVENT and COMPETING_EVENT"):
        derive_effect_references_v1(scenario_root=scenario, trial_output_root=output)


def test_public_replay_counts_terminal_events_as_competing_not_censoring(tmp_path: Path) -> None:
    scenario, output = _surface(tmp_path, events=(1.0, 0.0), competing_events=(0.0, 1.0))

    reference = derive_effect_references_v1(scenario_root=scenario, trial_output_root=output)[0]

    assert reference.estimate == pytest.approx(0.5)


def test_public_replay_rejects_overlapping_primary_and_competing_events(tmp_path: Path) -> None:
    scenario, output = _surface(tmp_path, events=(1.0, 0.0), competing_events=(1.0, 0.0))

    with pytest.raises(ValueError, match="mutually exclusive"):
        derive_effect_references_v1(scenario_root=scenario, trial_output_root=output)


def test_aalen_johansen_matches_hand_calculated_competing_risk_example() -> None:
    frame = pd.DataFrame(
        {
            "TIME": [1.0, 2.0, 3.0, 3.0],
            "EVENT": [0, 1, 0, 0],
            "COMPETING_EVENT": [1, 0, 0, 0],
        }
    )

    cif, variance = aalen_johansen_cif_variance_v1(frame=frame, horizon=3.0)

    assert cif == pytest.approx(0.25)
    assert 0.0 < variance < 1.0


def _direct_aalen_johansen_reference(frame: pd.DataFrame, horizon: float) -> tuple[float, float]:
    times = frame["TIME"].astype(float)
    events = frame["EVENT"].astype(int)
    competing = frame["COMPETING_EVENT"].astype(int)
    any_event = (events == 1) | (competing == 1)
    survival = 1.0
    cif = 0.0
    increments: list[tuple[float, float, int, int, int]] = []
    for event_time in sorted(set(float(value) for value in times[any_event & (times <= horizon)])):
        at_risk = int((times >= event_time).sum())
        primary_count = int(((times == event_time) & (events == 1)).sum())
        all_count = int(((times == event_time) & any_event).sum())
        survival_before = survival
        cif += survival_before * float(primary_count) / float(at_risk)
        survival *= 1.0 - float(all_count) / float(at_risk)
        increments.append((cif, survival_before, at_risk, primary_count, all_count))
    variance = 0.0
    for cif_after, survival_before, at_risk, primary_count, all_count in increments:
        remaining_cif = cif - cif_after
        if at_risk > all_count:
            variance += remaining_cif * remaining_cif * float(all_count) / float(at_risk * (at_risk - all_count))
        variance += (
            survival_before * survival_before * float(primary_count * (at_risk - primary_count)) / float(at_risk**3)
        )
        variance -= 2.0 * remaining_cif * survival_before * float(primary_count) / float(at_risk**2)
    return float(cif), max(0.0, float(variance))


@pytest.mark.parametrize("seed", range(10))
def test_vectorized_aalen_johansen_matches_direct_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    size = 200
    event_kind = rng.choice(3, size=size, p=(0.55, 0.30, 0.15))
    frame = pd.DataFrame(
        {
            "TIME": rng.integers(1, 31, size=size).astype(float),
            "EVENT": event_kind == 1,
            "COMPETING_EVENT": event_kind == 2,
        }
    )

    observed = aalen_johansen_cif_variance_v1(frame=frame, horizon=20.0)
    expected = _direct_aalen_johansen_reference(frame, 20.0)

    assert observed[0] == pytest.approx(expected[0], abs=1e-15)
    assert observed[1] == pytest.approx(expected[1], abs=1e-15)


def test_public_replay_rejects_arm_mapping_drift(tmp_path: Path) -> None:
    scenario, output = _surface(tmp_path)
    endpoints = pd.read_parquet(output / "endpoints.parquet")
    endpoints.loc[3, "ARM"] = "UNDECLARED"
    endpoints.to_parquet(output / "endpoints.parquet", index=False)

    with pytest.raises(ValueError, match="do not match arm_mapping"):
        derive_effect_references_v1(scenario_root=scenario, trial_output_root=output)


def test_interval_equivalence_requires_both_reference_endpoints() -> None:
    reference = interval_equivalence_score_v1(
        lower=-1.0,
        upper=1.0,
        reference_lower=-1.0,
        reference_upper=1.0,
        endpoint_tolerance=0.001,
    )
    rounded = interval_equivalence_score_v1(
        lower=-1.0005,
        upper=1.0005,
        reference_lower=-1.0,
        reference_upper=1.0,
        endpoint_tolerance=0.001,
    )
    wide = interval_equivalence_score_v1(
        lower=-2.0,
        upper=2.0,
        reference_lower=-1.0,
        reference_upper=1.0,
        endpoint_tolerance=0.001,
    )

    assert reference == 1.0
    assert rounded == 1.0
    assert wide == 0.0


def test_point_equivalence_uses_declared_reporting_precision_not_standard_error() -> None:
    tolerance = reporting_tolerance_v1(decimal_places=3)
    assert tolerance == 0.0005
    assert point_equivalence_score_v1(observed=2.0004, expected=2.0, absolute_tolerance=tolerance) == 1.0
    assert point_equivalence_score_v1(observed=2.0006, expected=2.0, absolute_tolerance=tolerance) == 0.0


def test_joint_numeric_equivalence_is_conjunctive() -> None:
    score = point_interval_equivalence_score_v1(
        estimate=0.0004,
        lower=-1.0,
        upper=1.001,
        reference_estimate=0.0,
        reference_lower=-1.0,
        reference_upper=1.0,
        absolute_tolerance=0.0005,
    )
    point = point_equivalence_score_v1(
        observed=0.0004,
        expected=0.0,
        absolute_tolerance=0.0005,
    )
    interval = interval_equivalence_score_v1(
        lower=-1.0,
        upper=1.001,
        reference_lower=-1.0,
        reference_upper=1.0,
        endpoint_tolerance=0.0005,
    )
    assert score == min(point, interval)
