from __future__ import annotations

from pathlib import Path

from trialagentbench_harness.contracts.trialdev.trialdev_public_phase_replay import (
    TrialDevPhaseReplayCaseV1,
)
from trialagentbench_harness.trialdev.grading.sequential import (
    _copy_fixed_phase_evidence_v1,
)
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1


def _request(*, sample_size: int) -> TrialDevelopmentRequestV1:
    return TrialDevelopmentRequestV1(
        scenario_id="s01",
        phase_id="phase2",
        candidate_drug_ids=("drug_a",),
        target_sample_size=sample_size,
        endpoint_id="E1",
        follow_up_days=90,
        enrollment_window_days=42,
        site_count_budget=8,
        allocation_ratio="1:1",
        design_cell_id="trialdev.phase2.fixed_final_operating_characteristics.v1",
        treatment_discontinuation_strategy="treatment_policy",
        interim_policy="fixed_final",
        site_strategy="high_enrolling",
        selection_objective="benefit_risk",
    )


def test_fixed_phase_evidence_is_selected_by_asset_and_phase_not_proposed_sample_size(
    tmp_path: Path,
) -> None:
    scenario = tmp_path / "scenario_s01"
    trajectory_root = tmp_path / "fixed_trajectories"
    canonical = _request(sample_size=120)
    case = TrialDevPhaseReplayCaseV1(
        scenario_root="scenario_s01",
        world_seed=101,
        program_objective_ids=("benefit_risk",),
        request=canonical,
    )
    trajectory_root.mkdir()
    (trajectory_root / "cases.jsonl").write_text(case.model_dump_json() + "\n", encoding="utf-8")
    source = trajectory_root / "materialized" / "world_101" / f"request_{canonical.checksum()}" / "trial_seed_2026"
    source.mkdir(parents=True)
    (source / "marker.txt").write_text("fixed evidence\n", encoding="utf-8")

    checksum, world_seed, trial_seed = _copy_fixed_phase_evidence_v1(
        scenario_root=scenario,
        request=_request(sample_size=180),
        out_dir=tmp_path / "selected",
    )

    assert checksum == canonical.checksum()
    assert world_seed == 101
    assert trial_seed == 2026
    assert (tmp_path / "selected" / "marker.txt").read_text(encoding="utf-8") == "fixed evidence\n"
