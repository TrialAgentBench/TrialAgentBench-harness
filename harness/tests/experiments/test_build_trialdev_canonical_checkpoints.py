"""Tests for deterministic TrialDev canonical checkpoint capture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trialagentbench_harness.contracts.experiments import (
    TrialDevCanonicalCheckpointSourcesV1,
    TrialDevCheckpointBlockPlanV1,
    TrialDevCheckpointSchedulePlanV1,
)
from trialagentbench_harness.experiments import build_trialdev_canonical_checkpoints as builder
from trialagentbench_harness.io import read_json_model


def _block(**updates: object) -> TrialDevCheckpointBlockPlanV1:
    payload: dict[str, object] = {
        "block_id": "block-1",
        "program_id": "scenario-1__benefit_risk",
        "scenario_id": "scenario-1",
        "objective_id": "benefit_risk",
        "replicate_id": "replicate-1",
        "decoding_seed": 17,
        "checkpoint_phase_id": "phase2",
        "checkpoint_step_id": "trial_design_request",
        "endogenous_program_relative_path": "endogenous/programs/scenario-1__benefit_risk",
        "canonical_program_relative_path": "canonical/programs/scenario-1__benefit_risk",
        "canonical_reference_id": "canonical-reference-1",
    }
    payload.update(updates)
    return TrialDevCheckpointBlockPlanV1.model_validate(payload)


def _write_observational_submission(reference: Path) -> None:
    source = reference / "obs_review" / "agent_obs_review_payload.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "response_branch": "estimable",
                "primary_resolution_evidence_class": "empirical_diagnosis",
                "ranked_drug_ids": ["drug-a"],
                "candidate_utility_estimates": [
                    {
                        "evidence_id": "utility-drug-a",
                        "method_route_id": "method-a",
                        "candidate_drug_id": "drug-a",
                        "objective_id": "benefit_risk",
                        "estimator_id": "estimator-a",
                        "utility_unit": "dimensionless_declared_net_benefit",
                        "estimate": 0.2,
                        "lower": 0.1,
                        "upper": 0.3,
                        "confidence_level": 0.95,
                        "analysis_covariate_ids": ["age"],
                        "source_artifact_checksums": {"public/data.parquet": "a" * 64},
                    }
                ],
                "supporting_evidence_ids": ["utility-drug-a"],
                "candidate_drug_id": "drug-a",
                "decision_action": "nominate_for_early_study",
                "decision_rationale": "The public estimate supports nomination.",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_builder_writes_checksum_bound_source_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participant = tmp_path / "participant"
    references = tmp_path / "references"
    output = tmp_path / "sources"
    participant.mkdir()
    references.mkdir()
    checkpoint = output / "canonical" / "programs" / "scenario-1__benefit_risk" / "checkpoints" / "00000001.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> Path:
        captured.update(kwargs)
        return checkpoint

    monkeypatch.setattr(builder, "_capture_block", capture)
    monkeypatch.setattr(
        builder,
        "sha256_dir_digest",
        lambda path: "a" * 64 if path == participant.resolve() else "b" * 64,
    )
    plan = TrialDevCheckpointSchedulePlanV1(
        experiment_id="experiment",
        blocks=(_block(),),
    )

    receipt = builder.build_trialdev_canonical_checkpoints_v1(
        participant_root=participant,
        reference_root=references,
        checkpoint_root=output,
        plan=plan,
        max_tokens=8192,
        max_turns_per_step=41,
        program_watchdog_seconds=2700,
    )

    stored = read_json_model(
        TrialDevCanonicalCheckpointSourcesV1,
        output / "canonical_checkpoint_sources.json",
    )
    assert stored == receipt
    assert stored.checksum is not None
    assert stored.records[0].checkpoint_relative_path.endswith("checkpoints/00000001.json")
    assert captured["max_tokens"] == 8192
    assert captured["max_turns_per_step"] == 41
    assert captured["program_watchdog_seconds"] == 2700


def test_recorded_provider_stages_then_submits_scratch_relative_file(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    _write_observational_submission(reference)
    provider = builder.RecordedTrialDevProvider(
        reference_program=reference,
        scenario_root=tmp_path / "scenario",
        program_workdir=tmp_path / "program_workdir",
    )

    tools = [
        {
            "type": "function",
            "function": {"name": "submit_obs_review_analysis_and_decision_file"},
        },
        {
            "type": "function",
            "function": {"name": "execute_code"},
        },
    ]
    stage = provider.generate_turn(
        [],
        tools,
        temperature=0.0,
        max_tokens=1,
    )
    submit = provider.generate_turn([], tools, temperature=0.0, max_tokens=1)

    assert len(stage.tool_calls) == 1
    assert stage.tool_calls[0].name == "execute_code"
    assert "scratch/canonical_reference/observational_review.json" in stage.tool_calls[0].arguments
    assert len(submit.tool_calls) == 1
    assert submit.tool_calls[0].arguments == '{"path": "canonical_reference/observational_review.json"}'


def test_recorded_provider_rejects_stale_phase_contract_before_replay(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    _write_observational_submission(reference)
    phase = reference / "agent_workdir" / "phase_phase1"
    phase.mkdir(parents=True)
    (phase / "request.json").write_text(
        json.dumps(
            {
                "scenario_id": "scenario-1",
                "phase_id": "phase1",
                "candidate_drug_ids": ["drug-a"],
                "eligibility_filters": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="eligibility_filters"):
        builder.RecordedTrialDevProvider(
            reference_program=reference,
            scenario_root=tmp_path / "scenario",
            program_workdir=tmp_path / "program_workdir",
        )


def test_builder_rejects_one_reference_for_different_checkpoints(
    tmp_path: Path,
) -> None:
    participant = tmp_path / "participant"
    references = tmp_path / "references"
    participant.mkdir()
    references.mkdir()
    plan = TrialDevCheckpointSchedulePlanV1(
        experiment_id="experiment",
        blocks=(
            _block(),
            _block(
                block_id="block-2",
                replicate_id="replicate-2",
                decoding_seed=18,
                checkpoint_phase_id="phase3",
            ),
        ),
    )

    with pytest.raises(ValueError, match="cannot describe multiple source checkpoints"):
        builder.build_trialdev_canonical_checkpoints_v1(
            participant_root=participant,
            reference_root=references,
            checkpoint_root=tmp_path / "sources",
            plan=plan,
        )
