"""Tests for model-produced TrialDev checkpoint-source capture."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.experiments import (
    TrialDevCheckpointBlockPlanV1,
    TrialDevCheckpointSchedulePlanV1,
    TrialDevEndogenousCheckpointSourceV1,
)
from trialagentbench_harness.experiments import (
    capture_trialdev_endogenous_checkpoints as capture,
)
from trialagentbench_harness.trialdev.schema import Program


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
        "endogenous_program_relative_path": ("endogenous/reference/programs/scenario-1__benefit_risk"),
        "canonical_program_relative_path": ("canonical/reference/programs/scenario-1__benefit_risk"),
        "canonical_reference_id": "canonical-reference-1",
    }
    payload.update(updates)
    return TrialDevCheckpointBlockPlanV1.model_validate(payload)


def _record(block: TrialDevCheckpointBlockPlanV1) -> TrialDevEndogenousCheckpointSourceV1:
    return TrialDevEndogenousCheckpointSourceV1(
        program_id=block.program_id,
        scenario_id=block.scenario_id,
        objective_id=block.objective_id,
        replicate_id=block.replicate_id,
        decoding_seed=block.decoding_seed,
        phase_id=block.checkpoint_phase_id,
        step_id=block.checkpoint_step_id,
        program_relative_path=block.endogenous_program_relative_path,
        checkpoint_relative_path=(f"{block.endogenous_program_relative_path}/checkpoints/00000006.json"),
        checkpoint_sha256="a" * 64,
        run_identity_sha256="b" * 64,
        provider_model="model",
        provider_route="provider:route",
        procedure_assistance="output_contract_only",
    )


def test_endogenous_receipt_rejects_checkpoint_outside_programme() -> None:
    payload = _record(_block()).model_dump(mode="json")
    payload["checkpoint_relative_path"] = "elsewhere/checkpoints/00000006.json"
    with pytest.raises(ValidationError, match="must be below"):
        TrialDevEndogenousCheckpointSourceV1.model_validate(payload)


def test_capture_rejects_one_path_for_different_semantic_sources(
    tmp_path: Path,
) -> None:
    participant = tmp_path / "participant"
    participant.mkdir()
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

    with pytest.raises(ValueError, match="cannot describe multiple checkpoints"):
        capture.capture_trialdev_endogenous_checkpoints_v1(
            participant_root=participant,
            checkpoint_root=tmp_path / "sources",
            plan=plan,
            provider_factory=lambda seed: SimpleNamespace(),
            model="model",
        )


def test_capture_writes_participant_bound_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participant = tmp_path / "participant"
    participant.mkdir()
    output = tmp_path / "sources"
    block = _block()
    plan = TrialDevCheckpointSchedulePlanV1(
        experiment_id="experiment",
        blocks=(block,),
    )
    monkeypatch.setattr(
        capture,
        "resolve_executor_environment",
        lambda: SimpleNamespace(image_id=None, limits=SimpleNamespace()),
    )
    monkeypatch.setattr(capture, "_implementation_sha256", lambda: "c" * 64)
    monkeypatch.setattr(capture, "sha256_dir_digest", lambda path: "d" * 64)
    monkeypatch.setattr(capture, "_capture_block", lambda **kwargs: _record(kwargs["block"]))

    receipt = capture.capture_trialdev_endogenous_checkpoints_v1(
        participant_root=participant,
        checkpoint_root=output,
        plan=plan,
        provider_factory=lambda seed: SimpleNamespace(),
        model="model",
    )

    assert receipt.participant_release_sha256 == "d" * 64
    assert receipt.records == (_record(block),)
    assert (output / "endogenous_checkpoint_sources.json").is_file()


def test_capture_rejects_provider_model_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participant = tmp_path / "participant"
    participant.mkdir()
    program = Program(
        program_id="scenario-1__benefit_risk",
        scenario_id="scenario-1",
        objective_id="benefit_risk",
        items_by_phase={},
    )
    monkeypatch.setattr(capture, "discover_programs", lambda root: [program])

    with pytest.raises(ValueError, match="model identity"):
        capture._capture_block(
            participant_root=participant,
            checkpoint_root=tmp_path / "sources",
            block=_block(),
            provider_factory=lambda seed: SimpleNamespace(
                model="different-model",
                telemetry_route="provider:route",
            ),
            model="model",
            procedure_assistance="output_contract_only",
            master_seed=42,
            max_tokens=4096,
            max_turns_per_step=20,
            request_timeout_seconds=300.0,
            program_watchdog_seconds=1800,
            executor_image=None,
            executor_limits=SimpleNamespace(),
            implementation_sha256="c" * 64,
        )
