"""Tests for the public TrialEval experiment-protocol boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trialagentbench_harness.contracts.experiments.trialeval_design import TrialEvalExperimentProtocolV1

_HARNESS_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL_PATH = _HARNESS_ROOT / "experiment_configs" / "trialeval_experiment_protocol_v1.json"


def test_standalone_harness_parses_the_frozen_experiment_protocol() -> None:
    protocol = TrialEvalExperimentProtocolV1.model_validate_json(_PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert protocol.precision.regime_cell_count == 25
    assert protocol.compute_envelope.participant_item_count == 500
    assert protocol.compute_envelope.factorial_item_count == 100
    assert protocol.compute_envelope.factorial_assignments_per_model == 1_200
    assert protocol.precision.retained_independent_base_trials == 100
    assert {row.contrast_id for row in protocol.contrasts if row.contrast_kind == "context"} == {
        "C1-C2",
        "C3-C1",
        "C3-C4",
        "C4-C2",
        "C5-C4",
    }


def test_standalone_harness_rejects_protocol_tampering() -> None:
    payload = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["precision"]["retained_independent_base_trials"] = 111

    with pytest.raises(ValidationError, match="Retained base-trial|checksum"):
        TrialEvalExperimentProtocolV1.model_validate(payload)


def test_standalone_harness_rejects_missing_context_contrast() -> None:
    payload = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["contrasts"] = [contrast for contrast in payload["contrasts"] if contrast["contrast_id"] != "C5-C4"]

    with pytest.raises(ValidationError, match="five prespecified context contrasts"):
        TrialEvalExperimentProtocolV1.model_validate(payload)


def test_public_protocol_excludes_construction_and_answer_fields() -> None:
    payload = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)

    assert "rows" not in payload
    for forbidden in (
        "assumption_states",
        "charter_checksum",
        "construction_checksum",
        "credit_eligible_primary_effect_scales",
        "default_invalidity",
        "default_method_id",
        "n_subjects_range",
        "n_visits_range",
        "reference_method_id",
    ):
        assert forbidden not in serialized
