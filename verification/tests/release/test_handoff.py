"""Tests for the neutral external-verification handoff contract."""

from __future__ import annotations

import hashlib
import json

import pytest

from trialagentbench_validation.contracts.external_verification_handoff import (
    ExternalVerificationResultHandoffV1,
    ExternalVerificationResultV1,
)


def _handoff() -> ExternalVerificationResultHandoffV1:
    result = ExternalVerificationResultV1(
        evidence_id="survival.process",
        scientific_family="survival process",
        artifact_manifest_path="verification/survival/artifact_manifest.json",
        artifact_manifest_sha256="1" * 64,
        result_status="qualified",
        reproducibility_class="public_replayable",
        supported_scope=("cox recovery", "kaplan-meier"),
        affected_components=("survival_generation",),
        required_qualification_ids=("survival_process",),
    )
    payload = {
        "schema_id": "trialagentbench.external_verification_result_handoff/v1",
        "candidate_id": "evh_" + "2" * 20,
        "validation_package_lock_sha256": "3" * 64,
        "results": [result.model_dump(mode="json")],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return ExternalVerificationResultHandoffV1(
        **payload,
        checksum=hashlib.sha256(encoded).hexdigest(),
    )


def test_handoff_requires_canonical_content_and_checksum() -> None:
    """A canonical handoff validates and detects content drift."""

    handoff = _handoff()
    assert handoff.results[0].result_status == "qualified"
    with pytest.raises(ValueError, match="checksum mismatch"):
        handoff.model_copy(update={"checksum": "0" * 64}).model_validate(
            handoff.model_dump(mode="python") | {"checksum": "0" * 64}
        )


def test_handoff_rejects_escaping_manifest_path() -> None:
    """Artifact references cannot escape the evidence root."""

    with pytest.raises(ValueError, match="confined relative"):
        ExternalVerificationResultV1(
            evidence_id="survival.process",
            scientific_family="survival process",
            artifact_manifest_path="../artifact_manifest.json",
            artifact_manifest_sha256="1" * 64,
            result_status="qualified",
            reproducibility_class="public_replayable",
            supported_scope=("kaplan-meier",),
            affected_components=("survival_generation",),
            required_qualification_ids=("survival_process",),
        )
