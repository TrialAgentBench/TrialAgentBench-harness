"""Runner-owned append-only custody for exact TrialDev continuation."""

from __future__ import annotations

from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue

from trialagentbench_harness.contracts.trialdev.runtime_checkpoint import (
    TrialDevCheckpointArtifactV1,
    TrialDevContinuationCheckpointV1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256


class TrialDevRunCheckpointPhaseV1(BaseModel):
    """Custody-bound artifacts produced while executing one phase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase_id: Literal["phase1", "phase2", "phase3"]
    request: TrialDevCheckpointArtifactV1 | None = None
    trial_output: TrialDevCheckpointArtifactV1 | None = None
    materialization_seed: int | None = None
    request_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    trial_output_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    analysis: TrialDevCheckpointArtifactV1 | None = None
    decision: TrialDevCheckpointArtifactV1 | None = None
    matched_item_id: str | None = None
    decision_action: str | None = None
    advance: bool | None = None
    candidate_drug_id: str | None = None

    @model_validator(mode="after")
    def validate_materialization_record(self) -> Self:
        """Require complete materialization custody or no materialization fields."""

        values = (
            self.trial_output,
            self.materialization_seed,
            self.request_checksum,
            self.trial_output_checksum,
        )
        if any(value is not None for value in values) and any(value is None for value in values):
            raise ValueError("TrialDev checkpoint materialization custody must be complete.")
        return self


class TrialDevRunCheckpointPayloadV1(BaseModel):
    """Complete runner and AgentLoop state at one durable continuation boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_run_checkpoint_payload/v1"] = (
        "trialagentbench.trialdev_run_checkpoint_payload/v1"
    )
    sequence: int = Field(..., ge=0)
    previous_checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    run_identity_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    pending_operation: Literal[
        "observational_review",
        "phase_request",
        "materialize",
        "phase_analysis",
        "phase_decision",
        "advance_state",
    ]
    continuation: TrialDevContinuationCheckpointV1
    completed_phases: tuple[TrialDevRunCheckpointPhaseV1, ...] = ()
    current_phase: TrialDevRunCheckpointPhaseV1 | None = None

    @model_validator(mode="after")
    def validate_chain_and_phase(self) -> Self:
        """Require a contiguous chain and operation-consistent current phase."""

        if (self.sequence == 0) != (self.previous_checkpoint_sha256 is None):
            raise ValueError("TrialDev checkpoint predecessor must match its sequence.")
        phase_operations = {
            "phase_request",
            "materialize",
            "phase_analysis",
            "phase_decision",
            "advance_state",
        }
        if (self.pending_operation in phase_operations) != (self.current_phase is not None):
            raise ValueError("TrialDev phase operations require exactly one current phase.")
        completed_ids = [phase.phase_id for phase in self.completed_phases]
        if len(completed_ids) != len(set(completed_ids)):
            raise ValueError("TrialDev completed checkpoint phases must be unique.")
        return self


class TrialDevRunCheckpointV1(BaseModel):
    """Checksum-bearing envelope for one append-only runner checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialdev_run_checkpoint/v1"] = "trialagentbench.trialdev_run_checkpoint/v1"
    payload: TrialDevRunCheckpointPayloadV1
    payload_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, payload: TrialDevRunCheckpointPayloadV1) -> TrialDevRunCheckpointV1:
        """Create a checksum-bearing runner checkpoint."""

        digest = canonical_payload_sha256(cast(JsonValue, payload.model_dump(mode="json")))
        return cls(payload=payload, payload_sha256=digest)

    @model_validator(mode="after")
    def validate_payload_checksum(self) -> Self:
        """Reject any mutation of runner continuation custody."""

        observed = canonical_payload_sha256(cast(JsonValue, self.payload.model_dump(mode="json")))
        if observed != self.payload_sha256:
            raise ValueError("TrialDev runner checkpoint payload checksum mismatch.")
        return self


__all__ = [
    "TrialDevRunCheckpointPayloadV1",
    "TrialDevRunCheckpointPhaseV1",
    "TrialDevRunCheckpointV1",
]
