"""Contracts for design-bound TrialEval experiment samples."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_harness.io.checksums import canonical_payload_sha256


class TrialEvalFactorialTaskSampleV1(BaseModel):
    """Opaque participant task sample selected from complete evaluator blocks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.trialeval_factorial_task_sample/v1"] = (
        "trialagentbench.trialeval_factorial_task_sample/v1"
    )
    experiment_design_sha256: str = Field(..., min_length=64, max_length=64)
    participant_release_sha256: str = Field(..., min_length=64, max_length=64)
    evaluator_labels_sha256: str = Field(..., min_length=64, max_length=64)
    selection_method: Literal["stratified_regime_cell_omission_v1"] = "stratified_regime_cell_omission_v1"
    task_ids: tuple[str, ...] = Field(..., min_length=1)
    context_allocation: tuple[int, int, int, int, int]
    checksum: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _canonicalize_and_hash(self) -> TrialEvalFactorialTaskSampleV1:
        ordered = tuple(sorted(self.task_ids))
        if len(ordered) != len(set(ordered)):
            raise ValueError("Factorial task sample contains duplicate task IDs.")
        if any(count < 1 for count in self.context_allocation):
            raise ValueError("Factorial task sample must retain every context stratum.")
        if sum(self.context_allocation) != len(ordered):
            raise ValueError("Factorial task sample context allocation differs from its task count.")
        object.__setattr__(self, "task_ids", ordered)
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum", None)
        digest = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Factorial task-sample checksum does not match its payload.")
        object.__setattr__(self, "checksum", digest)
        return self


__all__ = ["TrialEvalFactorialTaskSampleV1"]
