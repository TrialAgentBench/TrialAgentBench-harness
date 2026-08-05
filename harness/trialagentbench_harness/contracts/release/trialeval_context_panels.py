"""Independent contracts for TrialEval context-sibling panels."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _checksum(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


class TrialEvalContextPanelTaskV1(BaseModel):
    """One public context view of an underlying trial realization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    variant_id: str = Field(min_length=1)


class TrialEvalContextPanelV1(BaseModel):
    """One checksummed panel of context views sharing scorer reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    panel_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    design_tier: Literal["D1", "D2", "D3", "D4"]
    design_subtype: str = Field(min_length=1)
    assumption_tier: Literal["A1", "A2", "A3", "A4"]
    reference_signature_digest: str = Field(min_length=16, max_length=64)
    tasks: tuple[TrialEvalContextPanelTaskV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_panel(self) -> TrialEvalContextPanelV1:
        """Validate ordering, uniqueness, and checksum."""

        if self.tasks != tuple(sorted(self.tasks, key=lambda task: (task.context_tier, task.task_id))):
            raise ValueError("Context panel tasks must be sorted by context tier and task id.")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Context panel task identifiers must be unique.")
        if len({task.context_tier for task in self.tasks}) != len(self.tasks):
            raise ValueError("Context panel tiers must be unique.")
        if self.checksum != _checksum(self.model_dump(mode="json", exclude={"checksum"})):
            raise ValueError("Context panel checksum mismatch.")
        return self


class TrialEvalContextPanelManifestV1(BaseModel):
    """Checksummed evaluator manifest of context-sibling panels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[
        "trialagentbench.trial_benchmark.context_panel_manifest/v1",
        "trialagentbench.trialeval.context_panel_manifest/v1",
    ]
    version: Literal["v1"]
    panels: tuple[TrialEvalContextPanelV1, ...]
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_manifest(self) -> TrialEvalContextPanelManifestV1:
        """Validate panel ordering, task ownership, and checksum."""

        if self.panels != tuple(sorted(self.panels, key=lambda panel: panel.panel_id)):
            raise ValueError("Context panels must be sorted by panel_id.")
        task_ids = [task.task_id for panel in self.panels for task in panel.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Each task must belong to exactly one context panel.")
        if self.checksum != _checksum(self.model_dump(mode="json", exclude={"checksum"})):
            raise ValueError("Context panel manifest checksum mismatch.")
        return self


__all__ = [
    "TrialEvalContextPanelManifestV1",
    "TrialEvalContextPanelTaskV1",
    "TrialEvalContextPanelV1",
]
