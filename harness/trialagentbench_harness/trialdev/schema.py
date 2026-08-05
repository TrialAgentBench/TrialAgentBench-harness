"""Internal dataclasses for harness bookkeeping.

These are *not* the on-disk submission schemas (those are pydantic models in
the upstream packages). These types track per-run state inside our orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PhaseId = Literal["observational_review", "phase1", "phase2", "phase3"]
ProgramExecutionStatus = Literal[
    "running",
    "completed",
    "model_turn_limit",
    "model_invalid_submission",
    "infrastructure_timeout",
    "infrastructure_error",
]
_MATERIALIZING_PHASES: tuple[PhaseId, ...] = ("phase1", "phase2", "phase3")


@dataclass(frozen=True)
class BenchmarkItem:
    """One scoring checkpoint as listed in a suite items manifest."""

    item_id: str
    scenario_id: str
    phase_id: PhaseId
    objective_id: str
    endpoint_id: str | None
    task_definition_id: str
    allowed_endpoint_ids: tuple[str, ...] = field(default_factory=tuple)
    allowed_follow_up_days: tuple[int, ...] = field(default_factory=tuple)
    allowed_enrollment_window_days: tuple[int, ...] = field(default_factory=tuple)
    allowed_site_count_budgets: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Program:
    """An ordered sequence of phase checkpoints for one (scenario, objective).

    Programs are the unit the agent runs end-to-end. Items present in the
    manifest at each phase are listed under ``items_by_phase``; a phase may
    have zero items (skipped), one item, or multiple (different endpoints —
    the agent picks one and we score against the matching item).
    """

    program_id: str  # canonical: "{scenario_id}__{objective_id}"
    scenario_id: str
    objective_id: str
    items_by_phase: dict[PhaseId, tuple[BenchmarkItem, ...]]

    @property
    def has_obs_review(self) -> bool:
        return bool(self.items_by_phase.get("observational_review"))

    def materializing_phases(self) -> tuple[PhaseId, ...]:
        return tuple(phase for phase in _MATERIALIZING_PHASES if self.items_by_phase.get(phase))


@dataclass
class MaterializationRecord:
    """One call to ``materialize_phase_v1`` and what came back."""

    phase_id: PhaseId
    seed: int
    request_path: Path
    request_checksum: str
    trial_output_root: Path
    trial_output_checksum: str
    next_state_path: Path | None = None


@dataclass
class MaterializationUsage:
    """Observed per-program materialization calls."""

    materialize_calls_by_phase: dict[str, int] = field(default_factory=dict)

    def record(self, phase_id: str) -> None:
        """Record one successful phase materialization."""

        self.materialize_calls_by_phase[phase_id] = int(self.materialize_calls_by_phase.get(phase_id, 0)) + 1


@dataclass
class PhaseAttempt:
    """Per-phase artefacts for one attempt within a program run."""

    phase_id: PhaseId
    matched_item_id: str | None = None
    request_path: Path | None = None
    trial_output_root: Path | None = None
    analysis_path: Path | None = None
    decision_path: Path | None = None
    grade_report_path: Path | None = None
    decision_action: str | None = None
    advance: bool | None = None
    candidate_drug_id: str | None = None
    materializations: list[MaterializationRecord] = field(default_factory=list)


@dataclass
class ProgramRun:
    """Top-level result record for one program run."""

    program_id: str
    scenario_id: str
    objective_id: str
    workdir: Path
    obs_review_grade_path: Path | None = None
    phases: list[PhaseAttempt] = field(default_factory=list)
    trajectory_grade_path: Path | None = None
    final_program_grade_path: Path | None = None
    started_at_utc: str | None = None
    ended_at_utc: str | None = None
    wall_seconds_total: float | None = None
    stopped_at_phase: PhaseId | None = None  # None means full traversal
    execution_status: ProgramExecutionStatus = "running"
    error: str | None = None
    violations: list[dict] = field(default_factory=list)
