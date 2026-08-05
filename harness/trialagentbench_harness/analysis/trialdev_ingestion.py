"""Canonical typed ingestion for TrialDev publication analysis."""

from __future__ import annotations

import csv
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic.types import JsonValue

from trialagentbench_harness.analysis.run_identity import require_unique_run_ids
from trialagentbench_harness.contracts.core.runs import (
    TrialDevChainSummaryV1,
    TrialDevObsReviewSummaryV1,
    TrialDevPhaseStepSummaryV1,
    TrialDevRunConfigV1,
)
from trialagentbench_harness.contracts.trace.observable import TrialDevPhaseOutcomeRowV1
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevGradeRecordV1,
    TrialDevProgrammeAnalysisRowV1,
    TrialDevTrajectoryGradeV1,
)
from trialagentbench_harness.io import canonical_payload_sha256, read_json_model
from trialagentbench_harness.trialdev.grade_wrappers import (
    summarise_programme_analysis_quality,
)
from trialagentbench_harness.trialdev.grading.models import (
    TrialDevelopmentSubmissionV1,
    TrialDevPhaseResourceConsequenceV1,
)
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1
from trialagentbench_harness.util.provider_telemetry import (
    read_provider_terminal_events_v1,
)

TrialDevPhaseId = Literal["observational_review", "phase1", "phase2", "phase3"]
PHASE_ORDER: tuple[TrialDevPhaseId, ...] = ("observational_review", "phase1", "phase2", "phase3")
TrialDevProgrammeKey = tuple[str, str, str]
TrialDevPhaseKey = tuple[str, str, str, TrialDevPhaseId]
TrialDevPopulationName = Literal[
    "all_programme",
    "completed",
    "entered_phase",
    "phase_analysis_eligible",
    "design_eligible",
    "paired_assistance",
    "observable_trace",
]


class TrialDevAnalysisSourceError(RuntimeError):
    """Raised when persisted TrialDev sources cannot support valid analysis."""


@dataclass(frozen=True)
class TrialDevAnalysisPopulations:
    """Explicit, validated TrialDev analysis populations."""

    all_programme: frozenset[TrialDevProgrammeKey]
    completed: frozenset[TrialDevProgrammeKey]
    entered_phase: frozenset[TrialDevPhaseKey]
    phase_analysis_eligible: frozenset[TrialDevPhaseKey]
    design_eligible: frozenset[TrialDevPhaseKey]
    paired_assistance: frozenset[TrialDevProgrammeKey]
    observable_trace: frozenset[TrialDevPhaseKey]

    def keys(
        self, population: TrialDevPopulationName
    ) -> frozenset[TrialDevProgrammeKey] | frozenset[TrialDevPhaseKey]:
        """Return keys for one declared analysis population."""

        if population == "all_programme":
            return self.all_programme
        if population == "completed":
            return self.completed
        if population == "entered_phase":
            return self.entered_phase
        if population == "phase_analysis_eligible":
            return self.phase_analysis_eligible
        if population == "design_eligible":
            return self.design_eligible
        if population == "paired_assistance":
            return self.paired_assistance
        return self.observable_trace

    def count(self, population: TrialDevPopulationName, *, model_id: str | None = None) -> int:
        """Count units in a population, optionally within one model."""

        keys = self.keys(population)
        return len(keys) if model_id is None else sum(key[0] == model_id for key in keys)

    def phase_count(self, population: TrialDevPopulationName, *, model_id: str, phase_id: str) -> int:
        """Count phase-keyed units for one model and phase."""

        if population == "entered_phase":
            keys = self.entered_phase
        elif population == "phase_analysis_eligible":
            keys = self.phase_analysis_eligible
        elif population == "design_eligible":
            keys = self.design_eligible
        elif population == "observable_trace":
            keys = self.observable_trace
        else:
            raise ValueError(f"{population} is a programme-level population.")
        return sum(key[0] == model_id and key[3] == phase_id for key in keys)


@dataclass(frozen=True)
class TrialDevAnalysisDataset:
    """Canonical programme and phase rows loaded from persisted contracts."""

    programmes: tuple[TrialDevProgrammeAnalysisRowV1, ...]
    phases: tuple[TrialDevPhaseOutcomeRowV1, ...]


@dataclass(frozen=True)
class _ProgrammeProviderTelemetry:
    response_count: int
    responses_with_usage: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float


def _programme_provider_telemetry(
    program_dir: Path,
    *,
    program_id: str,
) -> _ProgrammeProviderTelemetry:
    path = program_dir / "provider_responses.jsonl"
    if not path.is_file():
        raise TrialDevAnalysisSourceError(f"TrialDev programme lacks provider telemetry: {path}")
    try:
        events = read_provider_terminal_events_v1(path)
    except ValueError as exc:
        raise TrialDevAnalysisSourceError(f"Invalid TrialDev provider telemetry: {path}: {exc}") from exc
    if not events:
        raise TrialDevAnalysisSourceError(f"TrialDev programme provider telemetry is empty: {path}")
    invalid_identity = [
        event.request_id for event in events if event.benchmark != "trialdev" or event.unit_id != program_id
    ]
    if invalid_identity:
        raise TrialDevAnalysisSourceError(
            f"TrialDev provider telemetry has invalid programme identity: {invalid_identity!r}"
        )
    failed = [event.request_id for event in events if event.status == "failed"]
    if failed:
        raise TrialDevAnalysisSourceError(
            f"Analysable TrialDev programme contains terminal provider failures: {failed!r}"
        )
    return _ProgrammeProviderTelemetry(
        response_count=len(events),
        responses_with_usage=sum(event.usage_status == "reported" for event in events),
        prompt_tokens=sum(event.prompt_tokens for event in events),
        completion_tokens=sum(event.completion_tokens for event in events),
        total_tokens=sum(event.total_tokens for event in events),
        elapsed_seconds=sum(float(event.elapsed_seconds or 0.0) for event in events),
    )


def trialdev_programme_key(row: TrialDevProgrammeAnalysisRowV1) -> TrialDevProgrammeKey:
    """Return the canonical identity for one TrialDev programme."""

    return row.model_id, row.run_id, row.program_id


def trialdev_phase_key(row: TrialDevPhaseOutcomeRowV1) -> TrialDevPhaseKey:
    """Return the canonical identity for one TrialDev phase."""

    return row.model_id, row.run_id, row.program_id, row.phase_id


def build_trialdev_analysis_populations(
    dataset: TrialDevAnalysisDataset,
    *,
    paired_assistance_keys: Collection[TrialDevProgrammeKey] = (),
    observable_trace_keys: Collection[TrialDevPhaseKey] = (),
) -> TrialDevAnalysisPopulations:
    """Build named populations from one canonical TrialDev dataset."""

    programme_by_key: dict[TrialDevProgrammeKey, TrialDevProgrammeAnalysisRowV1] = {}
    for programme_row in dataset.programmes:
        programme_key = trialdev_programme_key(programme_row)
        if programme_key in programme_by_key:
            raise ValueError(f"Duplicate TrialDev programme key: {programme_key!r}.")
        programme_by_key[programme_key] = programme_row
    phase_by_key: dict[TrialDevPhaseKey, TrialDevPhaseOutcomeRowV1] = {}
    for phase_row in dataset.phases:
        phase_key = trialdev_phase_key(phase_row)
        if phase_key in phase_by_key:
            raise ValueError(f"Duplicate TrialDev phase key: {phase_key!r}.")
        if phase_key[:3] not in programme_by_key:
            raise ValueError(f"TrialDev phase has no canonical programme row: {phase_key!r}.")
        phase_by_key[phase_key] = phase_row

    all_programme = frozenset(programme_by_key)
    expected_phase_keys = {(*programme_key, phase_id) for programme_key in all_programme for phase_id in PHASE_ORDER}
    if set(phase_by_key) != expected_phase_keys:
        raise ValueError(
            "TrialDev canonical phase surface must contain every declared programme-phase key: "
            f"missing={sorted(expected_phase_keys - set(phase_by_key))!r}, "
            f"unexpected={sorted(set(phase_by_key) - expected_phase_keys)!r}."
        )
    paired = frozenset(paired_assistance_keys)
    traces = frozenset(observable_trace_keys)
    unknown_paired = sorted(paired - all_programme)
    unknown_traces = sorted(traces - set(phase_by_key))
    if unknown_paired:
        raise ValueError(f"TrialDev paired-assistance population contains unknown programmes: {unknown_paired!r}.")
    if unknown_traces:
        raise ValueError(f"TrialDev observable-trace population contains unknown phases: {unknown_traces!r}.")

    entered = frozenset(key for key, row in phase_by_key.items() if row.phase_reached)
    randomized_entered = frozenset(key for key in entered if key[3] != "observational_review")
    return TrialDevAnalysisPopulations(
        all_programme=all_programme,
        completed=frozenset(key for key, row in programme_by_key.items() if row.completed),
        entered_phase=entered,
        phase_analysis_eligible=entered,
        design_eligible=randomized_entered,
        paired_assistance=paired,
        observable_trace=traces,
    )


def _phase_rank(phase_id: str | None) -> int:
    if phase_id is None:
        return -1
    try:
        return PHASE_ORDER.index(phase_id)
    except ValueError:
        return -1


def _phase_summary(program_dir: Path, phase_id: TrialDevPhaseId) -> TrialDevPhaseStepSummaryV1 | None:
    if phase_id == "observational_review":
        return None
    path = program_dir / "agent_workdir" / f"phase_{phase_id}" / "phase_step_summary.json"
    return read_json_model(TrialDevPhaseStepSummaryV1, path) if path.is_file() else None


def _canonical_phase_request(
    program_dir: Path,
    phase_id: TrialDevPhaseId,
    summary: TrialDevPhaseStepSummaryV1 | None,
) -> TrialDevelopmentRequestV1 | None:
    """Load a canonical phase request and verify its analysis sidecar."""

    if phase_id == "observational_review":
        return None
    path = program_dir / "agent_workdir" / f"phase_{phase_id}" / "request.json"
    summarized = summary is not None and summary.request is not None
    if path.is_file() != summarized:
        raise TrialDevAnalysisSourceError(
            f"TrialDev canonical request and phase summary disagree: {program_dir.name}:{phase_id}"
        )
    if not path.is_file():
        return None

    request = read_json_model(TrialDevelopmentRequestV1, path)
    if summary is None or summary.request is None:
        raise TrialDevAnalysisSourceError(f"TrialDev phase request summary is missing: {program_dir.name}:{phase_id}")
    witness = summary.request
    expected = {
        "phase_id": request.phase_id,
        "endpoint_id": request.endpoint_id,
        "selection_objective": request.selection_objective,
        "target_sample_size": request.target_sample_size,
        "follow_up_days": request.follow_up_days,
        "allocation_ratio": request.allocation_ratio,
        "site_count_budget": request.site_count_budget,
        "enrollment_window_days": request.enrollment_window_days,
    }
    observed = witness.model_dump(include=set(expected))
    if observed != expected:
        raise TrialDevAnalysisSourceError(
            f"TrialDev canonical request disagrees with its phase summary: {program_dir.name}:{phase_id}"
        )
    return request


def _grade_by_phase(
    program_dir: Path,
) -> tuple[TrialDevGradeRecordV1 | None, TrialDevTrajectoryGradeV1 | None, dict[str, TrialDevGradeRecordV1]]:
    obs_path = program_dir / "obs_review" / "grade_report.json"
    trajectory_path = program_dir / "trajectory_grade.json"
    obs_grade = read_json_model(TrialDevGradeRecordV1, obs_path) if obs_path.is_file() else None
    trajectory = read_json_model(TrialDevTrajectoryGradeV1, trajectory_path) if trajectory_path.is_file() else None
    reports: dict[str, TrialDevGradeRecordV1] = {}
    if trajectory is not None:
        for report in trajectory.phase_reports:
            if report.phase_id is None:
                raise TrialDevAnalysisSourceError(f"Trajectory phase report lacks phase_id: {trajectory_path}")
            if report.phase_id in reports:
                raise TrialDevAnalysisSourceError(
                    f"Trajectory contains duplicate phase report {report.phase_id!r}: {trajectory_path}"
                )
            reports[report.phase_id] = report
    return obs_grade, trajectory, reports


def summarise_trialdev_phase_power(
    phases: Sequence[TrialDevPhaseResourceConsequenceV1],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Summarize weakest efficacy and safety operating characteristics."""

    efficacy_powers = tuple(
        float(phase.achieved_power)
        for phase in phases
        if phase.achieved_power is not None and phase.target_power is not None
    )
    efficacy_shortfalls = tuple(
        max(0.0, float(phase.target_power) - float(phase.achieved_power))
        for phase in phases
        if phase.achieved_power is not None and phase.target_power is not None
    )
    safety_powers = tuple(
        min(
            float(phase.achieved_safety_absolute_risk_power),
            float(phase.achieved_safety_excess_risk_power),
        )
        for phase in phases
    )
    safety_shortfalls = tuple(
        max(0.0, float(phase.target_safety_decision_power) - safety_power)
        for phase, safety_power in zip(phases, safety_powers, strict=True)
    )
    return (
        min(efficacy_powers) if efficacy_powers else None,
        max(efficacy_shortfalls) if efficacy_shortfalls else None,
        min(safety_powers) if safety_powers else None,
        max(safety_shortfalls) if safety_shortfalls else None,
    )


def _audit_derived_csv(
    run_dir: Path,
    canonical: dict[tuple[str, str], TrialDevGradeRecordV1],
) -> None:
    """Reject disagreement in an optional derived CSV without using it as input."""

    path = run_dir / "results_full.csv"
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    csv_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("program_id") or "", row.get("phase_id") or "")
        if not all(key) or key in csv_by_key:
            raise TrialDevAnalysisSourceError(f"Derived TrialDev CSV has invalid or duplicate key {key!r}: {path}")
        csv_by_key[key] = row
    if set(csv_by_key) != set(canonical):
        raise TrialDevAnalysisSourceError(f"Derived TrialDev CSV keys disagree with canonical grader records: {path}")
    for key, report in canonical.items():
        try:
            csv_score = float(csv_by_key[key]["primary_score"])
        except (KeyError, ValueError) as exc:
            raise TrialDevAnalysisSourceError(f"Derived TrialDev CSV has invalid primary_score for {key!r}") from exc
        if abs(csv_score - report.primary_score) > 1e-12:
            raise TrialDevAnalysisSourceError(
                f"Derived TrialDev CSV primary_score disagrees with canonical grader record for {key!r}"
            )


def _candidate_eligibility(
    candidate_drug_id: str | None,
    feasibility_failures: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    """Derive candidate eligibility from typed grader failure codes."""

    if candidate_drug_id is None:
        return ()
    failure_reason = "not_applicable"
    if f"unknown_candidate:{candidate_drug_id}" in feasibility_failures:
        failure_reason = "not_in_public_catalog"
    elif f"ineligible_candidate:{candidate_drug_id}" in feasibility_failures:
        failure_reason = "not_in_current_eligible_set"
    elif any(reason.startswith("program_already_terminal:") for reason in feasibility_failures):
        failure_reason = "program_already_terminal"
    return (
        {
            "candidate_drug_id": candidate_drug_id,
            "catalog_member": failure_reason != "not_in_public_catalog",
            "sequentially_eligible": failure_reason == "not_applicable",
            "eligibility_source": "canonical_grader_record",
            "failure_reason": failure_reason,
        },
    )


def trialdev_design_signature_sha256(request: TrialDevelopmentRequestV1) -> str:
    """Hash design-defining choices while excluding scenario and asset identity."""

    payload = request.model_dump(
        mode="json",
        exclude={"version", "scenario_id", "candidate_drug_ids", "selection_objective"},
    )
    payload["candidate_count"] = len(request.candidate_drug_ids)
    return str(canonical_payload_sha256(cast(JsonValue, payload)))


def trialdev_programme_pairing_sha256(
    config: TrialDevRunConfigV1,
    *,
    program_id: str,
) -> str:
    """Hash every paired runtime coordinate except procedure assistance."""

    return str(
        canonical_payload_sha256(
            cast(
                JsonValue,
                {
                    "bundle_sha256": config.bundle_sha256,
                    "scorer_source_sha256": config.scorer_source_sha256,
                    "runner_source_sha256": config.runner_source_sha256,
                    "prompt_interface_sha256": config.prompt_interface_sha256,
                    "staging_source_sha256": config.staging_source_sha256,
                    "model": config.model,
                    "master_seed": config.master_seed,
                    "request_replicate_id": config.experiment_condition.request_replicate_id,
                    "seed_variants": config.seed_variants,
                    "decoding": config.decoding.model_dump(mode="json"),
                    "routing": config.routing.model_dump(mode="json"),
                    "executor": config.executor.model_dump(mode="json"),
                    "workers": config.workers,
                    "max_turns_per_step": config.max_turns_per_step,
                    "max_phase_retries": config.max_phase_retries,
                    "max_submission_attempts": config.max_submission_attempts,
                    "tool_choice": config.experiment_condition.tool_choice,
                    "program_watchdog_seconds": config.program_watchdog_seconds,
                    "selected_program_ids": sorted(config.selected_program_ids),
                    "program_id": program_id,
                },
            )
        )
    )


def _program_rows(
    *,
    run_dir: Path,
    run_id: str,
    config: TrialDevRunConfigV1,
    program_id: str,
) -> tuple[
    TrialDevProgrammeAnalysisRowV1, tuple[TrialDevPhaseOutcomeRowV1, ...], dict[tuple[str, str], TrialDevGradeRecordV1]
]:
    program_dir = run_dir / "programs" / program_id
    if not program_dir.is_dir():
        raise TrialDevAnalysisSourceError(f"Declared TrialDev programme directory is missing: {program_dir}")
    chain_path = program_dir / "chain_summary.json"
    chain = read_json_model(TrialDevChainSummaryV1, chain_path)
    if chain.program_id != program_id:
        raise TrialDevAnalysisSourceError(f"TrialDev chain/program identity mismatch: {chain_path}")
    if chain.execution_status in {"infrastructure_timeout", "infrastructure_error"}:
        raise TrialDevAnalysisSourceError(
            f"TrialDev infrastructure failure is not an analysable model outcome: {program_id}:{chain.execution_status}"
        )

    obs_grade, trajectory, phase_reports = _grade_by_phase(program_dir)
    if (
        chain.execution_status == "completed"
        and trajectory is None
        and not chain.trajectory_metrics.checkpoint_outcomes
    ):
        raise TrialDevAnalysisSourceError(f"Completed TrialDev programme lacks programme grade: {program_dir}")
    completed = chain.execution_status == "completed"
    if chain.wall_seconds_total is None:
        raise TrialDevAnalysisSourceError(f"Terminal TrialDev programme lacks wall-clock duration: {chain_path}")
    provider = _programme_provider_telemetry(program_dir, program_id=program_id)
    programme_primary_score = chain.trajectory_metrics.programme_primary_score
    if completed and programme_primary_score is None:
        raise TrialDevAnalysisSourceError(
            f"Completed TrialDev programme lacks checkpoint-complete programme score: {program_id}"
        )
    primary_score = float(programme_primary_score or 0.0)
    persisted_decision_score = chain.trajectory_metrics.trajectory_decision_score
    decision_score = float(persisted_decision_score or 0.0) if completed else 0.0
    resources = chain.trajectory_metrics.resource_summary
    if resources is None and completed:
        raise TrialDevAnalysisSourceError(f"TrialDev programme lacks its resource consequence vector: {program_id}")
    phase_resources = () if trajectory is None else tuple(trajectory.resource_consequence.phases)
    if completed and resources is not None and len(phase_resources) != resources.phase_count:
        raise TrialDevAnalysisSourceError(f"TrialDev phase-resource count drift for programme_id={program_id!r}.")
    phase_resource_by_id = {resource.phase_id: resource for resource in phase_resources}
    if len(phase_resource_by_id) != len(phase_resources):
        raise TrialDevAnalysisSourceError(
            f"TrialDev phase resources duplicate a phase for programme_id={program_id!r}."
        )
    (
        minimum_efficacy_power,
        maximum_efficacy_power_shortfall,
        minimum_safety_power,
        maximum_safety_power_shortfall,
    ) = summarise_trialdev_phase_power(phase_resources)
    attempted_phase_ids = {
        str(attempt.phase_id)
        for attempt in chain.phases_attempted
        if str(attempt.phase_id) in {"phase1", "phase2", "phase3"}
    }
    quality = summarise_programme_analysis_quality(
        observational_report=obs_grade,
        phase_reports=() if trajectory is None else trajectory.phase_reports,
        attempted_phase_ids=attempted_phase_ids,
    )
    persisted_quality = chain.trajectory_metrics.analysis_quality
    if persisted_quality is not None and persisted_quality != quality:
        raise TrialDevAnalysisSourceError(f"TrialDev analysis-quality denominator drift: {program_id!r}.")
    design_valid = resources is not None and (
        resources.phase_count > 0
        and resources.statistically_inadequate_phases == 0
        and resources.operationally_infeasible_phases == 0
    )
    design_nondominated = resources is not None and design_valid and resources.dominated_phases == 0
    programme = TrialDevProgrammeAnalysisRowV1(
        model_id=config.model,
        run_id=run_id,
        bundle_sha256=config.bundle_sha256,
        scorer_source_sha256=config.scorer_source_sha256,
        runner_source_sha256=config.runner_source_sha256,
        prompt_interface_sha256=config.prompt_interface_sha256,
        staging_source_sha256=config.staging_source_sha256,
        seed_variants=config.seed_variants,
        condition_id=config.experiment_condition.condition_id,
        replicate_id=config.experiment_condition.request_replicate_id,
        reasoning_effort=config.experiment_condition.reasoning.effort,
        maximum_turns_per_step=config.experiment_condition.maximum_turns_per_step,
        maximum_submission_attempts=config.experiment_condition.maximum_submission_attempts,
        tool_choice=config.experiment_condition.tool_choice,
        task_materialization_seed=config.master_seed,
        program_id=program_id,
        scenario_id=chain.scenario_id,
        objective_id=chain.objective_id,
        pairing_sha256=trialdev_programme_pairing_sha256(config, program_id=program_id),
        procedure_assistance=config.procedure_assistance,
        execution_status=chain.execution_status,
        completed=completed,
        trajectory_primary_score=primary_score,
        trajectory_decision_score=decision_score,
        observational_analysis_validity=quality.observational_analysis_validity,
        observational_analysis_score=quality.observational_analysis_score,
        randomized_primary_effect_point_agreement=quality.randomized_primary_effect_point_agreement,
        randomized_primary_effect_interval_agreement=quality.randomized_primary_effect_interval_agreement,
        safety_evidence_agreement=quality.safety_evidence_agreement,
        phase_evaluation_validity=quality.phase_evaluation_validity,
        programme_design_validity=design_valid,
        programme_design_nondominance=design_nondominated,
        randomized_phase_count=0 if resources is None else resources.phase_count,
        minimum_randomized_efficacy_power=minimum_efficacy_power,
        maximum_randomized_efficacy_power_shortfall=maximum_efficacy_power_shortfall,
        minimum_randomized_safety_power=minimum_safety_power,
        maximum_randomized_safety_power_shortfall=maximum_safety_power_shortfall,
        total_agent_turns=(
            chain.obs_review_path_stats.turns + sum(attempt.turns for attempt in chain.phases_attempted)
        ),
        total_execute_code_calls=(
            chain.obs_review_path_stats.execute_code
            + sum(attempt.execute_code_calls for attempt in chain.phases_attempted)
        ),
        total_inspect_parquet_calls=(
            chain.obs_review_path_stats.inspect_parquet
            + sum(attempt.inspect_parquet_calls for attempt in chain.phases_attempted)
        ),
        provider_response_count=provider.response_count,
        provider_responses_with_usage=provider.responses_with_usage,
        prompt_tokens=provider.prompt_tokens,
        completion_tokens=provider.completion_tokens,
        total_tokens=provider.total_tokens,
        provider_elapsed_seconds=provider.elapsed_seconds,
        wall_seconds_total=chain.wall_seconds_total,
        invalid_submission_attempts=chain.trajectory_metrics.n_invalid_attempts,
        phase_materialization_calls=sum(attempt.n_materializations for attempt in chain.phases_attempted),
        total_participants=0 if resources is None else resources.total_participants,
        total_protocol_follow_up_days=0 if resources is None else resources.total_protocol_follow_up_days,
        total_enrollment_window_days=0 if resources is None else resources.total_enrollment_window_days,
        total_site_phase_budget=0 if resources is None else resources.total_site_phase_budget,
        total_planned_phase_duration_days=0 if resources is None else resources.total_planned_phase_duration_days,
        total_participant_follow_up_days=0 if resources is None else resources.total_participant_follow_up_days,
        participant_excess_vs_minimum=0 if resources is None else resources.participant_excess_vs_minimum,
        participant_shortage_vs_minimum=0 if resources is None else resources.participant_shortage_vs_minimum,
        follow_up_excess_days_vs_minimum=0 if resources is None else resources.follow_up_excess_days_vs_minimum,
        follow_up_shortage_days_vs_minimum=0 if resources is None else resources.follow_up_shortage_days_vs_minimum,
        statistically_inadequate_phases=0 if resources is None else resources.statistically_inadequate_phases,
        operationally_infeasible_phases=0 if resources is None else resources.operationally_infeasible_phases,
        dominated_phases=0 if resources is None else resources.dominated_phases,
        avoidable_participants_min=0 if resources is None else resources.design_avoidable_participants_min,
        avoidable_participants_max=0 if resources is None else resources.design_avoidable_participants_max,
        avoidable_follow_up_days_min=0 if resources is None else resources.design_avoidable_follow_up_days_min,
        avoidable_follow_up_days_max=0 if resources is None else resources.design_avoidable_follow_up_days_max,
        avoidable_participant_follow_up_days_min=(
            0 if resources is None else resources.design_avoidable_participant_follow_up_days_min
        ),
        avoidable_participant_follow_up_days_max=(
            0 if resources is None else resources.design_avoidable_participant_follow_up_days_max
        ),
        late_continuation_participants=0 if resources is None else resources.late_continuation_participants,
        late_continuation_protocol_follow_up_days=(
            0 if resources is None else resources.late_continuation_protocol_follow_up_days
        ),
        late_continuation_enrollment_window_days=(
            0 if resources is None else resources.late_continuation_enrollment_window_days
        ),
        late_continuation_site_phase_budget=0 if resources is None else resources.late_continuation_site_phase_budget,
        late_continuation_participant_follow_up_days=(
            0 if resources is None else resources.late_continuation_participant_follow_up_days
        ),
    )

    attempt_by_phase = {attempt.phase_id: attempt for attempt in chain.phases_attempted}
    if len(attempt_by_phase) != len(chain.phases_attempted):
        raise TrialDevAnalysisSourceError(f"TrialDev chain contains duplicate phase attempts: {chain_path}")
    obs_summary_path = program_dir / "obs_review" / "obs_review_summary.json"
    obs_summary = read_json_model(TrialDevObsReviewSummaryV1, obs_summary_path) if obs_summary_path.is_file() else None
    obs_submission_path = program_dir / "obs_review" / "obs_review_submission.json"
    obs_submission = (
        read_json_model(TrialDevelopmentSubmissionV1, obs_submission_path) if obs_submission_path.is_file() else None
    )
    if (
        obs_summary is not None
        and obs_submission is not None
        and obs_summary.recommended_drug_id != obs_submission.program_decision.recommended_drug_id
    ):
        raise TrialDevAnalysisSourceError(
            f"TrialDev observational summary disagrees with its typed submission: {program_id}"
        )
    phases: list[TrialDevPhaseOutcomeRowV1] = []
    canonical_grades: dict[tuple[str, str], TrialDevGradeRecordV1] = {}
    for phase_id in PHASE_ORDER:
        report = obs_grade if phase_id == "observational_review" else phase_reports.get(phase_id)
        if report is not None:
            if report.phase_id not in {None, phase_id}:
                raise TrialDevAnalysisSourceError(
                    f"TrialDev grade phase identity mismatch for {program_id}:{phase_id}"
                )
            if report.scenario_id not in {None, chain.scenario_id}:
                raise TrialDevAnalysisSourceError(
                    f"TrialDev grade scenario identity mismatch for {program_id}:{phase_id}"
                )
        if report is not None:
            canonical_grades[(program_id, phase_id)] = report
        attempt = attempt_by_phase.get(phase_id)
        summary = _phase_summary(program_dir, phase_id)
        submitted = (
            obs_submission is not None
            if phase_id == "observational_review"
            else bool(summary and (summary.analysis is not None or summary.decision is not None))
        )
        attempted = attempt is not None or submitted
        after_stop = bool(chain.stopped_at_phase and _phase_rank(phase_id) > _phase_rank(chain.stopped_at_phase))
        reached = not after_stop and (attempted or report is not None)
        if completed and attempted and report is None:
            raise TrialDevAnalysisSourceError(
                f"Completed TrialDev programme has an attempted phase without a grader record: {program_id}:{phase_id}"
            )
        if report is not None and after_stop:
            raise TrialDevAnalysisSourceError(
                f"TrialDev programme has a grader record after its terminal stop: {program_id}:{phase_id}"
            )
        if report is not None:
            endpoint_state = "valid" if report.primary_score > 0.0 else "failed"
        elif after_stop:
            endpoint_state = "not_reached_after_stop"
        elif attempted:
            endpoint_state = "submission_present_score_absent"
        else:
            endpoint_state = "not_attempted_noncompletion"

        decision = summary.decision if summary is not None else None
        analysis = summary.analysis if summary is not None else None
        request = _canonical_phase_request(program_dir, phase_id, summary)
        resource = phase_resource_by_id.get(phase_id) if phase_id != "observational_review" else None
        if resource is not None and (request is None or resource.request_checksum != request.checksum()):
            raise TrialDevAnalysisSourceError(
                f"TrialDev phase resource disagrees with its submitted request: {program_id}:{phase_id}"
            )
        violation_kinds = tuple(
            sorted(str(item.get("kind") or "unknown") for item in chain.violations if item.get("phase_id") == phase_id)
        )
        feasibility_failures = tuple(report.feasibility_failures) if report is not None else ()
        candidate = (
            obs_submission.program_decision.recommended_drug_id
            if phase_id == "observational_review" and obs_submission is not None
            else decision.candidate_drug_id if decision is not None else attempt.candidate_drug_id if attempt else None
        )
        phases.append(
            TrialDevPhaseOutcomeRowV1(
                model_id=config.model,
                run_id=run_id,
                scenario_id=chain.scenario_id,
                program_id=program_id,
                objective_id=chain.objective_id,
                phase_id=phase_id,
                phase_attempted=attempted,
                phase_reached=reached,
                stopped_at_phase=chain.stopped_at_phase,
                decision_action=(
                    obs_submission.program_decision.decision_action
                    if phase_id == "observational_review" and obs_submission is not None
                    else (
                        decision.decision_action
                        if decision is not None
                        else attempt.decision_action if attempt else None
                    )
                ),
                advance=attempt.advance if attempt is not None else None,
                candidate_drug_id=candidate,
                endpoint_id=(request.endpoint_id if request is not None else None),
                design_signature_sha256=(trialdev_design_signature_sha256(request) if request is not None else None),
                target_sample_size=request.target_sample_size if request is not None else None,
                follow_up_days=request.follow_up_days if request is not None else None,
                allocation_ratio=request.allocation_ratio if request is not None else None,
                allocation_weights=request.allocation_weights if request is not None else (),
                design_status=resource.design_status if resource is not None else None,
                statistically_adequate=resource.statistically_adequate if resource is not None else None,
                operationally_feasible=resource.operationally_feasible if resource is not None else None,
                operational_support=resource.operational_support if resource is not None else None,
                operational_headroom=resource.operational_headroom if resource is not None else None,
                operational_shortage=resource.operational_shortage if resource is not None else None,
                participant_excess_vs_minimum=(
                    resource.participant_excess_vs_minimum if resource is not None else None
                ),
                participant_shortage_vs_minimum=(
                    resource.participant_shortage_vs_minimum if resource is not None else None
                ),
                avoidable_participant_follow_up_days_min=(
                    resource.avoidable_participant_follow_up_days_min if resource is not None else None
                ),
                avoidable_participant_follow_up_days_max=(
                    resource.avoidable_participant_follow_up_days_max if resource is not None else None
                ),
                n_materializations=attempt.n_materializations if attempt is not None else 0,
                execute_code_calls=(
                    chain.obs_review_path_stats.execute_code
                    if phase_id == "observational_review"
                    else attempt.execute_code_calls if attempt is not None else 0
                ),
                inspect_parquet_calls=(
                    chain.obs_review_path_stats.inspect_parquet
                    if phase_id == "observational_review"
                    else attempt.inspect_parquet_calls if attempt is not None else 0
                ),
                turns=(
                    chain.obs_review_path_stats.turns
                    if phase_id == "observational_review"
                    else attempt.turns if attempt is not None else 0
                ),
                matched_item_id=attempt.matched_item_id if attempt is not None else None,
                violations_n=len(violation_kinds),
                violation_kinds=violation_kinds,
                invalid_attempt_reasons=(
                    tuple(trajectory.invalid_attempt_reasons)
                    if trajectory is not None
                    else tuple(chain.trajectory_metrics.invalid_attempt_reasons)
                ),
                feasibility_failures=feasibility_failures,
                lane_failure_reasons=tuple(
                    sorted({reason.split(":", maxsplit=1)[0] for reason in feasibility_failures})
                ),
                candidate_eligibility_records=_candidate_eligibility(candidate, feasibility_failures),
                trajectory_decision_score=decision_score,
                trajectory_primary_score=primary_score,
                decision_regret=(
                    trajectory.decision_regret_by_phase.get(phase_id) if trajectory is not None else None
                ),
                selected_winner_drug_id=(
                    report.selected_winner_drug_id
                    if report is not None
                    else analysis.selected_winner_drug_id if analysis is not None else None
                ),
                best_candidate_drug_id=report.best_candidate_drug_id if report is not None else None,
                ranking_score=report.ranking_score if report is not None else None,
                program_score=report.program_score if report is not None else None,
                policy_reference_regret=report.policy_reference_regret if report is not None else None,
                in_set_regret=report.in_set_regret if report is not None else None,
                trialdev_result_source="results_full_score_export" if report is not None else "not_available",
                endpoint_state=endpoint_state,
                score_link_id=f"{run_id}:{program_id}:{phase_id}",
            )
        )
    return programme, tuple(phases), canonical_grades


def load_trialdev_analysis_dataset(run_dirs: Iterable[Path]) -> TrialDevAnalysisDataset:
    """Load the canonical TrialDev analysis dataset from stored run contracts."""

    resolved_dirs = [Path(path).resolve() for path in run_dirs]
    run_ids = require_unique_run_ids(resolved_dirs, suite="trialdev")
    programmes: list[TrialDevProgrammeAnalysisRowV1] = []
    phases: list[TrialDevPhaseOutcomeRowV1] = []
    for run_dir in sorted(resolved_dirs):
        config = read_json_model(TrialDevRunConfigV1, run_dir / "run_config.json")
        canonical_grades: dict[tuple[str, str], TrialDevGradeRecordV1] = {}
        for program_id in config.selected_program_ids:
            programme, programme_phases, grades = _program_rows(
                run_dir=run_dir,
                run_id=run_ids[run_dir],
                config=config,
                program_id=program_id,
            )
            programmes.append(programme)
            phases.extend(programme_phases)
            canonical_grades.update(grades)
        _audit_derived_csv(run_dir, canonical_grades)
    dataset = TrialDevAnalysisDataset(programmes=tuple(programmes), phases=tuple(phases))
    build_trialdev_analysis_populations(dataset)
    return dataset


__all__ = [
    "PHASE_ORDER",
    "TrialDevAnalysisDataset",
    "TrialDevAnalysisPopulations",
    "TrialDevAnalysisSourceError",
    "TrialDevPhaseId",
    "TrialDevPhaseKey",
    "TrialDevPopulationName",
    "TrialDevProgrammeKey",
    "build_trialdev_analysis_populations",
    "load_trialdev_analysis_dataset",
    "summarise_trialdev_phase_power",
    "trialdev_phase_key",
    "trialdev_programme_pairing_sha256",
    "trialdev_programme_key",
]
