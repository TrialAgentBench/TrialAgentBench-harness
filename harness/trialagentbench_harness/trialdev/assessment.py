"""Project persisted TrialDev run and grade evidence into public metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from trialagentbench_harness.contracts.core.runs import (
    TrialDevChainSummaryV1,
    TrialDevRunConfigV1,
)
from trialagentbench_harness.contracts.trialdev.metrics import (
    TRIALDEV_CAPABILITY_CHECKS_V1,
    TRIALDEV_CAPABILITY_IDS_V1,
    TRIALDEV_CHECKPOINT_INVENTORY_V1,
    TRIALDEV_REQUIRED_LANES_V1,
    TRIALDEV_TERMINAL_LANES_V1,
    TrialDevCapabilityAssessmentV1,
    TrialDevCapabilityCheckIdV1,
    TrialDevCapabilityCheckV1,
    TrialDevCapabilityIdV1,
    TrialDevCheckpointAssessmentV1,
    TrialDevLaneAssessmentV1,
    TrialDevLaneOutcomeV1,
    TrialDevMetricLaneIdV1,
    TrialDevProgrammeAssessmentV1,
    TrialDevProgrammeExecutionStatusV1,
    TrialDevSecondaryOutcomesV1,
    TrialDevSwitchTimingV1,
)
from trialagentbench_harness.contracts.trialdev.portfolio_submission import (
    TrialDevPortfolioCheckpointGradeV1,
    TrialDevPortfolioCheckpointSubmissionV1,
    TrialDevPortfolioRunSummaryV1,
)
from trialagentbench_harness.contracts.trialdev.programme import (
    TRIALDEV_PROGRAMME_STATE_ADAPTER_V1,
    TrialDevAnalysisStatusV1,
    TrialDevCheckpointIdV1,
    TrialDevCheckpointOutcomeV1,
    TrialDevPortfolioProgrammeStateV1,
    TrialDevProgrammeStateV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevGradeRecordV1,
    TrialDevLaneScoreRecordV1,
    TrialDevTrajectoryGradeV1,
)
from trialagentbench_harness.io import read_json, read_json_model, sha256_path
from trialagentbench_harness.trialdev.grading.models import TrialDevelopmentSubmissionV1
from trialagentbench_harness.util.provider_telemetry import read_provider_terminal_events_v1

_PHASE_TO_CHECKPOINT: dict[str, TrialDevCheckpointIdV1] = {
    "observational_review": "observational_review",
    "phase1": "early_safety_study",
    "phase2": "proof_of_concept",
    "phase3": "confirmation",
}
_STATE_ORDER = (
    "state_initial.json",
    "state_after_observational_review.json",
    "state_after_phase1.json",
    "state_after_phase2.json",
    "state_after_phase3.json",
)


def _load_states(program_dir: Path) -> tuple[TrialDevProgrammeStateV1, ...]:
    state_dir = Path(program_dir) / "states"
    states = tuple(
        TRIALDEV_PROGRAMME_STATE_ADAPTER_V1.validate_python(read_json(state_dir / name))
        for name in _STATE_ORDER
        if (state_dir / name).is_file()
    )
    if not states:
        raise FileNotFoundError(f"TrialDev programme has no persisted state: {state_dir}")
    for previous, current in zip(states, states[1:], strict=False):
        if current.previous_state_checksum != previous.checksum:
            raise ValueError("TrialDev persisted states do not form one checksum chain.")
        if current.history[:-1] != previous.history:
            raise ValueError("TrialDev persisted state rewrote prior checkpoint history.")
    return states


def _grade_reports(
    program_dir: Path,
) -> tuple[
    dict[TrialDevCheckpointIdV1, TrialDevGradeRecordV1],
    TrialDevTrajectoryGradeV1 | None,
    dict[TrialDevCheckpointIdV1, str],
]:
    reports: dict[TrialDevCheckpointIdV1, TrialDevGradeRecordV1] = {}
    source_sha256: dict[TrialDevCheckpointIdV1, str] = {}
    observational_path = Path(program_dir) / "obs_review" / "grade_report.json"
    if observational_path.is_file():
        reports["observational_review"] = read_json_model(TrialDevGradeRecordV1, observational_path)
        source_sha256["observational_review"] = sha256_path(observational_path)
    trajectory_path = Path(program_dir) / "trajectory_grade.json"
    trajectory = read_json_model(TrialDevTrajectoryGradeV1, trajectory_path) if trajectory_path.is_file() else None
    if trajectory is not None:
        for report in trajectory.phase_reports:
            checkpoint_id = _PHASE_TO_CHECKPOINT.get(str(report.phase_id))
            if checkpoint_id is None:
                raise ValueError(f"TrialDev grade has an unsupported phase: {report.phase_id!r}.")
            if checkpoint_id in reports:
                raise ValueError(f"TrialDev grade duplicates checkpoint: {checkpoint_id!r}.")
            reports[checkpoint_id] = report
            source_sha256[checkpoint_id] = sha256_path(trajectory_path)
    return reports, trajectory, source_sha256


def _history_sha256(state: TrialDevProgrammeStateV1, checkpoint_id: TrialDevCheckpointIdV1) -> str:
    matches = tuple(entry for entry in state.history if entry.checkpoint_id == checkpoint_id)
    if len(matches) != 1 or matches[0].checksum is None:
        raise ValueError(f"TrialDev history lacks one checksummed {checkpoint_id!r} transition.")
    return cast(str, matches[0].checksum)


def _capability(
    capability_id: TrialDevCapabilityIdV1,
    results: Mapping[TrialDevCapabilityCheckIdV1, tuple[bool, str]],
) -> TrialDevCapabilityAssessmentV1:
    required = TRIALDEV_CAPABILITY_CHECKS_V1[capability_id]
    if not set(required) <= set(results):
        return TrialDevCapabilityAssessmentV1(capability_id=capability_id, outcome="not_applicable")
    checks = tuple(
        TrialDevCapabilityCheckV1(
            check_id=check_id,
            passed=results[check_id][0],
            source_record_sha256=results[check_id][1],
        )
        for check_id in required
    )
    return TrialDevCapabilityAssessmentV1(
        capability_id=capability_id,
        outcome="passed" if all(check.passed for check in checks) else "failed",
        checks=checks,
    )


def _capabilities(
    report: TrialDevGradeRecordV1,
    *,
    checkpoint_id: TrialDevCheckpointIdV1,
    report_sha256: str,
    history_sha256: str,
    required_outputs_present: bool,
) -> tuple[TrialDevCapabilityAssessmentV1, ...]:
    assessment = report.scientific_assessment
    checks: dict[TrialDevCapabilityCheckIdV1, tuple[bool, str]] = {
        "evidence_integrity": (assessment.evidential_support == "passed", report_sha256),
        "method_eligibility": (assessment.assumptions == "passed", report_sha256),
        "identification_status": (
            assessment.analysis_classification == "uncertainty_qualified",
            report_sha256,
        ),
        "uncertainty_qualification": (assessment.uncertainty == "passed", report_sha256),
        "policy_conclusion_compatibility": (assessment.scientific_agreement == "passed", report_sha256),
        "transition_legality": (True, history_sha256),
        "history_immutability": (True, history_sha256),
        "required_output_presence": (required_outputs_present, report_sha256),
        "workflow_completion": (assessment.execution == "passed", report_sha256),
        "selected_action_membership": (assessment.action_admissibility == "passed", report_sha256),
    }
    if checkpoint_id != "observational_review":
        checks["safety_evidence"] = (assessment.scientific_agreement == "passed", report_sha256)
    return tuple(_capability(capability_id, checks) for capability_id in TRIALDEV_CAPABILITY_IDS_V1)


def _lane_outcome(record: TrialDevLaneScoreRecordV1) -> TrialDevLaneOutcomeV1:
    if record.artifact_status == "missing" or record.status == "missing_submission_zeroed":
        return "missing"
    if record.artifact_status == "invalid" or record.status == "invalid_submission_zeroed" or record.score != 1.0:
        return "invalid"
    if record.status == "not_applicable":
        raise ValueError("A required TrialDev lane cannot be not_applicable.")
    return "accepted"


def _lanes(
    report: TrialDevGradeRecordV1,
    *,
    stream_id: Literal["single_asset_development"],
    checkpoint_id: TrialDevCheckpointIdV1,
    terminal: bool,
    trajectory: TrialDevTrajectoryGradeV1 | None,
    report_sha256: str,
) -> tuple[TrialDevLaneAssessmentV1, ...]:
    records = list(report.lane_scores)
    if terminal and trajectory is not None:
        records.extend(trajectory.final_lane_scores)
    required = set(TRIALDEV_REQUIRED_LANES_V1[(stream_id, checkpoint_id)])
    if terminal:
        required.update(TRIALDEV_TERMINAL_LANES_V1)
    by_lane = {cast(TrialDevMetricLaneIdV1, record.lane_id): record for record in records}
    if len(by_lane) != len(records):
        raise ValueError(f"TrialDev grade duplicates a lane at checkpoint {checkpoint_id!r}.")
    output: list[TrialDevLaneAssessmentV1] = []
    for lane_id in sorted(required):
        record = by_lane.get(lane_id)
        if record is None:
            if terminal and trajectory is None and lane_id in TRIALDEV_TERMINAL_LANES_V1:
                output.append(
                    TrialDevLaneAssessmentV1(
                        lane_id=lane_id,
                        outcome="accepted" if report.program_score == 1.0 else "invalid",
                        source_record_sha256=report_sha256,
                    )
                )
                continue
            raise ValueError(f"TrialDev grade omits required lane {checkpoint_id!r}:{lane_id!r}.")
        if record.checksum is None:
            raise ValueError(f"TrialDev lane lacks a checksum: {checkpoint_id!r}:{lane_id!r}.")
        output.append(
            TrialDevLaneAssessmentV1(
                lane_id=cast(TrialDevMetricLaneIdV1, lane_id),
                outcome=_lane_outcome(record),
                source_record_sha256=report_sha256,
            )
        )
    return tuple(output)


def _execution_status(
    chain: TrialDevChainSummaryV1,
    latest_state: TrialDevProgrammeStateV1,
) -> TrialDevProgrammeExecutionStatusV1:
    if chain.execution_status in {"infrastructure_timeout", "infrastructure_error"}:
        return "infrastructure_failure"
    if chain.execution_status in {"model_turn_limit", "model_invalid_submission"}:
        return "model_noncompletion"
    if latest_state.terminal_disposition == "active":
        return "model_noncompletion"
    return "completed"


def _analysis_status(
    *,
    program_dir: Path,
    checkpoint_id: TrialDevCheckpointIdV1,
    report: TrialDevGradeRecordV1,
) -> TrialDevAnalysisStatusV1:
    if not report.analysis_quality.phase_evaluation_valid:
        return "invalid"
    if checkpoint_id != "observational_review":
        return "estimable"
    submission = read_json_model(
        TrialDevelopmentSubmissionV1,
        Path(program_dir) / "obs_review" / "obs_review_submission.json",
    )
    return "non_estimable" if submission.analysis_report.response_branch == "qualified_non_nomination" else "estimable"


def _secondary_outcomes(program_dir: Path, chain: TrialDevChainSummaryV1) -> TrialDevSecondaryOutcomesV1:
    provider_path = Path(program_dir) / "provider_responses.jsonl"
    events = read_provider_terminal_events_v1(provider_path) if provider_path.is_file() else ()
    costs = tuple(event.reported_cost_usd for event in events if event.reported_cost_usd is not None)
    all_costs_reported = bool(events) and len(costs) == len(events)
    return TrialDevSecondaryOutcomesV1(
        elapsed_seconds=float(chain.wall_seconds_total or 0.0),
        provider_calls=len(events),
        agent_turns=(chain.obs_review_path_stats.turns + sum(attempt.turns for attempt in chain.phases_attempted)),
        correction_count=chain.violations_n,
        execute_code_calls=(
            chain.obs_review_path_stats.execute_code
            + sum(attempt.execute_code_calls for attempt in chain.phases_attempted)
        ),
        inspect_data_calls=(
            chain.obs_review_path_stats.inspect_parquet
            + sum(attempt.inspect_parquet_calls for attempt in chain.phases_attempted)
        ),
        prompt_tokens=sum(event.prompt_tokens for event in events),
        completion_tokens=sum(event.completion_tokens for event in events),
        provider_reported_usd=sum(cast(Sequence[float], costs)) if all_costs_reported else None,
    )


def build_single_asset_programme_assessment_v1(
    *,
    program_dir: Path,
    run_config: TrialDevRunConfigV1,
) -> TrialDevProgrammeAssessmentV1:
    """Build one source-bound assessment from an ordinary graded programme."""

    root = Path(program_dir)
    chain = read_json_model(TrialDevChainSummaryV1, root / "chain_summary.json")
    states = _load_states(root)
    latest_state = states[-1]
    if latest_state.stream_id != "single_asset_development":
        raise ValueError("The single-asset assessment builder received another TrialDev stream.")
    if latest_state.programme_id != chain.program_id or latest_state.scenario_id != chain.scenario_id:
        raise ValueError("TrialDev state and chain-summary identities disagree.")
    reports, trajectory, report_sources = _grade_reports(root)
    execution_status = _execution_status(chain, latest_state)
    terminal_checkpoint = (
        latest_state.history[-1].checkpoint_id if execution_status == "completed" and latest_state.history else None
    )
    failure_checkpoint = latest_state.current_checkpoint_id if execution_status != "completed" else None
    checkpoints: list[TrialDevCheckpointAssessmentV1] = []
    for checkpoint_id in TRIALDEV_CHECKPOINT_INVENTORY_V1["single_asset_development"]:
        report = reports.get(checkpoint_id)
        if report is not None and checkpoint_id != failure_checkpoint:
            terminal = checkpoint_id == terminal_checkpoint
            lanes = _lanes(
                report,
                stream_id="single_asset_development",
                checkpoint_id=checkpoint_id,
                terminal=terminal,
                trajectory=trajectory,
                report_sha256=report_sources[checkpoint_id],
            )
            history_sha256 = _history_sha256(latest_state, checkpoint_id)
            required_outputs_present = all(lane.outcome != "missing" for lane in lanes)
            checkpoints.append(
                TrialDevCheckpointAssessmentV1(
                    checkpoint_id=checkpoint_id,
                    outcome=TrialDevCheckpointOutcomeV1(
                        reach_status="reached",
                        submission_status="accepted" if report.validity.valid else "invalid",
                        analysis_status=_analysis_status(
                            program_dir=root,
                            checkpoint_id=checkpoint_id,
                            report=report,
                        ),
                        execution_status="completed",
                    ),
                    lanes=lanes,
                    capabilities=_capabilities(
                        report,
                        checkpoint_id=checkpoint_id,
                        report_sha256=report_sources[checkpoint_id],
                        history_sha256=history_sha256,
                        required_outputs_present=required_outputs_present,
                    ),
                    scientific_assessment=report.scientific_assessment,
                    terminal_record_valid=(all(lane.outcome == "accepted" for lane in lanes) if terminal else None),
                )
            )
            continue
        if checkpoint_id == failure_checkpoint:
            failure_sha256 = sha256_path(root / "chain_summary.json")
            required_lanes = TRIALDEV_REQUIRED_LANES_V1[("single_asset_development", checkpoint_id)]
            checkpoints.append(
                TrialDevCheckpointAssessmentV1(
                    checkpoint_id=checkpoint_id,
                    outcome=TrialDevCheckpointOutcomeV1(
                        reach_status="reached",
                        submission_status="missing",
                        analysis_status="missing",
                        execution_status=execution_status,
                    ),
                    lanes=(
                        tuple(
                            TrialDevLaneAssessmentV1(
                                lane_id=lane_id,
                                outcome="missing",
                                source_record_sha256=failure_sha256,
                            )
                            for lane_id in required_lanes
                        )
                        if execution_status == "model_noncompletion"
                        else ()
                    ),
                )
            )
            continue
        checkpoints.append(
            TrialDevCheckpointAssessmentV1(
                checkpoint_id=checkpoint_id,
                outcome=TrialDevCheckpointOutcomeV1(
                    reach_status="structural_nonreach",
                    submission_status="not_applicable",
                    analysis_status="not_applicable",
                    execution_status="not_applicable",
                ),
            )
        )
    return TrialDevProgrammeAssessmentV1(
        model_id=run_config.model,
        condition_id=run_config.experiment_condition.condition_id,
        request_replicate_id=run_config.experiment_condition.request_replicate_id,
        reasoning_effort=run_config.experiment_condition.reasoning.effort,
        procedure_assistance=run_config.experiment_condition.procedure_assistance,
        maximum_turns_per_step=run_config.experiment_condition.maximum_turns_per_step,
        maximum_submission_attempts=run_config.experiment_condition.maximum_submission_attempts,
        tool_choice=run_config.experiment_condition.tool_choice,
        task_materialization_seed=run_config.master_seed,
        release_id=run_config.bundle_sha256,
        run_id=run_config.run_identity_sha256,
        grader_sha256=run_config.scorer_source_sha256,
        evaluation_unit_id=chain.program_id,
        programme_id=chain.program_id,
        scenario_family_id=chain.scenario_id,
        objective_variant_id=chain.objective_id,
        policy_variant_id=latest_state.policy_binding.action_policy_checksum,
        stream_id="single_asset_development",
        execution_status=execution_status,
        checkpoints=tuple(checkpoints),
        secondary_outcomes=_secondary_outcomes(root, chain),
    )


def _portfolio_capabilities(
    *,
    checkpoint_id: TrialDevCheckpointIdV1,
    grade: TrialDevPortfolioCheckpointGradeV1,
    grade_sha256: str,
    history_sha256: str,
) -> tuple[TrialDevCapabilityAssessmentV1, ...]:
    assessment = grade.scientific_assessment
    checks: dict[TrialDevCapabilityCheckIdV1, tuple[bool, str]] = {
        "evidence_integrity": (assessment.evidential_support == "passed", grade_sha256),
        "method_eligibility": (assessment.assumptions == "passed", grade_sha256),
        "identification_status": (
            assessment.analysis_classification == "uncertainty_qualified",
            grade_sha256,
        ),
        "uncertainty_qualification": (assessment.uncertainty == "passed", grade_sha256),
        "policy_conclusion_compatibility": (assessment.scientific_agreement == "passed", grade_sha256),
        "transition_legality": (True, history_sha256),
        "history_immutability": (True, history_sha256),
        "required_output_presence": (True, grade_sha256),
        "workflow_completion": (assessment.execution == "passed", grade_sha256),
        "selected_action_membership": (assessment.action_admissibility == "passed", grade_sha256),
    }
    if checkpoint_id != "observational_review":
        checks["safety_evidence"] = (assessment.scientific_agreement == "passed", grade_sha256)
    return tuple(_capability(capability_id, checks) for capability_id in TRIALDEV_CAPABILITY_IDS_V1)


def _portfolio_lanes(
    *,
    checkpoint_id: TrialDevCheckpointIdV1,
    grade: TrialDevPortfolioCheckpointGradeV1,
    grade_sha256: str,
    terminal: bool,
) -> tuple[TrialDevLaneAssessmentV1, ...]:
    required = set(TRIALDEV_REQUIRED_LANES_V1[("bounded_portfolio_reallocation", checkpoint_id)])
    if terminal:
        required.update(TRIALDEV_TERMINAL_LANES_V1)
    assessment = grade.scientific_assessment
    analysis_valid = bool(
        assessment.analysis_classification == "uncertainty_qualified"
        and assessment.assumptions == "passed"
        and assessment.scientific_agreement == "passed"
        and assessment.uncertainty == "passed"
    )
    values: dict[TrialDevMetricLaneIdV1, bool] = {
        "asset_nomination": assessment.action_admissibility == "passed",
        "phase_design": assessment.design in {"passed", "not_applicable"},
        "phase_analysis": analysis_valid,
        "safety_gate": assessment.scientific_agreement == "passed",
        "decision_action": assessment.action_admissibility == "passed",
        "portfolio_allocation": assessment.evidential_support == "passed"
        and assessment.action_admissibility == "passed",
        "resource_feasibility": assessment.design in {"passed", "not_applicable"},
        "route_timing": assessment.sequential_coherence == "passed",
        "final_recommendation": assessment.action_admissibility == "passed",
    }
    return tuple(
        TrialDevLaneAssessmentV1(
            lane_id=lane_id,
            outcome="accepted" if values[lane_id] else "invalid",
            source_record_sha256=grade_sha256,
        )
        for lane_id in sorted(required)
    )


def build_portfolio_programme_assessment_v1(
    *,
    program_dir: Path,
    run_config: TrialDevRunConfigV1,
    release_id: str,
) -> TrialDevProgrammeAssessmentV1:
    """Project one independently graded portfolio run into public metrics."""

    root = Path(program_dir)
    summary = read_json_model(TrialDevPortfolioRunSummaryV1, root / "portfolio_run_summary.json")
    states = tuple(
        read_json_model(TrialDevPortfolioProgrammeStateV1, root / relative_path)
        for relative_path in summary.state_relative_paths
    )
    if len(states) != len(summary.grade_relative_paths) + 1:
        raise ValueError("Portfolio assessment requires one post-decision state per grade.")
    for previous, current in zip(states, states[1:], strict=False):
        if current.previous_state_checksum != previous.checksum or current.history[:-1] != previous.history:
            raise ValueError("Portfolio persisted states do not form one immutable checksum chain.")
    grades = tuple(
        read_json_model(TrialDevPortfolioCheckpointGradeV1, root / relative_path)
        for relative_path in summary.grade_relative_paths
    )
    submissions = tuple(
        read_json_model(TrialDevPortfolioCheckpointSubmissionV1, root / relative_path)
        for relative_path in summary.submission_relative_paths
    )
    if len(grades) != len(submissions):
        raise ValueError("Portfolio assessment requires aligned submissions and grades.")
    reached_by_checkpoint = {
        grade.checkpoint_id: (grade, submission, summary.grade_relative_paths[index], states[index + 1])
        for index, (grade, submission) in enumerate(zip(grades, submissions, strict=True))
    }
    if len(reached_by_checkpoint) != len(grades):
        raise ValueError("A portfolio run cannot reach the same checkpoint twice.")
    last_reached = grades[-1].checkpoint_id if grades else None
    failure_checkpoint = states[-1].current_checkpoint_id if summary.execution_status != "completed" else None
    checkpoint_assessments = []
    for checkpoint_id in TRIALDEV_CHECKPOINT_INVENTORY_V1["bounded_portfolio_reallocation"]:
        record = reached_by_checkpoint.get(checkpoint_id)
        if record is None:
            if checkpoint_id == failure_checkpoint:
                execution_status = (
                    "model_noncompletion"
                    if summary.execution_status == "model_noncompletion"
                    else "infrastructure_failure"
                )
                required_lanes = TRIALDEV_REQUIRED_LANES_V1[("bounded_portfolio_reallocation", checkpoint_id)]
                checkpoint_assessments.append(
                    TrialDevCheckpointAssessmentV1(
                        checkpoint_id=checkpoint_id,
                        outcome=TrialDevCheckpointOutcomeV1(
                            reach_status="reached",
                            submission_status="missing",
                            analysis_status="missing",
                            execution_status=execution_status,
                        ),
                        lanes=(
                            tuple(
                                TrialDevLaneAssessmentV1(
                                    lane_id=lane_id,
                                    outcome="missing",
                                    source_record_sha256=sha256_path(root / "portfolio_run_summary.json"),
                                )
                                for lane_id in required_lanes
                            )
                            if execution_status == "model_noncompletion"
                            else ()
                        ),
                    )
                )
                continue
            checkpoint_assessments.append(
                TrialDevCheckpointAssessmentV1(
                    checkpoint_id=checkpoint_id,
                    outcome=TrialDevCheckpointOutcomeV1(
                        reach_status="structural_nonreach",
                        submission_status="not_applicable",
                        analysis_status="not_applicable",
                        execution_status="not_applicable",
                    ),
                )
            )
            continue
        grade, _submission, grade_relative_path, post_state = record
        grade_sha256 = sha256_path(root / grade_relative_path)
        history = post_state.history[-1]
        if history.checksum is None:
            raise ValueError("Portfolio transition history lacks a checksum.")
        terminal = checkpoint_id == last_reached
        scientific_assessment = grade.scientific_assessment.model_copy(
            update={
                "resources": (
                    "within_budget"
                    if post_state.policy_binding.resource_budget_units is not None
                    and post_state.resource_spent_units <= post_state.policy_binding.resource_budget_units
                    else "exceeded"
                )
            }
        )
        checkpoint_assessments.append(
            TrialDevCheckpointAssessmentV1(
                checkpoint_id=checkpoint_id,
                outcome=grade.outcome,
                lanes=_portfolio_lanes(
                    checkpoint_id=checkpoint_id,
                    grade=grade,
                    grade_sha256=grade_sha256,
                    terminal=terminal,
                ),
                capabilities=_portfolio_capabilities(
                    checkpoint_id=checkpoint_id,
                    grade=grade,
                    grade_sha256=grade_sha256,
                    history_sha256=cast(str, history.checksum),
                ),
                scientific_assessment=scientific_assessment,
                terminal_record_valid=True if terminal else None,
            )
        )
    final_state = states[-1]
    switch_timing = "none"
    if final_state.switch_count:
        promotion = next(
            entry.checkpoint_id
            for entry in final_state.history
            if entry.selected_action is not None
            and entry.selected_action.action_id == "promote_reserve_to_proof_of_concept"
        )
        switch_timing = "early" if promotion == "joint_early_study_review" else "late"
    return TrialDevProgrammeAssessmentV1(
        model_id=run_config.model,
        condition_id=run_config.experiment_condition.condition_id,
        request_replicate_id=run_config.experiment_condition.request_replicate_id,
        reasoning_effort=run_config.experiment_condition.reasoning.effort,
        procedure_assistance=run_config.experiment_condition.procedure_assistance,
        maximum_turns_per_step=run_config.experiment_condition.maximum_turns_per_step,
        maximum_submission_attempts=run_config.experiment_condition.maximum_submission_attempts,
        tool_choice=run_config.experiment_condition.tool_choice,
        task_materialization_seed=run_config.master_seed,
        release_id=release_id,
        run_id=run_config.run_identity_sha256,
        grader_sha256=run_config.scorer_source_sha256,
        evaluation_unit_id=summary.programme_id,
        programme_id=summary.programme_id,
        scenario_family_id=summary.scenario_id,
        objective_variant_id=summary.objective_id,
        policy_variant_id=f"resource_budget_{summary.resource_budget_units}",
        stream_id="bounded_portfolio_reallocation",
        execution_status=summary.execution_status,
        checkpoints=tuple(checkpoint_assessments),
        secondary_outcomes=TrialDevSecondaryOutcomesV1(
            elapsed_seconds=summary.wall_seconds_total,
            provider_calls=summary.provider_calls,
            agent_turns=summary.agent_turns,
            correction_count=summary.correction_count,
            execute_code_calls=summary.execute_code_calls,
            inspect_data_calls=summary.inspect_data_calls,
            prompt_tokens=summary.prompt_tokens,
            completion_tokens=summary.completion_tokens,
            provider_reported_usd=summary.provider_reported_usd,
            programme_resource_units=final_state.resource_spent_units,
            switch_count=final_state.switch_count,
            switch_timing=cast(TrialDevSwitchTimingV1, switch_timing),
        ),
    )


__all__ = ["build_single_asset_programme_assessment_v1"]
