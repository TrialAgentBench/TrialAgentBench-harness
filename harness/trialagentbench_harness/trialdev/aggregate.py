"""Cross-program rollups and result reporting.

Walks every program directory under a run output root and extracts per-item
grade rows from saved ``grade_report.json`` (obs_review) and
``trajectory_grade.json`` (phase reports), producing:

* ``results_full.csv`` — one row per scored item, with raw lane scores
* ``RESULTS_SUMMARY.md`` — human-readable complete-suite summary with
  per-scenario, per-objective, and per-phase rollups
* ``results_summary.json`` — machine-readable equivalent of the markdown

Headline program-level rollups are computed on the **declared program
denominator** for the run (the selected program set written into
``coverage_report.json`` / ``run_config.json``). Missing or unusable
program artifacts are counted as explicit failures (score 0) rather than
dropped, preventing survivorship inflation. Diagnostic “attempted-only”
views may be exported separately, but never replace denominator-preserving
headline metrics.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from trialagentbench_harness import __version__ as harness_version
from trialagentbench_harness.contracts.core.manifest import AggregateManifestV1, ToleratedFailureV1
from trialagentbench_harness.contracts.core.runs import (
    TrialDevChainSummaryV1,
    TrialDevPhaseStepSummaryV1,
    TrialDevRunConfigV1,
)
from trialagentbench_harness.contracts.core.summaries import (
    TrialDevCompletionMetricsV1,
    TrialDevGroupRollupV1,
    TrialDevObjectiveAlignmentByPrimaryV1,
    TrialDevObjectiveAlignmentSummaryV1,
    TrialDevProgramCompletionV1,
    TrialDevRankMetricsSummaryV1,
    TrialDevRankPhaseSummaryV1,
    TrialDevResultsPayloadV1,
    TrialDevResultsRollupV1,
    TrialDevResultsSummaryV1,
    TrialDevStickTwistSummaryV1,
    TrialDevViolationsSummaryV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_exports import (
    TrialDevLaneScoreExportRowV1,
    TrialDevObjectiveAlignmentRowV1,
    TrialDevPhaseResourceRowV1,
    TrialDevProgrammeResourceRowV1,
    TrialDevRankMetricRowV1,
    TrialDevResultsRowV1,
    TrialDevStickTwistRowV1,
    TrialDevViolationRowV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_grades import (
    TrialDevGradeRecordV1,
    TrialDevLaneScoreRecordV1,
    TrialDevTrajectoryGradeV1,
)
from trialagentbench_harness.failure_codes import TrialDevFailureCode
from trialagentbench_harness.io.csv import write_csv_models
from trialagentbench_harness.io.json import read_json_model, write_json_model
from trialagentbench_harness.policies import AggregatePolicy
from trialagentbench_harness.trialdev.data import discover_items, scenario_root
from trialagentbench_harness.trialdev.derived_metrics import (
    RankMetrics,
    objective_alignment_for_program,
    rank_metrics_for_obs_review,
    rank_metrics_for_phase,
    stick_twist_for_program,
)
from trialagentbench_harness.trialdev.grade_wrappers import wrap_lane_score_record
from trialagentbench_harness.trialdev.schema import BenchmarkItem
from trialagentbench_harness.trialdev.scoring import LANE_KEYS, extract_grade_record, iter_phase_reports

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _items_index(bundle_root: Path) -> dict[tuple[str, str, str, str | None], BenchmarkItem]:
    """Index manifest items by (scenario, phase, objective, endpoint)."""
    out: dict[tuple[str, str, str, str | None], BenchmarkItem] = {}
    for item in discover_items(bundle_root):
        out[(item.scenario_id, item.phase_id, item.objective_id, item.endpoint_id)] = item
    return out


def _match_item_for_row(
    items_idx: dict[tuple[str, str, str, str | None], BenchmarkItem],
    *,
    scenario_id: str,
    phase_id: str,
    objective_id: str,
    endpoint_id: str | None,
) -> BenchmarkItem | None:
    key = (scenario_id, phase_id, objective_id, endpoint_id)
    if key in items_idx:
        return items_idx[key]
    # Endpoint may be None on the row but the manifest carries one — fall
    # back to a phase-level match (any endpoint).
    for (s, p, o, _e), item in items_idx.items():
        if (s, p, o) == (scenario_id, phase_id, objective_id):
            return item
    return None


def _load_grade_record(
    path: Path | None,
    *,
    policy: AggregatePolicy,
    manifest: AggregateManifestV1,
    code: TrialDevFailureCode,
) -> TrialDevGradeRecordV1 | None:
    if path is None or not path.is_file():
        return None
    try:
        return read_json_model(TrialDevGradeRecordV1, path)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        if policy.allow_incomplete_artifacts:
            manifest.tolerated_failures.append(ToleratedFailureV1(code=code, message=str(exc), path=str(path)))
            return None
        raise


def _load_trajectory_grade(
    path: Path | None,
    *,
    policy: AggregatePolicy,
    manifest: AggregateManifestV1,
    code: TrialDevFailureCode,
) -> TrialDevTrajectoryGradeV1 | None:
    if path is None or not path.is_file():
        return None
    try:
        return read_json_model(TrialDevTrajectoryGradeV1, path)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        if policy.allow_incomplete_artifacts:
            manifest.tolerated_failures.append(ToleratedFailureV1(code=code, message=str(exc), path=str(path)))
            return None
        raise


def _read_chain_summary(
    program_dir: Path,
    *,
    policy: AggregatePolicy,
    manifest: AggregateManifestV1,
) -> TrialDevChainSummaryV1 | None:
    p = program_dir / "chain_summary.json"
    if not p.is_file():
        return None
    try:
        return read_json_model(TrialDevChainSummaryV1, p)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        if policy.allow_incomplete_artifacts:
            manifest.tolerated_failures.append(
                ToleratedFailureV1(
                    code=TrialDevFailureCode.chain_summary_invalid,
                    message=str(exc),
                    path=str(p),
                )
            )
            return None
        raise


def _request_endpoint(
    program_dir: Path,
    phase_id: str,
    *,
    policy: AggregatePolicy,
    manifest: AggregateManifestV1,
) -> str | None:
    summary_path = program_dir / f"phase_{phase_id}" / "phase_step_summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = read_json_model(TrialDevPhaseStepSummaryV1, summary_path)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        if policy.allow_incomplete_artifacts:
            manifest.tolerated_failures.append(
                ToleratedFailureV1(
                    code=TrialDevFailureCode.phase_request_invalid,
                    message=str(exc),
                    path=str(summary_path),
                )
            )
            return None
        raise
    if summary.request is None:
        return None
    return summary.request.endpoint_id


def _attach_active_lanes(row: dict[str, object], report: TrialDevGradeRecordV1) -> None:
    """Pull upstream's ``primary_score`` + ``active_lane_scores`` into the row.

    Upstream now scores natively over active lanes only — no ceilings. The
    ``primary_score`` field is the headline, equivalent to our previous
    primary-score recomputation. We also surface per-lane
    ``lane_status`` so post-hoc analysis can tell active from
    not_applicable lanes.
    """
    row["primary_score"] = float(report.primary_score)
    for lane, score in (report.active_lane_scores or {}).items():
        row[f"lane_active__{lane}"] = float(score)
    for lane, status in (report.lane_status or {}).items():
        row[f"lane_status__{lane}"] = str(status)


def collect_rows(
    output_root: Path,
    bundle_root: Path,
    *,
    policy: AggregatePolicy,
    manifest: AggregateManifestV1,
) -> tuple[
    list[TrialDevResultsRowV1],
    list[TrialDevLaneScoreExportRowV1],
    list[TrialDevRankMetricRowV1],
    list[TrialDevStickTwistRowV1],
    list[TrialDevObjectiveAlignmentRowV1],
]:
    """Walk every program directory and produce aggregate row collections.

    * ``rows`` — one per scored item (obs_review or trajectory phase) with
      raw + active lane scores
    * ``rank_rows`` — one per phase with rank-gap and bottom-N concordance
    * ``stick_twist_rows`` — one per program comparing obs_review pick to
      phase1 pick (stick = inherited, twist = the agent pivoted)
    * ``alignment_rows`` — one per program tracking whether the agent's
      ``selection_objective`` per phase matches the program's primary
      objective at the (phase2/3) phases where it has a free choice
    """
    items_idx = _items_index(bundle_root)
    reference_scenario_by_checksum = _reference_scenario_by_checksum(bundle_root)
    programs_root = output_root / "programs"
    rows: list[TrialDevResultsRowV1] = []
    lane_score_rows: list[TrialDevLaneScoreExportRowV1] = []
    rank_rows: list[TrialDevRankMetricRowV1] = []
    stick_twist_rows: list[TrialDevStickTwistRowV1] = []
    alignment_rows: list[TrialDevObjectiveAlignmentRowV1] = []
    if not programs_root.is_dir():
        return rows, lane_score_rows, rank_rows, stick_twist_rows, alignment_rows

    for program_dir in sorted(programs_root.iterdir()):
        if not program_dir.is_dir():
            continue
        chain = _read_chain_summary(program_dir, policy=policy, manifest=manifest)
        scenario_id = str(chain.scenario_id) if chain else ""
        objective_id = str(chain.objective_id) if chain else ""
        program_id = str(chain.program_id) if chain else program_dir.name

        # ── obs_review ─────────────────────────────────────────────────────
        obs_grade = _load_grade_record(
            program_dir / "obs_review" / "grade_report.json",
            policy=policy,
            manifest=manifest,
            code=TrialDevFailureCode.obs_grade_invalid,
        )
        if obs_grade is not None:
            row = extract_grade_record(obs_grade, source="obs_review_grade_item")
            row.update(
                {
                    "program_id": program_id,
                    "scenario_id": scenario_id,
                    "objective_id": objective_id,
                    "phase_id": "observational_review",
                    "endpoint_id": None,
                }
            )
            item = _match_item_for_row(
                items_idx,
                scenario_id=scenario_id,
                phase_id="observational_review",
                objective_id=objective_id,
                endpoint_id=None,
            )
            row["item_id"] = item.item_id if item else None
            _attach_active_lanes(row, obs_grade)
            rows.append(TrialDevResultsRowV1.model_validate(row))
            lane_score_rows.extend(
                _lane_score_rows(
                    report=obs_grade,
                    program_id=program_id,
                    scenario_key=scenario_id,
                    objective_id=objective_id,
                    reference_scenario_by_checksum=reference_scenario_by_checksum,
                    source="obs_review_grade_item",
                )
            )
            # Rank metrics for obs_review
            try:
                rm = rank_metrics_for_obs_review(program_dir, bundle_root=bundle_root)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                if policy.allow_incomplete_artifacts:
                    manifest.tolerated_failures.append(
                        ToleratedFailureV1(
                            code=TrialDevFailureCode.obs_rank_metrics_failed,
                            message=str(exc),
                            program_id=program_id,
                            path=str(program_dir),
                        )
                    )
                    rm = None
                else:
                    raise
            if rm is not None:
                rank_rows.append(_rank_row(rm))

        # ── phases (from trajectory) ───────────────────────────────────────
        traj = _load_trajectory_grade(
            program_dir / "trajectory_grade.json",
            policy=policy,
            manifest=manifest,
            code=TrialDevFailureCode.trajectory_grade_invalid,
        )
        if traj is not None:
            for phase_id, report in iter_phase_reports(traj):
                phase_id = str(phase_id) if phase_id else ""
                if not phase_id:
                    continue
                endpoint_id = _request_endpoint(
                    program_dir,
                    phase_id,
                    policy=policy,
                    manifest=manifest,
                )
                row = extract_grade_record(report, source="trajectory_phase_report")
                row.update(
                    {
                        "program_id": program_id,
                        "scenario_id": scenario_id,
                        "objective_id": objective_id,
                        "phase_id": phase_id,
                        "endpoint_id": endpoint_id,
                    }
                )
                item = _match_item_for_row(
                    items_idx,
                    scenario_id=scenario_id,
                    phase_id=phase_id,
                    objective_id=objective_id,
                    endpoint_id=endpoint_id,
                )
                row["item_id"] = item.item_id if item else None
                _attach_active_lanes(row, report)
                rows.append(TrialDevResultsRowV1.model_validate(row))
                lane_score_rows.extend(
                    _lane_score_rows(
                        report=report,
                        program_id=program_id,
                        scenario_key=scenario_id,
                        objective_id=objective_id,
                        reference_scenario_by_checksum=reference_scenario_by_checksum,
                        source="trajectory_phase_report",
                    )
                )
                # Per-phase rank metrics
                try:
                    rm = rank_metrics_for_phase(program_dir, bundle_root=bundle_root, phase_id=phase_id)
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    if policy.allow_incomplete_artifacts:
                        manifest.tolerated_failures.append(
                            ToleratedFailureV1(
                                code=TrialDevFailureCode.phase_rank_metrics_failed,
                                message=str(exc),
                                program_id=program_id,
                                path=str(program_dir),
                            )
                        )
                        rm = None
                    else:
                        raise
                if rm is not None:
                    rank_rows.append(_rank_row(rm))
            lane_score_rows.extend(
                _lane_score_export_rows_from_lane_records(
                    lanes=traj.final_lane_scores,
                    program_id=program_id,
                    scenario_key=scenario_id,
                    objective_id=objective_id,
                    reference_scenario_by_checksum=reference_scenario_by_checksum,
                    source="trajectory_final_decision",
                )
            )
        elif scenario_id and objective_id:
            try:
                from trialagentbench_harness.trialdev.grading.sequential import (
                    final_decision_lane_scores_from_trajectory,
                )

                final_lanes = final_decision_lane_scores_from_trajectory(
                    scenario_root=scenario_root(bundle_root, scenario_id),
                    scenario_id=scenario_id,
                    program_objective_id=objective_id,
                    terminal_action=None,
                    terminal_recommendation_score=None,
                    trajectory_decision_score=None,
                    artifact_status="missing",
                    failure_reason="missing_trajectory_grade",
                )
            except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
                if policy.allow_incomplete_artifacts:
                    manifest.tolerated_failures.append(
                        ToleratedFailureV1(
                            code=TrialDevFailureCode.trajectory_grade_invalid,
                            message=str(exc),
                            program_id=program_id,
                            path=str(program_dir),
                        )
                    )
                else:
                    raise
            else:
                lane_score_rows.extend(
                    _lane_score_export_rows_from_lane_records(
                        lanes=(wrap_lane_score_record(lane.model_dump(mode="json")) for lane in final_lanes),
                        program_id=program_id,
                        scenario_key=scenario_id,
                        objective_id=objective_id,
                        reference_scenario_by_checksum=reference_scenario_by_checksum,
                        source="trajectory_final_decision_missing",
                    )
                )

        # ── stop timing ────────────────────────────────────────────────────
        # ── stick/twist ────────────────────────────────────────────────────
        try:
            tw = stick_twist_for_program(program_dir)
            if tw is not None:
                stick_twist_rows.append(
                    TrialDevStickTwistRowV1(
                        program_id=tw.program_id,
                        obs_pick=tw.obs_pick,
                        phase1_pick=tw.phase1_pick,
                        pivoted=tw.pivoted,
                    )
                )
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
            if policy.allow_incomplete_artifacts:
                manifest.tolerated_failures.append(
                    ToleratedFailureV1(
                        code=TrialDevFailureCode.stick_twist_failed,
                        message=str(exc),
                        program_id=program_id,
                        path=str(program_dir),
                    )
                )
            else:
                raise

        # ── objective alignment ────────────────────────────────────────────
        try:
            al = objective_alignment_for_program(program_dir, bundle_root=bundle_root)
            if al is not None:
                alignment_rows.append(
                    TrialDevObjectiveAlignmentRowV1(
                        program_id=al.program_id,
                        primary_objective=al.primary_objective,
                        n_free_phases=al.n_free_phases,
                        n_aligned=al.n_aligned,
                        alignment_rate=al.alignment_rate,
                        phase1_selected=al.per_phase.get("phase1", {}).get("selected"),
                        phase1_forced=al.per_phase.get("phase1", {}).get("forced"),
                        phase2_selected=al.per_phase.get("phase2", {}).get("selected"),
                        phase2_aligned=al.per_phase.get("phase2", {}).get("aligned"),
                        phase3_selected=al.per_phase.get("phase3", {}).get("selected"),
                        phase3_aligned=al.per_phase.get("phase3", {}).get("aligned"),
                    )
                )
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
            if policy.allow_incomplete_artifacts:
                manifest.tolerated_failures.append(
                    ToleratedFailureV1(
                        code=TrialDevFailureCode.objective_alignment_failed,
                        message=str(exc),
                        program_id=program_id,
                        path=str(program_dir),
                    )
                )
            else:
                raise

    return rows, lane_score_rows, rank_rows, stick_twist_rows, alignment_rows


def _collect_resource_rows(
    output_root: Path,
    *,
    policy: AggregatePolicy,
    manifest: AggregateManifestV1,
) -> tuple[list[TrialDevPhaseResourceRowV1], list[TrialDevProgrammeResourceRowV1]]:
    """Extract canonical phase and programme resource vectors from trajectory grades."""

    phase_rows: list[TrialDevPhaseResourceRowV1] = []
    programme_rows: list[TrialDevProgrammeResourceRowV1] = []
    programs_root = Path(output_root) / "programs"
    if not programs_root.is_dir():
        return phase_rows, programme_rows
    for program_dir in sorted(path for path in programs_root.iterdir() if path.is_dir()):
        chain = _read_chain_summary(program_dir, policy=policy, manifest=manifest)
        grade = _load_trajectory_grade(
            program_dir / "trajectory_grade.json",
            policy=policy,
            manifest=manifest,
            code=TrialDevFailureCode.trajectory_grade_invalid,
        )
        if chain is None or grade is None:
            continue
        resources = grade.resource_consequence
        for phase in resources.phases:
            phase_rows.append(
                TrialDevPhaseResourceRowV1(
                    program_id=chain.program_id,
                    scenario_id=chain.scenario_id,
                    objective_id=chain.objective_id,
                    phase_id=phase.phase_id,
                    request_checksum=phase.request_checksum,
                    target_sample_size=phase.target_sample_size,
                    follow_up_days=phase.follow_up_days,
                    enrollment_window_days=phase.enrollment_window_days,
                    site_count_budget=phase.site_count_budget,
                    participant_follow_up_days=phase.participant_follow_up_days,
                    statistically_adequate=phase.statistically_adequate,
                    operationally_feasible=phase.operationally_feasible,
                    design_status=phase.design_status,
                    operational_support=phase.operational_support,
                    operational_headroom=phase.operational_headroom,
                    operational_shortage=phase.operational_shortage,
                    achieved_power=phase.achieved_power,
                    target_power=phase.target_power,
                    achieved_safety_absolute_risk_power=(phase.achieved_safety_absolute_risk_power),
                    achieved_safety_excess_risk_power=(phase.achieved_safety_excess_risk_power),
                    target_safety_decision_power=phase.target_safety_decision_power,
                    participant_excess_vs_minimum=phase.participant_excess_vs_minimum,
                    participant_shortage_vs_minimum=phase.participant_shortage_vs_minimum,
                    follow_up_excess_days_vs_minimum=phase.follow_up_excess_days_vs_minimum,
                    follow_up_shortage_days_vs_minimum=phase.follow_up_shortage_days_vs_minimum,
                    dominating_frontier_count=len(phase.dominating_frontier),
                    avoidable_participants_min=phase.avoidable_participants_min,
                    avoidable_participants_max=phase.avoidable_participants_max,
                    avoidable_follow_up_days_min=phase.avoidable_follow_up_days_min,
                    avoidable_follow_up_days_max=phase.avoidable_follow_up_days_max,
                    avoidable_participant_follow_up_days_min=(phase.avoidable_participant_follow_up_days_min),
                    avoidable_participant_follow_up_days_max=(phase.avoidable_participant_follow_up_days_max),
                    entered_after_unsupported_advance=(phase.entered_after_unsupported_advance),
                )
            )
        programme_rows.append(
            TrialDevProgrammeResourceRowV1(
                program_id=chain.program_id,
                scenario_id=chain.scenario_id,
                objective_id=chain.objective_id,
                phase_count=len(resources.phases),
                total_participants=resources.total_participants,
                total_protocol_follow_up_days=resources.total_protocol_follow_up_days,
                total_enrollment_window_days=resources.total_enrollment_window_days,
                total_site_phase_budget=resources.total_site_phase_budget,
                total_planned_phase_duration_days=resources.total_planned_phase_duration_days,
                total_participant_follow_up_days=(resources.total_participant_follow_up_days),
                participant_excess_vs_minimum=resources.participant_excess_vs_minimum,
                participant_shortage_vs_minimum=resources.participant_shortage_vs_minimum,
                follow_up_excess_days_vs_minimum=resources.follow_up_excess_days_vs_minimum,
                follow_up_shortage_days_vs_minimum=resources.follow_up_shortage_days_vs_minimum,
                statistically_inadequate_phases=resources.statistically_inadequate_phases,
                operationally_infeasible_phases=resources.operationally_infeasible_phases,
                dominated_phases=resources.dominated_phases,
                design_avoidable_participants_min=(resources.design_avoidable_participants_min),
                design_avoidable_participants_max=(resources.design_avoidable_participants_max),
                design_avoidable_follow_up_days_min=(resources.design_avoidable_follow_up_days_min),
                design_avoidable_follow_up_days_max=(resources.design_avoidable_follow_up_days_max),
                design_avoidable_participant_follow_up_days_min=(
                    resources.design_avoidable_participant_follow_up_days_min
                ),
                design_avoidable_participant_follow_up_days_max=(
                    resources.design_avoidable_participant_follow_up_days_max
                ),
                late_continuation_participants=resources.late_continuation_participants,
                late_continuation_protocol_follow_up_days=(resources.late_continuation_protocol_follow_up_days),
                late_continuation_enrollment_window_days=(resources.late_continuation_enrollment_window_days),
                late_continuation_site_phase_budget=resources.late_continuation_site_phase_budget,
                late_continuation_participant_follow_up_days=(resources.late_continuation_participant_follow_up_days),
                cost_status=resources.cost_status,
            )
        )
    return phase_rows, programme_rows


def _reference_scenario_by_checksum(bundle_root: Path) -> dict[str, str]:
    """Map every evaluation-target register checksum in a bundle to its semantic scenario id."""

    out: dict[str, str] = {}
    for scenario_dir in sorted(Path(bundle_root).glob("scenario_*")):
        register_path = scenario_dir / "grader" / "evaluation_target_register.jsonl"
        if not register_path.is_file():
            continue
        for line_number, line in enumerate(register_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"evaluation_target_register.jsonl row {line_number} must be an object: {register_path}"
                )
            checksum = payload.get("checksum")
            scenario_id = payload.get("scenario_id")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError(
                    f"evaluation_target_register.jsonl row {line_number} lacks a valid checksum: {register_path}"
                )
            if not isinstance(scenario_id, str) or not scenario_id:
                raise ValueError(
                    f"evaluation_target_register.jsonl row {line_number} lacks scenario_id: {register_path}"
                )
            previous = out.get(checksum)
            if previous is not None and previous != scenario_id:
                raise ValueError(f"reference checksum maps to multiple scenarios: {checksum}")
            out[checksum] = scenario_id
    if not out:
        raise ValueError(f"No TrialDev evaluation-target register checksums found under bundle: {bundle_root}")
    return out


def _lane_scenario_identity(
    *,
    lane: TrialDevLaneScoreRecordV1,
    scenario_key: str,
    reference_scenario_by_checksum: dict[str, str],
) -> tuple[str, str]:
    """Return (scenario_key, semantic_id) and reject inconsistent lane identity."""

    if not scenario_key:
        raise ValueError("Lane score export requires the scenario key from chain_summary.json.")
    checksum = str(lane.evaluation_target_checksum)
    if checksum not in reference_scenario_by_checksum:
        raise ValueError(f"Lane score references unknown evaluation_target_checksum: {checksum}")
    semantic_id = reference_scenario_by_checksum[checksum]
    lane_scenario_id = str(lane.scenario_id)
    if lane_scenario_id not in {scenario_key, semantic_id}:
        raise ValueError(
            "Lane score scenario_id does not match either the program scenario key or reference semantic id: "
            f"lane={lane_scenario_id!r}, scenario_key={scenario_key!r}, scenario_semantic_id={semantic_id!r}"
        )
    return scenario_key, semantic_id


def _lane_score_rows(
    *,
    report: TrialDevGradeRecordV1,
    program_id: str,
    scenario_key: str,
    objective_id: str,
    reference_scenario_by_checksum: dict[str, str],
    source: str,
) -> list[TrialDevLaneScoreExportRowV1]:
    """Flatten register lane-score records from one grade report."""

    rows: list[TrialDevLaneScoreExportRowV1] = []
    for lane in report.lane_scores:
        scenario_key, scenario_semantic_id = _lane_scenario_identity(
            lane=lane,
            scenario_key=scenario_key,
            reference_scenario_by_checksum=reference_scenario_by_checksum,
        )
        rows.append(
            TrialDevLaneScoreExportRowV1(
                program_id=program_id,
                scenario_id=scenario_semantic_id,
                scenario_key=scenario_key,
                scenario_semantic_id=scenario_semantic_id,
                objective_id=objective_id,
                phase_id=str(lane.phase_id),
                program_objective_id=str(lane.program_objective_id),
                phase_scoring_objective_id=str(lane.phase_scoring_objective_id),
                lane_id=str(lane.lane_id),
                evaluation_target_checksum=str(lane.evaluation_target_checksum),
                scoring_policy_id=str(lane.scoring_policy_id),
                recoverability_policy_id=str(lane.recoverability_policy_id),
                submitted_target_id=lane.submitted_target_id,
                reference_target_ids=",".join(str(value) for value in lane.reference_target_ids),
                credit_eligible_target_ids=",".join(str(value) for value in lane.credit_eligible_target_ids),
                score=float(lane.score),
                score_derivation=str(lane.score_derivation),
                derived_from_trajectory_metric=bool(lane.derived_from_trajectory_metric),
                terminal_action_observed=lane.terminal_action_observed,
                terminal_asset_observed=lane.terminal_asset_observed,
                terminal_phase_observed=lane.terminal_phase_observed,
                status=str(lane.status),
                artifact_status=str(lane.artifact_status),
                missing_reason=lane.missing_reason,
                failure_reason=lane.failure_reason,
                source=source,
            )
        )
    return rows


def _lane_score_export_rows_from_lane_records(
    *,
    lanes: Iterable[TrialDevLaneScoreRecordV1],
    program_id: str,
    scenario_key: str,
    objective_id: str,
    reference_scenario_by_checksum: dict[str, str],
    source: str,
) -> list[TrialDevLaneScoreExportRowV1]:
    """Flatten raw lane-score records from trajectory-level payloads."""

    rows: list[TrialDevLaneScoreExportRowV1] = []
    for lane in lanes:
        scenario_key, scenario_semantic_id = _lane_scenario_identity(
            lane=lane,
            scenario_key=scenario_key,
            reference_scenario_by_checksum=reference_scenario_by_checksum,
        )
        rows.append(
            TrialDevLaneScoreExportRowV1(
                program_id=program_id,
                scenario_id=scenario_semantic_id,
                scenario_key=scenario_key,
                scenario_semantic_id=scenario_semantic_id,
                objective_id=objective_id,
                phase_id=str(lane.phase_id),
                program_objective_id=str(lane.program_objective_id),
                phase_scoring_objective_id=str(lane.phase_scoring_objective_id),
                lane_id=str(lane.lane_id),
                evaluation_target_checksum=str(lane.evaluation_target_checksum),
                scoring_policy_id=str(lane.scoring_policy_id),
                recoverability_policy_id=str(lane.recoverability_policy_id),
                submitted_target_id=lane.submitted_target_id,
                reference_target_ids=",".join(str(value) for value in lane.reference_target_ids),
                credit_eligible_target_ids=",".join(str(value) for value in lane.credit_eligible_target_ids),
                score=float(lane.score),
                score_derivation=str(lane.score_derivation),
                derived_from_trajectory_metric=bool(lane.derived_from_trajectory_metric),
                terminal_action_observed=lane.terminal_action_observed,
                terminal_asset_observed=lane.terminal_asset_observed,
                terminal_phase_observed=lane.terminal_phase_observed,
                status=str(lane.status),
                artifact_status=str(lane.artifact_status),
                missing_reason=lane.missing_reason,
                failure_reason=lane.failure_reason,
                source=source,
            )
        )
    return rows


def _rank_row(rm: RankMetrics) -> TrialDevRankMetricRowV1:
    return TrialDevRankMetricRowV1(
        program_id=rm.program_id,
        scenario_id=rm.scenario_id,
        objective_id=rm.objective_id,
        phase_id=rm.phase_id,
        pick_type=rm.pick_type,
        agent_top_pick=rm.agent_top_pick,
        reference_top_pick=rm.reference_top_pick,
        agent_top_pick_rank_in_reference=rm.agent_top_pick_rank_in_reference,
        reference_ranking_size=rm.reference_ranking_size,
        bottom_n_concordance=rm.bottom_n_concordance,
        bottom_n=rm.bottom_n,
        utility_regret=rm.utility_regret,
        acceptable_pick=rm.acceptable_pick,
        acceptable_candidate_set=",".join(rm.acceptable_candidate_set or ()),
    )


# ---------------------------------------------------------------------------
# Reduction helpers
# ---------------------------------------------------------------------------


@dataclass
class GroupSummary:
    n: int
    overall_mean: float
    design_mean: float
    evaluation_mean: float
    program_mean: float
    ranking_mean: float
    lane_raw_means: dict[str, float]


def _summarise(rows: Iterable[TrialDevResultsRowV1]) -> GroupSummary:
    rows = list(rows)
    n = len(rows)
    if not n:
        zeros = {lane: 0.0 for lane in LANE_KEYS}
        return GroupSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, zeros)

    def avg(key: str) -> float:
        return float(sum(float(getattr(r, key)) for r in rows) / n)

    lane_means = {lane: avg(f"lane_raw__{lane}") for lane in LANE_KEYS}
    overall_mean = avg("primary_score")
    return GroupSummary(
        n=n,
        overall_mean=overall_mean,
        design_mean=avg("design_score"),
        evaluation_mean=avg("evaluation_score"),
        program_mean=avg("program_score"),
        ranking_mean=avg("ranking_score"),
        lane_raw_means=lane_means,
    )


def _group_by(rows: list[TrialDevResultsRowV1], key: str) -> dict[object, list[TrialDevResultsRowV1]]:
    out: dict[object, list[TrialDevResultsRowV1]] = defaultdict(list)
    for r in rows:
        out[getattr(r, key)].append(r)
    return dict(out)


def _active_lane_means(rows: list[TrialDevResultsRowV1]) -> dict[str, float]:
    """Compute mean lane_active__X across rows for lanes that are active."""
    if not rows:
        return {}
    active_means: dict[str, float] = {}
    for lane in LANE_KEYS:
        ak = f"lane_active__{lane}"
        active_vals = [float(getattr(r, ak)) for r in rows if getattr(r, ak) is not None]
        if active_vals:
            active_means[lane] = sum(active_vals) / len(active_vals)
    return active_means


def build_summary(rows: list[TrialDevResultsRowV1]) -> TrialDevResultsRollupV1:
    """Build complete-suite rollups for ``results_summary.json``."""
    summary = _summarise(rows)
    rollup = TrialDevResultsRollupV1(
        n_items=summary.n,
        overall_mean=summary.overall_mean,
        lane_raw_means=summary.lane_raw_means,
        lane_active_means=_active_lane_means(rows),
    )
    for field, destination in (
        ("phase_id", rollup.by_phase),
        ("scenario_id", rollup.by_scenario),
        ("objective_id", rollup.by_objective),
    ):
        for value, group_rows in _group_by(rows, field).items():
            group = _summarise(group_rows)
            destination[str(value)] = TrialDevGroupRollupV1(
                n=group.n,
                overall_mean=group.overall_mean,
                lane_raw_means=group.lane_raw_means,
            )
    return rollup


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_results_csv(rows: Sequence[BaseModel], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    write_csv_models(out_path, rows)


def write_summary_json(summary: TrialDevResultsSummaryV1, out_path: Path) -> None:
    write_json_model(out_path, summary)


def render_summary_md(summary: TrialDevResultsSummaryV1, run_config: TrialDevRunConfigV1 | None = None) -> str:
    lines: list[str] = []
    lines.append("# TrialDevBench Run Summary")
    lines.append("")
    if run_config is not None:
        run_config_payload = run_config.model_dump(mode="json", exclude_none=True)
        lines.append("## Run config")
        for key in sorted(run_config_payload):
            lines.append(f"- **{key}**: `{run_config_payload[key]}`")
        lines.append("")

    # Completion-aware headline metrics
    cm = summary.completion_metrics
    if cm.n_declared:
        lines.append("## Completion-aware headline")
        lines.append("")
        n_att = int(cm.n_declared)
        n_done = int(cm.n_completed or 0)
        comp_rate = cm.completion_rate
        comp_mean = cm.completed_mean
        fi_mean = cm.failure_imputed_mean
        lines.append(
            f"**Programs declared**: {n_att} | **completed (have trajectory score)**: {n_done} "
            f"| **completion rate**: {(comp_rate or 0):.1%}"
        )
        lines.append("")
        lines.append("| metric | value | interpretation |")
        lines.append("|---|---|---|")
        if comp_mean is not None:
            lines.append(
                f"| completed_mean | **{comp_mean:.3f}** | mean trajectory_primary_score over programs that finished — survivorship-biased, do not use to rank fragile vs robust models |"
            )
        if fi_mean is not None:
            lines.append(
                f"| failure_imputed_mean | **{fi_mean:.3f}** | failed programs scored as 0; corrects for survivorship bias and rewards completion |"
            )
        lines.append("")

    # Violations
    vs = summary.payload.violations
    if vs.n_violations:
        lines.append("## Violations (agent submitted something the harness rejected)")
        lines.append("")
        lines.append(
            f"**Total**: {vs.n_violations}. Each is an agent submission that "
            "failed pydantic / upstream / budget validation; the harness surfaced "
            "the error and the agent retried."
        )
        lines.append("")
        lines.append("| kind | count |")
        lines.append("|---|---|")
        for k, n in sorted(vs.by_kind.items()):
            lines.append(f"| {k} | {n} |")
        lines.append("")
        lines.append("| phase | count |")
        lines.append("|---|---|")
        for k, n in sorted(vs.by_phase.items()):
            lines.append(f"| {k} | {n} |")
        lines.append("")

    # Rank metrics
    rm = summary.payload.rank_metrics
    if rm.n_phases:
        lines.append("## Rank quality on top pick")
        lines.append("")
        if rm.mean_top_pick_rank is not None:
            mean_size = rm.mean_reference_ranking_size
            size_str = f" out of mean {mean_size:.0f}" if mean_size else ""
            std = rm.rank_std
            std_str = f" (std {std:.2f})" if std is not None else ""
            quality = rm.mean_top_pick_quality
            quality_str = (
                f"; mean top_pick_quality = {quality:.2f} (1.0 = always picked policy reference's best, 0 = worst)"
                if quality is not None
                else ""
            )
            lines.append(
                f"Mean rank of agent's top pick in policy reference's ranking: **{rm.mean_top_pick_rank:.2f}**"
                f"{size_str}{std_str}. Median: **{rm.median_top_pick_rank}**"
                f"{quality_str}. n with measurable rank: {rm.n_with_rank} of {rm.n_phases}."
            )
            lines.append("")
        if rm.by_phase:
            lines.append("Per phase:")
            lines.append("")
            lines.append("| phase | n | mean rank | median rank | rank std | mean top_pick_quality |")
            lines.append("|---|---|---|---|---|---|")
            for phase in sorted(rm.by_phase):
                row = rm.by_phase[phase]
                std = row.rank_std
                std_s = f"{std:.2f}" if std is not None else "—"
                qual = row.mean_top_pick_quality
                qual_s = f"{qual:.2f}" if qual is not None else "—"
                lines.append(f"| {phase} | {row.n} | {row.mean_rank:.2f} | {row.median_rank} | {std_s} | {qual_s} |")
            lines.append("")
        if rm.mean_bottom_n_concordance is not None:
            lines.append(
                f"Mean bottom-N concordance (overlap between agent's worst-N and policy reference's worst-N): "
                f"**{rm.mean_bottom_n_concordance:.2f}**. n with measurable bottom: "
                f"{rm.n_with_bottom_n}."
            )
            lines.append("")

    payload = summary.payload.results.model_dump(mode="python")
    n = int(payload.get("n_items", 0))
    lines.append(f"## Complete suite ({n} items)")
    overall = float(payload.get("overall_mean", 0.0))
    lines.append(f"**overall_mean (active-only)**: {overall:.3f}")
    lines.append("")
    lines.append("Per-lane scores (active-only):")
    lines.append("")
    lines.append("| lane | mean |")
    lines.append("|---|---|")
    active_means = payload.get("lane_active_means") or {}
    for lane in sorted(LANE_KEYS):
        active = active_means.get(lane)
        if active is None:
            active = payload.get("lane_raw_means", {}).get(lane, 0.0)
        lines.append(f"| {lane} | {float(active):.3f} |")
    lines.append("")

    by_phase = payload.get("by_phase", {})
    if by_phase:
        lines.append("Per phase:")
        lines.append("")
        lines.append("| phase | n | overall |")
        lines.append("|---|---|---|")
        for phase in sorted(by_phase):
            row = by_phase[phase]
            lines.append(f"| {phase} | {row.get('n', 0)} | {float(row.get('overall_mean', 0.0)):.3f} |")
        lines.append("")
    by_scenario = payload.get("by_scenario", {})
    if by_scenario:
        lines.append("Per scenario:")
        lines.append("")
        lines.append("| scenario | n | overall |")
        lines.append("|---|---|---|")
        for scenario in sorted(by_scenario):
            row = by_scenario[scenario]
            lines.append(f"| {scenario} | {row.get('n', 0)} | {float(row.get('overall_mean', 0.0)):.3f} |")
        lines.append("")
    by_objective = payload.get("by_objective", {})
    if by_objective:
        lines.append("Per objective:")
        lines.append("")
        lines.append("| objective | n | overall |")
        lines.append("|---|---|---|")
        for objective in sorted(by_objective):
            row = by_objective[objective]
            lines.append(f"| {objective} | {row.get('n', 0)} | {float(row.get('overall_mean', 0.0)):.3f} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def aggregate_run(
    output_root: Path,
    bundle_root: Path,
    *,
    run_config: TrialDevRunConfigV1 | None = None,
    policy: AggregatePolicy | None = None,
) -> TrialDevResultsSummaryV1:
    """Walk an output directory and produce all aggregate artefacts.

    Writes:
      * ``results_full.csv`` — one row per scored item with raw + active lanes
      * ``rank_metrics.csv`` — one row per phase with rank-gap + bottom-N concordance
      * ``results_summary.json``
      * ``RESULTS_SUMMARY.md``

    Returns the schema-bearing summary model.
    """
    policy = policy or AggregatePolicy()
    manifest = AggregateManifestV1(
        harness_version=harness_version,
        timestamp_utc=datetime.now(UTC),
        input_run_dir=str(output_root),
        bundle_dir=str(bundle_root),
        policy_strict=bool(policy.strict),
        allow_incomplete_artifacts=bool(policy.allow_incomplete_artifacts),
    )

    rows, lane_score_rows, rank_rows, stick_twist_rows, alignment_rows = collect_rows(
        output_root, bundle_root, policy=policy, manifest=manifest
    )
    phase_resource_rows, programme_resource_rows = _collect_resource_rows(
        output_root,
        policy=policy,
        manifest=manifest,
    )
    violation_rows = _collect_violations(output_root, policy=policy, manifest=manifest)
    completion_metrics = _compute_completion_metrics(output_root, policy=policy, manifest=manifest)
    write_results_csv(rows, output_root / "results_full.csv")
    write_results_csv(lane_score_rows, output_root / "lane_scores.csv")
    write_results_csv(rank_rows, output_root / "rank_metrics.csv")
    write_results_csv(violation_rows, output_root / "violations.csv")
    write_results_csv(stick_twist_rows, output_root / "stick_twist.csv")
    write_results_csv(alignment_rows, output_root / "objective_alignment.csv")
    write_results_csv(phase_resource_rows, output_root / "phase_resources.csv")
    write_results_csv(
        programme_resource_rows,
        output_root / "programme_resources.csv",
    )
    payload = TrialDevResultsPayloadV1(
        results=build_summary(rows),
        rank_metrics=_summarise_rank_rows(rank_rows),
        violations=_summarise_violations(violation_rows),
        stick_twist=_summarise_stick_twist(stick_twist_rows),
        objective_alignment=_summarise_objective_alignment(alignment_rows),
        completion_metrics=completion_metrics,
    )
    summary = TrialDevResultsSummaryV1(
        schema_id="trialagentbench_trialdev_results_summary_v1",
        schema_version=1,
        completion_metrics=completion_metrics,
        payload=payload,
    )
    write_summary_json(summary, output_root / "results_summary.json")
    md = render_summary_md(summary, run_config=run_config)
    (output_root / "RESULTS_SUMMARY.md").write_text(md, encoding="utf-8")
    write_json_model(output_root / "AGGREGATE_MANIFEST.json", manifest)
    return summary


def _compute_completion_metrics(
    output_root: Path,
    *,
    policy: AggregatePolicy,
    manifest: AggregateManifestV1,
) -> TrialDevCompletionMetricsV1:
    """Per-program checkpoint-complete score, plus failure-imputed completion.

    A program is "completed" if its trajectory_grade.json or chain_summary.json
    yields a per-program score over all required reached checkpoints. Programs that errored or timed
    out before producing a trajectory grade are counted as failures with score 0
    in the failure-imputed mean — which prevents survivorship bias from
    inflating the headline of fragile models.
    """
    programs_dir = output_root / "programs"
    if not programs_dir.is_dir():
        return TrialDevCompletionMetricsV1()

    declared_program_ids: list[str] | None = None
    coverage_path = output_root / "coverage_report.json"
    if not coverage_path.is_file():
        exc = FileNotFoundError(f"Missing required coverage_report.json: {coverage_path}")
        if policy.allow_incomplete_artifacts:
            manifest.tolerated_failures.append(
                ToleratedFailureV1(
                    code=TrialDevFailureCode.coverage_report_invalid,
                    message=str(exc),
                    path=str(coverage_path),
                )
            )
            declared_program_ids = sorted(p.name for p in programs_dir.iterdir() if p.is_dir())
        else:
            raise exc
    else:
        from trialagentbench_harness.contracts.core.coverage import TrialDevCoverageReportV1

        try:
            cov_v1 = read_json_model(TrialDevCoverageReportV1, coverage_path)
        except (FileNotFoundError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            if policy.allow_incomplete_artifacts:
                manifest.tolerated_failures.append(
                    ToleratedFailureV1(
                        code=TrialDevFailureCode.coverage_report_invalid,
                        message=str(exc),
                        path=str(coverage_path),
                    )
                )
                declared_program_ids = sorted(p.name for p in programs_dir.iterdir() if p.is_dir())
            else:
                raise
        else:
            if not cov_v1.programs:
                empty_coverage_error = ValueError(f"coverage_report.json has no declared programs: {coverage_path}")
                if policy.allow_incomplete_artifacts:
                    manifest.tolerated_failures.append(
                        ToleratedFailureV1(
                            code=TrialDevFailureCode.coverage_report_invalid,
                            message=str(empty_coverage_error),
                            path=str(coverage_path),
                        )
                    )
                    declared_program_ids = sorted(p.name for p in programs_dir.iterdir() if p.is_dir())
                else:
                    raise empty_coverage_error
            else:
                declared_program_ids = [p.program_id for p in cov_v1.programs]

    per_program_rows: list[TrialDevProgramCompletionV1] = []
    completed_score_sum = 0.0
    n_completed = 0
    n_present = 0

    for program_id in declared_program_ids:
        program_dir = programs_dir / program_id
        if not program_dir.is_dir():
            per_program_rows.append(
                TrialDevProgramCompletionV1(
                    program_id=program_id,
                    trajectory_primary_score=None,
                    completed=False,
                    program_status="missing_program_dir",
                )
            )
            continue
        n_present += 1

        traj_score: float | None = None
        chain: TrialDevChainSummaryV1 | None = None
        chain_path = program_dir / "chain_summary.json"
        if chain_path.is_file():
            chain = _read_chain_summary(program_dir, policy=policy, manifest=manifest)
            if chain and chain.trajectory_metrics:
                traj_score = chain.trajectory_metrics.programme_primary_score

        if chain is not None and chain.execution_status == "model_turn_limit":
            status = "model_noncompletion"
        elif chain is not None and chain.execution_status == "model_invalid_submission":
            status = "model_invalid_submission"
        elif chain is not None and chain.execution_status == "infrastructure_timeout":
            status = "infrastructure_timeout"
        elif chain is not None and chain.execution_status == "infrastructure_error":
            status = "infrastructure_error"
        elif traj_score is not None:
            completed_score_sum += float(traj_score)
            n_completed += 1
            status = "completed"
        else:
            status = "missing_or_unusable_grade"

        per_program_rows.append(
            TrialDevProgramCompletionV1(
                program_id=program_id,
                trajectory_primary_score=traj_score,
                completed=status == "completed",
                program_status=status,
            )
        )

    n_declared = len(declared_program_ids)
    return TrialDevCompletionMetricsV1(
        n_declared=n_declared,
        n_present=n_present,
        n_completed=n_completed,
        completion_rate=(n_completed / n_declared) if n_declared else None,
        completed_mean=(completed_score_sum / n_completed) if n_completed else None,
        failure_imputed_mean=(completed_score_sum / n_declared) if n_declared else None,
        per_program=per_program_rows,
    )


def _alignment_by_primary(
    rows: list[TrialDevObjectiveAlignmentRowV1],
) -> dict[str, TrialDevObjectiveAlignmentByPrimaryV1]:
    """Group alignment rows by primary_objective and report rate per group."""
    out: dict[str, TrialDevObjectiveAlignmentByPrimaryV1] = {}
    for r in rows:
        prim = str(r.primary_objective or "")
        if int(r.n_free_phases) == 0:
            continue
        bucket = out.setdefault(prim, TrialDevObjectiveAlignmentByPrimaryV1())
        bucket.n_programs += 1
        bucket.n_aligned += int(r.n_aligned or 0)
        bucket.n_free_total += int(r.n_free_phases or 0)
    for _prim, b in out.items():
        b.alignment_rate = b.n_aligned / b.n_free_total if b.n_free_total else None
    return out


def _collect_violations(
    output_root: Path,
    *,
    policy: AggregatePolicy,
    manifest: AggregateManifestV1,
) -> list[TrialDevViolationRowV1]:
    """Walk every chain_summary.json and flatten its `violations` list."""
    out: list[TrialDevViolationRowV1] = []
    programs_root = output_root / "programs"
    if not programs_root.is_dir():
        return out
    for program_dir in sorted(programs_root.iterdir()):
        if not program_dir.is_dir():
            continue
        chain_path = program_dir / "chain_summary.json"
        if not chain_path.is_file():
            continue
        chain = _read_chain_summary(program_dir, policy=policy, manifest=manifest)
        if chain is None:
            continue
        program_id = str(chain.program_id or program_dir.name)
        scenario_id = str(chain.scenario_id or "")
        objective_id = str(chain.objective_id or "")
        for v in chain.violations or []:
            out.append(
                TrialDevViolationRowV1(
                    program_id=program_id,
                    scenario_id=scenario_id,
                    objective_id=objective_id,
                    phase_id=v.get("phase_id"),
                    kind=v.get("kind"),
                    error=str(v.get("error", ""))[:300],
                )
            )
    return out


def _summarise_violations(violation_rows: list[TrialDevViolationRowV1]) -> TrialDevViolationsSummaryV1:
    if not violation_rows:
        return TrialDevViolationsSummaryV1()
    by_kind: dict[str, int] = defaultdict(int)
    by_phase: dict[str, int] = defaultdict(int)
    for row in violation_rows:
        by_kind[str(row.kind or "unknown")] += 1
        by_phase[str(row.phase_id or "unknown")] += 1
    return TrialDevViolationsSummaryV1(
        n_violations=len(violation_rows),
        by_kind=dict(by_kind),
        by_phase=dict(by_phase),
    )


def _summarise_rank_rows(rank_rows: list[TrialDevRankMetricRowV1]) -> TrialDevRankMetricsSummaryV1:
    if not rank_rows:
        return TrialDevRankMetricsSummaryV1()
    import math

    ranks = [
        int(r.agent_top_pick_rank_in_reference) for r in rank_rows if r.agent_top_pick_rank_in_reference is not None
    ]
    sizes = [int(r.reference_ranking_size) for r in rank_rows if r.reference_ranking_size is not None]
    # top_pick_quality = (size - rank + 1) / size: 1.00 = picked policy reference's best,
    # 0.10 (or wherever the bottom is) = picked the worst.
    qualities = [
        (int(r.reference_ranking_size) - int(r.agent_top_pick_rank_in_reference) + 1) / int(r.reference_ranking_size)
        for r in rank_rows
        if r.agent_top_pick_rank_in_reference is not None and r.reference_ranking_size is not None
    ]
    bottoms = [float(r.bottom_n_concordance) for r in rank_rows if r.bottom_n_concordance is not None]

    def _std(values: list[float]) -> float | None:
        if len(values) < 2:
            return None
        m = sum(values) / len(values)
        return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

    # Per-phase breakdown: lets us see e.g. "phase1 mean top_pick_quality is 0.65"
    by_phase: dict[str, TrialDevRankPhaseSummaryV1] = {}
    for phase in {str(r.phase_id) for r in rank_rows if r.phase_id}:
        rows = [r for r in rank_rows if str(r.phase_id) == phase]
        rk = [int(r.agent_top_pick_rank_in_reference) for r in rows if r.agent_top_pick_rank_in_reference is not None]
        qs = [
            (int(r.reference_ranking_size) - int(r.agent_top_pick_rank_in_reference) + 1)
            / int(r.reference_ranking_size)
            for r in rows
            if r.agent_top_pick_rank_in_reference is not None and r.reference_ranking_size is not None
        ]
        if rk:
            by_phase[phase] = TrialDevRankPhaseSummaryV1(
                n=len(rk),
                mean_rank=sum(rk) / len(rk),
                median_rank=sorted(rk)[len(rk) // 2],
                rank_std=_std([float(x) for x in rk]),
                mean_top_pick_quality=(sum(qs) / len(qs)) if qs else None,
            )

    return TrialDevRankMetricsSummaryV1(
        n_phases=len(rank_rows),
        n_with_rank=len(ranks),
        mean_top_pick_rank=(sum(ranks) / len(ranks)) if ranks else None,
        median_top_pick_rank=sorted(ranks)[len(ranks) // 2] if ranks else None,
        rank_std=_std([float(x) for x in ranks]),
        mean_top_pick_quality=(sum(qualities) / len(qualities)) if qualities else None,
        mean_reference_ranking_size=(sum(sizes) / len(sizes)) if sizes else None,
        n_with_bottom_n=len(bottoms),
        mean_bottom_n_concordance=(sum(bottoms) / len(bottoms)) if bottoms else None,
        by_phase=by_phase,
    )


def _summarise_stick_twist(rows: list[TrialDevStickTwistRowV1]) -> TrialDevStickTwistSummaryV1:
    if not rows:
        return TrialDevStickTwistSummaryV1()
    n_pivot = sum(1 for r in rows if r.pivoted)
    return TrialDevStickTwistSummaryV1(
        n_programs_with_both=len(rows),
        n_pivoted=n_pivot,
        pivot_rate=(n_pivot / len(rows)) if rows else None,
    )


def _summarise_objective_alignment(rows: list[TrialDevObjectiveAlignmentRowV1]) -> TrialDevObjectiveAlignmentSummaryV1:
    free_rows = [r for r in rows if int(r.n_free_phases) > 0]
    n_align_total = sum(int(r.n_aligned) for r in free_rows)
    n_free_total = sum(int(r.n_free_phases) for r in free_rows)
    return TrialDevObjectiveAlignmentSummaryV1(
        n_programs_with_free_phase=len(free_rows),
        alignment_rate_overall=(n_align_total / n_free_total) if n_free_total else None,
        by_primary_objective=_alignment_by_primary(rows),
    )


__all__ = [
    "collect_rows",
    "build_summary",
    "render_summary_md",
    "aggregate_run",
]
