"""Secondary per-program ranking and objective-alignment diagnostics.

* **Rank gap on top pick** — how good was the single drug the agent
  recommended? Measured as its rank in the route-specific reference ranking for that
  phase × objective. Rank 1 = picked policy reference's best.
* **Bottom-N concordance** — did the agent correctly identify the
  worst N drugs? Useful as a "did the agent recognise dead-end
  candidates" signal at obs_review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trialagentbench_harness.contracts.core.runs import (
    TrialDevChainSummaryV1,
    TrialDevPhaseStepSummaryV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_grades import TrialDevGradeRecordV1
from trialagentbench_harness.io import read_json, read_json_model
from trialagentbench_harness.trialdev.share.validate import candidate_ids_by_role_v1

PHASE_ORDER = ("observational_review", "phase1", "phase2", "phase3")


@dataclass
class RankMetrics:
    program_id: str
    scenario_id: str
    objective_id: str
    phase_id: str
    agent_top_pick: str | None
    reference_top_pick: str | None
    agent_top_pick_rank_in_reference: int | None  # 1-indexed; None if absent from the reference ranking
    reference_ranking_size: int
    bottom_n_concordance: (
        float | None
    )  # fraction of agent's last-N matching policy reference's last-N (None if can't compute)
    bottom_n: int  # the N used
    # Utility loss relative to the best publicly recoverable candidate.
    utility_regret: float | None = None
    acceptable_pick: bool | None = None
    acceptable_candidate_set: tuple[str, ...] = ()
    # "first_pick" = agent actively decides which drug to nominate (obs_review,
    # phase1). "locked" = the asset is locked from phase1; phase2/3 picks are
    # not fresh decisions, so positional rank/regret here just mirrors phase1.
    pick_type: str = "first_pick"


# ---------------------------------------------------------------------------
# Rank metrics
# ---------------------------------------------------------------------------


def _reference_ranking_for(
    bundle_root: Path,
    *,
    scenario_id: str,
    phase_id: str,
    objective_id: str,
    method_route_id: str | None = None,
    candidate_only: bool = True,
) -> list[tuple[str, float]]:
    """Policy-reference ranking for one scenario, phase, and objective.

    ``candidate_only`` retains candidates whose public catalog role is
    ``investigational``. The agent ranks treatment regimens to advance, so for objectives like
    ``cost_effective_best`` (where the control row legitimately tops the
    raw objective_score because doing nothing has zero cost) the
    apples-to-apples comparison is candidate-only.
    """
    scenario_root = bundle_root / f"scenario_{scenario_id}"
    path = scenario_root / "grader" / "drug_ranking_reference_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"TrialDev policy reference ranking is missing: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"drug_ranking_reference_manifest must be a JSON object: {path}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"drug_ranking_reference_manifest must contain a records array: {path}")
    matching_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"drug_ranking_reference_manifest records must be objects: {path}")
        if (
            str(record.get("phase_id")) != phase_id
            or str(record.get("objective_id")) != objective_id
            or str(record.get("metric")) != "objective_score"
        ):
            continue
        matching_records.append(record)
    available_method_routes = {
        str(record["method_route_id"])
        for record in matching_records
        if isinstance(record.get("method_route_id"), str) and str(record["method_route_id"])
    }
    if method_route_id is None and len(available_method_routes) > 1:
        raise ValueError(
            "Reference ranking requires a method_route_id when multiple method-specific reference results exist."
        )
    if method_route_id is not None:
        matching_records = [
            record for record in matching_records if str(record.get("method_route_id") or "") == method_route_id
        ]
        if not matching_records:
            raise ValueError(f"Policy reference ranking does not contain method_route_id={method_route_id!r}.")

    rows: list[tuple[str, float]] = []
    for record in matching_records:
        candidate_ids = record.get("candidate_drug_ids")
        value = record.get("value")
        if not isinstance(candidate_ids, list) or len(candidate_ids) != 1 or not isinstance(candidate_ids[0], str):
            raise ValueError("Each objective-score record must identify exactly one candidate.")
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError("Each objective-score record must contain one finite numeric value.")
        rows.append((candidate_ids[0], float(value)))
    if len({candidate_id for candidate_id, _ in rows}) != len(rows):
        raise ValueError("Policy reference ranking contains duplicate candidate identifiers within one method route.")
    if candidate_only:
        investigational = set(candidate_ids_by_role_v1(scenario_root=scenario_root)["investigational"])
        rows = [(candidate_id, value) for candidate_id, value in rows if candidate_id in investigational]
    rows.sort(key=lambda kv: -kv[1])
    return rows


def _recoverability_for(
    bundle_root: Path,
    *,
    scenario_id: str,
    phase_id: str,
    objective_id: str,
    method_route_id: str | None = None,
) -> dict[str, Any]:
    """Load the recoverability manifest record for one phase/objective.

    Returns a mapping ``candidate_drug_id -> record`` plus the record-level
    acceptable candidate/action sets.
    """
    path = bundle_root / f"scenario_{scenario_id}/grader/recoverability_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"TrialDev recoverability manifest is missing: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"recoverability_manifest must be a JSON object: {path}")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError(f"recoverability_manifest must contain records array: {path}")
    matching = [
        record
        for record in raw_records
        if isinstance(record, dict)
        and str(record.get("phase_id")) == str(phase_id)
        and str(record.get("objective_id")) == str(objective_id)
    ]
    available_method_routes = {
        str(record["method_route_id"])
        for record in matching
        if isinstance(record.get("method_route_id"), str) and str(record["method_route_id"])
    }
    if method_route_id is None and len(available_method_routes) > 1:
        raise ValueError(
            "Recoverability lookup requires a method_route_id when multiple method-specific records exist."
        )
    if method_route_id is not None:
        matching = [record for record in matching if str(record.get("method_route_id") or "") == method_route_id]
    if len(matching) != 1:
        raise ValueError(
            f"recoverability_manifest must contain exactly one record for "
            f"phase_id={phase_id!r}, objective_id={objective_id!r}: {path}"
        )
    record_payload = matching[0]
    by_drug: dict[str, dict[str, Any]] = {}
    for record in record_payload.get("candidate_records", []) or []:
        drug = str(record.get("candidate_drug_id") or "")
        if drug:
            by_drug[drug] = dict(record)
    return {
        "by_drug": by_drug,
        "acceptable_candidate_set": tuple(record_payload.get("acceptable_candidate_set") or ()),
        "acceptable_action_set": tuple(record_payload.get("acceptable_action_set") or ()),
        "policy": str(record_payload.get("policy") or ""),
    }


def _attach_recoverability(rm: RankMetrics, recovery: dict[str, Any]) -> RankMetrics:
    """Annotate rank metrics with raw utility regret and accepted-set status.

    Uses the agent's top pick to look up the per-drug record.
    """
    if rm.agent_top_pick is None:
        return rm
    by_drug: dict[str, dict[str, Any]] = recovery.get("by_drug") or {}
    acceptable_set = tuple(recovery.get("acceptable_candidate_set") or ())
    record = by_drug.get(str(rm.agent_top_pick))
    if record is None:
        rm.utility_regret = None
        rm.acceptable_pick = False
    else:
        raw_regret = record.get("policy_reference_regret")
        rm.utility_regret = None if raw_regret is None else float(raw_regret)
        rm.acceptable_pick = bool(record.get("acceptable_candidate", False))
    rm.acceptable_candidate_set = acceptable_set
    return rm


def rank_metrics_for_obs_review(program_dir: Path, *, bundle_root: Path, bottom_n: int = 3) -> RankMetrics | None:
    """Rank-quality metrics for the obs_review submission."""
    chain = read_json_model(TrialDevChainSummaryV1, program_dir / "chain_summary.json").model_dump(mode="json")
    summary_path = program_dir / "obs_review" / "obs_review_summary.json"
    if not summary_path.is_file():
        return None
    from trialagentbench_harness.contracts.core.runs import TrialDevObsReviewSummaryV1

    summary = read_json_model(TrialDevObsReviewSummaryV1, summary_path)
    if summary.method_route_id is None:
        return None
    program_id = str(chain.get("program_id") or program_dir.name)
    scenario_id = str(chain.get("scenario_id") or "")
    objective_id = str(chain.get("objective_id") or "")

    reference_ranking = _reference_ranking_for(
        bundle_root,
        scenario_id=scenario_id,
        phase_id="observational_review",
        objective_id=objective_id,
        method_route_id=summary.method_route_id,
    )
    if not reference_ranking:
        return None

    agent_ranked = list(summary.ranked_drug_ids or [])
    agent_top = summary.recommended_drug_id or (agent_ranked[0] if agent_ranked else None)
    reference_top = reference_ranking[0][0]
    reference_drugs = [d for d, _ in reference_ranking]
    rank_in_reference = (reference_drugs.index(agent_top) + 1) if agent_top in reference_drugs else None

    bottom_concordance: float | None = None
    if len(agent_ranked) >= bottom_n and len(reference_drugs) >= bottom_n:
        agent_bottom = set(agent_ranked[-bottom_n:])
        reference_bottom = set(reference_drugs[-bottom_n:])
        bottom_concordance = len(agent_bottom & reference_bottom) / float(bottom_n)

    rm = RankMetrics(
        program_id=program_id,
        scenario_id=scenario_id,
        objective_id=objective_id,
        phase_id="observational_review",
        agent_top_pick=str(agent_top) if agent_top else None,
        reference_top_pick=str(reference_top),
        agent_top_pick_rank_in_reference=rank_in_reference,
        reference_ranking_size=len(reference_drugs),
        bottom_n_concordance=bottom_concordance,
        bottom_n=bottom_n,
        pick_type="first_pick",
    )
    return _attach_recoverability(
        rm,
        _recoverability_for(
            bundle_root,
            scenario_id=scenario_id,
            phase_id="observational_review",
            objective_id=objective_id,
            method_route_id=summary.method_route_id,
        ),
    )


def rank_metrics_for_phase(
    program_dir: Path,
    *,
    bundle_root: Path,
    phase_id: str,
    bottom_n: int = 3,
) -> RankMetrics | None:
    """Rank-quality metrics for one phase trial submission (phase1/2/3)."""
    chain = read_json_model(TrialDevChainSummaryV1, program_dir / "chain_summary.json").model_dump(mode="json")
    workdir = program_dir / "agent_workdir"
    summary_path = workdir / f"phase_{phase_id}" / "phase_step_summary.json"
    if not summary_path.is_file():
        return None
    summary = read_json_model(TrialDevPhaseStepSummaryV1, summary_path)
    program_id = str(chain.get("program_id") or program_dir.name)
    scenario_id = str(chain.get("scenario_id") or "")
    objective_id = str(
        (summary.request.selection_objective if summary.request is not None else None)
        or chain.get("objective_id")
        or ""
    )

    reference_ranking = _reference_ranking_for(
        bundle_root, scenario_id=scenario_id, phase_id=phase_id, objective_id=objective_id
    )
    if not reference_ranking:
        return None

    agent_ranked = list((summary.analysis.ranked_drug_ids if summary.analysis is not None else []) or [])
    agent_top = (summary.analysis.selected_winner_drug_id if summary.analysis is not None else None) or (
        agent_ranked[0] if agent_ranked else None
    )
    reference_drugs = [d for d, _ in reference_ranking]
    rank_in_reference = (reference_drugs.index(agent_top) + 1) if agent_top in reference_drugs else None

    bottom_concordance: float | None = None
    if len(agent_ranked) >= bottom_n and len(reference_drugs) >= bottom_n:
        agent_bottom = set(agent_ranked[-bottom_n:])
        reference_bottom = set(reference_drugs[-bottom_n:])
        bottom_concordance = len(agent_bottom & reference_bottom) / float(bottom_n)

    # Pick semantics: phase1 is a fresh nomination; phase2/3 inherit the
    # phase1 lock so the "pick" there is structurally constrained.
    pick_type = "first_pick" if phase_id == "phase1" else "locked"
    rm = RankMetrics(
        program_id=program_id,
        scenario_id=scenario_id,
        objective_id=objective_id,
        phase_id=phase_id,
        agent_top_pick=str(agent_top) if agent_top else None,
        reference_top_pick=str(reference_drugs[0]),
        agent_top_pick_rank_in_reference=rank_in_reference,
        reference_ranking_size=len(reference_drugs),
        bottom_n_concordance=bottom_concordance,
        bottom_n=bottom_n,
        pick_type=pick_type,
    )
    return _attach_recoverability(
        rm,
        _recoverability_for(bundle_root, scenario_id=scenario_id, phase_id=phase_id, objective_id=objective_id),
    )


@dataclass
class StickTwistResult:
    program_id: str
    obs_pick: str | None
    phase1_pick: str | None
    pivoted: bool


@dataclass
class ObjectiveAlignmentResult:
    """Track whether per-phase ``selection_objective`` matches the program objective.

    Phase 1 forces ``benefit_risk`` for every program; it is flagged
    ``forced=True`` and excluded from the alignment rate.
    Phase 2/3 are free choice; they count toward alignment.
    Obs_review's ``selection_objective`` is auto-set to the program's
    primary by our harness (the agent doesn't pick), so it always aligns
    and is excluded.
    """

    program_id: str
    primary_objective: str
    per_phase: dict[str, dict[str, Any]]  # phase_id -> {selected, forced, aligned}
    n_free_phases: int
    n_aligned: int
    alignment_rate: float | None  # n_aligned / n_free_phases, or None if no free phases


def objective_alignment_for_program(
    program_dir: Path,
    *,
    bundle_root: Path,
) -> ObjectiveAlignmentResult | None:
    """Compute per-phase objective alignment for one program.

    For each phase request the agent submitted, compare:
      ``request.selection_objective``  vs  program's primary_objective
    Mark forced if the phase's eval_contract only allows one objective.
    """
    chain_path = program_dir / "chain_summary.json"
    if not chain_path.is_file():
        return None
    chain = read_json_model(TrialDevChainSummaryV1, chain_path).model_dump(mode="json")
    program_id = str(chain.get("program_id") or program_dir.name)
    scenario_id = str(chain.get("scenario_id") or "")
    primary = str(chain.get("objective_id") or "")
    if not scenario_id or not primary:
        return None

    eval_contract_path = bundle_root / f"scenario_{scenario_id}/public/eval_contract.json"
    allowed_per_phase: dict[str, list[str]] = {}
    if eval_contract_path.is_file():
        ec = read_json(eval_contract_path)
        if not isinstance(ec, dict):
            raise ValueError(f"eval_contract.json must be a JSON object: {eval_contract_path}")
        for m in ec.get("phase_modules", []) or []:
            allowed_per_phase[str(m.get("phase_id"))] = list(m.get("allowed_selection_objectives") or [])

    per_phase: dict[str, dict[str, Any]] = {}
    n_free = 0
    n_aligned = 0
    for phase_id in ("phase1", "phase2", "phase3"):
        summ_path = program_dir / "agent_workdir" / f"phase_{phase_id}" / "phase_step_summary.json"
        if not summ_path.is_file():
            continue
        summ = read_json_model(TrialDevPhaseStepSummaryV1, summ_path)
        selected = str((summ.request.selection_objective if summ.request is not None else "") or "")
        allowed = list(allowed_per_phase.get(phase_id) or [])
        forced = len(allowed) == 1
        aligned = selected == primary
        per_phase[phase_id] = {
            "selected": selected,
            "allowed": allowed,
            "forced": forced,
            "aligned": aligned,
        }
        if not forced:
            n_free += 1
            if aligned:
                n_aligned += 1
    rate = (n_aligned / n_free) if n_free else None
    return ObjectiveAlignmentResult(
        program_id=program_id,
        primary_objective=primary,
        per_phase=per_phase,
        n_free_phases=n_free,
        n_aligned=n_aligned,
        alignment_rate=rate,
    )


def stick_twist_for_program(program_dir: Path) -> StickTwistResult | None:
    """Compare the agent's obs_review top pick against its phase1 candidate_drug_id.

    Returns None if either obs or phase1 wasn't run. Useful for measuring
    whether phase1 is genuinely re-evaluating the obs pick or just inheriting it.
    """
    program_id = program_dir.name
    obs_path = program_dir / "obs_review/obs_review_submission.json"
    if not obs_path.is_file():
        return None
    obs_grade_path = program_dir / "obs_review/grade_report.json"
    obs_pick = None
    if obs_grade_path.is_file():
        obs_report = read_json_model(TrialDevGradeRecordV1, obs_grade_path)
        obs_pick = obs_report.selected_winner_drug_id

    p1_summary_path = program_dir / "agent_workdir" / "phase_phase1" / "phase_step_summary.json"
    if not p1_summary_path.is_file():
        return None
    p1 = read_json_model(TrialDevPhaseStepSummaryV1, p1_summary_path)
    p1_pick = p1.decision.candidate_drug_id if p1.decision is not None else None
    return StickTwistResult(
        program_id=program_id,
        obs_pick=obs_pick,
        phase1_pick=p1_pick,
        pivoted=bool(obs_pick and p1_pick and obs_pick != p1_pick),
    )


__all__ = [
    "RankMetrics",
    "StickTwistResult",
    "ObjectiveAlignmentResult",
    "rank_metrics_for_obs_review",
    "rank_metrics_for_phase",
    "stick_twist_for_program",
    "objective_alignment_for_program",
]
