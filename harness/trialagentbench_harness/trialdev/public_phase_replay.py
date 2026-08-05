"""Replay TrialDev randomized phases into public decision and design evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from trialagentbench_harness.contracts.trialdev.trialdev_public_phase_replay import (
    TrialDevPhaseReplayCaseV1,
    TrialDevPublicCandidateDecisionV1,
    TrialDevPublicIntervalV1,
    TrialDevPublicPhaseReplayRecordV1,
    TrialDevPublicSafetyComponentV1,
)
from trialagentbench_harness.trialdev.grading.decision_evidence import (
    TrialDevPhaseDecisionWitnessV1,
    derive_phase_decision_witness_v1,
    derive_phase_design_witness_v1,
)
from trialagentbench_harness.trialdev.grading.design_frontier import (
    derive_phase_design_efficiency_v1,
)
from trialagentbench_harness.trialdev.grading.hashing import sha256_file_hex
from trialagentbench_harness.trialdev.share.materialize import materialize_trial_view_v1


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Public phase replay requires object-valued {label}.")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"Public phase replay requires finite numeric {label}.")
    return float(value)


def _interval(*, estimate: object, interval: object, label: str) -> TrialDevPublicIntervalV1:
    if not isinstance(interval, list | tuple) or len(interval) != 2:
        raise ValueError(f"Public phase replay requires a two-value {label} interval.")
    return TrialDevPublicIntervalV1(
        estimate=_number(estimate, label=f"{label} estimate"),
        lower=_number(interval[0], label=f"{label} lower bound"),
        upper=_number(interval[1], label=f"{label} upper bound"),
    )


def _safety_component(
    evidence: Mapping[str, object],
    *,
    component_id: str,
) -> TrialDevPublicSafetyComponentV1:
    if component_id == "serious_ae":
        role = "hard_gate"
        treated_prefix = "treated_serious"
        control_prefix = "control_serious"
        excess_prefix = "serious_rate"
    elif component_id == "discontinuation":
        role_value = evidence.get("role")
        if role_value not in {"hard_gate", "diagnostic_only"}:
            raise ValueError("Discontinuation evidence requires a hard_gate or diagnostic_only role.")
        role = str(role_value)
        treated_prefix = "treated"
        control_prefix = "control"
        excess_prefix = "rate"
    else:
        raise ValueError(f"Unsupported safety component: {component_id!r}.")
    return TrialDevPublicSafetyComponentV1.model_validate(
        {
            "component_id": component_id,
            "role": role,
            "treated": _interval(
                estimate=evidence.get(f"{treated_prefix}_rate"),
                interval=evidence.get(f"{treated_prefix}_rate_interval"),
                label=f"{component_id} treated risk",
            ),
            "control": _interval(
                estimate=evidence.get(f"{control_prefix}_rate"),
                interval=evidence.get(f"{control_prefix}_rate_interval"),
                label=f"{component_id} control risk",
            ),
            "excess": _interval(
                estimate=evidence.get(f"{excess_prefix}_excess"),
                interval=evidence.get(f"{excess_prefix}_excess_interval"),
                label=f"{component_id} excess risk",
            ),
            "absolute_limit": _number(evidence.get("absolute_limit"), label="absolute safety limit"),
            "excess_limit": _number(evidence.get("excess_limit"), label="excess safety limit"),
        }
    )


def _candidate_decision_evidence(
    witness: TrialDevPhaseDecisionWitnessV1,
) -> tuple[TrialDevPublicCandidateDecisionV1, ...]:
    evidence = _mapping(witness.evidence, label="decision evidence")
    candidates = _mapping(evidence.get("candidates"), label="candidate evidence")
    records: list[TrialDevPublicCandidateDecisionV1] = []
    for candidate_arm_id, raw_candidate in sorted(candidates.items()):
        candidate = _mapping(raw_candidate, label=f"candidate {candidate_arm_id!r}")
        efficacy_raw = _mapping(candidate.get("efficacy"), label=f"candidate {candidate_arm_id!r} efficacy")
        efficacy = None
        minimum_benefit = None
        if efficacy_raw.get("evaluated") is True:
            efficacy = _interval(
                estimate=efficacy_raw.get("risk_difference_control_minus_treatment"),
                interval=efficacy_raw.get("confidence_interval"),
                label=f"candidate {candidate_arm_id!r} efficacy",
            )
            minimum_benefit = _number(
                efficacy_raw.get("minimum_benefit"),
                label=f"candidate {candidate_arm_id!r} minimum benefit",
            )
        safety = _mapping(candidate.get("safety"), label=f"candidate {candidate_arm_id!r} safety")
        components = [_safety_component(safety, component_id="serious_ae")]
        if safety.get("discontinuation") is not None:
            components.append(
                _safety_component(
                    _mapping(
                        safety.get("discontinuation"),
                        label=f"candidate {candidate_arm_id!r} discontinuation",
                    ),
                    component_id="discontinuation",
                )
            )
        acceptable = candidate.get("acceptable_action_ids")
        state = candidate.get("safety_state")
        if (
            not isinstance(acceptable, list | tuple)
            or not acceptable
            or state not in {"acceptable", "unacceptable", "indeterminate"}
        ):
            raise ValueError("Candidate witness lacks complete action or safety-state evidence.")
        records.append(
            TrialDevPublicCandidateDecisionV1(
                candidate_arm_id=str(candidate_arm_id),
                acceptable_action_ids=tuple(str(action) for action in acceptable),
                safety_state=str(state),
                efficacy=efficacy,
                minimum_efficacy_benefit=minimum_benefit,
                safety_components=tuple(components),
            )
        )
    if not records:
        raise ValueError("Public phase replay contains no candidate evidence.")
    return tuple(records)


def _sensitivity_action_sets(witness: TrialDevPhaseDecisionWitnessV1) -> dict[str, tuple[str, ...]]:
    result: dict[str, set[str]] = {"primary": set(witness.acceptable_action_ids)}
    per_candidate = _mapping(witness.evidence.get("candidates"), label="candidate evidence")
    for candidate_evidence in per_candidate.values():
        candidate = _mapping(candidate_evidence, label="candidate evidence")
        efficacy = candidate.get("efficacy")
        if isinstance(efficacy, dict):
            margins = efficacy.get("margin_sensitivity_action_sets")
            if isinstance(margins, dict):
                for margin, actions in margins.items():
                    if not isinstance(actions, list) or not actions:
                        raise ValueError("Efficacy sensitivity actions must be non-empty arrays.")
                    result.setdefault(f"efficacy_margin::{margin}", set()).update(str(action) for action in actions)
    for profile in ("strict", "primary", "permissive"):
        profile_actions: set[str] = set()
        for candidate_evidence in per_candidate.values():
            candidate = _mapping(candidate_evidence, label="candidate evidence")
            safety = _mapping(candidate.get("safety"), label="candidate safety evidence")
            efficacy_actions = candidate.get("efficacy_action_ids")
            safety_stop_actions = candidate.get("hard_safety_stop_action_ids")
            if not isinstance(efficacy_actions, list | tuple) or not isinstance(
                safety_stop_actions,
                list | tuple,
            ):
                raise ValueError("Candidate evidence lacks explicit safety and efficacy action domains.")
            efficacy_action_ids = tuple(str(action) for action in efficacy_actions)
            safety_stop_action_ids = tuple(str(action) for action in safety_stop_actions)
            states = safety.get("sensitivity_states")
            if not isinstance(states, dict) or states.get(profile) not in {
                "acceptable",
                "unacceptable",
                "indeterminate",
            }:
                raise ValueError("Candidate safety evidence lacks a complete sensitivity state profile.")
            state = str(states[profile])
            selected: tuple[str, ...]
            if witness.phase_id == "phase3":
                failure = tuple(action for action in safety_stop_action_ids if action == "declare_failure")
                inconclusive = tuple(action for action in safety_stop_action_ids if action == "declare_inconclusive")
                if len(failure) != 1 or len(inconclusive) != 1:
                    raise ValueError("Phase-3 policy requires one failure and one inconclusive action.")
                if state == "unacceptable" or efficacy_action_ids == failure:
                    selected = failure
                elif state == "indeterminate":
                    selected = inconclusive
                else:
                    selected = efficacy_action_ids
            elif state == "unacceptable":
                selected = safety_stop_action_ids
            elif state == "indeterminate":
                selected = tuple(set(safety_stop_action_ids) | set(efficacy_action_ids))
            else:
                selected = efficacy_action_ids
            profile_actions.update(str(action) for action in selected)
        result[f"safety_profile::{profile}"] = profile_actions
    return {label: tuple(sorted(actions)) for label, actions in sorted(result.items())}


def replay_trialdev_public_phases_v1(
    *,
    bundle_root: Path,
    materialized_root: Path,
    cases: Sequence[TrialDevPhaseReplayCaseV1],
    trial_seeds: Sequence[int],
) -> tuple[TrialDevPublicPhaseReplayRecordV1, ...]:
    """Materialize declared phase requests and derive public replay evidence."""

    if not cases or not trial_seeds:
        raise ValueError("Public phase replay requires cases and trial seeds.")
    if len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("Public phase-replay trial seeds must be unique.")
    case_keys = tuple((case.world_seed, case.request.checksum()) for case in cases)
    if len(case_keys) != len(set(case_keys)):
        raise ValueError("Public phase-replay cases must be unique by world seed and request.")
    root = Path(bundle_root).resolve()
    output_root = Path(materialized_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[TrialDevPublicPhaseReplayRecordV1] = []
    for case in cases:
        request = case.request
        phase_id = str(request.phase_id)
        if phase_id not in {"phase1", "phase2", "phase3"}:
            raise ValueError("Public phase replay only accepts randomized phase1/phase2/phase3 requests.")
        scenario_root = (root / case.scenario_root).resolve()
        if not scenario_root.is_relative_to(root) or not scenario_root.is_dir():
            raise ValueError("Public phase-replay scenario_root is absent or escapes bundle_root.")
        if request.follow_up_days is None or request.target_sample_size is None or request.allocation_ratio is None:
            raise ValueError("Public randomized phase replay requires a complete design request.")
        if phase_id in {"phase2", "phase3"} and request.treatment_discontinuation_strategy is None:
            raise ValueError("Phase2 and phase3 replay require a treatment-discontinuation strategy.")
        for trial_seed in trial_seeds:
            relative_output = Path(
                f"world_{case.world_seed}",
                f"request_{request.checksum()}",
                f"trial_seed_{trial_seed}",
            )
            trial_output = output_root / relative_output
            materialization = materialize_trial_view_v1(
                scenario_root=scenario_root,
                request=request,
                seed=int(trial_seed),
                out_dir=trial_output,
                overwrite=False,
            )
            if materialization.audit.feasibility_status != "accepted":
                raise ValueError(
                    "Public phase replay cannot qualify a rejected design: "
                    f"{materialization.audit.rejection_reason or 'unspecified rejection'}."
                )
            if materialization.trial_tables_dir is None:
                raise ValueError("Successful public phase replay did not return its trial-table directory.")
            trial_output = Path(materialization.trial_tables_dir)
            witness = derive_phase_decision_witness_v1(
                scenario_root=scenario_root,
                trial_output_root=trial_output,
                phase_id=phase_id,
            )
            design_witness = derive_phase_design_witness_v1(
                scenario_root=scenario_root,
                request=request,
                trial_output_root=trial_output,
                phase_id=phase_id,
            )
            design_efficiency = derive_phase_design_efficiency_v1(
                scenario_root=scenario_root,
                request=request,
                design_witness=design_witness,
            )
            source_checksums = dict(
                _mapping(
                    witness.evidence.get("source_checksums"),
                    label="source checksums",
                )
            )
            for relative_path in (
                "public/phase_design_frontiers.json",
                "public/phase_design_policy.json",
            ):
                source_checksums[relative_path] = sha256_file_hex(scenario_root / relative_path)
            records.append(
                TrialDevPublicPhaseReplayRecordV1(
                    scenario_id=str(request.scenario_id),
                    world_seed=case.world_seed,
                    trial_seed=int(trial_seed),
                    request_checksum=request.checksum(),
                    trial_output_path=relative_output.as_posix(),
                    phase_id=phase_id,
                    endpoint_id=request.endpoint_id,
                    treatment_discontinuation_strategy=request.treatment_discontinuation_strategy,
                    follow_up_days=int(request.follow_up_days),
                    target_sample_size=int(request.target_sample_size),
                    allocation_ratio=str(request.allocation_ratio),
                    objective_ids=case.program_objective_ids,
                    candidate_drug_ids=tuple(str(value) for value in request.candidate_drug_ids),
                    acceptable_action_ids=tuple(witness.acceptable_action_ids),
                    stop_action_ids=tuple(witness.stop_action_ids),
                    advance_action_ids=tuple(witness.advance_action_ids),
                    sensitivity_action_sets=_sensitivity_action_sets(witness),
                    public_decision_witness_checksum=witness.checksum,
                    public_source_checksums={str(path): str(checksum) for path, checksum in source_checksums.items()},
                    candidate_decision_evidence=_candidate_decision_evidence(witness),
                    public_safety_state=str(witness.safety_state),
                    design_adequate=bool(design_witness.adequate),
                    design_failures=tuple(design_witness.failures),
                    design_frontier=design_efficiency.frontier,
                    design_on_frontier=design_efficiency.on_frontier,
                    design_dominated_by_frontier=design_efficiency.dominated_by_frontier,
                    minimum_frontier_participants=design_efficiency.minimum_frontier_participants,
                    minimum_frontier_follow_up_days=design_efficiency.minimum_frontier_follow_up_days,
                    participant_excess_vs_minimum=design_efficiency.participant_excess_vs_minimum,
                    participant_shortage_vs_minimum=design_efficiency.participant_shortage_vs_minimum,
                    follow_up_excess_days_vs_minimum=design_efficiency.follow_up_excess_days_vs_minimum,
                    follow_up_shortage_days_vs_minimum=design_efficiency.follow_up_shortage_days_vs_minimum,
                    achieved_power=design_witness.achieved_power,
                    target_power=design_witness.target_power,
                    achieved_safety_absolute_risk_power=(design_witness.achieved_safety_absolute_risk_power),
                    achieved_safety_excess_risk_power=(design_witness.achieved_safety_excess_risk_power),
                    target_safety_decision_power=design_witness.target_safety_decision_power,
                )
            )
    return tuple(records)


__all__ = ["replay_trialdev_public_phases_v1"]
