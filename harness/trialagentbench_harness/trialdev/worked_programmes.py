"""Generate reproducible, non-score-bearing TrialDev worked programmes."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import cast

from trialagentbench_harness.contracts.trialdev.metrics import (
    TRIALDEV_CAPABILITY_CHECKS_V1,
    TRIALDEV_CHECKPOINT_INVENTORY_V1,
    TRIALDEV_REQUIRED_LANES_V1,
    TRIALDEV_TERMINAL_LANES_V1,
    TrialDevCapabilityAssessmentV1,
    TrialDevCapabilityCheckIdV1,
    TrialDevCapabilityCheckV1,
    TrialDevCheckpointAssessmentV1,
    TrialDevLaneAssessmentV1,
    TrialDevProgrammeAssessmentV1,
)
from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevActionSelectionV1,
    TrialDevCheckpointOutcomeV1,
    TrialDevDecisionRuleEvidenceV1,
    TrialDevEvidenceReferenceV1,
    TrialDevObservationalCandidateEvidenceV1,
    TrialDevObservationalDecisionEvidenceV1,
    TrialDevPairContrastEvidenceV1,
    TrialDevPolicyBindingV1,
    TrialDevPortfolioProgrammeStateV1,
    TrialDevProgrammeStateV1,
    TrialDevRandomizedDecisionEvidenceV1,
    TrialDevResourceScheduleV1,
    TrialDevRuleClassificationV1,
    TrialDevRuleDomainV1,
    TrialDevSingleAssetProgrammeStateV1,
    TrialDevStreamIdV1,
)
from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TrialDevScientificAssessmentV1,
    TrialDevScientificEnvelopeV1,
)
from trialagentbench_harness.contracts.trialdev.worked_programmes import (
    TrialDevWorkedCheckpointV1,
    TrialDevWorkedPackageV1,
    TrialDevWorkedProgrammeV1,
)
from trialagentbench_harness.io.checksums import sha256_file
from trialagentbench_harness.io.json import write_json_model
from trialagentbench_harness.trialdev.policy import derive_supported_action_set_v1
from trialagentbench_harness.trialdev.programme import (
    build_checkpoint_action_policy_v1,
    transition_programme_state_v1,
)
from trialagentbench_harness.trialdev.state_action_graph import write_trialdev_state_action_graph_v1

_QUALIFICATION_SEED = 20_260_802
_METHOD_ID = "prespecified_interval_method_v1"


def _write_evidence(
    output: Path,
    *,
    source_identity: str,
    programme_id: str,
    checkpoint_id: str,
    asset_id: str,
    measure_id: str,
    estimate: float,
    lower: float,
    upper: float,
) -> TrialDevEvidenceReferenceV1:
    relative = Path("evidence") / programme_id / f"{checkpoint_id}-{asset_id}-{measure_id}.csv"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("asset_id", "measure_id", "estimate", "lower", "upper"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "asset_id": asset_id,
                "measure_id": measure_id,
                "estimate": f"{estimate:.6f}",
                "lower": f"{lower:.6f}",
                "upper": f"{upper:.6f}",
            }
        )
    return TrialDevEvidenceReferenceV1(
        evidence_id=f"{programme_id}-{checkpoint_id}-{asset_id}-{measure_id}",
        evidence_kind="dataset",
        checkpoint_id=checkpoint_id,
        asset_id=asset_id,
        evidence_protocol_id="worked_programme_interval_protocol_v1",
        evidence_protocol_checksum=source_identity,
        source_family_id=source_identity,
        world_id=f"worked-{programme_id}",
        generation_seed=_QUALIFICATION_SEED,
        relative_path=relative.as_posix(),
        artifact_sha256=sha256_file(path),
    )


def _binding(*, stream_id: TrialDevStreamIdV1, source_identity: str) -> TrialDevPolicyBindingV1:
    portfolio = stream_id == "bounded_portfolio_reallocation"
    return TrialDevPolicyBindingV1(
        stream_id=stream_id,
        objective_id="benefit_risk",
        objective_policy_checksum=source_identity,
        action_policy_checksum=source_identity,
        design_menu_checksum=source_identity,
        resource_schedule=TrialDevResourceScheduleV1() if portfolio else None,
        resource_budget_units=10 if portfolio else None,
    )


def _observational_evidence(
    state: TrialDevProgrammeStateV1,
    *,
    utilities: dict[str, float],
    lead_eligible: set[str],
    reserve_eligible: set[str],
    withholding_supported: bool,
) -> TrialDevObservationalDecisionEvidenceV1:
    evidence_by_asset = {
        item.asset_id: item
        for item in state.evidence
        if item.asset_id is not None and item.evidence_id.endswith("-utility")
    }
    return TrialDevObservationalDecisionEvidenceV1(
        state_checksum=cast(str, state.checksum),
        analysis_method_id=_METHOD_ID,
        identification_status="identified",
        minimum_efficacy_gain=0.50,
        practical_equivalence_margin=0.05,
        candidates=tuple(
            TrialDevObservationalCandidateEvidenceV1(
                asset_id=asset,
                utility_estimate=utilities[asset],
                utility_lower_bound=utilities[asset] - 0.10,
                utility_upper_bound=utilities[asset] + 0.10,
                efficacy_estimate=(0.55 if asset in reserve_eligible else 0.30),
                efficacy_lower_bound=(
                    0.55
                    if asset in lead_eligible and not withholding_supported
                    else (0.45 if asset in reserve_eligible else 0.20)
                ),
                efficacy_upper_bound=(0.65 if asset in reserve_eligible else 0.40),
                evidence_reference_checksums=(cast(str, evidence_by_asset[asset].checksum),),
            )
            for asset in state.candidate_asset_ids
        ),
        pair_contrasts=tuple(
            TrialDevPairContrastEvidenceV1(
                lead_asset_id=first,
                reserve_asset_id=second,
                confidence_half_width=0.10,
            )
            for index, first in enumerate(sorted(state.candidate_asset_ids))
            for second in sorted(state.candidate_asset_ids)[index + 1 :]
        ),
    )


def _rule(
    *,
    asset_id: str,
    domain: TrialDevRuleDomainV1,
    classification: TrialDevRuleClassificationV1,
    evidence: TrialDevEvidenceReferenceV1,
) -> TrialDevDecisionRuleEvidenceV1:
    direction, values = _rule_values(domain=domain, classification=classification)
    return TrialDevDecisionRuleEvidenceV1(
        rule_id=f"{asset_id}-{domain}",
        asset_id=asset_id,
        domain=domain,
        direction=direction,
        estimate=values[0],
        lower_bound=values[1],
        upper_bound=values[2],
        threshold=0.50,
        evidence_reference_checksums=(cast(str, evidence.checksum),),
    )


def _rule_values(
    *,
    domain: TrialDevRuleDomainV1,
    classification: TrialDevRuleClassificationV1,
) -> tuple[str, tuple[float, float, float]]:
    direction = "maximum" if domain == "safety" else "minimum"
    values = {
        ("minimum", "clear_pass"): (0.80, 0.70, 0.90),
        ("minimum", "clear_fail"): (0.20, 0.10, 0.30),
        ("minimum", "indeterminate"): (0.50, 0.30, 0.70),
        ("maximum", "clear_pass"): (0.20, 0.10, 0.30),
        ("maximum", "clear_fail"): (0.70, 0.60, 0.80),
        ("maximum", "indeterminate"): (0.50, 0.30, 0.70),
    }[(direction, classification)]
    return direction, values


def _write_rule_evidence(
    output: Path,
    *,
    source_identity: str,
    programme_id: str,
    checkpoint_id: str,
    classifications: dict[str, dict[TrialDevRuleDomainV1, TrialDevRuleClassificationV1]],
) -> tuple[TrialDevEvidenceReferenceV1, ...]:
    records = []
    for asset_id, domains in classifications.items():
        for domain, classification in domains.items():
            _, values = _rule_values(domain=domain, classification=classification)
            records.append(
                _write_evidence(
                    output,
                    source_identity=source_identity,
                    programme_id=programme_id,
                    checkpoint_id=checkpoint_id,
                    asset_id=asset_id,
                    measure_id=domain,
                    estimate=values[0],
                    lower=values[1],
                    upper=values[2],
                )
            )
    return tuple(records)


def _randomized_evidence(
    state: TrialDevProgrammeStateV1,
    *,
    classifications: dict[str, dict[TrialDevRuleDomainV1, TrialDevRuleClassificationV1]],
) -> TrialDevRandomizedDecisionEvidenceV1:
    evidence_by_key = {
        (item.asset_id, domain): item
        for item in state.evidence
        if item.asset_id is not None
        for domain in ("efficacy", "safety")
        if item.evidence_id.endswith(f"-{domain}")
    }
    return TrialDevRandomizedDecisionEvidenceV1(
        state_checksum=cast(str, state.checksum),
        analysis_method_id=_METHOD_ID,
        rules=tuple(
            _rule(
                asset_id=asset,
                domain=domain,
                classification=classification,
                evidence=evidence_by_key[(asset, domain)],
            )
            for asset, domains in classifications.items()
            for domain, classification in domains.items()
        ),
    )


def _step(
    state: TrialDevProgrammeStateV1,
    *,
    decision_evidence: TrialDevObservationalDecisionEvidenceV1 | TrialDevRandomizedDecisionEvidenceV1,
    action_id: str,
    target_asset_id: str | None = None,
    reserve_asset_id: str | None = None,
    next_evidence: tuple[TrialDevEvidenceReferenceV1, ...] = (),
) -> TrialDevWorkedCheckpointV1:
    supported = derive_supported_action_set_v1(state=state, evidence=decision_evidence)
    selection = TrialDevActionSelectionV1.model_validate(
        {
            "state_checksum": state.checksum,
            "checkpoint_id": state.current_checkpoint_id,
            "action_id": action_id,
            "target_asset_id": target_asset_id,
            "reserve_asset_id": reserve_asset_id,
            "analysis_method_id": decision_evidence.analysis_method_id,
            "supporting_evidence_ids": tuple(item.evidence_id for item in state.evidence),
            "justification": "The selected action belongs to the set supported by the current interval evidence.",
        }
    )
    after = transition_programme_state_v1(
        state=state,
        action_policy=build_checkpoint_action_policy_v1(state=state),
        selection=selection,
        outcome=TrialDevCheckpointOutcomeV1(
            reach_status="reached",
            submission_status="accepted",
            analysis_status="estimable",
            execution_status="completed",
        ),
        next_evidence=next_evidence,
    )
    return TrialDevWorkedCheckpointV1(
        state_before=state,
        decision_evidence=decision_evidence,
        supported_action_set=supported,
        selected_action=selection,
        state_after=after,
    )


def _assessment(
    *,
    source_identity: str,
    programme_id: str,
    stream_id: TrialDevStreamIdV1,
    steps: tuple[TrialDevWorkedCheckpointV1, ...],
) -> TrialDevProgrammeAssessmentV1:
    by_checkpoint = {step.state_before.current_checkpoint_id: step for step in steps}
    checkpoints = []
    for checkpoint_id in TRIALDEV_CHECKPOINT_INVENTORY_V1[stream_id]:
        step = by_checkpoint.get(checkpoint_id)
        if step is None:
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
            continue
        terminal = step.state_after.terminal_disposition != "active"
        evidence_sha = cast(str, step.decision_evidence.checksum)
        supported_set_sha = cast(str, step.supported_action_set.checksum)
        transition_sha = cast(str, step.state_after.checksum)
        source_by_check: dict[TrialDevCapabilityCheckIdV1, str] = {
            "evidence_integrity": evidence_sha,
            "method_eligibility": evidence_sha,
            "identification_status": evidence_sha,
            "uncertainty_qualification": evidence_sha,
            "policy_conclusion_compatibility": supported_set_sha,
            "safety_evidence": evidence_sha,
            "transition_legality": transition_sha,
            "history_immutability": transition_sha,
            "required_output_presence": transition_sha,
            "workflow_completion": transition_sha,
            "selected_action_membership": supported_set_sha,
        }
        required_lanes = set(TRIALDEV_REQUIRED_LANES_V1[(stream_id, checkpoint_id)])
        if terminal:
            required_lanes.update(TRIALDEV_TERMINAL_LANES_V1)
        capabilities = []
        for capability_id, check_ids in TRIALDEV_CAPABILITY_CHECKS_V1.items():
            capabilities.append(
                TrialDevCapabilityAssessmentV1(
                    capability_id=capability_id,
                    outcome="passed",
                    checks=tuple(
                        TrialDevCapabilityCheckV1(
                            check_id=cast(TrialDevCapabilityCheckIdV1, check_id),
                            passed=True,
                            source_record_sha256=source_by_check[check_id],
                        )
                        for check_id in check_ids
                    ),
                )
            )
        if isinstance(step.decision_evidence, TrialDevObservationalDecisionEvidenceV1):
            scientific_envelope = TrialDevScientificEnvelopeV1(
                envelope_id="worked_programme_declared_utility_margin_v1",
                basis="declared_practical_equivalence_margin",
                absolute_margin=step.decision_evidence.practical_equivalence_margin,
                exact_reproduction_tolerance=0.0005,
            )
        else:
            scientific_envelope = TrialDevScientificEnvelopeV1(
                envelope_id="worked_programme_declared_decision_thresholds_v1",
                basis="declared_decision_thresholds",
                decision_thresholds=tuple(sorted({float(rule.threshold) for rule in step.decision_evidence.rules})),
                exact_reproduction_tolerance=0.0005,
            )
        scientific_assessment = TrialDevScientificAssessmentV1(
            execution="passed",
            question_estimand="passed",
            design="not_applicable" if checkpoint_id == "observational_review" else "passed",
            assumptions="passed",
            analysis_classification="uncertainty_qualified",
            scientific_agreement="passed",
            exact_reproduction="passed",
            uncertainty="passed",
            action_admissibility="passed",
            evidential_support="passed",
            sequential_coherence="passed",
            resources="within_budget" if stream_id == "bounded_portfolio_reallocation" else "not_assessed",
            scientific_envelope=scientific_envelope,
            decision_complete=True,
        )
        checkpoints.append(
            TrialDevCheckpointAssessmentV1(
                checkpoint_id=checkpoint_id,
                outcome=TrialDevCheckpointOutcomeV1(
                    reach_status="reached",
                    submission_status="accepted",
                    analysis_status="estimable",
                    execution_status="completed",
                ),
                lanes=tuple(
                    TrialDevLaneAssessmentV1(
                        lane_id=lane_id,
                        outcome="accepted",
                        source_record_sha256=(
                            transition_sha
                            if lane_id in {"route_timing", "final_recommendation"}
                            else supported_set_sha
                        ),
                    )
                    for lane_id in sorted(required_lanes)
                ),
                capabilities=tuple(capabilities),
                scientific_assessment=scientific_assessment,
                terminal_record_valid=True if terminal else None,
            )
        )
    return TrialDevProgrammeAssessmentV1(
        model_id="deterministic_public_witness",
        condition_id="deterministic-public-witness",
        request_replicate_id="request-1",
        reasoning_effort=None,
        procedure_assistance="output_contract_only",
        maximum_turns_per_step=90,
        maximum_submission_attempts=3,
        task_materialization_seed=45560,
        release_id="worked-programmes-v1",
        run_id=f"worked-{programme_id}",
        grader_sha256=source_identity,
        evaluation_unit_id=programme_id,
        programme_id=programme_id,
        scenario_family_id=f"worked-{programme_id}",
        objective_variant_id="benefit_risk",
        policy_variant_id="primary",
        stream_id=stream_id,
        execution_status="completed",
        checkpoints=tuple(checkpoints),
    )


def _single_programme(output: Path, *, source_identity: str) -> TrialDevWorkedProgrammeV1:
    programme_id = "single_asset_irreversible_chain"
    assets = ("A", "B", "C")
    initial_evidence = tuple(
        _write_evidence(
            output,
            source_identity=source_identity,
            programme_id=programme_id,
            checkpoint_id="observational_review",
            asset_id=asset,
            measure_id="utility",
            estimate=value,
            lower=value - 0.08,
            upper=value + 0.08,
        )
        for asset, value in zip(assets, (0.80, 0.72, 0.30), strict=True)
    )
    state: TrialDevProgrammeStateV1 = TrialDevSingleAssetProgrammeStateV1(
        programme_id=programme_id,
        scenario_id="worked-single",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=assets,
        policy_binding=_binding(stream_id="single_asset_development", source_identity=source_identity),
        evidence=initial_evidence,
    )
    steps = []
    next_evidence = _write_rule_evidence(
        output,
        source_identity=source_identity,
        programme_id=programme_id,
        checkpoint_id="early_safety_study",
        classifications={"A": {"safety": "clear_pass"}},
    )
    step = _step(
        state,
        decision_evidence=_observational_evidence(
            state,
            utilities={"A": 0.80, "B": 0.72, "C": 0.30},
            lead_eligible={"A", "B"},
            reserve_eligible={"A", "B"},
            withholding_supported=True,
        ),
        action_id="nominate_for_early_study",
        target_asset_id="A",
        next_evidence=next_evidence,
    )
    steps.append(step)
    state = step.state_after
    for checkpoint_id, action_id, domains, following in (
        (
            "early_safety_study",
            "advance_to_proof_of_concept",
            {"safety": "clear_pass"},
            "proof_of_concept",
        ),
        (
            "proof_of_concept",
            "advance_to_confirmation",
            {"efficacy": "clear_pass", "safety": "clear_pass"},
            "confirmation",
        ),
        ("confirmation", "declare_success", {"efficacy": "clear_pass", "safety": "clear_pass"}, None),
    ):
        if state.current_checkpoint_id != checkpoint_id:
            raise ValueError("Worked single-asset checkpoint drift.")
        following_domains = (
            None
            if following is None
            else (
                {"efficacy": "clear_pass", "safety": "clear_pass"}
                if following == "confirmation"
                else {"efficacy": "clear_pass", "safety": "clear_pass"}
            )
        )
        following_evidence = (
            ()
            if following_domains is None
            else _write_rule_evidence(
                output,
                source_identity=source_identity,
                programme_id=programme_id,
                checkpoint_id=cast(str, following),
                classifications={
                    "A": cast(
                        dict[TrialDevRuleDomainV1, TrialDevRuleClassificationV1],
                        following_domains,
                    )
                },
            )
        )
        step = _step(
            state,
            decision_evidence=_randomized_evidence(
                state,
                classifications={"A": cast(dict[TrialDevRuleDomainV1, TrialDevRuleClassificationV1], domains)},
            ),
            action_id=action_id,
            next_evidence=following_evidence,
        )
        steps.append(step)
        state = step.state_after
    typed_steps = tuple(steps)
    return TrialDevWorkedProgrammeV1(
        programme_id=programme_id,
        stream_id="single_asset_development",
        qualification_seed=_QUALIFICATION_SEED,
        checkpoints=typed_steps,
        assessment=_assessment(
            source_identity=source_identity,
            programme_id=programme_id,
            stream_id="single_asset_development",
            steps=typed_steps,
        ),
    )


def _portfolio_programme(output: Path, *, source_identity: str) -> TrialDevWorkedProgrammeV1:
    programme_id = "portfolio_reserve_promotion"
    assets = ("A", "B", "C")
    initial_evidence = tuple(
        _write_evidence(
            output,
            source_identity=source_identity,
            programme_id=programme_id,
            checkpoint_id="observational_review",
            asset_id=asset,
            measure_id="utility",
            estimate=value,
            lower=value - 0.08,
            upper=value + 0.08,
        )
        for asset, value in zip(assets, (0.80, 0.76, 0.30), strict=True)
    )
    state: TrialDevProgrammeStateV1 = TrialDevPortfolioProgrammeStateV1(
        programme_id=programme_id,
        scenario_id="worked-portfolio",
        current_checkpoint_id="observational_review",
        candidate_asset_ids=assets,
        policy_binding=_binding(stream_id="bounded_portfolio_reallocation", source_identity=source_identity),
        evidence=initial_evidence,
    )
    steps = []
    joint_evidence = _write_rule_evidence(
        output,
        source_identity=source_identity,
        programme_id=programme_id,
        checkpoint_id="joint_early_study_review",
        classifications={
            "A": {"safety": "clear_fail"},
            "B": {"safety": "clear_pass"},
        },
    )
    step = _step(
        state,
        decision_evidence=_observational_evidence(
            state,
            utilities={"A": 0.80, "B": 0.76, "C": 0.30},
            lead_eligible={"A", "B"},
            reserve_eligible={"A", "B"},
            withholding_supported=False,
        ),
        action_id="select_lead_and_reserve",
        target_asset_id="A",
        reserve_asset_id="B",
        next_evidence=joint_evidence,
    )
    steps.append(step)
    state = step.state_after
    promoted_evidence = _write_rule_evidence(
        output,
        source_identity=source_identity,
        programme_id=programme_id,
        checkpoint_id="promoted_reserve_proof_of_concept_review",
        classifications={"B": {"efficacy": "clear_pass", "safety": "clear_pass"}},
    )
    step = _step(
        state,
        decision_evidence=_randomized_evidence(
            state,
            classifications={
                "A": {"safety": "clear_fail"},
                "B": {"safety": "clear_pass"},
            },
        ),
        action_id="promote_reserve_to_proof_of_concept",
        next_evidence=promoted_evidence,
    )
    steps.append(step)
    state = step.state_after
    confirmation_evidence = _write_rule_evidence(
        output,
        source_identity=source_identity,
        programme_id=programme_id,
        checkpoint_id="confirmation",
        classifications={"B": {"efficacy": "clear_pass", "safety": "clear_pass"}},
    )
    step = _step(
        state,
        decision_evidence=_randomized_evidence(
            state,
            classifications={"B": {"efficacy": "clear_pass", "safety": "clear_pass"}},
        ),
        action_id="advance_active_to_confirmation",
        next_evidence=confirmation_evidence,
    )
    steps.append(step)
    state = step.state_after
    step = _step(
        state,
        decision_evidence=_randomized_evidence(
            state,
            classifications={"B": {"efficacy": "clear_pass", "safety": "clear_pass"}},
        ),
        action_id="declare_success",
    )
    steps.append(step)
    typed_steps = tuple(steps)
    return TrialDevWorkedProgrammeV1(
        programme_id=programme_id,
        stream_id="bounded_portfolio_reallocation",
        qualification_seed=_QUALIFICATION_SEED,
        checkpoints=typed_steps,
        assessment=_assessment(
            source_identity=source_identity,
            programme_id=programme_id,
            stream_id="bounded_portfolio_reallocation",
            steps=typed_steps,
        ),
    )


def build_trialdev_worked_programmes_v1(*, output_dir: Path, source_identity: str) -> TrialDevWorkedPackageV1:
    """Generate and persist one complete worked programme for each stream."""

    if len(source_identity) != 64 or any(character not in "0123456789abcdef" for character in source_identity):
        raise ValueError("Worked-programme source identity must be a SHA-256 hex digest.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    package = TrialDevWorkedPackageV1(
        programmes=(
            _single_programme(output, source_identity=source_identity),
            _portfolio_programme(output, source_identity=source_identity),
        )
    )
    write_json_model(output / "worked_programmes.json", package)
    write_trialdev_state_action_graph_v1(
        output_path=output / "state_action_graph.json",
        source_identity=source_identity,
    )
    return package


__all__ = ["build_trialdev_worked_programmes_v1"]
