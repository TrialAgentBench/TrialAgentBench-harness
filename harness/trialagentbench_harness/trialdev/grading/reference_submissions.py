"""Canonical TrialDev submissions composed from released evidence contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from trialagentbench_harness.contracts.trialdev.trialdev_public_recoverability import (
    TrialDevPublicRecoverabilityReportV1,
)
from trialagentbench_harness.io import read_json
from trialagentbench_harness.trialdev.grading.analysis_evidence import (
    derive_effect_references_v1,
)
from trialagentbench_harness.trialdev.grading.decision_evidence import (
    TrialDevPhaseDecisionWitnessV1,
    derive_phase_decision_witness_v1,
)
from trialagentbench_harness.trialdev.grading.design_frontier import (
    TrialDevDesignFrontierStratumV1,
    load_phase_design_frontiers_v1,
    select_operational_support_v1,
)
from trialagentbench_harness.trialdev.grading.grade import (
    load_public_observational_reference_v1,
)
from trialagentbench_harness.trialdev.grading.hashing import sha256_file_hex
from trialagentbench_harness.trialdev.share.models import (
    PhaseModuleSpecV1,
    TrialDevelopmentRequestV1,
)
from trialagentbench_harness.trialdev.share.public_method_design import (
    TrialDevPhaseAnalysisMethodCatalogV1,
    TrialDevPhaseDesignPolicyV1,
    TrialDevPublicObservationalMethodCatalogV1,
)
from trialagentbench_harness.trialdev.share.sequential import (
    PhaseActionIdV1,
    TrialDevelopmentCandidateUtilityEstimateV1,
    TrialDevelopmentEffectEstimateV1,
    TrialDevelopmentIdentificationEvidenceV1,
    TrialDevelopmentObservationalReviewSubmissionV1,
    TrialDevelopmentPhaseAnalysisSubmissionV1,
    TrialDevelopmentPhaseDecisionSubmissionV1,
    TrialDevelopmentSafetyEstimateV1,
    validate_trial_output_bundle_v1,
)

__all__ = [
    "build_observational_reference_submission_v1",
    "build_phase_reference_analysis_v1",
    "build_phase_reference_decision_v1",
    "build_phase_reference_request_v1",
]


def _phase_module(*, scenario_root: Path, phase_id: str) -> PhaseModuleSpecV1:
    payload = read_json(Path(scenario_root) / "public" / "phase_module_catalog.json")
    modules = payload.get("phase_modules") if isinstance(payload, dict) else None
    if not isinstance(modules, list):
        raise ValueError("Reference submission requires the public phase-module catalog.")
    matches = tuple(
        PhaseModuleSpecV1.model_validate(module)
        for module in modules
        if isinstance(module, dict) and str(module.get("phase_id")) == phase_id
    )
    if len(matches) != 1:
        raise ValueError(f"Reference submission requires one module for phase={phase_id!r}.")
    return matches[0]


def build_observational_reference_submission_v1(
    *,
    scenario_root: Path,
    objective_id: str,
) -> TrialDevelopmentObservationalReviewSubmissionV1:
    """Build one exact method-conditioned observational reference submission."""

    catalog = TrialDevPublicObservationalMethodCatalogV1.model_validate(
        read_json(Path(scenario_root) / "public" / "observational_method_catalog.json")
    )
    unrecorded = tuple(
        factor
        for factor in catalog.assignment_prognostic_factors
        if factor.used_in_treatment_assignment
        and factor.prognostic_for_primary_endpoint
        and not factor.recorded_in_observational_extract
    )
    if unrecorded:
        catalog_path = Path(scenario_root) / "public" / "observational_method_catalog.json"
        evidence_id = "observational_exchangeability_provenance"
        return TrialDevelopmentObservationalReviewSubmissionV1(
            response_branch="qualified_non_nomination",
            primary_resolution_evidence_class="design_or_provenance_reasoning",
            identification_evidence=(
                TrialDevelopmentIdentificationEvidenceV1(
                    evidence_id=evidence_id,
                    premise_id="measured_conditional_exchangeability",
                    premise_state="failed",
                    evidence_kind="factual_provenance",
                    public_artifact_path="public/observational_method_catalog.json",
                    public_artifact_sha256=sha256_file_hex(catalog_path),
                    source_record_id=unrecorded[0].factor_id,
                    interpretation=(
                        "A pretreatment prognostic factor used in treatment assignment is absent from the released "
                        "adjustment set, so the observational evidence does not support causal candidate ranking."
                    ),
                ),
            ),
            supporting_evidence_ids=(evidence_id,),
            decision_action="withhold_nomination",
            decision_rationale="The released measurement provenance does not support a causal candidate ranking.",
        )
    method = sorted(catalog.methods, key=lambda row: row.method_route_id)[0]
    recoverability = TrialDevPublicRecoverabilityReportV1.model_validate(
        read_json(Path(scenario_root) / "grader" / "public_recoverability_report.json")
    )
    method_results = tuple(
        result for result in recoverability.method_results if result.method_route_id == method.method_route_id
    )
    if len(method_results) != 1:
        raise ValueError("Reference submission requires one public recoverability result for its method route.")
    method_result = method_results[0]
    objective_policies = tuple(
        policy for policy in method_result.objective_policies if str(policy.objective_id) == objective_id
    )
    if len(objective_policies) != 1:
        raise ValueError("Reference submission requires one recoverability policy for its objective.")
    if objective_policies[0].policy == "insufficient_recoverability":
        comparisons = tuple(
            comparison
            for comparison in method_result.estimator_comparisons
            if str(comparison.objective_id) == objective_id and comparison.estimator_id == method.primary_estimator_id
        )
        if len(comparisons) != 1 or comparisons[0].status != "not_estimable":
            raise ValueError("Insufficient recoverability requires one typed primary-estimator failure.")
        reason = comparisons[0].failure_reason
        premise_id = "practical_positivity" if reason == "empirical_positivity_violation" else "method_estimability"
        extract_path = Path(scenario_root) / "public" / "observational_extract.parquet"
        evidence_id = f"{method.method_route_id}_support_failure"
        return TrialDevelopmentObservationalReviewSubmissionV1(
            response_branch="qualified_non_nomination",
            primary_resolution_evidence_class="evidence_insufficient",
            identification_evidence=(
                TrialDevelopmentIdentificationEvidenceV1(
                    evidence_id=evidence_id,
                    premise_id=premise_id,
                    premise_state="failed",
                    evidence_kind="empirical_diagnostic",
                    public_artifact_path="public/observational_extract.parquet",
                    public_artifact_sha256=sha256_file_hex(extract_path),
                    source_record_id=method.method_route_id,
                    interpretation=(
                        "The declared public method cannot produce a reproducible causal candidate ranking "
                        f"from the released observational support ({reason})."
                    ),
                ),
            ),
            supporting_evidence_ids=(evidence_id,),
            decision_action="withhold_nomination",
            decision_rationale=(
                "The declared public method is not estimable from the released observational support."
            ),
        )
    reference = load_public_observational_reference_v1(
        scenario_root=Path(scenario_root),
        objective_id=objective_id,
        method_route_id=method.method_route_id,
    )
    ranked = tuple(
        sorted(
            reference.candidate_utilities,
            key=lambda candidate_id: (
                -reference.candidate_utilities[candidate_id].estimate,
                candidate_id,
            ),
        )
    )
    estimates = tuple(
        TrialDevelopmentCandidateUtilityEstimateV1(
            evidence_id=f"utility_{candidate_id}",
            method_route_id=reference.method_route_id,
            candidate_drug_id=candidate_id,
            objective_id=objective_id,
            estimator_id=reference.estimator_id,
            estimate=reference.candidate_utilities[candidate_id].estimate,
            lower=reference.candidate_utilities[candidate_id].lower,
            upper=reference.candidate_utilities[candidate_id].upper,
            confidence_level=reference.confidence_level,
            analysis_covariate_ids=tuple(sorted(reference.adjustment_covariates)),
            source_artifact_checksums=reference.source_artifact_checksums,
        )
        for candidate_id in ranked
    )
    nominated = next(
        (target_id for target_id in reference.reference_target_ids if target_id != "withhold_nomination"),
        None,
    )
    if nominated is None:
        if set(reference.reference_target_ids) != {"withhold_nomination"}:
            raise ValueError("Observational reference action set is internally inconsistent.")
        decision_action = "withhold_nomination"
        supporting_id = f"utility_{ranked[0]}"
    else:
        if nominated not in set(ranked):
            raise ValueError("Observational reference nominates a candidate without a utility estimate.")
        decision_action = "nominate_for_early_study"
        supporting_id = f"utility_{nominated}"
    return TrialDevelopmentObservationalReviewSubmissionV1(
        response_branch="estimable",
        primary_resolution_evidence_class="empirical_diagnosis",
        ranked_drug_ids=ranked,
        candidate_utility_estimates=estimates,
        supporting_evidence_ids=(supporting_id,),
        candidate_drug_id=nominated,
        decision_action=decision_action,
        decision_rationale=(
            "Decision follows the declared objective, accepted observational method, "
            "and method-conditioned public action set."
        ),
    )


def _reference_stratum(
    *,
    scenario_root: Path,
    phase_id: str,
    candidate_drug_id: str,
) -> TrialDevDesignFrontierStratumV1:
    policy = TrialDevPhaseDesignPolicyV1.model_validate(
        read_json(Path(scenario_root) / "public" / "phase_design_policy.json")
    )
    rule = policy.rule_for_phase(phase_id)
    artifact = load_phase_design_frontiers_v1(scenario_root=Path(scenario_root))
    matches = tuple(
        stratum
        for stratum in artifact.strata
        if stratum.phase_id == phase_id
        and stratum.candidate_drug_ids == (candidate_drug_id,)
        and stratum.endpoint_id == rule.primary_endpoint_id
        and stratum.design_cell_id == rule.design_cell_id
        and stratum.interim_policy == rule.supported_interim_policy
    )
    if not matches:
        raise ValueError(
            f"Public design frontier has no reference stratum for phase={phase_id!r}, candidate={candidate_drug_id!r}."
        )
    return sorted(matches, key=lambda row: row.key())[0]


def build_phase_reference_request_v1(
    *,
    scenario_root: Path,
    phase_id: str,
    candidate_drug_id: str,
    objective_id: str,
) -> TrialDevelopmentRequestV1:
    """Select a deterministic nondominated design from the public frontier."""

    module = _phase_module(scenario_root=Path(scenario_root), phase_id=phase_id)
    stratum = _reference_stratum(
        scenario_root=Path(scenario_root),
        phase_id=phase_id,
        candidate_drug_id=candidate_drug_id,
    )
    point = min(
        stratum.frontier,
        key=lambda row: (
            row.target_sample_size,
            row.follow_up_days,
            row.allocation_ratio,
        ),
    )
    frontier_artifact = load_phase_design_frontiers_v1(scenario_root=Path(scenario_root))
    operation = select_operational_support_v1(
        artifact=frontier_artifact,
        phase_id=phase_id,
        target_sample_size=point.target_sample_size,
    )
    selection_objective = "benefit_risk" if phase_id == "phase1" else objective_id
    if selection_objective not in set(module.allowed_selection_objectives):
        raise ValueError(f"Reference objective {selection_objective!r} is unavailable for phase={phase_id!r}.")
    return TrialDevelopmentRequestV1(
        scenario_id=str(frontier_artifact.scenario_id),
        phase_id=cast(str, phase_id),
        candidate_drug_ids=(candidate_drug_id,),
        target_sample_size=point.target_sample_size,
        endpoint_id=stratum.endpoint_id,
        follow_up_days=point.follow_up_days,
        enrollment_window_days=operation.enrollment_window_days,
        site_count_budget=operation.site_count_budget,
        allocation_ratio=point.allocation_ratio,
        design_cell_id=stratum.design_cell_id,
        treatment_discontinuation_strategy=cast(str, stratum.treatment_discontinuation_strategy),
        interim_policy=cast(str, stratum.interim_policy),
        site_strategy=operation.site_strategy,
        selection_objective=cast(str, selection_objective),
    )


def _interval(payload: object, key: str) -> tuple[float, float]:
    if not isinstance(payload, dict):
        raise ValueError("Reference safety evidence must be an object.")
    value = payload.get(key)
    if (
        not isinstance(value, list | tuple)
        or len(value) != 2
        or isinstance(value[0], bool)
        or not isinstance(value[0], int | float)
        or isinstance(value[1], bool)
        or not isinstance(value[1], int | float)
    ):
        raise ValueError(f"Reference safety evidence requires a numeric interval {key!r}.")
    return float(value[0]), float(value[1])


def _number(payload: object, key: str) -> float:
    if not isinstance(payload, dict):
        raise ValueError("Reference safety evidence must be an object.")
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Reference safety evidence requires a numeric value {key!r}.")
    return float(value)


def _safety_estimate(
    *,
    scenario_root: Path,
    trial_output_root: Path,
    phase_id: str,
    candidate_drug_id: str,
    witness: TrialDevPhaseDecisionWitnessV1,
) -> TrialDevelopmentSafetyEstimateV1:
    method_catalog = TrialDevPhaseAnalysisMethodCatalogV1.model_validate(
        read_json(Path(scenario_root) / "public" / "phase_analysis_method_catalog.json")
    )
    method = method_catalog.method_for_phase(phase_id)
    request = TrialDevelopmentRequestV1.model_validate(read_json(Path(trial_output_root) / "request.json"))
    if request.follow_up_days is None:
        raise ValueError("Reference safety analysis requires a fixed follow-up horizon.")
    candidate = witness.candidate(candidate_drug_id)
    safety = candidate.evidence.get("safety")
    if not isinstance(safety, dict):
        raise ValueError("Decision witness lacks candidate-specific safety evidence.")
    discontinuation = safety.get("discontinuation")
    serious_treatment = _interval(safety, "treated_serious_rate_interval")
    serious_control = _interval(safety, "control_serious_rate_interval")
    serious_excess = _interval(safety, "serious_rate_excess_interval")
    disc_treatment = _interval(discontinuation, "treated_rate_interval")
    disc_control = _interval(discontinuation, "control_rate_interval")
    disc_excess = _interval(discontinuation, "rate_excess_interval")
    source_checksums = witness.evidence.get("source_checksums")
    if not isinstance(source_checksums, dict) or any(
        not isinstance(path, str) or not isinstance(checksum, str) for path, checksum in source_checksums.items()
    ):
        raise ValueError("Decision witness lacks source checksum provenance.")
    confidence = witness.evidence.get("confidence_level")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise ValueError("Decision witness lacks its confidence level.")
    return TrialDevelopmentSafetyEstimateV1(
        evidence_id=f"{phase_id}_safety",
        method_route_id=method.method_route_id,
        candidate_drug_id=candidate_drug_id,
        estimator_id=method.estimator_id,
        estimand_ids=tuple(
            method.safety_estimand_id_template.format(safety_component_id=component_id)
            for component_id in method.safety_component_ids
        ),
        absolute_risk_scale_id=method.safety_absolute_risk_scale_id,
        excess_risk_scale_id=method.safety_excess_risk_scale_id,
        orientation_id=method.safety_orientation_id,
        horizon_days=request.follow_up_days,
        analysis_population=method.analysis_population,
        serious_ae_treatment_rate=_number(safety, "treated_serious_rate"),
        serious_ae_treatment_lower=serious_treatment[0],
        serious_ae_treatment_upper=serious_treatment[1],
        serious_ae_control_rate=_number(safety, "control_serious_rate"),
        serious_ae_control_lower=serious_control[0],
        serious_ae_control_upper=serious_control[1],
        serious_ae_excess=_number(safety, "serious_rate_excess"),
        serious_ae_excess_lower=serious_excess[0],
        serious_ae_excess_upper=serious_excess[1],
        discontinuation_treatment_rate=_number(discontinuation, "treated_rate"),
        discontinuation_treatment_lower=disc_treatment[0],
        discontinuation_treatment_upper=disc_treatment[1],
        discontinuation_control_rate=_number(discontinuation, "control_rate"),
        discontinuation_control_lower=disc_control[0],
        discontinuation_control_upper=disc_control[1],
        discontinuation_excess=_number(discontinuation, "rate_excess"),
        discontinuation_excess_lower=disc_excess[0],
        discontinuation_excess_upper=disc_excess[1],
        confidence_level=float(confidence),
        source_artifact_checksums={str(path): str(checksum) for path, checksum in source_checksums.items()},
    )


def build_phase_reference_analysis_v1(
    *,
    scenario_root: Path,
    trial_output_root: Path,
    phase_id: str,
    candidate_drug_id: str,
) -> TrialDevelopmentPhaseAnalysisSubmissionV1:
    """Replay the accepted randomized analysis from participant-visible tables."""

    request = TrialDevelopmentRequestV1.model_validate(read_json(Path(trial_output_root) / "request.json"))
    manifest = validate_trial_output_bundle_v1(trial_output_root=Path(trial_output_root))
    witness = derive_phase_decision_witness_v1(
        scenario_root=Path(scenario_root),
        trial_output_root=Path(trial_output_root),
        phase_id=phase_id,
    )
    references = derive_effect_references_v1(
        scenario_root=Path(scenario_root),
        trial_output_root=Path(trial_output_root),
    )
    matches = tuple(reference for reference in references if reference.candidate_drug_id == candidate_drug_id)
    primary_effect = None
    if phase_id in {"phase2", "phase3"}:
        if len(matches) != 1:
            raise ValueError("Reference randomized analysis requires one candidate effect.")
        reference = matches[0]
        primary_effect = TrialDevelopmentEffectEstimateV1(
            evidence_id=f"{phase_id}_primary_effect",
            method_route_id=reference.method_route_id,
            candidate_drug_id=reference.candidate_drug_id,
            endpoint_id=reference.endpoint_id,
            estimand_id=reference.estimand_id,
            estimator_id=reference.estimator_id,
            effect_scale_id=reference.effect_scale_id,
            orientation_id="positive_values_favour_treatment",
            estimate=reference.estimate,
            lower=reference.lower,
            upper=reference.upper,
            confidence_level=reference.confidence_level,
            horizon_days=reference.horizon_days,
            analysis_population=reference.analysis_population,
            source_artifact_checksums=dict(reference.source_checksums),
        )
    return TrialDevelopmentPhaseAnalysisSubmissionV1(
        scenario_id=request.scenario_id,
        phase_id=phase_id,
        request_checksum=request.checksum(),
        trial_output_checksum=str(manifest.checksum),
        selected_winner_drug_id=candidate_drug_id,
        ranked_drug_ids=(candidate_drug_id,),
        primary_effect=primary_effect,
        safety_estimate=_safety_estimate(
            scenario_root=Path(scenario_root),
            trial_output_root=Path(trial_output_root),
            phase_id=phase_id,
            candidate_drug_id=candidate_drug_id,
            witness=witness,
        ),
        evidence_summary=(
            "Reference analysis applies the declared participant-visible method route to the materialized trial tables."
        ),
    )


def build_phase_reference_decision_v1(
    *,
    scenario_root: Path,
    trial_output_root: Path,
    analysis_path: Path,
    phase_id: str,
    candidate_drug_id: str,
) -> TrialDevelopmentPhaseDecisionSubmissionV1:
    """Select one evidence-supported action without consulting hidden construction state."""

    request = TrialDevelopmentRequestV1.model_validate(read_json(Path(trial_output_root) / "request.json"))
    analysis = TrialDevelopmentPhaseAnalysisSubmissionV1.model_validate(read_json(Path(analysis_path)))
    witness = derive_phase_decision_witness_v1(
        scenario_root=Path(scenario_root),
        trial_output_root=Path(trial_output_root),
        phase_id=phase_id,
    )
    candidate_actions = witness.candidate(candidate_drug_id).acceptable_action_ids
    advancing = tuple(action_id for action_id in witness.advance_action_ids if action_id in candidate_actions)
    if advancing:
        action = advancing[0]
        decision_candidate: str | None = candidate_drug_id
    else:
        stopping = tuple(
            action_id
            for action_id in witness.stop_action_ids
            if witness.action_is_acceptable(action_id=action_id, candidate_drug_id=None)
        )
        if not stopping:
            raise ValueError("Reference decision witness has no acceptable action.")
        action = stopping[0]
        decision_candidate = None
    return TrialDevelopmentPhaseDecisionSubmissionV1(
        scenario_id=request.scenario_id,
        phase_id=phase_id,
        request_checksum=request.checksum(),
        analysis_checksum=sha256_file_hex(Path(analysis_path)),
        decision_action=cast(PhaseActionIdV1, action),
        supporting_evidence_ids=tuple(sorted(analysis.evidence_ids())),
        candidate_drug_id=decision_candidate,
        decision_rationale=(
            "Action belongs to the participant-evidence-derived acceptable set "
            "under the released phase decision policy."
        ),
    )
