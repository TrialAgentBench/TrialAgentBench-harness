"""Grade bounded-portfolio decisions from disclosed policy and public evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

from trialagentbench_harness.contracts.trialdev.portfolio_release import (
    TrialDevPortfolioEpisodeDatasetV1,
    TrialDevPortfolioEpisodeFileV1,
    TrialDevPortfolioEpisodeMetadataFileV1,
)
from trialagentbench_harness.contracts.trialdev.portfolio_submission import (
    TrialDevPortfolioCheckpointGradeV1,
    TrialDevPortfolioCheckpointSubmissionV1,
)
from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevAnalysisStatusV1,
    TrialDevAssetEligibilityV1,
    TrialDevCheckpointIdV1,
    TrialDevCheckpointOutcomeV1,
    TrialDevDecisionRuleEvidenceV1,
    TrialDevEvidenceReferenceV1,
    TrialDevObservationalCandidateEvidenceV1,
    TrialDevObservationalDecisionEvidenceV1,
    TrialDevPairContrastEvidenceV1,
    TrialDevPortfolioActionSelectionV1,
    TrialDevPortfolioEvidenceIndexV1,
    TrialDevPortfolioProgrammeStateV1,
    TrialDevRandomizedDecisionEvidenceV1,
    TrialDevSupportedActionV1,
)
from trialagentbench_harness.contracts.trialdev.scientific_grade import (
    TrialDevScientificAssessmentV1,
    TrialDevScientificEnvelopeV1,
)
from trialagentbench_harness.contracts.trialdev.trialdev_public_recoverability import (
    TrialDevPublicObservationalActionPolicyV1,
    TrialDevPublicObservationalMethodResultV1,
    TrialDevPublicRecoverabilityReportV1,
)
from trialagentbench_harness.io import read_json, read_json_model, sha256_file
from trialagentbench_harness.trialdev.grading.analysis_evidence import (
    derive_effect_references_v1,
    reporting_tolerance_v1,
)
from trialagentbench_harness.trialdev.grading.decision_evidence import (
    TrialDevPhaseDecisionWitnessV1,
    derive_phase_decision_witness_v1,
)
from trialagentbench_harness.trialdev.grading.scientific_assessment import (
    scientific_bundle_agreement_v1,
)
from trialagentbench_harness.trialdev.policy import derive_supported_action_set_v1
from trialagentbench_harness.trialdev.portfolio_release import portfolio_evaluator_view_v1
from trialagentbench_harness.trialdev.share.models import TrialDevelopmentRequestV1
from trialagentbench_harness.trialdev.share.public_method_design import TrialDevPhaseAnalysisMethodCatalogV1

_CHECKPOINT_PHASE = {
    "joint_early_study_review": "phase1",
    "lead_proof_of_concept_review": "phase2",
    "promoted_reserve_proof_of_concept_review": "phase2",
    "confirmation": "phase3",
}


class PortfolioSubmissionError(ValueError):
    """Participant-correctable portfolio submission error."""


def _close(observed: float, expected: float, tolerance: float) -> bool:
    return math.isfinite(observed) and abs(float(observed) - float(expected)) <= tolerance


def _world_root(release_root: Path, state: TrialDevPortfolioProgrammeStateV1) -> Path:
    root: Path = Path(release_root) / "worlds" / str(state.scenario_id)
    if not (root / "public").is_dir():
        raise FileNotFoundError(f"Portfolio world public surface is absent: {root / 'public'}")
    return root


def _evaluator_inputs(
    release_root: Path,
    state: TrialDevPortfolioProgrammeStateV1,
) -> tuple[TrialDevPortfolioEvidenceIndexV1, TrialDevPublicRecoverabilityReportV1]:
    view = portfolio_evaluator_view_v1(release_root, state.programme_id)
    if view.participant_view_checksum is None:
        raise ValueError("Portfolio evaluator view lacks its participant binding.")
    index_path = Path(release_root) / view.evidence_index_relative_path
    reference_path = Path(release_root) / view.observational_reference_relative_path
    if sha256_file(index_path) != view.evidence_index_sha256:
        raise ValueError("Portfolio evidence index checksum mismatch.")
    if sha256_file(reference_path) != view.observational_reference_sha256:
        raise ValueError("Portfolio observational reference checksum mismatch.")
    index = read_json_model(TrialDevPortfolioEvidenceIndexV1, index_path)
    report = read_json_model(TrialDevPublicRecoverabilityReportV1, reference_path)
    return index, report


def _require_public_reference_checksums(world_root: Path, report: TrialDevPublicRecoverabilityReportV1) -> None:
    for record in report.public_input_checksums:
        path = Path(world_root) / record.path
        if not path.is_file() or sha256_file(path) != record.sha256:
            raise ValueError(f"Portfolio observational reference input mismatch: {record.path}.")


def _objective_margin(world_root: Path, objective_id: str) -> float:
    payload = read_json(Path(world_root) / "public" / "objective_charter.json")
    objectives = payload.get("objectives") if isinstance(payload, dict) else None
    if not isinstance(objectives, list):
        raise ValueError("Portfolio objective charter requires an objectives array.")
    matches = tuple(row for row in objectives if isinstance(row, dict) and row.get("objective_id") == objective_id)
    if len(matches) != 1:
        raise ValueError(f"Portfolio objective charter lacks one objective_id={objective_id!r}.")
    margin = matches[0].get("indifference_margin")
    if isinstance(margin, bool) or not isinstance(margin, int | float) or float(margin) < 0.0:
        raise ValueError("Portfolio objective indifference margin must be non-negative numeric.")
    return float(margin)


def _method_result(
    report: TrialDevPublicRecoverabilityReportV1,
    method_id: str,
) -> TrialDevPublicObservationalMethodResultV1:
    matches = tuple(row for row in report.method_results if row.method_route_id == method_id)
    if len(matches) != 1:
        raise PortfolioSubmissionError(f"Unknown observational analysis method route {method_id!r}.")
    return matches[0]


def _action_policy(
    method: TrialDevPublicObservationalMethodResultV1,
    objective_id: str,
) -> TrialDevPublicObservationalActionPolicyV1:
    matches = tuple(row for row in method.observational_action_policies if row.objective_id == objective_id)
    if len(matches) != 1:
        raise ValueError(f"Portfolio observational method lacks one objective policy {objective_id!r}.")
    return matches[0]


def _expected_observational_evidence(
    *,
    world_root: Path,
    state: TrialDevPortfolioProgrammeStateV1,
    submitted: TrialDevObservationalDecisionEvidenceV1,
    report: TrialDevPublicRecoverabilityReportV1,
) -> TrialDevObservationalDecisionEvidenceV1:
    method = _method_result(report, submitted.analysis_method_id)
    policy = _action_policy(method, state.policy_binding.objective_id)
    scores = tuple(row for row in method.candidate_scores if row.objective_id == state.policy_binding.objective_id)
    identified = bool(scores) and all(row.inference_estimable for row in scores)
    evidence_checksums = tuple(sorted(cast(str, item.checksum) for item in state.evidence))
    candidates: tuple[TrialDevObservationalCandidateEvidenceV1, ...] = ()
    pairs: tuple[TrialDevPairContrastEvidenceV1, ...] = ()
    if identified:
        if {row.candidate_drug_id for row in scores} != set(state.candidate_asset_ids):
            raise ValueError("Portfolio observational reference does not cover the complete candidate set.")
        candidates = tuple(
            TrialDevObservationalCandidateEvidenceV1(
                asset_id=row.candidate_drug_id,
                utility_estimate=cast(float, row.adjusted_utility),
                utility_lower_bound=cast(float, row.ci_low),
                utility_upper_bound=cast(float, row.ci_high),
                efficacy_estimate=cast(float, row.efficacy_gain),
                efficacy_lower_bound=cast(float, row.efficacy_gain_ci_low),
                efficacy_upper_bound=cast(float, row.efficacy_gain_ci_high),
                evidence_reference_checksums=evidence_checksums,
            )
            for row in sorted(scores, key=lambda item: item.candidate_drug_id)
        )
        if any(cast(float, row.efficacy_gain_ci_high) >= policy.minimum_efficacy_gain for row in scores):
            candidate_ids = tuple(sorted(state.candidate_asset_ids))
            expected_pair_keys = {
                f"{first}|{second}"
                for index, first in enumerate(candidate_ids)
                for second in candidate_ids[index + 1 :]
            }
            if not expected_pair_keys <= set(policy.pairwise_utility_contrast_half_widths):
                raise ValueError("Portfolio allocation reference lacks complete pairwise uncertainty.")
            pairs = tuple(
                TrialDevPairContrastEvidenceV1(
                    lead_asset_id=first,
                    reserve_asset_id=second,
                    confidence_half_width=policy.pairwise_utility_contrast_half_widths[f"{first}|{second}"],
                )
                for index, first in enumerate(candidate_ids)
                for second in candidate_ids[index + 1 :]
            )
    return TrialDevObservationalDecisionEvidenceV1(
        state_checksum=cast(str, state.checksum),
        analysis_method_id=submitted.analysis_method_id,
        identification_status="identified" if identified else "not_identified",
        minimum_efficacy_gain=policy.minimum_efficacy_gain,
        practical_equivalence_margin=_objective_margin(world_root, state.policy_binding.objective_id),
        candidates=candidates,
        pair_contrasts=pairs,
        identification_evidence_reference_checksums=evidence_checksums if not identified else (),
    )


def _observational_agreement(
    observed: TrialDevObservationalDecisionEvidenceV1,
    expected: TrialDevObservationalDecisionEvidenceV1,
    tolerance: float,
) -> bool:
    return not _observational_disagreement_paths(observed, expected, tolerance)


def _observational_disagreement_paths(
    observed: TrialDevObservationalDecisionEvidenceV1,
    expected: TrialDevObservationalDecisionEvidenceV1,
    tolerance: float,
) -> tuple[str, ...]:
    """Identify disagreeing observational fields without exposing reference values."""

    disagreements: list[str] = []
    if observed.identification_status != expected.identification_status:
        return ("decision_evidence.identification_status",)
    if not _close(observed.minimum_efficacy_gain, expected.minimum_efficacy_gain, tolerance):
        disagreements.append("decision_evidence.minimum_efficacy_gain")
    if not _close(observed.practical_equivalence_margin, expected.practical_equivalence_margin, tolerance):
        disagreements.append("decision_evidence.practical_equivalence_margin")
    if expected.identification_status == "not_identified":
        if observed.candidates:
            disagreements.append("decision_evidence.candidates")
        if observed.pair_contrasts:
            disagreements.append("decision_evidence.pair_contrasts")
        return tuple(disagreements)
    observed_candidates = {row.asset_id: row for row in observed.candidates}
    expected_candidates = {row.asset_id: row for row in expected.candidates}
    if set(observed_candidates) != set(expected_candidates):
        disagreements.append("decision_evidence.candidates.asset_ids")
    numeric_fields = (
        "utility_estimate",
        "utility_lower_bound",
        "utility_upper_bound",
        "efficacy_estimate",
        "efficacy_lower_bound",
        "efficacy_upper_bound",
    )
    for asset_id in sorted(set(observed_candidates) & set(expected_candidates)):
        candidate = observed_candidates[asset_id]
        reference = expected_candidates[asset_id]
        disagreements.extend(
            f"decision_evidence.candidates[{asset_id}].{field}"
            for field in numeric_fields
            if not _close(getattr(candidate, field), getattr(reference, field), tolerance)
        )
    observed_pairs = {tuple(sorted((row.lead_asset_id, row.reserve_asset_id))): row for row in observed.pair_contrasts}
    expected_pairs = {tuple(sorted((row.lead_asset_id, row.reserve_asset_id))): row for row in expected.pair_contrasts}
    if set(observed_pairs) != set(expected_pairs):
        disagreements.append("decision_evidence.pair_contrasts.asset_pairs")
    disagreements.extend(
        "decision_evidence.pair_contrasts" f"[{lead_asset_id},{reserve_asset_id}].confidence_half_width"
        for lead_asset_id, reserve_asset_id in sorted(set(observed_pairs) & set(expected_pairs))
        if not _close(
            observed_pairs[(lead_asset_id, reserve_asset_id)].confidence_half_width,
            expected_pairs[(lead_asset_id, reserve_asset_id)].confidence_half_width,
            tolerance,
        )
    )
    return tuple(disagreements)


def _episode_root(release_root: Path, reference: TrialDevEvidenceReferenceV1) -> Path:
    manifest_path: Path = Path(release_root) / str(reference.relative_path)
    if sha256_file(manifest_path) != reference.artifact_sha256:
        raise ValueError("Portfolio checkpoint evidence checksum mismatch.")
    dataset = read_json_model(TrialDevPortfolioEpisodeDatasetV1, manifest_path)
    artifacts: tuple[TrialDevPortfolioEpisodeFileV1 | TrialDevPortfolioEpisodeMetadataFileV1, ...] = (
        *dataset.files,
        *dataset.metadata_files,
    )
    for artifact in artifacts:
        artifact_path = Path(release_root) / artifact.relative_path
        if sha256_file(artifact_path) != artifact.sha256:
            raise ValueError(f"Portfolio checkpoint input checksum mismatch: {artifact.relative_path}.")
    return manifest_path.parent


def _safety_rules(
    *,
    asset_id: str,
    evidence_checksum: str,
    witness: TrialDevPhaseDecisionWitnessV1,
) -> tuple[TrialDevDecisionRuleEvidenceV1, ...]:
    candidate = next(row for row in witness.candidates if row.candidate_drug_id == asset_id)
    safety = candidate.evidence.get("safety")
    if not isinstance(safety, dict):
        raise ValueError("Portfolio phase witness lacks safety evidence.")
    rules = [
        TrialDevDecisionRuleEvidenceV1(
            rule_id=f"{asset_id}:serious_absolute",
            asset_id=asset_id,
            domain="safety",
            direction="maximum",
            estimate=float(safety["treated_serious_rate"]),
            lower_bound=float(safety["treated_serious_rate_interval"][0]),
            upper_bound=float(safety["treated_serious_rate_interval"][1]),
            threshold=float(safety["absolute_limit"]),
            evidence_reference_checksums=(evidence_checksum,),
        ),
        TrialDevDecisionRuleEvidenceV1(
            rule_id=f"{asset_id}:serious_excess",
            asset_id=asset_id,
            domain="safety",
            direction="maximum",
            estimate=float(safety["serious_rate_excess"]),
            lower_bound=float(safety["serious_rate_excess_interval"][0]),
            upper_bound=float(safety["serious_rate_excess_interval"][1]),
            threshold=float(safety["excess_limit"]),
            evidence_reference_checksums=(evidence_checksum,),
        ),
    ]
    discontinuation = safety.get("discontinuation")
    if isinstance(discontinuation, dict) and discontinuation.get("role") == "hard_gate":
        rules.extend(
            (
                TrialDevDecisionRuleEvidenceV1(
                    rule_id=f"{asset_id}:discontinuation_absolute",
                    asset_id=asset_id,
                    domain="safety",
                    direction="maximum",
                    estimate=float(discontinuation["treated_rate"]),
                    lower_bound=float(discontinuation["treated_rate_interval"][0]),
                    upper_bound=float(discontinuation["treated_rate_interval"][1]),
                    threshold=float(discontinuation["absolute_limit"]),
                    evidence_reference_checksums=(evidence_checksum,),
                ),
                TrialDevDecisionRuleEvidenceV1(
                    rule_id=f"{asset_id}:discontinuation_excess",
                    asset_id=asset_id,
                    domain="safety",
                    direction="maximum",
                    estimate=float(discontinuation["rate_excess"]),
                    lower_bound=float(discontinuation["rate_excess_interval"][0]),
                    upper_bound=float(discontinuation["rate_excess_interval"][1]),
                    threshold=float(discontinuation["excess_limit"]),
                    evidence_reference_checksums=(evidence_checksum,),
                ),
            )
        )
    return tuple(rules)


def _expected_randomized_evidence(
    *,
    release_root: Path,
    world_root: Path,
    state: TrialDevPortfolioProgrammeStateV1,
    submitted: TrialDevRandomizedDecisionEvidenceV1,
) -> TrialDevRandomizedDecisionEvidenceV1:
    phase = _CHECKPOINT_PHASE.get(state.current_checkpoint_id)
    if phase is None:
        raise ValueError("Randomized portfolio evidence is invalid at observational review.")
    method_catalog = TrialDevPhaseAnalysisMethodCatalogV1.model_validate(
        read_json(world_root / "public" / "phase_analysis_method_catalog.json")
    )
    expected_method_id = method_catalog.method_for_phase(phase).method_route_id
    if submitted.analysis_method_id != expected_method_id:
        raise PortfolioSubmissionError(
            f"Randomized {phase} evidence must use the declared method route {expected_method_id!r}."
        )
    rules: list[TrialDevDecisionRuleEvidenceV1] = []
    for reference in state.evidence:
        if reference.checkpoint_id != state.current_checkpoint_id or reference.asset_id is None:
            continue
        episode_root = _episode_root(release_root, reference)
        witness = derive_phase_decision_witness_v1(
            scenario_root=world_root,
            trial_output_root=episode_root,
            phase_id=phase,
        )
        evidence_checksum = cast(str, reference.checksum)
        rules.extend(_safety_rules(asset_id=reference.asset_id, evidence_checksum=evidence_checksum, witness=witness))
        if phase != "phase1":
            effects = derive_effect_references_v1(scenario_root=world_root, trial_output_root=episode_root)
            matches = tuple(row for row in effects if row.candidate_drug_id == reference.asset_id)
            if len(matches) != 1:
                raise ValueError("Portfolio randomized episode lacks one active-asset efficacy reference.")
            effect = matches[0]
            candidate = next(row for row in witness.candidates if row.candidate_drug_id == reference.asset_id)
            efficacy = candidate.evidence.get("efficacy")
            if not isinstance(efficacy, dict):
                raise ValueError("Portfolio phase witness lacks efficacy evidence.")
            rules.append(
                TrialDevDecisionRuleEvidenceV1(
                    rule_id=f"{reference.asset_id}:efficacy",
                    asset_id=reference.asset_id,
                    domain="efficacy",
                    direction="minimum",
                    estimate=effect.estimate,
                    lower_bound=effect.lower,
                    upper_bound=effect.upper,
                    threshold=float(efficacy["minimum_benefit"]),
                    evidence_reference_checksums=(evidence_checksum,),
                )
            )
    if not rules:
        raise ValueError("Portfolio randomized state has no current participant evidence.")
    return TrialDevRandomizedDecisionEvidenceV1(
        state_checksum=cast(str, state.checksum),
        analysis_method_id=submitted.analysis_method_id,
        rules=tuple(sorted(rules, key=lambda row: row.rule_id)),
    )


def _randomized_agreement(
    observed: TrialDevRandomizedDecisionEvidenceV1,
    expected: TrialDevRandomizedDecisionEvidenceV1,
    tolerance: float,
) -> bool:
    return not _randomized_disagreement_paths(observed, expected, tolerance)


def _randomized_disagreement_paths(
    observed: TrialDevRandomizedDecisionEvidenceV1,
    expected: TrialDevRandomizedDecisionEvidenceV1,
    tolerance: float,
) -> tuple[str, ...]:
    """Identify disagreeing randomized fields without exposing reference values."""

    observed_rules = {row.rule_id: row for row in observed.rules}
    expected_rules = {row.rule_id: row for row in expected.rules}
    disagreements: list[str] = []
    if set(observed_rules) != set(expected_rules):
        disagreements.append("decision_evidence.rules.rule_ids")
    for rule_id in sorted(set(observed_rules) & set(expected_rules)):
        rule = observed_rules[rule_id]
        reference = expected_rules[rule_id]
        for field in ("asset_id", "domain", "direction"):
            if getattr(rule, field) != getattr(reference, field):
                disagreements.append(f"decision_evidence.rules[{rule_id}].{field}")
        disagreements.extend(
            f"decision_evidence.rules[{rule_id}].{field}"
            for field in ("estimate", "lower_bound", "upper_bound", "threshold")
            if not _close(getattr(rule, field), getattr(reference, field), tolerance)
        )
    return tuple(disagreements)


def _observational_scientific_agreement(
    *,
    state: TrialDevPortfolioProgrammeStateV1,
    observed: TrialDevObservationalDecisionEvidenceV1,
    expected: TrialDevObservationalDecisionEvidenceV1,
    envelope: TrialDevScientificEnvelopeV1,
    exact_tolerance: float,
) -> bool:
    """Compare all evidence that can change the observational decision."""

    if observed.identification_status != expected.identification_status:
        return False
    if not _close(observed.minimum_efficacy_gain, expected.minimum_efficacy_gain, exact_tolerance):
        return False
    if not _close(
        observed.practical_equivalence_margin,
        expected.practical_equivalence_margin,
        exact_tolerance,
    ):
        return False
    if expected.identification_status == "not_identified":
        return not observed.candidates and not observed.pair_contrasts
    observed_candidates = {row.asset_id: row for row in observed.candidates}
    expected_candidates = {row.asset_id: row for row in expected.candidates}
    if set(observed_candidates) != set(expected_candidates):
        return False
    utility_agreement = all(
        scientific_bundle_agreement_v1(
            estimate=observed_candidates[asset_id].utility_estimate,
            lower=observed_candidates[asset_id].utility_lower_bound,
            upper=observed_candidates[asset_id].utility_upper_bound,
            reference_estimate=reference.utility_estimate,
            reference_lower=reference.utility_lower_bound,
            reference_upper=reference.utility_upper_bound,
            envelope=envelope,
        )
        for asset_id, reference in expected_candidates.items()
    )
    efficacy_envelope = TrialDevScientificEnvelopeV1(
        envelope_id="trialdev_observational_minimum_efficacy_threshold_v1",
        basis="declared_decision_thresholds",
        decision_thresholds=(float(expected.minimum_efficacy_gain),),
        exact_reproduction_tolerance=exact_tolerance,
    )
    efficacy_agreement = all(
        scientific_bundle_agreement_v1(
            estimate=observed_candidates[asset_id].efficacy_estimate,
            lower=observed_candidates[asset_id].efficacy_lower_bound,
            upper=observed_candidates[asset_id].efficacy_upper_bound,
            reference_estimate=reference.efficacy_estimate,
            reference_lower=reference.efficacy_lower_bound,
            reference_upper=reference.efficacy_upper_bound,
            envelope=efficacy_envelope,
        )
        for asset_id, reference in expected_candidates.items()
    )
    observed_pairs = {tuple(sorted((row.lead_asset_id, row.reserve_asset_id))): row for row in observed.pair_contrasts}
    expected_pairs = {tuple(sorted((row.lead_asset_id, row.reserve_asset_id))): row for row in expected.pair_contrasts}
    if set(observed_pairs) != set(expected_pairs):
        return False
    pairwise_agreement = all(
        abs(pair.confidence_half_width - expected_pairs[(lead_asset_id, reserve_asset_id)].confidence_half_width)
        <= float(envelope.absolute_margin or 0.0)
        and scientific_bundle_agreement_v1(
            estimate=(
                observed_candidates[lead_asset_id].utility_estimate
                - observed_candidates[reserve_asset_id].utility_estimate
            ),
            lower=(
                observed_candidates[lead_asset_id].utility_estimate
                - observed_candidates[reserve_asset_id].utility_estimate
                - pair.confidence_half_width
            ),
            upper=(
                observed_candidates[lead_asset_id].utility_estimate
                - observed_candidates[reserve_asset_id].utility_estimate
                + pair.confidence_half_width
            ),
            reference_estimate=(
                expected_candidates[lead_asset_id].utility_estimate
                - expected_candidates[reserve_asset_id].utility_estimate
            ),
            reference_lower=(
                expected_candidates[lead_asset_id].utility_estimate
                - expected_candidates[reserve_asset_id].utility_estimate
                - expected_pairs[(lead_asset_id, reserve_asset_id)].confidence_half_width
            ),
            reference_upper=(
                expected_candidates[lead_asset_id].utility_estimate
                - expected_candidates[reserve_asset_id].utility_estimate
                + expected_pairs[(lead_asset_id, reserve_asset_id)].confidence_half_width
            ),
            envelope=envelope,
        )
        for (lead_asset_id, reserve_asset_id), pair in observed_pairs.items()
    )
    if not (utility_agreement and efficacy_agreement and pairwise_agreement):
        return False
    observed_actions = derive_supported_action_set_v1(state=state, evidence=observed)
    expected_actions = derive_supported_action_set_v1(state=state, evidence=expected)
    return {_action_signature(action) for action in observed_actions.supported_actions} == {
        _action_signature(action) for action in expected_actions.supported_actions
    }


def _randomized_scientific_agreement(
    *,
    observed: TrialDevRandomizedDecisionEvidenceV1,
    expected: TrialDevRandomizedDecisionEvidenceV1,
    exact_tolerance: float,
) -> tuple[bool, TrialDevScientificEnvelopeV1]:
    """Compare randomized conclusions at the declared decision thresholds."""

    observed_rules = {row.rule_id: row for row in observed.rules}
    expected_rules = {row.rule_id: row for row in expected.rules}
    thresholds = tuple(sorted({float(rule.threshold) for rule in expected.rules}))
    envelope = TrialDevScientificEnvelopeV1(
        envelope_id="trialdev_randomized_declared_decision_thresholds_v1",
        basis="declared_decision_thresholds",
        decision_thresholds=thresholds,
        exact_reproduction_tolerance=exact_tolerance,
    )
    if set(observed_rules) != set(expected_rules):
        return False, envelope
    agreement = all(
        observed_rules[rule_id].asset_id == reference.asset_id
        and observed_rules[rule_id].domain == reference.domain
        and observed_rules[rule_id].direction == reference.direction
        and _close(observed_rules[rule_id].threshold, reference.threshold, exact_tolerance)
        and scientific_bundle_agreement_v1(
            estimate=observed_rules[rule_id].estimate,
            lower=observed_rules[rule_id].lower_bound,
            upper=observed_rules[rule_id].upper_bound,
            reference_estimate=reference.estimate,
            reference_lower=reference.lower_bound,
            reference_upper=reference.upper_bound,
            envelope=TrialDevScientificEnvelopeV1(
                envelope_id=f"trialdev_rule_{rule_id}_decision_threshold_v1",
                basis="declared_decision_thresholds",
                decision_thresholds=(float(reference.threshold),),
                exact_reproduction_tolerance=exact_tolerance,
            ),
        )
        for rule_id, reference in expected_rules.items()
    )
    return agreement, envelope


def _evidence_provenance_valid(
    observed: TrialDevObservationalDecisionEvidenceV1 | TrialDevRandomizedDecisionEvidenceV1,
    expected: TrialDevObservationalDecisionEvidenceV1 | TrialDevRandomizedDecisionEvidenceV1,
) -> bool:
    """Compare participant evidence links independently of numeric agreement."""

    if type(observed) is not type(expected):
        return False
    if isinstance(observed, TrialDevObservationalDecisionEvidenceV1):
        if not isinstance(expected, TrialDevObservationalDecisionEvidenceV1):
            return False
        expected_references = (
            set(expected.identification_evidence_reference_checksums)
            if expected.identification_status == "not_identified"
            else {checksum for candidate in expected.candidates for checksum in candidate.evidence_reference_checksums}
        )
        if observed.identification_status == "not_identified":
            return (
                bool(expected_references)
                and set(observed.identification_evidence_reference_checksums) == expected_references
            )
        return bool(observed.candidates) and all(
            set(candidate.evidence_reference_checksums) == expected_references for candidate in observed.candidates
        )
    if not isinstance(expected, TrialDevRandomizedDecisionEvidenceV1):
        return False
    expected_by_asset: dict[str, set[str]] = {}
    for rule in expected.rules:
        expected_by_asset.setdefault(rule.asset_id, set()).update(rule.evidence_reference_checksums)
    return bool(observed.rules) and all(
        set(rule.evidence_reference_checksums) == expected_by_asset.get(rule.asset_id, set())
        for rule in observed.rules
    )


def _asset_eligibility_v1(
    evidence: TrialDevObservationalDecisionEvidenceV1 | TrialDevRandomizedDecisionEvidenceV1,
) -> tuple[TrialDevAssetEligibilityV1, ...]:
    """Return permanent safety exclusions established at this checkpoint."""

    if isinstance(evidence, TrialDevObservationalDecisionEvidenceV1):
        return ()
    failed_by_asset: dict[str, list[TrialDevDecisionRuleEvidenceV1]] = {}
    for rule in evidence.rules:
        if rule.domain == "safety" and rule.classification == "clear_fail":
            failed_by_asset.setdefault(rule.asset_id, []).append(rule)
    return tuple(
        TrialDevAssetEligibilityV1(
            asset_id=asset_id,
            status="permanently_ineligible",
            reason="safety_clear_fail",
            policy_rule_id=";".join(sorted(rule.rule_id for rule in rules)),
            evidence_reference_checksums=tuple(
                sorted({checksum for rule in rules for checksum in rule.evidence_reference_checksums})
            ),
        )
        for asset_id, rules in sorted(failed_by_asset.items())
    )


def _action_signature(
    action: TrialDevPortfolioActionSelectionV1 | TrialDevSupportedActionV1,
) -> tuple[str, str | None, str | None]:
    return (str(action.action_id), action.target_asset_id, action.reserve_asset_id)


def _expected_next_references(
    *,
    state: TrialDevPortfolioProgrammeStateV1,
    index: TrialDevPortfolioEvidenceIndexV1,
    submission: TrialDevPortfolioCheckpointSubmissionV1,
) -> tuple[TrialDevEvidenceReferenceV1, ...]:
    action = submission.selected_action.action_id
    assets: tuple[str, ...]
    checkpoint: TrialDevCheckpointIdV1
    if action == "select_lead_and_reserve":
        assets = (
            cast(str, submission.selected_action.target_asset_id),
            cast(str, submission.selected_action.reserve_asset_id),
        )
        checkpoint = "joint_early_study_review"
    elif action == "advance_lead_to_proof_of_concept":
        assets = (cast(str, state.lead_asset_id),)
        checkpoint = "lead_proof_of_concept_review"
    elif action == "promote_reserve_to_proof_of_concept":
        assets = (cast(str, state.reserve_asset_id),)
        checkpoint = "promoted_reserve_proof_of_concept_review"
    elif action == "advance_active_to_confirmation":
        assets = (cast(str, state.active_asset_id),)
        checkpoint = "confirmation"
    else:
        return ()
    resolved: tuple[TrialDevEvidenceReferenceV1, ...] = index.resolve(
        checkpoint_id=checkpoint,
        asset_ids=assets,
    )
    return resolved


def _scheduled_designs_valid(
    *,
    release_root: Path,
    references: tuple[TrialDevEvidenceReferenceV1, ...],
    submission: TrialDevPortfolioCheckpointSubmissionV1,
) -> bool:
    expected = set()
    for reference in references:
        if reference.asset_id is None:
            raise ValueError("A scheduled portfolio study must identify one asset.")
        dataset = read_json_model(
            TrialDevPortfolioEpisodeDatasetV1,
            Path(release_root) / reference.relative_path,
        )
        request_record = next(item for item in dataset.metadata_files if item.metadata_id == "request")
        request_path = Path(release_root) / request_record.relative_path
        if sha256_file(request_path) != request_record.sha256:
            raise ValueError("Portfolio scheduled-study request checksum mismatch.")
        request = TrialDevelopmentRequestV1.model_validate(read_json(request_path))
        if request.candidate_drug_ids != (reference.asset_id,):
            raise ValueError("Portfolio scheduled-study request does not match its assigned asset.")
        expected.add((reference.asset_id, str(request.phase_id), str(request.design_cell_id)))
    observed = {(row.asset_id, row.phase_id, row.design_cell_id) for row in submission.scheduled_studies}
    return observed == expected


def grade_portfolio_checkpoint_v1(
    *,
    release_root: Path,
    state: TrialDevPortfolioProgrammeStateV1,
    submission: TrialDevPortfolioCheckpointSubmissionV1,
) -> TrialDevPortfolioCheckpointGradeV1:
    """Grade one checkpoint against independently recomputed numeric evidence."""

    if submission.state_checksum != state.checksum:
        raise PortfolioSubmissionError("Portfolio submission is stale or belongs to another programme state.")
    if submission.selected_action.checkpoint_id != state.current_checkpoint_id:
        raise PortfolioSubmissionError("Portfolio selected action does not identify the current checkpoint.")
    index, report = _evaluator_inputs(release_root, state)
    world_root = _world_root(release_root, state)
    _require_public_reference_checksums(world_root, report)
    objective_charter = read_json(world_root / "public" / "objective_charter.json")
    decimal_places = objective_charter.get("numeric_reporting_decimal_places")
    if isinstance(decimal_places, bool) or not isinstance(decimal_places, int):
        raise ValueError("Portfolio objective charter lacks numeric reporting precision.")
    tolerance = reporting_tolerance_v1(decimal_places=decimal_places)
    expected_evidence: TrialDevObservationalDecisionEvidenceV1 | TrialDevRandomizedDecisionEvidenceV1
    if isinstance(submission.decision_evidence, TrialDevObservationalDecisionEvidenceV1):
        if state.current_checkpoint_id != "observational_review":
            raise PortfolioSubmissionError("Observational evidence is valid only at portfolio observational review.")
        expected_evidence = _expected_observational_evidence(
            world_root=world_root,
            state=state,
            submitted=submission.decision_evidence,
            report=report,
        )
        numeric_disagreement_paths = _observational_disagreement_paths(
            submission.decision_evidence,
            expected_evidence,
            tolerance,
        )
        if expected_evidence.identification_status == "not_identified":
            scientific_envelope = TrialDevScientificEnvelopeV1(
                envelope_id="trialdev_qualified_nonidentification_v1",
                basis="qualified_nonidentification",
                exact_reproduction_tolerance=tolerance,
            )
            scientific_agreement = _observational_scientific_agreement(
                state=state,
                observed=submission.decision_evidence,
                expected=expected_evidence,
                envelope=scientific_envelope,
                exact_tolerance=tolerance,
            )
        else:
            scientific_envelope = TrialDevScientificEnvelopeV1(
                envelope_id="trialdev_declared_utility_indifference_margin_v1",
                basis="declared_practical_equivalence_margin",
                absolute_margin=expected_evidence.practical_equivalence_margin,
                exact_reproduction_tolerance=tolerance,
            )
            scientific_agreement = _observational_scientific_agreement(
                state=state,
                observed=submission.decision_evidence,
                expected=expected_evidence,
                envelope=scientific_envelope,
                exact_tolerance=tolerance,
            )
        analysis_status: TrialDevAnalysisStatusV1 = (
            "non_estimable" if expected_evidence.identification_status == "not_identified" else "estimable"
        )
    else:
        if state.current_checkpoint_id == "observational_review":
            raise PortfolioSubmissionError("Randomized evidence is invalid at portfolio observational review.")
        expected_evidence = _expected_randomized_evidence(
            release_root=release_root,
            world_root=world_root,
            state=state,
            submitted=submission.decision_evidence,
        )
        numeric_disagreement_paths = _randomized_disagreement_paths(
            submission.decision_evidence,
            expected_evidence,
            tolerance,
        )
        scientific_agreement, scientific_envelope = _randomized_scientific_agreement(
            observed=submission.decision_evidence,
            expected=expected_evidence,
            exact_tolerance=tolerance,
        )
        analysis_status = "estimable"
    numeric_agreement = not numeric_disagreement_paths
    supported = derive_supported_action_set_v1(state=state, evidence=expected_evidence)
    selected_supported = _action_signature(submission.selected_action) in {
        _action_signature(action) for action in supported.supported_actions
    }
    required_evidence_ids = {
        item.evidence_id for item in state.evidence if item.checkpoint_id == state.current_checkpoint_id
    }
    provenance_valid = (
        submission.selected_action.analysis_method_id == expected_evidence.analysis_method_id
        and set(submission.selected_action.supporting_evidence_ids) == required_evidence_ids
        and bool(required_evidence_ids)
        and _evidence_provenance_valid(submission.decision_evidence, expected_evidence)
    )
    next_references = _expected_next_references(state=state, index=index, submission=submission)
    designs_valid = _scheduled_designs_valid(
        release_root=release_root,
        references=next_references,
        submission=submission,
    )
    design_status = (
        "passed" if next_references and designs_valid else "failed" if next_references else "not_applicable"
    )
    failure_reasons = tuple(
        reason
        for failed, reason in (
            (not scientific_agreement, "scientific_disagreement"),
            (not provenance_valid, "evidence_provenance_invalid"),
            (not selected_supported, "action_not_supported"),
            (bool(next_references) and not designs_valid, "scheduled_design_invalid"),
        )
        if failed
    )
    scientific_assessment = TrialDevScientificAssessmentV1(
        execution="passed",
        question_estimand="passed",
        design=design_status,
        assumptions="passed",
        analysis_classification="uncertainty_qualified",
        scientific_agreement="passed" if scientific_agreement else "failed",
        exact_reproduction="passed" if numeric_agreement else "failed",
        uncertainty="passed",
        action_admissibility="passed" if selected_supported else "failed",
        evidential_support="passed" if provenance_valid else "failed",
        sequential_coherence="passed" if designs_valid else "failed",
        scientific_envelope=scientific_envelope,
        failure_reasons=failure_reasons,
        decision_complete=bool(scientific_agreement and provenance_valid and selected_supported and designs_valid),
    )
    return TrialDevPortfolioCheckpointGradeV1(
        checkpoint_id=state.current_checkpoint_id,
        state_checksum=cast(str, state.checksum),
        evidence_numeric_agreement=numeric_agreement,
        numeric_disagreement_paths=numeric_disagreement_paths,
        provenance_valid=provenance_valid,
        supported_action_set=supported,
        selected_action_supported=selected_supported,
        scheduled_designs_valid=designs_valid,
        scientific_assessment=scientific_assessment,
        outcome=TrialDevCheckpointOutcomeV1(
            reach_status="reached",
            submission_status="accepted",
            analysis_status=analysis_status,
            execution_status="completed",
            asset_eligibility=(
                _asset_eligibility_v1(expected_evidence) if scientific_assessment.decision_complete else ()
            ),
        ),
    )


__all__ = ["PortfolioSubmissionError", "grade_portfolio_checkpoint_v1"]
