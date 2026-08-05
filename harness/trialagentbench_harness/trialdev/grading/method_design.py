"""Load and resolve public TrialDev method and design cells."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from trialagentbench_harness.trialdev.grading.io import read_json
from trialagentbench_harness.trialdev.share.public_method_design import (
    TrialDevPhaseAnalysisMethodCatalogV1,
    TrialDevPhaseAnalysisMethodRouteV1,
    TrialDevPhaseDesignCellV1,
    TrialDevPhaseDesignPolicyV1,
    TrialDevPublicObjectiveCharterV1,
    TrialDevPublicObservationalMethodCatalogV1,
    TrialDevPublicObservationalMethodSpecV1,
)
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentCandidateUtilityEstimateV1,
    TrialDevelopmentEffectEstimateV1,
    TrialDevelopmentSafetyEstimateV1,
)


@dataclass(frozen=True, slots=True)
class TrialDevPublicMethodDesignContractsV1:
    """Validated public method/design contracts for one scenario."""

    objective_charter: TrialDevPublicObjectiveCharterV1
    observational_methods: TrialDevPublicObservationalMethodCatalogV1
    phase_methods: TrialDevPhaseAnalysisMethodCatalogV1
    phase_designs: TrialDevPhaseDesignPolicyV1


def load_public_method_design_contracts_v1(
    scenario_root: Path,
) -> TrialDevPublicMethodDesignContractsV1:
    """Load checksum-bound public contracts and enforce shared references."""

    public = Path(scenario_root) / "public"
    objective_charter = TrialDevPublicObjectiveCharterV1.model_validate(read_json(public / "objective_charter.json"))
    observational_methods = TrialDevPublicObservationalMethodCatalogV1.model_validate(
        read_json(public / "observational_method_catalog.json")
    )
    methods = TrialDevPhaseAnalysisMethodCatalogV1.model_validate(
        read_json(public / "phase_analysis_method_catalog.json")
    )
    designs = TrialDevPhaseDesignPolicyV1.model_validate(read_json(public / "phase_design_policy.json"))
    scenario_ids = {
        objective_charter.scenario_id,
        observational_methods.scenario_id,
        methods.scenario_id,
        designs.scenario_id,
    }
    if len(scenario_ids) != 1:
        raise ValueError("TrialDev public method/design contracts disagree on scenario_id.")
    confidence_levels = (
        objective_charter.confidence_level,
        observational_methods.confidence_level,
        methods.confidence_level,
        designs.confidence_level,
    )
    if any(
        not math.isclose(
            confidence,
            objective_charter.confidence_level,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for confidence in confidence_levels
    ):
        raise ValueError("TrialDev public method/design contracts disagree on confidence_level.")
    if (
        objective_charter.decision_charter_checksum != designs.decision_charter_checksum
        or observational_methods.decision_charter_checksum != designs.decision_charter_checksum
    ):
        raise ValueError("TrialDev objective and design contracts reference different charters.")
    charter = read_json(public / "decision_charter.json")
    if not isinstance(charter, dict):
        raise ValueError("decision_charter.json must contain an object.")
    if charter.get("checksum") != objective_charter.decision_charter_checksum:
        raise ValueError("TrialDev public method/design contracts reference the wrong charter.")
    if not math.isclose(
        float(charter.get("confidence_level", 0.0)),
        objective_charter.confidence_level,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("TrialDev public contract confidence differs from the charter.")
    return TrialDevPublicMethodDesignContractsV1(
        objective_charter=objective_charter,
        observational_methods=observational_methods,
        phase_methods=methods,
        phase_designs=designs,
    )


def observational_submission_matches_cell_v1(
    *,
    estimates: tuple[TrialDevelopmentCandidateUtilityEstimateV1, ...],
    contracts: TrialDevPublicMethodDesignContractsV1,
    source_checksums: dict[str, str],
) -> bool:
    """Return whether every utility estimate resolves to the reference method route."""

    if not estimates:
        return False
    matching_cells = tuple(
        cell
        for cell in contracts.observational_methods.methods
        if all(_observational_estimate_matches_cell(estimate, cell, source_checksums) for estimate in estimates)
    )
    return len(matching_cells) == 1


def resolve_observational_method_route_v1(
    *,
    estimates: tuple[TrialDevelopmentCandidateUtilityEstimateV1, ...],
    contracts: TrialDevPublicMethodDesignContractsV1,
) -> TrialDevPublicObservationalMethodSpecV1 | None:
    """Resolve one method from submitted semantics without consulting numeric reference."""

    if not estimates:
        return None
    matching_cells = tuple(
        cell
        for cell in contracts.observational_methods.methods
        if all(_observational_estimate_matches_semantics(estimate, cell) for estimate in estimates)
    )
    if len(matching_cells) != 1:
        return None
    return matching_cells[0]


def _observational_estimate_matches_cell(
    estimate: TrialDevelopmentCandidateUtilityEstimateV1,
    cell: TrialDevPublicObservationalMethodSpecV1,
    source_checksums: dict[str, str],
) -> bool:
    """Match one estimate to one observational cell by executable semantics."""

    return _observational_estimate_matches_semantics(estimate, cell) and (
        estimate.source_artifact_checksums == source_checksums
    )


def _observational_estimate_matches_semantics(
    estimate: TrialDevelopmentCandidateUtilityEstimateV1,
    cell: TrialDevPublicObservationalMethodSpecV1,
) -> bool:
    """Match method semantics without reading any evaluator numerical reference."""

    return (
        _optional_cell_id_agrees(estimate.method_route_id, cell.method_route_id)
        and estimate.estimator_id == cell.primary_estimator_id
        and estimate.utility_unit == cell.effect_scale_id
        and math.isclose(
            estimate.confidence_level,
            cell.confidence_level,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and frozenset(estimate.analysis_covariate_ids) == frozenset(cell.adjustment_covariates)
    )


def effect_submission_matches_cell_v1(
    *,
    effect: TrialDevelopmentEffectEstimateV1,
    cell: TrialDevPhaseAnalysisMethodRouteV1,
    phase_id: str,
    treatment_discontinuation_strategy: str,
    horizon_days: int,
    source_checksums: dict[str, str],
) -> bool:
    """Return whether an efficacy estimate is an exact member of a phase cell."""

    if cell.efficacy_estimand_id_template is None or cell.efficacy_effect_scale_id is None:
        return False
    expected_estimand = cell.efficacy_estimand_id_template.format(
        treatment_discontinuation_strategy=treatment_discontinuation_strategy,
    )
    return (
        phase_id == cell.phase_id
        and _optional_cell_id_agrees(effect.method_route_id, cell.method_route_id)
        and effect.estimator_id == cell.estimator_id
        and effect.estimand_id == expected_estimand
        and effect.effect_scale_id == cell.efficacy_effect_scale_id
        and effect.orientation_id == cell.efficacy_orientation_id
        and effect.horizon_days == horizon_days
        and effect.analysis_population == cell.analysis_population
        and math.isclose(
            effect.confidence_level,
            cell.confidence_level,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and effect.source_artifact_checksums == source_checksums
    )


def safety_submission_matches_cell_v1(
    *,
    safety: TrialDevelopmentSafetyEstimateV1,
    cell: TrialDevPhaseAnalysisMethodRouteV1,
    phase_id: str,
    horizon_days: int,
    source_checksums: dict[str, str],
) -> bool:
    """Return whether a safety bundle is an exact member of a phase cell."""

    expected_estimands = tuple(
        cell.safety_estimand_id_template.format(safety_component_id=component)
        for component in cell.safety_component_ids
    )
    return (
        phase_id == cell.phase_id
        and _optional_cell_id_agrees(safety.method_route_id, cell.method_route_id)
        and safety.estimator_id == cell.estimator_id
        and frozenset(safety.estimand_ids) == frozenset(expected_estimands)
        and safety.absolute_risk_scale_id == cell.safety_absolute_risk_scale_id
        and safety.excess_risk_scale_id == cell.safety_excess_risk_scale_id
        and safety.orientation_id == cell.safety_orientation_id
        and safety.horizon_days == horizon_days
        and safety.analysis_population == cell.analysis_population
        and math.isclose(
            safety.confidence_level,
            cell.confidence_level,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and safety.source_artifact_checksums == source_checksums
    )


def submitted_analysis_cell_id_v1(
    *,
    phase_id: str,
    utility_estimates: tuple[TrialDevelopmentCandidateUtilityEstimateV1, ...],
    effect: TrialDevelopmentEffectEstimateV1 | None,
    safety: TrialDevelopmentSafetyEstimateV1 | None,
) -> str | None:
    """Return the unique method route ID declared by the submission."""

    if phase_id == "observational_review":
        ids = {estimate.method_route_id for estimate in utility_estimates}
    else:
        ids = {
            value
            for value in (
                None if effect is None else effect.method_route_id,
                None if safety is None else safety.method_route_id,
            )
            if value is not None
        }
    if not ids:
        return None
    if len(ids) != 1:
        return "inconsistent_submitted_method_routes"
    return ids.pop()


def _optional_cell_id_agrees(submitted: str | None, expected: str) -> bool:
    """Require a supplied opaque ID to agree with the semantic method route."""

    return submitted is None or submitted == expected


def required_phase_cells_v1(
    *,
    contracts: TrialDevPublicMethodDesignContractsV1,
    phase_id: str,
) -> tuple[TrialDevPhaseAnalysisMethodRouteV1, TrialDevPhaseDesignCellV1]:
    """Return the reference method and design cells for one phase."""

    return (
        contracts.phase_methods.method_for_phase(phase_id),
        contracts.phase_designs.rule_for_phase(phase_id),
    )


__all__ = [
    "TrialDevPublicMethodDesignContractsV1",
    "effect_submission_matches_cell_v1",
    "load_public_method_design_contracts_v1",
    "observational_submission_matches_cell_v1",
    "resolve_observational_method_route_v1",
    "required_phase_cells_v1",
    "safety_submission_matches_cell_v1",
    "submitted_analysis_cell_id_v1",
]
