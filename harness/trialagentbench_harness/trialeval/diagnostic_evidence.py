"""Validation of submitted diagnostics against participant-recoverable evidence."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import cast

from trialagentbench_harness.contracts.release.trialeval_runtime_surface import (
    TrialEvalSemanticSubmissionContractV1,
)
from trialagentbench_harness.contracts.scoring.assumption_evidence import AssumptionEvidenceManifestV1
from trialagentbench_harness.contracts.scoring.diagnostic_registry import (
    diagnostic_key_by_assumption_id_v1,
    load_diagnostic_registry_v1,
)
from trialagentbench_harness.contracts.submission import (
    DiagnosticMeasureV1,
    DiagnosticSummaryResultV1,
    DiagnosticTestResultV1,
    FactualPremiseResultV1,
    ProbabilityMeasureV1,
    TrialEvalSubmissionV1,
)
from trialagentbench_harness.trialeval.schema import BenchmarkItem

_UNIT_EQUIVALENCE = {
    "chi_square": "chi_square_statistic",
    "chi_square_statistic": "chi_square_statistic",
    "chi_squared": "chi_square_statistic",
    "chi_squared_statistic": "chi_square_statistic",
}
_FACTUAL_PREMISE_BY_DIAGNOSTIC = {
    "randomization_integrity_public": "randomized_assignment_declared",
    "cluster_structure_public": "cluster_randomization_declared",
    "sequential_design_adjustment_public": "group_sequential_plan_declared",
    "censoring_followup_public": "unmeasured_prognostic_censoring_factor",
}


def _canonical_unit(unit: str) -> str:
    """Return the exact semantic unit used for diagnostic comparison."""

    return _UNIT_EQUIVALENCE.get(unit, unit)


def _participant_artifact_exists(item: BenchmarkItem, submitted_path: str) -> bool:
    relative = PurePosixPath(submitted_path)
    if (
        "\\" in submitted_path
        or relative.is_absolute()
        or submitted_path != relative.as_posix()
        or any(part in {"", ".", ".."} or part.startswith(".") for part in relative.parts)
        or ":" in submitted_path
        or {"hidden", "grader"} & {part.lower() for part in relative.parts}
        or not relative.name
    ):
        return False
    roots = [item.visible_dir]
    if item.visible_dir.parent.name == "items":
        roots.append(item.visible_dir.parent.parent)
    return any(root.joinpath(*relative.parts).is_file() for root in roots)


def _public_protocol(item: BenchmarkItem) -> dict[str, object] | None:
    """Load the participant-visible protocol summary when present."""

    path = item.visible_dir / "protocol_summary.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _factual_premise_replays(
    *,
    diagnostic_id: str,
    result: FactualPremiseResultV1,
    item: BenchmarkItem,
    source_artifacts: set[str],
) -> bool:
    """Independently evaluate one typed premise from the public protocol."""

    expected_premise = _FACTUAL_PREMISE_BY_DIAGNOSTIC.get(diagnostic_id)
    if (
        result.premise_id != expected_premise
        or result.conclusion != "supported"
        or "protocol_summary.json" not in source_artifacts
    ):
        return False
    protocol = _public_protocol(item)
    if protocol is None:
        return False
    if result.premise_id == "randomized_assignment_declared":
        arms = protocol.get("arms")
        return isinstance(arms, list) and len(arms) >= 2 and bool(str(protocol.get("design_family", "")))
    if result.premise_id == "cluster_randomization_declared":
        return protocol.get("design_family") in {
            "cluster_parallel_randomized",
            "stepped_wedge_cluster_rollout",
        }
    if result.premise_id == "group_sequential_plan_declared":
        return isinstance(protocol.get("group_sequential_plan"), dict)
    observation_process = protocol.get("observation_process")
    return (
        isinstance(observation_process, dict)
        and observation_process.get("loss_to_follow_up_role") == "observation_process_censoring"
        and observation_process.get("follow_up_decision_basis") == "clinician_assessed_prognostic_factor"
        and observation_process.get("factor_recorded_in_released_data") is False
        and observation_process.get("factor_associated_with_primary_endpoint") is True
    )


def _matching_expected_metric_ids(
    value: DiagnosticMeasureV1 | ProbabilityMeasureV1,
    *,
    expected: dict[str, float],
    expected_units: dict[str, str],
) -> tuple[str, ...]:
    """Return the exact replay-compatible evaluator metric for one measure.

    Scientific identity is checked before value comparison. A value cannot be
    reassigned to another expected metric merely because its unit and rounded
    value coincide.
    """

    canonical_target = expected.get(value.metric_id)
    canonical_unit = expected_units.get(value.metric_id)
    if (
        canonical_target is not None
        and canonical_unit is not None
        and _canonical_unit(canonical_unit) == _canonical_unit(value.unit)
        and round(float(canonical_target), value.decimal_places) == round(float(value.value), value.decimal_places)
    ):
        return (value.metric_id,)
    return ()


def _diagnostic_measures(
    result: DiagnosticTestResultV1 | DiagnosticSummaryResultV1,
) -> tuple[DiagnosticMeasureV1 | ProbabilityMeasureV1, ...]:
    if isinstance(result, DiagnosticTestResultV1):
        return (result.statistic, result.p_value)
    return cast(tuple[DiagnosticMeasureV1, ...], result.measures)


def _matched_metric_measures(
    result: DiagnosticTestResultV1 | DiagnosticSummaryResultV1,
    *,
    expected: dict[str, float],
    expected_units: dict[str, str],
) -> dict[str, DiagnosticMeasureV1 | ProbabilityMeasureV1]:
    """Return replayed evaluator metrics keyed by their declared identities.

    Supplementary measures outside the score-bearing diagnostic contract do
    not invalidate a correctly replayed required metric. Recognized metrics
    remain subject to exact identity, unit, value, and provenance checks.
    """

    measures: tuple[DiagnosticMeasureV1 | ProbabilityMeasureV1, ...] = _diagnostic_measures(result)
    matched: dict[str, DiagnosticMeasureV1 | ProbabilityMeasureV1] = {}
    for measure in measures:
        candidates = _matching_expected_metric_ids(
            measure,
            expected=expected,
            expected_units=expected_units,
        )
        if not candidates:
            continue
        metric_id = candidates[0]
        if metric_id in matched:
            return {}
        matched[metric_id] = measure
    return matched


def _precision_preserves_regime(
    value: DiagnosticMeasureV1 | ProbabilityMeasureV1,
    *,
    thresholds: tuple[float, float, float],
) -> bool:
    """Return whether reported rounding cannot change the diagnostic regime."""

    half_unit = 0.5 * 10.0 ** (-value.decimal_places)
    lower = float(value.value) - half_unit
    upper = float(value.value) + half_unit
    return not any(lower <= threshold <= upper for threshold in thresholds)


def validate_diagnostic_contract_v1(
    *,
    contract: TrialEvalSemanticSubmissionContractV1,
    assumption_evidence: AssumptionEvidenceManifestV1,
    required_diagnostic_ids: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Validate participant/evaluator diagnostic identity and score bindings."""

    records = {record.assumption_id: record for record in assumption_evidence.records}
    obligations = {obligation.assumption_id: obligation for obligation in contract.diagnostic_obligations}
    if obligations and set(obligations) != set(records):
        raise ValueError(
            "Participant diagnostic obligations do not exactly cover evaluator assumptions: "
            f"participant={sorted(obligations)!r} evaluator={sorted(records)!r}."
        )
    canonical_ids = diagnostic_key_by_assumption_id_v1()
    diagnostic_specs = load_diagnostic_registry_v1().diagnostic_keys
    result: dict[str, str] = {}
    for assumption_id, record in records.items():
        expected_diagnostic_id = canonical_ids.get(assumption_id)
        if expected_diagnostic_id is None:
            raise ValueError(f"Evaluator assumption has no canonical diagnostic identity: {assumption_id!r}.")
        obligation = obligations.get(assumption_id)
        if obligation is None:
            if record.diagnosability in {"directly_diagnosable", "partially_diagnosable"}:
                metric_id = record.severity_metric_name
                if metric_id is None:
                    raise ValueError(f"Empirical assumption {assumption_id!r} has no severity metric.")
                result[expected_diagnostic_id] = metric_id
            continue
        if expected_diagnostic_id != obligation.diagnostic_id:
            raise ValueError(
                "Participant diagnostic identity disagrees with the canonical registry: "
                f"{obligation.diagnostic_id!r} != {expected_diagnostic_id!r}."
            )
        if record.diagnosability == "design_declared":
            expected_requirement = "design_declaration"
        elif record.diagnosability == "not_identifiable":
            evidence_class = diagnostic_specs[assumption_id].nonidentified_evidence_class
            if evidence_class is None:
                raise ValueError(f"Nonidentified assumption {assumption_id!r} lacks an evidence class.")
            expected_requirement = "empirical_diagnostic" if evidence_class == "empirical_diagnostic" else "limitation"
        else:
            expected_requirement = "empirical_diagnostic"
        if obligation.evidence_requirement != expected_requirement:
            raise ValueError(
                "Participant diagnostic obligation type disagrees with evaluator diagnosability: "
                f"{obligation.evidence_requirement!r} != {expected_requirement!r}."
            )
        metric_paths = tuple(
            sorted(
                {
                    *(record.factual_public_evidence_basis or ()),
                    *(path for paths in record.metric_public_evidence_basis.values() for path in paths),
                }
            )
        )
        if metric_paths != obligation.public_evidence_basis:
            raise ValueError(
                f"Participant and evaluator diagnostic provenance disagree for {obligation.diagnostic_id!r}."
            )
        if expected_requirement != "empirical_diagnostic":
            continue
        metric_id = obligation.score_bearing_metric_id
        metric_unit = obligation.metric_unit
        if metric_id is None or metric_unit is None:  # pragma: no cover - contract invariant
            raise RuntimeError("Empirical diagnostic obligation lost its metric contract.")
        expected_metric_id = (
            record.decision_metric_names.get("broken")
            if record.diagnosability == "not_identifiable"
            else record.severity_metric_name
        )
        if expected_metric_id != metric_id:
            raise ValueError(
                "Participant and evaluator score-bearing diagnostic metrics disagree: "
                f"{metric_id!r} != {expected_metric_id!r}."
            )
        if record.metric_units.get(metric_id) != metric_unit:
            raise ValueError(
                "Participant and evaluator score-bearing diagnostic units disagree: "
                f"{metric_unit!r} != {record.metric_units.get(metric_id)!r}."
            )
        result[obligation.diagnostic_id] = metric_id
    missing_required = sorted(required_diagnostic_ids - set(result))
    if missing_required:
        raise ValueError(f"Method routes require unavailable diagnostics: {missing_required!r}.")
    return result


def validated_diagnostic_ids_v1(
    *,
    submission: TrialEvalSubmissionV1,
    item: BenchmarkItem,
    assumption_evidence: AssumptionEvidenceManifestV1,
) -> tuple[str, ...]:
    """Return diagnostics whose reported values and public provenance replay."""

    public_key_to_assumption = {
        public_key: assumption_id for assumption_id, public_key in diagnostic_key_by_assumption_id_v1().items()
    }
    expected_by_assumption = {record.assumption_id: record for record in assumption_evidence.records}
    contract = TrialEvalSemanticSubmissionContractV1.model_validate(item.submission_contract)
    required_metric_by_diagnostic = validate_diagnostic_contract_v1(
        contract=contract,
        assumption_evidence=assumption_evidence,
    )
    validated: list[str] = []
    linked_ids = set(submission.primary_analysis.evidence_ids)
    for evidence in submission.evidence:
        if evidence.evidence_id not in linked_ids or evidence.evidence_type not in {"diagnostic", "validity"}:
            continue
        if evidence.diagnostic_id is None:
            continue
        assumption_id = public_key_to_assumption.get(evidence.diagnostic_id)
        if assumption_id is None:
            continue
        expected_record = expected_by_assumption.get(assumption_id)
        if expected_record is None:
            continue
        expected_metrics = dict(expected_record.supporting_metrics)
        if expected_record.severity_metric is not None and expected_record.severity_metric_name is not None:
            expected_metrics[expected_record.severity_metric_name] = expected_record.severity_metric
        result = evidence.result
        submitted_sources = set(evidence.source_artifacts)
        if isinstance(result, FactualPremiseResultV1):
            if not all(_participant_artifact_exists(item, path) for path in submitted_sources):
                continue
            required_sources = set(expected_record.factual_public_evidence_basis or ())
            if required_sources <= submitted_sources and _factual_premise_replays(
                diagnostic_id=evidence.diagnostic_id,
                result=result,
                item=item,
                source_artifacts=submitted_sources,
            ):
                validated.append(evidence.diagnostic_id)
            continue
        if not isinstance(result, DiagnosticTestResultV1 | DiagnosticSummaryResultV1):
            continue
        measures = _diagnostic_measures(result)
        metric_ids = tuple(measure.metric_id for measure in measures)
        if len(metric_ids) != len(set(metric_ids)):
            continue
        if not all(_participant_artifact_exists(item, path) for path in submitted_sources):
            continue
        matched_measures = _matched_metric_measures(
            result,
            expected=expected_metrics,
            expected_units=expected_record.metric_units,
        )
        required_metric = required_metric_by_diagnostic.get(evidence.diagnostic_id)
        if not matched_measures or (required_metric is not None and required_metric not in matched_measures):
            continue
        if required_metric is not None:
            stressed = expected_record.threshold_stressed
            fragile = expected_record.threshold_fragile
            broken = expected_record.threshold_broken
            if stressed is None or fragile is None or broken is None:
                raise ValueError(f"Empirical diagnostic {evidence.diagnostic_id!r} lacks severity thresholds.")
            if not _precision_preserves_regime(
                matched_measures[required_metric],
                thresholds=(
                    float(stressed),
                    float(fragile),
                    float(broken),
                ),
            ):
                continue
        required_sources = {
            path for metric_id in matched_measures for path in expected_record.metric_public_evidence_basis[metric_id]
        }
        if required_sources <= submitted_sources:
            validated.append(evidence.diagnostic_id)
    return tuple(sorted(set(validated)))


__all__ = ["validate_diagnostic_contract_v1", "validated_diagnostic_ids_v1"]
