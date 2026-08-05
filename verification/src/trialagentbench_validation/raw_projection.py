"""Independent raw-to-canonical TrialEval witness projection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from trialagentbench_validation.grader_concordance import (
        CanonicalSubmissionV1,
        RouteV1,
    )


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class _HorizonV1(_Contract):
    value: float = Field(gt=0)
    unit: Literal["days"]


class _EstimandV1(_Contract):
    estimand_id: str
    population_id: str
    treatment_id: str
    comparator_id: str
    endpoint_id: str
    intercurrent_event_strategy_ids: tuple[str, ...] = ()
    horizon: _HorizonV1 | None = None
    horizon_not_applicable_reason: str | None = None

    @model_validator(mode="after")
    def _one_horizon_state(self) -> _EstimandV1:
        if (self.horizon is None) == (self.horizon_not_applicable_reason is None):
            raise ValueError("raw estimand must declare one horizon state")
        return self


class _IntervalV1(_Contract):
    lower: float
    upper: float
    confidence_level: float = Field(gt=0, lt=1)


class _ScalarResultV1(_Contract):
    kind: Literal["scalar"]
    value: float
    effect_scale: str
    unit: str
    interval: _IntervalV1


class _IdentifiedIntervalV1(_Contract):
    kind: Literal["identified_interval"]
    lower: float
    upper: float
    effect_scale: str
    unit: str
    interpretation: str


class _VectorPointV1(_Contract):
    component_id: str
    index: float
    value: float


class _VectorResultV1(_Contract):
    kind: Literal["vector"]
    points: tuple[_VectorPointV1, ...] = Field(min_length=2)
    index_unit: str
    effect_scale: str
    unit: str


class _TestResultV1(_Contract):
    kind: Literal["statistical_test"]
    statistic: float
    p_value: float = Field(ge=0, le=1)
    reject_null: bool
    effect_scale: str
    unit: str
    alternative: str
    rho: float
    gamma: float


class _NonIdentificationResultV1(_Contract):
    kind: Literal["non_identification"]
    conclusion_code: str
    effect_scale: str
    unit: str
    reason: str
    identified_set: _IdentifiedIntervalV1 | None = None
    additional_assumption_required: str


_PrimaryResultV1 = Annotated[
    _ScalarResultV1
    | _IdentifiedIntervalV1
    | _VectorResultV1
    | _TestResultV1
    | _NonIdentificationResultV1,
    Field(discriminator="kind"),
]


class _DiagnosticMeasureV1(_Contract):
    metric_id: str
    value: float
    unit: str
    decimal_places: int = Field(ge=0, le=12)


class _DiagnosticSummaryV1(_Contract):
    kind: Literal["diagnostic_summary"]
    measures: tuple[_DiagnosticMeasureV1, ...] = Field(min_length=1)


class _FactualPremiseV1(_Contract):
    kind: Literal["factual_premise"]
    premise_id: str
    conclusion: Literal["supported", "not_supported", "unresolved"]


_EvidenceResultV1 = Annotated[
    _ScalarResultV1
    | _IdentifiedIntervalV1
    | _VectorResultV1
    | _TestResultV1
    | _NonIdentificationResultV1
    | _DiagnosticSummaryV1
    | _FactualPremiseV1,
    Field(discriminator="kind"),
]


class _EstimatorV1(_Contract):
    analysis_method_id: str
    implementation: str | None = None
    qualifications: tuple[str, ...] = ()


class _SensitivityParameterV1(_Contract):
    value: float = Field(ge=0.0, le=1.0)
    unit: Literal["probability"]


class _EvidenceV1(_Contract):
    evidence_id: str
    evidence_type: Literal[
        "diagnostic", "validity", "sensitivity", "supporting_analysis", "data_quality"
    ]
    principle: str
    operation: str
    diagnostic_id: str | None = None
    sensitivity_parameter: _SensitivityParameterV1 | None = None
    estimator: _EstimatorV1 | None = None
    target: str
    result: _EvidenceResultV1
    interpretation: str
    source_artifacts: tuple[str, ...]


class _PrimaryAnalysisV1(_Contract):
    declared_primary: Literal[True]
    estimand: _EstimandV1
    estimator: _EstimatorV1
    result_kind: Literal[
        "numeric_point",
        "numeric_interval",
        "numeric_vector",
        "statistical_test",
        "sensitivity_set",
        "identification_bound",
        "limitation",
        "abstention",
        "decision",
    ]
    result: _PrimaryResultV1
    favorable_direction: Literal["higher", "lower", "neither"]
    evidence_ids: tuple[str, ...] = ()


class _IntegrityRecordV1(_Contract):
    condition_id: Literal["exact_transport_row_duplication_v1"]
    affected_domain: str
    compound_key_fields: tuple[str, ...]
    observed_duplicate_group_count: int
    observed_extra_row_count: int
    repair_action: Literal["remove_one_exact_duplicate_copy"]
    repair_status: Literal["repaired", "unexpected_data_integrity_state"]
    post_repair_data_checksum: str
    analysis_input_data_checksum: str


class _PlanningSensitivityV1(_Contract):
    event_probability: float = Field(gt=0.0, le=1.0)
    target_sample_size: int = Field(gt=0)


class _PlanningV1(_Contract):
    method_id: Literal["schoenfeld_logrank_v1"]
    estimand_id: str
    alpha_two_sided: float = Field(gt=0.0, lt=1.0)
    power: float = Field(gt=0.0, lt=1.0)
    treated_allocation_fraction: float = Field(gt=0.0, lt=1.0)
    event_probability: float = Field(gt=0.0, le=1.0)
    followup_horizon_dy: float = Field(gt=0.0)
    multiplicity_adjustment: Literal["none"]
    required_events: int = Field(gt=0)
    target_sample_size: int = Field(gt=0)
    sensitivity: tuple[_PlanningSensitivityV1, ...] = Field(min_length=2)


class _ReconstructionV1(_Contract):
    n_subjects: int = Field(ge=0)
    n_primary_population: int = Field(ge=0)
    n_events: int = Field(ge=0)
    n_censored: int = Field(ge=0)
    checks_performed: tuple[str, ...] = ()
    notes: str
    source_artifacts: tuple[str, ...] = Field(min_length=1)


class _RawSubmissionV1(_Contract):
    schema_id: Literal["trialagentbench.trialeval_submission/v1"]
    task_id: str
    primary_analysis: _PrimaryAnalysisV1
    evidence: tuple[_EvidenceV1, ...] = ()
    planning: _PlanningV1 | None = None
    reconstruction: _ReconstructionV1 | None = None
    data_integrity_record: _IntegrityRecordV1 | None = None
    limitations: tuple[str, ...] = ()


class RawTrialEvalRouteWitnessV1(_Contract):
    """Independent parser for one public raw route witness."""

    schema_id: Literal["trialagentbench.trialeval_raw_route_witness/v1"]
    release_id: str
    witness_id: str
    item_id: str
    route_id: str
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    primary_evidence_class: str
    repair_required: bool
    fixed_question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    route_signature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_input_checksums: dict[str, str] = Field(min_length=1)
    raw_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submission: _RawSubmissionV1


class _DiagnosticObligationV1(_Contract):
    assumption_id: str
    diagnostic_id: str
    evidence_requirement: str
    primary_credit_policy: str
    operation: str
    score_bearing_metric_id: str | None = None
    metric_unit: str | None = None
    public_evidence_basis: tuple[str, ...]
    interpretation: str


class _SubmissionContractV1(_Contract):
    schema_id: Literal["trialagentbench.trialeval_semantic_submission_contract/v1"]
    task_id: str
    submission_semantics_id: str
    required_deliverables: tuple[str, ...]
    diagnostic_obligations: tuple[_DiagnosticObligationV1, ...]
    checksum: str


class _AssumptionEvidenceV1(_Contract):
    assumption_id: str
    expected_status: str
    computed_status: str
    expected_band: str
    computed_band: str
    diagnosability: str
    severity_metric: float | None = None
    severity_metric_name: str | None = None
    threshold_stressed: float | None = None
    threshold_fragile: float | None = None
    threshold_broken: float | None = None
    decision_metric_names: dict[str, str] = {}
    supporting_metrics: dict[str, float] = {}
    metric_units: dict[str, str]
    metric_public_evidence_basis: dict[str, tuple[str, ...]]
    factual_public_evidence_basis: tuple[str, ...] | None = None
    notes: tuple[str, ...] = ()


class _AssumptionManifestV1(_Contract):
    version: Literal["v1"]
    schema_id: Literal["trial_benchmark_assumption_evidence_manifest_v1"]
    item_id: str
    base_case_id: str
    canonical_item_id: str
    variant_id: str
    context_tier: str
    replicate_index: int
    records: tuple[_AssumptionEvidenceV1, ...]
    checksum: str


_FACTUAL_PREMISE_BY_DIAGNOSTIC = {
    "randomization_integrity_public": "randomized_assignment_declared",
    "cluster_structure_public": "cluster_randomization_declared",
    "sequential_design_adjustment_public": "group_sequential_plan_declared",
    "censoring_followup_public": "unmeasured_prognostic_censoring_factor",
}
_UNIT_EQUIVALENCE = {
    "chi_square": "chi_square_statistic",
    "chi_square_statistic": "chi_square_statistic",
    "chi_squared": "chi_square_statistic",
    "chi_squared_statistic": "chi_square_statistic",
}


def _canonical_sha256(payload: object) -> str:
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _read_assumption_manifests(
    evaluator_root: Path,
) -> dict[str, _AssumptionManifestV1]:
    path = evaluator_root / "grader" / "domains" / "assumption_evidence.jsonl"
    manifests: dict[str, _AssumptionManifestV1] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("domain") != "assumption_evidence":
            raise ValueError(
                "independent raw projection found a malformed assumption-evidence row"
            )
        task_id = row.get("task_id")
        payload = row.get("payload")
        manifest = payload.get("manifest") if isinstance(payload, dict) else None
        if not isinstance(task_id, str) or not isinstance(manifest, dict):
            raise ValueError(
                "independent raw projection found incomplete assumption evidence"
            )
        if task_id in manifests:
            raise ValueError(f"duplicate independent assumption evidence: {task_id}")
        manifests[task_id] = _AssumptionManifestV1.model_validate(manifest)
    return manifests


def _canonical_unit(unit: str) -> str:
    return _UNIT_EQUIVALENCE.get(unit, unit)


def _participant_artifact_exists(
    *,
    participant_item_root: Path,
    participant_root: Path,
    submitted_path: str,
) -> bool:
    relative = PurePosixPath(submitted_path)
    if (
        "\\" in submitted_path
        or relative.is_absolute()
        or submitted_path != relative.as_posix()
        or any(
            part in {"", ".", ".."} or part.startswith(".") for part in relative.parts
        )
        or ":" in submitted_path
        or {"hidden", "grader"} & {part.lower() for part in relative.parts}
        or not relative.name
    ):
        return False
    return any(
        root.joinpath(*relative.parts).is_file()
        for root in (participant_item_root, participant_root)
    )


def _measure_matches(
    measure: _DiagnosticMeasureV1,
    *,
    expected: dict[str, float],
    expected_units: dict[str, str],
) -> bool:
    target = expected.get(measure.metric_id)
    unit = expected_units.get(measure.metric_id)
    return (
        target is not None
        and unit is not None
        and _canonical_unit(unit) == _canonical_unit(measure.unit)
        and round(float(target), measure.decimal_places)
        == round(float(measure.value), measure.decimal_places)
    )


def _precision_preserves_regime(
    measure: _DiagnosticMeasureV1,
    *,
    thresholds: tuple[float, float, float],
) -> bool:
    half_unit = 0.5 * 10.0 ** (-measure.decimal_places)
    lower = float(measure.value) - half_unit
    upper = float(measure.value) + half_unit
    return not any(lower <= threshold <= upper for threshold in thresholds)


def _factual_supported(
    *,
    diagnostic_id: str,
    premise: _FactualPremiseV1,
    protocol: dict[str, object],
) -> bool:
    if (
        premise.premise_id != _FACTUAL_PREMISE_BY_DIAGNOSTIC.get(diagnostic_id)
        or premise.conclusion != "supported"
    ):
        return False
    if premise.premise_id == "randomized_assignment_declared":
        arms = protocol.get("arms")
        return (
            isinstance(arms, list)
            and len(arms) >= 2
            and bool(protocol.get("design_family"))
        )
    if premise.premise_id == "cluster_randomization_declared":
        return protocol.get("design_family") in {
            "cluster_parallel_randomized",
            "stepped_wedge_cluster_rollout",
        }
    if premise.premise_id == "group_sequential_plan_declared":
        return isinstance(protocol.get("group_sequential_plan"), dict)
    observation = protocol.get("observation_process")
    return (
        isinstance(observation, dict)
        and observation.get("loss_to_follow_up_role") == "observation_process_censoring"
        and observation.get("follow_up_decision_basis")
        == "clinician_assessed_prognostic_factor"
        and observation.get("factor_recorded_in_released_data") is False
        and observation.get("factor_associated_with_primary_endpoint") is True
    )


def _validated_diagnostic_ids(
    *,
    witness: RawTrialEvalRouteWitnessV1,
    participant_item_root: Path,
    participant_root: Path,
    contract: _SubmissionContractV1,
    manifest: _AssumptionManifestV1,
    required_diagnostics: tuple[str, ...],
    identification_assumptions: tuple[str, ...],
) -> tuple[str, ...]:
    obligations = {row.diagnostic_id: row for row in contract.diagnostic_obligations}
    assumptions = {row.assumption_id: row for row in manifest.records}
    linked = set(witness.submission.primary_analysis.evidence_ids)
    output: list[str] = []
    for evidence in witness.submission.evidence:
        diagnostic_id = evidence.diagnostic_id
        if diagnostic_id is None or evidence.evidence_id not in linked:
            continue
        obligation = obligations.get(diagnostic_id)
        assumption_id = _route_diagnostic_assumption_id(
            diagnostic_id=diagnostic_id,
            evidence_target=evidence.target,
            obligation=obligation,
            required_diagnostics=required_diagnostics,
            identification_assumptions=identification_assumptions,
        )
        if assumption_id is None:
            continue
        assumption = assumptions.get(assumption_id)
        if assumption is None:
            continue
        sources = set(evidence.source_artifacts)
        if not all(
            _participant_artifact_exists(
                participant_item_root=participant_item_root,
                participant_root=participant_root,
                submitted_path=path,
            )
            for path in sources
        ):
            continue
        result = evidence.result
        if isinstance(result, _FactualPremiseV1):
            required = set(assumption.factual_public_evidence_basis or ())
            protocol_path = participant_item_root / "protocol_summary.json"
            if (
                required <= sources
                and protocol_path.is_file()
                and _factual_supported(
                    diagnostic_id=diagnostic_id,
                    premise=result,
                    protocol=json.loads(protocol_path.read_text(encoding="utf-8")),
                )
            ):
                output.append(diagnostic_id)
            continue
        if not isinstance(result, _DiagnosticSummaryV1):
            continue
        expected = dict(assumption.supporting_metrics)
        if (
            assumption.severity_metric_name is not None
            and assumption.severity_metric is not None
        ):
            expected[assumption.severity_metric_name] = assumption.severity_metric
        metric_ids = tuple(measure.metric_id for measure in result.measures)
        if len(metric_ids) != len(set(metric_ids)):
            continue
        matched: dict[str, _DiagnosticMeasureV1] = {}
        for measure in result.measures:
            if _measure_matches(
                measure,
                expected=expected,
                expected_units=assumption.metric_units,
            ):
                matched[measure.metric_id] = measure
        required_metric = (
            None if obligation is None else obligation.score_bearing_metric_id
        )
        if not matched or (
            required_metric is not None and required_metric not in matched
        ):
            continue
        if required_metric is not None:
            stressed = assumption.threshold_stressed
            fragile = assumption.threshold_fragile
            broken = assumption.threshold_broken
            if stressed is None or fragile is None or broken is None:
                raise ValueError(
                    f"empirical diagnostic {diagnostic_id!r} lacks severity thresholds"
                )
            if not _precision_preserves_regime(
                matched[required_metric],
                thresholds=(float(stressed), float(fragile), float(broken)),
            ):
                continue
        required_sources = {
            path
            for metric_id in matched
            for path in assumption.metric_public_evidence_basis[metric_id]
        }
        if required_sources <= sources:
            output.append(diagnostic_id)
    return tuple(sorted(set(output)))


def _route_diagnostic_assumption_id(
    *,
    diagnostic_id: str,
    evidence_target: str,
    obligation: _DiagnosticObligationV1 | None,
    required_diagnostics: tuple[str, ...],
    identification_assumptions: tuple[str, ...],
) -> str | None:
    """Resolve a diagnostic to its route-bound identification assumption."""

    if obligation is not None:
        return obligation.assumption_id
    if (
        diagnostic_id not in required_diagnostics
        or evidence_target not in identification_assumptions
    ):
        return None
    return evidence_target


def _project_result(result: _PrimaryResultV1) -> dict[str, object]:
    if isinstance(result, _ScalarResultV1):
        return {
            "kind": "numeric_point",
            "value": result.value,
            "result_unit": result.unit,
            "confidence_interval_lower": result.interval.lower,
            "confidence_interval_upper": result.interval.upper,
        }
    if isinstance(result, _IdentifiedIntervalV1):
        return {
            "kind": "numeric_interval",
            "lower": result.lower,
            "upper": result.upper,
            "result_unit": result.unit,
        }
    if isinstance(result, _VectorResultV1):
        return {
            "kind": "numeric_vector",
            "components": [
                {"name": point.component_id, "value": point.value}
                for point in result.points
            ],
            "result_unit": result.unit,
        }
    if isinstance(result, _TestResultV1):
        return {
            "kind": "statistical_test",
            "p_value": result.p_value,
            "reject_null": result.reject_null,
        }
    return {
        "kind": "categorical",
        "code": result.conclusion_code,
    }


def project_raw_witnesses_independently(
    *,
    raw_witnesses_path: Path,
    participant_root: Path,
    evaluator_root: Path,
    route_by_identity: Mapping[tuple[str, str], RouteV1],
    require_route_evidence: bool = True,
) -> tuple[tuple[str, CanonicalSubmissionV1], ...]:
    """Project every raw witness without importing harness projection code."""

    from trialagentbench_validation.grader_concordance import CanonicalSubmissionV1

    witnesses = tuple(
        RawTrialEvalRouteWitnessV1.model_validate_json(line)
        for line in raw_witnesses_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    manifests = _read_assumption_manifests(evaluator_root)
    projected: list[tuple[str, CanonicalSubmissionV1]] = []
    for witness in witnesses:
        item_root = participant_root / "items" / witness.item_id
        task_path = item_root / "task.json"
        if (
            hashlib.sha256(task_path.read_bytes()).hexdigest()
            != witness.fixed_question_sha256
        ):
            raise ValueError(
                f"independent projection found a stale fixed question: {witness.witness_id}"
            )
        for (
            relative_path,
            expected_checksum,
        ) in witness.participant_input_checksums.items():
            observed = hashlib.sha256(
                (item_root / relative_path).read_bytes()
            ).hexdigest()
            if observed != expected_checksum:
                raise ValueError(
                    f"independent projection found stale public evidence: {witness.witness_id}/{relative_path}"
                )
        raw_payload = witness.submission.model_dump(mode="json", exclude_none=True)
        if _canonical_sha256(raw_payload) != witness.raw_response_sha256:
            raise ValueError(
                f"independent projection found a stale raw response: {witness.witness_id}"
            )
        route = route_by_identity.get((witness.item_id, witness.route_id))
        if route is None:
            raise ValueError(
                f"independent projection found an unknown route witness: {witness.witness_id}"
            )
        contract = _SubmissionContractV1.model_validate_json(
            (item_root / "submission_contract.json").read_text(encoding="utf-8")
        )
        manifest = manifests.get(witness.item_id)
        if manifest is None:
            raise ValueError(
                f"independent projection lacks assumption evidence: {witness.witness_id}"
            )
        diagnostics = _validated_diagnostic_ids(
            witness=witness,
            participant_item_root=item_root,
            participant_root=participant_root,
            contract=contract,
            manifest=manifest,
            required_diagnostics=route.required_diagnostics,
            identification_assumptions=route.required_identification_assumptions,
        )
        primary = witness.submission.primary_analysis
        estimand = primary.estimand
        signature = {
            "analysis_population_id": estimand.population_id,
            "estimand_id": estimand.estimand_id,
            "intercurrent_event_strategy_ids": estimand.intercurrent_event_strategy_ids,
            "assessment_horizon_days": (
                None if estimand.horizon is None else estimand.horizon.value
            ),
            "treatment_id": estimand.treatment_id,
            "comparator_id": estimand.comparator_id,
            "endpoint_id": estimand.endpoint_id,
            "effect_scale": primary.result.effect_scale,
            "analysis_method_id": primary.estimator.analysis_method_id,
        }
        if _canonical_sha256(signature) != witness.route_signature_sha256:
            raise ValueError(
                f"independent projection resolved another route: {witness.witness_id}"
            )
        integrity = (
            None
            if witness.submission.data_integrity_record is None
            else witness.submission.data_integrity_record.model_dump(mode="json")
        )
        canonical = CanonicalSubmissionV1.model_validate(
            {
                "schema_id": "trialagentbench.canonical_submission/v1",
                "item_id": witness.item_id,
                "primary": signature,
                "diagnostic_ids": diagnostics,
                "data_integrity_record": integrity,
                "result": _project_result(primary.result),
            }
        )
        if require_route_evidence and not set(route.required_diagnostics) <= set(
            canonical.diagnostic_ids
        ):
            raise ValueError(
                f"independent projection could not validate route evidence: {witness.witness_id}"
            )
        projected.append((witness.witness_id, canonical))
    identities = tuple(witness_id for witness_id, _ in projected)
    if len(identities) != len(set(identities)):
        raise ValueError("independent raw projection contains duplicate witness IDs")
    return tuple(projected)


__all__ = ["RawTrialEvalRouteWitnessV1", "project_raw_witnesses_independently"]
