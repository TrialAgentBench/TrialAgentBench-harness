"""Generated single-fault stress census for the narrow TrialEval grader."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_validation.grader_concordance import (
    CanonicalSubmissionV1,
    CanonicalTrialEvalRouteWitnessV1,
    CategoricalTargetV1,
    GradeRecordV1,
    NumericIntervalTargetV1,
    NumericPointTargetV1,
    NumericVectorTargetV1,
    RouteV1,
    ScoringKeyV1,
    StatisticalTestTargetV1,
    grade_trialeval_independently,
)
from trialagentbench_validation.raw_projection import (
    RawTrialEvalRouteWitnessV1,
    _precision_preserves_regime,
    _read_assumption_manifests,
    _SubmissionContractV1,
    project_raw_witnesses_independently,
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialEvalMutationWitnessV1(_Contract):
    """One schema-valid single-coordinate mutation."""

    schema_id: Literal["trialagentbench.trialeval_mutation_witness/v1"] = (
        "trialagentbench.trialeval_mutation_witness/v1"
    )
    mutation_id: str
    base_witness_id: str
    item_id: str
    route_id: str
    mutated_coordinate: str
    expected_first_gate: str | None = None
    expected_failure_code: str | None = None
    submission: CanonicalSubmissionV1


class TrialEvalMutationGradeV1(_Contract):
    """One grade bound to its generated mutation."""

    schema_id: Literal["trialagentbench.trialeval_mutation_grade/v1"] = (
        "trialagentbench.trialeval_mutation_grade/v1"
    )
    mutation_id: str
    base_witness_id: str
    item_id: str
    route_id: str
    mutated_coordinate: str
    expected_first_gate: str | None = None
    expected_failure_code: str | None = None
    grade: GradeRecordV1


class RawTrialEvalEvidenceMutationV1(_Contract):
    """One raw-response evidence mutation and its expected grade behavior."""

    schema_id: Literal["trialagentbench.trialeval_raw_evidence_mutation/v1"] = (
        "trialagentbench.trialeval_raw_evidence_mutation/v1"
    )
    mutation_id: str
    base_witness_id: str
    mutated_witness_id: str
    item_id: str
    route_id: str
    mutated_coordinate: str
    removed_required_artifact: str | None = None
    replacement_artifact: str | None = None
    expected_first_gate: Literal["evidence"] = "evidence"
    expected_failure_code: Literal["missing_required_diagnostic"] = (
        "missing_required_diagnostic"
    )

    @model_validator(mode="after")
    def _artifact_replacement_is_complete(self) -> RawTrialEvalEvidenceMutationV1:
        if (self.removed_required_artifact is None) != (
            self.replacement_artifact is None
        ):
            raise ValueError(
                "raw evidence artifact mutation must bind both source paths"
            )
        if (
            self.removed_required_artifact is not None
            and self.removed_required_artifact == self.replacement_artifact
        ):
            raise ValueError(
                "raw evidence artifact mutation must change the cited source"
            )
        return self


class TrialEvalGateApplicabilityV1(_Contract):
    """Whether one ordered grade gate is meaningful for one witness."""

    schema_id: Literal["trialagentbench.trialeval_gate_applicability/v1"] = (
        "trialagentbench.trialeval_gate_applicability/v1"
    )
    base_witness_id: str
    gate_id: Literal[
        "submission",
        "question",
        "route",
        "evidence",
        "integrity",
        "result",
        "conformance",
        "decision",
    ]
    applicable: bool
    reason: str


class TrialEvalGraderStressReportV1(_Contract):
    """Complete generated mutation and gate-coverage result."""

    schema_id: Literal["trialagentbench.trialeval_grader_stress/v1"] = (
        "trialagentbench.trialeval_grader_stress/v1"
    )
    positive_witness_count: int = Field(ge=1)
    canonical_mutation_required_count: int = Field(ge=1)
    raw_evidence_mutation_required_count: int = Field(ge=1)
    required_artifact_removal_count: int = Field(ge=1)
    required_mutation_count: int = Field(ge=1)
    independently_graded_count: int = Field(ge=0)
    public_graded_count: int = Field(ge=0)
    independently_projected_raw_mutation_count: int = Field(ge=0)
    harness_projected_raw_mutation_count: int = Field(ge=0)
    raw_mutation_projection_mismatch_count: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)
    expected_behavior_failure_count: int = Field(ge=0)
    crashed_count: int = Field(ge=0)
    mutation_counts_by_gate: dict[str, int]
    applicable_counts_by_gate: dict[str, int]
    nonapplicable_counts_by_gate: dict[str, int]
    independent_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_command: tuple[str, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def _status_matches_census(self) -> TrialEvalGraderStressReportV1:
        passed = (
            self.required_mutation_count
            == self.canonical_mutation_required_count
            + self.raw_evidence_mutation_required_count
            and 0
            < self.required_artifact_removal_count
            <= self.raw_evidence_mutation_required_count
            and self.raw_evidence_mutation_required_count
            == self.independently_projected_raw_mutation_count
            == self.harness_projected_raw_mutation_count
            and self.required_mutation_count
            == self.independently_graded_count
            == self.public_graded_count
            and self.raw_mutation_projection_mismatch_count
            == self.mismatch_count
            == self.expected_behavior_failure_count
            == self.crashed_count
            == 0
        )
        if (self.status == "pass") != passed:
            raise ValueError("grader-stress status does not match its finite census")
        return self


def _submission_with(
    submission: CanonicalSubmissionV1,
    mutate: Callable[..., object],
) -> CanonicalSubmissionV1:
    payload = copy.deepcopy(submission.model_dump(mode="json", exclude_none=True))
    mutate(payload)
    return CanonicalSubmissionV1.model_validate(payload)


def _mutation(
    *,
    witness: CanonicalTrialEvalRouteWitnessV1,
    coordinate: str,
    gate: str | None,
    code: str | None,
    submission: CanonicalSubmissionV1,
) -> TrialEvalMutationWitnessV1:
    suffix = coordinate.replace(".", "-").replace("[", "-").replace("]", "")
    return TrialEvalMutationWitnessV1(
        mutation_id=f"{witness.witness_id}::{suffix}",
        base_witness_id=witness.witness_id,
        item_id=witness.item_id,
        route_id=witness.route_id,
        mutated_coordinate=coordinate,
        expected_first_gate=gate,
        expected_failure_code=code,
        submission=submission,
    )


def _representable_boundary_value(
    *,
    origin: float,
    tolerance: float,
    boundary: Literal["inside", "on", "outside"],
    direction: float = 1.0,
) -> float:
    """Return the nearest floating-point value on the requested side."""

    candidate = origin + direction * tolerance
    toward = origin
    away = math.inf if direction > 0 else -math.inf
    if boundary in {"inside", "on"}:
        while abs(candidate - origin) > tolerance:
            candidate = math.nextafter(candidate, toward)
    if boundary == "inside":
        while abs(candidate - origin) >= tolerance:
            candidate = math.nextafter(candidate, toward)
    elif boundary == "outside":
        while abs(candidate - origin) <= tolerance:
            candidate = math.nextafter(candidate, away)
    return candidate


def _numeric_boundary_submissions(
    *,
    witness: CanonicalTrialEvalRouteWitnessV1,
    target: (
        NumericPointTargetV1
        | NumericIntervalTargetV1
        | NumericVectorTargetV1
        | StatisticalTestTargetV1
    ),
) -> tuple[TrialEvalMutationWitnessV1, ...]:
    tolerance = target.acceptance_envelope.absolute_tolerance
    rows: list[TrialEvalMutationWitnessV1] = []
    boundary_cases: tuple[
        tuple[Literal["inside", "on", "outside"], str | None, str | None],
        ...,
    ] = (
        ("inside", None, None),
        ("on", None, None),
        (
            "outside",
            "conformance",
            {
                "numeric_point": "numeric_result_outside_tolerance",
                "numeric_interval": "interval_result_outside_tolerance",
                "numeric_vector": "vector_result_outside_tolerance",
                "statistical_test": "test_p_value_outside_tolerance",
            }[target.kind],
        ),
    )
    for boundary, gate, code in boundary_cases:

        if isinstance(target, NumericPointTargetV1):
            value = _representable_boundary_value(
                origin=target.value,
                tolerance=tolerance,
                boundary=boundary,
            )
            submission = _submission_with(
                witness.submission,
                lambda payload, replacement=value: payload["result"].update(
                    {"value": replacement}
                ),
            )
        elif isinstance(target, NumericIntervalTargetV1):
            lower = _representable_boundary_value(
                origin=target.lower,
                tolerance=tolerance,
                boundary=boundary,
            )
            upper = _representable_boundary_value(
                origin=target.upper,
                tolerance=tolerance,
                boundary=boundary,
            )
            submission = _submission_with(
                witness.submission,
                lambda payload, replacement_lower=lower, replacement_upper=upper: payload[
                    "result"
                ].update(
                    {
                        "lower": replacement_lower,
                        "upper": replacement_upper,
                    }
                ),
            )
        elif isinstance(target, NumericVectorTargetV1):
            value = _representable_boundary_value(
                origin=target.components[0].value,
                tolerance=tolerance,
                boundary=boundary,
            )
            submission = _submission_with(
                witness.submission,
                lambda payload, replacement=value: payload["result"]["components"][
                    0
                ].update({"value": replacement}),
            )
        else:
            direction = 1.0 if target.p_value <= 0.5 else -1.0
            value = _representable_boundary_value(
                origin=target.p_value,
                tolerance=tolerance,
                boundary=boundary,
                direction=direction,
            )
            submission = _submission_with(
                witness.submission,
                lambda payload, replacement=value: payload["result"].update(
                    {"p_value": replacement}
                ),
            )
        rows.append(
            _mutation(
                witness=witness,
                coordinate=f"conformance.{boundary}",
                gate=gate,
                code=code,
                submission=submission,
            )
        )
    return tuple(rows)


def generate_trialeval_mutations(
    *,
    witnesses: tuple[CanonicalTrialEvalRouteWitnessV1, ...],
    key_by_item: dict[str, ScoringKeyV1],
) -> tuple[
    tuple[TrialEvalMutationWitnessV1, ...], tuple[TrialEvalGateApplicabilityV1, ...]
]:
    """Derive the complete schema-valid mutation inventory from frozen keys."""

    mutations: list[TrialEvalMutationWitnessV1] = []
    applicability: list[TrialEvalGateApplicabilityV1] = []
    question_coordinates = {
        "primary.analysis_population_id": (
            "analysis_population_id",
            "__mutated_population__",
        ),
        "primary.estimand_id": ("estimand_id", "__mutated_estimand__"),
        "primary.intercurrent_event_strategy_ids": (
            "intercurrent_event_strategy_ids",
            ["__mutated_strategy__"],
        ),
        "primary.assessment_horizon_days": ("assessment_horizon_days", 999999.0),
        "primary.treatment_id": ("treatment_id", "__mutated_treatment__"),
        "primary.comparator_id": ("comparator_id", "__mutated_comparator__"),
        "primary.endpoint_id": ("endpoint_id", "__mutated_endpoint__"),
        "primary.effect_scale": ("effect_scale", "__mutated_effect_scale__"),
    }
    route_coordinates = {
        "primary.analysis_method_id": ("analysis_method_id", "__mutated_method__"),
    }
    for witness in witnesses:
        key = key_by_item[witness.item_id]
        route = next(
            row
            for row in key.credit_eligible_routes
            if row.route_id == witness.route_id
        )
        mutation = _submission_with(
            witness.submission,
            lambda payload: payload.update({"item_id": "__stale_item__"}),
        )
        mutations.append(
            _mutation(
                witness=witness,
                coordinate="submission.item_id",
                gate="submission",
                code="item_mismatch",
                submission=mutation,
            )
        )
        for coordinate, (field, value) in question_coordinates.items():
            mutation = _submission_with(
                witness.submission,
                lambda payload, name=field, replacement=value: payload[
                    "primary"
                ].update({name: replacement}),
            )
            mutations.append(
                _mutation(
                    witness=witness,
                    coordinate=coordinate,
                    gate="question",
                    code="unrecognized_primary_question",
                    submission=mutation,
                )
            )
        for coordinate, (field, value) in route_coordinates.items():
            mutation = _submission_with(
                witness.submission,
                lambda payload, name=field, replacement=value: payload[
                    "primary"
                ].update({name: replacement}),
            )
            mutations.append(
                _mutation(
                    witness=witness,
                    coordinate=coordinate,
                    gate="route",
                    code="unrecognized_primary_route",
                    submission=mutation,
                )
            )
        evidence_applicable = bool(route.required_diagnostics)
        if evidence_applicable:
            mutations.append(
                _mutation(
                    witness=witness,
                    coordinate="evidence.required_diagnostics",
                    gate="evidence",
                    code="missing_required_diagnostic",
                    submission=_submission_with(
                        witness.submission,
                        lambda payload: payload.update({"diagnostic_ids": []}),
                    ),
                )
            )
        integrity_applicable = key.data_integrity_target is not None
        if integrity_applicable:
            mutations.append(
                _mutation(
                    witness=witness,
                    coordinate="integrity.data_integrity_record",
                    gate="integrity",
                    code="missing_data_integrity_record",
                    submission=_submission_with(
                        witness.submission,
                        lambda payload: payload.pop("data_integrity_record", None),
                    ),
                )
            )
        target = route.target
        result_applicable = isinstance(
            target,
            NumericPointTargetV1 | NumericIntervalTargetV1 | NumericVectorTargetV1,
        )
        if result_applicable:
            if (
                isinstance(target, NumericPointTargetV1)
                and target.require_confidence_interval
            ):
                mutations.append(
                    _mutation(
                        witness=witness,
                        coordinate="result.confidence_interval",
                        gate="result",
                        code="missing_confidence_interval",
                        submission=_submission_with(
                            witness.submission,
                            lambda payload: (
                                payload["result"].pop(
                                    "confidence_interval_lower", None
                                ),
                                payload["result"].pop(
                                    "confidence_interval_upper", None
                                ),
                            ),
                        ),
                    )
                )
            else:
                mutations.append(
                    _mutation(
                        witness=witness,
                        coordinate="result.result_unit",
                        gate="result",
                        code="result_unit_mismatch",
                        submission=_submission_with(
                            witness.submission,
                            lambda payload: payload["result"].update(
                                {"result_unit": "__mutated_unit__"}
                            ),
                        ),
                    )
                )
        if isinstance(
            target,
            NumericPointTargetV1
            | NumericIntervalTargetV1
            | NumericVectorTargetV1
            | StatisticalTestTargetV1,
        ):
            mutations.extend(
                _numeric_boundary_submissions(witness=witness, target=target)
            )
        elif isinstance(target, CategoricalTargetV1):
            mutations.append(
                _mutation(
                    witness=witness,
                    coordinate="conformance.categorical_code",
                    gate="conformance",
                    code="categorical_result_not_credit_eligible",
                    submission=_submission_with(
                        witness.submission,
                        lambda payload: payload["result"].update(
                            {"code": "__mutated_code__"}
                        ),
                    ),
                )
            )
        gate_states = {
            "submission": (True, "all witnesses have a typed submission"),
            "question": (
                True,
                "the fixed question has eight independently mutable coordinates",
            ),
            "route": (
                True,
                "the declared method signature has independently mutable coordinates",
            ),
            "evidence": (
                evidence_applicable,
                "applicable exactly when the selected route requires diagnostic or factual evidence",
            ),
            "integrity": (
                integrity_applicable,
                "applicable exactly to C5 items with a frozen repair target",
            ),
            "result": (
                result_applicable,
                "canonical result-shape mutation is applicable to numeric result contracts",
            ),
            "conformance": (
                True,
                "every route has numeric boundaries or categorical membership",
            ),
            "decision": (False, "TrialEval has no programme-decision gate"),
        }
        for gate_id, (applicable, reason) in gate_states.items():
            applicability.append(
                TrialEvalGateApplicabilityV1(
                    base_witness_id=witness.witness_id,
                    gate_id=gate_id,
                    applicable=applicable,
                    reason=reason,
                )
            )
    mutation_ids = tuple(row.mutation_id for row in mutations)
    if len(mutation_ids) != len(set(mutation_ids)):
        raise ValueError("generated TrialEval mutation IDs are not unique")
    return tuple(mutations), tuple(applicability)


def _canonical_sha256(payload: object) -> str:
    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _raw_evidence_mutation(
    *,
    witness: RawTrialEvalRouteWitnessV1,
    coordinate: str,
    mutate: Callable[..., object],
    removed_required_artifact: str | None = None,
    replacement_artifact: str | None = None,
) -> tuple[RawTrialEvalRouteWitnessV1, RawTrialEvalEvidenceMutationV1]:
    payload = copy.deepcopy(witness.model_dump(mode="json", exclude_none=True))
    submission = payload["submission"]
    if not isinstance(submission, dict):
        raise ValueError("raw TrialEval witness lost its typed submission")
    mutate(submission)
    suffix = coordinate.replace(".", "-").replace("[", "-").replace("]", "")
    mutation_id = f"{witness.witness_id}::raw-{suffix}"
    payload["witness_id"] = mutation_id
    payload["raw_response_sha256"] = _canonical_sha256(submission)
    mutated = RawTrialEvalRouteWitnessV1.model_validate(payload)
    return (
        mutated,
        RawTrialEvalEvidenceMutationV1(
            mutation_id=mutation_id,
            base_witness_id=witness.witness_id,
            mutated_witness_id=mutation_id,
            item_id=witness.item_id,
            route_id=witness.route_id,
            mutated_coordinate=coordinate,
            removed_required_artifact=removed_required_artifact,
            replacement_artifact=replacement_artifact,
        ),
    )


def generate_raw_evidence_mutations(
    *,
    raw_witnesses_path: Path,
    participant_root: Path,
    evaluator_root: Path,
    route_by_identity: Mapping[tuple[str, str], RouteV1],
) -> tuple[
    tuple[RawTrialEvalRouteWitnessV1, ...],
    tuple[RawTrialEvalEvidenceMutationV1, ...],
]:
    """Mutate every score-bearing raw evidence coordinate that is applicable."""

    raw_witnesses = tuple(
        RawTrialEvalRouteWitnessV1.model_validate_json(line)
        for line in Path(raw_witnesses_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    mutated_witnesses: list[RawTrialEvalRouteWitnessV1] = []
    cases: list[RawTrialEvalEvidenceMutationV1] = []
    assumption_manifests = _read_assumption_manifests(evaluator_root)

    def add(
        *,
        witness: RawTrialEvalRouteWitnessV1,
        coordinate: str,
        mutate: Callable[..., object],
        removed_required_artifact: str | None = None,
        replacement_artifact: str | None = None,
    ) -> None:
        mutated, case = _raw_evidence_mutation(
            witness=witness,
            coordinate=coordinate,
            mutate=mutate,
            removed_required_artifact=removed_required_artifact,
            replacement_artifact=replacement_artifact,
        )
        mutated_witnesses.append(mutated)
        cases.append(case)

    for witness in raw_witnesses:
        route = route_by_identity.get((witness.item_id, witness.route_id))
        required_diagnostics = tuple(getattr(route, "required_diagnostics", ()))
        if not required_diagnostics:
            continue
        contract = _SubmissionContractV1.model_validate_json(
            (
                participant_root
                / "items"
                / witness.item_id
                / "submission_contract.json"
            ).read_text(encoding="utf-8")
        )
        obligation_by_diagnostic = {
            obligation.diagnostic_id: obligation
            for obligation in contract.diagnostic_obligations
        }
        assumption_manifest = assumption_manifests.get(witness.item_id)
        if assumption_manifest is None:
            raise ValueError(
                f"raw stress witness lacks assumption evidence: {witness.witness_id}"
            )
        assumption_by_id = {
            record.assumption_id: record for record in assumption_manifest.records
        }
        linked_ids = set(witness.submission.primary_analysis.evidence_ids)
        matched_diagnostics: set[str] = set()
        for evidence_index, evidence in enumerate(witness.submission.evidence):
            if (
                evidence.diagnostic_id not in required_diagnostics
                or evidence.evidence_id not in linked_ids
            ):
                continue
            diagnostic_id = str(evidence.diagnostic_id)
            matched_diagnostics.add(diagnostic_id)
            replacement_diagnostic = (
                "cluster_structure_public"
                if diagnostic_id != "cluster_structure_public"
                else "randomization_integrity_public"
            )
            add(
                witness=witness,
                coordinate=f"evidence[{evidence_index}].diagnostic_id",
                mutate=lambda submission, row=evidence_index, replacement=replacement_diagnostic: submission[
                    "evidence"
                ][
                    row
                ].update(
                    {
                        "diagnostic_id": replacement,
                    }
                ),
            )
            source_mutation_count = 0
            for source_index, source in enumerate(evidence.source_artifacts):
                replacement_source = next(
                    (
                        path
                        for path in sorted(witness.participant_input_checksums)
                        if path != source
                    ),
                    None,
                )
                if replacement_source is None:
                    raise ValueError(
                        "raw stress witness cannot isolate a required evidence artifact: "
                        f"{witness.witness_id}/{source}"
                    )
                add(
                    witness=witness,
                    coordinate=(
                        f"evidence[{evidence_index}]."
                        f"source_artifacts[{source_index}]"
                    ),
                    mutate=lambda submission, row=evidence_index, index=source_index, replacement=replacement_source: submission[
                        "evidence"
                    ][
                        row
                    ][
                        "source_artifacts"
                    ].__setitem__(
                        index,
                        replacement,
                    ),
                    removed_required_artifact=source,
                    replacement_artifact=replacement_source,
                )
                source_mutation_count += 1
            if source_mutation_count != len(evidence.source_artifacts):
                raise ValueError(
                    "raw stress witness lacks a necessity mutation for every cited source: "
                    f"{witness.witness_id}/{evidence.evidence_id}"
                )
            result = evidence.result
            if result.kind == "factual_premise":
                add(
                    witness=witness,
                    coordinate=f"evidence[{evidence_index}].result.conclusion",
                    mutate=lambda submission, row=evidence_index: submission[
                        "evidence"
                    ][row]["result"].update({"conclusion": "not_supported"}),
                )
                continue
            if result.kind != "diagnostic_summary":
                continue
            obligation = obligation_by_diagnostic.get(diagnostic_id)
            required_metric_id = (
                None if obligation is None else obligation.score_bearing_metric_id
            )
            for measure_index, measure in enumerate(result.measures):
                if measure.metric_id != required_metric_id:
                    continue
                measure_prefix = (
                    f"evidence[{evidence_index}].result.measures[{measure_index}]"
                )
                add(
                    witness=witness,
                    coordinate=f"{measure_prefix}.metric_id",
                    mutate=lambda submission, row=evidence_index, index=measure_index: submission[
                        "evidence"
                    ][
                        row
                    ][
                        "result"
                    ][
                        "measures"
                    ][
                        index
                    ].update(
                        {"metric_id": "wrong_metric_id"}
                    ),
                )
                add(
                    witness=witness,
                    coordinate=f"{measure_prefix}.unit",
                    mutate=lambda submission, row=evidence_index, index=measure_index: submission[
                        "evidence"
                    ][
                        row
                    ][
                        "result"
                    ][
                        "measures"
                    ][
                        index
                    ].update(
                        {"unit": "__wrong_unit__"}
                    ),
                )
                step = 10.0 ** (-measure.decimal_places)
                add(
                    witness=witness,
                    coordinate=f"{measure_prefix}.value",
                    mutate=lambda submission, row=evidence_index, index=measure_index, replacement=measure.value + step: submission[
                        "evidence"
                    ][
                        row
                    ][
                        "result"
                    ][
                        "measures"
                    ][
                        index
                    ].update(
                        {"value": replacement}
                    ),
                )
                assumption = (
                    None
                    if obligation is None
                    else assumption_by_id.get(obligation.assumption_id)
                )
                thresholds = (
                    None
                    if assumption is None
                    else (
                        assumption.threshold_stressed,
                        assumption.threshold_fragile,
                        assumption.threshold_broken,
                    )
                )
                ambiguous_precision = None
                if thresholds is not None:
                    stressed, fragile, broken = thresholds
                else:
                    stressed = fragile = broken = None
                if stressed is not None and fragile is not None and broken is not None:
                    numeric_thresholds = (
                        float(stressed),
                        float(fragile),
                        float(broken),
                    )
                    ambiguous_precision = next(
                        (
                            decimal_places
                            for decimal_places in range(measure.decimal_places)
                            if not _precision_preserves_regime(
                                measure.model_copy(
                                    update={"decimal_places": decimal_places}
                                ),
                                thresholds=numeric_thresholds,
                            )
                        ),
                        None,
                    )
                if ambiguous_precision is not None:
                    add(
                        witness=witness,
                        coordinate=f"{measure_prefix}.decimal_places",
                        mutate=lambda submission, row=evidence_index, index=measure_index, replacement=ambiguous_precision: submission[
                            "evidence"
                        ][
                            row
                        ][
                            "result"
                        ][
                            "measures"
                        ][
                            index
                        ].update(
                            {"decimal_places": replacement}
                        ),
                    )
        if matched_diagnostics != set(required_diagnostics):
            raise ValueError(
                "raw stress witness does not contain every required diagnostic: "
                f"{witness.witness_id}; missing="
                f"{sorted(set(required_diagnostics) - matched_diagnostics)!r}"
            )
    mutation_ids = tuple(case.mutation_id for case in cases)
    if not mutation_ids:
        raise ValueError(
            "TrialEval release has no score-bearing raw evidence coordinate to stress"
        )
    if len(mutation_ids) != len(set(mutation_ids)):
        raise ValueError("generated raw evidence mutation IDs are not unique")
    return tuple(mutated_witnesses), tuple(cases)


def _write_jsonl(path: Path, rows: Sequence[BaseModel]) -> str:
    body = "".join(
        json.dumps(
            row.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _expected_behavior_matches(row: TrialEvalMutationGradeV1) -> bool:
    grade = row.grade
    if row.expected_first_gate is None:
        return grade.passed and grade.first_failure_gate is None
    if grade.first_failure_gate != row.expected_first_gate:
        return False
    if grade.failure_codes != (row.expected_failure_code,):
        return False
    failure_index = next(
        index
        for index, gate in enumerate(grade.gates)
        if gate.gate_id == row.expected_first_gate
    )
    return all(
        gate.status == "not_reached" for gate in grade.gates[failure_index + 1 :]
    )


def run_trialeval_grader_stress(
    *,
    witnesses: tuple[CanonicalTrialEvalRouteWitnessV1, ...],
    key_by_item: dict[str, ScoringKeyV1],
    raw_witnesses_path: Path,
    participant_root: Path,
    route_by_identity: Mapping[tuple[str, str], RouteV1],
    evaluator_root: Path,
    output_dir: Path,
    harness_executable: str,
) -> TrialEvalGraderStressReportV1:
    """Generate, independently grade, publicly grade, and reconcile mutations."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    mutations, applicability = generate_trialeval_mutations(
        witnesses=witnesses,
        key_by_item=key_by_item,
    )
    raw_witnesses, raw_cases = generate_raw_evidence_mutations(
        raw_witnesses_path=raw_witnesses_path,
        participant_root=participant_root,
        evaluator_root=evaluator_root,
        route_by_identity=route_by_identity,
    )
    required_artifact_removal_count = sum(
        case.removed_required_artifact is not None for case in raw_cases
    )
    raw_witness_path = output / "trialeval_raw_evidence_mutations.jsonl"
    _write_jsonl(raw_witness_path, raw_witnesses)
    _write_jsonl(output / "trialeval_raw_evidence_mutation_cases.jsonl", raw_cases)
    independent_raw_pairs = project_raw_witnesses_independently(
        raw_witnesses_path=raw_witness_path,
        participant_root=participant_root,
        evaluator_root=evaluator_root,
        route_by_identity=route_by_identity,
        require_route_evidence=False,
    )
    independent_raw_by_id = dict(independent_raw_pairs)
    raw_by_id = {witness.witness_id: witness for witness in raw_witnesses}
    case_by_id = {case.mutated_witness_id: case for case in raw_cases}
    independent_raw_projected = tuple(
        CanonicalTrialEvalRouteWitnessV1(
            witness_id=witness_id,
            item_id=case_by_id[witness_id].item_id,
            route_id=case_by_id[witness_id].route_id,
            context_tier=raw_by_id[witness_id].context_tier,
            submission=submission,
        )
        for witness_id, submission in independent_raw_pairs
    )
    _write_jsonl(
        output / "independent_raw_evidence_mutation_projections.jsonl",
        independent_raw_projected,
    )
    public_raw_projection_path = (
        output / "public_raw_evidence_mutation_projections.jsonl"
    )
    raw_projection_command = (
        harness_executable,
        "grade",
        "project-trialeval-witnesses",
        "--participant-root",
        str(participant_root),
        "--evaluator-root",
        str(evaluator_root),
        "--witnesses",
        str(raw_witness_path),
        "--output",
        str(public_raw_projection_path),
    )
    raw_projection_completed = subprocess.run(
        raw_projection_command,
        check=False,
        text=True,
        capture_output=True,
    )
    raw_projection_crashed = int(raw_projection_completed.returncode != 0)
    public_raw_projected = (
        ()
        if raw_projection_crashed
        else tuple(
            CanonicalTrialEvalRouteWitnessV1.model_validate_json(line)
            for line in public_raw_projection_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    )
    public_raw_by_id = {witness.witness_id: witness for witness in public_raw_projected}
    independent_raw_projected_by_id = {
        witness.witness_id: witness for witness in independent_raw_projected
    }
    raw_projection_mismatches = sum(
        public_raw_by_id.get(case.mutated_witness_id)
        != independent_raw_projected_by_id.get(case.mutated_witness_id)
        for case in raw_cases
    )
    raw_grade_mutations = tuple(
        TrialEvalMutationWitnessV1(
            mutation_id=case.mutation_id,
            base_witness_id=case.base_witness_id,
            item_id=case.item_id,
            route_id=case.route_id,
            mutated_coordinate=case.mutated_coordinate,
            expected_first_gate=case.expected_first_gate,
            expected_failure_code=case.expected_failure_code,
            submission=independent_raw_by_id[case.mutated_witness_id],
        )
        for case in raw_cases
    )
    all_mutations = (*mutations, *raw_grade_mutations)
    mutation_path = output / "trialeval_mutations.jsonl"
    _write_jsonl(mutation_path, all_mutations)
    _write_jsonl(output / "trialeval_gate_applicability.jsonl", applicability)
    independent = tuple(
        TrialEvalMutationGradeV1(
            mutation_id=row.mutation_id,
            base_witness_id=row.base_witness_id,
            item_id=row.item_id,
            route_id=row.route_id,
            mutated_coordinate=row.mutated_coordinate,
            expected_first_gate=row.expected_first_gate,
            expected_failure_code=row.expected_failure_code,
            grade=grade_trialeval_independently(
                key_by_item[row.item_id], row.submission
            ),
        )
        for row in all_mutations
    )
    independent_path = output / "independent_mutation_grades.jsonl"
    independent_sha256 = _write_jsonl(independent_path, independent)
    public_path = output / "public_mutation_grades.jsonl"
    command = (
        harness_executable,
        "grade",
        "canonical-trialeval-mutations",
        "--evaluator-root",
        str(evaluator_root),
        "--mutations",
        str(mutation_path),
        "--output",
        str(public_path),
    )
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    crashed = raw_projection_crashed + int(completed.returncode != 0)
    public = (
        ()
        if completed.returncode != 0
        else tuple(
            TrialEvalMutationGradeV1.model_validate_json(line)
            for line in public_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )
    public_sha256 = hashlib.sha256(
        public_path.read_bytes() if public_path.is_file() else b""
    ).hexdigest()
    independent_by_id = {row.mutation_id: row for row in independent}
    public_by_id = {row.mutation_id: row for row in public}
    mismatches = sum(
        public_by_id.get(row.mutation_id) != independent_by_id[row.mutation_id]
        for row in all_mutations
    )
    behavior_failures = sum(not _expected_behavior_matches(row) for row in independent)
    mutation_counts = Counter(
        row.expected_first_gate or "inclusive_boundary_pass" for row in all_mutations
    )
    applicable_counts = Counter(row.gate_id for row in applicability if row.applicable)
    nonapplicable_counts = Counter(
        row.gate_id for row in applicability if not row.applicable
    )
    report = TrialEvalGraderStressReportV1(
        positive_witness_count=len(witnesses),
        canonical_mutation_required_count=len(mutations),
        raw_evidence_mutation_required_count=len(raw_cases),
        required_artifact_removal_count=required_artifact_removal_count,
        required_mutation_count=len(all_mutations),
        independently_graded_count=len(independent),
        public_graded_count=len(public),
        independently_projected_raw_mutation_count=len(independent_raw_projected),
        harness_projected_raw_mutation_count=len(public_raw_projected),
        raw_mutation_projection_mismatch_count=raw_projection_mismatches,
        mismatch_count=mismatches,
        expected_behavior_failure_count=behavior_failures,
        crashed_count=crashed,
        mutation_counts_by_gate=dict(sorted(mutation_counts.items())),
        applicable_counts_by_gate=dict(sorted(applicable_counts.items())),
        nonapplicable_counts_by_gate=dict(sorted(nonapplicable_counts.items())),
        independent_sha256=independent_sha256,
        public_sha256=public_sha256,
        public_command=(*raw_projection_command, "&&", *command),
        status=(
            "pass"
            if not crashed
            and not raw_projection_mismatches
            and not mismatches
            and not behavior_failures
            and len(independent) == len(public) == len(all_mutations)
            and len(independent_raw_projected)
            == len(public_raw_projected)
            == len(raw_cases)
            else "fail"
        ),
    )
    (output / "trialeval_grader_stress_report.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "TrialEvalGateApplicabilityV1",
    "TrialEvalGraderStressReportV1",
    "TrialEvalMutationGradeV1",
    "TrialEvalMutationWitnessV1",
    "generate_trialeval_mutations",
    "run_trialeval_grader_stress",
]
