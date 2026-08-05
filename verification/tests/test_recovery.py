"""End-to-end public-role recovery tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from pydantic import ValidationError

from trialagentbench_validation import recovery
from trialagentbench_validation.contracts.component_evidence import (
    TrialEvalComponentEvidenceInventoryV1,
)
from trialagentbench_validation.contracts.release_scope import (
    TrialEvalCanonicalComponentInventoryV1,
    TrialEvalReleaseScopeV1,
)
from trialagentbench_validation.contracts.route_replay import (
    PublicRouteReplayEvidenceV1,
    PublicRouteReplayRecordV1,
)
from trialagentbench_validation.contracts.scientific_sources import (
    ScientificSourceRegistryV1,
)
from trialagentbench_validation.contracts.scoring.route_reference_inputs import (
    RouteReferenceInputRecordV1,
    RouteReferenceInputTableRefV1,
)
from trialagentbench_validation.contracts.scoring.route_references import (
    RouteReferenceRecordV1,
    float64_equivalence_policy_v1,
)
from trialagentbench_validation.recovery import (
    RecoverabilityRouteV1,
    recover_release,
    write_recoverability_report,
)
from trialagentbench_validation.trialeval.references.numeric import (
    PublicEvidenceNumericReferenceCheckV1,
    PublicEvidenceNumericReferenceReportV1,
)


def _json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _record_checksum(payload: dict[str, object]) -> str:
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "checksum" and value is not None
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _checked_payload(payload: dict[str, object]) -> dict[str, object]:
    checked = dict(payload)
    checked["checksum"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return checked


def _public_component_fixture(
    *,
    release_id: str,
    source_ids: tuple[str, ...] = ("TAB-SRC-001", "TAB-SRC-011"),
    registry_source_ids: tuple[str, ...] | None = None,
    evidence_release_id: str | None = None,
) -> tuple[
    TrialEvalCanonicalComponentInventoryV1,
    TrialEvalReleaseScopeV1,
    ScientificSourceRegistryV1,
    TrialEvalComponentEvidenceInventoryV1,
]:
    tiers_by_series = {
        "TE-S01": ("A1", "A2", "A3"),
        "TE-S02": ("A1", "A2", "A3"),
        "TE-S03": ("A1", "A2"),
        "TE-S04": ("A1", "A2", "A3", "A4"),
        "TE-S05": ("A1", "A2", "A3"),
        "TE-S06": ("A1", "A2", "A3", "A4"),
        "TE-S07": ("A1", "A2", "A3"),
        "TE-S08": ("A1", "A2"),
        "TE-S09": ("A4",),
    }
    route_ids = tuple(f"method-{index:02d}" for index in range(1, 18))
    series_payloads: list[dict[str, object]] = []
    cell_payloads: list[dict[str, object]] = []
    evidence_series: list[dict[str, object]] = []
    for series_index, (series_id, tiers) in enumerate(tiers_by_series.items(), start=1):
        design_profile_id = f"TE-DP{min(series_index, 7):02d}"
        series_payloads.append(
            {
                "evaluation_series_id": series_id,
                "design_profile_id": design_profile_id,
                "label": f"Fixture series {series_index}",
                "public_question": "Estimate the declared treatment contrast.",
                "estimand_id": f"estimand-{series_index:02d}",
                "target_population": "all_randomized_participants",
                "endpoint": "death_by_tau",
                "contrast_direction": "treated_minus_control",
                "intercurrent_event_bindings": [],
                "competing_event_handling_id": "no_competing_event_component_in_primary_endpoint",
                "censoring_handling_id": "administrative_and_recorded_loss_to_follow_up_censoring",
                "missing_observation_handling_id": "no_endpoint_imputation",
                "safety_handling_id": "safety_reported_separately_from_primary_efficacy",
                "default_route_id": "method-01",
                "assumption_tiers": list(tiers),
                "source_ids": list(source_ids),
            }
        )
        evidence_cells: list[dict[str, object]] = []
        for tier in tiers:
            cell_id = f"{series_id}-{tier}"
            eligible_routes = route_ids if cell_id == "TE-S01-A1" else ("method-01",)
            cell_payloads.append(
                {
                    "regime_cell_id": cell_id,
                    "evaluation_series_id": series_id,
                    "assumption_tier": tier,
                    "controlled_condition": "Synthetic contract fixture.",
                    "default_response": "compatible",
                    "identification_class": "point_identified",
                    "eligible_route_ids": list(eligible_routes),
                    "excluded_default_route_id": None,
                    "excluded_default_reason": None,
                    "required_response": "Use an eligible analysis route.",
                    "context_ids": ["C1", "C2", "C3", "C4", "C5"],
                    "base_trial_replicates": 4,
                }
            )
            evidence_cells.append(
                {
                    "regime_cell_id": cell_id,
                    "eligible_route_ids": list(eligible_routes),
                    "excluded_default_route_id": None,
                    "excluded_default_reason": None,
                    "source_ids": list(source_ids),
                }
            )
        evidence_series.append(
            {
                "evaluation_series_id": series_id,
                "design_profile_id": design_profile_id,
                "public_question": "Estimate the declared treatment contrast.",
                "estimand_id": f"estimand-{series_index:02d}",
                "source_applications": [
                    {
                        "source_id": source_id,
                        "exact_locator": f"https://example.org/{source_id.lower()}",
                    }
                    for source_id in source_ids
                ],
                "cells": evidence_cells,
            }
        )
    component_payload = _checked_payload(
        {
            "schema_id": "trialagentbench.trialeval.canonical_components/v1",
            "statistical_methods_contract": "trialagentbench.trialeval.statistical_methods/v1",
            "design_profiles": [
                {
                    "design_profile_id": f"TE-DP{index:02d}",
                    "design_tier": ("D1", "D2", "D3", "D4")[min(index - 1, 3)],
                    "study_format": "parallel",
                    "allocation_unit": "participant",
                    "intervention_model": "parallel_group",
                    "analysis_defining_subdesign": "fixed",
                    "evaluation_series_ids": ["TE-S01"],
                }
                for index in range(1, 8)
            ],
            "evaluation_series": series_payloads,
            "regime_cells": cell_payloads,
            "route_archetypes": [
                {
                    "route_id": route_id,
                    "label": f"Fixture method {index}",
                    "eligible_series_ids": ["TE-S01"],
                    "estimator_family": "fixture",
                    "result_kind": "point_with_interval",
                    "effect_scale": "death_risk_difference_tau",
                    "uncertainty_method": "model_based",
                    "design_obligations": [],
                    "diagnostic_requirements": [],
                    "implementation_summary": "Synthetic contract fixture.",
                    "interpretation": "Exercises the public schema.",
                }
                for index, route_id in enumerate(route_ids, start=1)
            ],
            "base_trial_count": 100,
            "participant_context_item_count": 500,
        }
    )
    components = TrialEvalCanonicalComponentInventoryV1.model_validate(
        component_payload
    )
    scope = TrialEvalReleaseScopeV1.model_validate(
        _checked_payload(
            {
                "schema_id": "trialagentbench.trialeval.release_scope/v1",
                "release_id": release_id,
                "components": components.model_dump(mode="json"),
            }
        )
    )
    registered_source_ids = (
        source_ids if registry_source_ids is None else registry_source_ids
    )
    source_payload = _checked_payload(
        {
            "schema_id": "trialagentbench.scientific_source_registry/v1",
            "sources": [
                {
                    "source_id": source_id,
                    "title": f"Fixture source {source_id}",
                    "source_type": "journal_article",
                    "evidence_role": "methods_evidence",
                    "canonical_id": source_id,
                    "canonical_url": f"https://example.org/{source_id.lower()}",
                    "access_class": "open_access",
                    "verification_status": "verified",
                    "scope_note": "Synthetic contract fixture.",
                }
                for source_id in registered_source_ids
            ],
        }
    )
    sources = ScientificSourceRegistryV1.model_validate(source_payload)
    evidence = TrialEvalComponentEvidenceInventoryV1.model_validate(
        _checked_payload(
            {
                "schema_id": "trialagentbench.trialeval_component_evidence/v1",
                "release_id": evidence_release_id or release_id,
                "component_inventory_checksum": components.checksum,
                "source_registry_checksum": sources.checksum,
                "evaluation_series": evidence_series,
            }
        )
    )
    return components, scope, sources, evidence


def _allow_minimal_route_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep route tests focused while full-census validation is tested separately."""

    monkeypatch.setattr(recovery, "validate_release_scope", lambda **_: None)
    monkeypatch.setattr(recovery, "validate_route_component_evidence", lambda **_: None)


def test_qualified_replay_reuse_validates_packaged_route_and_table_bytes(
    tmp_path: Path,
) -> None:
    participant = tmp_path / "participant.zip"
    table_bytes = b"public analysis table"
    table_path = "items/TASK1/data/ADSL.parquet"
    with ZipFile(participant, "w") as archive:
        archive.writestr(table_path, table_bytes)

    route_payload = RouteReferenceRecordV1(
        schema_id="trialagentbench.trialeval.route_reference/v1",
        task_id="TASK1",
        item_id="ITEM1",
        lane_id="primary_numeric.v1",
        route_reference_id="reference-1",
        variant_role="required_primary",
        route_family="risk_difference",
        estimator_method_id="observed:km",
        effect_scale="risk_difference_tau",
        answer_shape="point",
        value=-0.1,
        standard_error=0.01,
        public_evidence_basis=(table_path,),
        required_modifiers=(),
        identification_class="point_identified",
        support_status="official_supported",
        support_rationale="Public table replay.",
        numerical_equivalence=float64_equivalence_policy_v1(),
    ).model_dump(mode="json", exclude_none=True)
    route_payload["checksum"] = _record_checksum(route_payload)
    route = RouteReferenceRecordV1.model_validate(route_payload)
    input_payload = RouteReferenceInputRecordV1(
        schema_id="trialagentbench.trialeval.route_reference_input/v1",
        task_id="TASK1",
        input_bundle_id="input-1",
        estimator_method_id="observed:km",
        effect_scale="risk_difference_tau",
        lane_ids=("primary_numeric.v1",),
        route_reference_ids=("reference-1",),
        required_table_refs=(
            RouteReferenceInputTableRefV1(
                rel_path=table_path,
                semantic_role="analysis_subject_level",
                sha256=hashlib.sha256(table_bytes).hexdigest(),
                row_count=1,
                column_names=("USUBJID",),
            ),
        ),
        source_role="public_surface_mirror",
    ).model_dump(mode="json", exclude_none=True)
    input_payload["checksum"] = _record_checksum(input_payload)
    reference_input = RouteReferenceInputRecordV1.model_validate(input_payload)
    evidence = PublicRouteReplayEvidenceV1(
        evaluator_sha256="a" * 64,
        participant_sha256="b" * 64,
        records=(
            PublicRouteReplayRecordV1(
                route_reference_id=route.route_reference_id,
                route_reference_checksum=str(route.checksum),
                input_bundle_id=reference_input.input_bundle_id,
                input_bundle_checksum=str(reference_input.checksum),
                max_abs_difference=2e-12,
            ),
        ),
    )
    verification = tmp_path / "verification.zip"
    with ZipFile(verification, "w") as archive:
        archive.writestr(
            "grader/domains/route_references.jsonl",
            route.model_dump_json(exclude_none=True) + "\n",
        )
        archive.writestr(
            "grader/domains/route_reference_inputs.jsonl",
            reference_input.model_dump_json(exclude_none=True) + "\n",
        )
        archive.writestr(
            "verification/public_route_replay_evidence.json",
            evidence.model_dump_json(exclude_none=True),
        )

    replays, parameterized = recovery._validated_qualified_replays(
        verification_release=verification,
        participant_release=participant,
    )

    assert replays["reference-1"].difference == pytest.approx(2e-12)
    assert replays["public-replay:input-1"] == replays["reference-1"]
    assert parameterized == {}


def _write_trialeval_roles(
    tmp_path: Path,
    *,
    independent_difference: float,
    categorical_code: str | None = None,
    replay_code: str | None = None,
    inventory_effect_scale: str = "log_hr",
    item_base_case_id: str = "TE-S01-A1",
    missing_registry_source_ids: tuple[str, ...] = (),
    phase_release_id: str = "release-1",
    phase_card_source_ids: tuple[str, ...] = ("TAB-SRC-001", "TAB-SRC-011"),
    context_tier: str = "C1",
) -> tuple[Path, Path, Path]:
    participant = tmp_path / "participant.zip"
    with ZipFile(participant, "w") as archive:
        archive.writestr("items/TASK1/task.json", "{}")

    verification = tmp_path / "verification.zip"
    item_index = {
        "schema_id": "trialagentbench.item_index/v1",
        "version": "v1",
        "checksum": "a" * 64,
        "entries": [
            {
                "task_id": "TASK1",
                "item_id": "ITEM1",
                "generation_seed": 101,
                "base_case_id": item_base_case_id,
                "variant_id": "VARIANT1",
                "factors": {
                    "design_archetype": "D1",
                    "design_subtype": "parallel_group",
                    "assumption_regime": "A1",
                    "context_configuration": context_tier,
                    "data_preparation": "analysis_ready",
                    "analysis_specification": "locked_sap",
                    "procedure_assistance": "output_contract_only",
                    "response_interface": "structured",
                    "regime_cell_id": item_base_case_id,
                    "evaluation_series_id": item_base_case_id.rsplit("-", 1)[0],
                },
                "scoring_row_offset": 0,
                "scoring_row_count": 1,
                "reconstruction_row_offset": 0,
                "reconstruction_row_count": 0,
                "data_integrity_reference_row_offset": 0,
                "data_integrity_reference_row_count": 0,
            }
        ],
    }
    with ZipFile(verification, "w") as archive:
        archive.writestr("grader/item_index.json", _json(item_index))
        if replay_code is not None:
            replay = {
                "schema_id": "trialagentbench.validation.non_numeric_replay/v1",
                "item_id": "TASK1",
                "route_id": "route-1",
                "result_kind": "limitation",
                "reproduced_codes": [replay_code],
                "conformance_rule": "categorical_code_membership",
                "algorithm_id": "public-identification-rule-v1",
                "participant_release_sha256": hashlib.sha256(
                    participant.read_bytes()
                ).hexdigest(),
                "participant_input_checksums": {
                    "items/TASK1/task.json": hashlib.sha256(b"{}").hexdigest(),
                },
            }
            archive.writestr(
                "trialeval/non_numeric_replay.jsonl", _json(replay) + b"\n"
            )

    analysis_method_id = (
        "qualified_nonidentification"
        if categorical_code is not None
        else "cox_ph_model_based"
    )
    route = {
        "route_id": "route-1",
        "signature": {
            "analysis_population_id": "itt",
            "estimand_id": "primary",
            "intercurrent_event_strategy_ids": ["death:treatment_policy"],
            "assessment_horizon_days": 365,
            "treatment_id": "treated",
            "comparator_id": "control",
            "endpoint_id": "death",
            "effect_scale": "log_hr",
            "analysis_method_id": analysis_method_id,
        },
        "method": {
            "analysis_method_id": analysis_method_id,
            "estimator_family": "cox_ph",
            "result_kind": (
                "limitation" if categorical_code is not None else "numeric_point"
            ),
            "uncertainty_method": (
                "not_applicable" if categorical_code is not None else "model_based"
            ),
            "sensitivity_parameters": [],
            "design_modifiers": [],
        },
        "required_identification_assumptions": ["randomization"],
        "required_diagnostics": [],
        "target": (
            {"kind": "categorical", "credit_eligible_codes": [categorical_code]}
            if categorical_code is not None
            else {
                "kind": "numeric_point",
                "value": -0.5,
                "result_unit": "log_hr",
                "require_confidence_interval": True,
                "confidence_interval_lower": -0.7,
                "confidence_interval_upper": -0.3,
                "acceptance_envelope": {
                    "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
                    "reporting_decimal_places": 3,
                    "independent_max_abs_difference": independent_difference,
                    "public_verification_id": "reference-1",
                    "independent_verification_ids": ["independent-replay"],
                },
            }
        ),
    }
    key = {
        "schema_id": "trialagentbench.scoring_key/v1",
        "release_id": "release-1",
        "item_id": "TASK1",
        "question_id": "question-1",
        "context_tier": context_tier,
        "credit_eligible_routes": [route],
    }
    body = _json(key) + b"\n"
    manifest = {
        "schema_id": "trialagentbench.scoring_key_manifest/v1",
        "release_id": "release-1",
        "specification_sha256": "b" * 64,
        "scoring_keys_sha256": hashlib.sha256(body).hexdigest(),
        "item_ids": ["TASK1"],
    }
    scientific_row = {
        "generation_unit_id": "world-1",
        "generation_seed": 101,
        "item_id": "TASK1",
        "question_id": "question-1",
        "route_id": "route-1",
        "design_tier": "D1",
        "design_subtype": "parallel_group",
        "assumption_tier": "A1",
        "context_tier": context_tier,
        "objective": "estimation",
        "analysis_role": "main",
        "scoring_role": "main_credit",
        "target_population_id": "itt",
        "analysis_population_id": "itt",
        "treatment_id": "treated",
        "comparator_id": "control",
        "endpoint_id": "death",
        "estimand_id": "primary",
        "assessment_horizon_days": 365.0,
        "time_origin_id": "randomization",
        "intercurrent_event_bindings": ["death:treatment_policy"],
        "safety_handling_id": "safety_endpoints_reported_separately_from_the_primary_efficacy_estimand",
        "competing_event_handling_id": "no_competing_event_component_in_declared_primary_endpoint",
        "censoring_handling_id": "unweighted_time_to_event_under_declared_censoring_assumption",
        "missing_observation_handling_id": (
            "observed_event_time_and_censoring_indicator_without_endpoint_imputation"
        ),
        "interpretation_constraints": ["Interpret only as the declared estimand."],
        "identification_class": "point_identified",
        "identification_assumptions": ["randomization"],
        "analysis_method_id": analysis_method_id,
        "estimator_family": "cox_ph",
        "effect_scale": inventory_effect_scale,
        "result_kind": (
            "limitation" if categorical_code is not None else "numeric_point"
        ),
        "uncertainty_method": (
            "not_applicable" if categorical_code is not None else "model_based"
        ),
        "sensitivity_parameters": [],
        "design_obligations": [],
        "required_diagnostics": [],
        "participant_evidence_paths": ["items/TASK1/task.json"],
        "evaluator_reference_kind": (
            "categorical" if categorical_code is not None else "numeric_point"
        ),
        "comparison_rule": (
            "categorical_code_membership"
            if categorical_code is not None
            else "numeric_envelope"
        ),
        "reporting_decimal_places": None if categorical_code is not None else 3,
        "independent_max_abs_difference": (
            None if categorical_code is not None else independent_difference
        ),
        "public_verification_id": "reference-1",
        "independent_verification_ids": ["independent-replay"],
        "verification_record_paths": ["verification/replay.json"],
        "normative_source_ids": ["TAB-SRC-001"],
        "method_source_ids": ["TAB-SRC-011"],
        "precedent_source_ids": [],
    }
    inventory = {
        "schema_id": "trialagentbench.trialeval.scientific_construction_inventory/v1",
        "release_id": "release-1",
        "specification_sha256": "b" * 64,
        "rows": [scientific_row],
    }
    inventory["checksum"] = hashlib.sha256(
        json.dumps(
            inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    retained_source_ids = tuple(
        source_id
        for source_id in ("TAB-SRC-001", "TAB-SRC-011")
        if source_id not in set(missing_registry_source_ids)
    )
    _, release_scope_model, source_registry_model, component_evidence_model = (
        _public_component_fixture(
            release_id="release-1",
            registry_source_ids=retained_source_ids,
            evidence_release_id=phase_release_id,
        )
    )
    release_scope = release_scope_model.model_dump(mode="json")
    source_registry = source_registry_model.model_dump(mode="json")
    component_evidence = component_evidence_model.model_dump(mode="json")
    if phase_card_source_ids != ("TAB-SRC-001", "TAB-SRC-011"):
        component_evidence["evaluation_series"][0]["cells"][0]["source_ids"] = list(
            phase_card_source_ids
        )
        component_evidence.pop("checksum")
        component_evidence = _checked_payload(component_evidence)
    with ZipFile(verification, "a") as archive:
        archive.writestr(
            "construction/scientific_construction_inventory.json",
            _json(inventory),
        )
        archive.writestr("construction/release_scope.json", _json(release_scope))
        archive.writestr(
            "construction/scientific_source_registry.json", _json(source_registry)
        )
        archive.writestr(
            "construction/trialeval_component_evidence.json",
            _json(component_evidence),
        )
    evaluator = tmp_path / "evaluator.zip"
    with ZipFile(evaluator, "w") as archive:
        archive.writestr("grader/scoring_keys.jsonl", body)
        archive.writestr("grader/scoring_key_manifest.json", _json(manifest))
    return participant, evaluator, verification


def _numeric_report(difference: float) -> PublicEvidenceNumericReferenceReportV1:
    check = PublicEvidenceNumericReferenceCheckV1(
        task_id="TASK1",
        item_id="ITEM1",
        lane_id="primary",
        route_reference_id="reference-1",
        route_reference_checksum="a" * 64,
        variant_role="required_primary",
        input_bundle_id="input-1",
        input_bundle_checksum="b" * 64,
        route_family="cox_ph",
        estimator_method_id="cox_ph",
        effect_scale="log_hr",
        answer_shape="point",
        identification_class="point_identified",
        support_status="official_supported",
        outcome="matched",
        old_value=-0.5,
        recomputed_value=-0.5 + difference,
        abs_diff=difference,
        abs_tolerance=0.01,
        public_table_paths=("items/TASK1/data/ADSL.parquet",),
        public_table_roles=("analysis_population",),
        public_surface_shape="analysis_ready",
        message="matched",
    )
    return PublicEvidenceNumericReferenceReportV1(
        evaluator_zip="verification.zip",
        public_zip="participant.zip",
        status="pass",
        route_reference_count=1,
        route_reference_input_count=1,
        supported_method_ids=("cox_ph",),
        supported_check_count=1,
        matched_count=1,
        mismatched_count=0,
        unsupported_calculator_count=0,
        not_numeric_count=0,
        missing_public_input_count=0,
        invalid_public_input_count=0,
        outcome_counts={"matched": 1},
        method_counts={"cox_ph": 1},
        method_outcome_counts={"cox_ph": {"matched": 1}},
        unsupported_surface_shape_counts={},
        unsupported_disposition_counts={},
        unsupported_method_dispositions=(),
        drift_classification_counts={},
        findings=(),
        checks=(check,),
        drift_dispositions=(),
    )


def test_sensitivity_replay_requires_every_declared_parameter() -> None:
    base = _numeric_report(1e-10).checks[0]
    checks = tuple(
        base.model_copy(
            update={
                "route_reference_id": f"bounded-{parameter:.2f}",
                "estimator_method_id": "observed:tau_bounds_bounded_deviation",
                "answer_shape": "bound",
                "sensitivity_parameter": parameter,
                "abs_diff": None,
                "lower_abs_diff": 1e-10,
                "upper_abs_diff": 2e-10,
            }
        )
        for parameter in (0.05, 0.10, 0.20)
    )
    report = _numeric_report(1e-10).model_copy(update={"checks": checks})

    _, parameterized = recovery._direct_numeric_replays(report)
    observed = recovery._complete_sensitivity_replays(
        parameterized_replays=parameterized,
        item_id="TASK1",
        estimator_method_id="observed:tau_bounds_bounded_deviation",
        sensitivity_parameters=(0.05, 0.10, 0.20),
    )
    assert tuple(replay.difference for replay in observed) == (2e-10, 2e-10, 2e-10)

    with pytest.raises(ValueError, match="complete declared parameter grid"):
        recovery._complete_sensitivity_replays(
            parameterized_replays={
                key: replay for key, replay in parameterized.items() if key[2] != 0.20
            },
            item_id="TASK1",
            estimator_method_id="observed:tau_bounds_bounded_deviation",
            sensitivity_parameters=(0.05, 0.10, 0.20),
        )


def test_sensitivity_replays_are_scoped_to_released_tasks() -> None:
    base = _numeric_report(1e-10).checks[0]
    checks = tuple(
        base.model_copy(
            update={
                "task_id": task_id,
                "item_id": "SHARED-BASE-ITEM",
                "route_reference_id": f"{task_id}:bounded-0.05",
                "estimator_method_id": "observed:tau_bounds_bounded_deviation",
                "answer_shape": "bound",
                "sensitivity_parameter": 0.05,
                "abs_diff": None,
                "lower_abs_diff": difference,
                "upper_abs_diff": difference,
                "public_table_paths": (f"items/{task_id}/data/ADSL.parquet",),
            }
        )
        for task_id, difference in (("TASK1", 1e-10), ("TASK2", 2e-10))
    )
    report = _numeric_report(1e-10).model_copy(update={"checks": checks})

    _, parameterized = recovery._direct_numeric_replays(report)

    assert set(parameterized) == {
        ("TASK1", "observed:tau_bounds_bounded_deviation", 0.05),
        ("TASK2", "observed:tau_bounds_bounded_deviation", 0.05),
    }
    assert parameterized[
        ("TASK1", "observed:tau_bounds_bounded_deviation", 0.05)
    ].difference == pytest.approx(1e-10)
    assert parameterized[
        ("TASK2", "observed:tau_bounds_bounded_deviation", 0.05)
    ].difference == pytest.approx(2e-10)


def test_scoring_route_estimator_identity_is_task_scoped() -> None:
    route_id = (
        "TASK1:primary_numeric.v1:max_recoverable:observed:tau_bounds_bounded_deviation"
    )

    assert (
        recovery._scoring_route_estimator_method_id(
            item_id="TASK1",
            route_id=route_id,
        )
        == "observed:tau_bounds_bounded_deviation"
    )
    with pytest.raises(ValueError, match="invalid task-scoped"):
        recovery._scoring_route_estimator_method_id(
            item_id="TASK2",
            route_id=route_id,
        )


def test_recovery_opens_scoring_keys_only_after_independent_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_minimal_route_fixture(monkeypatch)
    roles = _write_trialeval_roles(tmp_path, independent_difference=1e-6)
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(1e-6),
    )

    report = recover_release(
        participant_release=roles[0],
        evaluator_release=roles[1],
        verification_release=roles[2],
    )
    write_recoverability_report(tmp_path / "report", report)

    assert report.status == "pass"
    assert report.private_generating_state_used is False
    assert report.required_route_count == report.replayed_route_count == 1
    assert report.routes[0].declared_absolute_tolerance == 1e-6
    assert report.routes[0].difference_to_tolerance_ratio == 1.0
    assert report.routes[0].comparison_denominator == 1
    assert (tmp_path / "report" / "recoverability_routes.parquet").is_file()


def test_recovery_route_rejects_an_inconsistent_tolerance_ratio() -> None:
    with pytest.raises(ValueError, match="tolerance ratio"):
        RecoverabilityRouteV1(
            suite="trialeval",
            unit_id="ITEM1",
            context_or_checkpoint_id="C1",
            route_id="route-1",
            estimator_family="km",
            effect_scale="risk_difference",
            result_kind="numeric_point",
            comparison_denominator=1,
            maximum_absolute_difference=2e-6,
            declared_absolute_tolerance=1e-6,
            difference_to_tolerance_ratio=1.0,
            comparison_rule="numeric_envelope",
            recovery_path="direct_analysis_ready",
            public_input_paths=("items/ITEM1/data/ADSL.parquet",),
            expected_summary="expected",
            reproduced_summary="reproduced",
            status="fail",
        )


def test_numeric_recovery_accepts_exact_agreement_at_zero_tolerance() -> None:
    route = RecoverabilityRouteV1(
        suite="trialeval",
        unit_id="TASK1",
        context_or_checkpoint_id="C1",
        route_id="route-1",
        estimator_family="km",
        effect_scale="risk_difference_tau",
        result_kind="numeric_point",
        comparison_denominator=1,
        maximum_absolute_difference=0.0,
        declared_absolute_tolerance=0.0,
        difference_to_tolerance_ratio=0.0,
        comparison_rule="numeric_envelope",
        recovery_path="direct_analysis_ready",
        public_input_paths=("items/TASK1/data/ADTTE.parquet",),
        expected_summary="exact",
        reproduced_summary="exact",
        status="pass",
    )

    assert route.difference_to_tolerance_ratio == 0.0


def test_recovery_rejects_envelope_smaller_than_measured_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_minimal_route_fixture(monkeypatch)
    roles = _write_trialeval_roles(tmp_path, independent_difference=1e-6)
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(2e-6),
    )

    report = recover_release(
        participant_release=roles[0],
        evaluator_release=roles[1],
        verification_release=roles[2],
    )

    assert report.status == "fail"
    assert report.failed_route_count == 1


def test_recovery_includes_raw_reconstruction_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_minimal_route_fixture(monkeypatch)
    roles = _write_trialeval_roles(
        tmp_path,
        independent_difference=1e-6,
        context_tier="C3",
    )
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(1e-6),
    )

    report = recover_release(
        participant_release=roles[0],
        evaluator_release=roles[1],
        verification_release=roles[2],
    )

    assert report.status == "pass"
    assert report.routes[0].context_or_checkpoint_id == "C3"
    assert report.routes[0].recovery_path == "reconstruct_raw_domains"


def test_recovery_rejects_scientific_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_minimal_route_fixture(monkeypatch)
    roles = _write_trialeval_roles(
        tmp_path,
        independent_difference=1e-6,
        inventory_effect_scale="hazard_ratio",
    )
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(1e-6),
    )

    with pytest.raises(
        ValueError, match="scientific inventory disagrees.*effect_scale"
    ):
        recover_release(
            participant_release=roles[0],
            evaluator_release=roles[1],
            verification_release=roles[2],
        )


def test_canonical_component_evidence_is_bound_to_component_inventory() -> None:
    components, _, _, evidence = _public_component_fixture(release_id="release-1")

    assert evidence.component_inventory_checksum == components.checksum
    assert len(evidence.evaluation_series) == 9
    assert sum(len(series.cells) for series in evidence.evaluation_series) == 25


def test_recovery_rejects_family_outside_release_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = _write_trialeval_roles(
        tmp_path,
        independent_difference=1e-6,
        item_base_case_id="CASE_RESEARCH",
    )
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(1e-6),
    )

    with pytest.raises(ValueError, match="must equal the canonical regime-cell scope"):
        recover_release(
            participant_release=roles[0],
            evaluator_release=roles[1],
            verification_release=roles[2],
        )


def test_recovery_rejects_unresolved_scientific_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_minimal_route_fixture(monkeypatch)
    roles = _write_trialeval_roles(
        tmp_path,
        independent_difference=1e-6,
        missing_registry_source_ids=("TAB-SRC-011",),
    )
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(1e-6),
    )

    with pytest.raises(
        ValueError, match="absent or unverified scientific sources.*TAB-SRC-011"
    ):
        recover_release(
            participant_release=roles[0],
            evaluator_release=roles[1],
            verification_release=roles[2],
        )


def test_recovery_rejects_component_evidence_for_another_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_minimal_route_fixture(monkeypatch)
    roles = _write_trialeval_roles(
        tmp_path,
        independent_difference=1e-6,
        phase_release_id="other-release",
    )
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(1e-6),
    )

    with pytest.raises(ValueError, match="identifies a different release"):
        recover_release(
            participant_release=roles[0],
            evaluator_release=roles[1],
            verification_release=roles[2],
        )


def test_recovery_rejects_component_evidence_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_minimal_route_fixture(monkeypatch)
    roles = _write_trialeval_roles(
        tmp_path,
        independent_difference=1e-6,
        phase_card_source_ids=("TAB-SRC-999",),
    )
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(1e-6),
    )

    with pytest.raises(ValidationError, match="complete series source application set"):
        recover_release(
            participant_release=roles[0],
            evaluator_release=roles[1],
            verification_release=roles[2],
        )


def test_recovery_compares_categorical_route_against_independent_public_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_minimal_route_fixture(monkeypatch)
    roles = _write_trialeval_roles(
        tmp_path,
        independent_difference=0.0,
        categorical_code="point_effect_not_identified",
        replay_code="point_effect_not_identified",
    )
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(0.0),
    )

    report = recover_release(
        participant_release=roles[0],
        evaluator_release=roles[1],
        verification_release=roles[2],
    )

    assert report.status == "pass"
    assert report.routes[0].comparison_rule == "categorical_code_membership"
    assert report.routes[0].reproduced_summary == "point_effect_not_identified"


def test_recovery_derives_a4_nonidentification_from_released_item_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_minimal_route_fixture(monkeypatch)
    code = "point_not_identified_due_to_censoring_or_support_failure"
    roles = _write_trialeval_roles(
        tmp_path,
        independent_difference=0.0,
        categorical_code=code,
        item_base_case_id="TE-S04-A4",
    )
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(0.0),
    )

    report = recover_release(
        participant_release=roles[0],
        evaluator_release=roles[1],
        verification_release=roles[2],
    )

    assert report.status == "pass"
    assert report.routes[0].reproduced_summary == code


def test_recovery_rejects_categorical_route_without_independent_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_minimal_route_fixture(monkeypatch)
    roles = _write_trialeval_roles(
        tmp_path,
        independent_difference=0.0,
        categorical_code="point_effect_not_identified",
    )
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(0.0),
    )

    with pytest.raises(ValueError, match="lacks independent replay"):
        recover_release(
            participant_release=roles[0],
            evaluator_release=roles[1],
            verification_release=roles[2],
        )


def test_recovery_rejects_nonnumeric_replay_not_bound_to_participant_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = _write_trialeval_roles(
        tmp_path,
        independent_difference=0.0,
        categorical_code="point_effect_not_identified",
        replay_code="point_effect_not_identified",
    )
    monkeypatch.setattr(
        recovery,
        "recompute_trialeval_public_numeric_reference_v1",
        lambda **_: _numeric_report(0.0),
    )
    participant = roles[0]
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(participant, "a") as archive:
            archive.writestr("items/TASK1/task.json", '{"changed":true}')

    with pytest.raises(
        ValueError,
        match="not computed from the supplied participant release|duplicate archive members",
    ):
        recover_release(
            participant_release=participant,
            evaluator_release=roles[1],
            verification_release=roles[2],
        )


def test_recovery_rejects_unsafe_trialdev_archive_member(tmp_path: Path) -> None:
    participant = tmp_path / "participant.zip"
    evaluator = tmp_path / "evaluator.zip"
    verification = tmp_path / "verification.zip"
    with ZipFile(participant, "w") as archive:
        archive.writestr("../escape", "unsafe")
    with ZipFile(evaluator, "w") as archive:
        archive.writestr("scenario_s1/grader/evaluation_target_register.jsonl", "{}\n")
    with ZipFile(verification, "w") as archive:
        archive.writestr("phase_replay/cases.jsonl", "")

    with pytest.raises(ValueError, match="unsafe release archive member"):
        recover_release(
            participant_release=participant,
            evaluator_release=evaluator,
            verification_release=verification,
        )
