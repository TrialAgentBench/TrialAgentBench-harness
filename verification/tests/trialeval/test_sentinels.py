"""Tests for deterministic high-risk sentinel selection and audit semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from trialagentbench_validation.cli import main
from trialagentbench_validation.trialeval.references.calculators import (
    PublicIPCWArmSupportV1,
    PublicIPCWSupportDiagnosticsV1,
)
from trialagentbench_validation.trialeval.sentinels import (
    DEFAULT_SENTINEL_STRATA_V1,
    SentinelStratumV1,
    _observation_process_limitation,
    audit_trialeval_sentinels,
)


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, body in sorted(members.items()):
            archive.writestr(name, body)


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()


def _index_entry() -> dict[str, object]:
    return {
        "task_id": "TASK1",
        "item_id": "item1",
        "generation_seed": 101,
        "base_case_id": "d1a1_rct_clean",
        "variant_id": "d1a1_rct_clean__C1",
        "factors": {
            "design_archetype": "D1",
            "design_subtype": "individual_randomized",
            "assumption_regime": "A1",
            "context_configuration": "C1",
            "data_preparation": "analysis_ready",
            "analysis_specification": "locked_sap",
            "procedure_assistance": "output_contract_only",
            "response_interface": "structured",
            "regime_cell_id": "d1a1_rct_clean",
            "evaluation_series_id": "d1a1_rct_clean",
        },
        "scoring_row_offset": 0,
        "scoring_row_count": 1,
        "reconstruction_row_offset": 0,
        "reconstruction_row_count": 0,
        "data_integrity_reference_row_offset": 0,
        "data_integrity_reference_row_count": 0,
    }


def _scoring_members(*routes: tuple[str, str]) -> dict[str, bytes]:
    credit_eligible_routes = [
        {
            "route_id": f"route:{family}:{scale}",
            "signature": {
                "analysis_population_id": "intention_to_treat",
                "estimand_id": "estimand:primary",
                "intercurrent_event_strategy_ids": ["discontinuation:treatment_policy"],
                "assessment_horizon_days": 365.0,
                "treatment_id": "active",
                "comparator_id": "control",
                "endpoint_id": "death",
                "effect_scale": scale,
                "analysis_method_id": f"method:{family}:{scale}:bootstrap",
            },
            "method": {
                "analysis_method_id": f"method:{family}:{scale}:bootstrap",
                "estimator_family": family,
                "result_kind": "numeric_point",
                "uncertainty_method": "bootstrap",
                "design_modifiers": [],
                "sensitivity_parameters": [],
            },
            "required_identification_assumptions": ["randomization"],
            "required_diagnostics": [],
            "target": {
                "kind": "numeric_point",
                "value": 1.0,
                "result_unit": "fixture_unit",
                "acceptance_envelope": {
                    "schema_id": "trialagentbench.numerical_acceptance_envelope/v1",
                    "reporting_decimal_places": 3,
                    "independent_max_abs_difference": 0.001,
                    "public_verification_id": "fixture-public-replay",
                    "independent_verification_ids": ["fixture-independent-replay"],
                },
                "require_confidence_interval": True,
                "confidence_interval_lower": 0.8,
                "confidence_interval_upper": 1.2,
            },
        }
        for scale, family in routes
    ]
    key = {
        "schema_id": "trialagentbench.scoring_key/v1",
        "release_id": "fixture-release",
        "item_id": "TASK1",
        "question_id": "TASK1:primary",
        "context_tier": "C1",
        "credit_eligible_routes": credit_eligible_routes,
    }
    body = _jsonl([key])
    manifest = {
        "schema_id": "trialagentbench.scoring_key_manifest/v1",
        "release_id": "fixture-release",
        "specification_sha256": "d" * 64,
        "scoring_keys_sha256": hashlib.sha256(body).hexdigest(),
        "item_ids": ["TASK1"],
    }
    return {
        "grader/scoring_keys.jsonl": body,
        "grader/scoring_key_manifest.json": json.dumps(manifest).encode(),
    }


def test_default_sentinels_use_canonical_v1_cells_and_one_a4_construct() -> None:
    assert all(row.variant_id.startswith("TE-S") for row in DEFAULT_SENTINEL_STRATA_V1)
    unmeasured = tuple(
        row
        for row in DEFAULT_SENTINEL_STRATA_V1
        if row.required_observation_process_limitation
        == "unmeasured_dependent_censoring"
    )

    assert len(unmeasured) == 1
    limitation_field = SentinelStratumV1.model_fields[
        "required_observation_process_limitation"
    ]
    assert "practical_positivity_failure" not in str(limitation_field.annotation)


def test_observation_process_identifies_unmeasured_dependent_censoring(
    tmp_path: Path,
) -> None:
    participant = tmp_path / "participant.zip"
    _write_zip(
        participant,
        {
            "items/TASK1/protocol_summary.json": json.dumps(
                {
                    "observation_process": {
                        "factor_associated_with_primary_endpoint": True,
                        "factor_recorded_in_released_data": False,
                        "follow_up_decision_basis": "clinician_assessed_prognostic_factor",
                        "loss_to_follow_up_role": "observation_process_censoring",
                    }
                }
            ).encode()
        },
    )

    with ZipFile(participant) as archive:
        limitation = _observation_process_limitation(
            participant=archive,
            task_id="TASK1",
        )

    assert limitation == "unmeasured_dependent_censoring"


def test_ipcw_sentinel_replays_support_from_estimator_family(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.zip"
    participant = tmp_path / "participant.zip"
    index = {
        "schema_id": "trialagentbench.item_index/v1",
        "version": "v1",
        "checksum": "c" * 64,
        "entries": [_index_entry()],
    }
    _write_zip(
        evaluator,
        {
            "grader/item_index.json": json.dumps(index).encode(),
            **_scoring_members(("rmst_difference_tau", "km_ipcw")),
        },
    )
    _write_zip(
        participant,
        {
            "items/TASK1/task.json": b"{}",
            "items/TASK1/data/ADSL.parquet": b"fixture",
        },
    )
    arm = PublicIPCWArmSupportV1(
        evaluated_event_time_count=2,
        minimum_fitted_censoring_survival=0.8,
        maximum_weight=1.25,
        minimum_effective_sample_size_ratio=0.9,
    )
    support = PublicIPCWSupportDiagnosticsV1(
        support_by_arm={"active": arm, "control": arm},
    )

    with patch(
        "trialagentbench_validation.trialeval.sentinels.recompute_public_ipcw_support_v1",
        return_value=support,
    ) as replay:
        report = audit_trialeval_sentinels(
            evaluator_zip=evaluator,
            participant_zip=participant,
            strata=(
                SentinelStratumV1(
                    sentinel_id="ipcw",
                    variant_id="d1a1_rct_clean__C1",
                    context_tier="C1",
                    rationale="Fixture.",
                ),
            ),
        )

    replay.assert_called_once()
    assert report.records[0].ipcw_support == support
    assert not any(
        finding.startswith("ipcw_support") for finding in report.records[0].findings
    )


def test_locked_sap_plural_estimands_require_readjudication(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.zip"
    participant = tmp_path / "participant.zip"
    index = {
        "schema_id": "trialagentbench.item_index/v1",
        "version": "v1",
        "checksum": "c" * 64,
        "entries": [_index_entry()],
    }
    _write_zip(
        evaluator,
        {
            "grader/item_index.json": json.dumps(index).encode(),
            **_scoring_members(
                ("rmst_difference_tau", "rmst_contrast"),
                ("log_time_ratio", "aft_parametric"),
            ),
        },
    )
    _write_zip(
        participant,
        {
            "items/TASK1/task.json": json.dumps(
                {"analysis_tasks_file": "analysis_tasks.md"}
            ).encode(),
            "items/TASK1/data/ADSL.parquet": b"fixture",
        },
    )

    report = audit_trialeval_sentinels(
        evaluator_zip=evaluator,
        participant_zip=participant,
        strata=(
            SentinelStratumV1(
                sentinel_id="locked",
                variant_id="d1a1_rct_clean__C1",
                context_tier="C1",
                rationale="Fixture.",
            ),
        ),
    )

    assert report.status == "fail"
    assert "locked_sap_requires_exactly_one_route" in report.records[0].findings
    assert "locked_sap_missing_complete_primary_analysis" in report.records[0].findings
    assert "task_references_missing_public_file" in report.records[0].findings
    assert report.records[0].findings


def test_locked_sap_requires_full_participant_visible_route(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.zip"
    participant = tmp_path / "participant.zip"
    index = {
        "schema_id": "trialagentbench.item_index/v1",
        "version": "v1",
        "checksum": "c" * 64,
        "entries": [_index_entry()],
    }
    _write_zip(
        evaluator,
        {
            "grader/item_index.json": json.dumps(index).encode(),
            **_scoring_members(("rmst_difference_tau", "rmst_contrast")),
        },
    )
    analysis_plan = {
        "primary_estimand_id": "rmst_difference_365d",
        "primary_endpoint_id": "death",
        "primary_analysis": {
            "estimator_family": "rmst_contrast",
            "method_id": "observed:rmst_contrast",
            "effect_scale": "rmst_difference_tau",
            "implementation": "Restricted mean survival time contrast.",
            "uncertainty_method": "participant_bootstrap",
            "required_method_modifiers": [],
        },
        "diagnostic_requirements": [
            {
                "assumption_id": "independent_censoring",
                "diagnostic_methods": ["censoring_summary"],
            }
        ],
        "lane_rules": [
            {
                "estimand_id": "rmst_difference_365d",
                "endpoint_id": "death",
                "effect_scale": "rmst_difference_tau",
                "role": "primary",
                "mandatory": True,
            }
        ],
    }
    _write_zip(
        participant,
        {
            "items/TASK1/task.json": json.dumps(
                {
                    "primary_estimand_id": "rmst_difference_365d",
                    "primary_endpoint_id": "death",
                    "primary_effect_scale": "rmst_difference_tau",
                }
            ).encode(),
            "items/TASK1/analysis_plan.json": json.dumps(analysis_plan).encode(),
            "items/TASK1/data/ADSL.parquet": b"fixture",
        },
    )

    report = audit_trialeval_sentinels(
        evaluator_zip=evaluator,
        participant_zip=participant,
        strata=(
            SentinelStratumV1(
                sentinel_id="locked",
                variant_id="d1a1_rct_clean__C1",
                context_tier="C1",
                rationale="Fixture.",
            ),
        ),
    )

    assert (
        "locked_sap_missing_complete_primary_analysis" not in report.records[0].findings
    )
    assert "locked_sap_primary_lane_binding_mismatch" not in report.records[0].findings
    assert "locked_sap_task_binding_mismatch" not in report.records[0].findings


def test_cli_writes_failed_audit_record(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.zip"
    participant = tmp_path / "participant.zip"
    output = tmp_path / "audit.json"
    index = {
        "schema_id": "trialagentbench.item_index/v1",
        "version": "v1",
        "checksum": "c" * 64,
        "entries": [_index_entry()],
    }
    _write_zip(
        evaluator,
        {
            "grader/item_index.json": json.dumps(index).encode(),
            **_scoring_members(("rmst_difference_tau", "rmst_contrast")),
        },
    )
    _write_zip(
        participant,
        {
            "items/TASK1/task.json": b"{}",
            "items/TASK1/data/ADSL.parquet": b"fixture",
        },
    )

    sentinel = SentinelStratumV1(
        sentinel_id="locked",
        variant_id="d1a1_rct_clean__C1",
        context_tier="C1",
        rationale="Fixture.",
    )
    with patch(
        "trialagentbench_validation.cli.audit_trialeval_sentinels",
        side_effect=lambda **kwargs: audit_trialeval_sentinels(
            strata=(sentinel,), **kwargs
        ),
    ):
        status = main(
            [
                "trialeval-sentinels",
                "--evaluator",
                str(evaluator),
                "--participant",
                str(participant),
                "--output",
                str(output),
            ]
        )

    assert status == 1
    assert json.loads(output.read_text())["status"] == "fail"
