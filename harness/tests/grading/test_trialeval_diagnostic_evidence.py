"""Tests for public-evidence replay in TrialEval grading."""

from __future__ import annotations

from pathlib import Path

from trialagentbench_harness.contracts.scoring.assumption_evidence import (
    AssumptionEvidenceManifestV1,
)
from trialagentbench_harness.contracts.submission import TrialEvalSubmissionV1
from trialagentbench_harness.io import canonical_payload_sha256
from trialagentbench_harness.trialeval.diagnostic_evidence import (
    validated_diagnostic_ids_v1,
)
from trialagentbench_harness.trialeval.schema import BenchmarkItem


def _assumption_evidence() -> AssumptionEvidenceManifestV1:
    payload: dict[str, object] = {
        "version": "v1",
        "schema_id": "trial_benchmark_assumption_evidence_manifest_v1",
        "item_id": "endpoint-ascertainment",
        "base_case_id": "endpoint-ascertainment",
        "canonical_item_id": "endpoint-ascertainment",
        "variant_id": "endpoint-ascertainment",
        "context_tier": "C2",
        "replicate_index": 0,
        "records": [
            {
                "assumption_id": "endpoint_ascertainment",
                "expected_status": "holds",
                "computed_status": "holds",
                "expected_band": "holds",
                "computed_band": "holds",
                "diagnosability": "directly_diagnosable",
                "severity_metric": 0.02,
                "severity_metric_name": "validation_discordance_fraction",
                "threshold_stressed": 0.025,
                "threshold_fragile": 0.1,
                "threshold_broken": 0.1,
                "decision_metric_names": {
                    "stressed": "validation_discordance_fraction",
                    "fragile": "validation_discordance_fraction",
                    "broken": "unsupported_validation_stratum_fraction",
                },
                "supporting_metrics": {
                    "unsupported_validation_stratum_fraction": 0.0,
                },
                "metric_units": {
                    "validation_discordance_fraction": "validation-discordance-probability",
                    "unsupported_validation_stratum_fraction": "validation-discordance-probability",
                },
                "metric_public_evidence_basis": {
                    "validation_discordance_fraction": ["data/validation.parquet"],
                    "unsupported_validation_stratum_fraction": ["data/validation.parquet"],
                },
                "notes": [],
            }
        ],
    }
    payload["checksum"] = canonical_payload_sha256(payload)
    return AssumptionEvidenceManifestV1.model_validate(payload)


def _item(tmp_path: Path) -> BenchmarkItem:
    visible = tmp_path / "public" / "items" / "TASK1"
    (visible / "data").mkdir(parents=True)
    (visible / "data" / "validation.parquet").write_bytes(b"fixture")
    (visible.parent.parent / "diagnostic_dictionary.json").write_text("{}", encoding="utf-8")
    (visible / "protocol_summary.json").write_text(
        '{"arms":[{"arm_id":"control"},{"arm_id":"active"}],"design_family":"parallel_randomized"}',
        encoding="utf-8",
    )
    contract: dict[str, object] = {
        "schema_id": "trialagentbench.trialeval_semantic_submission_contract/v1",
        "task_id": "TASK1",
        "submission_semantics_id": "trialagentbench.trialeval_submission/v1",
        "required_deliverables": ["evidence", "limitations", "primary_analysis"],
        "diagnostic_obligations": [
            {
                "assumption_id": "endpoint_ascertainment",
                "diagnostic_id": "endpoint_ascertainment_public",
                "evidence_requirement": "empirical_diagnostic",
                "primary_credit_policy": "method_dependent",
                "operation": "Estimate endpoint discordance from the released validation sample.",
                "score_bearing_metric_id": "validation_discordance_fraction",
                "metric_unit": "validation-discordance-probability",
                "public_evidence_basis": ["data/validation.parquet"],
                "interpretation": "Inference is limited to represented validation cells.",
            }
        ],
    }
    contract["checksum"] = canonical_payload_sha256(contract)
    return BenchmarkItem(
        item_id="TASK1",
        task_id="TASK1",
        trial_name="Endpoint ascertainment",
        design_tier="D3",
        design_subtype="endpoint_ascertainment",
        assumption_tier="A3",
        context_tier="C2",
        visible_dir=visible,
        data_dir=visible / "data",
        task={},
        submission_contract=contract,
    )


def _submission(*, result: dict[str, object], source_artifacts: list[str]) -> TrialEvalSubmissionV1:
    return TrialEvalSubmissionV1.model_validate(
        {
            "task_id": "TASK1",
            "primary_analysis": {
                "declared_primary": True,
                "estimand": {
                    "estimand_id": "primary_itt",
                    "population_id": "intention_to_treat",
                    "treatment_id": "active",
                    "comparator_id": "control",
                    "endpoint_id": "primary_endpoint",
                    "intercurrent_event_strategy_ids": [],
                    "horizon": {"value": 365, "unit": "days"},
                },
                "estimator": {
                    "analysis_method_id": "validated_endpoint_bootstrap",
                    "qualifications": ["randomization_exchangeability"],
                },
                "result_kind": "numeric_point",
                "result": {
                    "kind": "scalar",
                    "value": 0.1,
                    "effect_scale": "risk_difference_tau",
                    "unit": "probability_difference",
                    "interval": {
                        "lower": 0.0,
                        "upper": 0.2,
                        "confidence_level": 0.95,
                    },
                },
                "favorable_direction": "higher",
                "evidence_ids": ["endpoint-check"],
            },
            "evidence": [
                {
                    "evidence_id": "endpoint-check",
                    "evidence_type": "diagnostic",
                    "principle": "data_quality",
                    "operation": "assessment",
                    "diagnostic_id": "endpoint_ascertainment_public",
                    "target": "endpoint discordance",
                    "result": result,
                    "interpretation": "Endpoint ascertainment was assessed from public evidence.",
                    "source_artifacts": source_artifacts,
                }
            ],
            "limitations": ["Validation inference is limited to represented cells."],
        }
    )


def test_endpoint_ascertainment_rejects_factual_premise_shortcut(tmp_path: Path) -> None:
    submission = _submission(
        result={
            "kind": "factual_premise",
            "premise_id": "randomized_assignment_declared",
            "conclusion": "supported",
        },
        source_artifacts=["protocol_summary.json"],
    )

    assert (
        validated_diagnostic_ids_v1(
            submission=submission,
            item=_item(tmp_path),
            assumption_evidence=_assumption_evidence(),
        )
        == ()
    )


def test_endpoint_ascertainment_accepts_replayed_numeric_diagnostic(tmp_path: Path) -> None:
    submission = _submission(
        result={
            "kind": "diagnostic_summary",
            "measures": [
                {
                    "metric_id": "validation_discordance_fraction",
                    "value": 0.02,
                    "unit": "validation-discordance-probability",
                    "decimal_places": 3,
                },
                {
                    "metric_id": "unsupported_validation_stratum_fraction",
                    "value": 0.0,
                    "unit": "validation-discordance-probability",
                    "decimal_places": 3,
                },
            ],
        },
        source_artifacts=["data/validation.parquet", "diagnostic_dictionary.json"],
    )

    assert validated_diagnostic_ids_v1(
        submission=submission,
        item=_item(tmp_path),
        assumption_evidence=_assumption_evidence(),
    ) == ("endpoint_ascertainment_public",)


def test_endpoint_ascertainment_accepts_required_metric_with_supplementary_measure(
    tmp_path: Path,
) -> None:
    submission = _submission(
        result={
            "kind": "diagnostic_summary",
            "measures": [
                {
                    "metric_id": "validation_discordance_fraction",
                    "value": 0.02,
                    "unit": "validation-discordance-probability",
                    "decimal_places": 3,
                },
                {
                    "metric_id": "validated_records",
                    "value": 250.0,
                    "unit": "records",
                    "decimal_places": 0,
                },
            ],
        },
        source_artifacts=["data/validation.parquet"],
    )

    assert validated_diagnostic_ids_v1(
        submission=submission,
        item=_item(tmp_path),
        assumption_evidence=_assumption_evidence(),
    ) == ("endpoint_ascertainment_public",)


def test_endpoint_ascertainment_compares_value_at_declared_precision(tmp_path: Path) -> None:
    submission = _submission(
        result={
            "kind": "diagnostic_summary",
            "measures": [
                {
                    "metric_id": "validation_discordance_fraction",
                    "value": 0.0200000001,
                    "unit": "validation-discordance-probability",
                    "decimal_places": 3,
                }
            ],
        },
        source_artifacts=["data/validation.parquet"],
    )

    assert validated_diagnostic_ids_v1(
        submission=submission,
        item=_item(tmp_path),
        assumption_evidence=_assumption_evidence(),
    ) == ("endpoint_ascertainment_public",)


def test_endpoint_ascertainment_rejects_wrong_metric_id_with_matching_value_and_unit(
    tmp_path: Path,
) -> None:
    """A numerically coincident value cannot acquire another metric's identity."""

    submission = _submission(
        result={
            "kind": "diagnostic_summary",
            "measures": [
                {
                    "metric_id": "unsupported_validation_stratum_fraction",
                    "value": 0.02,
                    "unit": "validation-discordance-probability",
                    "decimal_places": 3,
                }
            ],
        },
        source_artifacts=["data/validation.parquet"],
    )

    assert (
        validated_diagnostic_ids_v1(
            submission=submission,
            item=_item(tmp_path),
            assumption_evidence=_assumption_evidence(),
        )
        == ()
    )
