"""Tests for truth-independent TrialEval experiment contracts."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)

from trialagentbench_harness.contracts.core.config import DecodingConfigV1
from trialagentbench_harness.contracts.core.runs import (
    TrialEvalAblationRunConfigV1,
    TrialEvalConditionProvenanceV1,
)
from trialagentbench_harness.contracts.experiments import (
    ProcedureAssistanceExposureV1,
    TrialEvalAblationAnalysisConfigV1,
    TrialEvalAblationArmSummaryV1,
    TrialEvalAblationAssignmentV1,
    TrialEvalAblationEndpointRowV1,
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalAblationScheduleV1,
    TrialEvalExperimentProtocolV1,
    TrialEvalSubmissionInterfaceV1,
    procedure_assistance_exposure_v1,
    trialeval_publication_analysis_config_v1,
)
from trialagentbench_harness.contracts.submission import trialeval_submission_shape_catalogue
from trialagentbench_harness.io import read_json_model
from trialagentbench_harness.trialeval.agent import _build_system_prompt, _get_tools
from trialagentbench_harness.trialeval.conditions import (
    condition_contrasts_markdown_v1,
    procedure_assistance_v1,
    prompt_set_sha256_v1,
    response_contract_sha256_v1,
    schema_affordance_inventory_v1,
    stage_response_contract_v1,
    submission_instruction_v1,
)
from trialagentbench_harness.trialeval.data import discover_participant_items
from trialagentbench_harness.trialeval.schema import BenchmarkItem


def _assignment(
    *,
    condition: str,
    interface: str,
    suffix: str,
    assistance: str = "output_contract_only",
    specification: str = "locked_sap",
) -> dict[str, str]:
    return {
        "assignment_id": f"assignment-{suffix}",
        "task_id": "TASK1001",
        "context_tier": "C1",
        "data_preparation": "analysis_ready",
        "analysis_specification": specification,
        "analysis_surface_sha256": ("1" if specification == "protocol_only" else "2") * 64,
        "replicate_id": "seed-1",
        "decoding_seed": 101,
        "procedure_assistance": assistance,
        "prompt_condition": condition,
        "submission_interface": interface,
    }


def _factorial_assignments(*, suffix_prefix: str = "") -> list[dict[str, str]]:
    return [
        _assignment(
            condition="neutral",
            interface=interface,
            assistance=assistance,
            suffix=f"{suffix_prefix}{specification}-{assistance}-{interface}",
            specification=specification,
        )
        for specification in ("locked_sap",)
        for assistance in ("output_contract_only", "unordered_checklist", "ordered_sop")
        for interface in ("structured", "narrative")
    ]


def _schedule_payload(*, design: str, assignments: list[dict[str, str]]) -> dict[str, object]:
    randomized = sorted(assignments, key=lambda row: row["assignment_id"])
    random.Random(17).shuffle(randomized)
    return {
        "experiment_id": "experiment-1",
        "design": design,
        "execution_scope": "pilot",
        "experiment_design_sha256": "d" * 64,
        "participant_release_sha256": "a" * 64,
        "prompt_set_sha256": prompt_set_sha256_v1(),
        "analysis_config_sha256": "c" * 64,
        "randomization_seed": 17,
        "assignments": randomized,
    }


def _valid_endpoint_payload() -> dict[str, object]:
    return {
        "assignment_id": "assignment-endpoint",
        "task_id": "TASK1001",
        "context_tier": "C1",
        "data_preparation": "analysis_ready",
        "analysis_specification": "locked_sap",
        "model_id": "model-1",
        "replicate_id": "seed-1",
        "procedure_assistance": "output_contract_only",
        "prompt_condition": "neutral",
        "submission_interface": "structured",
        "normalization_source": "direct_structured",
        "usable_primary": True,
        "route_match": True,
        "obligations_met": True,
        "credit_eligible_route_count": 1,
        "numeric_result_available": True,
        "result_match": True,
        "numeric_absolute_error": 0.001,
        "numeric_tolerance_ratio": 0.2,
        "primary_analysis_conforms": True,
        "planning_applicable": False,
    }


def test_factorial_schedule_requires_the_complete_paired_design() -> None:
    assignments = _factorial_assignments()
    schedule = TrialEvalAblationScheduleV1.model_validate(
        _schedule_payload(design="factorial_interface", assignments=assignments)
    )
    assert schedule.checksum is not None
    assert len(schedule.checksum) == 64

    with pytest.raises(ValidationError, match="must contain exactly"):
        TrialEvalAblationScheduleV1.model_validate(
            _schedule_payload(design="factorial_interface", assignments=assignments[:-1])
        )


def test_analysis_config_rejects_duplicate_or_targeted_factorial_endpoints() -> None:
    base = {
        "design": "factorial_interface",
        "execution_scope": "pilot",
        "experiment_design_sha256": "d" * 64,
        "primary_estimand": {
            "metric": "usable_primary",
            "contrast_id": "P2-P1",
            "analysis_specification": "protocol_only",
        },
        "supporting_metrics": ("primary_analysis_conforms",),
        "confidence_level": 0.9,
        "bootstrap_resamples": 1000,
        "bootstrap_seed": 9,
        "min_base_trial_clusters": 2,
        "min_decoding_replicates": 2,
    }

    duplicate = dict(base)
    duplicate["supporting_metrics"] = ("usable_primary",)
    with pytest.raises(ValidationError, match="cannot be repeated"):
        TrialEvalAblationAnalysisConfigV1.model_validate(duplicate)

    targeted = dict(base)
    targeted["primary_estimand"] = {
        **base["primary_estimand"],
        "prompt_condition": "targeted_covariate_structure",
    }
    with pytest.raises(ValidationError, match="cannot declare targeted-prompt strata"):
        TrialEvalAblationAnalysisConfigV1.model_validate(targeted)

    underreplicated = dict(base)
    underreplicated.update(min_base_trial_clusters=1, min_decoding_replicates=1)
    with pytest.raises(ValidationError, match="require at least two trial and decoding clusters"):
        TrialEvalAblationAnalysisConfigV1.model_validate(underreplicated)


def test_targeted_analysis_config_requires_capability_and_applicability_strata() -> None:
    base = {
        "design": "targeted_control",
        "execution_scope": "pilot",
        "experiment_design_sha256": "d" * 64,
        "primary_estimand": {
            "metric": "usable_primary",
            "contrast_id": "targeted_vs_neutral",
            "analysis_specification": "protocol_only",
        },
        "confidence_level": 0.9,
        "bootstrap_resamples": 1000,
        "bootstrap_seed": 9,
        "min_base_trial_clusters": 2,
        "min_decoding_replicates": 2,
    }

    with pytest.raises(ValidationError, match="targeted capability prompt"):
        TrialEvalAblationAnalysisConfigV1.model_validate(base)

    with_prompt = dict(base)
    with_prompt["primary_estimand"] = {
        **base["primary_estimand"],
        "prompt_condition": "targeted_covariate_structure",
    }
    with pytest.raises(ValidationError, match="applicability stratum"):
        TrialEvalAblationAnalysisConfigV1.model_validate(with_prompt)


def test_public_factorial_config_is_derived_from_frozen_design() -> None:
    root = Path(__file__).resolve().parents[2]
    design = read_json_model(
        TrialEvalExperimentProtocolV1,
        root / "experiment_configs" / "trialeval_experiment_protocol_v1.json",
    )
    config = read_json_model(
        TrialEvalAblationAnalysisConfigV1,
        root / "experiment_configs" / "trialeval_factorial_protocol_primary_v1.json",
    )

    assert config == trialeval_publication_analysis_config_v1(design)
    assert config.primary_estimand.metric == "primary_analysis_conforms"
    assert config.execution_scope == "publication"
    assert config.primary_estimand.contrast_id == "P2-P0"
    assert config.primary_estimand.analysis_specification == "protocol_only"
    assert config.min_base_trial_clusters == design.precision.retained_independent_base_trials
    assert config.min_decoding_replicates == 2
    assert "obligations_met" in config.supporting_metrics
    assert "numeric_absolute_error" in config.supporting_metrics
    assert "primary_interval_agreement" in config.supporting_metrics


def test_endpoint_requires_complete_noncompensatory_cascade() -> None:
    invalid_obligations = {
        **_valid_endpoint_payload(),
        "route_match": False,
        "obligations_met": True,
        "result_match": False,
        "primary_analysis_conforms": False,
        "primary_failure_code": "unrecognized_primary_route",
        "numeric_absolute_error": None,
        "numeric_tolerance_ratio": None,
    }
    with pytest.raises(ValidationError, match="only for a matched route"):
        TrialEvalAblationEndpointRowV1.model_validate(invalid_obligations)

    invalid_result = {
        **_valid_endpoint_payload(),
        "obligations_met": False,
        "result_match": True,
        "primary_analysis_conforms": False,
        "primary_failure_code": "missing_required_diagnostic",
    }
    with pytest.raises(ValidationError, match="satisfied obligations"):
        TrialEvalAblationEndpointRowV1.model_validate(invalid_result)

    invalid_acceptance = {
        **_valid_endpoint_payload(),
        "obligations_met": False,
        "result_match": False,
    }
    with pytest.raises(ValidationError, match="complete noncompensatory cascade"):
        TrialEvalAblationEndpointRowV1.model_validate(invalid_acceptance)


def test_endpoint_rejects_incomplete_numeric_diagnostics_and_zero_routes() -> None:
    incomplete_error = {
        **_valid_endpoint_payload(),
        "numeric_tolerance_ratio": None,
    }
    with pytest.raises(ValidationError, match="must be reported together"):
        TrialEvalAblationEndpointRowV1.model_validate(incomplete_error)

    no_routes = {
        **_valid_endpoint_payload(),
        "credit_eligible_route_count": 0,
    }
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        TrialEvalAblationEndpointRowV1.model_validate(no_routes)


def test_public_metric_documentation_uses_prospective_interface_sign() -> None:
    """Keep the documented interface contrast aligned with the estimator."""

    documentation = (Path(__file__).resolve().parents[2] / "docs" / "SCORING.md").read_text(encoding="utf-8")

    assert "structured-minus-narrative" in documentation
    assert "narrative-minus-structured" not in documentation


def test_public_documentation_defines_locked_sap_as_complete_prespecified_plan() -> None:
    """Keep the declared intervention aligned with the executable SAP."""

    docs_root = Path(__file__).resolve().parents[2] / "docs"
    documentation = "\n".join(
        (docs_root / name).read_text(encoding="utf-8") for name in ("EXPERIMENTS.md", "SCORING.md")
    )

    assert "complete prespecified analysis plan" in documentation
    assert "analysis method" in documentation
    assert "uncertainty procedure" in documentation
    assert "does not prescribe an ordered work process" in documentation
    assert "controlled estimand-specification intervention" not in documentation


def test_metric_population_cannot_mislabel_survivor_conditioned_planning() -> None:
    with pytest.raises(ValidationError, match="planning_consequence_evaluable_assignments"):
        TrialEvalAblationArmSummaryV1(
            model_id="model",
            metric="planning_power_shortfall",
            analysis_population="all_scheduled_assignments",
            analysis_specification="protocol_only",
            procedure_assistance="output_contract_only",
            submission_interface="structured",
            n_assignments=6,
            n_base_trial_clusters=2,
            n_decoding_replicates=3,
            estimate=0.1,
            interval_low=0.0,
            interval_high=0.2,
            confidence_level=0.95,
        )


def test_public_experiment_documentation_does_not_claim_d1_is_a_ph_dose_series() -> None:
    """D1 changes censoring obligations and cannot identify an ordinal PH effect."""

    documentation = (Path(__file__).resolve().parents[2] / "docs" / "EXPERIMENTS.md").read_text(encoding="utf-8")

    assert "do not identify an ordinal PH-dose effect" in documentation
    assert "proportional-hazards series supports matched" not in documentation


def test_incremental_sop_holds_analysis_components_fixed() -> None:
    """Contract and SOP arms differ in ordering, not task-general content."""

    contract = procedure_assistance_v1("unordered_checklist")
    sop = procedure_assistance_v1("ordered_sop")
    shared_operations = (
        "define the scientific question, estimand, population, contrast, endpoint, horizon",
        "inspect the participant-visible data, assignment mechanism, dependence structure",
        "assess identification and the assumptions needed by candidate analyses",
        "execute a defensible primary analysis with uncertainty",
        "when the public task declares planning eligible, translate the result into the requested planning implication",
        "verify every submitted claim against executed code",
    )
    for operation in shared_operations:
        assert operation in contract
        assert operation in sop
    assert "operations in any order" in contract
    assert "required order" not in contract
    assert "same operations in this required order" in sop


def test_assistance_exposure_changes_only_checklist_ordering() -> None:
    trialeval_contract = procedure_assistance_exposure_v1(
        suite="trialeval",
        procedure_assistance="unordered_checklist",
    )
    trialdev_contract = procedure_assistance_exposure_v1(
        suite="trialdev",
        procedure_assistance="unordered_checklist",
    )
    trialdev_sop = procedure_assistance_exposure_v1(
        suite="trialdev",
        procedure_assistance="ordered_sop",
    )

    assert trialeval_contract.unordered_completeness_instruction is True
    assert trialeval_contract.ordered_analysis_procedure is False
    assert trialdev_contract.unordered_completeness_instruction is True
    assert trialdev_contract.ordered_analysis_procedure is False
    assert trialdev_sop.ordered_analysis_procedure is True

    with pytest.raises(ValueError, match="TrialEval does not define"):
        procedure_assistance_exposure_v1(
            suite="trialeval",
            procedure_assistance="catalog_only",
        )


def test_procedure_assistance_exposure_rejects_old_public_field() -> None:
    with pytest.raises(ValidationError, match="procedure_assistance"):
        ProcedureAssistanceExposureV1.model_validate(
            {
                "suite": "trialeval",
                "assistance": "unordered_checklist",
                "unordered_completeness_instruction": True,
                "ordered_analysis_procedure": False,
            }
        )


def test_submission_interfaces_share_exact_semantic_obligations() -> None:
    structured = submission_instruction_v1("structured")
    narrative = submission_instruction_v1("narrative")
    obligations = (
        "State exactly one primary analysis, including the complete estimand (population, treatment conditions "
        "and contrast, endpoint, horizon, and intercurrent-event handling), executed estimator, result shape, "
        "numerical value, unit and orientation, uncertainty, participant-evidence links, executed supporting "
        "analyses, and limitations."
    )

    assert structured.count(obligations) == 1
    assert narrative.count(obligations) == 1
    inventory = schema_affordance_inventory_v1()
    for token in inventory.shared_scientific_vocabulary:
        assert token in structured
        assert token in narrative


def test_trialeval_provenance_rejects_unknown_assistance_arm() -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        TrialEvalConditionProvenanceV1(
            procedure_assistance="catalog_only",
            analysis_specification="protocol_only",
            analysis_surface_sha256="a" * 64,
            prompt_condition="neutral",
            submission_interface="structured",
            max_turns=20,
            prompt_set_sha256="b" * 64,
            rendered_system_prompt_sha256="c" * 64,
            tool_schema_sha256="d" * 64,
            response_contract_sha256="e" * 64,
        )


@pytest.mark.parametrize(
    "field",
    ("scorer_source_sha256", "agent_source_sha256", "run_identity_sha256"),
)
def test_ablation_run_contract_rejects_omitted_authoritative_identity(field: str) -> None:
    assignments = _factorial_assignments()
    schedule = TrialEvalAblationScheduleV1.model_validate(
        _schedule_payload(design="factorial_interface", assignments=assignments)
    )
    config = TrialEvalAblationRunConfigV1.create(
        timestamp_utc=datetime.now(UTC),
        experiment_id=schedule.experiment_id,
        schedule_checksum=str(schedule.checksum),
        participant_release_sha256=schedule.participant_release_sha256,
        prompt_set_sha256=schedule.prompt_set_sha256,
        scorer_source_sha256="s" * 64,
        agent_source_sha256="a" * 64,
        model="model",
        max_context_characters=120_000,
        item_watchdog_seconds=3600,
        decoding=DecodingConfigV1(temperature=0.0, max_tokens=1024, send_temperature=True),
        routing={"provider": "openai", "request_timeout_seconds": 300.0},
        executor={
            "image_reference": "executor:test",
            "image_id": f"sha256:{'e' * 64}",
            "python_version": "3.12",
            "packages": [{"name": "pandas", "version": "2.2"}],
            "limits": {},
        },
        workers=1,
        n_assignments=len(assignments),
    )
    payload = config.model_dump(mode="python")
    del payload[field]

    with pytest.raises(ValidationError, match=field):
        TrialEvalAblationRunConfigV1.model_validate(payload)


def test_schedule_accepts_current_alphanumeric_opaque_task_ids() -> None:
    assignments = _factorial_assignments()
    for assignment in assignments:
        assignment["task_id"] = "TASK2F35B95DDAC94E8BA70C06C554D995DC"

    schedule = TrialEvalAblationScheduleV1.model_validate(
        _schedule_payload(design="factorial_interface", assignments=assignments)
    )

    assert {row.task_id for row in schedule.assignments} == {"TASK2F35B95DDAC94E8BA70C06C554D995DC"}

    endpoint = TrialEvalAblationEndpointRowV1.model_validate(
        {
            "assignment_id": "assignment-ns",
            "task_id": "TASK2F35B95DDAC94E8BA70C06C554D995DC",
            "context_tier": "C1",
            "data_preparation": "analysis_ready",
            "analysis_specification": "locked_sap",
            "model_id": "model-1",
            "replicate_id": "seed-1",
            "procedure_assistance": "output_contract_only",
            "prompt_condition": "neutral",
            "submission_interface": "structured",
            "normalization_source": "direct_structured",
            "usable_primary": True,
            "route_match": True,
            "obligations_met": True,
            "credit_eligible_route_count": 1,
            "numeric_result_available": True,
            "result_match": True,
            "primary_analysis_conforms": True,
            "planning_applicable": False,
        }
    )
    assert endpoint.task_id == "TASK2F35B95DDAC94E8BA70C06C554D995DC"

    evaluator = TrialEvalAblationEvaluatorLabelsV1.model_validate(
        {
            "evaluator_release_sha256": "e" * 64,
            "task_identities": [
                {
                    "task_id": "TASK2F35B95DDAC94E8BA70C06C554D995DC",
                    "base_trial_id": "trial-1",
                    "regime_cell_id": "family-1",
                    "evaluation_series_id": "randomized",
                    "design_tier": "D1",
                    "design_subtype": "individual_randomized",
                    "assumption_tier": "A1",
                    "context_tier": "C1",
                    "data_preparation": "analysis_ready",
                    "analysis_specification": "locked_sap",
                }
            ],
        }
    )
    assert evaluator.task_identities[0].task_id == "TASK2F35B95DDAC94E8BA70C06C554D995DC"


def test_ablation_contracts_reject_non_alphanumeric_task_ids() -> None:
    assignment = _assignment(condition="neutral", interface="structured", suffix="ns")
    assignment["task_id"] = "TASK/escape"
    with pytest.raises(ValidationError, match="task_id"):
        TrialEvalAblationAssignmentV1.model_validate(assignment)


def test_ablation_endpoint_rejects_planning_consequences_on_ineligible_tasks() -> None:
    with pytest.raises(ValidationError, match="must be absent"):
        TrialEvalAblationEndpointRowV1.model_validate(
            {
                "assignment_id": "assignment-planning",
                "task_id": "TASK1001",
                "context_tier": "C1",
                "data_preparation": "analysis_ready",
                "analysis_specification": "locked_sap",
                "model_id": "model-1",
                "replicate_id": "seed-1",
                "procedure_assistance": "output_contract_only",
                "prompt_condition": "neutral",
                "submission_interface": "structured",
                "normalization_source": "direct_structured",
                "usable_primary": True,
                "route_match": True,
                "obligations_met": True,
                "credit_eligible_route_count": 1,
                "numeric_result_available": True,
                "result_match": True,
                "primary_analysis_conforms": True,
                "planning_applicable": False,
                "planning_valid": True,
            }
        )


def test_ablation_endpoint_rejects_interval_agreement_without_valid_method_uncertainty() -> None:
    with pytest.raises(ValidationError, match="matched method route and valid uncertainty"):
        TrialEvalAblationEndpointRowV1.model_validate(
            {
                "assignment_id": "assignment-interval",
                "task_id": "TASK1001",
                "context_tier": "C1",
                "data_preparation": "analysis_ready",
                "analysis_specification": "locked_sap",
                "model_id": "model-1",
                "replicate_id": "seed-1",
                "procedure_assistance": "output_contract_only",
                "prompt_condition": "neutral",
                "submission_interface": "structured",
                "normalization_source": "direct_structured",
                "usable_primary": False,
                "route_match": False,
                "obligations_met": False,
                "credit_eligible_route_count": 1,
                "numeric_result_available": True,
                "result_match": False,
                "primary_uncertainty_valid": False,
                "primary_interval_agreement": 0.5,
                "primary_analysis_conforms": False,
                "primary_failure_code": "unrecognized_primary_route",
                "planning_applicable": False,
            }
        )


def test_targeted_schedule_crosses_every_control_over_every_task() -> None:
    conditions = (
        "neutral",
        "targeted_covariate_structure",
        "targeted_survival_assumptions",
        "targeted_design_structure",
        "targeted_data_integrity",
        "placebo_deliberation",
    )
    assignments = [
        _assignment(
            condition=condition,
            interface="structured",
            suffix=f"{specification}-{index}",
            specification=specification,
        )
        for specification in ("locked_sap",)
        for index, condition in enumerate(conditions)
    ]
    TrialEvalAblationScheduleV1.model_validate(_schedule_payload(design="targeted_control", assignments=assignments))

    assignments[1]["submission_interface"] = "narrative"
    with pytest.raises(ValidationError, match="must use the structured"):
        TrialEvalAblationScheduleV1.model_validate(
            _schedule_payload(design="targeted_control", assignments=assignments)
        )


def test_schedule_rejects_evaluator_metadata() -> None:
    payload = _assignment(condition="neutral", interface="structured", suffix="ns")
    payload["accepted_method_id"] = "coxph_binary"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrialEvalAblationAssignmentV1.model_validate(payload)


def test_evaluator_labels_require_every_capability_for_each_task() -> None:
    labels = [
        {
            "task_id": "TASK1001",
            "prompt_condition": condition,
            "applicability": "inapplicable",
            "evidence_basis": ["no_targeted_capability_domain_active"],
        }
        for condition in (
            "targeted_covariate_structure",
            "targeted_survival_assumptions",
            "targeted_design_structure",
            "targeted_data_integrity",
        )
    ]
    artifact = TrialEvalAblationEvaluatorLabelsV1.model_validate(
        {
            "evaluator_release_sha256": "e" * 64,
            "task_identities": [
                {
                    "task_id": "TASK1001",
                    "base_trial_id": "trial-1",
                    "regime_cell_id": "family-1",
                    "evaluation_series_id": "randomized",
                    "design_tier": "D1",
                    "design_subtype": "individual_randomized",
                    "assumption_tier": "A1",
                    "context_tier": "C1",
                    "data_preparation": "analysis_ready",
                    "analysis_specification": "locked_sap",
                }
            ],
            "labels": labels,
        }
    )
    assert artifact.checksum is not None

    with pytest.raises(ValidationError, match="requires all capability conditions"):
        TrialEvalAblationEvaluatorLabelsV1.model_validate(
            {
                "evaluator_release_sha256": "e" * 64,
                "task_identities": [
                    {
                        "task_id": "TASK1001",
                        "base_trial_id": "trial-1",
                        "regime_cell_id": "family-1",
                        "evaluation_series_id": "randomized",
                        "design_tier": "D1",
                        "design_subtype": "individual_randomized",
                        "assumption_tier": "A1",
                        "context_tier": "C1",
                        "data_preparation": "analysis_ready",
                        "analysis_specification": "locked_sap",
                    }
                ],
                "labels": labels[:-1],
            }
        )


def test_schedule_rejects_order_not_derived_from_randomization_seed() -> None:
    assignments = _factorial_assignments()
    payload = _schedule_payload(design="factorial_interface", assignments=assignments)
    randomized = payload["assignments"]
    assert isinstance(randomized, list)
    payload["assignments"] = list(reversed(randomized))

    with pytest.raises(ValidationError, match="does not match randomization_seed"):
        TrialEvalAblationScheduleV1.model_validate(payload)


def test_schedule_rejects_decoding_seed_drift_within_paired_block() -> None:
    assignments = _factorial_assignments()
    assignments[1]["decoding_seed"] = 102

    with pytest.raises(ValidationError, match="one decoding seed across all cells"):
        TrialEvalAblationScheduleV1.model_validate(
            _schedule_payload(design="factorial_interface", assignments=assignments)
        )


def test_schedule_rejects_replicate_seed_drift_across_tasks() -> None:
    first_task = _factorial_assignments(suffix_prefix="1-")
    second_task = [
        {
            **row,
            "assignment_id": row["assignment_id"].replace("assignment-1", "assignment-2"),
            "task_id": "TASK1002",
            "decoding_seed": 202,
        }
        for row in first_task
    ]

    with pytest.raises(ValidationError, match="one decoding seed across all tasks"):
        TrialEvalAblationScheduleV1.model_validate(
            _schedule_payload(
                design="factorial_interface",
                assignments=first_task + second_task,
            )
        )


def test_schedule_rejects_duplicate_seed_across_replicates() -> None:
    assignments: list[dict[str, str]] = []
    for replicate_id, suffix_prefix in (("seed-1", "1"), ("seed-2", "2")):
        for specification in ("locked_sap",):
            for assistance in ("output_contract_only", "unordered_checklist", "ordered_sop"):
                for interface in ("structured", "narrative"):
                    suffix = f"{specification}-{assistance}-{interface}"
                    assignment = _assignment(
                        condition="neutral",
                        interface=interface,
                        assistance=assistance,
                        suffix=f"{suffix_prefix}-{suffix}",
                        specification=specification,
                    )
                    assignment["replicate_id"] = replicate_id
                    assignments.append(assignment)

    with pytest.raises(ValidationError, match="decoding seeds must be unique"):
        TrialEvalAblationScheduleV1.model_validate(
            _schedule_payload(design="factorial_interface", assignments=assignments)
        )


def test_prompt_conditions_are_global_and_interfaces_have_distinct_affordance(tmp_path: Path) -> None:
    item = BenchmarkItem(
        item_id="TASK1001",
        trial_name="TASK1001",
        design_tier="undisclosed",
        design_subtype="individual_randomized",
        assumption_tier="undisclosed",
        context_tier="C1",
        data_preparation="analysis_ready",
        analysis_specification="locked_sap",
        visible_dir=tmp_path,
        data_dir=tmp_path / "data",
        task={},
        task_id="TASK1001",
    )
    neutral = _build_system_prompt(item, max_turns=7)
    narrative = _build_system_prompt(item, max_turns=7, submission_interface="narrative")
    contract = _build_system_prompt(item, procedure_assistance="unordered_checklist")
    sop = _build_system_prompt(item, procedure_assistance="ordered_sop")
    forbidden = ("cox", "rmst", "ipcw", "proportional hazards", "confound")
    assistance_text = " ".join(
        (
            procedure_assistance_v1("output_contract_only"),
            procedure_assistance_v1("unordered_checklist"),
            procedure_assistance_v1("ordered_sop"),
        )
    ).lower()
    assert all(term not in assistance_text for term in forbidden)
    for term in forbidden:
        assert neutral.lower().count(term) == contract.lower().count(term) == sop.lower().count(term)
    assert "at most 7 turns" in neutral
    assert "scikit-learn" in neutral
    assert "matplotlib" in neutral
    assert "interface/submission_shapes.json" in neutral
    assert "interface/submission_shapes.json" not in narrative
    assert "After choosing a method, consider only the diagnostics named by that method" in neutral
    assert "replace every example identifier, method, value, source, and interpretation" in neutral
    assert neutral != contract != sop
    assert "method selection must follow the protocol and supplied data" in contract
    assert "same operations in this required order" in sop
    assert "(1) define the scientific question" in sop
    prohibited = ("benchmark", "evaluator", "fake", "ground truth", "oracle", "simulat", "synthetic")
    for participant_instruction in (neutral, narrative, contract, sop):
        assert all(token not in participant_instruction.lower() for token in prohibited)

    tools = _get_tools("structured")
    assert [tool["function"]["name"] for tool in tools[-2:]] == [
        "submit_response",
        "submit_response_file",
    ]
    assert "strict" not in tools[-2]["function"]
    assert "estimator" in json.dumps(tools[-2]).lower()

    inventory = schema_affordance_inventory_v1()
    by_interface = {row.submission_interface: row for row in inventory.interfaces}
    assert "coxph_binary" not in by_interface["structured"].enum_vocabulary
    assert "coxph_binary" not in by_interface["narrative"].enum_vocabulary
    assert "coxph_binary" not in inventory.shared_scientific_vocabulary
    assert len(by_interface["structured"].field_paths) > len(by_interface["narrative"].field_paths)
    assert by_interface["structured"].tool_schema_sha256 != by_interface["narrative"].tool_schema_sha256
    assert by_interface["structured"].response_contract_sha256 != by_interface["narrative"].response_contract_sha256

    contrasts = condition_contrasts_markdown_v1()
    assert f"Canonical prompt-set SHA-256: `{prompt_set_sha256_v1()}`." in contrasts
    assert "## P0 versus P1" in contrasts
    assert "## P1 versus P2" in contrasts
    assert "## Structured versus narrative instruction" in contrasts
    assert "## Structured versus narrative interface affordance" in contrasts
    assert "+A complete analysis must perform these operations in any order" in contrasts
    assert "+Perform the same operations in this required order" in contrasts


@pytest.mark.parametrize("interface", ["structured", "narrative"])
def test_response_contract_is_staged_and_tamper_evident(
    tmp_path: Path,
    interface: TrialEvalSubmissionInterfaceV1,
) -> None:
    path = stage_response_contract_v1(tmp_path, interface)

    assert path == tmp_path / "interface" / "response_contract.json"
    assert response_contract_sha256_v1(interface) == next(
        row.response_contract_sha256
        for row in schema_affordance_inventory_v1().interfaces
        if row.submission_interface == interface
    )
    assert stage_response_contract_v1(tmp_path, interface) == path
    shapes_path = tmp_path / "interface" / "submission_shapes.json"
    if interface == "structured":
        assert json.loads(
            shapes_path.read_text(encoding="utf-8")
        ) == trialeval_submission_shape_catalogue().model_dump(mode="json")
    else:
        assert not shapes_path.exists()

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the declared response interface"):
        stage_response_contract_v1(tmp_path, interface)


def test_structured_submission_shape_catalogue_is_tamper_evident(tmp_path: Path) -> None:
    """The ergonomic catalogue remains generated from the canonical models."""

    stage_response_contract_v1(tmp_path, "structured")
    shapes_path = tmp_path / "interface" / "submission_shapes.json"
    shapes_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="shape catalogue differs"):
        stage_response_contract_v1(tmp_path, "structured")


def test_participant_discovery_does_not_require_evaluator_tree(tmp_path: Path) -> None:
    participant = tmp_path / "public"
    item_root = participant / "items" / "TASK1001"
    (item_root / "data").mkdir(parents=True)
    (participant / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_public_bundle_manifest/v1",
                "applied_baseline_profile_id": None,
                "applied_baseline_profile_sha256": None,
                "task_ids": ["TASK1001"],
                "task_evidence_factors": {
                    "TASK1001": {
                        "context_configuration": "C1",
                        "data_preparation": "analysis_ready",
                        "analysis_specification": "locked_sap",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (item_root / "task.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_task_v1",
                "task_id": "TASK1001",
                "design_subtype": "individual_randomized",
                "primary_endpoint_id": "endpoint",
                "primary_paramcd": "endpoint",
                "primary_estimand_id": "estimand",
                "primary_effect_scale": "risk_difference_tau",
                "estimand_mode": "fixed_declared_estimand",
                "primary_effect_scale_options": ["risk_difference_tau"],
                "primary_result_unit": "probability_difference",
                "primary_tau_dy": 365.0,
                "primary_population_id": "itt",
                "primary_intercurrent_event_strategy_ids": ["treatment_policy"],
                "primary_control_arm_id": "control",
                "primary_treated_arm_id": "treated",
            }
        ),
        encoding="utf-8",
    )
    (item_root / "submission_contract.json").write_text(
        json.dumps(minimal_participant_output_contract("TASK1001")),
        encoding="utf-8",
    )
    write_minimal_trialeval_release_dictionaries(participant)

    items = discover_participant_items(participant, task_ids=("TASK1001",))
    assert items["TASK1001"].context_tier == "C1"
    assert items["TASK1001"].suite_dir == participant
