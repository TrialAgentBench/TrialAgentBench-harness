"""Tests for participant-only TrialEval ablation schedule construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)

from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationAnalysisConfigV1,
    TrialEvalExperimentProtocolV1,
    TrialEvalFactorialTaskSampleV1,
)
from trialagentbench_harness.experiments.build_trialeval_ablation_schedule import (
    _parse_replicate_seeds,
    build_trialeval_ablation_schedule_v1,
)
from trialagentbench_harness.io import sha256_dir_digest

_HARNESS_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT_DESIGN = TrialEvalExperimentProtocolV1.model_validate_json(
    (_HARNESS_ROOT / "experiment_configs/trialeval_experiment_protocol_v1.json").read_text(encoding="utf-8")
)


def _analysis_config(
    *, execution_scope: Literal["canary", "pilot", "publication"] = "pilot"
) -> TrialEvalAblationAnalysisConfigV1:
    return TrialEvalAblationAnalysisConfigV1(
        design="factorial_interface",
        execution_scope=execution_scope,
        experiment_design_sha256=_EXPERIMENT_DESIGN.checksum,
        primary_estimand={
            "metric": "usable_primary",
            "contrast_id": "P2-P1",
            "analysis_specification": "protocol_only",
        },
        supporting_metrics=("primary_analysis_conforms",),
        confidence_level=0.9,
        bootstrap_resamples=1000,
        bootstrap_seed=9,
        min_base_trial_clusters=(
            _EXPERIMENT_DESIGN.precision.retained_independent_base_trials
            if execution_scope == "publication"
            else 1 if execution_scope == "canary" else 2
        ),
        min_decoding_replicates=1 if execution_scope == "canary" else 2,
    )


def _targeted_analysis_config() -> TrialEvalAblationAnalysisConfigV1:
    return TrialEvalAblationAnalysisConfigV1(
        design="targeted_control",
        execution_scope="canary",
        experiment_design_sha256=_EXPERIMENT_DESIGN.checksum,
        primary_estimand={
            "metric": "primary_analysis_conforms",
            "contrast_id": "targeted_vs_neutral",
            "analysis_specification": "protocol_only",
            "prompt_condition": "targeted_covariate_structure",
            "applicability": "applicable",
        },
        confidence_level=0.9,
        bootstrap_resamples=1000,
        bootstrap_seed=9,
        min_base_trial_clusters=1,
        min_decoding_replicates=1,
    )


def _participant_release(tmp_path: Path, *, task_contexts: dict[str, str] | None = None) -> Path:
    root = tmp_path / "public"
    contexts = task_contexts or {"TASK1001": "C1", "TASK1002": "C4"}
    factors = {
        "C1": ("analysis_ready", "locked_sap"),
        "C2": ("analysis_ready", "protocol_only"),
        "C3": ("raw_domains", "locked_sap"),
        "C4": ("raw_domains", "protocol_only"),
        "C5": ("raw_domains_declared_defect", "protocol_only"),
    }
    for task_id in contexts:
        (root / "items" / task_id).mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_public_bundle_manifest/v1",
                "applied_baseline_profile_id": None,
                "applied_baseline_profile_sha256": None,
                "task_ids": list(reversed(contexts)),
                "task_evidence_factors": {
                    task_id: {
                        "context_configuration": context,
                        "data_preparation": factors[context][0],
                        "analysis_specification": factors[context][1],
                    }
                    for task_id, context in contexts.items()
                },
            }
        ),
        encoding="utf-8",
    )
    for task_id in contexts:
        (root / "items" / task_id / "task.json").write_text(
            json.dumps(
                {
                    "schema_id": "trial_analysis_task_v1",
                    "task_id": task_id,
                    "design_subtype": "individual_randomized",
                    "primary_endpoint_id": "endpoint",
                    "primary_paramcd": "endpoint",
                    "primary_estimand_id": "estimand",
                    "primary_effect_scale": "risk_difference_tau",
                    "estimand_mode": "fixed_declared_estimand",
                    "primary_effect_scale_options": ["risk_difference_tau"],
                    "primary_result_unit": "probability_difference",
                    "primary_population_id": "itt",
                    "primary_intercurrent_event_strategy_ids": ["treatment_policy"],
                    "primary_tau_dy": 365.0,
                    "primary_control_arm_id": "control",
                    "primary_treated_arm_id": "treated",
                }
            ),
            encoding="utf-8",
        )
        (root / "items" / task_id / "submission_contract.json").write_text(
            json.dumps(
                minimal_participant_output_contract(
                    task_id,
                    data_preparation=factors[contexts[task_id]][0],
                )
            ),
            encoding="utf-8",
        )
    write_minimal_trialeval_release_dictionaries(root)
    return root


def _publication_task_contexts() -> dict[str, str]:
    contexts = tuple(
        context
        for context, count in zip(
            ("C1", "C2", "C3", "C4", "C5"),
            _EXPERIMENT_DESIGN.compute_envelope.factorial_context_allocation,
            strict=True,
        )
        for _ in range(count)
    )
    return {f"TASK{index:04d}": context for index, context in enumerate(contexts, start=1)}


def _factorial_task_sample(participant: Path, task_contexts: dict[str, str]) -> TrialEvalFactorialTaskSampleV1:
    return TrialEvalFactorialTaskSampleV1(
        experiment_design_sha256=_EXPERIMENT_DESIGN.checksum,
        participant_release_sha256=sha256_dir_digest(participant),
        evaluator_labels_sha256="e" * 64,
        task_ids=tuple(task_contexts),
        context_allocation=(
            sum(context == "C1" for context in task_contexts.values()),
            sum(context == "C2" for context in task_contexts.values()),
            sum(context == "C3" for context in task_contexts.values()),
            sum(context == "C4" for context in task_contexts.values()),
            sum(context == "C5" for context in task_contexts.values()),
        ),
    )


def test_schedule_builder_fully_crosses_participant_tasks_without_evaluator_metadata(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    schedule = build_trialeval_ablation_schedule_v1(
        participant_root=participant,
        experiment_id="factorial-1",
        design="factorial_interface",
        experiment_design=_EXPERIMENT_DESIGN,
        analysis_config=_analysis_config(),
        replicate_seeds={"seed-2": 202, "seed-1": 101},
        randomization_seed=17,
    )

    assert len(schedule.assignments) == 24
    assert {row.context_tier for row in schedule.assignments if row.task_id == "TASK1001"} == {"C1"}
    assert {row.context_tier for row in schedule.assignments if row.task_id == "TASK1002"} == {"C4"}
    assert {row.analysis_specification for row in schedule.assignments if row.task_id == "TASK1001"} == {"locked_sap"}
    assert {row.analysis_specification for row in schedule.assignments if row.task_id == "TASK1002"} == {
        "protocol_only"
    }
    assert all(row.analysis_surface_sha256 for row in schedule.assignments)
    assert len({row.analysis_surface_sha256 for row in schedule.assignments if row.task_id == "TASK1001"}) == 1
    assert schedule.participant_release_sha256
    assert schedule.prompt_set_sha256
    assert schedule.analysis_config_sha256 == _analysis_config().checksum
    assert schedule.randomization_seed == 17
    assert {row.decoding_seed for row in schedule.assignments if row.replicate_id == "seed-1"} == {101}
    assert {row.decoding_seed for row in schedule.assignments if row.replicate_id == "seed-2"} == {202}
    for assignment in schedule.assignments:
        assert assignment.assignment_id.startswith("A")
        assert len(assignment.assignment_id) == 33
        assert assignment.task_id not in assignment.assignment_id
        assert assignment.replicate_id not in assignment.assignment_id
        assert assignment.procedure_assistance not in assignment.assignment_id
        assert assignment.prompt_condition not in assignment.assignment_id
        assert assignment.submission_interface not in assignment.assignment_id


def test_canary_schedule_runs_one_complete_randomized_block(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    schedule = build_trialeval_ablation_schedule_v1(
        participant_root=participant,
        experiment_id="factorial-canary",
        design="factorial_interface",
        experiment_design=_EXPERIMENT_DESIGN,
        analysis_config=_analysis_config(execution_scope="canary"),
        replicate_seeds={"seed-1": 101},
        randomization_seed=17,
        task_ids=("TASK1002",),
    )

    assert schedule.execution_scope == "canary"
    assert len(schedule.assignments) == 6
    assert {(row.task_id, row.replicate_id) for row in schedule.assignments} == {("TASK1002", "seed-1")}


def test_targeted_canary_uses_prespecified_prompt_contrasts(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    schedule = build_trialeval_ablation_schedule_v1(
        participant_root=participant,
        experiment_id="targeted-canary",
        design="targeted_control",
        experiment_design=_EXPERIMENT_DESIGN,
        analysis_config=_targeted_analysis_config(),
        replicate_seeds={"seed-1": 101},
        randomization_seed=17,
        task_ids=("TASK1002",),
    )

    assert {row.prompt_condition for row in schedule.assignments} == {
        "neutral",
        "placebo_deliberation",
        "targeted_covariate_structure",
        "targeted_data_integrity",
        "targeted_design_structure",
        "targeted_survival_assumptions",
    }
    assert {row.submission_interface for row in schedule.assignments} == {"structured"}


def test_publication_schedule_matches_frozen_compute_and_context_design(tmp_path: Path) -> None:
    contexts = _publication_task_contexts()
    participant = _participant_release(tmp_path, task_contexts=contexts)
    schedule = build_trialeval_ablation_schedule_v1(
        participant_root=participant,
        experiment_id="factorial-publication",
        design="factorial_interface",
        experiment_design=_EXPERIMENT_DESIGN,
        analysis_config=_analysis_config(execution_scope="publication"),
        replicate_seeds={"seed-1": 101, "seed-2": 202},
        randomization_seed=17,
        task_sample=_factorial_task_sample(participant, contexts),
    )

    assert schedule.execution_scope == "publication"
    assert schedule.experiment_design_sha256 == _EXPERIMENT_DESIGN.checksum
    assert len(schedule.assignments) == _EXPERIMENT_DESIGN.compute_envelope.factorial_assignments_per_model
    observed = {
        context: len({row.task_id for row in schedule.assignments if row.context_tier == context})
        for context in ("C1", "C2", "C3", "C4", "C5")
    }
    assert tuple(observed[context] for context in ("C1", "C2", "C3", "C4", "C5")) == (
        _EXPERIMENT_DESIGN.compute_envelope.factorial_context_allocation
    )


def test_publication_schedule_rejects_unbalanced_context_sample(tmp_path: Path) -> None:
    contexts = _publication_task_contexts()
    c5_task = next(task_id for task_id, context in contexts.items() if context == "C5")
    contexts[c5_task] = "C4"
    participant = _participant_release(tmp_path, task_contexts=contexts)

    with pytest.raises(ValueError, match="context allocation"):
        build_trialeval_ablation_schedule_v1(
            participant_root=participant,
            experiment_id="factorial-publication",
            design="factorial_interface",
            experiment_design=_EXPERIMENT_DESIGN,
            analysis_config=_analysis_config(execution_scope="publication"),
            replicate_seeds={"seed-1": 101, "seed-2": 202},
            randomization_seed=17,
            task_sample=_factorial_task_sample(participant, contexts),
        )


def test_schedule_builder_rejects_incomplete_participant_context_index(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    manifest = json.loads((participant / "manifest.json").read_text(encoding="utf-8"))
    manifest["task_evidence_factors"].pop("TASK1002")
    (participant / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="task_evidence_factors"):
        build_trialeval_ablation_schedule_v1(
            participant_root=participant,
            experiment_id="factorial-1",
            design="factorial_interface",
            experiment_design=_EXPERIMENT_DESIGN,
            analysis_config=_analysis_config(),
            replicate_seeds={"seed-1": 101, "seed-2": 202},
            randomization_seed=17,
        )


def test_schedule_builder_rejects_task_subset_below_analysis_precision(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    with pytest.raises(ValueError, match="base-trial requirement"):
        build_trialeval_ablation_schedule_v1(
            participant_root=participant,
            experiment_id="factorial-1",
            design="factorial_interface",
            experiment_design=_EXPERIMENT_DESIGN,
            analysis_config=_analysis_config(),
            replicate_seeds={"seed-1": 101, "seed-2": 202},
            randomization_seed=17,
            task_ids=("TASK1002",),
        )


def test_schedule_builder_rejects_primary_estimand_outside_frozen_design(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    payload = _analysis_config().model_dump(mode="json", exclude={"checksum"})
    payload["primary_estimand"]["contrast_id"] = "invented-contrast"
    config = TrialEvalAblationAnalysisConfigV1.model_validate(
        payload,
    )

    with pytest.raises(ValueError, match="one frozen design contrast"):
        build_trialeval_ablation_schedule_v1(
            participant_root=participant,
            experiment_id="factorial-1",
            design="factorial_interface",
            experiment_design=_EXPERIMENT_DESIGN,
            analysis_config=config,
            replicate_seeds={"seed-1": 101, "seed-2": 202},
            randomization_seed=17,
        )


def test_schedule_builder_rejects_unknown_task_subset(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    with pytest.raises(ValueError, match="Unknown participant task_ids"):
        build_trialeval_ablation_schedule_v1(
            participant_root=participant,
            experiment_id="factorial-1",
            design="factorial_interface",
            experiment_design=_EXPERIMENT_DESIGN,
            analysis_config=_analysis_config(),
            replicate_seeds={"seed-1": 101, "seed-2": 202},
            randomization_seed=17,
            task_ids=("TASK9999",),
        )


def test_ablation_assignment_rejects_path_traversal_replicate(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)

    with pytest.raises(ValueError, match="replicate_id"):
        build_trialeval_ablation_schedule_v1(
            participant_root=participant,
            experiment_id="factorial-1",
            design="factorial_interface",
            experiment_design=_EXPERIMENT_DESIGN,
            analysis_config=_analysis_config(),
            replicate_seeds={"../escape": 101},
            randomization_seed=17,
        )


def test_schedule_builder_randomizes_reproducibly_and_checksums_execution_order(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    first = build_trialeval_ablation_schedule_v1(
        participant_root=participant,
        experiment_id="factorial-1",
        design="factorial_interface",
        experiment_design=_EXPERIMENT_DESIGN,
        analysis_config=_analysis_config(),
        replicate_seeds={"seed-1": 101, "seed-2": 202},
        randomization_seed=17,
    )
    repeated = build_trialeval_ablation_schedule_v1(
        participant_root=participant,
        experiment_id="factorial-1",
        design="factorial_interface",
        experiment_design=_EXPERIMENT_DESIGN,
        analysis_config=_analysis_config(),
        replicate_seeds={"seed-1": 101, "seed-2": 202},
        randomization_seed=17,
    )
    different = build_trialeval_ablation_schedule_v1(
        participant_root=participant,
        experiment_id="factorial-1",
        design="factorial_interface",
        experiment_design=_EXPERIMENT_DESIGN,
        analysis_config=_analysis_config(),
        replicate_seeds={"seed-1": 101, "seed-2": 202},
        randomization_seed=18,
    )

    assert first.assignments == repeated.assignments
    assert first.checksum == repeated.checksum
    assert first.assignments != different.assignments
    assert first.checksum != different.checksum


def test_schedule_builder_rejects_negative_randomization_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="randomization_seed"):
        build_trialeval_ablation_schedule_v1(
            participant_root=_participant_release(tmp_path),
            experiment_id="factorial-1",
            design="factorial_interface",
            experiment_design=_EXPERIMENT_DESIGN,
            analysis_config=_analysis_config(),
            replicate_seeds={"seed-1": 101, "seed-2": 202},
            randomization_seed=-1,
        )


@pytest.mark.parametrize(
    "replicate_seeds",
    (
        {},
        {"": 101},
        {"seed-1": -1},
        {"seed-1": True},
        {"seed-1": 101, "seed-2": 101},
    ),
)
def test_schedule_builder_rejects_invalid_replicate_seed_map(
    tmp_path: Path,
    replicate_seeds: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="replicate_seeds"):
        build_trialeval_ablation_schedule_v1(
            participant_root=_participant_release(tmp_path),
            experiment_id="factorial-1",
            design="factorial_interface",
            experiment_design=_EXPERIMENT_DESIGN,
            analysis_config=_analysis_config(),
            replicate_seeds=replicate_seeds,
            randomization_seed=17,
        )


def test_replicate_seed_cli_parser_rejects_ambiguous_declarations() -> None:
    assert _parse_replicate_seeds(("seed-1=101", "seed-2=202")) == {
        "seed-1": 101,
        "seed-2": 202,
    }

    for values in (
        ("seed-1",),
        ("=101",),
        ("seed-1=-1",),
        ("seed-1=1.5",),
        ("seed-1=101", "seed-1=202"),
    ):
        with pytest.raises(ValueError, match="replicate"):
            _parse_replicate_seeds(values)
