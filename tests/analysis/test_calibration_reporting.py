"""Denominator-complete calibration reporting tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from trialagentbench.analysis.calibration import (
    CalibrationConditionV1,
    CalibrationPanelV1,
    CalibrationUnitV1,
    _trialdev_turns_used,
    calibration_cells_v1,
    validate_calibration_panel_v1,
)


def _unit(
    unit_id: str,
    *,
    execution_status: str,
) -> CalibrationUnitV1:
    return CalibrationUnitV1.model_validate(
        {
            "benchmark": "trialdev",
            "run_id": "run-001",
            "run_identity_sha256": "a" * 64,
            "release_identity": "release_" + "b" * 64,
            "unit_id": unit_id,
            "model": "example/model",
            "provider": "openrouter",
            "reasoning_effort": "medium",
            "turn_limit": 12,
            "completion_token_limit": 4096,
            "context_character_limit": 120_000,
            "send_temperature": False,
            "request_timeout_seconds": 300.0,
            "watchdog_seconds": 1800.0,
            "provider_maximum_attempts": 5,
            "provider_retry_base_delay_seconds": 2.0,
            "reported_cost_stop_usd": None,
            "executor_image": "trialagentbench/executor:1.0.0@sha256:" + "e" * 64,
            "executor_limits_sha256": "f" * 64,
            "worker_count": 1,
            "execution_status": execution_status,
            "provider_response_count": 2,
            "provider_failure_count": int(execution_status == "evaluator_error"),
            "analysis_prompt_tokens": 100,
            "analysis_completion_tokens": 20,
            "analysis_reported_cost_usd": 0.01,
            "analysis_elapsed_seconds": 2.0,
        }
    )


def _condition(
    condition_id: str,
    *,
    procedure_assistance: Literal["output_contract_only", "unordered_checklist"] = "output_contract_only",
) -> CalibrationConditionV1:
    return CalibrationConditionV1(
        condition_id=condition_id,
        provider="openrouter",
        reasoning_effort="medium",
        procedure_assistance=procedure_assistance,
        turn_limit=12,
        completion_token_limit=4096,
        context_character_limit=120_000,
        send_temperature=False,
        request_timeout_seconds=300.0,
        watchdog_seconds=1800.0,
        provider_maximum_attempts=5,
        provider_retry_base_delay_seconds=2.0,
        executor_image="trialagentbench/executor:1.0.0@sha256:" + "e" * 64,
        executor_limits_sha256="f" * 64,
        worker_count=1,
    )


def test_calibration_cells_retain_failures_and_unfinished_schedule_rows() -> None:
    """Never calculate apparent performance from completed or scored units alone."""

    units = (
        _unit("td-000000000001", execution_status="completed"),
        _unit("td-000000000002", execution_status="completed"),
        _unit("td-000000000003", execution_status="model_noncompletion"),
        _unit("td-000000000004", execution_status="not_started"),
    )

    (cell,) = calibration_cells_v1(units)

    assert cell.scheduled_requests == 4
    assert cell.independent_unit_count == 4
    assert cell.execution_completed == 2
    assert cell.model_noncompletion == 1
    assert cell.analysis_tokens == 480
    assert cell.observed_cost_usd == 0.04
    assert not {"scientifically_scored", "passed", "request_pass_rate"} & cell.model_dump().keys()


def test_calibration_cells_keep_completed_execution_explicit() -> None:
    """Keep execution completion separate from scientific performance."""

    (cell,) = calibration_cells_v1(
        (
            _unit(
                "td-000000000001",
                execution_status="completed",
            ),
        )
    )

    assert cell.execution_completed == cell.scheduled_requests == 1


def test_calibration_resource_totals_preserve_small_contributions() -> None:
    """Retain resource totals across summation order and supported Python versions."""
    units = tuple(
        _unit(f"td-{index:012d}", execution_status="completed").model_copy(
            update={"analysis_reported_cost_usd": value, "analysis_elapsed_seconds": value}
        )
        for index, value in enumerate((1e16, 1.0, 1.0))
    )
    for ordered in (units, tuple(reversed(units))):
        (cell,) = calibration_cells_v1(ordered)
        assert cell.observed_cost_usd == 10000000000000002.0
        assert cell.observed_elapsed_seconds == 10000000000000002.0


def test_model_noncompletion_remains_in_resource_denominators() -> None:
    """Execution status and scientific score describe different properties."""

    (cell,) = calibration_cells_v1(
        (
            _unit(
                "td-000000000001",
                execution_status="model_noncompletion",
            ),
        )
    )

    assert cell.scheduled_requests == 1
    assert cell.model_noncompletion == 1


def test_calibration_cells_do_not_pool_distinct_interaction_conditions() -> None:
    """Keep memory, interface, and turn-budget interventions in separate cells."""

    bounded = _unit("td-000000000001", execution_status="completed")
    compacted = bounded.model_copy(
        update={
            "unit_id": "td-000000000002",
            "condition_id": "compacted",
            "memory_strategy": "openai_responses_compaction",
        }
    )

    cells = calibration_cells_v1((bounded, compacted))

    assert len(cells) == 2
    assert {cell.memory_strategy for cell in cells} == {"bounded", "openai_responses_compaction"}
    assert all(cell.scheduled_requests == 1 for cell in cells)


def test_calibration_cells_ignore_labels_but_separate_material_run_factors() -> None:
    """Human labels identify runs without acting as scientific factors."""

    first = _unit("td-000000000001", execution_status="completed")
    relabelled_repeat = first.model_copy(
        update={
            "run_identity_sha256": "c" * 64,
            "condition_id": "human-label-b",
            "replicate_id": "request-2",
        }
    )

    (pooled,) = calibration_cells_v1((first, relabelled_repeat))

    assert pooled.condition_ids == ("human-label-b", "unspecified")
    assert pooled.scheduled_requests == 2
    assert pooled.independent_unit_count == 1
    changed_route = relabelled_repeat.model_copy(
        update={
            "run_identity_sha256": "d" * 64,
            "provider": "openai_responses",
        }
    )
    changed_release = relabelled_repeat.model_copy(
        update={
            "run_identity_sha256": "e" * 64,
            "release_identity": "release_" + "f" * 64,
        }
    )
    changed_seed = relabelled_repeat.model_copy(
        update={
            "run_identity_sha256": "f" * 64,
            "decoding_seed": 19,
        }
    )
    cells = calibration_cells_v1((first, changed_route, changed_release, changed_seed))

    assert len(cells) == 4


def test_calibration_panel_requires_complete_matched_request_schedule() -> None:
    """Manual public-run invocations remain bound to one prospective panel."""

    units = tuple(
        _unit(unit_id, execution_status="completed").model_copy(
            update={
                "run_identity_sha256": f"{index:064x}",
                "condition_id": condition,
                "replicate_id": replicate,
            }
        )
        for index, (condition, replicate, unit_id) in enumerate(
            (
                ("clean", "request-1", "td-000000000001"),
                ("clean", "request-1", "td-000000000002"),
                ("checklist", "request-1", "td-000000000001"),
                ("checklist", "request-1", "td-000000000002"),
            ),
            start=1,
        )
    )
    panel = CalibrationPanelV1(
        panel_id="assistance",
        benchmark="trialdev",
        release_identity="release_" + "b" * 64,
        model="example/model",
        contrast_factor="procedure_assistance",
        conditions=(
            _condition("clean"),
            _condition("checklist", procedure_assistance="unordered_checklist"),
        ),
        request_replicate_ids=("request-1",),
        unit_ids=("td-000000000001", "td-000000000002"),
    )

    units = tuple(
        row.model_copy(
            update={
                "procedure_assistance": (
                    "unordered_checklist" if row.condition_id == "checklist" else "output_contract_only"
                )
            }
        )
        for row in units
    )

    validate_calibration_panel_v1(units, panel=panel)
    with pytest.raises(ValueError, match="missing requests"):
        validate_calibration_panel_v1(units[:-1], panel=panel)


def test_calibration_panel_rejects_condition_factor_drift() -> None:
    """A complete label grid cannot conceal an unmatched runtime intervention."""

    panel = CalibrationPanelV1(
        panel_id="assistance",
        benchmark="trialdev",
        release_identity="release_" + "b" * 64,
        model="example/model",
        contrast_factor="procedure_assistance",
        conditions=(
            _condition("clean"),
            _condition("checklist", procedure_assistance="unordered_checklist"),
        ),
        request_replicate_ids=("request-1",),
        unit_ids=("td-000000000001",),
    )
    clean = _unit("td-000000000001", execution_status="completed").model_copy(
        update={"condition_id": "clean", "replicate_id": "request-1"}
    )
    checklist = clean.model_copy(
        update={
            "run_identity_sha256": "c" * 64,
            "condition_id": "checklist",
            "procedure_assistance": "unordered_checklist",
            "memory_strategy": "openai_responses_compaction",
        }
    )

    with pytest.raises(ValueError, match="undeclared factor values"):
        validate_calibration_panel_v1((clean, checklist), panel=panel)


def test_calibration_panel_definition_allows_only_its_declared_contrast() -> None:
    """Reject prospectively confounded condition definitions before execution."""

    with pytest.raises(ValueError, match="differ outside contrast factor"):
        CalibrationPanelV1(
            panel_id="confounded",
            benchmark="trialdev",
            release_identity="release_" + "b" * 64,
            model="example/model",
            contrast_factor="procedure_assistance",
            conditions=(
                _condition("clean"),
                _condition("checklist", procedure_assistance="unordered_checklist").model_copy(
                    update={"memory_strategy": "openai_responses_compaction"}
                ),
            ),
            request_replicate_ids=("request-1",),
            unit_ids=("td-000000000001",),
        )


def test_documented_calibration_panel_is_executable() -> None:
    """Keep the release-native matched-panel example synchronized with its schema."""

    document = (Path(__file__).resolve().parents[2] / "examples" / "calibration_panel.json").read_text(
        encoding="utf-8"
    )
    panel = CalibrationPanelV1.model_validate_json(document)

    assert panel.contrast_factor == "procedure_assistance"
    assert tuple(condition.condition_id for condition in panel.conditions) == ("clean", "checklist")


def test_trialdev_turn_count_includes_resumed_conversation_segments(tmp_path) -> None:
    """Report total analysis turns across an interrupted and resumed task."""

    (tmp_path / "conversation.json").write_text(
        json.dumps([{"role": "user"}, {"role": "assistant"}, {"role": "tool"}]),
        encoding="utf-8",
    )
    (tmp_path / "conversation.resume-001.json").write_text(
        json.dumps([{"role": "assistant"}, {"role": "tool"}, {"role": "assistant"}]),
        encoding="utf-8",
    )

    assert _trialdev_turns_used(tmp_path) == 3
