from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from trialagentbench_harness.ports import CodeExecutionResultV1, ToolCall
from trialagentbench_harness.trialeval.agent import _handle_tool_call

_CORE_DELIVERABLES = ("evidence", "limitations", "primary_analysis")


class _Session:
    def execute(self, code: str) -> str:
        return code

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        return CodeExecutionResultV1(status="success", output=code, elapsed_seconds=0.1)

    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    "tool_call",
    [
        ToolCall(id="1", name="unknown", arguments="{}"),
        ToolCall(id="2", name="execute_code", arguments='{"code": 3}'),
        ToolCall(id="3", name="inspect_parquet", arguments='{"filename": "../grader/truth.parquet"}'),
        ToolCall(id="4", name="inspect_parquet", arguments='{"filename": "table.csv"}'),
        ToolCall(id="5", name="write_workspace_file", arguments='{"path": "../truth.json", "content": "x"}'),
        ToolCall(id="6", name="read_workspace_file", arguments='{"path": "x", "start_line": 2, "end_line": 1}'),
    ],
)
def test_trial_eval_tools_reject_invalid_calls(tool_call: ToolCall, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _handle_tool_call(
            tool_call,
            _Session(),
            tmp_path,
            False,
            submission_interface="structured",
            required_deliverables=_CORE_DELIVERABLES,
        )


def test_inspect_parquet_accepts_only_relative_parquet_paths(tmp_path: Path) -> None:
    output, submission, execution = _handle_tool_call(
        ToolCall(id="1", name="inspect_parquet", arguments='{"filename": "raw/adtte.parquet"}'),
        _Session(),
        tmp_path,
        False,
        submission_interface="structured",
        required_deliverables=_CORE_DELIVERABLES,
    )

    assert "pd.read_parquet('data/raw/adtte.parquet')" in output
    assert submission is None
    assert execution is not None
    assert execution.status == "success"


def test_execute_code_preserves_failure_status_for_runtime_trace(tmp_path: Path) -> None:
    class FailureSession(_Session):
        def execute_result(self, code: str) -> CodeExecutionResultV1:
            return CodeExecutionResultV1(
                status="execution_error",
                output="Traceback: failed",
                output_truncated=True,
                elapsed_seconds=0.25,
            )

    output, submission, execution = _handle_tool_call(
        ToolCall(id="1", name="execute_code", arguments='{"code": "raise RuntimeError()"}'),
        FailureSession(),
        tmp_path,
        False,
        submission_interface="structured",
        required_deliverables=_CORE_DELIVERABLES,
    )

    assert output == "Traceback: failed"
    assert submission is None
    assert execution is not None
    assert execution.status == "execution_error"
    assert execution.output_truncated is True


def _submission_payload() -> dict[str, object]:
    return {
        "task_id": "task-1",
        "primary_analysis": {
            "declared_primary": True,
            "estimand": {
                "estimand_id": "primary_itt",
                "population_id": "itt",
                "treatment_id": "treated",
                "comparator_id": "control",
                "endpoint_id": "death",
                "intercurrent_event_strategy_ids": ["death:composite"],
                "horizon_not_applicable_reason": "effect scale has no fixed horizon",
            },
            "estimator": {
                "analysis_method_id": "coxph_binary_wald",
                "implementation": "unadjusted Cox model",
                "qualifications": [
                    "independent_censoring",
                    "proportional_hazards",
                    "randomization_exchangeability",
                ],
            },
            "result_kind": "numeric_point",
            "result": {
                "kind": "scalar",
                "value": -0.2,
                "effect_scale": "log_hr",
                "unit": "log hazard ratio",
                "interval": {"lower": -0.4, "upper": 0.1, "confidence_level": 0.95},
            },
            "favorable_direction": "lower",
            "evidence_ids": ["support-1"],
        },
        "evidence": [
            {
                "evidence_id": "support-1",
                "evidence_type": "supporting_analysis",
                "principle": "uncertainty",
                "operation": "estimation",
                "estimator": {
                    "analysis_method_id": "coxph_binary_wald",
                    "implementation": "unadjusted Cox model",
                    "qualifications": [
                        "independent_censoring",
                        "proportional_hazards",
                        "randomization_exchangeability",
                    ],
                },
                "target": "robustness",
                "result": {
                    "kind": "scalar",
                    "value": -0.2,
                    "effect_scale": "log_hr",
                    "unit": "log hazard ratio",
                    "interval": {"lower": -0.4, "upper": 0.1, "confidence_level": 0.95},
                },
                "interpretation": "supports the reported primary estimate",
                "source_artifacts": ["data/adtte.parquet"],
            }
        ],
        "limitations": ["Finite event count."],
    }


class _SubmissionSession(_Session):
    def __init__(self, text: str) -> None:
        self.text = text

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        assert "scratch/" in code
        encoded = base64.b64encode(self.text.encode("utf-8")).decode("ascii")
        return CodeExecutionResultV1(status="success", output=encoded, elapsed_seconds=0.1)


def test_structured_response_file_uses_canonical_submission_contract(tmp_path: Path) -> None:
    output, submission, execution = _handle_tool_call(
        ToolCall(id="1", name="submit_response_file", arguments='{"path":"submission.json"}'),
        _SubmissionSession(json.dumps(_submission_payload())),
        tmp_path,
        False,
        submission_interface="structured",
        required_deliverables=_CORE_DELIVERABLES,
    )

    assert output == "Response submitted."
    assert submission is not None
    assert submission["task_id"] == "task-1"
    assert execution is None


def test_structured_response_file_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    text = json.dumps(_submission_payload())
    duplicated = text.replace('"task_id": "task-1"', '"task_id": "task-1", "task_id": "other"', 1)

    with pytest.raises(ValueError, match="Duplicate JSON field"):
        _handle_tool_call(
            ToolCall(id="1", name="submit_response_file", arguments='{"path":"submission.json"}'),
            _SubmissionSession(duplicated),
            tmp_path,
            False,
            submission_interface="structured",
            required_deliverables=_CORE_DELIVERABLES,
        )


def test_response_file_rejects_workspace_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="remain under scratch"):
        _handle_tool_call(
            ToolCall(id="1", name="submit_response_file", arguments='{"path":"../submission.json"}'),
            _SubmissionSession(json.dumps(_submission_payload())),
            tmp_path,
            False,
            submission_interface="structured",
            required_deliverables=_CORE_DELIVERABLES,
        )


def test_narrative_response_file_preserves_text(tmp_path: Path) -> None:
    report = "# Analysis\n\nPrimary estimate: -0.20 (95% CI -0.46 to 0.06).\n"
    output, submission, execution = _handle_tool_call(
        ToolCall(id="1", name="submit_response_file", arguments='{"path":"report.md"}'),
        _SubmissionSession(report),
        tmp_path,
        False,
        submission_interface="narrative",
        required_deliverables=_CORE_DELIVERABLES,
    )

    assert output == "Response submitted."
    assert submission == {"report": report}
    assert execution is None


def test_narrative_response_file_rejects_blank_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty narrative"):
        _handle_tool_call(
            ToolCall(id="1", name="submit_response_file", arguments='{"path":"report.md"}'),
            _SubmissionSession(" \n"),
            tmp_path,
            False,
            submission_interface="narrative",
            required_deliverables=_CORE_DELIVERABLES,
        )


def test_direct_response_uses_the_predeclared_interface(tmp_path: Path) -> None:
    structured_payload = _submission_payload()
    structured_text = json.dumps(structured_payload)
    _, structured, _ = _handle_tool_call(
        ToolCall(id="1", name="submit_response", arguments=structured_text),
        _Session(),
        tmp_path,
        False,
        submission_interface="structured",
        required_deliverables=_CORE_DELIVERABLES,
    )
    _, narrative, _ = _handle_tool_call(
        ToolCall(id="2", name="submit_response", arguments=json.dumps({"content": structured_text})),
        _Session(),
        tmp_path,
        False,
        submission_interface="narrative",
        required_deliverables=_CORE_DELIVERABLES,
    )

    assert structured is not None and structured["task_id"] == "task-1"
    assert narrative == {"report": structured_text}
