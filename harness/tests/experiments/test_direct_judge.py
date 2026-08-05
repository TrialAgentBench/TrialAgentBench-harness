"""Tests for semantic assessment of narrative analysis reports."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest
from trialagentbench_test_helpers import (
    minimal_trialeval_diagnostic_dictionary,
    minimal_trialeval_method_dictionary,
)

from trialagentbench_harness.analysis.experiments.interface_calibration import (
    InterfaceCalibrationUnitV1,
    analyse_interface_calibration_v1,
)
from trialagentbench_harness.contracts.experiments import NarrativeParticipantContextV1
from trialagentbench_harness.contracts.submission import TrialEvalSubmissionV1
from trialagentbench_harness.experiments.direct_judge import (
    DirectJudgeRequestV1,
    judge_narrative_report_v1,
)
from trialagentbench_harness.ports import LLMResponse, ToolCall
from trialagentbench_harness.ports.llm_provider import JsonObject


class _Provider:
    model = "judge-test"
    telemetry_route = "test"

    def __init__(self, arguments: str) -> None:
        self.arguments = arguments
        self.messages: Sequence[JsonObject] = ()

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        self.messages = messages
        assert tools is not None and len(tools) == 1
        assert tool_choice == "required"
        return LLMResponse(
            tool_calls=[ToolCall(id="judge-call", name="record_report_assessment", arguments=self.arguments)]
        )


def _request() -> DirectJudgeRequestV1:
    context = NarrativeParticipantContextV1(
        task_id="TASK1001",
        task_contract={"task_id": "TASK1001", "question": "Estimate the declared treatment contrast."},
        participant_submission_contract={"task_id": "TASK1001"},
        participant_diagnostic_dictionary=minimal_trialeval_diagnostic_dictionary(),
        participant_method_dictionary=minimal_trialeval_method_dictionary(),
        canonical_submission_schema=TrialEvalSubmissionV1.model_json_schema(),
    ).with_checksum()
    return DirectJudgeRequestV1(
        assignment_id="assignment-1",
        task_id="TASK1001",
        raw_response="I used the declared method.\nThe estimate was 2.0 days.",
        participant_context=context,
        judge_model="judge-test",
    )


def _decision(*, conforms: bool = True) -> str:
    statuses = ["passed"] * 6
    if not conforms:
        statuses[-1] = "failed"
    components = []
    for component_id, status in zip(
        ("question", "method", "evidence", "integrity", "result_structure", "result_support"),
        statuses,
        strict=True,
    ):
        components.append(
            {
                "component_id": component_id,
                "status": status,
                "report_line_ids": ["L000001"],
                "reason": "The report contains the relevant declaration.",
            }
        )
    return json.dumps({"parameters": {"conforms": conforms, "components": components}})


def test_direct_judge_uses_report_and_participant_contract() -> None:
    provider = _Provider(_decision())
    result = judge_narrative_report_v1(request=_request(), provider=provider)

    assert result.status == "completed"
    assert result.decision is not None and result.decision.conforms
    user_payload = json.loads(str(provider.messages[1]["content"]))
    assert user_payload["task_id"] == "TASK1001"
    assert user_payload["report_lines"][0]["text"] == "I used the declared method."
    assert user_payload["participant_context"]["participant_task_contract"]["question"] == (
        "Estimate the declared treatment contrast."
    )


def test_direct_judge_rejects_unknown_report_line() -> None:
    payload = json.loads(_decision())
    payload["parameters"]["components"][0]["report_line_ids"] = ["L999999"]
    result = judge_narrative_report_v1(request=_request(), provider=_Provider(json.dumps(payload)))

    assert result.status == "invalid_response"
    assert result.decision is None
    assert result.failure_reason is not None


def test_interface_analysis_reports_paired_error_latency_cost_and_failure() -> None:
    units = (
        InterfaceCalibrationUnitV1(
            assignment_id="a1",
            task_id="t1",
            reference_conforms=True,
            automated_normalization_status="complete",
            automated_normalization_conforms=False,
            omitted_score_fields=("evidence",),
            normalization_elapsed_seconds=2.0,
            normalization_cost_usd=0.02,
            direct_judge_status="completed",
            direct_judge_conforms=True,
            direct_judge_elapsed_seconds=1.0,
            direct_judge_cost_usd=0.01,
        ),
        InterfaceCalibrationUnitV1(
            assignment_id="a2",
            task_id="t2",
            reference_conforms=False,
            automated_normalization_status="failed",
            normalization_failure_reason="invalid tool call",
            normalization_elapsed_seconds=4.0,
            direct_judge_status="completed",
            direct_judge_conforms=True,
            ambiguous_score_fields=("method",),
            direct_judge_elapsed_seconds=3.0,
        ),
    )

    report = analyse_interface_calibration_v1(units)

    assert report.normalizer.false_rejection_rate == 1.0
    assert report.normalizer.failure_rate == 0.5
    assert report.direct_judge.false_acceptance_rate == 1.0
    assert report.direct_judge.agreement_rate == 0.5
    assert report.omission_rate == 0.5
    assert report.ambiguity_rate == 0.5
    assert report.normalizer.mean_latency_seconds == pytest.approx(3.0)
    assert report.normalizer.total_reported_cost_usd == pytest.approx(0.02)
