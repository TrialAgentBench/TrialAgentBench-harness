"""Tests for truth-blind TrialEval narrative normalization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

import pytest
from trialagentbench_test_helpers import (
    minimal_trialeval_diagnostic_dictionary,
    minimal_trialeval_method_dictionary,
)

from trialagentbench_harness.contracts.experiments import NarrativeParticipantContextV1
from trialagentbench_harness.contracts.submission import TrialEvalSubmissionV1
from trialagentbench_harness.experiments.narrative_normalization import (
    NarrativeNormalizationRequestV1,
    normalize_narrative_submission_v1,
)
from trialagentbench_harness.ports import LLMResponse, ToolCall
from trialagentbench_harness.ports.llm_provider import JsonObject


class _Provider:
    model = "normalizer-test"
    telemetry_route = "test"

    def __init__(
        self, content: str, *, emit_tool: bool = True, tool_name: str = "emit_narrative_normalization"
    ) -> None:
        self.content = content
        self.emit_tool = emit_tool
        self.tool_name = tool_name
        self.messages: Sequence[JsonObject] = ()
        self.tools: Sequence[JsonObject] | None = None

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        self.messages = messages
        self.tools = tools
        assert tools is not None and len(tools) == 1
        assert temperature == 0.0
        assert max_tokens == 4096
        assert timeout_seconds == 120.0
        if not self.emit_tool:
            return LLMResponse(content=self.content)
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="normalization-call",
                    name=self.tool_name,
                    arguments=self.content,
                )
            ]
        )


def _submission() -> dict[str, object]:
    return {
        "schema_id": "trialagentbench.trialeval_submission/v1",
        "task_id": "TASK1001",
        "primary_analysis": {
            "declared_primary": True,
            "estimand": {
                "estimand_id": "primary_itt",
                "population_id": "intention_to_treat",
                "treatment_id": "active",
                "comparator_id": "control",
                "endpoint_id": "time_to_event",
                "intercurrent_event_strategy_ids": ["rescue:treatment_policy"],
                "horizon": {"value": 365.0, "unit": "days"},
            },
            "estimator": {
                "analysis_method_id": "km_rmst_greenwood",
                "implementation": "Kaplan-Meier integration",
                "qualifications": ["independent_censoring"],
            },
            "result_kind": "numeric_point",
            "result": {
                "kind": "scalar",
                "value": 18.0,
                "effect_scale": "rmst_difference_tau",
                "unit": "days",
                "interval": {"lower": 4.0, "upper": 32.0, "confidence_level": 0.95},
            },
            "favorable_direction": "higher",
            "evidence_ids": [],
        },
        "limitations": ["Interpretation is limited to 365 days."],
    }


def _claim(
    *,
    claim_id: str,
    path: str,
    text: str,
    parsed_value: object,
    role: str = "primary",
    evidence_level: str = "executed",
) -> dict[str, object]:
    claim: dict[str, object] = {
        "claim_id": claim_id,
        "field_path": path,
        "claim_role": role,
        "evidence_level": evidence_level,
        "source_line_ids": ["L000001"],
        "parsed_value": parsed_value,
    }
    return claim


def _request(report: str) -> NarrativeNormalizationRequestV1:
    context = NarrativeParticipantContextV1(
        task_id="TASK1001",
        task_contract={"task_id": "TASK1001", "primary_estimand_id": "primary_itt"},
        participant_submission_contract={"task_id": "TASK1001"},
        participant_diagnostic_dictionary=minimal_trialeval_diagnostic_dictionary(),
        participant_method_dictionary=minimal_trialeval_method_dictionary(),
        canonical_submission_schema=TrialEvalSubmissionV1.model_json_schema(),
    ).with_checksum()
    return NarrativeNormalizationRequestV1(
        assignment_id="assignment-1",
        task_id="TASK1001",
        raw_response=report,
        participant_context=context,
        normalizer_model="normalizer-test",
    )


def _complete_payload() -> dict[str, object]:
    primary = cast(dict[str, object], _submission()["primary_analysis"])
    return {
        "status": "complete",
        "claims": [
            _claim(
                claim_id="estimand",
                path="primary_analysis.estimand",
                text="365-day ITT contrast",
                parsed_value=primary["estimand"],
                evidence_level="declared",
            ),
            _claim(
                claim_id="estimator",
                path="primary_analysis.estimator",
                text="Kaplan-Meier integration",
                parsed_value=primary["estimator"],
            ),
            _claim(
                claim_id="result",
                path="primary_analysis.result",
                text="18 days (95% CI 4 to 32)",
                parsed_value=primary["result"],
            ),
            _claim(
                claim_id="direction",
                path="primary_analysis.favorable_direction",
                text="Higher is favorable",
                parsed_value="higher",
                evidence_level="declared",
            ),
            _claim(
                claim_id="limitations",
                path="limitations",
                text="limited to 365 days",
                parsed_value=_submission()["limitations"],
                role="limitation",
            ),
            _claim(
                claim_id="result-kind",
                path="primary_analysis.result_kind",
                text="numeric point",
                parsed_value="numeric_point",
            ),
        ],
        "abstention_reason": None,
    }


def _tool_payload(payload: dict[str, object]) -> str:
    return json.dumps({"parameters": payload})


def test_normalizer_returns_exact_source_grounded_submission() -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration with Greenwood uncertainty under "
        "independent censoring: 18 days (95% CI 4 to 32), reported as the primary numeric point result. "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    provider = _Provider(_tool_payload(_complete_payload()))

    normalization = normalize_narrative_submission_v1(request=_request(report), provider=provider)
    result = normalization.transcription

    assert result.status == "complete"
    assert result.submission is not None
    primary_result = result.submission.primary_analysis.result
    assert primary_result.kind == "scalar"
    assert primary_result.value == 18.0
    assert result.source_identity == "test:normalizer-test"
    assert result.importer_prompt_sha256
    assert result.importer_schema_sha256
    assert result.importer_response_sha256
    assert normalization.raw_provider_response == provider.content
    assert provider.tools is not None
    assert normalization.checksum
    user_payload = json.loads(str(provider.messages[1]["content"]))
    assert set(user_payload) == {"assignment_id", "task_id", "report_lines", "schemas"}
    assert user_payload["report_lines"] == [{"line_id": "L000001", "text": report}]
    assert "accepted_cells" not in str(user_payload)
    assert "target_values" not in str(user_payload)


@pytest.mark.parametrize("content", ("", "not json", '{"status":"complete"}'))
def test_normalizer_abstains_on_unusable_provider_output(content: str) -> None:
    result = normalize_narrative_submission_v1(
        request=_request("No complete primary analysis was reported."),
        provider=_Provider(content),
    ).transcription

    assert result.status == "abstain"
    assert result.submission is None
    assert result.abstention_reason


def test_normalizer_rejects_unstructured_content_even_when_it_contains_valid_json() -> None:
    result = normalize_narrative_submission_v1(
        request=_request("No complete primary analysis was reported."),
        provider=_Provider(
            json.dumps({"status": "abstain", "claims": [], "abstention_reason": "No primary."}), emit_tool=False
        ),
    ).transcription

    assert result.status == "abstain"
    assert result.abstention_reason == "The normalizer did not emit exactly one structured tool call."


def test_normalizer_rejects_wrong_tool_identity() -> None:
    result = normalize_narrative_submission_v1(
        request=_request("No complete primary analysis was reported."),
        provider=_Provider(
            json.dumps({"status": "abstain", "claims": [], "abstention_reason": "No primary."}),
            tool_name="untrusted_tool",
        ),
    ).transcription

    assert result.status == "abstain"
    assert result.abstention_reason == "The normalizer emitted an invalid normalization tool call."


def test_normalizer_abstains_when_primary_is_only_a_rejected_claim() -> None:
    report = (
        "The 365-day ITT contrast rejected Kaplan-Meier integration: 18 days (95% CI 4 to 32). "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    payload = _complete_payload()
    claims = cast(list[dict[str, object]], payload["claims"])
    claims[1]["claim_role"] = "rejected"

    result = normalize_narrative_submission_v1(
        request=_request(report),
        provider=_Provider(_tool_payload(payload)),
    ).transcription

    assert result.status == "abstain"
    assert any(claim.claim_role == "rejected" for claim in result.claims)


@pytest.mark.parametrize("claim_index", (1, 2))
def test_normalizer_requires_executed_estimator_and_result(claim_index: int) -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration: 18 days (95% CI 4 to 32). "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    payload = _complete_payload()
    claims = cast(list[dict[str, object]], payload["claims"])
    claims[claim_index]["evidence_level"] = "declared"

    result = normalize_narrative_submission_v1(
        request=_request(report),
        provider=_Provider(_tool_payload(payload)),
    ).transcription

    assert result.status == "abstain"
    assert result.submission is None


def test_normalizer_abstains_on_unknown_source_line() -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration: 18 days (95% CI 4 to 32). "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    payload = _complete_payload()
    claims = cast(list[dict[str, object]], payload["claims"])
    claims[2]["source_line_ids"] = ["L999999"]

    result = normalize_narrative_submission_v1(
        request=_request(report),
        provider=_Provider(_tool_payload(payload)),
    ).transcription

    assert result.status == "abstain"
    assert result.abstention_reason == "The normalizer claims violated the canonical claim or source-line contract."


def test_normalizer_line_identity_is_unambiguous_when_text_repeats() -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration: 18 days (95% CI 4 to 32). "
        "Higher is favorable; interpretation is limited to 365 days. Higher is favorable."
    )

    result = normalize_narrative_submission_v1(
        request=_request(report),
        provider=_Provider(_tool_payload(_complete_payload())),
    ).transcription

    assert result.status == "complete"


def test_normalizer_canonicalizes_line_id_padding_and_order() -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration.\n"
        "The primary estimate was 18 days (95% CI 4 to 32).\n"
        "Higher is favorable; interpretation is limited to 365 days."
    )
    payload = _complete_payload()
    claims = cast(list[dict[str, object]], payload["claims"])
    for claim in claims:
        claim["source_line_ids"] = ["L1"]
    claims[2]["source_line_ids"] = ["L3", "L2"]

    result = normalize_narrative_submission_v1(
        request=_request(report),
        provider=_Provider(_tool_payload(payload)),
    ).transcription

    assert result.status == "complete"
    assert tuple(span.text for span in result.claims[2].spans) == (
        "The primary estimate was 18 days (95% CI 4 to 32).",
        "Higher is favorable; interpretation is limited to 365 days.",
    )


def test_normalizer_rejects_duplicate_numeric_line_identity() -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration: 18 days (95% CI 4 to 32). "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    payload = _complete_payload()
    claims = cast(list[dict[str, object]], payload["claims"])
    claims[2]["source_line_ids"] = ["L1", "L000001"]

    result = normalize_narrative_submission_v1(
        request=_request(report),
        provider=_Provider(_tool_payload(payload)),
    ).transcription

    assert result.status == "abstain"
    assert result.claims == ()
    assert result.abstention_reason == "The normalizer claims violated the canonical claim or source-line contract."


def test_normalizer_preserves_conflicts_but_never_resolves_them() -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration: 18 days (95% CI 4 to 32). "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    payload = _complete_payload()
    claims = cast(list[dict[str, object]], payload["claims"])
    claims[2]["conflict"] = True
    claims[2]["conflict_group_id"] = "primary-result-conflict"

    result = normalize_narrative_submission_v1(
        request=_request(report),
        provider=_Provider(_tool_payload(payload)),
    ).transcription

    assert result.status == "abstain"
    assert result.submission is None
    assert result.claims[2].conflict is True
    assert result.abstention_reason == (
        "The normalizer claims could not produce one complete canonical primary submission."
    )


def test_normalizer_rejects_malformed_nested_claim_value() -> None:
    report = (
        "The 365-day ITT contrast used Kaplan-Meier integration: 18 days (95% CI 4 to 32). "
        "Higher is favorable; interpretation is limited to 365 days."
    )
    payload = _complete_payload()
    claims = cast(list[dict[str, object]], payload["claims"])
    claims[1]["parsed_value"] = {"family": "km"}

    result = normalize_narrative_submission_v1(
        request=_request(report),
        provider=_Provider(_tool_payload(payload)),
    ).transcription

    assert result.status == "abstain"
    assert result.claims == ()
    assert result.abstention_reason == "The normalizer claims violated the canonical claim or source-line contract."


def test_normalizer_rejects_provider_identity_drift() -> None:
    request = _request("No complete primary analysis was reported.").model_copy(
        update={"normalizer_model": "other-model"}
    )

    with pytest.raises(ValueError, match="does not match"):
        normalize_narrative_submission_v1(request=request, provider=_Provider("{}"))
