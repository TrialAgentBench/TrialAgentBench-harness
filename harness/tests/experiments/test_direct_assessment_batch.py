"""Tests for resumable reference-blind semantic-assessment batches."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from trialagentbench_test_helpers import write_narrative_packet_set

from trialagentbench_harness.contracts.experiments import DirectAssessmentBatchConfigV1
from trialagentbench_harness.experiments import assess_trialeval_narrative_packets as batch
from trialagentbench_harness.io import sha256_file
from trialagentbench_harness.ports import LLMResponse, ToolCall
from trialagentbench_harness.ports.llm_provider import JsonObject

_REPORT = "The report states a question, method, evidence, integrity review, result structure, and result."


def _decision() -> str:
    components = [
        {
            "component_id": component_id,
            "status": "passed",
            "report_line_ids": ["L000001"],
            "reason": "The report contains the required statement.",
        }
        for component_id in (
            "question",
            "method",
            "evidence",
            "integrity",
            "result_structure",
            "result_support",
        )
    ]
    return json.dumps({"parameters": {"conforms": True, "components": components}})


class _Provider:
    model = "judge-test"
    telemetry_route = "test"

    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid
        self.calls = 0

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
        self.calls += 1
        assert messages and tools is not None and len(tools) == 1
        assert tool_choice == "required"
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id=f"assessment-{self.calls}",
                    name="record_report_assessment",
                    arguments=_decision() if self.valid else "not-json",
                )
            ]
        )


def _packet_root(tmp_path: Path) -> Path:
    return write_narrative_packet_set(tmp_path / "packets", report=_REPORT)


def _config(packet_root: Path) -> DirectAssessmentBatchConfigV1:
    return DirectAssessmentBatchConfigV1(
        packet_set_manifest_sha256=sha256_file(packet_root / "manifest.json"),
        provider="openai",
        judge_model="judge-test",
        decoding_seed=11,
        temperature=0.0,
        send_temperature=True,
        max_tokens=2048,
        timeout_seconds=120.0,
        repeats=2,
    ).with_checksum()


def test_batch_preserves_results_and_resumes_without_new_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet_root = _packet_root(tmp_path)
    output = tmp_path / "assessed"
    provider = _Provider()
    monkeypatch.setattr(batch, "get_provider", lambda *args, **kwargs: provider)

    manifest = batch.assess_packet_set(
        packet_root=packet_root,
        output_root=output,
        config=_config(packet_root),
        resume=False,
    )

    assert manifest.result_count == 2
    assert manifest.completed_count == 2
    assert manifest.invalid_response_count == 0
    assert provider.calls == 2
    assert (output / "results" / "masked-narrative-0001" / "repeat-0002.json").is_file()

    repeated = batch.assess_packet_set(
        packet_root=packet_root,
        output_root=output,
        config=_config(packet_root),
        resume=True,
    )
    assert repeated == manifest
    assert provider.calls == 2


def test_batch_retains_invalid_provider_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet_root = _packet_root(tmp_path)
    provider = _Provider(valid=False)
    monkeypatch.setattr(batch, "get_provider", lambda *args, **kwargs: provider)

    manifest = batch.assess_packet_set(
        packet_root=packet_root,
        output_root=tmp_path / "assessed",
        config=_config(packet_root),
        resume=False,
    )

    assert manifest.completed_count == 0
    assert manifest.invalid_response_count == 2
    assert provider.calls == 2


def test_batch_rejects_source_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet_root = _packet_root(tmp_path)
    config = _config(packet_root)
    (packet_root / "masked-narrative-0001" / "frozen_report.txt").write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(batch, "get_provider", lambda *args, **kwargs: _Provider())

    with pytest.raises(ValueError, match="report drift"):
        batch.assess_packet_set(
            packet_root=packet_root,
            output_root=tmp_path / "assessed",
            config=config,
            resume=False,
        )
