"""Tests for resumable truth-blind narrative-normalization batches."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from trialagentbench_test_helpers import write_narrative_packet_set

from trialagentbench_harness.contracts.experiments import NarrativePacketIndexRowV1
from trialagentbench_harness.experiments import normalize_trialeval_narrative_packets as batch
from trialagentbench_harness.io import sha256_file
from trialagentbench_harness.ports import LLMResponse, ToolCall
from trialagentbench_harness.ports.llm_provider import JsonObject

_REPORT = (
    "The 365-day ITT contrast used Kaplan-Meier integration with Greenwood uncertainty under "
    "independent censoring: 18 days (95% CI 4 to 32), reported as the primary numeric point result. "
    "Higher is favorable; interpretation is limited to 365 days."
)


def _claim(
    claim_id: str,
    field_path: str,
    text: str,
    parsed_value: object,
    *,
    role: str = "primary",
) -> dict[str, object]:
    claim: dict[str, object] = {
        "claim_id": claim_id,
        "field_path": field_path,
        "claim_role": role,
        "evidence_level": "executed",
        "source_line_ids": ["L000001"],
        "parsed_value": parsed_value,
    }
    return claim


def _provider_payload() -> str:
    estimand = {
        "estimand_id": "primary_itt",
        "population_id": "intention_to_treat",
        "treatment_id": "active",
        "comparator_id": "control",
        "endpoint_id": "time_to_event",
        "intercurrent_event_strategy_ids": ["rescue:treatment_policy"],
        "horizon": {"value": 365.0, "unit": "days"},
    }
    estimator = {
        "analysis_method_id": "km_rmst_greenwood",
        "implementation": "Kaplan-Meier integration",
        "qualifications": ["independent_censoring"],
    }
    result = {
        "kind": "scalar",
        "value": 18.0,
        "effect_scale": "rmst_difference_tau",
        "unit": "days",
        "interval": {"lower": 4.0, "upper": 32.0, "confidence_level": 0.95},
    }
    return json.dumps(
        {
            "parameters": {
                "status": "complete",
                "claims": [
                    _claim("estimand", "primary_analysis.estimand", "365-day ITT contrast", estimand),
                    _claim("estimator", "primary_analysis.estimator", "Kaplan-Meier integration", estimator),
                    _claim(
                        "result",
                        "primary_analysis.result",
                        "18 days (95% CI 4 to 32)",
                        result,
                    ),
                    _claim(
                        "result-kind",
                        "primary_analysis.result_kind",
                        "primary numeric point result",
                        "numeric_point",
                    ),
                    _claim(
                        "direction",
                        "primary_analysis.favorable_direction",
                        "Higher is favorable",
                        "higher",
                    ),
                    _claim(
                        "limitations",
                        "limitations",
                        "limited to 365 days",
                        ["Interpretation is limited to 365 days."],
                        role="limitation",
                    ),
                ],
                "abstention_reason": None,
            },
        }
    )


class _Provider:
    model = "normalizer-test"
    telemetry_route = "test"

    def __init__(self) -> None:
        self.calls = 0

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        self.calls += 1
        assert tools is not None and len(tools) == 1
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id=f"normalization-{self.calls}",
                    name="emit_narrative_normalization",
                    arguments=_provider_payload(),
                )
            ]
        )


def _packet_root(tmp_path: Path) -> Path:
    return write_narrative_packet_set(tmp_path / "packets", report=_REPORT)


def _config(packet_root: Path) -> batch.NarrativeNormalizationBatchConfigV1:
    return batch.NarrativeNormalizationBatchConfigV1(
        packet_set_manifest_sha256=sha256_file(packet_root / "manifest.json"),
        provider="openai",
        normalizer_model="normalizer-test",
        decoding_seed=7,
        temperature=0.0,
        send_temperature=True,
        max_tokens=4096,
        timeout_seconds=120.0,
        repeats=2,
    ).with_checksum()


@pytest.mark.parametrize("unsafe_identity", ("../escape", "nested/unit", "unit\\name"))
def test_packet_contract_rejects_filesystem_unsafe_identities(unsafe_identity: str) -> None:
    with pytest.raises(ValueError, match="string_pattern_mismatch"):
        NarrativePacketIndexRowV1(
            blinded_identity="masked-narrative-0001",
            qualification_unit_id=unsafe_identity,
            packet_manifest_sha256="p" * 64,
            report_sha256="r" * 64,
        )


def test_batch_preserves_raw_results_and_resumes_without_new_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet_root = _packet_root(tmp_path)
    output = tmp_path / "normalized"
    provider = _Provider()
    monkeypatch.setattr(batch, "get_provider", lambda *args, **kwargs: provider)

    manifest = batch.normalize_packet_set(
        packet_root=packet_root,
        output_root=output,
        config=_config(packet_root),
        resume=False,
    )

    assert manifest.result_count == 2
    assert manifest.complete_count == 2
    assert provider.calls == 2
    assert (output / "automated" / "assignment-1.json").is_file()
    assert (output / "results" / "masked-narrative-0001" / "repeat-0002.json").is_file()

    repeated = batch.normalize_packet_set(
        packet_root=packet_root,
        output_root=output,
        config=_config(packet_root),
        resume=True,
    )
    assert repeated == manifest
    assert provider.calls == 2


def test_batch_rejects_source_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet_root = _packet_root(tmp_path)
    config = _config(packet_root)
    (packet_root / "masked-narrative-0001" / "frozen_report.txt").write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(batch, "get_provider", lambda *args, **kwargs: _Provider())

    with pytest.raises(ValueError, match="report drift"):
        batch.normalize_packet_set(
            packet_root=packet_root,
            output_root=tmp_path / "normalized",
            config=config,
            resume=False,
        )


def test_batch_requires_explicit_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet_root = _packet_root(tmp_path)
    output = tmp_path / "normalized"
    output.mkdir()
    monkeypatch.setattr(batch, "get_provider", lambda *args, **kwargs: _Provider())

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        batch.normalize_packet_set(
            packet_root=packet_root,
            output_root=output,
            config=_config(packet_root),
            resume=False,
        )
