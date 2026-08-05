"""Run-level controls for bounded TrialDev live evaluations."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Literal

import pytest

from trialagentbench_harness.ports import LLMResponse, LLMResponseMetadata
from trialagentbench_harness.ports.llm_provider import JsonObject
from trialagentbench_harness.tools.run import trialdev as trialdev_cli
from trialagentbench_harness.util.reported_cost import (
    ReportedCostBoundProvider,
    ReportedCostBudget,
    ReportedCostThresholdReached,
    RunStopRequested,
    RunStopSignal,
    StoppableProvider,
)


class _CostedProvider:
    model = "test-model"
    telemetry_route = "test"

    def __init__(self, costs: Sequence[float | None]) -> None:
        self._costs = iter(costs)
        self.calls = 0

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> LLMResponse:
        del messages, tools, temperature, max_tokens, timeout_seconds, tool_choice
        self.calls += 1
        return LLMResponse(metadata=LLMResponseMetadata(reported_cost_usd=next(self._costs)))


def _generate(provider: ReportedCostBoundProvider | StoppableProvider) -> LLMResponse:
    return provider.generate_turn((), temperature=0.0, max_tokens=1)


def test_reported_cost_stop_is_post_response_and_reports_overshoot() -> None:
    raw = _CostedProvider((0.6, 0.6, 0.6))
    budget = ReportedCostBudget(threshold_usd=1.0)
    provider = ReportedCostBoundProvider(provider=raw, budget=budget)

    _generate(provider)
    _generate(provider)
    with pytest.raises(ReportedCostThresholdReached):
        _generate(provider)

    snapshot = budget.snapshot()
    assert raw.calls == 2
    assert snapshot.observed_usd == pytest.approx(1.2)
    assert snapshot.overshoot_usd == pytest.approx(0.2)
    assert snapshot.response_count == 2


def test_stop_signal_blocks_the_next_provider_request() -> None:
    raw = _CostedProvider((0.1,))
    signal = RunStopSignal()
    provider = StoppableProvider(provider=raw, signal=signal)
    signal.request()

    with pytest.raises(RunStopRequested):
        _generate(provider)

    assert raw.calls == 0


def test_bounded_batch_preserves_keyboard_stop_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    signal = RunStopSignal()
    entered = threading.Event()
    real_wait = trialdev_cli.wait
    wait_calls = 0

    def execute(job: str) -> dict[str, object]:
        entered.set()
        while not signal._event.is_set():
            signal._event.wait(timeout=0.01)
        signal.before_request()
        return {"program_id": job}

    def interrupt_once(*args: object, **kwargs: object) -> object:
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            assert entered.wait(timeout=1.0)
            raise KeyboardInterrupt
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(trialdev_cli, "wait", interrupt_once)
    result = trialdev_cli._run_bounded_batch(
        jobs=("unit-1", "unit-2", "unit-3"),
        workers=1,
        unit_id=str,
        execute=execute,
        reported_cost_budget=None,
        stop_signal=signal,
    )

    assert result == {
        "results": [],
        "interrupted_unit_ids": ["unit-1"],
        "not_started_unit_ids": ["unit-2", "unit-3"],
        "stop_reason": "keyboard_interrupt",
    }
