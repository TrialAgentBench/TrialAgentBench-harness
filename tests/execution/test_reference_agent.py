"""The scripted reference agent completes development tasks through participant tools only.

Requires Docker and development evaluator archives named by
``TRIALAGENTBENCH_REFERENCE_INPUTS`` (a directory with ``trialeval-evaluator.zip`` and/or
``trialdev-evaluator.zip``), plus optional comma-separated task selections in
``TRIALAGENTBENCH_REFERENCE_TRIALEVAL_TASKS`` and ``TRIALAGENTBENCH_REFERENCE_TRIALDEV_TASKS``.
Only development-seed worlds may be used.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from reference_agent.agent import ReferenceAgent
from reference_agent.run import run_trialdev, run_trialeval

INPUTS = os.environ.get("TRIALAGENTBENCH_REFERENCE_INPUTS")
IMAGE = os.environ.get("TRIALAGENTBENCH_EXECUTOR_IMAGE")


def _selection(name: str) -> list[str]:
    return [task for task in os.environ.get(name, "").split(",") if task]


def test_reference_agent_requests_only_offered_tools() -> None:
    """The agent refuses to call a tool the runtime did not offer."""
    agent = ReferenceAgent()
    messages = [
        {"role": "system", "content": "Data files: data/raw/a.parquet.\n"},
        {"role": "user", "content": "q"},
    ]
    with pytest.raises(ValueError, match="unavailable tool"):
        agent.generate_turn(
            messages, [{"type": "function", "function": {"name": "submit_response"}}], temperature=0, max_tokens=1
        )


@pytest.mark.executor
@pytest.mark.skipif(INPUTS is None, reason="Set TRIALAGENTBENCH_REFERENCE_INPUTS to development archives")
def test_reference_agent_completes_selected_development_tasks(tmp_path: Path) -> None:
    """Every selected task completes with an exact replay and no tool error attributable to the harness."""
    assert INPUTS is not None
    inputs = Path(INPUTS)
    rows = []
    if tasks := _selection("TRIALAGENTBENCH_REFERENCE_TRIALEVAL_TASKS"):
        rows += run_trialeval(inputs / "trialeval-evaluator.zip", tasks, tmp_path, IMAGE)
    if tasks := _selection("TRIALAGENTBENCH_REFERENCE_TRIALDEV_TASKS"):
        rows += run_trialdev(inputs / "trialdev-evaluator.zip", tasks, tmp_path, IMAGE)
    assert rows, "Select at least one development task"
    for row in rows:
        assert row["status"] in {"success", "completed"}, row
        assert row.get("replay", "reproduced") == "reproduced", row
        assert not [call for call in row["calls"] if call.get("error")], row["friction"]  # type: ignore[union-attr]
