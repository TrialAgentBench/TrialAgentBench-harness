"""First-run teaching assets execute and replay through the public native owners."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from trialagentbench.tools.demo import demo_assets


def test_demo_assets_are_bounded_and_external_adapter_is_available() -> None:
    """The offline installed surface contains its entire small teaching corpus."""
    root = demo_assets()
    archives = tuple(root.glob("demo-*-evaluator.zip"))
    assert len(archives) == 2
    assert sum(path.stat().st_size for path in archives) < 4_000_000
    assert (root / "run_external_provider.py").is_file()
    assert (root / "observed_analysis.py").is_file()


@pytest.mark.executor
def test_demo_native_reporting_acquisition_and_completed_resume(tmp_path: Path) -> None:
    """Run without provider SDK imports, then resume without another provider turn."""
    output = tmp_path / "teaching"
    driver = """
import importlib.abc
import sys
from pathlib import Path
class NoProviderSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'openai', 'httpx2', 'agents'}:
            raise AssertionError('Demo imported a provider SDK: ' + fullname)
sys.meta_path.insert(0, NoProviderSDK())
from trialagentbench.tools.demo import ScriptedDemoProvider, run_demo
class CompletedProvider(ScriptedDemoProvider):
    def generate_turn(self, *args, **kwargs):
        raise AssertionError('Completed resume requested another provider turn')
resume = len(sys.argv) > 2
run_demo(output=Path(sys.argv[1]), provider=CompletedProvider() if resume else ScriptedDemoProvider(),
         executor_image=None, resume=resume)
"""
    completed = subprocess.run(
        [sys.executable, "-c", driver, str(output)], capture_output=True, text=True, timeout=180, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    te = json.loads((output / "trialeval/replay.json").read_text())
    assert te["status"] == "reproduced"
    task = next((output / "trialdev").iterdir())
    programme = json.loads((task / "programme_run.json").read_text())
    assert programme["execution_status"] == "completed"
    assert len(programme["evaluation"]["studies"]) == 1
    assert programme["evaluation"]["initial_replay"]["status"] == "reproduced"
    assert programme["evaluation"]["studies"][0]["replay"]["status"] == "reproduced"
    estimates = list(csv.DictReader((task / "workspace/scratch/estimates.csv").read_text().splitlines()))
    assert len(estimates) == 2
    assert any(float(row["estimate"]) != 0 for row in estimates)
    assert all(float(row["lower"]) < float(row["upper"]) for row in estimates)
    events = [json.loads(line) for line in (task / "events.jsonl").read_text().splitlines()]
    assert any((event.get("cpu_seconds") or 0) > 0 for event in events)
    transcript = (task / "conversation.json").read_bytes()
    ledger = (task / "private/budget.json").read_bytes()
    resumed = subprocess.run(
        [sys.executable, "-c", driver, str(output), "resume"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert transcript == (task / "conversation.json").read_bytes()
    assert ledger == (task / "private/budget.json").read_bytes()
    report = (output / "index.html").read_text()
    grade = json.loads((output / "trialeval/grade-0.json").read_text())
    assert f'{grade["signal_loss"]:.4g}' in report
    assert f'{grade["point_performance"]["oracle_value"]:.4g}' in report
    assert f'${programme["evaluation"]["value"]["net_value_usd"]:,.2f}' in report
    assert "Compare its clinical estimates with the world-derived targets" in report
    assert "<summary>Execution details</summary>" in report
    assert "DAS28 points" in report and "risk proportions" in report and "mg/g" in report
    assert "Download every observed row as CSV" in report
    for relative in re.findall(r'href="([^"]+)"', report):
        assert (output / relative).is_file()


@pytest.mark.executor
def test_demo_preserves_interrupted_trialeval_work_before_restart(tmp_path: Path, capsys) -> None:
    """A restarted TrialEval attempt retains its original transcript, usage and workspace."""
    from trialagentbench.tools.demo import ScriptedDemoProvider, run_demo

    class InterruptedProvider(ScriptedDemoProvider):
        def generate_turn(self, messages, tools=None, **kwargs):
            if any(message.get("tool_calls") for message in messages):
                raise RuntimeError("interrupted after the first saved analysis")
            return super().generate_turn(messages, tools, **kwargs)

    output = tmp_path / "demo"
    with pytest.raises(RuntimeError, match="interrupted after"):
        run_demo(output=output, provider=InterruptedProvider(), executor_image=None)
    partial = output / "trialeval"
    before = {path.relative_to(partial): path.read_bytes() for path in partial.rglob("*") if path.is_file()}
    assert json.loads(before[Path("conversation.json")])[-1]["role"] == "tool"
    assert b'"status": "succeeded"' in before[Path("provider_responses.jsonl")]
    run_demo(output=output, provider=ScriptedDemoProvider(), executor_image=None, resume=True)
    archived = output / "failed_attempts/trialeval/attempt-1/trialeval.archived"
    assert {path.relative_to(archived): path.read_bytes() for path in archived.rglob("*") if path.is_file()} == before
    assert "Restarting TrialEval from its initial evidence" in capsys.readouterr().out
    report = (output / "index.html").read_text()
    assert "Interrupted TrialEval attempts: 1" in report
    assert "Include their provider charges" in report
    for relative in re.findall(r'href="([^"]+)"', report):
        assert (output / relative).is_file()
