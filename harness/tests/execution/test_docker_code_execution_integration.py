"""Real-container qualification for model-generated code execution."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)

from trialagentbench_harness.adapters.docker_code_execution import DockerPythonSession
from trialagentbench_harness.ports import LLMResponse, ToolCall
from trialagentbench_harness.ports.code_execution import CodeExecutionLimitsV1
from trialagentbench_harness.trialeval.agent import run_agent
from trialagentbench_harness.trialeval.schema import BenchmarkItem
from trialagentbench_harness.util.runtime_context import (
    bounded_provider_context,
    persist_bulky_tool_output,
)

pytestmark = pytest.mark.executor


def _limits() -> CodeExecutionLimitsV1:
    return CodeExecutionLimitsV1(
        timeout_seconds=3,
        memory_mb=512,
        cpu_count=1,
        process_limit=32,
        output_bytes=1024,
        workspace_mb=128,
    )


def _container_removed(name: str) -> bool:
    return subprocess.run(["docker", "inspect", name], capture_output=True, check=False).returncode != 0


def test_executor_isolates_evidence_network_environment_and_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TRIALAGENTBENCH_TEST_SECRET", "must-not-leak")
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("original", encoding="utf-8")
    session = DockerPythonSession(cwd=tmp_path, limits=_limits())
    name = session._container_name
    try:
        assert session.execute_result("x=41").status == "success"
        assert session.execute_result("print(x + 1)").output == "42"
        raw = session.execute_result("import os; os.write(1, b'raw-fd-output')")
        assert raw.output == "raw-fd-output"
        assert session.execute_result("print(open('evidence.txt').read())").output == "original"
        assert (
            session.execute_result(
                "import os; print(os.environ['HOME'], os.environ['XDG_CACHE_HOME'], os.environ['MPLCONFIGDIR'])"
            ).output
            == "/workspace /workspace/.cache /workspace/.config/matplotlib"
        )
        assert (
            session.execute_result(
                "import os; print(*(os.environ[name] for name in "
                "('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', "
                "'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')))"
            ).output
            == "1 1 1 1 1"
        )
        matplotlib_config = session.execute_result("import matplotlib; print(matplotlib.get_configdir())")
        assert matplotlib_config.status == "success"
        assert matplotlib_config.output == "/workspace/.config/matplotlib"
        session.execute_result("open('evidence.txt', 'w').write('changed')")
        assert evidence.read_text(encoding="utf-8") == "original"
        (tmp_path / "new_phase.txt").write_text("phase2", encoding="utf-8")
        assert session.execute_result("print(open('new_phase.txt').read())").output == "phase2"
        assert session.execute_result("import os; print(os.getenv('TRIALAGENTBENCH_TEST_SECRET'))").output == "None"
        host_home = Path("/", "home", "host-only-user").as_posix()
        assert (
            session.execute_result(f"from pathlib import Path; print(Path({host_home!r}).exists())").output == "False"
        )
        network = session.execute_result(
            "import socket\n"
            "try:\n"
            " socket.create_connection(('1.1.1.1', 53), timeout=.2)\n"
            " print('open')\n"
            "except OSError:\n"
            " print('denied')"
        )
        assert network.output == "denied"
        disk = session.execute_result(
            "try:\n"
            " open('fill.bin', 'wb').write(b'x' * (140 * 1024 * 1024))\n"
            " print('unexpected')\n"
            "except OSError:\n"
            " print('bounded')"
        )
        assert disk.output == "bounded"
    finally:
        session.close()
    assert _container_removed(name)


def test_executor_bounds_output_processes_memory_and_time(tmp_path: Path) -> None:
    session = DockerPythonSession(cwd=tmp_path, limits=_limits())
    name = session._container_name
    large = session.execute_result("print('x' * 10_000_000)")
    assert large.status == "success"
    assert large.output_truncated is True
    assert len(large.output.encode("utf-8")) < 1100
    pids = session.execute_result(
        "import subprocess\n"
        "children=[]\n"
        "for _ in range(80):\n"
        " try: children.append(subprocess.Popen(['sleep', '2']))\n"
        " except OSError: break\n"
        "print(len(children))\n"
        "for child in children: child.terminate()"
    )
    assert pids.status == "success"
    assert int(pids.output) < 32
    assert session.execute_result("while True: pass").status == "timeout"
    swallowed = session.execute_result(
        "try:\n" " while True: pass\n" "except Exception:\n" " print('timeout swallowed')"
    )
    assert swallowed.status == "timeout"
    assert "timeout swallowed" not in swallowed.output
    recovered = session.execute_result("print('session retained')")
    assert recovered.status == "success"
    assert recovered.output == "session retained"
    session.close()
    assert _container_removed(name)

    memory_session = DockerPythonSession(cwd=tmp_path, limits=_limits())
    memory_name = memory_session._container_name
    memory = memory_session.execute_result("payload = bytearray(b'x' * (900 * 1024 * 1024))")
    assert memory.status == "session_terminated"
    memory_session.close()
    assert _container_removed(memory_name)


def test_executor_retains_scratch_across_hard_timeout_and_close(tmp_path: Path) -> None:
    limits = _limits().model_copy(update={"timeout_seconds": 1.0})
    session = DockerPythonSession(cwd=tmp_path, limits=limits)
    first_name = session._container_name
    assert (
        session.execute_result(
            "from pathlib import Path; Path('scratch/analysis.py').write_text('answer = 42')"
        ).status
        == "success"
    )

    timed_out = session.execute_result(
        "import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nwhile True: pass"
    )
    assert timed_out.status == "timeout"
    assert "in-memory variables were lost" in timed_out.output
    assert _container_removed(first_name)
    assert session._container_name != first_name
    restored = session.execute_result("print(open('scratch/analysis.py').read())")
    assert restored.status == "success"
    assert restored.output == "answer = 42"

    session.close()
    assert (tmp_path / "scratch" / "analysis.py").read_text(encoding="utf-8") == "answer = 42"


def test_runtime_context_artifacts_share_the_persistent_session_workspace(tmp_path: Path) -> None:
    session = DockerPythonSession(cwd=tmp_path, limits=_limits())
    assert session.execute_result("_path = 'agent-owned'").status == "success"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old interaction"},
        {"role": "assistant", "content": "old response"},
        {"role": "user", "content": "active prompt"},
    ]

    context = bounded_provider_context(
        messages,
        session=session,
        active_prompt_index=3,
        max_chars=80,
    )
    rendered = persist_bulky_tool_output(
        "captured-" * 100,
        session=session,
        artifact_id="runtime-output",
        inline_chars=20,
    )
    output_path = rendered.split("retained at ", maxsplit=1)[1].split(";", maxsplit=1)[0]

    assert any("context_archive.json" in str(message.get("content")) for message in context)
    assert session.execute_result("print(open('scratch/context_archive.json').read())").status == "success"
    retained = session.execute_result(f"print(open('scratch/{output_path}').read())")
    assert retained.status == "success"
    assert retained.output == "captured-" * 100
    assert session.execute_result("print(_path)").output == "agent-owned"

    session.close()
    assert (tmp_path / "scratch" / "context_archive.json").is_file()
    assert (tmp_path / "scratch" / output_path).read_text(encoding="utf-8") == "captured-" * 100


def test_executor_rejects_linked_scratch_checkpoint(tmp_path: Path) -> None:
    session = DockerPythonSession(cwd=tmp_path, limits=_limits())
    name = session._container_name
    created = session.execute_result("from pathlib import Path; Path('scratch/link').symlink_to('/etc/passwd')")
    assert created.status == "success"
    with pytest.raises(ValueError, match="linked or special file"):
        session.close()
    assert _container_removed(name)


def test_executor_snapshots_live_scratch_without_ending_session(tmp_path: Path) -> None:
    session = DockerPythonSession(cwd=tmp_path, limits=_limits())
    try:
        assert (
            session.execute_result(
                "from pathlib import Path; x = 41; Path('scratch/live.txt').write_text('one')"
            ).status
            == "success"
        )
        assert session.snapshot_scratch() == tmp_path / "scratch"
        assert (tmp_path / "scratch" / "live.txt").read_text(encoding="utf-8") == "one"
        assert session.execute_result("print(x + 1)").output == "42"
        assert session.execute_result("Path('scratch/live.txt').write_text('two')").status == "success"
        session.snapshot_scratch()
        assert (tmp_path / "scratch" / "live.txt").read_text(encoding="utf-8") == "two"
    finally:
        session.close()


def test_executor_rejects_unsafe_evidence_added_after_start(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("safe", encoding="utf-8")
    session = DockerPythonSession(cwd=tmp_path, limits=_limits())
    name = session._container_name
    try:
        (tmp_path / "unsafe-link.txt").symlink_to(evidence)
        result = session.execute_result("print('must not run')")
        assert result.status == "session_terminated"
        assert "forbidden symlink" in result.output
        assert "must not run" not in result.output
    finally:
        session.close()
    assert _container_removed(name)


def test_executor_sessions_are_isolated_under_concurrent_load(tmp_path: Path) -> None:
    workdirs = tuple(tmp_path / f"program-{index}" for index in range(4))
    for index, workdir in enumerate(workdirs):
        workdir.mkdir()
        (workdir / "identity.txt").write_text(str(index), encoding="utf-8")

    def execute(index: int) -> tuple[str, str, str]:
        session = DockerPythonSession(cwd=workdirs[index], limits=_limits())
        name = session._container_name
        try:
            result = session.execute_result("print(open('identity.txt').read())")
            return result.status, result.output, name
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=len(workdirs)) as pool:
        results = tuple(pool.map(execute, range(len(workdirs))))

    assert [(status, output) for status, output, _ in results] == [
        ("success", str(index)) for index in range(len(workdirs))
    ]
    assert all(_container_removed(name) for _, _, name in results)


def test_trialeval_agent_loop_uses_the_isolated_executor(tmp_path: Path) -> None:
    visible = tmp_path / "visible"
    data = visible / "data"
    hidden = tmp_path / "hidden"
    data.mkdir(parents=True)
    hidden.mkdir()
    (data / "analysis.parquet").write_bytes(b"fixture")
    write_minimal_trialeval_release_dictionaries(tmp_path)
    item = BenchmarkItem(
        item_id="executor-smoke",
        trial_name="executor-smoke",
        design_tier="D1",
        design_subtype="individual_randomized",
        assumption_tier="A1",
        context_tier="C4",
        data_preparation="analysis_ready",
        visible_dir=visible,
        data_dir=data,
        task={},
        submission_contract=minimal_participant_output_contract("executor-smoke"),
        suite_dir=tmp_path,
    )

    class StubProvider:
        model = "stub"

        def __init__(self) -> None:
            self.calls = 0

        def generate_turn(self, **_: object) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="write-workspace",
                            name="write_workspace_file",
                            arguments='{"path":"analysis.py","content":"value = 6 * 7\\nprint(value)\\n"}',
                        ),
                        ToolCall(
                            id="read-workspace",
                            name="read_workspace_file",
                            arguments='{"path":"analysis.py","start_line":1,"end_line":2}',
                        ),
                        ToolCall(
                            id="invalid-execute",
                            name="execute_code",
                            arguments="{not-json",
                        ),
                        ToolCall(
                            id="execute",
                            name="execute_code",
                            arguments='{"code":"value = 6 * 7\\nprint(value)"}',
                        ),
                    ]
                )
            submission = {
                "task_id": "executor-smoke",
                "primary_analysis": {
                    "declared_primary": True,
                    "estimand": {
                        "estimand_id": "primary_itt",
                        "population_id": "all_participants",
                        "treatment_id": "active",
                        "comparator_id": "control",
                        "endpoint_id": "endpoint",
                        "intercurrent_event_strategy_ids": ["rescue_therapy:treatment_policy"],
                        "horizon_not_applicable_reason": "cross-sectional endpoint",
                    },
                    "estimator": {
                        "analysis_method_id": "km_rmst_greenwood",
                        "implementation": "Kaplan-Meier fixed-horizon restricted mean survival time",
                        "qualifications": ["randomization_exchangeability"],
                    },
                    "result_kind": "numeric_point",
                    "result": {
                        "kind": "scalar",
                        "value": 0.0,
                        "effect_scale": "rmst_difference_tau",
                        "unit": "days",
                        "interval": {"lower": -1.0, "upper": 1.0, "confidence_level": 0.95},
                    },
                    "favorable_direction": "higher",
                    "evidence_ids": ["support-1"],
                },
                "evidence": [
                    {
                        "evidence_id": "support-1",
                        "evidence_type": "supporting_analysis",
                        "principle": "uncertainty",
                        "operation": "estimation",
                        "estimator": {
                            "analysis_method_id": "km_rmst_greenwood",
                            "implementation": "Kaplan-Meier fixed-horizon restricted mean survival time",
                            "qualifications": ["randomization_exchangeability"],
                        },
                        "target": "Mean difference",
                        "result": {
                            "kind": "scalar",
                            "value": 0.0,
                            "effect_scale": "rmst_difference_tau",
                            "unit": "days",
                            "interval": {
                                "lower": -1.0,
                                "upper": 1.0,
                                "confidence_level": 0.95,
                            },
                        },
                        "interpretation": "Supports the primary estimate.",
                        "source_artifacts": ["data/analysis.parquet"],
                    }
                ],
                "limitations": ["Fixture analysis."],
            }
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="submit",
                        name="submit_response",
                        arguments=json.dumps(submission),
                    )
                ]
            )

    provider = StubProvider()
    conversation_path = tmp_path / "conversation.json"
    event_path = tmp_path / "events.jsonl"
    result = run_agent(
        item,
        provider,
        max_turns=4,
        verbose=False,
        conversation_log_path=conversation_path,
        event_log_path=event_path,
    )

    assert result["status"] == "success"
    assert provider.calls == 2
    assert any(row.get("tool") == "write_workspace_file" for row in result["conversation"])
    assert any(
        row.get("tool") == "read_workspace_file" and "0002: print(value)" in str(row.get("output"))
        for row in result["conversation"]
    )
    assert any(
        row.get("tool") == "execute_code" and str(row.get("output", "")).splitlines()[0] == "42"
        for row in result["conversation"]
        if str(row.get("output", "")).splitlines()
    )
    assert any(
        row.get("tool_call_id") == "invalid-execute" and str(row.get("output", "")).startswith("Tool input rejected:")
        for row in result["conversation"]
    )
    assert [event["event_index"] for event in result["events"]] == list(range(len(result["events"])))
    assert any(event["status"] == "invalid" for event in result["events"])
    assert any(event["event_type"] == "code_execution" for event in result["events"])
    assert any(
        event["event_type"] == "file_inspection" and event["file_accessed"] == "scratch/analysis.py"
        for event in result["events"]
    )
    submission_events = [event for event in result["events"] if event["event_type"] == "submission"]
    assert len(submission_events) == 1
    assert submission_events[0]["status"] == "observed"
    assert json.loads(conversation_path.read_text(encoding="utf-8")) == result["conversation"]
    persisted_events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert persisted_events == result["events"]
