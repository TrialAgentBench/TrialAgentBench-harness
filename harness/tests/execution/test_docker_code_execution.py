"""Contract tests for the isolated model-code executor."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from trialagentbench_harness.adapters.docker_code_execution import (
    DockerPythonSession,
    _docker_command,
    resolve_executor_environment,
)
from trialagentbench_harness.ports.code_execution import CodeExecutionLimitsV1


def test_execution_limits_reject_unbounded_values() -> None:
    with pytest.raises(ValidationError):
        CodeExecutionLimitsV1(timeout_seconds=0)
    with pytest.raises(ValidationError):
        CodeExecutionLimitsV1(memory_mb=128)
    with pytest.raises(ValidationError):
        CodeExecutionLimitsV1(output_bytes=17 * 1024 * 1024)


def test_executor_image_uses_one_fully_pinned_package_lock() -> None:
    executor_root = Path(__file__).resolve().parents[2] / "executor"
    dockerfile = (executor_root / "Dockerfile").read_text(encoding="utf-8")
    requirements = (executor_root / "requirements.lock").read_text(encoding="utf-8").splitlines()

    assert "COPY requirements.lock" in dockerfile
    assert "--no-deps -r" in dockerfile
    assert requirements
    assert all(line and "==" in line and not line.startswith(("-", "#")) for line in requirements)
    names = [line.split("==", maxsplit=1)[0].casefold() for line in requirements]
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_docker_command_enforces_isolation_and_read_only_host_mounts(tmp_path: Path) -> None:
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    command = _docker_command(
        workdir=tmp_path,
        scratch_root=scratch_root,
        image="executor:test",
        limits=CodeExecutionLimitsV1(),
        container_name="executor-test",
        protocol_token="opaque-token",
    )
    rendered = " ".join(command)
    assert "--network none" in rendered
    assert "--read-only" in command
    assert "--cap-drop ALL" in rendered
    assert "--security-opt no-new-privileges" in rendered
    assert "--pids-limit 128" in rendered
    assert "--memory 4096m" in rendered
    assert command.count("--mount") == 2
    assert any(value.endswith(f"src={tmp_path},dst=/evidence,readonly") for value in command)
    assert any(value.endswith(f"src={scratch_root},dst=/scratch-seed,readonly") for value in command)
    assert any(value.startswith("/workspace:rw,noexec") and "size=2048m" in value for value in command)
    assert "/var/run/docker.sock" not in rendered
    assert "TRIALAGENTBENCH_PROTOCOL_TOKEN=opaque-token" in command


def test_session_fails_closed_without_docker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("trialagentbench_harness.adapters.docker_code_execution.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="no host-Python fallback"):
        DockerPythonSession(cwd=tmp_path)


def test_session_rejects_ambiguous_mount_path(tmp_path: Path) -> None:
    workdir = tmp_path / "comma,path"
    workdir.mkdir()
    with pytest.raises(ValueError, match="unsupported by Docker mount syntax"):
        DockerPythonSession(cwd=workdir)


def test_session_rejects_symlinked_evidence(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("evidence", encoding="utf-8")
    (tmp_path / "alias.txt").symlink_to(target)
    with pytest.raises(ValueError, match="forbidden symlink"):
        DockerPythonSession(cwd=tmp_path)


def test_session_rejects_hard_linked_evidence(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("evidence", encoding="utf-8")
    os.link(target, tmp_path / "alias.txt")
    with pytest.raises(ValueError, match="forbidden hard link"):
        DockerPythonSession(cwd=tmp_path)


def test_session_rejects_special_file_evidence(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "named-pipe")
    with pytest.raises(ValueError, match="forbidden special file"):
        DockerPythonSession(cwd=tmp_path)


def test_session_rejects_evidence_resource_limit_violations(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="file-count limit"):
        DockerPythonSession(cwd=tmp_path, limits=CodeExecutionLimitsV1(evidence_file_count=1))

    (tmp_path / "large.bin").write_bytes(b"x" * (1024 * 1024))
    with pytest.raises(ValueError, match="byte limit"):
        DockerPythonSession(cwd=tmp_path, limits=CodeExecutionLimitsV1(evidence_mb=1))


def test_session_rejects_unicode_normalized_path_collisions(tmp_path: Path) -> None:
    (tmp_path / "caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt").write_text("one", encoding="utf-8")
    (tmp_path / "cafe\N{COMBINING ACUTE ACCENT}.txt").write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="Unicode-normalized path collision"):
        DockerPythonSession(cwd=tmp_path)


def test_executor_environment_records_immutable_image_and_complete_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_id = f"sha256:{'a' * 64}"
    packages = [
        {"name": name, "version": "1.0"}
        for name in (
            "lifelines",
            "matplotlib",
            "numpy",
            "pandas",
            "pyarrow",
            "scikit-learn",
            "scipy",
            "statsmodels",
        )
    ]

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=f'[{{"Id":"{image_id}"}}]', stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"python_version":"3.12.10","packages":' + json.dumps(packages) + "}",
            stderr="",
        )

    monkeypatch.setattr("trialagentbench_harness.adapters.docker_code_execution.shutil.which", lambda _: "docker")
    monkeypatch.setattr("trialagentbench_harness.adapters.docker_code_execution.subprocess.run", fake_run)

    environment = resolve_executor_environment(image="executor:release")

    assert environment.image_reference == "executor:release"
    assert environment.image_id == image_id
    assert environment.python_version == "3.12.10"
    assert [package.name for package in environment.packages] == [package["name"] for package in packages]
    assert environment.limits.timeout_seconds == 120.0


def test_executor_environment_rejects_missing_analysis_package(monkeypatch: pytest.MonkeyPatch) -> None:
    image_id = f"sha256:{'b' * 64}"

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=f'[{{"Id":"{image_id}"}}]', stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"python_version":"3.12.10","packages":[{"name":"numpy","version":"2.4.4"}]}',
            stderr="",
        )

    monkeypatch.setattr("trialagentbench_harness.adapters.docker_code_execution.shutil.which", lambda _: "docker")
    monkeypatch.setattr("trialagentbench_harness.adapters.docker_code_execution.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="missing required analysis packages"):
        resolve_executor_environment(image="executor:incomplete")


def test_executor_shutdown_falls_back_to_owned_process_when_group_kill_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed = []
    process = SimpleNamespace(pid=1234, poll=lambda: None, kill=lambda: killed.append(True))
    session = object.__new__(DockerPythonSession)
    session._proc = process
    monkeypatch.setattr(
        "trialagentbench_harness.adapters.docker_code_execution.os.killpg",
        lambda *_: (_ for _ in ()).throw(PermissionError("not process-group owner")),
    )

    session._kill_client_process()

    assert killed == [True]
