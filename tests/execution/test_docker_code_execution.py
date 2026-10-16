"""Contract tests for the isolated model-code executor."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from trialagentbench_verifier.executor import (
    DockerCodeSession,
    _docker_command,
    resolve_executor_environment,
    resolve_trusted_executor_environment,
)

from trialagentbench.contracts.core.runs import ExecutorEnvironmentV1, ExecutorPackageV1
from trialagentbench.ports.code_execution import CodeExecutionLimitsV1


def test_execution_limits_reject_unbounded_values() -> None:
    with pytest.raises(ValidationError):
        CodeExecutionLimitsV1(timeout_seconds=0)
    with pytest.raises(ValidationError):
        CodeExecutionLimitsV1(memory_mb=128)
    with pytest.raises(ValidationError):
        CodeExecutionLimitsV1(output_bytes=17 * 1024 * 1024)
    with pytest.raises(ValidationError):
        CodeExecutionLimitsV1(workspace_file_count=0)


def test_executor_image_uses_one_fully_pinned_package_lock() -> None:
    executor_root = Path(__file__).resolve().parents[2] / "executor"
    dockerfile = (executor_root / "Dockerfile").read_text(encoding="utf-8")
    requirements = (executor_root / "requirements.lock").read_text(encoding="utf-8").splitlines()

    assert "COPY requirements.lock" in dockerfile
    assert "COPY wheels" in dockerfile
    assert "r-base-core=4.5.0-3" in dockerfile
    assert "r-recommended=4.5.0-3" in dockerfile
    assert "r-cran-jsonlite=1.9.1+dfsg-1" in dockerfile
    assert "r-cran-sandwich=3.1-1-1" in dockerfile
    assert "r-cran-survey=4.4-2-2" in dockerfile
    assert "R CMD INSTALL --no-byte-compile trialagentbenchR" in dockerfile
    assert "--require-hashes -r" in dockerfile
    assert requirements
    assert requirements[0] == "--only-binary=:all:"
    assert requirements[1] == "--find-links=wheels"
    assert not any(line.startswith("--no-binary") for line in requirements)
    assert "--hash=sha256:70b354e1fd583051fcfc36aae5c5a295a9b340138384bf4b96ac5a78fa7b7699" in (
        line.strip() for line in requirements
    )
    requirement_lines = [line for line in requirements if line and not line.startswith((" ", "-", "#"))]
    assert all("==" in line and line.endswith(" \\") for line in requirement_lines)
    assert all(line.strip().startswith("--hash=sha256:") for line in requirements if line.startswith(" "))
    names = [line.split("==", maxsplit=1)[0].casefold() for line in requirement_lines]
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_docker_command_enforces_isolation_and_read_only_host_mounts(tmp_path: Path) -> None:
    participant_root = tmp_path / "participant-release"
    workdir = tmp_path / "projected-task"
    participant_root.mkdir()
    workdir.mkdir()
    (participant_root / "undeclared_truth.json").write_text("{}", encoding="utf-8")
    scratch_root = workdir / "scratch"
    scratch_root.mkdir()
    command = _docker_command(
        workdir=workdir,
        scratch_root=scratch_root,
        image="executor:test",
        limits=CodeExecutionLimitsV1(),
        container_name="executor-test",
    )
    rendered = " ".join(command)
    assert "--network none" in rendered
    # The non-dumpable supervisor is PID 1: no init process exposes the host pipes.
    assert "--init" not in command
    assert "--read-only" in command
    assert "--cap-drop ALL" in rendered
    assert "--security-opt no-new-privileges" in rendered
    assert "--pids-limit 128" in rendered
    assert "--memory 4096m" in rendered
    assert command.count("--mount") == 2
    assert any(value.endswith(f"src={workdir},dst=/evidence,readonly") for value in command)
    assert any(value.endswith(f"src={scratch_root},dst=/scratch-seed,readonly") for value in command)
    assert not any(f"src={participant_root}," in value for value in command)
    assert any(value.startswith("/workspace:rw,noexec") and "size=2048m" in value for value in command)
    assert "TRIALAGENTBENCH_WORKSPACE_FILE_COUNT=10000" in command
    assert "/var/run/docker.sock" not in rendered
    assert not any("PROTOCOL_TOKEN" in value for value in command)


def test_session_fails_closed_without_docker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("trialagentbench_verifier.executor.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="no host-Python fallback"):
        DockerCodeSession(cwd=tmp_path)


def test_original_executor_name_resolves_to_dual_language_boundary() -> None:
    assert DockerCodeSession is DockerCodeSession


def test_session_rejects_ambiguous_mount_path(tmp_path: Path) -> None:
    workdir = tmp_path / "comma,path"
    workdir.mkdir()
    with pytest.raises(ValueError, match="unsupported by Docker mount syntax"):
        DockerCodeSession(cwd=workdir)


def test_session_rejects_symlinked_evidence(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("evidence", encoding="utf-8")
    (tmp_path / "alias.txt").symlink_to(target)
    with pytest.raises(ValueError, match="forbidden symlink"):
        DockerCodeSession(cwd=tmp_path)


def test_session_rejects_hard_linked_evidence(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("evidence", encoding="utf-8")
    os.link(target, tmp_path / "alias.txt")
    with pytest.raises(ValueError, match="forbidden hard link"):
        DockerCodeSession(cwd=tmp_path)


def test_session_rejects_special_file_evidence(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "named-pipe")
    with pytest.raises(ValueError, match="forbidden special file"):
        DockerCodeSession(cwd=tmp_path)


def test_session_rejects_evidence_resource_limit_violations(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="file-count limit"):
        DockerCodeSession(cwd=tmp_path, limits=CodeExecutionLimitsV1(evidence_file_count=1))

    (tmp_path / "large.bin").write_bytes(b"x" * (1024 * 1024))
    with pytest.raises(ValueError, match="byte limit"):
        DockerCodeSession(cwd=tmp_path, limits=CodeExecutionLimitsV1(evidence_mb=1))


def test_scratch_archive_rejects_file_count_before_extraction(tmp_path: Path) -> None:
    archive_path = tmp_path / "scratch.tar"
    with tarfile.open(archive_path, "w") as archive:
        for index in range(3):
            archive.addfile(tarfile.TarInfo(f"empty-{index}.txt"))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    session = object.__new__(DockerCodeSession)
    session._limits = CodeExecutionLimitsV1(workspace_file_count=2)

    with pytest.raises(ValueError, match="file-count limit"):
        session._extract_scratch_archive(archive_path=archive_path, checkpoint=checkpoint)

    assert list(checkpoint.iterdir()) == []


def test_session_rejects_unicode_normalized_path_collisions(tmp_path: Path) -> None:
    (tmp_path / "caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt").write_text("one", encoding="utf-8")
    (tmp_path / "cafe\N{COMBINING ACUTE ACCENT}.txt").write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="Unicode-normalized path collision"):
        DockerCodeSession(cwd=tmp_path)


@pytest.mark.parametrize("bridge_mode", ["0644", "0600"])
def test_executor_environment_records_immutable_image_and_complete_inventory(
    monkeypatch: pytest.MonkeyPatch, bridge_mode: str
) -> None:
    image_id = f"sha256:{'a' * 64}"
    packages = [
        {"name": name, "version": "1.0"}
        for name in (
            "jsonschema",
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
    r_packages = [
        {"name": name, "version": "1.0"}
        for name in (
            "base",
            "jsonlite",
            "MASS",
            "Matrix",
            "sandwich",
            "survey",
            "survival",
            "trialagentbenchR",
        )
    ]

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=f'[{{"Id":"{image_id}"}}]', stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                '{"python_version":"3.12.10","operating_system":"Linux","architecture":"x86_64",'
                '"system_packages":[{"name":"libc6","version":"2.41-12"}],"packages":'
                + json.dumps(packages)
                + ',"r_version":"4.5.0","r_packages":'
                + json.dumps(r_packages)
                + ',"support_files":'
                + json.dumps(
                    [{"path": "/opt/trialagentbench/parquet_to_csv.py", "mode": bridge_mode, "sha256": "c" * 64}]
                )
                + ',"cgroup_metrics":{"cpu":true,"memory_peak":true,"pids_peak":true}}'
            ),
            stderr="",
        )

    monkeypatch.setattr("trialagentbench_verifier.executor.shutil.which", lambda _: "docker")
    monkeypatch.setattr("trialagentbench_verifier.executor.subprocess.run", fake_run)

    if bridge_mode == "0600":
        # A bridge the non-root participant cannot read breaks every R parquet load.
        with pytest.raises(RuntimeError, match="world-readable"):
            resolve_executor_environment(image="executor:release")
        return
    environment = resolve_executor_environment(image="executor:release")
    assert environment.support_files[0].mode == bridge_mode
    # The semantic hash sees file modes, not only package inventories.
    group_only = environment.model_copy(
        update={"support_files": (environment.support_files[0].model_copy(update={"mode": "0640"}),)}
    )
    assert group_only.semantic_sha256() != environment.semantic_sha256()

    assert environment.image_reference == "executor:release"
    assert environment.image_id == image_id
    assert environment.python_version == "3.12.10"
    assert environment.r_version == "4.5.0"
    assert [package.name for package in environment.packages] == [package["name"] for package in packages]
    assert [package.name for package in environment.r_packages] == [package["name"] for package in r_packages]
    assert environment.limits.timeout_seconds == 120.0


def test_executor_environment_rejects_missing_analysis_package(monkeypatch: pytest.MonkeyPatch) -> None:
    image_id = f"sha256:{'b' * 64}"

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=f'[{{"Id":"{image_id}"}}]', stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                '{"python_version":"3.12.10","operating_system":"Linux","architecture":"x86_64",'
                '"system_packages":[{"name":"libc6","version":"2.41-12"}],'
                '"packages":[{"name":"numpy","version":"2.4.4"}],'
                '"r_version":"4.5.0","r_packages":['
                '{"name":"base","version":"4.5.0"},'
                '{"name":"jsonlite","version":"1.9.1"},'
                '{"name":"MASS","version":"7.3"},'
                '{"name":"Matrix","version":"1.7"},'
                '{"name":"sandwich","version":"3.1"},'
                '{"name":"survey","version":"4.4"},'
                '{"name":"survival","version":"3.8"},'
                '{"name":"trialagentbenchR","version":"1.0"}],"support_files":[],'
                '"cgroup_metrics":{"cpu":true,"memory_peak":true,"pids_peak":true}}'
            ),
            stderr="",
        )

    monkeypatch.setattr("trialagentbench_verifier.executor.shutil.which", lambda _: "docker")
    monkeypatch.setattr("trialagentbench_verifier.executor.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="missing required analysis packages"):
        resolve_executor_environment(image="executor:incomplete")


def test_trusted_executor_environment_requires_both_external_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a wrong image or inventory before a collaborator run can start."""

    limits = CodeExecutionLimitsV1()
    environment = ExecutorEnvironmentV1(
        image_reference="executor:release",
        image_id=f"sha256:{'a' * 64}",
        python_version="3.12.10",
        packages=(ExecutorPackageV1(name="fixture", version="1.0"),),
        limits=limits,
    )
    monkeypatch.setattr(
        "trialagentbench_verifier.executor.resolve_executor_environment",
        lambda *, image, limits: environment,
    )
    expected_environment = environment.semantic_sha256()
    assert (
        resolve_trusted_executor_environment(
            image="executor:release",
            expected_image_id=environment.image_id,
            expected_environment_sha256=expected_environment,
            limits=limits,
        )
        == environment
    )
    with pytest.raises(ValueError, match="immutable ID"):
        resolve_trusted_executor_environment(
            image="executor:release",
            expected_image_id=f"sha256:{'b' * 64}",
            expected_environment_sha256=expected_environment,
            limits=limits,
        )
    with pytest.raises(ValueError, match="trusted SHA-256"):
        resolve_trusted_executor_environment(
            image="executor:release",
            expected_image_id=environment.image_id,
            expected_environment_sha256="c" * 64,
            limits=limits,
        )


def test_executor_shutdown_falls_back_to_owned_process_when_group_kill_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed = []
    process = SimpleNamespace(pid=1234, poll=lambda: None, kill=lambda: killed.append(True))
    session = object.__new__(DockerCodeSession)
    session._proc = process
    monkeypatch.setattr(
        "trialagentbench_verifier.executor.os.killpg",
        lambda *_: (_ for _ in ()).throw(PermissionError("not process-group owner")),
    )

    session._kill_client_process()

    assert killed == [True]


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [
        (1, "", "container not available"),
        (0, "usage_usec 100\n0\n2\n", ""),
        (0, "usage_usec 99\n1024\n2\n", ""),
        (0, "usage_usec 100\nusage_usec 101\n1024\n2\n", ""),
        (0, "usage_usec invalid\n1024\n2\n", ""),
    ],
)
def test_kernel_accounting_rejects_failed_missing_and_decreased_counters(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str, stderr: str
) -> None:
    session = object.__new__(DockerCodeSession)
    session._container_name = "test-accounting"
    session._cpu_usage_usec = 100
    monkeypatch.setattr(
        "trialagentbench_verifier.executor.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr),
    )
    with pytest.raises(RuntimeError, match="kernel accounting"):
        session._execution_usage()
    assert session._cpu_usage_usec == 100


@pytest.mark.parametrize("participant_uid", [0, 1000, 65534])
def test_kernel_accounting_uses_a_separate_identity_and_cumulative_cpu(
    monkeypatch: pytest.MonkeyPatch, participant_uid: int
) -> None:
    session = object.__new__(DockerCodeSession)
    session._container_name = "test-accounting"
    session._cpu_usage_usec = 100
    counters = iter(("usage_usec 150\n1024\n3\n", "usage_usec 175\n2048\n4\n"))

    def measured(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:2] == ["docker", "exec"]
        assert int(command[command.index("--user") + 1].split(":")[0]) != participant_uid
        assert command[command.index(session._container_name) + 1] == "/bin/cat"
        assert kwargs["timeout"] == 10
        assert kwargs["stdin"] == subprocess.DEVNULL
        return subprocess.CompletedProcess(command, 0, stdout=next(counters), stderr="")

    monkeypatch.setattr("trialagentbench_verifier.executor.os.getuid", lambda: participant_uid)
    monkeypatch.setattr("trialagentbench_verifier.executor.subprocess.run", measured)
    assert session._execution_usage() == (0.00005, 1024, 3)
    assert session._execution_usage() == (0.000025, 2048, 4)
