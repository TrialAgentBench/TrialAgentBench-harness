"""Docker-backed isolated Python execution for model-generated code."""

from __future__ import annotations

import base64
import json
import os
import queue
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Final, Literal, TextIO, cast

from trialagentbench_harness.contracts.core.runs import ExecutorEnvironmentV1
from trialagentbench_harness.ports.code_execution import CodeExecutionLimitsV1, CodeExecutionResultV1

DEFAULT_EXECUTOR_IMAGE: Final = "trialagentbench/executor:0.1.0"
REQUIRED_ANALYSIS_PACKAGES: Final = (
    "lifelines",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "statsmodels",
)
_RESULT_PREFIX: Final = "__TRIALAGENTBENCH_RESULT__"
_WORKER = r"""
import base64
import json
import os
import pathlib
import shutil
import signal
import stat
import sys
import threading
import traceback

protocol_token = os.environ.pop("TRIALAGENTBENCH_PROTOCOL_TOKEN")
result_prefix = "__TRIALAGENTBENCH_RESULT__" + protocol_token + ":"

def restore_scratch():
    source_root = pathlib.Path("/scratch-seed")
    destination_root = pathlib.Path("/workspace/scratch")
    destination_root.mkdir(parents=True, exist_ok=True)
    for source in source_root.rglob("*"):
        if source.is_symlink():
            raise RuntimeError(f"Scratch checkpoint contains a forbidden symlink: {source}")
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        source_info = source.stat(follow_symlinks=False)
        if not stat.S_ISREG(source_info.st_mode) or source_info.st_nlink != 1:
            raise RuntimeError(f"Scratch checkpoint contains a forbidden file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

def sync_evidence():
    source_root = pathlib.Path("/evidence")
    destination_root = pathlib.Path("/workspace")
    maximum_bytes = int(os.environ["TRIALAGENTBENCH_EVIDENCE_BYTES"])
    maximum_files = int(os.environ["TRIALAGENTBENCH_EVIDENCE_FILE_COUNT"])
    total_bytes = 0
    file_count = 0
    for source in source_root.rglob("*"):
        relative = source.relative_to(source_root)
        if relative.parts and relative.parts[0] == "scratch":
            continue
        if source.is_symlink():
            raise RuntimeError(f"Participant evidence contains a forbidden symlink: {source}")
        destination = destination_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        source_info = source.stat(follow_symlinks=False)
        if not stat.S_ISREG(source_info.st_mode):
            raise RuntimeError(f"Participant evidence contains a forbidden special file: {source}")
        if source_info.st_nlink != 1:
            raise RuntimeError(f"Participant evidence contains a forbidden hard link: {source}")
        file_count += 1
        total_bytes += source_info.st_size
        if file_count > maximum_files:
            raise RuntimeError("Participant evidence exceeds the configured file-count limit")
        if total_bytes > maximum_bytes:
            raise RuntimeError("Participant evidence exceeds the configured byte limit")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or source_info.st_size != destination.stat().st_size or source_info.st_mtime_ns != destination.stat().st_mtime_ns:
            shutil.copy2(source, destination)

class ExecutionTimedOut(BaseException):
    pass

def execute_captured(code, namespace, limit, timeout_seconds):
    read_fd, write_fd = os.pipe()
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    captured = bytearray()
    capture_state = {"truncated": False}

    def drain():
        while True:
            try:
                chunk = os.read(read_fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            remaining = limit - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])
            if len(chunk) > remaining:
                capture_state["truncated"] = True

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    status = "success"

    def timeout_handler(_signum, _frame):
        raise ExecutionTimedOut()

    def terminate_descendants():
        pending = [os.getpid()]
        descendants = []
        while pending:
            parent = pending.pop()
            children_path = pathlib.Path(f"/proc/{parent}/task/{parent}/children")
            try:
                children = [int(value) for value in children_path.read_text().split()]
            except (FileNotFoundError, ProcessLookupError):
                children = []
            descendants.extend(children)
            pending.extend(children)
        for process_id in reversed(descendants):
            try:
                os.kill(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        try:
            exec(compile(code, "<agent-code>", "exec"), namespace, namespace)
        except ExecutionTimedOut:
            status = "timeout"
            print(f"Execution timed out after {timeout_seconds:g}s.")
        except BaseException:
            status = "execution_error"
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)
        terminate_descendants()
    reader.join(timeout=1.0)
    if reader.is_alive():
        capture_state["truncated"] = True
    os.close(read_fd)
    return status, captured.decode("utf-8", errors="replace"), capture_state["truncated"]

namespace = {"__name__": "__main__"}
restore_scratch()
for raw in iter(input, ""):
    request = json.loads(raw)
    request_id = request["id"]
    code = base64.b64decode(request["code_b64"]).decode("utf-8")
    sync_evidence()
    status, output, truncated = execute_captured(
        code,
        namespace,
        int(os.environ["TRIALAGENTBENCH_OUTPUT_BYTES"]),
        float(request["timeout_seconds"]),
    )
    payload = json.dumps({"id": request_id, "status": status, "output": output, "truncated": truncated})
    print(result_prefix + base64.b64encode(payload.encode()).decode(), flush=True)
"""

_ENVIRONMENT_PROBE = r"""
import importlib.metadata
import json
import platform

packages = sorted(
    (
        {"name": distribution.metadata["Name"], "version": distribution.version}
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    ),
    key=lambda package: package["name"].casefold(),
)
print(json.dumps({"python_version": platform.python_version(), "packages": packages}, separators=(",", ":")))
"""


def resolve_executor_environment(
    *,
    image: str | None = None,
    limits: CodeExecutionLimitsV1 | None = None,
) -> ExecutorEnvironmentV1:
    """Resolve and validate the immutable isolated analysis environment."""

    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for isolated code execution; no host-Python fallback is permitted.")
    image_reference = image or os.environ.get("TRIALAGENTBENCH_EXECUTOR_IMAGE", DEFAULT_EXECUTOR_IMAGE)
    inspection = subprocess.run(
        ["docker", "image", "inspect", image_reference],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if inspection.returncode != 0:
        detail = inspection.stderr.strip() or "image not found"
        raise RuntimeError(
            f"Isolated executor image {image_reference!r} is unavailable ({detail}). "
            "Build executor/Dockerfile or set TRIALAGENTBENCH_EXECUTOR_IMAGE."
        )
    inspection_payload = json.loads(inspection.stdout)
    if not isinstance(inspection_payload, list) or len(inspection_payload) != 1:
        raise RuntimeError("Docker returned an invalid executor image inspection payload.")
    image_document = inspection_payload[0]
    if not isinstance(image_document, dict):
        raise RuntimeError("Docker returned a non-object executor image inspection record.")
    image_id = image_document.get("Id")
    if not isinstance(image_id, str):
        raise RuntimeError("Docker executor image inspection omitted its immutable image ID.")

    probe = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            image_id,
            "python",
            "-c",
            _ENVIRONMENT_PROBE,
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip() or "environment probe failed"
        raise RuntimeError(f"Executor environment probe failed: {detail}")
    probe_payload = json.loads(probe.stdout)
    if not isinstance(probe_payload, dict):
        raise RuntimeError("Executor environment probe returned a non-object payload.")
    environment = ExecutorEnvironmentV1.model_validate(
        {
            "image_reference": image_reference,
            "image_id": image_id,
            "python_version": probe_payload.get("python_version"),
            "packages": probe_payload.get("packages"),
            "limits": (limits or CodeExecutionLimitsV1()).model_dump(mode="json"),
        }
    )
    installed = {package.name.casefold() for package in environment.packages}
    missing = sorted(set(REQUIRED_ANALYSIS_PACKAGES) - installed)
    if missing:
        raise RuntimeError(f"Executor image is missing required analysis packages: {', '.join(missing)}")
    return environment


def _validate_evidence_tree(root: Path, limits: CodeExecutionLimitsV1) -> None:
    """Reject evidence trees that cannot be mounted and copied safely."""
    normalized_paths: set[str] = set()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        relative = path.relative_to(root).as_posix()
        if Path(relative).parts[0] == "scratch":
            continue
        normalized = unicodedata.normalize("NFC", relative)
        if normalized in normalized_paths:
            raise ValueError(f"Participant evidence has a Unicode-normalized path collision: {relative}")
        normalized_paths.add(normalized)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Participant evidence contains a forbidden symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Participant evidence contains a forbidden special file: {relative}")
        if info.st_nlink != 1:
            raise ValueError(f"Participant evidence contains a forbidden hard link: {relative}")
        file_count += 1
        total_bytes += info.st_size
        if file_count > limits.evidence_file_count:
            raise ValueError(
                f"Participant evidence exceeds the configured file-count limit ({limits.evidence_file_count})"
            )
        maximum_bytes = limits.evidence_mb * 1024 * 1024
        if total_bytes > maximum_bytes:
            raise ValueError(f"Participant evidence exceeds the configured byte limit ({limits.evidence_mb} MiB)")


def _validate_scratch_tree(root: Path, limits: CodeExecutionLimitsV1) -> None:
    """Reject scratch checkpoints that cannot be restored safely."""

    total_bytes = 0
    for path in root.rglob("*"):
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Agent scratch workspace contains a forbidden symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Agent scratch workspace contains a forbidden linked or special file: {relative}")
        total_bytes += info.st_size
        if total_bytes > limits.workspace_mb * 1024 * 1024:
            raise ValueError(f"Agent scratch workspace exceeds the configured limit ({limits.workspace_mb} MiB)")


def _docker_command(
    *,
    workdir: Path,
    scratch_root: Path,
    image: str,
    limits: CodeExecutionLimitsV1,
    container_name: str,
    protocol_token: str,
) -> list[str]:
    uid = os.getuid() if hasattr(os, "getuid") else 65534
    gid = os.getgid() if hasattr(os, "getgid") else 65534
    return [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--name",
        container_name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(limits.process_limit),
        "--memory",
        f"{limits.memory_mb}m",
        "--cpus",
        str(limits.cpu_count),
        "--user",
        f"{uid}:{gid}",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=256m",
        "--mount",
        f"type=bind,src={workdir},dst=/evidence,readonly",
        "--tmpfs",
        f"/workspace:rw,noexec,nosuid,nodev,size={limits.workspace_mb}m,mode=0770,uid={uid},gid={gid}",
        "--mount",
        f"type=bind,src={scratch_root},dst=/scratch-seed,readonly",
        "--workdir",
        "/workspace",
        "--env",
        "PYTHONHASHSEED=0",
        "--env",
        "HOME=/workspace",
        "--env",
        "XDG_CACHE_HOME=/workspace/.cache",
        "--env",
        "MPLCONFIGDIR=/workspace/.config/matplotlib",
        "--env",
        "OPENBLAS_NUM_THREADS=1",
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "MKL_NUM_THREADS=1",
        "--env",
        "NUMEXPR_NUM_THREADS=1",
        "--env",
        "VECLIB_MAXIMUM_THREADS=1",
        "--env",
        f"TRIALAGENTBENCH_OUTPUT_BYTES={limits.output_bytes}",
        "--env",
        f"TRIALAGENTBENCH_EVIDENCE_BYTES={limits.evidence_mb * 1024 * 1024}",
        "--env",
        f"TRIALAGENTBENCH_EVIDENCE_FILE_COUNT={limits.evidence_file_count}",
        "--env",
        f"TRIALAGENTBENCH_PROTOCOL_TOKEN={protocol_token}",
        image,
        "python",
        "-u",
        "-c",
        _WORKER,
    ]


class DockerPythonSession:
    """Persistent Python session contained by a locked-down Docker runtime."""

    def __init__(
        self,
        cwd: str | Path | None = None,
        timeout: int = 120,
        *,
        image: str | None = None,
        limits: CodeExecutionLimitsV1 | None = None,
    ) -> None:
        workdir = Path(cwd or Path.cwd()).resolve(strict=True)
        if not workdir.is_dir():
            raise NotADirectoryError(f"Code-execution workdir is not a directory: {workdir}")
        if any(character in str(workdir) for character in (",", "\n", "\r")):
            raise ValueError("Code-execution workdir contains characters unsupported by Docker mount syntax.")
        self._limits = limits or CodeExecutionLimitsV1(timeout_seconds=float(timeout))
        _validate_evidence_tree(workdir, self._limits)
        self._workdir = workdir
        self._scratch_root = workdir / "scratch"
        self._scratch_root.mkdir(mode=0o700, exist_ok=True)
        if self._scratch_root.is_symlink() or not self._scratch_root.is_dir():
            raise ValueError("Agent scratch workspace must be a regular directory.")
        _validate_scratch_tree(self._scratch_root, self._limits)
        if shutil.which("docker") is None:
            raise RuntimeError("Docker is required for isolated code execution; no host-Python fallback is permitted.")
        self._image = image or os.environ.get("TRIALAGENTBENCH_EXECUTOR_IMAGE", DEFAULT_EXECUTOR_IMAGE)
        self._closed = False
        self._verify_image()
        self._launch_runtime()
        prime = self.execute_result("import numpy as np\nimport pandas as pd")
        if prime.status != "success":
            self.close()
            raise RuntimeError(f"Executor image cannot import required analysis libraries: {prime.output}")

    def _launch_runtime(self) -> None:
        """Start one isolated interpreter against the retained scratch mount."""

        self._container_name = f"trialagentbench-{uuid.uuid4().hex}"
        self._protocol_token = uuid.uuid4().hex
        self._result_prefix = f"{_RESULT_PREFIX}{self._protocol_token}:"
        self._request_index = 0
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._proc = self._start(self._workdir)
        if self._proc.stdout is None:
            self.close()
            raise RuntimeError("Docker executor did not expose stdout.")
        self._reader = threading.Thread(
            target=self._read_stdout,
            args=(self._proc.stdout, self._lines),
            daemon=True,
        )
        self._reader.start()

    def _restart_after_hard_timeout(self) -> None:
        """Replace an unresponsive interpreter while retaining scratch files."""

        self._terminate_runtime(checkpoint_scratch=True)
        self._launch_runtime()
        prime = self.execute_result("import numpy as np\nimport pandas as pd")
        if prime.status != "success":
            self.close()
            raise RuntimeError(f"Restarted executor cannot import required analysis libraries: {prime.output}")

    def _verify_image(self) -> None:
        result = subprocess.run(
            ["docker", "image", "inspect", self._image],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "image not found"
            raise RuntimeError(
                f"Isolated executor image {self._image!r} is unavailable ({detail}). "
                "Build executor/Dockerfile or set TRIALAGENTBENCH_EXECUTOR_IMAGE."
            )

    def _start(self, workdir: Path) -> subprocess.Popen[str]:
        command = _docker_command(
            workdir=workdir,
            scratch_root=self._scratch_root,
            image=self._image,
            limits=self._limits,
            container_name=self._container_name,
            protocol_token=self._protocol_token,
        )
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env={"PATH": os.environ.get("PATH", "")},
        )

    @staticmethod
    def _read_stdout(stream: TextIO, lines: queue.Queue[str | None]) -> None:
        try:
            for line in stream:
                lines.put(line.rstrip("\n"))
        finally:
            lines.put(None)

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        """Execute one block and return its typed bounded result."""
        started = time.monotonic()
        if self._closed or self._proc.poll() is not None or self._proc.stdin is None:
            return CodeExecutionResultV1(
                status="session_terminated",
                output="Python execution session has terminated.",
                elapsed_seconds=time.monotonic() - started,
            )
        self._request_index += 1
        request_id = str(self._request_index)
        request = {
            "id": request_id,
            "code_b64": base64.b64encode(code.encode("utf-8")).decode("ascii"),
            "timeout_seconds": self._limits.timeout_seconds,
        }
        try:
            self._proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.close()
            return CodeExecutionResultV1(
                status="session_terminated",
                output="Python execution session pipe closed.",
                elapsed_seconds=time.monotonic() - started,
            )

        incidental: list[str] = []
        # The worker owns the scientific execution timeout and returns a typed
        # result. This outer margin is only a hard kill for an unresponsive
        # interpreter or native extension.
        deadline = started + self._limits.timeout_seconds + 5.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._restart_after_hard_timeout()
                return CodeExecutionResultV1(
                    status="timeout",
                    output=(
                        f"Execution timed out after {self._limits.timeout_seconds:g}s. "
                        "The Python session was reset; scratch files were retained but in-memory variables were lost."
                    ),
                    elapsed_seconds=time.monotonic() - started,
                )
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                self.close()
                detail = "\n".join(incidental).strip()
                return CodeExecutionResultV1(
                    status="session_terminated",
                    output=detail or "Python execution session terminated before returning a result.",
                    elapsed_seconds=time.monotonic() - started,
                )
            if not line.startswith(self._result_prefix):
                incidental.append(line)
                continue
            payload = json.loads(base64.b64decode(line.removeprefix(self._result_prefix)).decode("utf-8"))
            if payload.get("id") != request_id:
                incidental.append(line)
                continue
            output = "\n".join((*incidental, str(payload.get("output", "")))).strip()
            encoded = output.encode("utf-8")
            truncated = bool(payload.get("truncated")) or len(encoded) > self._limits.output_bytes
            if truncated:
                output = encoded[: self._limits.output_bytes].decode("utf-8", errors="ignore")
                output += "\n[output truncated]"
            status = str(payload["status"])
            if status not in {"success", "execution_error", "timeout"}:
                self.close()
                raise RuntimeError(f"Executor returned an invalid status: {status!r}")
            return CodeExecutionResultV1(
                status=cast(Literal["success", "execution_error", "timeout"], status),
                output=output,
                output_truncated=truncated,
                elapsed_seconds=time.monotonic() - started,
            )

    def execute(self, code: str) -> str:
        """Execute code and return the agent-facing text response."""
        result = self.execute_result(code)
        if result.output:
            return result.output
        if result.status == "success":
            return "(code executed successfully; no stdout produced)"
        return f"[{result.status}]"

    def close(self) -> None:
        """Kill the container and its client process group."""
        if self._closed:
            return
        self._closed = True
        self._terminate_runtime(checkpoint_scratch=True)

    def snapshot_scratch(self) -> Path:
        """Persist the live tmpfs scratch tree without ending the session."""

        if self._closed or self._proc.poll() is not None:
            raise RuntimeError("Cannot snapshot a terminated code-execution session.")
        checkpoint = self._copy_scratch_checkpoint()
        if checkpoint is None:
            raise RuntimeError("Code-execution session ended before scratch could be snapshotted.")
        self._install_scratch_checkpoint(checkpoint)
        return self._scratch_root

    def _terminate_runtime(self, *, checkpoint_scratch: bool) -> None:
        """Terminate the current container and local client process."""

        try:
            scratch_checkpoint = self._copy_scratch_checkpoint() if checkpoint_scratch else None
        finally:
            subprocess.run(
                ["docker", "rm", "--force", self._container_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if self._proc.poll() is None:
                self._kill_client_process()
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            if self._proc.stdout is not None:
                self._proc.stdout.close()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._kill_client_process()
                self._proc.wait(timeout=5)
        if scratch_checkpoint is not None:
            self._install_scratch_checkpoint(scratch_checkpoint)

    def _kill_client_process(self) -> None:
        """Terminate the executor process group, falling back to its owned child."""

        try:
            os.killpg(self._proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except PermissionError:
            if self._proc.poll() is None:
                self._proc.kill()

    def _install_scratch_checkpoint(self, checkpoint: Path) -> None:
        """Atomically replace the host scratch seed with a validated snapshot."""

        staged = self._scratch_root.with_name(f".{self._scratch_root.name}.{uuid.uuid4().hex}.tmp")
        previous = self._scratch_root.with_name(f".{self._scratch_root.name}.{uuid.uuid4().hex}.previous")
        try:
            shutil.copytree(checkpoint, staged)
            _validate_scratch_tree(staged, self._limits)
            self._scratch_root.rename(previous)
            try:
                staged.rename(self._scratch_root)
            except OSError:
                previous.rename(self._scratch_root)
                raise
            shutil.rmtree(previous)
        finally:
            shutil.rmtree(checkpoint, ignore_errors=True)
            shutil.rmtree(staged, ignore_errors=True)

    def _copy_scratch_checkpoint(self) -> Path | None:
        """Copy the bounded scratch tmpfs before terminating a live container."""

        if self._proc.poll() is not None:
            return None
        checkpoint = Path(tempfile.mkdtemp(prefix="trialagentbench-scratch-"))
        archive_path = checkpoint.with_suffix(".tar")
        archive_script = (
            "import sys, tarfile\n"
            "with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:\n"
            "    archive.add('/workspace/scratch', arcname='.')\n"
        )
        try:
            with archive_path.open("wb") as archive_stream:
                result = subprocess.run(
                    ["docker", "exec", self._container_name, "python", "-c", archive_script],
                    stdin=subprocess.DEVNULL,
                    stdout=archive_stream,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=max(30.0, min(300.0, float(self._limits.workspace_mb))),
                )
            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="replace").strip() or "scratch export failed"
                raise RuntimeError(f"Unable to checkpoint the agent scratch workspace: {detail}")
            self._extract_scratch_archive(archive_path=archive_path, checkpoint=checkpoint)
            _validate_scratch_tree(checkpoint, self._limits)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, tarfile.TarError, ValueError):
            shutil.rmtree(checkpoint)
            archive_path.unlink(missing_ok=True)
            raise
        archive_path.unlink()
        return checkpoint

    def _extract_scratch_archive(self, *, archive_path: Path, checkpoint: Path) -> None:
        """Extract regular scratch files without trusting archive paths or links."""

        total_bytes = 0
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Scratch archive contains an unsafe path: {member.name}")
                if member.isdir():
                    (checkpoint / relative).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"Scratch archive contains a linked or special file: {member.name}")
                total_bytes += member.size
                if total_bytes > self._limits.workspace_mb * 1024 * 1024:
                    raise ValueError(
                        f"Agent scratch workspace exceeds the configured limit ({self._limits.workspace_mb} MiB)"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Scratch archive omitted file content: {member.name}")
                destination = checkpoint / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

    def __enter__(self) -> DockerPythonSession:
        """Return the active Docker-backed Python session."""

        return self

    def __exit__(self, *_: object) -> None:
        """Close the Docker-backed Python session on context exit."""

        self.close()


__all__ = [
    "DEFAULT_EXECUTOR_IMAGE",
    "REQUIRED_ANALYSIS_PACKAGES",
    "DockerPythonSession",
    "resolve_executor_environment",
]
