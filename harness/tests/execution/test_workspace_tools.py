"""Contract tests for the persistent agent text workspace."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialagentbench_harness.adapters.docker_code_execution import DockerPythonSession
from trialagentbench_harness.ports import CodeExecutionResultV1
from trialagentbench_harness.ports.tool_input import ToolInputError
from trialagentbench_harness.tools.workspace import (
    WORKSPACE_TOOLS,
    handle_workspace_tool,
    read_workspace_submission_text,
)


class _Session:
    def execute(self, code: str) -> str:
        return code

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        return CodeExecutionResultV1(status="success", output=code, elapsed_seconds=0.1)

    def close(self) -> None:
        return None


def test_workspace_tool_schemas_are_closed_and_bounded() -> None:
    by_name = {tool["function"]["name"]: tool for tool in WORKSPACE_TOOLS}

    assert set(by_name) == {"write_workspace_file", "read_workspace_file", "list_workspace_files"}
    assert all(tool["function"]["parameters"]["additionalProperties"] is False for tool in by_name.values())
    read_properties = by_name["read_workspace_file"]["function"]["parameters"]["properties"]
    assert read_properties["start_line"]["minimum"] == 1
    assert read_properties["end_line"]["minimum"] == 1


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "../truth.json",
        "analysis/../../truth.json",
        Path("/", "tmp", "analysis.py").as_posix(),
        "windows\\analysis.py",
        "line\nbreak.py",
        "scratch/analysis.py",
        "x" * 241,
    ],
)
def test_workspace_rejects_paths_outside_its_posix_boundary(path: str) -> None:
    with pytest.raises(ToolInputError):
        handle_workspace_tool(
            name="write_workspace_file",
            arguments={"path": path, "content": "print('safe')"},
            session=_Session(),
        )


def test_workspace_explains_that_paths_are_scratch_relative() -> None:
    with pytest.raises(
        ToolInputError,
        match=r"use 'submission\.json' instead of 'scratch/submission\.json'",
    ):
        read_workspace_submission_text(
            path="scratch/submission.json",
            session=_Session(),
        )


def test_workspace_rejects_invalid_content_and_read_ranges() -> None:
    with pytest.raises(ToolInputError, match="content must be a string"):
        handle_workspace_tool(
            name="write_workspace_file",
            arguments={"path": "analysis.py", "content": 1},
            session=_Session(),
        )
    with pytest.raises(ToolInputError, match="greater than or equal"):
        handle_workspace_tool(
            name="read_workspace_file",
            arguments={"path": "analysis.py", "start_line": 5, "end_line": 4},
            session=_Session(),
        )
    with pytest.raises(ToolInputError, match="at most 400 lines"):
        handle_workspace_tool(
            name="read_workspace_file",
            arguments={"path": "analysis.py", "start_line": 1, "end_line": 401},
            session=_Session(),
        )


@pytest.mark.executor
def test_workspace_persists_files_for_review_and_code_execution(tmp_path: Path) -> None:
    session = DockerPythonSession(cwd=tmp_path)
    try:
        written = handle_workspace_tool(
            name="write_workspace_file",
            arguments={"path": "analysis/main.py", "content": "estimate = 6 * 7\nprint(estimate)\n"},
            session=session,
        )
        listed = handle_workspace_tool(name="list_workspace_files", arguments={}, session=session)
        read = handle_workspace_tool(
            name="read_workspace_file",
            arguments={"path": "analysis/main.py", "start_line": 1, "end_line": 2},
            session=session,
        )
        executed = session.execute_result("exec(open('scratch/analysis/main.py').read())")
    finally:
        session.close()

    assert written.status == "success"
    assert written.output == "Wrote analysis/main.py (33 characters)."
    assert "analysis/main.py" in listed.output
    assert "scratch/analysis/main.py" not in listed.output
    assert read.output == "0001: estimate = 6 * 7\n0002: print(estimate)"
    assert executed.output == "42"


@pytest.mark.executor
def test_submission_file_boundary_reads_regular_utf8_and_rejects_symlink(tmp_path: Path) -> None:
    session = DockerPythonSession(cwd=tmp_path)
    try:
        written = session.execute_result(
            "from pathlib import Path\n"
            "Path('scratch/submission.json').write_text('{\"task_id\":\"task-1\"}', encoding='utf-8')"
        )
        text = read_workspace_submission_text(path="submission.json", session=session)
        linked = session.execute_result(
            "from pathlib import Path\nPath('scratch/submission-link.json').symlink_to('submission.json')"
        )
        with pytest.raises(ToolInputError, match="regular non-symlink"):
            read_workspace_submission_text(path="submission-link.json", session=session)
        with pytest.raises(ValueError, match="linked or special file"):
            session.close()
    finally:
        session.close()

    assert written.status == "success"
    assert linked.status == "success"
    assert text == '{"task_id":"task-1"}'


@pytest.mark.executor
def test_executor_supports_propensity_model_workflow(tmp_path: Path) -> None:
    session = DockerPythonSession(cwd=tmp_path)
    try:
        result = session.execute_result("""
import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([[0.0], [0.2], [0.8], [1.0]])
treatment = np.array([0, 0, 1, 1])
model = LogisticRegression(random_state=0).fit(x, treatment)
propensity = model.predict_proba(x)[:, 1]
print(model.n_features_in_, len(propensity), bool(np.all((propensity > 0) & (propensity < 1))))
""")
    finally:
        session.close()

    assert result.status == "success"
    assert result.output == "1 4 True"
