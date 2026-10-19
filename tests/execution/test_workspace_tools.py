"""Contract tests for the persistent agent text workspace."""

from __future__ import annotations

from pathlib import Path

import pytest
from trialagentbench_verifier.executor import DockerCodeSession

from trialagentbench.ports import CodeExecutionResultV1
from trialagentbench.ports.tool_input import ToolInputError
from trialagentbench.tools.workspace import (
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
    assert read_properties["end_line"]["type"] == ["integer", "null"]


def _schema_nodes(schema: dict) -> list[dict]:
    """Return a parameter schema and every nested property and item schema."""
    nodes = [schema]
    for child in (*schema.get("properties", {}).values(), *([schema["items"]] if "items" in schema else [])):
        nodes.extend(_schema_nodes(child))
    return nodes


@pytest.mark.parametrize("mode", ["python_r", "python", "r"])
def test_every_tool_offered_to_a_provider_types_every_schema_node(mode: str) -> None:
    import json

    from trialagentbench.tools.analysis import analysis_tools
    from trialagentbench.trialdev.actions import trialdev_programme_tools
    from trialagentbench.trialdev.task_agent import ANALYSIS_TOOLS as TRIALDEV_TOOLS
    from trialagentbench.trialeval.agent import _get_submission_tools, _get_tools

    tools = [
        *_get_tools(mode),
        *_get_submission_tools(),
        *analysis_tools(mode),
        *TRIALDEV_TOOLS,
        *trialdev_programme_tools(),
    ]
    for tool in tools:
        parameters = tool["function"]["parameters"]
        text = json.dumps(parameters)
        assert not {"$ref", "$defs", "anyOf", "oneOf", "allOf"} & {
            key for node in _schema_nodes(parameters) for key in node
        }, text
        for node in _schema_nodes(parameters):
            assert "type" in node, (tool["function"]["name"], node)
            if node["type"] == "array":
                assert "type" in node["items"], (tool["function"]["name"], node)


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
        "scratch",
        "scratch/../truth.json",
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


def test_workspace_rejects_invalid_content_and_read_ranges() -> None:
    with pytest.raises(ToolInputError, match="Invalid write_workspace_file arguments"):
        handle_workspace_tool(
            name="write_workspace_file",
            arguments={"path": "analysis.py", "content": 1},
            session=_Session(),
        )


@pytest.mark.parametrize(
    ("name", "arguments"),
    (
        ("write_workspace_file", {"path": "analysis.py", "content": "pass", "unexpected": 1}),
        ("read_workspace_file", {"path": "analysis.py", "unexpected": 1}),
        ("list_workspace_files", {"unexpected": 1}),
    ),
)
def test_workspace_runtime_rejects_arguments_excluded_by_its_schema(
    name: str,
    arguments: dict[str, object],
) -> None:
    """Keep provider-facing JSON schemas and local execution exact."""

    with pytest.raises(ToolInputError, match=f"Invalid {name} arguments"):
        handle_workspace_tool(name=name, arguments=arguments, session=_Session())
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
    session = DockerCodeSession(cwd=tmp_path)
    try:
        written = handle_workspace_tool(
            name="write_workspace_file",
            arguments={"path": "analysis/main.py", "content": "estimate = 6 * 7\nprint(estimate)\n"},
            session=session,
        )
        listed = handle_workspace_tool(name="list_workspace_files", arguments={}, session=session)
        read = handle_workspace_tool(
            name="read_workspace_file",
            arguments={"path": "scratch/analysis/main.py", "start_line": 1, "end_line": 2},
            session=session,
        )
        executed = session.execute_result("exec(open('scratch/analysis/main.py').read())")
    finally:
        session.close()

    assert written.status == "success"
    assert written.output == "Wrote scratch/analysis/main.py (33 characters)."
    assert "scratch/analysis/main.py" in listed.output  # listed paths are the paths the other tools accept
    assert read.output == "0001: estimate = 6 * 7\n0002: print(estimate)"
    assert executed.output == "42"


@pytest.mark.executor
def test_workspace_reads_supplied_task_materials(tmp_path: Path) -> None:
    """Listed participant materials are readable without copying them to scratch."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "protocol.md").write_text("# Protocol\nEstimand: risk difference\n", encoding="utf-8")
    session = DockerCodeSession(cwd=tmp_path)
    try:
        read = handle_workspace_tool(
            name="read_workspace_file",
            arguments={"path": "documents/protocol.md", "start_line": 1, "end_line": 2},
            session=session,
        )
        missing = handle_workspace_tool(name="read_workspace_file", arguments={"path": "protocol.md"}, session=session)
    finally:
        session.close()

    assert read.output == "0001: # Protocol\n0002: Estimand: risk difference"
    # A wrong path names the files with that name instead of a misleading scratch/ path.
    assert "No file at protocol.md; files with that name: documents/protocol.md" in missing.output


@pytest.mark.executor
def test_submission_file_boundary_reads_regular_utf8_and_rejects_symlink(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path)
    try:
        written = session.execute_result(
            "from pathlib import Path\n"
            "Path('scratch/submission.json').write_text('{\"task_id\":\"task-1\"}', encoding='utf-8')"
        )
        text = read_workspace_submission_text(path="submission.json", session=session)
        prefixed_text = read_workspace_submission_text(path="scratch/submission.json", session=session)
        linked = session.execute_result(
            "from pathlib import Path\nPath('scratch/submission-link.json').symlink_to('submission.json')"
        )
        with pytest.raises(ToolInputError, match="session has terminated"):
            read_workspace_submission_text(path="submission-link.json", session=session)
        session.close()
    finally:
        session.close()

    assert written.status == "success"
    assert linked.status == "session_terminated"
    assert text == '{"task_id":"task-1"}'
    assert prefixed_text == text


@pytest.mark.executor
def test_executor_supports_propensity_model_workflow(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path)
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


def test_provider_tool_rejects_a_union_that_cannot_be_one_typed_node() -> None:
    from pydantic import BaseModel

    from trialagentbench.ports.tool_input import provider_tool

    class Ambiguous(BaseModel):
        value: int | str

    with pytest.raises(ValueError, match="one type"):
        provider_tool("ambiguous", "Ambiguous argument.", Ambiguous)
