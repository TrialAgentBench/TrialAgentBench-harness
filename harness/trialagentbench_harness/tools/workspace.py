"""Persistent text-file workspace tools shared by both benchmark agents."""

from __future__ import annotations

import base64
import binascii
from pathlib import PurePosixPath
from typing import Final

from trialagentbench_harness.ports import CodeExecutionResultV1, CodeExecutionSession
from trialagentbench_harness.ports.llm_provider import JsonValue
from trialagentbench_harness.ports.tool_input import ToolInputError

_MAX_FILE_CHARS: Final = 262_144
_MAX_READ_LINES: Final = 400
_MAX_PATH_CHARS: Final = 240
_MAX_SUBMISSION_FILE_BYTES: Final = 65_536

WORKSPACE_TOOLS: Final[list[dict[str, JsonValue]]] = [
    {
        "type": "function",
        "function": {
            "name": "write_workspace_file",
            "description": (
                "Write a UTF-8 text file under the persistent scratch/ workspace. "
                "Use this for reusable Python scripts, notes, intermediate JSON, or CSV."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "description": "Relative path under scratch/, e.g. analysis.py"},
                    "content": {"type": "string", "description": "Complete UTF-8 file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_workspace_file",
            "description": (
                "Read a line range from a UTF-8 text file under scratch/. "
                "Use this to review or debug saved analysis code and notes."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "description": "Relative path under scratch/"},
                    "start_line": {"type": "integer", "minimum": 1, "description": "First line, inclusive"},
                    "end_line": {"type": "integer", "minimum": 1, "description": "Last line, inclusive"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_workspace_files",
            "description": "List persistent files under scratch/ with their byte sizes.",
            "parameters": {"type": "object", "additionalProperties": False, "properties": {}},
        },
    },
]


def handle_workspace_tool(
    *,
    name: str,
    arguments: dict[str, object],
    session: CodeExecutionSession,
) -> CodeExecutionResultV1:
    """Execute one validated workspace operation in the persistent container."""

    if name == "write_workspace_file":
        path = _workspace_path(arguments.get("path"))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ToolInputError("write_workspace_file content must be a string.")
        if len(content) > _MAX_FILE_CHARS:
            raise ToolInputError(f"write_workspace_file content exceeds {_MAX_FILE_CHARS} characters.")
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        code = (
            "import base64\n"
            "from pathlib import Path\n"
            f"_path = Path({str(path)!r})\n"
            "_relative = _path.relative_to('scratch')\n"
            "_path.parent.mkdir(parents=True, exist_ok=True)\n"
            f"_content = base64.b64decode({encoded!r}).decode('utf-8')\n"
            "_path.write_text(_content, encoding='utf-8')\n"
            "print(f'Wrote {_relative.as_posix()} ({len(_content)} characters).')"
        )
        return session.execute_result(code)

    if name == "read_workspace_file":
        path = _workspace_path(arguments.get("path"))
        start = _positive_integer(arguments.get("start_line"), default=1, field_name="start_line")
        end = _positive_integer(
            arguments.get("end_line"),
            default=start + _MAX_READ_LINES - 1,
            field_name="end_line",
        )
        if end < start:
            raise ToolInputError("read_workspace_file end_line must be greater than or equal to start_line.")
        if end - start + 1 > _MAX_READ_LINES:
            raise ToolInputError(f"read_workspace_file may return at most {_MAX_READ_LINES} lines.")
        code = (
            "from pathlib import Path\n"
            f"_path = Path({str(path)!r})\n"
            "if not _path.is_file():\n"
            "    raise FileNotFoundError(_path)\n"
            "_lines = _path.read_text(encoding='utf-8').splitlines()\n"
            f"_start, _end = {start}, {end}\n"
            "for _number, _line in enumerate(_lines[_start - 1:_end], start=_start):\n"
            "    print(f'{_number:04d}: {_line}')"
        )
        return session.execute_result(code)

    if name == "list_workspace_files":
        code = (
            "from pathlib import Path\n"
            "_root = Path('scratch')\n"
            "_root.mkdir(parents=True, exist_ok=True)\n"
            "_files = sorted(path for path in _root.rglob('*') if path.is_file())\n"
            "for _path in _files[:200]:\n"
            "    print(f'{_path.relative_to(_root).as_posix()}\\t{_path.stat().st_size} bytes')\n"
            "if len(_files) > 200:\n"
            "    print(f'... {len(_files) - 200} additional files')\n"
            "if not _files:\n"
            "    print('(scratch workspace is empty)')"
        )
        return session.execute_result(code)

    raise ToolInputError(f"Unknown workspace tool: {name}")


def read_workspace_submission_text(
    *,
    path: object,
    session: CodeExecutionSession,
) -> str:
    """Read one bounded regular UTF-8 submission file from scratch/."""
    workspace_path = _workspace_path(path)
    code = (
        "import base64\n"
        "from pathlib import Path\n"
        f"_path = Path({str(workspace_path)!r})\n"
        "if _path.is_symlink() or not _path.is_file():\n"
        "    raise ValueError('Submission path must identify a regular non-symlink file.')\n"
        f"if _path.stat().st_size > {_MAX_SUBMISSION_FILE_BYTES}:\n"
        f"    raise ValueError('Submission file exceeds {_MAX_SUBMISSION_FILE_BYTES} bytes.')\n"
        "print(base64.b64encode(_path.read_bytes()).decode('ascii'))"
    )
    result = session.execute_result(code)
    if result.status != "success":
        raise ToolInputError(f"Unable to read workspace submission file: {result.output or result.status}")
    try:
        raw = base64.b64decode(result.output.strip(), validate=True)
        text = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ToolInputError("Workspace submission file must contain valid UTF-8 text.") from exc
    if len(raw) > _MAX_SUBMISSION_FILE_BYTES:
        raise ToolInputError(f"Workspace submission file exceeds {_MAX_SUBMISSION_FILE_BYTES} bytes.")
    return text


def _workspace_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError("Workspace path must be a non-empty string.")
    if len(value) > _MAX_PATH_CHARS:
        raise ToolInputError(f"Workspace path exceeds {_MAX_PATH_CHARS} characters.")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise ToolInputError("Workspace path must use printable POSIX path characters.")
    relative = PurePosixPath(value)
    if relative.parts and relative.parts[0] == "scratch":
        corrected = PurePosixPath(*relative.parts[1:]).as_posix()
        raise ToolInputError(
            f"Workspace paths are already relative to scratch/; use {corrected!r} instead of {relative.as_posix()!r}."
        )
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative == PurePosixPath(".")
    ):
        raise ToolInputError("Workspace path must be relative and remain under scratch/.")
    return PurePosixPath("scratch") / relative


def _positive_integer(value: object, *, default: int, field_name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ToolInputError(f"{field_name} must be a positive integer.")
    return value


__all__ = ["WORKSPACE_TOOLS", "handle_workspace_tool", "read_workspace_submission_text"]
