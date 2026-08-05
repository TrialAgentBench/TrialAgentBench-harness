"""Render a per-program conversation log as human-readable Markdown.

The raw ``conversation.json`` is a faithful but verbose record of the
chat-completions message list. ``transcript.md`` reformats it for skimming:

* tool calls are rendered as fenced code blocks with truncated arguments
* tool replies are summarised — full text is kept for short replies,
  long replies are quoted in collapsible details
* the system prompt is included once at the top
* errors and decisions stand out

This convenience view does not infer phase or evidence use from prose.
Runner-native ``events.jsonl`` is the source of phase and step identity for
analysis.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_MAX_INLINE_OUTPUT = 600
_MAX_INLINE_TOOL_ARGS = 800


def _truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... (truncated, {len(s) - limit} more chars)"


def _format_tool_args(raw: str) -> str:
    if not raw:
        return "(no arguments)"
    try:
        parsed = json.loads(raw)
        return json.dumps(parsed, indent=2, sort_keys=True)
    except (json.JSONDecodeError, ValueError, TypeError):
        return raw


def _render_assistant(message: dict[str, Any]) -> list[str]:
    out: list[str] = []
    content = message.get("content")
    if content:
        out.append("**Agent:**")
        out.append("")
        out.append(_truncate(str(content), 4000))
        out.append("")
    for tc in message.get("tool_calls", []) or []:
        fn = tc.get("function", {}) or {}
        name = fn.get("name", "?")
        args = _format_tool_args(fn.get("arguments") or "")
        if name in {"execute_code", "inspect_parquet"}:
            args = _truncate(args, _MAX_INLINE_TOOL_ARGS)
        out.append(f"**Tool call:** `{name}`")
        out.append("")
        out.append("```json")
        out.append(args)
        out.append("```")
        out.append("")
    return out


def _render_tool_reply(message: dict[str, Any]) -> list[str]:
    raw = str(message.get("content") or "")
    out = ["**Tool reply:**"]
    if len(raw) <= _MAX_INLINE_OUTPUT:
        out.append("")
        out.append("```")
        out.append(raw or "(empty)")
        out.append("```")
        out.append("")
    else:
        out.append("")
        out.append(f"<details><summary>tool reply ({len(raw)} chars — click to expand)</summary>")
        out.append("")
        out.append("```")
        out.append(_truncate(raw, 6000))
        out.append("```")
        out.append("")
        out.append("</details>")
        out.append("")
    return out


def _render_user(message: dict[str, Any]) -> list[str]:
    content = str(message.get("content") or "")
    return [
        "**Harness → agent:**",
        "",
        "```",
        _truncate(content, 4000),
        "```",
        "",
    ]


def render_transcript_md(messages: Iterable[dict[str, Any]]) -> str:
    """Render a conversation list as a human-readable markdown transcript."""
    messages = list(messages)
    lines: list[str] = ["# Program transcript", ""]

    # System prompt
    if messages and messages[0].get("role") == "system":
        lines.append("<details><summary>system prompt</summary>")
        lines.append("")
        lines.append("```")
        lines.append(_truncate(str(messages[0].get("content") or ""), 8000))
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    for msg in messages[1:]:
        role = msg.get("role")
        if role == "user":
            lines.extend(_render_user(msg))
        elif role == "assistant":
            lines.extend(_render_assistant(msg))
        elif role == "tool":
            lines.extend(_render_tool_reply(msg))
        else:
            lines.append(f"_{role}: {_truncate(str(msg.get('content') or ''), 500)}_")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_transcript_md(messages: Iterable[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_transcript_md(messages), encoding="utf-8")


def render_transcript_from_file(conversation_path: Path) -> str:
    messages = json.loads(Path(conversation_path).read_text(encoding="utf-8"))
    return render_transcript_md(messages)


__all__ = [
    "render_transcript_md",
    "write_transcript_md",
    "render_transcript_from_file",
]
