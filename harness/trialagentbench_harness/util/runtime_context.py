"""Shared deadline and lossless provider-context utilities."""

from __future__ import annotations

import base64
import hashlib
import json
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from trialagentbench_harness.ports.code_execution import CodeExecutionSession
from trialagentbench_harness.ports.llm_provider import JsonObject

R = TypeVar("R")

FINAL_TURN_SUBMISSION_REMINDER = (
    "This is the final available turn. Submit the best complete response supported by your executed work now. "
    "Do not start another analysis step; an unsubmitted response is recorded as missing."
)
SUBMISSION_WINDOW_REMINDER = (
    "The analysis-tool window is now closed. Use the remaining turns to submit the best complete response "
    "supported by your executed work; an unsubmitted response is recorded as missing."
)


def turn_budget_tag(*, turn: int, maximum: int) -> str:
    """Render suite-independent remaining-turn metadata."""

    if maximum < 1:
        raise ValueError("Maximum turns must be positive.")
    if not 1 <= turn <= maximum:
        raise ValueError("Current turn must be within the declared turn budget.")
    return f"\n[turn {turn}/{maximum}, {maximum - turn} left]"


@dataclass(frozen=True)
class RuntimeDeadline:
    """One monotonic wall-clock deadline shared by transport and local tools."""

    monotonic: float | None
    label: str

    @classmethod
    def after(cls, seconds: float | None, *, label: str) -> RuntimeDeadline:
        """Construct a deadline from a positive duration."""

        if seconds is not None and seconds <= 0.0:
            raise ValueError("Deadline duration must be positive when provided.")
        return cls(
            monotonic=time.monotonic() + seconds if seconds is not None else None,
            label=label,
        )

    def remaining(self) -> float | None:
        """Return remaining seconds, failing once no work may start."""

        if self.monotonic is None:
            return None
        remaining = self.monotonic - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f"{self.label} exceeded its declared wall-clock budget.")
        return remaining

    def run_blocking(
        self,
        operation: Callable[[], R],
        *,
        operation_name: str,
        on_timeout: Callable[[], None] | None = None,
    ) -> R:
        """Run a local blocking operation for no longer than the remaining budget."""

        remaining = self.remaining()
        if remaining is None:
            return operation()
        result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def target() -> None:
            try:
                result.put((True, operation()))
            except BaseException as error:
                result.put((False, error))

        thread = threading.Thread(
            target=target,
            name=f"trialagentbench-{operation_name}",
            daemon=True,
        )
        thread.start()
        try:
            succeeded, value = result.get(timeout=remaining)
        except queue.Empty as exc:
            if on_timeout is not None:
                on_timeout()
            raise TimeoutError(f"{self.label} exhausted while blocking on {operation_name}.") from exc
        if succeeded:
            return value  # type: ignore[return-value]
        if not isinstance(value, BaseException):
            raise TypeError("Blocking operation returned an invalid exception payload.")
        raise value


def bounded_provider_context(
    messages: Sequence[dict],
    *,
    session: CodeExecutionSession,
    active_prompt_index: int,
    max_chars: int,
    required_message_indices: Sequence[int] = (),
) -> list[JsonObject]:
    """Build deterministic bounded context without splitting tool-call pairs."""

    if max_chars < 1:
        raise ValueError("max_chars must be at least 1.")
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Provider context must start with one system message.")
    if not 0 <= active_prompt_index < len(messages):
        raise ValueError("active_prompt_index is outside the conversation.")
    if messages[active_prompt_index].get("role") != "user":
        raise ValueError("The active-step prompt must be a user message.")
    if any(not 0 <= index < len(messages) for index in required_message_indices):
        raise ValueError("A required message index is outside the conversation.")

    units = _complete_message_units(messages)
    required_indices = {0, active_prompt_index, *required_message_indices}
    pinned_units = {index for index, unit in enumerate(units) if required_indices.intersection(unit)}
    selected = set(pinned_units)
    used = sum(_message_chars(messages[index]) for unit in pinned_units for index in units[unit])
    for unit_index in range(len(units) - 1, -1, -1):
        if unit_index in selected:
            continue
        unit_size = sum(_message_chars(messages[index]) for index in units[unit_index])
        if used + unit_size <= max_chars:
            selected.add(unit_index)
            used += unit_size

    omitted_indices = [index for unit_index, unit in enumerate(units) if unit_index not in selected for index in unit]
    if omitted_indices:
        _write_session_text(
            session=session,
            relative_path="context_archive.json",
            content=(
                json.dumps(
                    {
                        "schema_id": "trialagentbench.context_archive/v1",
                        "messages": [messages[index] for index in omitted_indices],
                    },
                    indent=2,
                    ensure_ascii=True,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            ),
        )

    context: list[JsonObject] = []
    reference_inserted = False
    for unit_index, unit in enumerate(units):
        if unit_index not in selected:
            continue
        for index in unit:
            context.append(messages[index])
            if index == active_prompt_index and omitted_indices:
                context.append(
                    {
                        "role": "user",
                        "content": (
                            "Earlier complete interaction records are retained at "
                            "context_archive.json. Use read_workspace_file if "
                            "those prior details are needed; no LLM summary was substituted."
                        ),
                    }
                )
                reference_inserted = True
    if omitted_indices and not reference_inserted:
        raise RuntimeError("Context archive reference was not inserted after the active prompt.")
    return context


def persist_bulky_tool_output(
    output: str,
    *,
    session: CodeExecutionSession,
    artifact_id: str,
    inline_chars: int,
) -> str:
    """Persist bulky output losslessly and return a bounded workspace reference."""

    if inline_chars < 1:
        raise ValueError("inline_chars must be at least 1.")
    if len(output) <= inline_chars:
        return output
    digest = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()[:16]
    relative = f"tool_outputs/{digest}.txt"
    _write_session_text(
        session=session,
        relative_path=relative,
        content=output,
    )
    return (
        output[:inline_chars] + f"\n\n[Output truncated in context. Full captured output retained at {relative}; "
        "use read_workspace_file to inspect it.]"
    )


def _write_session_text(
    *,
    session: CodeExecutionSession,
    relative_path: str,
    content: str,
) -> None:
    """Write runtime-owned text through the authoritative isolated workspace."""

    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    script = (
        "import base64\n"
        "from pathlib import Path\n"
        f"_path = Path('scratch') / {relative_path!r}\n"
        "_path.parent.mkdir(parents=True, exist_ok=True)\n"
        f"_path.write_bytes(base64.b64decode({encoded!r}))"
    )
    result = session.execute_result(f"exec({script!r}, {{}})")
    if result.status != "success":
        raise RuntimeError(
            f"Unable to persist runtime workspace artifact {relative_path!r}: {result.output or result.status}"
        )


def _complete_message_units(messages: Sequence[dict]) -> list[tuple[int, ...]]:
    units: list[tuple[int, ...]] = []
    observed_call_ids: set[str] = set()
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        raw_calls = message.get("tool_calls")
        if role == "tool":
            raise ValueError(f"Unmatched tool response at conversation index {index}.")
        if role != "assistant" or not isinstance(raw_calls, list) or not raw_calls:
            units.append((index,))
            index += 1
            continue
        call_ids: list[str] = []
        for call in raw_calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                raise ValueError(f"Invalid assistant tool call at conversation index {index}.")
            call_id = call["id"]
            if call_id in observed_call_ids or call_id in call_ids:
                raise ValueError(f"Duplicate assistant tool_call_id in conversation: {call_id}.")
            call_ids.append(call_id)
        observed_call_ids.update(call_ids)
        end = index + 1
        response_ids: list[str] = []
        while end < len(messages) and messages[end].get("role") == "tool":
            response_id = messages[end].get("tool_call_id")
            if not isinstance(response_id, str):
                raise ValueError(f"Tool response at conversation index {end} has no tool_call_id.")
            response_ids.append(response_id)
            end += 1
        if len(response_ids) != len(set(response_ids)):
            raise ValueError(f"Duplicate tool responses for assistant at conversation index {index}.")
        if set(response_ids) != set(call_ids):
            raise ValueError(
                f"Assistant/tool response IDs do not match at conversation index {index}: "
                f"calls={sorted(call_ids)}, responses={sorted(response_ids)}."
            )
        units.append(tuple(range(index, end)))
        index = end
    return units


def _message_chars(message: dict) -> int:
    return len(json.dumps(message, ensure_ascii=True, sort_keys=True, default=str))


__all__ = [
    "FINAL_TURN_SUBMISSION_REMINDER",
    "SUBMISSION_WINDOW_REMINDER",
    "RuntimeDeadline",
    "bounded_provider_context",
    "persist_bulky_tool_output",
    "turn_budget_tag",
]
