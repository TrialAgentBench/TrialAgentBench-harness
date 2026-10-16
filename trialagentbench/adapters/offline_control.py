"""Participant-only deterministic installation control, with no model or network.

Zero estimates deliberately have no scientific interpretation. This adapter
exercises the same tools, persistent executors, result capture and study service
as a model while making no claim about model performance.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from trialagentbench.ports.llm_provider import JsonObject, LLMResponse, LLMResponseMetadata, ToolCall


class _Function(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str


class _Tool(BaseModel):
    model_config = ConfigDict(extra="ignore")
    function: _Function


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | None = None
    tool_calls: tuple[_Tool, ...] = ()


class OfflineControlProvider:
    """Save a zero-result script and, for programmes, acquire one public study."""

    def __init__(self, model: str) -> None:
        if model not in {"smoke-zero-python", "smoke-zero-r"}:
            raise ValueError("Unknown offline installation control")
        self._model = model

    @property
    def model(self) -> str:
        """Identify the control explicitly in every retained run."""
        return self._model

    @property
    def telemetry_route(self) -> str:
        """Distinguish local control steps from provider requests."""
        return "offline_control"

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> LLMResponse:
        """Choose solely from participant-visible history and offered tools."""
        offered = {_Tool.model_validate(tool).function.name for tool in tools or ()}
        history = [_Message.model_validate(message) for message in messages]
        calls = [call for message in history for call in message.tool_calls]
        outputs = [message.content for message in history if message.role == "tool" and message.content]
        programme = "start_study" in offered
        arguments: dict[str, JsonValue]
        if not calls:
            language = "r" if self.model.endswith("-r") else "python"
            script = (
                "source('interface/reporting.R')\n"
                "report(data.frame(target_id=targets(),estimate=0,lower=-1,upper=1))\n"
                if language == "r"
                else "from interface.reporting import targets, report\n"
                "report([dict(target_id=t,estimate=0,lower=-1,upper=1) for t in targets()])\n"
            )
            filename = "analysis.R" if language == "r" else "analysis.py"
            code = (
                "from pathlib import Path\nimport csv, json\n"
                "assert not Path('evaluation').exists()\n"
                f"Path('scratch/{filename}').write_text({script!r})\n"
            )
            if programme:
                code += (
                    "c=next(csv.DictReader(open('interface/targets.csv')))['target_id'].split(':')[0]\n"
                    "print(json.dumps(dict(phase='early_safety',sites=2,control_participants=30,"
                    "arms=[dict(compound=c,dose='standard',participants=30)],"
                    "follow_up_days=56,looks=[dict(deadline_day=365,completed_participants=60)])))\n"
                )
            else:
                code += "print('Saved installation-control analysis for native result capture.')\n"
            name = "execute_code"
            arguments = {"code": code, "language": "python", "purpose": "Save an offline zero-result control."}
        elif not programme:
            name = "submit_response" if "submit_response" in offered else "submit_analysis_report"
            arguments = {
                "content": "Offline zero-result installation control; these estimates are not clinical or model-performance evidence."
            }
        else:
            prior = [call.function.name for call in calls[1:]]
            if not outputs:
                raise ValueError("Offline study control received no tool output")
            if prior.count("start_study") < 2:
                design: dict[str, JsonValue] = json.loads(outputs[0].splitlines()[0])
                name = "start_study"
                arguments = design | {"preview": not prior}
            elif "request_readout" not in prior:
                name, arguments = "request_readout", {"study_id": "study-101"}
            else:
                if "evidence/study-101-look-1/" not in outputs[-1] or "| `" not in outputs[-1]:
                    raise ValueError("Offline study control did not acquire reportable records")
                name, arguments = "stop_development", {}
        if name not in offered:
            raise ValueError(f"Offline control tool is unavailable: {name}")
        return LLMResponse(
            tool_calls=[ToolCall(id=f"control-{len(calls)}", name=name, arguments=json.dumps(arguments))],
            metadata=LLMResponseMetadata(
                response_id=f"offline-{len(calls)}",
                returned_model=self.model,
                finish_reason="tool_calls",
                reported_cost_usd=0.0,
                request_attempts=1,
            ),
        )
