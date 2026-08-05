"""Typed boundary for executing model-generated analysis code."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class CodeExecutionLimitsV1(BaseModel):
    """Resource limits applied to one isolated Python session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)
    memory_mb: int = Field(default=4096, ge=256, le=65536)
    cpu_count: float = Field(default=2.0, gt=0.0, le=64.0)
    process_limit: int = Field(default=128, ge=16, le=4096)
    output_bytes: int = Field(default=65536, ge=1024, le=16 * 1024 * 1024)
    workspace_mb: int = Field(default=2048, ge=128, le=32768)
    evidence_mb: int = Field(default=8192, ge=1, le=131072)
    evidence_file_count: int = Field(default=10000, ge=1, le=1_000_000)


class CodeExecutionResultV1(BaseModel):
    """Result of one code block executed inside an isolated session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["success", "execution_error", "timeout", "session_terminated"]
    output: str
    output_truncated: bool = False
    elapsed_seconds: float = Field(ge=0.0)


class CodeExecutionSession(Protocol):
    """Stateful isolated code-execution session used by benchmark agents."""

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        """Execute one code block while preserving session state."""

    def execute(self, code: str) -> str:
        """Execute one code block and return bounded agent-facing text."""

    def snapshot_scratch(self) -> Path:
        """Persist the current isolated scratch tree for checkpoint custody."""

    def close(self) -> None:
        """Terminate the session and all descendant processes."""


__all__ = ["CodeExecutionLimitsV1", "CodeExecutionResultV1", "CodeExecutionSession"]
