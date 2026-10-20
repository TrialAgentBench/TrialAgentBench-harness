"""Stable classification of failures shared by benchmark runtimes."""

from __future__ import annotations

from trialagentbench.contracts.trace.observable import BenchmarkRuntimeFailureTypeV1
from trialagentbench.ports import ModelResponseNoncompletion, ProviderRequestError, RuntimeDeadlineExceeded
from trialagentbench.ports.tool_input import ToolInputError


def common_runtime_failure_type_v1(
    error: BaseException | None,
) -> BenchmarkRuntimeFailureTypeV1:
    """Classify failures that have the same meaning in every benchmark runtime."""

    if isinstance(error, RuntimeDeadlineExceeded):
        return "timeout"
    if isinstance(error, ProviderRequestError):
        return "provider_error"
    if isinstance(error, (ToolInputError, ModelResponseNoncompletion)):
        return "contract_error"
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "cancelled"
    return "runtime_error"


__all__ = ["common_runtime_failure_type_v1"]
