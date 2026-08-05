"""Release execution budgets shared by the public harness and release metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_harness.ports.code_execution import CodeExecutionLimitsV1

TRIALEVAL_DEFAULT_WORKERS = 1
TRIALEVAL_DIAGNOSTIC_PROOF_DEFAULT_WORKERS = 8
TRIALEVAL_DIAGNOSTIC_WORKER_INVALID_INPUT_EXIT_CODE = 2
TRIALDEV_DEFAULT_WORKERS = 4


class ExecutionBudgetProfileV1(BaseModel):
    """One immutable suite-level execution budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.execution_budget_profile/v1"] = "trialagentbench.execution_budget_profile/v1"
    profile_id: str = Field(min_length=1)
    suite_id: Literal["trialeval", "trialdev"]
    turn_scope: Literal["item", "submission_step"]
    maximum_turns: int = Field(ge=1)
    maximum_completion_tokens_per_turn: int = Field(ge=1)
    provider_request_timeout_seconds: float = Field(gt=0.0, le=900.0)
    wall_time_limit_seconds: int = Field(ge=1)
    maximum_context_characters: int = Field(ge=1)
    code_execution: CodeExecutionLimitsV1
    network_access: Literal[False] = False


TRIALEVAL_RELEASE_BUDGET_V1 = ExecutionBudgetProfileV1(
    profile_id="trialeval_release_default_v1",
    suite_id="trialeval",
    turn_scope="item",
    maximum_turns=90,
    maximum_completion_tokens_per_turn=4096,
    provider_request_timeout_seconds=300.0,
    wall_time_limit_seconds=3600,
    maximum_context_characters=120_000,
    code_execution=CodeExecutionLimitsV1(),
)

TRIALDEV_RELEASE_BUDGET_V1 = ExecutionBudgetProfileV1(
    profile_id="trialdev_release_default_v1",
    suite_id="trialdev",
    turn_scope="submission_step",
    maximum_turns=45,
    maximum_completion_tokens_per_turn=4096,
    provider_request_timeout_seconds=300.0,
    wall_time_limit_seconds=1800,
    maximum_context_characters=120_000,
    code_execution=CodeExecutionLimitsV1(),
)


def release_execution_budget_profiles_v1() -> tuple[ExecutionBudgetProfileV1, ...]:
    """Return the complete immutable public release budget registry."""

    return (TRIALEVAL_RELEASE_BUDGET_V1, TRIALDEV_RELEASE_BUDGET_V1)


__all__ = [
    "ExecutionBudgetProfileV1",
    "TRIALDEV_DEFAULT_WORKERS",
    "TRIALEVAL_DIAGNOSTIC_PROOF_DEFAULT_WORKERS",
    "TRIALEVAL_DIAGNOSTIC_WORKER_INVALID_INPUT_EXIT_CODE",
    "TRIALEVAL_RELEASE_BUDGET_V1",
    "TRIALEVAL_DEFAULT_WORKERS",
    "TRIALDEV_RELEASE_BUDGET_V1",
    "release_execution_budget_profiles_v1",
]
