"""Tests for the released TrialAgentBench semantic charter."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from trialagentbench_test_helpers import minimal_benchmark_charter_payload

from trialagentbench_harness.contracts.release.benchmark_charter import (
    TrialAgentBenchCharterV1,
)
from trialagentbench_harness.io.checksums import canonical_payload_sha256


def _charter() -> TrialAgentBenchCharterV1:
    return TrialAgentBenchCharterV1.model_validate(minimal_benchmark_charter_payload())


def test_released_charter_requires_complete_grading_policy() -> None:
    payload = minimal_benchmark_charter_payload()
    policy = payload["trialeval_grading_policy"]
    assert isinstance(policy, dict)
    policy.pop("credit_eligible_set_closure")
    payload.pop("checksum")
    payload["checksum"] = canonical_payload_sha256(payload)

    with pytest.raises(ValidationError, match="credit_eligible_set_closure"):
        TrialAgentBenchCharterV1.model_validate(payload)
