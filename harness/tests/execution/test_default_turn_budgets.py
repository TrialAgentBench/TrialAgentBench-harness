"""Tests for suite-owned default execution budgets."""

from trialagentbench_harness.trialdev.agent import DEFAULT_MAX_TURNS_PER_STEP
from trialagentbench_harness.trialeval.agent import DEFAULT_MAX_TURNS


def test_trialeval_default_supports_bounded_long_form_analysis() -> None:
    """The default must not truncate the qualified long-form analysis path."""

    assert DEFAULT_MAX_TURNS == 90


def test_trialdev_default_supports_bounded_multistep_analysis() -> None:
    """Each semantic step receives the suite-owned bounded-run turn ceiling."""

    assert DEFAULT_MAX_TURNS_PER_STEP == 45
