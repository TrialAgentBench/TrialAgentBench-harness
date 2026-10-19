"""The decision layer is proper: no distortion of a reported analysis gains in expectation."""

from __future__ import annotations

import numpy as np
import pytest

from trialagentbench_verifier.decision_consequence import (
    COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS,
    COCHRANE_SNR_MIXTURE_WEIGHTS,
    DecisionEconomicsV1,
    DecisionReportV1,
    _posterior,
    decision_consequence_v1,
    optimal_action_v1,
)

SCALE, PARTICIPANTS, WORLDS = 0.1, 400, 3_000
ECONOMICS = DecisionEconomicsV1(value_of_success_musd=1_000.0, replicate_cost_musd=30.0, fixed_trial_cost_musd=6.0)


def _worlds(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    component = rng.choice(len(COCHRANE_SNR_MIXTURE_WEIGHTS), size=WORLDS, p=COCHRANE_SNR_MIXTURE_WEIGHTS)
    truth = rng.normal(0.0, np.asarray(COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS)[component] * SCALE)
    return truth, truth + SCALE * rng.normal(size=WORLDS)


def _mean_regret(
    shift: float, width: float, direction: str = "higher", *, missing: bool = False, realised: bool = False
) -> float:
    truth, estimate = _worlds(7)
    sign = 1.0 if direction == "higher" else -1.0
    regrets = []
    for value, observed in zip(truth * sign, estimate * sign, strict=True):
        valid = DecisionReportV1(estimate=observed, lower=observed - 1.96 * SCALE, upper=observed + 1.96 * SCALE)
        reported = observed + sign * shift * SCALE
        answer = DecisionReportV1(
            estimate=reported, lower=reported - 1.96 * SCALE * width, upper=reported + 1.96 * SCALE * width
        )
        consequence = decision_consequence_v1(
            None if missing else answer,
            (valid,),
            truth=value,
            scale=SCALE,
            participants=PARTICIPANTS,
            favorable_direction=direction,  # type: ignore[arg-type]
            economics=ECONOMICS,
        )
        regrets.append(consequence.realised_regret_musd if realised else consequence.regret_musd)
    return float(np.mean(regrets))


@pytest.mark.parametrize(
    ("shift", "width"),
    ((1.0, 1.0), (-1.0, 1.0), (2.0, 1.0), (0.0, 0.5), (0.0, 0.7), (0.0, 1.5)),
)
@pytest.mark.parametrize("realised", (False, True))
def test_no_distortion_of_a_report_gains_in_expectation(shift: float, width: float, realised: bool) -> None:
    assert _mean_regret(shift, width, realised=realised) > 0.0


def test_pedantic_differences_cost_far_less_than_consequential_ones() -> None:
    assert abs(_mean_regret(0.1, 1.0)) < 0.1 * _mean_regret(1.0, 1.0)


def test_lower_is_better_targets_are_oriented_and_missing_answers_use_the_prior() -> None:
    assert _mean_regret(1.0, 1.0, "lower") > 0.0
    assert _mean_regret(0.0, 1.0, missing=True) > _mean_regret(1.0, 1.0)


def test_posterior_regret_is_the_expected_regret_at_the_truth_given_the_data() -> None:
    """Rao-Blackwellisation: averaging realised regret over the credited posterior recovers the headline."""
    valid = DecisionReportV1(estimate=0.21, lower=0.21 - 1.96 * SCALE, upper=0.21 + 1.96 * SCALE)
    answer = DecisionReportV1(estimate=0.02, lower=0.02 - 1.96 * SCALE, upper=0.02 + 1.96 * SCALE)
    weights, means, variances = _posterior(valid.estimate, SCALE, SCALE)
    rng = np.random.default_rng(3)
    component = rng.choice(len(weights), size=20_000, p=weights)
    truths = rng.normal(means[component], np.sqrt(variances[component]))
    common = {"scale": SCALE, "participants": PARTICIPANTS, "favorable_direction": "higher", "economics": ECONOMICS}
    headline = decision_consequence_v1(answer, (valid,), truth=0.0, **common).regret_musd  # type: ignore[arg-type]
    realised = np.mean(
        [
            decision_consequence_v1(answer, (valid,), truth=float(truth), **common).realised_regret_musd  # type: ignore[arg-type]
            for truth in truths
        ]
    )
    assert headline > 10.0
    assert realised == pytest.approx(headline, rel=0.05)


def test_any_credited_analysis_costs_nothing_and_strong_evidence_confirms() -> None:
    first = DecisionReportV1(estimate=0.30, lower=0.10, upper=0.50)
    second = DecisionReportV1(estimate=0.20, lower=0.00, upper=0.40)
    consequence = decision_consequence_v1(
        second,
        (first, second),
        truth=0.25,
        scale=SCALE,
        participants=PARTICIPANTS,
        favorable_direction="higher",
        economics=ECONOMICS,
    )
    assert consequence.regret_musd == 0.0 and not consequence.consequential
    action = optimal_action_v1(first, scale=SCALE, favorable_direction="higher", economics=ECONOMICS)
    assert action.confirm and action.size_multiple > 0
    harmful = optimal_action_v1(
        DecisionReportV1(estimate=-0.3, lower=-0.5, upper=-0.1),
        scale=SCALE,
        favorable_direction="higher",
        economics=ECONOMICS,
    )
    assert not harmful.confirm
    assert not optimal_action_v1(
        DecisionReportV1(estimate=0.05, lower=-0.15, upper=0.25),
        scale=SCALE,
        favorable_direction="higher",
        economics=ECONOMICS.scaled(value=0.1),
    ).confirm


def test_confirmatory_trials_are_reported_in_participants_and_power() -> None:
    """The decision depends on information and economics; headcount only converts multiples to participants."""
    report = DecisionReportV1(estimate=0.30, lower=0.10, upper=0.50)
    small, large = (
        decision_consequence_v1(
            report,
            (report,),
            truth=0.25,
            scale=SCALE,
            participants=count,
            favorable_direction="higher",
            economics=ECONOMICS,
        )
        for count in (300, 150_000)
    )
    assert small.agent == large.agent and small.agent.confirm
    assert large.agent_confirmatory_participants == small.agent.size_multiple * 150_000
    assert small.reference_confirmatory_participants == small.agent_confirmatory_participants
    assert small.agent_power_at_truth is not None and 0.5 < small.agent_power_at_truth < 1.0


def test_dearer_confirmation_shrinks_or_stops_the_confirmatory_trial() -> None:
    report = DecisionReportV1(estimate=0.15, lower=-0.05, upper=0.35)
    cheap, dear = (
        optimal_action_v1(report, scale=SCALE, favorable_direction="higher", economics=ECONOMICS.scaled(cost=cost))
        for cost in (0.5, 4.0)
    )
    assert cheap.confirm and (not dear.confirm or dear.size_multiple < cheap.size_multiple)
