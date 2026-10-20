"""Price a TrialEval answer by the development decision it would drive.

The analysis an agent reports, its estimate and 95% interval, feeds one published downstream decision: stop,
or run a confirmatory trial of a chosen size. The decision layer takes the action with the largest expected
value under the disclosed effect prior and the trial's declared economics, treating the report as the trial's
likelihood. Because the action is optimal given the report, no distortion of the report (bias, a narrower or
wider interval, or prior shrinkage applied by the agent) gains in expectation: the layer already applies the
prior, so an agent reports what its trial's data say. Textbook sizing (power at the observed estimate) would
not keep this property, so the confirmatory size is always the expected-value-optimal one.

An answer's consequence is its regret against the credited reference analyses of the same data: the expected
value, under a credited analysis' posterior, of the action that analysis drives minus that of the action the
answer drives, taking the closest credited analysis (grading plurality: any credited analysis costs nothing).
This posterior-expected regret is the expectation of the regret at the true effect given the data; it is never
negative and does not depend on the drawn effect, so it ranks answers with far less noise. The regret at the
task's true effect is reported alongside. A missing answer drives the action the prior alone drives. Values are
in millions of US dollars; each grade also reports both confirmatory trials in participants and in power at the
true effect.

The economics are declared per analysed trial from published sources (construction ``trialeval.economics``):
a confirmatory success is worth ``value_of_success_musd``; a confirmatory trial costs ``fixed_trial_cost_musd``
plus ``replicate_cost_musd`` for each multiple of the analysed trial's participants; success is a one-sided 2.5%
test. A confirmatory trial is sized as a multiple of the analysed trial, so its standard error is the target's
normaliser over the square root of that multiple.
"""

from __future__ import annotations

import math
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import norm

_Array = npt.NDArray[np.float64]

COCHRANE_SNR_MIXTURE_WEIGHTS: tuple[float, ...] = (0.32, 0.31, 0.30, 0.07)
"""Disclosed effect-prior mixture weights (van Zwet, Schwab and Senn, Stat Med 2021;40:6107-6117)."""
COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS: tuple[float, ...] = (0.61, 1.42, 2.16, 5.64)
"""Standard deviations of the signal-to-noise ratio in each mixture component."""
CONFIRMATORY_SIZE_MULTIPLES: tuple[float, ...] = tuple(float(value) for value in np.geomspace(0.5, 25.0, 48))
"""Confirmatory trial sizes, as multiples of the analysed trial's participants."""
_SUCCESS_CRITICAL_Z = float(norm.isf(0.025))
_Z95 = float(norm.isf(0.025))


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DecisionEconomicsV1(_Model):
    """Declared economics of one analysed trial's confirmatory decision (US$ millions)."""

    value_of_success_musd: float = Field(gt=0)
    replicate_cost_musd: float = Field(gt=0)
    """Cost of a confirmatory trial the size of the analysed trial, beyond its fixed cost."""
    fixed_trial_cost_musd: float = Field(ge=0)

    def scaled(self, *, value: float = 1.0, cost: float = 1.0) -> DecisionEconomicsV1:
        """Return the economics with the value and the costs multiplied (the disclosed sensitivity band)."""
        return DecisionEconomicsV1(
            value_of_success_musd=self.value_of_success_musd * value,
            replicate_cost_musd=self.replicate_cost_musd * cost,
            fixed_trial_cost_musd=self.fixed_trial_cost_musd * cost,
        )


class DecisionReportV1(_Model):
    """A reported analysis on the comparison scale (log scale for ratios): estimate and 95% interval."""

    estimate: float
    lower: float
    upper: float

    @model_validator(mode="after")
    def ordered_interval(self) -> Self:
        """Reject reversed bounds before interpreting the normal likelihood."""
        if self.lower > self.upper:
            raise ValueError("decision report lower limit exceeds upper limit")
        return self


class DecisionActionV1(_Model):
    """The optimal downstream action: stop (multiple 0) or confirm at a multiple of the analysed trial."""

    confirm: bool
    size_multiple: float = Field(ge=0)
    expected_value_musd: float


class DecisionConsequenceV1(_Model):
    """Consequence of one answer against its closest credited analysis of the same data."""

    agent: DecisionActionV1
    reference: DecisionActionV1
    regret_musd: float = Field(ge=0)
    """Posterior-expected regret under the closest credited analysis (the headline)."""
    realised_regret_musd: float
    """Regret at the task's true effect against the closest credited analysis there."""
    go_decision_differs: bool
    agent_confirmatory_participants: float = Field(ge=0)
    reference_confirmatory_participants: float = Field(ge=0)
    agent_power_at_truth: float | None = None
    reference_power_at_truth: float | None = None
    consequential: bool


def _benefit(value: float, favorable_direction: Literal["higher", "lower"]) -> float:
    return value if favorable_direction == "higher" else -value


def _posterior(estimate: float, standard_error: float, scale: float) -> tuple[_Array, _Array, _Array]:
    """Return the mixture posterior (weights, means, variances) of the benefit given one normal report."""
    weights = np.asarray(COCHRANE_SNR_MIXTURE_WEIGHTS, dtype=float)
    deviations = np.asarray(COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS, dtype=float)
    if math.isinf(standard_error):
        return weights, np.zeros_like(weights), (deviations * scale) ** 2
    if standard_error == 0:
        return weights, np.full_like(weights, estimate), np.zeros_like(weights)
    log_prior_sd = np.log(deviations) + math.log(scale)
    log_se = math.log(standard_error)
    log_total_sd = np.logaddexp(2 * log_prior_sd, 2 * log_se) / 2
    # Subtract the broadest component's quadratic term before exponentiating.
    # Squaring a large finite report, or subtracting two infinite log weights,
    # otherwise crashes grading or turns a valid likelihood into NaN.
    penalty = np.zeros_like(weights)
    if estimate != 0:
        with np.errstate(over="ignore", under="ignore"):
            penalty[:-1] = np.exp(
                2 * math.log(abs(estimate))
                + 2 * math.log(scale)
                + np.log(deviations[-1] ** 2 - deviations[:-1] ** 2)
                - 2 * log_total_sd[:-1]
                - 2 * log_total_sd[-1]
            )
    log_weight = np.log(weights) - log_total_sd - penalty / 2
    posterior_weight = np.exp(log_weight - log_weight.max())
    posterior_weight /= posterior_weight.sum()
    ratio = np.exp(log_prior_sd - log_total_sd)
    posterior_sd = np.exp(log_prior_sd + log_se - log_total_sd)
    return posterior_weight, ratio * estimate * ratio, posterior_sd**2


def _report_posterior(
    report: DecisionReportV1 | None, scale: float, favorable_direction: Literal["higher", "lower"]
) -> tuple[_Array, _Array, _Array]:
    """Return the posterior of the benefit given a report (``None`` uses the prior alone)."""
    if report is None:
        return _posterior(0.0, math.inf, scale)
    standard_error = report.upper / (2 * _Z95) - report.lower / (2 * _Z95)
    return _posterior(_benefit(report.estimate, favorable_direction), standard_error, scale)


def _value_curve(
    weights: _Array,
    means: _Array,
    variances: _Array,
    *,
    scale: float,
    economics: DecisionEconomicsV1,
) -> _Array:
    """Return the expected value of confirming at every declared size multiple."""
    multiples = np.asarray(CONFIRMATORY_SIZE_MULTIPLES, dtype=float)[:, None]
    slope = np.sqrt(multiples) / scale
    with np.errstate(over="ignore"):
        power = norm.cdf((slope * means[None, :] - _SUCCESS_CRITICAL_Z) / np.sqrt(1.0 + slope**2 * variances[None, :]))
    cost = multiples[:, 0] * economics.replicate_cost_musd + economics.fixed_trial_cost_musd
    values: _Array = economics.value_of_success_musd * (power @ weights) - cost
    return values


def optimal_action_v1(
    report: DecisionReportV1 | None,
    *,
    scale: float,
    favorable_direction: Literal["higher", "lower"],
    economics: DecisionEconomicsV1,
) -> DecisionActionV1:
    """Return the expected-value-maximising action given a report (``None`` uses the prior alone).

    Parameters
    ----------
    report
        Reported estimate and 95% interval on the comparison scale.
    scale
        The target's grading normaliser: the comparison's sampling SD at the analysed trial's size.
    favorable_direction
        Direction of benefit on the comparison scale.
    economics
        Declared development economics.

    Returns
    -------
    DecisionActionV1
        Stop, or confirm at the size multiple of largest expected value.
    """
    curve = _value_curve(*_report_posterior(report, scale, favorable_direction), scale=scale, economics=economics)
    best = int(np.argmax(curve))
    if curve[best] <= 0.0:
        return DecisionActionV1(confirm=False, size_multiple=0.0, expected_value_musd=0.0)
    return DecisionActionV1(
        confirm=True, size_multiple=CONFIRMATORY_SIZE_MULTIPLES[best], expected_value_musd=float(curve[best])
    )


def _power(action: DecisionActionV1, true_benefit: float, scale: float) -> float:
    return float(norm.cdf(true_benefit * math.sqrt(action.size_multiple) / scale - _SUCCESS_CRITICAL_Z))


def realised_value_v1(
    action: DecisionActionV1,
    *,
    true_benefit: float,
    scale: float,
    economics: DecisionEconomicsV1,
) -> float:
    """Return an action's expected value (US$ millions) when the true benefit is known."""
    if not action.confirm:
        return 0.0
    cost = action.size_multiple * economics.replicate_cost_musd
    return (
        economics.value_of_success_musd * _power(action, true_benefit, scale) - cost - economics.fixed_trial_cost_musd
    )


def _posterior_value(
    action: DecisionActionV1, posterior: tuple[_Array, _Array, _Array], *, scale: float, economics: DecisionEconomicsV1
) -> float:
    """Return an action's expected value under a posterior of the benefit."""
    if not action.confirm:
        return 0.0
    weights, means, variances = posterior
    slope = math.sqrt(action.size_multiple) / scale
    power = norm.cdf((slope * means - _SUCCESS_CRITICAL_Z) / np.sqrt(1.0 + slope**2 * variances)) @ weights
    cost = action.size_multiple * economics.replicate_cost_musd + economics.fixed_trial_cost_musd
    return float(economics.value_of_success_musd * power - cost)


def decision_consequence_v1(
    answer: DecisionReportV1 | None,
    credited: tuple[DecisionReportV1, ...],
    *,
    truth: float,
    scale: float,
    participants: int,
    favorable_direction: Literal["higher", "lower"],
    economics: DecisionEconomicsV1,
    consequential_regret_musd: float = 5.0,
) -> DecisionConsequenceV1:
    """Return an answer's decision consequence against the closest credited analysis of the same data.

    Parameters
    ----------
    answer
        The agent's report on the comparison scale, or ``None`` when no valid answer was submitted.
    credited
        Every credited reference analysis of the same data, on the comparison scale.
    truth
        The task's true effect on the comparison scale.
    scale, favorable_direction, economics
        As for ``optimal_action_v1``.
    participants
        Participants in the analysed trial, to report confirmatory participants in natural units.
    consequential_regret_musd
        Posterior-expected regret at which a changed stop-or-confirm decision counts as a consequential error.

    Returns
    -------
    DecisionConsequenceV1
        Posterior-expected and realised regret, natural-unit decisions and the consequential-error flag.

    Raises
    ------
    ValueError
        If no credited analysis is supplied.
    """
    if not credited:
        raise ValueError("decision consequence requires at least one credited analysis")
    benefit = _benefit(truth, favorable_direction)
    agent = optimal_action_v1(answer, scale=scale, favorable_direction=favorable_direction, economics=economics)
    agent_value = realised_value_v1(agent, true_benefit=benefit, scale=scale, economics=economics)
    expected, realised = [], []
    for report in credited:
        posterior = _report_posterior(report, scale, favorable_direction)
        action = optimal_action_v1(report, scale=scale, favorable_direction=favorable_direction, economics=economics)
        regret = _posterior_value(action, posterior, scale=scale, economics=economics) - _posterior_value(
            agent, posterior, scale=scale, economics=economics
        )
        expected.append((max(0.0, regret), action))
        value = realised_value_v1(action, true_benefit=benefit, scale=scale, economics=economics)
        realised.append((abs(value - agent_value), value - agent_value))
    regret, reference = min(expected, key=lambda item: item[0])
    differs = all(action.confirm != agent.confirm for _, action in expected)
    return DecisionConsequenceV1(
        agent=agent,
        reference=reference,
        regret_musd=regret,
        realised_regret_musd=min(realised)[1],
        go_decision_differs=differs,
        agent_confirmatory_participants=agent.size_multiple * participants,
        reference_confirmatory_participants=reference.size_multiple * participants,
        agent_power_at_truth=_power(agent, benefit, scale) if agent.confirm else None,
        reference_power_at_truth=_power(reference, benefit, scale) if reference.confirm else None,
        consequential=differs and regret >= consequential_regret_musd,
    )


__all__ = [
    "COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS",
    "COCHRANE_SNR_MIXTURE_WEIGHTS",
    "CONFIRMATORY_SIZE_MULTIPLES",
    "DecisionActionV1",
    "DecisionConsequenceV1",
    "DecisionEconomicsV1",
    "DecisionReportV1",
    "decision_consequence_v1",
    "optimal_action_v1",
    "realised_value_v1",
]
