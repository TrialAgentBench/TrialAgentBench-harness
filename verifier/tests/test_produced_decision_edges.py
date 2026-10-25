"""Decision limits exercised on targets from the produced development release."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from trialagentbench_verifier.campaign import _decision_consequence, trialeval_target_consequence_v1
from trialagentbench_verifier.decision_consequence import DecisionReportV1
from trialagentbench_verifier.trialeval import _PointTarget

_CASES = json.loads((Path(__file__).parent / "fixtures/produced_decision_targets.json").read_text())


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["target"]["binding"]["effect_measure"])
def test_zero_width_has_the_same_decision_as_its_point_mass_limit(case):
    """A finite exact report carries information rather than selecting the prior."""
    target = _PointTarget.model_validate(case["target"])
    direction = 1 if target.binding.favorable_direction == "higher" else -1
    estimate = direction * 10 * target.reference_sampling_standard_deviation
    exact = DecisionReportV1(estimate=estimate, lower=estimate, upper=estimate)
    narrow = DecisionReportV1(
        estimate=estimate,
        lower=math.nextafter(estimate, -math.inf),
        upper=math.nextafter(estimate, math.inf),
    )
    observed = trialeval_target_consequence_v1(target, exact)
    limit = trialeval_target_consequence_v1(target, narrow)
    assert observed is not None and limit is not None
    assert observed.agent.confirm == limit.agent.confirm
    assert observed.agent.size_multiple == limit.agent.size_multiple
    assert observed.regret_musd == pytest.approx(limit.regret_musd, abs=1e-10)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["target"]["binding"]["effect_measure"])
@pytest.mark.parametrize("benefit", [1e200, -1e200, 1e308, -1e308])
def test_extreme_finite_estimate_has_a_finite_decision(case, benefit):
    """A very large finite report must reach its power limit without overflow."""
    target = _PointTarget.model_validate(case["target"])
    direction = 1 if target.binding.favorable_direction == "higher" else -1
    report = DecisionReportV1(estimate=direction * benefit, lower=-1.0, upper=1.0)
    result = trialeval_target_consequence_v1(target, report)
    assert result is not None
    assert result.agent.confirm == (benefit > 0)
    assert result.agent.size_multiple == (0.5 if benefit > 0 else 0.0)
    assert math.isfinite(result.regret_musd)
    assert math.isfinite(result.realised_regret_musd)
    assert result.model_dump_json() == trialeval_target_consequence_v1(target, report).model_dump_json()


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["target"]["binding"]["effect_measure"])
@pytest.mark.parametrize("width", [1e200, 1e308])
def test_extremely_wide_interval_converges_to_the_prior(case, width):
    """An uninformative report has the prior's decision and finite regret."""
    target = _PointTarget.model_validate(case["target"])
    report = DecisionReportV1(estimate=0.0, lower=-width, upper=width)
    observed = trialeval_target_consequence_v1(target, report)
    prior = trialeval_target_consequence_v1(target, None)
    assert observed is not None and prior is not None
    assert observed.agent.confirm == prior.agent.confirm
    assert observed.agent.size_multiple == prior.agent.size_multiple
    assert observed.regret_musd == pytest.approx(prior.regret_musd)


def test_decision_report_rejects_reversed_bounds():
    """A reversed interval is invalid rather than an implicit missing answer."""
    with pytest.raises(ValueError, match="lower.*upper"):
        DecisionReportV1(estimate=0, lower=1, upper=-1)


def test_campaign_does_not_overflow_when_reconstructing_a_finite_report():
    """A very large estimate and finite width need no synthetic infinite bound."""
    target = _PointTarget.model_validate(_CASES[0]["target"])
    row = SimpleNamespace(
        target=target,
        outcome=SimpleNamespace(verification_metrics={"comparison_signed_error": 1.7e308, "interval_width": 1e308}),
    )
    result = _decision_consequence(row)
    assert result is not None
    assert math.isfinite(result.regret_musd)
