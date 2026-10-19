"""Numerical recovery uses consistent units and penalizes missed reference values."""

import pytest

from trialagentbench_verifier.numerical_scoring import interval_score, numerical_loss


def test_recovery_is_invariant_to_units_and_direction():
    original = numerical_loss(estimate=1.2, lower=0.5, upper=1.5, target=1, scale=2, confidence_level=0.95)
    assert numerical_loss(
        estimate=120, lower=50, upper=150, target=100, scale=200, confidence_level=0.95
    ) == pytest.approx(original)
    assert numerical_loss(
        estimate=-1.2, lower=-1.5, upper=-0.5, target=-1, scale=2, confidence_level=0.95
    ) == pytest.approx(original)


def test_interval_noncoverage_and_point_error_both_increase_loss():
    assert interval_score(0, 2, 1, 0.95) == 2
    assert interval_score(2, 3, 1, 0.95) == pytest.approx(41)
    assert numerical_loss(estimate=1, lower=0, upper=2, target=1, scale=1, confidence_level=0.95) == 1
    assert numerical_loss(estimate=2, lower=2, upper=3, target=1, scale=1, confidence_level=0.95) == pytest.approx(21)
    # The estimate is not required to lie inside the reported interval.
    assert numerical_loss(estimate=1, lower=2, upper=3, target=1, scale=1, confidence_level=0.95) == pytest.approx(
        20.5
    )


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_normalization_is_an_evaluator_error(scale):
    with pytest.raises(ValueError, match="scale"):
        numerical_loss(estimate=0, lower=-1, upper=1, target=0, scale=scale, confidence_level=0.95)


@pytest.mark.parametrize(
    "estimate,lower,upper,target,scale",
    [
        (1e308, 0, 1e308, 0, 1e308),
        (1e308, -1e308, 1e308, 0, 0.01),
        (-1e308, -1e308, 0, 1e308, 1e308),
        (1e200, -1, 1, 0, 1),
        (1e308, -1e308, 1e308, 0, 5e-324),
    ],
)
def test_extreme_finite_answers_match_decimal_reference(
    estimate: float, lower: float, upper: float, target: float, scale: float
) -> None:
    import math
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 80
        e, lo, hi, t, s, c = map(Decimal.from_float, (estimate, lower, upper, target, scale, 0.95))
        width = hi - lo
        penalty = 2 * max(lo - t, t - hi, Decimal(0)) / (1 - c)
        expected = float((abs(e - t) + width + penalty) / (2 * s))
    actual = numerical_loss(
        estimate=estimate, lower=lower, upper=upper, target=target, scale=scale, confidence_level=0.95
    )
    if math.isfinite(expected):
        assert actual == pytest.approx(expected, rel=1e-12, abs=5e-324)
    else:
        assert actual is None


def test_mean_loss_preserves_interval_incentive_and_explicit_missingness():
    """Aggregation must not reproduce the bounded-score undercoverage incentive."""
    from math import log

    from trialagentbench_verifier.authored_results import TrialDevNumericalRecovery, mean_numerical_loss

    def expected_loss(width):
        # Exact integration for T ~ Uniform[-1, 1], estimate=0, scale=1.
        return 0.25 + width + 10 * (1 - width) ** 2

    def expected_old_credit(width):
        return 2 * log((2 + 3 * width) / (2 + 2 * width)) + 2 / 41 * log((43 - 38 * width) / (2 + 3 * width))

    assert expected_old_credit(0.8837) > expected_old_credit(0.95)
    assert expected_loss(0.95) < expected_loss(0.8837)
    assert min(range(10001), key=lambda i: expected_loss(i / 10000)) == 9500
    scores = tuple(
        TrialDevNumericalRecovery(
            target_id=str(index),
            status="reported",
            reference_value=0,
            normalization_scale=1,
            normalized_absolute_error=point,
            normalized_interval_loss=interval,
        )
        for index, (point, interval) in enumerate(((0, 0.2), (2, 8)))
    )
    assert mean_numerical_loss(scores) == pytest.approx(2.55)
    assert 1 / (1 + mean_numerical_loss(scores)) != pytest.approx(sum(1 / (1 + score.loss) for score in scores) / 2)
    missing = scores[1].model_copy(
        update={
            "status": "missing",
            "normalized_absolute_error": None,
            "normalized_interval_loss": None,
        }
    )
    assert mean_numerical_loss((scores[0], missing)) is None
    assert mean_numerical_loss(()) is None
