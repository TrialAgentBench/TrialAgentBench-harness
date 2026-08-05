import numpy as np
import pytest

from trialagentbench_validation.characterisation.group_sequential_response import (
    simulate_group_sequential_operating_characteristics,
    simulate_group_sequential_response,
)


def test_group_sequential_response_controls_null_and_tracks_signal() -> None:
    result = simulate_group_sequential_response(
        information_fractions=(0.5, 0.75, 1.0),
        efficacy_boundaries=(2.7718076487, 2.2980858347, 2.0425913079),
        signal_to_final_boundary=(0.0, 0.5, 1.0, 1.5),
        world_count=50_000,
        seed=825_451,
    )

    final = result.loc[result["look"].eq(3)].sort_values("signal_to_final_boundary")
    assert final.iloc[0]["cumulative_rejection_probability"] == pytest.approx(
        0.05, abs=0.004
    )
    assert np.all(np.diff(final["cumulative_rejection_probability"]) > 0.0)
    totals = result.groupby("signal_to_final_boundary")["stopping_probability"].sum()
    assert np.allclose(
        totals,
        final.set_index("signal_to_final_boundary")["cumulative_rejection_probability"],
    )


def test_group_sequential_repeated_inference_controls_coverage() -> None:
    result = simulate_group_sequential_operating_characteristics(
        information_fractions=(0.5, 0.75, 1.0),
        efficacy_boundaries=(2.7718076487, 2.2980858347, 2.0425913079),
        signal_to_final_boundary=(0.0, 0.5, 1.0, 1.5),
        world_count=50_000,
        seed=825_451,
    )

    assert result["failure_count"].eq(0).all()
    null = result.iloc[0]
    assert null["repeated_interval_coverage"] == pytest.approx(0.95, abs=0.004)
    assert null["ordinary_interval_coverage"] < null["repeated_interval_coverage"]
    assert result["repeated_interval_coverage"].ge(0.94).all()
    assert null["mean_bias_ci_low"] < 0 < null["mean_bias_ci_high"]
    assert result.iloc[1:]["mean_bias"].gt(0).all()
    assert result["mean_information_fraction"].is_monotonic_decreasing


@pytest.mark.parametrize(
    ("information", "boundaries"),
    [
        ((0.5, 1.0), (2.5,)),
        ((0.75, 0.5), (2.5, 2.0)),
        ((0.5, 0.75), (2.5, 2.0)),
    ],
)
def test_group_sequential_response_rejects_invalid_plans(
    information: tuple[float, ...],
    boundaries: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        simulate_group_sequential_response(
            information_fractions=information,
            efficacy_boundaries=boundaries,
            world_count=10_000,
        )
