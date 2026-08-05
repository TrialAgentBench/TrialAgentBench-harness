"""Tests for manifest-driven TrialDev release denominators."""

from __future__ import annotations

import pytest

from trialagentbench_harness.contracts.trialdev.trialdev_run_grid import (
    TrialDevProgramGridEntryV1,
    TrialDevProgramGridV1,
)
from trialagentbench_harness.trialdev.data import require_complete_trialdev_grid

_OBJECTIVES = (
    "benefit_risk",
    "pure_efficacy",
    "cost_effective_best",
    "net_clinical_value_under_budget",
)


def _grid(*, scenarios: tuple[str, ...]) -> TrialDevProgramGridV1:
    return TrialDevProgramGridV1(
        release_id="fixture",
        programs=tuple(
            TrialDevProgramGridEntryV1(
                program_id=f"{scenario}__{objective}",
                scenario_key=scenario,
                scenario_semantic_id=f"mechanism-{scenario}",
                objective_id=objective,
            )
            for scenario in scenarios
            for objective in _OBJECTIVES
        ),
    )


@pytest.mark.parametrize("scenario_count", (1, 5, 8))
def test_release_grid_accepts_any_complete_manifest_declared_scenario_set(
    scenario_count: int,
) -> None:
    require_complete_trialdev_grid(_grid(scenarios=tuple(f"s{index:02d}" for index in range(scenario_count))))


def test_release_grid_rejects_an_incomplete_scenario_objective_product() -> None:
    grid = _grid(scenarios=("s01", "s02"))
    incomplete = grid.model_copy(update={"programs": grid.programs[:-1]})

    with pytest.raises(ValueError, match="complete scenario-by-objective product"):
        require_complete_trialdev_grid(incomplete)
