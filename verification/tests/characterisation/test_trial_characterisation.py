"""Tests for reusable clinical-trial characterisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trialagentbench_validation.characterisation import (
    BinaryOutcomeSpec,
    CategoricalVariableSpec,
    CompetingRiskOutcomeSpec,
    ContinuousOutcomeSpec,
    ContinuousVariableSpec,
    DependenceSpec,
    LongitudinalOutcomeSpec,
    OrdinalOutcomeSpec,
    RecurrentEventOutcomeSpec,
    SurvivalOutcomeSpec,
    TrialCharacterisationSpec,
    TrialData,
    characterise_trial,
    summarise_characterisations,
    write_characterisation_csv,
)


def _data() -> TrialData:
    participants = pd.DataFrame(
        {
            "participant_id": [f"P{index:03d}" for index in range(40)],
            "arm": ["Control"] * 20 + ["Treatment"] * 20,
            "cluster": [f"S{index // 4:02d}" for index in range(40)],
            "age": np.linspace(40.0, 78.0, 40),
            "bmi": np.linspace(21.0, 34.0, 40) + np.tile([0.0, 0.4], 20),
            "sex": np.tile(["Female", "Male"], 20),
            "response": np.asarray([index % 3 == 0 for index in range(40)], dtype=int),
            "score": np.linspace(10.0, 20.0, 40) + np.repeat([0.0, 1.5], 20),
            "ordinal": np.tile(["low", "middle", "high", "middle"], 10),
            "time": 4.0 + np.mod(np.arange(40), 12),
            "event": np.asarray([index % 4 != 0 for index in range(40)], dtype=int),
            "first_event_time": 2.0 + np.mod(np.arange(40), 9),
            "event_type": np.tile([0, 1, 2, 1], 10),
        }
    )
    longitudinal = pd.DataFrame(
        [
            {
                "participant_id": f"P{participant:03d}",
                "visit": float(visit),
                "value": 20.0 - 0.5 * visit + 0.1 * participant,
            }
            for participant in range(40)
            for visit in (0, 1, 2)
            if not (visit == 2 and participant % 7 == 0)
        ]
    )
    recurrent = pd.DataFrame(
        [
            {
                "participant_id": f"P{participant:03d}",
                "event_time": float(event_number + 1),
            }
            for participant in range(40)
            for event_number in range(participant % 3)
        ]
    )
    return TrialData(
        participants=participants,
        observation_tables={
            "longitudinal": longitudinal,
            "recurrent": recurrent,
        },
    )


def _spec(
    *,
    trial_id: str = "trial-01",
    programme_id: str = "programme-01",
    design_profile_id: str = "TE-DP05",
    design_family: str = "cluster_parallel",
) -> TrialCharacterisationSpec:
    return TrialCharacterisationSpec(
        trial_id=trial_id,
        programme_id=programme_id,
        design_profile_id=design_profile_id,
        design_family=design_family,
        participant_id_column="participant_id",
        arm_column="arm",
        cluster_id_column=(
            "cluster"
            if design_family in {"cluster_parallel", "stepped_wedge"}
            else None
        ),
        continuous_variables=(
            ContinuousVariableSpec(variable_id="age", column="age", unit="years"),
            ContinuousVariableSpec(variable_id="bmi", column="bmi", unit="kg/m2"),
        ),
        categorical_variables=(
            CategoricalVariableSpec(
                variable_id="sex",
                column="sex",
                unit="proportion",
                categories=("Female", "Male"),
            ),
        ),
        dependence=(
            DependenceSpec(
                dependence_id="age_bmi",
                left_column="age",
                right_column="bmi",
            ),
        ),
        outcomes=(
            BinaryOutcomeSpec(
                outcome_id="response",
                table="participants",
                participant_id_column="participant_id",
                value_column="response",
                event_value=1,
            ),
            ContinuousOutcomeSpec(
                outcome_id="score",
                table="participants",
                participant_id_column="participant_id",
                value_column="score",
                unit="points",
            ),
            OrdinalOutcomeSpec(
                outcome_id="severity",
                table="participants",
                participant_id_column="participant_id",
                value_column="ordinal",
                categories=("low", "middle", "high"),
            ),
            SurvivalOutcomeSpec(
                outcome_id="survival",
                table="participants",
                participant_id_column="participant_id",
                duration_column="time",
                event_column="event",
                horizons=(5.0, 10.0),
                unit="days",
            ),
            LongitudinalOutcomeSpec(
                outcome_id="trajectory",
                table="longitudinal",
                participant_id_column="participant_id",
                time_column="visit",
                value_column="value",
                scheduled_times=(0.0, 1.0, 2.0),
                time_unit="visits",
                value_unit="points",
            ),
            RecurrentEventOutcomeSpec(
                outcome_id="episodes",
                table="recurrent",
                participant_id_column="participant_id",
                event_time_column="event_time",
                horizons=(1.0, 2.0),
                unit="days",
            ),
            CompetingRiskOutcomeSpec(
                outcome_id="first_event",
                table="participants",
                participant_id_column="participant_id",
                duration_column="first_event_time",
                event_type_column="event_type",
                primary_event_code=1,
                competing_event_codes=(2,),
                horizons=(3.0, 8.0),
                unit="days",
            ),
        ),
        bootstrap_replicates=200,
        seed=41,
    )


def test_complete_characterisation_is_deterministic_and_described() -> None:
    first = characterise_trial(_spec(), _data())
    second = characterise_trial(_spec(), _data())
    assert first == second
    property_ids = {row.property_id for row in first.estimates}
    assert {
        "baseline.age.mean",
        "baseline.sex.proportion.female",
        "dependence.age_bmi.spearman",
        "outcome.response.event_probability",
        "outcome.score.mean",
        "outcome.severity.cumulative_probability.high",
        "outcome.survival.survival_probability",
        "outcome.trajectory.mean",
        "observation.trajectory.attendance_probability",
        "outcome.episodes.mean_cumulative_count",
        "outcome.first_event.cumulative_incidence",
        "trial.cluster_count",
    } <= property_ids
    assert all(row.unit for row in first.estimates)
    assert all(row.independent_unit for row in first.estimates)
    assert all(row.estimator for row in first.estimates)
    assert all(row.uncertainty_method for row in first.estimates)
    assert all(row.observed + row.missing == row.denominator for row in first.estimates)


@pytest.mark.parametrize(
    ("design_profile_id", "design_family"),
    [
        ("TE-DP01", "individual_randomized"),
        ("TE-DP02", "pragmatic_randomized"),
        ("TE-DP03", "covariate_subdesign"),
        ("TE-DP04", "ascertainment_subdesign"),
        ("TE-DP05", "cluster_parallel"),
        ("TE-DP06", "stepped_wedge"),
        ("TE-DP07", "group_sequential"),
    ],
)
def test_every_released_design_family_is_explicitly_supported(
    design_profile_id: str,
    design_family: str,
) -> None:
    result = characterise_trial(
        _spec(
            design_profile_id=design_profile_id,
            design_family=design_family,
        ),
        _data(),
    )
    assert result.design_profile_id == design_profile_id
    assert result.design_family == design_family


def test_collection_and_csv_cover_trial_programme_and_portfolio(tmp_path: Path) -> None:
    trial_1 = characterise_trial(
        _spec(trial_id="trial-01", programme_id="programme-a"), _data()
    )
    trial_2 = characterise_trial(
        _spec(trial_id="trial-02", programme_id="programme-a"), _data()
    )
    trial_3 = characterise_trial(
        _spec(trial_id="trial-03", programme_id="programme-b"), _data()
    )
    collection = summarise_characterisations((trial_1, trial_2, trial_3))
    assert {row.programme_id for row in collection.programme_estimates} == {
        "programme-a",
        "programme-b",
    }
    assert {row.evidence_level for row in collection.portfolio_estimates} == {
        "portfolio"
    }
    output = tmp_path / "characterisation.csv"
    write_characterisation_csv(output, collection)
    exported = pd.read_csv(output)
    assert set(exported["evidence_level"]) == {
        "participant_distribution",
        "trial",
        "programme",
        "portfolio",
    }
    with pytest.raises(FileExistsError):
        write_characterisation_csv(output, collection)


def test_ambiguous_or_unsupported_inputs_fail_loudly() -> None:
    data = _data()
    duplicated = data.participants.copy()
    duplicated.loc[1, "participant_id"] = duplicated.loc[0, "participant_id"]
    with pytest.raises(ValueError, match="one row per participant"):
        characterise_trial(
            _spec(),
            TrialData(
                participants=duplicated, observation_tables=data.observation_tables
            ),
        )

    constant = data.participants.copy()
    constant["bmi"] = 25.0
    with pytest.raises(ValueError, match="rank dependence is undefined"):
        characterise_trial(
            _spec(),
            TrialData(
                participants=constant, observation_tables=data.observation_tables
            ),
        )

    invalid_cluster = _spec(design_family="cluster_parallel").model_dump()
    invalid_cluster["cluster_id_column"] = None
    with pytest.raises(ValidationError, match="cluster_id_column"):
        TrialCharacterisationSpec.model_validate(invalid_cluster)

    missing_table = TrialData(participants=data.participants)
    with pytest.raises(ValueError, match="requires observation table"):
        characterise_trial(_spec(), missing_table)
