"""Statistical characterisation of clinical-trial participant data."""

from __future__ import annotations

import csv
import hashlib
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, KaplanMeierFitter
from scipy.stats import bootstrap, chi2, spearmanr, t

from trialagentbench_validation.characterisation.contracts import (
    BinaryOutcomeSpec,
    CategoricalVariableSpec,
    CharacterisationCollection,
    CompetingRiskOutcomeSpec,
    ContinuousOutcomeSpec,
    ContinuousVariableSpec,
    DependenceSpec,
    LongitudinalOutcomeSpec,
    OrdinalOutcomeSpec,
    OutcomeSpec,
    RecurrentEventOutcomeSpec,
    SurvivalOutcomeSpec,
    TidyEstimate,
    TrialCharacterisation,
    TrialCharacterisationSpec,
    TrialData,
)
from trialagentbench_validation.statistics import proportion_interval


def characterise_trial(
    spec: TrialCharacterisationSpec,
    data: TrialData,
) -> TrialCharacterisation:
    """Characterise one trial from participant and observation tables.

    Parameters
    ----------
    spec
        Typed definition of the design, variables, outcomes, and uncertainty.
    data
        Participant and optional long-form observation tables.

    Returns
    -------
    TrialCharacterisation
        Deterministically ordered, fully described estimates.

    Raises
    ------
    ValueError
        If a required column, key, value, category, or design property is
        missing, ambiguous, or invalid.
    """

    participants = data.participants.copy()
    _validate_participants(spec, participants)
    estimates: list[TidyEstimate] = []
    estimates.extend(_structure_estimates(spec, participants))
    groups = _groups(participants, spec.arm_column)
    for continuous_variable in spec.continuous_variables:
        estimates.extend(
            _continuous_estimates(spec, participants, groups, continuous_variable)
        )
    for categorical_variable in spec.categorical_variables:
        estimates.extend(
            _categorical_estimates(spec, participants, groups, categorical_variable)
        )
    for dependence in spec.dependence:
        estimates.extend(_dependence_estimates(spec, participants, groups, dependence))
    for outcome in spec.outcomes:
        frame = _outcome_frame(spec, data, outcome)
        estimates.extend(_outcome_estimates(spec, participants, frame, outcome))
    ordered = tuple(
        sorted(
            estimates, key=lambda row: (row.property_id, row.group, row.time or -1.0)
        )
    )
    return TrialCharacterisation(
        trial_id=spec.trial_id,
        programme_id=spec.programme_id,
        design_profile_id=spec.design_profile_id,
        design_family=spec.design_family,
        estimates=ordered,
    )


def summarise_characterisations(
    trials: Sequence[TrialCharacterisation],
) -> CharacterisationCollection:
    """Summarise trial estimates at programme and portfolio levels.

    Parameters
    ----------
    trials
        One or more independently generated trial characterisations.

    Returns
    -------
    CharacterisationCollection
        Trial results plus range-described programme and portfolio summaries.

    Raises
    ------
    ValueError
        If trial identities are duplicated or comparable estimates disagree on
        their units or estimators.
    """

    rows = tuple(trials)
    if not rows:
        raise ValueError("at least one trial characterisation is required")
    trial_ids = tuple(row.trial_id for row in rows)
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("trial characterisation IDs must be unique")
    by_programme: dict[str, list[TrialCharacterisation]] = defaultdict(list)
    for trial in rows:
        by_programme[trial.programme_id].append(trial)
    programme_estimates = tuple(
        estimate
        for programme_id, programme_trials in sorted(by_programme.items())
        for estimate in _aggregate_estimates(
            (estimate for trial in programme_trials for estimate in trial.estimates),
            evidence_level="programme",
            trial_id="__programme__",
            programme_id=programme_id,
            uncertainty_method="range_across_trials",
            independent_unit="trial",
        )
    )
    portfolio_estimates = _aggregate_estimates(
        programme_estimates,
        evidence_level="portfolio",
        trial_id="__portfolio__",
        programme_id="__portfolio__",
        uncertainty_method="range_across_programmes",
        independent_unit="programme",
    )
    return CharacterisationCollection(
        trials=rows,
        programme_estimates=programme_estimates,
        portfolio_estimates=portfolio_estimates,
    )


def write_characterisation_csv(
    path: Path,
    collection: CharacterisationCollection,
) -> None:
    """Write a collection as a deterministic tidy CSV file.

    Parameters
    ----------
    path
        Destination CSV path. Its parent directory must already exist.
    collection
        Validated characterisation collection.

    Raises
    ------
    FileExistsError
        If ``path`` already exists.
    """

    if path.exists():
        raise FileExistsError(path)
    rows = (
        [estimate for trial in collection.trials for estimate in trial.estimates]
        + list(collection.programme_estimates)
        + list(collection.portfolio_estimates)
    )
    fieldnames = tuple(TidyEstimate.model_fields)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _validate_participants(
    spec: TrialCharacterisationSpec, frame: pd.DataFrame
) -> None:
    required = {spec.participant_id_column, spec.arm_column}
    required.update(row.column for row in spec.continuous_variables)
    required.update(row.column for row in spec.categorical_variables)
    required.update(row.left_column for row in spec.dependence)
    required.update(row.right_column for row in spec.dependence)
    if spec.cluster_id_column is not None:
        required.add(spec.cluster_id_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"participant table lacks required columns: {missing!r}")
    ids = frame[spec.participant_id_column]
    if ids.isna().any() or ids.astype("string").str.strip().eq("").any():
        raise ValueError("participant IDs must be non-missing, non-empty values")
    if ids.astype("string").duplicated().any():
        raise ValueError("participant table must contain one row per participant")
    arms = frame[spec.arm_column]
    if arms.isna().any() or arms.astype("string").str.strip().eq("").any():
        raise ValueError("treatment arms must be non-missing, non-empty values")
    if arms.astype("string").nunique() < 2:
        raise ValueError("trial characterisation requires at least two treatment arms")
    if spec.cluster_id_column is not None:
        clusters = frame[spec.cluster_id_column]
        if clusters.isna().any() or clusters.astype("string").str.strip().eq("").any():
            raise ValueError("cluster IDs must be non-missing, non-empty values")


def _structure_estimates(
    spec: TrialCharacterisationSpec,
    participants: pd.DataFrame,
) -> list[TidyEstimate]:
    n = len(participants)
    rows = [
        _estimate(
            spec,
            evidence_level="trial",
            property_id="trial.participant_count",
            group="overall",
            estimate=float(n),
            unit="participants",
            independent_unit="participant",
            estimator="exact_count",
            uncertainty_method="none_descriptive",
            denominator=n,
            observed=n,
            missing=0,
            missingness_disposition="not_applicable",
        ),
        _estimate(
            spec,
            evidence_level="trial",
            property_id="trial.arm_count",
            group="overall",
            estimate=float(participants[spec.arm_column].nunique()),
            unit="arms",
            independent_unit="arm",
            estimator="exact_count",
            uncertainty_method="none_descriptive",
            denominator=n,
            observed=n,
            missing=0,
            missingness_disposition="not_applicable",
        ),
    ]
    if spec.cluster_id_column is not None:
        rows.append(
            _estimate(
                spec,
                evidence_level="trial",
                property_id="trial.cluster_count",
                group="overall",
                estimate=float(participants[spec.cluster_id_column].nunique()),
                unit="clusters",
                independent_unit="cluster",
                estimator="exact_count",
                uncertainty_method="none_descriptive",
                denominator=n,
                observed=n,
                missing=0,
                missingness_disposition="not_applicable",
            )
        )
    return rows


def _groups(
    frame: pd.DataFrame, arm_column: str
) -> tuple[tuple[str, pd.DataFrame], ...]:
    groups: list[tuple[str, pd.DataFrame]] = [("overall", frame)]
    groups.extend(
        (str(arm), arm_frame)
        for arm, arm_frame in frame.groupby(arm_column, observed=True, sort=True)
    )
    return tuple(groups)


def _continuous_estimates(
    spec: TrialCharacterisationSpec,
    participants: pd.DataFrame,
    groups: tuple[tuple[str, pd.DataFrame], ...],
    variable: ContinuousVariableSpec,
) -> list[TidyEstimate]:
    rows: list[TidyEstimate] = []
    for group, frame in groups:
        values, missing = _finite_numeric(frame[variable.column], label=variable.column)
        low, high = _mean_interval(values, spec.confidence_level)
        sd = float(np.std(values, ddof=1))
        sd_low, sd_high = _sd_interval(sd, len(values), spec.confidence_level)
        denominator = len(frame)
        rows.extend(
            (
                _estimate(
                    spec,
                    evidence_level="participant_distribution",
                    property_id=f"{variable.role}.{variable.variable_id}.mean",
                    group=group,
                    estimate=float(np.mean(values)),
                    interval_low=low,
                    interval_high=high,
                    unit=variable.unit,
                    independent_unit="participant",
                    estimator="arithmetic_mean",
                    uncertainty_method=f"student_t_{spec.confidence_level:.3f}",
                    denominator=denominator,
                    observed=len(values),
                    missing=missing,
                    missingness_disposition=variable.missingness,
                ),
                _estimate(
                    spec,
                    evidence_level="participant_distribution",
                    property_id=f"{variable.role}.{variable.variable_id}.sd",
                    group=group,
                    estimate=sd,
                    interval_low=sd_low,
                    interval_high=sd_high,
                    unit=variable.unit,
                    independent_unit="participant",
                    estimator="sample_standard_deviation",
                    uncertainty_method=f"chi_square_{spec.confidence_level:.3f}",
                    denominator=denominator,
                    observed=len(values),
                    missing=missing,
                    missingness_disposition=variable.missingness,
                ),
            )
        )
    return rows


def _categorical_estimates(
    spec: TrialCharacterisationSpec,
    participants: pd.DataFrame,
    groups: tuple[tuple[str, pd.DataFrame], ...],
    variable: CategoricalVariableSpec,
) -> list[TidyEstimate]:
    rows: list[TidyEstimate] = []
    observed_categories = tuple(
        sorted(participants[variable.column].dropna().astype(str).unique())
    )
    categories = variable.categories or observed_categories
    unknown = sorted(set(observed_categories) - set(categories))
    if unknown:
        raise ValueError(
            f"{variable.column} contains undeclared categories: {unknown!r}"
        )
    for group, frame in groups:
        values = frame[variable.column]
        if variable.missingness == "explicit_level":
            strings = values.astype("string").fillna("(missing)").astype(str)
            current_categories = categories + (
                ("(missing)",) if "(missing)" not in categories else ()
            )
            missing = 0
        else:
            strings = values.dropna().astype(str)
            current_categories = categories
            missing = int(values.isna().sum())
        denominator = len(frame)
        observed = denominator - missing
        if observed == 0:
            raise ValueError(
                f"{variable.column} has no observed values in group {group!r}"
            )
        for category in current_categories:
            successes = int((strings == category).sum())
            low, high = proportion_interval(
                successes, observed, alpha=1.0 - spec.confidence_level
            )
            rows.append(
                _estimate(
                    spec,
                    evidence_level="participant_distribution",
                    property_id=f"{variable.role}.{variable.variable_id}.proportion.{_slug(category)}",
                    group=group,
                    estimate=successes / observed,
                    interval_low=low,
                    interval_high=high,
                    unit=variable.unit,
                    independent_unit="participant",
                    estimator="sample_proportion",
                    uncertainty_method=f"wilson_{spec.confidence_level:.3f}",
                    denominator=denominator,
                    observed=observed,
                    missing=missing,
                    missingness_disposition=variable.missingness,
                )
            )
    return rows


def _dependence_estimates(
    spec: TrialCharacterisationSpec,
    participants: pd.DataFrame,
    groups: tuple[tuple[str, pd.DataFrame], ...],
    dependence: DependenceSpec,
) -> list[TidyEstimate]:
    rows: list[TidyEstimate] = []
    selected_groups = groups if dependence.stratify_by_arm else groups[:1]
    for group, frame in selected_groups:
        pairs = frame[[dependence.left_column, dependence.right_column]].apply(
            pd.to_numeric, errors="coerce"
        )
        complete = pairs.dropna()
        if len(complete) < 4:
            raise ValueError(
                f"{dependence.dependence_id} requires at least four complete pairs in {group!r}"
            )
        values = complete.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"{dependence.dependence_id} contains non-finite complete pairs"
            )
        estimate = _spearman(values[:, 0], values[:, 1])
        result = bootstrap(
            (values[:, 0], values[:, 1]),
            _spearman,
            paired=True,
            vectorized=False,
            n_resamples=spec.bootstrap_replicates,
            confidence_level=spec.confidence_level,
            method="percentile",
            rng=np.random.default_rng(
                _derived_seed(spec.seed, dependence.dependence_id, group)
            ),
        )
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
        if not np.isfinite((low, high)).all():
            raise ValueError(
                f"{dependence.dependence_id} bootstrap produced non-finite limits"
            )
        rows.append(
            _estimate(
                spec,
                evidence_level="participant_distribution",
                property_id=f"dependence.{dependence.dependence_id}.spearman",
                group=group,
                estimate=estimate,
                interval_low=min(low, estimate),
                interval_high=max(high, estimate),
                unit="spearman_rho",
                independent_unit="participant",
                estimator="spearman_rank_correlation",
                uncertainty_method=f"paired_percentile_bootstrap_{spec.bootstrap_replicates}",
                denominator=len(frame),
                observed=len(complete),
                missing=len(frame) - len(complete),
                missingness_disposition=dependence.missingness,
            )
        )
    return rows


def _outcome_frame(
    spec: TrialCharacterisationSpec,
    data: TrialData,
    outcome: OutcomeSpec,
) -> pd.DataFrame:
    frame = (
        data.participants.copy()
        if outcome.table == "participants"
        else data.observation_tables.get(outcome.table)
    )
    if frame is None:
        raise ValueError(
            f"{outcome.outcome_id} requires observation table {outcome.table!r}"
        )
    if outcome.participant_id_column not in frame:
        raise ValueError(
            f"{outcome.outcome_id} table lacks participant key {outcome.participant_id_column!r}"
        )
    participant_arms = data.participants[
        [spec.participant_id_column, spec.arm_column]
    ].rename(columns={spec.participant_id_column: outcome.participant_id_column})
    if outcome.table == "participants":
        if outcome.participant_id_column != spec.participant_id_column:
            raise ValueError(
                "participant-table outcomes must use the trial participant ID column"
            )
        return frame
    if spec.arm_column in frame:
        raise ValueError(
            "observation tables must not duplicate the participant treatment-arm column"
        )
    merged = frame.merge(
        participant_arms,
        on=outcome.participant_id_column,
        how="left",
        validate="many_to_one",
    )
    if merged[spec.arm_column].isna().any():
        raise ValueError(
            f"{outcome.outcome_id} contains participant IDs absent from the participant table"
        )
    return merged


def _outcome_estimates(
    spec: TrialCharacterisationSpec,
    participants: pd.DataFrame,
    frame: pd.DataFrame,
    outcome: OutcomeSpec,
) -> list[TidyEstimate]:
    if isinstance(outcome, BinaryOutcomeSpec):
        return _binary_outcome(spec, frame, outcome)
    if isinstance(outcome, ContinuousOutcomeSpec):
        return _continuous_outcome(spec, frame, outcome)
    if isinstance(outcome, OrdinalOutcomeSpec):
        return _ordinal_outcome(spec, frame, outcome)
    if isinstance(outcome, SurvivalOutcomeSpec):
        return _survival_outcome(spec, frame, outcome)
    if isinstance(outcome, LongitudinalOutcomeSpec):
        return _longitudinal_outcome(spec, participants, frame, outcome)
    if isinstance(outcome, RecurrentEventOutcomeSpec):
        return _recurrent_outcome(spec, participants, frame, outcome)
    if isinstance(outcome, CompetingRiskOutcomeSpec):
        return _competing_risk_outcome(spec, frame, outcome)
    raise TypeError(f"unsupported outcome specification: {type(outcome).__name__}")


def _binary_outcome(
    spec: TrialCharacterisationSpec,
    frame: pd.DataFrame,
    outcome: BinaryOutcomeSpec,
) -> list[TidyEstimate]:
    _require_one_row_per_participant(
        frame, outcome.participant_id_column, outcome.outcome_id
    )
    _require_columns(frame, (spec.arm_column, outcome.value_column), outcome.outcome_id)
    rows: list[TidyEstimate] = []
    for group, group_frame in _groups(frame, spec.arm_column):
        values = group_frame[outcome.value_column]
        observed_values = values.dropna()
        if observed_values.empty:
            raise ValueError(
                f"{outcome.outcome_id} has no observed values in {group!r}"
            )
        successes = int((observed_values == outcome.event_value).sum())
        low, high = proportion_interval(
            successes,
            len(observed_values),
            alpha=1.0 - spec.confidence_level,
        )
        rows.append(
            _estimate(
                spec,
                evidence_level="trial",
                property_id=f"outcome.{outcome.outcome_id}.event_probability",
                group=group,
                estimate=successes / len(observed_values),
                interval_low=low,
                interval_high=high,
                unit=outcome.unit,
                independent_unit="participant",
                estimator="sample_proportion",
                uncertainty_method=f"wilson_{spec.confidence_level:.3f}",
                denominator=len(group_frame),
                observed=len(observed_values),
                missing=int(values.isna().sum()),
                missingness_disposition=outcome.missingness,
            )
        )
    return rows


def _continuous_outcome(
    spec: TrialCharacterisationSpec,
    frame: pd.DataFrame,
    outcome: ContinuousOutcomeSpec,
) -> list[TidyEstimate]:
    _require_one_row_per_participant(
        frame, outcome.participant_id_column, outcome.outcome_id
    )
    _require_columns(frame, (spec.arm_column, outcome.value_column), outcome.outcome_id)
    rows: list[TidyEstimate] = []
    for group, group_frame in _groups(frame, spec.arm_column):
        values, missing = _finite_numeric(
            group_frame[outcome.value_column], label=outcome.value_column
        )
        low, high = _mean_interval(values, spec.confidence_level)
        rows.append(
            _estimate(
                spec,
                evidence_level="trial",
                property_id=f"outcome.{outcome.outcome_id}.mean",
                group=group,
                estimate=float(np.mean(values)),
                interval_low=low,
                interval_high=high,
                unit=outcome.unit,
                independent_unit="participant",
                estimator="arithmetic_mean",
                uncertainty_method=f"student_t_{spec.confidence_level:.3f}",
                denominator=len(group_frame),
                observed=len(values),
                missing=missing,
                missingness_disposition=outcome.missingness,
            )
        )
    return rows


def _ordinal_outcome(
    spec: TrialCharacterisationSpec,
    frame: pd.DataFrame,
    outcome: OrdinalOutcomeSpec,
) -> list[TidyEstimate]:
    _require_one_row_per_participant(
        frame, outcome.participant_id_column, outcome.outcome_id
    )
    _require_columns(frame, (spec.arm_column, outcome.value_column), outcome.outcome_id)
    rows: list[TidyEstimate] = []
    declared = set(outcome.categories)
    observed_categories = set(frame[outcome.value_column].dropna().astype(str))
    unknown = sorted(observed_categories - declared)
    if unknown:
        raise ValueError(
            f"{outcome.outcome_id} contains undeclared ordinal categories: {unknown!r}"
        )
    for group, group_frame in _groups(frame, spec.arm_column):
        values = group_frame[outcome.value_column]
        observed = values.dropna().astype(str)
        if observed.empty:
            raise ValueError(
                f"{outcome.outcome_id} has no observed values in {group!r}"
            )
        cumulative = 0
        for category in outcome.categories:
            count = int((observed == category).sum())
            cumulative += count
            for metric, successes in (
                ("probability", count),
                ("cumulative_probability", cumulative),
            ):
                low, high = proportion_interval(
                    successes,
                    len(observed),
                    alpha=1.0 - spec.confidence_level,
                )
                rows.append(
                    _estimate(
                        spec,
                        evidence_level="trial",
                        property_id=f"outcome.{outcome.outcome_id}.{metric}.{_slug(category)}",
                        group=group,
                        estimate=successes / len(observed),
                        interval_low=low,
                        interval_high=high,
                        unit=outcome.unit,
                        independent_unit="participant",
                        estimator="sample_proportion",
                        uncertainty_method=f"wilson_{spec.confidence_level:.3f}",
                        denominator=len(group_frame),
                        observed=len(observed),
                        missing=int(values.isna().sum()),
                        missingness_disposition=outcome.missingness,
                    )
                )
    return rows


def _survival_outcome(
    spec: TrialCharacterisationSpec,
    frame: pd.DataFrame,
    outcome: SurvivalOutcomeSpec,
) -> list[TidyEstimate]:
    _require_one_row_per_participant(
        frame, outcome.participant_id_column, outcome.outcome_id
    )
    _require_columns(
        frame,
        (spec.arm_column, outcome.duration_column, outcome.event_column),
        outcome.outcome_id,
    )
    rows: list[TidyEstimate] = []
    for group, group_frame in _groups(frame, spec.arm_column):
        complete = (
            group_frame[[outcome.duration_column, outcome.event_column]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        missing = len(group_frame) - len(complete)
        durations = complete[outcome.duration_column].to_numpy(dtype=float)
        events = complete[outcome.event_column].to_numpy(dtype=int)
        if (
            len(durations) < 2
            or not np.isfinite(durations).all()
            or np.any(durations < 0)
        ):
            raise ValueError(
                f"{outcome.outcome_id} requires finite nonnegative durations in {group!r}"
            )
        if not set(np.unique(events)).issubset({0, 1}):
            raise ValueError(
                f"{outcome.outcome_id} event indicator must contain only zero and one"
            )
        fit = KaplanMeierFitter(alpha=1.0 - spec.confidence_level).fit(
            durations,
            event_observed=events,
            label=group,
        )
        for horizon in outcome.horizons:
            estimate = float(fit.predict(horizon))
            low, high = _step_interval(
                fit.confidence_interval_,
                horizon,
                lower_column=f"{group}_lower_{spec.confidence_level:.2f}",
                upper_column=f"{group}_upper_{spec.confidence_level:.2f}",
            )
            rows.append(
                _estimate(
                    spec,
                    evidence_level="trial",
                    property_id=f"outcome.{outcome.outcome_id}.survival_probability",
                    group=group,
                    time=horizon,
                    estimate=estimate,
                    interval_low=min(low, estimate),
                    interval_high=max(high, estimate),
                    unit="probability",
                    independent_unit="participant",
                    estimator="kaplan_meier",
                    uncertainty_method=f"greenwood_log_log_{spec.confidence_level:.3f}",
                    denominator=len(group_frame),
                    observed=len(complete),
                    missing=missing,
                    missingness_disposition=outcome.missingness,
                )
            )
    return rows


def _longitudinal_outcome(
    spec: TrialCharacterisationSpec,
    participants: pd.DataFrame,
    frame: pd.DataFrame,
    outcome: LongitudinalOutcomeSpec,
) -> list[TidyEstimate]:
    _require_columns(
        frame,
        (spec.arm_column, outcome.time_column, outcome.value_column),
        outcome.outcome_id,
    )
    times = pd.to_numeric(frame[outcome.time_column], errors="coerce")
    if times.isna().any() or not np.isfinite(times.to_numpy(dtype=float)).all():
        raise ValueError(f"{outcome.outcome_id} times must be finite numeric values")
    frame = frame.assign(**{outcome.time_column: times})
    duplicate = frame.duplicated([outcome.participant_id_column, outcome.time_column])
    if duplicate.any():
        raise ValueError(
            f"{outcome.outcome_id} requires at most one record per participant and time"
        )
    rows: list[TidyEstimate] = []
    for group, participant_group in _groups(participants, spec.arm_column):
        if group == "overall":
            group_frame = frame
        else:
            group_frame = frame.loc[frame[spec.arm_column].astype(str) == group]
        denominator = len(participant_group)
        for scheduled_time in outcome.scheduled_times:
            current = group_frame.loc[
                group_frame[outcome.time_column] == scheduled_time
            ]
            values, _ = _finite_numeric(
                current[outcome.value_column], label=outcome.value_column
            )
            participant_observed = current.loc[
                current[outcome.value_column].notna(), outcome.participant_id_column
            ].nunique()
            if participant_observed != len(values):
                raise ValueError(
                    f"{outcome.outcome_id} has ambiguous participant records at time {scheduled_time}"
                )
            low, high = _mean_interval(values, spec.confidence_level)
            rows.append(
                _estimate(
                    spec,
                    evidence_level="trial",
                    property_id=f"outcome.{outcome.outcome_id}.mean",
                    group=group,
                    time=scheduled_time,
                    estimate=float(np.mean(values)),
                    interval_low=low,
                    interval_high=high,
                    unit=outcome.value_unit,
                    independent_unit="participant",
                    estimator="visit_specific_arithmetic_mean",
                    uncertainty_method=f"student_t_{spec.confidence_level:.3f}",
                    denominator=denominator,
                    observed=len(values),
                    missing=denominator - len(values),
                    missingness_disposition=outcome.missingness,
                )
            )
            attendance_low, attendance_high = proportion_interval(
                len(values),
                denominator,
                alpha=1.0 - spec.confidence_level,
            )
            rows.append(
                _estimate(
                    spec,
                    evidence_level="trial",
                    property_id=f"observation.{outcome.outcome_id}.attendance_probability",
                    group=group,
                    time=scheduled_time,
                    estimate=len(values) / denominator,
                    interval_low=attendance_low,
                    interval_high=attendance_high,
                    unit="proportion",
                    independent_unit="participant",
                    estimator="sample_proportion",
                    uncertainty_method=f"wilson_{spec.confidence_level:.3f}",
                    denominator=denominator,
                    observed=len(values),
                    missing=denominator - len(values),
                    missingness_disposition=outcome.missingness,
                )
            )
    return rows


def _recurrent_outcome(
    spec: TrialCharacterisationSpec,
    participants: pd.DataFrame,
    frame: pd.DataFrame,
    outcome: RecurrentEventOutcomeSpec,
) -> list[TidyEstimate]:
    _require_columns(
        frame, (spec.arm_column, outcome.event_time_column), outcome.outcome_id
    )
    event_times = pd.to_numeric(frame[outcome.event_time_column], errors="coerce")
    if (
        event_times.isna().any()
        or not np.isfinite(event_times.to_numpy(dtype=float)).all()
        or (event_times < 0).any()
    ):
        raise ValueError(
            f"{outcome.outcome_id} event times must be finite and nonnegative"
        )
    frame = frame.assign(**{outcome.event_time_column: event_times})
    rows: list[TidyEstimate] = []
    for group, participant_group in _groups(participants, spec.arm_column):
        group_frame = (
            frame
            if group == "overall"
            else frame.loc[frame[spec.arm_column].astype(str) == group]
        )
        participant_ids = participant_group[spec.participant_id_column].astype(str)
        for horizon in outcome.horizons:
            counts = (
                group_frame.loc[group_frame[outcome.event_time_column] <= horizon]
                .groupby(outcome.participant_id_column, observed=True)
                .size()
                .reindex(participant_ids, fill_value=0)
                .to_numpy(dtype=float)
            )
            low, high = _mean_interval(counts, spec.confidence_level)
            rows.append(
                _estimate(
                    spec,
                    evidence_level="trial",
                    property_id=f"outcome.{outcome.outcome_id}.mean_cumulative_count",
                    group=group,
                    time=horizon,
                    estimate=float(np.mean(counts)),
                    interval_low=low,
                    interval_high=high,
                    unit=f"events_per_participant_by_{outcome.unit}",
                    independent_unit="participant",
                    estimator="mean_cumulative_count",
                    uncertainty_method=f"student_t_{spec.confidence_level:.3f}",
                    denominator=len(counts),
                    observed=len(counts),
                    missing=0,
                    missingness_disposition=outcome.missingness,
                )
            )
    return rows


def _competing_risk_outcome(
    spec: TrialCharacterisationSpec,
    frame: pd.DataFrame,
    outcome: CompetingRiskOutcomeSpec,
) -> list[TidyEstimate]:
    _require_one_row_per_participant(
        frame, outcome.participant_id_column, outcome.outcome_id
    )
    _require_columns(
        frame,
        (spec.arm_column, outcome.duration_column, outcome.event_type_column),
        outcome.outcome_id,
    )
    rows: list[TidyEstimate] = []
    permitted_codes = {0, outcome.primary_event_code, *outcome.competing_event_codes}
    for group, group_frame in _groups(frame, spec.arm_column):
        complete = (
            group_frame[[outcome.duration_column, outcome.event_type_column]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        durations = complete[outcome.duration_column].to_numpy(dtype=float)
        event_types = complete[outcome.event_type_column].to_numpy(dtype=int)
        if (
            len(durations) < 2
            or not np.isfinite(durations).all()
            or np.any(durations < 0)
        ):
            raise ValueError(
                f"{outcome.outcome_id} requires finite nonnegative durations in {group!r}"
            )
        unknown = sorted(set(np.unique(event_types)) - permitted_codes)
        if unknown:
            raise ValueError(
                f"{outcome.outcome_id} contains undeclared event codes: {unknown!r}"
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Tied event times were detected.*",
                module=r"lifelines\.fitters\.aalen_johansen_fitter",
            )
            fit = AalenJohansenFitter(
                alpha=1.0 - spec.confidence_level,
                seed=_derived_seed(spec.seed, outcome.outcome_id, group),
            ).fit(
                durations,
                event_observed=event_types,
                event_of_interest=outcome.primary_event_code,
            )
        estimate_column = str(fit.cumulative_density_.columns[0])
        lower_column, upper_column = map(str, fit.confidence_interval_.columns)
        for horizon in outcome.horizons:
            estimate = _step_value(fit.cumulative_density_[estimate_column], horizon)
            low, high = _step_interval(
                fit.confidence_interval_,
                horizon,
                lower_column=lower_column,
                upper_column=upper_column,
            )
            rows.append(
                _estimate(
                    spec,
                    evidence_level="trial",
                    property_id=f"outcome.{outcome.outcome_id}.cumulative_incidence",
                    group=group,
                    time=horizon,
                    estimate=estimate,
                    interval_low=min(low, estimate),
                    interval_high=max(high, estimate),
                    unit="probability",
                    independent_unit="participant",
                    estimator="aalen_johansen",
                    uncertainty_method=f"aalen_johansen_{spec.confidence_level:.3f}",
                    denominator=len(group_frame),
                    observed=len(complete),
                    missing=len(group_frame) - len(complete),
                    missingness_disposition=outcome.missingness,
                )
            )
    return rows


def _aggregate_estimates(
    estimates: Iterable[TidyEstimate],
    *,
    evidence_level: str,
    trial_id: str,
    programme_id: str,
    uncertainty_method: str,
    independent_unit: str,
) -> tuple[TidyEstimate, ...]:
    grouped: dict[tuple[str, str, float | None], list[TidyEstimate]] = defaultdict(list)
    for row in estimates:
        grouped[(row.property_id, row.group, row.time)].append(row)
    output: list[TidyEstimate] = []
    for (property_id, group, time), values in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2] or -1.0),
    ):
        units = {row.unit for row in values}
        estimators = {row.estimator for row in values}
        if len(units) != 1 or len(estimators) != 1:
            raise ValueError(
                f"incomparable estimates for {property_id!r}/{group!r}/{time!r}"
            )
        observed = np.asarray([row.estimate for row in values], dtype=float)
        estimate = float(np.median(observed))
        output.append(
            TidyEstimate(
                trial_id=trial_id,
                programme_id=programme_id,
                evidence_level=evidence_level,
                property_id=property_id,
                group=group,
                time=time,
                estimate=estimate,
                interval_low=float(np.min(observed)),
                interval_high=float(np.max(observed)),
                unit=next(iter(units)),
                independent_unit=independent_unit,
                estimator=f"median_of_{next(iter(estimators))}",
                uncertainty_method=uncertainty_method,
                denominator=len(values),
                observed=len(values),
                missing=0,
                missingness_disposition="not_applicable",
            )
        )
    if not output:
        raise ValueError("characterisation aggregation requires at least one estimate")
    return tuple(output)


def _finite_numeric(series: pd.Series, *, label: str) -> tuple[np.ndarray, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    observed = numeric.dropna().to_numpy(dtype=float)
    if len(observed) < 2:
        raise ValueError(f"{label} requires at least two observed numeric values")
    if not np.isfinite(observed).all():
        raise ValueError(f"{label} contains non-finite observed values")
    return observed, int(numeric.isna().sum())


def _mean_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("mean uncertainty requires at least two observations")
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    if standard_error == 0:
        return mean, mean
    critical = float(t.ppf((1.0 + confidence_level) / 2.0, df=len(values) - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def _sd_interval(
    standard_deviation: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float]:
    if sample_size < 2:
        raise ValueError(
            "standard-deviation uncertainty requires at least two observations"
        )
    if standard_deviation == 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence_level
    variance = standard_deviation**2
    low = np.sqrt(
        (sample_size - 1) * variance / chi2.ppf(1.0 - alpha / 2.0, sample_size - 1)
    )
    high = np.sqrt(
        (sample_size - 1) * variance / chi2.ppf(alpha / 2.0, sample_size - 1)
    )
    return float(low), float(high)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        raise ValueError("rank dependence is undefined for a constant variable")
    estimate = float(spearmanr(left, right).statistic)
    if not np.isfinite(estimate):
        raise ValueError("rank dependence is undefined for a constant variable")
    return estimate


def _step_interval(
    frame: pd.DataFrame,
    time: float,
    *,
    lower_column: str,
    upper_column: str,
) -> tuple[float, float]:
    if lower_column not in frame or upper_column not in frame:
        raise ValueError("fitted interval columns are unavailable")
    return _step_value(frame[lower_column], time), _step_value(
        frame[upper_column], time
    )


def _step_value(series: pd.Series, time: float) -> float:
    eligible = series.loc[series.index.astype(float) <= time]
    value = float(series.iloc[0] if eligible.empty else eligible.iloc[-1])
    if not np.isfinite(value):
        raise ValueError("fitted step function produced a non-finite value")
    return value


def _require_one_row_per_participant(
    frame: pd.DataFrame, column: str, outcome_id: str
) -> None:
    if frame[column].isna().any() or frame[column].astype("string").duplicated().any():
        raise ValueError(f"{outcome_id} requires one non-missing row per participant")


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], outcome_id: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{outcome_id} table lacks required columns: {missing!r}")


def _derived_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(":".join((str(seed), *parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _slug(value: str) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "_" for character in value
    ).strip("_")
    if not slug:
        raise ValueError("category labels must contain an alphanumeric character")
    return slug


def _estimate(
    spec: TrialCharacterisationSpec,
    *,
    evidence_level: str,
    property_id: str,
    group: str,
    estimate: float,
    unit: str,
    independent_unit: str,
    estimator: str,
    uncertainty_method: str,
    denominator: int,
    observed: int,
    missing: int,
    missingness_disposition: str,
    time: float | None = None,
    interval_low: float | None = None,
    interval_high: float | None = None,
) -> TidyEstimate:
    return TidyEstimate(
        trial_id=spec.trial_id,
        programme_id=spec.programme_id,
        evidence_level=evidence_level,
        property_id=property_id,
        group=group,
        time=time,
        estimate=estimate,
        interval_low=interval_low,
        interval_high=interval_high,
        unit=unit,
        independent_unit=independent_unit,
        estimator=estimator,
        uncertainty_method=uncertainty_method,
        denominator=denominator,
        observed=observed,
        missing=missing,
        missingness_disposition=missingness_disposition,
    )


__all__ = [
    "characterise_trial",
    "summarise_characterisations",
    "write_characterisation_csv",
]
