"""Independent public-data replay for validated endpoint outcomes."""

from __future__ import annotations

from zipfile import ZipFile

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit

from trialagentbench_validation.contracts.scoring.route_reference_inputs import (
    RouteReferenceInputRecordV1,
)
from trialagentbench_validation.contracts.v1_scope import (
    TRIALEVAL_PARTIAL_IDENTIFICATION_DELTA_GRID_V1,
    TRIALEVAL_VALIDATED_ENDPOINT_BOOTSTRAP_REPLICATES_V1,
    TRIALEVAL_VALIDATED_ENDPOINT_BOOTSTRAP_SEED_V1,
    TRIALEVAL_VALIDATED_ENDPOINT_CELL_COLUMNS_V1,
    TRIALEVAL_VALIDATED_ENDPOINT_MAXIMUM_CLASSIFICATION_PROBABILITY_V1,
    TRIALEVAL_VALIDATED_ENDPOINT_MINIMUM_CLASSIFICATION_PROBABILITY_V1,
    TRIALEVAL_VALIDATED_ENDPOINT_MINIMUM_CONVERGED_BOOTSTRAPS_V1,
    TRIALEVAL_VALIDATED_ENDPOINT_OPTIMIZER_GRADIENT_TOLERANCE_V1,
    TRIALEVAL_VALIDATED_ENDPOINT_OPTIMIZER_MAXIMUM_ITERATIONS_V1,
    TRIALEVAL_VALIDATED_ENDPOINT_OPTIMIZER_MAXIMUM_LINE_SEARCH_STEPS_V1,
    TRIALEVAL_VALIDATED_ENDPOINT_PROBABILITY_CLIP_V1,
)
from trialagentbench_validation.trialeval.references.io import (
    read_json_from_public_v1,
    read_required_table_by_suffix_v1,
    read_treatment_surface_table_v1,
    required_str_v1,
)

_FREQUENCY_COLUMN = "_frequency"
_MINIMUM_CLASSIFICATION_PROBABILITY_V1 = (
    TRIALEVAL_VALIDATED_ENDPOINT_MINIMUM_CLASSIFICATION_PROBABILITY_V1
)
_MAXIMUM_CLASSIFICATION_PROBABILITY_V1 = (
    TRIALEVAL_VALIDATED_ENDPOINT_MAXIMUM_CLASSIFICATION_PROBABILITY_V1
)
_CLASSIFICATION_LOGIT_BOUNDS_V1 = (
    float(logit(_MINIMUM_CLASSIFICATION_PROBABILITY_V1)),
    float(logit(_MAXIMUM_CLASSIFICATION_PROBABILITY_V1)),
)


def _analysis_frame(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> pd.DataFrame:
    task = read_json_from_public_v1(
        public, f"items/{reference_input.task_id}/task.json"
    )
    design = read_json_from_public_v1(
        public,
        f"items/{reference_input.task_id}/ascertainment_model.json",
    )
    endpoint_id = required_str_v1(task, "primary_paramcd")
    if required_str_v1(design, "endpoint_id") != endpoint_id:
        raise ValueError(
            "Endpoint-validation design and primary endpoint do not match."
        )
    adsl = read_treatment_surface_table_v1(
        public=public, reference_input=reference_input
    )
    adtte = read_required_table_by_suffix_v1(
        public=public,
        reference_input=reference_input,
        suffix="ADTTE.parquet",
    )
    primary = adtte.loc[
        adtte["PARAMCD"]
        .astype("string")
        .isin({endpoint_id, "__RECONSTRUCTED_PRIMARY__"}),
        :,
    ].copy()
    if primary.empty or primary["USUBJID"].duplicated().any():
        raise ValueError(
            "Validated-endpoint replay requires one primary endpoint row per participant."
        )
    arm_column = "TRTA" if "TRTA" in adsl.columns else "ARMCD"
    if arm_column not in adsl.columns:
        raise ValueError(
            "Validated-endpoint replay requires a public treatment assignment."
        )
    frame = adsl.loc[:, ["USUBJID", arm_column]].rename(
        columns={arm_column: "treatment_arm"}
    )
    frame = frame.merge(
        primary.loc[:, ["USUBJID", "CNSR"]],
        on="USUBJID",
        how="left",
        validate="one_to_one",
    )
    released = {"VALSTRAT", "OBSEVNT", "VALIDFL", "ADJEVNT"}
    if released.issubset(primary.columns):
        frame = frame.merge(
            primary.loc[:, ["USUBJID", *sorted(released)]],
            on="USUBJID",
            how="left",
            validate="one_to_one",
        )
    else:
        baseline = read_required_table_by_suffix_v1(
            public=public,
            reference_input=reference_input,
            suffix="baseline_characteristics.parquet",
        )
        stratum_variable = required_str_v1(design, "prognostic_stratum_variable")
        if {"USUBJID", stratum_variable} - set(baseline.columns):
            raise ValueError(
                "Raw endpoint-validation replay lacks its declared prognostic variable."
            )
        values = pd.to_numeric(baseline[stratum_variable], errors="raise")
        strata = baseline.loc[:, ["USUBJID"]].copy()
        strata["VALSTRAT"] = values.ge(float(values.median())).map(
            {
                False: "lower_risk",
                True: "higher_risk",
            }
        )
        adjudication = read_required_table_by_suffix_v1(
            public=public,
            reference_input=reference_input,
            suffix="endpoint_adjudication.parquet",
        )
        missing = sorted({"USUBJID", "VALIDFL", "ADJEVNT"} - set(adjudication.columns))
        if missing:
            raise ValueError(
                f"Raw endpoint-validation replay lacks adjudication fields: {missing!r}."
            )
        selected_rows = adjudication.loc[
            pd.to_numeric(adjudication["VALIDFL"], errors="coerce").eq(1),
            ["USUBJID", "VALIDFL", "ADJEVNT"],
        ].drop_duplicates("USUBJID")
        frame = frame.merge(strata, on="USUBJID", how="left", validate="one_to_one")
        frame = frame.merge(
            selected_rows, on="USUBJID", how="left", validate="one_to_one"
        )
        frame["VALIDFL"] = frame["VALIDFL"].fillna(0).astype("int64")
        frame["OBSEVNT"] = (
            1 - pd.to_numeric(frame["CNSR"], errors="raise").astype("int64")
        ).astype("int64")
    frame = frame.rename(
        columns={
            "VALSTRAT": "prognostic_stratum",
            "OBSEVNT": "observed_event",
            "VALIDFL": "validation_selected",
            "ADJEVNT": "validated_event",
        }
    )
    required = {
        "treatment_arm",
        "prognostic_stratum",
        "observed_event",
        "validation_selected",
        "validated_event",
    }
    if frame.loc[:, list(required - {"validated_event"})].isna().any().any():
        raise ValueError("Validated-endpoint replay contains incomplete public inputs.")
    selected_mask = frame["validation_selected"].astype("int64").eq(1)
    if (
        frame.loc[selected_mask, "validated_event"].isna().any()
        or frame.loc[~selected_mask, "validated_event"].notna().any()
    ):
        raise ValueError(
            "Adjudicated endpoint values must be exposed exactly for selected validation records."
        )
    return frame.loc[:, sorted(required)]


def _objective(
    parameters: NDArray[np.float64],
    *,
    arm_index: NDArray[np.int64],
    stratum_index: NDArray[np.int64],
    observed: NDArray[np.int64],
    selected: NDArray[np.bool_],
    validated: NDArray[np.int64],
    frequency: NDArray[np.int64],
    arm_count: int,
    stratum_count: int,
) -> float:
    risk_count = arm_count * stratum_count
    risk = expit(parameters[:risk_count]).reshape(arm_count, stratum_count)[
        arm_index, stratum_index
    ]
    sensitivity = expit(parameters[risk_count : risk_count + arm_count])[arm_index]
    specificity = expit(parameters[risk_count + arm_count :])[arm_index]
    event = (
        np.log(risk)
        + observed * np.log(sensitivity)
        + (1 - observed) * np.log1p(-sensitivity)
    )
    nonevent = (
        np.log1p(-risk)
        + (1 - observed) * np.log(specificity)
        + observed * np.log1p(-specificity)
    )
    known = validated * event + (1 - validated) * nonevent
    unknown = np.logaddexp(event, nonevent)
    return -float(
        np.sum(known[selected] * frequency[selected])
        + np.sum(unknown[~selected] * frequency[~selected])
    )


def _objective_gradient(
    parameters: NDArray[np.float64],
    *,
    arm_index: NDArray[np.int64],
    stratum_index: NDArray[np.int64],
    observed: NDArray[np.int64],
    selected: NDArray[np.bool_],
    validated: NDArray[np.int64],
    frequency: NDArray[np.int64],
    arm_count: int,
    stratum_count: int,
) -> NDArray[np.float64]:
    """Evaluate the exact score of the public validation likelihood."""

    risk_count = arm_count * stratum_count
    risk_cell = arm_index * stratum_count + stratum_index
    risk = expit(parameters[:risk_count])[risk_cell]
    sensitivity = expit(parameters[risk_count : risk_count + arm_count])[arm_index]
    specificity = expit(parameters[risk_count + arm_count :])[arm_index]
    event_log_likelihood = (
        np.log(risk)
        + observed * np.log(sensitivity)
        + (1 - observed) * np.log1p(-sensitivity)
    )
    nonevent_log_likelihood = (
        np.log1p(-risk)
        + (1 - observed) * np.log(specificity)
        + observed * np.log1p(-specificity)
    )
    posterior_event = np.exp(
        event_log_likelihood
        - np.logaddexp(event_log_likelihood, nonevent_log_likelihood)
    )
    expected_event = np.where(selected, validated, posterior_event)
    weights = frequency.astype(np.float64)
    score = np.zeros_like(parameters, dtype=np.float64)
    np.add.at(score, risk_cell, -weights * (expected_event - risk))
    np.add.at(
        score,
        risk_count + arm_index,
        -weights * expected_event * (observed - sensitivity),
    )
    np.add.at(
        score,
        risk_count + arm_count + arm_index,
        -weights * (1.0 - expected_event) * ((1 - observed) - specificity),
    )
    return score


def _fit_arm_risks(data: pd.DataFrame) -> dict[str, float]:
    frame = data.copy()
    if _FREQUENCY_COLUMN not in frame.columns:
        frame[_FREQUENCY_COLUMN] = 1

    def weighted_mean(rows: pd.DataFrame, column: str) -> float:
        return float(
            np.average(
                rows[column].to_numpy(dtype=float),
                weights=rows[_FREQUENCY_COLUMN].to_numpy(dtype=float),
            )
        )

    observed_arms = set(frame["treatment_arm"].astype(str).unique())
    if observed_arms != {"control", "treated"}:
        raise ValueError(
            "Validated-endpoint replay requires canonical control and treated arm identifiers."
        )
    arms = ("control", "treated")
    strata = tuple(sorted(frame["prognostic_stratum"].astype(str).unique()))
    if len(arms) != 2 or not strata:
        raise ValueError(
            "Validated-endpoint replay requires two arms and at least one stratum."
        )
    arm_lookup = {arm: index for index, arm in enumerate(arms)}
    stratum_lookup = {stratum: index for index, stratum in enumerate(strata)}
    likelihood_rows = pd.DataFrame(
        {
            "arm_index": frame["treatment_arm"]
            .astype(str)
            .map(arm_lookup)
            .to_numpy(dtype=np.int64),
            "stratum_index": frame["prognostic_stratum"]
            .astype(str)
            .map(stratum_lookup)
            .to_numpy(dtype=np.int64),
            "observed": frame["observed_event"].to_numpy(dtype=np.int64),
            "selected": frame["validation_selected"].to_numpy(dtype=np.int64),
            "validated": frame["validated_event"].fillna(0).to_numpy(dtype=np.int64),
            "frequency": frame[_FREQUENCY_COLUMN].to_numpy(dtype=np.int64),
        }
    )
    likelihood_cells = (
        likelihood_rows.groupby(
            [column for column in likelihood_rows.columns if column != "frequency"],
            observed=True,
            sort=True,
        )["frequency"]
        .sum()
        .reset_index()
    )
    arm_index = likelihood_cells["arm_index"].to_numpy(dtype=np.int64)
    stratum_index = likelihood_cells["stratum_index"].to_numpy(dtype=np.int64)
    observed = likelihood_cells["observed"].to_numpy(dtype=np.int64)
    selected_cells = likelihood_cells["selected"].to_numpy(dtype=bool)
    validated = likelihood_cells["validated"].to_numpy(dtype=np.int64)
    frequency = likelihood_cells["frequency"].to_numpy(dtype=np.int64)
    starts: list[float] = []
    for arm in arms:
        for stratum in strata:
            group = frame.loc[
                frame["treatment_arm"].astype(str).eq(arm)
                & frame["prognostic_stratum"].astype(str).eq(stratum)
            ]
            adjudicated = group.loc[group["validation_selected"].eq(1)]
            estimate = (
                weighted_mean(adjudicated, "validated_event")
                if not adjudicated.empty
                else weighted_mean(group, "observed_event")
            )
            starts.append(
                float(
                    np.clip(
                        estimate,
                        TRIALEVAL_VALIDATED_ENDPOINT_PROBABILITY_CLIP_V1,
                        1 - TRIALEVAL_VALIDATED_ENDPOINT_PROBABILITY_CLIP_V1,
                    )
                )
            )
    for arm in arms:
        group = frame.loc[
            frame["treatment_arm"].astype(str).eq(arm)
            & frame["validation_selected"].eq(1)
        ]
        events = group.loc[group["validated_event"].eq(1)]
        estimate = weighted_mean(events, "observed_event") if not events.empty else 0.5
        starts.append(
            float(
                np.clip(
                    estimate,
                    TRIALEVAL_VALIDATED_ENDPOINT_PROBABILITY_CLIP_V1,
                    1 - TRIALEVAL_VALIDATED_ENDPOINT_PROBABILITY_CLIP_V1,
                )
            )
        )
    for arm in arms:
        group = frame.loc[
            frame["treatment_arm"].astype(str).eq(arm)
            & frame["validation_selected"].eq(1)
        ]
        nonevents = group.loc[group["validated_event"].eq(0)]
        estimate = (
            1 - weighted_mean(nonevents, "observed_event")
            if not nonevents.empty
            else 0.5
        )
        starts.append(
            float(
                np.clip(
                    estimate,
                    TRIALEVAL_VALIDATED_ENDPOINT_PROBABILITY_CLIP_V1,
                    1 - TRIALEVAL_VALIDATED_ENDPOINT_PROBABILITY_CLIP_V1,
                )
            )
        )

    def objective(parameters: NDArray[np.float64]) -> float:
        return _objective(
            parameters,
            arm_index=arm_index,
            stratum_index=stratum_index,
            observed=observed,
            selected=selected_cells,
            validated=validated,
            frequency=frequency,
            arm_count=len(arms),
            stratum_count=len(strata),
        )

    def gradient(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return _objective_gradient(
            parameters,
            arm_index=arm_index,
            stratum_index=stratum_index,
            observed=observed,
            selected=selected_cells,
            validated=validated,
            frequency=frequency,
            arm_count=len(arms),
            stratum_count=len(strata),
        )

    fits = []
    risk_count = len(arms) * len(strata)
    parameter_bounds = (
        *((None, None),) * risk_count,
        *(_CLASSIFICATION_LOGIT_BOUNDS_V1,) * (2 * len(arms)),
    )
    for start in (
        np.zeros(len(starts), dtype=np.float64),
        logit(np.asarray(starts, dtype=np.float64)),
    ):
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=gradient,
            bounds=parameter_bounds,
            options={
                "ftol": 0.0,
                "gtol": TRIALEVAL_VALIDATED_ENDPOINT_OPTIMIZER_GRADIENT_TOLERANCE_V1,
                "maxiter": TRIALEVAL_VALIDATED_ENDPOINT_OPTIMIZER_MAXIMUM_ITERATIONS_V1,
                "maxls": TRIALEVAL_VALIDATED_ENDPOINT_OPTIMIZER_MAXIMUM_LINE_SEARCH_STEPS_V1,
            },
        )
        if result.success and np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            fits.append(result)
    if not fits:
        raise ValueError("Validated-endpoint public likelihood did not converge.")
    fitted = min(fits, key=lambda result: float(result.fun))
    risk = expit(fitted.x[: len(arms) * len(strata)]).reshape(len(arms), len(strata))
    counts = frame.groupby("prognostic_stratum", observed=True)[_FREQUENCY_COLUMN].sum()
    participant_count = int(frame[_FREQUENCY_COLUMN].sum())
    weights = np.asarray(
        [float(counts[stratum]) / participant_count for stratum in strata]
    )
    standardized = risk @ weights
    return {arm: float(standardized[index]) for index, arm in enumerate(arms)}


def _point(data: pd.DataFrame) -> float:
    risks = _fit_arm_risks(data)
    return float(risks["treated"] - risks["control"])


def validated_endpoint_point_and_standard_error_v1(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
) -> tuple[float, float]:
    """Replay the joint likelihood and fixed stratified bootstrap."""

    data = _analysis_frame(public=public, reference_input=reference_input)
    point = _point(data)
    arms = ("control", "treated")
    strata = tuple(sorted(data["prognostic_stratum"].astype(str).unique()))
    data[_FREQUENCY_COLUMN] = 1
    cell_columns = list(TRIALEVAL_VALIDATED_ENDPOINT_CELL_COLUMNS_V1)
    cells = (
        data.groupby(cell_columns, observed=True, sort=True, dropna=False)[
            _FREQUENCY_COLUMN
        ]
        .sum()
        .reset_index()
    )
    groups = tuple(
        cells.loc[
            cells["treatment_arm"].astype(str).eq(arm)
            & cells["prognostic_stratum"].astype(str).eq(stratum)
        ].reset_index(drop=True)
        for arm in arms
        for stratum in strata
    )
    rng = np.random.RandomState(TRIALEVAL_VALIDATED_ENDPOINT_BOOTSTRAP_SEED_V1)
    estimates: list[float] = []
    for _ in range(TRIALEVAL_VALIDATED_ENDPOINT_BOOTSTRAP_REPLICATES_V1):
        replicate_groups: list[pd.DataFrame] = []
        for group in groups:
            frequencies = group[_FREQUENCY_COLUMN].to_numpy(dtype=np.int64)
            participant_count = int(frequencies.sum())
            sampled = rng.multinomial(
                participant_count,
                frequencies.astype(float) / participant_count,
            )
            retained = group.loc[sampled > 0].copy()
            retained[_FREQUENCY_COLUMN] = sampled[sampled > 0]
            replicate_groups.append(retained)
        replicate = pd.concat(replicate_groups, ignore_index=True)
        try:
            estimates.append(_point(replicate))
        except ValueError:
            continue
    if len(estimates) < TRIALEVAL_VALIDATED_ENDPOINT_MINIMUM_CONVERGED_BOOTSTRAPS_V1:
        raise ValueError(
            "Validated-endpoint public bootstrap requires at least "
            f"{TRIALEVAL_VALIDATED_ENDPOINT_MINIMUM_CONVERGED_BOOTSTRAPS_V1} converged replicates."
        )
    return point, float(np.asarray(estimates, dtype=np.float64).std(ddof=1))


def validated_endpoint_bounds_v1(
    *,
    public: ZipFile,
    reference_input: RouteReferenceInputRecordV1,
    delta: float | None,
) -> tuple[float, float]:
    """Replay bounded-deviation or worst-case support for one unsupported stratum."""

    if delta is not None and float(delta) not in set(
        TRIALEVAL_PARTIAL_IDENTIFICATION_DELTA_GRID_V1
    ):
        raise ValueError(
            "Validated-endpoint public bounds require the fixed sensitivity grid "
            f"{TRIALEVAL_PARTIAL_IDENTIFICATION_DELTA_GRID_V1!r}."
        )
    data = _analysis_frame(public=public, reference_input=reference_input)
    arms = ("control", "treated")
    strata = tuple(sorted(data["prognostic_stratum"].astype(str).unique()))
    unsupported = tuple(
        stratum
        for stratum in strata
        if data.loc[
            data["prognostic_stratum"].astype(str).eq(stratum)
            & data["validation_selected"].eq(1)
        ].empty
    )
    if len(unsupported) != 1:
        raise ValueError(
            "Validated-endpoint bounds require exactly one wholly unsupported stratum."
        )
    supported = tuple(stratum for stratum in strata if stratum not in unsupported)
    supported_risks = _fit_arm_risks(
        data.loc[data["prognostic_stratum"].astype(str).isin(supported)]
    )
    weights = data.groupby("prognostic_stratum", observed=True).size() / len(data)
    unsupported_stratum = unsupported[0]
    arm_bounds: dict[str, tuple[float, float]] = {}
    for arm in arms:
        observed_risk = float(
            data.loc[
                data["treatment_arm"].astype(str).eq(arm)
                & data["prognostic_stratum"].astype(str).eq(unsupported_stratum),
                "observed_event",
            ].mean()
        )
        supported_weight = float(sum(float(weights[stratum]) for stratum in supported))
        unsupported_weight = float(weights[unsupported_stratum])
        unsupported_bounds = (
            (0.0, 1.0)
            if delta is None
            else (
                max(0.0, observed_risk - float(delta)),
                min(1.0, observed_risk + float(delta)),
            )
        )
        arm_bounds[arm] = (
            supported_weight * supported_risks[arm]
            + unsupported_weight * unsupported_bounds[0],
            supported_weight * supported_risks[arm]
            + unsupported_weight * unsupported_bounds[1],
        )
    return (
        float(arm_bounds["treated"][0] - arm_bounds["control"][1]),
        float(arm_bounds["treated"][1] - arm_bounds["control"][0]),
    )


__all__ = [
    "validated_endpoint_bounds_v1",
    "validated_endpoint_point_and_standard_error_v1",
]
