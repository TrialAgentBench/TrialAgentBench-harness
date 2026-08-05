"""Measure realised trial designs from public release records."""

from __future__ import annotations

import csv
import json
import math
from io import BytesIO
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from statsmodels.duration.survfunc import SurvfuncRight

from trialagentbench_validation.characterisation.cluster_statistics import (
    one_way_cluster_information,
)
from trialagentbench_validation.characterisation.contracts import (
    DesignAnalysisComparison,
    DesignProperty,
    DesignReleaseCharacterisation,
    ReleaseCharacterisation,
    TrialProfile,
)
from trialagentbench_validation.contracts.scoring.route_references import (
    RouteReferenceRecordV1,
)
from trialagentbench_validation.contracts.v1_scope import (
    TRIALEVAL_STANDARDIZED_RISK_BASELINE_COVARIATES_V1,
)
from trialagentbench_validation.io import sha256_file
from trialagentbench_validation.trialeval.public_archive import resolve_public_member_v1
from trialagentbench_validation.trialeval.references.standardized import (
    cox_linear_standardized_risk_difference_tau_reference_with_uncertainty_v1,
)
from trialagentbench_validation.trialeval.references.stepped_wedge import (
    stepped_wedge_period_adjusted_baseline_rates_v1,
    stepped_wedge_unadjusted_risk_difference_tau_with_uncertainty_v1,
)

_Z_95 = float(stats.norm.ppf(0.975))


def characterise_design_release(
    *,
    participant_archive: Path,
    verification_archive: Path,
    release: ReleaseCharacterisation,
) -> DesignReleaseCharacterisation:
    """Measure the Design axis and paired design-aware analyses.

    Parameters
    ----------
    participant_archive
        Public TrialEval participant ZIP.
    verification_archive
        Public TrialEval verification ZIP.
    release
        Complete release characterisation for the same archives.

    Returns
    -------
    DesignReleaseCharacterisation
        Realised properties and same-estimand design comparisons.

    Raises
    ------
    FileNotFoundError
        If either archive is absent.
    ValueError
        If archive identities or required public design records are invalid.
    """

    for path in (participant_archive, verification_archive):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(participant_archive) != release.participant_archive_sha256:
        raise ValueError(
            "participant archive does not match the release characterisation"
        )
    if sha256_file(verification_archive) != release.verification_archive_sha256:
        raise ValueError(
            "verification archive does not match the release characterisation"
        )
    with (
        ZipFile(participant_archive) as participant,
        ZipFile(verification_archive) as verification,
    ):
        references = _route_references(verification)
        properties: list[DesignProperty] = []
        comparisons: list[DesignAnalysisComparison] = []
        for profile in release.profiles:
            prefix = f"items/{profile.task_id}"
            adsl = _parquet(participant, f"{prefix}/data/ADSL.parquet")
            adtte = _parquet(participant, f"{prefix}/data/ADTTE.parquet")
            flags = _parquet(
                participant, f"{prefix}/data/subject_operational_flags.parquet"
            )
            task = _json_object(participant, f"{prefix}/task.json")
            protocol = _json_object(participant, f"{prefix}/protocol_summary.json")
            properties.extend(
                _design_properties(profile, adsl, adtte, flags, task, protocol)
            )
            reference_covariates = (
                _parquet(
                    participant,
                    f"{prefix}/data/reference_population_covariates.parquet",
                )
                if profile.design_profile_id == "TE-DP03"
                else None
            )
            comparison = _design_comparison(
                profile,
                adsl,
                adtte,
                task,
                protocol,
                references,
                reference_covariates,
            )
            if comparison is not None:
                comparisons.append(comparison)
    return DesignReleaseCharacterisation(
        release_id=release.release_id,
        participant_archive_sha256=release.participant_archive_sha256,
        catalogue_sha256=release.catalogue_sha256,
        verification_archive_sha256=release.verification_archive_sha256,
        independent_trial_count=release.independent_trial_count,
        properties=tuple(properties),
        comparisons=tuple(comparisons),
    )


def write_design_release(
    output_dir: Path, result: DesignReleaseCharacterisation
) -> None:
    """Write canonical and tidy Design-axis results.

    Parameters
    ----------
    output_dir
        New output directory.
    result
        Validated design characterisation.

    Raises
    ------
    FileExistsError
        If the output directory already exists.
    """

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "design_characterisation.json").write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    _write_models(
        output_dir / "design_properties.csv", DesignProperty, result.properties
    )
    _write_models(
        output_dir / "design_comparisons.csv",
        DesignAnalysisComparison,
        result.comparisons,
    )


def _design_properties(
    profile: TrialProfile,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    flags: pd.DataFrame,
    task: dict[str, object],
    protocol: dict[str, object],
) -> list[DesignProperty]:
    _require_columns(adsl, {"USUBJID", "TRTA", "SITEID"})
    _require_columns(flags, {"USUBJID", "ATTENDANCE_RATE"})
    if (
        len(adsl) != profile.participant_count
        or len(flags) != profile.participant_count
    ):
        raise ValueError(
            f"{profile.task_id} public design tables disagree on participant count"
        )
    arm_counts = adsl.groupby("TRTA", dropna=False).size()
    if bool(pd.isna(arm_counts.index).any()) or len(arm_counts) < 2:
        raise ValueError(
            f"{profile.task_id} requires at least two complete allocation groups"
        )
    expected_fraction = 1.0 / len(arm_counts)
    rows = [
        _property(
            profile,
            "allocation.arm_count",
            len(arm_counts),
            "arms",
            "participant_table",
            "randomized groups",
        ),
        _property(
            profile,
            "allocation.maximum_fraction_deviation",
            float(
                np.max(np.abs(arm_counts.to_numpy() / len(adsl) - expected_fraction))
            ),
            "proportion",
            "participant_table",
            "maximum absolute deviation from equal allocation",
        ),
        _property(
            profile,
            "allocation.site_count",
            adsl["SITEID"].nunique(),
            "sites",
            "participant_table",
            "unique released site or cluster identifiers",
        ),
        _property(
            profile,
            "follow_up.horizon",
            profile.follow_up_horizon_days,
            "days",
            "protocol",
            "declared primary follow-up horizon",
        ),
        _property(
            profile,
            "observation.mean_attendance",
            _finite_mean(flags["ATTENDANCE_RATE"]),
            "proportion",
            "participant_table",
            "mean attended proportion of scheduled visits",
        ),
    ]
    if profile.design_profile_id == "TE-DP01":
        balance = _baseline_randomization_balance(
            adsl,
            seed=int.from_bytes(
                profile.independence_unit_id.encode("utf-8")[:4].ljust(4, b"\0"), "big"
            ),
        )
        rows.extend(
            (
                _property(
                    profile,
                    "allocation.maximum_baseline_smd",
                    balance["observed"],
                    "standardized mean difference",
                    "participant_table",
                    "largest absolute treated-control standardized mean difference across the five "
                    "prespecified baseline prognostic covariates",
                ),
                _property(
                    profile,
                    "allocation.randomization_p95_maximum_baseline_smd",
                    balance["randomization_p95"],
                    "standardized mean difference",
                    "participant_table",
                    "95th percentile after preserving arm counts and randomly reassigning treatment 499 times",
                ),
                _property(
                    profile,
                    "allocation.randomization_percentile",
                    balance["randomization_percentile"],
                    "percentile",
                    "participant_table",
                    "percentile of observed maximum baseline imbalance under arm-count-preserving randomization",
                ),
            )
        )
    elif profile.design_profile_id == "TE-DP02":
        _require_columns(adsl, {"PPFL"})
        _require_columns(
            flags,
            {
                "ANY_NONADHERENT_EX",
                "MEAN_EXADH",
                "N_ICE_RECORDS",
                "ANY_DISCONTINUATION_ICE",
                "ANY_RESCUE_THERAPY_ICE",
                "ANY_TREATMENT_SWITCH_ICE",
            },
        )
        rows.extend(
            (
                _property(
                    profile,
                    "pragmatic.mean_exposure_adherence",
                    _finite_mean(flags["MEAN_EXADH"]),
                    "proportion of prescribed exposure",
                    "participant_table",
                    "mean participant-level proportion of prescribed exposure received",
                ),
                _property(
                    profile,
                    "pragmatic.per_protocol_fraction",
                    float(adsl["PPFL"].astype("string").eq("Y").mean()),
                    "proportion",
                    "participant_table",
                    "participants meeting the released per-protocol flag",
                ),
                _property(
                    profile,
                    "pragmatic.nonadherence_fraction",
                    float(flags["ANY_NONADHERENT_EX"].astype("string").eq("Y").mean()),
                    "proportion",
                    "participant_table",
                    "participants with any nonadherent exposure record",
                ),
                _property(
                    profile,
                    "pragmatic.intercurrent_event_fraction",
                    float(
                        pd.to_numeric(flags["N_ICE_RECORDS"], errors="raise")
                        .gt(0)
                        .mean()
                    ),
                    "proportion",
                    "participant_table",
                    "participants with at least one intercurrent event",
                ),
                _property(
                    profile,
                    "pragmatic.discontinuation_fraction",
                    _yes_fraction(flags["ANY_DISCONTINUATION_ICE"]),
                    "proportion",
                    "participant_table",
                    "participants with a treatment-discontinuation intercurrent event",
                ),
                _property(
                    profile,
                    "pragmatic.rescue_fraction",
                    _yes_fraction(flags["ANY_RESCUE_THERAPY_ICE"]),
                    "proportion",
                    "participant_table",
                    "participants with a rescue-therapy intercurrent event",
                ),
                _property(
                    profile,
                    "pragmatic.switch_fraction",
                    _yes_fraction(flags["ANY_TREATMENT_SWITCH_ICE"]),
                    "proportion",
                    "participant_table",
                    "participants with a treatment-switch intercurrent event",
                ),
            )
        )
        rows.extend(_pragmatic_arm_properties(profile, adsl, flags, task))
    elif profile.design_profile_id == "TE-DP03":
        endpoint = _primary_endpoint(adtte, profile.primary_paramcd)
        frame = adsl[["USUBJID", "AGE", "BMI"]].merge(
            endpoint, on="USUBJID", validate="one_to_one"
        )
        rows.extend(
            (
                _property(
                    profile,
                    "covariate.age_bmi_spearman",
                    _spearman(frame["AGE"], frame["BMI"]),
                    "correlation coefficient",
                    "participant_table",
                    "participant-level age-BMI rank dependence",
                ),
                _property(
                    profile,
                    "covariate.age_event_spearman",
                    _spearman(frame["AGE"], frame["event"]),
                    "correlation coefficient",
                    "endpoint_table",
                    "age and primary-event rank dependence",
                ),
                _property(
                    profile,
                    "covariate.bmi_event_spearman",
                    _spearman(frame["BMI"], frame["event"]),
                    "correlation coefficient",
                    "endpoint_table",
                    "BMI and primary-event rank dependence",
                ),
            )
        )
    elif profile.design_profile_id == "TE-DP04":
        primary = adtte.loc[
            adtte["PARAMCD"].astype("string").eq(profile.primary_paramcd)
        ].copy()
        _require_columns(primary, {"VALIDFL", "OBSEVNT", "ADJEVNT"})
        validated = primary.loc[
            pd.to_numeric(primary["VALIDFL"], errors="raise").eq(1)
        ].copy()
        observed = pd.to_numeric(validated["OBSEVNT"], errors="raise")
        adjudicated = pd.to_numeric(validated["ADJEVNT"], errors="raise")
        positive = adjudicated.eq(1)
        negative = adjudicated.eq(0)
        if not positive.any() or not negative.any():
            raise ValueError(
                f"{profile.task_id} validation sample lacks both endpoint states"
            )
        rows.extend(
            (
                _property(
                    profile,
                    "ascertainment.validation_fraction",
                    len(validated) / len(primary),
                    "proportion",
                    "endpoint_table",
                    "participants with blinded adjudication",
                ),
                _property(
                    profile,
                    "ascertainment.sensitivity",
                    float(observed.loc[positive].eq(1).mean()),
                    "proportion",
                    "endpoint_table",
                    "observed endpoint positive among adjudicated positives",
                ),
                _property(
                    profile,
                    "ascertainment.specificity",
                    float(observed.loc[negative].eq(0).mean()),
                    "proportion",
                    "endpoint_table",
                    "observed endpoint negative among adjudicated negatives",
                ),
                _property(
                    profile,
                    "ascertainment.observed_event_fraction",
                    float(observed.eq(1).mean()),
                    "proportion",
                    "endpoint_table",
                    "validated participants positive under the routinely observed endpoint",
                ),
                _property(
                    profile,
                    "ascertainment.adjudicated_event_fraction",
                    float(adjudicated.eq(1).mean()),
                    "proportion",
                    "endpoint_table",
                    "validated participants positive after blinded adjudication",
                ),
            )
        )
    elif profile.design_profile_id == "TE-DP05":
        rows.extend(_cluster_properties(profile, adsl, adtte))
    elif profile.design_profile_id == "TE-DP06":
        _require_columns(adsl, {"INTERVENTION_START_DY"})
        switch = pd.to_numeric(adsl["INTERVENTION_START_DY"], errors="raise")
        cluster_switch_counts = (
            adsl.assign(_switch=switch).groupby("SITEID")["_switch"].nunique()
        )
        switch_days = sorted(float(value) for value in switch.unique())
        fine_period_length = _stepped_period_length(protocol)
        fine_period_rates = stepped_wedge_period_adjusted_baseline_rates_v1(
            adsl=adsl,
            adtte=adtte,
            paramcd=profile.primary_paramcd,
            tau=profile.follow_up_horizon_days,
            period_length_dy=fine_period_length,
        )
        rollout_period_rates = _aggregate_period_rates(
            fine_rates=fine_period_rates,
            fine_period_length=fine_period_length,
            boundaries=(*switch_days, profile.follow_up_horizon_days),
        )
        rows.extend(
            (
                *_cluster_properties(profile, adsl, adtte),
                _property(
                    profile,
                    "rollout.sequence_count",
                    switch.nunique(),
                    "sequences",
                    "participant_table",
                    "unique intervention adoption days",
                ),
                _property(
                    profile,
                    "rollout.switch_day_span",
                    float(switch.max() - switch.min()),
                    "days",
                    "participant_table",
                    "latest minus earliest intervention adoption day",
                ),
                _property(
                    profile,
                    "rollout.cluster_schedule_purity",
                    float(cluster_switch_counts.eq(1).mean()),
                    "proportion",
                    "participant_table",
                    "clusters assigned exactly one adoption day",
                ),
                _property(
                    profile,
                    "rollout.log_baseline_rate_slope_per_period",
                    _log_rate_slope(rollout_period_rates),
                    "log rate ratio per rollout period",
                    "endpoint_table",
                    "linear change in treatment-adjusted log baseline event rate across rollout periods",
                ),
                *(
                    _property(
                        profile,
                        f"rollout.switch_day.sequence_{sequence}",
                        switch_day,
                        "days since randomization",
                        "participant_table",
                        f"intervention adoption day for rollout sequence {sequence}",
                    )
                    for sequence, switch_day in enumerate(switch_days, start=1)
                ),
                *(
                    _property(
                        profile,
                        f"rollout.event_rate.period_{period}",
                        rate,
                        "events per 1000 participant-days",
                        "endpoint_table",
                        f"treatment-adjusted primary-event baseline rate during calendar period {period}",
                    )
                    for period, rate in enumerate(rollout_period_rates, start=1)
                ),
            )
        )
    elif profile.design_profile_id == "TE-DP07":
        plan = protocol.get("group_sequential_plan")
        if not isinstance(plan, dict):
            raise ValueError(f"{profile.task_id} lacks a public group-sequential plan")
        looks = plan.get("looks")
        alpha = plan.get("nominal_two_sided_alpha_by_look")
        critical = plan.get("z_critical_by_look")
        if (
            not isinstance(looks, list)
            or not isinstance(alpha, list)
            or not isinstance(critical, list)
        ):
            raise ValueError(f"{profile.task_id} has an invalid group-sequential plan")
        look_index = _group_sequential_look_index(plan)
        z_critical = _finite_index(critical, look_index)
        estimate, standard_error = _km_difference(
            adsl=adsl,
            adtte=adtte,
            paramcd=profile.primary_paramcd,
            tau=profile.follow_up_horizon_days,
            control_arm=_required_text(task, "primary_control_arm_id"),
            treated_arm=_required_text(task, "primary_treated_arm_id"),
        )
        observed_z = abs(estimate / standard_error)
        expected_early_stop = look_index < len(looks) - 1 and observed_z >= z_critical
        rows.extend(
            (
                _property(
                    profile,
                    "interim.look_count",
                    len(looks),
                    "looks",
                    "protocol",
                    "planned analyses",
                ),
                _property(
                    profile,
                    "interim.analysis_nominal_alpha",
                    _finite_index(alpha, look_index),
                    "probability",
                    "protocol",
                    "two-sided nominal alpha at the released analysis look",
                ),
                _property(
                    profile,
                    "interim.analysis_critical_z",
                    z_critical,
                    "standard normal units",
                    "protocol",
                    "critical value at the released analysis look",
                ),
                _property(
                    profile,
                    "interim.analysis_information_fraction",
                    _finite_number(plan.get("analysis_information_fraction")),
                    "proportion",
                    "protocol",
                    "planned information fraction at the released analysis",
                ),
                _property(
                    profile,
                    "interim.stopped_early",
                    float(bool(plan.get("stopped_early"))),
                    "indicator",
                    "protocol",
                    "whether the released trial stopped before the final look",
                ),
                _property(
                    profile,
                    "interim.observed_absolute_z",
                    observed_z,
                    "standard normal units",
                    "endpoint_table",
                    "absolute independently recomputed fixed-horizon risk-difference statistic",
                ),
                _property(
                    profile,
                    "interim.boundary_margin",
                    observed_z - z_critical,
                    "standard normal units",
                    "endpoint_table",
                    "observed absolute statistic minus the prespecified efficacy boundary",
                ),
                _property(
                    profile,
                    "interim.early_stop_decision_reproduced",
                    float(expected_early_stop == bool(plan.get("stopped_early"))),
                    "indicator",
                    "endpoint_table",
                    "agreement between the independently recomputed boundary crossing and released early-stop status",
                ),
                *(
                    _property(
                        profile,
                        f"interim.information_fraction.look_{index}",
                        _finite_number(information_fraction),
                        "proportion",
                        "protocol",
                        f"planned information fraction at analysis look {index}",
                    )
                    for index, information_fraction in enumerate(looks, start=1)
                ),
                *(
                    _property(
                        profile,
                        f"interim.critical_z.look_{index}",
                        _finite_number(boundary),
                        "standard normal units",
                        "protocol",
                        f"prespecified two-sided efficacy boundary at analysis look {index}",
                    )
                    for index, boundary in enumerate(critical, start=1)
                ),
            )
        )
    return rows


def _pragmatic_arm_properties(
    profile: TrialProfile,
    adsl: pd.DataFrame,
    flags: pd.DataFrame,
    task: dict[str, object],
) -> list[DesignProperty]:
    arm_ids = {
        "control": _required_text(task, "primary_control_arm_id"),
        "treated": _required_text(task, "primary_treated_arm_id"),
    }
    participant = adsl.loc[:, ["USUBJID", "TRTA", "PPFL"]].merge(
        flags.loc[
            :,
            [
                "USUBJID",
                "MEAN_EXADH",
                "N_ICE_RECORDS",
                "ANY_DISCONTINUATION_ICE",
                "ANY_RESCUE_THERAPY_ICE",
                "ANY_TREATMENT_SWITCH_ICE",
            ],
        ],
        on="USUBJID",
        how="inner",
        validate="one_to_one",
    )
    if len(participant) != len(adsl):
        raise ValueError(
            f"{profile.task_id} operational records do not cover every randomized participant"
        )
    rows: list[DesignProperty] = []
    for label, arm_id in arm_ids.items():
        arm = participant.loc[participant["TRTA"].astype("string").eq(arm_id)]
        if arm.empty:
            raise ValueError(
                f"{profile.task_id} has no participants in declared {label} arm {arm_id!r}"
            )
        values = (
            (
                "mean_exposure_adherence",
                _finite_mean(arm["MEAN_EXADH"]),
                "proportion of prescribed exposure",
                "mean participant-level proportion of prescribed exposure received",
            ),
            (
                "discontinuation_fraction",
                _yes_fraction(arm["ANY_DISCONTINUATION_ICE"]),
                "proportion",
                "participants with a treatment-discontinuation intercurrent event",
            ),
            (
                "rescue_fraction",
                _yes_fraction(arm["ANY_RESCUE_THERAPY_ICE"]),
                "proportion",
                "participants with a rescue-therapy intercurrent event",
            ),
            (
                "switch_fraction",
                _yes_fraction(arm["ANY_TREATMENT_SWITCH_ICE"]),
                "proportion",
                "participants with a treatment-switch intercurrent event",
            ),
            (
                "intercurrent_event_fraction",
                float(pd.to_numeric(arm["N_ICE_RECORDS"], errors="raise").gt(0).mean()),
                "proportion",
                "participants with at least one intercurrent event",
            ),
            (
                "per_protocol_fraction",
                float(arm["PPFL"].astype("string").eq("Y").mean()),
                "proportion",
                "participants meeting the released per-protocol flag",
            ),
        )
        rows.extend(
            _property(
                profile,
                f"pragmatic.{property_name}.{label}",
                estimate,
                unit,
                "participant_table",
                f"{definition} in the {label} arm",
            )
            for property_name, estimate, unit, definition in values
        )
    return rows


def _design_comparison(
    profile: TrialProfile,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    task: dict[str, object],
    protocol: dict[str, object],
    references: tuple[RouteReferenceRecordV1, ...],
    reference_covariates: pd.DataFrame | None,
) -> DesignAnalysisComparison | None:
    included_tiers = {
        "TE-DP03": {"A1"},
        "TE-DP04": {"A3"},
        "TE-DP05": {"A1"},
        "TE-DP06": {"A1", "A2"},
        "TE-DP07": {"A4"},
    }
    if profile.design_profile_id not in included_tiers:
        return None
    if profile.assumption_tier not in included_tiers[profile.design_profile_id]:
        return None
    reference = _primary_reference(profile, references)
    if reference.value is None or reference.ci_low is None or reference.ci_high is None:
        raise ValueError(
            f"{profile.task_id} primary design route lacks a point and interval"
        )
    qualified_estimate = float(reference.value)
    qualified_low = float(reference.ci_low)
    qualified_high = float(reference.ci_high)
    if profile.design_profile_id == "TE-DP03":
        if reference_covariates is None:
            raise ValueError(
                f"{profile.task_id} lacks its reference-population covariates"
            )
        covariate_columns = (
            "USUBJID",
            *TRIALEVAL_STANDARDIZED_RISK_BASELINE_COVARIATES_V1,
        )
        _require_columns(adsl, set(covariate_columns))
        qualified_estimate, qualified_se = (
            cox_linear_standardized_risk_difference_tau_reference_with_uncertainty_v1(
                adsl=adsl,
                adtte=adtte,
                analysis_covariates=cast(pd.DataFrame, adsl.loc[:, covariate_columns]),
                reference_covariates=reference_covariates,
                paramcd=profile.primary_paramcd,
                tau=profile.follow_up_horizon_days,
                control_arm_id=_required_text(task, "primary_control_arm_id"),
                treated_arm_id=_required_text(task, "primary_treated_arm_id"),
            )
        )
        qualified_low = qualified_estimate - _Z_95 * qualified_se
        qualified_high = qualified_estimate + _Z_95 * qualified_se
        naive, naive_se = _km_difference(
            adsl=adsl,
            adtte=adtte,
            paramcd=profile.primary_paramcd,
            tau=profile.follow_up_horizon_days,
            control_arm=_required_text(task, "primary_control_arm_id"),
            treated_arm=_required_text(task, "primary_treated_arm_id"),
        )
        comparison_id = "covariate_adjusted_vs_unadjusted"
        qualified_method = (
            "Cox standardization to participants with baseline BMI >= 35 kg/m^2"
        )
        naive_method = "Unadjusted Kaplan-Meier risk difference"
        independent_unit = "randomized participant"
    elif profile.design_profile_id == "TE-DP04":
        naive, naive_se = _km_difference(
            adsl=adsl,
            adtte=adtte,
            paramcd=profile.primary_paramcd,
            tau=profile.follow_up_horizon_days,
            control_arm=_required_text(task, "primary_control_arm_id"),
            treated_arm=_required_text(task, "primary_treated_arm_id"),
        )
        comparison_id = "validation_corrected_vs_observed"
        qualified_method = profile.primary_method_id
        naive_method = (
            "Kaplan-Meier risk difference using the routinely observed endpoint"
        )
        independent_unit = "randomized participant"
    elif profile.design_profile_id == "TE-DP05":
        naive, naive_se = _km_difference(
            adsl=adsl,
            adtte=adtte,
            paramcd=profile.primary_paramcd,
            tau=profile.follow_up_horizon_days,
            control_arm=_required_text(task, "primary_control_arm_id"),
            treated_arm=_required_text(task, "primary_treated_arm_id"),
        )
        comparison_id = "cluster_robust_vs_independent"
        qualified_method = profile.primary_method_id
        naive_method = "Kaplan-Meier with participant-independent Greenwood uncertainty"
        independent_unit = (
            "randomized cluster for qualified analysis; participant for naive analysis"
        )
    elif profile.design_profile_id == "TE-DP06":
        naive, naive_se = (
            stepped_wedge_unadjusted_risk_difference_tau_with_uncertainty_v1(
                adsl=adsl,
                adtte=adtte,
                paramcd=profile.primary_paramcd,
                tau=profile.follow_up_horizon_days,
                period_length_dy=_stepped_period_length(protocol),
            )
        )
        comparison_id = "period_adjusted_vs_unadjusted"
        qualified_method = profile.primary_method_id
        naive_method = "time-varying Poisson model without calendar-period adjustment"
        independent_unit = "randomized cluster"
    else:
        plan = protocol.get("group_sequential_plan")
        if not isinstance(plan, dict):
            raise ValueError(f"{profile.task_id} lacks a public group-sequential plan")
        critical = plan.get("z_critical_by_look")
        if not isinstance(critical, list):
            raise ValueError(
                f"{profile.task_id} has invalid group-sequential critical values"
            )
        naive, naive_se = _km_difference(
            adsl=adsl,
            adtte=adtte,
            paramcd=profile.primary_paramcd,
            tau=profile.follow_up_horizon_days,
            control_arm=_required_text(task, "primary_control_arm_id"),
            treated_arm=_required_text(task, "primary_treated_arm_id"),
        )
        comparison_id = "group_sequential_vs_fixed"
        qualified_method = profile.primary_method_id
        naive_method = (
            "fixed-analysis normal interval without alpha-spending adjustment"
        )
        independent_unit = "trial at the released information fraction"
    return DesignAnalysisComparison(
        task_id=profile.task_id,
        independence_unit_id=profile.independence_unit_id,
        matched_set_id=profile.matched_set_id,
        design_profile_id=profile.design_profile_id,
        design_tier=cast(str, profile.design_tier),
        assumption_tier=profile.assumption_tier,
        comparison_id=comparison_id,
        qualified_method=qualified_method,
        qualified_estimate=qualified_estimate,
        qualified_interval_low=qualified_low,
        qualified_interval_high=qualified_high,
        naive_method=naive_method,
        naive_estimate=naive,
        naive_interval_low=naive - _Z_95 * naive_se,
        naive_interval_high=naive + _Z_95 * naive_se,
        unit=profile.primary_result_unit,
        independent_unit=independent_unit,
    )


def _cluster_properties(
    profile: TrialProfile,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
) -> tuple[DesignProperty, ...]:
    cluster_sizes = adsl.groupby("SITEID").size().astype(float)
    cluster_arm_counts = adsl.groupby("SITEID")["TRTA"].nunique()
    endpoint = _primary_endpoint(adtte, profile.primary_paramcd)
    clustered = adsl[["USUBJID", "SITEID", "TRTA"]].merge(
        endpoint[["USUBJID", "event"]],
        on="USUBJID",
        validate="one_to_one",
    )
    cluster_information = one_way_cluster_information(
        clustered["event"],
        clustered["SITEID"],
        strata=clustered["TRTA"],
    )
    return (
        _property(
            profile,
            "cluster.count",
            len(cluster_sizes),
            "clusters",
            "participant_table",
            "unique randomized cluster identifiers",
        ),
        _property(
            profile,
            "cluster.size_coefficient_of_variation",
            float(cluster_sizes.std(ddof=1) / cluster_sizes.mean()),
            "ratio",
            "participant_table",
            "between-cluster standard deviation divided by mean cluster size",
        ),
        _property(
            profile,
            "cluster.arm_purity",
            float(cluster_arm_counts.eq(1).mean()),
            "proportion",
            "participant_table",
            "clusters assigned to exactly one randomized group",
        ),
        _property(
            profile,
            "cluster.event_intraclass_correlation",
            cluster_information.intraclass_correlation,
            "correlation coefficient",
            "endpoint_table",
            "one-way random-effects correlation of arm-residualized primary events within cluster",
        ),
        _property(
            profile,
            "cluster.effective_cluster_size",
            cluster_information.effective_cluster_size,
            "participants",
            "participant_table",
            "unequal-size effective cluster size in the one-way variance decomposition",
        ),
        _property(
            profile,
            "cluster.design_effect",
            cluster_information.variance_inflation,
            "ratio",
            "endpoint_table",
            "variance inflation implied by effective cluster size and event intraclass correlation",
        ),
        _property(
            profile,
            "cluster.information_loss_fraction",
            cluster_information.information_loss_fraction,
            "proportion",
            "endpoint_table",
            "fractional information loss implied by the cluster design effect",
        ),
    )


def _km_difference(
    *,
    adsl: pd.DataFrame,
    adtte: pd.DataFrame,
    paramcd: str,
    tau: float,
    control_arm: str,
    treated_arm: str,
) -> tuple[float, float]:
    endpoint = _primary_endpoint(adtte, paramcd)
    frame = adsl[["USUBJID", "TRTA"]].merge(
        endpoint, on="USUBJID", validate="one_to_one"
    )
    estimates: dict[str, tuple[float, float]] = {}
    for arm in (control_arm, treated_arm):
        rows = frame.loc[frame["TRTA"].astype("string").eq(arm)]
        if rows.empty:
            raise ValueError(f"Kaplan-Meier comparator lacks arm {arm!r}")
        fit = SurvfuncRight(
            rows["time"].to_numpy(dtype=float), rows["event"].to_numpy(dtype=int)
        )
        index = int(np.searchsorted(fit.surv_times, tau, side="right") - 1)
        survival = 1.0 if index < 0 else float(fit.surv_prob[index])
        standard_error = 0.0 if index < 0 else float(fit.surv_prob_se[index])
        estimates[arm] = (1.0 - survival, standard_error)
    value = estimates[treated_arm][0] - estimates[control_arm][0]
    standard_error = math.sqrt(
        estimates[treated_arm][1] ** 2 + estimates[control_arm][1] ** 2
    )
    if (
        not math.isfinite(value)
        or not math.isfinite(standard_error)
        or standard_error <= 0
    ):
        raise ValueError("Kaplan-Meier design comparator produced an invalid result")
    return value, standard_error


def _baseline_randomization_balance(
    adsl: pd.DataFrame, *, seed: int
) -> dict[str, float]:
    covariates = TRIALEVAL_STANDARDIZED_RISK_BASELINE_COVARIATES_V1
    _require_columns(adsl, {"TRTA", *covariates})
    treatment = adsl["TRTA"].astype("string")
    arms = tuple(sorted(str(value) for value in treatment.unique()))
    if len(arms) != 2:
        raise ValueError("baseline balance requires exactly two randomized groups")
    values = (
        adsl.loc[:, covariates]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=float)
    )
    if not np.isfinite(values).all():
        raise ValueError("baseline balance requires finite age and BMI")
    group_size = int(treatment.eq(arms[0]).sum())
    if group_size < 2 or len(values) - group_size < 2:
        raise ValueError(
            "baseline balance requires at least two participants per randomized group"
        )

    def _maximum_smd(left_index: npt.NDArray[np.bool_]) -> float:
        left = values[left_index]
        right = values[np.logical_not(left_index)]
        pooled_variance = (
            (len(left) - 1) * left.var(axis=0, ddof=1)
            + (len(right) - 1) * right.var(axis=0, ddof=1)
        ) / (len(values) - 2)
        if not np.isfinite(pooled_variance).all() or np.any(pooled_variance <= 0):
            raise ValueError("baseline variable has invalid pooled variance")
        return float(
            np.max(
                np.abs(right.mean(axis=0) - left.mean(axis=0))
                / np.sqrt(pooled_variance)
            )
        )

    observed = _maximum_smd(treatment.eq(arms[0]).to_numpy(dtype=bool))
    rng = np.random.default_rng(seed)
    randomized = np.empty(499, dtype=float)
    for replicate in range(len(randomized)):
        membership = np.zeros(len(values), dtype=bool)
        membership[rng.permutation(len(values))[:group_size]] = True
        randomized[replicate] = _maximum_smd(membership)
    return {
        "observed": observed,
        "randomization_p95": float(np.quantile(randomized, 0.95)),
        "randomization_percentile": float(
            (np.count_nonzero(randomized <= observed) + 1) / (len(randomized) + 1)
        ),
    }


def _group_sequential_look_index(plan: dict[str, object]) -> int:
    value = plan.get("analysis_look_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("group-sequential analysis look index is invalid")
    return value


def _aggregate_period_rates(
    *,
    fine_rates: tuple[float, ...],
    fine_period_length: float,
    boundaries: tuple[float, ...],
) -> tuple[float, ...]:
    if len(boundaries) < 3 or boundaries[0] != 0.0:
        raise ValueError(
            "rollout-period rates require at least two periods beginning at day zero"
        )
    if any(
        right <= left
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        raise ValueError("rollout-period boundaries must be strictly increasing")
    coverage = len(fine_rates) * fine_period_length
    if coverage + 1e-9 < boundaries[-1]:
        raise ValueError("fine calendar-period rates do not cover the rollout horizon")
    aggregated: list[float] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        weighted_rate = 0.0
        observed_time = 0.0
        for index, rate in enumerate(fine_rates):
            fine_left = index * fine_period_length
            fine_right = fine_left + fine_period_length
            overlap = max(0.0, min(right, fine_right) - max(left, fine_left))
            weighted_rate += overlap * rate
            observed_time += overlap
        if observed_time <= 0.0:
            raise ValueError("rollout period has no fitted calendar time")
        aggregated.append(weighted_rate / observed_time)
    return tuple(aggregated)


def _log_rate_slope(rates: tuple[float, ...]) -> float:
    values = np.asarray(rates, dtype=float)
    if values.size < 2 or not np.isfinite(values).all() or bool((values <= 0.0).any()):
        raise ValueError(
            "rollout log-rate slope requires at least two positive finite rates"
        )
    result = stats.linregress(
        np.arange(1, values.size + 1, dtype=float), np.log(values)
    )
    if not math.isfinite(float(result.slope)):
        raise ValueError("rollout log-rate slope is not estimable")
    return float(result.slope)


def _finite_index(values: list[object], index: int) -> float:
    if index >= len(values):
        raise ValueError("group-sequential analysis look exceeds the declared plan")
    return _finite_number(values[index])


def _yes_fraction(values: pd.Series) -> float:
    return float(values.astype("string").fillna("").eq("Y").mean())


def _primary_endpoint(adtte: pd.DataFrame, paramcd: str) -> pd.DataFrame:
    _require_columns(adtte, {"USUBJID", "PARAMCD", "AVAL", "CNSR"})
    rows = adtte.loc[
        adtte["PARAMCD"].astype("string").eq(paramcd), ["USUBJID", "AVAL", "CNSR"]
    ].copy()
    if rows.empty or rows["USUBJID"].duplicated().any():
        raise ValueError(
            f"primary endpoint {paramcd!r} must have one row per participant"
        )
    rows["time"] = pd.to_numeric(rows.pop("AVAL"), errors="raise")
    rows["event"] = 1 - pd.to_numeric(rows.pop("CNSR"), errors="raise")
    if not rows["event"].isin((0, 1)).all():
        raise ValueError("primary endpoint event indicator must be binary")
    return rows


def _primary_reference(
    profile: TrialProfile,
    references: tuple[RouteReferenceRecordV1, ...],
) -> RouteReferenceRecordV1:
    matches = [
        row
        for row in references
        if row.task_id == profile.task_id
        and row.estimator_method_id == profile.primary_method_id
        and row.effect_scale == profile.primary_effect_scale
        and row.answer_shape == "point"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{profile.task_id} requires exactly one primary route reference"
        )
    return matches[0]


def _route_references(archive: ZipFile) -> tuple[RouteReferenceRecordV1, ...]:
    member = resolve_public_member_v1(archive, "grader/domains/route_references.jsonl")
    return tuple(
        RouteReferenceRecordV1.model_validate_json(line)
        for line in archive.read(member).splitlines()
        if line.strip()
    )


def _json_object(archive: ZipFile, member: str) -> dict[str, object]:
    resolved = resolve_public_member_v1(archive, member)
    payload = json.loads(archive.read(resolved))
    if not isinstance(payload, dict):
        raise ValueError(f"public member is not a JSON object: {member}")
    return payload


def _parquet(archive: ZipFile, member: str) -> pd.DataFrame:
    resolved = resolve_public_member_v1(archive, member)
    return cast(pd.DataFrame, pd.read_parquet(BytesIO(archive.read(resolved))))


def _property(
    profile: TrialProfile,
    property_id: str,
    estimate: float,
    unit: str,
    source: str,
    definition: str,
) -> DesignProperty:
    return DesignProperty(
        task_id=profile.task_id,
        independence_unit_id=profile.independence_unit_id,
        matched_set_id=profile.matched_set_id,
        design_profile_id=profile.design_profile_id,
        design_tier=profile.design_tier,
        assumption_tier=profile.assumption_tier,
        property_id=property_id,
        estimate=float(estimate),
        unit=unit,
        source=source,
        definition=definition,
    )


def _write_models(
    path: Path,
    model: type[DesignProperty] | type[DesignAnalysisComparison],
    rows: tuple[DesignProperty | DesignAnalysisComparison, ...],
) -> None:
    fieldnames = tuple(model.model_fields)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if not isinstance(row, model):
                raise TypeError(
                    f"cannot write {type(row).__name__} with {model.__name__}"
                )
            writer.writerow(row.model_dump(mode="json"))


def _require_columns(data: pd.DataFrame, required: set[str]) -> None:
    if missing := sorted(required - set(data.columns)):
        raise ValueError(f"public design table lacks columns: {missing!r}")


def _finite_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("design property requires finite values")
    return float(np.mean(numeric))


def _spearman(left: pd.Series, right: pd.Series) -> float:
    result = stats.spearmanr(
        pd.to_numeric(left, errors="raise").to_numpy(dtype=float),
        pd.to_numeric(right, errors="raise").to_numpy(dtype=float),
    )
    value = float(result.statistic)
    if not math.isfinite(value):
        raise ValueError("design rank correlation is not estimable")
    return value


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"public design record requires string field {key!r}")
    return value


def _finite_number(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError("public design record requires a finite number")
    return float(value)


def _stepped_period_length(protocol: dict[str, object]) -> float:
    plan = protocol.get("stepped_wedge_plan")
    if not isinstance(plan, dict):
        raise ValueError("stepped-wedge protocol lacks a plan")
    return _finite_number(plan.get("period_length_dy"))


__all__ = ["characterise_design_release", "write_design_release"]
