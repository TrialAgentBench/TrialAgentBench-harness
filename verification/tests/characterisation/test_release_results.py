"""Checks for the packaged TrialEval release characterisation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from scipy import stats

from trialagentbench_validation.characterisation import (
    AssumptionReleaseCharacterisation,
    DesignReleaseCharacterisation,
    MatchedAssumptionDesign,
    ReleaseCharacterisation,
)
from trialagentbench_validation.external.sources.cdisc import CDISCReferenceEvidenceV1


def _result_root() -> Path:
    return Path(__file__).resolve().parents[2] / "validation_results"


def test_release_characterisation_covers_each_independent_trial_once() -> None:
    """The packaged census distinguishes independent trials from context views."""

    root = _result_root()
    result = ReleaseCharacterisation.model_validate_json(
        (root / "data/release_characterisation.json").read_text(encoding="utf-8")
    )
    profiles = pd.read_csv(root / "data/programme_profiles.csv")
    assert (
        result.independent_trial_count == len(result.profiles) == len(profiles) == 100
    )
    assert {profile.design_subtype for profile in result.profiles} == {
        "individual_randomized",
        "pragmatic",
        "covariate_structure",
        "endpoint_ascertainment",
        "cluster_parallel",
        "stepped_wedge",
        "group_sequential",
    }
    assert result.context_view_count == 500
    assert profiles["independence_unit_id"].is_unique
    assert set(profiles["design_profile_id"]) == {
        f"TE-DP0{index}" for index in range(1, 8)
    }
    assert int(profiles["participant_count"].sum()) == 610_190


def test_worked_trial_data_reconcile_with_analysis_lineage() -> None:
    """The participant export and public analysis record describe the same trial."""

    root = _result_root()
    participants = pd.read_csv(root / "data/worked_trial_participants.csv")
    lineage = json.loads(
        (root / "data/worked_trial_lineage.json").read_text(encoding="utf-8")
    )
    assert (
        len(participants)
        == lineage["participant_rows"]
        == lineage["linked_rows"]
        == 7_691
    )
    assert participants["participant_id"].is_unique
    assert set(participants["event"]) == {0, 1}
    assert int(participants["event"].sum()) == 447
    assert lineage["estimate"] == pytest.approx(-0.016520535924455126)
    assert lineage["interval_low"] < lineage["estimate"] < lineage["interval_high"] < 0


def test_design_characterisation_covers_every_profile_from_public_records() -> None:
    """The packaged design census exposes all designs and paired complex analyses."""

    root = _result_root()
    result = DesignReleaseCharacterisation.model_validate_json(
        (root / "data/design_characterisation.json").read_text(encoding="utf-8")
    )
    properties = pd.read_csv(root / "data/design_properties.csv")
    comparisons = pd.read_csv(root / "data/design_comparisons.csv")
    assert result.independent_trial_count == 100
    assert len(result.properties) == len(properties) == 1_436
    assert len(result.comparisons) == len(comparisons) == 24
    assert set(properties["design_profile_id"]) == {
        f"TE-DP0{index}" for index in range(1, 8)
    }
    assert set(comparisons["design_profile_id"]) == {
        "TE-DP03",
        "TE-DP04",
        "TE-DP05",
        "TE-DP06",
        "TE-DP07",
    }
    balance = properties.loc[
        properties["property_id"].isin(
            (
                "allocation.maximum_baseline_smd",
                "allocation.randomization_p95_maximum_baseline_smd",
            )
        )
    ].pivot(index="independence_unit_id", columns="property_id", values="estimate")
    assert len(balance) == 24
    assert (
        balance["allocation.maximum_baseline_smd"]
        <= balance["allocation.randomization_p95_maximum_baseline_smd"]
    ).sum() == 22
    pragmatic_by_arm = properties.loc[
        properties["property_id"].str.match(
            r"pragmatic\.(mean_exposure_adherence|discontinuation_fraction|rescue_fraction|"
            r"switch_fraction|intercurrent_event_fraction|per_protocol_fraction)\.(control|treated)"
        )
    ]
    assert len(pragmatic_by_arm) == 24 * 6 * 2
    pragmatic = pragmatic_by_arm.pivot(
        index="independence_unit_id",
        columns="property_id",
        values="estimate",
    )
    assert pragmatic["pragmatic.switch_fraction.control"].eq(0).all()
    assert (
        pragmatic["pragmatic.intercurrent_event_fraction.treated"].median()
        > pragmatic["pragmatic.intercurrent_event_fraction.control"].median()
    )
    decisions = properties.loc[
        properties["property_id"].eq("interim.early_stop_decision_reproduced"),
        "estimate",
    ]
    assert len(decisions) == 4
    assert decisions.eq(1).all()
    period_slopes = properties.loc[
        properties["property_id"].eq("rollout.log_baseline_rate_slope_per_period"),
        ["assumption_tier", "estimate"],
    ]
    assert len(period_slopes) == 8
    assert (
        period_slopes.loc[period_slopes["assumption_tier"].eq("A2"), "estimate"]
        .gt(0.0)
        .all()
    )
    assert (
        period_slopes.loc[period_slopes["assumption_tier"].eq("A2"), "estimate"].mean()
        > period_slopes.loc[
            period_slopes["assumption_tier"].eq("A1"), "estimate"
        ].mean()
    )


def test_assumption_characterisation_exposes_ordered_mechanisms_and_analysis_response() -> (
    None
):
    """The finite release exposes every assumption series through public analyses."""

    root = _result_root()
    result = AssumptionReleaseCharacterisation.model_validate_json(
        (root / "data/assumption_characterisation.json").read_text(encoding="utf-8")
    )
    bridges = pd.read_csv(root / "data/assumption_bridges.csv")
    summaries = pd.read_csv(root / "data/assumption_summaries.csv")

    assert result.release_id == "trialagentbench-collaborator-single-seed-016"
    assert result.analysis_count == len(result.bridges) == len(bridges) == 100
    assert len(result.summaries) == len(summaries) == 25
    assert not bridges["analysis_failure"].any()
    assert bridges["qualified_replay_abs_error"].max() < 1e-12

    expected_tiers = {
        "TE-S01": ("A1", "A2", "A3"),
        "TE-S02": ("A1", "A2", "A3"),
        "TE-S03": ("A1", "A2"),
        "TE-S04": ("A1", "A2", "A3", "A4"),
        "TE-S05": ("A1", "A2", "A3"),
        "TE-S06": ("A1", "A2", "A3", "A4"),
        "TE-S07": ("A1", "A2", "A3"),
        "TE-S08": ("A1", "A2"),
        "TE-S09": ("A4",),
    }
    actual_tiers = (
        summaries.groupby("series_id", sort=False)["assumption_tier"]
        .apply(tuple)
        .to_dict()
    )
    assert actual_tiers == expected_tiers

    numeric = summaries.dropna(subset=["mechanism_value_mean"])
    graded_tiers = {
        series_id: tuple(tier for tier in tiers if tier != "A4")
        for series_id, tiers in expected_tiers.items()
        if tiers[0] == "A1"
    }
    for series_id, tiers in graded_tiers.items():
        values = numeric.loc[
            numeric["series_id"].eq(series_id) & numeric["assumption_tier"].isin(tiers),
            "mechanism_value_mean",
        ].tolist()
        assert len(values) == len(tiers)
        assert values == sorted(values)
        assert len(set(values)) == len(values)

    treatment_policy = bridges["series_id"].eq("TE-S03")
    assert bridges.loc[treatment_policy, "absolute_analysis_difference"].eq(0).all()
    incompatible = bridges["assumption_tier"].eq("A4")
    assert set(bridges.loc[incompatible, "series_id"]) == {"TE-S04", "TE-S06", "TE-S09"}
    assert bridges.loc[incompatible, "default_status"].eq("incompatible").all()


def test_matched_assumption_response_is_paired_complete_and_directional() -> None:
    """Matched worlds isolate graded mechanisms from random trial variation."""

    root = _result_root()
    response = root / "data/assumption_response"
    design = MatchedAssumptionDesign.model_validate_json(
        (response / "matched_assumption_design.json").read_text(encoding="utf-8")
    )
    bridges = pd.read_csv(response / "assumption_bridges.csv")
    summaries = pd.read_csv(response / "assumption_summaries.csv")
    contrasts = pd.read_csv(response / "assumption_paired_contrasts.csv")
    figure = pd.read_csv(root / "figures/assumption_response.csv")

    assert design.analysis_count == len(bridges) == 88
    assert len(design.identities) == 32
    assert len(summaries) == len(figure) == 22
    assert len(contrasts) == 14
    assert summaries.groupby("series_id", sort=False)["assumption_tier"].apply(
        tuple
    ).to_dict() == {
        "TE-S01": ("A1", "A2", "A3"),
        "TE-S02": ("A1", "A2", "A3"),
        "TE-S03": ("A1", "A2"),
        "TE-S04": ("A1", "A2", "A3"),
        "TE-S05": ("A1", "A2", "A3"),
        "TE-S06": ("A1", "A2", "A3"),
        "TE-S07": ("A1", "A2", "A3"),
        "TE-S08": ("A1", "A2"),
    }
    assert contrasts["trial_pair_count"].eq(4).all()
    assert contrasts["mechanism_change_mean"].gt(0).all()
    assert contrasts["mechanism_change_interval_low"].gt(0).all()
    absolute_gap_changes = []
    for _, group in bridges.groupby("series_id", sort=False):
        tier_means = (
            group.groupby("assumption_tier", sort=True)["absolute_analysis_difference"]
            .mean()
            .to_numpy()
        )
        absolute_gap_changes.extend(
            upper - lower
            for lower, upper in zip(tier_means, tier_means[1:], strict=False)
        )
    assert sum(change > 0 for change in absolute_gap_changes) == 13
    assert sum(change == 0 for change in absolute_gap_changes) == 1
    assert not bridges["analysis_failure"].any()
    assert bridges["qualified_replay_abs_error"].max() < 1e-12

    consequence_orientation = {
        "TE-S01": -1.0,
        "TE-S02": 1.0,
        "TE-S04": -1.0,
        "TE-S05": -1.0,
        "TE-S06": 1.0,
        "TE-S07": -1.0,
        "TE-S08": 1.0,
    }
    nonadherence_excluded = bridges.loc[bridges["series_id"].ne("TE-S03")].copy()
    nonadherence_excluded["expected_consequence"] = nonadherence_excluded[
        "series_id"
    ].map(consequence_orientation) * (
        nonadherence_excluded["default_value"]
        - nonadherence_excluded["qualified_value"]
    )
    expected_means = nonadherence_excluded.groupby(
        ["series_id", "assumption_tier"],
        sort=False,
    )["expected_consequence"].mean()
    observed_means = figure.set_index(["series_id", "assumption_tier"])[
        "consequence_value_mean"
    ]
    assert observed_means.loc[expected_means.index].to_numpy() == pytest.approx(
        expected_means.to_numpy()
    )
    nonadherence = bridges.loc[bridges["series_id"].eq("TE-S03")].copy()
    reference_effect = (
        nonadherence.loc[nonadherence["assumption_tier"].eq("A1")]
        .set_index("replicate_index")["default_value"]
        .abs()
    )
    nonadherence["expected_consequence"] = (
        nonadherence["replicate_index"].map(reference_effect)
        - nonadherence["default_value"].abs()
    )
    expected_nonadherence = nonadherence.groupby(
        ["series_id", "assumption_tier"],
        sort=False,
    )["expected_consequence"].mean()
    assert observed_means.loc[expected_nonadherence.index].to_numpy() == pytest.approx(
        expected_nonadherence.to_numpy()
    )
    assert figure["mechanism_value_mean"].to_numpy() == pytest.approx(
        summaries["mechanism_value_mean"].to_numpy()
    )
    assert figure["consequence_unit"].tolist() == summaries["result_unit"].tolist()


def test_context_reconstruction_and_standards_results_cover_the_complete_workflow() -> (
    None
):
    """Matched contexts, public replay, repair, and standards checks reconcile."""

    root = _result_root()
    contexts = pd.read_csv(root / "data/context_invariance.csv")
    routes = pd.read_csv(root / "data/context_route_recovery.csv")
    integrity = pd.read_csv(root / "data/context_integrity.csv")
    standards = CDISCReferenceEvidenceV1.model_validate_json(
        (root / "data/cdisc_reference_evidence.json").read_text(encoding="utf-8")
    )

    assert len(contexts) == 100
    assert contexts["status"].eq("pass").all()
    assert contexts["context_count"].eq(5).all()
    assert contexts["generation_seed_count"].eq(1).all()
    assert contexts["estimand_count"].eq(1).all()
    assert contexts["analysis_ready_hash_count_c1_c2"].eq(1).all()
    assert contexts["raw_domain_hash_count_c3_c4"].eq(1).all()

    assert len(routes) == 692
    assert routes["status"].eq("pass").all()
    assert routes.groupby("context_or_checkpoint_id").size().to_dict() == {
        "C1": 100,
        "C2": 164,
        "C3": 100,
        "C4": 164,
        "C5": 164,
    }
    assert routes["maximum_absolute_difference"].max() < 1e-12
    assert integrity.to_dict("records") == [
        {
            "condition": "C5 exact transport duplication",
            "required": 100,
            "repaired": 100,
            "mismatched": 0,
            "unsupported": 0,
            "status": "pass",
        }
    ]

    assert standards.rows_compared == 7_942
    assert standards.cells_compared == 179_903
    assert standards.mismatched_cells == 0
    assert standards.key_violations == standards.subject_reference_violations == 0
    assert (
        standards.define_xml_datasets_present
        == standards.define_xml_datasets_required
        == 8
    )
    assert standards.analysis_absolute_difference == 0
    assert (
        standards.negative_controls_detected == standards.negative_controls_total == 5
    )


def test_external_construct_comparison_retains_all_prespecified_results() -> None:
    """The overview reports favorable and imprecise constructs on one scale."""

    data = pd.read_csv(_result_root() / "figures/generator_realism.csv")
    assert len(data) == 8
    assert set(data["validation_domain"]) == {
        "Marginal distributions",
        "Joint structure",
        "Analysis impact",
    }
    assert data["relative_distance"].lt(1).all()
    assert (
        data.loc[
            data["validation_domain"].eq("Marginal distributions"), "interval_high"
        ]
        .lt(1)
        .all()
    )
    age_bmi = data.loc[data["construct_id"].eq("age_bmi_spearman")].squeeze()
    assert age_bmi["relative_distance"] < 1 < age_bmi["interval_high"]
    assert set(data["external_trials"]) == {10, 29}
    assert data["synthetic_trials"].eq(75).all()


def test_complete_operating_characteristic_tables_are_packaged() -> None:
    """The installed evidence includes complete mechanism-level result tables."""

    root = _result_root() / "data/operating_characteristics"
    expected = {
        "clustered_design/cluster_response_summary.csv",
        "clustered_design/cluster_response_worlds.csv",
        "competing_risks/cause_specific_recovery.csv",
        "competing_risks/competing_risk_information_response.csv",
        "competing_risks/competing_risk_response.csv",
        "confounding/confounding_dose_response.csv",
        "confounding/confounding_failures.csv",
        "confounding/confounding_information_response.csv",
        "confounding/confounding_operating_characteristics.csv",
        "confounding/overlap_response.csv",
        "cross_domain_linkage/linkage_response.csv",
        "cross_domain_linkage/portfolio_response.csv",
        "group_sequential/group_sequential_operating_characteristics.csv",
        "group_sequential/group_sequential_response.csv",
        "stepped_wedge/stepped_wedge_response_summary.csv",
        "stepped_wedge/stepped_wedge_response_worlds.csv",
        "longitudinal/longitudinal_linkage_curve.csv",
        "longitudinal/longitudinal_source_anchored_recovery.csv",
        "longitudinal/longitudinal_treatment_recovery.csv",
        "longitudinal/tereco_linkage_response.csv",
        "longitudinal/tereco_treatment_recovery.csv",
        "observation_process/dropout_recovery.csv",
        "observation_process/dropout_response.csv",
        "observation_process/paired_route_contrasts.csv",
        "observation_process/treatment_recovery.csv",
        "observation_process/treatment_response.csv",
        "outcome_replication/headsoar_proportional_odds_dose_recovery.csv",
        "outcome_replication/headsoar_safety_predictive.csv",
        "outcome_replication/patency_cox_dose_recovery.csv",
        "outcome_replication/patency_rmst_predictive.csv",
        "recurrent_events/frailty_operating_characteristics.csv",
        "recurrent_events/frailty_realization.csv",
        "recurrent_events/frailty_realized_process.csv",
        "recurrent_events/frailty_response.csv",
        "treatment_heterogeneity/hte_dose_response.csv",
        "treatment_heterogeneity/hte_information_response.csv",
        "treatment_heterogeneity/hte_operating_characteristics.csv",
    }
    observed = {path.relative_to(root).as_posix() for path in root.rglob("*.csv")}
    assert observed == expected
    for relative_path in sorted(expected):
        frame = pd.read_csv(root / relative_path)
        assert not frame.empty, relative_path
        assert not frame.columns.empty, relative_path


def test_mechanism_response_figure_reconciles_with_operating_characteristics() -> None:
    """The mechanism display preserves source-level estimates and uncertainty."""

    root = _result_root()
    figure = pd.read_csv(root / "figures/mechanism_response.csv")
    operating = root / "data/operating_characteristics"
    assert set(figure["panel"]) == {
        "treatment_heterogeneity",
        "competing_events",
        "confounding",
        "dropout",
        "recurrent_events",
        "cross_domain_linkage",
    }

    heterogeneity = pd.read_csv(
        operating / "treatment_heterogeneity/hte_dose_response.csv"
    )
    binary = heterogeneity.loc[
        heterogeneity["outcome_kind"].eq("binary")
        & heterogeneity["sample_size_multiplier"].eq(1.0),
        "mean_slope",
    ]
    expected_half_width = stats.t.ppf(0.975, len(binary) - 1) * stats.sem(binary)
    plotted = figure.loc[
        figure["panel"].eq("treatment_heterogeneity")
        & figure["setting"].eq("Binary, N")
    ].squeeze()
    assert plotted["estimate"] == pytest.approx(binary.mean())
    assert plotted["interval_low"] == pytest.approx(binary.mean() - expected_half_width)
    assert plotted["interval_high"] == pytest.approx(
        binary.mean() + expected_half_width
    )

    dropout = pd.read_csv(operating / "observation_process/paired_route_contrasts.csv")
    plotted_dropout = figure.loc[figure["panel"].eq("dropout")].sort_values(
        ["series", "dose"]
    )
    expected_dropout = dropout.loc[
        dropout["correction_route"].eq("estimated_ipcw"),
        [
            "lagged_outcome_coefficient",
            "mean_absolute_error_reduction",
            "reduction_ci_low",
            "reduction_ci_high",
        ],
    ].sort_values(["mean_absolute_error_reduction", "lagged_outcome_coefficient"])
    assert len(plotted_dropout) == len(expected_dropout) == 12
    assert sorted(plotted_dropout["estimate"]) == pytest.approx(
        sorted(expected_dropout["mean_absolute_error_reduction"])
    )
    assert (
        plotted_dropout.loc[
            plotted_dropout["series"].eq("Skin barrier")
            & plotted_dropout["dose"].eq(1.0),
            "interval_high",
        ].item()
        > 0
    )

    cross_domain = pd.read_csv(
        operating / "cross_domain_linkage/portfolio_response.csv"
    ).sort_values("mean_slope")
    plotted_cross_domain = figure.loc[
        figure["panel"].eq("cross_domain_linkage")
    ].sort_values("estimate")
    assert plotted_cross_domain["estimate"].to_numpy() == pytest.approx(
        cross_domain["mean_slope"].to_numpy()
    )
    assert plotted_cross_domain["independent_units"].eq(8).all()


def test_tereco_results_link_trajectory_to_analysis_and_joint_dependence() -> None:
    """TERECO evidence covers the randomized analysis and multivariate process."""

    root = _result_root() / "data/operating_characteristics/longitudinal"
    treatment = pd.read_csv(root / "tereco_treatment_recovery.csv")
    linkage = pd.read_csv(root / "tereco_linkage_response.csv").sort_values(
        "linkage_retention"
    )

    assert len(treatment) == 6
    assert treatment["worlds"].eq(200).all()
    assert (treatment["standardized_bias_simultaneous_ci_low"] <= 0).all()
    assert (treatment["standardized_bias_simultaneous_ci_high"] >= 0).all()
    assert treatment["coverage"].between(0.9, 0.97).all()

    assert linkage["linkage_retention"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert linkage["worlds"].eq(200).all()
    assert linkage["correlation_mae_mean"].is_monotonic_decreasing
    production = linkage.iloc[-1]
    assert production["source_bootstrap_low"] <= production["correlation_mae_mean"]
    assert production["correlation_mae_mean"] <= production["source_bootstrap_high"]


def test_cluster_dependence_has_a_graded_repeated_world_response() -> None:
    """Increasing cluster frailty increases event dependence and information loss."""

    root = _result_root() / "data/operating_characteristics/clustered_design"
    summary = pd.read_csv(root / "cluster_response_summary.csv")
    worlds = pd.read_csv(root / "cluster_response_worlds.csv")
    variance = summary.loc[
        summary["measure"].eq("event_variance_inflation")
    ].sort_values("cluster_log_hazard_sd")

    assert len(worlds) == 5_000
    assert worlds.groupby("setting").size().eq(1_000).all()
    assert set(worlds["setting"]) == {
        "zero",
        "low",
        "source_anchored",
        "moderate",
        "strong",
    }
    assert variance["world_count"].eq(1_000).all()
    assert variance["failure_count"].eq(0).all()
    assert variance["mean"].is_monotonic_increasing
    previous_high = variance["ci_high"].to_numpy(dtype=float)[1:-1]
    next_low = variance["ci_low"].to_numpy(dtype=float)[2:]
    assert (previous_high < next_low).all()


def test_report_binds_characterisation_figures_and_source_data() -> None:
    """The public reading path links each result family to exact data."""

    documents = tuple(sorted((_result_root() / "reports").glob("*.md")))
    report = "\n".join(path.read_text(encoding="utf-8") for path in documents)
    required_links = {
        "figures/worked_trial.svg",
        "figures/trial_programme.svg",
        "data/worked_trial_participants.csv",
        "data/worked_trial.csv",
        "data/worked_trial_lineage.json",
        "data/programme_profiles.csv",
        "data/programme_estimates.csv",
        "data/release_characterisation.json",
        "figures/trial_designs.svg",
        "figures/design_consequences.svg",
        "data/design_properties.csv",
        "data/design_comparisons.csv",
        "data/design_characterisation.json",
        "data/operating_characteristics/clustered_design/cluster_response_summary.csv",
        "data/assumption_bridges.csv",
        "data/assumption_summaries.csv",
        "data/assumption_characterisation.json",
        "data/assumption_response/assumption_bridges.csv",
        "data/assumption_response/assumption_summaries.csv",
        "data/assumption_response/assumption_paired_contrasts.csv",
        "data/assumption_response/matched_assumption_design.json",
        "figures/context_workflow.svg",
        "figures/context_workflow.csv",
        "data/context_invariance.csv",
        "data/context_route_recovery.csv",
        "data/context_integrity.csv",
        "data/cdisc_reference_evidence.json",
        "figures/generator_realism.svg",
        "figures/generator_realism.csv",
        "figures/mechanism_response.svg",
        "figures/mechanism_response.csv",
        "data/operating_characteristics/cross_domain_linkage/portfolio_response.csv",
        "data/operating_characteristics/cross_domain_linkage/linkage_response.csv",
        "data/operating_characteristics",
    }
    for relative_path in required_links:
        assert f"](../{relative_path})" in report
