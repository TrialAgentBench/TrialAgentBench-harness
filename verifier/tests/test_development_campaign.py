"""Independent finite-world counterexamples for paired TrialDev comparisons."""

from __future__ import annotations

import pytest

from trialagentbench_verifier.campaign_contracts import CampaignAnalysisV1
from trialagentbench_verifier.development_campaign import compare_trialdev_rankings
from trialagentbench_verifier.response import TrialDevRankingItemV1, TrialDevRunRankingV1

STRATA = (("clinical", "scenario", 3), ("clinical", "scenario", 5))


def ranking(model: str, *, delta: float = 0, fault: bool = False) -> TrialDevRunRankingV1:
    rows = []
    for size, count in ((3, 2), (5, 4)):
        for world in range(count):
            for stream in ("evidence_analysis", "portfolio_management"):
                value = (0.1 * world if size == 3 else 0.5) + delta
                rows.append(
                    TrialDevRankingItemV1(
                        task_id=f"{size}-{world}-{stream}",
                        world_id=f"world-{size}-{world}",
                        stream=stream,
                        clinical_profile_id="clinical",
                        scenario_profile_id="scenario",
                        asset_count=size,
                        execution_status="evaluator_error" if fault and world == 0 else "completed",
                        run_checksum="a" * 64,
                        observational_loss=value,
                        initial_target_statuses={"effect": "reported"},
                    )
                )
    return TrialDevRunRankingV1(
        release_id="release",
        release_manifest_sha256="a" * 64,
        invocation_checksum="b" * 64,
        model_id=model,
        scope="scheduled_panel",
        items=tuple(rows),
        profile_means=(),
    )


def evidence_only(row: TrialDevRunRankingV1) -> TrialDevRunRankingV1:
    return row.model_copy(update={"items": tuple(item for item in row.items if item.stream == "evidence_analysis")})


def test_campaign_profiles_preserve_completion_and_equal_world_weight(tmp_path):
    import csv
    import json

    from trialagentbench_verifier.development_campaign import (
        TrialDevCampaignAnalysisV1,
        render_trialdev_comparisons,
        summarize_trialdev_campaign_profiles,
    )
    from trialagentbench_verifier.public_results import public_trialdev_campaign

    a = evidence_only(ranking("a"))
    incomplete = a.model_copy(
        update={
            "items": (
                a.items[0].model_copy(
                    update={
                        "execution_status": "model_noncompletion",
                        "observational_loss": None,
                        "initial_target_statuses": {"effect": "missing"},
                    }
                ),
                *a.items[1:],
            )
        }
    )
    summaries = summarize_trialdev_campaign_profiles((a, incomplete))
    three = next(row for row in summaries if row.asset_count == 3)
    assert three.scheduled_tasks == 4 and three.scored_tasks == 3
    assert three.scheduled_worlds == three.scored_worlds == 2
    assert three.unscored_task_counts == {"model_noncompletion": 1}
    assert three.target_status_counts == {"reported": 3, "missing": 1}
    assert three.mean is None
    assert three.observed_mean == pytest.approx(0.05)
    complete = next(row for row in summaries if row.asset_count == 5)
    assert complete.mean == complete.observed_mean == pytest.approx(0.5)
    assert complete.scheduled_tasks == 8 and complete.scheduled_worlds == 4
    empty = a.model_copy(
        update={
            "items": tuple(
                row.model_copy(
                    update={
                        "execution_status": "model_noncompletion",
                        "observational_loss": None,
                        "initial_target_statuses": {"effect": "missing"},
                    }
                )
                for row in a.items
            )
        }
    )
    missing = summarize_trialdev_campaign_profiles((empty,))[0].model_dump()
    assert missing["scored_worlds"] == missing["scored_tasks"] == 0
    assert missing["unscored_task_counts"] == {"model_noncompletion": 2}
    assert missing["mean"] is None and missing["observed_mean"] is None
    report = TrialDevCampaignAnalysisV1(
        campaign_plan_sha256="a" * 64,
        release_sha256="b" * 64,
        exports=(),
        comparisons=(),
        profile_summaries=summaries,
    )
    assert public_trialdev_campaign(report, policy=CampaignAnalysisV1()).profile_summaries == summaries
    render_trialdev_comparisons(report, tmp_path / "review")
    with (tmp_path / "review/profiles.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["model_id"] == "a"
    assert json.loads(rows[0]["unscored_task_counts"]) == {"model_noncompletion": 1}
    assert json.loads(rows[0]["target_status_counts"]) == {"reported": 3, "missing": 1}
    assert rows[0]["mean"] == "" and float(rows[0]["observed_mean"]) == pytest.approx(0.05)


def compare(*rows: TrialDevRunRankingV1):
    return compare_trialdev_rankings(tuple(rows), required_strata=STRATA, policy=CampaignAnalysisV1())


def test_paired_covariance_and_unequal_world_counts():
    from pathlib import Path

    from trialagentbench_verifier.development_campaign import TrialDevCampaignAnalysisV1, TrialDevCampaignExportV1
    from trialagentbench_verifier.public_results import public_trialdev_campaign

    a, b = evidence_only(ranking("a", delta=0.1)), evidence_only(ranking("b"))
    row = compare(a, b)[0]
    assert row.estimate == pytest.approx(0.1)
    assert row.confidence_low == pytest.approx(0.1)
    assert row.confidence_high == pytest.approx(0.1)
    assert row.worlds == 6
    assert row.strata == 2
    assert row.adjusted_p_value == pytest.approx(1 / 2001)
    # Duplicating the same within-world observations changes neither support nor uncertainty.
    repeated = compare(a, a, b, b)[0]
    assert repeated == row
    report = TrialDevCampaignAnalysisV1(
        campaign_plan_sha256="a" * 64,
        release_sha256="b" * 64,
        exports=(TrialDevCampaignExportV1(job_id="a", path=Path("/private/custody.zip"), sha256="c" * 64),),
        comparisons=(row,),
    )
    public = public_trialdev_campaign(report, policy=CampaignAnalysisV1())
    assert public.comparisons == (row,)
    assert public.export_sha256 == ("c" * 64,)
    assert "/private" not in public.model_dump_json()


def test_strata_are_equally_weighted_and_ties_do_not_force_ordering():
    a = evidence_only(ranking("a"))
    b = evidence_only(ranking("b"))
    b = b.model_copy(update={"items": tuple(row.model_copy(update={"observational_loss": 0.0}) for row in b.items)})
    assert compare(a, b)[0].estimate == pytest.approx((0.05 + 0.5) / 2)
    tied = compare(a, a.model_copy(update={"model_id": "b"}))[0]
    assert tied.estimate == 0
    assert tied.confidence_low == tied.confidence_high == 0
    assert tied.p_value == tied.adjusted_p_value == 1


def test_faults_missing_strata_and_pending_primaries_are_explicit():
    a, b = evidence_only(ranking("a")), evidence_only(ranking("b", fault=True))
    assert compare(a, b)[0].unavailable_reason == "evaluator_fault"
    assert compare(a, b)[0].estimate is None
    b = evidence_only(ranking("b"))
    shortened = tuple(row.model_copy(update={"items": row.items[:1]}) for row in (a, b))
    assert compare(*shortened)[0].unavailable_reason == "missing_required_stratum"
    with pytest.raises(ValueError, match="same scheduled task panel"):
        compare(a, shortened[1])
    for status in ("not_started", "incomplete_run"):
        pending = b.model_copy(
            update={"items": (b.items[0].model_copy(update={"execution_status": status}), *b.items[1:])}
        )
        with pytest.raises(ValueError, match="terminal primary"):
            compare(a, pending)


def test_model_noncompletion_withholds_loss_and_retains_holm_family():
    rows = []
    for model in ("a", "b", "c"):
        row = evidence_only(ranking(model))
        rows.append(
            row.model_copy(
                update={
                    "items": tuple(
                        item.model_copy(
                            update={
                                "execution_status": "model_noncompletion",
                                "observational_loss": None,
                                "initial_target_statuses": {"effect": "missing"},
                            }
                        )
                        for item in row.items
                    )
                }
            )
        )
    result = compare(*rows)
    recovery = [row for row in result if row.metric == "observational_loss"]
    assert len(recovery) == 3
    assert all(row.estimate is None and row.adjusted_p_value is None for row in recovery)
    assert all(row.unavailable_reason == "model_noncompletion" for row in recovery)
    fabricated = rows[0].model_copy(
        update={"items": tuple(row.model_copy(update={"observational_loss": 0.0}) for row in rows[0].items)}
    )
    with pytest.raises(ValueError, match="uncompleted evidence analysis"):
        compare(fabricated, rows[1])


def test_common_world_sensitivity_preserves_missing_panel_and_equal_strata():
    a, b = evidence_only(ranking("a")), evidence_only(ranking("b"))
    b = b.model_copy(update={"items": tuple(row.model_copy(update={"observational_loss": 0.0}) for row in b.items)})
    a = a.model_copy(
        update={
            "items": (
                a.items[0].model_copy(
                    update={
                        "execution_status": "model_noncompletion",
                        "observational_loss": None,
                        "initial_target_statuses": {"effect": "missing"},
                    }
                ),
                *a.items[1:],
            )
        }
    )
    row = compare(a, b)[0]
    assert row.estimate is None and row.unavailable_reason == "model_noncompletion"
    assert row.worlds == 6 and row.strata == 2
    assert row.common_worlds == 5 and row.common_strata == 2
    assert row.common_world_estimate == pytest.approx((0.1 + 0.5) / 2)
    repeated = compare(a, a, b, b)[0]
    assert repeated == row
    fault = evidence_only(ranking("fault", fault=True))
    blocked = compare(a, fault)[0]
    assert blocked.common_world_estimate is None
    assert blocked.common_worlds == blocked.common_strata == 0


def test_programme_metric_uses_portfolio_only_and_evidence_fault_is_local():
    runs = []
    for model, advantage in (("a", 4.0), ("b", 0.0)):
        run = ranking(model)
        items = []
        for row in run.items:
            if row.stream == "portfolio_management":
                items.append(
                    row.model_copy(
                        update={
                            "authored_programme": True,
                            "net_development_value_usd": advantage + row.asset_count,
                        }
                    )
                )
            else:
                items.append(row.model_copy(update={"execution_status": "evaluator_error"}))
        runs.append(run.model_copy(update={"items": tuple(items)}))
    recovery, programme_recovery, value = compare(*runs)
    assert recovery.unavailable_reason == "evaluator_fault"
    assert programme_recovery.unavailable_reason is None
    assert programme_recovery.estimate == 0
    assert value.estimate == value.confidence_low == value.confidence_high == 4
    assert value.unavailable_reason is None


def test_single_model_and_small_complete_strata():
    assert compare(evidence_only(ranking("a"))) == ()
    rows = []
    for model in ("a", "b"):
        run = evidence_only(ranking(model))
        rows.append(run.model_copy(update={"items": (run.items[0], run.items[2])}))
    result = compare(*rows)[0]
    assert result.estimate == 0
    assert result.unavailable_reason == "insufficient_independent_worlds"
    assert result.confidence_low is result.confidence_high is result.p_value is None


def authored_ranking(model: str, delta: float = 0) -> TrialDevRunRankingV1:
    """Represent current phased results without any legacy decision object."""
    run = ranking(model)
    return run.model_copy(
        update={
            "items": tuple(
                row.model_copy(
                    update={
                        "authored_programme": True,
                        "net_development_value_usd": row.asset_count * 1000 + delta,
                    }
                )
                for row in run.items
                if row.stream == "portfolio_management"
            )
        }
    )


def test_authored_campaign_preserves_dollars_and_withholds_noncompletion():
    a, b = authored_ranking("a", 250_000), authored_ranking("b")
    result = compare(a, b)
    value = next(row for row in result if row.metric == "net_development_value_usd")
    assert value.estimate == pytest.approx(250_000)
    assert value.confidence_low == value.confidence_high == pytest.approx(250_000)
    assert value.worlds == 6
    assert not any(row.metric == "development_value" for row in result)
    interrupted = b.model_copy(
        update={
            "items": (
                b.items[0].model_copy(
                    update={"execution_status": "model_noncompletion", "net_development_value_usd": None}
                ),
                *b.items[1:],
            )
        }
    )
    result = compare(a, interrupted)
    value = next(row for row in result if row.metric == "net_development_value_usd")
    assert value.worlds == 6
    assert value.estimate is value.confidence_low is value.p_value is None
    assert value.unavailable_reason == "model_noncompletion"
    assert result[0].estimate is not None
    invalid = interrupted.model_copy(
        update={"items": (interrupted.items[0].model_copy(update={"net_development_value_usd": 0}), *b.items[1:])}
    )
    with pytest.raises(ValueError, match="cannot have terminal"):
        compare(a, invalid)


@pytest.mark.parametrize("field,value", [("data_view", "analysis_ready"), ("budget_usd", 15_000_000)])
def test_campaign_pairing_binds_presentation_and_resources(field: str, value: str | int):
    a, b = authored_ranking("a"), authored_ranking("b")
    changed = b.model_copy(update={"items": (b.items[0].model_copy(update={field: value}), *b.items[1:])})
    with pytest.raises(ValueError, match="same scheduled task panel"):
        compare(a, changed)


def test_authored_value_fault_is_not_noncompletion_or_legacy_value():
    a, b = authored_ranking("a"), authored_ranking("b")
    faulty = b.model_copy(
        update={"items": (b.items[0].model_copy(update={"execution_status": "evaluator_error"}), *b.items[1:])}
    )
    assert all(row.estimate is None and row.unavailable_reason == "evaluator_fault" for row in compare(a, faulty))
    mixed = a.model_copy(update={"items": (a.items[0].model_copy(update={"authored_programme": False}), *a.items[1:])})
    with pytest.raises(ValueError, match="requires authored development values"):
        compare(mixed, mixed.model_copy(update={"model_id": "b"}))


def test_current_rankings_reject_historical_programme_outputs():
    """Old normalized values cannot enter current dollar-valued comparisons."""
    from pydantic import ValidationError

    current = authored_ranking("a").model_dump(mode="json")
    assert current["schema_id"] == "trialagentbench.trialdev_run_ranking/v5"
    historical = {**current, "schema_id": "trialagentbench.trialdev_run_ranking/v1"}
    with pytest.raises(ValidationError, match="schema_id"):
        TrialDevRunRankingV1.model_validate(historical)
    current["items"][0]["programme"] = {"ranking_value": 0.8}
    with pytest.raises(ValidationError, match="programme"):
        TrialDevRunRankingV1.model_validate(current)


def test_opposite_presentation_and_stream_effects_do_not_cancel():
    """Separate easy-view gains and standalone recovery from raw development."""
    models = []
    for model in ("a", "b"):
        source = authored_ranking(model)
        items = []
        for row in source.items:
            for stream in ("evidence_analysis", "portfolio_management"):
                for view in ("raw", "analysis_ready"):
                    difference = (0.2 if view == "raw" else -0.2) * (1 if stream == "evidence_analysis" else -1)
                    items.append(
                        row.model_copy(
                            update={
                                "task_id": f"{row.task_id}-{stream}-{view}",
                                "stream": stream,
                                "data_view": view,
                                "observational_loss": 0.5 + (difference if model == "a" else 0),
                            }
                        )
                    )
        models.append(source.model_copy(update={"items": tuple(items)}))
    results = compare(*models)
    recovery = {(row.stream, row.data_view): row for row in results if row.metric == "observational_loss"}
    assert len(results) == 6
    assert len(recovery) == 4
    for (stream, view), result in recovery.items():
        expected = (0.2 if view == "raw" else -0.2) * (1 if stream == "evidence_analysis" else -1)
        assert result.estimate == pytest.approx(expected)
        assert result.worlds == 6


def test_completed_initial_analysis_survives_later_programme_noncompletion():
    runs = []
    for model in ("a", "b"):
        row = ranking(model)
        runs.append(
            row.model_copy(
                update={
                    "items": tuple(
                        item.model_copy(update={"execution_status": "model_noncompletion", "authored_programme": True})
                        for item in row.items
                        if item.stream == "portfolio_management"
                    )
                }
            )
        )
    numerical, value = compare(*runs)
    assert numerical.estimate == 0
    assert numerical.unavailable_reason is None
    assert value.estimate is None
    assert value.unavailable_reason == "model_noncompletion"


def test_development_missing_metrics_retain_distinct_execution_states():
    """An omitted analysis does not erase a completed programme's value."""
    from trialagentbench_verifier.development import ranking_means

    source = ranking("a").items[1]
    rows = [
        source.model_copy(
            update={
                "task_id": f"task-{index}",
                "execution_status": status,
                "authored_programme": True,
                "observational_loss": None,
                "net_development_value_usd": 100.0 if status == "completed" else None,
            }
        )
        for index, status in enumerate(
            ("completed", "model_noncompletion", "evaluator_error", "incomplete_run", "not_started")
        )
    ]
    summaries = {row.metric: row for row in ranking_means(rows)}
    recovery = summaries["observational_loss"]
    value = summaries["net_development_value_usd"]
    assert recovery.scored_tasks == 0 and recovery.observed_mean is None
    assert recovery.unscored_task_counts == {
        "unscored": 1,
        "model_noncompletion": 1,
        "evaluator_error": 1,
        "incomplete_run": 1,
        "not_started": 1,
    }
    assert value.scored_tasks == 1 and value.observed_mean == 100.0
    assert value.unscored_task_counts == {
        key: count for key, count in recovery.unscored_task_counts.items() if key != "unscored"
    }
    assert recovery.mean is value.mean is None
    with pytest.raises(ValueError, match="every missing metric"):
        type(value).model_validate({**value.model_dump(), "unscored_task_counts": {}})


def test_partial_targets_are_counted_without_a_partial_task_score():
    from trialagentbench_verifier.development import ranking_means

    row = (
        ranking("a")
        .items[0]
        .model_copy(
            update={
                "observational_loss": None,
                "initial_target_statuses": {"benefit": "reported", "safety": "missing"},
            }
        )
    )
    (summary,) = ranking_means([row])
    assert summary.target_status_counts == {"reported": 1, "missing": 1}
    assert summary.scored_tasks == 0
    assert summary.mean is summary.observed_mean is None
    with pytest.raises(ValueError, match="every initial target"):
        type(row).model_validate({**row.model_dump(), "observational_loss": 0.1})


def test_campaign_rejects_different_initial_target_inventory():
    a, b = evidence_only(ranking("a")), evidence_only(ranking("b"))
    changed = b.model_copy(
        update={
            "items": (
                b.items[0].model_copy(update={"initial_target_statuses": {"different": "reported"}}),
                *b.items[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="same scheduled task panel"):
        compare(a, changed)


@pytest.mark.parametrize(
    ("selectors", "expected", "expected_error"),
    [
        ((), (0, 2), None),
        (("td-000000000001",), (1,), None),
        (("tdw-0000",), (0, 1), None),
        (("evidence_analysis",), (0, 1), None),
        (("tdw-0001:portfolio_management",), (2, 3), None),
        (("missing",), (0,), "selectors match no task"),
        ((), (0,), "frozen model or task panel"),
        ((), (0, 1, 2, 3), "frozen model or task panel"),
        (("tdw-0000",), (0,), "frozen model or task panel"),
    ],
)
@pytest.mark.parametrize("transport", ["zip", "assets", "incomplete_assets"])
def test_campaign_archive_resolves_native_selectors(
    tmp_path, monkeypatch, selectors, expected, expected_error, transport
):
    """Default and shorthand panels must reconcile with authenticated native ranks."""
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock
    from zipfile import ZipFile

    from trialagentbench_verifier import development_campaign as owner
    from trialagentbench_verifier.campaign_contracts import CampaignPlanV1
    from trialagentbench_verifier.io import sha256_file

    tasks = [
        dict(
            task_id=f"td-{index:012x}",
            case_id=f"tdw-{index // 2:04x}",
            stream="evidence_analysis" if index < 2 else "portfolio_management",
            data_view="raw" if index % 2 == 0 else "analysis_ready",
            available_materials=[],
        )
        for index in range(4)
    ]
    release = tmp_path / "release.zip"
    with ZipFile(release, "w") as archive:
        for task in tasks:
            archive.writestr(f"participant/tasks/{task['task_id']}/task.json", json.dumps(task))
        archive.writestr(
            "evaluation/worlds/tdw-0000/world.json",
            json.dumps(
                dict(
                    world_id="tdw-0000",
                    clinical_profile=dict(profile_id="clinical", candidate_labels=["a", "b", "c"]),
                    difficulty_profile_id="scenario",
                )
            ),
        )
    if transport != "zip":
        from trialagentbench_verifier.assets import pack_release_assets

        with ZipFile(release, "a") as archive:
            archive.writestr("release_manifest.json", '{"release_id":"fixture"}')
            archive.writestr("participant/manifest.json", "{}")
            archive.writestr("evaluation/manifest.json", "{}")
            for index in range(2):
                archive.writestr(f"participant/worlds/tdw-{index:04x}/observed.csv", "patient,value\n1,0\n")
            for index in ((1,) if transport == "assets" else ()):
                archive.writestr(
                    f"evaluation/worlds/tdw-{index:04x}/world.json",
                    json.dumps(
                        dict(
                            world_id=f"tdw-{index:04x}",
                            clinical_profile=dict(profile_id="clinical", candidate_labels=["a", "b", "c"]),
                            difficulty_profile_id="scenario",
                        )
                    ),
                )
        _, release = pack_release_assets(release, expected_sha256=sha256_file(release), output=tmp_path / "assets")
    plan = CampaignPlanV1.model_validate(
        dict(
            configuration=dict(
                release_directory=tmp_path,
                expected_release_manifest_sha256="a" * 64,
                executor_image_id="sha256:" + "b" * 64,
                models=[
                    dict(
                        name="control",
                        model="fixture/control",
                        upstream_provider="fixture",
                        spend_limit_micro_usd=10,
                        reservation_micro_usd=1,
                        reservation_tokens=1000,
                        maximum_completion_tokens=100,
                        pricing_premise="Offline test",
                        requests_per_minute=10,
                        tokens_per_minute=1000,
                    )
                ],
                panel=dict(benchmarks=["trialdev"], trialdev_tasks=selectors),
                operations=dict(spend_limit_micro_usd=10),
            ),
            runs=[
                dict(
                    job_id="control-trialdev-1",
                    benchmark="trialdev",
                    model_name="control",
                    repetition=1,
                    relative_run_root="runs/control/run",
                    invocation_sha256="c" * 64,
                )
            ],
        )
    )
    items = tuple(
        ranking("fixture/control")
        .items[0]
        .model_copy(
            update=dict(
                task_id=tasks[index]["task_id"],
                world_id=tasks[index]["case_id"],
                stream=tasks[index]["stream"],
                data_view=tasks[index]["data_view"],
                authored_programme=index >= 2,
                net_development_value_usd=0.0 if index >= 2 else None,
            )
        )
        for index in expected
    )
    observed = ranking("fixture/control").model_copy(update=dict(items=items))
    verify = Mock(
        return_value=SimpleNamespace(status="pass", checks=[SimpleNamespace(observed=observed.model_dump())])
    )
    monkeypatch.setattr(owner, "verify_trialdev_ranking", verify)
    export = owner.TrialDevCampaignExportV1(
        job_id="control-trialdev-1", path=tmp_path / "results.zip", sha256="d" * 64
    )

    def analyse():
        return owner.analyse_trialdev_campaign(
            release,
            expected_release_sha256=sha256_file(release),
            plan=plan,
            expected_plan_checksum=plan.checksum,
            exports=(export,),
        )

    if transport == "incomplete_assets" and selectors != ("missing",):
        expected_error = "complete shared world metadata"
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            analyse()
        return
    result = analyse()
    assert sum(row.scheduled_tasks for row in result.profile_summaries if row.metric == "observational_loss") == len(
        expected
    )
    assert verify.call_args.kwargs["expected_invocation_sha256"] == "c" * 64


def test_leaderboard_uses_the_missing_answer_rule_and_world_clustered_intervals():
    from statistics import mean

    from trialagentbench_verifier.development_campaign import trialdev_leaderboard_rows

    policy = CampaignAnalysisV1()
    complete = evidence_only(ranking("model-a"))
    (row,) = trialdev_leaderboard_rows((complete,), required_strata=STRATA, policy=policy)
    expected = mean((mean((1.0, 1 / 1.1)), 1 / 1.5))
    assert (row.metric, row.track, row.scheduled_tasks, row.independent_worlds) == (
        "bounded_recovery",
        "evidence_analysis/raw",
        6,
        6,
    )
    assert row.estimate == pytest.approx(expected)
    assert row.interval_low is not None and row.interval_high is not None
    assert row.interval_low <= row.estimate <= row.interval_high

    missing_item = complete.items[0].model_copy(
        update={
            "execution_status": "model_noncompletion",
            "observational_loss": None,
            "initial_target_statuses": {"effect": "missing"},
        }
    )
    incomplete = complete.model_copy(update={"items": (missing_item, *complete.items[1:])})
    (row,) = trialdev_leaderboard_rows((incomplete,), required_strata=STRATA, policy=policy)
    # The missing answer scores 0 rather than voiding the headline.
    assert row.scored_tasks == 5
    assert row.estimate == pytest.approx(mean((mean((0.0, 1 / 1.1)), 1 / 1.5)))

    authored = authored_ranking("model-a")
    value = next(
        row
        for row in trialdev_leaderboard_rows((authored,), required_strata=STRATA, policy=policy)
        if row.metric == "net_development_value_usd"
    )
    assert (value.estimate, value.interval_low, value.interval_high) == (4000.0, 4000.0, 4000.0)
    unfinished = authored.model_copy(
        update={
            "items": (authored.items[0].model_copy(update={"net_development_value_usd": None}), *authored.items[1:])
        }
    )
    value = next(
        row
        for row in trialdev_leaderboard_rows((unfinished,), required_strata=STRATA, policy=policy)
        if row.metric == "net_development_value_usd"
    )
    assert (value.estimate, value.interval_low, value.scored_tasks) == (None, None, value.scheduled_tasks - 1)


def _censoring(**censored: tuple[str, ...]):
    from trialagentbench_verifier.campaign_contracts import BudgetCensoringV1, CensoredRunV1

    tasks = tuple(row.task_id for row in evidence_only(ranking("a")).items)
    return BudgetCensoringV1(
        campaign_plan_sha256="c" * 64,
        exhausted_models=tuple(name for name, rows in censored.items() if rows),
        runs=tuple(
            CensoredRunV1(
                job_id=f"{name}-trialdev-1",
                model_name=name,
                model_id=name,
                benchmark="trialdev",
                repetition=1,
                scheduled_task_ids=tasks,
                censored_task_ids=rows,
            )
            for name, rows in censored.items()
        ),
    )


def _unrun(row: TrialDevRunRankingV1, task_ids: set[str]) -> TrialDevRunRankingV1:
    return row.model_copy(
        update={
            "items": tuple(
                (
                    item.model_copy(
                        update={
                            "execution_status": "incomplete_run",
                            "observational_loss": None,
                            "initial_target_statuses": {"effect": "incomplete_run"},
                        }
                    )
                    if item.task_id in task_ids
                    else item
                )
                for item in row.items
            )
        }
    )


def test_budget_censoring_compares_models_on_the_paired_panel_only():
    """Censored tasks leave every model's comparison panel; participant failures keep the missing-answer rule."""
    from trialagentbench_verifier.development_campaign import paired_trialdev_results, trialdev_leaderboard_rows

    policy = CampaignAnalysisV1(bootstrap_repetitions=200)
    complete_a, complete_b = evidence_only(ranking("a")), evidence_only(ranking("b", delta=0.2))
    censored = {"5-2-evidence_analysis", "5-3-evidence_analysis"}
    failed = complete_b.items[0].model_copy(
        update={
            "execution_status": "model_noncompletion",
            "observational_loss": None,
            "initial_target_statuses": {"effect": "missing"},
        }
    )
    complete_b = complete_b.model_copy(update={"items": (failed, *complete_b.items[1:])})
    rankings = {"a-trialdev-1": complete_a, "b-trialdev-1": _unrun(complete_b, censored)}
    censoring = _censoring(a=(), b=tuple(sorted(censored)))
    comparisons, profiles, leaderboard = paired_trialdev_results(
        rankings, censoring=censoring, required_strata=STRATA, policy=policy
    )
    paired = frozenset(row.task_id for row in complete_a.items) - censored
    assert censoring.paired_task_ids("trialdev") == paired
    retained = tuple(
        row.model_copy(update={"items": tuple(i for i in row.items if i.task_id in paired)})
        for row in (complete_a, complete_b)
    )
    assert comparisons == compare_trialdev_rankings(retained, required_strata=STRATA, policy=policy)
    expected = trialdev_leaderboard_rows(retained, required_strata=STRATA, policy=policy)
    assert [row.model_copy(update={"budget_censored_tasks": 0}) for row in leaderboard] == list(expected)
    by_model = {row.model_id: row for row in leaderboard}
    assert (by_model["a"].scheduled_tasks, by_model["a"].budget_censored_tasks) == (4, 0)
    # The participant failure scores 0 under the missing-answer rule; censored tasks are never scored.
    assert (by_model["b"].scheduled_tasks, by_model["b"].scored_tasks, by_model["b"].budget_censored_tasks) == (
        4,
        3,
        2,
    )
    # Single-model summaries cover each model's own completed tasks.
    completed = {
        model: sum(
            row.scheduled_tasks for row in profiles if row.model_id == model and row.metric == "observational_loss"
        )
        for model in ("a", "b")
    }
    assert completed == {"a": 6, "b": 4}
    assert not any(row.unscored_task_counts.get("incomplete_run") for row in profiles)
    assert [(row.model_id, row.censoring_rate) for row in censoring.models] == [("a", 0.0), ("b", 1 / 3)]
    assert censoring.warnings and "differs materially" in censoring.warnings[0]

    with pytest.raises(ValueError, match="only recorded budget censoring"):
        paired_trialdev_results(rankings, censoring=None, required_strata=STRATA, policy=policy)
    with pytest.raises(ValueError, match="only recorded budget censoring"):
        paired_trialdev_results(
            {"a-trialdev-1": complete_a, "b-trialdev-1": complete_b},
            censoring=censoring,
            required_strata=STRATA,
            policy=policy,
        )
    everything = _censoring(a=(), b=tuple(sorted(row.task_id for row in complete_a.items)))
    unrun = {"a-trialdev-1": complete_a, "b-trialdev-1": _unrun(complete_b, set(everything.runs[1].censored_task_ids))}
    comparisons, profiles, leaderboard = paired_trialdev_results(
        unrun, censoring=everything, required_strata=STRATA, policy=policy
    )
    assert comparisons == () and leaderboard == () and {row.model_id for row in profiles} == {"a"}
    assert any("no scheduled task was completed by every model" in text for text in everything.warnings)


def test_budget_censoring_record_requires_recorded_exhaustion_and_derives_its_summaries():
    from trialagentbench_verifier.campaign_contracts import BudgetCensoringV1

    # Equal censoring rates need no informative-censoring warning; the paired panel drops both tasks.
    equal = _censoring(a=("3-0-evidence_analysis",), b=("3-1-evidence_analysis",))
    assert equal.warnings == () and len(equal.paired_task_ids("trialdev")) == 4
    record = _censoring(a=("3-0-evidence_analysis",), b=())
    assert "a 17%, b 0%" in record.warnings[0]
    payload = record.model_dump(mode="json")
    with pytest.raises(ValueError, match="recorded spending-limit exhaustion"):
        BudgetCensoringV1.model_validate({**payload, "exhausted_models": []})
    assert BudgetCensoringV1.model_validate(
        {**payload, "exhausted_models": [], "campaign_budget_exhausted": True}
    ).paired_task_ids("trialdev") == frozenset(record.runs[0].scheduled_task_ids) - {"3-0-evidence_analysis"}
    with pytest.raises(ValueError, match="differ from their runs"):
        BudgetCensoringV1.model_validate({**payload, "warnings": ["edited"]})
    with pytest.raises(ValueError, match="unique scheduled tasks"):
        _censoring(a=("not-scheduled",), b=())
    assert BudgetCensoringV1.model_validate_json(record.model_dump_json()) == record
