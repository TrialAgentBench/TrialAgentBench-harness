"""Campaign point estimates and intervals share canonical finite-cell weights."""

import pytest

from trialagentbench_verifier.campaign import EffectState, TrialEvalCampaignCellV1, _bootstrap, _cell_mean


def test_unbalanced_effect_labels_resample_worlds_within_the_series() -> None:
    """Effect labels are descriptive; worlds of a series are exchangeable under the continuous prior."""
    values: dict[tuple[str, EffectState, str], float] = {
        ("TE-S01-A1", "favourable", "a"): 0,
        ("TE-S01-A1", "favourable", "b"): 0,
        ("TE-S01-A1", "near_null", "c"): 1,
        ("TE-S01-A1", "unfavourable", "d"): 1,
    }
    assert _cell_mean(values) == 0.5
    cell = TrialEvalCampaignCellV1(regime_cell_id="TE-S01-A1", series_id="TE-S01")
    estimate, low, high, p_value = _bootstrap(values, cells=(cell,), resamples=200, confidence_level=0.95, seed=7)
    assert estimate == 0.5 and p_value is not None
    assert low is not None and high is not None and 0.0 <= low < 0.5 < high <= 1.0


def test_a_singleton_series_withholds_its_interval() -> None:
    values: dict[tuple[str, EffectState, str], float] = {("TE-S01-A1", "favourable", "a"): 0.3}
    cell = TrialEvalCampaignCellV1(regime_cell_id="TE-S01-A1", series_id="TE-S01")
    assert _bootstrap(values, cells=(cell,), resamples=100, confidence_level=0.95, seed=7) == (
        pytest.approx(0.3),
        None,
        None,
        None,
    )


def test_trialeval_leaderboard_leads_with_decision_consequence_and_keeps_loss_secondary():
    from types import SimpleNamespace

    from trialagentbench_verifier.campaign import TrialEvalLossRankingV1
    from trialagentbench_verifier.public_results import trialeval_leaderboard_rows

    ranking = TrialEvalLossRankingV1(
        model_id="model-a",
        target_kind="point",
        scheduled_attempts=4,
        scored_attempts=3,
        scheduled_targets=4,
        scored_targets=3,
        unscored_targets=1,
        target_failure_counts={"not_started": 1},
        observed_mean_loss=0.5,
        scored_worlds=2,
        scored_regime_cells=("TE-S01-A1",),
        independent_worlds=2,
        regime_cells=("TE-S01-A1",),
        mean_loss=None,
        display_recovery=None,
        interval_low=None,
        interval_high=None,
        bounded_recovery=0.4,
        bounded_recovery_low=0.2,
        bounded_recovery_high=0.6,
    )

    def summary(metric_id: str, estimate: float, low: float, high: float, context: str = "C5"):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            model_id="model-a",
            context=context,
            metric=SimpleNamespace(metric_id=metric_id),
            scheduled_attempts=4,
            observed_attempts=4,
            equal_cell_effect_mean=estimate,
            interval_low=low,
            interval_high=high,
        )

    summaries = (
        summary("decision_regret_musd", 12.5, 3.0, 22.0),
        summary("consequential_error", 0.25, 0.0, 0.5),
        summary("decision_regret_musd", 99.0, 90.0, 110.0, context="C2"),
    )
    regret, errors, loss = trialeval_leaderboard_rows(
        SimpleNamespace(loss_rankings=(ranking,), metric_summaries=summaries)
    )
    # The headline prices answers by their decisions on the main C5 view; missing answers drive the
    # prior-only decision, so the decision rows count every scheduled answer.
    assert (regret.metric, regret.estimate, regret.interval_low, regret.interval_high, regret.scored_tasks) == (
        "decision_regret_musd",
        12.5,
        3.0,
        22.0,
        4,
    )
    assert (errors.metric, errors.estimate, errors.direction) == ("consequential_error_rate", 0.25, "lower_is_better")
    # One missing attempt withholds the complete-panel loss, the secondary row.
    assert (loss.metric, loss.estimate, loss.scored_tasks, loss.scheduled_tasks) == ("mean_loss", None, 3, 4)
