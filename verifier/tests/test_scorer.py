"""Scientific artifacts cannot silently acquire a new scorer identity."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trialagentbench_verifier.campaign_contracts import CampaignAnalysisV1
from trialagentbench_verifier.development_campaign import compare_trialdev_rankings, trialdev_leaderboard_rows
from trialagentbench_verifier.response import TrialDevRunRankingV1
from trialagentbench_verifier.scorer import ScorerIdentityV1, current_scorer, require_same_scorer


def _ranking() -> TrialDevRunRankingV1:
    return TrialDevRunRankingV1(
        scorer=current_scorer(),
        release_id="release",
        release_manifest_sha256="a" * 64,
        invocation_checksum="b" * 64,
        model_id="model",
        scope="scheduled_panel",
        items=(),
        profile_means=(),
    )


@pytest.mark.parametrize("field,value", [("version", "earlier-scorer"), ("sources_sha256", "0" * 64)])
def test_reading_and_pooling_refuse_changed_scorer(field, value):
    ranking = _ranking()
    earlier = ScorerIdentityV1.model_validate(current_scorer().model_dump() | {field: value})
    changed = ranking.model_copy(update={"scorer": earlier})
    with pytest.raises(ValidationError, match="regrade retained outputs"):
        TrialDevRunRankingV1.model_validate_json(changed.model_dump_json())
    with pytest.raises(ValueError, match="different scorer identities"):
        require_same_scorer([ranking.scorer, changed.scorer])
    for operation in (compare_trialdev_rankings, trialdev_leaderboard_rows):
        with pytest.raises(ValueError, match="different scorer identities"):
            operation((ranking, changed), required_strata=(("clinical", "scenario", 3),), policy=CampaignAnalysisV1())


def test_missing_identity_is_refused_instead_of_assuming_current():
    payload = _ranking().model_dump(mode="json", exclude={"scorer"})
    with pytest.raises(ValidationError, match="scorer"):
        TrialDevRunRankingV1.model_validate(payload)


@pytest.mark.parametrize("operation", ["public_trialeval_campaign", "trialeval_leaderboard_rows"])
def test_trialeval_public_export_refuses_a_copied_stale_report(operation):
    from types import SimpleNamespace

    from trialagentbench_verifier import public_results

    report = SimpleNamespace(
        scorer=current_scorer().model_copy(update={"version": "earlier-scorer"}),
        between_model_contrasts=(),
    )
    kwargs = {"split": "public"} if operation == "public_trialeval_campaign" else {}
    with pytest.raises(ValueError, match="regrade retained outputs"):
        getattr(public_results, operation)(report, **kwargs)


def test_leaderboard_export_refuses_rows_from_different_scorers():
    from trialagentbench_verifier.campaign_contracts import LeaderboardRowV1
    from trialagentbench_verifier.public_results import CampaignLeaderboardV1

    row = LeaderboardRowV1(
        scorer=current_scorer(),
        benchmark="trialeval",
        track="point",
        model_id="model-a",
        metric="mean_loss",
        direction="lower_is_better",
        estimate=None,
        scheduled_tasks=1,
        scored_tasks=0,
        independent_worlds=1,
    )
    earlier = row.model_copy(update={"scorer": current_scorer().model_copy(update={"sources_sha256": "0" * 64})})
    with pytest.raises(ValueError, match="different scorer identities"):
        CampaignLeaderboardV1(
            scorer=current_scorer(),
            campaign_plan_sha256="a" * 64,
            release_set_manifest_sha256="b" * 64,
            split="public",
            confidence_level=0.95,
            bootstrap_resamples=100,
            bootstrap_seed=0,
            rows=(row, earlier),
        )
