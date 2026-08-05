"""Tests for released-portfolio observational replay discovery."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from trialagentbench_validation.trialdev import portfolio_observational_replay
from trialagentbench_validation.trialdev.portfolio_observational_replay import (
    released_portfolio_observational_worlds_v1,
)


def _view(*, world_id: str, reference: str, programme_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        world_id=world_id,
        observational_reference_relative_path=reference,
        programme_id=programme_id,
    )


def _mock_loader(
    *,
    evaluator_views: tuple[SimpleNamespace, ...],
    participant_views: tuple[SimpleNamespace, ...],
):
    def _load(model: type[object], path: Path) -> SimpleNamespace:
        if model.__name__ == "TrialDevPortfolioReleaseManifestV1":
            return SimpleNamespace(evaluator_views=evaluator_views)
        return SimpleNamespace(views=participant_views)

    return _load


def test_replay_discovery_has_one_reference_per_released_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    participant_views = tuple(
        _view(
            world_id=f"portfolio-world-{index:02d}",
            reference=f"evaluator/worlds/portfolio-world-{index:02d}/observational_reference.json",
            programme_id=f"programme-{index}-{variant}",
        )
        for index in range(1, 13)
        for variant in range(8)
    )
    evaluator_views = tuple(
        SimpleNamespace(
            programme_id=view.programme_id,
            observational_reference_relative_path=view.observational_reference_relative_path,
        )
        for view in participant_views
    )
    monkeypatch.setattr(
        portfolio_observational_replay,
        "read_json_model",
        _mock_loader(
            evaluator_views=evaluator_views, participant_views=participant_views
        ),
    )
    root = tmp_path / "release" / "evaluator"
    root.mkdir(parents=True)
    (root / "release_manifest.json").write_text("{}\n", encoding="utf-8")

    worlds = released_portfolio_observational_worlds_v1(release_root=root.parent)

    assert len(worlds) == 12
    assert len({world_id for world_id, _ in worlds}) == 12
    assert all(relative.startswith("evaluator/worlds/") for _, relative in worlds)


def test_replay_discovery_rejects_conflicting_reference_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    participant_views = (
        _view(
            world_id="portfolio-world-01",
            reference="evaluator/worlds/portfolio-world-01/observational_reference.json",
            programme_id="programme-1",
        ),
        _view(
            world_id="portfolio-world-01",
            reference="evaluator/conflict.json",
            programme_id="programme-2",
        ),
    )
    evaluator_views = tuple(
        SimpleNamespace(
            programme_id=view.programme_id,
            observational_reference_relative_path=view.observational_reference_relative_path,
        )
        for view in participant_views
    )
    monkeypatch.setattr(
        portfolio_observational_replay,
        "read_json_model",
        _mock_loader(
            evaluator_views=evaluator_views, participant_views=participant_views
        ),
    )
    root = tmp_path / "release" / "evaluator"
    root.mkdir(parents=True)
    (root / "release_manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        released_portfolio_observational_worlds_v1(release_root=root.parent)
