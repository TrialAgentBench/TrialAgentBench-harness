"""Replay every portfolio observational analysis from exact released inputs."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from trialagentbench_harness.contracts.trialdev.portfolio_release import (
    TrialDevPortfolioParticipantCatalogueV1,
    TrialDevPortfolioReleaseManifestV1,
)
from trialagentbench_harness.io.json import read_json_model, write_json_model

from trialagentbench_validation.process_pool import (
    single_threaded_numerical_process_pool,
)
from trialagentbench_validation.trialdev.replay import (
    replay_trialdev_observational_reference,
)


class TrialDevPortfolioObservationalReplayCensusV1(BaseModel):
    """Identity and status of a complete released-world replay census."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[
        "trialagentbench.validation.trialdev_portfolio_observational_replay/v1"
    ] = "trialagentbench.validation.trialdev_portfolio_observational_replay/v1"
    release_source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_world_count: int = Field(ge=1)
    passing_world_count: int = Field(ge=0)
    report_paths: tuple[str, ...] = Field(min_length=1)
    status: Literal["pass", "fail"]


def released_portfolio_observational_worlds_v1(
    *, release_root: Path, world_ids: tuple[str, ...] | None = None
) -> tuple[tuple[str, str], ...]:
    """Resolve unique worlds and evaluator references from a release manifest."""

    root = Path(release_root).resolve(strict=True)
    manifest = read_json_model(
        TrialDevPortfolioReleaseManifestV1,
        root / "evaluator" / "release_manifest.json",
    )
    catalogue = read_json_model(
        TrialDevPortfolioParticipantCatalogueV1,
        root / "participant_catalogue.json",
    )
    world_by_programme = {view.programme_id: view.world_id for view in catalogue.views}
    if len(world_by_programme) != len(catalogue.views):
        raise ValueError(
            "Portfolio participant catalogue contains duplicate programme identities."
        )
    selected = None if world_ids is None else set(world_ids)
    if world_ids is not None and len(set(world_ids)) != len(world_ids):
        raise ValueError("Requested portfolio replay worlds must be unique.")
    references: dict[str, str] = {}
    for view in manifest.evaluator_views:
        world_id = world_by_programme.get(view.programme_id)
        if world_id is None:
            raise ValueError(
                f"Evaluator view has no participant view: {view.programme_id!r}."
            )
        if selected is not None and world_id not in selected:
            continue
        existing = references.setdefault(
            world_id, view.observational_reference_relative_path
        )
        if existing != view.observational_reference_relative_path:
            raise ValueError(
                f"Portfolio world has conflicting observational references: {world_id!r}."
            )
    if selected is not None:
        missing = selected - set(references)
        if missing:
            raise ValueError(
                f"Requested portfolio replay worlds are absent: {sorted(missing)!r}."
            )
    if not references:
        raise ValueError("Portfolio replay selected no released worlds.")
    return tuple(sorted(references.items()))


def _replay_world(task: tuple[Path, str, str, Path]) -> tuple[str, str]:
    release_root, world_id, reference_relative_path, output_root = task
    with tempfile.TemporaryDirectory(
        prefix=f"trialdev-replay-{world_id}-"
    ) as temporary:
        scenario = Path(temporary)
        shutil.copytree(
            release_root / "worlds" / world_id / "public", scenario / "public"
        )
        grader = scenario / "grader"
        grader.mkdir()
        shutil.copyfile(
            release_root / reference_relative_path,
            grader / "public_recoverability_report.json",
        )
        report = replay_trialdev_observational_reference(scenario)
    output_name = f"observational_replay_{world_id.removeprefix('portfolio-').replace('-', '_')}.json"
    write_json_model(output_root / output_name, report)
    return output_name, report.status


def replay_trialdev_portfolio_observational_release_v1(
    *,
    release_root: Path,
    output_dir: Path,
    workers: int,
    world_ids: tuple[str, ...] | None = None,
) -> TrialDevPortfolioObservationalReplayCensusV1:
    """Replay selected or all portfolio worlds and publish one atomic census."""

    if workers < 1:
        raise ValueError("Portfolio observational replay workers must be positive.")
    release = Path(release_root).resolve(strict=True)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = read_json_model(
        TrialDevPortfolioReleaseManifestV1,
        release / "evaluator" / "release_manifest.json",
    )
    worlds = released_portfolio_observational_worlds_v1(
        release_root=release, world_ids=world_ids
    )
    with tempfile.TemporaryDirectory(
        prefix="trialdev-replay-census-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        tasks = tuple(
            (release, world_id, reference, staging) for world_id, reference in worlds
        )
        if workers == 1:
            results = tuple(_replay_world(task) for task in tasks)
        else:
            with single_threaded_numerical_process_pool(
                workers=min(workers, len(tasks))
            ) as pool:
                results = tuple(pool.map(_replay_world, tasks))
        passing = sum(status == "pass" for _, status in results)
        census = TrialDevPortfolioObservationalReplayCensusV1(
            release_source_identity=manifest.source_identity,
            requested_world_count=len(results),
            passing_world_count=passing,
            report_paths=tuple(sorted(name for name, _ in results)),
            status="pass" if passing == len(results) else "fail",
        )
        write_json_model(staging / "observational_replay_census.json", census)
        os.replace(staging, output)
    return census


__all__ = [
    "TrialDevPortfolioObservationalReplayCensusV1",
    "released_portfolio_observational_worlds_v1",
    "replay_trialdev_portfolio_observational_release_v1",
]
