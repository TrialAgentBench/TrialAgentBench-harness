"""Validate and stage immutable bounded-portfolio TrialDev evidence."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

from trialagentbench_harness.contracts.trialdev.portfolio_release import (
    TrialDevPortfolioEpisodeDatasetV1,
    TrialDevPortfolioEvaluatorViewV1,
    TrialDevPortfolioParticipantCatalogueV1,
    TrialDevPortfolioParticipantViewV1,
    TrialDevPortfolioReleaseManifestV1,
)
from trialagentbench_harness.contracts.trialdev.programme import (
    TrialDevEvidenceReferenceV1,
    TrialDevPortfolioProgrammeStateV1,
)
from trialagentbench_harness.io import sha256_file, write_json
from trialagentbench_harness.io.json import read_json_model, write_json_model
from trialagentbench_harness.trialdev.grading.sequential import phase_summary_v1


def _release_member(root: Path, relative_path: str) -> Path:
    path = (Path(root) / relative_path).resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ValueError(f"Portfolio release path escapes its root: {relative_path!r}.") from exc
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Portfolio release member is absent or not a regular file: {path}")
    return path


def _require_checksum(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"Portfolio release checksum mismatch for {path}: expected={expected}, observed={observed}.")


def load_portfolio_catalogue_v1(release_root: Path) -> TrialDevPortfolioParticipantCatalogueV1:
    """Load the strict participant catalogue from an extracted release."""

    return cast(
        TrialDevPortfolioParticipantCatalogueV1,
        read_json_model(
            TrialDevPortfolioParticipantCatalogueV1,
            Path(release_root) / "participant_catalogue.json",
        ),
    )


def load_portfolio_manifest_v1(release_root: Path) -> TrialDevPortfolioReleaseManifestV1:
    """Load the evaluator manifest from an extracted release."""

    return cast(
        TrialDevPortfolioReleaseManifestV1,
        read_json_model(
            TrialDevPortfolioReleaseManifestV1,
            Path(release_root) / "evaluator" / "release_manifest.json",
        ),
    )


def validate_portfolio_release_v1(release_root: Path) -> TrialDevPortfolioReleaseManifestV1:
    """Validate every role binding and checksummed evidence input in a release."""

    root = Path(release_root).resolve(strict=True)
    catalogue = load_portfolio_catalogue_v1(root)
    manifest = load_portfolio_manifest_v1(root)
    if catalogue.release_id != manifest.release_id or catalogue.source_identity != manifest.source_identity:
        raise ValueError("Portfolio participant and evaluator release identities disagree.")
    if catalogue.views != manifest.participant_views:
        raise ValueError("Portfolio catalogue and evaluator manifest contain different participant views.")
    for view in manifest.evaluator_views:
        for relative_path, expected in (
            (view.world_manifest_relative_path, view.world_manifest_sha256),
            (view.evidence_index_relative_path, view.evidence_index_sha256),
            (view.observational_reference_relative_path, view.observational_reference_sha256),
        ):
            _require_checksum(_release_member(root, relative_path), expected)
    for episode in manifest.episodes:
        _require_checksum(_release_member(root, episode.protocol_relative_path), episode.protocol_sha256)
        dataset_path = _release_member(root, episode.episode_manifest_relative_path)
        _require_checksum(dataset_path, episode.episode_manifest_sha256)
        dataset = read_json_model(TrialDevPortfolioEpisodeDatasetV1, dataset_path)
        if (
            dataset.episode_id != episode.episode_id
            or dataset.world_id != episode.world_id
            or dataset.asset_id != episode.asset_id
            or dataset.phase_id != episode.phase_id
            or dataset.generation_seed != episode.generation_seed
            or dataset.files[0].row_count != episode.row_count
        ):
            raise ValueError(f"Portfolio episode and dataset manifests disagree: {episode.episode_id}.")
        for artifacts in (dataset.files, dataset.metadata_files):
            for artifact in artifacts:
                _require_checksum(_release_member(root, artifact.relative_path), artifact.sha256)
    return manifest


def portfolio_participant_view_v1(
    release_root: Path,
    programme_id: str,
) -> TrialDevPortfolioParticipantViewV1:
    """Return one uniquely identified participant programme view."""

    catalogue = load_portfolio_catalogue_v1(release_root)
    matches = tuple(view for view in catalogue.views if view.programme_id == programme_id)
    if len(matches) != 1:
        raise ValueError(f"Portfolio release does not contain one programme_id={programme_id!r}.")
    return matches[0]


def portfolio_evaluator_view_v1(
    release_root: Path,
    programme_id: str,
) -> TrialDevPortfolioEvaluatorViewV1:
    """Return the evaluator binding for one participant programme view."""

    manifest = load_portfolio_manifest_v1(release_root)
    matches = tuple(view for view in manifest.evaluator_views if view.programme_id == programme_id)
    if len(matches) != 1:
        raise ValueError(f"Portfolio evaluator manifest does not contain one programme_id={programme_id!r}.")
    return matches[0]


def initial_portfolio_state_v1(view: TrialDevPortfolioParticipantViewV1) -> TrialDevPortfolioProgrammeStateV1:
    """Build the checksum-bound initial state for a portfolio programme."""

    return TrialDevPortfolioProgrammeStateV1(
        programme_id=view.programme_id,
        scenario_id=view.scenario_id,
        current_checkpoint_id="observational_review",
        candidate_asset_ids=view.candidate_asset_ids,
        policy_binding=view.policy_binding,
        evidence=view.initial_evidence,
    )


def stage_portfolio_public_view_v1(
    *,
    release_root: Path,
    view: TrialDevPortfolioParticipantViewV1,
    workdir: Path,
) -> Path:
    """Stage one flat participant workspace without evaluator or future evidence."""

    root = Path(release_root).resolve(strict=True)
    destination = Path(workdir)
    if destination.exists():
        raise FileExistsError(f"Portfolio participant workspace already exists: {destination}")
    public_root = (root / view.public_scenario_relative_path).resolve(strict=True)
    try:
        public_root.relative_to(root)
    except ValueError as exc:
        raise ValueError("Portfolio public scenario path escapes its release root.") from exc
    destination.mkdir(parents=True)
    for source in sorted(public_root.iterdir(), key=lambda path: path.name):
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Portfolio public scenario members must be regular files: {source}")
        shutil.copy2(source, destination / source.name)
    write_json_model(destination / "programme_view.json", view)
    write_json_model(destination / "programme_state.json", initial_portfolio_state_v1(view))
    return destination


def _episode_dataset_for_reference(
    *,
    release_root: Path,
    evidence: TrialDevEvidenceReferenceV1,
) -> TrialDevPortfolioEpisodeDatasetV1:
    if evidence.evidence_kind != "dataset" or evidence.asset_id is None:
        raise ValueError("Randomized portfolio evidence must identify one asset-specific dataset.")
    manifest_path = _release_member(release_root, evidence.relative_path)
    _require_checksum(manifest_path, evidence.artifact_sha256)
    return cast(
        TrialDevPortfolioEpisodeDatasetV1,
        read_json_model(TrialDevPortfolioEpisodeDatasetV1, manifest_path),
    )


def stage_portfolio_checkpoint_evidence_v1(
    *,
    release_root: Path,
    scenario_public_root: Path,
    evidence: tuple[TrialDevEvidenceReferenceV1, ...],
    destination: Path,
) -> tuple[Path, ...]:
    """Stage only the asset episodes released at the current checkpoint."""

    if not evidence or any(item.asset_id is None for item in evidence):
        raise ValueError("Portfolio randomized checkpoint requires asset-specific evidence.")
    if len({item.asset_id for item in evidence}) != len(evidence):
        raise ValueError("Portfolio checkpoint evidence contains duplicate assets.")
    root = Path(release_root).resolve(strict=True)
    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"Portfolio checkpoint evidence destination already exists: {output}")
    staged: list[Path] = []
    for reference in evidence:
        dataset = _episode_dataset_for_reference(release_root=root, evidence=reference)
        if dataset.asset_id != reference.asset_id or dataset.phase_id not in {"phase1", "phase2", "phase3"}:
            raise ValueError("Portfolio evidence reference and episode manifest disagree.")
        asset_root = output / dataset.asset_id
        asset_root.mkdir(parents=True)
        for artifacts in (dataset.files, dataset.metadata_files):
            for artifact in artifacts:
                source = _release_member(root, artifact.relative_path)
                _require_checksum(source, artifact.sha256)
                shutil.copy2(source, asset_root / source.name)
        summary = phase_summary_v1(
            scenario_root=Path(scenario_public_root).parent,
            trial_output_root=asset_root,
        )
        write_json(asset_root / "phase_summary_public.json", summary)
        staged.append(asset_root)
    return tuple(staged)


__all__ = [
    "initial_portfolio_state_v1",
    "load_portfolio_catalogue_v1",
    "load_portfolio_manifest_v1",
    "portfolio_evaluator_view_v1",
    "portfolio_participant_view_v1",
    "stage_portfolio_checkpoint_evidence_v1",
    "stage_portfolio_public_view_v1",
    "validate_portfolio_release_v1",
]
