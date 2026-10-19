"""Read public TrialAgentBench release files the way a user does, for the worked demonstrations.

Every demonstration reads the published dataset and nothing else. Small files (checksums, catalogs,
task inventories and the release-evidence record) come from a local copy of the dataset repository
(``--release-root``) or are downloaded once from the Hugging Face dataset into the cache. Every one of
them is checked against the dataset's root ``SHA256SUMS``. Task payloads are acquired shard by shard
with the verifier's asset cache, which authenticates each shard against its catalog before use.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict
from trialagentbench_verifier.asset_cache import acquire_asset_selection
from trialagentbench_verifier.assets import ReleaseAssets, read_asset_catalog

DATASET_URL = "https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/resolve"
DEFAULT_REVISION = "v1.0.0"
_SMALL_FILE_LIMIT = 256 * 1024**2

Benchmark = Literal["trialeval", "trialdev"]
Audience = Literal["participant", "evaluator"]


def add_release_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared release-source options to a demonstration's command line.

    Parameters
    ----------
    parser
        The demonstration's argument parser.
    """
    parser.add_argument(
        "--release-root",
        type=Path,
        help="Local copy of the dataset repository (with SHA256SUMS and full/); omit to download.",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Dataset revision to download.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("TRIALAGENTBENCH_CACHE_DIR", Path.home() / ".cache" / "trialagentbench")),
        help="Operator cache for downloaded files and verified shards (default: $TRIALAGENTBENCH_CACHE_DIR).",
    )


class _Record(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class TrapCost(_Record):
    """Expected regret of the excluded (routine) analysis in one trap cell, over the policy-panel worlds."""

    regime_cell_id: str
    regret_musd: float
    regret_se_musd: float
    consequential_share: float


class OverCorrection(_Record):
    """Expected regret of the tempting correction in one twin cell, where the default is valid."""

    regime_cell_id: str
    consequential_share: float
    mean_regret_musd: float


class TrapConsequence(_Record):
    """How far the excluded analysis moves the estimate in a trap cell, by tertile of the true effect."""

    regime_cell_id: str
    shift_consequential: bool
    interval_consequential: bool
    width_ratio: float
    tertile_bias_sd: tuple[float, float, float]


class PolicyPanel(_Record):
    """Fixed answer policies run over prior-drawn worlds of every public cell."""

    worlds_per_cell: int
    cells: int
    failing_cells: dict[str, int]
    naive_costs: tuple[TrapCost, ...]


class Economics(_Record):
    """Failing cells of every policy at each point of the value and cost sensitivity band."""

    band_failing_cells: dict[str, dict[str, int]]


class TrialEvalEvidence(_Record):
    """The TrialEval part of the release-evidence record read by the demonstrations."""

    consequence: tuple[TrapConsequence, ...]
    over_corrections: tuple[OverCorrection, ...]
    policy_public: PolicyPanel
    economics: Economics


class GradientStep(_Record):
    """Mean change in realised value between two budget rungs, with its 95% interval over worlds."""

    mean_musd: float
    lower_musd: float
    upper_musd: float
    worlds: int


class PolicyGradient(_Record):
    """One TrialDev policy's realised value at the three budget rungs of a programme."""

    realised_mean_by_rung_musd: tuple[float, float, float]
    constrained_to_lean: GradientStep
    lean_to_generous: GradientStep


class ProgrammeGradient(_Record):
    """The resource-gradient panel of one clinical programme at the declared profit scale."""

    ladder_usd: tuple[float, float, float]
    oracle: PolicyGradient
    reference: PolicyGradient
    mean_musd_by_policy: dict[str, float]


class TrialDevEvidence(_Record):
    """The TrialDev part of the release-evidence record read by the demonstrations."""

    gradient: dict[str, ProgrammeGradient] | None = None


class ReleaseEvidence(_Record):
    """The fields of ``release-evidence/release-evidence.json`` read by the demonstrations.

    Field names follow the published record; fields the demonstrations do not read are ignored. The
    TrialDev gradient is read only by the budget demonstration, which stops if the record lacks it.
    """

    trialeval: TrialEvalEvidence
    trialdev: TrialDevEvidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PublicRelease:
    """One published dataset revision, local or downloaded on demand.

    Parameters
    ----------
    root
        Directory holding the dataset files (the local copy, or the cache mirror of a revision).
    cache
        Operator cache for verified shards.
    base_url
        HTTPS directory of the revision, or ``None`` to use local files only.

    Examples
    --------
    >>> release = PublicRelease.open(release_root=None, revision="v1.0.0", cache=Path("cache"))  # doctest: +SKIP
    >>> record = release.evidence()  # doctest: +SKIP
    """

    root: Path
    cache: Path
    base_url: str | None

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> PublicRelease:
        """Open the release named by ``add_release_arguments`` options.

        Parameters
        ----------
        arguments
            Parsed command line.

        Returns
        -------
        PublicRelease
            The selected release.
        """
        return cls.open(release_root=arguments.release_root, revision=arguments.revision, cache=arguments.cache)

    @classmethod
    def open(cls, *, release_root: Path | None, revision: str, cache: Path) -> PublicRelease:
        """Select a local dataset copy, or a cache mirror of a published revision.

        Parameters
        ----------
        release_root
            Local copy of the dataset repository, or ``None`` to download ``revision``.
        revision
            Published dataset revision (a tag or commit).
        cache
            Operator cache directory.

        Returns
        -------
        PublicRelease
            The selected release.

        Raises
        ------
        FileNotFoundError
            If a local copy lacks its root checksum file.
        """
        cache = cache.absolute()
        if release_root is not None:
            root = release_root.absolute()
            if not (root / "SHA256SUMS").is_file():
                raise FileNotFoundError(f"{root} is not a dataset copy: SHA256SUMS is missing")
            return cls(root=root, cache=cache, base_url=None)
        return cls(
            root=cache / "dataset" / revision.replace("/", "_"),
            cache=cache,
            base_url=f"{DATASET_URL}/{revision}",
        )

    def _checksums(self) -> dict[str, str]:
        path = self.root / "SHA256SUMS"
        if not path.is_file():
            self._download("SHA256SUMS", path)
        lines = path.read_text(encoding="ascii").splitlines()
        return {name: digest for digest, name in (line.split("  ", 1) for line in lines if line)}

    def _download(self, relative: str, target: Path) -> None:
        if self.base_url is None:
            raise FileNotFoundError(f"{relative} is missing from the local dataset copy {self.root}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", delete=False) as partial:
            try:
                with urlopen(f"{self.base_url}/{relative}", timeout=60) as response:
                    total = 0
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > _SMALL_FILE_LIMIT:
                            raise ValueError(f"{relative} exceeds the small-file download limit")
                        partial.write(block)
                partial.close()
                os.replace(partial.name, target)
            finally:
                if os.path.exists(partial.name):
                    os.unlink(partial.name)

    def file(self, relative: str) -> Path:
        """Return a small dataset file after checking it against the root ``SHA256SUMS``.

        Parameters
        ----------
        relative
            Path inside the dataset repository, for example ``release-evidence/release-evidence.json``.

        Returns
        -------
        Path
            The verified local file.

        Raises
        ------
        ValueError
            If the file is not listed in, or differs from, the root checksums.
        """
        expected = self._checksums().get(relative)
        if expected is None:
            raise ValueError(f"{relative} is not listed in the dataset's SHA256SUMS")
        path = self.root / relative
        if not path.is_file():
            self._download(relative, path)
        if _sha256(path) != expected:
            raise ValueError(f"{relative} differs from its published SHA-256")
        return path

    def evidence(self) -> ReleaseEvidence:
        """Return the release-evidence record behind every published number.

        Returns
        -------
        ReleaseEvidence
            The fields of ``release-evidence/release-evidence.json`` that the demonstrations read.
        """
        return ReleaseEvidence.model_validate_json(self.file("release-evidence/release-evidence.json").read_bytes())

    def catalog(self, benchmark: Benchmark, audience: Audience) -> tuple[Path, str, ReleaseAssets]:
        """Return an authenticated public asset catalog.

        Parameters
        ----------
        benchmark
            ``trialeval`` or ``trialdev``.
        audience
            ``participant`` or ``evaluator`` (public evaluator records, never mounted for an agent).

        Returns
        -------
        tuple
            Catalog path, its published SHA-256 and the parsed catalog.
        """
        relative = f"full/{benchmark}-{audience}.assets.json"
        path = self.file(relative)
        digest = self._checksums()[relative]
        return path, digest, read_asset_catalog(path, expected_sha256=digest)

    def members(self, benchmark: Benchmark, audience: Audience, paths: Iterable[str]) -> dict[str, bytes]:
        """Acquire only the shards holding ``paths`` and return their verified bytes.

        Parameters
        ----------
        benchmark, audience
            As for ``catalog``.
        paths
            Logical release paths, for example ``evaluation/grader/task_index.json``.

        Returns
        -------
        dict
            Exact bytes of each requested path.

        Raises
        ------
        KeyError
            If a path is not part of the catalog.
        ValueError
            If a payload differs from its catalog digest.
        """
        catalog_path, digest, catalog = self.catalog(benchmark, audience)
        index = {member.path: member for member in catalog.members}
        wanted = [index[path] for path in paths]
        # Shared groups come with every selection; task groups need a task that names them.
        needed = {member.group for member in wanted if not member.group.endswith("-shared")}
        tasks = tuple(task.task_id for task in catalog.tasks if needed.intersection(task.groups)) or (
            catalog.tasks[0].task_id,
        )
        shard_base = None if self.base_url is None else f"{self.base_url}/full"
        cached = acquire_asset_selection(
            catalog_path, expected_sha256=digest, cache=self.cache, task_ids=tasks, base_url=shard_base
        )
        shards = {shard.group: cached.parent / shard.filename for shard in catalog.shards}
        payloads: dict[str, bytes] = {}
        for member in wanted:
            with zipfile.ZipFile(shards[member.group]) as archive:
                body = archive.read(f"objects/{member.sha256}")
            if len(body) != member.byte_length or hashlib.sha256(body).hexdigest() != member.sha256:
                raise ValueError(f"{member.path} differs from its catalog digest")
            payloads[member.path] = body
        return payloads
