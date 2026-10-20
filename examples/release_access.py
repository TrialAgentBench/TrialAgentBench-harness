"""Read public TrialAgentBench release files the way a user does, for the worked demonstrations.

Every demonstration that reads release data reads the published dataset and nothing else. Its small files
(the checksums and the release-evidence record) come from a local copy of the dataset repository
(``--release-root``) or are downloaded once from the Hugging Face dataset into the cache, and every one of
them is checked against the dataset's root ``SHA256SUMS``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, model_validator

DATASET_URL = "https://huggingface.co/datasets/TrialAgentBench/TrialAgentBench/resolve"
DEFAULT_REVISION = "v1.0.0"
_SMALL_FILE_LIMIT = 256 * 1024**2


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
        help="Local copy of the dataset repository (with SHA256SUMS); omit to download.",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Dataset revision to download.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("TRIALAGENTBENCH_CACHE_DIR", Path.home() / ".cache" / "trialagentbench")),
        help="Operator cache for downloaded files (default: $TRIALAGENTBENCH_CACHE_DIR).",
    )


class _Record(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class PolicyPanel(_Record):
    """Fixed answer policies run over prior-drawn worlds of every cell of one split."""

    worlds_per_cell: int
    cells: int
    failing_cells: dict[str, int]


class Economics(_Record):
    """Public cells failed by every policy at each point of the value and cost sensitivity band."""

    band_failing_cells: dict[str, dict[str, int]]


class WithheldPolicyOutcomes(_Record):
    """Whether each policy fails at least one withheld cell at each band point; no failed-cell counts."""

    fails_a_cell: dict[str, dict[str, bool]]


class TrialEvalEvidence(_Record):
    """The TrialEval part of the release-evidence record read by the demonstrations."""

    policy_public: PolicyPanel
    policy_withheld: WithheldPolicyOutcomes
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
    gradient_status: Literal["pending", "present"] = "pending"

    @model_validator(mode="after")
    def consistent_gradient_status(self) -> Self:
        """Refuse a pending state that carries a result; infer the state for older public records."""
        expected = "pending" if self.gradient is None else "present"
        if "gradient_status" in self.model_fields_set and self.gradient_status != expected:
            raise ValueError("TrialDev gradient status disagrees with its result")
        object.__setattr__(self, "gradient_status", expected)
        return self


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
        Operator cache for downloaded files.
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
