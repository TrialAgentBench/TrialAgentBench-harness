"""Safe materialization of paired participant and evaluator release archives."""

from __future__ import annotations

import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

from trialagentbench_harness.contracts.release.artifacts import (
    TRIALDEV_EVALUATOR_ARCHIVE_NAME,
    TRIALDEV_PARTICIPANT_ARCHIVE_NAME,
)


def _validate_member(info: ZipInfo, *, archive_path: Path) -> None:
    """Reject archive members that could escape or alter the extraction root."""

    member = PurePosixPath(info.filename)
    if not info.filename or member.is_absolute() or ".." in member.parts or "\x00" in info.filename:
        raise ValueError(f"Unsafe ZIP member in {archive_path}: {info.filename!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"ZIP symlinks are not allowed in {archive_path}: {info.filename!r}")


def _extract_zip(archive_path: Path, destination: Path) -> None:
    """Extract one validated ZIP into an existing destination."""

    try:
        with ZipFile(archive_path) as archive:
            infos = archive.infolist()
            names = tuple(info.filename for info in infos)
            if len(names) != len(set(names)):
                raise ValueError(f"ZIP contains duplicate members: {archive_path}")
            for info in infos:
                _validate_member(info, archive_path=archive_path)
            archive.extractall(destination)
    except BadZipFile as exc:
        raise ValueError(f"Invalid release ZIP: {archive_path}") from exc


@contextmanager
def materialized_paired_release_root(
    source: Path,
    *,
    evaluator_archive_name: str,
    participant_archive_name: str,
    participant_subdirectory: str | None = None,
) -> Iterator[Path]:
    """Yield a directory release or safely merge one canonical archive pair."""

    source = Path(source).resolve()
    if source.is_dir():
        yield source
        return
    if not source.is_file():
        raise FileNotFoundError(f"Release root not found: {source}")
    if source.name != evaluator_archive_name:
        raise ValueError(f"Evaluator ZIP must be named {evaluator_archive_name!r}: {source}")
    participant = source.with_name(participant_archive_name)
    if not participant.is_file():
        raise FileNotFoundError(f"Paired participant ZIP not found beside evaluator ZIP: {participant}")
    with tempfile.TemporaryDirectory(prefix="trialagentbench_release_") as temporary:
        root = Path(temporary) / "release"
        root.mkdir(parents=True, exist_ok=False)
        participant_root = root if participant_subdirectory is None else root / participant_subdirectory
        participant_root.mkdir(parents=True, exist_ok=participant_subdirectory is None)
        _extract_zip(participant, participant_root)
        _extract_zip(source, root)
        yield root


@contextmanager
def materialized_trialdev_release_root(source: Path) -> Iterator[Path]:
    """Yield the scenario-root directory from one TrialDev release pair."""

    with materialized_paired_release_root(
        source,
        evaluator_archive_name=TRIALDEV_EVALUATOR_ARCHIVE_NAME,
        participant_archive_name=TRIALDEV_PARTICIPANT_ARCHIVE_NAME,
    ) as root:
        if any(child.is_dir() and child.name.startswith("scenario_") for child in root.iterdir()):
            yield root
            return
        children = [child for child in root.iterdir() if child.is_dir()]
        if len(children) == 1 and any(
            child.is_dir() and child.name.startswith("scenario_") for child in children[0].iterdir()
        ):
            yield children[0]
            return
        raise FileNotFoundError(f"TrialDev release does not contain scenario_* roots: {source}")


__all__ = ["materialized_paired_release_root", "materialized_trialdev_release_root"]
