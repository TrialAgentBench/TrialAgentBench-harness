"""Fail-closed inspection of public release ZIP archives."""

from __future__ import annotations

import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

NESTED_ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".7z",
    ".bz2",
    ".gz",
    ".rar",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
)


@dataclass(frozen=True)
class ReleaseArchiveIssue:
    """One structural or integrity issue in a release archive."""

    code: str
    member: str | None
    message: str


def _canonical_member_path(name: str) -> PurePosixPath:
    if not name or "\\" in name or re.match(r"^[A-Za-z]:", name):
        raise ValueError("archive member path must be a non-empty POSIX relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("archive member path must not be absolute or traverse parents")
    if path.as_posix() != name.rstrip("/"):
        raise ValueError("archive member path must be canonically normalized")
    return path


def inspect_release_zip(path: Path) -> tuple[ReleaseArchiveIssue, ...]:
    """Return structural and integrity issues for one release ZIP."""

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        return (
            ReleaseArchiveIssue(
                code="invalid",
                member=None,
                message="release archive must be a regular non-symlink file",
            ),
        )
    issues: list[ReleaseArchiveIssue] = []
    try:
        with zipfile.ZipFile(path) as archive:
            observed: set[str] = set()
            for info in archive.infolist():
                try:
                    member = _canonical_member_path(info.filename)
                except ValueError as exc:
                    issues.append(ReleaseArchiveIssue(code="unsafe_path", member=info.filename, message=str(exc)))
                    continue
                normalized = member.as_posix()
                if normalized in observed:
                    issues.append(
                        ReleaseArchiveIssue(
                            code="duplicate_member",
                            member=info.filename,
                            message="archive contains a duplicate normalized member path",
                        )
                    )
                observed.add(normalized)
                file_type = stat.S_IFMT(info.external_attr >> 16)
                if file_type == stat.S_IFLNK:
                    issues.append(
                        ReleaseArchiveIssue(
                            code="link",
                            member=info.filename,
                            message="archive members must not be symbolic links",
                        )
                    )
                elif file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    issues.append(
                        ReleaseArchiveIssue(
                            code="special_file",
                            member=info.filename,
                            message="archive members must be regular files or directories",
                        )
                    )
                if info.flag_bits & 0x1:
                    issues.append(
                        ReleaseArchiveIssue(
                            code="encrypted_member",
                            member=info.filename,
                            message="release archives must not contain encrypted members",
                        )
                    )
                if not info.is_dir() and normalized.lower().endswith(NESTED_ARCHIVE_SUFFIXES):
                    issues.append(
                        ReleaseArchiveIssue(
                            code="nested_archive",
                            member=info.filename,
                            message="release role archives must not contain nested archives",
                        )
                    )
            corrupt = archive.testzip()
            if corrupt is not None:
                issues.append(
                    ReleaseArchiveIssue(
                        code="corrupt_member",
                        member=corrupt,
                        message="archive member failed its CRC check",
                    )
                )
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        issues.append(ReleaseArchiveIssue(code="invalid", member=None, message=f"cannot read release archive: {exc}"))
    return tuple(issues)


__all__ = ["ReleaseArchiveIssue", "inspect_release_zip"]
