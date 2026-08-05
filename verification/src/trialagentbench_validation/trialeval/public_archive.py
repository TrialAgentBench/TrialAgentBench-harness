"""Canonical participant-archive member resolution."""

from __future__ import annotations

from pathlib import PurePosixPath
from zipfile import ZipFile


def _canonical_semantic_member_v1(member: str) -> str:
    if not member or "\\" in member or ":" in member:
        raise ValueError(
            "participant semantic path must be a non-empty POSIX relative path"
        )
    path = PurePosixPath(member)
    if (
        path.is_absolute()
        or path.as_posix() != member
        or any(part in {"", ".", ".."} or part.startswith(".") for part in path.parts)
    ):
        raise ValueError(
            "participant semantic path must be canonical and remain inside its archive root"
        )
    return member


def resolve_public_member_v1(public: ZipFile, member: str) -> str:
    """Resolve one semantic participant path across supported archive roots."""

    member = _canonical_semantic_member_v1(member)
    candidates = (member, f"public/{member}")
    present = tuple(
        candidate for candidate in candidates if candidate in public.NameToInfo
    )
    if not present:
        raise FileNotFoundError(f"Missing public input: {member}")
    if len(present) != 1:
        raise ValueError(f"Ambiguous public input path: {member}")
    return present[0]


def public_member_exists_v1(public: ZipFile, member: str) -> bool:
    """Return whether one semantic path has exactly one participant member."""

    member = _canonical_semantic_member_v1(member)
    present = tuple(
        candidate
        for candidate in (member, f"public/{member}")
        if candidate in public.NameToInfo
    )
    if len(present) > 1:
        raise ValueError(f"Ambiguous public input path: {member}")
    return bool(present)


def participant_semantic_member_names_v1(public: ZipFile) -> frozenset[str]:
    """Return participant member names without the optional public-root prefix."""

    names: set[str] = set()
    for member in public.namelist():
        if member.endswith("/"):
            continue
        _canonical_semantic_member_v1(member)
        semantic = member.removeprefix("public/")
        _canonical_semantic_member_v1(semantic)
        if semantic in names:
            raise ValueError(f"Ambiguous participant archive member: {semantic}")
        names.add(semantic)
    return frozenset(names)
