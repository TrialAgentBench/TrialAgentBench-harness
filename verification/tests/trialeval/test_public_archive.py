"""Boundary tests for participant semantic archive paths."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from trialagentbench_validation.trialeval.public_archive import (
    participant_semantic_member_names_v1,
    public_member_exists_v1,
    resolve_public_member_v1,
)


def _archive(path: Path, *members: str) -> Path:
    with ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, b"example")
    return path


@pytest.mark.parametrize(
    "member",
    (
        "/etc/passwd",
        "../task.json",
        "items/../task.json",
        r"items\\task.json",
        "C:/task.json",
        "./task.json",
        ".hidden",
    ),
)
def test_semantic_member_resolution_rejects_noncanonical_paths(
    tmp_path: Path, member: str
) -> None:
    path = _archive(tmp_path / "participant.zip", "items/TASK1/task.json")
    with ZipFile(path) as archive:
        with pytest.raises(ValueError, match="participant semantic path"):
            resolve_public_member_v1(archive, member)
        with pytest.raises(ValueError, match="participant semantic path"):
            public_member_exists_v1(archive, member)


def test_semantic_member_resolution_supports_exactly_one_declared_root(
    tmp_path: Path,
) -> None:
    flat = _archive(tmp_path / "flat.zip", "items/TASK1/task.json")
    rooted = _archive(tmp_path / "rooted.zip", "public/items/TASK1/task.json")

    with ZipFile(flat) as archive:
        assert (
            resolve_public_member_v1(archive, "items/TASK1/task.json")
            == "items/TASK1/task.json"
        )
    with ZipFile(rooted) as archive:
        assert (
            resolve_public_member_v1(archive, "items/TASK1/task.json")
            == "public/items/TASK1/task.json"
        )


def test_semantic_member_resolution_rejects_ambiguous_roots(tmp_path: Path) -> None:
    path = _archive(
        tmp_path / "ambiguous.zip",
        "items/TASK1/task.json",
        "public/items/TASK1/task.json",
    )
    with ZipFile(path) as archive:
        with pytest.raises(ValueError, match="Ambiguous"):
            resolve_public_member_v1(archive, "items/TASK1/task.json")
        with pytest.raises(ValueError, match="Ambiguous"):
            participant_semantic_member_names_v1(archive)


def test_semantic_name_inventory_rejects_unsafe_archive_members(tmp_path: Path) -> None:
    path = _archive(tmp_path / "unsafe.zip", "../task.json")
    with ZipFile(path) as archive:
        with pytest.raises(ValueError, match="participant semantic path"):
            participant_semantic_member_names_v1(archive)
