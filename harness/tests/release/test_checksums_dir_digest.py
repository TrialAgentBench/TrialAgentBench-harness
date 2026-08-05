from __future__ import annotations

from pathlib import Path

import pytest

from trialagentbench_harness.io import sha256_dir_digest, sha256_dir_files, sha256_file, sha256_path


def test_sha256_dir_files_and_digest_are_deterministic(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("c", encoding="utf-8")

    m1 = sha256_dir_files(tmp_path)
    m2 = sha256_dir_files(tmp_path)
    assert m1 == m2

    d1 = sha256_dir_digest(tmp_path)
    d2 = sha256_dir_digest(tmp_path)
    assert d1 == d2

    # Mutating any file must change the digest.
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    assert sha256_dir_digest(tmp_path) != d1


def test_sha256_path_dispatches_without_ambiguity(tmp_path: Path) -> None:
    file_path = tmp_path / "input.json"
    file_path.write_text("{}", encoding="utf-8")

    assert sha256_path(file_path) == sha256_file(file_path)
    assert sha256_path(tmp_path) == sha256_dir_digest(tmp_path)


def test_sha256_path_rejects_symlinks_and_missing_paths(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        sha256_path(link)
    with pytest.raises(FileNotFoundError):
        sha256_path(tmp_path / "missing")
