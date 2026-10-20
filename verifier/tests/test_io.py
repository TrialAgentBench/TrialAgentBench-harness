"""Hostile-input tests for public verifier file custody."""

from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from pathlib import Path

import pytest

from trialagentbench_verifier import io as verifier_io


@pytest.mark.parametrize("shared", [False, True])
def test_authenticated_archive_parses_the_bytes_opened_for_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared: bool,
) -> None:
    """Reject a caller path swapped before the verifier acquires its private copy."""

    archive = tmp_path / "release.zip"
    original = b"trusted release bytes"
    replacement = b"substituted release bytes"
    archive.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    real_open = os.open
    swapped = False

    def swapping_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if Path(path) == archive and not swapped:
            swapped = True
            archive.write_bytes(replacement)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(verifier_io.os, "open", swapping_open)
    with verifier_io.shared_release_archives(tmp_path) if shared else nullcontext():
        with pytest.raises(verifier_io.VerificationInputError, match="externally trusted SHA-256"):
            with verifier_io.authenticated_archive(archive, expected_sha256=expected):
                pytest.fail("a substituted archive must not be yielded")


def test_authenticated_archive_rejects_symlinks_and_special_files(tmp_path: Path) -> None:
    """Reject filesystem indirection before attempting archive reads."""

    target = tmp_path / "target.zip"
    target.write_bytes(b"zip")
    linked = tmp_path / "linked.zip"
    linked.symlink_to(target)
    with pytest.raises(verifier_io.VerificationInputError, match="regular non-symlinked"):
        with verifier_io.authenticated_archive(linked, expected_sha256=hashlib.sha256(b"zip").hexdigest()):
            pytest.fail("a symlinked archive must not be yielded")

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "release.fifo"
        os.mkfifo(fifo)
        with pytest.raises(verifier_io.VerificationInputError, match="regular non-symlinked"):
            with verifier_io.authenticated_archive(fifo, expected_sha256="0" * 64):
                pytest.fail("a FIFO must not be yielded")


@pytest.mark.parametrize("fail", [False, True])
def test_shared_archives_reuse_authenticated_extraction_across_threads_and_cleanup(tmp_path, monkeypatch, fail):
    """One consumer action extracts once and releases custody on success or failure."""
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", "{}")
        handle.writestr("data.txt", "trusted observed data")
    expected = verifier_io.sha256_file(archive)
    extracted = []
    original = zipfile.ZipFile.extractall

    def record(self, path, *args, **kwargs):
        extracted.append(Path(path))
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "extractall", record)

    def read(_):
        with verifier_io.authenticated_archive(archive, expected_sha256=expected) as snapshot:
            with verifier_io.materialized_release(snapshot) as root:
                assert (root / "data.txt").read_text() == "trusted observed data"
                return snapshot, root

    class ConsumerFailure(RuntimeError):
        pass

    try:
        with verifier_io.shared_release_archives(tmp_path):
            with ThreadPoolExecutor(max_workers=4) as pool:
                outputs = list(pool.map(read, range(8)))
            assert len(set(outputs)) == 1
            assert len(extracted) == 1
            assert all(path.exists() for path in outputs[0])
            if fail:
                raise ConsumerFailure()
    except ConsumerFailure:
        assert fail
    assert all(not path.exists() for path in outputs[0])
    with verifier_io.shared_release_archives(tmp_path):
        second = read(0)
        assert second != outputs[0]
    assert len(extracted) == 2
    assert all(not path.exists() for path in second)


def test_shared_archives_reject_changed_source_and_wrong_trust_anchor(tmp_path):
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"trusted")
    expected = verifier_io.sha256_file(archive)
    with verifier_io.shared_release_archives(tmp_path):
        with verifier_io.authenticated_archive(archive, expected_sha256=expected) as snapshot:
            assert snapshot.read_bytes() == b"trusted"
        with pytest.raises(verifier_io.VerificationInputError, match="trusted SHA-256"):
            with verifier_io.authenticated_archive(archive, expected_sha256="0" * 64):
                pytest.fail("different trust anchor reused a cached snapshot")
        # Length changes are observable even on filesystems with coarse timestamps.
        archive.write_bytes(b"changed-source")
        with pytest.raises(verifier_io.VerificationInputError, match="changed during"):
            with verifier_io.authenticated_archive(archive, expected_sha256=expected):
                pytest.fail("changed source silently reused a snapshot")
        assert snapshot.read_bytes() == b"trusted"


def test_shared_scope_never_caches_an_unauthenticated_zip(tmp_path):
    import zipfile

    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", "{}")
    with verifier_io.shared_release_archives(tmp_path):
        with verifier_io.materialized_release(archive) as first:
            assert first.is_dir()
        assert not first.exists()
        with verifier_io.materialized_release(archive) as second:
            assert second != first
        with pytest.raises(RuntimeError, match="already owns"):
            with verifier_io.shared_release_archives(tmp_path):
                pytest.fail("overlapping scope replaced another consumer's custody")


def test_shared_inputs_do_not_retain_result_archives_and_reuse_nested_authentication(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    archive = inputs / "release.zip"
    archive.write_bytes(b"input")
    result = tmp_path / "result.zip"
    result.write_bytes(b"result")
    with verifier_io.shared_release_archives(inputs):
        with verifier_io.authenticated_archive(archive, expected_sha256=verifier_io.sha256_file(archive)) as first:
            with verifier_io.authenticated_archive(first, expected_sha256=verifier_io.sha256_file(first)) as nested:
                assert nested == first
        with verifier_io.authenticated_archive(result, expected_sha256=verifier_io.sha256_file(result)) as transient:
            assert transient.read_bytes() == b"result"
        assert not transient.exists()
        assert first.exists()
    assert not first.exists()
