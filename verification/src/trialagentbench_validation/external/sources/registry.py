"""Verification of frozen external-source bytes."""

from __future__ import annotations

import subprocess
from pathlib import Path

from trialagentbench_validation.external.contracts import (
    ExternalSourceManifestV1,
    LicenseStatus,
)
from trialagentbench_validation.io import sha256_file


def verify_source_manifest(
    manifest: ExternalSourceManifestV1,
    *,
    local_paths: dict[str, Path],
) -> None:
    """Verify source checksums, repository commits, and redistribution boundaries."""

    expected_ids = {source.source_id for source in manifest.sources}
    if set(local_paths) != expected_ids:
        raise ValueError("local source paths must exactly match the source manifest")
    for source in manifest.sources:
        path = local_paths[source.source_id].resolve()
        if source.source_type == "rct_bench":
            if not path.is_dir():
                raise ValueError(f"RCT Bench source is not a directory: {path}")
            commit = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if commit != source.snapshot_identity:
                raise ValueError(
                    f"RCT Bench commit drift: expected {source.snapshot_identity}, observed {commit}"
                )
            tree_digest = _tree_digest(path)
            if tree_digest != source.sha256:
                raise ValueError(
                    f"RCT Bench tree checksum drift: expected {source.sha256}, observed {tree_digest}"
                )
        else:
            if not path.is_file():
                raise ValueError(f"external source is not a file: {path}")
            observed = sha256_file(path)
            if observed != source.sha256:
                raise ValueError(
                    f"source checksum drift for {source.source_id}: expected {source.sha256}, observed {observed}"
                )
        if (
            source.license_status == LicenseStatus.ACQUISITION_ONLY
            and "not redistribut" not in (source.redistribution_rationale.lower())
        ):
            raise ValueError(
                "acquisition-only sources must explicitly prohibit redistribution"
            )


def _tree_digest(root: Path) -> str:
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    import hashlib

    return hashlib.sha256(listing.encode()).hexdigest()


__all__ = ["verify_source_manifest"]
