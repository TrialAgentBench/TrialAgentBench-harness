"""Tests for the published validation distribution."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.mark.slow
def test_wheel_contains_a_verifiable_validation_bundle(tmp_path: Path) -> None:
    """A wheel built through the sdist contains every checksummed result."""

    package_root = Path(__file__).resolve().parents[2]
    dist = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(dist)],
        cwd=package_root,
        check=True,
    )
    wheels = tuple(dist.glob("*.whl"))
    assert len(wheels) == 1

    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
        assert "trialagentbench_validation/validation_results/SOURCES.md" in names
        assert (
            "trialagentbench_validation/validation_results/figures/outcome_survival.pdf"
            in names
        )
        archive.extractall(installed)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from trialagentbench_validation.external.release.bundle import "
                "verify_installed_validation_bundle; "
                "verify_installed_validation_bundle()"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    regenerated = tmp_path / "regenerated"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "trialagentbench_validation.validation_figures.report",
            "--output-dir",
            str(regenerated),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    installed_figures = (
        installed / "trialagentbench_validation" / "validation_results" / "figures"
    )
    for path in regenerated.iterdir():
        assert path.read_bytes() == (installed_figures / path.name).read_bytes()
