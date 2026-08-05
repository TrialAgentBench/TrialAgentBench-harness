"""Installed-package reproduction test for trial characterisation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_public_example_regenerates_equivalent_result(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[2]
    example_root = package_root / "examples" / "characterisation"
    output = tmp_path / "characterisation.csv"
    subprocess.run(
        [
            sys.executable,
            str(example_root / "run.py"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
    )
    observed = pd.read_csv(output)
    expected = pd.read_csv(example_root / "expected_characterisation.csv")
    assert tuple(observed.columns) == tuple(expected.columns)
    assert observed.shape == expected.shape
    numeric = tuple(expected.select_dtypes(include="number").columns)
    categorical = tuple(column for column in expected.columns if column not in numeric)
    pd.testing.assert_frame_equal(
        observed.loc[:, categorical],
        expected.loc[:, categorical],
        check_dtype=False,
    )
    np.testing.assert_allclose(
        observed.loc[:, numeric],
        expected.loc[:, numeric],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
