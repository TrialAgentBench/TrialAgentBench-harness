"""Scientific reconstruction checks tolerate storage changes, not data changes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from trialagentbench_verifier.reconstruction import verify_analysis_data_v1


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_reconstruction_accepts_equivalent_layouts(tmp_path: Path, extension: str) -> None:
    expected = pd.DataFrame(
        {"participant_id": ["001", "002"], "occurrence_time": [10.0, 10.0], "endpoint": [np.nan, 2.0]}
    )
    observed = expected.iloc[::-1, ::-1].assign(extra_analysis_term=42)
    path = tmp_path / f"analysis_data.{extension}"
    if extension == "csv":
        observed.to_csv(path, index=False)
    else:
        observed.to_parquet(path, index=False)
    result = verify_analysis_data_v1(
        scratch=tmp_path, expected=expected, key_fields=("participant_id", "occurrence_time")
    )
    assert result.status == "verified"


@pytest.mark.parametrize("change", ["population", "duplicate", "endpoint", "missingness", "visit", "absent_column"])
def test_reconstruction_rejects_changed_scientific_records(tmp_path: Path, change: str) -> None:
    expected = pd.DataFrame(
        {"participant_id": ["p1", "p2"], "occurrence_time": [10.0, 10.0], "endpoint": [np.nan, 2.0]}
    )
    observed = expected.copy()
    if change == "population":
        observed = observed.iloc[:1]
    elif change == "duplicate":
        observed = pd.concat([observed, observed.iloc[:1]])
    elif change == "endpoint":
        observed.loc[1, "endpoint"] = 3.0
    elif change == "missingness":
        observed.loc[0, "endpoint"] = 0.0
    elif change == "visit":
        observed.loc[1, "occurrence_time"] = 20.0
    else:
        observed = observed.drop(columns="endpoint")
    observed.to_parquet(tmp_path / "analysis_data.parquet", index=False)
    result = verify_analysis_data_v1(
        scratch=tmp_path, expected=expected, key_fields=("participant_id", "occurrence_time")
    )
    assert result.status != "verified"


def test_reconstruction_requires_one_regular_table_and_valid_reference(tmp_path: Path) -> None:
    expected = pd.DataFrame({"participant_id": ["p1"], "endpoint": [1.0]})
    assert (
        verify_analysis_data_v1(scratch=tmp_path, expected=expected, key_fields=("participant_id",)).status
        == "missing"
    )
    expected.to_csv(tmp_path / "elsewhere.csv", index=False)
    (tmp_path / "analysis_data.csv").symlink_to(tmp_path / "elsewhere.csv")
    assert (
        verify_analysis_data_v1(scratch=tmp_path, expected=expected, key_fields=("participant_id",)).status
        == "invalid"
    )
    expected.to_parquet(tmp_path / "analysis_data.parquet", index=False)
    assert (
        verify_analysis_data_v1(scratch=tmp_path, expected=expected, key_fields=("participant_id",)).status
        == "ambiguous"
    )
    with pytest.raises(ValueError, match="unique scientific keys"):
        verify_analysis_data_v1(
            scratch=tmp_path, expected=pd.concat([expected, expected]), key_fields=("participant_id",)
        )
