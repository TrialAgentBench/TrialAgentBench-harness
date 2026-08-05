"""CDISC reference workflow tests."""

from __future__ import annotations

import pandas as pd

from trialagentbench_validation.external.sources.cdisc import (
    _key_violations,
    _transport_mismatches,
)


def test_transport_comparison_handles_sas_dates_and_detects_cell_drift() -> None:
    xpt = pd.DataFrame(
        {
            "USUBJID": ["S1", "S2"],
            "TRTSDT": [0.0, 1.0],
            "AVAL": [1.0, 2.0],
        }
    )
    dataset_json = pd.DataFrame(
        {
            "USUBJID": ["S1", "S2"],
            "TRTSDT": ["1960-01-01", "1960-01-02"],
            "AVAL": [1.0, 2.0],
        }
    )
    metadata = {
        "USUBJID": {"dataType": "string"},
        "TRTSDT": {"dataType": "date"},
        "AVAL": {"dataType": "float"},
    }

    assert _transport_mismatches(xpt, dataset_json, metadata=metadata) == 0

    dataset_json.loc[1, "AVAL"] = 3.0
    assert _transport_mismatches(xpt, dataset_json, metadata=metadata) == 1


def test_key_comparison_detects_missing_and_duplicate_identifiers() -> None:
    valid = pd.DataFrame(
        {
            "STUDYID": ["A", "A"],
            "USUBJID": ["S1", "S2"],
        }
    )
    invalid = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
    invalid.loc[1, "USUBJID"] = None

    assert _key_violations(valid, keys=("STUDYID", "USUBJID")) == 0
    assert _key_violations(invalid, keys=("STUDYID", "USUBJID")) == 2
