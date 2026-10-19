"""Verify retained analysis inputs against released scientific records."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, ConfigDict

from trialagentbench_verifier.execution_contracts import (
    TRIALEVAL_REPLAY_ABS_TOLERANCE_V1,
    TRIALEVAL_REPLAY_REL_TOLERANCE_V1,
)


class TrialEvalAnalysisDataVerificationV1(BaseModel):
    """Content verification of a derived table, without a repair-method label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["verified", "missing", "ambiguous", "invalid", "different_population", "different_values"]
    differing_columns: tuple[str, ...] = ()


def analysis_input_reference_v1(data: pd.DataFrame) -> pd.DataFrame:
    """Select scientific input columns without transfer or internal labels.

    Parameters
    ----------
    data
        Reconstructed participant/endpoint table in evaluator custody.

    Returns
    -------
    pandas.DataFrame
        Clinical source values, timing, population membership and assignment.
        Extra participant-generated columns never affect comparison.
    """
    return data.drop(
        columns=["source_value_status", "reference_population_id", "event_type", "outcome_id"], errors="ignore"
    )


def verify_analysis_data_v1(
    *, scratch: Path, expected: pd.DataFrame, key_fields: tuple[str, ...]
) -> TrialEvalAnalysisDataVerificationV1:
    """Compare scientific input values irrespective of table order or format.

    Parameters
    ----------
    scratch
        Retained or freshly replayed participant output directory.
    expected
        Evaluator-owned scientific columns; never mounted in the executor.
    key_fields
        Publicly declared row identity, for example participant and visit.

    Returns
    -------
    TrialEvalAnalysisDataVerificationV1
        Verification status. Extra derived columns are allowed. Numeric values
        use the published replay tolerance; identifiers are compared exactly.

    Raises
    ------
    ValueError
        If the evaluator reference has absent or nonunique scientific keys.
    """
    if (
        not key_fields
        or not set(key_fields) <= set(expected)
        or expected.duplicated(list(key_fields)).any()
        or expected.loc[:, list(key_fields)].isna().any().any()
    ):
        raise ValueError("analysis-data reference requires complete unique scientific keys")
    candidates = tuple(
        path for path in (scratch / "analysis_data.parquet", scratch / "analysis_data.csv") if path.exists()
    )
    if not candidates:
        return TrialEvalAnalysisDataVerificationV1(status="missing")
    if len(candidates) != 1:
        return TrialEvalAnalysisDataVerificationV1(status="ambiguous")
    path = candidates[0]
    if path.is_symlink() or not path.is_file():
        return TrialEvalAnalysisDataVerificationV1(status="invalid")
    try:
        observed = (
            pd.read_parquet(path)
            if path.suffix == ".parquet"
            else pd.read_csv(path, dtype={field: str for field in key_fields if not is_numeric_dtype(expected[field])})
        )
    except (OSError, ValueError, pd.errors.ParserError):
        return TrialEvalAnalysisDataVerificationV1(status="invalid")
    columns = tuple(str(column) for column in expected.columns)
    if not all(isinstance(column, str) for column in observed.columns):
        return TrialEvalAnalysisDataVerificationV1(status="invalid")
    absent = tuple(sorted(set(columns) - {str(column) for column in observed.columns}))
    if absent:
        return TrialEvalAnalysisDataVerificationV1(status="invalid", differing_columns=absent)
    if observed.duplicated(list(key_fields)).any() or len(observed) != len(expected):
        return TrialEvalAnalysisDataVerificationV1(status="different_population")
    left = expected.sort_values(list(key_fields)).reset_index(drop=True)
    try:
        right = observed.sort_values(list(key_fields)).reset_index(drop=True)
    except TypeError:
        return TrialEvalAnalysisDataVerificationV1(status="invalid")
    different: list[str] = []
    for column in columns:
        a, b = left[column], right[column]
        if column not in key_fields and is_numeric_dtype(a) and is_numeric_dtype(b):
            equal = bool(
                np.allclose(
                    a.to_numpy(dtype=float),
                    b.to_numpy(dtype=float),
                    rtol=TRIALEVAL_REPLAY_REL_TOLERANCE_V1,
                    atol=TRIALEVAL_REPLAY_ABS_TOLERANCE_V1,
                    equal_nan=True,
                )
            )
        else:
            equal = bool(((a == b) | (a.isna() & b.isna())).all())
        if not equal:
            different.append(column)
    if set(different) & set(key_fields):
        return TrialEvalAnalysisDataVerificationV1(
            status="different_population", differing_columns=tuple(sorted(different))
        )
    return TrialEvalAnalysisDataVerificationV1(
        status="different_values" if different else "verified", differing_columns=tuple(sorted(different))
    )
