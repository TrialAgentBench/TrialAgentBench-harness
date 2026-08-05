"""Tests for matched Assumption-axis experiment contracts."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import pandas as pd
import pytest
from pydantic import ValidationError
from pydantic.types import JsonValue

from trialagentbench_validation.characterisation import (
    AssumptionAnalysisBridge,
    AssumptionSeriesIdentity,
    MatchedAssumptionDesign,
)
from trialagentbench_validation.characterisation.assumption import (
    _SERIES,
    _assumption_response_rows,
    _mechanism,
)
from trialagentbench_validation.io import canonical_payload_sha256


def _identity(*, replicate_index: int, stream: str) -> AssumptionSeriesIdentity:
    payload = {
        "series_id": "TE-S01",
        "replicate_index": replicate_index,
        "design_profile_id": "TE-DP01",
        "population": "ITT",
        "endpoint_id": "death",
        "estimand_id": "primary_itt",
        "effect_scale": "risk_difference_tau",
        "default_method": "observed:coxph_binary_breslow_risk_tau",
        "participant_count": 3400,
        "follow_up_horizon_days": 1095.0,
    }
    return AssumptionSeriesIdentity(
        assumption_tiers=("A1", "A2", "A3"),
        task_ids={
            "A1": f"TASK{'1' * 31}{replicate_index}",
            "A2": f"TASK{'2' * 31}{replicate_index}",
            "A3": f"TASK{'3' * 31}{replicate_index}",
        },
        random_stream_id=stream,
        identity_sha256=canonical_payload_sha256(cast(JsonValue, payload)),
        **payload,
    )


def test_matched_design_requires_unique_streams_and_complete_count() -> None:
    first = _identity(replicate_index=1, stream="series-01-world-01")
    second = _identity(replicate_index=2, stream="series-01-world-02")
    design = MatchedAssumptionDesign(
        release_id="matched-assumption-example",
        analysis_count=6,
        identities=(first, second),
    )
    assert design.analysis_count == 6

    with pytest.raises(ValidationError, match="random stream IDs must be unique"):
        MatchedAssumptionDesign(
            release_id="matched-assumption-example",
            analysis_count=6,
            identities=(
                first,
                second.model_copy(update={"random_stream_id": first.random_stream_id}),
            ),
        )
    with pytest.raises(ValidationError, match="analysis count"):
        MatchedAssumptionDesign(
            release_id="matched-assumption-example",
            analysis_count=5,
            identities=(first, second),
        )


def test_matched_identity_requires_one_task_per_tier() -> None:
    identity = _identity(replicate_index=1, stream="series-01-world-01")
    with pytest.raises(ValidationError, match="task IDs must cover"):
        AssumptionSeriesIdentity.model_validate(
            identity.model_dump(exclude={"task_ids"})
            | {"task_ids": {"A1": identity.task_ids["A1"]}}
        )


def _bridge(
    *,
    series_id: str,
    tier: str,
    replicate_index: int,
    mechanism: float,
    default: float,
    qualified: float,
) -> AssumptionAnalysisBridge:
    return AssumptionAnalysisBridge.model_validate(
        {
            "task_id": f"TASK{replicate_index:032X}",
            "independence_unit_id": f"{series_id}-world-{replicate_index}",
            "series_id": series_id,
            "replicate_index": replicate_index,
            "assumption_tier": tier,
            "design_profile_id": "TE-DP02",
            "participant_count": 2000,
            "follow_up_horizon_days": 365.0,
            "endpoint_id": "death",
            "estimand_id": "primary_itt",
            "effect_scale": "risk_difference_tau",
            "mechanism_id": "test_mechanism",
            "mechanism_label": "test mechanism",
            "mechanism_value": mechanism,
            "mechanism_unit": "proportion",
            "mechanism_band": "observed",
            "diagnostic_status": "observed",
            "default_method": "observed:km",
            "default_status": "estimated",
            "default_value": default,
            "default_standard_error": 0.01,
            "default_interval_low": default - 0.02,
            "default_interval_high": default + 0.02,
            "qualified_method": "observed:km",
            "qualified_shape": "point",
            "qualified_value": qualified,
            "qualified_standard_error": 0.01,
            "qualified_interval_low": qualified - 0.02,
            "qualified_interval_high": qualified + 0.02,
            "result_unit": "risk difference",
            "absolute_analysis_difference": abs(default - qualified),
            "default_rejects_null": False,
            "qualified_rejects_null": False,
            "qualified_replay_abs_error": 0.0,
        }
    )


def test_response_rows_link_nonadherence_to_treatment_policy_attenuation() -> None:
    """The nonadherence display measures effect attenuation, not a duplicate route."""

    bridges = tuple(
        _bridge(
            series_id="TE-S03",
            tier=tier,
            replicate_index=replicate_index,
            mechanism=mechanism,
            default=estimate,
            qualified=estimate,
        )
        for replicate_index in (1, 2)
        for tier, mechanism, estimate in (
            ("A1", 0.05, -0.12),
            ("A2", 0.35, -0.08),
        )
    )

    rows = {
        (row.series_id, row.assumption_tier): row
        for row in _assumption_response_rows(bridges)
    }
    assert rows[("TE-S03", "A1")].consequence_value_mean == pytest.approx(0.0)
    assert rows[("TE-S03", "A2")].consequence_value_mean == pytest.approx(0.04)
    assert (
        rows[("TE-S03", "A2")].mechanism_label
        == "treated-arm recorded-dose nonadherence"
    )
    assert (
        rows[("TE-S03", "A2")].consequence_label
        == "Treatment-effect attenuation from reference"
    )


def test_nonadherence_mechanism_uses_the_active_treatment_arm(tmp_path: Path) -> None:
    """Placebo adherence cannot determine active-treatment exposure intensity."""

    frame = pd.DataFrame(
        {
            "TRTA": ["control", "control", "treated", "treated"],
            "MEAN_EXADH": [0.0, 0.0, 0.8, 0.6],
        }
    )
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    archive_path = tmp_path / "participant.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "items/TASK00000000000000000000000000000000/data/subject_operational_flags.parquet",
            buffer.getvalue(),
        )
    with ZipFile(archive_path) as archive:
        mechanism, unit, status = _mechanism(
            participant=archive,
            task_id="TASK00000000000000000000000000000000",
            definition=_SERIES["TE-S03"],
            records=(),
        )
    assert mechanism == pytest.approx(0.3)
    assert unit == "proportion"
    assert status == "observed"


def test_response_rows_use_signed_same_estimand_contrasts() -> None:
    """Compatible estimators are centred at zero rather than a positive noise floor."""

    bridges = tuple(
        _bridge(
            series_id="TE-S05",
            tier=tier,
            replicate_index=replicate_index,
            mechanism=mechanism,
            default=default,
            qualified=qualified,
        )
        for replicate_index, offset in ((1, -0.01), (2, 0.01))
        for tier, mechanism, default, qualified in (
            ("A1", 0.0, -0.10 + offset, -0.10),
            ("A2", 0.2, -0.14 + offset, -0.10),
        )
    )

    rows = {
        (row.series_id, row.assumption_tier): row
        for row in _assumption_response_rows(bridges)
    }
    assert rows[("TE-S05", "A1")].consequence_value_mean == pytest.approx(0.0)
    assert rows[("TE-S05", "A2")].consequence_value_mean == pytest.approx(0.04)
    assert (
        rows[("TE-S05", "A2")].consequence_label
        == "Spline minus linear standardized risk"
    )
