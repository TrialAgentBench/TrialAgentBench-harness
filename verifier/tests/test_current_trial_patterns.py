"""Current-trial summaries retain study pairing and observation denominators."""

from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples/scientific"


def test_current_trial_records_reproduce_mechanism_patterns(
    tmp_path: Path, require_records: Callable[..., None]
) -> None:
    """Public data alone reproduce distinct trials, rollout and missing outcomes."""
    require_records("current-trial-records.zip")
    subprocess.run(
        [sys.executable, str(_EXAMPLES / "review_current_trials.py"), "--output", str(tmp_path)],
        check=True,
        timeout=180,
    )
    census = pd.read_csv(tmp_path / "trial_census.csv")
    observation = pd.read_csv(tmp_path / "observation.csv")
    allocation = pd.read_csv(tmp_path / "allocation.csv")
    monitoring = pd.read_csv(tmp_path / "monitoring.csv")
    with zipfile.ZipFile(_EXAMPLES / "current-trial-records.zip") as archive:
        studies = pd.read_csv(archive.open("studies.csv"))
        assert len(census) == len(studies)
        assert not census.duplicated(["cell", "paired_world"]).any()
        for family in ("TE-S08", "TE-S12", "TE-S14"):
            study = studies.loc[studies.family_id.eq(family)].iloc[0]
            participants = pd.read_parquet(io.BytesIO(archive.read(f"{study.study}/participant_records.parquet")))
            reported = census.loc[census.study.eq(study.study)].iloc[0]
            assert reported.participants == len(participants)
            assert reported.occupied_sites == participants.site_id.nunique()
            if family == "TE-S14":
                analysis = pd.read_parquet(io.BytesIO(archive.read(f"{study.study}/analysis.parquet")))
                # Analysis-ready rows omit patients without a measured outcome.
                expected = analysis[study.endpoint_id].notna().sum() / len(participants)
                assert np.isclose(reported.final_known_fraction, expected)
                arms = observation.loc[observation.study.eq(study.study)]
                assert arms.participants.sum() == len(participants)
                assert arms.known.sum() == analysis[study.endpoint_id].notna().sum()
            if family == "TE-S08":
                groups = allocation.loc[allocation.study.eq(study.study)]
                assert groups.participants.sum() == len(participants)
                assert set(groups.active_fraction) == {0, 1}
                assert groups.groupby("cluster").active_fraction.apply(lambda x: x.is_monotonic_increasing).all()
        for _, study in studies.loc[studies.cell.isin(["TE-S06-A1", "TE-S15-A3", "TE-S16-A3"])].iterrows():
            analysis = pd.read_parquet(io.BytesIO(archive.read(f"{study.study}/analysis.parquet")))
            participants = pd.read_parquet(io.BytesIO(archive.read(f"{study.study}/participant_records.parquet")))
            reported = census.loc[census.study.eq(study.study)].iloc[0]
            routine = "routine_endpoint" if "routine_endpoint" in analysis else study.endpoint_id
            routine_fraction = analysis[routine].notna().sum() / len(participants)
            adjudicated_fraction = analysis.adjudicated_endpoint.notna().sum() / len(participants)
            assert reported.routine_endpoint_field == routine
            assert np.isclose(reported.routine_recorded_fraction, routine_fraction)
            assert np.isclose(reported.final_known_fraction, routine_fraction)
            assert np.isclose(reported.adjudicated_fraction, adjudicated_fraction)
            # Routine outcomes exist for everyone; adjudication samples a subset.
            assert routine_fraction == 1
            assert 0 < adjudicated_fraction < routine_fraction
            arms = observation.loc[observation.study.eq(study.study)]
            assert arms.known.sum() == analysis[routine].notna().sum()
            assert arms.adjudicated.sum() == analysis.adjudicated_endpoint.notna().sum()
    assert monitoring.study.nunique() == census.loc[census.family_id.eq("TE-S09"), "study"].nunique()
