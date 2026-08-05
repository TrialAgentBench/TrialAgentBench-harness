"""Tests for independent clustered ordinal verification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

from trialagentbench_validation.cli import main
from trialagentbench_validation.external.recovery import (
    clustered_ordinal as clustered_ordinal_qualification,
)
from trialagentbench_validation.external.recovery.clustered_ordinal import (
    ClusteredOrdinalArmReferenceV1,
    ClusteredOrdinalDoseDistributionV1,
    ClusteredOrdinalQualificationDesignV1,
    _validate_world,
)


class _Result(BaseModel):
    value: int


def test_clustered_ordinal_public_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clustered outcome verifier is available through the public CLI."""

    release_dir = tmp_path / "release"
    output = tmp_path / "report.json"

    def evaluate(*, release_dir: Path, minimum_worlds: int) -> _Result:
        assert release_dir == tmp_path / "release"
        assert minimum_worlds == 2
        return _Result(value=2)

    monkeypatch.setattr(
        clustered_ordinal_qualification,
        "evaluate_clustered_ordinal_qualification",
        evaluate,
    )

    assert (
        main(
            [
                "clustered-ordinal-validation",
                "--release-dir",
                str(release_dir),
                "--output",
                str(output),
                "--minimum-worlds",
                "2",
            ]
        )
        == 0
    )
    assert _Result.model_validate_json(output.read_text()) == _Result(value=2)


def _design() -> ClusteredOrdinalQualificationDesignV1:
    arms = (
        ClusteredOrdinalArmReferenceV1(
            arm_id="control",
            participants=20,
            cluster_size_probabilities=(0.5, 0.5),
            observation_probability=0.8,
            category_probabilities=(0.7, 0.2, 0.1),
            source_kendall_tau=0.2,
            latent_correlation=0.4,
        ),
        ClusteredOrdinalArmReferenceV1(
            arm_id="treatment",
            participants=20,
            cluster_size_probabilities=(0.5, 0.5),
            observation_probability=0.8,
            category_probabilities=(0.75, 0.18, 0.07),
            source_kendall_tau=0.2,
            latent_correlation=0.4,
        ),
    )
    doses = (0.0, 1.0, 2.0)
    return ClusteredOrdinalQualificationDesignV1(
        trial_id="fixture",
        source_sha256="a" * 64,
        worlds=2,
        seed=1,
        categories=(1, 2, 3),
        control_arm_id="control",
        treatment_arm_id="treatment",
        source_log_occlusion_odds_ratio=-0.3,
        dose_multipliers=doses,
        arms=arms,
        fitted_distributions=tuple(
            ClusteredOrdinalDoseDistributionV1(
                dose_multiplier=dose,
                arm_id=arm,
                category_probabilities=(0.7, 0.2, 0.1),
            )
            for dose in doses
            for arm in ("control", "treatment")
        ),
    )


def test_world_validation_rejects_nonconsecutive_graft_keys() -> None:
    """Independent verification rejects malformed within-participant keys."""

    rows = []
    for dose in (0.0, 1.0, 2.0):
        for arm in ("control", "treatment"):
            for participant in range(20):
                rows.append(
                    {
                        "world_id": "world_0123456789abcdefabcd",
                        "dose_multiplier": dose,
                        "participant_id": f"{dose}-{arm}-{participant}",
                        "arm": arm,
                        "graft_index": 2,
                        "observed": True,
                        "three_year_ct_result": 1,
                    }
                )
    with pytest.raises(ValueError, match="indices must be consecutive"):
        _validate_world(
            pd.DataFrame(rows),
            design=_design(),
            world_id="world_0123456789abcdefabcd",
        )
