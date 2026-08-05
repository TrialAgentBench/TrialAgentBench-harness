"""Regenerate the public trial-characterisation example."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trialagentbench_validation.characterisation import (
    TrialCharacterisationSpec,
    TrialData,
    characterise_trial,
    summarise_characterisations,
    write_characterisation_csv,
)


def main() -> None:
    """Run the deterministic example from its typed specification."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    base = TrialCharacterisationSpec.model_validate_json(
        (root / "spec.json").read_text(encoding="utf-8")
    )
    participants = pd.read_csv(root / "participants.csv")
    trials = []
    for (programme_id, trial_id), frame in participants.groupby(
        ["programme_id", "trial_id"],
        observed=True,
        sort=True,
    ):
        payload = base.model_dump()
        payload.update(
            {
                "programme_id": str(programme_id),
                "trial_id": str(trial_id),
            }
        )
        spec = TrialCharacterisationSpec.model_validate(payload)
        trial_frame = frame.drop(columns=["programme_id", "trial_id"]).reset_index(
            drop=True
        )
        trials.append(characterise_trial(spec, TrialData(participants=trial_frame)))
    write_characterisation_csv(args.output, summarise_characterisations(tuple(trials)))


if __name__ == "__main__":
    main()
