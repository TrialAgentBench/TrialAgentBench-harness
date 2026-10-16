"""Choose a small early-safety study from the participant's first reporting targets."""

import csv
import json
from pathlib import Path

with Path("interface/targets.csv").open(newline="") as handle:
    compound = next(csv.DictReader(handle))["target_id"].split(":")[0]
study = dict(
    phase="early_safety",
    sites=12,
    arms=[dict(compound=compound, dose="standard", participants=30, name="active")],
    control_participants=30,
    follow_up_days=56,
    assessment_days=[28, 56],
    surveillance_days=[56],
    looks=[dict(deadline_day=365, completed_participants=60)],
)
print("DEMO_STUDY=" + json.dumps(study, sort_keys=True))
