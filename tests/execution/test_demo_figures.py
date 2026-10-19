"""The worked demonstrations compute, select and draw from public release files, without a network."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def examples(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    """Import the demonstration scripts from their public location."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return {
        name: importlib.import_module(name)
        for name in ("figure_style", "release_access", "trap_decision", "release_panel", "trialdev_budget")
    }


def _record() -> dict[str, object]:
    policies = {"scripted_reference": 0, "always_naive": 2, "always_robust": 1, "prior_only": 3}
    band = {point: dict(policies) for point in ("declared", "value x0.5", "value x2", "cost x0.5", "cost x2")}
    band["value x2"]["always_robust"] = 2
    step = {"mean_musd": 5.0, "lower_musd": 1.0, "upper_musd": 9.0, "worlds": 96}
    policy = {
        "realised_mean_by_rung_musd": [100.0, 105.0, 110.0],
        "constrained_to_lean": step,
        "lean_to_generous": step,
    }
    return {
        "schema_id": "trialagentbench.release_evidence/v1",
        "trialeval": {
            "consequence": [
                {
                    "regime_cell_id": "TE-S06-A3",
                    "shift_consequential": True,
                    "interval_consequential": False,
                    "width_ratio": 0.7,
                    "tertile_bias_sd": [-2.0, -1.7, -1.3],
                },
                {
                    "regime_cell_id": "TE-S08-A3",
                    "shift_consequential": False,
                    "interval_consequential": True,
                    "width_ratio": 0.5,
                    "tertile_bias_sd": [-1.0, -0.9, -0.8],
                },
            ],
            "over_corrections": [
                {"regime_cell_id": "TE-S06-A2", "consequential_share": 0.24, "mean_regret_musd": 6.4}
            ],
            "policy_public": {
                "worlds_per_cell": 1200,
                "cells": 4,
                "failing_cells": policies,
                "naive_costs": [
                    {
                        "regime_cell_id": "TE-S06-A3",
                        "regret_musd": 49.8,
                        "regret_se_musd": 1.8,
                        "consequential_share": 0.3,
                    }
                ],
            },
            "economics": {"band_failing_cells": band},
        },
        "trialdev": {
            "gradient": {
                "rheumatoid_arthritis": {
                    "ladder_usd": [32e6, 47e6, 106e6],
                    "oracle": policy,
                    "reference": policy | {"realised_mean_by_rung_musd": [80.0, 81.0, 79.0]},
                    "mean_musd_by_policy": {"readout_rank": 40.0, "stop": 0.0},
                }
            }
        },
    }


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Write a dataset copy holding only a checksummed release-evidence record."""
    root = tmp_path / "dataset"
    record = root / "release-evidence/release-evidence.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps(_record()))
    digest = hashlib.sha256(record.read_bytes()).hexdigest()
    (root / "SHA256SUMS").write_text(f"{digest}  release-evidence/release-evidence.json\n")
    return root


def _frame(sensitivity: tuple[float, float], seed: int = 3) -> pd.DataFrame:
    """Simulate routine and audited outcomes with arm-specific routine sensitivity."""
    rng = np.random.default_rng(seed)
    n = 6000
    arm = np.repeat(["control", "active"], n // 2)
    stratum = rng.choice(["lower", "higher"], size=n, p=(0.6, 0.4))
    risk = np.where(stratum == "higher", 0.35, 0.15) - 0.05 * (arm == "active")
    truth = rng.random(n) < risk
    sense = np.where(arm == "active", sensitivity[1], sensitivity[0])
    routine = truth & (rng.random(n) < sense) | (~truth & (rng.random(n) < 0.01))
    selected = rng.random(n) < 0.3
    return pd.DataFrame(
        {
            "participant_id": np.arange(n),
            "assigned_arm": arm,
            "validation_stratum": stratum,
            "routine": routine.astype(float),
            "selected": selected,
            "adjudicated": np.where(selected, truth.astype(float), np.nan),
        }
    )


def test_trap_estimators_separate_differential_from_equal_ascertainment(examples: dict[str, ModuleType]) -> None:
    """The audit correction recovers the truth under differential misses; the routine contrast does not."""
    trap = examples["trap_decision"]
    differential = _frame((0.95, 0.75))
    truth = -0.05
    routine = trap.routine_risk_difference(differential)
    audit = trap.audit_corrected_risk_difference(differential)
    assert audit.lower < truth < audit.upper
    assert abs(routine.estimate - truth) > 2 * (routine.upper - routine.estimate) / 1.96
    equal = _frame((0.9, 0.9), seed=5)
    joint = trap.joint_likelihood_risk_difference(equal)
    corrected = trap.audit_corrected_risk_difference(equal)
    assert joint.lower < truth < joint.upper
    assert joint.upper - joint.lower < corrected.upper - corrected.lower


def test_raw_tables_follow_the_documented_identity_rules(examples: dict[str, ModuleType]) -> None:
    """Former source identifiers are mapped; a second assessment of one kind is refused."""
    trap = examples["trap_decision"]
    participants = pd.DataFrame(
        {"participant_id": [1, 2], "assigned_arm": ["active", "control"], "validation_stratum": ["lower", "higher"]}
    )
    events = pd.DataFrame(
        {
            "participant_id": ["legacy-a", "2", "2"],
            "event_kind": ["endpoint_assessment"] * 3,
            "event_id": ["routine_hf_assessment", "routine_hf_assessment", "adjudicated_hf_assessment"],
            "event_by_contact": [True, False, True],
        }
    )
    crosswalk = pd.DataFrame({"source_identifier": ["legacy-a"], "current_identifier": [1]})

    def tables(frame: pd.DataFrame) -> dict[str, bytes]:
        return {
            "participant_records.parquet": participants.to_parquet(),
            "event_history.parquet": frame.to_parquet(),
            "identifier_crosswalk.parquet": crosswalk.to_parquet(),
        }

    frame = trap.analysis_frame(tables(events))
    assert frame["routine"].tolist() == [1.0, 0.0]
    assert frame["selected"].tolist() == [False, True]
    with pytest.raises(ValueError, match="two assessments"):
        trap.analysis_frame(tables(pd.concat([events, events.iloc[[1]]])))


def test_selection_rule_finds_the_trap_and_the_twin_on_shared_worlds(examples: dict[str, ModuleType]) -> None:
    """The twin is found by shared base worlds, whatever its tier is called."""
    trap = examples["trap_decision"]
    entries = [
        trap.IndexEntry(
            task_id=f"te_{cell}{world}",
            regime_cell_id=f"TE-S06-{cell}",
            evaluation_series_id="TE-S06",
            assumption_tier=cell,
            context_tier=context,
            base_world_id=f"world_{world}",
        )
        for cell in ("A2", "A3")
        for world in range(4)
        for context in ("C1", "C5")
    ]
    selected = trap.select_tasks(entries)
    assert {entry.regime_cell_id for entry in selected["trap"]} == {"TE-S06-A3"}
    assert {entry.regime_cell_id for entry in selected["twin"]} == {"TE-S06-A2"}
    assert {entry.context_tier for group in selected.values() for entry in group} == {"C5"}
    with pytest.raises(ValueError, match="sharing the trap's base worlds"):
        trap.select_tasks(
            [entry for entry in entries if entry.base_world_id != "world_0" or entry.assumption_tier == "A3"]
        )


def test_trap_and_twin_figures_draw_from_a_decision_table(
    examples: dict[str, ModuleType], dataset: Path, tmp_path: Path
) -> None:
    """Both figures and their source data are written from tidy decision rows."""
    trap = examples["trap_decision"]
    rows = [
        {
            "role": role,
            "world": world,
            "cell": cell,
            "tier": cell[-2:],
            "analysis": analysis,
            "credited": analysis == credited,
            "true_risk_difference": 0.01 * world,
            "estimate": 0.01 * world - 0.01,
            "lower": 0.01 * world - 0.03,
            "upper": 0.01 * world + 0.01,
            "confirm": world < 3,
            "size_multiple": 2.0 if world < 3 else 0.0,
            "confirmatory_participants": 8700.0 if world < 3 else 0.0,
            "regret_musd": 0.0 if analysis == credited else 10.0 * world,
            "consequential": analysis != credited and world == 3,
        }
        for role, cell, credited in (("trap", "TE-S06-A3", "audit"), ("twin", "TE-S06-A2", "joint"))
        for world in (1, 2, 3, 4)
        for analysis in ("routine", "audit", "joint")
    ]
    table = pd.DataFrame(rows)
    release = examples["release_access"].PublicRelease.open(release_root=dataset, revision="local", cache=tmp_path)
    scale = trap.release_scale(release.evidence(), "TE-S06-A3", "TE-S06-A2")
    assert scale["trap_naive_regret_musd"] == 49.8
    outputs = examples["figure_style"].save_figure(trap.trap_figure(table, scale), table, tmp_path / "out", "trap")
    assert all(path.stat().st_size > 0 for path in outputs)
    trap.twin_figure(table, scale)


def test_record_demonstrations_run_offline_from_a_dataset_copy(
    examples: dict[str, ModuleType], dataset: Path, tmp_path: Path
) -> None:
    """The panel and budget demonstrations read only the checksummed record."""
    common = ["--release-root", str(dataset), "--cache", str(tmp_path / "cache")]
    assert examples["release_panel"].main([*common, "--output", str(tmp_path / "panel")]) == 0
    assert examples["trialdev_budget"].main([*common, "--output", str(tmp_path / "budget")]) == 0
    panel = pd.read_csv(tmp_path / "panel/release_panel.csv")
    assert set(panel.loc[panel["panel"].eq("a"), "band"]) == {
        "declared",
        "value x0.5",
        "value x2",
        "cost x0.5",
        "cost x2",
    }
    assert panel.loc[panel["policy"].eq("scripted_reference"), "failing_cells"].eq(0).all()
    budget = pd.read_csv(tmp_path / "budget/trialdev_budget.csv")
    assert budget.loc[budget["policy"].eq("oracle"), "realised_value_musd"].tolist() == [100.0, 105.0, 110.0]
    for name in ("release_panel.svg", "release_panel.png"):
        assert (tmp_path / "panel" / name).stat().st_size > 0


def test_a_changed_record_is_refused(examples: dict[str, ModuleType], dataset: Path, tmp_path: Path) -> None:
    """A record that differs from the dataset's checksums is never read."""
    record = dataset / "release-evidence/release-evidence.json"
    record.write_text(record.read_text() + " ")
    release = examples["release_access"].PublicRelease.open(release_root=dataset, revision="local", cache=tmp_path)
    with pytest.raises(ValueError, match="differs from its published SHA-256"):
        release.evidence()


@pytest.mark.slow
def test_demonstrations_regenerate_from_the_published_release(examples: dict[str, ModuleType], tmp_path: Path) -> None:
    """Download (or read a local copy of) the published release and rebuild every demonstration figure."""
    root = os.environ.get("TRIALAGENTBENCH_DEMO_RELEASE_ROOT")
    common = ["--cache", str(tmp_path / "cache")] + (["--release-root", root] if root else [])
    release = examples["release_access"].PublicRelease.open(
        release_root=Path(root) if root else None,
        revision=examples["release_access"].DEFAULT_REVISION,
        cache=tmp_path / "cache",
    )
    # The budget demonstration draws the record's resource-gradient panel, which a release may publish after its data.
    budget = ("trialdev_budget",) if release.evidence().trialdev.gradient is not None else ()
    for name in ("trap_decision", "release_panel", *budget):
        assert examples[name].main([*common, "--output", str(tmp_path / "figures")]) == 0
    decisions = pd.read_csv(tmp_path / "figures/twin_overcorrection.csv")
    assert decisions.groupby("role")["world"].nunique().tolist() == [4, 4]
    assert decisions.loc[decisions["credited"], "regret_musd"].eq(0).all()
