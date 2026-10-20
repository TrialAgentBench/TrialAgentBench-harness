"""The worked demonstrations compute, select and draw from public release files, without a network."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from pydantic import BaseModel

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
BAND = ("declared", "value x0.5", "value x2", "cost x0.5", "cost x2")


@pytest.fixture
def examples(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    """Import the demonstration scripts from their public location."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return {
        name: importlib.import_module(name)
        for name in ("figure_style", "release_access", "decision_layer", "release_panel", "trialdev_budget")
    }


def _record(examples: dict[str, ModuleType]) -> BaseModel:
    """Build demonstration evidence using only the public example's own models."""
    policies = {"scripted_reference": 0, "always_naive": 2, "always_robust": 1, "prior_only": 3}
    public = {point: dict(policies) for point in BAND}
    public["value x2"]["always_robust"] = 2
    withheld = {point: {policy: count > 0 for policy, count in policies.items()} for point in BAND}
    withheld["cost x0.5"]["always_robust"] = False
    step = {"mean_musd": 5.0, "lower_musd": 1.0, "upper_musd": 9.0, "worlds": 96}
    policy = {
        "realised_mean_by_rung_musd": [100.0, 105.0, 110.0],
        "constrained_to_lean": step,
        "lean_to_generous": step,
    }
    return examples["release_access"].ReleaseEvidence.model_validate(
        {
            "trialeval": {
                "policy_public": {"worlds_per_cell": 1200, "cells": 4, "failing_cells": policies},
                "policy_withheld": {"fails_a_cell": withheld},
                "economics": {"band_failing_cells": public},
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
    )


@pytest.fixture
def dataset(tmp_path: Path, examples: dict[str, ModuleType]) -> Path:
    """Write a dataset copy holding only a checksummed release-evidence record."""
    root = tmp_path / "dataset"
    record = root / "release-evidence/release-evidence.json"
    record.parent.mkdir(parents=True)
    record.write_text(_record(examples).model_dump_json())
    digest = hashlib.sha256(record.read_bytes()).hexdigest()
    (root / "SHA256SUMS").write_text(f"{digest}  release-evidence/release-evidence.json\n")
    return root


def test_the_decision_layer_stops_on_weak_evidence_and_confirms_on_strong(examples: dict[str, ModuleType]) -> None:
    """An honest report stops on adverse evidence and confirms on favourable; strong evidence needs a smaller trial.

    A wider interval leaves more to learn, so a confirmatory trial stays worth running on weaker evidence.
    """
    layer = examples["decision_layer"]
    actions = layer.action_table(pd.Series([-2.0, -1.0, 1.0, 2.5, 5.0]).to_numpy())
    honest = actions.loc[actions["width"].eq(1.0)].set_index("estimate_se")
    assert not honest.loc[-2.0, "confirm"] and not honest.loc[-1.0, "confirm"]
    assert honest.loc[2.5, "confirm"] and 0 < honest.loc[5.0, "size_multiple"] < honest.loc[2.5, "size_multiple"]
    first = {width: rows.loc[rows["confirm"], "estimate_se"].min() for width, rows in actions.groupby("width")}
    assert first[2.0] <= first[1.0] <= first[0.5]


def test_no_distortion_of_an_honest_report_gains(examples: dict[str, ModuleType]) -> None:
    """Every distortion, the agent's own shrinkage included, costs regret against the honest report."""
    layer = examples["decision_layer"]
    table = layer.distortion_table(worlds=150).set_index("report")
    assert set(table.index) == set(layer.DISTORTIONS)
    assert (table["mean_regret_musd"] > 0).all()
    shrunk = layer.shrunk_report(2.0)
    assert 0 < shrunk.estimate < 2.0 and shrunk.upper - shrunk.lower < 2 * 1.96


def test_the_decision_layer_figure_writes_its_data(examples: dict[str, ModuleType], tmp_path: Path) -> None:
    """The figure, its plotted values and the printed summary come from one run."""
    assert examples["decision_layer"].main(["--output", str(tmp_path / "layer"), "--worlds", "60"]) == 0
    data = pd.read_csv(tmp_path / "layer/decision_layer.csv")
    assert set(data["panel"]) == {"a", "b"}
    for name in ("decision_layer.svg", "decision_layer.png"):
        assert (tmp_path / "layer" / name).stat().st_size > 0


def test_record_demonstrations_run_offline_from_a_dataset_copy(
    examples: dict[str, ModuleType], dataset: Path, tmp_path: Path
) -> None:
    """The panel and budget demonstrations read only the checksummed record."""
    common = ["--release-root", str(dataset), "--cache", str(tmp_path / "cache")]
    assert examples["release_panel"].main([*common, "--output", str(tmp_path / "panel")]) == 0
    assert examples["trialdev_budget"].main([*common, "--output", str(tmp_path / "budget")]) == 0
    panel = pd.read_csv(tmp_path / "panel/release_panel.csv")
    assert set(panel["split"]) == {"public", "withheld"}
    assert set(panel["band"]) == set(BAND)
    public = panel.loc[panel["split"].eq("public")]
    assert public.loc[public["policy"].eq("scripted_reference"), "failing_cells"].eq(0).all()
    assert public["fails_a_cell"].isna().all()
    withheld = panel.loc[panel["split"].eq("withheld")]
    assert withheld[["failing_cells", "cells"]].isna().all().all()
    assert set(withheld["fails_a_cell"]) == {True, False}
    cheap = withheld.loc[withheld["band"].eq("cost x0.5")].set_index("policy")
    assert not cheap.loc["always_robust", "fails_a_cell"]
    assert cheap.loc["prior_only", "fails_a_cell"]
    svg = ET.parse(tmp_path / "panel/release_panel.svg")
    axis = svg.find(".//*[@id='axes_2']")
    assert axis is not None
    texts = ["".join(node.itertext()) for node in axis.iter("{http://www.w3.org/2000/svg}text")]
    assert "b  Withheld: fails at least one cell" in texts
    assert not any(re.fullmatch(r"\d+(?:\.\d+)?", text) for text in texts)
    assert texts.count("fails") == withheld["fails_a_cell"].sum()
    assert texts.count("passes") == (~withheld["fails_a_cell"].astype(bool)).sum()
    budget = pd.read_csv(tmp_path / "budget/trialdev_budget.csv")
    assert budget.loc[budget["policy"].eq("oracle"), "realised_value_musd"].tolist() == [100.0, 105.0, 110.0]
    for name in ("release_panel.svg", "release_panel.png"):
        assert (tmp_path / "panel" / name).stat().st_size > 0


def test_a_panel_that_disagrees_with_its_band_is_refused(examples: dict[str, ModuleType]) -> None:
    """The public declared band point must be its own policy panel."""
    record = _record(examples)
    record.trialeval.economics.band_failing_cells["declared"]["prior_only"] = 4
    evidence = examples["release_access"].ReleaseEvidence.model_validate(record.model_dump())
    with pytest.raises(ValueError, match="public band's declared point"):
        examples["release_panel"].policy_table(evidence)


@pytest.mark.parametrize("missing", ["band", "policy"])
def test_withheld_outcomes_must_cover_the_public_band_and_policies(
    examples: dict[str, ModuleType], missing: str
) -> None:
    """A missing withheld outcome fails before drawing an incomplete panel."""
    record = _record(examples)
    withheld = record.trialeval.policy_withheld.fails_a_cell
    if missing == "band":
        withheld.pop("cost x0.5")
    else:
        withheld["cost x0.5"].pop("prior_only")
    evidence = examples["release_access"].ReleaseEvidence.model_validate(record.model_dump())
    with pytest.raises(ValueError, match="differ in band points|withheld band point omits a policy"):
        examples["release_panel"].policy_table(evidence)


def test_a_changed_record_is_refused(examples: dict[str, ModuleType], dataset: Path, tmp_path: Path) -> None:
    """A record that differs from the dataset's checksums is never read."""
    record = dataset / "release-evidence/release-evidence.json"
    record.write_text(record.read_text() + " ")
    release = examples["release_access"].PublicRelease.open(release_root=dataset, revision="local", cache=tmp_path)
    with pytest.raises(ValueError, match="differs from its published SHA-256"):
        release.evidence()


def test_pending_budget_demo_has_no_retained_measurements(
    examples: dict[str, ModuleType], dataset: Path, tmp_path: Path
) -> None:
    """A pending release still renders all three artifact formats, without an earlier value."""
    path = dataset / "release-evidence/release-evidence.json"
    record = examples["release_access"].ReleaseEvidence.model_validate_json(path.read_bytes())
    td = record.trialdev.model_copy(update={"gradient": None, "gradient_status": "pending"})
    path.write_text(record.model_copy(update={"trialdev": td}).model_dump_json())
    (dataset / "SHA256SUMS").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  release-evidence/release-evidence.json\n"
    )
    output = tmp_path / "figures"
    assert examples["trialdev_budget"].main(["--release-root", str(dataset), "--output", str(output)]) == 0
    table = pd.read_csv(output / "trialdev_budget.csv")
    assert list(table) == ["status"]
    assert table.iloc[0, 0] == "pending: published as an evidence update with unchanged data"
    assert (output / "trialdev_budget.svg").is_file() and (output / "trialdev_budget.png").is_file()


@pytest.mark.slow
@pytest.mark.network
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
    for name in ("release_panel", *budget):
        assert examples[name].main([*common, "--output", str(tmp_path / "figures")]) == 0
    assert examples["decision_layer"].main(["--output", str(tmp_path / "figures")]) == 0
    panel = pd.read_csv(tmp_path / "figures/release_panel.csv")
    reference = panel.loc[panel["policy"].eq("scripted_reference")]
    assert reference.loc[reference["split"].eq("public"), "failing_cells"].eq(0).all()
    assert reference.loc[reference["split"].eq("withheld"), "fails_a_cell"].eq(False).all()
