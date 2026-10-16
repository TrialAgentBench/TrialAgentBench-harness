"""Public calibration and mechanism comparisons preserve adverse scientific results."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "scientific"


def test_compact_mechanism_and_registry_readers_have_no_benchmark_dependency(tmp_path: Path) -> None:
    """The small public inputs reproduce every condition and the negative controls."""
    command = """
import importlib.abc
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
class RejectPrivate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('ariadne', 'trialagentbench', 'clinical_data_tables')):
            raise RuntimeError('Public reader attempted a benchmark import: ' + fullname)
sys.meta_path.insert(0, RejectPrivate())
sys.argv = [sys.argv[1], '--output', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    for script in ("review_registry.py", "review_mechanisms.py"):
        subprocess.run(
            [sys.executable, "-I", "-c", command, str(_EXAMPLES / script), str(tmp_path)],
            cwd=tmp_path,
            check=True,
            timeout=90,
            capture_output=True,
            text=True,
        )
    comparisons = pd.read_csv(tmp_path / "registry_comparisons.csv")
    assert len(comparisons) == 34 and comparisons.family_id.nunique() == 17
    archived = pd.read_csv(tmp_path / "registry_archived_comparisons.csv")
    assert len(archived) == 60 and archived.generation_role.eq("archived_teaching_snapshot").all()
    assert comparisons.generation_role.eq("current_generator_probe").all()
    enrollment = comparisons.loc[comparisons.metric.eq("Randomized participants")].set_index("cell")
    assert enrollment.loc["TE-S15-A1", "native_value"] == 3196
    assert enrollment.loc["TE-S15-A1", "registry_upper_quartile"] == 151
    assert enrollment.loc["TE-S15-A1", "position_relative_to_iqr"] == "above"
    assert enrollment.loc["TE-S08-A1", "position_relative_to_iqr"] == "sparse_reference"
    assert enrollment.loc["TE-S08-A1", "native_value"] == 143600
    assert set(comparisons.comparison_role) == {"calibration_source_comparison"}
    availability = pd.read_csv(tmp_path / "registry_observation_loss.csv")
    assert len(availability) == 16 and availability.family_id.nunique() == 8
    oncology = availability.set_index(["cell", "arm"]).loc[("TE-S15-A1", "active")]
    assert oncology.observed_endpoint_unavailable_count == 135
    assert oncology.randomized_participants == 1598
    assert oncology.calibration_origin == "authored"
    summary = pd.read_csv(tmp_path / "mechanism_recovery_summary.csv")
    assert len(summary) == 18 and summary.case.nunique() == 11 and summary.studies.eq(32).all()
    paired = pd.read_csv(tmp_path / "mechanism_paired_comparisons.csv")
    assert len(paired) == 11
    controls = paired.loc[
        paired.kind.eq("same_trial_retrieval_probability") & paired.first_method.eq("Crude event risk")
    ]
    assert len(controls) == 2 and controls.mean_loss_difference.eq(0).all()
    adverse = paired.loc[paired.first_case.eq("TE-S04-A4-unfavourable") & paired.kind.eq("same_trial_method")].iloc[0]
    assert adverse.mean_loss_difference == pytest.approx(-0.21367360890878279)
    assert adverse.conditional_upper < 0
    source_cells = pd.read_csv(tmp_path / "mechanism_clinical_cells.csv")
    assert len(source_cells) == 30 and source_cells.family_id.nunique() == 17
    changes = pd.read_csv(tmp_path / "mechanism_condition_changes.csv")
    assert len(changes) == 13
    kidney = changes.set_index("cell").loc["TE-S02-A2"]
    assert kidney.change_in_finite_complete_contrast == 0
    assert kidney.change_in_shortcut_estimate > kidney.change_in_estimate
    for stem in ("registry_comparison", "mechanism_recovery"):
        for extension in ("png", "pdf", "svg"):
            assert (tmp_path / f"{stem}.{extension}").stat().st_size > 1000


def test_registry_finite_counts_do_not_treat_infinity_as_a_trial_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quantiles and contributing counts use the same finite registry observations."""
    monkeypatch.syspath_prepend(str(_EXAMPLES))
    reader = importlib.import_module("review_registry")
    native = pd.read_csv(_EXAMPLES / "registry/native_trials.csv").iloc[:1]
    registry = pd.DataFrame(
        {
            "family_id": ["TE-S01"] * 7,
            "nct_id": [f"trial-{index}" for index in range(7)],
            "enrollment": [1, 2, 3, 4, 5, np.inf, np.nan],
            "sites": [1] * 7,
        }
    )
    targets = json.loads((_EXAMPLES / "registry/calibration_targets.json").read_text())
    result = reader.compare_trials(native, registry, targets).set_index("metric")
    assert result.loc["Randomized participants", "registry_trials"] == 5
    assert result.loc["Randomized participants", "registry_missing_or_nonfinite"] == 2
    assert result.loc["Randomized participants", "registry_median"] == 3
    with pytest.raises(ValueError, match="identities must be unique"):
        reader.compare_trials(native, pd.concat([registry, registry.iloc[:1]]), targets)


def test_repeated_comparison_rejects_unpaired_or_reference_mismatched_studies(monkeypatch: pytest.MonkeyPatch) -> None:
    """A convenient difference cannot silently become an unpaired comparison."""
    monkeypatch.syspath_prepend(str(_EXAMPLES))
    reader = importlib.import_module("review_mechanisms")
    rows, _ = reader.summarize_repetitions(
        pd.read_csv(_EXAMPLES / "mechanisms/study_estimates.csv"),
        pd.read_csv(_EXAMPLES / "mechanisms/population_references.csv"),
    )
    with pytest.raises(ValueError, match="same independent study seeds"):
        reader.paired_comparisons(rows.iloc[1:])
    changed = rows.copy()
    changed.loc[0, "reference"] += 1
    with pytest.raises(ValueError, match="matching target"):
        reader.paired_comparisons(changed)


def test_native_family_projection_reproduces_bundled_small_inputs(
    monkeypatch: pytest.MonkeyPatch, require_records, tmp_path: Path
) -> None:
    """All clinical-cell values can be reacquired from the original public records."""
    require_records("family-records.zip", "current-records.zip")
    monkeypatch.syspath_prepend(str(_EXAMPLES))
    for module, function, relative, archive in (
        ("review_registry", "project_native_trials", "registry/native_trials.csv", "current-records.zip"),
        ("review_registry", "project_native_trials", "registry/archived_native_trials.csv", "family-records.zip"),
        ("review_mechanisms", "project_cells", "mechanisms/clinical_cells.csv", "family-records.zip"),
    ):
        reader = importlib.import_module(module)
        observed = getattr(reader, function)(_EXAMPLES / archive)
        expected = pd.read_csv(_EXAMPLES / relative)
        pd.testing.assert_frame_equal(observed, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    registry_reader = importlib.import_module("review_registry")
    with pytest.raises(ValueError, match="differs from the declared source"):
        registry_reader.review(_EXAMPLES, tmp_path, family_archive=_EXAMPLES / "family-records.zip")
