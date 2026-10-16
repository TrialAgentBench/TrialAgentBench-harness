"""Consumer reconstruction and value accounting for disclosed native examples."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "scientific"
_RECORDS = (
    "current-records.zip",
    "design-records.zip",
    "development-records.zip",
    "family-records.zip",
    "recovery-records.zip",
)


@pytest.mark.parametrize("compact", [True, False], ids=["bundled", "complete"])
def test_disclosed_examples_recalculate_without_benchmark_imports(
    tmp_path: Path, require_records: Callable[..., None], compact: bool
) -> None:
    """Real numerical libraries reproduce source reconstruction and native value."""
    if not compact:
        require_records(*_RECORDS)
    output = tmp_path / "review"
    script = _EXAMPLES / "review.py"
    source = tmp_path / "disclosed-data"
    shutil.copytree(_EXAMPLES, source, ignore=shutil.ignore_patterns("*.py", "*.R", "*.md", "reports", "__pycache__"))
    assert not tuple(source.rglob("*.py")) and not tuple(source.rglob("*.R"))
    command = """
import importlib.abc
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
class RejectPrivate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('ariadne', 'trialagentbench', 'clinical_data_tables')):
            raise RuntimeError('Example attempted a benchmark import: ' + fullname)
sys.meta_path.insert(0, RejectPrivate())
sys.argv = [sys.argv[1], '--output', sys.argv[2], '--source', sys.argv[3], *sys.argv[4:]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            command,
            str(script),
            str(output),
            str(source),
            *(["--compact"] if compact else []),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    analyses = pd.read_csv(output / "trialeval_analyses.csv")
    assert len(analyses) == 3
    assert analyses.complete.all()
    assert analyses[["estimate", "lower", "upper", "mean_loss"]].notna().all().all()
    sensitivity = pd.read_csv(output / "trialeval_reference_sensitivity.csv")
    adjusted = sensitivity.loc[sensitivity.analysis.eq("Current records: adjusted HC3")]
    assert len(adjusted) == 2
    assert (adjusted.minimum_loss_difference <= adjusted.maximum_loss_difference).all()
    superseded = sensitivity.loc[sensitivity.analysis.eq("Superseded records: Welch")]
    assert len(superseded) == 2
    assert sensitivity[["minimum_loss_difference", "maximum_loss_difference"]].notna().all().all()
    assert (sensitivity.minimum_loss_difference <= sensitivity.maximum_loss_difference).all()
    law = pd.read_csv(output / "known_law.csv").iloc[0]
    assert abs(law.pre_rounding_law_contrast - (-6.098048)) < 1e-9
    costs = pd.read_csv(output / "trialdev_resources.csv")
    assert costs.enrolled.tolist() == [20, 120, 600]
    assert costs.cost_usd.sum() == 30_128_800
    # The approved fixed programme earns discounted market value; the declared
    # stopping rule stops on the noisy 20-patient early readout and keeps its costs.
    policy = pd.read_csv(output / "trialdev_policy_comparison.csv")
    assert policy.decision.tolist() == ["stop", "advance", "advance", "stop"]
    assert policy.net_value_usd.tolist() == pytest.approx(
        [0, 653_418_593.347052, 653_418_593.347052, -3_903_600], rel=0, abs=1e-3
    )
    phases = pd.read_csv(output / "trialdev_readout_estimates.csv")
    timeline = pd.read_csv(output / "trialdev_participant_timeline.csv")
    assert timeline.participant_id.nunique() == 20 and len(timeline) == 40
    assert timeline.groupby("participant_id").site_id.nunique().eq(1).all()
    assert timeline.occurrence_time.equals(timeline.availability_time)
    elapsed = timeline.occurrence_time - timeline.enrollment_day
    assert np.isclose(elapsed, 0).sum() == 20
    final = timeline.loc[np.isclose(elapsed, 28)]
    assert len(final) == 20 and final.groupby("arm")["observed"].sum().to_dict() == {"active": 10, "control": 10}
    assert timeline.availability_time.le(150).all()
    for extension in ("png", "pdf", "svg"):
        assert (output / f"trialdev_participant_timeline.{extension}").stat().st_size > 0
    # Execute the actual copyable tutorial in a workspace containing only one
    # visible readout. Separate processes cannot inherit interactive state.
    report = (output / "trialdev_study.md").read_text()
    saved_analysis = report.split("```python\n", 1)[1].split("```", 1)[0]
    for phase in phases.itertuples():
        workspace = tmp_path / f"submission-{phase.phase}"
        scratch = workspace / "scratch"
        scratch.mkdir(parents=True)
        shutil.copytree(source / "trialdev/readouts" / f"{phase.phase}-look", workspace / "readout")
        (scratch / "analysis.py").write_text(saved_analysis)
        replays = []
        for _ in range(2):
            subprocess.run([sys.executable, "-I", "scratch/analysis.py"], cwd=workspace, check=True, timeout=30)
            replays.append((scratch / "estimates.csv").read_bytes())
        assert replays[0] == replays[1]
        submitted = pd.read_csv(scratch / "estimates.csv")
        assert submitted.columns.tolist() == ["target_id", "estimate", "lower", "upper"]
        assert submitted.target_id.str.rsplit(":", n=1).str[-1].tolist() == ["clinical_outcome", "safety_risk"]
        np.testing.assert_allclose(
            submitted[["estimate", "lower", "upper"]],
            [
                [phase.estimate, phase.lower, phase.upper],
                [phase.safety_difference, phase.safety_lower, phase.safety_upper],
            ],
            rtol=0,
            atol=1e-12,
        )
    early = phases.iloc[0]
    assert early.safety_difference == 0
    assert early.safety_lower < 0 < early.safety_upper
    assert phases.finite_outcome_reference.nunique() == 3
    controls = pd.read_csv(output / "trialdev_time_controls.csv")
    assert len(controls) == 11
    current = controls.loc[controls.method.eq("scheduled_visit")].reset_index(drop=True)
    np.testing.assert_allclose(
        current[["estimate", "lower", "upper", "clinical_loss"]],
        phases[["estimate", "lower", "upper", "clinical_loss"]],
    )
    wrong_clock = controls.loc[controls.method.eq("programme_clock_as_duration")]
    assert wrong_clock.status.eq("insufficient_observations").all()
    assert wrong_clock[["observed_active", "observed_control"]].eq(0).all().all()
    assert wrong_clock[["estimate", "lower", "upper", "clinical_loss"]].isna().all().all()
    baseline_controls = controls.loc[controls.method.eq("baseline_as_followup")]
    for row in baseline_controls.itertuples():
        values = []
        for arm in ("active", "control"):
            # The disclosed CSV stores each baseline first, followed by its visit.
            # This independent extraction does not use the reader's time predicate.
            records = pd.read_csv(
                source / "trialdev/readouts" / f"{row.study_id}-look/arms" / arm / "measurements.csv"
            )
            first = records.groupby("participant_id", sort=False).head(1)
            values.append(first.loc[first.observed, "observed_outcome"].mean())
        assert row.estimate == pytest.approx(values[0] - values[1])
    stale = controls.loc[controls.method.eq("previous_study")].reset_index(drop=True)
    assert stale.source_study_id.tolist() == current.study_id.tolist()[:-1]
    assert stale.study_id.tolist() == current.study_id.tolist()[1:]
    np.testing.assert_allclose(stale.estimate, current.estimate.iloc[:-1])
    np.testing.assert_allclose(stale.finite_outcome_reference, current.finite_outcome_reference.iloc[1:])
    associations = pd.read_csv(output / "trialdev_initial_associations.csv")
    assert len(associations) == 18
    assert associations[["estimate", "lower", "upper", "association_reference"]].notna().all().all()
    derivation = pd.read_csv(output / "trialdev_initial_reference_derivation.csv")
    assert derivation.filter(like="native_error").abs().to_numpy().max() < 1e-11
    assert (output / "trialeval_analyses.pdf").is_file()
    latent = pd.read_csv(output / "latent_selection_control.csv")
    assert len(latent) == 18
    assert latent.native_error.abs().max() < 1e-8
    assert latent.normal_tail_error_bound.max() < 1e-18
    continuous = latent.loc[latent.family.eq("continuous") & latent.participant_id.eq(1)].set_index("asset_id")
    observed = continuous.observed_conditional_mean["asset_b"] - continuous.observed_conditional_mean["control"]
    intervention = continuous.intervention_mean["asset_b"] - continuous.intervention_mean["control"]
    assert abs(observed - intervention) > 0.1

    episode = pd.read_csv(output / "interaction-timeline.csv")
    readouts = episode.loc[episode.action.eq("readout")]
    np.testing.assert_allclose(readouts[["estimate", "lower", "upper"]], phases[["estimate", "lower", "upper"]])
    assert episode.programme_day.is_monotonic_increasing
    assert readouts.available_outcomes.tolist() == [20, 134, 713]
    assert episode.iloc[-1].study_cost_usd == pytest.approx(30_128_800)
    assert episode.iloc[-1].budget_remaining_usd == pytest.approx(19_871_200)
    # Readouts precede same-day setup charges and cannot contain later evidence.
    assert episode.iloc[1].study_cost_usd == pytest.approx(1_653_600)
    assert episode.iloc[2].study_cost_usd == pytest.approx(5_153_600)
    assert episode.iloc[2].available_outcomes == episode.iloc[1].available_outcomes
    assert (output / "index.html").is_file()
    if compact:
        return

    paired_survival = pd.read_csv(output / "paired_survival_analyses.csv")
    assert len(paired_survival) == 4 and paired_survival.target_id.nunique() == 2
    assert paired_survival[["estimate", "lower", "upper", "loss", "reference_mcse"]].notna().all().all()
    assert (paired_survival.lower <= paired_survival.estimate).all()
    assert (paired_survival.estimate <= paired_survival.upper).all()
    curves = pd.read_csv(output / "families/paired_survival_curves.csv").pivot(
        index="time_day", columns="arm", values="survival"
    )
    assert curves.index.min() == 0 and curves.index.max() == 730
    assert curves.diff().dropna().le(0).all().all()
    km = paired_survival.loc[paired_survival.method.eq("Kaplan–Meier")].set_index("effect_measure")
    assert curves.control.iloc[-1] - curves.active.iloc[-1] == pytest.approx(km.loc["risk_difference", "estimate"])
    area = np.sum((curves.active - curves.control).iloc[:-1].to_numpy() * np.diff(curves.index))
    assert area == pytest.approx(km.loc["restricted_mean_survival_time_difference", "estimate"])
    for extension in ("png", "pdf", "svg", "md"):
        assert (output / f"paired_survival.{extension}").stat().st_size > 0
    assert (output / "reference_agreement.pdf").is_file()
    coverage = pd.read_csv(output / "proof_coverage.csv")
    assert len(coverage) == 30
    assert coverage.source_reconstruction.eq("passed").all()
    materiality = pd.read_csv(output / "interpretation-materiality.csv")
    five = materiality.loc[materiality.threshold_pp.eq(5)]
    assert len(five) == 64 and five.seed.nunique() == 32
    assert (five.loc[five.method.eq("Routine endpoint"), "conclusion"] == "rules out target").sum() == 14
    assert (five.loc[five.method.eq("Stratified adjudication"), "conclusion"] == "rules out target").sum() == 0
    recovery = pd.read_csv(output / "recovery_operating_characteristics.csv")
    assert recovery.completed.eq(32).all() and recovery.attempted.eq(32).all()
    assert recovery.coverage.between(0, 1).all()
    losses = pd.read_csv(output / "mean_loss_summary.csv")
    assert len(losses) == 18 and losses.n.eq(32).all()
    assert losses.mean_loss.ge(0).all()
    paired = pd.read_csv(output / "paired_loss_contrasts.csv")
    assert len(paired) == 11 and paired.n.eq(32).all()
    assert (paired.sampling_lower <= paired.mean_paired_loss_difference).all()
    assert (paired.mean_paired_loss_difference <= paired.sampling_upper).all()
    assert (recovery.reference_coverage_min <= recovery.coverage).all()
    assert (recovery.reference_coverage_max >= recovery.coverage).all()
    references = pd.read_csv(output / "population_references.csv")
    null = references.loc[references.case.eq("TE-S10-A1-null")].iloc[0]
    assert null.reference == 0 and null.reference_se == 0
    development = pd.read_csv(output / "trialdev_recovery_estimates.csv")
    assert len(development) == 1920 and development.profile.nunique() == 4
    assert development.groupby(["profile", "checkpoint", "method"]).size().eq(32).all()
    assert development.status.value_counts().to_dict() == {"estimated": 1536, "insufficient_observations": 384}
    assert development.loc[development.status.ne("estimated"), ["lower", "upper", "loss"]].isna().all().all()
    development_summary = pd.read_csv(output / "trialdev_recovery_summary.csv")
    assert len(development_summary) == 60 and development_summary.attempted.eq(32).all()
    development_contrasts = pd.read_csv(output / "trialdev_recovery_contrasts.csv")
    assert len(development_contrasts) == 44
    assert development_contrasts.attempted.eq(32).all()
    assert development_contrasts.loc[development_contrasts.paired.eq(0), "mean_loss_difference"].isna().all()
    programmes = pd.read_csv(output / "authored_programme_values.csv")
    assert len(programmes) == 120 and programmes.complete.all()
    assert programmes.world.nunique() == 6
    studies = pd.read_csv(output / "authored_study_estimates.csv")
    staged = studies.loc[studies.design.eq("staged") & studies.policy.eq("fixed")]
    assert set(staged.loc[staged.phase.ne("confirmation"), "activated_sites"]) == {6}
    final = staged.loc[staged.phase.eq("confirmation")]
    assert final.patients.eq(900).all() and final.activated_sites.eq(60).all()
    assert final.cumulative_cost.le(50_000_000).all()
    assert final.represented_sites.le(final.target_sites).all()
    assert studies.site_total_variation.between(0, 1).all()
    assert set(programmes.decision) == {"advance", "stop"}
    assert programmes.value_usd.min() < 0 < programmes.value_usd.max()
    for name in ("study_design_recovery", "repeated_study_recovery", "authored_design_utility"):
        assert (output / f"{name}.pdf").is_file()
    report = (output / "index.html").read_text()
    assert "not executed in this run" in report
    assert "reference_agreement.csv" in report
    assert "clinical-reference-results.csv" not in report
    from html.parser import HTMLParser

    class LocalLinks(HTMLParser):
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            for name, value in attrs:
                if name in {"href", "src"} and value and not value.startswith("#"):
                    assert (output / value).is_file(), value

    LocalLinks().feed(report)


def test_disclosed_raw_ready_disagreement_is_not_silently_accepted(tmp_path: Path) -> None:
    """A damaged current source value cannot pass the paired reconstruction check."""
    copied = tmp_path / "example"
    shutil.copytree(_EXAMPLES, copied)
    path = copied / "trialeval" / "endpoint_observations.csv"
    records = pd.read_csv(path)
    records.loc[0, "motor_disability_score"] += 1
    records.to_csv(path, index=False)
    result = subprocess.run(
        [sys.executable, str(copied / "review.py"), "--output", str(tmp_path / "review")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "DataFrame.iloc" in result.stderr


def test_disclosed_initial_reference_requires_its_independently_derived_law(tmp_path: Path) -> None:
    """A substituted reference cannot pass merely because the example reads a reference table."""
    copied = tmp_path / "example"
    shutil.copytree(_EXAMPLES, copied)
    path = copied / "trialdev" / "initial_references.csv"
    reference = pd.read_csv(path)
    reference.loc[0, "outcome_mean_difference"] += 0.1
    reference.to_csv(path, index=False)
    result = subprocess.run(
        [sys.executable, str(copied / "review.py"), "--output", str(tmp_path / "review")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "Native association reference disagrees" in result.stderr


def test_family_source_checks_reject_missing_and_changed_terminal_records(tmp_path: Path) -> None:
    """Source reconstruction fails on silent loss or alteration of terminal records."""
    command = """
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
import pandas as pd
import numpy as np
scope = runpy.run_path(sys.argv[1])
check = scope['reconstruct_survival']
analysis = pd.DataFrame({'participant_id': [1, 2, 3], 'observed_time': [3., 5., 9.],
                         'event_observed': [True, False, False]})
events = pd.DataFrame({'participant_id': [1, 2, 3, 1],
                       'event_id': ['primary', 'loss', 'administrative', 'safety'],
                       'event_kind': ['endpoint', 'follow_up', 'follow_up', 'safety_event'],
                       'occurrence_time': [3., 5., 9., 1.]})
check(analysis, events, 'primary')
try:
    check(analysis, events.loc[events.participant_id != 3], 'primary')
except ValueError:
    pass
else:
    raise AssertionError('Missing terminal record was accepted')
events.loc[events.participant_id == 2, 'occurrence_time'] = 6
try:
    check(analysis, events, 'primary')
except AssertionError:
    pass
else:
    raise AssertionError('Changed terminal time was accepted')
validated = pd.DataFrame({'assigned_arm': ['active'] * 4 + ['control'] * 4,
    'validation_stratum': ['low', 'low', 'high', 'high'] * 2,
    'validation_selected': [True, False] * 4,
    'adjudicated_endpoint': [2., np.nan, 6., np.nan, 4., np.nan, 8., np.nan],
    'count': [1., 1., 2., 2., 3., 3., 4., 4.]})
result = scope['point_estimates'](validated, endpoint='count', family='count',
                                 scale='rate_ratio', horizon=365., cell='test', covariates=())
assert np.isclose(result[1], 4 / 6)
assert np.isclose(result[3], 1.5 / 3.5)
# Exact enumeration over two independent retrieval coins checks HT weighting.
from itertools import product
retrieval = pd.DataFrame({'assigned_arm': ['active', 'active', 'control', 'control'],
    'routine_observed': [True, False, True, False],
    'endpoint_recovered': [False] * 4,
    'outcome': [10., np.nan, 5., np.nan]})
estimates = []
for first, second in product((False, True), repeat=2):
    copy = retrieval.copy()
    copy.loc[1, ['endpoint_recovered', 'outcome']] = [first, 30. if first else np.nan]
    copy.loc[3, ['endpoint_recovered', 'outcome']] = [second, 15. if second else np.nan]
    estimates.append(scope['recovery_estimate'](copy, 'outcome', 0.5)[0])
assert np.isclose(np.mean(estimates), 10.)
trial = pd.DataFrame({'assigned_arm': ['active'] * 4 + ['control'] * 4,
    'observed_time': [2., 2., 10., 10., 4., 10., 10., 10.],
    'event_observed': [True, True, False, False, True, False, False, False]})
estimate, se = scope['horizon_risk'](trial, 10.)
assert np.isclose(estimate, .25)
assert np.isclose(se**2, .5 * .5 / 4 + .25 * .75 / 4)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", command, str(_EXAMPLES / "review_families.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_bundled_family_reader_reproduces_clinical_report_without_private_imports(
    tmp_path: Path, require_records: Callable[..., None]
) -> None:
    """The distributed reader needs no benchmark installation."""
    require_records("family-records.zip")
    output = tmp_path / "clinical-review"
    command = """
import importlib.abc
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
class RejectPrivate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('ariadne', 'trialagentbench', 'clinical_data_tables')):
            raise RuntimeError('Private import: ' + fullname)
sys.meta_path.insert(0, RejectPrivate())
sys.argv = [sys.argv[1], '--output', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", command, str(_EXAMPLES / "review_families.py"), str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    flow = pd.read_csv(output / "asthma_flow.csv")
    assert (flow.routine_status + flow.status_retrieved + flow.still_missing).equals(flow.randomized)
    planning = pd.read_csv(output / "asthma_sample_size.csv").iloc[0]
    assert planning.planned_n == flow.randomized.sum()
    reconstructed = pd.read_csv(output / "asthma_reconstructed.csv")
    assert len(reconstructed) == planning.planned_n
    assert reconstructed.horizon_endpoint.isna().sum() == flow.still_missing.sum()
    assert len(pd.read_csv(output / "asthma_results.csv")) == 3
    assert len(pd.read_csv(output / "family_diagnostics.csv")) == 30
    report = (output / "asthma_trial.md").read_text()
    assert f"randomized {len(reconstructed):,} adults" in report
    assert (output / "asthma_trial.pdf").is_file()
    categories = pd.read_csv(output / "clinical_baseline_categories.csv")
    maternity_sex = categories.loc[categories.cell.eq("TE-S08-A1") & categories.field.eq("sex")]
    assert len(maternity_sex) > 0
    assert maternity_sex.observed_probability.sum() == pytest.approx(1)
    assert maternity_sex.observed_probability.max() == 1
    dependence = pd.read_csv(output / "clinical_baseline_dependence.csv")
    constant = dependence.loc[dependence.comparison_status.eq("constant_margin")]
    assert not constant.empty and constant.observed_spearman.isna().all()
    incomparable = dependence.loc[dependence.comparison_status.eq("different_scales")]
    assert not incomparable.empty and incomparable.target_spearman_before_recording.isna().all()
    comparable = dependence.loc[dependence.comparison_status.eq("before_recording_reference")]
    assert not comparable.empty and comparable.target_spearman_before_recording.notna().all()


def test_cluster_deletions_match_independent_survival_library(
    tmp_path: Path, require_records: Callable[..., None]
) -> None:
    """The count-based acceleration preserves exact leave-cluster-out KM estimates."""
    require_records("recovery-records.zip")
    command = r"""
import sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[1])
from review_families import cluster_risk_deletions, horizon_risk
with zipfile.ZipFile(Path(sys.argv[1]) / 'recovery-records.zip') as archive:
    data = pd.read_parquet(archive.open('TE-S07-A1-favourable/observed-0.parquet'))
values = cluster_risk_deletions(data, 180.)
clusters = data.cluster_id.unique()
for index in (0, 7, 48, len(clusters)-1):
    conventional = horizon_risk(data.loc[data.cluster_id.ne(clusters[index])], 180.)[0]
    np.testing.assert_allclose(values[index], conventional, rtol=0, atol=5e-13)
"""
    subprocess.run([sys.executable, "-I", "-c", command, str(_EXAMPLES)], cwd=tmp_path, check=True, timeout=30)


def test_paired_survival_reader_reconstructs_both_targets_and_rejects_tampering(tmp_path, monkeypatch) -> None:
    """Complete event flags distinguish events at the horizon from event-free follow-up."""
    import runpy

    import pytest

    monkeypatch.syspath_prepend(str(_EXAMPLES))
    read = runpy.run_path(str(_EXAMPLES / "review_families.py"))["paired_survival_targets"]
    complete = pd.DataFrame(
        {
            "participant_id": range(8),
            "assigned_arm": ["active"] * 4 + ["control"] * 4,
            "complete_event_by_horizon": [True, False, False, False, True, True, False, False],
            "complete_event_free_time": [10.0, 10.0, 10.0, 10.0, 2.0, 6.0, 10.0, 10.0],
        }
    )
    totals = pd.DataFrame(
        {
            "target_id": ["risk", "risk", "rmst", "rmst"],
            "effect_measure": ["risk_difference"] * 2 + ["restricted_mean_survival_time_difference"] * 2,
            "unit": ["proportion"] * 2 + ["day"] * 2,
            "arm": ["active", "control"] * 2,
            "total": [1.0, 2.0, 40.0, 28.0],
            "n": [4] * 4,
        }
    )
    complete.to_parquet(tmp_path / "disclosed_complete_outcomes.parquet", index=False)
    totals.to_csv(tmp_path / "paired_oracle_totals.csv", index=False)
    result = read(tmp_path, horizon=10).set_index("target_id")
    assert result.loc["risk", "finite_complete_contrast"] == -0.25
    assert result.loc["rmst", "finite_complete_contrast"] == 3.0
    for column, value in (("total", 2.0), ("n", 5), ("unit", "day"), ("target_id", "rmst")):
        changed = totals.copy()
        changed.loc[0, column] = value
        changed.to_csv(tmp_path / "paired_oracle_totals.csv", index=False)
        with pytest.raises((ValueError, AssertionError)):
            read(tmp_path, horizon=10)
    totals.to_csv(tmp_path / "paired_oracle_totals.csv", index=False)
    for column, value in (("participant_id", 1), ("complete_event_free_time", 11.0)):
        changed = complete.copy()
        changed.loc[0, column] = value
        changed.to_parquet(tmp_path / "disclosed_complete_outcomes.parquet", index=False)
        with pytest.raises(ValueError):
            read(tmp_path, horizon=10)


def test_paired_conditional_law_matches_closed_form_exponential_moments(tmp_path, monkeypatch) -> None:
    """Independent integration recovers known risk, RMST and sampling variance."""
    import math
    import runpy

    import pytest

    monkeypatch.syspath_prepend(str(_EXAMPLES))
    read = runpy.run_path(str(_EXAMPLES / "review_families.py"))["paired_survival_law"]
    arms = ["active", "active", "control", "control"]
    complete = pd.DataFrame(
        {
            "participant_id": range(4),
            "assigned_arm": arms,
            "complete_event_by_horizon": [False] * 4,
            "complete_event_free_time": [10.0] * 4,
        }
    )
    complete.to_parquet(tmp_path / "disclosed_complete_outcomes.parquet", index=False)
    pd.DataFrame(
        {
            "target_id": ["risk", "risk", "rmst", "rmst"],
            "effect_measure": ["risk_difference"] * 2 + ["restricted_mean_survival_time_difference"] * 2,
            "unit": ["proportion"] * 2 + ["day"] * 2,
            "arm": ["active", "control"] * 2,
            "total": [0.0, 0.0, 20.0, 20.0],
            "n": [2] * 4,
        }
    ).to_csv(tmp_path / "paired_oracle_totals.csv", index=False)
    covariates = pd.DataFrame({"participant_id": range(4), "assigned_arm": arms, "log_hazard_multiplier": 0.0})
    covariates["baseline"] = 0.0
    pd.DataFrame({"variable": ["baseline"], "coefficient": [0.5]}).to_csv(
        tmp_path / "survival_risk_terms.csv", index=False
    )
    covariates.to_csv(tmp_path / "survival_covariates.csv", index=False)
    law = pd.DataFrame(
        {
            "start": [0.0],
            "end": [10.0],
            "shape": [1.0],
            "scale": [20.0],
            "reference_risk": [0.5],
            "control_log_hazard": [0.0],
            "active_log_hazard": [-math.log(2)],
        }
    )
    law.to_csv(tmp_path / "survival_law.csv", index=False)
    result = read(tmp_path, horizon=10).set_index("effect_measure")
    risks, means, variances = [], [], []
    for rate in (math.log(2) / 20, math.log(2) / 10):
        risk = -math.expm1(-10 * rate)
        mean = risk / rate
        second = 2 * (1 - (1 + 10 * rate) * math.exp(-10 * rate)) / rate**2
        risks.append(risk)
        means.append(mean)
        variances.append(second - mean**2)
    risk_row = result.loc["risk_difference"]
    rmst_row = result.loc["restricted_mean_survival_time_difference"]
    assert risk_row.conditional_expected_contrast == pytest.approx(risks[0] - risks[1], abs=1e-10)
    assert rmst_row.conditional_expected_contrast == pytest.approx(means[0] - means[1], abs=1e-10)
    assert risk_row.conditional_sampling_se**2 == pytest.approx(sum(p * (1 - p) / 2 for p in risks), abs=1e-10)
    assert rmst_row.conditional_sampling_se**2 == pytest.approx(sum(variances) / 2, abs=1e-10)
    altered = covariates.assign(baseline=1.0)
    altered.to_csv(tmp_path / "survival_covariates.csv", index=False)
    with pytest.raises(AssertionError):
        read(tmp_path, horizon=10)
    for changed in (covariates.iloc[:-1], covariates.assign(assigned_arm=list(reversed(arms)))):
        changed.to_csv(tmp_path / "survival_covariates.csv", index=False)
        with pytest.raises(ValueError):
            read(tmp_path, horizon=10)
    covariates.to_csv(tmp_path / "survival_covariates.csv", index=False)
    law.assign(start=1.0).to_csv(tmp_path / "survival_law.csv", index=False)
    with pytest.raises(ValueError, match="period support"):
        read(tmp_path, horizon=10)


def test_disclosed_csv_preserves_absent_and_unknown_seriousness(tmp_path: Path, monkeypatch) -> None:
    """CSV interchange must distinguish known negative criteria from missing evidence."""
    import io
    import json
    import runpy
    import zipfile

    monkeypatch.syspath_prepend(str(_EXAMPLES))
    reconcile = runpy.run_path(str(_EXAMPLES / "review_standards.py"))["reconcile_readout"]
    source = _EXAMPLES / "trialdev" / "readouts" / "confirmation-look"
    readout = tmp_path / "readout"
    shutil.copytree(source, readout)
    first_arm = json.loads((readout / "study.json").read_text())["arms"][0]["arm_id"]
    for arm in ("control", "active"):
        path = readout / "arms" / arm / "safety_events.csv"
        frame = pd.read_csv(path)
        frame["seriousness_criteria"] = ""
        if arm == first_arm:
            frame.loc[frame.index[0], "seriousness_criteria"] = None
        frame.to_csv(path, index=False, na_rep="NA")
    archive = tmp_path / "standards.zip"
    with zipfile.ZipFile(source / "standards.zip") as original, zipfile.ZipFile(archive, "w") as revised:
        for name in original.namelist():
            content = original.read(name)
            if name == "AE.parquet":
                ae = pd.read_parquet(io.BytesIO(content))
                for column in ("AESCONG", "AESDISAB", "AESDTH", "AESHOSP", "AESLIFE", "AESMIE"):
                    ae[column] = "N"
                    ae.loc[ae.index[0], column] = ""
                buffer = io.BytesIO()
                ae.to_parquet(buffer, index=False)
                content = buffer.getvalue()
            revised.writestr(name, content)
    reconcile(readout, archive, extension="csv")
    import pytest

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(tampered, "w") as revised:
        for name in original.namelist():
            content = original.read(name)
            if name == "SUPPAE.parquet":
                supplemental = pd.read_parquet(io.BytesIO(content))
                index = supplemental.index[supplemental.QNAM.eq("ELAPDAY")][0]
                supplemental.loc[index, "QVAL"] = str(float(supplemental.loc[index, "QVAL"]) + 1e-6)
                buffer = io.BytesIO()
                supplemental.to_parquet(buffer, index=False)
                content = buffer.getvalue()
            revised.writestr(name, content)
    with pytest.raises(AssertionError):
        reconcile(readout, tampered, extension="csv")


def test_explicit_scientific_source_never_falls_back_to_bundled_data(tmp_path: Path) -> None:
    """An incomplete selected corpus cannot silently produce a bundled-data report."""
    source = tmp_path / "empty-source"
    source.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(_EXAMPLES / "review.py"),
            "--source",
            str(source),
            "--output",
            str(tmp_path / "report"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert str(source / "trialeval" / "participants.csv") in result.stderr
    assert not (tmp_path / "report" / "index.html").exists()


def test_disclosed_context_reconstructs_intercepts_slopes_and_rejects_missing_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent table joins reproduce context terms and reject incomplete laws."""
    import runpy

    import pytest

    monkeypatch.syspath_prepend(str(_EXAMPLES))
    scope = runpy.run_path(str(_EXAMPLES / "review_trialdev.py"))
    rows = []
    for endpoint, identity, covariate, coefficients in (
        ("outcome", "intercept", "", (1.0, -1.0)),
        ("outcome", "slope", "raw_score", (2.0, 4.0)),
        ("safety", "safety", "", (-0.3, 0.7)),
    ):
        for level, coefficient in zip(("site_a", "site_b"), coefficients, strict=True):
            rows.append(
                dict(
                    endpoint=endpoint,
                    effect_id=identity,
                    group_field="site_id",
                    group_level=level,
                    coefficient=coefficient,
                    input_field=covariate,
                )
            )
    context = pd.DataFrame(rows)
    path = tmp_path / "initial_context.csv"
    context.to_csv(path, index=False)
    observed = pd.DataFrame({"site_id": ["site_a", "site_b"], "raw_score": [999, 999]})
    decoded = pd.DataFrame({"raw_score": [3.0, 5.0]})
    actual = scope["initial_context_offsets"](tmp_path, observed, decoded)
    pd.testing.assert_frame_equal(actual, pd.DataFrame({"outcome": [7.0, 19.0], "safety": [-0.3, 0.7]}))
    pd.concat([context, context.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="repeats"):
        scope["initial_context_offsets"](tmp_path, observed, decoded)
    context.loc[context.group_level.eq("site_a")].to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite coefficient"):
        scope["initial_context_offsets"](tmp_path, observed, decoded)


def test_paired_survival_intervals_match_complete_empirical_moments(tmp_path, monkeypatch) -> None:
    """Both marginal KM variances reduce to empirical means without censoring."""
    import runpy

    import numpy as np

    monkeypatch.syspath_prepend(str(_EXAMPLES))
    calculate = runpy.run_path(str(_EXAMPLES / "review_families.py"))["paired_survival_estimates"]
    data = pd.DataFrame(
        {
            "assigned_arm": ["active"] * 4 + ["control"] * 4,
            "observed_time": [1.0, 2.0, 4.0, 4.0] * 2,
            "event_observed": [True, True, False, False] * 2,
        }
    )
    result = calculate(data, horizon=4.0)
    km = result.loc[result.method.eq("Kaplan–Meier")].set_index("effect_measure")
    assert km.estimate.tolist() == [0.0, 0.0]
    assert km.loc["risk_difference", "se"] == pytest.approx(np.sqrt(2 * 0.5 * 0.5 / 4))
    assert km.loc["restricted_mean_survival_time_difference", "se"] == pytest.approx(
        np.sqrt(2 * np.var([1.0, 2.0, 4.0, 4.0]) / 4)
    )
    with pytest.raises(ValueError, match="horizon support"):
        calculate(data.assign(observed_time=data.observed_time.clip(upper=3.0)), horizon=4.0)
    with pytest.raises(ValueError, match="two arms"):
        calculate(data.loc[data.assigned_arm.eq("active")], horizon=4.0)


def test_development_clocks_preserve_duration_and_reject_adjacent_visits(tmp_path: Path) -> None:
    """Elapsed follow-up is invariant; a calendar denominator and wrong visit are not."""
    command = r"""
import sys
sys.path.insert(0,sys.argv[1])
import numpy as np
import pandas as pd
from review_recovery import development_samples
baseline = pd.DataFrame({'participant_id':['p'], 'enrollment_day':[100.]})
events = pd.DataFrame({'participant_id':['p']*3, 'event':[1,1,0], 'occurrence_time':[102.,105.,110.]})
a = development_samples(baseline,events,20)
assert a['observed_duration'].iloc[0] == 73
assert a['full_horizon_denominator'].iloc[0] == 36.5
assert np.isclose(a['programme_clock_denominator'].iloc[0], 730/110)
baseline.enrollment_day += 10000
b = development_samples(baseline,events.assign(occurrence_time=events.occurrence_time + 10000),20)
assert b['observed_duration'].equals(a['observed_duration'])
assert b['full_horizon_denominator'].equals(a['full_horizon_denominator'])
assert not b['programme_clock_denominator'].equals(a['programme_clock_denominator'])
measurements = pd.DataFrame({'participant_id':['p','p'], 'outcome_id':['clinical_outcome']*2,
    'occurrence_time':[10120.,np.nextafter(10120.,np.inf)],'observed':[True,True],'observed_outcome':[3.,99.]})
values = development_samples(baseline,measurements,20)
assert values['scheduled_visit'].tolist() == [3.]
assert values['programme_clock_as_duration'].empty
"""
    subprocess.run([sys.executable, "-I", "-c", command, str(_EXAMPLES)], cwd=tmp_path, check=True, timeout=30)


def test_compact_review_rejects_unexecuted_r_request(tmp_path: Path) -> None:
    """An explicitly requested R comparison cannot silently disappear in compact mode."""
    output = tmp_path / "review"
    result = subprocess.run(
        [sys.executable, str(_EXAMPLES / "review.py"), "--compact", "--rscript", "Rscript", "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert "--rscript requires the complete review" in result.stderr
    assert not output.exists()
