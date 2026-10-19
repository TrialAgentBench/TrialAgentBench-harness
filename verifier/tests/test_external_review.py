"""Independent numerical and consumer checks for the public real-trial reader."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "scientific"


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the ordinary-library reader directly from its public location."""
    monkeypatch.syspath_prepend(str(_EXAMPLES))
    return importlib.import_module("review_external")


@pytest.fixture
def trial() -> pd.DataFrame:
    """Create a deterministic heteroscedastic two-arm trial with linked measures."""
    rng = np.random.RandomState(926)
    arm = np.repeat(["control", "treated"], [48, 52])
    age = rng.normal(55, 10, len(arm))
    bmi = 22 + age / 10 + rng.normal(size=len(arm))
    outcome = 3 * (arm == "treated") + 0.7 * age - bmi + rng.normal(size=len(arm)) * (age / 20)
    return pd.DataFrame({"arm": arm, "outcome": outcome, "age": age, "bmi": bmi})


def test_external_reader_imports_without_private_packages(tmp_path: Path) -> None:
    """The actual CLI imports and explains its contract with private modules blocked."""
    command = """
import importlib.abc
import runpy
import sys
class RejectPrivate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('ariadne', 'trialagentbench', 'clinical_data_tables')):
            raise RuntimeError('Unexpected benchmark import: ' + fullname)
sys.meta_path.insert(0, RejectPrivate())
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", command, str(_EXAMPLES / "review_external.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--source" in result.stdout and "--reference" in result.stdout


def test_adjusted_hc1_matches_independent_sandwich(reader: ModuleType, trial: pd.DataFrame) -> None:
    """Recalculate the coefficient, HC1 covariance and finite-df interval with NumPy."""
    x = np.column_stack((np.ones(len(trial)), trial.arm.eq("treated"), trial[["age", "bmi"]]))
    y = trial.outcome.to_numpy()
    coefficient = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coefficient
    inverse = np.linalg.inv(x.T @ x)
    covariance = inverse @ np.einsum("ni,n,nj->ij", x, residual**2, x) @ inverse
    covariance *= len(trial) / (len(trial) - x.shape[1])
    se = np.sqrt(covariance[1, 1])
    margin = t.ppf(0.975, len(trial) - x.shape[1]) * se
    expected = [coefficient[1], se, coefficient[1] - margin, coefficient[1] + margin]
    np.testing.assert_allclose(reader._fit(trial), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reader._fit(trial, reverse=True), [-expected[0], se, -expected[3], -expected[2]])
    with pytest.raises(ValueError, match="rank deficient"):
        reader._fit(trial.assign(bmi=trial.age))


def test_structural_screen_checks_design_and_measurements(reader: ModuleType, trial: pd.DataFrame) -> None:
    """Source design and support determine inclusion before treatment contrasts."""
    data = trial.rename(columns={"arm": "Treatment", "outcome": "YP_value", "age": "X_age", "bmi": "X_bmi"})
    assert reader._screen(data, "Continuous", False) == ("eligible", ["YP_value", "X_age"])
    assert reader._screen(data, "Continuous", True) == ("eligible", ["YP_value", "X_age", "X_bmi"])
    assert reader._screen(data, "Time-to-event", False)[0] == "unsupported_source_endpoint_type"
    assert reader._screen(data.assign(Participant_ID=np.arange(len(data))), "Continuous", False)[0] == (
        "explicit_repeated_or_crossover_design"
    )
    assert reader._screen(data.assign(X_age=np.inf), "Continuous", True)[0] == "nonfinite_measurement"
    assert reader._screen(data.iloc[:52], "Continuous", True)[0] == "insufficient_complete_case_arm_support"


def test_paired_permutation_preserves_arm_marginals(
    reader: ModuleType, trial: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observe actual fitted draws: only participant linkage differs within each pair."""
    observed: list[pd.DataFrame] = []
    original = reader._fit

    def capture(data: pd.DataFrame) -> tuple[float, float, float, float]:
        observed.append(data.copy())
        return original(data)

    monkeypatch.setattr(reader, "_fit", capture)
    metrics = pd.DataFrame(reader._resamples(trial, 24, {"seed": 451013, "resamples_per_trial": 3}))
    assert len(observed) == 7
    for whole, permuted in zip(observed[1::2], observed[2::2], strict=True):
        assert whole.groupby("arm").size().equals(trial.groupby("arm").size())
        for arm, block in whole.groupby("arm"):
            other = permuted.loc[permuted.arm.eq(arm)]
            for column in trial.columns[1:]:
                np.testing.assert_array_equal(np.sort(block[column]), np.sort(other[column]))
        assert not np.allclose(reader._correlations(whole), reader._correlations(permuted))
    marginals = metrics.pivot(index="replicate", columns="method", values="standardized_wasserstein")
    np.testing.assert_array_equal(marginals.whole_subject, marginals.permuted_linkage)
