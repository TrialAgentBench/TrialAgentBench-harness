"""Independent share-package binary data contract tests."""

import pandas as pd
import pytest

from trialagentbench_harness.trialdev.grading.statistics import complete_binary_indicator_v1
from trialagentbench_harness.trialdev.share.inspect import _risk_at_horizon


@pytest.mark.parametrize("values", [[0.0, 0.4, 1.0], [0.0, None, 1.0]])
def test_event_indicator_rejects_nonbinary_or_missing_values(values: list[float | None]) -> None:
    """Public materialization cannot round or impute event indicators."""

    with pytest.raises(ValueError, match="complete binary 0/1"):
        complete_binary_indicator_v1(pd.Series(values))


def test_scenario_inspection_rejects_fractional_event_values() -> None:
    """Release summaries cannot hide malformed event state through thresholding."""

    with pytest.raises(ValueError, match="complete binary 0/1"):
        _risk_at_horizon(
            time=pd.Series([10.0, 20.0]),
            event=pd.Series([0.0, 0.6]),
            horizon=30.0,
        )
