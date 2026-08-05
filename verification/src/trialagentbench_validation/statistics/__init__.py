"""Statistical primitives shared by independent validation analyses."""

from trialagentbench_validation.statistics.operating_characteristics import (
    proportion_interval,
    scale_aware_tolerance,
)

__all__ = ["proportion_interval", "scale_aware_tolerance"]
