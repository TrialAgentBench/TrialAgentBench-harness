"""Exact-marginal cross-domain linkage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trialagentbench_validation.external.realism.cross_domain_linkage import (
    analyze_cross_domain_linkage,
    combine_cross_domain_reports,
)


def test_linkage_breakage_preserves_marginals_and_changes_analysis() -> None:
    rng = np.random.default_rng(451)
    latent = rng.normal(size=300)
    frame = pd.DataFrame(
        {
            "assessment_count": rng.poisson(np.exp(1.0 + 0.4 * latent)),
            "biosample_count": rng.poisson(np.exp(0.5 + 0.3 * latent)),
            "adverse_event_count": rng.poisson(np.exp(0.2 + 0.5 * latent)),
            "intervention_count": rng.poisson(np.exp(0.7 + 0.2 * latent)),
        }
    )

    first = analyze_cross_domain_linkage(
        frame,
        source_object_id="source-one",
        source_sha256="a" * 64,
        worlds=20,
    )
    second = analyze_cross_domain_linkage(
        frame,
        source_object_id="source-two",
        source_sha256="b" * 64,
        worlds=20,
    )
    combined = combine_cross_domain_reports((first, second))

    assert len(first.estimates) == 100
    assert len(combined.studies) == 2
    assert all(row.mean_slope > 0 for row in first.responses)
    assert all(row.positive_slope_fraction >= 0.9 for row in first.responses)
    intact = [row for row in first.estimates if row.linkage_retention == 1.0]
    assert all(row.association_divergence == 0 for row in intact)
    assert all(row.safety_analysis_perturbation < 1e-12 for row in intact)
