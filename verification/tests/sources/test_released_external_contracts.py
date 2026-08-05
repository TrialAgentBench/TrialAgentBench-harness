"""Checks for the released external-validation contracts."""

from pathlib import Path

from trialagentbench_validation.external.contracts import (
    ConstructMapV1,
    ExternalSourceManifestV1,
    ExternalValidationDesignV1,
)

CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "external_validation"


def test_released_external_validation_contracts_are_valid() -> None:
    """The public source, construct, and design records satisfy their contracts."""
    source_manifest = ExternalSourceManifestV1.model_validate_json(
        (CONTRACT_ROOT / "quantitative_source_manifest.json").read_text()
    )
    construct_map = ConstructMapV1.model_validate_json(
        (CONTRACT_ROOT / "construct_map.json").read_text()
    )
    validation_design = ExternalValidationDesignV1.model_validate_json(
        (CONTRACT_ROOT / "validation_design.json").read_text()
    )

    source_ids = {source.source_id for source in source_manifest.sources}
    assert source_ids == {"aact_20260701", "rct_bench_125"}
    assert construct_map.constructs
    assert validation_design.profile_id == (
        "trialagentbench-observable-profile-20260724"
    )
    assert validation_design.bootstrap_replicates == 2000
