"""Shared data types for the TrialEvalBench runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trialagentbench_harness.contracts.experiments import TrialEvalAnalysisSpecificationV1


@dataclass
class BenchmarkItem:
    """One immutable benchmark task surface."""

    item_id: str  # e.g. "TE-S01-A1_02__C1", or TASK_ID for an opaque participant release
    trial_name: str
    design_tier: str  # "D1", "D2", "D3", "D4"
    design_subtype: str
    assumption_tier: str  # "A1", "A2", "A3", "A4"
    context_tier: str  # "C1" .. "C5"
    visible_dir: Path
    data_dir: Path  # visible_dir / "data" or visible_dir / "data" / "raw"
    task: dict[str, object]  # parsed task.json
    estimand_mode: str = ""
    data_preparation: str = ""
    analysis_specification: TrialEvalAnalysisSpecificationV1 = "locked_sap"

    data_version: str = "trialagentbench_v1"
    submission_contract: dict[str, object] | None = None  # parsed submission_contract.json
    reconstruction_task: dict[str, object] | None = None  # parsed reconstruction_task.json (C3/C4)
    raw_data_dir: Path | None = None  # visible_dir / "data" / "raw" when present
    task_id: str = ""  # anonymised TASK id, e.g. "TASK137364"
    variant_id: str = ""  # e.g. "TE-S01-A1__C5"
    suite_dir: Path | None = None  # participant release root
