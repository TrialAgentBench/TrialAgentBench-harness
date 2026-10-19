"""Tests for role-separated TrialEval release custody."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trialagentbench.release import (
    ParticipantReleaseError,
    load_evaluation_release,
    load_evaluation_task_index,
)


def _write_evaluation_release(root: Path) -> Path:
    evaluation = root / "evaluation"
    grader = evaluation / "grader"
    grader.mkdir(parents=True)
    artifact = grader / "scoring_keys.jsonl"
    artifact.write_text('{"item_id":"te_0123456789abcdef0123"}\n', encoding="utf-8")
    task_index = {
        "schema_id": "ariadne.trialeval.task_index/v2",
        "release_id": f"release_{'0' * 64}",
        "entries": [
            {
                "task_id": "te_0123456789abcdef0123",
                "trial_id": "trial_0123456789abcdef0123",
                "base_world_id": "world_0123456789abcdef0123",
                "design_tier": "D1",
                "design_subtype": "Individually randomized parallel-group trial",
                "design_family": "individual_randomized",
                "assumption_tier": "A1",
                "context_tier": "C1",
                "evaluation_series_id": "TE-S01",
                "regime_cell_id": "TE-S01-A1",
                "effect_state": "favourable",
                "data_preparation": "analysis_ready",
                "analysis_specification": "locked_sap",
                "reconstruction_family": None,
                "data_integrity_reference_row_count": 0,
            }
        ],
    }
    canonical_index = json.dumps(task_index, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    task_index["checksum"] = hashlib.sha256(canonical_index).hexdigest()
    index_path = grader / "task_index.json"
    index_path.write_text(json.dumps(task_index), encoding="utf-8")
    artifacts = []
    for path in sorted(grader.iterdir()):
        payload = path.read_bytes()
        artifacts.append(
            {
                "relative_path": f"evaluation/grader/{path.name}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_length": len(payload),
            }
        )
    manifest = {
        "schema_id": "ariadne.benchmark_role_manifest/v1",
        "release_id": f"release_{'0' * 64}",
        "role": "evaluation",
        "trial_count": 1,
        "task_count": 1,
        "artifacts": artifacts,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    manifest["checksum"] = hashlib.sha256(canonical).hexdigest()
    (evaluation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return evaluation


def test_evaluation_release_rejects_changed_bytes(tmp_path: Path) -> None:
    evaluation = _write_evaluation_release(tmp_path)

    loaded = load_evaluation_release(evaluation)
    assert loaded.manifest.role == "evaluation"
    index = load_evaluation_task_index(loaded)
    assert index.entries[0].context_tier == "C1"

    (evaluation / "grader" / "scoring_keys.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ParticipantReleaseError, match="artifact changed"):
        load_evaluation_release(evaluation)


def test_evaluation_release_rejects_undeclared_bytes(tmp_path: Path) -> None:
    evaluation = _write_evaluation_release(tmp_path)
    (evaluation / "grader" / "extra.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ParticipantReleaseError, match="files differ"):
        load_evaluation_release(evaluation)


def test_evaluation_release_rejects_missing_bytes(tmp_path: Path) -> None:
    """Require every artifact named by the role manifest."""

    evaluation = _write_evaluation_release(tmp_path)
    (evaluation / "grader" / "scoring_keys.jsonl").unlink()

    with pytest.raises(ParticipantReleaseError, match="artifact is missing"):
        load_evaluation_release(evaluation)


def test_evaluation_release_rejects_symbolic_link_artifacts(tmp_path: Path) -> None:
    """Reject artifacts that escape byte custody through symbolic links."""

    evaluation = _write_evaluation_release(tmp_path)
    artifact = evaluation / "grader" / "scoring_keys.jsonl"
    external = tmp_path / "external.jsonl"
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(external)

    with pytest.raises(ParticipantReleaseError, match="artifact is missing"):
        load_evaluation_release(evaluation)
