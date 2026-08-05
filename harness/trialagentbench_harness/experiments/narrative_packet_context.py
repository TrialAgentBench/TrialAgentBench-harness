"""Build reference-blind narrative-normalization context from participant bytes."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic.types import JsonValue

from trialagentbench_harness.contracts.experiments import NarrativeParticipantContextV1
from trialagentbench_harness.contracts.submission import trialeval_submission_schema
from trialagentbench_harness.io import read_json, sha256_dir_digest
from trialagentbench_harness.trialeval.data import (
    discover_participant_items,
    load_participant_diagnostic_dictionary,
    load_participant_method_dictionary,
)


def load_narrative_participant_contexts_v1(
    *,
    participant_root: Path,
    expected_release_sha256: str,
    task_ids: tuple[str, ...],
) -> dict[str, NarrativeParticipantContextV1]:
    """Load exact participant contracts for a frozen set of narrative tasks."""

    root = Path(participant_root)
    if root.is_symlink():
        raise ValueError(f"Participant release path must not be a symbolic link: {root}")
    root = root.resolve(strict=True)
    if sha256_dir_digest(root) != expected_release_sha256:
        raise ValueError("Narrative packet participant release does not match the source run.")
    items = discover_participant_items(root, task_ids=task_ids)
    _, diagnostic_dictionary = load_participant_diagnostic_dictionary(root)
    _, method_dictionary = load_participant_method_dictionary(root)
    canonical_schema = trialeval_submission_schema()
    contexts: dict[str, NarrativeParticipantContextV1] = {}
    for task_id, item in items.items():
        submission_contract_path = item.visible_dir / "submission_contract.json"
        if submission_contract_path.is_symlink() or not submission_contract_path.is_file():
            raise FileNotFoundError(f"Participant submission contract is missing: {submission_contract_path}")
        task_contract = cast(dict[str, JsonValue], item.task)
        submission_contract = read_json(submission_contract_path)
        if not isinstance(submission_contract, dict):
            raise ValueError(f"Participant submission contract must be an object: {submission_contract_path}")
        contexts[task_id] = NarrativeParticipantContextV1(
            task_id=task_id,
            task_contract=task_contract,
            participant_submission_contract=cast(dict[str, JsonValue], submission_contract),
            participant_diagnostic_dictionary=diagnostic_dictionary,
            participant_method_dictionary=method_dictionary,
            canonical_submission_schema=canonical_schema,
        ).with_checksum()
    return contexts


__all__ = ["load_narrative_participant_contexts_v1"]
