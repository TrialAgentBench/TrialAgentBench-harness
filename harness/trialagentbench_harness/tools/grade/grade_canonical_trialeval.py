"""Grade a complete canonical TrialEval submission census."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_harness.grading import CanonicalSubmissionV1, ScoringKeyStoreV1, grade


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_submissions(path: Path) -> tuple[CanonicalSubmissionV1, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical TrialEval submission census is missing: {path}")
    submissions = tuple(
        CanonicalSubmissionV1.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not submissions:
        raise ValueError("Canonical TrialEval submission census cannot be empty.")
    item_ids = tuple(submission.item_id for submission in submissions)
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Canonical TrialEval submission census contains duplicate item IDs.")
    return submissions


def main(argv: Sequence[str] | None = None) -> int:
    """Grade every canonical submission and write stable JSON Lines records."""

    args = _parser().parse_args(argv)
    submissions = _read_submissions(args.submissions)
    item_ids = tuple(submission.item_id for submission in submissions)
    store = ScoringKeyStoreV1.from_release(args.evaluator_root, expected_item_ids=item_ids)
    if set(item_ids) != set(store.manifest.item_ids):
        missing = sorted(set(store.manifest.item_ids) - set(item_ids))
        extra = sorted(set(item_ids) - set(store.manifest.item_ids))
        raise ValueError(f"Canonical TrialEval grader census is incomplete: missing={missing!r}, extra={extra!r}.")
    by_item = {submission.item_id: submission for submission in submissions}
    records = tuple(grade(store.for_item(item_id), by_item[item_id]) for item_id in store.manifest.item_ids)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite canonical grade census: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(record.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    output.write_text(body, encoding="utf-8")
    return 0


__all__ = ["main"]
