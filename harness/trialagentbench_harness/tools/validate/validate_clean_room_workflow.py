"""Validate TrialAgentBench clean-room participant/evaluator/audit boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trialagentbench_harness.contracts.release.clean_room_workflow import (
    clean_room_workflow_markdown,
    validate_clean_room_workflow,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, help="Unpacked HF-style TrialAgentBench package root.")
    parser.add_argument("--harness-root", required=True, help="Unpacked GitHub-style harness root.")
    parser.add_argument(
        "--audit-root",
        action="append",
        default=[],
        help="Optional independent witness root. May be supplied multiple times.",
    )
    parser.add_argument("--write-report", help="Optional JSON report path.")
    parser.add_argument("--write-markdown", help="Optional Markdown report path.")
    parser.add_argument("--max-findings", type=int, default=100)
    args = parser.parse_args(argv)
    if args.max_findings < 0:
        parser.error("--max-findings must be non-negative")

    report = validate_clean_room_workflow(
        package_root=Path(args.package_root),
        harness_root=Path(args.harness_root),
        audit_roots=[Path(root) for root in args.audit_root],
    )
    if args.write_report:
        out = Path(args.write_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_markdown:
        out = Path(args.write_markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(clean_room_workflow_markdown(report), encoding="utf-8")
    for finding in report.findings[: args.max_findings]:
        print(f"{finding.path}: {finding.code}: {finding.message}")
    hidden_count = len(report.findings) - min(len(report.findings), args.max_findings)
    if hidden_count:
        print(f"... {hidden_count} additional finding(s) omitted; use --write-report for the full report.")
    if report.status == "pass":
        print("clean-room workflow validation passed")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
