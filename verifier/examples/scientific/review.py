"""Reconstruct clinical analyses, calculate reference effects and plot recovery."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
from review_environment import review_environment
from review_interaction import interpretation_materiality, research_episode
from review_report import proof_report, reference_display
from review_scoring import numerical_loss
from review_trialdev import (
    development,
    development_report,
    development_timeline,
    initial_associations,
    latent_selection_control,
)
from review_trialeval import (
    fixed_trial,
    paired_survival_report,
    population_display,
    recovery_loss_report,
)

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})


def main() -> None:
    """Write exact numerical tables and a scientific comparison figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Scientific data directory; defaults to the bundled examples.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Run the bundled trial, development and selection examples without optional full-panel archives.",
    )
    parser.add_argument(
        "--rscript", help="Rscript executable with survival installed for fresh planning and clinical comparisons."
    )
    args = parser.parse_args()
    if args.compact and args.rscript:
        parser.error("--rscript requires the complete review; omit --compact and supply the optional records")
    root = args.source.resolve(strict=True)
    code_root = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    fixed_trial(root / "trialeval", args.output)
    population_display(args.output)
    initial_associations(root / "trialdev", args.output)
    development(root / "trialdev", args.output)
    development_timeline(root / "trialdev", args.output)
    development_report(root / "trialdev", args.output)
    research_episode(root / "trialdev", args.output)
    subprocess.run(
        [
            sys.executable,
            str(code_root / "review_standards.py"),
            "--source",
            str(root / "trialdev"),
            "--output",
            str(args.output),
        ],
        check=True,
    )
    latent_selection_control(root / "latent_selection", args.output)
    if args.compact:
        proof_report(root, args.output, compact=True)
        print(f"Open the research episode: {args.output / 'index.html'}")
        return
    review_environment(root, args.output)
    if (root / "current-trial-records.zip").exists():
        from review_current_trials import run as current_trials

        current_trials(root, args.output / "current-trials")
    subprocess.run(
        [
            sys.executable,
            str(code_root / "review_families.py"),
            "--source",
            str(root / "family-records.zip"),
            "--output",
            str(args.output / "families"),
        ],
        check=True,
    )
    if args.rscript:
        subprocess.run(
            [
                args.rscript,
                str(code_root / "review_planning.R"),
                str(root / "references/planning.csv"),
                str(args.output / "planning-results.csv"),
                str(args.output),
            ],
            check=True,
        )
    reference_display(root, args.output)
    from review_designs import compare_designs
    from review_mechanisms import review as review_mechanisms
    from review_recovery import design_display, development_recovery, recovery
    from review_registry import review as review_registry

    design_display(root, args.output)
    recovery(root, args.output)
    interpretation_materiality(args.output)
    recovery_loss_report(args.output)
    paired_survival_report(args.output)
    compare_designs(root, args.output)
    review_registry(root, args.output, family_archive=root / "current-records.zip")
    review_mechanisms(root, args.output, family_archive=root / "family-records.zip")
    development_recovery(root, args.output, score=numerical_loss)
    proof_report(root, args.output, r_executed=bool(args.rscript))
    print(f"Open the trial review: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
