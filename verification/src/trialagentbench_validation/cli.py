"""Command-line entry points for independent release validation."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from trialagentbench_validation.contracts.v1_scope import (
    RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
    TRIALDEV_REPLAY_CALIBRATION_TOLERANCE_V1,
    TRIALEVAL_C5_REFERENCE_COUNT_V1,
)
from trialagentbench_validation.trialeval.sentinels import audit_trialeval_sentinels


def main(argv: Sequence[str] | None = None) -> int:
    """Run independent validation commands."""

    parser = argparse.ArgumentParser(prog="trialagentbench-validate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sentinels = subparsers.add_parser(
        "trialeval-sentinels",
        help="Audit the high-risk TrialEval sentinel strata from release archives.",
    )
    sentinels.add_argument("--evaluator", type=Path, required=True)
    sentinels.add_argument("--participant", type=Path, required=True)
    sentinels.add_argument("--output", type=Path, required=True)
    trialeval_replay = subparsers.add_parser(
        "trialeval-replay",
        help="Independently replay TrialEval references from participant-facing evidence.",
    )
    trialeval_replay.add_argument("--evaluator", type=Path, required=True)
    trialeval_replay.add_argument("--participant", type=Path, required=True)
    trialeval_replay.add_argument("--output-dir", type=Path, required=True)
    trialeval_replay.add_argument("--workers", type=int, default=1)
    c5_integrity = subparsers.add_parser(
        "trialeval-c5-integrity",
        help="Independently repair every declared C5 transport duplication and prove equality to C4.",
    )
    c5_integrity.add_argument("--verification", type=Path, required=True)
    c5_integrity.add_argument("--participant", type=Path, required=True)
    c5_integrity.add_argument("--output", type=Path, required=True)
    c5_integrity.add_argument(
        "--expected-items", type=int, default=TRIALEVAL_C5_REFERENCE_COUNT_V1
    )
    c5_integrity.add_argument("--workers", type=int, default=1)
    route_evidence = subparsers.add_parser(
        "trialeval-route-evidence",
        help="Write compact checksum-bound public replay evidence for release construction.",
    )
    route_evidence.add_argument("--evaluator", type=Path, required=True)
    route_evidence.add_argument("--participant", type=Path, required=True)
    route_evidence.add_argument("--output", type=Path, required=True)
    route_evidence.add_argument("--workers", type=int, default=1)
    analysis_reliability = subparsers.add_parser(
        "trialeval-analysis-reliability",
        help="Recompute routine and prespecified-alternative reliability from repeated-trial records.",
    )
    analysis_reliability.add_argument("--world-records", type=Path, required=True)
    analysis_reliability.add_argument(
        "--operating-characteristics", type=Path, required=True
    )
    analysis_reliability.add_argument("--output", type=Path, required=True)
    analysis_reliability.add_argument(
        "--bootstrap-replicates", type=int, default=10_000
    )
    analysis_reliability.add_argument("--seed", type=int, default=20_260_802)
    identification_reliability = subparsers.add_parser(
        "trialeval-identification-reliability",
        help="Recompute A4 identified-range and sequential-analysis reliability.",
    )
    identification_reliability.add_argument("--world-records", type=Path, required=True)
    identification_reliability.add_argument(
        "--operating-characteristics", type=Path, required=True
    )
    identification_reliability.add_argument("--output", type=Path, required=True)
    identification_reliability.add_argument(
        "--bootstrap-replicates", type=int, default=10_000
    )
    identification_reliability.add_argument("--seed", type=int, default=20_260_803)
    trialdev_replay = subparsers.add_parser(
        "trialdev-replay",
        help="Independently replay TrialDev observational references from public scenario files.",
    )
    trialdev_replay.add_argument("--scenario-root", type=Path, required=True)
    trialdev_replay.add_argument("--output", type=Path, required=True)
    trialdev_replay.add_argument(
        "--absolute-tolerance",
        type=float,
        default=RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
    )
    trialdev_sentinels = subparsers.add_parser(
        "trialdev-sentinels",
        help="Replay and audit the high-risk TrialDev sentinel strata from release archives.",
    )
    trialdev_sentinels.add_argument("--participant-release", type=Path, required=True)
    trialdev_sentinels.add_argument("--evaluator-release", type=Path, required=True)
    trialdev_sentinels.add_argument("--verification-release", type=Path, required=True)
    trialdev_sentinels.add_argument("--output", type=Path, required=True)
    trialdev_sentinels.add_argument(
        "--absolute-tolerance",
        type=float,
        default=RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
    )
    trialdev_reachability = subparsers.add_parser(
        "trialdev-reachability",
        help="Verify every TrialDev programme action against available fixed randomized evidence.",
    )
    trialdev_reachability.add_argument(
        "--participant-release", type=Path, required=True
    )
    trialdev_reachability.add_argument("--evaluator-release", type=Path, required=True)
    trialdev_reachability.add_argument("--output", type=Path, required=True)
    trialdev_programme_census = subparsers.add_parser(
        "trialdev-programme-census",
        help="Independently verify the finite TrialDev state, action, evidence, and transition census.",
    )
    trialdev_programme_census.add_argument("--census", type=Path, required=True)
    trialdev_programme_census.add_argument("--release-root", type=Path, required=True)
    trialdev_programme_census.add_argument("--output", type=Path, required=True)
    trialdev_portfolio_audit = subparsers.add_parser(
        "trialdev-portfolio-release-audit",
        help="Audit every TrialDev portfolio table for integrity, chronology, realism, and usability.",
    )
    trialdev_portfolio_audit.add_argument("--release-root", type=Path, required=True)
    trialdev_portfolio_audit.add_argument("--output", type=Path, required=True)
    trialdev_portfolio_replay = subparsers.add_parser(
        "trialdev-portfolio-observational-replay",
        help="Independently replay observational analyses for released TrialDev portfolio worlds.",
    )
    trialdev_portfolio_replay.add_argument("--release-root", type=Path, required=True)
    trialdev_portfolio_replay.add_argument("--output-dir", type=Path, required=True)
    trialdev_portfolio_replay.add_argument("--workers", type=int, default=1)
    trialdev_portfolio_replay.add_argument("--world", action="append")
    trialdev_grader_controls = subparsers.add_parser(
        "trialdev-portfolio-grader-controls",
        help="Run accepted and orthogonal single-fault grader controls over every portfolio view.",
    )
    trialdev_grader_controls.add_argument("--release-root", type=Path, required=True)
    trialdev_grader_controls.add_argument("--controls-csv", type=Path, required=True)
    trialdev_grader_controls.add_argument("--output", type=Path, required=True)
    trialdev_difficulty = subparsers.add_parser(
        "trialdev-portfolio-difficulty",
        help="Measure released-view action-set diversity and prespecified shortcut performance.",
    )
    trialdev_difficulty.add_argument("--release-root", type=Path, required=True)
    trialdev_difficulty.add_argument("--output", type=Path, required=True)
    trialdev_routes = subparsers.add_parser(
        "trialdev-portfolio-routes",
        help="Traverse every method-conditioned action route supported by a TrialDev portfolio release.",
    )
    trialdev_routes.add_argument("--release-root", type=Path, required=True)
    trialdev_routes.add_argument("--output", type=Path, required=True)
    policy_value_audit = subparsers.add_parser(
        "trialdev-policy-value-audit",
        help="Independently reconstruct TrialDev policy-value qualification results.",
    )
    policy_value_audit.add_argument("--policy-value-root", type=Path, required=True)
    policy_value_audit.add_argument("--output", type=Path, required=True)
    trialdev_worked = subparsers.add_parser(
        "trialdev-worked-programmes",
        help="Independently reconstruct the public TrialDev worked programmes.",
    )
    trialdev_worked.add_argument("--package-root", type=Path, required=True)
    trialdev_worked.add_argument("--output", type=Path, required=True)
    trialdev_figures = subparsers.add_parser(
        "trialdev-scientific-figures",
        help="Build TrialDev result figures from exact verification outputs.",
    )
    trialdev_figures.add_argument("--operating-summary", type=Path, required=True)
    trialdev_figures.add_argument(
        "--observational-replay-root", type=Path, required=True
    )
    trialdev_figures.add_argument("--release-audit", type=Path, required=True)
    trialdev_figures.add_argument("--grader-controls", type=Path, required=True)
    trialdev_figures.add_argument("--decision-boundary", type=Path, required=True)
    trialdev_figures.add_argument("--portfolio-difficulty", type=Path, required=True)
    trialdev_figures.add_argument("--portfolio-routes", type=Path, required=True)
    trialdev_figures.add_argument("--policy-value-csv", type=Path, required=True)
    trialdev_figures.add_argument("--output-dir", type=Path, required=True)
    trialdev_scientific = subparsers.add_parser(
        "trialdev-scientific-package",
        help="Build the self-contained TrialDev scientific verification package.",
    )
    trialdev_scientific.add_argument("--worked-root", type=Path, required=True)
    trialdev_scientific.add_argument("--operating-root", type=Path, required=True)
    trialdev_scientific.add_argument(
        "--decision-boundary-report", type=Path, required=True
    )
    trialdev_scientific.add_argument("--policy-value-root", type=Path, required=True)
    trialdev_scientific.add_argument("--release-audit-report", type=Path, required=True)
    trialdev_scientific.add_argument(
        "--grader-control-report", type=Path, required=True
    )
    trialdev_scientific.add_argument(
        "--portfolio-difficulty-report", type=Path, required=True
    )
    trialdev_scientific.add_argument(
        "--portfolio-route-report", type=Path, required=True
    )
    trialdev_scientific.add_argument(
        "--observational-replay-root", type=Path, required=True
    )
    trialdev_scientific.add_argument("--figures-root", type=Path, required=True)
    trialdev_scientific.add_argument("--diagram-root", type=Path, required=True)
    trialdev_scientific.add_argument("--source-manifest", type=Path, required=True)
    trialdev_scientific.add_argument("--output-dir", type=Path, required=True)
    trialdev_scientific_verify = subparsers.add_parser(
        "trialdev-scientific-package-verify",
        help="Verify every checksummed artifact and scientific input in a TrialDev package.",
    )
    trialdev_scientific_verify.add_argument("--package-root", type=Path, required=True)
    trialdev_scientific_verify.add_argument("--output", type=Path, required=True)
    phase_replay = subparsers.add_parser(
        "trialdev-phase-replay",
        help="Independently verify randomized TrialDev decisions from retained public tables.",
    )
    phase_replay.add_argument("--bundle-root", type=Path, required=True)
    phase_replay.add_argument("--materialized-root", type=Path, required=True)
    phase_replay.add_argument("--cases", type=Path, required=True)
    phase_replay.add_argument("--records", type=Path, required=True)
    phase_replay.add_argument("--output", type=Path, required=True)
    phase_replay.add_argument(
        "--absolute-tolerance",
        type=float,
        default=TRIALDEV_REPLAY_CALIBRATION_TOLERANCE_V1,
    )
    external = subparsers.add_parser(
        "external-validate",
        help="Fit observable profiles and evaluate frozen held-out external studies.",
    )
    external.add_argument("--source-manifest", type=Path, required=True)
    external.add_argument("--construct-map", type=Path, required=True)
    external.add_argument("--design", type=Path, required=True)
    external.add_argument("--aact", type=Path, required=True)
    external.add_argument("--rct-bench", type=Path, required=True)
    external.add_argument("--output-dir", type=Path, required=True)
    concordance = subparsers.add_parser(
        "synthetic-concordance",
        help="Compare public synthetic trials with frozen held-out external studies.",
    )
    concordance.add_argument("--source-manifest", type=Path, required=True)
    concordance.add_argument("--construct-map", type=Path, required=True)
    concordance.add_argument("--design", type=Path, required=True)
    concordance.add_argument("--partition", type=Path, required=True)
    concordance.add_argument("--profile", type=Path, required=True)
    concordance.add_argument("--aact", type=Path, required=True)
    concordance.add_argument("--rct-bench", type=Path, required=True)
    concordance.add_argument("--participant", type=Path, required=True)
    concordance.add_argument("--output", type=Path, required=True)
    paired_concordance = subparsers.add_parser(
        "compare-synthetic-concordance",
        help="Compare matched pre-fit and selected-profile concordance reports.",
    )
    paired_concordance.add_argument("--prefit", type=Path, required=True)
    paired_concordance.add_argument("--selected", type=Path, required=True)
    paired_concordance.add_argument("--output", type=Path, required=True)
    recoverability = subparsers.add_parser(
        "recoverability",
        help="Independently replay all score-bearing routes from public release roles.",
    )
    recoverability.add_argument("--participant-release", type=Path, required=True)
    recoverability.add_argument("--evaluator-release", type=Path, required=True)
    recoverability.add_argument("--verification-release", type=Path, required=True)
    recoverability.add_argument("--output-dir", type=Path, required=True)
    recoverability.add_argument("--workers", type=int, default=1)
    recoverability.add_argument(
        "--reuse-qualified-trialeval-replay",
        action="store_true",
        help=(
            "Reuse the checksum-bound route-admission replay after validating every "
            "packaged route, input bundle, and participant table checksum."
        ),
    )
    recoverability.add_argument(
        "--absolute-tolerance",
        type=float,
        default=RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
    )
    grader_concordance = subparsers.add_parser(
        "grader-concordance",
        help="Independently reconstruct and compare the complete canonical public-grader census.",
    )
    grader_concordance.add_argument("--release-root", type=Path, required=True)
    grader_concordance.add_argument("--canonical-submissions", type=Path, required=True)
    grader_concordance.add_argument("--output-dir", type=Path, required=True)
    grader_concordance.add_argument("--harness-executable", default="trialagentbench")
    grader_behavior = subparsers.add_parser(
        "grader-behavior",
        help="Compare accepted, alternative, rejected, abstaining, malformed, and non-identification cases.",
    )
    grader_behavior.add_argument("--release-id", required=True)
    grader_behavior.add_argument("--release-root", type=Path, required=True)
    grader_behavior.add_argument("--canonical-submissions", type=Path, required=True)
    grader_behavior.add_argument("--output-dir", type=Path, required=True)
    grader_behavior.add_argument("--harness-executable", default="trialagentbench")
    candidate_release = subparsers.add_parser(
        "candidate-release",
        help="Build the complete finite-census analysis bundle for one released candidate.",
    )
    candidate_release.add_argument("--release-root", type=Path, required=True)
    candidate_release.add_argument("--output-dir", type=Path, required=True)
    candidate_release.add_argument("--verifier-lock", type=Path, required=True)
    candidate_release.add_argument(
        "--absolute-tolerance",
        type=float,
        default=RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
    )
    candidate_verify = subparsers.add_parser(
        "candidate-release-verify",
        help="Verify every checksummed artifact in an existing candidate-analysis bundle.",
    )
    candidate_verify.add_argument("--bundle-root", type=Path, required=True)
    candidate_replay = subparsers.add_parser(
        "candidate-replay-compare",
        help="Compare repository-independent installed-wheel replay with one immutable candidate.",
    )
    candidate_replay.add_argument("--release-root", type=Path, required=True)
    candidate_replay.add_argument("--replay-root", type=Path, required=True)
    candidate_replay.add_argument("--validation-wheel", type=Path, required=True)
    candidate_replay.add_argument("--harness-wheel", type=Path, required=True)
    candidate_replay.add_argument("--installed-environment", type=Path, required=True)
    candidate_replay.add_argument(
        "--installation-constraints", type=Path, required=True
    )
    candidate_replay.add_argument("--transcript", type=Path, required=True)
    candidate_replay.add_argument(
        "--import-audit", type=Path, action="append", required=True
    )
    candidate_replay.add_argument("--output", type=Path, required=True)
    candidate_replay.add_argument(
        "--absolute-tolerance",
        type=float,
        default=RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
    )
    worked_example = subparsers.add_parser(
        "worked-example",
        help="Verify and export public worked-example records from release roles.",
    )
    worked_example.add_argument("--participant-release", type=Path, required=True)
    worked_example.add_argument("--evaluator-release", type=Path, required=True)
    worked_example.add_argument("--verification-release", type=Path, required=True)
    worked_example.add_argument("--case-id", action="append")
    worked_example.add_argument("--output-dir", type=Path, required=True)
    worked_example.add_argument("--workers", type=int, default=1)
    worked_example.add_argument(
        "--absolute-tolerance",
        type=float,
        default=RELEASE_RECOVERY_ABSOLUTE_TOLERANCE_V1,
    )
    matched_assumption = subparsers.add_parser(
        "matched-assumption",
        help="Analyse pair-matched Assumption-axis trials from public release roles.",
    )
    matched_assumption.add_argument("--participant-release", type=Path, required=True)
    matched_assumption.add_argument("--verification-release", type=Path, required=True)
    matched_assumption.add_argument("--design", type=Path, required=True)
    matched_assumption.add_argument("--output-dir", type=Path, required=True)
    distributional_bridge = subparsers.add_parser(
        "distributional-bridge-summary",
        help="Recompute linked-subject distribution and analysis replication.",
    )
    distributional_bridge.add_argument("--worlds", type=Path, required=True)
    distributional_bridge.add_argument("--output", type=Path, required=True)
    distributional_bridge.add_argument(
        "--bootstrap-replicates", type=int, default=2_000
    )
    distributional_bridge.add_argument("--seed", type=int, default=451012)
    generator_realism = subparsers.add_parser(
        "generator-realism-summary",
        help="Compare generated trial fingerprints with external trials.",
    )
    generator_realism.add_argument("--external-baseline", type=Path, required=True)
    generator_realism.add_argument("--external-analysis", type=Path, required=True)
    generator_realism.add_argument("--synthetic", type=Path, required=True)
    generator_realism.add_argument("--output", type=Path, required=True)
    generator_realism.add_argument("--bootstrap-replicates", type=int, default=2_000)
    generator_realism.add_argument("--seed", type=int, default=451014)
    longitudinal = subparsers.add_parser(
        "longitudinal-fingerprint",
        help="Summarize a standardized participant-by-time clinical trial panel.",
    )
    longitudinal.add_argument("--input", type=Path, required=True)
    longitudinal.add_argument("--trial-id", required=True)
    longitudinal.add_argument("--source", required=True)
    longitudinal.add_argument("--measurement", required=True)
    longitudinal.add_argument("--measurement-unit", required=True)
    longitudinal.add_argument("--time-unit", required=True)
    longitudinal.add_argument("--output", type=Path, required=True)
    cdisc_reference = subparsers.add_parser(
        "cdisc-reference",
        help="Verify official CDISC pilot transport and analysis coherence.",
    )
    cdisc_reference.add_argument("--package-root", type=Path, required=True)
    cdisc_reference.add_argument("--source-commit", required=True)
    cdisc_reference.add_argument("--output", type=Path, required=True)
    process_replication = subparsers.add_parser(
        "process-replication",
        help="Compare released empirical visit counts with aggregate source fingerprints.",
    )
    process_replication.add_argument("--release-dir", type=Path, required=True)
    process_replication.add_argument("--source-fingerprints", type=Path, required=True)
    process_replication.add_argument("--output", type=Path, required=True)
    artifact_manifest = subparsers.add_parser(
        "artifact-manifest-verify",
        help="Verify exact file membership and checksums in an evidence directory.",
    )
    artifact_manifest.add_argument("--directory", type=Path, required=True)
    generator_core = subparsers.add_parser(
        "generator-core-recovery",
        help="Independently analyse clinical-trial generator replicates.",
    )
    generator_core.add_argument("--release-dir", type=Path, required=True)
    generator_core.add_argument("--output", type=Path, required=True)
    generator_core.add_argument("--minimum-worlds-per-cell", type=int, default=100)
    generator_core.add_argument(
        "--workers", type=int, default=min(16, os.cpu_count() or 1)
    )
    generator_comparison = subparsers.add_parser(
        "generator-core-compare",
        help="Compare request-matched generator recovery reports.",
    )
    generator_comparison.add_argument("--reference", type=Path, required=True)
    generator_comparison.add_argument("--comparison", type=Path, required=True)
    generator_comparison.add_argument("--output", type=Path, required=True)
    native_stress = subparsers.add_parser(
        "native-stress-recovery",
        help="Independently recover native clinical-mechanism stress worlds.",
    )
    native_stress.add_argument("--release-dir", type=Path, required=True)
    native_stress.add_argument("--output", type=Path, required=True)
    native_stress.add_argument(
        "--minimum-null-worlds-per-anchor", type=int, default=100
    )
    native_stress.add_argument(
        "--minimum-nonnull-worlds-per-anchor", type=int, default=50
    )
    native_stress.add_argument(
        "--workers", type=int, default=min(16, os.cpu_count() or 1)
    )
    longitudinal_qualification = subparsers.add_parser(
        "longitudinal-validation",
        help="Independently verify source-fitted longitudinal trial worlds.",
    )
    longitudinal_qualification.add_argument("--release-dir", type=Path, required=True)
    longitudinal_qualification.add_argument("--output", type=Path, required=True)
    longitudinal_qualification.add_argument(
        "--minimum-worlds-per-trial", type=int, default=100
    )
    multivariate_longitudinal = subparsers.add_parser(
        "multivariate-longitudinal-validation",
        help="Independently verify joint longitudinal trial worlds.",
    )
    multivariate_longitudinal.add_argument("--release-dir", type=Path, required=True)
    multivariate_longitudinal.add_argument("--output", type=Path, required=True)
    multivariate_longitudinal.add_argument("--minimum-worlds", type=int, default=100)
    survival_qualification = subparsers.add_parser(
        "survival-validation",
        help="Independently verify source-fitted survival trial worlds.",
    )
    survival_qualification.add_argument("--release-dir", type=Path, required=True)
    survival_qualification.add_argument("--output", type=Path, required=True)
    survival_qualification.add_argument("--minimum-worlds", type=int, default=100)
    ordinal_qualification = subparsers.add_parser(
        "ordinal-validation",
        help="Independently verify source-fitted ordinal trial worlds.",
    )
    ordinal_qualification.add_argument("--release-dir", type=Path, required=True)
    ordinal_qualification.add_argument("--output", type=Path, required=True)
    ordinal_qualification.add_argument("--minimum-worlds", type=int, default=100)
    clustered_ordinal = subparsers.add_parser(
        "clustered-ordinal-validation",
        help="Independently verify clustered ordinal trial worlds.",
    )
    clustered_ordinal.add_argument("--release-dir", type=Path, required=True)
    clustered_ordinal.add_argument("--output", type=Path, required=True)
    clustered_ordinal.add_argument("--minimum-worlds", type=int, default=100)
    longitudinal_observation = subparsers.add_parser(
        "longitudinal-observation",
        help="Independently verify native longitudinal dropout and IPCW recovery.",
    )
    longitudinal_observation.add_argument("--release-dir", type=Path, required=True)
    longitudinal_observation.add_argument("--output", type=Path, required=True)
    longitudinal_observation.add_argument(
        "--minimum-worlds-per-trial-cell",
        type=int,
        default=100,
    )
    hte_qualification = subparsers.add_parser(
        "hte-validation",
        help="Independently verify treatment-effect heterogeneity response worlds.",
    )
    hte_qualification.add_argument("--release-dir", type=Path, required=True)
    hte_qualification.add_argument("--output", type=Path, required=True)
    hte_qualification.add_argument("--minimum-null-worlds", type=int, default=100)
    hte_qualification.add_argument("--minimum-nonnull-worlds", type=int, default=50)
    competing_risk = subparsers.add_parser(
        "competing-risk-validation",
        help="Independently verify native competing-risk response worlds.",
    )
    competing_risk.add_argument("--release-dir", type=Path, required=True)
    competing_risk.add_argument("--output", type=Path, required=True)
    competing_risk.add_argument("--minimum-null-worlds", type=int, default=100)
    competing_risk.add_argument("--minimum-nonnull-worlds", type=int, default=50)
    confounding = subparsers.add_parser(
        "confounding-validation",
        help="Independently verify confounding and limited-overlap response worlds.",
    )
    confounding.add_argument("--release-dir", type=Path, required=True)
    confounding.add_argument("--output", type=Path, required=True)
    confounding.add_argument("--minimum-null-worlds", type=int, default=100)
    confounding.add_argument("--minimum-nonnull-worlds", type=int, default=50)
    rctbench_qualification = subparsers.add_parser(
        "rctbench-validation",
        help="Independently verify source-sized RCT worlds and response curves.",
    )
    rctbench_qualification.add_argument("--release-dir", type=Path, required=True)
    rctbench_qualification.add_argument("--source-root", type=Path, required=True)
    rctbench_qualification.add_argument("--output", type=Path, required=True)
    rctbench_qualification.add_argument(
        "--minimum-worlds-per-trial", type=int, default=100
    )
    args = parser.parse_args(argv)

    if args.command == "trialeval-sentinels":
        sentinel_report = audit_trialeval_sentinels(
            evaluator_zip=args.evaluator,
            participant_zip=args.participant,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            sentinel_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if sentinel_report.status == "pass" else 1
    if args.command == "trialeval-replay":
        from trialagentbench_validation.trialeval.references.drift import (
            write_public_evidence_reference_drift_validation_artifacts_v1,
        )
        from trialagentbench_validation.trialeval.references.numeric import (
            write_public_evidence_numeric_reference_artifacts_v1,
        )
        from trialagentbench_validation.trialeval.references.replay import (
            write_public_evidence_reference_replay_artifacts_v1,
        )

        replay_report = write_public_evidence_reference_replay_artifacts_v1(
            evaluator_zip=args.evaluator,
            public_zip=args.participant,
            out_dir=args.output_dir,
        )
        numeric_report = write_public_evidence_numeric_reference_artifacts_v1(
            evaluator_zip=args.evaluator,
            public_zip=args.participant,
            out_dir=args.output_dir,
            workers=args.workers,
        )
        drift_validation_report = (
            write_public_evidence_reference_drift_validation_artifacts_v1(
                drift_jsonl=args.output_dir
                / "public_evidence_reference_drift_dispositions.jsonl",
                out_dir=args.output_dir,
            )
        )
        statuses = (
            replay_report.status,
            numeric_report.status,
            drift_validation_report.status,
        )
        return 0 if all(status == "pass" for status in statuses) else 1
    if args.command == "trialeval-c5-integrity":
        from trialagentbench_validation.trialeval.integrity import (
            write_c5_integrity_recovery,
        )

        report = write_c5_integrity_recovery(
            participant_zip=args.participant,
            verification_zip=args.verification,
            output=args.output,
            expected_item_count=args.expected_items,
            workers=args.workers,
        )
        return 0 if report.status == "pass" else 1
    if args.command == "trialeval-route-evidence":
        from trialagentbench_validation.contracts.route_replay import (
            write_public_route_replay_evidence,
        )

        write_public_route_replay_evidence(
            evaluator_zip=args.evaluator,
            participant_zip=args.participant,
            output_path=args.output,
            workers=args.workers,
        )
        return 0
    if args.command == "trialeval-analysis-reliability":
        from trialagentbench_validation.trialeval.analysis_reliability import (
            verify_analysis_reliability,
            write_analysis_reliability_csv,
        )

        analysis_results = verify_analysis_reliability(
            world_records_path=args.world_records,
            operating_characteristics_path=args.operating_characteristics,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        write_analysis_reliability_csv(path=args.output, results=analysis_results)
        return 0
    if args.command == "trialeval-identification-reliability":
        from trialagentbench_validation.trialeval.identification_reliability import (
            verify_identification_reliability,
            write_identification_reliability_csv,
        )

        identification_results = verify_identification_reliability(
            world_records_path=args.world_records,
            operating_characteristics_path=args.operating_characteristics,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        write_identification_reliability_csv(
            path=args.output,
            results=identification_results,
        )
        return 0
    if args.command == "trialdev-replay":
        from trialagentbench_validation.trialdev.replay import (
            replay_trialdev_observational_reference,
        )

        observational_report = replay_trialdev_observational_reference(
            args.scenario_root,
            absolute_tolerance=args.absolute_tolerance,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            observational_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if observational_report.status == "pass" else 1
    if args.command == "trialdev-sentinels":
        from trialagentbench_validation.trialdev.sentinel_audit import (
            audit_trialdev_release_sentinels,
        )

        trialdev_sentinel_report = audit_trialdev_release_sentinels(
            participant_release=args.participant_release,
            evaluator_release=args.evaluator_release,
            verification_release=args.verification_release,
            absolute_tolerance=args.absolute_tolerance,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            trialdev_sentinel_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if trialdev_sentinel_report.status == "pass" else 1
    if args.command == "trialdev-reachability":
        from trialagentbench_validation.trialdev.reachability import (
            audit_trialdev_reachability,
        )

        reachability_report = audit_trialdev_reachability(
            participant_release=args.participant_release,
            evaluator_release=args.evaluator_release,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            reachability_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if reachability_report.status == "pass" else 1
    if args.command == "trialdev-programme-census":
        from trialagentbench_validation.trialdev.programme_census import (
            audit_trialdev_programme_census,
        )

        programme_report = audit_trialdev_programme_census(
            census_path=args.census,
            release_root=args.release_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            programme_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if programme_report.status == "pass" else 1
    if args.command == "trialdev-portfolio-release-audit":
        from trialagentbench_validation.trialdev.portfolio_release_audit import (
            audit_trialdev_portfolio_release_v1,
        )

        portfolio_report = audit_trialdev_portfolio_release_v1(args.release_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            portfolio_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if portfolio_report.status == "pass" else 1
    if args.command == "trialdev-portfolio-observational-replay":
        from trialagentbench_validation.trialdev.portfolio_observational_replay import (
            replay_trialdev_portfolio_observational_release_v1,
        )

        portfolio_observational_report = (
            replay_trialdev_portfolio_observational_release_v1(
                release_root=args.release_root,
                output_dir=args.output_dir,
                workers=args.workers,
                world_ids=None if args.world is None else tuple(args.world),
            )
        )
        return 0 if portfolio_observational_report.status == "pass" else 1
    if args.command == "trialdev-portfolio-grader-controls":
        from trialagentbench_validation.trialdev.portfolio_grader_controls import (
            run_trialdev_portfolio_grader_controls_v1,
        )

        control_report = run_trialdev_portfolio_grader_controls_v1(
            release_root=args.release_root,
            controls_csv=args.controls_csv,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            control_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if control_report.status == "pass" else 1
    if args.command == "trialdev-portfolio-difficulty":
        from trialagentbench_validation.trialdev.portfolio_difficulty import (
            audit_trialdev_portfolio_difficulty_v1,
        )

        difficulty_report = audit_trialdev_portfolio_difficulty_v1(
            release_root=args.release_root
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            difficulty_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if difficulty_report.status == "pass" else 1
    if args.command == "trialdev-portfolio-routes":
        from trialagentbench_validation.trialdev.portfolio_routes import (
            audit_trialdev_portfolio_routes_v1,
        )

        route_report = audit_trialdev_portfolio_routes_v1(
            release_root=args.release_root
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            route_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if route_report.status == "pass" else 1
    if args.command == "trialdev-worked-programmes":
        from trialagentbench_validation.trialdev.worked_programmes import (
            audit_trialdev_worked_programmes,
        )

        worked_report = audit_trialdev_worked_programmes(package_root=args.package_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            worked_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if worked_report.status == "pass" else 1
    if args.command == "trialdev-policy-value-audit":
        from trialagentbench_validation.trialdev.policy_value_audit import (
            audit_trialdev_policy_value_v1,
        )

        policy_report = audit_trialdev_policy_value_v1(
            policy_value_root=args.policy_value_root
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            policy_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if policy_report.status == "pass" else 1
    if args.command == "trialdev-scientific-figures":
        from trialagentbench_validation.trialdev.figure_rendering import (
            build_trialdev_scientific_figures_v1,
        )

        build_trialdev_scientific_figures_v1(
            operating_summary_csv=args.operating_summary,
            observational_replay_root=args.observational_replay_root,
            release_audit_json=args.release_audit,
            grader_controls_csv=args.grader_controls,
            decision_boundary_json=args.decision_boundary,
            portfolio_difficulty_json=args.portfolio_difficulty,
            portfolio_routes_json=args.portfolio_routes,
            policy_value_csv=args.policy_value_csv,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "trialdev-scientific-package":
        from trialagentbench_validation.trialdev.scientific_package import (
            build_trialdev_scientific_package,
        )

        build_trialdev_scientific_package(
            worked_root=args.worked_root,
            operating_root=args.operating_root,
            decision_boundary_report=args.decision_boundary_report,
            policy_value_root=args.policy_value_root,
            release_audit_report=args.release_audit_report,
            grader_control_report=args.grader_control_report,
            portfolio_difficulty_report=args.portfolio_difficulty_report,
            portfolio_route_report=args.portfolio_route_report,
            observational_replay_root=args.observational_replay_root,
            figures_root=args.figures_root,
            diagram_root=args.diagram_root,
            source_manifest=args.source_manifest,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "trialdev-scientific-package-verify":
        from trialagentbench_validation.trialdev.scientific_package import (
            verify_trialdev_scientific_package,
        )

        package_root = args.package_root.resolve(strict=True)
        output = args.output.resolve(strict=False)
        if output == package_root or package_root in output.parents:
            raise ValueError(
                "Scientific-package verification output must be outside the package root."
            )
        scientific_report = verify_trialdev_scientific_package(
            package_root=package_root
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            scientific_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if scientific_report.status == "pass" else 1
    if args.command == "trialdev-phase-replay":
        from trialagentbench_validation.trialdev.phase_replay import (
            validate_trialdev_phase_replay,
        )

        phase_report = validate_trialdev_phase_replay(
            bundle_root=args.bundle_root,
            materialized_root=args.materialized_root,
            cases_path=args.cases,
            records_path=args.records,
            absolute_tolerance=args.absolute_tolerance,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            phase_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return 0 if phase_report.status == "pass" else 1
    if args.command == "external-validate":
        from trialagentbench_validation.external.workflow import run_external_validation

        run_external_validation(
            source_manifest_path=args.source_manifest,
            construct_map_path=args.construct_map,
            design_path=args.design,
            aact_path=args.aact,
            rct_bench_path=args.rct_bench,
            output_dir=args.output_dir,
        )
        return 0
    if args.command == "synthetic-concordance":
        from trialagentbench_validation.external.workflow import (
            run_synthetic_concordance,
        )

        run_synthetic_concordance(
            source_manifest_path=args.source_manifest,
            construct_map_path=args.construct_map,
            design_path=args.design,
            partition_path=args.partition,
            profile_path=args.profile,
            aact_path=args.aact,
            rct_bench_path=args.rct_bench,
            participant_release_path=args.participant,
            output_path=args.output,
        )
        return 0
    if args.command == "compare-synthetic-concordance":
        from trialagentbench_validation.external.workflow import (
            compare_synthetic_concordance,
        )

        compare_synthetic_concordance(
            prefit_report_path=args.prefit,
            selected_report_path=args.selected,
            output_path=args.output,
        )
        return 0
    if args.command == "recoverability":
        from trialagentbench_validation.recovery import (
            recover_release,
            write_recoverability_report,
        )

        recovery_report = recover_release(
            participant_release=args.participant_release,
            evaluator_release=args.evaluator_release,
            verification_release=args.verification_release,
            workers=args.workers,
            absolute_tolerance=args.absolute_tolerance,
            reuse_qualified_trialeval_replay=args.reuse_qualified_trialeval_replay,
        )
        write_recoverability_report(args.output_dir, recovery_report)
        return 0 if recovery_report.status == "pass" else 1
    if args.command == "grader-concordance":
        from trialagentbench_validation.grader_concordance import (
            run_grader_concordance,
        )

        grader_report = run_grader_concordance(
            release_root=args.release_root,
            canonical_submissions=args.canonical_submissions,
            output_dir=args.output_dir,
            harness_executable=args.harness_executable,
        )
        return 0 if grader_report.passed else 1
    if args.command == "grader-behavior":
        from trialagentbench_validation.grader_behavior import (
            run_grader_behavior_census,
        )

        behavior_report = run_grader_behavior_census(
            release_id=args.release_id,
            release_root=args.release_root,
            canonical_submissions=args.canonical_submissions,
            output_dir=args.output_dir,
            harness_executable=args.harness_executable,
        )
        return 0 if behavior_report.status == "pass" else 1
    if args.command == "candidate-release":
        from trialagentbench_validation.candidate_release import (
            CandidateAnalysisConfigV1,
            build_candidate_validation_bundle,
        )

        build_candidate_validation_bundle(
            config=CandidateAnalysisConfigV1(
                release_root=args.release_root,
                output_dir=args.output_dir,
                verifier_lock=args.verifier_lock,
                absolute_tolerance=args.absolute_tolerance,
            )
        )
        return 0
    if args.command == "candidate-release-verify":
        from trialagentbench_validation.contracts.candidate_release import (
            CandidateValidationBundleV1,
            verify_candidate_validation_bundle,
        )

        bundle_root = args.bundle_root.resolve()
        bundle = CandidateValidationBundleV1.model_validate_json(
            (bundle_root / "candidate_validation_bundle.json").read_text(
                encoding="utf-8"
            )
        )
        verify_candidate_validation_bundle(bundle_root, bundle)
        print(f"Candidate analysis verified: {bundle_root}")
        return 0
    if args.command == "candidate-replay-compare":
        from trialagentbench_validation.candidate_clean_replay import (
            compare_candidate_clean_replay,
        )

        compare_candidate_clean_replay(
            release_root=args.release_root,
            replay_root=args.replay_root,
            validation_wheel=args.validation_wheel,
            harness_wheel=args.harness_wheel,
            installed_environment=args.installed_environment,
            installation_constraints=args.installation_constraints,
            transcript=args.transcript,
            import_audits=tuple(args.import_audit),
            output=args.output,
            absolute_tolerance=args.absolute_tolerance,
        )
        return 0
    if args.command == "worked-example":
        from trialagentbench_validation.demonstrations import (
            verify_worked_examples,
        )

        demonstration_report = verify_worked_examples(
            participant_release=args.participant_release,
            evaluator_release=args.evaluator_release,
            verification_release=args.verification_release,
            case_ids=None if args.case_id is None else tuple(args.case_id),
            output_dir=args.output_dir,
            workers=args.workers,
            absolute_tolerance=args.absolute_tolerance,
        )
        return 0 if demonstration_report.status == "pass" else 1
    if args.command == "matched-assumption":
        from trialagentbench_validation.characterisation import (
            MatchedAssumptionDesign,
            characterise_matched_assumption_release,
            write_assumption_release,
        )

        design = MatchedAssumptionDesign.model_validate_json(
            args.design.read_text(encoding="utf-8")
        )
        result = characterise_matched_assumption_release(
            participant_archive=args.participant_release,
            verification_archive=args.verification_release,
            design=design,
        )
        write_assumption_release(args.output_dir, result)
        return 0
    if args.command == "distributional-bridge-summary":
        from trialagentbench_validation.external.realism.distributional_bridge import (
            read_trial_replication_worlds,
            summarize_distributional_bridge,
            write_distributional_bridge_summary,
        )

        summary = summarize_distributional_bridge(
            read_trial_replication_worlds(args.worlds),
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        write_distributional_bridge_summary(args.output, summary)
        return 0
    if args.command == "generator-realism-summary":
        from trialagentbench_validation.external.realism.generator_realism import (
            compare_generator_realism,
            read_trial_baseline_fingerprints,
            read_trial_realism_fingerprints,
        )
        from trialagentbench_validation.io import write_model

        realism_summary = compare_generator_realism(
            read_trial_baseline_fingerprints(args.external_baseline),
            read_trial_realism_fingerprints(args.external_analysis),
            read_trial_realism_fingerprints(args.synthetic),
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        write_model(args.output, realism_summary)
        return 0
    if args.command == "longitudinal-fingerprint":
        import pandas as pd

        from trialagentbench_validation.external.realism.longitudinal import (
            fingerprint_longitudinal_trial,
        )
        from trialagentbench_validation.io import write_model

        if args.input.suffix == ".csv":
            frame = pd.read_csv(args.input)
        elif args.input.suffix == ".parquet":
            frame = pd.read_parquet(args.input)
        else:
            raise ValueError("Longitudinal input must be CSV or Parquet")
        fingerprint = fingerprint_longitudinal_trial(
            frame,
            trial_id=args.trial_id,
            source=args.source,
            measurement=args.measurement,
            measurement_unit=args.measurement_unit,
            time_unit=args.time_unit,
        )
        write_model(args.output, fingerprint)
        return 0
    if args.command == "cdisc-reference":
        from trialagentbench_validation.external.sources.cdisc import (
            verify_cdisc_reference,
        )
        from trialagentbench_validation.io import write_model

        cdisc_report: BaseModel = verify_cdisc_reference(
            args.package_root,
            source_commit=args.source_commit,
        )
        write_model(args.output, cdisc_report)
        return 0
    if args.command == "process-replication":
        from trialagentbench_validation.external.realism.process_replication import (
            evaluate_visit_count_replication,
        )
        from trialagentbench_validation.io import write_model

        visit_report = evaluate_visit_count_replication(
            release_dir=args.release_dir,
            source_fingerprints=args.source_fingerprints,
        )
        write_model(args.output, visit_report)
        return 0
    if args.command == "artifact-manifest-verify":
        from trialagentbench_validation.external.release.artifacts import (
            verify_external_artifact_manifest,
        )

        verify_external_artifact_manifest(args.directory)
        return 0
    if args.command == "generator-core-recovery":
        from trialagentbench_validation.external.recovery.production import (
            evaluate_production_core_release,
        )
        from trialagentbench_validation.io import write_model

        production_report = evaluate_production_core_release(
            release_dir=args.release_dir,
            minimum_worlds_per_cell=args.minimum_worlds_per_cell,
            workers=args.workers,
        )
        write_model(args.output, production_report)
        return 0
    if args.command == "generator-core-compare":
        from trialagentbench_validation.external.recovery.production import (
            ProductionCoreRecoveryReportV1,
            compare_production_core_candidates,
        )
        from trialagentbench_validation.io import write_model

        reference = ProductionCoreRecoveryReportV1.model_validate_json(
            args.reference.read_text(encoding="utf-8")
        )
        comparison = ProductionCoreRecoveryReportV1.model_validate_json(
            args.comparison.read_text(encoding="utf-8")
        )
        write_model(
            args.output,
            compare_production_core_candidates(reference, comparison),
        )
        return 0
    if args.command == "native-stress-recovery":
        from trialagentbench_validation.external.recovery.native_stress import (
            evaluate_native_stress_release,
        )
        from trialagentbench_validation.io import write_model

        native_report = evaluate_native_stress_release(
            release_dir=args.release_dir,
            minimum_null_worlds_per_anchor=args.minimum_null_worlds_per_anchor,
            minimum_nonnull_worlds_per_anchor=args.minimum_nonnull_worlds_per_anchor,
            workers=args.workers,
        )
        write_model(args.output, native_report)
        return 0
    if args.command == "longitudinal-validation":
        from trialagentbench_validation.external.recovery.longitudinal import (
            evaluate_longitudinal_qualification,
        )
        from trialagentbench_validation.io import write_model

        longitudinal_report = evaluate_longitudinal_qualification(
            release_dir=args.release_dir,
            minimum_worlds_per_trial=args.minimum_worlds_per_trial,
        )
        write_model(args.output, longitudinal_report)
        return 0
    if args.command == "multivariate-longitudinal-validation":
        from trialagentbench_validation.external.recovery.multivariate_longitudinal import (
            evaluate_multivariate_longitudinal_qualification,
        )
        from trialagentbench_validation.io import write_model

        multivariate_report = evaluate_multivariate_longitudinal_qualification(
            release_dir=args.release_dir,
            minimum_worlds=args.minimum_worlds,
        )
        write_model(args.output, multivariate_report)
        return 0
    if args.command == "survival-validation":
        from trialagentbench_validation.external.recovery.survival import (
            evaluate_survival_qualification,
        )
        from trialagentbench_validation.io import write_model

        survival_report = evaluate_survival_qualification(
            release_dir=args.release_dir,
            minimum_worlds=args.minimum_worlds,
        )
        write_model(args.output, survival_report)
        return 0
    if args.command == "ordinal-validation":
        from trialagentbench_validation.external.recovery.ordinal import (
            evaluate_ordinal_qualification,
        )
        from trialagentbench_validation.io import write_model

        ordinal_report = evaluate_ordinal_qualification(
            release_dir=args.release_dir,
            minimum_worlds=args.minimum_worlds,
        )
        write_model(args.output, ordinal_report)
        return 0
    if args.command == "clustered-ordinal-validation":
        from trialagentbench_validation.external.recovery.clustered_ordinal import (
            evaluate_clustered_ordinal_qualification,
        )
        from trialagentbench_validation.io import write_model

        clustered_report = evaluate_clustered_ordinal_qualification(
            release_dir=args.release_dir,
            minimum_worlds=args.minimum_worlds,
        )
        write_model(args.output, clustered_report)
        return 0
    if args.command == "longitudinal-observation":
        from trialagentbench_validation.external.recovery.longitudinal_observation import (
            evaluate_longitudinal_observation,
        )
        from trialagentbench_validation.io import write_model

        observation_report = evaluate_longitudinal_observation(
            release_dir=args.release_dir,
            minimum_worlds_per_trial_cell=args.minimum_worlds_per_trial_cell,
        )
        write_model(args.output, observation_report)
        return 0
    if args.command == "hte-validation":
        from trialagentbench_validation.external.recovery.hte import (
            evaluate_hte_qualification,
        )
        from trialagentbench_validation.io import write_model

        hte_report = evaluate_hte_qualification(
            release_dir=args.release_dir,
            minimum_null_worlds=args.minimum_null_worlds,
            minimum_nonnull_worlds=args.minimum_nonnull_worlds,
        )
        write_model(args.output, hte_report)
        return 0
    if args.command == "competing-risk-validation":
        from trialagentbench_validation.external.recovery.competing_risk import (
            evaluate_competing_risk_qualification,
        )
        from trialagentbench_validation.io import write_model

        competing_risk_report = evaluate_competing_risk_qualification(
            release_dir=args.release_dir,
            minimum_null_worlds=args.minimum_null_worlds,
            minimum_nonnull_worlds=args.minimum_nonnull_worlds,
        )
        write_model(args.output, competing_risk_report)
        return 0
    if args.command == "confounding-validation":
        from trialagentbench_validation.external.recovery.confounding import (
            evaluate_confounding_qualification,
        )
        from trialagentbench_validation.io import write_model

        confounding_report = evaluate_confounding_qualification(
            release_dir=args.release_dir,
            minimum_null_worlds=args.minimum_null_worlds,
            minimum_nonnull_worlds=args.minimum_nonnull_worlds,
        )
        write_model(args.output, confounding_report)
        return 0
    if args.command == "rctbench-validation":
        from trialagentbench_validation.external.recovery.rctbench import (
            evaluate_rctbench_qualification,
        )
        from trialagentbench_validation.io import write_model

        rct_report = evaluate_rctbench_qualification(
            release_dir=args.release_dir,
            source_root=args.source_root,
            minimum_worlds_per_trial=args.minimum_worlds_per_trial,
        )
        write_model(args.output, rct_report)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


__all__ = ["main"]
