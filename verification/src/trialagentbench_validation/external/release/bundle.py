"""Build and verify clinical-trial simulation validation results."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from trialagentbench_validation.contracts.simulation_validation_bundle import (
    SimulationValidationBundleV1,
    ValidationArtifactV1,
    ValidationFigureV1,
    read_validation_results,
    verify_simulation_validation_bundle,
)
from trialagentbench_validation.io import sha256_file, write_model

MediaType = Literal[
    "application/json",
    "application/pdf",
    "image/svg+xml",
    "text/csv",
    "text/markdown",
]


@dataclass(frozen=True)
class _FigureDefinition:
    figure_id: str
    title: str
    scientific_question: str
    independent_unit: str
    estimand: str
    comparator: str
    uncertainty: str
    interpretation: tuple[str, ...]
    files: tuple[tuple[str, MediaType], ...]


_FIGURES = (
    _FigureDefinition(
        figure_id="characterisation.programme",
        title="TrialEval base-trial properties",
        scientific_question="What range of trial properties is present across the release?",
        independent_unit="100 independent trials; the five contexts are matched views",
        estimand="participant count, age-BMI rank correlation, attendance, and declared follow-up",
        comparator="all seven design profiles",
        uncertainty="complete TrialEval base-trial census with profile medians",
        interpretation=(
            "All 100 independent trials and seven design profiles contribute to the display.",
            "Baseline dependence remains stable while attendance and follow-up vary across declared trial settings.",
        ),
        files=(
            ("data/programme_estimates.csv", "text/csv"),
            ("data/programme_profiles.csv", "text/csv"),
            ("figures/trial_programme.pdf", "application/pdf"),
            ("figures/trial_programme.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="design.properties",
        title="Trial-design representation",
        scientific_question=(
            "Are the declared trial designs visible in the released participant, endpoint, and protocol records?"
        ),
        independent_unit="independent trial",
        estimand=(
            "allocation, adherence, covariate dependence, endpoint ascertainment, "
            "clustering, rollout, and interim information"
        ),
        comparator="complete census within each applicable design profile",
        uncertainty="complete finite census plus 499 arm-count-preserving reassignments per individually randomized trial",
        interpretation=(
            "Observed allocation balance is compared with a trial-specific randomization distribution.",
            "Cluster dependence, treatment rollout, calendar trend, and monitoring-boundary decisions "
            "are computed from the released participant and endpoint records.",
        ),
        files=(
            ("data/design_properties.csv", "text/csv"),
            ("figures/trial_designs.pdf", "application/pdf"),
            ("figures/trial_designs.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="design.analysis",
        title="Trial-design consequences",
        scientific_question="What changes when an analysis omits an encoded trial-design feature?",
        independent_unit="randomized participant or cluster; independent repeated-trial path",
        estimand="treatment effect, bias, and interval coverage",
        comparator=(
            "prespecified analysis versus the same-estimand analysis omitting covariate adjustment, "
            "endpoint correction, cluster dependence, calendar period, or sequential monitoring"
        ),
        uncertainty=(
            "method-compatible 95% intervals for release analyses; Wilson intervals "
            "across repeated cluster, stepped-wedge, and monitoring trials"
        ),
        interpretation=(
            "At the benchmark stepped-wedge trend, period-adjusted coverage remains near 95% "
            "while period-omitting coverage falls to 33.0%.",
            "Cluster-aware interval coverage remains between 95.2% and 95.8% "
            "while participant-independent coverage falls as cluster "
            "heterogeneity strengthens.",
            "The repeated confidence interval retains at least 95.0% coverage "
            "across the tested signal range while expected information falls "
            "as the signal strengthens.",
        ),
        files=(
            ("data/design_comparisons.csv", "text/csv"),
            (
                "data/operating_characteristics/clustered_design/cluster_response_summary.csv",
                "text/csv",
            ),
            (
                "data/operating_characteristics/group_sequential/"
                "group_sequential_operating_characteristics.csv",
                "text/csv",
            ),
            (
                "data/operating_characteristics/stepped_wedge/stepped_wedge_response_summary.csv",
                "text/csv",
            ),
            ("figures/design_consequences.pdf", "application/pdf"),
            ("figures/design_consequences.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="assumption.response",
        title="Assumption response",
        scientific_question=(
            "Are analysis-relevant mechanisms visible, and does the corresponding "
            "analysis result respond as each mechanism strengthens?"
        ),
        independent_unit=(
            "independent TrialEval base trial; matched generated-trial "
            "replicate in the response experiment"
        ),
        estimand=(
            "mechanism-specific diagnostic and analysis consequence on the "
            "declared estimand scale"
        ),
        comparator="six A1-A3 series and two A1-A2 series",
        uncertainty="95% intervals across independent trials or matched trial replicates",
        interpretation=(
            "All 14 adjacent-tier comparisons increase the observed mechanism, with paired intervals above zero.",
            "All 704 tier-specific analyses completed and independently reproduced the distributed references.",
        ),
        files=(
            ("figures/assumption_response.csv", "text/csv"),
            ("figures/assumption_response.pdf", "application/pdf"),
            ("figures/assumption_response.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="assumption.limits",
        title="Supported A4 conclusions",
        scientific_question=(
            "What result remains supported when an A4 condition makes the routine "
            "point analysis incompatible with the trial question?"
        ),
        independent_unit="independently generated trial",
        estimand="treated-minus-control event-risk difference at the prespecified horizon",
        comparator=(
            "identified ranges for dependent censoring and incomplete endpoint-validation "
            "support; repeated intervals after group-sequential monitoring"
        ),
        uncertainty=(
            "complete identified range for nonpoint conclusions; repeated 95% confidence "
            "interval for group-sequential point conclusions"
        ),
        interpretation=(
            "Dependent censoring and incomplete validation support require ranges rather than point estimates.",
            "Sequential monitoring requires the prespecified repeated interval at the realized analysis look.",
        ),
        files=(
            ("data/assumption_identification_results.csv", "text/csv"),
            ("figures/assumption_limits.pdf", "application/pdf"),
            ("figures/assumption_limits.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="context.workflow",
        title="Context reconstruction and standards",
        scientific_question=(
            "Does each information context preserve the trial and estimand while changing "
            "the declared analysis input or reconstruction task?"
        ),
        independent_unit="matched base trial, analysis route, dataset cell, or corruption control",
        estimand=(
            "matched identity, reconstruction parity, route recovery, C5 repair, "
            "and bounded standards workflow"
        ),
        comparator="complete C1-C5 panels and official CDISC pilot datasets",
        uncertainty="exact finite census and deterministic numerical comparison",
        interpretation=(
            "All 100 five-context panels preserve one generation seed and one estimand.",
            "All 692 analysis routes replay and all 100 C5 duplications repair exactly.",
        ),
        files=(
            ("figures/context_workflow.csv", "text/csv"),
            ("figures/context_workflow.pdf", "application/pdf"),
            ("figures/context_workflow.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="external.realism",
        title="External trial distributions",
        scientific_question=(
            "Are marginal, dependence, randomization, and adjustment-sensitive "
            "properties within the variation observed between external trials?"
        ),
        independent_unit="external trial",
        estimand="standardized distributional distance relative to external trial-split variation",
        comparator="75 generated trials versus 29 baseline and 10 analysis-compatible external trials",
        uncertainty="2,000-replicate trial bootstrap",
        interpretation=(
            "All eight point discrepancies are below the external 95th-percentile split reference.",
            "Uncertainty remains wider for dependence and analysis-impact constructs than for marginals.",
        ),
        files=(
            ("figures/generator_realism.csv", "text/csv"),
            ("figures/generator_realism.pdf", "application/pdf"),
            ("figures/generator_realism.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="characterisation.worked_trial",
        title="Worked TrialEval example",
        scientific_question=(
            "Do treatment exposure, post-randomization events, follow-up, and the stated analysis form one "
            "coherent trial?"
        ),
        independent_unit="participant for process distributions; trial for the treatment contrast",
        estimand=(
            "exposure received, intercurrent events, analysis eligibility, event-free survival, "
            "and 365-day risk difference"
        ),
        comparator="randomized arms within one generated trial",
        uncertainty="complete participant census and Greenwood/delta-method interval for the treatment contrast",
        interpretation=(
            "The participant and endpoint tables link one-to-one for all 7,691 participants.",
            "The analysis records reproduce a 365-day risk difference of -0.0165 probability units.",
        ),
        files=(
            ("data/worked_trial_participants.csv", "text/csv"),
            ("figures/worked_trial.pdf", "application/pdf"),
            ("figures/worked_trial.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="joint.structure",
        title="Participant-linkage preservation",
        scientific_question=(
            "Does keeping each participant's measurements together preserve correlations "
            "and adjusted treatment analyses?"
        ),
        independent_unit="trial for portfolio summaries and generated-trial replicate within source",
        estimand=(
            "standardized Wasserstein distance in source standard deviations, "
            "absolute Spearman-correlation error, and absolute adjusted-treatment "
            "bias in source standard errors"
        ),
        comparator=(
            "linked-subject resampling versus a control that preserves each variable "
            "but samples the columns independently"
        ),
        uncertainty="trial-level bootstrap intervals and complete trial-specific results",
        interpretation=(
            "Median standardized Wasserstein distance was nearly unchanged: 0.121 "
            "source standard deviations with linked-subject resampling and 0.122 "
            "with independent columns.",
            "Linked-subject resampling reduced absolute Spearman-correlation error "
            "by 30% and adjusted-treatment bias in source standard errors by 58%.",
        ),
        files=(
            ("figures/joint_structure.pdf", "application/pdf"),
            ("figures/joint_structure.svg", "image/svg+xml"),
            ("figures/joint_structure_methods.csv", "text/csv"),
            ("figures/joint_structure_trials.csv", "text/csv"),
        ),
    ),
    _FigureDefinition(
        figure_id="parameter.recovery",
        title="Effect recovery",
        scientific_question="Do independent analyses follow known changes in the generating treatment effect, including no effect?",
        independent_unit="generated-trial replicate with the source-trial sample size",
        estimand="independently estimated Cox log hazard ratio and proportional-odds log odds ratio",
        comparator="four generating effect settings spanning the null and stronger effects",
        uncertainty=(
            "Monte Carlo intervals for mean estimates plus bias, RMSE, "
            "model-interval coverage, power, and estimability"
        ),
        interpretation=(
            "Estimated effects changed in the configured direction in all 1,000 "
            "paired survival and ordinal trial replicates.",
            "The paired response slopes were -0.1150 versus -0.1141 configured "
            "for survival and -0.08341 versus -0.08321 for the ordinal outcome.",
        ),
        files=(
            ("figures/parameter_recovery.csv", "text/csv"),
            (
                "data/operating_characteristics/outcome_replication/patency_cox_dose_recovery.csv",
                "text/csv",
            ),
            (
                "data/operating_characteristics/outcome_replication/"
                "headsoar_proportional_odds_dose_recovery.csv",
                "text/csv",
            ),
            ("figures/parameter_recovery.pdf", "application/pdf"),
            ("figures/parameter_recovery.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="outcome.longitudinal",
        title="Longitudinal outcomes",
        scientific_question="Does simulation reproduce change over time and relationships among repeated clinical measurements?",
        independent_unit="generated-trial replicate",
        estimand="mean six-minute walk distance by randomized arm and visit",
        comparator="source trial and repeated trials fitted to the source trial",
        uncertainty=(
            "simulation intervals for displayed means, with companion analyses of "
            "retention, within-participant correlations, and treatment-by-time effects"
        ),
        interpretation=(
            "All 36 source arm-visit means and all 36 follow-up counts lie within "
            "their repeated-trial 95% predictive intervals.",
            "Across six outcomes, the largest absolute treatment-effect bias is "
            "0.016 source standard deviations; complete-linkage correlation error "
            "is 0.089 correlation units.",
        ),
        files=(
            ("figures/outcome_longitudinal.csv", "text/csv"),
            (
                "data/operating_characteristics/longitudinal/tereco_linkage_response.csv",
                "text/csv",
            ),
            (
                "data/operating_characteristics/longitudinal/tereco_treatment_recovery.csv",
                "text/csv",
            ),
            ("figures/outcome_longitudinal.pdf", "application/pdf"),
            ("figures/outcome_longitudinal.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="outcome.ordinal",
        title="Ordinal outcomes",
        scientific_question="Does simulation reproduce the complete ordered outcome distribution used in the trial analysis?",
        independent_unit="generated-trial replicate",
        estimand="arm-specific modified Rankin Scale category probability",
        comparator="observed trial and simulations fitted to the source trial",
        uncertainty="simulation intervals across 1,000 source-size trials",
        interpretation=(
            "Mean absolute error was 0.00443 across all seven categories and 0.00358 "
            "across cumulative category probabilities.",
            "Proportional-odds bias at the source effect was -0.00293, with 0.947 "
            "coverage of the generating effect.",
        ),
        files=(
            ("figures/outcome_ordinal.csv", "text/csv"),
            (
                "data/operating_characteristics/outcome_replication/headsoar_safety_predictive.csv",
                "text/csv",
            ),
            ("figures/outcome_ordinal.pdf", "application/pdf"),
            ("figures/outcome_ordinal.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="outcome.survival",
        title="Survival",
        scientific_question=(
            "Does simulation reproduce when events occur and how the number at "
            "risk changes, rather than only the final event proportion?"
        ),
        independent_unit="generated-trial replicate",
        estimand="arm-specific Kaplan-Meier survival probability through 1,095 days",
        comparator="observed trial and simulations fitted to the source trial",
        uncertainty="95% simulation envelope across replicates",
        interpretation=(
            "Across 1,000 source-size trials, mean survival-probability error was "
            "0.00134 and restricted mean survival differed by 0.84 days.",
            "Risk sets differed by 4.18 participants per arm-time cell; source-effect "
            "Cox bias was -0.00050 with 0.962 coverage.",
        ),
        files=(
            ("figures/outcome_survival.csv", "text/csv"),
            (
                "data/operating_characteristics/outcome_replication/patency_rmst_predictive.csv",
                "text/csv",
            ),
            ("figures/outcome_survival.pdf", "application/pdf"),
            ("figures/outcome_survival.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="mechanism.response",
        title="Mechanism response",
        scientific_question=(
            "Do independently fitted analyses respond in the expected direction "
            "when analysis-relevant trial mechanisms are strengthened or disrupted?"
        ),
        independent_unit=(
            "source trial or study for portfolio intervals; matched generated-trial "
            "replicate for dropout intervals"
        ),
        estimand=(
            "mechanism-specific response slope, analysis-error reduction, or "
            "recovered frailty variance"
        ),
        comparator=(
            "graded treatment heterogeneity, competing-event intensity, exposure "
            "assignment, informative dropout, recurrent-event heterogeneity, "
            "and cross-domain linkage disruption"
        ),
        uncertainty=(
            "equal-source t intervals or Monte Carlo intervals over matched "
            "generated-trial replicates"
        ),
        interpretation=(
            "Treatment heterogeneity, competing-event probabilities, confounding, "
            "recurrent-event frailty, and cross-domain linkage move in their "
            "prespecified directions.",
            "Attendance weighting reduces treatment-trajectory error in the PENG "
            "settings but not in the skin-barrier setting, despite recovery of the "
            "configured dropout mechanism in both sources.",
        ),
        files=(
            (
                "data/operating_characteristics/competing_risks/competing_risk_response.csv",
                "text/csv",
            ),
            (
                "data/operating_characteristics/confounding/confounding_dose_response.csv",
                "text/csv",
            ),
            (
                "data/operating_characteristics/cross_domain_linkage/linkage_response.csv",
                "text/csv",
            ),
            (
                "data/operating_characteristics/cross_domain_linkage/portfolio_response.csv",
                "text/csv",
            ),
            (
                "data/operating_characteristics/observation_process/paired_route_contrasts.csv",
                "text/csv",
            ),
            (
                "data/operating_characteristics/recurrent_events/frailty_realization.csv",
                "text/csv",
            ),
            (
                "data/operating_characteristics/treatment_heterogeneity/hte_dose_response.csv",
                "text/csv",
            ),
            ("figures/mechanism_response.csv", "text/csv"),
            ("figures/mechanism_response.pdf", "application/pdf"),
            ("figures/mechanism_response.svg", "image/svg+xml"),
        ),
    ),
    _FigureDefinition(
        figure_id="negative.control",
        title="Structural controls",
        scientific_question=(
            "Does breaking analysis-relevant outcome linkage increase source-scale "
            "error beyond trial-to-trial sampling variation?"
        ),
        independent_unit="matched generated trial",
        estimand=(
            "paired change in mean absolute survival-curve or ordinal-category "
            "probability error"
        ),
        comparator=(
            "intact data versus controls that permute event times or treatment-outcome "
            "links while preserving the observed values"
        ),
        uncertainty="95% confidence interval for the paired mean change over 1,000 trials",
        interpretation=(
            "Both targeted permutations increase source-scale probability error.",
            "The paired contrasts isolate event-time and treatment-outcome linkage "
            "while retaining the values and sample size of each generated trial.",
        ),
        files=(
            ("figures/negative_control.csv", "text/csv"),
            (
                "data/operating_characteristics/outcome_replication/open_outcome_summary.json",
                "application/json",
            ),
            ("figures/negative_control.pdf", "application/pdf"),
            ("figures/negative_control.svg", "image/svg+xml"),
        ),
    ),
)


def build_simulation_validation_bundle(
    *,
    validation_root: Path,
    verifier_lock: Path,
) -> SimulationValidationBundleV1:
    """Build and verify the clinical-trial simulation validation bundle."""

    root = validation_root.resolve()
    read_validation_results(root / "RESULTS.csv")
    figures = tuple(
        sorted(
            (
                ValidationFigureV1(
                    figure_id=definition.figure_id,
                    title=definition.title,
                    scientific_question=definition.scientific_question,
                    independent_unit=definition.independent_unit,
                    estimand=definition.estimand,
                    comparator=definition.comparator,
                    uncertainty=definition.uncertainty,
                    interpretation=definition.interpretation,
                    artifacts=tuple(
                        sorted(
                            (
                                _artifact(root, relative_path, media_type)
                                for relative_path, media_type in definition.files
                            ),
                            key=lambda artifact: artifact.relative_path,
                        )
                    ),
                )
                for definition in _FIGURES
            ),
            key=lambda figure: figure.figure_id,
        )
    )
    figure_paths = {
        artifact.relative_path for figure in figures for artifact in figure.artifacts
    }
    supporting_data = tuple(
        _artifact(root, path.relative_to(root).as_posix(), _media_type(path))
        for path in sorted((root / "data").rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in figure_paths
    )
    chapters = tuple(
        _artifact(root, path.relative_to(root).as_posix(), "text/markdown")
        for path in sorted((root / "reports").glob("*.md"))
    )
    payload = {
        "schema_id": "trialagentbench.simulation_validation_bundle/v1",
        "verifier_lock_sha256": sha256_file(verifier_lock),
        "figures": [figure.model_dump(mode="json") for figure in figures],
        "supporting_data": [
            artifact.model_dump(mode="json") for artifact in supporting_data
        ],
        "methods": _artifact(root, "METHODS.md", "text/markdown").model_dump(
            mode="json"
        ),
        "report": _artifact(root, "REPORT.md", "text/markdown").model_dump(mode="json"),
        "chapters": [artifact.model_dump(mode="json") for artifact in chapters],
        "results": _artifact(root, "RESULTS.csv", "text/csv").model_dump(mode="json"),
        "sources": _artifact(root, "SOURCES.md", "text/markdown").model_dump(
            mode="json"
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    bundle = SimulationValidationBundleV1(
        **payload,
        checksum=hashlib.sha256(encoded).hexdigest(),
    )
    write_model(root / "validation_bundle.json", bundle)
    verify_simulation_validation_bundle(root, bundle)
    return bundle


def installed_validation_root() -> Path:
    """Return the validation-results directory embedded in an installed wheel."""

    root = Path(__file__).resolve().parents[2] / "validation_results"
    if not root.is_dir():
        raise FileNotFoundError(
            "The installed package does not contain simulation validation results."
        )
    return root


def verify_installed_validation_bundle() -> SimulationValidationBundleV1:
    """Load and verify the simulation validation results embedded in an installed wheel."""

    root = installed_validation_root()
    bundle_path = root / "validation_bundle.json"
    bundle = SimulationValidationBundleV1.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    verify_simulation_validation_bundle(root, bundle)
    return bundle


def _artifact(
    root: Path, relative_path: str, media_type: MediaType
) -> ValidationArtifactV1:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Required validation artifact is missing: {relative_path!r}"
        )
    return ValidationArtifactV1(
        relative_path=relative_path,
        sha256=sha256_file(path),
        media_type=media_type,
    )


def _media_type(path: Path) -> MediaType:
    suffixes: dict[str, MediaType] = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".pdf": "application/pdf",
        ".svg": "image/svg+xml",
    }
    try:
        return suffixes[path.suffix.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported validation artifact type: {path}") from error


def main() -> None:
    """Build the clinical-trial simulation validation bundle."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--verifier-lock", type=Path, required=True)
    args = parser.parse_args()
    build_simulation_validation_bundle(
        validation_root=args.validation_root or installed_validation_root(),
        verifier_lock=args.verifier_lock,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "build_simulation_validation_bundle",
    "installed_validation_root",
    "verify_installed_validation_bundle",
]
