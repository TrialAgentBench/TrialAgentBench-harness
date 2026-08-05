"""Build a neutral handoff from verified external-evidence packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict

from trialagentbench_validation.contracts.external_verification_handoff import (
    ExternalVerificationResultHandoffV1,
    ExternalVerificationResultV1,
)
from trialagentbench_validation.external.release.artifacts import (
    verify_external_artifact_manifest,
    write_external_artifact_manifest,
)
from trialagentbench_validation.io import sha256_file, write_model


class _ResultDefinition(TypedDict):
    evidence_id: str
    scientific_family: str
    artifact_manifest_path: str
    result_status: Literal[
        "qualified", "qualified_with_estimator_limitation", "unsupported"
    ]
    reproducibility_class: Literal[
        "public_replayable", "credentialed_reacquirable", "derived_only"
    ]
    supported_scope: tuple[str, ...]
    limitations: tuple[str, ...]
    affected_components: tuple[str, ...]
    required_qualification_ids: tuple[str, ...]


def build_external_result_handoff(
    *,
    evidence_root: Path,
    validation_package_root: Path,
    output_dir: Path,
) -> ExternalVerificationResultHandoffV1:
    """Build and verify the reusable scientific-result inventory."""

    rows = []
    for definition in _result_definitions():
        manifest_path = evidence_root / definition["artifact_manifest_path"]
        verify_external_artifact_manifest(manifest_path.parent)
        rows.append(
            ExternalVerificationResultV1(
                **definition,
                artifact_manifest_sha256=sha256_file(manifest_path),
            )
        )
    rows.sort(key=lambda row: row.evidence_id)
    candidate_payload = {row.evidence_id: row.artifact_manifest_sha256 for row in rows}
    candidate_encoded = json.dumps(
        candidate_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    payload = {
        "schema_id": "trialagentbench.external_verification_result_handoff/v1",
        "candidate_id": f"evh_{hashlib.sha256(candidate_encoded).hexdigest()[:20]}",
        "validation_package_lock_sha256": sha256_file(
            validation_package_root / "uv.lock"
        ),
        "results": [row.model_dump(mode="json") for row in rows],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    handoff = ExternalVerificationResultHandoffV1(
        **payload,
        checksum=hashlib.sha256(encoded).hexdigest(),
    )
    expected_outputs = {
        "README.md",
        "artifact_manifest.json",
        "external_verification_result_handoff.json",
    }
    existing = (
        {path.name for path in output_dir.iterdir()} if output_dir.exists() else set()
    )
    if unexpected := sorted(existing - expected_outputs):
        raise ValueError(
            f"External-result handoff directory contains unexpected files: {unexpected}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_model(output_dir / "external_verification_result_handoff.json", handoff)
    (output_dir / "README.md").write_text(
        "# External Verification Result Handoff\n\n"
        "This package identifies reusable, independently verified scientific "
        "result families by the checksum of each artifact manifest. Downstream "
        "builds can select relevant results and rerun the listed qualification "
        "families after changing an affected production component. Status and "
        "scope are scientific rather than publication decisions.\n",
        encoding="utf-8",
    )
    write_external_artifact_manifest(output_dir)
    return handoff


def _result_definitions() -> tuple[_ResultDefinition, ...]:
    return (
        _definition(
            "cdisc.workflow",
            "CDISC transport and analysis workflow",
            "verification/cdisc_reference/artifact_manifest.json",
            ("analysis reconstruction", "corruption detection", "transport parity"),
            ("clinical_workflow_export",),
            ("cdisc_reference",),
            reproducibility_class="derived_only",
            limitations=(
                "Artificial reference data do not establish clinical-distribution realism.",
            ),
        ),
        _definition(
            "clinical_process.production",
            "empirical clinical process and proportional hazards",
            "production_core/artifact_manifest.json",
            (
                "empirical linked-subject resampling",
                "information response",
                "proportional-hazards recovery",
            ),
            ("empirical_clinical_process", "survival_generation"),
            ("production_core",),
        ),
        _definition(
            "competing_risk.recovery",
            "competing-risk recovery",
            "verification/competing_risk_qualification/artifact_manifest.json",
            (
                "cause-specific recovery",
                "competing-cause response",
                "cumulative incidence",
            ),
            ("competing_risk_generation",),
            ("competing_risk",),
        ),
        _definition(
            "confounding.recovery",
            "confounding and overlap recovery",
            "verification/confounding_qualification/artifact_manifest.json",
            ("adjustment response", "overlap diagnostics", "propensity weighting"),
            ("assignment_generation", "causal_analysis"),
            ("confounding",),
        ),
        _definition(
            "frailty.recovery",
            "recurrent-event frailty recovery",
            "verification/frailty_qualification/artifact_manifest.json",
            ("frailty response", "null behavior", "recurrent-event clustering"),
            ("recurrent_event_generation",),
            ("frailty",),
            status="qualified_with_estimator_limitation",
            limitations=(
                "HC3 intervals show modest non-null undercoverage at the evaluated doses.",
            ),
        ),
        _definition(
            "generator.external_compatibility",
            "external generator compatibility",
            "verification/generator_realism/artifact_manifest.json",
            ("arm balance", "baseline dependence", "baseline marginals"),
            ("baseline_generation",),
            ("generator_realism",),
            limitations=(
                "External split references are descriptive compatibility diagnostics, not equivalence margins.",
            ),
        ),
        _definition(
            "hte.recovery",
            "heterogeneous treatment-effect recovery",
            "verification/hte_qualification/artifact_manifest.json",
            ("interaction recovery", "null behavior", "sample-information response"),
            ("outcome_effect_generation",),
            ("heterogeneous_treatment_effect",),
        ),
        _definition(
            "immport.cross_domain",
            "cross-domain participant linkage",
            "verification/immport_cross_domain/artifact_manifest.json",
            ("analysis perturbation", "exact-marginal linkage response"),
            ("empirical_clinical_process",),
            ("cross_domain_linkage",),
            reproducibility_class="credentialed_reacquirable",
            limitations=(
                "The evidence concerns linked process structure, not treatment-effect realism.",
            ),
        ),
        _definition(
            "immport.recurrent",
            "empirical recurrent-event process",
            "verification/immport_recurrent_events/artifact_manifest.json",
            ("event-count dispersion", "recurrent-event source range"),
            ("recurrent_event_generation",),
            ("recurrent_event",),
            reproducibility_class="credentialed_reacquirable",
        ),
        _definition(
            "longitudinal.external",
            "external longitudinal process",
            "verification/zenodo_longitudinal/artifact_manifest.json",
            ("attendance", "linked trajectory dependence", "marginal trajectory"),
            ("longitudinal_generation",),
            ("longitudinal_external",),
        ),
        _definition(
            "native_stress.recovery",
            "non-proportional hazard and recurrent-event stress",
            "native_stress/artifact_manifest.json",
            (
                "estimand-changing controls",
                "non-proportional-hazard recovery",
                "recurrent-rate recovery",
            ),
            ("recurrent_event_generation", "survival_generation"),
            ("native_stress",),
        ),
        _definition(
            "observation.recovery",
            "informative observation recovery",
            "verification/observation_qualification/artifact_manifest.json",
            (
                "dropout-law realization",
                "inverse-probability weighting",
                "observation response",
            ),
            ("longitudinal_observation",),
            ("informative_observation",),
        ),
        _definition(
            "open_outcome.clustered_graft",
            "participant-clustered ordinal graft outcomes",
            "verification/patency_graft_qualification/artifact_manifest.json",
            (
                "cluster-size and ordinal fidelity",
                "clustered GEE dose recovery",
                "within-participant dependence",
            ),
            ("baseline_generation", "ordinal_generation"),
            ("clustered_ordinal_graft",),
        ),
        _definition(
            "open_outcome.primary",
            "native-scale survival and ordinal outcomes",
            "verification/open_outcomes/artifact_manifest.json",
            (
                "ordinal and safety fidelity",
                "survival process fidelity",
                "treatment-effect dose recovery",
            ),
            ("ordinal_generation", "survival_generation"),
            ("open_outcome",),
            limitations=(
                "PATENCY does not declare mutually exclusive first-event precedence for a source-specific cumulative-incidence estimand.",
            ),
        ),
        _definition(
            "rctbench.distributional",
            "linked-subject distributional bridge",
            "verification/rctbench_distributional_bridge/artifact_manifest.json",
            (
                "adjusted-analysis impact",
                "joint-distribution fidelity",
                "marginal fidelity",
            ),
            ("baseline_generation", "empirical_clinical_process"),
            ("distributional_bridge",),
        ),
        _definition(
            "rctbench.production",
            "source-sized endpoint production",
            "verification/production_qualification/artifact_manifest.json",
            (
                "analysis-concordance intervals",
                "endpoint fidelity",
                "treatment and prognostic recovery",
            ),
            ("baseline_generation", "outcome_effect_generation"),
            ("rctbench_production",),
        ),
        _definition(
            "tereco.multivariate",
            "multivariate longitudinal outcome process",
            "verification/tereco_multivariate/artifact_manifest.json",
            (
                "cross-outcome covariance",
                "joint linkage response",
                "simultaneous treatment recovery",
            ),
            ("longitudinal_generation",),
            ("tereco_multivariate",),
        ),
    )


def _definition(
    evidence_id: str,
    scientific_family: str,
    artifact_manifest_path: str,
    supported_scope: tuple[str, ...],
    affected_components: tuple[str, ...],
    required_qualification_ids: tuple[str, ...],
    *,
    status: Literal[
        "qualified", "qualified_with_estimator_limitation", "unsupported"
    ] = "qualified",
    reproducibility_class: Literal[
        "public_replayable", "credentialed_reacquirable", "derived_only"
    ] = ("public_replayable"),
    limitations: tuple[str, ...] = (),
) -> _ResultDefinition:
    return {
        "evidence_id": evidence_id,
        "scientific_family": scientific_family,
        "artifact_manifest_path": artifact_manifest_path,
        "result_status": status,
        "reproducibility_class": reproducibility_class,
        "supported_scope": tuple(sorted(supported_scope)),
        "limitations": tuple(sorted(limitations)),
        "affected_components": tuple(sorted(affected_components)),
        "required_qualification_ids": tuple(sorted(required_qualification_ids)),
    }


def main() -> None:
    """Run the external-result handoff builder."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--validation-package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_external_result_handoff(
        evidence_root=args.evidence_root,
        validation_package_root=args.validation_package_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


__all__ = ["build_external_result_handoff"]
