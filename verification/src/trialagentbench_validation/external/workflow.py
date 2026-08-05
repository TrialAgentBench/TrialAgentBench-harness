"""End-to-end external fitting and held-out validation workflow."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from trialagentbench_validation.external.analysis import (
    evaluate_held_out,
    evaluate_synthetic_concordance,
    fit_observable_profile,
    split_studies,
)
from trialagentbench_validation.external.contracts import (
    ConstructMapV1,
    ExternalSourceManifestV1,
    ExternalValidationDesignV1,
    PairedSyntheticConcordanceReportV1,
    SelectedObservableProfileV1,
    StudyPartitionV1,
    StudySummaryV1,
    SyntheticConcordanceDifferenceV1,
    SyntheticConcordanceReportV1,
)
from trialagentbench_validation.external.sources.aact import (
    extract_aact_interventional_trials,
)
from trialagentbench_validation.external.sources.rctbench import extract_rct_bench
from trialagentbench_validation.external.sources.registry import verify_source_manifest
from trialagentbench_validation.external.synthetic import (
    extract_public_synthetic_trials,
)
from trialagentbench_validation.io import (
    read_json,
    sha256_file,
    sha256_model,
    write_model,
)


def run_external_validation(
    *,
    source_manifest_path: Path,
    construct_map_path: Path,
    design_path: Path,
    aact_path: Path,
    rct_bench_path: Path,
    output_dir: Path,
) -> None:
    """Verify sources, fit on calibration studies, and evaluate held-out studies."""

    source_manifest = ExternalSourceManifestV1.model_validate(
        read_json(source_manifest_path)
    )
    construct_map = ConstructMapV1.model_validate(read_json(construct_map_path))
    design = ExternalValidationDesignV1.model_validate(read_json(design_path))
    studies = _verified_external_studies(
        source_manifest=source_manifest,
        design=design,
        aact_path=aact_path,
        rct_bench_path=rct_bench_path,
    )
    partition = split_studies(
        studies,
        seed=design.split_seed,
        held_out_fraction=design.held_out_fraction,
    )
    profile = fit_observable_profile(
        studies,
        partition=partition,
        construct_map=construct_map,
        profile_id=design.profile_id,
        source_manifest_sha256=sha256_model(source_manifest),
    )
    report = evaluate_held_out(
        studies,
        partition=partition,
        construct_map=construct_map,
        profile=profile,
        bootstrap_replicates=design.bootstrap_replicates,
        seed=design.bootstrap_seed,
    )
    write_model(output_dir / "partition.json", partition)
    write_model(output_dir / "selected_observable_profile.json", profile)
    write_model(output_dir / "held_out_validation.json", report)


def run_synthetic_concordance(
    *,
    source_manifest_path: Path,
    construct_map_path: Path,
    design_path: Path,
    partition_path: Path,
    profile_path: Path,
    aact_path: Path,
    rct_bench_path: Path,
    participant_release_path: Path,
    output_path: Path,
) -> SyntheticConcordanceReportV1:
    """Compare public synthetic trial observables with frozen held-out studies."""

    source_manifest = ExternalSourceManifestV1.model_validate(
        read_json(source_manifest_path)
    )
    construct_map = ConstructMapV1.model_validate(read_json(construct_map_path))
    design = ExternalValidationDesignV1.model_validate(read_json(design_path))
    partition = StudyPartitionV1.model_validate(read_json(partition_path))
    profile = SelectedObservableProfileV1.model_validate(read_json(profile_path))
    if profile.source_manifest_sha256 != sha256_model(source_manifest):
        raise ValueError("selected profile does not match the source manifest")
    if profile.construct_map_sha256 != sha256_model(construct_map):
        raise ValueError("selected profile does not match the construct map")
    if profile.partition_sha256 != sha256_model(partition):
        raise ValueError("selected profile does not match the study partition")
    studies = _verified_external_studies(
        source_manifest=source_manifest,
        design=design,
        aact_path=aact_path,
        rct_bench_path=rct_bench_path,
    )
    synthetic_trials, task_count, applied_profile_id, applied_profile_sha256 = (
        extract_public_synthetic_trials(participant_release_path)
    )
    results = evaluate_synthetic_concordance(
        synthetic_trials,
        studies,
        partition=partition,
        construct_map=construct_map,
        bootstrap_replicates=design.bootstrap_replicates,
        seed=design.bootstrap_seed + 10_000,
    )
    report = SyntheticConcordanceReportV1(
        reference_profile_id=profile.profile_id,
        reference_profile_sha256=sha256_model(profile),
        applied_profile_id=applied_profile_id,
        applied_profile_sha256=applied_profile_sha256,
        participant_release_sha256=sha256_file(participant_release_path),
        participant_task_count=task_count,
        independent_synthetic_trial_count=len(synthetic_trials),
        synthetic_trial_identity_sha256=_study_identity_sha256(synthetic_trials),
        results=results,
        limitations=(
            "Comparisons concern declared observable distributions, not latent causal-truth validity.",
            "Context siblings are repeated evidence views of one trial and are counted once.",
            "Prepared C1/C2 baseline views define the clean demographic comparison surface.",
            "No equivalence claim is made without a prospective construct-specific margin.",
        ),
    )
    write_model(output_path, report)
    return report


def compare_synthetic_concordance(
    *,
    prefit_report_path: Path,
    selected_report_path: Path,
    output_path: Path,
) -> PairedSyntheticConcordanceReportV1:
    """Compare matched pre-fit and selected-profile concordance reports."""

    prefit = SyntheticConcordanceReportV1.model_validate(read_json(prefit_report_path))
    selected = SyntheticConcordanceReportV1.model_validate(
        read_json(selected_report_path)
    )
    shared_fields = (
        "reference_profile_id",
        "reference_profile_sha256",
        "participant_task_count",
        "independent_synthetic_trial_count",
        "synthetic_trial_identity_sha256",
    )
    mismatched = [
        field
        for field in shared_fields
        if getattr(prefit, field) != getattr(selected, field)
    ]
    if mismatched:
        raise ValueError(
            f"paired concordance reports differ on matched fields: {mismatched}"
        )
    if prefit.applied_profile_id is not None:
        raise ValueError("pre-fit concordance report must not apply a fitted profile")
    if selected.applied_profile_id != selected.reference_profile_id:
        raise ValueError("selected release did not apply the reference profile")
    if selected.applied_profile_sha256 != selected.reference_profile_sha256:
        raise ValueError(
            "selected release profile checksum differs from the reference profile"
        )
    prefit_by_id = {row.construct_id: row for row in prefit.results}
    selected_by_id = {row.construct_id: row for row in selected.results}
    if set(prefit_by_id) != set(selected_by_id):
        raise ValueError(
            "paired concordance reports have different construct inventories"
        )
    differences: list[SyntheticConcordanceDifferenceV1] = []
    for construct_id in sorted(prefit_by_id):
        before = prefit_by_id[construct_id]
        after = selected_by_id[construct_id]
        if before.role != after.role or before.status != after.status:
            raise ValueError(f"paired construct metadata differs for {construct_id!r}")
        if before.status == "unsupported":
            differences.append(
                SyntheticConcordanceDifferenceV1(
                    construct_id=construct_id,
                    role=before.role,
                    status="unsupported",
                    prefit=before,
                    selected=after,
                )
            )
            continue
        assert before.wasserstein_distance is not None
        assert after.wasserstein_distance is not None
        differences.append(
            SyntheticConcordanceDifferenceV1(
                construct_id=construct_id,
                role=before.role,
                status="supported",
                prefit=before,
                selected=after,
                selected_minus_prefit_distance=(
                    after.wasserstein_distance - before.wasserstein_distance
                ),
            )
        )
    report = PairedSyntheticConcordanceReportV1(
        reference_profile_id=selected.reference_profile_id,
        reference_profile_sha256=selected.reference_profile_sha256,
        synthetic_trial_identity_sha256=selected.synthetic_trial_identity_sha256,
        participant_task_count=selected.participant_task_count,
        independent_synthetic_trial_count=selected.independent_synthetic_trial_count,
        prefit_participant_release_sha256=prefit.participant_release_sha256,
        selected_participant_release_sha256=selected.participant_release_sha256,
        selected_applied_profile_id=selected.applied_profile_id,
        selected_applied_profile_sha256=selected.applied_profile_sha256,
        results=tuple(differences),
        interpretation=(
            "Negative selected-minus-pre-fit distance indicates closer alignment with held-out "
            "external studies for the matched construct; no global realism score is defined."
        ),
    )
    write_model(output_path, report)
    return report


def _study_identity_sha256(studies: tuple[StudySummaryV1, ...]) -> str:
    identities = tuple(sorted(study.study_id for study in studies))
    return sha256(("\n".join(identities) + "\n").encode()).hexdigest()


def _verified_external_studies(
    *,
    source_manifest: ExternalSourceManifestV1,
    design: ExternalValidationDesignV1,
    aact_path: Path,
    rct_bench_path: Path,
) -> tuple[StudySummaryV1, ...]:
    aact_sources = tuple(
        source for source in source_manifest.sources if source.source_type == "aact"
    )
    rct_sources = tuple(
        source
        for source in source_manifest.sources
        if source.source_type == "rct_bench"
    )
    if len(aact_sources) != 1 or len(rct_sources) != 1:
        raise ValueError(
            "external workflow requires exactly one AACT and one RCT Bench source"
        )
    aact_source = aact_sources[0]
    rct_source = rct_sources[0]
    local_paths = {
        aact_source.source_id: aact_path,
        rct_source.source_id: rct_bench_path,
    }
    verify_source_manifest(
        source_manifest,
        local_paths=local_paths,
    )
    rct_studies = extract_rct_bench(rct_bench_path, source_id=rct_source.source_id)
    aact_studies = extract_aact_interventional_trials(
        aact_path,
        source_id=aact_source.source_id,
        inclusion=design.aact_inclusion,
    )
    return (*rct_studies, *aact_studies)


__all__ = [
    "compare_synthetic_concordance",
    "run_external_validation",
    "run_synthetic_concordance",
]
