"""Tests for joining scored normalizer qualification observations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from trialagentbench_test_helpers import (
    minimal_trialeval_diagnostic_dictionary,
    minimal_trialeval_method_dictionary,
)

from trialagentbench_harness.contracts.experiments import (
    NarrativeNormalizationBatchConfigV1,
    NarrativeNormalizationBatchManifestV1,
    NarrativeNormalizationBatchRecordV1,
    NarrativePacketIndexRowV1,
    NarrativePacketManifestV1,
    NarrativeParticipantContextV1,
    NarrativeQualificationPacketSetManifestV1,
    TrialEvalAblationEndpointRowV1,
    TrialEvalNarrativeTranscriptionV1,
    TrialEvalNormalizerSampleUnitV1,
    TrialEvalNormalizerSampleV1,
)
from trialagentbench_harness.contracts.submission import TrialEvalSubmissionV1
from trialagentbench_harness.experiments import build_trialeval_normalizer_observations as builder
from trialagentbench_harness.experiments.narrative_normalization import NarrativeNormalizationResultV1
from trialagentbench_harness.io import sha256_file, write_json_model


def _sample(report_sha256: str) -> TrialEvalNormalizerSampleV1:
    unit = TrialEvalNormalizerSampleUnitV1(
        unit_id="unit-1",
        run_identity_sha256="r" * 64,
        assignment_id="assignment-1",
        task_id="TASK1001",
        base_trial_id="base-1",
        report_sha256=report_sha256,
        regime_cell_id="family-1",
        design_tier="D1",
        assumption_tier="A1",
        context_configuration="C1",
        data_preparation="analysis_ready",
        analysis_specification="locked_sap",
        result_shape="scalar",
        model_id="model-1",
        stratum_id="regime_cell_id=family-1|context_configuration=C1",
        frame_base_trial_count=1,
        sampled_base_trial_count=1,
        base_trial_candidate_report_count=1,
        base_trial_inclusion_probability=1.0,
        within_base_report_inclusion_probability=1.0,
        inclusion_probability=1.0,
        selected_without_normalizer_or_score_outcomes=True,
    )
    return TrialEvalNormalizerSampleV1(
        experiment_design_checksum="d" * 64,
        frame_checksum="f" * 64,
        selection_method="stratified_base_trial_then_within_base_hash_rank_v1",
        selection_seed=1,
        units=(unit,),
    ).with_checksum()


def _abstention(*, source: str, report_sha256: str) -> TrialEvalNarrativeTranscriptionV1:
    return TrialEvalNarrativeTranscriptionV1.model_validate(
        {
            "assignment_id": "assignment-1",
            "report_sha256": report_sha256,
            "source": source,
            "source_identity": "masked-panel" if source == "manual_masked" else "provider:model",
            "transcriber_identities": ["transcriber-a", "transcriber-b"] if source == "manual_masked" else [],
            "transcription_disposition": "independent_exact_agreement" if source == "manual_masked" else None,
            "blinded_to_model_identity": True,
            "blinded_to_evaluator_reference": True,
            "importer_prompt_sha256": None if source == "manual_masked" else "p" * 64,
            "importer_schema_sha256": None if source == "manual_masked" else "s" * 64,
            "importer_response_sha256": None if source == "manual_masked" else "x" * 64,
            "status": "abstain",
            "claims": [],
            "abstention_reason": "No scoreable primary analysis was present.",
        }
    )


def _endpoint(source: str) -> TrialEvalAblationEndpointRowV1:
    return TrialEvalAblationEndpointRowV1(
        assignment_id="assignment-1",
        task_id="TASK1001",
        context_tier="C1",
        data_preparation="analysis_ready",
        analysis_specification="locked_sap",
        model_id="model-1",
        replicate_id="seed-1",
        procedure_assistance="output_contract_only",
        prompt_condition="neutral",
        submission_interface="narrative",
        normalization_source=source,
        normalization_status="abstain",
        normalization_failure_reason="No scoreable primary analysis was present.",
        primary_failure_code="missing_primary_submission",
        usable_primary=False,
        route_match=False,
        obligations_met=False,
        credit_eligible_route_count=1,
        numeric_result_available=False,
        result_match=False,
        primary_analysis_conforms=0.0,
        planning_applicable=False,
    )


def test_observation_builder_joins_exact_sample_and_repeated_normalizations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = "No completed primary analysis."
    report_path = tmp_path / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    report_sha256 = sha256_file(report_path)
    sample = _sample(report_sha256)

    packet_root = tmp_path / "packets"
    packet_dir = packet_root / "masked-normalizer-0001"
    packet_dir.mkdir(parents=True)
    frozen_report = packet_dir / "frozen_report.txt"
    frozen_report.write_text(report, encoding="utf-8")
    context = NarrativeParticipantContextV1(
        task_id="TASK1001",
        task_contract={"task_id": "TASK1001"},
        participant_submission_contract={"task_id": "TASK1001"},
        participant_diagnostic_dictionary=minimal_trialeval_diagnostic_dictionary(),
        participant_method_dictionary=minimal_trialeval_method_dictionary(),
        canonical_submission_schema=TrialEvalSubmissionV1.model_json_schema(),
    ).with_checksum()
    context_path = packet_dir / "participant_context.json"
    write_json_model(context_path, context)
    packet = NarrativePacketManifestV1(
        blinded_identity=packet_dir.name,
        participant_task_id="TASK1001",
        assignment_id="assignment-1",
        report_state="present",
        report_sha256=report_sha256,
        participant_context_sha256=sha256_file(context_path),
    )
    packet_path = packet_dir / "packet.json"
    write_json_model(packet_path, packet)
    packet_manifest = NarrativeQualificationPacketSetManifestV1(
        sample_checksum=str(sample.checksum),
        schedule_sha256="q" * 64,
        participant_release_sha256="p" * 64,
        source_run_identity_sha256s=("r" * 64,),
        source_files_sha256={"source": "a" * 64},
        packets=(
            NarrativePacketIndexRowV1(
                blinded_identity=packet_dir.name,
                qualification_unit_id="unit-1",
                packet_manifest_sha256=sha256_file(packet_path),
                report_sha256=report_sha256,
            ),
        ),
    ).with_checksum()
    packet_manifest_path = packet_root / "manifest.json"
    write_json_model(packet_manifest_path, packet_manifest)

    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    write_json_model(manual_dir / "unit-1.json", _abstention(source="manual_masked", report_sha256=report_sha256))
    batch_root = tmp_path / "batch"
    result_root = batch_root / "results" / packet_dir.name
    result_root.mkdir(parents=True)
    records = []
    for repeat in (1, 2):
        result = NarrativeNormalizationResultV1(
            request_sha256=str(repeat) * 64,
            transcription=_abstention(source="automated_importer", report_sha256=report_sha256),
            raw_provider_response='{"status":"abstain"}',
        ).with_checksum()
        result_path = result_root / f"repeat-{repeat:04d}.json"
        write_json_model(result_path, result)
        records.append(
            NarrativeNormalizationBatchRecordV1(
                blinded_identity=packet_dir.name,
                qualification_unit_id="unit-1",
                assignment_id="assignment-1",
                repeat_index=repeat,
                result_file=str(result_path.relative_to(batch_root)),
                result_sha256=sha256_file(result_path),
                status="abstain",
            )
        )
    config = NarrativeNormalizationBatchConfigV1(
        packet_set_manifest_sha256=sha256_file(packet_manifest_path),
        provider="openai",
        normalizer_model="normalizer",
        temperature=0.0,
        send_temperature=True,
        max_tokens=1024,
        timeout_seconds=30.0,
        repeats=2,
    ).with_checksum()
    write_json_model(batch_root / "batch_config.json", config)
    batch_manifest = NarrativeNormalizationBatchManifestV1(
        config_checksum=str(config.checksum),
        packet_count=1,
        repeat_count=2,
        result_count=2,
        complete_count=0,
        abstain_count=2,
        records=tuple(records),
    ).with_checksum()
    write_json_model(batch_root / "manifest.json", batch_manifest)

    fake_run = SimpleNamespace(
        run_config=SimpleNamespace(run_identity_sha256="r" * 64),
        result_by_assignment=lambda: {"assignment-1": object()},
        assert_unchanged=lambda: None,
    )
    monkeypatch.setattr(builder, "load_completed_trialeval_ablation_run", lambda _: fake_run)
    monkeypatch.setattr(builder, "discover_items", lambda _: (SimpleNamespace(task_id="TASK1001"),))
    monkeypatch.setattr(
        builder.ScoringKeyStoreV1,
        "from_release",
        lambda *args, **kwargs: SimpleNamespace(for_item=lambda _: object()),
    )
    monkeypatch.setattr(
        builder,
        "read_assumption_evidence_domains",
        lambda **kwargs: {"TASK1001": object()},
    )
    monkeypatch.setattr(
        builder,
        "score_trialeval_ablation_submission_v1",
        lambda **kwargs: _endpoint(kwargs["normalization_source"]),
    )
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()

    observation_set = builder.build_trialeval_normalizer_observations_v1(
        sample=sample,
        packet_root=packet_root,
        manual_transcriptions_dir=manual_dir,
        normalization_batch_root=batch_root,
        run_dirs=(tmp_path / "run",),
        evaluator_root=evaluator,
    )

    assert len(observation_set.observations) == 1
    assert len(observation_set.observations[0].automated_repeats) == 2
    assert observation_set.sample_checksum == sample.checksum
    assert observation_set.checksum
