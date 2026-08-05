from __future__ import annotations

import json
from pathlib import Path

from trialagentbench_harness.analysis.evidence_classifier import (
    EvidenceClassificationError,
    classify_evidence_source,
)


def _release_root(tmp_path: Path) -> Path:
    root = tmp_path / "release" / "TrialDevBench"
    public = root / "scenario_s01" / "public"
    public.mkdir(parents=True)
    (public / "candidate_drug_catalog.json").write_text("{}", encoding="utf-8")
    (public / "trial_request_schema.json").write_text("{}", encoding="utf-8")
    return root


def _program_dir(tmp_path: Path) -> Path:
    program = tmp_path / "runs" / "m" / "r" / "programs" / "s01__benefit_risk"
    (program / "obs_review").mkdir(parents=True)
    (program / "conversation.json").write_text("[]", encoding="utf-8")
    (program / "obs_review" / "obs_review_submission.json").write_text("{}", encoding="utf-8")
    return program


def test_bare_public_filename_resolves_to_release_public_file(tmp_path: Path) -> None:
    result = classify_evidence_source(
        "candidate_drug_catalog.json",
        program_dir=_program_dir(tmp_path),
        trialdev_release_root=_release_root(tmp_path),
        scenario_id="s01",
    )

    assert result.source_role == "release_public_file"
    assert result.evidence_category == "public_catalog"
    assert result.supports_positive_method_claim is True
    assert result.canonical_source_path and result.canonical_source_path.endswith(
        "scenario_s01/public/candidate_drug_catalog.json"
    )


def test_submitted_payload_resolves_as_submitted_payload(tmp_path: Path) -> None:
    program = _program_dir(tmp_path)
    submission = program / "obs_review" / "obs_review_submission.json"
    result = classify_evidence_source(
        submission.as_posix(),
        program_dir=program,
        submission_paths=(submission.as_posix(),),
    )

    assert result.source_role == "submitted_payload"
    assert result.evidence_category == "analysis_or_submission_workfile"
    assert result.supports_positive_method_claim is True


def test_conversation_path_resolves_as_conversation_event(tmp_path: Path) -> None:
    program = _program_dir(tmp_path)
    result = classify_evidence_source((program / "conversation.json").as_posix(), program_dir=program)

    assert result.source_role == "conversation_event"
    assert result.evidence_category == "analysis_or_submission_workfile"
    assert result.supports_positive_method_claim is True


def test_transient_tmp_reference_is_not_positive_method_support() -> None:
    result = classify_evidence_source(Path("/", "tmp", "model_result.csv").as_posix())

    assert result.source_role == "agent_scratch_file"
    assert result.scratch_artifact_kind == "transient_unresolved_workfile"
    assert result.evidence_category == "scratch_or_diagnostic_file"
    assert result.supports_positive_method_claim is False


def test_classified_scratch_file_is_not_positive_method_support(tmp_path: Path) -> None:
    program = _program_dir(tmp_path)
    scratch = program / "agent_workdir"
    scratch.mkdir()
    (scratch / "model.csv").write_text("coef,p_value\n1,0.1\n", encoding="utf-8")

    result = classify_evidence_source("agent_workdir/model.csv", program_dir=program)

    assert result.scratch_artifact_kind == "model_result"
    assert result.supports_positive_method_claim is False


def test_shell_literal_is_pseudo_path_and_not_positive_support() -> None:
    result = classify_evidence_source("cat candidate_drug_catalog.json")

    assert result.source_role == "shell_literal_or_pseudo_path"
    assert result.evidence_category == "shell_literal_or_pseudo_path"
    assert result.supports_positive_method_claim is False


def test_hidden_path_is_release_ineligible() -> None:
    result = classify_evidence_source("grader/domains/route_references.jsonl")

    assert result.source_role == "hidden_or_grader_file"
    assert result.evidence_category == "protected_reference_or_grader_artifact"
    assert result.participant_facing is False
    assert result.hidden_or_grader is True


def test_content_classifies_schema_model_survival_uncertainty_and_listing(tmp_path: Path) -> None:
    program = _program_dir(tmp_path)
    scratch = program / "agent_workdir"
    scratch.mkdir()
    schema = scratch / "x.json"
    schema.write_text(json.dumps({"properties": {"age": {"type": "number"}}, "required": ["age"]}), encoding="utf-8")
    model = scratch / "model.csv"
    model.write_text("coef,p_value,propensity\n1,0.1,0.5\n", encoding="utf-8")
    survival = scratch / "survival.csv"
    survival.write_text("time,event,hazard\n1,0,0.1\n", encoding="utf-8")
    interval = scratch / "interval.csv"
    interval.write_text("estimate,lower,upper\n1,0,2\n", encoding="utf-8")
    listing = scratch / "listing.txt"
    listing.write_text("cwd=/tmp\nfiles: a.csv b.json\n", encoding="utf-8")

    assert classify_evidence_source("agent_workdir/x.json", program_dir=program).scratch_artifact_kind == (
        "schema_or_dictionary"
    )
    assert classify_evidence_source("agent_workdir/model.csv", program_dir=program).scratch_artifact_kind == (
        "model_result"
    )
    assert classify_evidence_source("agent_workdir/survival.csv", program_dir=program).scratch_artifact_kind == (
        "survival_result"
    )
    assert classify_evidence_source("agent_workdir/interval.csv", program_dir=program).scratch_artifact_kind == (
        "uncertainty_result"
    )
    assert classify_evidence_source("agent_workdir/listing.txt", program_dir=program).scratch_artifact_kind == (
        "diagnostic_listing"
    )


def test_unresolved_bare_filename_without_release_context_fails() -> None:
    try:
        classify_evidence_source("candidate_drug_catalog.json")
    except EvidenceClassificationError as exc:
        assert "Cannot classify unresolved evidence source" in str(exc)
    else:
        raise AssertionError("unresolved bare filenames must fail without release context")


def test_trialeval_participant_members_use_canonical_roles() -> None:
    prepared = classify_evidence_source("data/ADSL.parquet", participant_release_relative=True)
    source = classify_evidence_source("data/raw/ADSL.parquet", participant_release_relative=True)
    protocol = classify_evidence_source("protocol_summary.json", participant_release_relative=True)

    assert prepared.evidence_category == "trial_population_table"
    assert source.evidence_category == "trial_population_table"
    assert protocol.evidence_category == "protocol_or_program_contract"


def test_trialeval_unknown_participant_member_fails() -> None:
    try:
        classify_evidence_source("data/derived/unknown.parquet", participant_release_relative=True)
    except ValueError as exc:
        assert "Unknown TrialEval item data member" in str(exc)
    else:
        raise AssertionError("unknown TrialEval participant members must fail")


def test_trialeval_scratch_path_is_not_misclassified_as_participant_evidence() -> None:
    result = classify_evidence_source("scratch/submission.json", participant_release_relative=True)

    assert result.source_role == "agent_scratch_file"
    assert result.evidence_category == "scratch_or_diagnostic_file"
    assert result.supports_positive_method_claim is False
