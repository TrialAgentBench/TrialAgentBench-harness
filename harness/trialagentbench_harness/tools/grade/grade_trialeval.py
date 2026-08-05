"""Deterministically grade a canonical TrialEvalBench run."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from trialagentbench_harness.contracts.core.manifest import GradeManifestV1
from trialagentbench_harness.contracts.core.runs import TrialEvalItemResultV1, TrialEvalRunConfigV1
from trialagentbench_harness.contracts.release.artifacts import (
    TRIALEVAL_EVALUATOR_ARCHIVE_NAME,
    TRIALEVAL_PARTICIPANT_ARCHIVE_NAME,
)
from trialagentbench_harness.contracts.scoring.assumption_evidence import (
    read_assumption_evidence_domains,
)
from trialagentbench_harness.contracts.scoring.trialeval_scores import TrialEvalItemScoresV1
from trialagentbench_harness.grading import ScoringKeyStoreV1
from trialagentbench_harness.grading.reporting import write_trialeval_grade_summary
from trialagentbench_harness.io import read_json_model, sha256_path, staged_directory, write_json_model
from trialagentbench_harness.tools.grade.release_pair import materialized_paired_release_root
from trialagentbench_harness.trialeval.data import discover_items
from trialagentbench_harness.trialeval.grade_submission import (
    grade_trialeval_submission_v1,
)
from trialagentbench_harness.trialeval.planning import assess_trialeval_route_planning_v1
from trialagentbench_harness.util import head_sha


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Canonical TrialEvalBench run directory.")
    parser.add_argument(
        "--suite-dir",
        required=True,
        help="Paired TrialEval release root, or the evaluator ZIP beside its participant ZIP.",
    )
    parser.add_argument("--out-dir", required=True, help="New directory for graded artifacts.")
    return parser


def grade_trialeval_run(argv: list[str] | None = None) -> int:
    """Grade one canonical run against one explicit evaluator release."""
    args = _parser().parse_args(argv)
    source = Path(args.run_dir).resolve()
    suite_source = Path(args.suite_dir).resolve()
    output = Path(args.out_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"TrialEval run directory not found: {source}")
    if not suite_source.exists():
        raise FileNotFoundError(f"TrialEval evaluator release not found: {suite_source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite grading output: {output}")

    with materialized_paired_release_root(
        suite_source,
        evaluator_archive_name=TRIALEVAL_EVALUATOR_ARCHIVE_NAME,
        participant_archive_name=TRIALEVAL_PARTICIPANT_ARCHIVE_NAME,
        participant_subdirectory="public",
    ) as suite_dir:
        if not (suite_dir / "public").is_dir() or not (suite_dir / "grader").is_dir():
            raise FileNotFoundError(f"TrialEval release must contain public/ and grader/: {suite_source}")
        return _grade_materialized_run(
            source=source,
            suite_dir=suite_dir,
            suite_source=suite_source,
            output=output,
        )


def _grade_materialized_run(
    *,
    source: Path,
    suite_dir: Path,
    suite_source: Path,
    output: Path,
) -> int:
    """Grade a TrialEval run against one materialized paired release."""

    run_config = read_json_model(TrialEvalRunConfigV1, source / "run_config.json")
    if run_config.data_format != "trialagentbench_v1" or run_config.data_version != "trialagentbench_v1":
        raise ValueError("Run does not use the canonical TrialEval release contract.")

    public_release = suite_dir / "public"
    if sha256_path(public_release) != run_config.participant_release_sha256:
        raise ValueError(
            "TrialEval participant release checksum does not match the evaluator's paired public surface."
        )
    item_by_id = {item.task_id: item for item in discover_items(suite_dir)}
    expected_ids = list(run_config.task_ids)
    unknown = sorted(set(expected_ids) - set(item_by_id))
    if unknown:
        raise ValueError(f"TrialEval run references task IDs absent from the evaluator release: {unknown}")
    item_files = {path.stem: path for path in sorted((source / "items").glob("*.json"))}
    missing = sorted(set(expected_ids) - set(item_files))
    extra = sorted(set(item_files) - set(expected_ids))
    if missing or extra:
        raise ValueError(f"TrialEval run denominator mismatch: missing={missing}, extra={extra}")

    scoring_keys = ScoringKeyStoreV1.from_release(
        suite_dir,
        expected_item_ids=tuple(expected_ids),
    )
    assumption_evidence = read_assumption_evidence_domains(release_root=suite_dir)
    missing_assumption_evidence = sorted(set(expected_ids) - set(assumption_evidence))
    if missing_assumption_evidence:
        raise ValueError(f"assumption evidence is missing requested tasks: {missing_assumption_evidence}")

    with staged_directory(output) as staging:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        scores: list[dict[str, object]] = []
        for result_id in expected_ids:
            record_path = staging / "items" / f"{result_id}.json"
            record = read_json_model(TrialEvalItemResultV1, record_path)
            if record.item_id != result_id:
                raise ValueError(f"TrialEval item identity mismatch in {record_path}: {record.item_id!r}")
            item = item_by_id[result_id]
            scoring_key = scoring_keys.for_item(result_id)
            grade_record = grade_trialeval_submission_v1(
                item=item,
                scoring_key=scoring_key,
                assumption_evidence=assumption_evidence[result_id],
                submission=record.agent_output.result,
            )
            planning = assess_trialeval_route_planning_v1(
                item=item,
                scoring_key=scoring_key,
                matched_route_id=grade_record.matched_route_id,
                submission=record.agent_output.result,
            )
            score_payload = {
                "item_id": result_id,
                "design_tier": item.design_tier,
                "design_subtype": item.design_subtype,
                "assumption_tier": item.assumption_tier,
                "context_tier": item.context_tier,
                "credit_eligible_route_count": len(scoring_key.credit_eligible_routes),
                "model": run_config.model,
                "agent_status": str(record.agent_output.status or ""),
                "turns_used": int(record.agent_output.turns_used or 0),
                **grade_record.model_dump(mode="json"),
                "planning_applicable": planning.applicable,
                "planning_valid": planning.valid,
                "planning_achieved_power": planning.matched_achieved_power,
                "planning_power_shortfall": planning.matched_power_shortfall,
                "planning_underpowered": planning.matched_underpowered,
                "planning_proportional_participant_deviation": (planning.matched_proportional_participant_deviation),
                "planning_log_sample_size_ratio": planning.matched_log_sample_size_ratio,
                "planning_event_shortage": planning.matched_event_shortage,
                "planning_excess_events": planning.matched_excess_events,
                "planning_excess_participants": planning.matched_excess_participants,
                "planning_participant_shortage": planning.matched_participant_shortage,
            }
            score_model = TrialEvalItemScoresV1(
                item_id=result_id,
                task_id=result_id,
                trial_name=item.trial_name,
                design_tier=item.design_tier,
                design_subtype=item.design_subtype,
                assumption_tier=item.assumption_tier,
                context_tier=item.context_tier,
                estimand_mode=item.estimand_mode,
                model=run_config.model,
                output_mode=record.agent_output.condition_provenance.submission_interface,
                turns_used=int(record.agent_output.turns_used or 0),
                agent_status=str(record.agent_output.status or ""),
                credit_eligible_route_count=len(scoring_key.credit_eligible_routes),
                grade=grade_record,
                planning=planning,
            )
            write_json_model(record_path, record.model_copy(update={"scores": score_model}))
            scores.append(score_payload)

        write_trialeval_grade_summary(rows=scores, output_dir=staging)
        from trialagentbench_harness import __version__ as harness_version

        manifest = GradeManifestV1(
            suite="trialeval",
            harness_version=harness_version,
            harness_git_sha=head_sha(Path(__file__).resolve().parent),
            timestamp_utc=run_config.timestamp_utc,
            input_run_dir=str(source),
            output_run_dir=str(output),
            input_run_sha256=sha256_path(source),
            evaluator_release_sha256=sha256_path(suite_source),
            suite_dir=str(suite_source),
            data_format=run_config.data_format,
            score_profile_id="accepted_route_primary_v1",
            score_profile_ids_available=["accepted_route_primary_v1"],
            scorer_surface_sha256=scoring_keys.manifest.scoring_keys_sha256,
            suite_task_count=len(expected_ids),
            graded_task_count=len(scores),
            method_route_count=sum(
                len(scoring_keys.for_item(item_id).credit_eligible_routes) for item_id in expected_ids
            ),
            notes=[
                "Canonical primary submissions were graded against checksum-bound accepted routes without "
                "model or network calls."
            ],
        )
        write_json_model(staging / "GRADE_MANIFEST.json", manifest)
    print(f"graded {len(scores)} TrialEval items in {output}")
    return 0


__all__ = ["grade_trialeval_run"]


if __name__ == "__main__":
    raise SystemExit(grade_trialeval_run())
