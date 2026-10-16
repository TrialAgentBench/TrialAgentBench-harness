"""A budget-capped campaign finishes on its paired panel through the real native lifecycle.

Both offline-control routes run the scripted teaching analysis, whose study design fits the
small teaching world; every turn reserves one microdollar through the campaign's real spending
ledger, so a model spending limit censors part of the campaign exactly as a provider route would. The declared input set is the shipped
teaching corpus, pinned as development inputs in ``tmp_path``; the test requires Docker and the
local executor image.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path

import pytest
from trialagentbench_verifier.campaign import TrialEvalCampaignUnavailableV1
from trialagentbench_verifier.campaign_contracts import BudgetCensoringV1
from trialagentbench_verifier.campaign_inputs import (
    DevelopmentArchiveV1,
    DevelopmentCampaignInputsV1,
    DevelopmentExecutorV1,
)
from trialagentbench_verifier.development_campaign import TrialDevCampaignAnalysisV1
from trialagentbench_verifier.execution_contracts import RELEASE_EXECUTION_LIMITS_V1
from trialagentbench_verifier.executor import resolve_executor_environment
from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.public_results import PublicTrialDevCampaignV1

from trialagentbench.tools import campaign
from trialagentbench.tools.demo import ScriptedDemoProvider, demo_assets
from trialagentbench.tools.run import canonical_trialdev, trialeval
from trialagentbench.util.campaign_control import CampaignRequestControl

pytestmark = pytest.mark.executor
# Two analysis-ready teaching programmes, in run order, and the teaching TrialEval task.
FIRST, SECOND = "td-90b226601cd9", "td-eb87f71556a0"
TRIALEVAL_TASK = "te_63e13c2099f324d78672"
# Scripted turns: the first programme takes 6; the r control's limit stops it inside the second.
R_TURN_LIMIT = 8
TEACHING_TRIALEVAL = demo_assets() / "demo-trialeval-evaluator.zip"
TEACHING_TRIALDEV = demo_assets() / "demo-trialdev-evaluator.zip"


class _MeteredControl:
    """Reserve one microdollar per scripted turn in the campaign ledger before delegating."""

    def __init__(self, provider: object, control: CampaignRequestControl) -> None:
        self._provider = provider
        self._control = control

    def __getattr__(self, name: str) -> object:
        return getattr(self._provider, name)

    def generate_turn(self, *args: object, **kwargs: object) -> object:
        model = self._control.model
        assert model is not None
        if self._control.try_reserve(model=model, now=time.time(), tokens=1, micro_usd=1) is None:
            raise AssertionError("Offline control capacity windows are unbounded")
        return self._provider.generate_turn(*args, **kwargs)  # type: ignore[attr-defined]


def _metered(model: str, *, request_control: CampaignRequestControl | None = None, **kwargs: object) -> object:
    assert request_control is not None, "campaign execution binds the spending ledger"
    return _MeteredControl(ScriptedDemoProvider(), request_control)


def _model(name: str, limit: int) -> dict[str, object]:
    return {
        "name": name,
        "model": f"smoke-zero-{name}",
        "provider": "offline_control",
        "maximum_completion_tokens": 1000,
        "spend_limit_micro_usd": limit,
        "reservation_micro_usd": 1,
        "reservation_tokens": 200000,
        "pricing_premise": "Offline control metered at one microdollar per turn to exercise spending limits.",
        "requests_per_minute": 100000,
        "tokens_per_minute": 10000000,
    }


def _development_inputs(
    root: Path,
    *,
    trialeval_evaluator: Path = TEACHING_TRIALEVAL,
    trialdev_evaluator: Path | None = TEACHING_TRIALDEV,
) -> tuple[str, str]:
    """Pin evaluator archives (the teaching ones by default) and the local executor as development inputs."""
    root.mkdir()
    if trialdev_evaluator is not None:
        shutil.copyfile(trialdev_evaluator, root / "trialdev-evaluator.zip")
    evaluator = root / "trialeval-evaluator.zip"
    shutil.copyfile(trialeval_evaluator, evaluator)
    # Project the paired participant archive with the release writer's canonical member encoding.
    with zipfile.ZipFile(evaluator) as source, zipfile.ZipFile(root / "trialeval-participant.zip", "x") as target:
        identity = json.loads(source.read("participant_archive_identity.json"))
        names = [name for name in source.namelist() if name.startswith("participant/")] + ["release_manifest.json"]
        for name in names:
            info = zipfile.ZipInfo(name.removeprefix("participant/"), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o100644 << 16
            target.writestr(info, source.read(name), compresslevel=6)
    assert sha256_file(root / "trialeval-participant.zip") == identity["participant_archive_sha256"]
    executor = resolve_executor_environment(image=None, limits=RELEASE_EXECUTION_LIMITS_V1)
    manifest = DevelopmentCampaignInputsV1(
        artifacts=tuple(
            DevelopmentArchiveV1(filename=path.name, sha256=sha256_file(path), byte_length=path.stat().st_size)
            for path in sorted(root.glob("*.zip"))
        ),
        executor=DevelopmentExecutorV1(image_id=executor.image_id, environment_sha256=executor.semantic_sha256()),
    )
    (root / "development-inputs.json").write_text(manifest.model_dump_json())
    return sha256_file(root / "development-inputs.json"), executor.image_id


def test_budget_capped_campaign_finishes_with_paired_comparisons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    release = tmp_path / "development"
    manifest_sha256, image_id = _development_inputs(release)
    config = tmp_path / "campaign-config.json"
    config.write_text(
        json.dumps(
            {
                "release_directory": str(release),
                "input_mode": "development",
                "expected_release_manifest_sha256": manifest_sha256,
                "executor_image_id": image_id,
                "models": [_model("python", 1000), _model("r", R_TURN_LIMIT)],
                "panel": {
                    "benchmarks": ["trialdev", "trialeval"],
                    "trialdev_tasks": [FIRST, SECOND],
                    "trialeval_tasks": [TRIALEVAL_TASK],
                    "trialdev_turns_per_step": 12,
                    "trialeval_turns": 12,
                    "watchdog_seconds": 900,
                    "condition_id": "budget-censoring-lifecycle",
                },
                "operations": {
                    "model_workers": 1,
                    "task_workers": 1,
                    "executor_workers": 2,
                    "grading_workers": 1,
                    "spend_limit_micro_usd": 2000,
                },
            }
        )
    )
    monkeypatch.setattr(canonical_trialdev, "get_provider", _metered)
    monkeypatch.setattr(trialeval, "get_provider", _metered)
    root = tmp_path / "campaign"
    assert campaign.main(["prepare", "--config", str(config), "--campaign", str(root)]) == 0
    assert campaign.main(["run", "--campaign", str(root)]) == 3

    def ledger() -> list[tuple[str, int]]:
        with sqlite3.connect(root / "requests.sqlite") as connection:
            return connection.execute("SELECT model,COUNT(*) FROM attempts GROUP BY model ORDER BY model").fetchall()

    spent = ledger()
    assert spent == [("python", spent[0][1]), ("r", R_TURN_LIMIT)]
    # Budgets never refill: resuming re-censors without a single new request.
    assert campaign.main(["resume", "--campaign", str(root)]) == 3
    assert ledger() == spent
    capsys.readouterr()

    assert campaign.main(["finish", "--campaign", str(root)]) == 0
    assert "WARNING: trialdev: budget censoring differs materially" in capsys.readouterr().out
    comparisons = root / "comparisons"
    censoring = BudgetCensoringV1.model_validate_json((comparisons / "censoring.json").read_bytes())
    assert censoring.exhausted_models == ("r",)
    assert censoring.censored_by_job("trialdev") == {"python-trialdev-1": frozenset(), "r-trialdev-1": {SECOND}}
    assert censoring.censored_by_job("trialeval") == {
        "python-trialeval-1": frozenset(),
        "r-trialeval-1": {TRIALEVAL_TASK},
    }
    assert censoring.paired_task_ids("trialdev") == {FIRST}

    development = TrialDevCampaignAnalysisV1.model_validate_json((comparisons / "trialdev/analysis.json").read_bytes())
    assert development.budget_censoring == censoring
    assert development.comparisons and {row.worlds for row in development.comparisons} == {1}
    assert (comparisons / "trialdev/comparisons.csv").is_file()
    unavailable = TrialEvalCampaignUnavailableV1.model_validate_json(
        (comparisons / "trialeval/analysis.json").read_bytes()
    )
    assert "completed by every model" in unavailable.reason

    # Development inputs publish an unqualified aggregate instead of a leaderboard; it carries the record.
    results = json.loads((comparisons / "trialdev/development-results.json").read_bytes())
    assert results["status"] == "unqualified" and results["input_manifest_sha256"] == manifest_sha256
    assert PublicTrialDevCampaignV1.model_validate(results["result"]).budget_censoring == censoring
    index = (root / "index.html").read_text()
    assert "Budget censoring" in index and "budget-censored 1" in index
    # Finishing is repeatable: retained stages of censored runs are verified and reused byte for byte.
    retained = (comparisons / "trialdev/development-results.json").read_bytes()
    assert campaign.main(["finish", "--campaign", str(root)]) == 0
    assert (comparisons / "trialdev/development-results.json").read_bytes() == retained


@pytest.mark.skipif(
    "TRIALAGENTBENCH_REFERENCE_INPUTS" not in os.environ,
    reason="Set TRIALAGENTBENCH_REFERENCE_INPUTS to a development TrialEval evaluator archive directory",
)
def test_campaign_with_decision_tasks_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A campaign with TE-S02 fixed-sequence, TE-S03 non-inferiority and TE-S09 superiority tasks finishes.

    The scripted reference agent analyses every decision task of a development release; finishing
    reconciles each seventh score check (score.decision) with its graded component.
    """
    from reference_agent.agent import ReferenceAgent

    from trialagentbench.contracts.release.archive_safety import materialized_release_archive
    from trialagentbench.trialeval.data import discover_participant_items
    from trialagentbench.trialeval.outcome_submission import TrialEvalEvaluatorRecordStoreV1

    archive = Path(os.environ["TRIALAGENTBENCH_REFERENCE_INPUTS"]) / "trialeval-evaluator.zip"
    with materialized_release_archive(archive, expected_sha256=sha256_file(archive)) as materialized:
        items = discover_participant_items(materialized.root / "participant")
        records = TrialEvalEvaluatorRecordStoreV1.from_release(
            materialized.root / "evaluation", expected_task_ids=tuple(items)
        )
        rules = {task: records.for_task(task).decision for task in sorted(items)}
    tasks = [task for task, rule in rules.items() if rule is not None]
    assert {"fixed_sequence", "non_inferiority"} <= {rule.kind for rule in rules.values() if rule is not None}
    release = tmp_path / "development"
    manifest_sha256, image_id = _development_inputs(release, trialeval_evaluator=archive, trialdev_evaluator=None)
    config = tmp_path / "campaign-config.json"
    config.write_text(
        json.dumps(
            {
                "release_directory": str(release),
                "input_mode": "development",
                "expected_release_manifest_sha256": manifest_sha256,
                "executor_image_id": image_id,
                "models": [_model("python", 1000)],
                "panel": {
                    "benchmarks": ["trialeval"],
                    "trialeval_tasks": tasks,
                    "trialeval_turns": 12,
                    "condition_id": "decision-task-finish",
                },
                "operations": {
                    "model_workers": 1,
                    "task_workers": 2,
                    "executor_workers": 4,
                    "grading_workers": 2,
                    "spend_limit_micro_usd": 1000,
                },
            }
        )
    )
    monkeypatch.setattr(trialeval, "get_provider", lambda model, **kwargs: ReferenceAgent())
    root = tmp_path / "campaign"
    assert campaign.main(["prepare", "--config", str(config), "--campaign", str(root)]) == 0
    assert campaign.main(["run", "--campaign", str(root)]) == 0
    assert campaign.main(["finish", "--campaign", str(root)]) == 0
    decisions = [
        check
        for path in root.glob("results/*/verification/**/*.json")
        for check in json.loads(path.read_bytes()).get("checks", [])
        if check.get("check_id") == "score.decision"
    ]
    assert len(decisions) == len(tasks)
