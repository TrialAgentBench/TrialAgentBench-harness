"""Build and verify the self-contained TrialDev scientific evidence package."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self  # noqa: UP035

from trialagentbench_validation.trialdev.contracts import (
    TrialDevObservationalReplayReportV1,
)
from trialagentbench_validation.trialdev.policy_value_audit import (
    audit_trialdev_policy_value_v1,
)
from trialagentbench_validation.trialdev.portfolio_difficulty import (
    TrialDevPortfolioDifficultyReportV1,
)
from trialagentbench_validation.trialdev.portfolio_grader_controls import (
    TrialDevPortfolioGraderControlReportV1,
)
from trialagentbench_validation.trialdev.portfolio_release_audit import (
    TrialDevPortfolioReleaseAuditV1,
)
from trialagentbench_validation.trialdev.portfolio_routes import (
    TrialDevPortfolioRouteAuditV1,
)
from trialagentbench_validation.trialdev.worked_programmes import (
    audit_trialdev_worked_programmes,
)

_OPERATING_FILES = (
    "experiment_manifest.json",
    "summary_results.csv",
    "world_results.csv",
    "performance.json",
)
_DISPLAY_IDS = (
    "01_state_action",
    "02_identification_uncertainty",
    "03_policy_response",
    "04_mechanism_response",
    "05_clinical_realism",
    "06_operating_characteristics",
    "07_grader_controls",
    "08_decision_difficulty",
    "09_policy_value",
    "10_portfolio_routes",
)
_SOURCE_IDS = (
    "TAB-SRC-001",
    "TAB-SRC-005",
    "TAB-SRC-018",
    "TAB-SRC-019",
    "TAB-SRC-024",
    "TAB-SRC-028",
    "TAB-SRC-029",
    "TAB-SRC-032",
    "TAB-SRC-043",
)
_PUBLIC_SOURCE_FIELDS = (
    "source_id",
    "title",
    "source_type",
    "evidence_role",
    "canonical_id",
    "canonical_url",
    "full_text_url",
    "verification_status",
    "notes",
)


class ScientificPackageArtifactV1(BaseModel):
    """One checksummed file in the reproducible scientific package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class TrialDevScientificPackageManifestV1(BaseModel):
    """Checksum boundary for the integrated TrialDev evidence package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench.validation.trialdev_scientific_package/v1"] = (
        "trialagentbench.validation.trialdev_scientific_package/v1"
    )
    source_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_count_per_arm: int = Field(ge=1)
    matched_experiment_count: int = Field(ge=1)
    released_world_count: int = Field(ge=1)
    participant_view_count: int = Field(ge=1)
    randomized_episode_count: int = Field(ge=1)
    grader_control_count: int = Field(ge=1)
    observational_replay_count: int = Field(ge=1)
    display_count: int = Field(ge=1)
    artifacts: tuple[ScientificPackageArtifactV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        """Require a canonical unique artifact inventory."""

        paths = tuple(item.relative_path for item in self.artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("Scientific-package artifacts must be sorted and unique.")
        if "manifest.json" in paths:
            raise ValueError("A checksum manifest cannot include itself.")
        return self


class TrialDevScientificPackageVerificationV1(BaseModel):
    """Independent integrity result for a built scientific package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal[
        "trialagentbench.validation.trialdev_scientific_package_verification/v1"
    ] = "trialagentbench.validation.trialdev_scientific_package_verification/v1"
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_count: int = Field(ge=0)
    findings: tuple[str, ...]
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        """Bind pass status to an empty finding set."""

        if self.findings != tuple(sorted(set(self.findings))):
            raise ValueError("Scientific-package findings must be sorted and unique.")
        if (self.status == "pass") != (not self.findings):
            raise ValueError("Scientific-package status disagrees with its findings.")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=delimiter))
    if not rows:
        raise ValueError(f"Required scientific input contains no rows: {path}")
    return rows


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
    *,
    delimiter: str = ",",
) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty scientific table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="raise", delimiter=delimiter
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_operating_inputs(
    operating_root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str, int]:
    for name in _OPERATING_FILES:
        if not (operating_root / name).is_file():
            raise FileNotFoundError(
                f"Operating-characteristic input is missing {name}."
            )
    manifest = _read_json_object(operating_root / "experiment_manifest.json")
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError(
            "Operating-characteristic manifest is not a valid experiment inventory."
        )
    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, str) or len(source_identity) != 64:
        raise ValueError(
            "Operating-characteristic manifest lacks a SHA-256 source identity."
        )
    summary = _read_csv(operating_root / "summary_results.csv")
    worlds = _read_csv(operating_root / "world_results.csv")
    experiment_ids = {row["experiment_id"] for row in summary}
    declared_ids = {
        str(item["experiment_id"]) for item in experiments if isinstance(item, dict)
    }
    if experiment_ids != declared_ids:
        raise ValueError(
            "Operating summary does not cover the prespecified experiment inventory."
        )
    world_count = int(summary[0]["world_count"])
    if world_count <= 0 or any(
        int(row["world_count"]) != world_count for row in summary
    ):
        raise ValueError(
            "Operating experiments require one positive common world count."
        )
    grouped: Counter[tuple[str, str]] = Counter(
        (row["experiment_id"], row["arm_id"]) for row in worlds
    )
    expected = {
        (experiment_id, arm)
        for experiment_id in experiment_ids
        for arm in ("reference", "intervention")
    }
    if set(grouped) != expected or any(grouped[key] != world_count for key in expected):
        raise ValueError(
            "Operating results do not form complete reference/intervention arms."
        )
    matched_ids: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in worlds:
        key = (row["experiment_id"], row["arm_id"])
        matched_ids.setdefault(key, set()).add((row["world_index"], row["world_seed"]))
    for experiment_id in experiment_ids:
        if (
            matched_ids[(experiment_id, "reference")]
            != matched_ids[(experiment_id, "intervention")]
        ):
            raise ValueError(
                f"Operating experiment {experiment_id!r} is not pair matched."
            )
    if any(
        row["missing_world_count"] != "0" or row["failed_world_count"] != "0"
        for row in summary
    ):
        raise ValueError("Operating evidence contains missing or failed worlds.")
    return summary, worlds, source_identity, world_count


def _source_rows(source_manifest: Path) -> list[dict[str, str]]:
    rows = _read_csv(source_manifest, delimiter="\t")
    selected = [row for row in rows if row.get("source_id") in _SOURCE_IDS]
    if {row["source_id"] for row in selected} != set(_SOURCE_IDS):
        raise ValueError("Scientific source manifest lacks a required TrialDev source.")
    if any(row.get("verification_status") != "verified" for row in selected):
        raise ValueError("Scientific package cannot cite an unverified source record.")
    if any(not set(_PUBLIC_SOURCE_FIELDS).issubset(row) for row in selected):
        raise ValueError("Scientific source manifest lacks a required public field.")
    return sorted(
        ({field: row[field] for field in _PUBLIC_SOURCE_FIELDS} for row in selected),
        key=lambda row: row["source_id"],
    )


def _load_observational_replays(
    replay_root: Path,
) -> tuple[TrialDevObservationalReplayReportV1, ...]:
    paths = tuple(sorted(Path(replay_root).glob("observational_replay_world_*.json")))
    reports = tuple(
        TrialDevObservationalReplayReportV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for path in paths
    )
    if not reports or any(report.status != "pass" for report in reports):
        raise ValueError("Observational replay package requires only passing reports.")
    scenarios = tuple(report.scenario_id for report in reports)
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("Observational replay scenarios must be unique.")
    return reports


def _validate_decision_boundary(
    path: Path, *, source_identity: str
) -> dict[str, object]:
    report = _read_json_object(path)
    cells = report.get("cells")
    if (
        report.get("status") != "pass"
        or report.get("source_identity") != source_identity
        or not isinstance(cells, list)
        or not cells
    ):
        raise ValueError(
            "Decision-boundary evidence must pass and share the release source identity."
        )
    return report


def _validate_policy_value(path: Path, *, source_identity: str) -> dict[str, object]:
    report = _read_json_object(path)
    cells = report.get("cells")
    if (
        report.get("status") != "pass"
        or report.get("source_identity") != source_identity
        or report.get("role") != "qualification_only_not_primary_grade"
        or not isinstance(cells, list)
        or not cells
    ):
        raise ValueError(
            "Policy-value evidence must pass, retain its role, and share the release source identity."
        )
    return report


def _worked_tables(
    worked_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    payload = _read_json_object(worked_root / "worked_programmes.json")
    graph = _read_json_object(worked_root / "state_action_graph.json")
    programmes = payload.get("programmes")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if (
        not isinstance(programmes, list)
        or not isinstance(nodes, list)
        or not isinstance(edges, list)
    ):
        raise ValueError(
            "Worked programme input lacks programmes or a state-action graph."
        )
    checkpoints: list[dict[str, object]] = []
    capability_counts: Counter[str] = Counter()
    capability_passes: Counter[str] = Counter()
    for programme in programmes:
        if not isinstance(programme, dict) or not isinstance(
            programme.get("checkpoints"), list
        ):
            raise ValueError("Worked programme record is malformed.")
        assessment = programme.get("assessment")
        if not isinstance(assessment, dict) or not isinstance(
            assessment.get("checkpoints"), list
        ):
            raise ValueError("Worked programme assessment is malformed.")
        assessment_by_checkpoint = {
            str(row["checkpoint_id"]): row
            for row in assessment["checkpoints"]
            if isinstance(row, dict)
        }
        for index, step in enumerate(programme["checkpoints"]):
            if not isinstance(step, dict):
                raise ValueError("Worked checkpoint is malformed.")
            before, after = step.get("state_before"), step.get("state_after")
            supported, selected = step.get("supported_action_set"), step.get(
                "selected_action"
            )
            evidence = step.get("decision_evidence")
            if not all(
                isinstance(item, dict)
                for item in (before, after, supported, selected, evidence)
            ):
                raise ValueError("Worked checkpoint records are malformed.")
            before_record = cast(dict[str, object], before)
            after_record = cast(dict[str, object], after)
            supported_record = cast(dict[str, object], supported)
            selected_record = cast(dict[str, object], selected)
            evidence_record = cast(dict[str, object], evidence)
            checkpoint = str(before_record["current_checkpoint_id"])
            supported_actions = supported_record.get("supported_actions")
            if not isinstance(supported_actions, list):
                raise ValueError("Worked checkpoint lacks a supported-action set.")
            checkpoints.append(
                {
                    "programme_id": str(programme["programme_id"]),
                    "stream_id": str(programme["stream_id"]),
                    "step_index": index,
                    "checkpoint_id": checkpoint,
                    "identification_status": str(
                        evidence_record.get("identification_status", "randomized")
                    ),
                    "supported_action_count": len(supported_actions),
                    "selected_action_id": str(selected_record["action_id"]),
                    "target_asset_id": str(
                        selected_record.get("target_asset_id") or ""
                    ),
                    "reserve_asset_id": str(
                        selected_record.get("reserve_asset_id") or ""
                    ),
                    "next_checkpoint_id": str(after_record["current_checkpoint_id"]),
                    "terminal_disposition": str(after_record["terminal_disposition"]),
                    "state_before_sha256": str(before_record["checksum"]),
                    "supported_set_sha256": str(supported_record["checksum"]),
                    "state_after_sha256": str(after_record["checksum"]),
                }
            )
            assessed = assessment_by_checkpoint.get(checkpoint)
            if not isinstance(assessed, dict) or not isinstance(
                assessed.get("capabilities"), list
            ):
                raise ValueError("Worked checkpoint lacks capability assessments.")
            for capability in assessed["capabilities"]:
                if not isinstance(capability, dict):
                    raise ValueError("Worked capability assessment is malformed.")
                capability_id = str(capability["capability_id"])
                capability_counts[capability_id] += 1
                capability_passes[capability_id] += int(
                    capability["outcome"] == "passed"
                )
    metrics = [
        {
            "capability_id": capability,
            "passed_count": capability_passes[capability],
            "evaluated_count": capability_counts[capability],
            "failed_count": capability_counts[capability]
            - capability_passes[capability],
            "rate": capability_passes[capability] / capability_counts[capability],
            "interpretation": "worked_example_reconstruction",
        }
        for capability in sorted(capability_counts)
    ]
    graph_rows = [
        dict(record_kind="node", **row) for row in nodes if isinstance(row, dict)
    ] + [dict(record_kind="edge", **row) for row in edges if isinstance(row, dict)]
    return checkpoints, metrics, graph_rows


def _write_result_tables(
    *,
    output: Path,
    audit: TrialDevPortfolioReleaseAuditV1,
    controls: TrialDevPortfolioGraderControlReportV1,
    difficulty: TrialDevPortfolioDifficultyReportV1,
    replays: tuple[TrialDevObservationalReplayReportV1, ...],
    boundary: dict[str, object],
    policy_value: dict[str, object],
    routes: TrialDevPortfolioRouteAuditV1,
    checkpoints: list[dict[str, object]],
    metrics: list[dict[str, object]],
    graph_rows: list[dict[str, object]],
    operating_summary: list[dict[str, str]],
    sources: list[dict[str, str]],
) -> None:
    _write_csv(
        output / "results" / "worked_checkpoints.csv",
        checkpoints,
        tuple(checkpoints[0]),
    )
    _write_csv(
        output / "results" / "capability_metrics.csv", metrics, tuple(metrics[0])
    )
    graph_fields = tuple(sorted({key for row in graph_rows for key in row}))
    _write_csv(output / "results" / "state_action_graph.csv", graph_rows, graph_fields)
    _write_csv(
        output / "results" / "operating_characteristics.csv",
        [dict(row) for row in operating_summary],
        tuple(operating_summary[0]),
    )
    _write_csv(
        output / "results" / "randomized_episode_realism.csv",
        [row.model_dump(mode="json") for row in audit.episode_realism],
        tuple(audit.episode_realism[0].model_dump(mode="json")),
    )
    _write_csv(
        output / "results" / "observational_realism.csv",
        [
            {
                **row.model_dump(mode="json", exclude={"treatment_counts"}),
                "treatment_counts": json.dumps(
                    row.treatment_counts, sort_keys=True, separators=(",", ":")
                ),
            }
            for row in audit.observational_realism
        ],
        tuple(
            {
                **audit.observational_realism[0].model_dump(
                    mode="json", exclude={"treatment_counts"}
                ),
                "treatment_counts": "",
            }
        ),
    )
    control_rows = [row.model_dump(mode="json") for row in controls.controls]
    _write_csv(
        output / "results" / "grader_controls.csv", control_rows, tuple(control_rows[0])
    )
    strategy_rows = [row.model_dump(mode="json") for row in difficulty.strategies]
    _write_csv(
        output / "results" / "shortcut_strategies.csv",
        strategy_rows,
        tuple(strategy_rows[0]),
    )
    view_rows = [row.model_dump(mode="json") for row in difficulty.views]
    _write_csv(
        output / "results" / "released_view_difficulty.csv",
        view_rows,
        tuple(view_rows[0]),
    )
    cells = boundary["cells"]
    if not isinstance(cells, list) or not all(isinstance(row, dict) for row in cells):
        raise ValueError("Decision-boundary cells must be objects.")
    _write_csv(
        output / "results" / "decision_boundary_cells.csv", cells, tuple(cells[0])
    )
    policy_cells = policy_value["cells"]
    if not isinstance(policy_cells, list) or not all(
        isinstance(row, dict) for row in policy_cells
    ):
        raise ValueError("Policy-value cells must be objects.")
    _write_csv(
        output / "results" / "policy_value_cells.csv",
        policy_cells,
        tuple(policy_cells[0]),
    )
    route_rows = [row.model_dump(mode="json") for row in routes.families]
    _write_csv(
        output / "results" / "portfolio_routes.csv", route_rows, tuple(route_rows[0])
    )
    replay_rows: list[dict[str, object]] = []
    for replay in replays:
        for method in replay.methods:
            replay_rows.append(
                {
                    "scenario_id": replay.scenario_id,
                    "method_route_id": method.method_route_id,
                    "result_form": method.result_form,
                    "bootstrap_replicates": method.bootstrap_replicates,
                    "maximum_utility_absolute_error": method.maximum_utility_absolute_error,
                    "maximum_efficacy_gain_absolute_error": method.maximum_efficacy_gain_absolute_error,
                    "maximum_standard_error_absolute_error": method.maximum_standard_error_absolute_error,
                    "maximum_interval_endpoint_absolute_error": method.maximum_interval_endpoint_absolute_error,
                    "maximum_pairwise_contrast_absolute_error": method.maximum_pairwise_contrast_absolute_error,
                    "ranking_match": method.ranking_match,
                    "action_match": method.action_match,
                    "uncertainty_policy_match": method.uncertainty_policy_match,
                    "status": method.status,
                }
            )
    _write_csv(
        output / "results" / "observational_replay.csv",
        replay_rows,
        tuple(replay_rows[0]),
    )
    _write_csv(
        output / "results" / "sources.csv",
        [dict(row) for row in sources],
        tuple(sources[0]),
    )


def _copy_file(source: Path, destination: Path) -> Path:
    source_path = Path(source).resolve(strict=True)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination


def _validated_export(
    *,
    source_artifact: Path,
    source_report: Path,
    source_preview: Path,
    artifact_destination: Path,
) -> None:
    report = _read_json_object(source_report)
    if report.get("objective_errors") != []:
        raise ValueError(f"Display export has objective errors: {source_report}")
    if report.get("sha256") != _sha256(source_artifact):
        raise ValueError(
            f"Display validation report does not bind the current artifact: {source_artifact}"
        )
    if not Path(source_preview).resolve(strict=True).is_file():
        raise FileNotFoundError(source_preview)
    _copy_file(source_artifact, artifact_destination)


def _copy_displays(
    *,
    output: Path,
    figures_root: Path,
    diagram_root: Path,
) -> int:
    figure_output = output / "figures"
    for display_id in _DISPLAY_IDS:
        if display_id == "01_state_action":
            source_stem = Path(diagram_root) / "trialdev_state_action"
            qa_root = Path(diagram_root)
            _copy_file(
                output / "results" / "state_action_graph.csv",
                figure_output / "source_data" / f"{display_id}.csv",
            )
        else:
            source_stem = Path(figures_root) / display_id
            qa_root = Path(figures_root) / "qa" / display_id
            _copy_file(
                Path(figures_root) / "source_data" / f"{display_id}.csv",
                figure_output / "source_data" / f"{display_id}.csv",
            )
        for extension in ("pdf", "svg"):
            _validated_export(
                source_artifact=source_stem.with_suffix(f".{extension}"),
                source_report=qa_root / f"{extension}_validation.json",
                source_preview=qa_root / f"{extension}_preview.png",
                artifact_destination=figure_output / f"{display_id}.{extension}",
            )
    return len(_DISPLAY_IDS)


def _write_documents(
    *,
    output: Path,
    world_count: int,
    experiment_count: int,
    audit: TrialDevPortfolioReleaseAuditV1,
    controls: TrialDevPortfolioGraderControlReportV1,
    difficulty: TrialDevPortfolioDifficultyReportV1,
    policy_value: dict[str, object],
    routes: TrialDevPortfolioRouteAuditV1,
    replays: tuple[TrialDevObservationalReplayReportV1, ...],
    operating_summary: list[dict[str, str]],
    sources: list[dict[str, str]],
) -> None:
    strategy_rates = {
        row.strategy_id: row.supported_view_rate for row in difficulty.strategies
    }
    policy_cells = policy_value.get("cells")
    if (
        not isinstance(policy_cells, list)
        or not policy_cells
        or not all(isinstance(row, dict) for row in policy_cells)
    ):
        raise ValueError("Policy-value report lacks summary cells.")
    oracle_coverage_lower = min(
        float(row["oracle_action_supported_lower"]) for row in policy_cells
    )
    maximum_action_range = max(
        float(row["worst_supported_regret"]) for row in policy_cells
    )
    maximum_label_regret = max(
        float(row["alphabetical_regret"]) for row in policy_cells
    )
    total_policy_worlds = sum(int(row["world_count"]) for row in policy_cells)
    if total_policy_worlds <= 0:
        raise ValueError("Policy-value report has no simulated programmes.")

    def policy_mean(field: str) -> float:
        return (
            sum(float(row[field]) * int(row["world_count"]) for row in policy_cells)
            / total_policy_worlds
        )

    best_supported_success = policy_mean("best_supported_terminal_success_probability")
    adjusted_point_success = policy_mean("adjusted_point_terminal_success_probability")
    alphabetical_success = policy_mean("alphabetical_terminal_success_probability")
    best_supported_resources = policy_mean("best_supported_expected_resource_units")
    adjusted_point_resources = policy_mean("adjusted_point_expected_resource_units")
    alphabetical_resources = policy_mean("alphabetical_expected_resource_units")
    maximum_replay_error = max(
        max(
            method.maximum_utility_absolute_error,
            method.maximum_efficacy_gain_absolute_error,
            method.maximum_standard_error_absolute_error,
            method.maximum_interval_endpoint_absolute_error,
            method.maximum_pairwise_contrast_absolute_error,
        )
        for replay in replays
        for method in replay.methods
    )
    operating_rows = [
        "| Axis | Outcome | Reference | Intervention | Paired difference (95% interval) |",
        "|---|---|---:|---:|---:|",
    ]
    operating_rows.extend(
        "| {axis} | {metric} | {reference:.3f} | {intervention:.3f} | "
        "{difference:.3f} ({low:.3f} to {high:.3f}) |".format(
            axis=row["axis"].replace("_", " ").title(),
            metric=row["primary_metric"].replace("_", " "),
            reference=float(row["reference_mean"]),
            intervention=float(row["intervention_mean"]),
            difference=float(row["paired_difference"]),
            low=float(row["paired_bootstrap_lower"]),
            high=float(row["paired_bootstrap_upper"]),
        )
        for row in operating_summary
    )
    (output / "README.md").write_text(
        "# TrialDev verification\n\n"
        "TrialDev evaluates whether an analysis system carries a qualified statistical conclusion into a supported "
        "clinical-development decision. The [report](REPORT.md) presents the evidence across programme structure, "
        "independent reconstruction, controlled response, decision difficulty, and decision consequences. The "
        "[methods](METHODS.md), [numerical results](results/), and [verification command](REPRODUCE.md) provide the "
        "corresponding analysis definitions and reproducibility path.\n",
        encoding="utf-8",
    )
    (output / "METHODS.md").write_text(
        "# Methods\n\n"
        "TrialDev contains two sequential programmes: irreversible development of one candidate and development of "
        "three candidates with one permitted lead-reserve reallocation. At each reached checkpoint, the analysis "
        "produces an estimate, uncertainty interval, and evidence status. The declared decision rule maps that "
        "evidence to one or more supported actions. An answer is statistically coherent when its analysis is "
        "identified by the available data and its selected action belongs to that supported set.\n\n"
        "Observational checkpoints first assess identification and empirical support. Lead selection incorporates "
        "uncertainty around utility; reserve selection requires the declared screening evidence and comparison with "
        "the remaining candidates. Randomized checkpoints classify efficacy and noncompensatory safety from "
        "prespecified intervals. Study-design feasibility is determined by the public design menu and operational "
        "bounds; portfolio feasibility is determined by the disclosed resource schedule. Neither is a participant-"
        "reported statistical interval. The worked programmes independently recompute every estimate, action set, "
        "selected action, and state transition.\n\n"
        f"Operating characteristics comprise {experiment_count} matched interventions with {world_count} "
        "common-random-number worlds per arm. Binary arm rates use Wilson intervals, paired contrasts use world-level "
        "bootstrap intervals, and the boundary study uses 400 repeated worlds per cell. Analyses of the released task "
        "are finite censuses. The secondary decision-consequence analysis propagates each supported lead-reserve action "
        "through the known efficacy, safety, resource, and switch mechanisms and reports reference-action coverage and "
        "terminal-success regret. Its best-supported summary selects the action with the highest known "
        "terminal-success probability within the evidence-supported set; it measures the value retained by that set "
        "rather than defining an executable policy for choosing among supported actions.\n\n"
        "An exhaustive finite traversal additionally follows every supported action under each declared analysis "
        "route. It verifies that all checkpoints, actions, and terminal dispositions are reachable; that both "
        "identified and non-identified withholding controls exist; and that the resource schedule permits early "
        "and late reserve promotion. These are exact structural counts, not independent observations or model "
        "performance estimates.\n",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        "# TrialDev verification\n\n"
        "TrialDev asks whether an analysis system can turn the evidence available at each stage of a clinical "
        "development programme into an identified statistical result, a supported decision, and a coherent "
        "programme history. The verification follows that chain directly: programme structure, clinical data, "
        "independent analysis, response to controlled changes, evaluation controls, and downstream consequences.\n\n"
        "## Programme structure\n\n"
        "The single-candidate programme has irreversible stage transitions. The portfolio programme adds a lead and "
        "reserve, a finite resource budget, and one possible reserve promotion after lead failure. Explicit terminal "
        "and nonprogression actions prevent an inconclusive analysis from being converted automatically into "
        "progression.\n\n"
        "![Programme states and actions](figures/01_state_action.svg)\n\n"
        "**Figure 1. Programme states and actions.** Nodes are reachable programme states and arrows are legal "
        "actions under the declared resource schedule. The graph is reconstructed from "
        "[state_action_graph.csv](results/state_action_graph.csv).\n\n"
        "## Clinical data\n\n"
        f"The release contains {audit.world_count} worlds, {audit.participant_view_count} objective-by-budget views, "
        f"and {audit.randomized_episode_count} randomized episodes. The exact audit covers "
        f"{audit.observational_row_count:,} observational and {audit.randomized_row_count:,} randomized rows. "
        "Every world contains a control and three investigational candidates; randomized episodes compare one "
        "candidate with its concurrent control. The data census measures phase-specific sample size and follow-up, "
        "treatment contrasts, safety events, retention, and observational support.\n\n"
        "![Clinical data structure](figures/05_clinical_realism.svg)\n\n"
        "**Figure 2. Clinical data structure.** Each panel reports a released, participant-visible property; points "
        "summarize episodes or worlds rather than design inputs. The panels show that later phases contain more "
        "participants and longer follow-up, while treatment effects, serious events, discontinuation, loss to "
        "follow-up, and observational support vary across programmes. Numerical values are in "
        "[randomized_episode_realism.csv](results/randomized_episode_realism.csv) and "
        "[observational_realism.csv](results/observational_realism.csv).\n\n"
        "## Independent analysis\n\n"
        f"Independent replay passed for {len(replays)} world-level reports, covering both declared observational "
        f"methods where estimable. The largest absolute discrepancy across utility, efficacy gain, standard error, "
        f"interval endpoint, and pairwise contrast was {maximum_replay_error:.2g}. Unsupported observational "
        "comparisons retain non-estimability rather than receiving a numerical substitute.\n\n"
        "![Independent numerical reconstruction](figures/02_identification_uncertainty.svg)\n\n"
        "**Figure 3. Independent numerical reconstruction.** Released results and independent replays are shown "
        "on the same scale. Agreement demonstrates that the numerical results can be recovered from the released "
        "participant evidence; explicit non-estimability demonstrates that unsupported comparisons are not forced "
        "to produce an estimate. Source values "
        "are in [observational_replay.csv](results/observational_replay.csv).\n\n"
        "## Controlled response\n\n"
        f"All {experiment_count} matched experiments changed their prespecified outcome in the expected direction, "
        f"with {world_count} paired worlds per arm and no missing or failed worlds.\n\n"
        + "\n".join(operating_rows)
        + "\n\n"
        "![Policy response](figures/03_policy_response.svg)\n\n"
        "**Figure 4. Policy response.** Matched experiments isolate resource, stopping, and reallocation conditions "
        "while preserving the remaining programme state. Each intervention changes the corresponding programme "
        "decision in all paired worlds, showing that these controls have an observable consequence rather than "
        "serving as descriptive metadata.\n\n"
        "![Mechanism response](figures/04_mechanism_response.svg)\n\n"
        "**Figure 5. Mechanism response.** Efficacy, safety, information, confounding, overlap, operations, and "
        "cross-candidate dependence are changed one at a time. The resulting shifts occur in the intended clinical "
        "or analytical quantity; intervals use the paired world as the independent unit. Complete estimates are in "
        "[operating_characteristics.csv](results/operating_characteristics.csv).\n\n"
        "![Operating characteristics](figures/06_operating_characteristics.svg)\n\n"
        "**Figure 6. Operating characteristics.** Reference and intervention rates are shown with their uncertainty, "
        "so the intended direction, effect size, and sampling precision can be assessed separately across all ten "
        "controlled changes.\n\n"
        "## Evaluation controls\n\n"
        f"The exact evaluation census contains {len(controls.controls)} positive and isolated-fault controls. "
        "Reference submissions produced complete scientific grades; numeric, provenance, action, and design faults "
        "were isolated in their owning responsibilities, and stale states were rejected before grading.\n\n"
        "![Evaluation controls](figures/07_grader_controls.svg)\n\n"
        "**Figure 7. Evaluation controls.** Each point gives the exact agreement rate and denominator for one "
        "positive or single-fault control. Full records are in "
        "[grader_controls.csv](results/grader_controls.csv).\n\n"
        "## Decision difficulty\n\n"
        "Repeated-world boundary experiments show that evidence becomes more decisive as information increases away "
        "from the efficacy and safety thresholds, while evidence at the thresholds remains predominantly "
        "indeterminate. Across the exact 96 views, complete evidence-and-policy analysis is supported in every view; "
        f"adjusted point ranking reaches {strategy_rates['adjusted_point_pair']:.1%}, always withholding reaches "
        f"{strategy_rates['always_withhold']:.1%}, raw point ranking reaches "
        f"{strategy_rates['raw_observed_pair']:.1%}, and alphabetical selection reaches "
        f"{strategy_rates['alphabetical_pair']:.1%}.\n\n"
        "![Decision difficulty](figures/08_decision_difficulty.svg)\n\n"
        "**Figure 8. Decision difficulty.** Boundary cells show how information and distance from the efficacy or "
        "safety threshold change clear-pass, clear-fail, and indeterminate evidence. The released-view census "
        "compares complete analysis with prespecified uncertainty-blind strategies. Values are in "
        "[decision_boundary_cells.csv](results/decision_boundary_cells.csv) and "
        "[shortcut_strategies.csv](results/shortcut_strategies.csv).\n\n"
        "## Decision consequences\n\n"
        "Evidence may support more than one action, and those actions can lead to different programme outcomes. "
        "The known programme probabilities give the chance of successful programme completion for every supported "
        "action. The analysis identifies the supported action with the highest probability and measures how often "
        "the supported set retains it. These probabilities evaluate the supported options; they are not an "
        "additional decision rule available to the analysis system.\n\n"
        f"Across {total_policy_worlds:,} simulated programmes, the terminal-success-maximising supported action had "
        f"mean terminal-success probability {best_supported_success:.3f}, compared with "
        f"{adjusted_point_success:.3f} for adjusted point-estimate selection and "
        f"{alphabetical_success:.3f} for selection by asset label. All strategies operated under the same 8- or "
        f"10-unit resource budgets. Their mean realised resource use was {best_supported_resources:.3f}, "
        f"{adjusted_point_resources:.3f}, and {alphabetical_resources:.3f} units, respectively.\n\n"
        "| Decision rule | Mean terminal-success probability | Mean realised resource use |\n"
        "|---|---:|---:|\n"
        f"| Terminal-success-maximising supported action | {best_supported_success:.3f} | "
        f"{best_supported_resources:.3f} |\n"
        f"| Adjusted point-estimate selection | {adjusted_point_success:.3f} | "
        f"{adjusted_point_resources:.3f} |\n"
        f"| Selection by asset label | {alphabetical_success:.3f} | "
        f"{alphabetical_resources:.3f} |\n\n"
        "Across the prespecified grid, the lower 95% confidence bound for retaining the "
        f"terminal-success-maximising action is at least {oracle_coverage_lower:.1%}. The largest mean gap between "
        "the best and worst supported action "
        f"is {maximum_action_range:.3f} in terminal-success probability, showing why support and selection within "
        "the supported set remain distinct. The 10-unit schedule increases terminal success by permitting a late "
        "reserve switch after lead failure. Asset identities are permuted across repeated "
        f"worlds; the largest mean regret of alphabetical selection is {maximum_label_regret:.3f}, so stable labels do "
        "not encode candidate quality.\n\n"
        "![Decision consequences](figures/09_policy_value.svg)\n\n"
        "**Figure 9. Decision consequences.** Reference-action coverage and terminal-success regret are reported across "
        "candidate separation, information size, and resource budget with Wilson or paired-bootstrap intervals. "
        "Numerical results are in [policy_value_cells.csv](results/policy_value_cells.csv).\n\n"
        "## Supported programme routes\n\n"
        f"Exhaustive traversal evaluated {routes.evaluated_state_count:,} method-conditioned states and "
        f"{routes.terminal_route_count:,} terminal routes across all {routes.participant_view_count} released "
        "views. It reached all nine action types, all five checkpoints, and all five terminal dispositions. The "
        f"census includes {routes.joint_safety_stop_state_count} joint safety-stop states, "
        f"{routes.early_reserve_promotion_route_count} early and {routes.late_reserve_promotion_route_count} late "
        "reserve-promotion routes, and both identified and non-identified withholding controls.\n\n"
        "![Supported programme routes](figures/10_portfolio_routes.svg)\n\n"
        "**Figure 10. Supported programme routes.** The action matrix gives exact reachability by trial "
        "family; ranges give the minimum and maximum number of terminal routes across the eight objective-by-budget "
        "views of each family. Views sharing a world are not independent. Numerical values are in "
        "[portfolio_routes.csv](results/portfolio_routes.csv).\n\n"
        "Together, these analyses establish that the released task contains recognizable clinical-development data, "
        "its declared analyses can be reproduced from participant evidence, controlled changes produce the intended "
        "statistical response, incomplete strategies leave material cases unresolved, and supported actions retain "
        "high-value options across the tested programme conditions.\n",
        encoding="utf-8",
    )
    source_lines = [
        "# Sources",
        "",
        "The verified source records used by this package are reproduced in `results/sources.csv`.",
        "",
    ]
    source_lines.extend(
        f"- {row['source_id']}: [{row['title'].replace('—', ': ')}]"
        f"({row['canonical_url']}). {row['notes'].rstrip('.')}."
        for row in sources
    )
    (output / "SOURCES.md").write_text("\n".join(source_lines) + "\n", encoding="utf-8")
    (output / "REPRODUCE.md").write_text(
        "# Verify the package\n\n"
        "Install `trialagentbench-validation`, then run:\n\n"
        "```bash\n"
        "trialagentbench-validate trialdev-scientific-package-verify \\\n"
        "  --package-root . \\\n"
        "  --output verification.json\n"
        "```\n\n"
        "The verifier checks exact membership and hashes, independently reconstructs the worked programmes and "
        "decision-consequence results, validates the matched operating study, and parses every world-level replay and "
        "evaluation control. Figure QA is completed before package construction; the released PDF, SVG, and source "
        "table for each display are bound by `manifest.json`.\n",
        encoding="utf-8",
    )


def _manifest(
    output: Path,
    *,
    source_identity: str,
    world_count: int,
    experiment_count: int,
    audit: TrialDevPortfolioReleaseAuditV1,
    grader_control_count: int,
    observational_replay_count: int,
    display_count: int,
) -> TrialDevScientificPackageManifestV1:
    artifacts = tuple(
        ScientificPackageArtifactV1(
            relative_path=path.relative_to(output).as_posix(),
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(
            item
            for item in output.rglob("*")
            if item.is_file() and item.name != "manifest.json"
        )
    )
    return TrialDevScientificPackageManifestV1(
        source_identity=source_identity,
        world_count_per_arm=world_count,
        matched_experiment_count=experiment_count,
        released_world_count=audit.world_count,
        participant_view_count=audit.participant_view_count,
        randomized_episode_count=audit.randomized_episode_count,
        grader_control_count=grader_control_count,
        observational_replay_count=observational_replay_count,
        display_count=display_count,
        artifacts=artifacts,
    )


def build_trialdev_scientific_package(
    *,
    worked_root: Path,
    operating_root: Path,
    decision_boundary_report: Path,
    policy_value_root: Path,
    release_audit_report: Path,
    grader_control_report: Path,
    portfolio_difficulty_report: Path,
    portfolio_route_report: Path,
    observational_replay_root: Path,
    figures_root: Path,
    diagram_root: Path,
    source_manifest: Path,
    output_dir: Path,
) -> TrialDevScientificPackageManifestV1:
    """Build the complete public TrialDev scientific verification package."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite scientific package: {output}")
    worked = audit_trialdev_worked_programmes(package_root=worked_root)
    if worked.status != "pass":
        raise ValueError(
            f"Worked programmes failed independent reconstruction: {worked.findings!r}"
        )
    summary, _, source_identity, world_count = _validate_operating_inputs(
        Path(operating_root)
    )
    if worked.source_identity != source_identity:
        raise ValueError(
            "Worked programmes and operating characteristics use different source identities."
        )
    audit = TrialDevPortfolioReleaseAuditV1.model_validate_json(
        Path(release_audit_report).read_text(encoding="utf-8")
    )
    controls = TrialDevPortfolioGraderControlReportV1.model_validate_json(
        Path(grader_control_report).read_text(encoding="utf-8")
    )
    difficulty = TrialDevPortfolioDifficultyReportV1.model_validate_json(
        Path(portfolio_difficulty_report).read_text(encoding="utf-8")
    )
    routes = TrialDevPortfolioRouteAuditV1.model_validate_json(
        Path(portfolio_route_report).read_text(encoding="utf-8")
    )
    if any(report.status != "pass" for report in (audit, controls, difficulty, routes)):
        raise ValueError(
            "Exact-release audit, grader controls, difficulty, and route analysis must pass."
        )
    if {
        audit.release_source_identity,
        controls.release_source_identity,
        difficulty.release_source_identity,
        routes.release_source_identity,
    } != {source_identity}:
        raise ValueError("Exact-release evidence uses inconsistent source identities.")
    if (
        controls.participant_view_count != audit.participant_view_count
        or difficulty.participant_view_count != audit.participant_view_count
        or routes.participant_view_count != audit.participant_view_count
    ):
        raise ValueError(
            "Exact-release evidence disagrees on the participant-view census."
        )
    boundary = _validate_decision_boundary(
        Path(decision_boundary_report), source_identity=source_identity
    )
    policy_root = Path(policy_value_root)
    policy_value = _validate_policy_value(
        policy_root / "policy_value_report.json", source_identity=source_identity
    )
    policy_audit = audit_trialdev_policy_value_v1(policy_value_root=policy_root)
    if policy_audit.status != "pass" or policy_audit.source_identity != source_identity:
        raise ValueError(
            "Policy-value evidence failed independent reconstruction or source binding."
        )
    replays = _load_observational_replays(Path(observational_replay_root))
    if {report.scenario_id for report in replays} != {
        row.world_id for row in audit.observational_realism
    }:
        raise ValueError(
            "Observational replays do not cover the exact released world census."
        )
    sources = _source_rows(Path(source_manifest))
    checkpoints, metrics, graph_rows = _worked_tables(Path(worked_root))

    output.mkdir(parents=True)
    shutil.copytree(worked_root, output / "inputs" / "worked_programmes")
    operating_input = output / "inputs" / "operating_characteristics"
    operating_input.mkdir(parents=True)
    for name in _OPERATING_FILES:
        _copy_file(Path(operating_root) / name, operating_input / name)
    _write_csv(
        output / "inputs" / "sources" / "SOURCE_MANIFEST.tsv",
        [dict(row) for row in sources],
        _PUBLIC_SOURCE_FIELDS,
        delimiter="\t",
    )
    verification = output / "verification"
    _copy_file(
        Path(release_audit_report), verification / "portfolio_release_audit.json"
    )
    _copy_file(Path(grader_control_report), verification / "grader_controls.json")
    _copy_file(
        Path(portfolio_difficulty_report), verification / "portfolio_difficulty.json"
    )
    _copy_file(Path(portfolio_route_report), verification / "portfolio_routes.json")
    _copy_file(
        Path(decision_boundary_report), verification / "decision_boundary_report.json"
    )
    policy_input = output / "inputs" / "policy_value"
    for name in (
        "policy_value_report.json",
        "policy_value_cells.csv",
        "policy_value_worlds.csv",
        "policy_value_candidates.csv",
    ):
        _copy_file(policy_root / name, policy_input / name)
    (verification / "independent_policy_value_audit.json").write_text(
        policy_audit.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    for path in sorted(
        Path(observational_replay_root).glob("observational_replay_world_*.json")
    ):
        _copy_file(path, verification / "observational_replays" / path.name)
    (verification / "worked_programmes.json").write_text(
        worked.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    _write_result_tables(
        output=output,
        audit=audit,
        controls=controls,
        difficulty=difficulty,
        replays=replays,
        boundary=boundary,
        policy_value=policy_value,
        routes=routes,
        checkpoints=checkpoints,
        metrics=metrics,
        graph_rows=graph_rows,
        operating_summary=summary,
        sources=sources,
    )
    display_count = _copy_displays(
        output=output,
        figures_root=Path(figures_root),
        diagram_root=Path(diagram_root),
    )
    _write_documents(
        output=output,
        world_count=world_count,
        experiment_count=len(summary),
        audit=audit,
        controls=controls,
        difficulty=difficulty,
        policy_value=policy_value,
        routes=routes,
        replays=replays,
        operating_summary=summary,
        sources=sources,
    )
    manifest = _manifest(
        output,
        source_identity=source_identity,
        world_count=world_count,
        experiment_count=len(summary),
        audit=audit,
        grader_control_count=len(controls.controls),
        observational_replay_count=len(replays),
        display_count=display_count,
    )
    (output / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_trialdev_scientific_package(
    *, package_root: Path
) -> TrialDevScientificPackageVerificationV1:
    """Verify the exact file inventory and self-contained scientific inputs."""

    root = Path(package_root).resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = TrialDevScientificPackageManifestV1.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    findings: list[str] = []
    declared = {item.relative_path: item for item in manifest.artifacts}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    for missing in sorted(set(declared) - observed):
        findings.append(f"missing_artifact:{missing}")
    for extra in sorted(observed - set(declared)):
        findings.append(f"undeclared_artifact:{extra}")
    for relative in sorted(set(declared) & observed):
        path = root / relative
        record = declared[relative]
        if _sha256(path) != record.sha256 or path.stat().st_size != record.size_bytes:
            findings.append(f"artifact_identity_disagreement:{relative}")
    try:
        worked = audit_trialdev_worked_programmes(
            package_root=root / "inputs" / "worked_programmes"
        )
        summary, _, source_identity, world_count = _validate_operating_inputs(
            root / "inputs" / "operating_characteristics"
        )
        audit = TrialDevPortfolioReleaseAuditV1.model_validate_json(
            (root / "verification" / "portfolio_release_audit.json").read_text(
                encoding="utf-8"
            )
        )
        controls = TrialDevPortfolioGraderControlReportV1.model_validate_json(
            (root / "verification" / "grader_controls.json").read_text(encoding="utf-8")
        )
        difficulty = TrialDevPortfolioDifficultyReportV1.model_validate_json(
            (root / "verification" / "portfolio_difficulty.json").read_text(
                encoding="utf-8"
            )
        )
        routes = TrialDevPortfolioRouteAuditV1.model_validate_json(
            (root / "verification" / "portfolio_routes.json").read_text(
                encoding="utf-8"
            )
        )
        boundary = _validate_decision_boundary(
            root / "verification" / "decision_boundary_report.json",
            source_identity=source_identity,
        )
        policy_value = _validate_policy_value(
            root / "inputs" / "policy_value" / "policy_value_report.json",
            source_identity=source_identity,
        )
        policy_audit = audit_trialdev_policy_value_v1(
            policy_value_root=root / "inputs" / "policy_value"
        )
        replays = _load_observational_replays(
            root / "verification" / "observational_replays"
        )
    except (FileNotFoundError, KeyError, OSError, UnicodeError, ValueError) as error:
        findings.append(f"scientific_reconstruction_failed:{error}")
    else:
        if worked.status != "pass" or worked.source_identity != source_identity:
            findings.append("worked_programme_identity_disagreement")
        if (
            policy_audit.status != "pass"
            or policy_audit.source_identity != source_identity
        ):
            findings.append("policy_value_reconstruction_disagreement")
        if routes.status != "pass":
            findings.append("portfolio_route_audit_failed")
        expected_counts = (
            world_count,
            len(summary),
            audit.world_count,
            audit.participant_view_count,
            audit.randomized_episode_count,
            len(controls.controls),
            len(replays),
            len(_DISPLAY_IDS),
        )
        manifest_counts = (
            manifest.world_count_per_arm,
            manifest.matched_experiment_count,
            manifest.released_world_count,
            manifest.participant_view_count,
            manifest.randomized_episode_count,
            manifest.grader_control_count,
            manifest.observational_replay_count,
            manifest.display_count,
        )
        if expected_counts != manifest_counts:
            findings.append("scientific_census_disagreement")
        if (
            {
                audit.release_source_identity,
                controls.release_source_identity,
                difficulty.release_source_identity,
                routes.release_source_identity,
            }
            != {manifest.source_identity}
            or boundary.get("source_identity") != manifest.source_identity
            or policy_value.get("source_identity") != manifest.source_identity
        ):
            findings.append("scientific_source_identity_disagreement")
        if routes.participant_view_count != audit.participant_view_count:
            findings.append("portfolio_route_census_disagreement")
        if {report.scenario_id for report in replays} != {
            row.world_id for row in audit.observational_realism
        }:
            findings.append("observational_replay_census_disagreement")
    for display_id in _DISPLAY_IDS:
        for relative_path in (
            f"figures/{display_id}.pdf",
            f"figures/{display_id}.svg",
            f"figures/source_data/{display_id}.csv",
        ):
            if relative_path not in declared:
                findings.append(f"display_artifact_missing:{relative_path}")
    ordered = tuple(sorted(set(findings)))
    return TrialDevScientificPackageVerificationV1(
        manifest_sha256=_sha256(manifest_path),
        artifact_count=len(manifest.artifacts),
        findings=ordered,
        status="fail" if ordered else "pass",
    )


__all__ = [
    "TrialDevScientificPackageManifestV1",
    "TrialDevScientificPackageVerificationV1",
    "build_trialdev_scientific_package",
    "verify_trialdev_scientific_package",
]
