"""Export evaluator-owned task identities and targeted-control applicability."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from trialagentbench_harness.contracts.experiments import (
    TrialEvalAblationEvaluatorLabelsV1,
    TrialEvalAblationTaskIdentityV1,
    TrialEvalPromptConditionV1,
    TrialEvalTargetedApplicabilityLabelV1,
    trialeval_capability_prompt_conditions_v1,
)
from trialagentbench_harness.contracts.scoring.assumption_evidence import (
    AssumptionEvidenceManifestV1,
    read_assumption_evidence_domains,
)
from trialagentbench_harness.io import sha256_dir_digest, write_json_model

_SURVIVAL_ASSUMPTIONS = frozenset({"censoring_ignorability", "proportional_hazards"})
_PROMPT_DOMAIN: dict[TrialEvalPromptConditionV1, str] = {
    "targeted_covariate_structure": "covariate_structure",
    "targeted_survival_assumptions": "survival_assumptions",
    "targeted_design_structure": "design_structure",
    "targeted_data_integrity": "data_integrity",
}


def _index_entries(evaluator_root: Path) -> tuple[dict[str, object], ...]:
    path = evaluator_root / "grader" / "item_index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_id") != "trialagentbench.trial_benchmark.grader_item_index/v1"
    ):
        raise ValueError("TrialEval evaluator item index has an unsupported schema_id.")
    rows = payload.get("entries")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("TrialEval evaluator item index requires a non-empty entries array.")
    task_ids = [row.get("task_id") for row in rows]
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise ValueError("TrialEval evaluator item index contains an invalid task_id.")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("TrialEval evaluator item index contains duplicate task_id values.")
    return tuple(cast(dict[str, object], row) for row in rows)


def _active_domains(
    *,
    entry: dict[str, object],
    assumption_evidence: AssumptionEvidenceManifestV1,
) -> dict[str, tuple[str, ...]]:
    factors = entry.get("factors")
    if not isinstance(factors, dict):
        raise ValueError(f"Task {entry.get('task_id')!r} lacks evaluator-owned factor metadata.")
    design_subtype = factors.get("design_subtype")
    valid_design_subtypes = {
        "individual_randomized",
        "pragmatic",
        "covariate_structure",
        "endpoint_ascertainment",
        "cluster_parallel",
        "stepped_wedge",
        "group_sequential",
    }
    if design_subtype not in valid_design_subtypes:
        raise ValueError(f"Task {entry.get('task_id')!r} has an invalid design_subtype.")
    defect_count = entry.get("data_integrity_reference_row_count")
    if isinstance(defect_count, bool) or not isinstance(defect_count, int) or defect_count < 0:
        raise ValueError(f"Task {entry.get('task_id')!r} has an invalid data_integrity_reference_row_count.")

    domains: dict[str, tuple[str, ...]] = {}
    if design_subtype == "covariate_structure":
        domains["covariate_structure"] = (
            "item_index.factors.design_subtype=covariate_structure",
            "retained_D3_scope=analysis_defining_covariate_structure",
        )
    if design_subtype in {"cluster_parallel", "stepped_wedge", "group_sequential"}:
        domains["design_structure"] = (f"item_index.factors.design_subtype={design_subtype}",)
    if defect_count > 0:
        domains["data_integrity"] = (f"item_index.data_integrity_reference_row_count={defect_count}",)

    stressed = tuple(
        sorted(
            record.assumption_id
            for record in assumption_evidence.records
            if record.assumption_id in _SURVIVAL_ASSUMPTIONS
            and record.computed_status != "holds"
            and record.diagnosability != "not_identifiable"
        )
    )
    if stressed:
        domains["survival_assumptions"] = tuple(
            f"assumption_evidence.{assumption_id}.computed_status!=holds" for assumption_id in stressed
        )
    return domains


def build_trialeval_ablation_evaluator_labels_v1(
    *,
    evaluator_root: Path,
) -> TrialEvalAblationEvaluatorLabelsV1:
    """Build post-response labels solely from frozen evaluator facts."""

    root = Path(evaluator_root)
    entries = _index_entries(root)
    evidence = read_assumption_evidence_domains(release_root=root)
    indexed_task_ids = {cast(str, entry["task_id"]) for entry in entries}
    if set(evidence) != indexed_task_ids:
        missing = sorted(indexed_task_ids.difference(evidence))
        extra = sorted(set(evidence).difference(indexed_task_ids))
        raise ValueError(
            f"Assumption-evidence coverage differs from item index: missing={missing!r}, extra={extra!r}."
        )

    identities: list[TrialEvalAblationTaskIdentityV1] = []
    labels: list[TrialEvalTargetedApplicabilityLabelV1] = []
    capability_conditions = trialeval_capability_prompt_conditions_v1()
    for entry in entries:
        task_id = cast(str, entry["task_id"])
        base_trial_id = entry.get("item_id")
        if not isinstance(base_trial_id, str) or not base_trial_id:
            raise ValueError(f"Task {task_id!r} lacks an evaluator-owned base trial identity.")
        regime_cell_id = entry.get("base_case_id")
        if not isinstance(regime_cell_id, str) or not regime_cell_id:
            raise ValueError(f"Task {task_id!r} lacks an evaluator-owned regime-cell identity.")
        factors = entry.get("factors")
        if not isinstance(factors, dict):
            raise ValueError(f"Task {task_id!r} lacks evaluator-owned factor metadata.")
        identities.append(
            TrialEvalAblationTaskIdentityV1.model_validate(
                {
                    "task_id": task_id,
                    "base_trial_id": base_trial_id,
                    "regime_cell_id": regime_cell_id,
                    "evaluation_series_id": regime_cell_id.rsplit("-", maxsplit=1)[0],
                    "design_tier": factors.get("design_archetype"),
                    "design_subtype": factors.get("design_subtype"),
                    "assumption_tier": factors.get("assumption_regime"),
                    "context_tier": factors.get("context_configuration"),
                    "data_preparation": factors.get("data_preparation"),
                    "analysis_specification": factors.get("analysis_specification"),
                }
            )
        )
        active = _active_domains(entry=entry, assumption_evidence=evidence[task_id])
        for condition in capability_conditions:
            domain = _PROMPT_DOMAIN[condition]
            if domain in active:
                applicability: Literal["applicable", "mismatched", "inapplicable"] = "applicable"
                basis = active[domain]
            elif active:
                applicability = "mismatched"
                basis = tuple(basis_entry for active_domain in sorted(active) for basis_entry in active[active_domain])
            else:
                applicability = "inapplicable"
                basis = ("no_targeted_capability_domain_active",)
            labels.append(
                TrialEvalTargetedApplicabilityLabelV1(
                    task_id=task_id,
                    prompt_condition=condition,
                    applicability=applicability,
                    evidence_basis=basis,
                )
            )

    return TrialEvalAblationEvaluatorLabelsV1(
        evaluator_release_sha256=sha256_dir_digest(root),
        task_identities=tuple(identities),
        labels=tuple(labels),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-dir", required=True, help="Extracted TrialEval evaluator release root.")
    parser.add_argument("--out", required=True, help="New evaluator-label JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export one immutable evaluator-label artifact."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite TrialEval evaluator labels: {output}")
    labels = build_trialeval_ablation_evaluator_labels_v1(evaluator_root=Path(args.evaluator_dir))
    write_json_model(output, labels)
    print(f"Wrote TrialEval ablation evaluator labels: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
