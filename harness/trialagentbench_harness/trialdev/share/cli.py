"""CLI entrypoints for the share package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from trialagentbench_harness.trialdev.share.inspect import inspect_scenario_bundle_v1
from trialagentbench_harness.trialdev.share.io import read_json
from trialagentbench_harness.trialdev.share.materialize import materialize_trial_view_v1
from trialagentbench_harness.trialdev.share.models import (
    PhaseModuleSpecV1,
    TrialDevelopmentEvalContractV1,
    TrialDevelopmentRequestV1,
)
from trialagentbench_harness.trialdev.share.sequential import (
    TrialDevelopmentProgramLoopManifestV1,
    validate_design_request_file_v1,
    validate_phase_action_policy_file_v1,
    validate_phase_analysis_file_v1,
    validate_phase_decision_against_policy_v1,
    validate_phase_decision_file_v1,
    validate_trial_output_bundle_v1,
)
from trialagentbench_harness.trialdev.share.validate import (
    candidate_ids_by_role_v1,
    validate_public_scenario_bundle_v1,
    validate_request_against_scenario_file_v1,
    validate_request_file_v1,
    validate_request_shape_file_v1,
    validate_scenario_bundle_v1,
    validate_submission_shape_file_v1,
)

__all__ = ["main"]

T = TypeVar("T")


def _cmd_validate(args: argparse.Namespace) -> int:
    validate_scenario_bundle_v1(scenario_root=Path(args.scenario_root))
    print("ok")
    return 0


def _cmd_validate_public(args: argparse.Namespace) -> int:
    validate_public_scenario_bundle_v1(scenario_root=Path(args.scenario_root))
    print("ok")
    return 0


def _cmd_validate_request(args: argparse.Namespace) -> int:
    validate_request_file_v1(request_path=Path(args.request))
    print("ok")
    return 0


def _cmd_validate_request_shape(args: argparse.Namespace) -> int:
    validate_request_shape_file_v1(request_path=Path(args.request))
    print("ok")
    return 0


def _first_non_control_candidate(scenario_root: Path) -> str:
    return candidate_ids_by_role_v1(scenario_root=scenario_root)["investigational"][0]


def _eval_contract(scenario_root: Path) -> TrialDevelopmentEvalContractV1:
    validate_public_scenario_bundle_v1(scenario_root=scenario_root)
    return TrialDevelopmentEvalContractV1.model_validate(read_json(scenario_root / "public" / "eval_contract.json"))


def _phase_module(scenario_root: Path, phase_id: str) -> PhaseModuleSpecV1:
    contract = _eval_contract(scenario_root)
    for module in contract.phase_modules:
        if module.phase_id == phase_id:
            return module
    raise ValueError(f"Phase {phase_id!r} not found in the public evaluation contract.")


def _first_declared(values: tuple[T, ...], *, field: str, phase_id: str) -> T:
    if not values:
        raise ValueError(f"Phase {phase_id!r} does not declare any values for {field}.")
    return values[0]


def _cmd_make_smoke_request(args: argparse.Namespace) -> int:
    scenario_root = Path(args.scenario_root)
    phase_id = str(args.phase)
    module = _phase_module(scenario_root, phase_id)
    scenario_id = scenario_root.name.removeprefix("scenario_")
    payload: dict[str, object] = {
        "version": "v1",
        "scenario_id": scenario_id,
        "phase_id": phase_id,
        "candidate_drug_ids": [_first_non_control_candidate(scenario_root)],
        "target_sample_size": int(args.target_sample_size),
        "follow_up_days": int(
            _first_declared(module.allowed_follow_up_days, field="allowed_follow_up_days", phase_id=phase_id)
        ),
        "enrollment_window_days": int(
            _first_declared(
                module.allowed_enrollment_window_days,
                field="allowed_enrollment_window_days",
                phase_id=phase_id,
            )
        ),
        "site_count_budget": int(
            _first_declared(module.allowed_site_count_budgets, field="allowed_site_count_budgets", phase_id=phase_id)
        ),
        "allocation_ratio": _first_declared(
            module.allowed_allocation_ratios, field="allowed_allocation_ratios", phase_id=phase_id
        ),
        "interim_policy": _first_declared(
            module.allowed_interim_policies, field="allowed_interim_policies", phase_id=phase_id
        ),
        "site_strategy": _first_declared(
            module.allowed_site_strategies, field="allowed_site_strategies", phase_id=phase_id
        ),
        "selection_objective": _first_declared(
            module.allowed_selection_objectives, field="allowed_selection_objectives", phase_id=phase_id
        ),
    }
    if module.allowed_treatment_discontinuation_strategies:
        payload["treatment_discontinuation_strategy"] = module.allowed_treatment_discontinuation_strategies[0]
    if module.allowed_endpoint_ids:
        payload["endpoint_id"] = module.allowed_endpoint_ids[0]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    validate_request_against_scenario_file_v1(scenario_root=scenario_root, request_path=out_path)
    print(json.dumps({"request": str(out_path), "valid": True}, sort_keys=True, indent=2))
    return 0


def _cmd_validate_design_request(args: argparse.Namespace) -> int:
    request = validate_design_request_file_v1(request_path=Path(args.request))
    print(json.dumps({"valid": True, "request_checksum": request.checksum()}, sort_keys=True, indent=2))
    return 0


def _cmd_validate_request_against_scenario(args: argparse.Namespace) -> int:
    request = validate_request_against_scenario_file_v1(
        scenario_root=Path(args.scenario_root),
        request_path=Path(args.request),
    )
    print(json.dumps({"valid": True, "request_checksum": request.checksum()}, sort_keys=True, indent=2))
    return 0


def _cmd_inspect_phase_menu(args: argparse.Namespace) -> int:
    scenario_root = Path(args.scenario_root)
    contract = _eval_contract(scenario_root)
    loop_manifest = TrialDevelopmentProgramLoopManifestV1.model_validate(
        read_json(scenario_root / "public" / "program_loop_manifest.json")
    )
    modules = list(contract.phase_modules)
    if args.phase is not None:
        modules = [module for module in modules if module.phase_id == str(args.phase)]
    if not modules:
        raise ValueError("No matching phase modules found.")
    rows = [
        {
            "phase_id": module.phase_id,
            "includes_control_arm": module.includes_control_arm,
            "program_archetype": loop_manifest.program_archetype,
            "phase_policy_mode": loop_manifest.phase_policy_modes[module.phase_id],
            "endpoint_required": module.phase_id in {"phase2", "phase3"},
            "allowed_endpoint_ids": list(module.allowed_endpoint_ids),
            "allowed_follow_up_days": list(module.allowed_follow_up_days),
            "allowed_enrollment_window_days": list(module.allowed_enrollment_window_days),
            "allowed_site_count_budgets": list(module.allowed_site_count_budgets),
            "allowed_allocation_ratios": list(module.allowed_allocation_ratios),
            "allowed_treatment_discontinuation_strategies": list(module.allowed_treatment_discontinuation_strategies),
            "allowed_interim_policies": list(module.allowed_interim_policies),
            "allowed_site_strategies": list(module.allowed_site_strategies),
            "allowed_selection_objectives": list(module.allowed_selection_objectives),
            "max_sample_size": module.max_sample_size,
            "max_analysis_covariates": module.max_analysis_covariates,
            "max_subgroup_splits": module.max_subgroup_splits,
        }
        for module in modules
    ]
    print(json.dumps({"phase_modules": rows}, sort_keys=True, indent=2))
    return 0


def _cmd_validate_submission_shape(args: argparse.Namespace) -> int:
    validate_submission_shape_file_v1(submission_path=Path(args.submission))
    print("ok")
    return 0


def _cmd_validate_trial_output(args: argparse.Namespace) -> int:
    manifest = validate_trial_output_bundle_v1(trial_output_root=Path(args.trial_output_root))
    print(json.dumps(manifest.model_dump(mode="json", exclude_none=True), sort_keys=True, indent=2))
    return 0


def _cmd_validate_phase_analysis(args: argparse.Namespace) -> int:
    submission = validate_phase_analysis_file_v1(submission_path=Path(args.submission))
    print(json.dumps(submission.model_dump(mode="json", exclude_none=True), sort_keys=True, indent=2))
    return 0


def _cmd_validate_phase_action_policy(args: argparse.Namespace) -> int:
    policy = validate_phase_action_policy_file_v1(scenario_root=Path(args.scenario_root))
    print(json.dumps(policy.model_dump(mode="json", exclude_none=True), sort_keys=True, indent=2))
    return 0


def _cmd_validate_phase_decision(args: argparse.Namespace) -> int:
    if args.scenario_root is None:
        submission = validate_phase_decision_file_v1(submission_path=Path(args.submission))
    else:
        submission = validate_phase_decision_against_policy_v1(
            scenario_root=Path(args.scenario_root),
            submission_path=Path(args.submission),
        )
    print(json.dumps(submission.model_dump(mode="json", exclude_none=True), sort_keys=True, indent=2))
    return 0


def _cmd_materialize(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    request = TrialDevelopmentRequestV1.model_validate(payload)
    result = materialize_trial_view_v1(
        scenario_root=Path(args.scenario_root),
        request=request,
        seed=int(args.seed),
        out_dir=Path(args.out),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result.model_dump(mode="json", exclude_none=True), sort_keys=True, indent=2))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    summary = inspect_scenario_bundle_v1(
        scenario_root=Path(args.scenario_root),
        include_hidden_diagnostics=bool(args.include_hidden_diagnostics),
    )
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trial-benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate = sub.add_parser("validate", help="Validate a scenario bundle using its checksummed manifest.")
    validate.add_argument("--scenario-root", required=True)
    validate.set_defaults(_fn=_cmd_validate)

    validate_public = sub.add_parser("validate-public", help="Validate participant-visible scenario files.")
    validate_public.add_argument("--scenario-root", required=True)
    validate_public.set_defaults(_fn=_cmd_validate_public)

    validate_request = sub.add_parser("validate-request", help="Validate one trial request JSON file.")
    validate_request.add_argument("--request", required=True)
    validate_request.set_defaults(_fn=_cmd_validate_request)

    validate_request_shape = sub.add_parser("validate-request-shape", help="Validate one request-shape JSON file.")
    validate_request_shape.add_argument("--request", required=True)
    validate_request_shape.set_defaults(_fn=_cmd_validate_request_shape)

    make_smoke_request = sub.add_parser("make-smoke-request", help="Create a minimal executable request.")
    make_smoke_request.add_argument("--scenario-root", required=True)
    make_smoke_request.add_argument("--out", required=True)
    make_smoke_request.add_argument("--phase", required=True, choices=("phase1", "phase2", "phase3"))
    make_smoke_request.add_argument("--target-sample-size", required=True, type=int)
    make_smoke_request.set_defaults(_fn=_cmd_make_smoke_request)

    validate_design_request = sub.add_parser("validate-design-request", help="Validate one stepwise design request.")
    validate_design_request.add_argument("--request", required=True)
    validate_design_request.set_defaults(_fn=_cmd_validate_design_request)

    validate_request_against_scenario = sub.add_parser(
        "validate-request-against-scenario",
        help="Validate one request against a scenario's public phase menus.",
    )
    validate_request_against_scenario.add_argument("--scenario-root", required=True)
    validate_request_against_scenario.add_argument("--request", required=True)
    validate_request_against_scenario.set_defaults(_fn=_cmd_validate_request_against_scenario)

    inspect_phase_menu = sub.add_parser(
        "inspect-phase-menu",
        help="Print participant-safe phase arm-count and request-menu semantics.",
    )
    inspect_phase_menu.add_argument("--scenario-root", required=True)
    inspect_phase_menu.add_argument("--phase", default=None)
    inspect_phase_menu.set_defaults(_fn=_cmd_inspect_phase_menu)

    validate_submission_shape = sub.add_parser(
        "validate-submission-shape",
        help="Validate participant submission structure without scoring.",
    )
    validate_submission_shape.add_argument("--submission", required=True)
    validate_submission_shape.set_defaults(_fn=_cmd_validate_submission_shape)

    validate_trial_output = sub.add_parser("validate-trial-output", help="Validate a returned trial-output bundle.")
    validate_trial_output.add_argument("--trial-output-root", required=True)
    validate_trial_output.set_defaults(_fn=_cmd_validate_trial_output)

    validate_phase_analysis = sub.add_parser("validate-phase-analysis", help="Validate one phase analysis submission.")
    validate_phase_analysis.add_argument("--submission", required=True)
    validate_phase_analysis.set_defaults(_fn=_cmd_validate_phase_analysis)

    validate_phase_action_policy = sub.add_parser(
        "validate-phase-action-policy",
        help="Validate one scenario's public phase-action policy.",
    )
    validate_phase_action_policy.add_argument("--scenario-root", required=True)
    validate_phase_action_policy.set_defaults(_fn=_cmd_validate_phase_action_policy)

    validate_phase_decision = sub.add_parser("validate-phase-decision", help="Validate one phase decision submission.")
    validate_phase_decision.add_argument("--submission", required=True)
    validate_phase_decision.add_argument("--scenario-root", default=None)
    validate_phase_decision.set_defaults(_fn=_cmd_validate_phase_decision)

    materialize = sub.add_parser("materialize", help="Materialize a governed trial view deterministically.")
    materialize.add_argument("--scenario-root", required=True)
    materialize.add_argument("--request", required=True, help="Path to TrialDevelopmentRequestV1 JSON.")
    materialize.add_argument("--out", required=True, help="Output directory for trial tables.")
    materialize.add_argument("--seed", type=int, required=True)
    materialize.add_argument("--overwrite", action="store_true")
    materialize.set_defaults(_fn=_cmd_materialize)

    inspect = sub.add_parser("inspect", help="Summarize a scenario bundle for review/debugging.")
    inspect.add_argument("--scenario-root", required=True)
    inspect.add_argument("--include-hidden-diagnostics", action="store_true")
    inspect.set_defaults(_fn=_cmd_inspect)

    args = parser.parse_args(argv)
    try:
        code = int(args._fn(args))
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        parser.error(str(exc))
        raise SystemExit(2) from exc
    raise SystemExit(code)
